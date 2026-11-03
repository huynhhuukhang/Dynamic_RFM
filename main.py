"""End-to-end pipeline: load → clean → rolling RFM → adaptive window + K → tracking → drift → eval."""
from __future__ import annotations
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    PipelineConfig, DataConfig, PreprocessConfig, RollingRFMConfig,
    AdaptiveKConfig, AdaptiveWindowConfig, TrackingConfig, DriftConfig,
)
from .data_loader import load_all
from .preprocessing import clean_dataframe
from .rolling_rfm import build_snapshots
from .adaptive_kmeans import adaptive_kmeans, _evaluate_k, _normalize_scores, _composite_score
from .adaptive_window import AdaptiveWindowSelector
from .segment_tracking import SegmentTracker
from .drift_detection import DriftMonitor
from .evaluation import compare_all


def run_pipeline(cfg: PipelineConfig, verbose: bool = True) -> Dict[str, object]:
    if verbose:
        print(f"[1/7] Loading dataset: {cfg.data.path}")
    df = load_all(cfg.data)
    if verbose:
        print(f"      raw rows: {len(df)}")

    if verbose:
        print("[2/7] Cleaning / preprocessing")
    df = clean_dataframe(df, cfg.preprocess)
    if verbose:
        print(f"      clean rows: {len(df)} | customers: {df['CustomerID'].nunique()}")

    if verbose:
        print("[3/7] Building rolling RFM snapshots")
    snaps_by_window = build_snapshots(df, cfg.rolling_rfm)
    for w, s in snaps_by_window.items():
        if verbose:
            print(f"      window={w}d snapshots rows: {len(s)}")

    if verbose:
        print("[4/7] Adaptive window + adaptive K clustering per snapshot")
    window_selector = AdaptiveWindowSelector(cfg.adaptive_window)
    tracker = SegmentTracker(cfg.tracking)
    drift_monitor = DriftMonitor(cfg.drift)

    snapshot_records: List[Dict] = []
    rfm_records: List[Dict] = []
    drift_log: List[Dict] = []

    # For each snapshot_date (unified across windows), select best window then cluster
    all_dates = sorted({d for s in snaps_by_window.values() if not s.empty
                        for d in s["snapshot_date"].unique()})

    for snap_date in all_dates:
        quality_by_window: Dict[int, float] = {}
        labels_by_window: Dict[int, tuple] = {}
        for w, snapdf in snaps_by_window.items():
            if snapdf.empty:
                continue
            group = snapdf[snapdf["snapshot_date"] == snap_date]
            if len(group) < cfg.adaptive_k.k_min + 1:
                continue
            X = StandardScaler().fit_transform(group[["Recency", "Frequency", "Monetary"]].values)
            labels, k_sel, metrics = adaptive_kmeans(X, cfg.adaptive_k)
            quality_by_window[w] = _composite_score(_normalize_scores({k_sel: metrics}).get(k_sel, metrics), cfg.adaptive_k)
            labels_by_window[w] = (group, X, labels)

        if not quality_by_window:
            continue

        chosen_w = window_selector.select(quality_by_window)
        group, X, labels = labels_by_window[chosen_w]

        # Build raw-RFM centroids per cluster
        centroids = []
        for lab in sorted(set(labels)):
            centroids.append(group.loc[labels == lab, ["Recency", "Frequency", "Monetary"]].values.mean(axis=0))
        centroids = np.array(centroids)

        id_map = tracker.update(
            snapshot_label=str(pd.Timestamp(snap_date).date()),
            raw_labels=labels,
            user_ids=group["CustomerID"].values,
            centroids_raw=centroids,
        )

        # Record per-segment stats + drift
        for raw_lab, seg_id in id_map.items():
            mask = labels == raw_lab
            sub = group[mask]
            if sub.empty:
                continue
            snapshot_records.append({
                "snapshot_date": snap_date, "seg_id": seg_id,
                "size": int(mask.sum()),
                "window_days": chosen_w,
            })
            rfm_records.append({
                "snapshot_date": snap_date, "seg_id": seg_id,
                "R": float(sub["Recency"].mean()),
                "F": float(sub["Frequency"].mean()),
                "M": float(sub["Monetary"].mean()),
            })
            centroid = sub[["Recency", "Frequency", "Monetary"]].values.mean(axis=0)
            summary = float(sub["Monetary"].sum())
            dres = drift_monitor.update(seg_id, centroid, summary)
            if dres["drift"]:
                drift_log.append({
                    "snapshot_date": snap_date, "seg_id": seg_id,
                    "centroid_drift": dres["centroid_drift"],
                    "ph": dres["ph"], "adwin": dres["adwin"],
                })

    if verbose:
        print(f"      snapshots processed: {len(all_dates)}")
        print(f"      evolution events: {len(tracker.history)}")
        print(f"      drift events: {len(drift_log)}")

    if verbose:
        print("[5/7] Benchmarking")
    benchmark_df = compare_all(df, cfg.rolling_rfm, cfg.adaptive_k)
    if verbose:
        print(benchmark_df.to_string(index=False))

    if verbose:
        print("[6/7] Compiling results")
    return {
        "data": df,
        "snapshot_records": pd.DataFrame(snapshot_records),
        "rfm_records": pd.DataFrame(rfm_records),
        "evolution_events": tracker.history,
        "drift_log": pd.DataFrame(drift_log),
        "benchmark": benchmark_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Customer Segment Monitoring (adaptive).")
    parser.add_argument("--data", required=True, help="Path to Online Retail II CSV/XLSX")
    parser.add_argument("--window-min", type=int, default=28)
    parser.add_argument("--window-mid", type=int, default=56)
    parser.add_argument("--window-max", type=int, default=91)
    parser.add_argument("--stride-days", type=int, default=7)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data=DataConfig(path=args.data),
        rolling_rfm=RollingRFMConfig(
            windows_days=[args.window_min, args.window_mid, args.window_max],
            snapshot_stride_days=args.stride_days,
        ),
        adaptive_k=AdaptiveKConfig(
            k_min=args.k_min, k_max=args.k_max,
            min_cluster_size=args.min_cluster_size,
        ),
    )

    results = run_pipeline(cfg, verbose=True)

    if args.plot:
        from .visualization import (
            plot_segment_sizes_over_time, plot_rfm_centroids_over_time,
            plot_evolution_timeline,
        )
        plot_segment_sizes_over_time(results["snapshot_records"].to_dict("records"))
        plot_rfm_centroids_over_time(results["rfm_records"].to_dict("records"))
        plot_evolution_timeline(results["evolution_events"])
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()