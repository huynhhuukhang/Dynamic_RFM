"""Benchmarking: static RFM+KMeans vs fixed rolling vs adaptive system."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from .rolling_rfm import compute_lifetime_rfm, build_snapshots
from .adaptive_kmeans import adaptive_kmeans, _evaluate_k, _normalize_scores, _composite_score
from .config import AdaptiveKConfig, RollingRFMConfig
from .segment_tracking import jaccard


def compute_quality_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(set(labels)) < 2 or X.shape[0] < 3:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def stability_across_snapshots(prev_assign: Dict[int, str], curr_assign: Dict[int, str]) -> float:
    """Average Jaccard over matched segment ids (simple global stability proxy)."""
    if not prev_assign or not curr_assign:
        return 0.0
    segs_prev: Dict[str, set] = {}
    for uid, sid in prev_assign.items():
        segs_prev.setdefault(sid, set()).add(uid)
    segs_curr: Dict[str, set] = {}
    for uid, sid in curr_assign.items():
        segs_curr.setdefault(sid, set()).add(uid)
    sims = []
    for sid, users in segs_curr.items():
        if sid in segs_prev:
            sims.append(jaccard(segs_prev[sid], users))
    return float(np.mean(sims)) if sims else 0.0


@dataclass
class BenchmarkResult:
    method: str
    quality: Dict[str, float] = field(default_factory=dict)
    stability: float = 0.0
    runtime_sec: float = 0.0
    n_snapshots: int = 0
    extra: Dict[str, object] = field(default_factory=dict)


def _assign_user_segments(rfm_df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    return {int(uid): f"c{int(lab)}" for uid, lab in zip(rfm_df["CustomerID"].values, labels)}


def benchmark_static_kmeans(df: pd.DataFrame, k: int = 5) -> BenchmarkResult:
    t0 = time.perf_counter()
    reference = df["InvoiceDate"].max()
    rfm = compute_lifetime_rfm(df, reference=reference)
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].values)
    model = KMeans(n_clusters=min(k, max(2, len(rfm) - 1)), random_state=42, n_init=20)
    labels = model.fit_predict(X)
    q = compute_quality_metrics(X, labels)
    return BenchmarkResult(method="static_kmeans", quality=q, stability=1.0,
                           runtime_sec=time.perf_counter() - t0, n_snapshots=1)


def benchmark_fixed_rolling(
    df: pd.DataFrame,
    window_days: int = 28,
    stride_days: int = 7,
    k: int = 5,
) -> BenchmarkResult:
    t0 = time.perf_counter()
    cfg = RollingRFMConfig(windows_days=[window_days], snapshot_stride_days=stride_days)
    snaps = build_snapshots(df, cfg)[window_days]
    if snaps.empty:
        return BenchmarkResult(method=f"fixed_rolling_{window_days}d", runtime_sec=time.perf_counter() - t0)

    q_list: List[Dict[str, float]] = []
    prev_assign: Dict[int, str] = {}
    stab: List[float] = []
    for _, group in snaps.groupby("snapshot_date"):
        if len(group) < k + 1:
            continue
        X = StandardScaler().fit_transform(group[["Recency", "Frequency", "Monetary"]].values)
        model = KMeans(n_clusters=min(k, len(group) - 1), random_state=42, n_init=20)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        q_list.append(compute_quality_metrics(X, labels))
        curr_assign = _assign_user_segments(group, labels)
        stab.append(stability_across_snapshots(prev_assign, curr_assign))
        prev_assign = curr_assign

    quality = {m: float(np.nanmean([q[m] for q in q_list])) for m in q_list[0]} if q_list else {}
    return BenchmarkResult(
        method=f"fixed_rolling_{window_days}d",
        quality=quality,
        stability=float(np.mean(stab)) if stab else 0.0,
        runtime_sec=time.perf_counter() - t0,
        n_snapshots=len(q_list),
    )


def benchmark_adaptive(
    df: pd.DataFrame,
    rolling_cfg: RollingRFMConfig,
    adaptive_cfg: AdaptiveKConfig,
) -> BenchmarkResult:
    """Adaptive window + adaptive K over rolling snapshots."""
    t0 = time.perf_counter()
    snaps_by_w = build_snapshots(df, rolling_cfg)
    best_quality_by_w: Dict[int, List[Dict[str, float]]] = {w: [] for w in rolling_cfg.windows_days}
    stability: Dict[int, List[float]] = {w: [] for w in rolling_cfg.windows_days}
    chosen_ks: List[int] = []
    prev_assigns: Dict[int, Dict[int, str]] = {w: {} for w in rolling_cfg.windows_days}

    for w in rolling_cfg.windows_days:
        snapdf = snaps_by_w[w]
        if snapdf.empty:
            continue
        for _, group in snapdf.groupby("snapshot_date"):
            if len(group) < adaptive_cfg.k_min + 1:
                continue
            X = StandardScaler().fit_transform(group[["Recency", "Frequency", "Monetary"]].values)
            labels, k_sel, metrics = adaptive_kmeans(X, adaptive_cfg)
            q = compute_quality_metrics(X, labels)
            if not np.isnan(q["silhouette"]):
                best_quality_by_w[w].append(q)
            chosen_ks.append(k_sel)
            curr_assign = _assign_user_segments(group, labels)
            stability[w].append(stability_across_snapshots(prev_assigns[w], curr_assign))
            prev_assigns[w] = curr_assign

    # Aggregate across all windows as an "ensemble" view
    all_q = [q for w, qs in best_quality_by_w.items() for q in qs]
    all_s = [s for w, ss in stability.items() for s in ss]
    quality = {m: float(np.nanmean([q[m] for q in all_q])) for m in ("silhouette", "davies_bouldin", "calinski_harabasz")} if all_q else {}
    return BenchmarkResult(
        method="adaptive_window_k",
        quality=quality,
        stability=float(np.mean(all_s)) if all_s else 0.0,
        runtime_sec=time.perf_counter() - t0,
        n_snapshots=len(all_q),
        extra={"mean_k": float(np.mean(chosen_ks)) if chosen_ks else None,
               "windows": list(rolling_cfg.windows_days)},
    )


def compare_all(df: pd.DataFrame, rolling_cfg: RollingRFMConfig, adaptive_cfg: AdaptiveKConfig) -> pd.DataFrame:
    results: List[BenchmarkResult] = []
    results.append(benchmark_static_kmeans(df))
    for w in rolling_cfg.windows_days:
        results.append(benchmark_fixed_rolling(df, window_days=w, stride_days=rolling_cfg.snapshot_stride_days))
    results.append(benchmark_adaptive(df, rolling_cfg, adaptive_cfg))

    rows = []
    for r in results:
        row = {
            "method": r.method,
            "runtime_sec": round(r.runtime_sec, 3),
            "n_snapshots": r.n_snapshots,
            "stability": round(r.stability, 3),
        }
        for m in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            v = r.quality.get(m, float("nan"))
            row[m] = round(v, 3) if v == v else "NaN"
        if r.extra:
            row.update(r.extra)
        rows.append(row)
    return pd.DataFrame(rows)