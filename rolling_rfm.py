"""Rolling RFM computation over configurable windows, with optional lifetime RFM."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from .config import RollingRFMConfig


def _rfm_for_window(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    reference: pd.Timestamp,
) -> pd.DataFrame:
    """RFM restricted to [window_start, window_end], with recency measured from `reference`."""
    mask = (df["InvoiceDate"] > window_start) & (df["InvoiceDate"] <= window_end)
    sub = df.loc[mask]
    if sub.empty:
        return pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

    last_purchase = sub.groupby("CustomerID")["InvoiceDate"].max()
    recency = (reference - last_purchase).dt.total_seconds() / 86400.0
    frequency = sub.groupby("CustomerID")["InvoiceNo"].nunique().astype(float)
    monetary = sub.groupby("CustomerID")["Amount"].sum().astype(float)

    rfm = pd.concat([recency.rename("Recency"),
                     frequency.rename("Frequency"),
                     monetary.rename("Monetary")], axis=1)
    return rfm.dropna()


def compute_lifetime_rfm(df: pd.DataFrame, reference: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
    recency = (reference - last_purchase).dt.total_seconds() / 86400.0
    frequency = df.groupby("CustomerID")["InvoiceNo"].nunique().astype(float)
    monetary = df.groupby("CustomerID")["Amount"].sum().astype(float)
    return pd.concat([recency.rename("Recency"),
                      frequency.rename("Frequency"),
                      monetary.rename("Monetary")], axis=1).dropna()


def build_snapshots(df: pd.DataFrame, cfg: RollingRFMConfig) -> Dict[int, pd.DataFrame]:
    """
    Build a dict:
        {window_days: DataFrame indexed by (snapshot_date, CustomerID) with RFM columns}
    Each window produces a time series of rolling RFM snapshots at stride `snapshot_stride_days`.
    """
    if df.empty:
        return {w: pd.DataFrame() for w in cfg.windows_days}

    start = df["InvoiceDate"].min().normalize()
    end = df["InvoiceDate"].max().normalize()

    snapshot_dates = pd.date_range(
        start=start + pd.Timedelta(days=cfg.snapshot_stride_days),
        end=end,
        freq=f"{cfg.snapshot_stride_days}D",
    )
    if len(snapshot_dates) == 0:
        snapshot_dates = pd.DatetimeIndex([end])

    out: Dict[int, list[pd.DataFrame]] = {w: [] for w in cfg.windows_days}

    for snap in snapshot_dates:
        for w in cfg.windows_days:
            ws = snap - pd.Timedelta(days=w)
            rfm = _rfm_for_window(df, ws, snap, reference=snap)
            if rfm.empty:
                continue
            rfm = rfm.reset_index().rename(columns={"index": "CustomerID"})
            rfm["snapshot_date"] = snap
            rfm["window_days"] = w
            out[w].append(rfm)

    result: Dict[int, pd.DataFrame] = {}
    for w in cfg.windows_days:
        if out[w]:
            result[w] = pd.concat(out[w], ignore_index=True)
        else:
            result[w] = pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary",
                                              "snapshot_date", "window_days"])
    return result


def attach_lifetime(snapshots: Dict[int, pd.DataFrame], df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """For each window snapshot, join lifetime RFM (computed at that snapshot date)."""
    if df.empty:
        return snapshots
    enriched: Dict[int, pd.DataFrame] = {}
    for w, snapdf in snapshots.items():
        if snapdf.empty:
            enriched[w] = snapdf
            continue
        frames = []
        for snap, group in snapdf.groupby("snapshot_date"):
            life = compute_lifetime_rfm(df[df["InvoiceDate"] <= snap], reference=snap)
            life = life.rename(columns={"Recency": "L_Recency",
                                        "Frequency": "L_Frequency",
                                        "Monetary": "L_Monetary"})
            merged = group.merge(life, left_on="CustomerID", right_index=True, how="left")
            frames.append(merged)
        enriched[w] = pd.concat(frames, ignore_index=True)
    return enriched