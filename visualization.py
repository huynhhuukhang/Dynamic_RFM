"""Visualizations: segment composition, RFM shifts, and evolution timeline."""
from __future__ import annotations
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .segment_tracking import EvolutionRecord


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def plot_segment_sizes_over_time(snapshot_records: List[Dict]) -> plt.Figure:
    """
    snapshot_records: list of {"snapshot_date": ..., "seg_id": ..., "size": ...}
    """
    df = pd.DataFrame(snapshot_records)
    if df.empty:
        fig, ax = plt.subplots()
        ax.set_title("No snapshots")
        return fig
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (sid, grp) in enumerate(df.groupby("seg_id")):
        grp = grp.sort_values("snapshot_date")
        ax.plot(grp["snapshot_date"], grp["size"],
                marker="o", label=sid, color=COLORS[i % len(COLORS)])
    ax.set_title("Segment Sizes Over Time")
    ax.set_ylabel("Users")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_rfm_centroids_over_time(snapshot_records: List[Dict]) -> plt.Figure:
    """
    snapshot_records: list of {"snapshot_date": ..., "seg_id": ..., "R": ..., "F": ..., "M": ...}
    """
    df = pd.DataFrame(snapshot_records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (sid, grp) in enumerate(df.groupby("seg_id")):
        grp = grp.sort_values("snapshot_date")
        color = COLORS[i % len(COLORS)]
        for ax, metric in zip(axes, ("R", "F", "M")):
            ax.plot(grp["snapshot_date"], grp[metric], marker="o", color=color, label=sid)
    for ax, title in zip(axes, ("Recency", "Frequency", "Monetary")):
        ax.set_title(f"{title} centroid drift")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[0].legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_evolution_timeline(records: List[EvolutionRecord]) -> plt.Figure:
    if not records:
        fig, ax = plt.subplots()
        ax.set_title("No evolution events")
        return fig
    events = sorted(records, key=lambda r: r.snapshot)
    labels = [f"{r.snapshot} · {r.event}" for r in events]
    counts = {e: sum(1 for r in events if r.event == e)
              for e in {r.event for r in events}}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.bar(list(counts.keys()), list(counts.values()), color=COLORS[: len(counts)])
    ax1.set_title("Event type counts")
    ax1.set_ylabel("count")
    # simple timeline of events by category
    event_types = list(counts.keys())
    for i, ev in enumerate(event_types):
        snaps = [j for j, r in enumerate(events) if r.event == ev]
        ax2.scatter(snaps, [i] * len(snaps), s=40, marker="|")
    ax2.set_yticks(range(len(event_types)))
    ax2.set_yticklabels(event_types)
    ax2.set_title("Event timeline (ordered by snapshot)")
    ax2.set_xlabel("event index")
    fig.tight_layout()
    return fig


def plot_segment_composition(snapshot_df: pd.DataFrame, snapshot_date) -> plt.Figure:
    """
    Stacked bar of user counts per segment at a given snapshot.
    snapshot_df: columns include CustomerID, seg_id
    """
    sub = snapshot_df[snapshot_df["snapshot_date"] == snapshot_date]
    counts = sub.groupby("seg_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color=COLORS[: len(counts)])
    ax.set_title(f"Segment composition @ {snapshot_date}")
    ax.set_ylabel("Users")
    fig.tight_layout()
    return fig