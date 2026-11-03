"""Drift detection: Page-Hinkley, ADWIN-lite, and centroid/distribution drift."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from .config import DriftConfig


class PageHinkley:
    def __init__(self, delta: float = 0.005, threshold: float = 0.05, min_window: int = 5):
        self.delta = delta
        self.threshold = threshold
        self.min_window = min_window
        self.mean = 0.0
        self.cum = 0.0
        self.n = 0
        self.drift_detected = False
        self.drift_point: Optional[int] = None

    def update(self, value: float) -> bool:
        self.drift_detected = False
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        if self.n < self.min_window:
            return False
        if self.cum > self.threshold:
            self.drift_detected = True
            self.drift_point = self.n
            self.mean = 0.0
            self.cum = 0.0
            self.n = 0
        return self.drift_detected


class ADWINLite:
    """Simplified ADWIN: two-window mean comparison with adaptive threshold."""
    def __init__(self, max_window: int = 64, threshold: float = 0.05):
        self.max_window = max_window
        self.threshold = threshold
        self.window: List[float] = []

    def update(self, value: float) -> bool:
        self.window.append(value)
        if len(self.window) > self.max_window:
            self.window.pop(0)
        n = len(self.window)
        if n < 4:
            return False
        split = n // 2
        left = np.array(self.window[:split])
        right = np.array(self.window[split:])
        diff = abs(left.mean() - right.mean())
        eps = np.sqrt(np.log(2.0 / self.threshold) * (1.0 / max(1, len(left)) + 1.0 / max(1, len(right)))) * 0.5
        if diff > eps and diff > self.threshold:
            self.window = list(right)
            return True
        return False


class CentroidDriftMonitor:
    """Track L2 drift of a segment's centroid relative to its running mean."""
    def __init__(self, threshold: float = 0.25, alpha: float = 0.1):
        self.threshold = threshold
        self.alpha = alpha
        self.mean: Optional[np.ndarray] = None

    def update(self, centroid: np.ndarray) -> float:
        c = np.asarray(centroid, dtype=float)
        if self.mean is None:
            self.mean = c.copy()
            return 0.0
        drift = float(np.linalg.norm(c - self.mean))
        self.mean = (1 - self.alpha) * self.mean + self.alpha * c
        return drift

    def is_drift(self, drift: float) -> bool:
        return drift > self.threshold


class DriftMonitor:
    """Unified drift monitor for RFM distributions per segment."""
    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.ph = PageHinkley(delta=cfg.delta, threshold=cfg.threshold, min_window=cfg.min_window)
        self.adwin = ADWINLite(threshold=cfg.threshold)
        self.centroids: Dict[str, CentroidDriftMonitor] = {}
        self.feature_drift: Dict[str, float] = {}

    def update(self, seg_id: str, centroid: np.ndarray, summary_value: float) -> Dict[str, object]:
        result = {"ph": False, "adwin": False, "centroid_drift": 0.0, "drift": False}
        if self.cfg.method == "page_hinkley":
            result["ph"] = self.ph.update(summary_value)
        else:
            result["adwin"] = self.adwin.update(summary_value)

        mon = self.centroids.setdefault(seg_id, CentroidDriftMonitor(self.cfg.feature_drift_threshold))
        d = mon.update(centroid)
        result["centroid_drift"] = d
        result["drift"] = bool(result["ph"] or result["adwin"] or mon.is_drift(d))
        return result