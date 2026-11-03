"""Adaptive rolling window selection with quality + stability + hysteresis."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from .config import AdaptiveWindowConfig


class AdaptiveWindowSelector:
    def __init__(self, cfg: AdaptiveWindowConfig):
        self.cfg = cfg
        self.current_window: Optional[int] = None
        self.history: Dict[int, list[float]] = {w: [] for w in cfg.candidate_windows}

    def _stability(self, window: int) -> float:
        h = self.history[window]
        if len(h) < 2:
            return 1.0
        return float(1.0 - min(1.0, np.std(h[-5:]) if len(h) >= 2 else 0.0))

    def select(self, quality_by_window: Dict[int, float]) -> int:
        """
        quality_by_window: composite clustering quality score per candidate window.
        Returns chosen window, considering stability and hysteresis.
        """
        if not quality_by_window:
            return self.current_window or self.cfg.candidate_windows[0]

        combined = {}
        for w, q in quality_by_window.items():
            stab = self._stability(w)
            combined[w] = self.cfg.quality_weight * q + self.cfg.stability_weight * stab
            self.history[w].append(q)

        best_w = max(combined, key=combined.get)

        if self.current_window is None:
            self.current_window = best_w
            return best_w

        current_score = combined.get(self.current_window, -np.inf)
        best_score = combined[best_w]

        # Hysteresis: only switch if improvement is significant
        if best_w != self.current_window and (best_score - current_score) >= self.cfg.min_switch_improvement:
            self.current_window = best_w
        return self.current_window