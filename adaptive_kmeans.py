"""Adaptive K-Means: selects k by combining Silhouette, Davies-Bouldin, Calinski-Harabasz."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from .config import AdaptiveKConfig


def _normalize_scores(raw: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """Min-max normalize each metric across k so we can combine them."""
    out = {k: dict(v) for k, v in raw.items()}
    for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        vals = [v[metric] for v in raw.values() if np.isfinite(v[metric])]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            for k in out:
                out[k][metric] = 0.5
            continue
        for k in out:
            x = raw[k][metric]
            # silhouette + calinski are higher-better, davies_bouldin lower-better
            if metric == "davies_bouldin":
                out[k][metric] = 1.0 - (x - lo) / (hi - lo)
            else:
                out[k][metric] = (x - lo) / (hi - lo)
    return out


def _composite_score(scores: Dict[str, float], cfg: AdaptiveKConfig) -> float:
    return (
        cfg.silhouette_weight * scores.get("silhouette", 0.0)
        + cfg.davies_bouldin_weight * scores.get("davies_bouldin", 0.0)
        + cfg.calinski_harabasz_weight * scores.get("calinski_harabasz", 0.0)
    )


def _evaluate_k(X: np.ndarray, k: int, cfg: AdaptiveKConfig) -> Tuple[float, Dict[str, float]]:
    model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=cfg.n_init)
    labels = model.fit_predict(X)
    if len(set(labels)) < 2:
        return -1.0, {"silhouette": -1.0, "davies_bouldin": np.inf, "calinski_harabasz": 0.0}
    try:
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        sil, db, ch = -1.0, np.inf, 0.0
    return sil, {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


def adaptive_kmeans(
    X: np.ndarray,
    cfg: AdaptiveKConfig,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """
    Returns (labels, chosen_k, metrics).
    Filters out candidates whose minimum cluster size < cfg.min_cluster_size.
    Uses composite normalized score.
    """
    n = X.shape[0]
    if n < cfg.k_min + 1:
        k_eff = max(2, min(cfg.k_min, n - 1)) if n >= 3 else 1
        if k_eff < 2:
            return np.zeros(n, dtype=int), 1, {"silhouette": -1.0}
        model = KMeans(n_clusters=k_eff, random_state=cfg.random_state, n_init=cfg.n_init)
        return model.fit_predict(X), k_eff, {"silhouette": -1.0}

    k_max = min(cfg.k_max, n - 1)
    raw: Dict[int, Dict[str, float]] = {}
    valid_k: Dict[int, np.ndarray] = {}

    for k in range(cfg.k_min, k_max + 1):
        sil, metrics = _evaluate_k(X, k, cfg)
        model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=cfg.n_init)
        labels = model.fit_predict(X)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < cfg.min_cluster_size:
            continue
        raw[k] = metrics
        valid_k[k] = labels

    if not raw:
        # Fall back: smallest k in range
        k = cfg.k_min
        model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=cfg.n_init)
        labels = model.fit_predict(X)
        sil, metrics = _evaluate_k(X, k, cfg)
        return labels, k, metrics

    norm = _normalize_scores(raw)
    best_k = cfg.k_min
    best_score = -np.inf
    for k, sc in norm.items():
        c = _composite_score(sc, cfg)
        if c > best_score:
            best_score = c
            best_k = k

    return valid_k[best_k], best_k, raw[best_k]


def scale_rfm(rfm_df, feature_cols=("Recency", "Frequency", "Monetary")):
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm_df[list(feature_cols)].values)
    return X, scaler