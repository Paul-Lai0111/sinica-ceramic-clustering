"""K-Means with elbow and silhouette curves, and DBSCAN with a silhouette-based parameter search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from . import config


@dataclass
class KPoint:
    k: int
    sse: float
    silhouette: float


@dataclass
class GridCell:
    eps: float
    min_samples: int
    silhouette: float | None
    clusters: int
    noise: int
    labels: np.ndarray


def kmeans(coords: np.ndarray, k: int) -> np.ndarray:
    """Cluster PCA coordinates with K-Means; returns one label per sherd."""
    model = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE)
    return model.fit(coords).labels_


def k_curve(coords: np.ndarray) -> list[KPoint]:
    """SSE (elbow method) and silhouette score for every K in config.K_RANGE."""
    points = []
    for k in config.K_RANGE:
        model = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE).fit(coords)
        points.append(KPoint(k, float(model.inertia_), float(silhouette_score(coords, model.labels_))))
    return points


def dbscan_search(coords: np.ndarray) -> tuple[list[GridCell], GridCell]:
    """Run DBSCAN over the eps × min_samples grid on min-max scaled coordinates.

    Noise keeps its own label (-1) in the silhouette, as in the original analysis. Returns every cell
    and the one with the highest silhouette.
    """
    scaled = MinMaxScaler().fit_transform(coords)
    grid = []
    for eps in config.DBSCAN_EPS_GRID:
        for min_samples in config.DBSCAN_MIN_SAMPLES_GRID:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled).labels_
            score = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 else None
            grid.append(GridCell(eps, min_samples, score, len(set(labels) - {-1}), int((labels == -1).sum()), labels))
    best = max((cell for cell in grid if cell.silhouette is not None), key=lambda cell: cell.silhouette)
    return grid, best
