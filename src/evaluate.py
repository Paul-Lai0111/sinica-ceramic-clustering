"""Silhouette scores and per-cluster mineral signatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .preprocess import MINERALS


def silhouette(coords: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (−1 to 1; above 0.5 indicates strong structure)."""
    return float(silhouette_score(coords, labels))


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters by size (largest = 0) so colours follow a stable rule; noise stays −1."""
    clusters = [c for c in np.unique(labels) if c != -1]
    order = sorted(clusters, key=lambda c: -(labels == c).sum())
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping.get(label, -1) for label in labels])


def cluster_medians(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Median share of each mineral per cluster (rows: minerals, columns: clusters), plus cluster sizes."""
    minerals = table[MINERALS]
    medians = {f"Cluster {c + 1}": minerals[labels == c].median() for c in sorted(set(labels) - {-1})}
    result = pd.DataFrame(medians)
    result.loc["Sherds"] = [int((labels == c).sum()) for c in sorted(set(labels) - {-1})]
    return result
