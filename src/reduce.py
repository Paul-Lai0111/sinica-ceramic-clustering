"""Standardize the variables and project them onto two principal components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardize_pca(features: pd.DataFrame, n_components: int = 2) -> np.ndarray:
    """Z-score every variable, then return the first `n_components` principal-component scores.

    Clustering runs in this 2D space, which is also the space the plots show.
    """
    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=n_components).fit_transform(scaled)


def align_signs(reference: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Flip PCA axes (their sign is arbitrary) so `coords` points the same way as `reference`."""
    aligned = coords.copy()
    for axis in range(aligned.shape[1]):
        if np.corrcoef(reference[:, axis], aligned[:, axis])[0, 1] < 0:
            aligned[:, axis] *= -1
    return aligned
