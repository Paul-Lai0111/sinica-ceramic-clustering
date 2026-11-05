"""Figures for results/figures/: PCA scatters, the K curves, the DBSCAN grid, correlations and cluster medians.

Every figure shows derived values only (PCA coordinates, labels, scores, correlations, medians).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config  # noqa: E402
from .cluster import GridCell, KPoint  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": config.CANVAS,
    "axes.facecolor": config.SURFACE,
    "axes.edgecolor": "none",
    "axes.labelcolor": config.TEXT_MUTED,
    "axes.titlecolor": config.TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": config.TEXT_MUTED,
    "ytick.color": config.TEXT_MUTED,
    "text.color": config.TEXT,
    "font.size": 10,
    "savefig.dpi": 200,
    "savefig.facecolor": config.CANVAS,
})

GOLD_RAMP = LinearSegmentedColormap.from_list("gold", [config.SURFACE, config.ACCENT])
DIVERGING = LinearSegmentedColormap.from_list("blue_gold", ["#3987e5", config.SURFACE, config.ACCENT])


def _scatter(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    ax.axhline(0, color=config.TEXT_MUTED, lw=0.6, ls=(0, (3, 4)), alpha=0.4)
    ax.axvline(0, color=config.TEXT_MUTED, lw=0.6, ls=(0, (3, 4)), alpha=0.4)
    for cluster in sorted(set(labels)):
        points = coords[labels == cluster]
        if cluster == -1:
            ax.scatter(points[:, 0], points[:, 1], s=26, facecolors="none", edgecolors=config.NOISE_COLOR, linewidths=1.2, label="Noise")
        else:
            ax.scatter(points[:, 0], points[:, 1], s=30, color=config.CLUSTER_COLORS[cluster], edgecolors=config.SURFACE, linewidths=0.6, label=f"Cluster {cluster + 1}")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, fontsize=8, labelcolor=config.TEXT_MUTED, loc="best")


def pca_pruning(coords13: np.ndarray, labels13: np.ndarray, coords11: np.ndarray, labels11: np.ndarray, out: Path) -> None:
    """Nanke sherds in PCA space before (13 variables) and after (11 variables) pruning, by K-Means cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    _scatter(axes[0], coords13, labels13, "13 variables (site code and body total included)")
    _scatter(axes[1], coords11, labels11, "11 compositional variables")
    fig.suptitle("Nanke sherds in PCA space, before and after feature pruning (K-Means)", color=config.TEXT, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)


def all_scatters(panels: list[tuple[str, np.ndarray, np.ndarray]], out: Path) -> None:
    """Every dataset × method in one grid (2 regions × 2 feature sets × 2 methods)."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, (title, coords, labels) in zip(axes.flat, panels):
        _scatter(ax, coords, labels, title)
    fig.savefig(out)
    plt.close(fig)


def choose_k(curve: list[KPoint], selected: int, title: str, out: Path) -> None:
    """Elbow (SSE) and silhouette curves side by side, with the selected K marked."""
    ks = [p.k for p in curve]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, values, name in ((axes[0], [p.sse for p in curve], "SSE (elbow method)"), (axes[1], [p.silhouette for p in curve], "Silhouette score")):
        ax.grid(axis="y", color=config.TEXT_MUTED, alpha=0.15)
        ax.plot(ks, values, color=config.TEXT_MUTED, lw=2, marker="o", ms=5)
        ax.axvline(selected, color=config.ACCENT, lw=1, ls=(0, (3, 4)), alpha=0.7)
        ax.plot([selected], [values[ks.index(selected)]], "o", ms=10, color=config.ACCENT, mec=config.SURFACE)
        ax.set_title(name)
        ax.set_xlabel("Number of clusters (K)")
        ax.set_xticks(ks)
    fig.suptitle(title, color=config.TEXT, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)


def dbscan_grid(grid: list[GridCell], best: GridCell, title: str, out: Path) -> None:
    """Silhouette over the eps × min_samples grid; the best cell is outlined."""
    eps = list(config.DBSCAN_EPS_GRID)
    mins = list(config.DBSCAN_MIN_SAMPLES_GRID)
    values = np.full((len(eps), len(mins)), np.nan)
    for cell in grid:
        if cell.silhouette is not None:
            values[eps.index(cell.eps), mins.index(cell.min_samples)] = cell.silhouette
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    image = ax.imshow(values, cmap=GOLD_RAMP, aspect="auto")
    for (row, col), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color=config.CANVAS if value > np.nanmean(values) else config.TEXT)
    ax.add_patch(plt.Rectangle((mins.index(best.min_samples) - 0.5, eps.index(best.eps) - 0.5), 1, 1, fill=False, ec=config.TEXT, lw=2))
    ax.set_xticks(range(len(mins)), mins)
    ax.set_yticks(range(len(eps)), [f"{e:.2f}" for e in eps])
    ax.set_xlabel("min_samples")
    ax.set_ylabel("eps (min-max scaled PCA space)")
    fig.suptitle(title, color=config.TEXT, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Silhouette")
    fig.savefig(out)
    plt.close(fig)


def correlation(corr: pd.DataFrame, title: str, out: Path) -> None:
    """Correlation heatmap of the mineral variables (blue negative, gold positive)."""
    fig, ax = plt.subplots(figsize=(9, 7.4), constrained_layout=True)
    image = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    for (row, col), value in np.ndenumerate(corr.values):
        ax.text(col, row, "n/a" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=7, color=config.TEXT_MUTED if np.isnan(value) or abs(value) < 0.6 else config.CANVAS)
    ax.set_xticks(range(len(corr.columns)), [str(i + 1) for i in range(len(corr.columns))])
    ax.set_yticks(range(len(corr.index)), [f"{i + 1}  {name}" for i, name in enumerate(corr.index)])
    fig.suptitle(title, color=config.TEXT, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.savefig(out)
    plt.close(fig)


def fingerprints(medians: pd.DataFrame, title: str, out: Path) -> None:
    """Median share (%) of each mineral per cluster, as a heatmap table."""
    shares = medians.drop(index="Sherds")
    sizes = medians.loc["Sherds"]
    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    ax.imshow(shares.values, cmap=GOLD_RAMP, vmin=0, vmax=80, aspect="auto")
    for (row, col), value in np.ndenumerate(shares.values):
        ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=9, color=config.CANVAS if value > 45 else config.TEXT)
    ax.set_xticks(range(len(shares.columns)), [f"{name}\n{int(sizes[name])} sherds" for name in shares.columns])
    ax.set_yticks(range(len(shares.index)), shares.index)
    ax.xaxis.tick_top()
    fig.suptitle(title, color=config.TEXT, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(out)
    plt.close(fig)
