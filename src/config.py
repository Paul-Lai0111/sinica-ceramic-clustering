"""Every parameter of the pipeline in one place."""

from __future__ import annotations

REGIONS: tuple[str, ...] = ("penghu", "nanke")
FEATURE_SETS: tuple[int, ...] = (13, 11)

# K-Means: K per dataset, as chosen in the original analysis (elbow + silhouette).
KMEANS_K: dict[tuple[str, int], int] = {
    ("penghu", 13): 3,
    ("nanke", 13): 4,
    ("penghu", 11): 3,
    ("nanke", 11): 3,
}
KMEANS_N_INIT = 10
RANDOM_STATE = 0
K_RANGE = range(2, 11)  # K values scanned for the elbow and silhouette curves

# DBSCAN: grid searched on min-max scaled PCA coordinates; the best silhouette wins.
DBSCAN_EPS_GRID: tuple[float, ...] = (0.06, 0.07, 0.08, 0.09, 0.10, 0.11)
DBSCAN_MIN_SAMPLES_GRID: tuple[int, ...] = (3, 4, 5, 6, 7)

# Silhouette scores from the original written report, for comparison.
REPORT_SILHOUETTE: dict[tuple[str, int, str], float] = {
    ("penghu", 13, "kmeans"): 0.48, ("penghu", 13, "dbscan"): 0.40,
    ("nanke", 13, "kmeans"): 0.50, ("nanke", 13, "dbscan"): 0.46,
    ("penghu", 11, "kmeans"): 0.57, ("penghu", 11, "dbscan"): 0.49,
    ("nanke", 11, "kmeans"): 0.64, ("nanke", 11, "dbscan"): 0.58,
}

# Figures: dark surface, gold / blue / aqua clusters (checked for colour-vision separation),
# a light neutral for a fourth cluster, and hollow grey rings for DBSCAN noise.
SURFACE = "#192a38"
CANVAS = "#0b1219"
TEXT = "#f2f1ed"
TEXT_MUTED = "#b0bcc5"
ACCENT = "#dec8a4"
CLUSTER_COLORS: tuple[str, ...] = ("#b98a2e", "#3987e5", "#1fa37a", "#e3ddd0")
NOISE_COLOR = "#7d8a96"
