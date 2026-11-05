"""Reproduce the analysis: read the two regional tables, write results/silhouette.csv and results/figures/.

The data is not included in this repository (see data/README.md). Point the pipeline at your copies with
environment variables, or place them in data/ as penghu.xlsx and nanke.xlsx:

    SINICA_PENGHU=/path/to/penghu.xlsx SINICA_NANKE=/path/to/nanke.xlsx python run_pipeline.py

All parameters live in src/config.py.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from src import config
from src.cluster import dbscan_search, k_curve, kmeans
from src.evaluate import cluster_medians, relabel_by_size, silhouette
from src.figures import all_scatters, choose_k, correlation, dbscan_grid, fingerprints, pca_pruning
from src.preprocess import MINERALS, feature_set, load_region
from src.reduce import align_signs, standardize_pca

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LABELS = {"penghu": "Penghu", "nanke": "Nanke"}


def data_paths() -> dict[str, Path]:
    """Locate the two tables from the environment, falling back to data/."""
    paths = {
        "penghu": Path(os.environ.get("SINICA_PENGHU", ROOT / "data" / "penghu.xlsx")),
        "nanke": Path(os.environ.get("SINICA_NANKE", ROOT / "data" / "nanke.xlsx")),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        print("Data not included. See data/README.md.")
        print("Missing: " + ", ".join(missing))
        sys.exit(1)
    return paths


def main() -> None:
    paths = data_paths()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    scatter_panels = []
    nanke: dict[str, object] = {}

    for region in config.REGIONS:
        table = load_region(paths[region])
        reference = None
        for n in config.FEATURE_SETS:
            features = feature_set(table, n)
            coords = standardize_pca(features)
            if reference is not None:
                coords = align_signs(reference, coords)  # same orientation for both feature sets in the plots
            reference = coords if reference is None else reference

            k = config.KMEANS_K[(region, n)]
            km_labels = kmeans(coords, k)
            km_score = silhouette(coords, km_labels)
            grid, best = dbscan_search(coords)

            name = f"{LABELS[region]}, {n} variables"
            rows.append({"region": LABELS[region], "variables": n, "method": "K-Means", "parameters": f"k={k}",
                         "groups": k, "silhouette": round(km_score, 4), "report": config.REPORT_SILHOUETTE[(region, n, "kmeans")]})
            rows.append({"region": LABELS[region], "variables": n, "method": "DBSCAN", "parameters": f"eps={best.eps}, min_samples={best.min_samples}",
                         "groups": f"{best.clusters} + noise ({best.noise})", "silhouette": round(best.silhouette, 4),
                         "report": config.REPORT_SILHOUETTE[(region, n, "dbscan")]})
            scatter_panels.append((f"{name}: K-Means", coords, relabel_by_size(km_labels)))
            scatter_panels.append((f"{name}: DBSCAN", coords, relabel_by_size(best.labels)))

            if region == "nanke":
                nanke[f"coords{n}"], nanke[f"labels{n}"] = coords, relabel_by_size(km_labels)
                if n == 11:
                    nanke.update(table=table, curve=k_curve(coords), grid=grid, best=best)

    with open(RESULTS / "silhouette.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pca_pruning(nanke["coords13"], nanke["labels13"], nanke["coords11"], nanke["labels11"], FIGURES / "pca_pruning_nanke.png")
    all_scatters(scatter_panels, FIGURES / "all_clusterings.png")
    choose_k(nanke["curve"], config.KMEANS_K[("nanke", 11)], "Choosing K for K-Means (Nanke, 11 variables)", FIGURES / "choose_k_nanke_11.png")
    dbscan_grid(nanke["grid"], nanke["best"], "DBSCAN parameter search (Nanke, 11 variables)", FIGURES / "dbscan_grid_nanke_11.png")
    table = nanke["table"]
    correlation(table[MINERALS].corr(), "Correlation between the 11 mineral variables (Nanke)", FIGURES / "correlation_nanke.png")
    fingerprints(cluster_medians(table, nanke["labels11"]), "Median mineral share by cluster (Nanke, 11 variables, K-Means)", FIGURES / "fingerprints_nanke_11.png")

    print(f"{'Dataset':<22}{'Method':<9}{'Silhouette':>11}{'Report':>8}  Parameters")
    for row in rows:
        print(f"{row['region'] + ', ' + str(row['variables']) + ' variables':<22}{row['method']:<9}{row['silhouette']:>11.3f}{row['report']:>8.2f}  {row['parameters']}")
    print(f"Wrote {RESULTS / 'silhouette.csv'} and {len(list(FIGURES.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
