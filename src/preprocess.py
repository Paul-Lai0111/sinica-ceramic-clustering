"""Load a region's sherd table and build the 13- and 11-variable feature sets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The source tables have 15 columns in this order (see data/README.md).
ID_COLUMNS: list[str] = ["lab_id", "site_code", "site_name", "body_total"]
MINERALS: list[str] = [
    "Quartz + sericite",
    "K-feldspar",
    "Basaltic minerals (plagioclase, pyroxene)",
    "Weathered igneous rock",
    "Granite",
    "Sedimentary rock fragments",
    "Metamorphic rock fragments",
    "Quartzite / chert / meta-igneous rock",
    "Carbonates",
    "Iron-rich clay pellets",
    "Opaque minerals",
]
COLUMNS: list[str] = ID_COLUMNS + MINERALS


def load_region(path: Path) -> pd.DataFrame:
    """Read one region's table (.xlsx or .csv), rename its columns by position, fill blanks with 0."""
    table = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    if table.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, found {table.shape[1]}. See data/README.md.")
    table.columns = COLUMNS
    numeric = ["site_code", "body_total", *MINERALS]
    table[numeric] = table[numeric].fillna(0)
    return table


def feature_set(table: pd.DataFrame, n_variables: int) -> pd.DataFrame:
    """Return the modelling variables.

    13 variables: drop the non-numeric lab ID and site name; the site is kept as a numeric site code.
    11 variables: also drop the site code (it leaks where a sherd was found, and coding categories as
    integers creates distances that don't exist) and the body total (texture, not mineral composition).
    """
    features = table.drop(columns=["lab_id", "site_name"])
    if n_variables == 11:
        features = features.drop(columns=["site_code", "body_total"])
    elif n_variables != 13:
        raise ValueError("n_variables must be 13 or 11")
    return features
