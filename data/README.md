# Data

**The data is not included in this repository.** The sherd measurements belong to the research
institutions (Academia Sinica and the Institute of History and Philology) and are not public.

The pipeline expects two tables, one per region, as `.xlsx` or `.csv`:

| Region | Sherds | Excavation sites |
|---|---|---|
| Penghu | 52 | 11 |
| Nanke | 205 | 17 |

Place them at `data/penghu.xlsx` and `data/nanke.xlsx`, or pass their paths with the
`SINICA_PENGHU` and `SINICA_NANKE` environment variables. Files in `data/` are ignored by git.

## Columns

Each table has 15 columns, in this order (the header text is not used; columns are read by position):

| # | Column | Type | Used in |
|---|---|---|---|
| 1 | Lab ID | text | dropped |
| 2 | Site code | integer | 13-variable set only |
| 3 | Site name | text | dropped |
| 4 | Body total | number | 13-variable set only |
| 5 | Quartz + sericite | number (%) | both sets |
| 6 | K-feldspar | number (%) | both sets |
| 7 | Basaltic minerals (plagioclase, pyroxene) | number (%) | both sets |
| 8 | Weathered igneous rock | number (%) | both sets |
| 9 | Granite | number (%) | both sets |
| 10 | Sedimentary rock fragments | number (%) | both sets |
| 11 | Metamorphic rock fragments | number (%) | both sets |
| 12 | Quartzite / chert / meta-igneous rock | number (%) | both sets |
| 13 | Carbonates | number (%) | both sets |
| 14 | Iron-rich clay pellets | number (%) | both sets |
| 15 | Opaque minerals | number (%) | both sets |

Columns 5–15 are the proportions of each mineral or rock component in a sherd's clay.

## Preprocessing

1. Blank cells are filled with 0.
2. **13 variables:** the lab ID and site name are dropped; the site is kept as its numeric site code.
3. **11 variables:** the site code and body total are also dropped, leaving the 11 compositional variables.
4. Both sets are standardized (z-scores) and projected onto two principal components; clustering runs in that 2D space.
