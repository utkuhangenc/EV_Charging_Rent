# Analysis data — 2017–2024

The large datasets used for the revised analyses are distributed separately as an **analysis-ready Zenodo data package**. GitHub tracks the Python scripts, documentation, selected numerical results, and figures; do **not** commit a copy of the downloaded data or full GIS source files into this repository.

**Dataset landing DOI:** https://doi.org/10.5281/zenodo.21346452. The **2017–2024 v2** data version must be published and selected before these paths will be available; the previous version covers 2017–2023.

## How to unpack

1. Download the Zenodo version labeled **2017–2024**. Its ZIP contains top-level `README.md`, `FILE_MANIFEST.csv`, `DATA_DICTIONARY.csv`, `SOURCE_AND_REPRODUCIBILITY_NOTES.md`, and `data/`.
2. Extract it to a location **outside the repository**, such as `~/datasets/evcs_2017_2024/`. **Do not extract it straight into the GitHub repository root** or it could overwrite this repository's `README.md`.
3. Set `EVCS_DATA_ROOT` to the extracted **`data/` directory**, not the directory containing the ZIP. See the root `README.md` for Linux/macOS and PowerShell examples.
4. Verify against the archive's `FILE_MANIFEST.csv` and `DATA_DICTIONARY.csv` if independently checking the released inputs. The manifest contains source-relative dataset names, sizes, row/column counts, and SHA-256 checksums.

## Expected eight files

Paths below are **relative to `EVCS_DATA_ROOT`**.

| Relative path | Role | Validated rows |
|---|---|---:|
| `output/native_bg_rf_replication/panel_twfe_2017_2024/analysis_panel_2017_2024.parquet` | Broader 2017–2024 TWFE analysis panel | 36,328 |
| `reference/balanced_twfe_complete_case_2017_2024.parquet` | Balanced eight-year reference: 2,903 CBGs | 23,224 |
| `intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet` | Harmonized ACS, 8,566 CBGs per year, 2013–2024 | 102,792 |
| `intermediate/evcs/afdc_ev_master_with_geography.parquet` | Reduced AFDC historical station snapshots, 2017–2024; 19 columns | 316,503 |
| `output/native_bg_rf_replication/exact_replication_panel_2017_2024.csv` | Cross-sectional RF/Poisson analysis panel | 36,328 |
| `output/native_bg_rf_replication/panel_twfe_2017_2024/evcs_24h_access_twfe_2017_2024/panel_charger_types_2017_2024.parquet` | Charging-category exposure panel | 36,328 |
| `output/native_bg_rf_replication/exact_replication_panel_2017_2023.csv` | Original 2017–2023 cross-sectional model support panel | 32,240 |
| `intermediate/analysis_data_2020bg/20_bg_2010_to_2020_interpolation_weights.parquet` | Historical BG interpolation weights | 11,966 |

The two 36,328-row analysis panels are **not** a balanced 36,328-row regression sample. Baseline eight-wave complete-case TWFE analyses use 2,903 CBGs × 8 years = 23,224 CBG-years. Do not mix up the full-panel and complete-case counts.

## Not included in the Zenodo package

- National Census block-group polygon shapefile, generally configured through `EVCS_BG_SHP` (the scripts default to `shapefiles/BG_shapefile_cb/cb_2024_us_bg_500k.shp` under the data root).
- CBSA geometry (`EVCS_CBSA_SHP`), Census source relationship downloads, and full raw ACS pulls.
- Original EPA Smart Location Database geodatabase and full raw historical AFDC exports.
- Every archived historical model intermediate and proprietary/local GIS project file.

The necessary upstream source files must be obtained independently for GIS reconstruction or full raw-data processing; the deposited tables and input panels can be used for analyses that do not require such reconstruction. Consult `docs/REPRODUCIBILITY.md` and the Zenodo file `SOURCE_AND_REPRODUCIBILITY_NOTES.md`.

## License and privacy

Use and attribution remain subject to the terms of each upstream data provider. The historical AFDC file is a deliberately reduced, selected-field export, not a verbatim release of the full station database. Do not include private credentials or machine-specific source exports in commits.
