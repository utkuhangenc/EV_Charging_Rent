# 2017–2024 GitHub publication inventory

The files below are **required candidates**, not a claim that the present archive contains or has tested them. Collect the final versions that produced the reported manuscript and SI tables. Do not substitute older drafts simply because their filenames match.

## A. Code (collect from Gautschi)

Likely location: `/depot/john1808/Utku/projects/ev_charging_rent/code/`.

- [ ] Final ACS downloads/harmonization and AFDC data-preparation scripts/notebooks (audit for embedded API keys)
- [ ] Final modified Poisson and random forest 2017–2024 scripts/notebooks and plotting scripts
- [ ] `rebuild_evcs_panel_2017_2024_b25070_diagnostic.py` (main balanced TWFE + grid)
- [ ] `evcs_24h_access_twfe_2017_2024.py` (reported-hour charging categories)
- [ ] `evcs_charger_type_twfe_2017_2024.py` (only if its results were used)
- [ ] `two_period_twfe_evcs_revision_2019_2024.py` (nonoverlap)
- [ ] `evcs_revision_remaining_analyses.py` (historical, ACS MOE, jurisdiction, sample selection)
- [ ] `evcs_tract_sensitivity_2017_2024.py` (tract sensitivity)
- [ ] `export_figure2_arcgis_2017_2024.py` (map exports)
- [ ] Every imported helper file actually used by these scripts
- [ ] Full `requirements.txt` / recorded environment versions

## B. Candidate data files (do not auto-publish without checking size, licensing, and contents)

- [ ] `/depot/john1808/Utku/data/ev_charging/output/native_bg_rf_replication/panel_twfe_2017_2024/analysis_panel_2017_2024.parquet`
- [ ] `/depot/john1808/Utku/data/ev_charging/intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet`
- [ ] `/depot/john1808/Utku/data/ev_charging/intermediate/evcs/afdc_ev_master_with_geography.parquet`
- [ ] `/depot/john1808/Utku/EV_Charging/Analysis_Data/04_acs_harmonized_bg_2013_2024.parquet` (or moved current path)
- [ ] `/depot/john1808/Utku/EV_Charging/Analysis_Data/05_evcs_access_harmonized_bg_2014_2025.parquet` (or moved current path)
- [ ] Tract reconstruction outputs and geographic source/crosswalk instructions

## C. Final outputs and documentation

- [ ] Main tables 1–3 and SI.1–SI.13 final CSVs
- [ ] Figures 1–4 and SI.1–SI.4 final outputs / plotted data
- [ ] Record counts and checksums for final analysis inputs
- [ ] Validated output-to-script crosswalk (main/SI table to script and file)
- [ ] Relative/configurable paths rather than hard-coded Gautschi-only paths
- [ ] updated Zenodo data/version and access checks
- [ ] check README, manuscript links, and regression results against generated outputs

**Caution:** Earlier work documents a historical monthly-rent/annual-income ratio and notes that earlier RF geospatial buffers might have used a different CRS. Preserve the actual executed final code and explicitly validate method consistency before release. The historical HBG selection analysis uses a different reference-point exposure definition than the main CBG polygon-buffer TWFE.
