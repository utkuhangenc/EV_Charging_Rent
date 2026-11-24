# Reproducibility guide (2017–2024)

This guide explains the **different levels** of reproducibility supported by the current repository and Zenodo dataset. It is intended to distinguish directly accessible results, data-based reruns, and geographic rebuilds requiring external source files.

## 1. Install, unpack, and configure

- Install Python, ideally **3.12** as used in the Gautschi preparation checks. Create an isolated environment and install `requirements.txt`.
- Download the **2017–2024** Zenodo dataset version and extract it **outside** this repository. Its `data/` folder is the value for `EVCS_DATA_ROOT`.
- Optionally provide paths to shapefiles using `EVCS_SHAPE_ROOT`, `EVCS_BG_SHP` and `EVCS_CBSA_SHP`. These are interpreted by `code/evcs_paths.py` when the Python process starts.
- For the original 2024 cross-sectional extension and GIS construction, obtain the appropriate Census block-group polygons and EPA Smart Location Database geodatabase independently. Follow the source notes in the Zenodo archive.

Example (Linux/macOS shell, executed from repository root):

```bash
python -m pip install -r requirements.txt
export EVCS_DATA_ROOT="/path/to/extracted/zenodo/data"
python verify_release.py
python code/two_period_twfe_evcs_2017_2024.py --self-test
python code/evcs_tract_sensitivity_2017_2024.py --stage self-test
```

`verify_release.py` checks Python syntax and screens for certain known hard-coded paths and credential patterns. It is **not** a security audit and does **not** test statistical results. The two self-tests use synthetic data; they do not validate the empirical estimates.

For a non-destructive inventory of the downloaded ACS and panel inputs:

```bash
python code/evcs_revision_remaining_analyses_v2.py --stage audit
```

**Output locations:** Analysis scripts generally write results under `$EVCS_DATA_ROOT/output/`. That directory is deliberately ignored by Git when `$EVCS_DATA_ROOT` is placed under the repository. Final reviewed outputs are separately committed under the repository's `results/` directory; new run outputs are not automatically publication-approved.

## 2. Which files correspond to manuscript exhibits?

| Exhibit or analysis | Existing release files | Relevant code (where available) |
|---|---|---|
| Main Figure 1, housing descriptives | `results/figures/main/Figure_1.*`, `results/figure_data/figure1/` | Descriptive plotting outputs retained in `results/`; full historical notebook plotting context is not packaged as one stand-alone reproduction script |
| Main Figure 2, 2017–2024 maps | `results/figures/main/Figure_2.*`, `results/figure_data/figure2/` | `code/export_figure2_arcgis_2017_2024.py`, `code/render_figure2_2017_2024.py` (GIS sources and map construction needed) |
| Main Figure 3, pooled annual AUC | `results/figures/main/Figure_3.*`, `results/model_outputs/cross_sectional_summary/` | `code/02_run_original_models.py` (historical 2017–2023), `code/cross_sectional_2017_2024_extension.py` (2024 extension) |
| Main Figure 4, two-year lead–lag coefficients | `results/figures/main/Figure_4.*`, `results/figure_data/figure4/` | `code/evcs_24h_access_twfe_2017_2024.py` and `code/evcs_full_robustness_grid_2017_2024.py` |
| Main Table 1, RF importance ranks | `results/main_tables/table1/` | Historical cross-sectional models and 2024 extension; Python and R random forests need not match numerically |
| Main Table 2, charging classes × TWFE | `results/main_tables/table2/` | `code/rebuild_evcs_panel_2017_2024_b25070_diagnostic.py`, `code/evcs_24h_access_twfe_2017_2024.py` |
| Main Table 3, 810-model robustness grid | `results/main_tables/table3/`, `results/model_outputs/leadlag/` | `code/evcs_full_robustness_grid_2017_2024.py` |
| Supplementary Tables SI.1–SI.13 | `results/supplemental_tables/si01/` through `si13/` | Cross-sectional, panel, `code/two_period_twfe_evcs_2017_2024.py`, `code/evcs_tract_sensitivity_2017_2024.py`, and `code/evcs_revision_remaining_analyses_v2.py` |

Consult `results/README.md` for *legacy filenames* that do not match final SI numbering (for example, `Table_SI_1_Modified_Poisson...` is published SI.3; `Table_SI_2_Random_Forest...` is SI.4).

## 3. Reruns from deposited analysis-ready data

These commands illustrate reruns from the deposited panel, provided the Python dependencies and input columns are compatible:

```bash
# Three nonoverlapping ACS comparisons used for supplementary Table SI.8.
python code/two_period_twfe_evcs_2017_2024.py \
    --pairs 2017-2024 2017-2023 2018-2024

# 2019–2024 ACS margin-of-error diagnostics (SI.9) from harmonized ACS data.
python code/evcs_revision_remaining_analyses_v2.py --stage precision

# Robustness grid from the deposited precomputed 24-hour charging panel;
# this requests 810 models and can be computationally expensive.
python code/evcs_full_robustness_grid_2017_2024.py --stage leadlag
```

These commands have **not** all been executed end to end from a fresh public Zenodo download. Review generated model diagnostics and compare estimates to the committed `results/` files rather than assuming byte-for-byte or numerical identity.

For the older cross-sectional models, `code/02_run_original_models.py` explicitly covers 2017–2023 (the 2024 extension is implemented separately). Some Python outputs approximate the original R model setup, but differing random number generation and random forest implementations can affect results.

## 4. GIS and source reconstruction require extra inputs

The following cannot be reconstructed solely from the eight deposited datasets:

- Station buffers against Census polygons and some tract-level geographic exposure calculations, when rebuilding from raw station/geographic inputs.
- The Figure 2 map geometry and full spatial source pipeline.
- Initial ACS extraction, 2010–2020 harmonization from raw Census downloads, and the full EPA Smart Location Database join.

Relevant scripts include `code/rebuild_evcs_panel_2017_2024_b25070_diagnostic.py`, `code/evcs_24h_access_twfe_2017_2024.py`, `code/evcs_tract_sensitivity_2017_2024.py`, and the Figure 2 scripts. Obtain the required block-group polygon/CBSA shapefiles and original EPA source independently and check CRS handling before running these steps.

**CRS distinction:** The earlier cross-sectional extension uses EPSG:3857 one-mile-buffer conventions for historical comparability, while final TWFE station-buffer construction uses EPSG:5070. The methods are not automatically interchangeable. This release does not assert that the two buffer definitions yield identical exposure values.

## 5. Published-result verification and known correction

- **Main Table 2:** pooled public 24-hour DC-fast proximity coefficient ×10⁴ = **1.355**. The earlier manuscript typo of 1.335 was identified and confirmed against the saved model result; do not replace the correctly stored CSV with the typo.
- **SI.5:** nine charging-category rows (both moderate and severe burden outcomes) and confidence intervals were verified against saved models and stored as `results/supplemental_tables/si05/Table_SI_5_verified_from_models.csv`.
- **SI.13:** the final pooled descriptive file contains 18 variables, stored as `results/supplemental_tables/si13/Table_SI_13_pooled_descriptive_statistics.csv`.
- **Sample:** complete-case TWFE reference is 2,903 CBGs × 8 years = 23,224 observations; broader analysis panel is 36,328 CBG-year rows.
- **Grid:** Table 3 summarizes 810 joint lead/current/lag models, producing 2,430 EV-access-term coefficients, with unadjusted p < 0.05 used for significance frequency summaries.

To validate file-level **integrity** of the published `results/` folder against its committed SHA-256 manifest, run this from the repository root (Python standard library only):

```python
from pathlib import Path
from hashlib import sha256
import csv

root = Path("results")
with (root / "RESULTS_MANIFEST.csv").open(newline="", encoding="utf-8") as fh:
    entries = list(csv.DictReader(fh))

errors = []
for row in entries:
    path = root / row["relative_path"]
    if not path.is_file():
        errors.append(f"Missing: {path}")
        continue
    if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        errors.append(f"Checksum mismatch: {path}")
print("PASS: all results hashes match" if not errors else "\n".join(errors))
```

The ZIP's own `FILE_MANIFEST.csv` provides SHA-256 hashes for the deposited **data**, separate from the repository's `RESULTS_MANIFEST.csv` (which covers **results**). Passing hashes confirms bytes, not scientific validity.

## 6. Scope of validation

On Gautschi, source-data counts and AFDC historical subsets were audited; selected pooled baseline TWFE fits, charging-hour classification, SI.9 and SI.11 results were independently retested. All nine SI.5 rows and all 18 SI.13 variables were subsequently checked, and the four main figures were visually compared with the revised manuscript. The result package was checked for ZIP and SHA-256 integrity.

The release is **not certified as a complete clean-environment end-to-end replication**. Additional work would be needed to lock all Python package versions, fully rebuild geospatial inputs from federal sources, rerun every cross-sectional model and 810 robustness specifications from the deposited data, and compare numerical outputs systematically to all published tables.
