# EV-Charging Access, Neighborhood Characteristics, and Housing Costs

This repository contains the reproducible data-processing and statistical
analysis workflow supporting the manuscript on EV-charging access,
neighborhood characteristics, and housing-cost outcomes in the
Washington, DC–Maryland–Virginia region and the Philadelphia region.

The analysis covers **2017–2023** at the Census block-group level.

## Repository structure

```text
.
├── README.md
├── code/
│   ├── 00_project_setup.Rmd
│   ├── 01_build_panel_dataset.Rmd
│   ├── 02_ev_siting_models.Rmd
│   ├── 03_panel_twfe_models.Rmd
│   └── 04_tables_and_figures.Rmd
├── data/
│   ├── raw/
│   │   ├── acs/
│   │   ├── acs_rent_burden/
│   │   ├── census_boundaries/
│   │   │   ├── 2010/
│   │   │   └── 2020/
│   │   ├── ev_stations/
│   │   └── smart_location/
│   └── processed/
└── results/
    ├── figures/
    └── tables/
```

## Recommended execution order

1. `code/00_project_setup.Rmd`
2. `code/01_build_panel_dataset.Rmd`
3. `code/02_ev_siting_models.Rmd`
4. `code/03_panel_twfe_models.Rmd`
5. `code/04_tables_and_figures.Rmd`

All paths are repository-relative and are constructed with the `here`
package. Open the repository's `.Rproj` file, or set the repository root
as the working directory before knitting the files.

## Required R packages

The setup file checks for and installs the following packages:

`here`, `sf`, `dplyr`, `tidyr`, `purrr`, `readr`, `stringr`,
`ggplot2`, `caret`, `sandwich`, `lmtest`, `car`, `pROC`,
`randomForest`, `plm`, `tibble`, and `forcats`.

For a publication archive, recording package versions with `renv` is
recommended:

```r
install.packages("renv")
renv::init()
renv::snapshot()
```

## Datasets required to reproduce the study

### 1. Annual ACS analysis files, 2017–2023


```text
2017ACS-Data_DMV_Philly_Combined.csv
2018ACS-Data_DMV_Philly_Combined.csv
2019ACS-Data_DMV_Philly_Combined.csv
2020ACS-Data_DMV_Philly_Combined.csv
2021ACS-Data_DMV_Philly_Combined.csv
2022ACS-Data_DMV_Philly_Combined.csv
2023ACS-Data_DMV_Philly_Combined.csv
```

The workflow expects variables including:

- `GEO_ID`
- `NAME`
- `MedianRent`
- `MedianHouseholdIncome`
- `TotalPopulation`
- `WhitePopulation`
- `BlackPopulation`
- `AsianPopulation`
- `HispanicPopulation`
- `NonHispanicPopulation`
- `TotalStudents`
- `HighSchoolDiploma`
- `BachelorDegree`
- `MasterDegreeAndPlus`
- `TotalHH`
- `OwnerHH`
- `RenterHH`
- `HouseholdUnderPoverty`
- `HousingUnits`
- `VacantUnits`


### 2. ACS Table B25070 rent-burden files, 2017–2023



```text
ACSDT5Y2017.B25070-Data.csv
ACSDT5Y2018.B25070-Data.csv
ACSDT5Y2019.B25070-Data.csv
ACSDT5Y2020.B25070-Data.csv
ACSDT5Y2021.B25070-Data.csv
ACSDT5Y2022.B25070-Data.csv
ACSDT5Y2023.B25070-Data.csv
```

The workflow uses:

- `B25070_001E`: renter-occupied units paying cash rent
- `B25070_007E`: 30.0–34.9% of income
- `B25070_008E`: 35.0–39.9% of income
- `B25070_009E`: 40.0–49.9% of income
- `B25070_010E`: 50% or more of income


### 3. Annual EV-station files, 2017–2023


```text
dmv_ev_2017.csv
dmv_ev_2018.csv
dmv_ev_2019.csv
dmv_ev_2020.csv
dmv_ev_2021.csv
dmv_ev_2022.csv
dmv_ev_2023.csv

philadelphia_ev_2017.csv
philadelphia_ev_2018.csv
philadelphia_ev_2019.csv
philadelphia_ev_2020.csv
philadelphia_ev_2021.csv
philadelphia_ev_2022.csv
philadelphia_ev_2023.csv
```

Each file must contain:

- `EV.Network`
- `Latitude`
- `Longitude`
- `Open.Date`


### 4. 2010 Census block-group boundaries


```text
data/raw/census_boundaries/2010/
```

The workflow currently filters to:

- District of Columbia (`11`)
- Maryland (`24`)
- Pennsylvania (`42`)
- Virginia (`51`)



### 5. 2020 Census block-group boundaries


```text
data/raw/census_boundaries/2020/block_groups.shp
```


### 6. EPA Smart Location Database


```text
data/raw/smart_location/SmartLocationDatabase.gdb/
```

Required fields are:

- `GEOID10`
- `STATEFP`
- `D1B`
- `D2A_EPHHM`
- `D4A`
- `NatWalkInd`
- polygon geometry


## Primary processed dataset


```text
data/processed/analysis_panel_2017_2023.csv
```

The processed file should include:

- Census block-group identifier and year
- EV access indicators at 0.5, 1, and 2 miles
- EV station counts at 0.5, 1, and 2 miles
- demographic and socioeconomic percentages
- Smart Location variables
- median rent and household income
- rent-burden outcomes
- poverty and renter-household measures

## Analytical workflow

### Dataset construction

`01_build_panel_dataset.Rmd`:

1. Reads 2010 and 2020 block-group boundaries.
2. transfers 2019 Smart Location variables to 2020 geography using
   maximum area overlap;
3. creates 0.5-, 1-, and 2-mile EV exposure variables;
4. joins annual ACS and rent-burden data; and
5. writes the 2017–2023 panel.

Area and distance operations use **NAD83 / Conus Albers (EPSG:5070)**.

### EV-siting models

`02_ev_siting_models.Rmd` estimates annual:

- modified Poisson models with HC1 robust standard errors; and
- random-forest models with 10-fold cross-validation.

The default dependent variable is EV access within one mile. Change
`outcome` near the beginning of the file to analyze another distance.

### Housing-cost panel models

`03_panel_twfe_models.Rmd` estimates two-way fixed-effects models with:

- Census block-group fixed effects;
- year fixed effects;
- Census block-group-clustered standard errors;
- two-year leads and lags;
- EV exposure measures at 0.5, 1, and 2 miles; and
- specifications with and without median household income.

### Tables and figures

`04_tables_and_figures.Rmd` reads saved result tables and creates
publication-ready model-comparison and coefficient plots.

## Important variable-definition note

The variable `rent_afford` is retained as:

```text
monthly median gross rent / annual median household income
```

