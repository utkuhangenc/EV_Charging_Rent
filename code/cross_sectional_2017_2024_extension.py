#!/usr/bin/env python3
"""2024 EVCS replication extension: build 2024, run 3 models, aggregate 2017–2024, figure and Table 1.

Keeps 2017–2023 benchmark unchanged and retains EPSG:3857 buffers for comparability only.
"""

# Build 2024 panel using the EXACT same approach as the original 2017-2023 replication.
# Reuses the user's 2020-BG annual panel, AFDC master and Smart Location Database.
# NOTE: EPSG:3857 buffer retained ONLY for comparability; not a ground-distance-correct one-mile buffer.

from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# PATHS / SETTINGS
# ============================================================

ROOT = DATA_ROOT
SHAPE_ROOT = SHAPE_ROOT

ANALYSIS_DIR = ROOT / "intermediate/analysis_data_2020bg"
OUTPUT_DIR = ROOT / "output/native_bg_rf_replication"

ANNUAL_PANEL_PATH = (
    ANALYSIS_DIR
    / "23_analysis_2020bg_annual.parquet"
)

WEIGHTS_PATH = (
    ANALYSIS_DIR
    / "20_bg_2010_to_2020_interpolation_weights.parquet"
)

EV_PATH = (
    ROOT
    / "intermediate/evcs"
    / "afdc_ev_master_with_geography.parquet"
)

BG_PATH = (
    SHAPE_ROOT
    / "BG_shapefile_cb"
    / "cb_2024_us_bg_500k.shp"
)

SMART_LOCATION_GDB = (
    ROOT
    / "input/smartlocation"
    / "SmartLocationDatabase.gdb"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2024]

PROJECTED_CRS = "EPSG:3857"  # Historical R benchmark, NOT a distance-accurate CRS
BUFFER_METERS = 1609.344

METROS = [
    "Philadelphia-Camden-Wilmington",
    "Washington-Arlington-Alexandria",
]


# ============================================================
# HELPERS
# ============================================================

def require(*paths):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )


def clean_geoid(series, width=12):
    return (
        series.astype("string")
        .str.replace("1500000US", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(width)
    )


def choose(frame, candidates, required=True):
    lookup = {
        str(column).lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise KeyError(
            "Missing all candidate columns: "
            + ", ".join(candidates)
        )

    return None


def area_weighted_average(data, variable):
    values = pd.to_numeric(
        data[variable],
        errors="coerce",
    )

    area = pd.to_numeric(
        data["arealand_part"],
        errors="coerce",
    )

    valid = values.notna() & area.gt(0)

    work = data.loc[
        valid,
        ["bg_geoid_2020"]
    ].copy()

    work["numerator"] = (
        values.loc[valid]
        * area.loc[valid]
    )

    work["denominator"] = area.loc[valid]

    result = (
        work.groupby(
            "bg_geoid_2020",
            as_index=False,
        )
        .agg(
            numerator=("numerator", "sum"),
            denominator=("denominator", "sum"),
        )
    )

    result[variable] = (
        result["numerator"]
        / result["denominator"]
    )

    return result[
        ["bg_geoid_2020", variable]
    ]


# ============================================================
# CHECK INPUTS
# ============================================================

require(
    ANNUAL_PANEL_PATH,
    WEIGHTS_PATH,
    EV_PATH,
    BG_PATH,
    SMART_LOCATION_GDB,
)


# ============================================================
# LOAD ANALYSIS PANEL + 2020 BG GEOMETRY
# ============================================================

panel = pd.read_parquet(
    ANNUAL_PANEL_PATH
)

panel["bg_geoid_2020"] = clean_geoid(
    panel["bg_geoid_2020"]
)

panel = panel[
    panel["acs_year"].isin(YEARS)
    & panel["study_metro"].isin(METROS)
].copy()


bg = gpd.read_file(BG_PATH)

bg_geoid = choose(
    bg,
    ["GEOID", "GEOID20"],
)

bg = (
    bg[[bg_geoid, "geometry"]]
    .rename(
        columns={
            bg_geoid: "bg_geoid_2020"
        }
    )
)

bg["bg_geoid_2020"] = clean_geoid(
    bg["bg_geoid_2020"]
)

bg = bg[
    bg["bg_geoid_2020"].isin(
        panel["bg_geoid_2020"].unique()
    )
].copy()

bg = bg.to_crs(PROJECTED_CRS)

metro_lookup = (
    panel[
        ["bg_geoid_2020", "study_metro"]
    ]
    .drop_duplicates()
)

bg = bg.merge(
    metro_lookup,
    on="bg_geoid_2020",
    how="inner",
    validate="one_to_one",
)


# ============================================================
# SMART LOCATION VARIABLES
# ============================================================

try:
    layer_name = (
        gpd.list_layers(
            SMART_LOCATION_GDB
        )
        .iloc[0]["name"]
    )
except Exception:
    layer_name = None


smart = gpd.read_file(
    SMART_LOCATION_GDB,
    layer=layer_name,
    ignore_geometry=True,
)

smart_geoid = choose(
    smart,
    ["GEOID10", "GEOID"],
)

smart = smart.rename(
    columns={
        smart_geoid: "bg_geoid_2010"
    }
)

smart["bg_geoid_2010"] = clean_geoid(
    smart["bg_geoid_2010"]
)


smart_columns = {
    choose(smart, ["D1B"]):
        "PopDens",

    choose(smart, ["D2A_EPHHM"]):
        "EmployeeDiversity",

    choose(smart, ["D4A"]):
        "DistanceTransit",

    choose(
        smart,
        ["NatWalkInd", "NATWALKIND"]
    ):
        "Walkability",
}


smart = (
    smart[
        ["bg_geoid_2010"]
        + list(smart_columns)
    ]
    .rename(columns=smart_columns)
    .drop_duplicates("bg_geoid_2010")
)


SLD_VARIABLES = [
    "PopDens",
    "EmployeeDiversity",
    "DistanceTransit",
    "Walkability",
]

for column in SLD_VARIABLES:

    smart[column] = pd.to_numeric(
        smart[column],
        errors="coerce",
    )

    smart.loc[
        smart[column] < 0,
        column,
    ] = np.nan


# ============================================================
# CROSSWALK SMART LOCATION DATA TO 2020 BGs
# ============================================================

weights = pd.read_parquet(
    WEIGHTS_PATH
)

for column in [
    "bg_geoid_2010",
    "bg_geoid_2020",
]:
    weights[column] = clean_geoid(
        weights[column]
    )


smart_joined = (
    weights[
        [
            "bg_geoid_2010",
            "bg_geoid_2020",
            "arealand_part",
        ]
    ]
    .merge(
        smart,
        on="bg_geoid_2010",
        how="left",
        validate="many_to_one",
    )
)


smart_2020 = pd.DataFrame({
    "bg_geoid_2020":
        sorted(
            smart_joined[
                "bg_geoid_2020"
            ]
            .dropna()
            .unique()
        )
})


for column in SLD_VARIABLES:

    smart_2020 = smart_2020.merge(
        area_weighted_average(
            smart_joined,
            column,
        ),
        on="bg_geoid_2020",
        how="left",
        validate="one_to_one",
    )


# ============================================================
# EV CHARGING STATIONS
# ============================================================

ev = pd.read_parquet(EV_PATH)

# Use the year-end AFDC snapshot when the master stores multiple snapshots
# within a calendar year. This prevents arbitrary deduplication of multiple
# snapshots for the same station and year. If year-end coverage is incomplete,
# select the latest available dated snapshot in each year and report it.
if "snapshot_year" in ev.columns:
    ev["snapshot_year"] = pd.to_numeric(ev["snapshot_year"], errors="coerce")
if "is_year_end_snapshot" in ev.columns:
    flag = ev["is_year_end_snapshot"].astype("string").str.lower().eq("true")
    years_with_yearend = set(ev.loc[flag, "snapshot_year"].dropna().astype(int))
    if set(YEARS).issubset(years_with_yearend):
        ev = ev.loc[flag].copy()
        print("AFDC annual basis: December 31 snapshots for all model years")
    elif "snapshot_date" in ev.columns:
        ev["_snapshot_date"] = pd.to_datetime(ev["snapshot_date"], errors="coerce")
        latest = ev.groupby("snapshot_year")["_snapshot_date"].transform("max")
        ev = ev.loc[ev["_snapshot_date"].eq(latest)].copy()
        print("WARNING: some years lack year-end snapshots; using latest dated snapshot per year")
    else:
        raise ValueError(
            "Incomplete year-end flags and no snapshot_date to choose an annual snapshot"
        )
elif "snapshot_date" in ev.columns:
    ev["_snapshot_date"] = pd.to_datetime(ev["snapshot_date"], errors="coerce")
    if ev["_snapshot_date"].notna().any():
        latest = ev.groupby("snapshot_year")["_snapshot_date"].transform("max")
        ev = ev.loc[ev["_snapshot_date"].eq(latest)].copy()
        print("AFDC annual basis: latest dated snapshot per year")

print("EV rows after selecting annual snapshots:", len(ev))

ev_columns = {
    "station_id":
        choose(
            ev,
            ["station_id", "id", "ID"],
        ),

    "snapshot_year":
        choose(
            ev,
            ["snapshot_year"],
        ),

    "status_code":
        choose(
            ev,
            ["status_code", "Status.Code"],
        ),

    "access_code":
        choose(
            ev,
            ["access_code", "Access.Code"],
            required=False,
        ),

    "fuel_type":
        choose(
            ev,
            [
                "fuel_type_code",
                "Fuel.Type.Code",
            ],
            required=False,
        ),

    "latitude":
        choose(
            ev,
            ["latitude", "Latitude"],
        ),

    "longitude":
        choose(
            ev,
            ["longitude", "Longitude"],
        ),
}


stations = pd.DataFrame({
    "station_id":
        ev[
            ev_columns["station_id"]
        ].astype("string"),

    "Year":
        pd.to_numeric(
            ev[
                ev_columns["snapshot_year"]
            ],
            errors="coerce",
        ),

    "status_code":
        ev[
            ev_columns["status_code"]
        ]
        .astype("string")
        .str.upper()
        .str.strip(),

    "latitude":
        pd.to_numeric(
            ev[
                ev_columns["latitude"]
            ],
            errors="coerce",
        ),

    "longitude":
        pd.to_numeric(
            ev[
                ev_columns["longitude"]
            ],
            errors="coerce",
        ),
})


# The original R study excludes private stations. Apply the same requirement here.
# Refuse to proceed without a usable access_code instead of silently including private stations.
if ev_columns["access_code"] is None:
    raise KeyError(
        "AFDC master is missing access_code. Original R replication requires public stations."
    )
stations["access_code"] = (
    ev[ev_columns["access_code"]].astype("string").str.lower().str.strip()
)

if ev_columns["fuel_type"] is None:
    stations["fuel_type"] = "ELEC"

else:
    stations["fuel_type"] = (
        ev[
            ev_columns["fuel_type"]
        ]
        .astype("string")
        .str.upper()
        .str.strip()
    )


stations = (
    stations[
        stations["Year"].isin(YEARS)
        & stations["status_code"].eq("E")
        & stations["access_code"].eq("public")
        & stations["fuel_type"].eq("ELEC")
    ]
    .dropna(
        subset=[
            "station_id",
            "latitude",
            "longitude",
        ]
    )
    .copy()
)


stations["Year"] = (
    stations["Year"]
    .astype(int)
)


stations = (
    stations
    .sort_values(
        ["Year", "station_id"]
    )
    .drop_duplicates(
        ["Year", "station_id"],
        keep="last",
    )
)


stations_geo = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(
        stations["longitude"],
        stations["latitude"],
    ),
    crs="EPSG:4326",
).to_crs(PROJECTED_CRS)


# Retain stations within the two study metro boundaries.
metro_boundaries = (
    bg[
        ["study_metro", "geometry"]
    ]
    .dissolve(by="study_metro")
    .reset_index()
)


stations_geo = (
    gpd.sjoin(
        stations_geo,
        metro_boundaries,
        how="inner",
        predicate="within",
    )
    .drop(columns="index_right")
    .drop_duplicates(
        ["Year", "station_id"]
    )
)


# ============================================================
# ORIGINAL ONE-MILE POLYGON-BUFFER ACCESS MEASURE
# ============================================================

access_frames = []

for year in YEARS:

    stations_year = stations_geo.loc[
        stations_geo["Year"] == year,
        ["station_id", "geometry"],
    ].copy()

    station_buffers = (
        stations_year.copy()
    )

    station_buffers["geometry"] = (
        station_buffers.geometry.buffer(
            BUFFER_METERS
        )
    )


    intersections = gpd.sjoin(
        bg[
            [
                "bg_geoid_2020",
                "geometry",
            ]
        ],
        station_buffers,
        how="left",
        predicate="intersects",
    )


    counts = (
        intersections
        .groupby(
            "bg_geoid_2020"
        )["station_id"]
        .nunique()
        .rename("PointCount")
        .reset_index()
    )


    counts["Year"] = year

    counts["WithinBuffer"] = (
        counts["PointCount"] > 0
    ).astype(int)

    access_frames.append(counts)

    print(
        f"Access finished: {year}"
    )


access = pd.concat(
    access_frames,
    ignore_index=True,
)


# ============================================================
# RECREATE ORIGINAL ANALYSIS VARIABLES
# ============================================================

replication = (
    panel
    .merge(
        smart_2020,
        on="bg_geoid_2020",
        how="left",
        validate="many_to_one",
    )
    .merge(
        access,
        left_on=[
            "bg_geoid_2020",
            "acs_year",
        ],
        right_on=[
            "bg_geoid_2020",
            "Year",
        ],
        how="left",
        validate="one_to_one",
    )
)


replication = replication.assign(

    GEO_ID=(
        "1500000US"
        + replication["bg_geoid_2020"]
    ),

    TotalPopulation=
        replication["total_population"],

    MedianRent=
        replication[
            "median_gross_rent_approx"
        ],

    MedianHouseholdIncome=
        replication[
            "median_household_income_approx"
        ],

    black_per=
        replication["black_pct"],

    hispanic_per=
        replication["hispanic_pct"],

    asian_per=
        replication["asian_pct"],

    renter_per=
        replication["renter_pct"],

    hs_per=
        replication[
            "high_school_or_equivalent_pct"
        ],

    bachelor_per=
        replication[
            "bachelors_degree_pct"
        ],

    master_per=
        replication[
            "graduate_degree_pct"
        ],
)


replication["white_per"] = np.where(
    replication["race_population"] > 0,
    (
        100
        * replication["white_alone"]
        / replication["race_population"]
    ),
    np.nan,
)


# Preserve original benchmark definition.
replication["rent_afford"] = np.where(
    replication[
        "MedianHouseholdIncome"
    ] > 0,

    replication["MedianRent"]
    / replication[
        "MedianHouseholdIncome"
    ],

    np.nan,
)


# ============================================================
# ORIGINAL ANALYSIS SAMPLE
# ============================================================

columns = [
    "GEO_ID",
    "bg_geoid_2020",
    "study_metro",
    "state_city",
    "tract_geoid_2020",
    "Year",
    "WithinBuffer",
    "PointCount",
    "TotalPopulation",
    "MedianRent",
    "PopDens",
    "Walkability",
    "black_per",
    "hispanic_per",
    "white_per",
    "asian_per",
    "renter_per",
    "MedianHouseholdIncome",
    "EmployeeDiversity",
    "DistanceTransit",
    "hs_per",
    "bachelor_per",
    "master_per",
    "rent_afford",
    "rent_burden_30_pct",
]


replication = (
    replication[
        columns
    ]
    .loc[
        lambda x:
            x["MedianRent"].gt(0)
            & x[
                "MedianHouseholdIncome"
            ].gt(0)
            & x["PopDens"].ge(0)
            & x[
                "EmployeeDiversity"
            ].ge(0)
            & x[
                "DistanceTransit"
            ].ge(0)
            & x["Walkability"].ge(0)
    ]
    .copy()
)


# ============================================================
# SAVE
# ============================================================

replication.to_csv(
    OUTPUT_DIR
    / "exact_replication_panel_2024.csv",
    index=False,
)


map_output = (
    bg[
        [
            "bg_geoid_2020",
            "geometry",
        ]
    ]
    .merge(
        replication.loc[
            replication["Year"].isin(
                [2024]
            ),
            [
                "bg_geoid_2020",
                "study_metro",
                "Year",
                "MedianRent",
                "WithinBuffer",
                "PointCount",
            ],
        ],
        on="bg_geoid_2020",
        how="inner",
        validate="one_to_many",
    )
)


map_output.to_file(
    OUTPUT_DIR
    / "exact_replication_map_inputs_2024.gpkg",
    layer="bg_year",
    driver="GPKG",
)


quality = (
    replication
    .groupby(
        ["Year", "study_metro"],
        as_index=False,
    )
    .agg(
        block_groups=(
            "bg_geoid_2020",
            "nunique",
        ),

        access_share=(
            "WithinBuffer",
            "mean",
        ),

        median_station_count=(
            "PointCount",
            "median",
        ),
    )
)


quality.to_csv(
    OUTPUT_DIR
    / "exact_replication_panel_quality_2024.csv",
    index=False,
)


print(
    "\nSaved exact replication under:",
    OUTPUT_DIR,
)

print(
    quality.to_string(index=False)
)



# ============================================================
# RUN ONLY THE 2024 MODELS, REAGGREGATE ALL 8 YEARS AND PLOT
# ============================================================

import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD_PANEL = OUTPUT_DIR / "exact_replication_panel_2017_2023.csv"
NEW_PANEL = OUTPUT_DIR / "exact_replication_panel_2024.csv"
COMBINED_PANEL = OUTPUT_DIR / "exact_replication_panel_2017_2024.csv"
MODEL_SCRIPT = CODE_DIR / "02_run_original_models.py"
MODEL_OUTPUT = OUTPUT_DIR / "metro_original_models"
SUMMARY = MODEL_OUTPUT / "summary"
FIGURES = MODEL_OUTPUT / "figures"
FIGURES.mkdir(exist_ok=True, parents=True)

for path in (OLD_PANEL, NEW_PANEL, MODEL_SCRIPT):
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")

old = pd.read_csv(OLD_PANEL, dtype={"bg_geoid_2020":"string"}, low_memory=False)
new = pd.read_csv(NEW_PANEL, dtype={"bg_geoid_2020":"string"}, low_memory=False)
if not set(old.columns).issubset(new.columns):
    raise ValueError(f"2024 panel is missing earlier columns: {sorted(set(old.columns)-set(new.columns))}")
old["Year"] = pd.to_numeric(old["Year"], errors="raise").astype(int)
new["Year"] = pd.to_numeric(new["Year"], errors="raise").astype(int)
if set(old["Year"]) != set(range(2017, 2024)) or set(new["Year"]) != {2024}:
    raise ValueError("Expected original panel years 2017–2023 and new panel year 2024 only")
combined = pd.concat([old, new[old.columns]], ignore_index=True)
if combined.duplicated(["bg_geoid_2020", "study_metro", "Year"]).any():
    raise ValueError("Duplicate BG × metro × year rows in combined panel")
if not set(combined["study_metro"]) == set(METROS):
    raise ValueError("Both metropolitan areas should be present")
combined.to_csv(COMBINED_PANEL, index=False)
print("\nCombined panel by year and metro:")
print(combined.groupby(["Year", "study_metro"]).size().to_string())
print("Saved:", COMBINED_PANEL)

spec = importlib.util.spec_from_file_location("evcs_model_2024", MODEL_SCRIPT)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
model.YEARS = list(range(2017, 2025))  # 8-year aggregation validation
model.TARGETS = [(s,y) for s in model.SCOPES for y in model.YEARS]

expected_old = [(scope,year) for scope in model.SCOPES for year in range(2017,2024)]
missing_old = [f"{scope.lower()}/{year}_performance.csv" for scope,year in expected_old
               if not (MODEL_OUTPUT / scope.lower() / f"{year}_performance.csv").is_file()]
if missing_old:
    raise FileNotFoundError("Existing annual model outputs are missing; refusing to compute misleading 8-year averages:\n" + "\n".join(missing_old))

for scope in ("Pooled", "Philadelphia", "DMV"):
    print(f"\nRUNNING 2024 — {scope}", flush=True)
    model.run_one(combined, scope, 2024, MODEL_OUTPUT, trees=500, folds=10, workers=4)

model.aggregate(MODEL_OUTPUT)

performance = pd.read_csv(SUMMARY / "annual_auc_and_performance.csv")
expected = {(scope,year,method) for scope in model.SCOPES for year in range(2017,2025)
            for method in ("Random Forest", "Modified Poisson")}
actual = set(map(tuple, performance[["Scope","Year","Model"]].itertuples(index=False, name=None)))
if expected != actual:
    raise ValueError(f"Performance table does not have all 48 rows; missing: {sorted(expected-actual)}")

# Pooled AUC comparison with eight years
pooled = performance.loc[performance["Scope"].eq("Pooled")]
fig,ax = plt.subplots(figsize=(8.2,5))
for method, color, marker, ls in [
        ("Random Forest", "#0072B2", "o", "-"),
        ("Modified Poisson", "#D55E00", "s", "--")]:
    a = pooled.loc[pooled["Model"].eq(method)].sort_values("Year")
    ax.plot(a["Year"],a["AUC"],label=method,color=color,marker=marker,
            linestyle=ls,linewidth=2.2,markersize=6)
ax.set_xlabel("Year",fontsize=12)
ax.set_ylabel("Held-out AUC",fontsize=12)
ax.set_xticks(range(2017,2025))
ax.set_ylim(0.70,0.85)
ax.grid(alpha=0.15)
ax.legend(frameon=False)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(FIGURES / f"pooled_auc_comparison_2017_2024.{ext}",dpi=300,bbox_inches="tight")
plt.close(fig)

# Sort one manuscript-ready wide table by POOLED average importance rank.
imp = pd.read_csv(SUMMARY / "all_variable_importance_ranks.csv")
if not imp.groupby("Scope")["Years"].min().eq(8).all():
    raise ValueError("Variable-importance summaries are not all based on 8 years")
readable = {
    "master_per": "Percentage of Population 25+ with Master's Degree or Above",
    "hs_per": "Percentage of Population 25+ with High School Degree",
    "MedianRent": "Median Rent",
    "bachelor_per": "Percentage of Population 25+ with Bachelor's Degree",
    "EmployeeDiversity": "Employee Diversity",
    "PopDens": "Population Density",
    "renter_per": "Percentage of Renter Occupied Households",
    "asian_per": "Percentage of Population Identifying as Asian",
    "black_per": "Percentage of Population Identifying as Black",
    "MedianHouseholdIncome": "Median Household Income",
    "Walkability": "Walkability",
    "hispanic_per": "Percentage of Population Identifying as Hispanic",
    "DistanceTransit": "Distance to Transit",
    "TotalPopulation": "Total Population",
}

tables = []
for scope in ("Pooled", "DMV", "Philadelphia"):
    subset = imp.loc[imp["Scope"].eq(scope), ["Variable","AverageRank","RankStdDev"]].copy()
    subset.rename(columns={"AverageRank":f"{scope} Average Rank",
                           "RankStdDev":f"{scope} Rank Standard Deviation"},inplace=True)
    tables.append(subset)
wide = tables[0]
for t in tables[1:]:
    wide = wide.merge(t,on="Variable",validate="one_to_one")
wide.sort_values(["Pooled Average Rank","Variable"],inplace=True)
wide.insert(0,"Variable Label",wide["Variable"].map(readable).fillna(wide["Variable"]))
wide.drop(columns="Variable",inplace=True)
wide.reset_index(drop=True,inplace=True)
path_csv=SUMMARY / "table_1_pooled_dmv_philadelphia_2017_2024.csv"
path_tsv=SUMMARY / "table_1_pooled_dmv_philadelphia_2017_2024.tsv"
wide.round(2).to_csv(path_csv,index=False)
wide.round(2).to_csv(path_tsv,index=False,sep="\t")

# Separate already-created eight-year tables: summary/table_1_pooled.csv, table_1b_regional.csv
print("\nPOOLED 2024 RESULTS:")
print(pooled.loc[pooled["Year"].eq(2024),["Scope","Model","N","AUC","Accuracy"]].to_string(index=False))
print("\nVARIABLE IMPORTANCE — SORTED BY POOLED RANK (2017–2024)")
print(wide.round(2).to_string(index=False))
print("\nSaved figure:", FIGURES / "pooled_auc_comparison_2017_2024.png")
print("Saved table:",path_csv)
print("Saved tab-separated table:",path_tsv)
