#!/usr/bin/env python3
"""ArcGIS-ready 2017/2024 housing/EVCS mapping dataset for the two full CBSAs.

Inputs are the SAME harmonized ACS/AFDC/geography sources as the final TWFE
analysis. Output GPKGs have 2020 Census block-group polygon geometries.

Run on Gautschi:
    python code/export_figure2_arcgis_2017_2024.py

Requires packages: pandas, numpy, pyarrow, geopandas, shapely, pyogrio.
Requires source scripts in CODE directory:
    evcs_24h_access_twfe_2017_2024.py
    rebuild_evcs_panel_2017_2024_b25070_diagnostic.py

The entire CBSA geography is mapped (based on its constituent counties), not
only the 2,903 complete-case panel CBGs. Missing ACS outcome data remain NULL;
charging counts are 0 where there are no observed public AFDC stations nearby.
"""
from __future__ import annotations

import sys
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = DATA_ROOT
CODE = CODE_DIR
ACS_FILE = ROOT / 'intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet'
AFDC_FILE = ROOT / 'intermediate/evcs/afdc_ev_master_with_geography.parquet'
BG_FILE = BG_SHP
ENRICHED_FILE = (ROOT / 'output/native_bg_rf_replication/panel_twfe_2017_2024'
                 / 'evcs_24h_access_twfe_2017_2024/panel_charger_types_2017_2024.parquet')
OUT = ROOT / 'output/native_bg_rf_replication/figure2_arcgis_2017_2024'
YEARS = (2017, 2024)
EPSG_METERS = 'EPSG:5070'  # Same distance-accurate continental CRS as final exposure creation.
BUFFER_METERS = 1609.344
METROS = {
    'Philadelphia': 'Philadelphia-Camden-Wilmington',
    'DMV': 'Washington-Arlington-Alexandria',
}
# Approximate local State Plane projections for ArcGIS display; each CBSA
# crosses multiple state boundaries, so no one State Plane zone fits every state.
LOCAL_CRS = {'Philadelphia': 'EPSG:32129', # NAD83 / Pennsylvania South
             'DMV': 'EPSG:26985'}        # NAD83 / Maryland


def clean_geoid(col):
    return (col.astype('string').str.replace('1500000US', '', regex=False)
            .str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(12))


def require(paths):
    absent = [str(p) for p in paths if not Path(p).exists()]
    if absent:
        raise FileNotFoundError('Input missing on Gautschi:\n' + '\n'.join(absent))


def nonnegative_numeric(col):
    s = pd.to_numeric(col, errors='coerce')
    return s.where(s >= 0)


def make_outcome_long(path):
    acs = pd.read_parquet(path)
    needed = ['bg_geoid_2020', 'acs_year', 'study_metro',
              'median_gross_rent_approx', 'median_household_income_approx',
              'cash_renters_total', 'rent_burden_30_n', 'rent_burden_50_n']
    missing = set(needed) - set(acs.columns)
    if missing:
        raise KeyError(f'ACS parquet missing fields: {sorted(missing)}')
    acs['bg'] = clean_geoid(acs['bg_geoid_2020'])
    acs['year'] = pd.to_numeric(acs['acs_year'], errors='coerce')
    acs = acs[acs.year.isin(YEARS) & acs.study_metro.isin(METROS.values())].copy()
    if acs.duplicated(['bg', 'year']).any():
        raise ValueError('Repeated block group/year records in harmonized ACS')
    acs['metro'] = acs.study_metro.map({v:k for k,v in METROS.items()})
    acs['rent'] = nonnegative_numeric(acs.median_gross_rent_approx).where(
        lambda s: s > 0)
    acs['income'] = nonnegative_numeric(acs.median_household_income_approx).where(
        lambda s: s > 0)
    acs['renters'] = nonnegative_numeric(acs.cash_renters_total)
    denom = acs['renters'].where(acs['renters'] > 0)
    for threshold in (30, 50):
        numerator = nonnegative_numeric(acs[f'rent_burden_{threshold}_n'])
        # Denominator and numerator match the published panel calculation.
        share = 100 * numerator / denom
        acs[f'b{threshold}'] = share.where(share.between(0, 100.00001))
    return acs[['bg', 'year', 'metro', 'rent', 'income', 'renters', 'b30', 'b50']]


def geography_from_cbsa_counties(outcome, bg_path):
    """Use the harmonized panel's full CBSA county roster, not complete cases."""
    roster = outcome[['bg', 'metro']].drop_duplicates()
    if roster.groupby('bg').metro.nunique().gt(1).any():
        raise ValueError('Same block group assigned to both metro regions')
    roster['county_fips'] = roster.bg.str[:5]
    counties = roster[['county_fips', 'metro']].drop_duplicates()
    if counties.groupby('county_fips').metro.nunique().gt(1).any():
        raise ValueError('CBSA county appears in both metros')
    counties.sort_values(['metro','county_fips']).to_csv(OUT/'cbsa_counties_used.csv', index=False)
    print('CBSA counties derived from harmonized panel:')
    print(counties.groupby('metro').size().to_string())
    bg = gpd.read_file(bg_path)
    key = 'GEOID' if 'GEOID' in bg.columns else 'GEOID20'
    if key not in bg:
        raise KeyError('CBG file requires GEOID or GEOID20')
    bg['bg'] = clean_geoid(bg[key])
    bg = bg[['bg', 'geometry']].copy()
    bg['county_fips'] = bg.bg.str[:5]
    bg = bg.merge(counties, how='inner', on='county_fips', validate='many_to_one')
    bg = bg.drop(columns=['county_fips'])
    if bg.bg.duplicated().any():
        raise ValueError('Duplicate CBG geometry')
    if bg.crs is None:
        raise ValueError('Census geography has no declared coordinate reference system')
    if bg.empty:
        raise ValueError('No block groups match county roster')
    print('Full CBSA map block groups:', bg.groupby('metro').size().to_dict())
    return bg.to_crs(EPSG_METERS)


def add_housing_columns(bg, outcome):
    src = outcome.copy()
    wide = []
    for yr in YEARS:
        one = src[src.year.eq(yr)].copy()
        one = one[['bg', 'metro', 'rent', 'income', 'renters', 'b30', 'b50']]
        one = one.drop(columns='metro').rename(columns={
            'rent': f'rent{yr % 100}',
            'income': f'inc{yr % 100}',
            'renters': f'renters{yr % 100}',
            'b30': f'b30_{yr % 100}',
            'b50': f'b50_{yr % 100}',
        })
        wide.append(one)
    out = bg.copy()
    for one in wide:
        out = out.merge(one, how='left', on='bg', validate='one_to_one')
    out['rent_diff'] = out.rent24 - out.rent17
    out['rent_pct'] = 100 * out.rent_diff / out.rent17.where(out.rent17.gt(0))
    out['inc_diff'] = out.inc24 - out.inc17
    out['b30_diff'] = out.b30_24 - out.b30_17  # percentage points
    out['b50_diff'] = out.b50_24 - out.b50_17  # percentage points
    out['rent_status'] = np.select(
        [out.rent17.isna() & out.rent24.isna(), out.rent17.isna(), out.rent24.isna()],
        ['Missing both years','Missing 2017','Missing 2024'], default='Observed both years')
    out['burden_status'] = np.select(
        [out.b50_17.isna() & out.b50_24.isna(), out.b50_17.isna(), out.b50_24.isna()],
        ['Missing both years','Missing 2017','Missing 2024'], default='Observed both years')
    # Separate status from observed zero, especially in renter-sparse suburbs.
    out['zero_renters17'] = out.renters17.eq(0).fillna(False).astype('int8')
    out['zero_renters24'] = out.renters24.eq(0).fillna(False).astype('int8')
    return out


def compute_station_counts(bg, afdc_path):
    """Match the final paper's public/24h/fast EVCS station-buffer definition."""
    sys.path.insert(0, str(CODE))
    try:
        from evcs_24h_access_twfe_2017_2024 import prepare_stations
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('Place the script in the project CODE folder next to '
                                  'evcs_24h_access_twfe_2017_2024.py') from exc
    st = prepare_stations(afdc_path, OUT)
    st = st[st.snapshot_year.isin(YEARS)].copy()
    st = gpd.GeoDataFrame(st,
                          geometry=gpd.points_from_xy(st.longitude, st.latitude),
                          crs='EPSG:4326').to_crs(EPSG_METERS)
    bg_geo = bg[['bg','geometry']]
    station_points = []
    exposure_parts = []
    for yr in YEARS:
        yrst = st[st.snapshot_year.eq(yr)].copy()
        yrst['is_24h'] = yrst['is_24h'].fillna(False).astype(bool)
        yrst['is_24h_fast'] = yrst['is_24h_fast'].fillna(False).astype(bool)
        stations_for_map = yrst[['station_id', 'snapshot_year', 'study_metro',
                                 'is_24h', 'is_24h_fast', 'geometry']].copy()
        stations_for_map['station_id'] = stations_for_map.station_id.astype(str)
        station_points.append(stations_for_map)
        buf = yrst[['station_id','is_24h','is_24h_fast','geometry']].copy()
        buf['geometry'] = buf.geometry.buffer(BUFFER_METERS)
        matching = gpd.sjoin(bg_geo, buf, how='inner', predicate='intersects')
        o = pd.DataFrame({'bg': bg.bg.to_numpy()})
        for label, condition in [
            ('public', pd.Series(True,index=matching.index)),
            ('pub24h', matching['is_24h']),
            ('fast24h', matching['is_24h_fast']),
        ]:
            matched = matching.loc[condition.fillna(False)]
            cnt = matched.groupby('bg').station_id.nunique()
            o[f'{label}{yr % 100}'] = o.bg.map(cnt).fillna(0).astype('int32')
        exposure_parts.append(o)
        print(f'{yr}: matched {len(yrst):,} AFDC public stations to all {len(bg):,} CBSA block groups')
    return exposure_parts, gpd.GeoDataFrame(pd.concat(station_points, ignore_index=True),
                                            geometry='geometry', crs=EPSG_METERS)


def add_access_columns(bg, annual_exposures):
    df = bg.copy()
    for item in annual_exposures:
        df = df.merge(item, how='left', on='bg', validate='one_to_one')
    if df[['public17','public24','pub24h17','pub24h24','fast24h17','fast24h24']].isna().any().any():
        raise AssertionError('Station counts should be known (0 or more) for ALL map CBGs')
    for type_ in ('public','pub24h','fast24h'):
        df[f'{type_}_diff'] = df[f'{type_}24'] - df[f'{type_}17']
        p = df[f'{type_}17'] > 0
        q = df[f'{type_}24'] > 0
        # 0 = no stations at both dates; 1=gained; 2=both; 3=lost.
        df[f'{type_}_tr'] = np.select(
            [~p & ~q, ~p & q, p & q, p & ~q],
            [0, 1, 2, 3], default=-1).astype('int8')
    df['access_class'] = df.public_tr.map({
        0: 'No charging station (both years)',
        1: 'Gained access by 2024',
        2: 'Access in both years',
        3: 'Lost access by 2024',
    })
    return df


def add_model_sample_flag(df, enriched_path):
    """Clearly distinguish map universe from 2,903-CBG balanced model sample.

    Also verify our reconstructed map station counts match the final model's
    exposure counts for all matching endpoint observations.
    """
    enriched = pd.read_parquet(enriched_path)
    if 'bg_geoid_2020' in enriched:
        enriched['bg'] = clean_geoid(enriched.bg_geoid_2020)
    else:
        enriched['bg'] = clean_geoid(enriched.GEO_ID)
    enriched['Year'] = pd.to_numeric(enriched.Year, errors='raise').astype(int)
    if enriched.duplicated(['bg','Year']).any():
        raise ValueError('Repeated block group / year in model exposure panel')
    full_years = set(range(2017, 2025))
    complete = enriched.groupby('bg').Year.agg(lambda v: set(v) == full_years)
    model_bgs = set(complete.index[complete])
    df['model_bg'] = df.bg.isin(model_bgs).astype('int8')
    if len(model_bgs) != 2903:
        raise AssertionError(f'Final study is 2,903 balanced BGs, but panel has {len(model_bgs)}')
    for yr in YEARS:
        compare = enriched[enriched.Year.eq(yr)].copy()
        cols = {'PointCount_1mi':'public',
                'PointCount_Open24hPublic_1mi':'pub24h',
                'PointCount_Open24hFast_1mi':'fast24h'}
        if missing := set(cols) - set(compare.columns):
            raise KeyError(f'Final panel missing exposure columns: {sorted(missing)}')
        compare = compare[['bg',*cols]].merge(
            df[['bg',*(f'{name}{yr%100}' for name in cols.values())]],
            on='bg',how='left',validate='one_to_one')
        for existing, label in cols.items():
            mismatched = (pd.to_numeric(compare[existing],errors='coerce')
                          != pd.to_numeric(compare[f'{label}{yr%100}'],errors='coerce'))
            if mismatched.any():
                path = OUT/f'ERROR_station_count_mismatch_{yr}_{label}.csv'
                compare.loc[mismatched].to_csv(path,index=False)
                raise AssertionError(f'{mismatched.sum()} 1-mi {label} count mismatches '
                                     f'against regression panel; see {path}')
    return df


def classify_rent_change(df):
    # Same pooled cutpoints for BOTH metropolitan areas, not within-metro quintiles.
    values = df.rent_diff.dropna()
    if values.size < 10:
        raise ValueError('Too few valid 2017-2024 rent changes to classify')
    qs = np.quantile(values, [0, .2, .4, .6, .8, 1])
    breaks = np.unique(qs)
    if breaks.size < 3:
        raise ValueError('Too few distinct rent change quantile breakpoints')
    df['rent_class'] = pd.cut(df.rent_diff, bins=breaks,
                              include_lowest=True,labels=False).add(1).astype('Int64')
    # More stable than relative-change quintiles where original rents are small.
    pd.DataFrame({'class': range(1,len(breaks)),
                  'lower_dollars': breaks[:-1], 'upper_dollars': breaks[1:]})\
      .to_csv(OUT/'pooled_rent_change_class_breaks.csv', index=False)
    # Common and straightforward across metro/years; 100% is included.
    for col in ('b30_17','b30_24','b50_17','b50_24'):
        df[f'{col}_class'] = pd.cut(
            df[col], bins=[-1e-8,20,40,60,80,100.00001],labels=[1,2,3,4,5],
            right=False).astype('Int64')
    return df


def export_files(data, station_points, export_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    combined = OUT/'Figure2_all_CBGs_EPSG5070.gpkg'
    if not export_only:
        data.to_file(combined, layer='CBG_2017_2024',driver='GPKG')
        data.drop(columns='geometry').to_csv(OUT/'Figure2_CBG_attributes.csv',index=False)
        station_points.to_file(OUT/'Figure2_AFDC_public_station_points_2017_2024.gpkg',
                               layer='Public_stations', driver='GPKG')
    stats = []
    for metro, sub in data.groupby('metro'):
        filename = OUT/f'Figure2_{metro}_StatePlane.gpkg'
        projected = sub.to_crs(LOCAL_CRS[metro])
        projected.to_file(filename,layer='block_groups',driver='GPKG')
        bound = projected[['metro','geometry']].dissolve(by='metro',as_index=False)
        # Fiona on some HPC installations fails when appending a second layer
        # to an existing GeoPackage; write the boundary to its own GeoPackage.
        outline_file = OUT / f'Figure2_{metro}_outline_StatePlane.gpkg'
        bound.to_file(outline_file, layer='metro_outline', driver='GPKG')
        stats.append(dict(metro=metro,full_CBSA_CBGs=len(sub),balanced_model_CBGs=int(sub.model_bg.sum()),
                          rent_change_available=int(sub.rent_diff.notna().sum()),
                          rent_change_missing=int(sub.rent_diff.isna().sum()),
                          severe_burden_2017_available=int(sub.b50_17.notna().sum()),
                          severe_burden_2024_available=int(sub.b50_24.notna().sum()),
                          zero_chargers_both_dates=int(sub.public_tr.eq(0).sum()),
                          gained_1mi_access=int(sub.public_tr.eq(1).sum()),
                          retained_1mi_access=int(sub.public_tr.eq(2).sum()),
                          lost_1mi_access=int(sub.public_tr.eq(3).sum())))
        print(f'{metro}: {len(sub):,} geographic BGs, {sub.model_bg.sum():,} balanced model BGs')
    pd.DataFrame(stats).to_csv(OUT/'map_coverage_QA.csv',index=False)
    (OUT/'ArcGIS_FIELD_GUIDE.txt').write_text(
        "FIGURE 2: GIS EXPORT, 2017-2024\n"
        "DATA: harmonized ACS 5-year estimates (2017, 2024); AFDC operating public station snapshots.\n"
        "GEOGRAPHY: 2020 Census block groups, complete CBSA COUNTY footprints.\n"
        "OUTLINE: Figure2_<metro>_outline_StatePlane.gpkg (separate from block groups).\n"
        "PROJECTION: all BG layer EPSG:5070; Philadelphia display EPSG:32129 "
        "(PA South); DMV display EPSG:26985 (Maryland). These CBSAs span several states.\n\n"
        "rent17,rent24: harmonized ACS median gross monthly rent ($).\n"
        "rent_diff = rent24 - rent17 (nominal $); rent_pct = percent change from 2017.\n"
        "rent_class: pooled rent_diff quantile class (1=smallest, up to 5=largest); "
        "see pooled_rent_change_class_breaks.csv. NULL = missing rent in either year.\n"
        "b30_17,b30_24: percent of renter households spending >=30% income on rent.\n"
        "b50_17,b50_24: percent spending >=50%; b30_diff,b50_diff percentage-point changes.\n"
        "b50_17_class,b50_24_class: [0,20),[20,40),[40,60),[60,80),[80,100]%.\n"
        "renters17,renters24: ACS renter denominator (zero implies burden N/A, not 0%).\n"
        "public17,public24: counts of operating public AFDC stations within 1 mile.\n"
        "pub24h17,pub24h24: 24h-public station counts; fast24h17,fast24h24: 24h DC-fast counts.\n"
        "*_diff: count change; *_tr: 0=no access both years, 1=gained, 2=access both, 3=lost.\n"
        "access_class: explicit human-readable public access change; no stations != missing ACS.\n"
        "model_bg: 1=balanced 2017-2024 regression sample; 0=outside regression sample.\n"
        "rent_status,burden_status: reasons outcome maps contain null values.\n\n"
        "PREFERRED MAIN MAPS: classified rent_diff (no data visibly shaded), b50_24 "
        "or b50_diff (no data visibly shaded), and access_class (no stations explicitly "
        "labelled). Share each map's legend across both metros.\n"
        "Alternative density map: public_diff; optional pub24h_diff or fast24h_diff.\n"
        "DO NOT present access_class as a 3x3 bivariate map.\n"
        "CAUTION: station-buffer intersects CBG polygon, not road-network travel distance.\n"
        "24-hour availability is AFDC-reported; known dealerships are NOT universally "
        "excluded from these categories.\n"
        "Endpoint ACS 5-year windows are 2013-2017 and 2020-2024, so they do not overlap.\n"
        "Comparison is descriptive and does not by itself establish gentrification or displacement.\n",
        encoding='utf-8')
    print('\nExported ArcGIS-ready layers to:', OUT)
    print('Validation summary:')
    print(pd.DataFrame(stats).to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--export-only' in sys.argv[1:]:
        # Recover all unfinished metro outputs from the GeoPackages saved
        # before the original Fiona append error; no ACS/AFDC work is repeated.
        combined = OUT / 'Figure2_all_CBGs_EPSG5070.gpkg'
        points = OUT / 'Figure2_AFDC_public_station_points_2017_2024.gpkg'
        require([combined, points])
        data = gpd.read_file(combined, layer='CBG_2017_2024')
        stations = gpd.read_file(points, layer='Public_stations')
        print(f'Reusing {len(data):,} saved block groups and '
              f'{len(stations):,} saved endpoint station records; '
              'skipping ACS/AFDC matching.')
        export_files(data, stations, export_only=True)
        return
    require([ACS_FILE, AFDC_FILE, BG_FILE, ENRICHED_FILE,
             CODE/'evcs_24h_access_twfe_2017_2024.py',
             CODE/'rebuild_evcs_panel_2017_2024_b25070_diagnostic.py'])
    OUT.mkdir(parents=True,exist_ok=True)
    acs = make_outcome_long(ACS_FILE)
    bg = geography_from_cbsa_counties(acs,BG_FILE)
    mapped = add_housing_columns(bg,acs)
    counts, stations = compute_station_counts(bg, AFDC_FILE)
    mapped = add_access_columns(mapped,counts)
    mapped = add_model_sample_flag(mapped, ENRICHED_FILE)
    mapped = classify_rent_change(mapped)
    export_files(mapped,stations)


if __name__ == '__main__':
    main()
