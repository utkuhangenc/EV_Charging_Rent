#!/usr/bin/env python3
"""Audit expanded Philly/DMV block-group coverage against Census 2025 CBSA polygons.

Input panel may lack BGs with insufficient ACS data: the script distinguishes
unexpected out-of-CBSA units from missing-in-panel units and reports both.
"""
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import geopandas as gpd
import pandas as pd

ROOT = DATA_ROOT
SHAPES = SHAPE_ROOT
PANEL = ROOT / 'intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet'
BG = SHAPES / 'BG_shapefile_cb/cb_2024_us_bg_500k.shp'
CBSA = CBSA_SHP
OUTPUT = ROOT / 'output/native_bg_rf_replication/metro_boundary_audit.csv'
METROS = ['Philadelphia-Camden-Wilmington', 'Washington-Arlington-Alexandria']


def normalize_geoid(x):
    return x.astype('string').str.replace('1500000US', '', regex=False).str.replace(r'\.0$', '', regex=True).str.zfill(12)


def main():
    for p in [PANEL, BG, CBSA]:
        if not p.exists():
            raise FileNotFoundError(p)
    panel = pd.read_parquet(PANEL, columns=['bg_geoid_2020', 'study_metro', 'acs_year'])
    panel['bg_geoid_2020'] = normalize_geoid(panel['bg_geoid_2020'])
    panel = panel.loc[panel.study_metro.isin(METROS), ['bg_geoid_2020', 'study_metro']].drop_duplicates()
    if panel.duplicated('bg_geoid_2020').any():
        raise ValueError('A BG is assigned to multiple metropolitan regions')
    cbsa = gpd.read_file(CBSA)
    namecol = next((x for x in ['NAME', 'NAMELSAD'] if x in cbsa), None)
    if namecol is None:
        raise KeyError('Missing CBSA NAME or NAMELSAD field')
    cbsa['study_metro'] = pd.NA
    for metro in METROS:
        cbsa.loc[cbsa[namecol].astype(str).str.startswith(metro), 'study_metro'] = metro
    selected = cbsa.dropna(subset=['study_metro'])[['study_metro', 'geometry']].copy()
    if set(selected.study_metro) != set(METROS):
        raise ValueError(f'CBSA name lookup incomplete; names: {selected.study_metro.tolist()}')
    selected = selected.to_crs(5070)
    bg = gpd.read_file(BG)
    idcol = next((x for x in ['GEOID', 'GEOID20'] if x in bg), None)
    if idcol is None:
        raise KeyError('2020 BG shapefile missing GEOID')
    bg = bg[[idcol, 'geometry']].rename(columns={idcol:'bg_geoid_2020'})
    bg['bg_geoid_2020'] = normalize_geoid(bg['bg_geoid_2020'])
    bg = bg.to_crs(5070)
    # A representative point is guaranteed to fall within its BG polygon.
    bg['geometry'] = bg.geometry.representative_point()
    expected = gpd.sjoin(bg, selected, how='inner', predicate='within')
    expected = expected[['bg_geoid_2020', 'study_metro']].drop_duplicates()
    if expected.duplicated('bg_geoid_2020').any():
        raise ValueError('One BG maps to more than one CBSA')
    merged = expected.merge(panel, on='bg_geoid_2020', how='outer',
                            suffixes=['_cbsa','_panel'], indicator=True)
    problems = merged.loc[(merged['_merge'] == 'right_only') |
                          ((merged['_merge'] == 'both') &
                           (merged.study_metro_cbsa != merged.study_metro_panel))]
    rows = []
    for metro in METROS:
        ex = expected.loc[expected.study_metro.eq(metro), 'bg_geoid_2020']
        actual = panel.loc[panel.study_metro.eq(metro), 'bg_geoid_2020']
        rows.append({'study_metro': metro, 'expected_CBSA_BGs': len(ex),
                     'panel_BGs': len(actual), 'matching_BGs': len(set(ex) & set(actual)),
                     'missing_from_panel': len(set(ex) - set(actual)),
                     'outside_CBSA_or_wrong_metro': len(set(actual) - set(ex))})
    quality = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(OUTPUT, index=False)
    print(quality.to_string(index=False))
    print('Saved:', OUTPUT)
    if len(problems):
        raise ValueError(f'{len(problems)} panel BGs outside claimed CBSA or assigned to wrong CBSA')
    if quality.missing_from_panel.sum():
        print('Note: Missing CBSA BGs may reflect ACS data availability or upstream selection; investigate.')


if __name__ == '__main__':
    main()
