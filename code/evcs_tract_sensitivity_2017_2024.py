#!/usr/bin/env python3
"""Tract-level sensitivity for EVCS/rental-affordability revision, 2017-2024.

Uses ALREADY harmonized ACS *block-group count estimates*; sums numerators and
B25070 denominators before computing tract burden shares. Does not average BG
percentages. This is an aggregate of harmonized BG estimates, not a direct ACS
tract extract. Tract identifiers = first 11 digits of 2020 BG GEOID.

Rebuilds the existing all-public station polygon-buffer exposure for every BG
in both study metros, then uses FIXED 2017 BG population weights to represent
tract residents' average BG exposure without mechanically treating a large
tract as having access everywhere when only its boundary is near a charger.

The model fits reuse the exact TWFE estimator and controls from the final BG
analysis (rebuild_evcs_panel_2017_2024_b25070_diagnostic.py).

Usage on Gautschi:
 python evcs_tract_sensitivity_2017_2024.py --stage audit
 python evcs_tract_sensitivity_2017_2024.py --stage build
 python evcs_tract_sensitivity_2017_2024.py --stage models
 python evcs_tract_sensitivity_2017_2024.py --stage all

No causal identification is claimed. ACS 5-year waves overlap in annual models.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = DATA_ROOT
ANNUAL = ROOT/'intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet'
EV_MASTER = ROOT/'intermediate/evcs/afdc_ev_master_with_geography.parquet'
BG_SHP = BG_SHP
MAIN_BG_PANEL = ROOT/'output/native_bg_rf_replication/panel_twfe_2017_2024/analysis_panel_2017_2024.parquet'
OUT = ROOT/'output/native_bg_rf_replication/reviewer_revision_checks_2026/tract_sensitivity'
YEARS = tuple(range(2017, 2025))
METROS = {'DMV':'Washington-Arlington-Alexandria',
          'Philadelphia':'Philadelphia-Camden-Wilmington'}
MEASURES = ('PointCount_1mi','SharePopulationWithAccess_1mi')
OUTCOMES = ('rent_burden_30p','rent_burden_50p')
REQUIRED_BG = ['bg_geoid_2020','acs_year','study_metro','cash_renters_total',
               'rent_burden_30_n','rent_burden_50_n','total_population',
               'race_population','black_alone','asian_alone','ethnicity_population',
               'hispanic_population','population_25plus','high_school_or_equivalent_n',
               'bachelors_degree_n','graduate_degree_n','occupied_housing_units',
               'renter_occupied_units','households_below_poverty',
               'median_household_income_approx']
CONTROL_COUNTS = ['total_population','race_population','black_alone','asian_alone',
                  'ethnicity_population','hispanic_population','population_25plus',
                  'high_school_or_equivalent_n','bachelors_degree_n',
                  'graduate_degree_n','occupied_housing_units','renter_occupied_units',
                  'households_below_poverty']
SUM_COUNTS = ['cash_renters_total','rent_burden_30_n','rent_burden_50_n']+CONTROL_COUNTS
CONTROL_NAMES = ['TotalPopulation','black_per','hispanic_per','asian_per','hs_per',
                 'bachelor_per','master_per','renter_per','poverty_hh_per']


def norm_bg(s):
    return (s.astype('string').str.replace('1500000US','',regex=False)
            .str.replace(r'\.0$', '',regex=True).str.strip().str.zfill(12))


def metro_short(s):
    mapping = {v:k for k,v in METROS.items()}
    mapping.update({'DMV':'DMV','Philadelphia':'Philadelphia'})
    return s.map(mapping)


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing {path}')


def check_acs_bg(acs):
    absent=set(REQUIRED_BG)-set(acs.columns)
    if absent: raise KeyError(f'Harmonized ACS panel missing columns: {sorted(absent)}')
    out=acs.copy()
    out['bg']=norm_bg(out.bg_geoid_2020)
    out['tract']=out.bg.str[:11]
    out['Year']=pd.to_numeric(out.acs_year,errors='coerce')
    out['metro']=metro_short(out.study_metro)
    out=out.loc[out.Year.isin(YEARS)&out.metro.notna()].copy()
    out['Year']=out.Year.astype(int)
    if out.duplicated(['bg','Year']).any():
        raise ValueError('Harmonized ACS contains duplicate BG-years')
    if out.groupby('bg').metro.nunique().gt(1).any():
        raise ValueError('Block group changes metro')
    for name in SUM_COUNTS+['median_household_income_approx']:
        out[name]=pd.to_numeric(out[name],errors='coerce').replace([np.inf,-np.inf],np.nan)
        # ACS suppressed counts should not become zeros in tract totals.
        out.loc[out[name]<0,name]=np.nan
    return out


def get_shapes(bg_ids):
    import geopandas as gpd
    if not BG_SHP.exists():raise FileNotFoundError(BG_SHP)
    raw=gpd.read_file(BG_SHP)
    col=next((c for c in ('GEOID','GEOID20') if c in raw.columns),None)
    if col is None: raise KeyError('Shapefile needs GEOID or GEOID20')
    geom=raw[[col,'geometry']].rename(columns={col:'bg'}).copy()
    geom['bg']=norm_bg(geom.bg)
    geom=geom.loc[geom.bg.isin(bg_ids)].copy()
    if geom.bg.duplicated().any():raise ValueError('Duplicate BG polygon GEOIDs')
    missing=set(bg_ids)-set(geom.bg)
    if missing:raise ValueError(f'{len(missing)} harmonized ACS BGs absent from BG geometry; examples {list(missing)[:5]}')
    if geom.crs is None: raise ValueError('Input BG shapefile lacks CRS')
    geom=geom.to_crs('EPSG:5070')
    geom['tract']=geom.bg.str[:11]
    if geom.geometry.isna().any() or geom.geometry.is_empty.any():
        raise ValueError('Null or empty BG geometry')
    return geom


def read_stations():
    import geopandas as gpd
    require_file(EV_MASTER)
    # Reuse the *same* snapshot selector as the final BG analysis.
    try:
        from rebuild_evcs_panel_2017_2024_b25070_diagnostic import select_annual_snapshots
    except ImportError as ex:
        raise ImportError('Place this script alongside rebuild_evcs_panel_2017_2024_b25070_diagnostic.py') from ex
    raw=pd.read_parquet(EV_MASTER)
    st=select_annual_snapshots(raw)
    cols=['snapshot_year','station_id','latitude','longitude',
          'status_code','access_code','fuel_type_code','study_metro']
    absent=set(cols)-set(st.columns)
    if absent:raise KeyError(f'AFDC missing {sorted(absent)}')
    for c in ('status_code','access_code','fuel_type_code'):
        st[c]=st[c].astype('string').str.strip().str.upper()
    st=st.loc[st.status_code.eq('E') & st.access_code.eq('PUBLIC') &
              st.fuel_type_code.eq('ELEC') & st.study_metro.isin(METROS.values())].copy()
    for c in ('latitude','longitude'):
        st[c]=pd.to_numeric(st[c],errors='coerce')
    st=st.dropna(subset=['station_id','latitude','longitude'])
    st=st.loc[st.latitude.between(-90,90)&st.longitude.between(-180,180)]
    st=st.drop_duplicates(['snapshot_year','station_id'])
    st=gpd.GeoDataFrame(st,geometry=gpd.points_from_xy(st.longitude,st.latitude),crs='EPSG:4326').to_crs('EPSG:5070')
    return st


def reconstruct_bg_exposure(geom, stations):
    import geopandas as gpd
    expected={2017:612,2024:2828}
    rows=[]
    for yr in YEARS:
        subset=stations.loc[stations.snapshot_year.eq(yr),['station_id','geometry']].copy()
        if yr in expected and len(subset)!=expected[yr]:
            warnings.warn(f'AFDC station count for {yr} is {len(subset)} vs prior {expected[yr]}: investigate before publication')
        buffers=subset.copy()
        buffers['geometry']=buffers.geometry.buffer(1609.344)
        join=gpd.sjoin(geom[['bg','geometry']],buffers,how='inner',predicate='intersects')
        counts=join.groupby('bg').station_id.nunique()
        t=pd.DataFrame({'bg':geom.bg.to_numpy(),'Year':yr})
        t['bg_count_1mi']=t.bg.map(counts).fillna(0).astype('int32')
        t['bg_access_1mi']=(t.bg_count_1mi>0).astype('int8')
        rows.append(t)
        print(f'  {yr}: {len(subset):,} stations; {int(t.bg_access_1mi.sum()):,} BGs with 1-mile access',flush=True)
    result=pd.concat(rows,ignore_index=True)
    if result.duplicated(['bg','Year']).any():raise ValueError('Duplicated exposure rows')
    return result


def verify_main_bg_exposures(bg_exposure):
    require_file(MAIN_BG_PANEL)
    old=pd.read_parquet(MAIN_BG_PANEL,columns=['bg_geoid_2020','Year','PointCount_1mi'])
    old['bg']=norm_bg(old.bg_geoid_2020)
    merged=old.merge(bg_exposure,on=['bg','Year'],how='left',validate='one_to_one')
    a=pd.to_numeric(merged.PointCount_1mi,errors='coerce')
    b=pd.to_numeric(merged.bg_count_1mi,errors='coerce')
    bad=a.isna()|b.isna()|a.ne(b)
    if bad.any():
        OUT.mkdir(parents=True,exist_ok=True)
        merged.loc[bad].to_csv(OUT/'FAILED_BG_exposure_check.csv',index=False)
        raise ValueError(f'{int(bad.sum())} BG-year station counts disagree with final main panel. STOP. Inspect FAILED_BG_exposure_check.csv')
    print(f'PASS: {len(merged):,} BG-year one-mile station counts match final TWFE panel exactly.')


def get_fixed_population_weights(acs):
    base=acs.loc[acs.Year.eq(2017),['bg','total_population']].copy()
    if base.bg.duplicated().any():raise ValueError('Duplicated 2017 population BG')
    base=base.rename(columns={'total_population':'pop17'})
    base['pop17']=base.pop17.where(base.pop17.gt(0))
    return base


def valid_burden_counts(acs):
    d=acs.copy()
    den=d.cash_renters_total
    n30=d.rent_burden_30_n
    n50=d.rent_burden_50_n
    tol=1e-8*np.maximum(1.0,den.abs())
    valid=(den.notna()&n30.notna()&n50.notna()&den.ge(0)&n30.ge(0)&n50.ge(0)&
           n30.le(den+tol)&n50.le(den+tol)&n50.le(n30+tol))
    print('B25070 BG-years valid:',int(valid.sum()),'/',len(valid))
    for c in ('cash_renters_total','rent_burden_30_n','rent_burden_50_n'):
        d.loc[~valid,c]=np.nan
    # Do not let minor harmonization rounding produce slightly > 100% shares.
    for c in ('rent_burden_30_n','rent_burden_50_n'):
        q=d[c].notna() & d[c].gt(d.cash_renters_total)
        d.loc[q,c]=d.loc[q,'cash_renters_total']
    d['bg_burden_valid']=valid.astype(int)
    return d


def aggregate_tracts(acs, geo, bg_exposure):
    expected_bg=geo[['bg','tract']].groupby('tract').size().rename('bg_geom_count')
    # Every BG-year appears once; a missing BG's contribution is reported,
    # never silently added as zero.
    acs=acs.merge(geo[['bg','tract']],on='bg',how='inner',suffixes=('','_geometry'),validate='many_to_one')
    if 'tract_geometry' in acs and acs.tract.ne(acs.tract_geometry).any():
        raise ValueError('Inconsistent tract GEOID / polygon GEOID')
    acs=acs.drop(columns=['tract_geometry'],errors='ignore')
    acs=valid_burden_counts(acs)
    acs=acs.merge(bg_exposure,on=['bg','Year'],how='left',validate='one_to_one')
    if acs.bg_count_1mi.isna().any(): raise ValueError('Missing reconstructed station counts')
    acs=acs.merge(get_fixed_population_weights(acs),on='bg',how='left',validate='many_to_one')

    # Exposure: population-weighted average of final-method BG station counts
    # and population share in BGs intersecting at least one station buffer.
    acs['bg_controls_valid']=acs[CONTROL_COUNTS].notna().all(axis=1).astype(int)
    acs['weighted_count']=acs.pop17*acs.bg_count_1mi
    acs['weighted_access']=acs.pop17*acs.bg_access_1mi
    acs['pop_weight_valid']=acs.pop17.where(acs.pop17.gt(0),0)

    # Household-weighted AVG of BG median income is NOT a tract median.
    # Included solely as a proxy control to check comparability with BG TWFE.
    household_weight=acs.occupied_housing_units.where(acs.occupied_housing_units.gt(0))
    proxy_ok=household_weight.notna() & acs.median_household_income_approx.gt(0)
    acs['income_weight']=household_weight.where(proxy_ok,0)
    acs['income_product']=(acs.median_household_income_approx*acs.income_weight).fillna(0)
    acs['income_weight_possible']=household_weight.fillna(0)

    # Count fields for controls must be complete and logically meaningful.
    # Recompute percentages AFTER summing, not as means of BG percentages.
    for c in CONTROL_COUNTS:
        acs[c]=acs[c].where(acs[c].ge(0))
    group=acs.groupby(['metro','tract','Year'],sort=True,observed=True)
    # min_count=1 ensures all-NA input yields NA, unlike default sum=0.
    sums=group[SUM_COUNTS].sum(min_count=1)
    audit=group.agg(n_bg_obs=('bg','nunique'),n_bg_burden_valid=('bg_burden_valid','sum'),
                    n_bg_controls_valid=('bg_controls_valid','sum'),
                    n_pop17=('pop17','count'),n_income_valid=('income_weight',lambda x:int(x.gt(0).sum())),
                    pop_weights=('pop_weight_valid','sum'),
                    weighted_count=('weighted_count','sum'),
                    weighted_access=('weighted_access','sum'),
                    income_weight=('income_weight','sum'),
                    income_product=('income_product','sum'),
                    income_weight_possible=('income_weight_possible','sum'))
    out=sums.join(audit).reset_index()
    out=out.merge(expected_bg.reset_index(),on='tract',how='left',validate='many_to_one')
    out['bg_coverage']=out.n_bg_obs/out.bg_geom_count
    out['burden_bg_coverage']=out.n_bg_burden_valid/out.bg_geom_count
    out['control_bg_coverage']=out.n_bg_controls_valid/out.bg_geom_count
    out['income_household_coverage']=out.income_weight/out.income_weight_possible.replace(0,np.nan)
    out['PointCount_1mi']=out.weighted_count/out.pop_weights.where(out.pop_weights.gt(0))
    out['SharePopulationWithAccess_1mi']=out.weighted_access/out.pop_weights.where(out.pop_weights.gt(0))
    out['MedianHouseholdIncome']=out.income_product/out.income_weight.where(out.income_weight.gt(0))
    out['TotalPopulation']=out.total_population

    def pct(n,d):
        return 100*n/d.where(d.gt(0))
    out['black_per']=pct(out.black_alone,out.race_population)
    out['asian_per']=pct(out.asian_alone,out.race_population)
    out['hispanic_per']=pct(out.hispanic_population,out.ethnicity_population)
    out['hs_per']=pct(out.high_school_or_equivalent_n,out.population_25plus)
    out['bachelor_per']=pct(out.bachelors_degree_n,out.population_25plus)
    out['master_per']=pct(out.graduate_degree_n,out.population_25plus)
    out['renter_per']=pct(out.renter_occupied_units,out.occupied_housing_units)
    out['poverty_hh_per']=pct(out.households_below_poverty,out.occupied_housing_units)
    out['rent_burden_30p']=pct(out.rent_burden_30_n,out.cash_renters_total)
    out['rent_burden_50p']=pct(out.rent_burden_50_n,out.cash_renters_total)

    valid_num=out.rent_burden_30_n.le(out.cash_renters_total + 1e-6)&out.rent_burden_50_n.le(out.rent_burden_30_n+1e-6)
    bad=out.cash_renters_total.gt(0)&(~valid_num | out.rent_burden_30p.gt(100.00001) | out.rent_burden_50p.gt(100.00001))
    if bad.any():raise ValueError(f'{int(bad.sum())} invalid tract burden sums: investigate input counts')
    if out.SharePopulationWithAccess_1mi.dropna().lt(0).any() or out.SharePopulationWithAccess_1mi.dropna().gt(1+1e-9).any():
        raise ValueError('Tract access share outside [0,1]')
    out['GEO_ID']='1400000US'+out.tract
    out['study_metro']=out.metro.map(METROS)
    out=out.rename(columns={'n_bg_obs':'n_bg_acs_year'})
    return out


def make_model_panel(all_tracts, coverage:float, income:bool):
    d=all_tracts.copy()
    d.loc[d.burden_bg_coverage.lt(coverage) | d.bg_coverage.lt(coverage),list(OUTCOMES)]=np.nan
    # Never treat missing BG control counts as zeros in a tract control.
    d.loc[d.control_bg_coverage.lt(coverage),CONTROL_NAMES]=np.nan
    if income:
        # For an included income control, require sufficient valid household weights.
        d.loc[d.income_household_coverage.lt(coverage),'MedianHouseholdIncome']=np.nan
    d=d.sort_values(['GEO_ID','Year']).copy()
    # Ensure lead/lag terms are actually two *calendar* years apart.
    g=d.groupby('GEO_ID',sort=False)
    yr_forward=g.Year.shift(-2)
    yr_backward=g.Year.shift(2)
    for col in MEASURES:
        d[f'{col}_lead2']=g[col].shift(-2).where(yr_forward.eq(d.Year+2))
        d[f'{col}_lag2']=g[col].shift(2).where(yr_backward.eq(d.Year-2))
    return d


def annual_models(panel, coverage, income, leadlag=False):
    try:
        from rebuild_evcs_panel_2017_2024_b25070_diagnostic import fit_twfe
    except ImportError as exc:
        raise ImportError('Place beside rebuild_evcs_panel_2017_2024_b25070_diagnostic.py') from exc
    timing='leadlag_2' if leadlag else 'baseline'
    rows=[]
    for scope in ('Pooled','DMV','Philadelphia'):
        sample=panel if scope=='Pooled' else panel.loc[panel.metro.eq(scope)]
        for outcome in OUTCOMES:
            for exposure in MEASURES:
                try:
                    estimates,diag=fit_twfe(sample,outcome,exposure,income,timing)
                    for r in estimates:
                        rows.append(dict(Scope=scope,Outcome=outcome,Exposure=exposure,
                                         Coverage=coverage,IncomeProxy=income,Timing=timing,
                                         term=r['term'],estimate=r['estimate'],std_error=r['std_error'],
                                         p_value=r['p_value'],conf_low=r['conf_low'],conf_high=r['conf_high'],
                                         N=diag['N'],Tracts=diag['N_groups'],Years=diag['N_years'],
                                         status='ok',removed_collinear=diag['removed_collinear']))
                except Exception as err:
                    rows.append(dict(Scope=scope,Outcome=outcome,Exposure=exposure,
                                     Coverage=coverage,IncomeProxy=income,Timing=timing,
                                     status='FAILED',error=str(err)))
    return pd.DataFrame(rows)


def first_difference_models(panel, coverage, income):
    # Two nonoverlapping ACS 5-year windows: 2015-2019 vs 2020-2024.
    # A single pair per tract is an end-to-end change; no assumed causal timing.
    controls=CONTROL_NAMES+(['MedianHouseholdIncome'] if income else [])
    a=panel.loc[panel.Year.eq(2019),['GEO_ID','metro',*OUTCOMES,*MEASURES,*controls]].copy()
    b=panel.loc[panel.Year.eq(2024),['GEO_ID','metro',*OUTCOMES,*MEASURES,*controls]].copy()
    pair=a.merge(b,on=['GEO_ID','metro'],how='inner',suffixes=('_19','_24'),validate='one_to_one')
    rows=[]
    for col in set(OUTCOMES+MEASURES+tuple(controls)):
        pair[f'd_{col}']=pair[f'{col}_24']-pair[f'{col}_19']
    for scope in ('Pooled','DMV','Philadelphia'):
        sub=pair if scope=='Pooled' else pair.loc[pair.metro.eq(scope)]
        for outcome in OUTCOMES:
            for exposure in MEASURES:
                xcols=[f'd_{exposure}']+[f'd_{c}' for c in controls]
                df=sub[['GEO_ID','metro',f'd_{outcome}',*xcols]].replace([np.inf,-np.inf],np.nan).dropna()
                row=dict(Scope=scope,Outcome=outcome,Exposure=exposure,Coverage=coverage,
                         IncomeProxy=income,Timing='2019-2024_first_difference',
                         N=len(df),Tracts=df.GEO_ID.nunique(),Years=2,status='ok')
                try:
                    if len(df)<max(50,5*(len(xcols)+2)):raise ValueError('Too few complete tract pairs')
                    X=sm.add_constant(df[xcols].astype(float),has_constant='add')
                    if scope=='Pooled':
                        # Distinct metro changes are allowed, analogous to metro-specific period trends.
                        X['Philly_period']=df.metro.eq('Philadelphia').astype(float).to_numpy()
                    if np.linalg.matrix_rank(X.to_numpy())<X.shape[1]:
                        raise ValueError('Rank-deficient first-difference controls')
                    fit=sm.OLS(df[f'd_{outcome}'].astype(float),X).fit(cov_type='HC1',use_t=True)
                    key=f'd_{exposure}'
                    ci=fit.conf_int().loc[key]
                    row.update(estimate=float(fit.params[key]),std_error=float(fit.bse[key]),
                               p_value=float(fit.pvalues[key]),conf_low=float(ci.iloc[0]),
                               conf_high=float(ci.iloc[1]))
                except Exception as err:
                    row.update(status='FAILED',error=str(err))
                rows.append(row)
    return pd.DataFrame(rows)


def save_coverage_summary(tracts):
    t=tracts.copy()
    t['complete_burden_100']=t.burden_bg_coverage.ge(1)
    t['complete_burden_80']=t.burden_bg_coverage.ge(.8)
    t['has_burden_outcome']=t.rent_burden_30p.notna()&t.rent_burden_50p.notna()
    table=t.groupby(['metro','Year']).agg(
        all_tracts=('tract','nunique'),
        n_all_bg_complete=('complete_burden_100','sum'),
        n_80pct_bg_complete=('complete_burden_80','sum'),
        n_raw_burden=('has_burden_outcome','sum'),
        n_control_bg_complete=('control_bg_coverage',lambda s:int(s.ge(1).sum())),
        n_with_renters=('cash_renters_total',lambda x:int(x.gt(0).sum())),
        median_bg_coverage=('burden_bg_coverage','median'),
        median_renter_households=('cash_renters_total','median')).reset_index()
    table.to_csv(OUT/'tract_coverage_by_metro_year.csv',index=False)
    print('\nACS tract coverage by metro and year:\n',table.to_string(index=False))
    return table


def build():
    for f in (ANNUAL,EV_MASTER,MAIN_BG_PANEL):require_file(f)
    if not BG_SHP.exists():raise FileNotFoundError(BG_SHP)
    OUT.mkdir(parents=True,exist_ok=True)
    acs=check_acs_bg(pd.read_parquet(ANNUAL))
    print(f'ACS: {len(acs):,} BG-years; {acs.bg.nunique():,} BGs; {acs.tract.nunique():,} tracts')
    geom=get_shapes(set(acs.bg))
    stations=read_stations()
    ex=reconstruct_bg_exposure(geom,stations)
    verify_main_bg_exposures(ex)
    ex.to_parquet(OUT/'all_public_BG_exposure_1mi_2017_2024.parquet',index=False)
    tracts=aggregate_tracts(acs,geom,ex)
    if tracts.duplicated(['GEO_ID','Year']).any():raise ValueError('Duplicate tract-year')
    tracts.to_parquet(OUT/'tract_panel_2017_2024.parquet',index=False)
    save_coverage_summary(tracts)
    print(f'\nSaved {len(tracts):,} tract-years / {tracts.GEO_ID.nunique():,} tracts at {OUT}')
    return tracts


def models():
    saved=OUT/'tract_panel_2017_2024.parquet'
    require_file(saved)
    tracts=pd.read_parquet(saved)
    results=[]
    for coverage in (1.,.8):
        for income in (True,False):
            panel=make_model_panel(tracts,coverage,income)
            # Annual TWFE baseline on ALL years; 2-yr joint leadlag on 2019-22.
            a=annual_models(panel,coverage,income,leadlag=False)
            results.append(a)
            print('Annual finished:',coverage,income, 'failed:',int(a.status.ne('ok').sum()),flush=True)
            # Lag sensitivity only in strict coverage models with income proxy.
            if coverage==1. and income:
                d=annual_models(panel,coverage,income,leadlag=True)
                results.append(d)
                print('Leadlag finished:',coverage,income,'failed:',int(d.status.ne('ok').sum()),flush=True)
            # Two-period comparison for both coverages, with income proxy.
            if income:
                f=first_difference_models(panel,coverage,income)
                results.append(f)
                print('Nonoverlapping ACS finished:',coverage,income,'failed:',int(f.status.ne('ok').sum()),flush=True)
    out=pd.concat(results,ignore_index=True)
    out.to_csv(OUT/'tract_TWFE_and_nonoverlap_estimates.csv',index=False)
    # Compact comparison table: all three scopes, two outcomes, two exposure metrics.
    subset=out.loc[out.status.eq('ok') & out.Timing.isin(['baseline','2019-2024_first_difference']) &
                   out.IncomeProxy.eq(True) & out.Coverage.eq(1.)].copy()
    if not subset.empty:
        subset['b_se_p']=subset.apply(lambda r:f"{r.estimate:.6g} (SE {r.std_error:.3g}, p={r.p_value:.3f})",axis=1)
        compact=subset[['Scope','Outcome','Exposure','Timing','Tracts','N','b_se_p']]
        compact.to_csv(OUT/'tract_primary_results_readable.csv',index=False)
        print('\nPrimary tract sensitivity results:\n'+compact.to_string(index=False))
    print('\nModel status by coverage, income, timing:\n',
          out.groupby(['Coverage','IncomeProxy','Timing','status'],dropna=False).size().to_string())
    print('\nSaved:',OUT/'tract_TWFE_and_nonoverlap_estimates.csv')
    return out


def audit():
    for label,path in [('Harmonized ACS',ANNUAL),('AFDC master',EV_MASTER),
                       ('Final BG TWFE',MAIN_BG_PANEL),('BG geometry',BG_SHP)]:
        print(label,':',path,'exists=',path.exists())
    for path,cols in [(ANNUAL,REQUIRED_BG),
                      (EV_MASTER,['snapshot_year','station_id','latitude','longitude','status_code','access_code','fuel_type_code','study_metro'])]:
        if path.exists():
            import pyarrow.parquet as pq
            available=set(pq.read_schema(path).names)
            missing=set(cols)-available
            print(f'{path.name}: columns={len(available)}, missing required={sorted(missing)}')
            if missing:raise KeyError(f'Missing required columns in {path}')
    print('Map geometry EPSG:5070; 1 mile = 1609.344 metres; 2020 tract GEOID = BG GEOID[:11].')
    print('IMPORTANT: rent burden is summed B25070 numerator / summed B25070 total renters,')
    print('not mean of BG burden percentages. Annual models use overlapping ACS 5yr releases.')


def self_test():
    # Minimal deterministic test for aggregation numerators/denominators and fixed weights.
    vals=[]
    for i in range(4):
        for yr in YEARS:
            vals.append(dict(bg_geoid_2020=f'11001000100{i+1}',acs_year=yr,study_metro=METROS['DMV'],
                             cash_renters_total=100+i*25,rent_burden_30_n=30+i*10,
                             rent_burden_50_n=10+i*5,total_population=200+i*60,
                             race_population=200+i*60,black_alone=20,asian_alone=10,
                             ethnicity_population=200+i*60,hispanic_population=25,
                             population_25plus=120,high_school_or_equivalent_n=30,
                             bachelors_degree_n=40,graduate_degree_n=20,
                             occupied_housing_units=100+i*25,renter_occupied_units=50+i*10,
                             households_below_poverty=12,median_household_income_approx=50000+i*5000))
    bg=check_acs_bg(pd.DataFrame(vals))
    geo=pd.DataFrame({'bg':sorted(bg.bg.unique()),'tract':['11001000100']*4})
    ex=bg[['bg','Year']].copy();ex['bg_count_1mi']=np.select([ex.bg.eq(sorted(bg.bg.unique())[0])],[2],default=1)
    ex['bg_access_1mi']=(ex.bg_count_1mi>0).astype(int)
    o=aggregate_tracts(bg,geo,ex)
    first=o.loc[o.Year.eq(2017)].iloc[0]
    assert first.bg_geom_count==4 and first.n_bg_burden_valid==4 and first.n_bg_controls_valid==4
    assert np.isclose(first.cash_renters_total,100+125+150+175)
    assert np.isclose(first.rent_burden_30p,100*(30+40+50+60)/(100+125+150+175))
    assert 0<=first.SharePopulationWithAccess_1mi<=1
    print('PASS: tract burden aggregation and population-weighted access checks')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['audit','build','models','all','self-test'],default='audit')
    opts=ap.parse_args()
    if opts.stage=='self-test':self_test()
    elif opts.stage=='audit':audit()
    elif opts.stage=='build':build()
    elif opts.stage=='models':models()
    elif opts.stage=='all':build();models()

if __name__=='__main__':main()
