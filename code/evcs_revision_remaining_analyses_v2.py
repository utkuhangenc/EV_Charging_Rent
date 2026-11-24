#!/usr/bin/env python3
"""Focused EVCS manuscript-revision analyses. Run on Gautschi (not an IV design).

Input sources were identified in uploaded notebooks:
  Prepare_Analysis_Dataset.ipynb (historical harmonized ACS and EV access)
  Pull_ACS_Data.ipynb (ACS 2010-2024; B25002 vacancy and B25070 burden)
  rebuild_evcs_panel_2017_2024_b25070_diagnostic.py (final TWFE panel)
  Create_Harmonized_CBG_Geometries.ipynb (full CBSA map coverage)

Stages:
  audit         inspect expected input files and their actual columns
  history       2014->2019 nonoverlapping ACS changes predicting 2019->2024 EVCS access (2020-BG by default)
  missingness   coverage/inclusion audit by metro; not a test for missing-at-random
  vacancy       baseline-vacancy interactions in 2017-2024 TWFE
  precision     2019->2024 nonoverlapping ACS burden-MOE audit
  jurisdiction  compare year FE with state-by-year FE in 2017-2024 TWFE

All analyses are observational. Default historical analysis uses the existing
2020-BG annual panel. EV counts in that panel are based on reference-point
distances, not final-TWFE polygon-buffer intersections. Legacy HBG files
remain selectable using --history-dir. Report as supplementary checks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = DATA_ROOT
LEGACY_HISTORY = HISTORY_DIR
HIST_ALTERNATIVE = ROOT / 'intermediate/analysis_data_harmonized'
ANNUAL = ROOT / 'intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet'
PANEL = ROOT / 'output/native_bg_rf_replication/panel_twfe_2017_2024/analysis_panel_2017_2024.parquet'
OUTPUT = ROOT / 'output/native_bg_rf_replication/reviewer_revision_checks_2026'
METROS = {'DMV':'Washington-Arlington-Alexandria',
          'Philadelphia':'Philadelphia-Camden-Wilmington'}
CONTROLS = ['TotalPopulation', 'black_per', 'hispanic_per', 'asian_per',
            'hs_per', 'bachelor_per', 'master_per', 'renter_per',
            'poverty_hh_per', 'MedianHouseholdIncome']
OUTCOMES = ['rent_afford','rent_burden_30p','rent_burden_50p']


def clean_bg(x):
    return (x.astype('string').str.replace('1500000US','',regex=False)
            .str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(12))


def choose_history_dir(override=None):
    paths = [Path(override)] if override else [LEGACY_HISTORY, HIST_ALTERNATIVE]
    for d in paths:
        if (d/'04_acs_harmonized_bg_2013_2024.parquet').exists() and \
           (d/'05_evcs_access_harmonized_bg_2014_2025.parquet').exists():
            return d
    raise FileNotFoundError('Historical 04_* and 05_* files not found. Checked: '
                            + '; '.join(map(str,paths)) + '\n'
                            'Run the historical harmonization cell in Prepare_Analysis_Dataset.ipynb '
                            'or use --history-dir PATH pointing at its output directory.')


def read_parquet(path, needed=None):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Missing: {path}')
    df=pd.read_parquet(path)
    if needed:
        absent=set(needed)-set(df.columns)
        if absent: raise KeyError(f'{path.name}: missing {sorted(absent)}; have {list(df.columns)}')
    return df


def file_audit(history_dir=None):
    sources={'Main 2017-2024 TWFE panel':PANEL,'All-CBSA ACS 2020 BG panel':ANNUAL}
    if history_dir:
        d=Path(history_dir)
        sources.update({'Historical ACS':d/'04_acs_harmonized_bg_2013_2024.parquet',
                        'Historical EV':d/'05_evcs_access_harmonized_bg_2014_2025.parquet',
                        'Historical geography':d/'03_harmonized_bg_geography_lookup.parquet'})
    for name, path in sources.items():
        print(f'\n{name}\n  {path}\n  exists={path.exists()}')
        if path.exists():
            x=pd.read_parquet(path)
            print(f'  shape={x.shape}  columns={x.columns.tolist()}')
            for y in ('Year','acs_year','snapshot_year'):
                if y in x: print(f'  {y}={sorted(pd.to_numeric(x[y],errors="coerce").dropna().unique().tolist())}')


def standard_metro(s):
    return s.map({**{v:k for k,v in METROS.items()}, 'DMV':'DMV','Philadelphia':'Philadelphia'})


def fit_glm_binomial(df, target, predictors, cluster=None):
    cols=[target]+predictors+([cluster] if cluster else [])
    t=df[cols].copy()
    for col in [target]+predictors: t[col]=pd.to_numeric(t[col], errors='coerce')
    t=t.replace([np.inf,-np.inf],np.nan).dropna()
    if len(t)<40 or t[target].nunique()<2:
        return None, len(t), 'insufficient_cases'
    x=sm.add_constant(t[predictors].astype(float),has_constant='add')
    if np.linalg.matrix_rank(x.to_numpy())<x.shape[1]: return None,len(t),'collinear'
    try:
        kw={'cov_type':'HC1'}
        if cluster and t[cluster].nunique()>=20:
            kw={'cov_type':'cluster','cov_kwds':{'groups':t[cluster]}}
        fit=sm.GLM(t[target],x,family=sm.families.Binomial()).fit(**kw)
        return fit,len(t),'ok'
    except Exception as exc: return None,len(t),f'error:{exc}'


def historical_selection(history_dir):
    d=Path(history_dir)
    acs=read_parquet(d/'04_acs_harmonized_bg_2013_2024.parquet',
        ['harmonized_bg_id','acs_year','study_metro',
         'median_gross_rent_approx','rent_burden_50_pct','rent_burden_30_pct',
         'median_household_income_approx','vacancy_pct','renter_pct',
         'bachelors_or_higher_pct'])
    ev=read_parquet(d/'05_evcs_access_harmonized_bg_2014_2025.parquet',
        ['harmonized_bg_id','snapshot_year','stations_within_1mi'])
    acs['acs_year']=pd.to_numeric(acs.acs_year,errors='coerce')
    ev['snapshot_year']=pd.to_numeric(ev.snapshot_year,errors='coerce')
    acs=acs.loc[acs.acs_year.isin([2014,2019])].copy()
    ev=ev.loc[ev.snapshot_year.isin([2019,2024])].copy()
    acs['metro']=standard_metro(acs.study_metro)
    acs=acs.loc[acs.metro.notna()].copy()
    if acs.duplicated(['harmonized_bg_id','acs_year']).any(): raise ValueError('ACS duplicate zone-year')
    if ev.duplicated(['harmonized_bg_id','snapshot_year']).any(): raise ValueError('EV duplicate zone-year')
    cols=['median_gross_rent_approx','rent_burden_50_pct','rent_burden_30_pct',
          'median_household_income_approx','vacancy_pct','renter_pct','bachelors_or_higher_pct']
    a14=acs.loc[acs.acs_year.eq(2014),['harmonized_bg_id']+cols].rename(columns={c:f'{c}_14' for c in cols})
    a19=acs.loc[acs.acs_year.eq(2019),['harmonized_bg_id','metro']+cols].rename(columns={c:f'{c}_19' for c in cols})
    e19=ev.loc[ev.snapshot_year.eq(2019),['harmonized_bg_id','stations_within_1mi']].rename(columns={'stations_within_1mi':'ev_19'})
    e24=ev.loc[ev.snapshot_year.eq(2024),['harmonized_bg_id','stations_within_1mi']].rename(columns={'stations_within_1mi':'ev_24'})
    x=a19.merge(a14,on='harmonized_bg_id',how='left',validate='one_to_one').merge(e19,on='harmonized_bg_id',how='left',validate='one_to_one').merge(e24,on='harmonized_bg_id',how='left',validate='one_to_one')
    lk=d/'03_harmonized_bg_geography_lookup.parquet'
    if lk.exists():
        lookup=read_parquet(lk)
        aux=[c for c in ['harmonized_bg_id','cluster_tract_geoid_2020','county_geoid','state_fips'] if c in lookup]
        x=x.merge(lookup[aux].drop_duplicates('harmonized_bg_id'),on='harmonized_bg_id',how='left',validate='one_to_one')
    x['pre_rent_change_pct']=np.where(x['median_gross_rent_approx_14'].gt(0),
        100*(x['median_gross_rent_approx_19']/x['median_gross_rent_approx_14']-1),np.nan)
    x['pre_severe_change_pp']=x['rent_burden_50_pct_19']-x['rent_burden_50_pct_14']
    x['pre_moderate_change_pp']=x['rent_burden_30_pct_19']-x['rent_burden_30_pct_14']
    x['pre_income_change_pct']=np.where(x['median_household_income_approx_14'].gt(0),
        100*(x['median_household_income_approx_19']/x['median_household_income_approx_14']-1),np.nan)
    x['future_stock_change']=x.ev_24-x.ev_19
    x['future_gain']=np.where(x.ev_19.eq(0) & x.ev_24.notna(),(x.ev_24>0).astype(float),np.nan)
    x['baseline_no_access']=x.ev_19.eq(0)
    x['log_income_2019']=np.where(x.median_household_income_approx_19.gt(0),
            np.log(x.median_household_income_approx_19), np.nan)
    x['log_rent_2019']=np.where(x.median_gross_rent_approx_19.gt(0),
            np.log(x.median_gross_rent_approx_19), np.nan)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    x.to_csv(OUTPUT/'01_historical_2014_2019_pretrend_vs_2019_2024_ev.csv',index=False)
    no=x.loc[x.baseline_no_access].copy()
    summary=no.groupby('metro',dropna=False).agg(
        zones=('harmonized_bg_id','nunique'),
        gained_access=('future_gain','sum'),
        gain_rate=('future_gain','mean'),
        median_pre_rent_change_pct=('pre_rent_change_pct','median'),
        median_pre_severe_change_pp=('pre_severe_change_pp','median'),
        prior_rent_change_available=('pre_rent_change_pct','count'),
        prior_severe_change_available=('pre_severe_change_pp','count')).reset_index()
    summary.to_csv(OUTPUT/'01_historical_coverage.csv',index=False)
    print('\n2019 zones without EV access; did they gain by 2024?\n',summary.to_string(index=False))
    rows=[]
    for scope in ('Pooled','DMV','Philadelphia'):
        sub=no if scope=='Pooled' else no.loc[no.metro.eq(scope)]
        sub=sub.copy()
        sub['metro_philly']=sub.metro.eq('Philadelphia').astype(float)
        # Each predictor is assessed separately to limit multicollinearity
        # across closely related historical rent/income changes.
        for pre in ('pre_rent_change_pct','pre_severe_change_pp','pre_moderate_change_pp'):
            pred=[pre,'log_income_2019','vacancy_pct_19','renter_pct_19']
            if scope=='Pooled': pred.append('metro_philly')
            cluster='cluster_tract_geoid_2020' if 'cluster_tract_geoid_2020' in sub else None
            fit,n,status=fit_glm_binomial(sub,'future_gain',pred,cluster)
            if fit is None:
                rows.append({'Scope':scope,'pretrend':pre,'N':n,'status':status});continue
            term=fit.params[pre]
            rows.append({'Scope':scope,'pretrend':pre,'N':n,
                         'N_gaining':int(sub.loc[sub[pred+['future_gain']].notna().all(axis=1),'future_gain'].sum()),
                         'coefficient_log_odds':float(term),
                         'odds_ratio_per_unit':float(np.exp(term)),
                         'conf_low_or':float(np.exp(fit.conf_int().loc[pre,0])),
                         'conf_high_or':float(np.exp(fit.conf_int().loc[pre,1])),
                         'p_value':float(fit.pvalues[pre]),'status':'ok'})
    res=pd.DataFrame(rows)
    res.to_csv(OUTPUT/'01_historical_selection_logit.csv',index=False)
    print('\nHistorical selection models (odds ratios per 1 pp pretrend):\n',res.to_string(index=False))
    return x,summary,res


def acs_precision_audit(history_dir):
    """Descriptive independent-window ACS uncertainty; no model of IV/displacement.

    2019 ACS5 = 2015-2019; 2024 ACS5 = 2020-2024.
    The original harmonization combines count MOEs by root-sum-of-squares and
    approximates ratio MOEs. These are approximate 90% error margins, not a
    formal simultaneous multiple-comparison correction.
    """
    f=read_parquet(Path(history_dir)/'04_acs_harmonized_bg_2013_2024.parquet',
       ['acs_year','harmonized_bg_id','study_metro',
        'rent_burden_30_pct','rent_burden_30_pct_moe',
        'rent_burden_50_pct','rent_burden_50_pct_moe'])
    f['acs_year']=pd.to_numeric(f.acs_year,errors='coerce')
    f['metro']=standard_metro(f.study_metro)
    f=f.loc[f.acs_year.isin([2019,2024])&f.metro.notna()].copy()
    years=[]
    for year in (2019,2024):
        g=f.loc[f.acs_year.eq(year)].copy()
        names=['harmonized_bg_id','metro','rent_burden_30_pct','rent_burden_30_pct_moe',
               'rent_burden_50_pct','rent_burden_50_pct_moe']
        years.append(g[names].rename(columns={c:f'{c}_{year}' for c in names if c not in ['harmonized_bg_id','metro']}))
    wide=years[0].merge(years[1],on=['harmonized_bg_id','metro'],validate='one_to_one',how='outer')
    for threshold in (30,50):
        c=f'rent_burden_{threshold}_pct'
        for year in (2019,2024):
            wide[f'{c}_{year}']=pd.to_numeric(wide[f'{c}_{year}'],errors='coerce')
            wide[f'{c}_moe_{year}']=pd.to_numeric(wide[f'{c}_moe_{year}'],errors='coerce')
        wide[f'{c}_change_pp']=wide[f'{c}_2024']-wide[f'{c}_2019']
        wide[f'{c}_change_moe_90']=np.sqrt(wide[f'{c}_moe_2019']**2+wide[f'{c}_moe_2024']**2)
        wide[f'{c}_change_exceeds_90pct_moe']=np.where(
            wide[f'{c}_change_moe_90'].gt(0)&wide[f'{c}_change_pp'].notna(),
            (wide[f'{c}_change_pp'].abs()>wide[f'{c}_change_moe_90']).astype(float),np.nan)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    wide.to_csv(OUTPUT/'05_acs_2019_2024_nonoverlap_moe_by_zone.csv',index=False)
    rows=[]
    for metro,g in wide.groupby('metro'):
        for threshold in (30,50):
            c=f'rent_burden_{threshold}_pct'
            ch=g[f'{c}_change_pp']; moe=g[f'{c}_change_moe_90']; measurable=g[f'{c}_change_exceeds_90pct_moe']
            rows.append(dict(metro=metro,burden_threshold=threshold,all_zones=len(g),
               n_with_two_estimates=int(ch.notna().sum()),n_with_change_moe=int(moe.notna().sum()),
               median_abs_change_pp=float(ch.abs().median()),median_change_moe_90_pp=float(moe.median()),
               share_changes_exceed_approx_90pct_moe=float(measurable.mean())))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTPUT/'05_acs_uncertainty_summary.csv',index=False)
    print('\nACS nonoverlap uncertainty summary (90% MOEs):\n',summary.to_string(index=False))
    return wide,summary


def panel_2017_2024():
    f=read_parquet(PANEL,['GEO_ID','Year','study_metro',*OUTCOMES,
                           'PointCount_1mi','WithinBuffer_1mi',*CONTROLS])
    f['bg']=clean_bg(f['GEO_ID'])
    f['Year']=pd.to_numeric(f.Year,errors='raise').astype(int)
    f['metro']=standard_metro(f.study_metro)
    f=f[f.Year.between(2017,2024)&f.metro.notna()].copy()
    if f.duplicated(['bg','Year']).any():raise ValueError('Duplicate panel bg-year')
    ids=f.groupby('bg').Year.agg(lambda y:set(y)==set(range(2017,2025)))
    f=f.loc[f.bg.isin(ids.index[ids])].copy()
    print(f'Balanced panel: {f.bg.nunique():,} BGs x 8 years = {len(f):,} rows')
    # Calendar-year completeness alone is not the regression complete-case sample.
    print('Note: complete cases depend on the outcome, controls and access metric;\n'
          '      2,903 is the previously reported model sample, NOT a required raw-panel row count.')
    return f


def annual_acs():
    a=read_parquet(ANNUAL,['bg_geoid_2020','acs_year','study_metro',
                             'median_gross_rent_approx','cash_renters_total'])
    a['bg']=clean_bg(a.bg_geoid_2020)
    a['acs_year']=pd.to_numeric(a.acs_year,errors='coerce').astype('Int64')
    a['metro']=standard_metro(a.study_metro)
    a=a.loc[a.metro.notna()].copy()
    if a.duplicated(['bg','acs_year']).any(): raise ValueError('Duplicate ACS bg-year')
    return a


def missingness():
    a=annual_acs()
    panel=panel_2017_2024()
    roster=a.loc[a.acs_year.isin([2017,2024])].copy()
    roster['in_balanced']=roster.bg.isin(set(panel.bg.unique()))
    # outcome availability is not treated as zero; summarize 2017, 2024 by metro.
    rows=[]
    for (metro,year,inc),sub in roster.groupby(['metro','acs_year','in_balanced'],dropna=False):
        r={'metro':metro,'year':int(year),'in_balanced_panel':bool(inc),
           'CBGs':sub.bg.nunique(),
           'valid_median_rent':int(sub.median_gross_rent_approx.notna().sum()),
           'valid_renter_denominator':int(pd.to_numeric(sub.cash_renters_total,errors='coerce').gt(0).sum())}
        for name in ['median_household_income_approx','renter_pct','vacancy_pct',
                     'rent_burden_50_pct','rent_burden_30_pct','total_population']:
            if name in sub:
                vals=pd.to_numeric(sub[name],errors='coerce')
                r[f'{name}_nonmissing']=int(vals.notna().sum())
                r[f'{name}_median']=vals.median()
        rows.append(r)
    out=pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUTPUT/'02_representativeness_missingness.csv',index=False)
    print('\nCoverage and sample selection:\n',out.to_string(index=False))
    return out


def add_vacancy(panel, annual):
    v=annual.loc[annual.acs_year.eq(2017)].copy()
    if 'vacancy_pct' in v:
        v['vacancy17']=pd.to_numeric(v.vacancy_pct,errors='coerce')
    elif {'vacant_units','housing_units'}.issubset(v.columns):
        vacant=pd.to_numeric(v.vacant_units,errors='coerce')
        total=pd.to_numeric(v.housing_units,errors='coerce')
        v['vacancy17']=np.where(total.gt(0),100*vacant/total,np.nan)
    else: raise KeyError('Annual ACS missing vacancy_pct or vacant_units/housing_units')
    if v.duplicated('bg').any():raise ValueError('Duplicate 2017 vacancy bg')
    matched=panel.merge(v[['bg','vacancy17']],on='bg',how='left',validate='many_to_one')
    matched['vacancy17_10pp']=matched.vacancy17/10.0
    print('Matched 2017 vacancy to TWFE: ',matched.drop_duplicates('bg').vacancy17.notna().sum(),
          '/',matched.bg.nunique(), 'BGs')
    return matched


def fe_residualize(matrix, group_ids, tol=1e-10, maxiter=350):
    """Numerical Frisch-Waugh-Lovell absorption for arbitrary category FE groups."""
    work=np.asarray(matrix,dtype=float).copy()
    if work.ndim==1:work=work[:,None]
    if not np.isfinite(work).all(): raise ValueError('nonfinite FE data')
    codes=[]
    for key in group_ids:
        code,_=pd.factorize(pd.Series(key,dtype='string'),sort=False)
        if (code<0).any(): raise ValueError('missing FE group')
        codes.append(code)
    for iteration in range(maxiter):
        change=0.0
        for g in codes:
            cnt=np.bincount(g)
            means=np.vstack([np.bincount(g,weights=work[:,i])/cnt for i in range(work.shape[1])]).T
            delta=means[g]
            work-=delta
            change=max(change,float(np.max(np.abs(delta))))
        if change<tol:break
    if iteration>=maxiter-1: warnings.warn('FE absorption did not converge')
    return work


def fit_twfe_fwl(d, outcome, exposure, extra=None, state_year=False):
    extra=extra or []
    predictors=[exposure]+extra+CONTROLS
    need=list(dict.fromkeys([outcome]+predictors+['bg','Year','metro']))
    t=d[need].copy()
    for c in [outcome]+predictors:t[c]=pd.to_numeric(t[c],errors='coerce')
    t=t.replace([np.inf,-np.inf],np.nan).dropna()
    if t.bg.nunique()<40 or t.Year.nunique()<4:
        return None,dict(N=len(t),BGs=t.bg.nunique(),status='insufficient_cases')
    group2=(t.bg.str[:2].astype(str)+'_'+t.Year.astype(str)) if state_year else t.Year.astype(str)
    res=fe_residualize(t[[outcome]+predictors].to_numpy(dtype=float),[t.bg,group2])
    x=res[:,1:];y=res[:,0]
    keep=(np.std(x,axis=0)>1e-10)
    # If any intended exposure/interaction is absorbed, this model is not estimable.
    if not all(keep[:1+len(extra)]):
        return None,dict(N=len(t),BGs=t.bg.nunique(),status='exposure_absorbed')
    x=x[:,keep]
    names=np.array(predictors)[keep].tolist()
    if np.linalg.matrix_rank(x)<x.shape[1]:
        return None,dict(N=len(t),BGs=t.bg.nunique(),status='collinear')
    fit=sm.OLS(y,x).fit(cov_type='cluster',cov_kwds={'groups':t.bg})
    result=dict(N=len(t),BGs=t.bg.nunique(),status='ok')
    for name,coef,se,pval in zip(names,fit.params,fit.bse,fit.pvalues):
        if name in [exposure]+extra:
            result[f'{name}_beta']=float(coef)
            result[f'{name}_se']=float(se)
            result[f'{name}_p']=float(pval)
    result['FE']='BG + state-by-year' if state_year else 'BG + common year'
    return fit,result


def vacancy_models():
    panel=panel_2017_2024()
    panel=add_vacancy(panel,annual_acs())
    output=[]
    for scope in ('Pooled','DMV','Philadelphia'):
        sub=panel if scope=='Pooled' else panel.loc[panel.metro.eq(scope)]
        # Center at the sample baseline vacancy rate to interpret access main effects.
        average=sub.drop_duplicates('bg').vacancy17_10pp.mean()
        sub=sub.copy()
        sub['vacancy_centered']=sub.vacancy17_10pp-average
        for exposure in ('PointCount_1mi','WithinBuffer_1mi'):
            sub['charger_x_vacancy']=pd.to_numeric(sub[exposure],errors='coerce')*sub.vacancy_centered
            for outcome in OUTCOMES:
                _,row=fit_twfe_fwl(sub,outcome,exposure,extra=['charger_x_vacancy'])
                row.update(Scope=scope,Outcome=outcome,Exposure=exposure,
                           baseline_vacancy_pct_at_center=float(average*10),
                           interaction_unit='per 10 percentage point increase in 2017 vacancy')
                output.append(row)
    result=pd.DataFrame(output)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUTPUT/'03_vacancy_interactions_twfe.csv',index=False)
    print('\nVacancy interaction results:\n',result.to_string(index=False))
    return result


def jurisdiction_models():
    panel=panel_2017_2024()
    rows=[]
    for scope in ('Pooled','DMV','Philadelphia'):
        sub=panel if scope=='Pooled' else panel.loc[panel.metro.eq(scope)]
        for outcome in OUTCOMES:
            for exposure in ('PointCount_1mi','WithinBuffer_1mi'):
                for state_year in (False,True):
                    _,row=fit_twfe_fwl(sub,outcome,exposure,state_year=state_year)
                    row.update(Scope=scope,Outcome=outcome,Exposure=exposure)
                    rows.append(row)
    result=pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUTPUT/'04_jurisdiction_state_year_sensitivity.csv',index=False)
    print('\nState-year fixed-effects sensitivity:\n',result.to_string(index=False))
    return result


# ================================================================
# NEW: already-existing 2020-CBG historical panel (no legacy files)
# ================================================================
def historical_panel_2020bg():
    """Read the completed ACS/charging panel; do not require old 04/05 files.

    Historical charging counts here are based on 2020-BG reference-point
    distances as constructed upstream. They should NOT be described as
    1-mile polygon-buffer intersection counts from the main TWFE.
    """
    require = ['bg_geoid_2020', 'acs_year', 'study_metro',
               'median_gross_rent_approx', 'median_household_income_approx',
               'rent_burden_30_pct', 'rent_burden_50_pct',
               'vacancy_pct', 'renter_pct', 'bachelors_or_higher_pct',
               'stations_within_1mi', 'snapshot_year']
    f = read_parquet(ANNUAL, require).copy()
    f['bg'] = clean_bg(f['bg_geoid_2020'])
    f['acs_year'] = pd.to_numeric(f['acs_year'], errors='coerce')
    f['snapshot_year'] = pd.to_numeric(f['snapshot_year'], errors='coerce')
    f['metro'] = standard_metro(f['study_metro'])
    f = f.loc[f['acs_year'].isin([2014, 2019, 2024]) & f.metro.notna()].copy()
    if f.duplicated(['bg', 'acs_year']).any():
        dups = f.loc[f.duplicated(['bg', 'acs_year'], keep=False), ['bg', 'acs_year']].head()
        raise ValueError(f'Historical 2020 BG panel has duplicate BG-year rows:\n{dups}')
    return f


def historical_selection_2020bg():
    """2014->2019 nonoverlapping ACS windows predicting 2019->2024 access.

    Restricted to BGs without 1-mile access in 2019 to compare subsequent
    first gains. Observational association, not an IV or causal model.
    """
    f = historical_panel_2020bg()
    vars = ['median_gross_rent_approx', 'median_household_income_approx',
            'rent_burden_30_pct', 'rent_burden_50_pct', 'vacancy_pct',
            'renter_pct', 'bachelors_or_higher_pct']
    a14 = f.loc[f.acs_year.eq(2014), ['bg'] + vars].rename(
        columns={c:f'{c}_14' for c in vars})
    a19 = f.loc[f.acs_year.eq(2019), ['bg','metro','county_geoid','state_abbr'] + vars +
                ['stations_within_1mi','snapshot_year']].rename(
        columns={**{c:f'{c}_19' for c in vars}, 'stations_within_1mi':'ev_19',
                 'snapshot_year':'ev_snapshot_2019'})
    a24 = f.loc[f.acs_year.eq(2024), ['bg','stations_within_1mi','snapshot_year']].rename(
        columns={'stations_within_1mi':'ev_24','snapshot_year':'ev_snapshot_2024'})
    x = a19.merge(a14,on='bg',how='left',validate='one_to_one').merge(
        a24,on='bg',how='left',validate='one_to_one')
    # Never silently take the charger count from a different snapshot year.
    for y in [2019, 2024]:
        ok = x[f'ev_snapshot_{y}'].eq(y)
        x.loc[~ok, f'ev_{str(y)[-2:]}'] = np.nan
    numeric = [c for c in x.columns if c.startswith(('median_gross_rent_approx_',
                'median_household_income_approx_', 'rent_burden_30_pct_',
                'rent_burden_50_pct_', 'vacancy_pct_', 'renter_pct_',
                'bachelors_or_higher_pct_','ev_'))]
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    x['pre_rent_change_pct'] = np.where(x.median_gross_rent_approx_14.gt(0),
        100*(x.median_gross_rent_approx_19 / x.median_gross_rent_approx_14 - 1), np.nan)
    x['pre_severe_change_pp'] = x.rent_burden_50_pct_19 - x.rent_burden_50_pct_14
    x['pre_moderate_change_pp'] = x.rent_burden_30_pct_19 - x.rent_burden_30_pct_14
    x['pre_income_change_pct'] = np.where(x.median_household_income_approx_14.gt(0),
        100*(x.median_household_income_approx_19/x.median_household_income_approx_14 - 1),np.nan)
    x['future_stock_change'] = x.ev_24 - x.ev_19
    x['baseline_no_access'] = x.ev_19.eq(0)
    x['future_gain'] = np.where(x.baseline_no_access & x.ev_24.notna(),
                                (x.ev_24>0).astype(float), np.nan)
    x['log_income_2019'] = np.where(x.median_household_income_approx_19.gt(0),
                                    np.log(x.median_household_income_approx_19),np.nan)
    x['log_rent_2019'] = np.where(x.median_gross_rent_approx_19.gt(0),
                                 np.log(x.median_gross_rent_approx_19),np.nan)
    x['cluster_tract_geoid_2020'] = x.bg.str[:11]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    x.to_csv(OUTPUT/'01_historical_2020bg_2014_2019_vs_2019_2024.csv',index=False)
    no = x.loc[x.baseline_no_access & x.future_gain.notna()].copy()
    summ = no.groupby('metro',dropna=False).agg(
        bg=('bg','nunique'), n_gain=('future_gain','sum'),
        gain_rate=('future_gain','mean'),
        n_pre_rent_change=('pre_rent_change_pct','count'),
        n_pre_severe_change=('pre_severe_change_pp','count'),
        median_pre_rent_change_pct=('pre_rent_change_pct','median'),
        median_pre_severe_change_pp=('pre_severe_change_pp','median')).reset_index()
    summ.to_csv(OUTPUT/'01_historical_2020bg_coverage.csv',index=False)
    results = []
    for scope in ('Pooled','DMV','Philadelphia'):
        sub = no if scope=='Pooled' else no.loc[no.metro.eq(scope)]
        sub = sub.copy()
        sub['metro_philly'] = sub.metro.eq('Philadelphia').astype(float)
        for pre in ['pre_rent_change_pct','pre_severe_change_pp','pre_moderate_change_pp']:
            pred = [pre,'log_income_2019','vacancy_pct_19','renter_pct_19']
            if scope=='Pooled': pred.append('metro_philly')
            fit,n,status = fit_glm_binomial(sub,'future_gain',pred,'cluster_tract_geoid_2020')
            row = dict(Scope=scope,pretrend=pre,N=n,status=status)
            if fit is not None:
                valid = sub[['future_gain']+pred].apply(pd.to_numeric,errors='coerce').replace(
                    [np.inf,-np.inf],np.nan).notna().all(axis=1)
                row.update(N_gaining=int(sub.loc[valid,'future_gain'].sum()),
                           coefficient_log_odds=float(fit.params[pre]),
                           odds_ratio_per_unit=float(np.exp(fit.params[pre])),
                           conf_low_or=float(np.exp(fit.conf_int().loc[pre,0])),
                           conf_high_or=float(np.exp(fit.conf_int().loc[pre,1])),
                           p_value=float(fit.pvalues[pre]))
            results.append(row)
    res = pd.DataFrame(results)
    res.to_csv(OUTPUT/'01_historical_2020bg_selection_logit.csv',index=False)
    print('\nHistorical 2020-CBG selection (nonoverlapping ACS 2014/2019):\n',summ.to_string(index=False))
    print('\nModels (odds ratios per 1 percentage point pre-period change):\n',res.to_string(index=False))
    print('Historical station counts are reference-point proximity measures, NOT polygon-buffer intersections.')
    return x,summ,res


def acs_precision_audit_2020bg():
    cols = ['bg_geoid_2020','acs_year','study_metro',
            'rent_burden_30_pct','rent_burden_30_pct_moe',
            'rent_burden_50_pct','rent_burden_50_pct_moe']
    f = read_parquet(ANNUAL,cols)[cols].copy()
    f['bg'] = clean_bg(f.bg_geoid_2020)
    f['acs_year'] = pd.to_numeric(f.acs_year,errors='coerce')
    f['metro'] = standard_metro(f.study_metro)
    f = f.loc[f.acs_year.isin([2019,2024]) & f.metro.notna()].copy()
    if f.duplicated(['bg','acs_year']).any():raise ValueError('Duplicate precision BG-year')
    frames=[]
    for year in (2019,2024):
        t = f.loc[f.acs_year.eq(year),['bg','metro']+cols[3:]].copy()
        t = t.rename(columns={c:f'{c}_{year}' for c in cols[3:]})
        frames.append(t)
    wide = frames[0].merge(frames[1],on=['bg','metro'],how='outer',validate='one_to_one')
    for threshold in (30,50):
        c=f'rent_burden_{threshold}_pct'
        for year in (2019,2024):
            for cc in (f'{c}_{year}',f'{c}_moe_{year}'):
                wide[cc] = pd.to_numeric(wide[cc],errors='coerce')
        wide[f'{c}_change_pp'] = wide[f'{c}_2024'] - wide[f'{c}_2019']
        wide[f'{c}_change_moe_90'] = np.sqrt(
            wide[f'{c}_moe_2019']**2 + wide[f'{c}_moe_2024']**2)
        complete = (wide[f'{c}_change_moe_90'].gt(0) &
                    wide[f'{c}_change_pp'].notna())
        wide[f'{c}_change_exceeds_90pct_moe'] = np.where(complete,
            wide[f'{c}_change_pp'].abs() > wide[f'{c}_change_moe_90'], np.nan)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    wide.to_csv(OUTPUT/'05_acs_2020bg_2019_2024_moe_by_bg.csv',index=False)
    rows=[]
    for metro,sub in wide.groupby('metro'):
        for threshold in (30,50):
            c=f'rent_burden_{threshold}_pct'
            rows.append(dict(metro=metro,threshold=threshold,all_BG=len(sub),
              n_with_two_estimates=int(sub[f'{c}_change_pp'].notna().sum()),
              n_with_change_moe=int(sub[f'{c}_change_moe_90'].notna().sum()),
              median_abs_change_pp=float(sub[f'{c}_change_pp'].abs().median()),
              median_change_moe_90_pp=float(sub[f'{c}_change_moe_90'].median()),
              share_changes_exceed_approx_90pct_moe=float(sub[f'{c}_change_exceeds_90pct_moe'].mean())))
    summ=pd.DataFrame(rows)
    summ.to_csv(OUTPUT/'05_acs_2020bg_uncertainty_summary.csv',index=False)
    print('\nNonoverlapping ACS 2019 (2015-19) / 2024 (2020-24) uncertainty:\n',summ.to_string(index=False))
    return wide,summ


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['audit','history','precision','missingness','vacancy','jurisdiction','all'],default='audit')
    p.add_argument('--history-dir',default=None)
    args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    hd=None
    if args.history_dir:
        hd=choose_history_dir(args.history_dir)
        print('Using explicitly requested legacy historical HBG data:',hd)
    elif args.stage in ('audit','history','precision','all'):
        print('Using existing 2020-BG ACS/EV annual panel for historical analyses:',ANNUAL)
    if args.stage=='audit':
        file_audit(hd)
        if ANNUAL.exists():
            f=historical_panel_2020bg()
            print('\nHistorical 2020-BG availability by year/metro:')
            print(f.groupby(['metro','acs_year']).agg(
              BGs=('bg','nunique'),n_rent=('median_gross_rent_approx','count'),
              n_severe=('rent_burden_50_pct','count'),
              n_chargers=('stations_within_1mi','count')).to_string())
    if args.stage in ('history','all'):
        if hd:historical_selection(hd)
        else:historical_selection_2020bg()
    if args.stage in ('precision','all'):
        if hd:acs_precision_audit(hd)
        else:acs_precision_audit_2020bg()
    if args.stage in ('missingness','all'):missingness()
    if args.stage in ('vacancy','all'):vacancy_models()
    if args.stage in ('jurisdiction','all'):jurisdiction_models()
    print('\nOutputs:',OUTPUT)

if __name__=='__main__':main()
