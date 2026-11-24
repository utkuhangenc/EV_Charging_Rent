#!/usr/bin/env python3
"""Rebuild original EVCS/rental-affordability TWFE study for expanded Philly/DMV CBSAs, 2017-2024.

Sources of specification: original 01_build_panel_dataset_fixed.Rmd and 03_panel_twfe_models.Rmd.

Stage A: ACS + 0.5, 1, 2 mile station-polygon proximity in EPSG:5070.
Stage B: exact original R controls, three outcomes, five exposures, income/no-income,
         both contemporaneous TWFE and (t+2,t,t-2) joint TWFE specifications.

Requires: numpy, pandas, pyarrow, geopandas, shapely, scipy.
NO updates to the original 2017-2024 RF/modified-Poisson outputs.

Run: python rebuild_evcs_panel_2017_2024.py --stage all --scopes Pooled
     python rebuild_evcs_panel_2017_2024.py --stage models --scopes Pooled DMV Philadelphia

Output is deliberately kept in a new folder, panel_twfe_2017_2024.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import numpy as np
import pandas as pd

ROOT = DATA_ROOT
BASE_DIR = ROOT / 'output/native_bg_rf_replication'
BASE_PANEL = BASE_DIR / 'exact_replication_panel_2017_2024.csv'
ANNUAL = ROOT / 'intermediate/analysis_data_2020bg/23_analysis_2020bg_annual.parquet'
EV = ROOT / 'intermediate/evcs/afdc_ev_master_with_geography.parquet'
BG = BG_SHP
OUT = BASE_DIR / 'panel_twfe_2017_2024'
YEARS = list(range(2017,2025))
METROS = {'Philadelphia':'Philadelphia-Camden-Wilmington', 'DMV':'Washington-Arlington-Alexandria'}
TAGS = {'0_5mi':.5, '1mi':1.0, '2mi':2.0}
OUTCOMES = ['rent_afford','rent_burden_30p','rent_burden_50p']
CONTROLS = ['TotalPopulation','black_per','hispanic_per','asian_per',
            'hs_per','bachelor_per','master_per','renter_per','poverty_hh_per']
BASES = ([f'WithinBuffer_{t}' for t in TAGS] +
         [f'PointCount_{t}' for t in TAGS] +
         [f'dPointCount_{t}' for t in TAGS] +
         [f'logPointCount_{t}' for t in TAGS] +
         [f'logNewAdds_{t}' for t in TAGS])


def norm_id(s):
    return (s.astype('string').str.replace('1500000US','',regex=False)
             .str.replace(r'\.0$','',regex=True).str.strip().str.zfill(12))


def required_path(p):
    if not p.exists():
        raise FileNotFoundError(f'Required file missing: {p}')
    return p


def numeric(df, col, candidates, required=True):
    found = next((name for name in candidates if name in df.columns), None)
    if found is None:
        if required:
            raise KeyError(f'Cannot construct {col}. Need one of: {candidates}; columns available: {list(df.columns)}')
        df[col] = np.nan
    else:
        df[col] = pd.to_numeric(df[found], errors='coerce')
    return found


def select_annual_snapshots(df):
    if 'snapshot_year' not in df:
        raise KeyError('AFDC master needs snapshot_year')
    df = df.copy()
    df['snapshot_year'] = pd.to_numeric(df['snapshot_year'],errors='coerce')
    df = df[df.snapshot_year.isin(YEARS)].copy()
    chosen=[]
    for year in YEARS:
        d = df[df.snapshot_year.eq(year)].copy()
        if d.empty:
            raise ValueError(f'No AFDC snapshot for {year}')
        if 'is_year_end_snapshot' in d and d['is_year_end_snapshot'].astype('string').str.lower().isin(['true','1']).any():
            d = d[d['is_year_end_snapshot'].astype('string').str.lower().isin(['true','1'])]
            basis='year-end'
        elif 'snapshot_date' in d:
            date=pd.to_datetime(d.snapshot_date,errors='coerce')
            if date.notna().sum()==0:
                raise ValueError(f'No valid AFDC snapshot date for {year}')
            d=d[date.eq(date.max())]
            basis=f'latest {date.max().date()}'
        else:
            raise ValueError(f'Cannot select one snapshot for {year}')
        print(f'AFDC {year}: {basis} snapshot, {len(d):,} rows',flush=True)
        chosen.append(d)
    return pd.concat(chosen,ignore_index=True)


def make_panel(out_dir:Path):
    import geopandas as gpd
    for p in [BASE_PANEL, ANNUAL, EV, BG]: required_path(p)
    out_dir.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(BASE_PANEL,dtype={'bg_geoid_2020':'string'},low_memory=False)
    base['bg_geoid_2020']=norm_id(base.bg_geoid_2020)
    base['Year']=pd.to_numeric(base.Year,errors='raise').astype(int)
    base=base[base.Year.isin(YEARS)&base.study_metro.isin(METROS.values())]
    if base.duplicated(['bg_geoid_2020','Year']).any():
        raise ValueError('Duplicate BG/Year in exact_replication_panel_2017_2024.csv')
    keep=['bg_geoid_2020','Year','study_metro']
    base=base[keep].copy()
    original_counts=base.groupby(['study_metro','Year']).size().rename('N_source_panel')
    print('Original replicate sample:\n',original_counts.to_string(),flush=True)

    annual=pd.read_parquet(ANNUAL)
    needed_base=['bg_geoid_2020','acs_year','study_metro']
    if not set(needed_base).issubset(annual):
        raise KeyError('2020BG annual panel missing keys: '+str(set(needed_base)-set(annual)))
    annual['bg_geoid_2020']=norm_id(annual.bg_geoid_2020)
    annual['Year']=pd.to_numeric(annual.acs_year,errors='coerce')
    annual=annual[annual.Year.isin(YEARS)&annual.study_metro.isin(METROS.values())].copy()
    if annual.duplicated(['bg_geoid_2020','Year']).any():
        raise ValueError('Duplicate BG/Year in annual ACS panel')
    df=base.merge(annual.drop(columns=['study_metro']),on=['bg_geoid_2020','Year'],
                  how='left',indicator=True,validate='one_to_one')
    if df['_merge'].ne('both').any():
        raise ValueError(f"{df['_merge'].ne('both').sum()} base panel rows absent from harmonized annual ACS panel")
    df=df.drop(columns=['_merge'])
    names={
      'TotalPopulation':['total_population'],
      'MedianRent':['median_gross_rent_approx'],
      'MedianHouseholdIncome':['median_household_income_approx'],
      'black_per':['black_pct'], 'hispanic_per':['hispanic_pct'],
      'asian_per':['asian_pct'], 'renter_per':['renter_pct'],
      'hs_per':['high_school_or_equivalent_pct'],
      'bachelor_per':['bachelors_degree_pct'],
      'master_per':['graduate_degree_pct'],
      'poverty_n':['households_below_poverty'],
      'households_total':['occupied_housing_units'],
      'rent_burden_30_n':['rent_burden_30_n'],
      'rent_burden_50_n':['rent_burden_50_n'],
      'renters_total':['cash_renters_total'],
    }
    for target,sources in names.items(): numeric(df,target,sources,required=True)
    # Original R: 100 * HouseholdUnderPoverty / TotalHH (occupied households),
    # NOT the revised ACS poverty-universe share used for another sensitivity design.
    df['poverty_hh_per']=100*df.poverty_n/df.households_total.where(df.households_total.gt(0))
    # B25070 components should be nonnegative and must not exceed B25070_001E.
    # Density-weighted interpolation to 2020 BGs introduces tiny FP rounding
    # errors, so do not reject harmless 1e-12 differences.
    burden_fields=['renters_total','rent_burden_30_n','rent_burden_50_n']
    negative_counts={c:int(df[c].lt(0).sum()) for c in burden_fields}
    if any(negative_counts.values()):
        warnings.warn('Negative ACS burden counts (likely suppression codes): '
                      +str(negative_counts)+'; recoding them to missing, not zero')
        for c in burden_fields:
            df.loc[df[c].lt(0),c]=np.nan
    denom=df['renters_total']
    tolerance=1e-8*np.maximum(1.0,denom.abs())
    for c in ['rent_burden_30_n','rent_burden_50_n']:
        excess=df[c]-denom
        floating_error=denom.notna() & df[c].notna() & excess.gt(0) & excess.le(tolerance)
        if floating_error.any():
            print(f'Corrected {int(floating_error.sum())} tiny floating-point excesses in {c}',flush=True)
            df.loc[floating_error,c]=denom.loc[floating_error]
    # Any substantial violation should be traced to upstream input/harmonization.
    # Save offending BG-years and stop, rather than creating >100% burdens.
    bad30=(df['rent_burden_30_n'].gt(denom+tolerance) & denom.notna())
    bad50=(df['rent_burden_50_n'].gt(denom+tolerance) & denom.notna())
    badnested=(df['rent_burden_50_n'].gt(df['rent_burden_30_n']+tolerance)
               & df['rent_burden_30_n'].notna())
    bad0=((denom.eq(0)) & (df['rent_burden_30_n'].gt(0) | df['rent_burden_50_n'].gt(0)))
    problematic=bad30|bad50|badnested|bad0
    if problematic.any():
        audit=df.loc[problematic,
                     ['study_metro','bg_geoid_2020','Year', 'renters_total',
                      'rent_burden_30_n','rent_burden_50_n']].copy()
        audit['excess_30']=audit['rent_burden_30_n']-audit['renters_total']
        audit['excess_50']=audit['rent_burden_50_n']-audit['renters_total']
        audit['ratio_30']=audit['rent_burden_30_n']/audit['renters_total'].replace(0,np.nan)
        report=out_dir/'b25070_inconsistent_counts_audit.csv'
        audit.to_csv(report,index=False)
        print('B25070 inconsistent counts by metro and year:\n'
              +audit.groupby(['study_metro','Year']).size().to_string(),flush=True)
        print('Worst rows:\n'+audit.sort_values('excess_30',ascending=False).head(15).to_string(index=False),flush=True)
        raise ValueError(f'{len(audit)} B25070 BG-years have substantive count inconsistencies. '
                         f'Inspect {report} before changing the denominator or dropping rows.')
    # R original: numerator / B25070_001E (all renter occupied households in that table),
    # rather than the alternative share among only households with computable burdens.
    denom=df.renters_total.where(df.renters_total.gt(0))
    df['rent_burden_30p']=100*df.rent_burden_30_n/denom
    df['rent_burden_50p']=100*df.rent_burden_50_n/denom
    df['rent_afford']=df.MedianRent/df.MedianHouseholdIncome.where(df.MedianHouseholdIncome.gt(0))
    # This reproduces the original R's monthly-rent / annual-income ratio (NOT true annual rent burden).

    # Full study-area BG polygons: use all selected BGs, even if a BG is absent in some years.
    bgs=gpd.read_file(BG)
    geoidcol='GEOID' if 'GEOID' in bgs else 'GEOID20'
    bgs=bgs[[geoidcol,'geometry']].rename(columns={geoidcol:'bg_geoid_2020'})
    bgs['bg_geoid_2020']=norm_id(bgs.bg_geoid_2020)
    bgs=bgs[bgs.bg_geoid_2020.isin(df.bg_geoid_2020.unique())].copy()
    if bgs.bg_geoid_2020.duplicated().any():raise ValueError('Duplicate 2020 BG polygon IDs')
    missing_geo=set(df.bg_geoid_2020)-set(bgs.bg_geoid_2020)
    if missing_geo:raise ValueError(f'{len(missing_geo)} BGs not matched to Census polygons')
    bgs=bgs.to_crs('EPSG:5070')
    
    stations=select_annual_snapshots(pd.read_parquet(EV))
    needed_ev=['station_id','latitude','longitude','status_code','access_code','fuel_type_code','study_metro']
    missing_ev=set(needed_ev)-set(stations.columns)
    if missing_ev:raise KeyError('AFDC data missing '+str(missing_ev))
    for col in ['status_code','access_code','fuel_type_code']:
        stations[col]=stations[col].astype('string').str.strip().str.upper()
    stations=stations[(stations.status_code=='E')&(stations.access_code=='PUBLIC')&
                      (stations.fuel_type_code=='ELEC')&stations.study_metro.isin(METROS.values())].copy()
    for col in ['latitude','longitude']:
        stations[col]=pd.to_numeric(stations[col],errors='coerce')
    stations=stations.dropna(subset=['station_id','latitude','longitude'])
    stations=stations[stations.latitude.between(-90,90)&stations.longitude.between(-180,180)]
    stations=stations.drop_duplicates(['snapshot_year','station_id'])
    stations=gpd.GeoDataFrame(stations,geometry=gpd.points_from_xy(stations.longitude,stations.latitude),crs='EPSG:4326').to_crs('EPSG:5070')
    print(f'Filtered public active EVCS snapshots: {len(stations):,}',flush=True)
    # EPSG:5070 follows the original R panel builder, unlike the earlier EPSG:3857
    # preliminary random-forest dataset. Never mix their proximity variables.
    exposure_frames=[]
    for year in YEARS:
        yr=stations[stations.snapshot_year.eq(year)][['station_id','geometry']].copy()
        ex=bgs[['bg_geoid_2020']].copy()
        ex['Year']=year
        for tag,miles in TAGS.items():
            buffer=yr.copy()
            buffer['geometry']=buffer.geometry.buffer(miles*1609.344)
            joined=gpd.sjoin(bgs[['bg_geoid_2020','geometry']],buffer,
                             how='left',predicate='intersects')
            counts=joined.groupby('bg_geoid_2020')['station_id'].nunique()
            point_col=f'PointCount_{tag}'
            ex[point_col]=ex.bg_geoid_2020.map(counts).fillna(0).astype(int)
            ex[f'WithinBuffer_{tag}']=(ex[point_col]>0).astype(int)
        exposure_frames.append(pd.DataFrame(ex.drop(columns='geometry',errors='ignore')))
        print(f'Computed 0.5/1/2 mile polygon buffer exposures for {year}',flush=True)
    exposures=pd.concat(exposure_frames,ignore_index=True)
    df=df.merge(exposures,on=['bg_geoid_2020','Year'],how='left',validate='one_to_one')
    if df['PointCount_1mi'].isna().any():raise ValueError('Missing EV proximity after geometry join')
    df['GEO_ID']='1500000US'+df.bg_geoid_2020
    cols=['GEO_ID','bg_geoid_2020','Year','study_metro']+OUTCOMES+CONTROLS+['MedianHouseholdIncome','MedianRent','renters_total']
    for tag in TAGS:
        cols.extend([f'WithinBuffer_{tag}',f'PointCount_{tag}'])
    panel=df[cols].sort_values(['bg_geoid_2020','Year']).reset_index(drop=True)
    panel.to_parquet(out_dir/'analysis_panel_2017_2024.parquet',index=False)
    panel.to_csv(out_dir/'analysis_panel_2017_2024.csv',index=False)
    qc=panel.groupby(['study_metro','Year']).agg(N=('GEO_ID','size'),
        N_rent30=('rent_burden_30p','count'),N_rent50=('rent_burden_50p','count'),
        share_1mi=('WithinBuffer_1mi','mean'),median_count_1mi=('PointCount_1mi','median')).reset_index()
    qc.to_csv(out_dir/'sample_audit_by_year_metro.csv',index=False)
    print('\nSample and outcome coverage:\n'+qc.to_string(index=False),flush=True)
    return panel


def make_transforms(panel:pd.DataFrame, balanced=True):
    frame=panel.copy()
    frame['Year']=pd.to_numeric(frame.Year,errors='raise').astype(int)
    if frame.duplicated(['GEO_ID','Year']).any():raise ValueError('Duplicate group-year rows')
    if balanced:
        all_years=set(YEARS)
        valid=frame.groupby('GEO_ID')['Year'].agg(lambda s:set(s)==all_years)
        ids=set(valid[valid].index)
        frame=frame[frame.GEO_ID.isin(ids)].copy()
        print(f'Balanced {len(YEARS)}-year panel: {len(ids):,} BGs / {len(frame):,} rows',flush=True)
        if not len(ids):raise ValueError('No eight-year balanced BG panels')
    frame=frame.sort_values(['GEO_ID','Year']).reset_index(drop=True)
    g=frame.groupby('GEO_ID',sort=False)
    for tag in TAGS:
        col=f'PointCount_{tag}'
        frame[f'logPointCount_{tag}']=np.log1p(frame[col])
        frame[f'dPointCount_{tag}']=g[col].diff()
        frame[f'logNewAdds_{tag}']=np.log1p(frame[f'dPointCount_{tag}'].clip(lower=0))
    g=frame.groupby('GEO_ID',sort=False)
    for base in BASES:
        frame[f'{base}_lead2']=g[base].shift(-2)
        frame[f'{base}_lag2']=g[base].shift(2)
    if not balanced:
        # Shift(-2) on a missing-year panel is NOT necessarily a two-calendar-year lead.
        # Require actual 2-year steps whenever EV terms are observed.
        g=frame.groupby('GEO_ID',sort=False)
        lead_year=g.Year.shift(-2);lag_year=g.Year.shift(2)
        for base in BASES:
            frame.loc[lead_year.ne(frame.Year+2),f'{base}_lead2']=np.nan
            frame.loc[lag_year.ne(frame.Year-2),f'{base}_lag2']=np.nan
        for tag in TAGS:
            first_year=g.Year.shift(1)
            frame.loc[first_year.ne(frame.Year-1),f'dPointCount_{tag}']=np.nan
            frame.loc[first_year.ne(frame.Year-1),f'logNewAdds_{tag}']=np.nan
    return frame


def residualize_two_way(a:np.ndarray, gid:np.ndarray, tid:np.ndarray, tol=1e-9, max_iter=500):
    """FWL residualization against entity and year dummies, even for unbalanced listwise samples."""
    import scipy.sparse as sp
    n,q=a.shape
    one=np.ones(n)
    u=np.arange(n)
    eg=sp.csr_matrix((one,(gid,u)),shape=(gid.max()+1,n))
    et=sp.csr_matrix((one,(tid,u)),shape=(tid.max()+1,n))
    cg=np.asarray(eg@one).reshape(-1,1)
    ct=np.asarray(et@one).reshape(-1,1)
    r=a.copy()
    for iteration in range(max_iter):
        last=r.copy()
        r-=(eg@r/cg)[gid,:]
        r-=(et@r/ct)[tid,:]
        if np.max(np.abs(r-last))<tol:
            return r,eg,iteration+1
    raise RuntimeError(f'Two-way FE absorption did not converge in {max_iter} cycles')


def fit_twfe(frame, outcome, ev_base, income, timing):
    """OLS with absorbed BG+year FE and BG cluster-HC1 covariance.

    HC1 n/(n-k) where k is the number of non-absorbed slopes. Different plm
    finite-sample factor conventions may cause small numerical differences.
    """
    ev_terms=([f'{ev_base}_lead2',ev_base,f'{ev_base}_lag2']
              if timing=='leadlag_2' else [ev_base])
    xcols=ev_terms+CONTROLS+(['MedianHouseholdIncome'] if income else [])
    data=frame[['GEO_ID','Year',outcome]+xcols].copy()
    for c in [outcome]+xcols:
        data[c]=pd.to_numeric(data[c],errors='coerce')
    data=data.replace([np.inf,-np.inf],np.nan).dropna()
    n=len(data);groups=data.GEO_ID.nunique(); ny=data.Year.nunique()
    if n < len(xcols)+10 or groups<10 or ny<3:
        raise ValueError(f'insufficient complete cases (n={n}, groups={groups}, years={ny})')
    gid=pd.factorize(data.GEO_ID)[0]
    tid=pd.factorize(data.Year)[0]
    columns=[outcome]+xcols
    a=data[columns].to_numpy(dtype=float)
    r,eg,iterations=residualize_two_way(a,gid,tid)
    y=r[:,0];X=r[:,1:]
    # Block groups with no within-year residual variation -> coefficient not identified.
    xnorm=np.linalg.norm(X,axis=0)
    ok=xnorm>1e-8
    remaining=np.flatnonzero(ok)
    if len(remaining)==0:raise ValueError('No identifiable regressors after two-way FE')
    rankselect=[]
    for index in remaining:
        proposal=rankselect+[index]
        if np.linalg.matrix_rank(X[:,proposal],tol=1e-9)==len(proposal):
            rankselect=proposal
    if not rankselect:raise ValueError('Singular after two-way FE')
    x=X[:,rankselect]
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    e=y-x@beta
    xtxi=np.linalg.pinv(x.T@x)
    # Clustered meat: one score per BG, sum within BG before outer product.
    meat_scores=eg@(x*e[:,None])
    meat=meat_scores.T@meat_scores
    k=x.shape[1]
    cov=xtxi@meat@xtxi*(n/(n-k))
    se=np.sqrt(np.maximum(np.diag(cov),0))
    from scipy.stats import t as tdist
    stat=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    # lmtest::coeftest(plm) uses plm's model residual degrees of freedom.
    # For two-way FE on a connected panel, df = n - (G + T - 1) - k.
    residual_df=n-groups-ny+1-k
    if residual_df<=0: raise ValueError('Nonpositive two-way FE residual degrees of freedom')
    p=2*tdist.sf(np.abs(stat),df=residual_df)
    by_idx={i:(float(b),float(s),float(pv)) for i,b,s,pv in zip(rankselect,beta,se,p)}
    rows=[]
    for evterm in ev_terms:
        idx=xcols.index(evterm)
        b,s,pv=by_idx.get(idx,(np.nan,np.nan,np.nan))
        rows.append(dict(term=evterm,estimate=b,std_error=s,p_value=pv,
                         significant_05=bool(pv<0.05) if np.isfinite(pv) else False,
                         conf_low=b-1.96*s,conf_high=b+1.96*s))
    diagn=dict(N=n, N_groups=groups, N_years=ny, N_slopes=k,
               dof_residual=residual_df,absorption_iterations=iterations,
               removed_collinear=';'.join([c for i,c in enumerate(xcols) if i not in rankselect]),
               R2_within=1-float(e@e)/(float(y@y) if y@y>0 else np.nan))
    return rows,diagn


def write_access_fe_diagnostics(frame: pd.DataFrame, out_dir: Path, scope: str, suffix: str):
    """Replicates FE-only fit share and BG switching patterns for 1-mile access."""
    rows=[]; sw=[]
    for measure in ['WithinBuffer_1mi','PointCount_1mi']:
        d=frame[['GEO_ID','Year',measure]].dropna().copy()
        if d.empty: continue
        g=pd.factorize(d.GEO_ID)[0]; t=pd.factorize(d.Year)[0]
        a=d[[measure]].to_numpy(dtype=float)
        r,_,_=residualize_two_way(a,g,t)
        mean=float(a.mean()); sst=float(np.sum((a-mean)**2))
        rsq=1-float(np.sum(r**2))/sst if sst>0 else np.nan
        rows.append({'Scope':scope,'Variable':measure,'FE_only_R2':rsq,
                     'N':len(d),'N_groups':d.GEO_ID.nunique(),
                     'N_years':d.Year.nunique()})
    g=frame.groupby('GEO_ID')['WithinBuffer_1mi']
    stat=g.agg(['min','max']).dropna()
    never=int(stat['max'].eq(0).sum())
    always=int(stat['min'].eq(1).sum())
    switcher=int((stat['min'].eq(0)&stat['max'].eq(1)).sum())
    sw.append({'Scope':scope,'Groups':len(stat),'Never':never,'Always':always,
               'Switchers':switcher,'SwitcherShare':switcher/len(stat) if len(stat) else np.nan})
    pd.DataFrame(rows).to_csv(out_dir/f'access_fe_only_r2_{scope.lower()}_{suffix}.csv',index=False)
    pd.DataFrame(sw).to_csv(out_dir/f'access_switcher_groups_{scope.lower()}_{suffix}.csv',index=False)


def run_models(out_dir:Path, scopes:list[str], balanced=True, modes:list[str]|None=None):
    modes=modes or ['baseline','leadlag_2']
    src=required_path(out_dir/'analysis_panel_2017_2024.parquet')
    panel=pd.read_parquet(src)
    panel=make_transforms(panel, balanced=balanced)
    all_rows=[]; diagnostics=[]
    for scope in scopes:
        sub=panel if scope=='Pooled' else panel[panel.study_metro.eq(METROS[scope])]
        if sub.empty:raise ValueError(f'No rows for scope {scope}')
        write_access_fe_diagnostics(sub, out_dir, scope, 'balanced' if balanced else 'unbalanced')
        for outcome in OUTCOMES:
            for ev_base in BASES:
                for income in (False,True):
                    for mode in modes:
                        ident=f'{scope}__{outcome}__{ev_base}__'+('withIncome' if income else 'noIncome')+'__'+mode
                        ev_terms=([f'{ev_base}_lead2',ev_base,f'{ev_base}_lag2']
                                  if mode=='leadlag_2' else [ev_base])
                        try:
                            rows,diagn=fit_twfe(sub,outcome,ev_base,income,mode)
                            status='ok';error=''
                        except Exception as exc:
                            rows=[dict(term=t,estimate=np.nan,std_error=np.nan,p_value=np.nan,
                                       significant_05=False,conf_low=np.nan,conf_high=np.nan) for t in ev_terms]
                            diagn={};status='FAILED';error=str(exc)
                        base_info=dict(Scope=scope,dv=outcome,ev_base=ev_base,
                                       include_income=income,dynamic=mode,model_id=ident,
                                       effect='twoways',status=status)
                        diagnostics.append({**base_info,**diagn,'error':error})
                        for row in rows:
                            all_rows.append({**base_info,**row})
                        if mode=='leadlag_2':
                            print(f'{scope:12s} {outcome:16s} {ev_base:26s} income={income} {status}',flush=True)
    results=pd.DataFrame(all_rows)
    diag=pd.DataFrame(diagnostics)
    suffix='_'.join(s.lower() for s in scopes)
    bal_name='balanced' if balanced else 'unbalanced'
    stem=f'{suffix}_{bal_name}_2017_2024'
    results.to_csv(out_dir/f'twfe_all_ev_terms_{stem}.csv',index=False)
    diag.to_csv(out_dir/f'twfe_model_diagnostics_{stem}.csv',index=False)
    results[(results.dynamic=='baseline')].to_csv(out_dir/f'twfe_baseline_results_{stem}.csv',index=False)
    dynamic=results[results.dynamic.eq('leadlag_2')].copy()
    dynamic['timing']=np.select([dynamic.term.str.endswith('_lead2'),dynamic.term.str.endswith('_lag2')],
                               ['Lead (t+2)','Lag (t-2)'],default='Current')
    dynamic.to_csv(out_dir/f'twfe_timing_long_{stem}.csv',index=False)
    wide=dynamic.pivot(index=['Scope','dv','ev_base','include_income','model_id'],
                        columns='timing',values=['estimate','std_error','p_value'])
    wide.columns=[f'{metric}__{timing}' for metric,timing in wide.columns]
    wide.reset_index().to_csv(out_dir/f'twfe_timing_wide_{stem}.csv',index=False)
    results[results.significant_05].to_csv(out_dir/f'twfe_significant_{stem}.csv',index=False)
    print('\nSuccessful models:',int(diag.status.eq('ok').sum()),'of',len(diag))
    print('Results:',out_dir)
    return results,diag


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','prepare','models'],default='all')
    p.add_argument('--scopes',nargs='+',choices=['Pooled','DMV','Philadelphia'],default=['Pooled'])
    p.add_argument('--unbalanced',action='store_true',help='Optional sensitivity (balanced is original final R default)')
    p.add_argument('--modes',nargs='+',choices=['baseline','leadlag_2'],default=['baseline','leadlag_2'])
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    if args.stage in ['all','prepare']:
        make_panel(args.output)
    if args.stage in ['all','models']:
        run_models(args.output,args.scopes,balanced=not args.unbalanced,modes=args.modes)

if __name__=='__main__': main()
