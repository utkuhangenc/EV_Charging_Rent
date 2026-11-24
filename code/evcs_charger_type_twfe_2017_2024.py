#!/usr/bin/env python3
"""EVCS charger-type sensitivity: strictly preserve the published annual TWFE panel/model.

Three new station subsets of the ORIGINAL operating PUBLIC AFDC station universe:
  FastPublic        ev_dc_fast_count > 0
  PublicNonDealer   facility_type != CAR_DEALER (unknown facility types retained and audited)
  FastNonDealer     intersection of the above.

DC fast *stations* counted; not DC-fast ports. Every retained station counts once per BG.
Proximity is polygon intersects buffered station, EPSG:5070 (same as existing panel).

The comparison all-public station count must reproduce every stored PointCount across
all 2017-2024 BG-years; otherwise exit BEFORE any models are estimated.

For annual models, call the EXISTING fit_twfe function without changing model controls,
BG/year fixed effects, cluster covariance, or model sample. Balanced 2017-2024 default.
Optional nonoverlapping 2017->2024 first-difference sensitivity follows prior design.

On Gautschi, keep this file alongside
  rebuild_evcs_panel_2017_2024_b25070_diagnostic.py
and, for optional first differences, two_period_twfe_evcs_2017_2024.py.

Examples:
 python evcs_charger_type_twfe_2017_2024.py --stage audit
 python evcs_charger_type_twfe_2017_2024.py --stage models --distances 1mi
 python evcs_charger_type_twfe_2017_2024.py --stage models --distances 0_5mi 1mi 2mi
 python evcs_charger_type_twfe_2017_2024.py --stage two-period --distances 1mi
 python evcs_charger_type_twfe_2017_2024.py --stage models --timings baseline leadlag_2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import numpy as np
import pandas as pd

from rebuild_evcs_panel_2017_2024_b25070_diagnostic import (
    fit_twfe, select_annual_snapshots, norm_id, CONTROLS, METROS, OUTCOMES,
    YEARS, TAGS, EV, BG, OUT as PANEL_DIR,
)

SOURCE_PANEL = PANEL_DIR / 'analysis_panel_2017_2024.parquet'
OUTPUT = PANEL_DIR / 'charger_type_twfe_2017_2024'
CATEGORIES = {
    'FastPublic': 'is_fast',
    'PublicNonDealer': 'is_nondealer',
    'FastNonDealer': 'is_fast_nondealer',
}
KINDS = ('WithinBuffer', 'PointCount')


def load_panel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Prepared annual TWFE panel not found: {path}')
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'GEO_ID': 'string'}, low_memory=False)
    need = {'GEO_ID', 'Year', 'study_metro', *OUTCOMES, *CONTROLS,
            'MedianHouseholdIncome', 'PointCount_0_5mi', 'PointCount_1mi', 'PointCount_2mi'}
    if missing := need-set(df.columns):
        raise KeyError(f'Existing prepared panel missing: {sorted(missing)}')
    df['GEO_ID'] = df.GEO_ID.astype('string')
    df['bg_geoid_2020'] = norm_id(df.GEO_ID)
    df['Year'] = pd.to_numeric(df.Year, errors='raise').astype(int)
    df = df[df.study_metro.isin(METROS.values()) & df.Year.isin(YEARS)].copy()
    if df.duplicated(['bg_geoid_2020', 'Year']).any():
        raise ValueError('Duplicate BG-year in prepared panel')
    if df.groupby('bg_geoid_2020').study_metro.nunique().gt(1).any():
        raise ValueError('Block group changes metropolitan area')
    return df


def prepare_stations(ev_path: Path, output: Path) -> pd.DataFrame:
    if not ev_path.is_file():
        raise FileNotFoundError(f'AFDC file not found: {ev_path}')
    raw = pd.read_parquet(ev_path)
    required = {'station_id','latitude','longitude','status_code','access_code',
                'fuel_type_code','study_metro','ev_dc_fast_count','facility_type'}
    if missing := required-set(raw.columns):
        raise KeyError('AFDC master missing necessary charger-type fields: '+str(sorted(missing)))
    stations = select_annual_snapshots(raw)
    for col in ('status_code','access_code','fuel_type_code'):
        stations[col]=stations[col].astype('string').str.strip().str.upper()
    stations = stations.loc[(stations.status_code=='E') &
                            (stations.access_code=='PUBLIC') &
                            (stations.fuel_type_code=='ELEC') &
                            stations.study_metro.isin(METROS.values())].copy()
    for col in ['latitude','longitude','ev_dc_fast_count']:
        stations[col]=pd.to_numeric(stations[col],errors='coerce')
    stations=stations.dropna(subset=['station_id','latitude','longitude'])
    stations=stations.loc[stations.latitude.between(-90,90)&stations.longitude.between(-180,180)]
    # Mirror original station-level filtering and duplicate logic.
    stations=stations.drop_duplicates(['snapshot_year','station_id']).copy()
    ft=(stations.facility_type.astype('string').str.strip().str.upper()
        .str.replace(r'[\s\-]+','_',regex=True).replace({'':'<UNKNOWN>'}))
    stations['facility_type_norm']=ft
    stations['facility_unknown']=ft.isna()|ft.eq('<UNKNOWN>')
    stations['is_dealer']=ft.eq('CAR_DEALER').fillna(False)
    stations['is_fast']=stations.ev_dc_fast_count.gt(0).fillna(False)
    stations['fast_unknown']=stations.ev_dc_fast_count.isna()
    stations['is_nondealer']=~stations.is_dealer
    stations['is_fast_nondealer']=stations.is_fast & stations.is_nondealer
    if 'station_name' in stations.columns:
        # Audit only: station names are NOT used in primary treatment definitions.
        names=stations.station_name.astype('string').str.upper().fillna('')
        stations['dealer_name_flag']=(names.str.contains(r'\b(?:DEALERSHIP|AUTO DEALER|CAR DEALER)\b',regex=True) &
                                     ~stations.is_dealer)
    else:
        stations['dealer_name_flag']=False
    cols=[]
    for (year, metro), ss in stations.groupby(['snapshot_year','study_metro']):
        cols.append(dict(year=int(year),study_metro=metro,
            n_all_public=len(ss),n_fast_public=int(ss.is_fast.sum()),
            n_dealer=int(ss.is_dealer.sum()),
            n_public_nondealer=int(ss.is_nondealer.sum()),
            n_fast_nondealer=int(ss.is_fast_nondealer.sum()),
            n_missing_facility_type=int(ss.facility_unknown.sum()),
            n_missing_fast_port_count=int(ss.fast_unknown.sum()),
            n_name_indicates_dealer_but_facility_not=int(ss.dealer_name_flag.sum())))
    audit=pd.DataFrame(cols).sort_values(['year','study_metro'])
    audit.to_csv(output/'station_classification_audit.csv',index=False)
    print('\nSTATION CATEGORY AUDIT (public, active, electric only):')
    print(audit.to_string(index=False),flush=True)
    fac=stations.groupby(['snapshot_year','study_metro','facility_type_norm'],dropna=False).size().rename('stations').reset_index()
    fac.to_csv(output/'facility_type_counts_by_year_metro.csv',index=False)
    flagged=stations.loc[stations.dealer_name_flag,
           [c for c in ['snapshot_year','study_metro','station_id','station_name','facility_type_norm'] if c in stations]]
    flagged.to_csv(output/'review_possible_dealers_misclassified.csv',index=False)
    unknown_rate=float(stations.facility_unknown.mean()) if len(stations) else np.nan
    print(f'Unknown facility_type in {unknown_rate:.1%} of public station-snapshots; '
          'these are retained in PublicNonDealer unless explicitly coded CAR_DEALER.',flush=True)
    fastmiss=float(stations.fast_unknown.mean()) if len(stations) else np.nan
    print(f'Missing ev_dc_fast_count in {fastmiss:.1%} of public station-snapshots; '
          'these are NOT assumed to have DC-fast access.',flush=True)
    return stations


def create_exposures(panel:pd.DataFrame, ev_path:Path, shp_path:Path, output:Path):
    import geopandas as gpd
    output.mkdir(parents=True,exist_ok=True)
    bgs=gpd.read_file(shp_path)
    key='GEOID' if 'GEOID' in bgs else 'GEOID20'
    if key not in bgs: raise KeyError('BG polygons must have GEOID or GEOID20')
    bgs=bgs[[key,'geometry']].rename(columns={key:'bg_geoid_2020'}).copy()
    bgs['bg_geoid_2020']=norm_id(bgs.bg_geoid_2020)
    bgs=bgs.loc[bgs.bg_geoid_2020.isin(panel.bg_geoid_2020.unique())].copy()
    if bgs.bg_geoid_2020.duplicated().any() or len(bgs)!=panel.bg_geoid_2020.nunique():
        raise ValueError('BG geometries are missing or duplicated')
    bgs=bgs.to_crs('EPSG:5070')
    st=prepare_stations(ev_path,output)
    if st.empty: raise ValueError('AFDC contains no valid public operating stations')
    st=gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.longitude,st.latitude),
                        crs='EPSG:4326').to_crs('EPSG:5070')
    outputs=[]
    for year in YEARS:
        yr=st.loc[st.snapshot_year.eq(year),
            ['station_id','geometry',*CATEGORIES.values()]].copy()
        out=pd.DataFrame({'bg_geoid_2020':bgs.bg_geoid_2020.to_numpy(),'Year':year})
        for tag,miles in TAGS.items():
            buffers=yr.copy()
            buffers['geometry']=buffers.geometry.buffer(miles*1609.344)
            joined=gpd.sjoin(bgs[['bg_geoid_2020','geometry']],buffers,
                             how='inner',predicate='intersects')
            base=joined.groupby('bg_geoid_2020')['station_id'].nunique()
            out[f'PointCount_AllPublic_{tag}']=out.bg_geoid_2020.map(base).fillna(0).astype('int32')
            for category,flag in CATEGORIES.items():
                subset=joined.loc[joined[flag].fillna(False)]
                counts=subset.groupby('bg_geoid_2020')['station_id'].nunique()
                cnt=f'PointCount_{category}_{tag}'
                out[cnt]=out.bg_geoid_2020.map(counts).fillna(0).astype('int32')
                out[f'WithinBuffer_{category}_{tag}']=out[cnt].gt(0).astype('int8')
                if (out[cnt]>out[f'PointCount_AllPublic_{tag}']).any():
                    raise RuntimeError(f'{category} has >all public count: {tag}, year={year}')
        outputs.append(out)
        print(f'Computed station-type exposures {year}: {len(yr):,} all-public stations, '
              f'{len(out):,} BGs, 3 distances',flush=True)
    exposure=pd.concat(outputs,ignore_index=True)
    checked=panel[['bg_geoid_2020','Year']+[f'PointCount_{t}' for t in TAGS]].merge(
        exposure[['bg_geoid_2020','Year']+[f'PointCount_AllPublic_{t}' for t in TAGS]],
        on=['bg_geoid_2020','Year'],how='left',validate='one_to_one')
    failed=[]
    for tag in TAGS:
        a=pd.to_numeric(checked[f'PointCount_{tag}'],errors='coerce')
        b=pd.to_numeric(checked[f'PointCount_AllPublic_{tag}'],errors='coerce')
        bad=(a.isna() | b.isna() | a.ne(b))
        if bad.any():
            d=checked.loc[bad,['bg_geoid_2020','Year',f'PointCount_{tag}',f'PointCount_AllPublic_{tag}']].copy()
            d['distance']=tag
            failed.append(d)
    if failed:
        pd.concat(failed,ignore_index=True).to_csv(output/'FAILED_original_exposure_reconstruction.csv',index=False)
        raise ValueError('New spatial matching does not exactly reproduce original all-public '
                         'station counts. See FAILED_original_exposure_reconstruction.csv. NO MODELS FIT.')
    print(f'PASS: All-public exposures agree exactly with {len(panel):,} existing BG-year rows at all 3 distances',flush=True)
    exposure.to_parquet(output/'charger_type_exposures_2017_2024.parquet',index=False)
    enriched=panel.merge(exposure.drop(columns=[f'PointCount_AllPublic_{t}' for t in TAGS]),
                         on=['bg_geoid_2020','Year'],how='left',validate='one_to_one')
    enriched.to_parquet(output/'panel_charger_types_2017_2024.parquet',index=False)
    audit=[]
    for (year,metro), df in enriched.groupby(['Year','study_metro']):
        for category in CATEGORIES:
            for tag in TAGS:
                c=f'PointCount_{category}_{tag}'
                audit.append(dict(Year=year,study_metro=metro,category=category,distance=tag,
                                  N=len(df),share_with_access=float(df[c].gt(0).mean()),
                                  mean_stations=float(df[c].mean()),
                                  median_stations=float(df[c].median()),
                                  n_with_access=int(df[c].gt(0).sum())))
    pd.DataFrame(audit).to_csv(output/'exposure_coverage_by_year_metro.csv',index=False)
    return enriched


def load_extended_panel(output:Path):
    path=output/'panel_charger_types_2017_2024.parquet'
    if not path.is_file():
        raise FileNotFoundError(f'Run --stage audit first. Missing {path}')
    p=pd.read_parquet(path)
    if p.duplicated(['GEO_ID','Year']).any(): raise ValueError('Duplicate BG-year')
    return p


def balanced_sample(df:pd.DataFrame):
    nyr=len(YEARS)
    n=df.groupby('GEO_ID').Year.nunique()
    chosen=n[n.eq(nyr)].index
    panel=df.loc[df.GEO_ID.isin(chosen)].copy()
    # Guard: every BG truly present for exactly the intended eight years.
    chk=panel.groupby('GEO_ID').Year.agg(lambda x:set(x)==set(YEARS))
    if not chk.all():raise ValueError('Unbalanced years in selected sample')
    print(f'Original balanced 2017-2024 panel: {len(chosen):,} BGs; {len(panel):,} BG-years',flush=True)
    return panel.sort_values(['GEO_ID','Year']).reset_index(drop=True)


def exposures_to_fit(categories, distances):
    return [(cat,kind,tag,f'{kind}_{cat}_{tag}')
            for cat in categories for kind in KINDS for tag in distances]



def audit_model_sample_access(p:pd.DataFrame, output:Path, scopes, categories, distances):
    """Station-type exposure switching on EXACT eight-year balanced TWFE sample."""
    rows=[]
    for scope in scopes:
        d=p if scope=='Pooled' else p.loc[p.study_metro.eq(METROS[scope])]
        for category in categories:
            for tag in distances:
                c=f'PointCount_{category}_{tag}'
                wide=d.pivot(index='GEO_ID',columns='Year',values=c)
                start=wide[2017];end=wide[2024]
                a0=start.gt(0);a1=end.gt(0)
                switches=(wide.gt(0).nunique(axis=1)>1)
                rows.append(dict(Scope=scope,category=category,distance=tag,
                    N_BGs=len(wide),initial_access_share=float(a0.mean()),
                    final_access_share=float(a1.mean()),
                    gain_access=int((~a0 & a1).sum()),
                    loss_access=int((a0 & ~a1).sum()),
                    always_at_endpoints=int((a0 & a1).sum()),
                    never_at_endpoints=int((~a0 & ~a1).sum()),
                    switchers_any_year=int(switches.sum()),
                    start_mean_stations=float(start.mean()),
                    end_mean_stations=float(end.mean()),
                    N_increased_station_count=int(end.gt(start).sum()),
                    N_decreased_station_count=int(end.lt(start).sum())))
    result=pd.DataFrame(rows)
    result.to_csv(output/'balanced_model_sample_access_switchers.csv',index=False)
    print('Balanced-sample exposure-switcher diagnostics saved.',flush=True)
    return result


def run_annual(panel, output, scopes, categories, distances, timings):
    p=balanced_sample(panel)
    audit_model_sample_access(p,output,scopes,categories,distances)
    bases=exposures_to_fit(categories,distances)
    if 'leadlag_2' in timings:
        grouped=p.groupby('GEO_ID',sort=False)
        for _,_,_,col in bases:
            p[f'{col}_lead2']=grouped[col].shift(-2)
            p[f'{col}_lag2']=grouped[col].shift(2)
    results=[]; diagnostics=[]
    for scope in scopes:
        sub=p if scope=='Pooled' else p.loc[p.study_metro.eq(METROS[scope])]
        for category,kind,tag,col in bases:
            for outcome in OUTCOMES:
                for income in (False,True):
                    for mode in timings:
                        ident=f'{scope}__{outcome}__{col}__{int(income)}__{mode}'
                        common=dict(Scope=scope,category=category,exposure_kind=kind,
                                    distance=tag,dv=outcome,ev_base=col,
                                    include_income=income,dynamic=mode,model_id=ident)
                        terms=[f'{col}_lead2',col,f'{col}_lag2'] if mode=='leadlag_2' else [col]
                        try:
                            rows,diag=fit_twfe(sub,outcome,col,income,mode)
                            current=next((z for z in rows if z['term']==col),None)
                            if current is None or not np.isfinite(current['estimate']):
                                raise ValueError('EV exposure term not identified after TWFE absorption')
                            status='ok'; error=''
                        except Exception as exc:
                            rows=[dict(term=x,estimate=np.nan,std_error=np.nan,p_value=np.nan,
                                       significant_05=False,conf_low=np.nan,conf_high=np.nan) for x in terms]
                            diag={}; status='FAILED';error=str(exc)
                        results.extend({**common,**r,'status':status} for r in rows)
                        diagnostics.append({**common,**diag,'status':status,'error':error})
        n_ok=sum(x['status']=='ok' for x in diagnostics if x['Scope']==scope)
        n_all=sum(x['Scope']==scope for x in diagnostics)
        print(f'{scope}: successful annual TWFE specifications {n_ok}/{n_all}',flush=True)
    r=pd.DataFrame(results);d=pd.DataFrame(diagnostics)
    r.to_csv(output/'annual_twfe_charger_type_all_coefficients.csv',index=False)
    d.to_csv(output/'annual_twfe_charger_type_model_diagnostics.csv',index=False)
    main=r.loc[(r.dynamic=='baseline')&(r.distance=='1mi')&(r.term==r.ev_base)].copy()
    main.to_csv(output/'annual_twfe_charger_type_1mi_baseline.csv',index=False)
    compare_against_saved_all_public(r, d, output)
    if (d.status!='ok').any():
        print('WARNING: annual model failures:',d.status.ne('ok').sum(),flush=True)
        print(d.loc[d.status!='ok',['Scope','ev_base','dv','include_income','dynamic','error']].to_string(index=False),flush=True)
    else:
        print(f'PASS: {len(d)} annual TWFE specifications completed',flush=True)
    print('\nONE MILE, INCOME-ADJUSTED BASELINE:')
    display=main.loc[main.include_income,
                     ['Scope','category','dv','ev_base','estimate','std_error','p_value']]
    print(display.to_string(index=False, float_format=lambda x: f'{x:.6g}'),flush=True)
    return r,d



def compare_against_saved_all_public(new_rows:pd.DataFrame, diagnostics:pd.DataFrame, output:Path):
    """Place original public charging baseline and new categories into ONE table.

    The old coefficients are read, not refitted. Original files are optional; if
    absent we still write the charger-type estimates, but warn about missing old baseline.
    """
    old_files=[PANEL_DIR/'twfe_all_ev_terms_pooled_balanced_2017_2024.csv',
               PANEL_DIR/'twfe_all_ev_terms_dmv_philadelphia_balanced_2017_2024.csv']
    pieces=[]
    for f in old_files:
        if f.exists():
            z=pd.read_csv(f)
            z=z.loc[z.dynamic.eq('baseline') & z.ev_base.str.fullmatch(
                r'(?:WithinBuffer|PointCount)_(?:0_5mi|1mi|2mi)') & z.term.eq(z.ev_base)].copy()
            z['category']='AllPublic'
            z['exposure_kind']=z.ev_base.str.extract(r'^(WithinBuffer|PointCount)_')[0]
            z['distance']=z.ev_base.str.extract(r'_(0_5mi|1mi|2mi)$')[0]
            pieces.append(z)
        else:
            print(f'Optional original result missing (comparison omits it): {f}',flush=True)
    new=new_rows.loc[new_rows.dynamic.eq('baseline') & new_rows.term.eq(new_rows.ev_base)].copy()
    sample=diagnostics[['model_id','N','N_groups']].drop_duplicates()
    new=new.merge(sample,on='model_id',how='left',validate='many_to_one')
    fields=['Scope','category','exposure_kind','distance','dv','include_income',
            'estimate','std_error','p_value','N','N_groups','status']
    for p in pieces+[new]:
        for field in fields:
            if field not in p:p[field]=np.nan
    both=pd.concat(pieces+[new],ignore_index=True)[fields].sort_values(
        ['dv','distance','exposure_kind','Scope','include_income','category'])
    both.to_csv(output/'comparison_all_public_vs_charger_types.csv',index=False)
    print(f'Comparison saved: {output/"comparison_all_public_vs_charger_types.csv"}',flush=True)
    return both


def run_two_period(panel, output, scopes, categories, distances):
    # Depend on the already-approved first-difference estimator to preserve its controls,
    # model terms, HC1 SE and pooled-metro trend implementation.
    from two_period_twfe_evcs_2017_2024 import pair_data, fit_first_difference
    results=[]; paired_n=[]
    for cat in categories:
        p=panel.copy()
        for tag in TAGS:
            p[f'PointCount_{tag}']=p[f'PointCount_{cat}_{tag}']
            p[f'WithinBuffer_{tag}']=p[f'WithinBuffer_{cat}_{tag}']
            p[f'logPointCount_{tag}']=np.log1p(p[f'PointCount_{cat}_{tag}'])
        paired=pair_data(p,2017,2024)
        for scope in scopes:
            frame=paired if scope.startswith('Pooled') else paired.loc[paired.study_metro.eq(METROS[scope])]
            paired_n.append(dict(category=cat,Scope=scope,matched_BGs=len(frame)))
            for dv in OUTCOMES:
                for tag in distances:
                    for kind in ('WithinBuffer','PointCount','logPointCount'):
                        exp=f'{kind}_{tag}'
                        for inc in (False,True):
                            d,_=fit_first_difference(frame,scope,dv,exp,inc,
                                                     metro_trends=(scope=='Pooled_MetroTrends'))
                            d.update(category=cat,distance=tag,exposure_kind=kind,
                                     original_exposure=f'{kind}_{cat}_{tag}',
                                     pair='2017-2024')
                            results.append(d)
        print(f'Completed two-period models for {cat}',flush=True)
    out=pd.DataFrame(results)
    out.to_csv(output/'two_period_2017_2024_charger_type_results.csv',index=False)
    pd.DataFrame(paired_n).to_csv(output/'two_period_matched_BG_counts.csv',index=False)
    print(f'Two-period results: {int(out.status.eq("ok").sum())}/{len(out)} models OK.',flush=True)
    if out.status.ne('ok').any():
        print(out.loc[out.status.ne('ok'),['Scope','category','original_exposure','outcome','error']].to_string(index=False))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['audit','models','two-period','all'],default='all')
    ap.add_argument('--source-panel',type=Path,default=SOURCE_PANEL)
    ap.add_argument('--afdc',type=Path,default=EV)
    ap.add_argument('--bg-shp',type=Path,default=BG)
    ap.add_argument('--output',type=Path,default=OUTPUT)
    ap.add_argument('--scopes',nargs='+',choices=['Pooled','DMV','Philadelphia','Pooled_MetroTrends'],
                    default=['Pooled','DMV','Philadelphia'])
    ap.add_argument('--categories',nargs='+',choices=list(CATEGORIES),default=list(CATEGORIES))
    ap.add_argument('--distances',nargs='+',choices=list(TAGS),default=['1mi'])
    ap.add_argument('--timings',nargs='+',choices=['baseline','leadlag_2'],default=['baseline'])
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ('audit','all'):
        panel=load_panel(args.source_panel)
        create_exposures(panel,args.afdc,args.bg_shp,args.output)
    if args.stage in ('models','all'):
        if 'Pooled_MetroTrends' in args.scopes:
            raise ValueError('Pooled_MetroTrends is a two-period-only scope. '
                             'For annual model, use --scopes Pooled DMV Philadelphia.')
        run_annual(load_extended_panel(args.output),args.output,args.scopes,
                   args.categories,args.distances,args.timings)
    if args.stage=='two-period':
        scopes=list(args.scopes)
        if 'Pooled' in scopes and 'Pooled_MetroTrends' not in scopes:
            scopes.append('Pooled_MetroTrends')
        run_two_period(load_extended_panel(args.output),args.output,
                       scopes,args.categories,args.distances)

if __name__=='__main__':
    main()
