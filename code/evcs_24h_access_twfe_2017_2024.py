#!/usr/bin/env python3
"""24/7 public-access EVCS sensitivity: preserve existing annual TWFE panel/model.

Four subsets of the ORIGINAL operating PUBLIC AFDC station universe:
  Open24hPublic            publicly listed stations explicitly operating 24/7
  Open24hFast              24/7 public locations with EV DC fast ports
  Open24hNonDealer         24/7 public locations without explicit dealer coding
  Open24hFastNonDealer     24/7 public fast locations without explicit dealer coding.

Strict filter: 24 hours daily/24-7 or complete seven-day 24-hour schedule.
Near-24h schedules (e.g. 12am-11:58pm) and conflicting descriptions are audited
but excluded. Opening hours do NOT establish unrestricted public usability.

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
 python evcs_24h_access_twfe_2017_2024.py --stage audit
 python evcs_24h_access_twfe_2017_2024.py --stage models --distances 1mi
 python evcs_24h_access_twfe_2017_2024.py --stage models --distances 0_5mi 1mi 2mi
 python evcs_24h_access_twfe_2017_2024.py --stage two-period --distances 1mi
 python evcs_24h_access_twfe_2017_2024.py --stage models --timings baseline leadlag_2
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
OUTPUT = PANEL_DIR / 'evcs_24h_access_twfe_2017_2024'
CATEGORIES = {
    'Open24hPublic': 'is_24h',
    'Open24hFast': 'is_24h_fast',
    'Open24hNonDealer': 'is_24h_nondealer',
    'Open24hFastNonDealer': 'is_24h_fast_nondealer',
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


# The three day-name variations encountered in AFDC historical CSVs are
# handled explicitly. Only a complete 7-day, all-24-hours schedule qualifies.
import re
DAY_MAP = {
    'MO': 'mon', 'MON': 'mon', 'MONDAY': 'mon',
    'TU': 'tue', 'TUE': 'tue', 'TUESDAY': 'tue',
    'WE': 'wed', 'WED': 'wed', 'WEDNESDAY': 'wed',
    'TH': 'thu', 'THU': 'thu', 'THURSDAY': 'thu',
    'FR': 'fri', 'FRI': 'fri', 'FRIDAY': 'fri',
    'SA': 'sat', 'SAT': 'sat', 'SATURDAY': 'sat',
    'SU': 'sun', 'SUN': 'sun', 'SUNDAY': 'sun',
}
DAYS = set(DAY_MAP.values())
DAY_TOKEN = re.compile(r'^([a-z]+)\s*:\s*(.+)$', flags=re.I)
DAY_FULL = re.compile(r'^24\s*(?:hours?|hrs?)\s*$', flags=re.I)
DAY_NEAR = re.compile(r'^12\s*:\s*00\s*am\s*[-–]\s*11\s*:\s*5[89]\s*pm$',flags=re.I)
DAILY_24 = re.compile(
    r'^(?:open\s+)?24\s*(?:hours?|hrs?)\s*(?:daily|every\s*day|7\s*days\s*a\s*week|a\s*day\s*,?\s*7\s*days\s*a\s*week)\b|'
    r'^(?:open\s+)?24\s*[/x]\s*7\b|'
    r'^24\s*(?:hours?|hrs?)\s*$',flags=re.I)
TIME_RANGE = re.compile(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b',flags=re.I)
CONFLICT = re.compile(
    r'\b(?:limited\s+access|restricted\s+to|business\s+hours|garage\s+hours|'
    r'not\s+available|closed\s+(?:at|after|on|during)|except\s+(?:on|during)|'
    r'one\s+station\s+open|only\s+open|no\s+overnight\s+parking)\b',flags=re.I)
# These restrictions do not necessarily change the operating HOURS. We
# nevertheless flag them because 24-hour access != universal public usability.
CONDITION = re.compile(
    r'\b(?:guest(?:s)?\s+(?:use\s+)?only|hotel\s+guest|customer\s+use\s+only|'
    r'for\s+tesla\s+use\s+only|employee|member(?:s|ship)?\s+only|'
    r'parking\s+permit\s+required|permit\s+required|parking\s+pass\s+required|'
    r'monthly\s+contract(?:ed)?|volunteer\s+or\s+member)\b',flags=re.I)


def classify_hours(value):
    """Return (category, restriction_flag) from AFDC Access Days Time text.

    Categories: confirmed_24h, near_24h, conflicting_24h,
                not_24h, unspecified. Missing never implies 24h.
    """
    if pd.isna(value):
        return 'unspecified', False
    raw=re.sub(r'\s+', ' ', str(value).strip())
    if not raw or raw.lower() in ('nan','none','unknown','not specified','n/a'):
        return 'unspecified', False
    condition=bool(CONDITION.search(raw))
    # Complete structured schedule: recognize AFDC MON/TUE ... | and MO/TU ... ;
    pieces=re.split(r'\s*[|;]\s*',raw)
    weekly={}; n_recognized=0
    for piece in pieces:
        m=DAY_TOKEN.match(piece)
        if m and m.group(1).upper() in DAY_MAP:
            n_recognized+=1
            day=DAY_MAP[m.group(1).upper()]
            if day in weekly:
                return 'conflicting_24h',condition
            hours=m.group(2).strip()
            weekly[day]='full' if DAY_FULL.fullmatch(hours) else ('near' if DAY_NEAR.fullmatch(hours) else 'other')
    if n_recognized:
        if len(weekly)==7 and set(weekly)==DAYS:
            if set(weekly.values())=={'full'} and len(pieces)==7:
                return 'confirmed_24h',condition
            if set(weekly.values())=={'near'} and len(pieces)==7:
                return 'near_24h',condition
        return 'not_24h',condition
    if DAILY_24.search(raw):
        # Detect text internally contradicting '24 hours daily', rather than
        # treating a 24-hour phrase in an irregular schedule as full 24/7.
        if CONFLICT.search(raw) or TIME_RANGE.search(raw) or re.search(r'\b(?:M-F|Mon-Fri|Sat-Sun)\b',raw,re.I):
            return 'conflicting_24h',condition
        return 'confirmed_24h',condition
    if re.search(r'\b24\s*(?:hours?|hrs?)\b|\b24\s*[/x]\s*7\b',raw,re.I):
        return 'conflicting_24h',condition
    return 'not_24h',condition


def find_access_days_time_column(columns):
    candidates=[]
    for c in columns:
        k=re.sub(r'[^a-z0-9]+','_',str(c).lower()).strip('_')
        if k in ('access_days_time','access_days_time_en','access_days_time_english'):
            candidates.append(c)
    # Require the English field, not a French translation.
    return candidates[0] if candidates else None


def prepare_stations(ev_path: Path, output: Path) -> pd.DataFrame:
    if not ev_path.is_file():
        raise FileNotFoundError(f'AFDC master not found: {ev_path}')
    raw=pd.read_parquet(ev_path)
    hours_col=find_access_days_time_column(raw.columns)
    if hours_col is None:
        raise KeyError('The AFDC master has no Access Days Time column. The 2014 sample '
                       'cannot be used to classify 2017–2024 stations. Restore the '
                       'year-specific field from the ORIGINAL snapshots (join by '
                       'snapshot_year and station_id), then rerun. Candidate fields: '
                       +str([c for c in raw if ('access' in c.lower() or 'day' in c.lower())]))
    print(f'AFDC charging hours source field: {hours_col!r}',flush=True)
    required={'station_id','latitude','longitude','status_code','access_code',
              'fuel_type_code','study_metro','ev_dc_fast_count','facility_type'}
    if missing:=required-set(raw.columns):
        raise KeyError(f'AFDC master missing required fields: {sorted(missing)}')
    stations=select_annual_snapshots(raw)
    for col in ('status_code','access_code','fuel_type_code'):
        stations[col]=stations[col].astype('string').str.strip().str.upper()
    stations=stations.loc[(stations.status_code=='E') &
        (stations.access_code=='PUBLIC') & (stations.fuel_type_code=='ELEC') &
        stations.study_metro.isin(METROS.values())].copy()
    for col in ('latitude','longitude','ev_dc_fast_count'):
        stations[col]=pd.to_numeric(stations[col],errors='coerce')
    stations=stations.dropna(subset=['station_id','latitude','longitude'])
    stations=stations.loc[stations.latitude.between(-90,90)&stations.longitude.between(-180,180)]
    stations=stations.drop_duplicates(['snapshot_year','station_id']).copy()
    norm=(stations.facility_type.astype('string').str.strip().str.upper()
          .str.replace(r'[\s\-]+','_',regex=True).replace({'':'<UNKNOWN>'}))
    stations['facility_type_norm']=norm
    stations['is_dealer']=norm.isin(['CAR_DEALER','CRDLR']).fillna(False)
    stations['is_fast']=stations.ev_dc_fast_count.gt(0).fillna(False)
    stations['is_nondealer']=~stations.is_dealer
    classifications=stations[hours_col].map(classify_hours)
    stations['hours_class']=[v[0] for v in classifications]
    stations['hours_user_condition']=[v[1] for v in classifications]
    stations['is_24h']=stations.hours_class.eq('confirmed_24h')
    stations['is_24h_fast']=stations.is_24h & stations.is_fast
    stations['is_24h_nondealer']=stations.is_24h & stations.is_nondealer
    stations['is_24h_fast_nondealer']=stations.is_24h_fast & stations.is_nondealer
    station_meta=[c for c in ('snapshot_year','study_metro','station_id','station_name','facility_type_norm',hours_col,'hours_class','hours_user_condition','is_dealer','is_fast') if c in stations]
    stations.loc[stations.hours_class.isin(['near_24h','conflicting_24h']),station_meta].to_csv(
        output/'review_ambiguous_access_hours.csv',index=False)
    summary=stations.groupby(['snapshot_year','study_metro','hours_class'],dropna=False).size().rename('N').reset_index()
    summary.to_csv(output/'hours_classification_by_year_metro.csv',index=False)
    pd.DataFrame([
        dict(year=int(year),study_metro=metro,n_public=len(ss),
             n_open24h=int(ss.is_24h.sum()),
             n_open24h_fast=int(ss.is_24h_fast.sum()),
             n_open24h_nondealer=int(ss.is_24h_nondealer.sum()),
             n_open24h_fast_nondealer=int(ss.is_24h_fast_nondealer.sum()),
             n_near24h=int(ss.hours_class.eq('near_24h').sum()),
             n_conflicting=int(ss.hours_class.eq('conflicting_24h').sum()),
             n_unspecified=int(ss.hours_class.eq('unspecified').sum()),
             n_restriction_flag_24h=int((ss.is_24h & ss.hours_user_condition).sum()),
             n_explicit_dealer_24h=int((ss.is_24h & ss.is_dealer).sum()),
             n_unknown_facility_24h=int((ss.is_24h & ss.facility_type_norm.isna()).sum()))
        for (year,metro),ss in stations.groupby(['snapshot_year','study_metro'])
    ]).to_csv(output/'station_classification_audit.csv',index=False)
    patt=stations.groupby([hours_col,'hours_class','hours_user_condition'],dropna=False).size().rename('n_station_years').reset_index().sort_values('n_station_years',ascending=False)
    patt.to_csv(output/'hours_string_patterns.csv',index=False)
    print('24-hour station audit:\n'+pd.read_csv(output/'station_classification_audit.csv').to_string(index=False),flush=True)
    if not stations.is_24h.any():
        raise ValueError('No 24h stations found: inspect hours_string_patterns.csv and source field')
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
