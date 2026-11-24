#!/usr/bin/env python3
"""Full 2017-2024 TWFE robustness grid, on the *original* eight-wave balanced CBG panel.

Variation: 3 outcomes x 3 buffers x 5 metrics x 2 income x 3 station categories
x 3 geographic scopes = 810 joint lead/current/lag TWFE regressions and 2430
potential EV-term tests. An optional baseline stage adds 810 separate regressions.

Requires the 24h-exposure panel built by evcs_24h_access_twfe_2017_2024.py
and the original TWFE estimator from rebuild_evcs_panel_2017_2024_b25070_diagnostic.py.

Run on Gautschi:
  python evcs_full_robustness_grid_2017_2024.py --stage leadlag
  python evcs_full_robustness_grid_2017_2024.py --stage both
"""
from __future__ import annotations
import argparse
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR
import sys
import numpy as np
import pandas as pd

ROOT = DATA_ROOT / 'output/native_bg_rf_replication/panel_twfe_2017_2024'
INFILE = ROOT / 'evcs_24h_access_twfe_2017_2024/panel_charger_types_2017_2024.parquet'
OUTDIR = ROOT / 'evcs_24h_access_twfe_2017_2024/full_robustness_grid'
CODE = CODE_DIR

SCOPES = ('Pooled', 'DMV', 'Philadelphia')
CATEGORIES = ('AllPublic', 'Open24hPublic', 'Open24hFast')
OUTCOMES = ('rent_afford', 'rent_burden_30p', 'rent_burden_50p')
DISTANCES = ('0_5mi', '1mi', '2mi')
METRICS = ('WithinBuffer', 'PointCount', 'dPointCount', 'logPointCount', 'logNewAdds')
INCOME = (False, True)
YEARS = tuple(range(2017, 2025))
TIMING = {'leadlag_2': (('Lead 2y','_lead2'),('Current',''),('Lag 2y','_lag2')),
          'baseline': (('Current',''),)}


def add_transforms(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the original transforms separately for each charging category."""
    df = raw.copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype('int64')
    if df.duplicated(['GEO_ID','Year']).any():
        raise ValueError('Duplicate CBG-year rows')
    valid = df.groupby('GEO_ID')['Year'].agg(lambda s: set(s) == set(YEARS))
    df = df.loc[df.GEO_ID.isin(valid.index[valid])].sort_values(['GEO_ID','Year']).copy()
    if len(df) != 23224 or df.GEO_ID.nunique() != 2903:
        raise ValueError(f'Unexpected balanced panel: {len(df):,} rows, {df.GEO_ID.nunique():,} CBGs; expected 23,224 / 2,903')
    for cat in CATEGORIES:
        for dist in DISTANCES:
            suffix = f'{cat}_{dist}' if cat != 'AllPublic' else dist
            count = f'PointCount_{suffix}'
            proximity = f'WithinBuffer_{suffix}'
            if count not in df.columns or proximity not in df.columns:
                raise KeyError(f'Missing exposure {count} or {proximity}; run the 24-hour exposure audit first.')
            group = df.groupby('GEO_ID', sort=False)[count]
            df[f'dPointCount_{suffix}'] = group.diff()
            df[f'logPointCount_{suffix}'] = np.log1p(df[count])
            # EXACT earlier R/Python convention: log(1 + positive additions),
            # NOT a log of negative net changes.
            df[f'logNewAdds_{suffix}'] = np.log1p(df[f'dPointCount_{suffix}'].clip(lower=0))
    # Shift strictly within each CBG; all eight calendar years are present.
    grp = df.groupby('GEO_ID', sort=False)
    exposure_cols = []
    for cat in CATEGORIES:
        for dist in DISTANCES:
            suffix = f'{cat}_{dist}' if cat != 'AllPublic' else dist
            exposure_cols.extend(f'{metric}_{suffix}' for metric in METRICS)
    for col in exposure_cols:
        df[col + '_lead2'] = grp[col].shift(-2)
        df[col + '_lag2'] = grp[col].shift(2)
    return df


def run_grid(panel, stage, output, fit_twfe, METROS):
    modes = ('baseline','leadlag_2') if stage == 'both' else (stage,)
    expected_fits = (len(SCOPES)*len(CATEGORIES)*len(OUTCOMES)*len(DISTANCES)
                     *len(METRICS)*len(INCOME)*len(modes))
    rows=[]; diagnostics=[]
    iteration=0
    for scope in SCOPES:
        sub = panel if scope=='Pooled' else panel.loc[panel['study_metro'].eq(METROS[scope])]
        for cat in CATEGORIES:
            for outcome in OUTCOMES:
                for dist in DISTANCES:
                    for metric in METRICS:
                        suffix = f'{cat}_{dist}' if cat!='AllPublic' else dist
                        base = f'{metric}_{suffix}'
                        for include_income in INCOME:
                            for mode in modes:
                                iteration += 1
                                model_info = dict(Scope=scope,Category=cat,Outcome=outcome,
                                    Distance=dist,Metric=metric,IncludeIncome=include_income,
                                    Model=mode,Exposure=base,
                                    ModelID=f'{scope}|{cat}|{outcome}|{dist}|{metric}|{include_income}|{mode}')
                                ev_terms = TIMING[mode]
                                try:
                                    fitted, diag=fit_twfe(sub,outcome,base,include_income,mode)
                                    if len(fitted)!=len(ev_terms):
                                        raise ValueError(f'Expected {len(ev_terms)} terms; got {len(fitted)}')
                                    status='ok' if all(np.isfinite(r['estimate']) and np.isfinite(r['p_value']) for r in fitted) else 'partial'
                                    error = '' if status=='ok' else 'At least one EV term is not identifiable'
                                except Exception as exc:
                                    fitted=[dict(term=base+suffix,estimate=np.nan,std_error=np.nan,
                                      p_value=np.nan,conf_low=np.nan,conf_high=np.nan) for label,suffix in ev_terms]
                                    diag={};status='FAILED';error=f'{type(exc).__name__}: {exc}'
                                diagnostics.append({**model_info, 'status':status,'error':error,**diag})
                                for (label, _), fitted_row in zip(ev_terms, fitted):
                                    rows.append({**model_info,'Timing':label,
                                      'status':status,
                                      'estimate':fitted_row['estimate'],
                                      'std_error':fitted_row['std_error'],
                                      'conf_low':fitted_row['conf_low'],
                                      'conf_high':fitted_row['conf_high'],
                                      'p_value':fitted_row['p_value'],
                                      'significant_05':bool(np.isfinite(fitted_row['p_value']) and fitted_row['p_value']<0.05)})
        print(f'{scope}: completed {iteration}/{expected_fits} models; '
              f'failed/partial so far: {sum(d["status"]!="ok" for d in diagnostics)}',flush=True)
    if len(diagnostics)!=expected_fits:
        raise AssertionError(f'Expected {expected_fits} fits; made {len(diagnostics)}')
    res=pd.DataFrame(rows)
    diag=pd.DataFrame(diagnostics)
    output.mkdir(parents=True,exist_ok=True)
    res.to_csv(output/'robustness_grid_EV_coefficients.csv',index=False)
    diag.to_csv(output/'robustness_grid_model_diagnostics.csv',index=False)
    return res,diag


def summarize(results, output):
    lag=results.loc[results.Model.eq('leadlag_2')].copy()
    if lag.empty:
        print('No lead/lag models in --stage baseline; no Table 3 summary generated.')
        return None
    lag['tested'] = lag['p_value'].notna()
    lag['significant'] = lag['tested'] & lag['p_value'].lt(.05)
    groupings = {
        'Table3_overall': ['Outcome','Timing'],
        'Table3_by_charger_type': ['Outcome','Category','Timing'],
        'Table3_by_metro': ['Outcome','Scope','Timing'],
        'Table3_charger_type_by_metro': ['Outcome','Category','Scope','Timing'],
    }
    tables={}
    for filename, keys in groupings.items():
        table=(lag.groupby(keys,dropna=False,sort=False)
          .agg(N_planned=('ModelID','count'),N_tested=('tested','sum'),
               N_significant=('significant','sum')).reset_index())
        table['Percent_of_tested'] = np.where(table['N_tested']>0,
                                             100*table.N_significant/table.N_tested, np.nan)
        if filename=='Table3_overall':
            table['Expected_n'] = 270
            if not table.N_planned.eq(270).all():
                raise AssertionError('Expected 270 unique model/term combinations per outcome/timing')
        table.to_csv(output/f'{filename}.csv',index=False)
        tables[filename]=table
    return tables


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['leadlag','baseline','both'],default='leadlag')
    p.add_argument('--panel',type=Path,default=INFILE)
    p.add_argument('--output',type=Path,default=OUTDIR)
    p.add_argument('--code-dir',type=Path,default=CODE)
    args=p.parse_args()
    if not args.panel.is_file():
        raise FileNotFoundError(f'Enriched charging panel is missing: {args.panel}')
    sys.path.insert(0,str(args.code_dir))
    from rebuild_evcs_panel_2017_2024_b25070_diagnostic import fit_twfe, METROS
    raw=pd.read_parquet(args.panel)
    panel=add_transforms(raw)
    print(f'Balanced panel: {panel.GEO_ID.nunique():,} CBGs / {len(panel):,} rows',flush=True)
    stage={'leadlag':'leadlag_2', 'baseline':'baseline', 'both':'both'}[args.stage]
    res,diag=run_grid(panel,stage,args.output,fit_twfe,METROS)
    print(f'Successful fits: {(diag.status=="ok").sum():,} / {len(diag):,}',flush=True)
    tables=summarize(res,args.output)
    if tables:
        print('\nNEW TABLE 3 — p < 0.05, unadjusted for multiple testing:')
        print(tables['Table3_overall'].to_string(index=False))
        print('\nBy charging category:')
        print(tables['Table3_by_charger_type'].to_string(index=False))
        print('\nOutputs:',args.output)

if __name__=='__main__': main()
