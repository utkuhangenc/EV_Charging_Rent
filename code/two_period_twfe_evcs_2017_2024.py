#!/usr/bin/env python3
"""Two-period, nonoverlapping ACS sensitivity analysis for EVCS and rental affordability.

Uses the already-prepared 2020-block-group annual panel from the existing
2017–2024 Python TWFE replication; does NOT reconstruct ACS or EV buffers.

Primary: 2017->2024. Equal-duration sensitivity: 2017->2023, 2018->2024.
Each pair's ACS five-year windows are disjoint, but different pairs are NOT
statistically independent of each other.

With two waves, y_it = a_i + lambda_t + beta x_it + gamma c_it + u_it
is equivalent in coefficients to
  (y_i,end - y_i,start) = alpha + beta (x_i,end - x_i,start)
    + gamma (c_i,end - c_i,start) + error_i,
where alpha absorbs the common year difference. Pooled model uses one common
trend, as in the original TWFE; metro-specific trends can be added optionally.

Robust HC1 covariance across unique BG first differences; one row per BG,
so there is no additional within-BG serial correlation to estimate.

Exposures used here: WithinBuffer, PointCount, log1p(PointCount), at 0.5/1/2mi.
The original annual dPointCount and logNewAdds are intentionally NOT
regressed as differences: that would estimate differences in annual addition
rates at endpoints, not total charging expansion between the two periods.

No models are causally identified by these observational specifications.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = DATA_ROOT
PANEL_DIR = ROOT / 'output/native_bg_rf_replication/panel_twfe_2017_2024'
PANEL = PANEL_DIR / 'analysis_panel_2017_2024.parquet'
OUTPUT = PANEL_DIR / 'two_period_twfe_nonoverlap'

PAIRS = {'2017-2024': (2017, 2024), '2017-2023': (2017, 2023),
         '2018-2024': (2018, 2024)}
METROS = {'DMV': 'Washington-Arlington-Alexandria',
          'Philadelphia': 'Philadelphia-Camden-Wilmington'}
OUTCOMES = ['rent_afford', 'rent_burden_30p', 'rent_burden_50p']
CONTROLS = ['TotalPopulation', 'black_per', 'hispanic_per', 'asian_per',
            'hs_per', 'bachelor_per', 'master_per', 'renter_per', 'poverty_hh_per']
TAGS = ['0_5mi', '1mi', '2mi']
EXPOSURES = ([f'WithinBuffer_{t}' for t in TAGS]
             + [f'PointCount_{t}' for t in TAGS]
             + [f'logPointCount_{t}' for t in TAGS])
RAW_EXPOSURES = ([f'WithinBuffer_{t}' for t in TAGS]
                 + [f'PointCount_{t}' for t in TAGS])
NUMERIC_COLS = list(dict.fromkeys(OUTCOMES + CONTROLS +
                                  ['MedianHouseholdIncome'] + RAW_EXPOSURES))


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Prepared panel not found: {path}\n'
                                'First run --stage prepare using the existing 2017–2024 panel builder.')
    panel = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(
        path, dtype={'GEO_ID': 'string'}, low_memory=False)
    needed = {'GEO_ID', 'Year', 'study_metro', *NUMERIC_COLS}
    missing = needed.difference(panel.columns)
    if missing:
        raise KeyError(f'Prepared panel is missing columns: {sorted(missing)}')
    panel['GEO_ID'] = panel['GEO_ID'].astype('string')
    panel['Year'] = pd.to_numeric(panel['Year'], errors='raise').astype(int)
    if panel[['GEO_ID', 'Year', 'study_metro']].isna().any().any():
        raise ValueError('Missing BG identifier, Year, or study metro.')
    if panel.duplicated(['GEO_ID', 'Year']).any():
        raise ValueError('Duplicate GEO_ID-Year rows: cannot construct paired data.')
    if (panel.groupby('GEO_ID')['study_metro'].nunique() > 1).any():
        raise ValueError('A GEO_ID changes metropolitan area between years.')
    panel = panel[panel.study_metro.isin(METROS.values())].copy()
    for c in NUMERIC_COLS:
        panel[c] = pd.to_numeric(panel[c], errors='coerce')
        panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
    for c in RAW_EXPOSURES:
        if panel[c].lt(0).any():
            raise ValueError(f'Negative EV exposure values found in {c}.')
    for tag in TAGS:
        c = f'WithinBuffer_{tag}'
        if not panel[c].dropna().isin([0, 1]).all():
            raise ValueError(f'Invalid binary access coding in {c}.')
        raw = panel[f'PointCount_{tag}']
        panel[f'logPointCount_{tag}'] = np.log1p(raw)
        disagreement = raw.notna() & panel[c].notna() & (raw.gt(0) != panel[c].eq(1))
        if disagreement.any():
            raise ValueError(f'{int(disagreement.sum())} rows disagree about access / counts at {tag}.')
    return panel


def pair_data(panel: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    keep = ['GEO_ID', 'study_metro'] + OUTCOMES + CONTROLS + ['MedianHouseholdIncome'] + EXPOSURES
    # Preserve only block groups present in BOTH endpoint datasets; NO 8-year balance rule.
    a = panel.loc[panel.Year.eq(start), keep]
    b = panel.loc[panel.Year.eq(end), keep]
    if a.empty or b.empty:
        raise ValueError(f'Missing endpoint-year data for {start}->{end}')
    both = a.merge(b, on=['GEO_ID', 'study_metro'], how='inner',
                   suffixes=('_start', '_end'), validate='one_to_one')
    if both.empty:
        raise ValueError(f'No block groups observed in both {start} and {end}')
    both['start_year'] = start
    both['end_year'] = end
    for c in list(dict.fromkeys(OUTCOMES + CONTROLS + ['MedianHouseholdIncome'] + EXPOSURES)):
        both[f'd_{c}'] = both[f'{c}_end'] - both[f'{c}_start']
    return both


def access_summary(pair: pd.DataFrame, label: str, scope: str) -> list[dict]:
    records = []
    for tag in TAGS:
        access = f'WithinBuffer_{tag}'
        counts = f'PointCount_{tag}'
        v = pair[[f'{access}_start', f'{access}_end',
                  f'{counts}_start', f'{counts}_end']].dropna()
        initial = v[f'{access}_start'].to_numpy()
        final = v[f'{access}_end'].to_numpy()
        records.append(dict(pair=label, Scope=scope, distance=tag,
             matched_BGs=len(pair), usable_exposure_BGs=len(v),
             initial_access_share=float(np.mean(initial)) if len(v) else np.nan,
             final_access_share=float(np.mean(final)) if len(v) else np.nan,
             newly_accessible=int(np.sum((initial == 0) & (final == 1))),
             lost_access=int(np.sum((initial == 1) & (final == 0))),
             never_access=int(np.sum((initial == 0) & (final == 0))),
             always_access=int(np.sum((initial == 1) & (final == 1))),
             start_avg_station_count=float(v[f'{counts}_start'].mean()),
             end_avg_station_count=float(v[f'{counts}_end'].mean())))
    return records


def fit_first_difference(pair: pd.DataFrame, scope: str, dv: str,
                         exposure: str, include_income: bool,
                         metro_trends: bool = False) -> tuple[dict, dict]:
    # Outcome and covariates are in ORIGINAL UNITS, with difference end-minus-start.
    controls = CONTROLS + (['MedianHouseholdIncome'] if include_income else [])
    regressors = [exposure] + controls
    use = ['GEO_ID', 'study_metro', f'd_{dv}'] + [f'd_{c}' for c in regressors]
    df = pair[use].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(df)
    n_dmv = int(df.study_metro.eq(METROS['DMV']).sum())
    n_philly = int(df.study_metro.eq(METROS['Philadelphia']).sum())
    model_info = dict(Scope=scope, outcome=dv, exposure=exposure,
                      include_income=include_income, N=n, N_BGs=n,
                      N_DMV=n_dmv, N_Philadelphia=n_philly,
                      metro_specific_trend=metro_trends,
                      n_controls=len(controls), status='ok', error='')
    output = dict(**model_info, estimate=np.nan, std_error=np.nan,
                  t_stat=np.nan, p_value=np.nan, conf_low=np.nan,
                  conf_high=np.nan, significant_05=False, r2=np.nan,
                  within_exposure_sd=np.nan)
    try:
        if n < max(40, 5 * (len(regressors) + 2)):
            raise ValueError('Insufficient complete endpoint pairs')
        y = df[f'd_{dv}'].to_numpy(dtype=float)
        cols = [f'd_{c}' for c in regressors]
        X = df[cols].copy()
        if metro_trends:
            X['DMV_period_trend'] = df.study_metro.eq(METROS['DMV']).astype(float).to_numpy()
        X = sm.add_constant(X, has_constant='add')
        x = X.to_numpy(dtype=float)
        rank = np.linalg.matrix_rank(x)
        if rank != x.shape[1]:
            raise ValueError(f'Rank-deficient design matrix: rank={rank}, columns={x.shape[1]}; '
                             'inspect exposure switchers and covariates')
        # With two complete observations per BG, each BG contributes ONE
        # independent first difference. HC1 is the heteroskedasticity-robust
        # sandwich for that transformed regression; df-residual t inference.
        fitted = sm.OLS(y, X).fit(cov_type='HC1', use_t=True)
        col = f'd_{exposure}'
        if col not in fitted.params:
            raise ValueError(f'Exposure {exposure} not found in estimated model')
        b = float(fitted.params[col])
        se = float(fitted.bse[col])
        p = float(fitted.pvalues[col])
        ci = fitted.conf_int(alpha=0.05).loc[col]
        output.update(estimate=b, std_error=se, t_stat=float(fitted.tvalues[col]),
                      p_value=p, conf_low=float(ci.iloc[0]),
                      conf_high=float(ci.iloc[1]), significant_05=bool(p < 0.05),
                      r2=float(fitted.rsquared),
                      within_exposure_sd=float(df[col].std(ddof=1)))
    except Exception as exc:
        output.update(status='FAILED', error=str(exc))
    return output, model_info


def run(panel: pd.DataFrame, pairs: list[str], scopes: list[str],
        output: Path, metro_trends=False):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    samples = []
    access_rows = []
    missing_rows = []
    for label in pairs:
        start, end = PAIRS[label]
        paired = pair_data(panel, start, end)
        print(f'\n=== {label}: {len(paired):,} matched BGs (all geographies) ===', flush=True)
        paired[['GEO_ID', 'study_metro', 'start_year', 'end_year']].to_csv(
            output / f'matched_BG_ids_{label}.csv', index=False)
        configurations = list(scopes)
        if metro_trends and 'Pooled' in configurations:
            configurations.append('Pooled_MetroTrends')
        for scope in configurations:
            part = paired if scope.startswith('Pooled') else paired.loc[
                paired.study_metro.eq(METROS[scope])].copy()
            if part.empty:
                raise ValueError(f'No observations in {scope} for {label}')
            access_rows.extend(access_summary(part, label, scope))
            for dv in OUTCOMES:
                valid = part[f'd_{dv}'].notna()
                missing_rows.append(dict(pair=label, Scope=scope, outcome=dv,
                       matched_BGs=len(part), complete_DV_pairs=int(valid.sum()),
                       missing_DV_pairs=int((~valid).sum())))
            for dv in OUTCOMES:
                for exposure in EXPOSURES:
                    for income in (False, True):
                        result, _ = fit_first_difference(
                            part, scope, dv, exposure, income,
                            metro_trends=(scope == 'Pooled_MetroTrends'))
                        result.update(pair=label, start_year=start,
                                      end_year=end, interval_years=end - start,
                                      matched_BGs_before_modelwise_drops=len(part))
                        results.append(result)
            scope_results = [x for x in results if x['pair'] == label and x['Scope'] == scope]
            ok = sum(x['status'] == 'ok' for x in scope_results)
            print(f'{scope:20s} paired BGs={len(part):4d}; models OK={ok}/{len(scope_results)}', flush=True)
            example = [x for x in scope_results if x['exposure'] == 'WithinBuffer_1mi'
                       and x['include_income'] and x['outcome'] in OUTCOMES]
            for row in example:
                b = row['estimate']; p = row['p_value']; n = row['N']
                print(f'  {row["outcome"]:17s}: b={b: .6f}, p={p:.4f}, N={n}', flush=True)
    result_df = pd.DataFrame(results)
    access_df = pd.DataFrame(access_rows)
    sample_df = pd.DataFrame(missing_rows)
    result_df.to_csv(output / 'two_period_twfe_all_results.csv', index=False)
    access_df.to_csv(output / 'two_period_access_switchers.csv', index=False)
    sample_df.to_csv(output / 'two_period_sample_audit.csv', index=False)
    result_df.loc[(result_df.exposure == 'WithinBuffer_1mi') & result_df.include_income].to_csv(
        output / 'two_period_main_1mi_comparison.csv', index=False)
    failed = result_df[result_df.status != 'ok']
    if len(failed):
        failed.to_csv(output / 'two_period_FAILED_model_diagnostics.csv', index=False)
        print(f'\nWARNING: {len(failed)} specifications failed. Inspect the diagnostic CSV.', flush=True)
    else:
        print(f'\nAll {len(result_df):,} two-period TWFE/FD specifications completed.', flush=True)
    print('Saved to:', output, flush=True)
    return result_df, access_df


def self_test():
    rng = np.random.default_rng(25)
    n = 90
    ids = np.repeat(np.arange(n), 2)
    t = np.tile([0, 1], n)
    alpha = rng.normal(size=n)
    x = rng.normal(size=2*n)
    z = rng.normal(size=2*n)
    yy = alpha[ids] + .6*t - .47*x + .25*z + rng.normal(scale=.15, size=2*n)
    wide = pd.DataFrame({'id':ids, 'time':t, 'y':yy, 'x':x, 'z':z})
    p = wide.pivot(index='id', columns='time', values=['y','x','z'])
    dd = pd.DataFrame({k: (p[k][1] - p[k][0]) for k in ['y','x','z']})
    b_fd = sm.OLS(dd.y, sm.add_constant(dd[['x','z']])).fit().params['x']
    import statsmodels.formula.api as smf
    b_twfe = smf.ols('y ~ x + z + C(id) + C(time)', data=wide).fit().params['x']
    assert np.isclose(b_fd, b_twfe, rtol=0, atol=1e-11), (b_fd, b_twfe)
    print(f'PASS: two-period first differences = TWFE coefficient (b={b_fd:.9f})')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, default=PANEL,
                   help='Existing prepared 2017-2024 TWFE panel (parquet or CSV)')
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--pairs', nargs='+', choices=list(PAIRS), default=list(PAIRS),
                   help='Select comparisons (default: all three)')
    p.add_argument('--scopes', nargs='+', choices=['Pooled', 'DMV', 'Philadelphia'],
                   default=['Pooled', 'DMV', 'Philadelphia'])
    p.add_argument('--metro-trends', action='store_true',
                   help='Also fit pooled models allowing a separate DMV-specific 2017-to-end change.')
    p.add_argument('--self-test', action='store_true', help='Test FD/TWFE coefficient equivalence only')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    panel = load_panel(args.panel)
    result, access = run(panel, args.pairs, args.scopes, args.output,
                         metro_trends=args.metro_trends)
    if result.status.ne('ok').any():
        raise SystemExit('Some models failed; see two_period_FAILED_model_diagnostics.csv')


if __name__ == '__main__':
    main()
