#!/usr/bin/env python3
"""Annual 2017-2023 original-model replication for full Philly and DMV CBSAs.

- Poisson GLM with log link and HC1 robust covariance
- Random forest with 10-fold mtry tuning (ROC AUC), 500 trees, 80/20 holdout
- Tree-level out-of-bag permutation mean-decrease-accuracy, scaled by its SE
- Pooled and separate metro model scopes

Does NOT claim exact numerical equality to caret/randomForest: Python's forest
algorithm, CV folds, and RNG differ from R; the statistical specification matches.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from evcs_paths import DATA_ROOT, SHAPE_ROOT, BG_SHP, CBSA_SHP, HISTORY_DIR, CODE_DIR

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
import statsmodels.api as sm

ROOT = DATA_ROOT
DEFAULT_PANEL = ROOT / 'output/native_bg_rf_replication/exact_replication_panel_2017_2023.csv'
DEFAULT_OUTPUT = ROOT / 'output/native_bg_rf_replication/metro_original_models'
YEARS = list(range(2017, 2024))
METROS = ['Philadelphia-Camden-Wilmington', 'Washington-Arlington-Alexandria']
SCOPES = ['Pooled', 'Philadelphia', 'DMV']
PREDICTORS = [
    'TotalPopulation', 'MedianRent', 'PopDens', 'Walkability',
    'black_per', 'hispanic_per', 'asian_per', 'renter_per',
    'MedianHouseholdIncome', 'EmployeeDiversity', 'DistanceTransit',
    'hs_per', 'bachelor_per', 'master_per',
]
TARGETS = [(scope, year) for scope in SCOPES for year in YEARS]


def oob_scaled_mda(forest, x_train: pd.DataFrame, y_train: pd.Series, seed=42):
    """Approximate R randomForest importance(type=1), scaled, via tree OOB samples.

    Mirrors user's prior Python notebook; sklearn and R bootstrap RNGs differ.
    """
    x = x_train.to_numpy(dtype=np.float64)
    y = y_train.to_numpy(dtype=np.int32)
    n, p = x.shape
    losses = [[] for _ in range(p)]
    for ti, tree in enumerate(forest.estimators_):
        # sklearn bootstrap uses its tree's random_state and n_samples draws.
        rng = np.random.RandomState(tree.random_state)
        sampled = rng.randint(0, n, n)
        inbag = np.zeros(n, dtype=bool)
        inbag[np.unique(sampled)] = True
        idx = np.flatnonzero(~inbag)
        if not len(idx):
            continue
        xo, yo = x[idx], y[idx]
        base = np.mean(tree.predict(xo) == yo)
        for j in range(p):
            xp = xo.copy()
            prng = np.random.RandomState(seed + ti * p + j)
            xp[:, j] = prng.permutation(xp[:, j])
            losses[j].append(base - np.mean(tree.predict(xp) == yo))
    rows = []
    for col, arr in zip(x_train.columns, losses):
        a = np.asarray(arr, dtype=float)
        mean = float(np.mean(a)) if len(a) else np.nan
        se = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan
        scaled = mean / se if np.isfinite(se) and se > 0 else mean
        rows.append({'Variable': col, 'Importance': scaled,
                     'RawMeanDecreaseAccuracy': mean, 'ImportanceSE': se})
    out = pd.DataFrame(rows)
    out['Rank'] = out['Importance'].rank(ascending=False, method='average')
    return out


def performance(y_true, scores, threshold=None):
    true = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    auc = roc_auc_score(true, score)
    fpr, tpr, cuts = roc_curve(true, score)
    if threshold is None:
        # Youden threshold, as in the original R Poisson code.
        eligible = np.isfinite(cuts)
        ix = np.argmax(np.where(eligible, tpr - fpr, -np.inf))
        threshold = float(cuts[ix])
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    return {
        'AUC': auc, 'Accuracy': accuracy_score(true, pred),
        'TPR': tp/(tp+fn) if tp+fn else np.nan,
        'FPR': fp/(fp+tn) if fp+tn else np.nan,
        'Precision': tp/(tp+fp) if tp+fp else np.nan,
        'Threshold': threshold,
        'TestN': len(true), 'TestPositiveN': int((true == 1).sum())
    }


def get_scope(df, scope):
    if scope == 'Pooled':
        return df[df.study_metro.isin(METROS)]
    if scope == 'Philadelphia':
        return df[df.study_metro.eq(METROS[0])]
    if scope == 'DMV':
        return df[df.study_metro.eq(METROS[1])]
    raise ValueError(scope)


def run_one(df, scope, year, outdir, *, trees=500, folds=10, workers=4):
    sample = get_scope(df, scope)
    sample = sample.loc[sample.Year.eq(year), ['WithinBuffer'] + PREDICTORS].copy()
    n_before = len(sample)
    for col in sample:
        sample[col] = pd.to_numeric(sample[col], errors='coerce')
    sample.replace([np.inf, -np.inf], np.nan, inplace=True)
    sample.dropna(inplace=True)
    sample = sample.loc[sample.WithinBuffer.isin([0, 1])]
    sample['WithinBuffer'] = sample['WithinBuffer'].astype(int)
    if sample['WithinBuffer'].nunique() != 2:
        raise ValueError(f'{scope} {year}: need both access classes')
    train, test = train_test_split(sample, test_size=.20, stratify=sample['WithinBuffer'], random_state=2)
    if train.WithinBuffer.value_counts().min() < folds:
        raise ValueError(f'{scope} {year}: insufficient minority-class cases for {folds}-fold CV')
    x_tr, x_te = train[PREDICTORS], test[PREDICTORS]
    y_tr, y_te = train.WithinBuffer, test.WithinBuffer

    # Modified Poisson with HC1 heteroskedasticity-robust standard errors.
    x_poisson = sm.add_constant(x_tr, has_constant='add')
    fit = sm.GLM(y_tr, x_poisson, family=sm.families.Poisson()).fit(cov_type='HC1')
    pred_poisson = fit.predict(sm.add_constant(x_te, has_constant='add'))
    poisson_perf = performance(y_te, pred_poisson)
    pcoef = pd.DataFrame({'Variable': fit.params.index, 'Coefficient': fit.params.values,
                          'HC1_SE': fit.bse.values, 'PValue': fit.pvalues.values})
    pcoef['ExpCoefficient'] = np.exp(np.clip(pcoef.Coefficient, -700, 700))
    pcoef.insert(0, 'Year', year)
    pcoef.insert(0, 'Scope', scope)

    # Reproduce R mtry grid: unique(pmax(1, round(seq(2, sqrt(p)*1.5, 6))))
    p = len(PREDICTORS)
    mtry = sorted(set(np.maximum(1, np.rint(np.linspace(2, np.sqrt(p)*1.5, 6)).astype(int)).tolist()))
    model = RandomForestClassifier(
        n_estimators=trees, criterion='gini', max_features='sqrt',
        min_samples_leaf=1, bootstrap=True, class_weight=None,
        random_state=42, n_jobs=workers,
    )
    grid = GridSearchCV(
        model, {'max_features': mtry}, cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        scoring='roc_auc', refit=True, n_jobs=1, error_score='raise',
    )
    grid.fit(x_tr, y_tr)
    best = grid.best_estimator_
    pred_rf = best.predict_proba(x_te)[:, list(best.classes_).index(1)]
    # R RF uses final 0.5 class-decision cutoff, not the Poisson Youden cutoff.
    rf_perf = performance(y_te, pred_rf, threshold=.5)
    importance = oob_scaled_mda(best, x_tr, y_tr)
    importance.insert(0, 'Year', year)
    importance.insert(0, 'Scope', scope)

    metrics = []
    for m, vals in [('Modified Poisson', poisson_perf), ('Random Forest', rf_perf)]:
        metrics.append({'Scope': scope, 'Year': year, 'Model': m,
                        'N': len(sample), 'NBeforeCompleteCases': n_before,
                        'TrainN': len(train), 'AccessShare': sample.WithinBuffer.mean(),
                        'BestMtry': int(grid.best_params_['max_features']) if m == 'Random Forest' else np.nan,
                        'CVAUC': float(grid.best_score_) if m == 'Random Forest' else np.nan,
                        **vals})
    target_dir = outdir / scope.lower().replace(' ', '_')
    target_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(target_dir / f'{year}_rf_importance.csv', index=False)
    pcoef.to_csv(target_dir / f'{year}_poisson_coefs.csv', index=False)
    pd.DataFrame(metrics).to_csv(target_dir / f'{year}_performance.csv', index=False)
    print(pd.DataFrame(metrics)[['Scope','Year','Model','N','AUC','Accuracy','BestMtry']].to_string(index=False), flush=True)
    return metrics


def aggregate(outdir):
    impfiles = list(outdir.glob('*/20??_rf_importance.csv'))
    perffiles = list(outdir.glob('*/20??_performance.csv'))
    if not impfiles or not perffiles:
        raise FileNotFoundError('No model result files found; run annual models first.')
    importance = pd.concat([pd.read_csv(p) for p in impfiles], ignore_index=True)
    perf = pd.concat([pd.read_csv(p) for p in perffiles], ignore_index=True)
    if importance.duplicated(['Scope', 'Year', 'Variable']).any():
        raise ValueError('Repeated RF importance entries')
    table = importance.groupby(['Scope', 'Variable'], as_index=False).agg(
        AverageRank=('Rank', 'mean'), RankStdDev=('Rank', 'std'),
        MeanImportance=('Importance', 'mean'), Years=('Year', 'nunique'))
    table.sort_values(['Scope', 'AverageRank', 'Variable'], inplace=True)
    pooled = table.loc[table.Scope.eq('Pooled')].copy()
    regional = table.loc[table.Scope.ne('Pooled')].copy()
    (outdir / 'summary').mkdir(exist_ok=True, parents=True)
    table.to_csv(outdir / 'summary' / 'all_variable_importance_ranks.csv', index=False)
    pooled.to_csv(outdir / 'summary' / 'table_1_pooled.csv', index=False)
    regional.to_csv(outdir / 'summary' / 'table_1b_regional.csv', index=False)
    perf.sort_values(['Scope', 'Year', 'Model']).to_csv(
        outdir / 'summary' / 'annual_auc_and_performance.csv', index=False)
    importance.to_csv(outdir / 'summary' / 'rf_importance_by_year.csv', index=False)
    counts = perf[perf.Model.eq('Random Forest')].groupby('Scope').Year.nunique()
    for scope in SCOPES:
        if int(counts.get(scope, 0)) != len(YEARS):
            warnings.warn(f'{scope}: only {counts.get(scope, 0)}/7 annual models; tables are PROVISIONAL')
    print('Summary outputs:', outdir / 'summary', flush=True)
    print(pooled[['Variable','AverageRank','RankStdDev','Years']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=DEFAULT_PANEL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scope', choices=SCOPES, default='Pooled')
    parser.add_argument('--year', type=int, choices=YEARS, default=2017)
    parser.add_argument('--array-task', type=int, default=None, choices=range(len(TARGETS)))
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.output)
        return
    scope, year = (TARGETS[args.array_task] if args.array_task is not None
                   else (args.scope, args.year))
    if not args.panel.exists():
        raise FileNotFoundError(f'Missing input panel: {args.panel}. Run 01_build_metro_panel.py first.')
    panel = pd.read_csv(args.panel, low_memory=False)
    missing = set(['study_metro', 'Year', 'WithinBuffer'] + PREDICTORS) - set(panel.columns)
    if missing:
        raise KeyError(f'Missing columns: {sorted(missing)}')
    panel['Year'] = pd.to_numeric(panel['Year'], errors='coerce')
    run_one(panel, scope, year, args.output,
            trees=args.trees, folds=args.folds, workers=args.workers)


if __name__ == '__main__':
    main()
