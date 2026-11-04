"""Reproduce a small, frozen credit-risk evaluation. Run from the project root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_risk.data import read_data, model_features
from credit_risk.prepare import fetch_data, sha256
from credit_risk.models import build_model
from credit_risk.evaluation import capacity_threshold, metrics, subgroup_metrics, load_partitions


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def run(root: Path, run_dir: Path) -> dict:
    """Never overwrite a completed evaluation; use a new directory to reproduce it."""
    if run_dir.exists():
        raise FileExistsError(f'Run directory exists: {run_dir}. Use a new --run-dir for reproduction.')
    warnings.filterwarnings('error', category=ConvergenceWarning)
    start = time.perf_counter()
    source = fetch_data(root / 'data/raw')  # verify original checksums before reuse
    frame = read_data(source)
    manifest_path = root / 'data/processed/split_manifest.csv'
    partitions = load_partitions(frame, pd.read_csv(manifest_path))
    X = {s: model_features(v) for s, v in partitions.items()}
    y = {s: v['default'].to_numpy() for s, v in partitions.items()}
    run_dir.mkdir(parents=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    configurations = [
        ('logistic', 'unweighted', {'balanced': False}),
        ('logistic', 'balanced', {'balanced': True}),
        ('forest', 'depth6', {'depth': 6}),
        ('forest', 'depth12', {'depth': 12}),
    ]
    cv_rows, options = [], {}
    for family, label, params in configurations:
        estimator = build_model(family, **params)
        result = cross_validate(estimator, X['train'], y['train'], cv=cv, n_jobs=1,
            scoring={'ap': 'average_precision', 'roc_auc': 'roc_auc'}, error_score='raise')
        row = {'family': family, 'configuration': label, 'parameters': params,
               'mean_ap': float(result['test_ap'].mean()), 'std_ap': float(result['test_ap'].std()),
               'mean_roc_auc': float(result['test_roc_auc'].mean()),
               'fold_ap': result['test_ap'].tolist()}
        cv_rows.append(row)
        options[(family, label)] = estimator
        print(f'Training CV: {family}/{label}: AP={row["mean_ap"]:.4f}', flush=True)
    pd.DataFrame(cv_rows).to_csv(run_dir / 'training_cv.csv', index=False)
    best = {family: max([r for r in cv_rows if r['family'] == family], key=lambda r: r['mean_ap'])
            for family in ['logistic', 'forest']}
    # Family is chosen using training CV, before looking at validation/test results.
    selected_family = max(best, key=lambda f: best[f]['mean_ap'])
    models = {'dummy': DummyClassifier(strategy='prior')}
    models['dummy'].fit(X['train'], y['train'])
    for family, row in best.items():
        estimator = options[(family, row['configuration'])]
        estimator.fit(X['train'], y['train'])
        models[family] = estimator
        calibrated = CalibratedClassifierCV(
            build_model(family, **row['parameters']), method='sigmoid', cv=cv, ensemble=False)
        calibrated.fit(X['train'], y['train'])
        models[family + '_sigmoid'] = calibrated
        print(f'Fitted {family} and train-only sigmoid calibration', flush=True)
    validation_scores = {n: m.predict_proba(X['validation'])[:, 1] for n, m in models.items()}
    validation_rows = [{'model': n, **metrics(y['validation'], p)} for n, p in validation_scores.items()]
    pd.DataFrame(validation_rows).to_csv(run_dir / 'validation_metrics.csv', index=False)
    briers = {r['model']: r['brier'] for r in validation_rows}
    selected = selected_family
    if briers[selected_family + '_sigmoid'] < briers[selected_family]:
        selected += '_sigmoid'
    threshold = capacity_threshold(validation_scores[selected], .2)
    decision = {'seed': 42, 'training_folds': 3, 'selected_family': selected_family,
        'selected_model': selected, 'selected_configuration': best[selected_family],
        'selection_rule': 'Family/configuration: max training-CV AP; calibration: min validation Brier (raw wins ties).',
        'capacity_assumption': .2, 'threshold': threshold,
        'validation_working_point': metrics(y['validation'], validation_scores[selected], threshold),
        'validation_calibration_brier': {n: briers[n] for n in [selected_family, selected_family+'_sigmoid']},
        'source_xls_sha256': sha256(source), 'split_sha256': sha256(manifest_path),
        'source_code_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'test_policy': 'All model variants predeclared; no test-driven change or refit on train+validation.'}
    # Freeze both fitted objects and decisions BEFORE scoring the test sample.
    write_json(run_dir / 'decision.json', decision)
    joblib.dump(models, run_dir / 'models.joblib')
    print(f'Frozen selection: {selected}; threshold={threshold:.6f}. Evaluating test now.', flush=True)
    test_scores = {n: m.predict_proba(X['test'])[:, 1] for n, m in models.items()}
    test_rows = [{'model': n, 'policy': '0.5', **metrics(y['test'], p)} for n, p in test_scores.items()]
    working = metrics(y['test'], test_scores[selected], threshold)
    test_rows.append({'model': selected, 'policy': 'validation_capacity', **working})
    pd.DataFrame(test_rows).to_csv(run_dir / 'test_metrics.csv', index=False)
    pd.DataFrame({'ID': partitions['test']['ID'], 'default': y['test'],
                  **test_scores}).to_csv(run_dir / 'test_predictions.csv', index=False)
    # Predeclared confidence intervals: sampling uncertainty, not model-selection uncertainty.
    rng, boot = np.random.default_rng(42), []
    for _ in range(200):
        index = rng.integers(0, len(y['test']), len(y['test']))
        if len(np.unique(y['test'][index])) == 2:
            boot.append([roc_auc_score(y['test'][index], test_scores[selected][index]),
                         average_precision_score(y['test'][index], test_scores[selected][index])])
    interval = np.quantile(boot, [.025, .975], axis=0)
    write_json(run_dir / 'bootstrap_intervals.json', {
        'replicates': len(boot), 'roc_auc_95_percentile': interval[:, 0].tolist(),
        'ap_95_percentile': interval[:, 1].tolist(),
        'scope': 'Conditional on the frozen model and this sample; excludes selection and population-shift uncertainty.'})
    subgroup_rows = []
    test_frame = partitions['test']
    groupings = {'sex_source_code': test_frame['SEX'].map({1: 'male (source code 1)', 2: 'female (source code 2)'}).fillna('other/unknown'),
        'age_band': pd.cut(test_frame['AGE'], [-np.inf, 30, 50, np.inf], right=False,
                           labels=['under 30', '30 to 49', '50 and older']).astype(str)}
    for attribute, groups in groupings.items():
        subgroup_rows += [{'attribute': attribute, **r} for r in
            subgroup_metrics(y['test'], test_scores[selected], groups, threshold)]
    pd.DataFrame(subgroup_rows).to_csv(run_dir / 'test_subgroups.csv', index=False)
    write_json(run_dir / 'test_subgroups.json', subgroup_rows)
    logistic = models['logistic']
    coefficients = pd.DataFrame({'feature': logistic.named_steps['preprocess'].get_feature_names_out(),
        'coefficient': logistic.named_steps['classifier'].coef_[0]})
    coefficients['absolute_coefficient'] = coefficients['coefficient'].abs()
    coefficients.sort_values('absolute_coefficient', ascending=False).to_csv(run_dir / 'logistic_coefficients.csv', index=False)
    importance = permutation_importance(models['forest'], X['validation'], y['validation'],
                                       scoring='average_precision', n_repeats=3, random_state=42, n_jobs=1)
    pd.DataFrame({'feature': X['validation'].columns, 'ap_decrease': importance.importances_mean,
        'std': importance.importances_std}).sort_values('ap_decrease', ascending=False).to_csv(
            run_dir / 'forest_validation_importance.csv', index=False)
    pd.DataFrame([{'threshold': float(t), **metrics(y['validation'], validation_scores[selected], t)}
                  for t in np.unique(np.r_[np.linspace(.05, .95, 19), threshold])]).to_csv(
                      run_dir / 'validation_thresholds.csv', index=False)
    from credit_risk.plots import save_figures
    save_figures(run_dir, y['test'], test_scores, validation_scores[selected], y['validation'], selected, threshold)
    summary = {'selected_model': selected, 'test_working_point': working,
               'elapsed_seconds': time.perf_counter() - start}
    write_json(run_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/baseline'))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.cache/matplotlib'))
    run(root, args.run_dir.resolve())


if __name__ == '__main__':
    main()
