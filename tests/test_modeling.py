"""Artificial fixtures check behavior; no synthetic observations enter real reports."""
import numpy as np
import pandas as pd
import pytest
from credit_risk.data import FEATURE_COLUMNS
from credit_risk.models import build_model
from credit_risk.evaluation import capacity_threshold, metrics, subgroup_metrics, load_partitions


def test_pipeline_fits_only_training_statistics_and_accepts_unknown_codes():
    X = pd.DataFrame({c: [0., 2., 0., 2.] for c in FEATURE_COLUMNS})
    X['EDUCATION'] = [1, 2, 1, 2]
    model = build_model('logistic')
    model.fit(X, [0, 1, 0, 1])
    holdout = X.iloc[:1].copy()
    holdout['LIMIT_BAL'] = 999999
    holdout['EDUCATION'] = 99
    p = model.predict_proba(holdout)
    assert np.isfinite(p).all()
    scaler = model.named_steps['preprocess'].named_transformers_['numeric']
    assert scaler.mean_[0] == 1.0
    assert 99 not in model.named_steps['preprocess'].named_transformers_['categorical'].categories_[0]


def test_tree_model_is_reproducible_and_outputs_probabilities():
    X = pd.DataFrame({c: np.arange(80) % 7 for c in FEATURE_COLUMNS})
    y = np.arange(80) % 2
    a, b = build_model('forest'), build_model('forest')
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
    assert a.predict_proba(X).shape == (80, 2)


@pytest.mark.parametrize('scores,capacity,flagged', [
    ([.9, .8, .8, .1], .5, 1), ([.9, .8, .7, .1], .5, 2),
    ([1., 1., 1.], .2, 0), ([.1, .2], 0, 0), ([.1, .2], 1, 2)])
def test_capacity_respects_entire_tie_groups(scores, capacity, flagged):
    threshold = capacity_threshold(scores, capacity)
    assert int((np.array(scores) >= threshold).sum()) == flagged


@pytest.mark.parametrize('scores,capacity', [([], .2), ([np.nan], .2), ([1.1], .2), ([.2], -.1)])
def test_threshold_rejects_invalid_inputs(scores, capacity):
    with pytest.raises(ValueError):
        capacity_threshold(scores, capacity)


def test_confusion_matrix_orientation_and_metrics():
    result = metrics([0, 0, 1, 1], [.1, .8, .7, .9], .5)
    assert [result[k] for k in ['tn', 'fp', 'fn', 'tp']] == [1, 1, 0, 2]
    assert result['recall'] == 1
    assert result['precision'] == pytest.approx(2/3)
    assert result['flag_rate'] == .75


def test_subgroups_keep_counts_but_suppress_unstable_rates():
    rows = subgroup_metrics([0, 0, 1, 1], [.1, .9, .8, .2], ['A', 'A', 'B', 'B'], .5, minimum=2)
    a, b = rows
    assert a['n'] == 2 and a['positives'] == 0 and a['negatives'] == 2
    assert a['recall'] is None and a['fpr'] == .5
    assert b['recall'] == .5 and b['fpr'] is None
    assert a['precision'] is None  # one flagged case, below minimum denominator


def test_partition_loader_rejects_changed_or_duplicate_ids():
    frame = pd.DataFrame({'ID': [1, 2, 3], 'default': [0, 1, 0]})
    manifest = pd.DataFrame({'ID': [1, 2, 4], 'split': ['train', 'validation', 'test']})
    with pytest.raises(ValueError):
        load_partitions(frame, manifest)
    manifest['ID'] = [1, 2, 2]
    with pytest.raises(ValueError):
        load_partitions(frame, manifest)
