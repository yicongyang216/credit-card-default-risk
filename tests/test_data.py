"""Artificial fixtures test contracts only; these are not portfolio observations."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import make_splits, model_features, validate_data
from credit_risk.prepare import build_audit


@pytest.fixture
def sample():
    n = 100
    columns = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0",
               "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    columns += [f"BILL_AMT{i}" for i in range(1, 7)]
    columns += [f"PAY_AMT{i}" for i in range(1, 7)]
    frame = pd.DataFrame({c: np.ones(n, dtype=int) for c in columns})
    frame.insert(0, "ID", np.arange(1, n + 1))
    frame["default"] = [0] * 75 + [1] * 25
    return frame


def test_feature_boundary_excludes_target_id_and_audit_fields(sample):
    # A broad 'drop target only' selection would accidentally expose IDs/audit fields.
    sample["future_collection_amount"] = 999
    features = model_features(sample)
    assert not {"default", "ID", "SEX", "MARRIAGE", "future_collection_amount"} & set(features)
    assert features.shape == (100, 21)
    assert "AGE" in features


@pytest.mark.parametrize("column,value", [("default", 2), ("default", None),
    ("ID", 0), ("ID", 1.5), ("AGE", None), ("LIMIT_BAL", float("inf")),
    ("PAY_0", "not numeric")])
def test_invalid_required_values_fail_closed(sample, column, value):
    sample[column] = sample[column].astype(object)
    sample.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_data(sample)


def test_repeated_customer_id_is_rejected(sample):
    sample.loc[1, "ID"] = 1
    with pytest.raises(ValueError, match="ID"):
        validate_data(sample)


def test_missing_and_unknown_columns_are_rejected(sample):
    with pytest.raises(ValueError, match="columns"):
        validate_data(sample.drop(columns="PAY_AMT6"))
    with pytest.raises(ValueError, match="columns"):
        validate_data(sample.assign(post_outcome=1))


def test_undocumented_numeric_codes_are_preserved_for_audit(sample):
    sample.loc[0, "EDUCATION"] = 6
    sample.loc[0, "PAY_0"] = -2
    validate_data(sample)
    assert model_features(sample).loc[0, "EDUCATION"] == 6
    assert model_features(sample).loc[0, "PAY_0"] == -2


def test_split_covers_every_id_once_and_preserves_class_proportion(sample):
    manifest = make_splits(sample)
    assert manifest.shape == (100, 2)
    assert manifest["ID"].is_unique
    assert set(manifest["ID"]) == set(range(1, 101))
    assert manifest["split"].value_counts().to_dict() == {"train": 60, "validation": 20, "test": 20}
    counts = manifest.merge(sample[["ID", "default"]], on="ID").groupby("split")["default"].sum()
    assert counts.to_dict() == {"test": 5, "train": 15, "validation": 5}


def test_split_is_stable_under_input_row_reordering(sample):
    first = make_splits(sample)
    second = make_splits(sample.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(make_splits(sample, seed=43))


def test_single_class_data_cannot_be_split(sample):
    sample["default"] = 0
    with pytest.raises(ValueError):
        make_splits(sample)


def test_training_audit_does_not_read_holdout_outcomes(sample):
    manifest = pd.DataFrame({"ID": range(1, 101),
        "split": ["train"] * 60 + ["validation"] * 20 + ["test"] * 20})
    original = build_audit(sample, manifest)["training"]
    changed = sample.copy()
    held_out = manifest.loc[manifest["split"] != "train", "ID"]
    changed.loc[changed["ID"].isin(held_out), "default"] = 1 - changed.loc[
        changed["ID"].isin(held_out), "default"]
    assert build_audit(changed, manifest)["training"] == original
