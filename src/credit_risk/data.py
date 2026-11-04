"""Data contract and model feature boundary; no learned preprocessing here."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC_COLUMNS = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [
    f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLUMNS = ["EDUCATION"] + STATUS_COLUMNS
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
AUDIT_COLUMNS = ["SEX", "MARRIAGE"]
SOURCE_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + STATUS_COLUMNS + [
    f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
REQUIRED_COLUMNS = ["ID"] + SOURCE_FEATURES + ["default"]


def validate_data(frame: pd.DataFrame) -> None:
    """Fail closed on schema/identity/target errors; do not guess category meanings."""
    if not frame.columns.is_unique or set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError("Unexpected columns: expected the official schema with normalized target")
    if frame.empty:
        raise ValueError("Empty dataset")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in REQUIRED_COLUMNS):
        raise ValueError("Required columns must be numeric")
    if not np.isfinite(frame[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Required values must be finite and non-missing")
    ids = frame["ID"]
    if not ids.is_unique or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("ID must contain unique positive integers")
    if set(frame["default"].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1")


def read_data(path: str | Path) -> pd.DataFrame:
    """The official XLS has an X1/Y row followed by its actual column headers."""
    frame = pd.read_excel(path, header=1, engine="xlrd")
    frame = frame.rename(columns={"default payment next month": "default"})
    validate_data(frame)
    return frame


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit allowlist prevents an added outcome-derived field entering a model."""
    return frame.loc[:, FEATURE_COLUMNS].copy()


def make_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Sort identities for order invariance; stratify once, then save assignments."""
    validate_data(frame)
    ordered = frame.sort_values("ID").reset_index(drop=True)
    train, holdout = train_test_split(
        ordered, test_size=0.4, stratify=ordered["default"], random_state=seed)
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=holdout["default"], random_state=seed)
    manifest = pd.concat([
        part[["ID"]].assign(split=name)
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    ])
    return manifest.sort_values("ID").reset_index(drop=True)
