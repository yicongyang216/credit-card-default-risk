"""Retrieve the official source, save split IDs, and audit training data only."""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

from credit_risk.data import (
    CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, SOURCE_FEATURES,
    STATUS_COLUMNS, make_splits, read_data,
)

SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
ARCHIVE_MEMBER = "default of credit card clients.xls"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_data(raw_dir: Path, source_url: str = SOURCE_URL) -> Path:
    """Cache verified source bytes; local file URLs are used by offline unit tests."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls, archive, metadata = [raw_dir / n for n in ["source.xls", "source.zip", "source.json"]]
    if metadata.exists():
        recorded = json.loads(metadata.read_text(encoding="utf-8"))
        if recorded["source_url"] != source_url:
            raise ValueError("Cached source URL differs from the requested source")
        for path, key in [(xls, "xls_sha256"), (archive, "archive_sha256")]:
            if not path.exists() or sha256(path) != recorded[key]:
                raise ValueError(f"Cached source checksum mismatch: {path.name}")
        return xls
    if xls.exists() or archive.exists():
        raise ValueError("Unverified cache files exist without source.json; investigate before reuse")
    with urlopen(source_url, timeout=60) as response:
        payload = response.read()
    try:
        with ZipFile(BytesIO(payload)) as zipped:
            if zipped.namelist() != [ARCHIVE_MEMBER]:
                raise ValueError("Unexpected source archive contents")
            # Read this named file; never extract arbitrary archive paths.
            workbook = zipped.read(ARCHIVE_MEMBER)
    except BadZipFile as exc:
        raise ValueError("Invalid source archive") from exc
    archive.write_bytes(payload)
    xls.write_bytes(workbook)
    recorded = {
        "dataset": "UCI Default of Credit Card Clients",
        "doi": "10.24432/C55S3H", "license": "CC BY 4.0",
        "source_url": source_url, "archive_member": ARCHIVE_MEMBER,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256(archive), "xls_sha256": sha256(xls),
    }
    metadata.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    return xls


def build_audit(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Global structural checks and train-only statistics; no holdout label report."""
    assigned = frame.merge(manifest, on="ID", how="left", validate="one_to_one")
    if assigned["split"].isna().any() or len(manifest) != len(frame):
        raise ValueError("Split manifest does not cover the dataset exactly")
    train = assigned.loc[assigned["split"] == "train"]
    if train.empty or set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Expected nonempty train, validation and test partitions")
    labels = train["default"].value_counts()
    # Overlap uses predictors, not outcomes. Repeated profiles do not prove identity.
    shared = assigned.groupby(FEATURE_COLUMNS, dropna=False)["split"].nunique()
    counts = lambda series: {str(k): int(v) for k, v in series.value_counts().sort_index().items()}
    return {
        "stage": "data foundation; no trained model",
        "seed": 42,
        "structural": {
            "rows": len(frame), "source_predictors": len(SOURCE_FEATURES),
            "model_predictors": len(FEATURE_COLUMNS),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_ids": int(frame["ID"].duplicated().sum()),
            "duplicate_rows_beyond_first_excluding_id": int(frame.drop(columns="ID").duplicated().sum()),
            "duplicate_source_profiles_beyond_first": int(frame[SOURCE_FEATURES].duplicated().sum()),
            "model_profiles_shared_across_partitions": int(shared.gt(1).sum()),
            "split_rows": {str(k): int(v) for k, v in manifest["split"].value_counts().sort_index().items()},
        },
        "training": {
            "rows": len(train), "defaults": int(labels.get(1, 0)),
            "non_defaults": int(labels.get(0, 0)), "default_rate": float(train["default"].mean()),
            "numeric_summary": json.loads(train[NUMERIC_COLUMNS].describe().to_json()),
            "category_counts": {c: counts(train[c]) for c in CATEGORICAL_COLUMNS + ["SEX", "MARRIAGE"]},
        },
        "limits": [
            "Historical Taiwan snapshot; no temporal or US-population validation.",
            "Distinct IDs with shared profiles retained; hidden entity duplication cannot be excluded.",
            "SEX and MARRIAGE audit-only; removing them does not establish fairness.",
            "Undocumented category semantics are not inferred.",
            "Model metrics and business impact: [需要真实信息].",
        ],
    }


def save_report(frame: pd.DataFrame, manifest: pd.DataFrame, report_dir: Path) -> dict:
    """Write a concise audit and a training-only figure."""
    os.environ.setdefault("MPLCONFIGDIR", str(report_dir.parent / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    audit = build_audit(frame, manifest)
    (report_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    structural, training = audit["structural"], audit["training"]
    lines = ["# Stage 1 data audit", "", "Generated from the official workbook. No model has been trained.", "",
             "## Structural checks", "", "| Check | Observed value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in structural.items() if k != "split_rows"]
    lines += ["", "## Split assignment", "", "| Partition | Rows |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in structural["split_rows"].items()]
    lines += ["", "Seed: 42. IDs sorted before stratification. No chronological meaning is assigned to ID.",
              "", "## Training subset only", "",
              f"- Default: {training['defaults']:,}", f"- No default: {training['non_defaults']:,}",
              f"- Observed default rate: {training['default_rate']:.2%}", "",
              "![Training overview](figures/training_overview.png)", "",
              "## Limitations", ""]
    lines += [f"- {v}" for v in audit["limits"]]
    lines += ["", "Code counts and training numeric summaries are in `data_audit.json`.",
              "Shared-profile counts describe groups of identical allowed predictor values, not proven duplicate people."]
    (report_dir / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    train = frame.loc[frame["ID"].isin(manifest.loc[manifest["split"] == "train", "ID"])]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        counts = [training["non_defaults"], training["defaults"]]
        bars = axes[0].bar(["No default", "Default"], counts, color=["#396985", "#c47842"])
        axes[0].bar_label(bars, labels=[f"{v:,}" for v in counts], padding=4)
        axes[0].set_ylim(0, max(counts) * 1.18)
        axes[0].set_title("Supplied next-month label")
        axes[0].set_ylabel("Training customers")
        status = train["PAY_0"].value_counts().sort_index()
        axes[1].bar(status.index.astype(str), status.to_numpy(), color="#396985")
        axes[1].set_title("September repayment status (PAY_0)")
        axes[1].set_xlabel("Source code; meanings of -2 and 0 undocumented")
        axes[1].set_ylabel("Training customers")
        fig.suptitle(f"Training data only | n = {len(train):,} | seed = 42", fontsize=14)
        fig.savefig(figure_dir / "training_overview.png", dpi=160)
        plt.close(fig)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_dir.resolve()
    frame = read_data(fetch_data(root / "data" / "raw"))
    manifest = make_splits(frame)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(processed / "split_manifest.csv", index=False, lineterminator="\n")
    audit = save_report(frame, manifest, root / "reports")
    print(json.dumps({"rows": len(frame), "split_rows": audit["structural"]["split_rows"],
                      "training_default_rate": audit["training"]["default_rate"]}, indent=2))


if __name__ == "__main__":
    main()
