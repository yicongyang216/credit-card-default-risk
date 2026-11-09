# Credit Card Default Risk

Predict next-month default for existing credit card customers, then examine how
model choice, probability calibration and decision thresholds affect evaluation.
This is an educational portfolio study, not a loan-approval or production system.

**Status:** implemented and run on the official data. Two executed notebooks,
reusable Python source, tests, saved figures and a frozen evaluation are included.

## Results at a glance

On the fixed 6,000-customer test set, the selected random forest achieves
**ROC-AUC 0.783** and **average precision 0.563**
(test default prevalence 22.1%). The logistic baseline obtains
ROC-AUC 0.774 and AP 0.557. These describe this historical split, not future US borrowers.

The validation-derived threshold gives 56.1% precision and
53.1% recall, flagging 20.93% of test customers.
The assumed 20% validation review capacity is not a guaranteed future quota.
Sigmoid calibration did not improve the forest's validation Brier score and was
not selected. No financial savings or formal fairness claims are made.

Read [the full results and limitations](reports/model_results.md),
[the model card](docs/model_card.md), or [the interview guide](docs/interview_guide.md).
Verification evidence: [33 passing tests, executed notebooks and matching reproduction](docs/final_verification.md).

## Data and task

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
Yeh, I. (2009), [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0.
30,000 Taiwan customer records, 23 source predictors, April–September 2005 history.
The supplied outcome is next-month default; the loader normalizes its name to `default`.

The scoring assumption is after September information is available. See the
[data dictionary](docs/data_dictionary.md) for undocumented codes and timing limits.
There are 21 modeled predictors: ID/target excluded, SEX/MARRIAGE audit-only, AGE retained.

## Method

- Saved seed-42 stratified 60/20/20 split by customer ID; no invented chronology.
- Train-fitted standardization and one-hot encoding inside pipelines.
- Logistic regression: C=1, unweighted versus balanced class weights.
- Random forest: 150 trees, leaf minimum 20, depth 6 versus 12.
- Three-fold training CV selects configurations and family using average precision.
- Train-only out-of-fold sigmoid calibration; validation Brier chooses raw/calibrated.
- Validation scores determine a <=20% flag threshold, excluding overflowing ties.
- Decisions/models saved before test scoring. No train+validation refit afterward.
- ROC-AUC, AP, precision/recall/F1, confusion matrices, Brier/log loss,
  calibration, validation importance, and sex/age diagnostics with denominators.

AP means `average_precision_score`, not trapezoidal PR integration. Class imbalance
is handled through informative metrics, a class-weight comparison and threshold
analysis; no resampling changes validation/test prevalence.

## Reproduce

Verified using Python 3.12.14 on Windows. Run from this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m credit_risk.prepare
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m credit_risk.train --run-dir artifacts/reproduction
.\.venv\Scripts\python.exe scripts/execute_notebook.py
```

For macOS/Linux, substitute `.venv/bin/python`; those platforms have not been
independently verified. First preparation requires internet to UCI; cached reuse
verifies source checksums. The tested package versions are locked. No GPU is needed.
Training uses one CPU worker for deterministic, uncomplicated execution.

The shipped results are in `artifacts/baseline`. Reproduction writes a new directory;
existing run directories are refused to avoid silently overwriting evidence.
For another repetition, choose `artifacts/reproduction-2`. Notebooks read the frozen
baseline artifacts and do not refit models or choose thresholds.

## Structure

```text
notebooks/01_data_understanding.ipynb   # train-only EDA and prediction boundary
notebooks/02_model_evaluation.ipynb     # saved evaluation and interpretation
src/credit_risk/data.py                # schema, allowlist and splitting
src/credit_risk/prepare.py             # source retrieval and data audit
src/credit_risk/models.py              # train-fitted LR/RF pipelines
src/credit_risk/evaluation.py          # metrics, capacity and subgroup diagnostics
src/credit_risk/train.py               # CV, frozen choices and final evaluation
src/credit_risk/plots.py               # figures from saved predictions
tests/                                # artificial test fixtures only
artifacts/baseline/                    # real metrics, decisions, model files and figures
reports/                              # readable audit and model results
docs/                                 # data dictionary, model card and interview preparation
```

