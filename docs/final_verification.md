# Completed-project verification

The full model run used the original stage-1 split. The source workbook and split
checksums match the recorded source and stage-1 evidence. No configuration was
changed after viewing test results. The second run was deterministic reproduction,
not a new model search or a new independent test sample.

## Executed checks

- `python -m pytest -q`: **33 passed**, zero failures.
- `python -m pip check`: no broken requirements found.
- Both notebooks executed via the supplied runner: **12 code cells**, no error outputs.
- Both notebooks passed nbformat validation.
- Relative Markdown file/image links checked; none broken.
- Five model figures and the stage-1 figure inspected visually; readable and unclipped.
- A second run into `artifacts/reproduction` matched the original bytes for:
  `training_cv.csv`, `decision.json`, `validation_metrics.csv`, `test_metrics.csv`,
  `test_subgroups.csv`, `logistic_coefficients.csv`, `forest_validation_importance.csv`,
  and `bootstrap_intervals.json`. Runtime duration is intentionally not deterministic.
- Baseline training/evaluation took approximately 23 seconds in this environment;
  the reproduction took approximately 25 seconds. These are measurements of this
  local run, not a performance guarantee on other machines.

## Independent code review

A separate review agent checked the source, tests, decision artifacts and outputs
without refitting. It found no blocking leakage or metrics defect. It independently
recomputed every test metric from saved predictions to within 1e-12, verified source
and split hashes, and checked that subgroup confusion counts sum to test totals.

Optional hardening suggestions were recorded, not treated as observed defects:

- Add a runner-level regression test for training/test call order. Current checks
  cover pipeline boundaries and the actual run was independently reviewed.
- Reject any structurally valid but reassigned split using a separately anchored
  checksum. This actual run was checked against the preserved stage-1 checksum;
  the runner records rather than independently anchors its supplied manifest.
- Narrow the subgroup helper's generic rate-suppression docstring. The implemented
  policy suppresses precision, recall and FPR below denominator 20; prevalence and
  flag rate remain visible. The report states this policy explicitly. Current
  reported groups all exceed the minimum denominators.

## Limits of verification

Tests and code review do not prove that the source contains no hidden entity
duplication, that probabilities transfer to another population or period, or that
group comparisons establish fairness. No deployed service, operational review
benefit or dollar impact has been verified. Personal review and understanding of
the project, repository URL and software license remain [需要真实信息].

No Git repository existed in this workspace; no remote, commit, pull request or
deployment was created. Dependency versions are in `requirements-lock.txt`.
