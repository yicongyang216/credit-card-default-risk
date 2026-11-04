# Stage 1 verification

Verified on 2026-10-20 (America/New_York), Windows, Python 3.12.14.
This verifies the data foundation only, not a trained prediction model.

- Full suite: `python -m pytest -q` — **19 passed**.
- `python -m pip check` — no broken requirements found.
- Official XLS loaded with its actual second-row header: 30,000 rows, 23 source predictors.
- No null cells and no repeated IDs in the validated input.
- Manifest: 18,000 train / 6,000 validation / 6,000 test.
- Training outcomes: 3,982 default and 14,018 no default (22.1222% default).
- Preparation rerun from cache: audit JSON and split CSV hashes unchanged.
- Split CSV SHA-256: `3f3d4a83a618d39317472802aeaa48f8a10d965d5e50b83d6d7f8fa449382593`.
- Notebook executed successfully: 6 code cells, zero error outputs; nbformat validation passed.
- Saved PNG inspected visually: readable labels and no clipping.
- Source digests and actual retrieval time are in `data/raw/source.json`.

Tests were first observed failing against unimplemented boundaries, then passing
after implementation. Synthetic test fixtures are used only in tests and are
never treated as dataset observations.

## Review notes

The author checked feature exclusions, identity validation, class stratification,
order stability, holdout-label isolation in the training audit, and cache integrity.
There was no independent code review in this stage.

Different IDs with identical predictor profiles are retained, documented and counted.
There are 59 allowed-feature profile groups shared across partitions. This is not
proof of duplicate identities; hidden duplication remains a limitation. No temporal,
geographic, fairness or production claim follows from these checks.

## Environment notes

The local bundled interpreter was used through an isolated environment. The initial
venv setup needed a workspace-local temporary directory. Dependency and official
data downloads required network access. Jupyter execution required local loopback
sockets outside the restricted sandbox; no public service or global kernel was
registered. The notebook runner explicitly uses the invoking Python environment.
