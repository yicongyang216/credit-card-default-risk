# Credit Card Default Risk — approved design

Approved in chat on 2026-10-20. This document records the agreed design;
it does not claim that future stages have been implemented.

## Purpose
An entry-level portfolio project demonstrating supervised learning and honest
evaluation, complementing an existing analytics/engineering project.
Predict next-month default for existing card customers, not loan approval.
Use UCI Default of Credit Card Clients (DOI 10.24432/C55S3H, CC BY 4.0).

## Scope
- Interpretable logistic regression, one random forest comparison, dummy benchmark.
- Fixed seed 42; stratified 60/20/20 train/validation/test assignment by customer ID.
- No ID or target in predictors. SEX and MARRIAGE reserved for audit; AGE retained.
- Fit all learned preprocessing inside training folds only.
- Check entity IDs and identical records before splitting; identical financial
  profiles alone do not establish that two IDs represent the same customer.
- No fabricated dates or temporal validation. Historical features span April–September
  2005; the intended scoring point is after September, for the supplied next-month label.
- Train-only cross-validation for small configuration comparisons and calibration.
- Validation-only model/threshold decisions; one final test evaluation after choices freeze.
- ROC-AUC, average precision (explicit PR-AUC convention), precision, recall, F1,
  confusion matrices, Brier score and calibration diagrams.
- Illustrative 20% review capacity; no invented dollar savings or business costs.
- Coefficients and permutation importance, without causal claims.
- Limited sex/age subgroup diagnostics with counts and common threshold;
  no formal fairness/compliance claim.
- Two notebooks calling reusable source; English project materials; tested run commands.
- Missing external facts and unmeasured results use [需要真实信息].

## Stage 1 delivery (completed)
Official source retrieval with checksums; schema validation; deterministic split
manifest; train-only data audit and saved figure; one executed learning notebook;
tests and a README. No trained model or performance claim in this stage.
Test labels are not included in EDA. Global schema and duplication checks are
structural checks, not model selection.

## Subsequent stages (implemented; see reports/model_results.md)
2. Logistic regression and dummy baseline with leakage-safe preprocessing.
3. Random forest, validation comparisons, calibration and threshold choice.
4. Frozen final test report, interpretation, subgroup diagnostics and interview notes.

## Non-goals
Web deployment, SQL duplication, deep learning, SMOTE, SHAP, large tuning grids,
automated lending decisions, and production-readiness claims.
