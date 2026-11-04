# Model card

**Intended use:** educational analysis of existing-customer next-month default risk.
**Out of scope:** automated lending, approval/denial, production servicing, or compliance decisions.

## Data and evaluation

UCI Default of Credit Card Clients; 30,000 historical Taiwan records. Source:
https://doi.org/10.24432/C55S3H (CC BY 4.0). Historical feature window April–September
2005; supplied next-month binary label. Operational default definition/reporting
lag: [需要真实信息]. Seed-42 stratified ID split: 18,000/6,000/6,000.
No temporal backtest is possible from the supplied snapshot.

## Selection and inputs

21 allowed predictors; ID/outcome excluded. SEX and MARRIAGE audit-only; AGE retained.
Numeric inputs are standardized for logistic regression; nominal education and
repayment codes are one-hot encoded. Unknown codes map to an all-zero encoding
block at prediction time; their meaning is not guessed. Schema validation rejects
missing/nonfinite raw values; no unneeded imputation step is introduced.

Training-only three-fold AP selects the family/configuration. Selected model:
forest, forest with 150 trees, depth 12, minimum leaf 20.
Train-only sigmoid calibration was tested and rejected using validation Brier.
Validation scores fix threshold 0.307695 for an illustrative 20% budget.
Model objects and decisions were saved before test scoring; no post-selection refit.

## Test performance

ROC-AUC 0.7829, AP 0.5630, Brier
0.1340. Working-point precision 56.1%, recall
53.1%, flag rate 20.93%. Test n=6,000;
prevalence 22.12%. The fixed threshold is not a hard future quota.
Raw tables and assumptions: `../reports/model_results.md`.

## Risks and limitations

- Source-coded binary sex groups and fixed age bands are limited diagnostics.
  Same threshold across groups; small denominators (<20) suppressed.
  No fairness certification or formal compliance conclusion.
- Population/period shifts, existing-customer selection, ambiguous codes and hidden
  duplication can invalidate apparent performance. Different IDs with identical
  profiles are retained and overlap documented.
- Permutation importance and coefficients are associations, not causes.
- Costs, loss severity, review effectiveness and economic impact: [需要真实信息].
- No production endpoint, monitoring or retraining policy is implemented.

## Reproducibility

`artifacts/baseline/decision.json` contains source/split/code hashes, configuration,
validation criterion and threshold. Package versions are locked. Fitted objects
are local generated artifacts; source and run steps are the reproducible record.
Code and documentation were developed with AI assistance; personal review and
understanding: [需要真实信息].
