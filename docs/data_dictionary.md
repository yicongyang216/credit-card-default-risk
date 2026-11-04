# Data dictionary and timing contract

Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
Yeh, I. (2009), DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
License: CC BY 4.0; original files retained without editing.

One row represents a customer record in the supplied historical Taiwan sample.
The workbook labels the outcome `default payment next month`; source values are
1 = default, 0 = no default. The project normalizes that name to `default`.
The operational definition of default (e.g., days past due) is not fully documented
on the repository page: [需要真实信息]. Do not equate this label to a regulatory definition.

| Source column | Meaning | Intended treatment |
|---|---|---|
| ID | Record identifier | Join/split key only; never a predictor |
| LIMIT_BAL | Credit limit, NT dollars | Numeric predictor |
| SEX | Source codes: 1 male, 2 female | Audit only; source coding does not capture all gender identities |
| EDUCATION | 1 graduate school, 2 university, 3 high school, 4 others | Categorical; other observed codes remain explicit unknown codes |
| MARRIAGE | 1 married, 2 single, 3 others | Audit only; other observed codes not guessed |
| AGE | Age in years | Numeric predictor; fixed audit bands: <30, 30–49, >=50 |
| PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | Repayment status, September back to April 2005 | Categorical predictors; source uses PAY_0, not PAY_1 |
| BILL_AMT1 … BILL_AMT6 | Monthly bill amounts, September back to April 2005, NT dollars | Numeric predictors; do not silently remove negative balances |
| PAY_AMT1 … PAY_AMT6 | Payments, September back to April 2005, NT dollars | Numeric predictors |
| default payment next month | Supplied future binary outcome | Target only |

For repayment status, UCI documents -1 as pay duly and 1–9 as months of delay
(9 = nine months or more). Codes 0 and -2 are not explained in that description;
their meaning is [需要真实信息]. Preserve them as separate categories rather than
inventing an interpretation, forcing ordinal distances, or treating them as null.

## Prediction boundary

The educational assumption is scoring after all September information is available,
using the supplied next-month outcome. Actual operational reporting lags are
[需要真实信息]. This is existing-customer behavioral scoring, not origination underwriting.
No customer-level series of scoring dates is available for temporal backtesting.
Random stratification measures generalization within this historical sample only.

## Duplicate policy

Repeated IDs are rejected. Exact records and identical predictor profiles are counted;
different IDs with identical values are retained because identity cannot be inferred
from shared values alone. Cross-split identical predictor profiles are reported as a
limitation, not hidden. This choice does not prove that the source contains no hidden
entity duplication. Do not use outcome labels to deduplicate customers.

## What stage 1 does not establish

Stage 1 alone establishes no model performance, calibration or subgroup conclusion.
The completed model runs are now documented in `../reports/model_results.md`.
Real business savings, intervention effectiveness and deployment benefit remain
[需要真实信息]; no formal fairness/compliance conclusion is established.
