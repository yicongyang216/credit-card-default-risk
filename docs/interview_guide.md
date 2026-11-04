# Interview preparation

## 90-second overview to understand and personalize

I studied next-month default risk for existing credit card customers using a
historical UCI dataset with 30,000 records. I reserved a fixed training, validation
and test split before model development and placed learned preprocessing inside
pipelines. I compared interpretable logistic regression with a constrained random
forest using training-only cross-validation and average precision. The forest
was selected and reached test ROC-AUC 0.783 and AP 0.563. I evaluated sigmoid
calibration but kept raw forest probabilities because validation Brier was better.
An illustrative validation review budget produced a frozen threshold; on test it
flagged 20.9% of customers with 56.1% precision and 53.1% recall. I also documented
subgroup differences, ambiguous codes and the absence of temporal validation.
This was a learning project with AI-assisted implementation, not a deployed lender.

Your own motivation, decisions you personally verified and remaining gaps:
[需要真实信息]. Do not memorize statements you cannot explain.

## Questions and defensible answers

**1. What does a prediction mean?**
The supplied probability concerns next-month default for an existing customer,
assuming all September history is available at scoring time. It is not a probability
of loan approval, intervention success or a legally defined default rate.

**2. What would be leakage?**
Post-outcome recovery amounts, outcome-derived features or fitting preprocessing
on holdout data. Earlier payment history is allowed under the explicit prediction
boundary. ID is excluded even though it is available.

**3. Why three partitions?**
Training/CV fits models and compares a few configurations. Validation selects
calibration and threshold. The untouched test sample evaluates those frozen choices.
I did not then tune the model to its test score or refit on train+validation.

**4. Why a random split rather than a temporal split?**
Each record has the same historical feature window. The monthly columns are not
multiple independent scoring dates. Sorting IDs chronologically would invent time.
The evaluation supports only this sampled historical setting.

**5. Why logistic regression and random forest?**
Logistic regression models additive linear log-odds after encoding, is regularized
and offers interpretable coefficients. The forest checks whether nonlinear splits
and interactions add useful predictive ranking without introducing another library.
Its test AP advantage here is small; no claim of universal superiority is justified.

**6. How did you handle imbalance?**
I reported prevalence, AP and threshold precision/recall, compared balanced versus
unweighted logistic training, and kept validation/test prevalence unchanged.
Balanced weighting was not selected by training CV. Resampling is not mandatory.
Accuracy alone would reward predicting the majority class.

**7. ROC-AUC versus AP?**
ROC-AUC summarizes positive-versus-negative ranking across thresholds. AP summarizes
precision over recall increases and has a prevalence-dependent baseline. Here test
prevalence is about 22.1%, so the constant-probability baseline AP is about 0.221.
Neither tells me the precision or recall at one operating threshold.

**8. Why is 0.5 not automatically the right threshold?**
It does not encode a review budget or relative error costs. Lowering the threshold
here raises recall but increases flags and false positives. The 20% capacity was
an illustrative validation assumption. A fixed threshold flagged 20.9% on test;
a hard daily top-k budget would require a separate operational policy.

**9. What is calibration?**
Among cases assigned similar probabilities, observed event rates should roughly
agree with those probabilities. A model can rank well yet be poorly calibrated.
I used reliability diagrams and Brier/log loss, recognizing Brier also reflects
discrimination and uncertainty. Training-only sigmoid calibration did not improve
the forest's validation Brier, so I retained the raw model.

**10. How do you interpret coefficients and importance?**
Numeric LR coefficients are for one training-standard-deviation increases. Nominal
variables use full one-hot coding with L2 regularization, so compare category
coefficients within the same variable rather than inventing a dropped reference.
Permutation importance measures a validation AP drop after shuffling a column.
Correlated histories can share importance. Neither method identifies causes.

**11. Is the model fair?**
I cannot conclude that. I compared common-threshold group counts, recall, false
positive rate and precision, with small-denominator suppression. The dataset's
groups and population are limited; excluding sex/marital fields does not eliminate
proxies. These are diagnostics, not causal or legal compliance findings.

**12. What would you improve with more suitable data?**
Use verified customer identifiers, multiple scoring dates, explicit outcome horizons,
operational costs and intervention outcomes. Evaluate a future-period holdout and
check stability and subgroup uncertainty. These are proposed extensions, not completed work.

## Minimal reading order

1. `data_dictionary.md`: row, label, available information and unknown codes.
2. `models.py`: pipeline, scaling, one-hot encoding and the two estimators.
3. `decision.json`: configuration, calibration decision and frozen threshold.
4. `reports/model_results.md`: actual counts, tradeoffs and limits.
5. `test_modeling.py`: what the safeguards demonstrate and what they cannot prove.
