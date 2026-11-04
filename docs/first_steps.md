# Stage 1 — understand the problem before fitting a model

Read `data_dictionary.md`, then the executed notebook
`notebooks/01_data_understanding.ipynb`. The notebook uses training data for EDA;
the validation and test IDs are reserved now so later choices cannot redefine them.

## Questions to answer in your own words

1. What does one row represent, what is the target, and when would the score be produced?
2. Why is historical late payment useful information rather than automatically leakage?
3. Why is an ID inappropriate as a predictor even if it appears correlated with default?
4. Why do we split before fitting scalers, imputers, encoders or choosing thresholds?
5. Why can we not claim that this random split tests performance on future US borrowers?
6. Why do unexplained numeric category codes need documentation rather than guessing?

## Interview explanation to understand, not memorize

> I defined this as next-month default prediction for existing card customers.
> I reserved customer IDs for training, validation and testing before model development.
> I used historical repayment information only under an explicit scoring-time assumption.
> Because the dataset is a historical snapshot, the holdout evaluates the same sampled
> setting, not temporal or geographic transfer. I documented ambiguous category codes
> rather than assigning them unsupported meanings.

## Your own observations

- Why you selected this problem: [需要真实信息]
- What you learned from inspecting the training data: [需要真实信息]
- One modeling decision you can independently explain: [需要真实信息]
- AI assistance used: code drafting and verification assistance in this chat;
  your independent review and understanding: [需要真实信息].

## Next stage

The modeling stage is implemented in `notebooks/02_model_evaluation.ipynb` and
`../reports/model_results.md`. After understanding this first notebook, read the
logistic baseline and constant-probability benchmark before examining the forest.
Learn ROC-AUC versus average precision before using the resume draft.
