# Retired one-off scripts

This file records one-off operational scripts that have been removed from
the repo. Each entry exists so future contributors can understand why a
script disappeared without having to dig through git history.

## `backfill-prior-source.sql` — removed in Task #506

A one-off backfill that tagged pre-Task #107 QUANT-prior rows in
`predictions` and `model_predictions` with `source = 'prior'`. It joined
`model_predictions` back to `predictions` via
`model_predictions.llm_prediction_id = predictions.id`.

Task #506 dropped the `llm_prediction_id` column (along with the rest
of the dead LLM-era join columns) from `model_predictions`, which made
the join unrunnable. The script had already executed successfully
during the Task #107 / #111 rollout and is idempotent in that completed
state, so it has no remaining operational value.
