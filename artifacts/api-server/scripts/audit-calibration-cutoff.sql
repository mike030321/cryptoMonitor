-- Audit: confirm the confidence calibrator cannot ingest pre-fleet-reset rows.
--
-- Run with:
--   psql "$DATABASE_URL" -f artifacts/api-server/scripts/audit-calibration-cutoff.sql
--
-- Expected output (post-reset, healthy):
--   pre_reset_rows                    = 0
--   pre_reset_resolved_rows           = 0
--   earliest_row >= '2026-04-21T00:00:00Z'
--
-- If any of the pre_reset_* counts is > 0, the calibrator's defensive
-- `gte(predictions.created_at, PREDICTION_FLEET_RESET_AT)` filter is doing
-- real work and the rows should be investigated/purged.

\set CUTOFF '2026-04-21T00:00:00Z'

SELECT
  COUNT(*)                                                                                                     AS total_predictions,
  COUNT(*) FILTER (WHERE created_at <  TIMESTAMP WITH TIME ZONE :'CUTOFF')                                     AS pre_reset_rows,
  COUNT(*) FILTER (WHERE created_at >= TIMESTAMP WITH TIME ZONE :'CUTOFF')                                     AS post_reset_rows,
  COUNT(*) FILTER (WHERE outcome IN ('correct','wrong')         AND created_at < TIMESTAMP WITH TIME ZONE :'CUTOFF') AS pre_reset_resolved_rows,
  MIN(created_at)                                                                                              AS earliest_row,
  MAX(created_at)                                                                                              AS latest_row
FROM predictions;

-- Per-bucket sanity: confirm no resolved bucket includes a pre-reset id.
SELECT
  width_bucket(COALESCE(raw_confidence, confidence), 0, 1, 10) AS confidence_bucket,
  COUNT(*)                                                       AS bucket_total,
  COUNT(*) FILTER (WHERE created_at < TIMESTAMP WITH TIME ZONE :'CUTOFF') AS pre_reset_in_bucket,
  MIN(id) FILTER (WHERE created_at < TIMESTAMP WITH TIME ZONE :'CUTOFF')  AS first_pre_reset_id
FROM predictions
WHERE outcome IN ('correct','wrong')
GROUP BY 1
ORDER BY 1;
