-- Add SEQUENCE + DEFAULT constraint for every serial id column
-- so Drizzle's INSERT ... VALUES (DEFAULT, ...) works

DECLARE @tables TABLE (tbl NVARCHAR(128));
INSERT INTO @tables VALUES
  ('agents'),('auto_deploy_attribution_snapshots'),('coin_correlations'),
  ('coin_insights'),('drift_snapshots'),('feature_lab_candidates'),
  ('feature_lab_reports'),('feature_lab_unquarantine_events'),('fingerprint_buffers'),
  ('journal_rollup'),('market_signals'),('model_predictions'),('model_registry'),
  ('monitoring_state'),('paper_portfolios'),('paper_position_marks'),('paper_positions'),
  ('paper_trades'),('prediction_journal'),('predictions'),('price_history'),
  ('quarantine_events'),('skip_events'),('strategy_settings_history'),
  ('strategy_snapshots'),('trade_journal');

DECLARE @tbl NVARCHAR(128), @sql NVARCHAR(MAX), @maxId BIGINT;

DECLARE cur CURSOR FOR SELECT tbl FROM @tables;
OPEN cur;
FETCH NEXT FROM cur INTO @tbl;

WHILE @@FETCH_STATUS = 0
BEGIN
  -- Drop existing sequence if any
  SET @sql = 'IF EXISTS (SELECT 1 FROM sys.sequences WHERE name = ''seq_' + @tbl + ''') DROP SEQUENCE [seq_' + @tbl + ']';
  EXEC sp_executesql @sql;

  -- Get current max id
  SET @sql = 'SELECT @out = ISNULL(MAX(id), 0) FROM [' + @tbl + ']';
  EXEC sp_executesql @sql, N'@out BIGINT OUTPUT', @out = @maxId OUTPUT;

  -- Create sequence starting after max id
  SET @sql = 'CREATE SEQUENCE [seq_' + @tbl + '] AS BIGINT START WITH ' + CAST(@maxId + 1 AS NVARCHAR) + ' INCREMENT BY 1 NO CYCLE';
  EXEC sp_executesql @sql;

  -- Drop existing default constraint on id if any
  SET @sql = '
  DECLARE @cname NVARCHAR(128);
  SELECT @cname = dc.name FROM sys.default_constraints dc
    JOIN sys.columns c ON dc.parent_object_id = c.object_id AND dc.parent_column_id = c.column_id
    JOIN sys.tables t ON c.object_id = t.object_id
  WHERE t.name = ''' + @tbl + ''' AND c.name = ''id'';
  IF @cname IS NOT NULL
    EXEC (''ALTER TABLE [' + @tbl + '] DROP CONSTRAINT ['' + @cname + '']'');';
  EXEC sp_executesql @sql;

  -- Add DEFAULT constraint using the sequence
  SET @sql = 'ALTER TABLE [' + @tbl + '] ADD CONSTRAINT [df_' + @tbl + '_id] DEFAULT (NEXT VALUE FOR [seq_' + @tbl + ']) FOR [id]';
  EXEC sp_executesql @sql;

  PRINT 'Done: ' + @tbl + ' (start=' + CAST(@maxId + 1 AS NVARCHAR) + ')';
  FETCH NEXT FROM cur INTO @tbl;
END;

CLOSE cur;
DEALLOCATE cur;
