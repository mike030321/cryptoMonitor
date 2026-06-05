/**
 * Fake pg-compatible pool backed by MSSQL.
 * Drizzle generates PostgreSQL SQL; this layer translates it to T-SQL and
 * executes it via the `mssql` package. The app code and schema files stay
 * 100% unchanged.
 */
import sql from "mssql";

// ── connection ────────────────────────────────────────────────────────────────

const config: sql.config = {
  server: process.env.MSSQL_HOST ?? "sql6034.site4now.net",
  database: process.env.MSSQL_DB ?? "db_aca32a_cryptoai",
  user: process.env.MSSQL_USER ?? "db_aca32a_cryptoai_admin",
  password: process.env.MSSQL_PASSWORD ?? "Ciyvi.123",
  port: 1433,
  options: { encrypt: true, trustServerCertificate: true },
  pool: { min: 2, max: 10 },
};

let _pool: sql.ConnectionPool | null = null;

export async function getMssqlPool(): Promise<sql.ConnectionPool> {
  if (!_pool) {
    _pool = await new sql.ConnectionPool(config).connect();
  }
  return _pool;
}

// ── SQL translation: PostgreSQL → T-SQL ───────────────────────────────────────

function translatePgToMssql(pgSql: unknown): string {
  const s0 = typeof pgSql === "string" ? pgSql : (pgSql as any)?.text ?? (pgSql as any)?.sql ?? String(pgSql);
  let s: string = s0;

  // 1. Double-quoted identifiers → square-bracket identifiers
  //    "table"."column" → [table].[column]
  s = s.replace(/"(\w+)"/g, "[$1]");

  // 2. Parameter placeholders: $1 → @p1
  s = s.replace(/\$(\d+)/g, "@p$1");

  // 2b. Remove PostgreSQL type casts: ::int, ::text, ::numeric, ::boolean, ::regclass, etc.
  s = s.replace(/::[a-zA-Z_]+(\[\])?(\(\d+(?:,\d+)?\))?/g, "");

  // 3. Boolean literals
  s = s.replace(/\bTRUE\b/g, "1").replace(/\bFALSE\b/g, "0");

  // 4. NOW() → GETDATE()
  s = s.replace(/\bNOW\(\)/gi, "GETDATE()");

  // 5. ILIKE → LIKE (MSSQL is case-insensitive by default)
  s = s.replace(/\bILIKE\b/gi, "LIKE");

  // 5b. INTERVAL expressions: expr ± INTERVAL 'N unit' → DATEADD(unit, ±N, expr)
  s = rewriteIntervals(s);

  // 6. RETURNING → OUTPUT INSERTED.*
  //    Pattern: "... VALUES (...) RETURNING col1, col2"
  //    →        "... OUTPUT INSERTED.col1, INSERTED.col2 VALUES (...)"
  s = rewriteReturning(s);

  // 6b. Strip DEFAULT values from INSERT column/value lists
  //     Drizzle: INSERT INTO [t] ([id], [col]) VALUES (DEFAULT, @p1)
  //     → INSERT INTO [t] ([col]) VALUES (@p1)
  //     MSSQL sequence DEFAULT constraint handles the id.
  s = rewriteInsertDefaults(s);

  // 7. ON CONFLICT DO UPDATE → MERGE
  s = rewriteOnConflict(s);

  // 8. LIMIT [n] / LIMIT [n] OFFSET [m]
  //    Must appear after ORDER BY; add ORDER BY (SELECT NULL) if missing
  s = rewriteLimit(s);

  // 9. Row-value comparisons: (a, b) > ($p1, $p2) → expanded form
  s = rewriteRowValueCmp(s);

  // 10. NULLS FIRST / NULLS LAST (not supported in MSSQL)
  s = s.replace(/\bNULLS\s+(?:FIRST|LAST)\b/gi, "");

  return s.trim();
}

// ── INSERT DEFAULT stripping ──────────────────────────────────────────────────

function rewriteInsertDefaults(s: string): string {
  // Match: INSERT INTO [tbl] ([col1], [col2], ...) VALUES (DEFAULT, @p1, ...), ...
  // Find the column list and all value tuples, then strip DEFAULT columns.
  const insertRe = /^(INSERT\s+INTO\s+\[\w+\])\s+\(([^)]+)\)\s+((?:OUTPUT[^V]+)?VALUES\s+)((?:\([^)]+\)(?:\s*,\s*)?)+)/im;
  const m = s.match(insertRe);
  if (!m) return s;

  const cols = m[2].split(",").map(c => c.trim());
  const defaultIdxs = cols.reduce<number[]>((acc, c, i) => {
    // Check if this column has DEFAULT in ANY of the value tuples
    return acc;
  }, []);

  // Check first value tuple for DEFAULT positions
  const firstTupleMatch = m[4].match(/\(([^)]+)\)/);
  if (!firstTupleMatch) return s;
  const firstVals = firstTupleMatch[1].split(",").map(v => v.trim());

  const keepIdxs: number[] = [];
  firstVals.forEach((v, i) => {
    if (!/^DEFAULT$/i.test(v)) keepIdxs.push(i);
  });

  if (keepIdxs.length === firstVals.length) return s; // no DEFAULT cols

  const newCols = keepIdxs.map(i => cols[i]).join(", ");

  // Rewrite all value tuples
  const newVals = m[4].replace(/\(([^)]+)\)/g, (_, inner) => {
    const vals = inner.split(",").map((v: string) => v.trim());
    return "(" + keepIdxs.map(i => vals[i]).join(", ") + ")";
  });

  return s.replace(insertRe, `$1 (${newCols}) $3${newVals}`);
}

// ── RETURNING ─────────────────────────────────────────────────────────────────

function rewriteReturning(s: string): string {
  // Match: INSERT INTO [tbl] ([cols]) VALUES (...) RETURNING <cols>
  // Or with ON CONFLICT (handle RETURNING only — ON CONFLICT handled separately)
  const re = /\)\s+RETURNING\s+(.+?)(?:$|(?=\s+ON\b))/si;
  const m = s.match(re);
  if (!m) return s;

  const retCols = m[1]
    .trim()
    .split(",")
    .map((c) => `INSERTED.${c.trim()}`)
    .join(", ");

  // Find the position right after INSERT INTO [tbl] ([cols])
  // Insert OUTPUT clause before VALUES
  s = s.replace(re, ")"); // remove RETURNING first
  s = s.replace(/\)\s+VALUES\s+\(/i, `) OUTPUT ${retCols} VALUES (`);
  return s;
}

// ── ON CONFLICT DO UPDATE ─────────────────────────────────────────────────────

function rewriteOnConflict(s: string): string {
  // Pattern:
  //   INSERT INTO [tbl] ([c1],[c2],...) VALUES (@p1,@p2,...) ON CONFLICT ([target]) DO UPDATE SET [c1]=EXCLUDED.[c1],...
  const insertRe =
    /INSERT INTO (\[\w+\])\s+\(([^)]+)\)\s+(?:OUTPUT[^V]+)?VALUES\s+\(([^)]+)\)\s+ON CONFLICT\s+\(([^)]+)\)\s+DO UPDATE SET\s+(.+?)(?:;|$)/si;

  const m = s.match(insertRe);
  if (!m) return s;

  const [, table, colsRaw, valsRaw, targetRaw, setRaw] = m;
  const cols = colsRaw.split(",").map((c) => c.trim());
  const vals = valsRaw.split(",").map((v) => v.trim());
  const targets = targetRaw.split(",").map((t) => t.trim());

  // Build MERGE ON clause
  const onClause = targets
    .map((t) => `target.${t} = source.${t}`)
    .join(" AND ");

  // Build UPDATE SET (replace EXCLUDED. with source.)
  const updateSet = setRaw
    .trim()
    .replace(/EXCLUDED\./g, "source.")
    .split(",")
    .map((p) => p.trim())
    .join(", ");

  // Build source columns/values
  const sourceCols = cols.join(", ");
  const sourceVals = vals.join(", ");

  const merge =
    `MERGE INTO ${table} AS target ` +
    `USING (VALUES (${sourceVals})) AS source(${sourceCols}) ` +
    `ON ${onClause} ` +
    `WHEN MATCHED THEN UPDATE SET ${updateSet} ` +
    `WHEN NOT MATCHED THEN INSERT (${sourceCols}) VALUES (${sourceVals});`;

  return s.replace(insertRe, merge);
}

// ── LIMIT / OFFSET ────────────────────────────────────────────────────────────

function rewriteLimit(s: string): string {
  // LIMIT n OFFSET m
  s = s.replace(
    /\bLIMIT\s+(@p\d+|\d+)\s+OFFSET\s+(@p\d+|\d+)\b/gi,
    "OFFSET $2 ROWS FETCH NEXT $1 ROWS ONLY"
  );
  // OFFSET m LIMIT n (alternate ordering)
  s = s.replace(
    /\bOFFSET\s+(@p\d+|\d+)\s+LIMIT\s+(@p\d+|\d+)\b/gi,
    "OFFSET $1 ROWS FETCH NEXT $2 ROWS ONLY"
  );
  // Bare LIMIT n (no existing OFFSET)
  s = s.replace(/\bLIMIT\s+(@p\d+|\d+)\b/gi, (_, n) => {
    // If no ORDER BY present, add a dummy one (MSSQL requires it)
    if (!/ORDER BY/i.test(s)) {
      s = s.replace(/\bFROM\b/i, "FROM"); // no-op just to trigger rebuild
      return `ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT ${n} ROWS ONLY`;
    }
    return `OFFSET 0 ROWS FETCH NEXT ${n} ROWS ONLY`;
  });
  return s;
}

// ── INTERVAL expressions ──────────────────────────────────────────────────────

const INTERVAL_UNIT: Record<string, string> = {
  second: "second", seconds: "second",
  minute: "minute", minutes: "minute",
  hour: "hour", hours: "hour",
  day: "day", days: "day",
  week: "week", weeks: "week",
  month: "month", months: "month",
  year: "year", years: "year",
};

function rewriteIntervals(s: string): string {
  // Pattern: expr - INTERVAL 'N unit'  or  expr + INTERVAL 'N unit'
  return s.replace(
    /(\S+)\s*([+-])\s*INTERVAL\s+'(\d+(?:\.\d+)?)\s+(\w+)'/gi,
    (_, expr, op, n, unit) => {
      const mssqlUnit = INTERVAL_UNIT[unit.toLowerCase()] ?? unit;
      const amount = op === "-" ? `-${n}` : n;
      return `DATEADD(${mssqlUnit}, ${amount}, ${expr})`;
    }
  );
}

// ── Row-value comparisons ─────────────────────────────────────────────────────

function rewriteRowValueCmp(s: string): string {
  // (col1, col2) > (val1, val2)  →  (col1 > val1 OR (col1 = val1 AND col2 > val2))
  return s.replace(
    /\((\S+),\s*(\S+)\)\s*>\s*\((\S+),\s*(\S+)\)/g,
    "($1 > $3 OR ($1 = $3 AND $2 > $4))"
  );
}

// ── Fake pg Pool (Drizzle expects this interface) ─────────────────────────────

interface PgResult {
  rows: Record<string, unknown>[];
  rowCount: number | null;
}

export class MssqlPgPool {
  // Drizzle calls pool.connect() to get a client; we return ourselves
  async connect(): Promise<this> { return this; }
  release(): void { /* no-op */ }

  async query(
    textOrConfig: any,
    valuesArg?: unknown[]
  ): Promise<PgResult> {
    // Drizzle may pass a string, {text,values}, or a prepared query object
    let text: string;
    let values: unknown[];
    if (typeof textOrConfig === "string") {
      text = textOrConfig;
      values = valuesArg ?? [];
    } else if (textOrConfig && typeof textOrConfig.text === "string") {
      text = textOrConfig.text;
      values = textOrConfig.values ?? valuesArg ?? [];
    } else if (textOrConfig && typeof textOrConfig.sql === "string") {
      text = textOrConfig.sql;
      values = textOrConfig.params ?? valuesArg ?? [];
    } else {
      // Last resort: stringify and log
      console.error("[mssql-pool] unknown query format:", typeof textOrConfig, JSON.stringify(textOrConfig)?.slice(0, 200));
      throw new Error(`[mssql-pool] Cannot extract SQL from: ${typeof textOrConfig}`);
    }
    const pool = await getMssqlPool();
    const mssqlSql = translatePgToMssql(text);

    const req = pool.request();
    if (values) {
      values.forEach((v, i) => {
        // Map JS types to MSSQL types
        if (v === null || v === undefined) {
          req.input(`p${i + 1}`, sql.NVarChar, null);
        } else if (typeof v === "boolean") {
          req.input(`p${i + 1}`, sql.Bit, v ? 1 : 0);
        } else if (typeof v === "number") {
          req.input(`p${i + 1}`, Number.isInteger(v) ? sql.Int : sql.Float, v);
        } else if (v instanceof Date) {
          req.input(`p${i + 1}`, sql.DateTimeOffset, v);
        } else {
          req.input(`p${i + 1}`, sql.NVarChar(sql.MAX), String(v));
        }
      });
    }

    try {
      const result = await req.query(mssqlSql);
      const recordset = result.recordset ?? [];
      const rowCount = result.rowsAffected[0] ?? recordset.length;

      // Drizzle uses rowMode:'array' for typed selects — it expects each row as
      // an array of values in column order. Convert MSSQL objects to arrays.
      const rowMode = typeof textOrConfig === "object" ? textOrConfig?.rowMode : undefined;
      if (rowMode === "array" && recordset.length > 0) {
        const colMeta = (recordset as any).columns as Record<string, { index: number }> | undefined;
        let orderedCols: string[];
        if (colMeta) {
          orderedCols = Object.entries(colMeta)
            .sort(([, a], [, b]) => a.index - b.index)
            .map(([name]) => name);
        } else {
          orderedCols = Object.keys(recordset[0]);
        }
        return {
          rows: recordset.map((row) => orderedCols.map((col) => row[col])),
          rowCount,
        };
      }

      return { rows: recordset as Record<string, unknown>[], rowCount };
    } catch (err) {
      const e = err as Error;
      // Re-throw with original + translated SQL for easier debugging
      throw Object.assign(new Error(`MSSQL error: ${e.message}\nOriginal SQL: ${text}\nTranslated SQL: ${mssqlSql}`), { cause: err });
    }
  }
}
