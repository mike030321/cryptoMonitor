"""
MSSQL-backed drop-in replacement for db.py's asyncpg interface.
Activated when MSSQL_HOST env var is set.
Uses pymssql on Linux/Render (no ODBC driver required),
pyodbc on Windows (local dev with ODBC Driver 18 installed).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import sys
import struct
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger("ml-engine.db.mssql")

# Auto-detect driver: pymssql on Linux (Render), pyodbc on Windows (local)
_USE_PYMSSQL: bool = sys.platform != "win32" or os.environ.get("MSSQL_USE_PYMSSQL") == "1"

if _USE_PYMSSQL:
    import pymssql as _db_driver  # type: ignore
else:
    import pyodbc as _db_driver  # type: ignore

def _conn_params() -> dict:
    return {
        "host": os.environ.get("MSSQL_HOST", "sql6034.site4now.net"),
        "database": os.environ.get("MSSQL_DB", "db_aca32a_cryptoai"),
        "user": os.environ.get("MSSQL_USER", "db_aca32a_cryptoai_admin"),
        "password": os.environ.get("MSSQL_PASSWORD", "Ciyvi.123"),
    }

# ── DATETIMEOFFSET converter for pyodbc (Windows only) ───────────────────────

def _handle_datetimeoffset(raw: bytes) -> datetime:
    tup = struct.unpack("<6hI2h", raw)
    yr, mo, dy, hr, mn, sc, ns, tz_h, tz_m = tup
    tz = timezone(timedelta(hours=tz_h, minutes=tz_m))
    return datetime(yr, mo, dy, hr, mn, sc, ns // 1000, tzinfo=tz)

# ── thread-local connection pool ──────────────────────────────────────────────

_local = threading.local()

def _get_conn():
    """Return a per-thread connection. pymssql on Linux, pyodbc on Windows."""
    conn = getattr(_local, "conn", None)
    try:
        if conn is not None:
            if _USE_PYMSSQL:
                conn.cursor().execute("SELECT 1")
            else:
                conn.execute("SELECT 1")
            return conn
    except Exception:
        pass

    p = _conn_params()
    if _USE_PYMSSQL:
        conn = _db_driver.connect(
            server=p["host"], user=p["user"],
            password=p["password"], database=p["database"],
            tds_version="7.4", autocommit=True,
        )
    else:
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={p['host']};DATABASE={p['database']};"
            f"UID={p['user']};PWD={p['password']};"
            f"TrustServerCertificate=yes;Encrypt=yes;"
        )
        conn = _db_driver.connect(conn_str, autocommit=True)
        conn.add_output_converter(-155, _handle_datetimeoffset)

    _local.conn = conn
    return conn

async def _run(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)

# ── SQL translation: PostgreSQL → T-SQL ───────────────────────────────────────

def _translate(sql: str) -> str:
    s = sql.strip()
    # $1,$2... → ? (pyodbc) or %s (pymssql)
    placeholder = "%s" if _USE_PYMSSQL else "?"
    s = re.sub(r'\$\d+', placeholder, s)
    # double-quoted identifiers → square brackets
    s = re.sub(r'"(\w+)"', r'[\1]', s)
    # Quote unquoted MSSQL reserved words used as column names
    for rw in ('key', 'value', 'timestamp', 'open', 'close', 'level', 'status',
               'name', 'type', 'source', 'data', 'schema', 'identity'):
        s = re.sub(
            r'(?<![.\["\w])' + rw + r'(?!["\]\w])',
            f'[{rw}]', s, flags=re.IGNORECASE
        )
    # boolean literals
    s = s.replace(' true', ' 1').replace(' false', ' 0')
    # COALESCE(col, false) / COALESCE(col, true)
    s = re.sub(r'COALESCE\s*\(([^,]+),\s*false\)', r'COALESCE(\1, 0)', s, flags=re.IGNORECASE)
    s = re.sub(r'COALESCE\s*\(([^,]+),\s*true\)', r'COALESCE(\1, 1)', s, flags=re.IGNORECASE)
    # NOW() → GETDATE()
    s = re.sub(r'\bNOW\(\)', 'GETDATE()', s, flags=re.IGNORECASE)
    # is_synthetic = false/true
    s = re.sub(r'\bIS_SYNTHETIC\b\s*=\s*false', '[is_synthetic] = 0', s, flags=re.IGNORECASE)
    s = re.sub(r'\bIS_SYNTHETIC\b\s*=\s*true', '[is_synthetic] = 1', s, flags=re.IGNORECASE)
    # pg advisory lock → always true/no-op
    s = re.sub(r'SELECT\s+pg_try_advisory_lock\([^)]+\)', 'SELECT 1', s, flags=re.IGNORECASE)
    s = re.sub(r'SELECT\s+pg_advisory_unlock\([^)]+\)', 'SELECT 1', s, flags=re.IGNORECASE)
    # LIMIT n → OFFSET 0 ROWS FETCH NEXT n ROWS ONLY (needs ORDER BY)
    # Use lookahead instead of \b so it matches LIMIT ? (? is non-word char)
    def rewrite_limit(m):
        n = m.group(1)
        if not re.search(r'\bORDER\s+BY\b', s, re.IGNORECASE):
            return f'ORDER BY (SELECT NULL) OFFSET 0 ROWS FETCH NEXT {n} ROWS ONLY'
        return f'OFFSET 0 ROWS FETCH NEXT {n} ROWS ONLY'
    s = re.sub(r'\bLIMIT\s+(\?|\d+)(?=\s|$|;|\))', rewrite_limit, s, flags=re.IGNORECASE)
    return s

# ── Row adapter (makes pyodbc rows behave like asyncpg Records) ───────────────

class _Row(dict):
    """Dict subclass so callers can use both row['col'] and row.col."""
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def _rows_from_cursor(cursor) -> list[_Row]:
    if cursor.description is None:
        return []
    rows = cursor.fetchall()
    if not rows:
        return []
    # pymssql with as_dict=True returns dicts; pyodbc returns tuples
    if isinstance(rows[0], dict):
        return [_Row({k.lower(): v for k, v in r.items()}) for r in rows]
    cols = [d[0].lower() for d in cursor.description]
    return [_Row(zip(cols, row)) for row in rows]

def _one_from_cursor(cursor) -> Optional[_Row]:
    if cursor.description is None:
        return None
    row = cursor.fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return _Row({k.lower(): v for k, v in row.items()})
    cols = [d[0].lower() for d in cursor.description]
    return _Row(zip(cols, row))

# ── Pool-like object ──────────────────────────────────────────────────────────

class _MssqlPool:
    async def fetch(self, query: str, *args) -> list[_Row]:
        # Handle PostgreSQL unnest() bulk-array insert (used by backfill scripts)
        if 'unnest' in query.lower() and args and isinstance(args[0], list):
            return await self._handle_unnest_insert(query, args)
        sql = _translate(query)
        def _do():
            conn = _get_conn()
            if _USE_PYMSSQL:
                cur = conn.cursor(as_dict=True)
                cur.execute(sql, list(args) if args else None)
            else:
                cur = conn.execute(sql, list(args))
            return _rows_from_cursor(cur)
        try:
            return await _run(_do)
        except Exception as exc:
            logger.warning("mssql_fetch_failed", extra={"error": str(exc), "sql": sql[:120]})
            raise

    async def _handle_unnest_insert(self, query: str, args) -> list[_Row]:
        """Convert PostgreSQL unnest(arr1, arr2, ...) bulk insert to MSSQL executemany."""
        # Extract table and column names from the INSERT
        insert_m = re.search(
            r'INSERT INTO (\w+)\s*\(([^)]+)\)', query, re.IGNORECASE
        )
        returning_m = re.search(r'RETURNING\s+(\w+)', query, re.IGNORECASE)
        conflict_col_m = re.search(r'ON CONFLICT\s*\(([^)]+)\)', query, re.IGNORECASE)

        if not insert_m:
            raise ValueError("Cannot parse unnest INSERT query")

        table = insert_m.group(1)
        cols = [c.strip() for c in insert_m.group(2).split(',')]
        ret_col = returning_m.group(1) if returning_m else None
        conflict_cols = [c.strip() for c in conflict_col_m.group(1).split(',')] if conflict_col_m else []

        # Build rows from the N parallel arrays
        rows = list(zip(*args))

        quoted_cols = ', '.join(f'[{c}]' if c.lower() in
            ('key','value','timestamp','open','close','level','status','name','type','source','data')
            else c for c in cols)
        placeholders = ', '.join(['?'] * len(cols))

        results = []

        def _do_bulk():
            conn = _get_conn()
            inserted = []
            for row in rows:
                if conflict_cols:
                    # Check existence first (ON CONFLICT DO NOTHING equivalent)
                    where = ' AND '.join(
                        f'[{c}] = ?' if c.lower() in ('timestamp','key','value','source')
                        else f'{c} = ?'
                        for c in conflict_cols
                    )
                    idx = {c: cols.index(c) for c in conflict_cols if c in cols}
                    where_vals = [row[cols.index(c)] for c in conflict_cols if c in cols]
                    try:
                        exists_cur = conn.execute(
                            f"SELECT 1 FROM [{table}] WHERE {where}", where_vals
                        )
                        if exists_cur.fetchone():
                            continue
                    except Exception:
                        pass

                try:
                    conn.execute(
                        f"INSERT INTO [{table}] ({quoted_cols}) VALUES ({placeholders})",
                        list(row)
                    )
                    if ret_col and ret_col in cols:
                        inserted.append(_Row({ret_col: row[cols.index(ret_col)]}))
                except Exception as e:
                    if 'duplicate' in str(e).lower() or '2627' in str(e) or '2601' in str(e):
                        pass  # ON CONFLICT DO NOTHING
                    else:
                        raise
            return inserted

        try:
            results = await _run(_do_bulk)
            return results
        except Exception as exc:
            logger.warning("mssql_fetch_failed", extra={"error": str(exc), "sql": query[:120]})
            raise

    async def fetchrow(self, query: str, *args) -> Optional[_Row]:
        sql = _translate(query)
        def _do():
            conn = _get_conn()
            if _USE_PYMSSQL:
                cur = conn.cursor(as_dict=True)
                cur.execute(sql, list(args) if args else None)
            else:
                cur = conn.execute(sql, list(args))
            return _one_from_cursor(cur)
        try:
            return await _run(_do)
        except Exception as exc:
            logger.warning("mssql_fetchrow_failed", extra={"error": str(exc)})
            raise

    async def fetchval(self, query: str, *args) -> Any:
        sql = _translate(query)
        def _do():
            conn = _get_conn()
            if _USE_PYMSSQL:
                cur = conn.cursor()
                cur.execute(sql, list(args) if args else None)
            else:
                cur = conn.execute(sql, list(args))
            row = cur.fetchone()
            return (row[0] if isinstance(row, (list, tuple)) else next(iter(row.values()))) if row else None
        try:
            return await _run(_do)
        except Exception as exc:
            logger.warning("mssql_fetchval_failed", extra={"error": str(exc)})
            raise

    async def execute(self, query: str, *args) -> None:
        sql = _translate(query)
        def _do():
            conn = _get_conn()
            if _USE_PYMSSQL:
                cur = conn.cursor()
                cur.execute(sql, list(args) if args else None)
            else:
                conn.execute(sql, list(args))
        try:
            await _run(_do)
        except Exception as exc:
            logger.warning("mssql_execute_failed", extra={"error": str(exc)})
            raise

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield _MssqlConn(self)

class _MssqlConn:
    def __init__(self, pool: _MssqlPool):
        self._pool = pool
    async def fetchval(self, query: str, *args) -> Any:
        return await self._pool.fetchval(query, *args)
    async def fetch(self, query: str, *args) -> list[_Row]:
        return await self._pool.fetch(query, *args)
    async def fetchrow(self, query: str, *args) -> Optional[_Row]:
        return await self._pool.fetchrow(query, *args)
    async def execute(self, query: str, *args) -> None:
        return await self._pool.execute(query, *args)

# ── Public API (mirrors db.py) ────────────────────────────────────────────────

# Exposed so main.py's training detach/reattach pattern works
_pool: Optional[_MssqlPool] = None
_pool_loop = None  # asyncpg compat stub — MSSQL shim is not loop-bound

def _get_dispatcher_pool():
    """Return app.db._pool — main.py uses db_mod._pool = None to force re-init."""
    import sys
    db_mod = sys.modules.get("app.db")
    if db_mod is not None:
        return getattr(db_mod, "_pool", _pool)
    return _pool

def _set_dispatcher_pool(p):
    global _pool
    _pool = p
    import sys
    db_mod = sys.modules.get("app.db")
    if db_mod is not None:
        db_mod._pool = p

async def init_pool() -> _MssqlPool:
    global _pool
    # Respect main.py's detach pattern: if app.db._pool was set to None, reset ours too
    ext = _get_dispatcher_pool()
    if ext is None:
        _pool = None
    elif ext is not None and not isinstance(ext, _MssqlPool):
        # main.py restored a saved pool that's actually a _MssqlPool
        _pool = ext

    if _pool is None:
        def _test():
            conn = _get_conn()
            conn.execute("SELECT 1")
        await _run(_test)
        _pool = _MssqlPool()
        _set_dispatcher_pool(_pool)
        logger.info("mssql_pool_ready")
    return _pool

async def close_pool() -> None:
    global _pool, _CONN
    _pool = None
    if _CONN is not None:
        try:
            _CONN.close()
        except Exception:
            pass
        _CONN = None

def advisory_lock_key(label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)

@contextlib.asynccontextmanager
async def try_advisory_lock(key: int) -> AsyncIterator[bool]:
    """MSSQL has no advisory locks — always yield True (single-process)."""
    yield True

# Re-export the same fetch functions as db.py using the MSSQL pool

async def fetch_real_ticks(
    coin_id: str, lookback_ms: int, now: Optional[datetime] = None
) -> list[tuple[datetime, float]]:
    pool = await init_pool()
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(milliseconds=lookback_ms)
    rows = await pool.fetch(
        """
        SELECT [timestamp], price
        FROM price_history
        WHERE coin_id = ?
          AND ([is_synthetic] IS NULL OR [is_synthetic] = 0)
          AND [timestamp] >= ?
          AND [timestamp] <= ?
        ORDER BY [timestamp] ASC
        """,
        coin_id, start, end,
    )
    return [(r["timestamp"], float(r["price"])) for r in rows]

async def fetch_real_candles(
    coin_id: str, timeframe: str, lookback_ms: int, now: Optional[datetime] = None,
) -> list[tuple[datetime, float]]:
    try:
        pool = await init_pool()
        end = now or datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=lookback_ms)
        rows = await pool.fetch(
            """
            SELECT bucket_start, [close]
            FROM price_candles
            WHERE coin_id = ?
              AND timeframe = ?
              AND bucket_start >= ?
              AND bucket_start <= ?
            ORDER BY bucket_start ASC
            """,
            coin_id, timeframe, start, end,
        )
    except Exception:
        return []
    return [(r["bucket_start"], float(r["close"])) for r in rows]

async def fetch_real_ticks_with_provenance(
    coin_id: str, lookback_ms: int, now: Optional[datetime] = None,
) -> dict:
    pool = await init_pool()
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(milliseconds=lookback_ms)
    rows = await pool.fetch(
        """
        SELECT [timestamp], price, COALESCE([is_synthetic], 0) AS is_synthetic
        FROM price_history
        WHERE coin_id = ?
          AND [timestamp] >= ?
          AND [timestamp] <= ?
        ORDER BY [timestamp] ASC
        """,
        coin_id, start, end,
    )
    ticks, n_real, n_synth = [], 0, 0
    for r in rows:
        if r["is_synthetic"]:
            n_synth += 1
            continue
        n_real += 1
        ticks.append((r["timestamp"], float(r["price"])))
    return {
        "ticks": ticks, "rows_real": n_real, "rows_synthetic": n_synth,
        "rejected_synthetic": n_synth > 0,
        "window_start": start, "window_end": end,
    }

async def fetch_recent_news_tags(
    coin_id: str, limit: int = 20, now: Optional[datetime] = None
) -> list[str]:
    return []  # news_tags table not present

async def fetch_market_signals(
    coin_id: str, lookback_ms: int, now: Optional[datetime] = None,
) -> list[dict]:
    try:
        pool = await init_pool()
        end = now or datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=lookback_ms)
        rows = await pool.fetch(
            """
            SELECT [timestamp], funding_rate, open_interest_usd,
                   liquidations_1h_usd, bid_ask_spread_bps, mid_price
            FROM market_signals
            WHERE coin_id = ?
              AND [timestamp] >= ?
              AND [timestamp] <= ?
            ORDER BY [timestamp] ASC
            """,
            coin_id, start, end,
        )
    except Exception:
        return []
    out = []
    for r in rows:
        out.append({
            "timestamp_ms": int(r["timestamp"].timestamp() * 1000),
            "funding_rate": float(r["funding_rate"]) if r["funding_rate"] is not None else None,
            "open_interest_usd": float(r["open_interest_usd"]) if r["open_interest_usd"] is not None else None,
            "liquidations_1h_usd": float(r["liquidations_1h_usd"]) if r["liquidations_1h_usd"] is not None else None,
            "bid_ask_spread_bps": float(r["bid_ask_spread_bps"]) if r["bid_ask_spread_bps"] is not None else None,
            "mid_price": float(r["mid_price"]) if r["mid_price"] is not None else None,
        })
    return out

async def fetch_lead_price_series(
    reference_coin_id: str, lookback_ms: int, now: Optional[datetime] = None,
) -> list[tuple[int, float]]:
    rows = await fetch_market_signals(reference_coin_id, lookback_ms, now=now)
    return [
        (r["timestamp_ms"], r["mid_price"])
        for r in rows
        if r["mid_price"] is not None and r["mid_price"] > 0
    ]

async def fetch_scope_for_active_champion(
    coin_id: str, timeframe: str, model_version: str,
) -> Optional[dict]:
    try:
        pool = await init_pool()
        row = await pool.fetchrow(
            """
            SELECT scope_constraint FROM model_registry
            WHERE coin_id = ? AND timeframe = ? AND model_version = ?
              AND [state] = 'champion' AND is_active = 1
              AND scope_constraint IS NOT NULL
            """,
            coin_id, timeframe, model_version,
        )
    except Exception:
        return None
    if row is None:
        return None
    scope = row["scope_constraint"]
    if scope is None:
        return None
    if isinstance(scope, str):
        try:
            import json as _json
            scope = _json.loads(scope)
        except Exception:
            return None
    return scope if isinstance(scope, dict) else None

async def fetch_quarantined_versions() -> set[tuple[str, str, str]]:
    try:
        pool = await init_pool()
        rows = await pool.fetch(
            """
            SELECT coin_id, timeframe, model_version FROM model_registry
            WHERE [state] = 'quarantined' AND is_active = 1
            """
        )
    except Exception:
        return set()
    return {(str(r["coin_id"]), str(r["timeframe"]), str(r["model_version"])) for r in rows}
