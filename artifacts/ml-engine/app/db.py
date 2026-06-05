"""Database access dispatcher.

When MSSQL_HOST is set, routes all DB calls through the MSSQL shim (db_mssql.py)
using pyodbc — no PostgreSQL / asyncpg / DATABASE_URL needed.
Otherwise falls back to the original asyncpg/PostgreSQL implementation (db_pg.py).
"""
import os as _os

if _os.environ.get("MSSQL_HOST"):
    from .db_mssql import *  # noqa: F401,F403
    from .db_mssql import _pool, _pool_loop, _get_conn  # noqa: F401 — needed by main.py
else:
    from .db_pg import *  # noqa: F401,F403
    from .db_pg import _pool, _pool_loop  # noqa: F401 — needed by main.py
