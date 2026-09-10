"""
📦  FORECAST STORAGE — Postgres (Railway), replacing Firestore for this
one collection.

Why: a saved forecast document (predictions + historical snapshot) can
comfortably exceed Firestore's 1 MiB per-document limit once a schema has
many groups (products/stores) and/or a long horizon — that's exactly what
was silently breaking "View Insights" and the Past Forecasts dropdown.
Postgres has no such per-row size ceiling (JSONB columns are TOASTed
automatically), so forecasts now live here instead. Everything else
(auth, models, billing, users) stays on Firestore/Firebase — only this
one table moved.

Connects via DATABASE_URL (set this on the FastAPI service in Railway —
see config.py). Uses a small psycopg2 connection pool since this is a
single-process server (same assumption ratelimit.py already makes).
"""
import json
import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

from config import settings

logger = logging.getLogger("forecastiq")

_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        if not settings.DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set — add it to this service's env vars "
                "(Railway: reference your Postgres service's DATABASE_URL)."
            )
        _pool = SimpleConnectionPool(1, settings.DB_POOL_MAX, dsn=settings.DATABASE_URL)
    return _pool


@contextmanager
def _cursor(commit: bool = False):
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db() -> None:
    """Creates the forecasts table/index if they don't exist yet. Safe to
    call on every startup (CREATE ... IF NOT EXISTS)."""
    with _cursor(commit=True) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                uid            TEXT NOT NULL,
                owner_email    TEXT,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                months         INTEGER,
                model_id       TEXT,
                model_name     TEXT,
                train_rmse     DOUBLE PRECISION,
                val_rmse       DOUBLE PRECISION,
                baseline_rmse  DOUBLE PRECISION,
                schema         JSONB,
                predictions    JSONB,
                historical     JSONB
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecasts_uid_created "
            "ON forecasts (uid, created_at DESC);"
        )
        # gen_random_uuid() needs pgcrypto on older Postgres — Railway's
        # managed Postgres (15+) ships it enabled, but this is a harmless
        # no-op if it's already there.
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        except Exception as exc:
            logger.warning(f"Could not ensure pgcrypto extension (likely already fine): {exc}")
    logger.info("✅ Postgres forecasts table ready.")


def save_forecast(
    uid: str,
    email: str,
    months: int,
    metrics: dict,
    predictions: list,
    historical: list,
    sch: dict,
    model_id: str = None,
) -> str:
    """Saves one forecast run. Returns the new row's id (str) — handed
    straight to the frontend as X-Forecast-Id, same as the old Firestore
    doc id was."""
    with _cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO forecasts
                (uid, owner_email, months, model_id, model_name,
                 train_rmse, val_rmse, baseline_rmse, schema, predictions, historical)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                uid,
                email,
                months,
                model_id,
                metrics.get("model_name"),
                metrics.get("train_rmse"),
                metrics.get("val_rmse"),
                metrics.get("baseline_rmse"),
                json.dumps(sch),
                json.dumps(predictions),
                json.dumps(historical),
            ),
        )
        row = cur.fetchone()
        return str(row["id"])


def list_forecasts(uid: str, limit: int = 20) -> list:
    """Lightweight list (no predictions/historical) for the history dropdown."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT id, created_at, months, model_id, model_name, val_rmse
            FROM forecasts
            WHERE uid = %s
            ORDER BY created_at DESC
            LIMIT %s;
            """,
            (uid, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "id": str(r["id"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "months": r["months"],
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "val_rmse": r["val_rmse"],
        }
        for r in rows
    ]


def get_forecast(uid: str, forecast_id: str) -> dict | None:
    """Full detail for one forecast, scoped to its owner (uid) — returns
    None if it doesn't exist OR belongs to someone else, same 404-either-way
    behavior the Firestore version had via the users/{uid}/... subcollection path."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT id, created_at, months, model_id, model_name,
                   train_rmse, val_rmse, baseline_rmse, schema, predictions, historical
            FROM forecasts
            WHERE id = %s AND uid = %s;
            """,
            (forecast_id, uid),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": str(row["id"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "months": row["months"],
        "model_id": row["model_id"],
        "model_name": row["model_name"],
        "train_rmse": row["train_rmse"],
        "val_rmse": row["val_rmse"],
        "baseline_rmse": row["baseline_rmse"],
        "schema": row["schema"],
        "predictions": row["predictions"],
        "historical": row["historical"],
    }


def delete_forecast(uid: str, forecast_id: str) -> bool:
    """Deletes one forecast owned by uid. Returns True if a row was deleted."""
    with _cursor(commit=True) as cur:
        cur.execute(
            "DELETE FROM forecasts WHERE id = %s AND uid = %s RETURNING id;",
            (forecast_id, uid),
        )
        return cur.fetchone() is not None
