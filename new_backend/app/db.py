"""永続データベース(Render Postgres)への接続とスキーマ管理。

【設計方針・2026-09-24追加】
- これまでの参照実装は _USAGE / _PLAN_QUOTA をメモリ上の辞書で持つだけで、
  サーバー再起動(Renderの無料プランでは頻繁に起こりうる)のたびに消えていた。
  さらに致命的な問題として、クライアントが送ってくる plan フィールドを
  そのまま信用しており、誰でも自己申告だけで有料プラン扱いを得られる状態
  だった。この2点を解消するため、ユーザー・プラン・購入・利用実績を
  Postgresに永続化する。
- ORMは導入せず psycopg2 を直接使う(既存コードの「参照実装として薄く保つ」
  方針に合わせる)。
- DATABASE_URL 環境変数が無い場合、このモジュールは import 時にはエラーに
  せず、AVAILABLE = False を返すだけにする。呼び出し側(api.py)はこの
  フラグを見て、DB未設定時は安全側(従来のメモリ管理・実質FREE相当)に
  倒す。これは「動くふりをしない」方針とは別軸の話で、DBの設定漏れで
  サービス全体を落とさないための可用性優先の判断。ただし挙動は必ず
  ログに明示し、黙って握りつぶすことはしない。
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("videogen.db")

DATABASE_URL = os.environ.get("DATABASE_URL")
AVAILABLE = bool(DATABASE_URL)

if not AVAILABLE:
    logger.warning(
        "DATABASE_URL が設定されていません。ユーザー/プラン/利用実績の永続化は "
        "無効化され、レガシーのメモリ管理(サーバー再起動で消える・plan自己申告を "
        "信用する挙動)にフォールバックします。"
    )

_psycopg2 = None
if AVAILABLE:
    import psycopg2  # noqa: E402
    import psycopg2.extras  # noqa: E402

    _psycopg2 = psycopg2

_conn_lock = threading.Lock()
_conn = None


def _get_conn():
    global _conn
    with _conn_lock:
        if _conn is None or _conn.closed:
            _conn = _psycopg2.connect(DATABASE_URL, sslmode="require")
            _conn.autocommit = True
        return _conn


@contextmanager
def _cursor():
    conn = _get_conn()
    with conn.cursor(cursor_factory=_psycopg2.extras.RealDictCursor) as cur:
        yield cur


def init_schema() -> None:
    """起動時に一度呼び出す。テーブルが無ければ作成する(IF NOT EXISTS)。"""
    if not AVAILABLE:
        return
    with _cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                google_sub TEXT UNIQUE NOT NULL,
                email TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_plans (
                user_id TEXT PRIMARY KEY REFERENCES users(user_id),
                plan TEXT NOT NULL DEFAULT 'FREE',
                expires_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS purchases (
                purchase_token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                product_id TEXT NOT NULL,
                plan TEXT NOT NULL,
                verified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                raw_response JSONB
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_counters (
                user_id TEXT NOT NULL REFERENCES users(user_id),
                year_month TEXT NOT NULL,
                mode TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, year_month, mode)
            );
            """
        )
    logger.info("DBスキーマ初期化を確認しました(必要なテーブルは揃っています)。")


def get_or_create_user(google_sub: str, email: Optional[str]) -> str:
    user_id = f"g:{google_sub}"
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (user_id, google_sub, email)
            VALUES (%s, %s, %s)
            ON CONFLICT (google_sub) DO UPDATE SET email = EXCLUDED.email
            RETURNING user_id;
            """,
            (user_id, google_sub, email),
        )
        row = cur.fetchone()
        cur.execute(
            """
            INSERT INTO user_plans (user_id, plan)
            VALUES (%s, 'FREE')
            ON CONFLICT (user_id) DO NOTHING;
            """,
            (row["user_id"],),
        )
        return row["user_id"]


def get_plan(user_id: str) -> dict:
    """サーバー側DBに記録された「本当の」プランを返す。期限切れならFREE扱い。"""
    with _cursor() as cur:
        cur.execute(
            "SELECT plan, expires_at FROM user_plans WHERE user_id = %s;",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {"plan": "FREE", "expires_at": None}
        if row["expires_at"] is not None and row["expires_at"] < datetime.now(timezone.utc):
            return {"plan": "FREE", "expires_at": row["expires_at"]}
        return {"plan": row["plan"], "expires_at": row["expires_at"]}


def user_exists(user_id: str) -> bool:
    with _cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE user_id = %s;", (user_id,))
        return cur.fetchone() is not None


def get_usage(user_id: str, mode: str, year_month: str) -> int:
    with _cursor() as cur:
        cur.execute(
            "SELECT count FROM usage_counters WHERE user_id=%s AND year_month=%s AND mode=%s;",
            (user_id, year_month, mode),
        )
        row = cur.fetchone()
        return row["count"] if row else 0


def increment_usage(user_id: str, mode: str, year_month: str) -> int:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO usage_counters (user_id, year_month, mode, count)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (user_id, year_month, mode)
            DO UPDATE SET count = usage_counters.count + 1
            RETURNING count;
            """,
            (user_id, year_month, mode),
        )
        return cur.fetchone()["count"]


def record_purchase(
    purchase_token: str,
    user_id: str,
    product_id: str,
    plan: str,
    expires_at,
    raw_response: dict,
) -> None:
    """Google Play Billingの購入検証が通った際に呼び出す(タスク#7で使用予定)。
    まだ購入検証エンドポイント自体は実装していないため、現時点では未使用。
    """
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO purchases (purchase_token, user_id, product_id, plan, raw_response)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (purchase_token) DO NOTHING;
            """,
            (purchase_token, user_id, product_id, plan, _psycopg2.extras.Json(raw_response)),
        )
        cur.execute(
            """
            INSERT INTO user_plans (user_id, plan, expires_at, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                plan = EXCLUDED.plan,
                expires_at = EXCLUDED.expires_at,
                updated_at = now();
            """,
            (user_id, plan, expires_at),
        )
