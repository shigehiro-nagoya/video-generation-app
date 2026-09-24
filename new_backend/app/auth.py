"""Googleサインインで受け取ったIDトークンをサーバー側で検証するモジュール。

【設計方針・2026-09-24追加】
- google-auth ライブラリを新規依存に追加せず、Googleの公開tokeninfo
  エンドポイント(https://oauth2.googleapis.com/tokeninfo)にそのままHTTPで
  問い合わせて検証する。既存コードが外部連携にhttpxをそのまま使うスタイルに
  合わせ、依存を増やさないため。
- 参照実装としては十分だが、本番で高頻度にアクセスする場合は
  google-authライブラリの公開鍵キャッシュ方式(毎回Googleに問い合わせない)
  のほうが速く低コストなので、将来の置き換え候補として明記しておく。
- GOOGLE_OAUTH_CLIENT_ID が未設定の場合は「動くふりをしない」方針に従い、
  明示的にエラーを返す(誰でもサインインできたことにする、を避ける)。
"""
from __future__ import annotations

import os

import httpx

GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")


class GoogleAuthError(Exception):
    pass


async def verify_google_id_token(id_token: str) -> dict:
    if not GOOGLE_OAUTH_CLIENT_ID:
        raise GoogleAuthError(
            "サーバーにGOOGLE_OAUTH_CLIENT_IDが設定されていないため、Googleサインインを"
            "検証できません。管理者に連絡してください。"
        )
    if not id_token:
        raise GoogleAuthError("id_tokenが空です")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": id_token},
            )
        except httpx.HTTPError as e:
            raise GoogleAuthError(f"Googleへの問い合わせに失敗しました: {e}")

    if resp.status_code != 200:
        raise GoogleAuthError("IDトークンが無効です(Google側の検証でエラーが返されました)")

    payload = resp.json()

    if payload.get("aud") != GOOGLE_OAUTH_CLIENT_ID:
        raise GoogleAuthError(
            "IDトークンのaudienceがこのアプリのOAuthクライアントIDと一致しません"
            "(別アプリ向けのトークンの可能性があります)"
        )
    if payload.get("email_verified") not in ("true", True):
        raise GoogleAuthError("メールアドレスが未確認のGoogleアカウントです")
    sub = payload.get("sub")
    if not sub:
        raise GoogleAuthError("IDトークンにsub(ユーザー識別子)が含まれていません")

    return {"sub": sub, "email": payload.get("email")}
