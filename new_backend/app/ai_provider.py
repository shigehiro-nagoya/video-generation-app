"""ai_premium (高画質) レンダリングの実体: Runway ML / Kling AI 連携。

設計方針(このプロジェクト全体を貫く「写真保全(strict photo)」の考え方を、
生成AIモデルにも適用する):

1. どちらのプロバイダも「画像 → 動画」(image-to-video)のみを使う。テキストのみ
   から動画を生成する機能(text_to_video等)は使わない。これは、入力写真の内容を
   土台にしてカメラワークだけを足す使い方に限定するため。
2. プロンプトは、この参照実装が固定で用意する _SAFE_MOTION_PROMPT のみを使う。
   ユーザーが自由入力したテキストをそのままプロンプトへ流し込むことはしない。
   これは、生成AIモデルに「写真の内容を変える/足す」ような指示を人間が(意図せず
   とも)与えてしまうリスクを避けるため(standard モードの renderer.py が
   「クロップしない・合成しない」を徹底しているのと同じ思想)。
3. どちらのプロバイダも非同期タスク方式(作成→ポーリング→結果取得)。ポーリング
   間隔は各社ドキュメントが明示する目安(5秒に1回程度)を守る。
4. 生成された動画は一度ローカルにダウンロードしてから、既存の
   renderer.verify_output() で(0バイト/黒画面でないか)検証する。「動くふりを
   しない」というこのプロジェクトの一貫した方針を、AI生成でも適用するため。

コスト面の注意(monetization_plan.md 参照): Kling/Runwayともに動画1本ごとに
実費が発生する(概算 Kling ≈$0.04/秒、Runway ≈$0.05〜0.12/秒)。本参照実装では
写真1枚につき5秒のクリップを1本生成する設計にしている(Klingがduration=5|10の
固定値しか受け付けないため、両プロバイダで挙動を揃えている)。複数写真の
ai_premiumスライドショーは、そのぶん生成本数=実費が線形に増える点に注意。
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from .models import AiProvider


class AiProviderError(RuntimeError):
    """Runway/Kling連携のいずれかの失敗(設定不備・API側エラー・タイムアウト等)。
    呼び出し側(jobs.py)はこれを捕捉してjob.statusをFAILEDにすること。
    """


# 生成AIモデルに渡す唯一のプロンプト。ユーザー自由入力は使わない(理由は上記docstring)。
_SAFE_MOTION_PROMPT = (
    "Add subtle, natural camera motion (gentle pan or slow zoom) to this exact photo. "
    "Do not alter, add, or remove any people, objects, text, or background elements. "
    "Keep the original photo's content, people, and composition exactly as shown."
)

# Kling は duration に 5 か 10 の固定値しか受け付けない。Runwayは2〜10の任意の
# 整数秒を受け付けるが、この参照実装では2社の挙動を揃えるため両方とも5秒に統一する。
AI_CLIP_SECONDS = 5

# ポーリング設定。各社ドキュメントが「5秒に1回程度」を推奨しているため合わせる。
_POLL_INTERVAL_SECONDS = 5.0
_POLL_MAX_ATTEMPTS = 72  # 5秒 x 72 = 最大6分/クリップ

_HTTP_TIMEOUT = httpx.Timeout(30.0, read=60.0)

# 生成AIに送る画像はここまで縮小・圧縮する(base64化した際にRunwayの5MB上限に
# 余裕を持って収まるようにするため)。
_MAX_IMAGE_BYTES = 3_000_000


def generate_ai_clip(
    provider: AiProvider,
    image_path: Path,
    out_path: Path,
) -> None:
    """1枚の写真から、指定プロバイダでAI生成した動画クリップを作り、out_pathに保存する。
    失敗時はAiProviderErrorを送出する(呼び出し側はcatchしてjob.status=FAILEDにすること)。
    """
    data_uri = _prepare_image_data_uri(image_path)

    if provider == AiProvider.RUNWAY:
        video_url = _runway_generate(data_uri)
    elif provider == AiProvider.KLING:
        video_url = _kling_generate(data_uri)
    else:  # pragma: no cover - 予期しない値は設定ミスとして扱う
        raise AiProviderError(f"未対応のai_providerです: {provider}")

    _download_to(video_url, out_path)


# ---- 画像の前処理 -----------------------------------------------------------


def _prepare_image_data_uri(path: Path) -> str:
    """画像をRGB/JPEGに正規化し、必要なら縮小・再圧縮してから、
    生成AI各社のAPIがそのまま受け付けられる data:image/jpeg;base64,... 形式にする。
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size

            quality = 90
            scale = 1.0
            encoded: bytes | None = None

            for _ in range(8):
                w = max(1, int(width * scale))
                h = max(1, int(height * scale))
                candidate = im.resize((w, h), Image.LANCZOS) if scale != 1.0 else im
                buf = io.BytesIO()
                candidate.save(buf, format="JPEG", quality=quality)
                data = buf.getvalue()
                if len(data) <= _MAX_IMAGE_BYTES:
                    encoded = data
                    break
                # まだ大きい場合は、縮小と画質低下を交互に強めてもう一度試す。
                if quality > 60:
                    quality -= 10
                else:
                    scale *= 0.8

            if encoded is None:
                # 最後の手段として最小サイズ・最低画質で強制的に収める。
                buf = io.BytesIO()
                im.resize((max(1, int(width * 0.3)), max(1, int(height * 0.3))), Image.LANCZOS).save(
                    buf, format="JPEG", quality=50
                )
                encoded = buf.getvalue()
    except Exception as e:  # noqa: BLE001
        raise AiProviderError(f"画像の読み込み/変換に失敗しました: {e}")

    b64 = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _download_to(url: str, out_path: Path) -> None:
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            raw = resp.content
    except httpx.HTTPError as e:
        raise AiProviderError(f"生成された動画のダウンロードに失敗しました: {e}")
    if not raw:
        raise AiProviderError("生成された動画のダウンロード結果が空でした。")
    out_path.write_bytes(raw)


# ---- Runway ML --------------------------------------------------------------

_RUNWAY_BASE = "https://api.dev.runwayml.com/v1"
_RUNWAY_VERSION = "2024-11-06"
_RUNWAY_MODEL = "gen4_turbo"
_RUNWAY_RATIO = "720:1280"  # 縦動画(9:16)。TARGET_W=1080/TARGET_H=1920と同じ比率。


def _runway_headers() -> dict[str, str]:
    api_key = os.environ.get("RUNWAYML_API_SECRET")
    if not api_key:
        raise AiProviderError("サーバーに RUNWAYML_API_SECRET が設定されていません。")
    return {
        "Authorization": f"Bearer {api_key}",
        "X-Runway-Version": _RUNWAY_VERSION,
        "Content-Type": "application/json",
    }


def _runway_generate(image_data_uri: str) -> str:
    headers = _runway_headers()
    body = {
        "model": _RUNWAY_MODEL,
        "promptImage": image_data_uri,
        "promptText": _SAFE_MOTION_PROMPT,
        "ratio": _RUNWAY_RATIO,
        "duration": AI_CLIP_SECONDS,
    }

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(f"{_RUNWAY_BASE}/image_to_video", headers=headers, json=body)
        if resp.status_code >= 400:
            raise AiProviderError(f"Runwayへのタスク作成に失敗しました (HTTP {resp.status_code}): {resp.text[:500]}")
        task = resp.json()
        task_id = task.get("id")
        if not task_id:
            raise AiProviderError(f"Runwayのレスポンスにタスクidがありませんでした: {task}")

        for _ in range(_POLL_MAX_ATTEMPTS):
            time.sleep(_POLL_INTERVAL_SECONDS)
            poll = client.get(f"{_RUNWAY_BASE}/tasks/{task_id}", headers=headers)
            if poll.status_code >= 400:
                raise AiProviderError(f"Runwayのタスク確認に失敗しました (HTTP {poll.status_code}): {poll.text[:500]}")
            status_body = poll.json()
            status = status_body.get("status")
            if status == "SUCCEEDED":
                outputs = status_body.get("output") or []
                if not outputs:
                    raise AiProviderError(f"Runwayは成功と応答しましたが、出力URLがありませんでした: {status_body}")
                return outputs[0]
            if status == "FAILED":
                failure = status_body.get("failure") or status_body.get("failureCode") or "詳細不明"
                raise AiProviderError(f"Runwayでの動画生成に失敗しました: {failure}")
            # PENDING / THROTTLED / RUNNING はポーリング継続。

        raise AiProviderError("Runwayでの動画生成がタイムアウトしました。")


# ---- Kling AI ----------------------------------------------------------------

_KLING_BASE = "https://api-singapore.klingai.com"
_KLING_MODEL = "kling-2.5-turbo"


def _kling_headers() -> dict[str, str]:
    api_key = os.environ.get("KLING_API_KEY")
    if not api_key:
        raise AiProviderError("サーバーに KLING_API_KEY が設定されていません。")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _kling_generate(image_data_uri: str) -> str:
    headers = _kling_headers()
    body = {
        "contents": [
            {"type": "prompt", "text": _SAFE_MOTION_PROMPT},
            {"type": "first_frame", "url": image_data_uri},
        ],
        "settings": {
            "resolution": "720p",
            "duration": AI_CLIP_SECONDS,
        },
        "options": {
            "watermark_info": {"enabled": False},
        },
    }

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(f"{_KLING_BASE}/image-to-video/{_KLING_MODEL}", headers=headers, json=body)
        if resp.status_code >= 400:
            raise AiProviderError(f"Klingへのタスク作成に失敗しました (HTTP {resp.status_code}): {resp.text[:500]}")
        created = resp.json()
        if created.get("code", 0) != 0:
            raise AiProviderError(f"Klingへのタスク作成に失敗しました: {created.get('message')}")
        task_id = created.get("data", {}).get("id")
        if not task_id:
            raise AiProviderError(f"Klingのレスポンスにタスクidがありませんでした: {created}")

        for _ in range(_POLL_MAX_ATTEMPTS):
            time.sleep(_POLL_INTERVAL_SECONDS)
            poll = client.get(f"{_KLING_BASE}/tasks", headers=headers, params={"task_ids": task_id})
            if poll.status_code >= 400:
                raise AiProviderError(f"Klingのタスク確認に失敗しました (HTTP {poll.status_code}): {poll.text[:500]}")
            poll_body = poll.json()
            if poll_body.get("code", 0) != 0:
                raise AiProviderError(f"Klingのタスク確認に失敗しました: {poll_body.get('message')}")
            data_list = poll_body.get("data") or []
            if not data_list:
                continue
            task_status = data_list[0]
            status = task_status.get("status")
            if status == "succeeded":
                outputs = task_status.get("outputs") or []
                video_outputs = [o for o in outputs if o.get("type") == "video"]
                if not video_outputs:
                    raise AiProviderError(f"Klingは成功と応答しましたが、動画出力がありませんでした: {task_status}")
                return video_outputs[0]["url"]
            if status == "failed":
                raise AiProviderError(f"Klingでの動画生成に失敗しました: {task_status.get('message')}")
            # submitted / processing はポーリング継続。

        raise AiProviderError("Klingでの動画生成がタイムアウトしました。")
