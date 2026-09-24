"""HTTPルーティング層(参照実装)

【2026-09-23 修正】当初 FastAPI で書いていたが、このセッションのサンドボックス
環境が PyPI から fastapi パッケージ自体を取得できない(403)という制約があるため、
FastAPI が内部で使っている ASGI フレームワーク Starlette (+ 手動の pydantic
バリデーション)に直接書き換えた。ルーティング/バリデーション/レスポンスの
振る舞いはFastAPIを使った場合と機能的に同一になるようにしている。
これにより、このAPI層を実際にuvicornで起動し、実際のHTTPリクエストで
動作検証することができるようになった(以前の版では「未実行・未検証」だったが、
今回は実際に起動してエンドツーエンドで検証済み)。
"""
from __future__ import annotations

import base64
import binascii
import mimetypes
import uuid
from pathlib import Path

import httpx
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from .jobs import STORAGE_DIR, create_job, get_job
from .models import (
    AssetKind,
    CreateVideoRequestV2,
    JobStatus,
    ResultResponseV2,
    StatusResponseV2,
    UsageInfo,
    VideoCreateResponseV2,
)

INBOUND_DIR = Path("/tmp/videogen_v2_inbound")
INBOUND_DIR.mkdir(parents=True, exist_ok=True)

# 参照実装なので簡易的な月次クオータのみ(実際の決済連携は対象外)。
_PLAN_QUOTA = {"FREE": 5, "LITE": 50, "PREMIUM": 1000}
_USAGE: dict[str, int] = {}


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


async def create_video(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "リクエストボディがJSONとして解析できません"}, status_code=422)

    try:
        req = CreateVideoRequestV2.model_validate(body)
    except ValidationError as e:
        return JSONResponse({"detail": e.errors()}, status_code=422)

    try:
        used = _USAGE.get(req.user_id, 0)
        quota = _PLAN_QUOTA.get(req.plan, _PLAN_QUOTA["FREE"])
        if used >= quota:
            raise ApiError(402, f"今月の生成枠({quota}本)を使い切りました。プランをアップグレードしてください。")

        if req.generation_type.value == "PHOTO_SLIDESHOW" and len(req.assets) < 2:
            raise ApiError(422, "PHOTO_SLIDESHOW には2枚以上のassetsが必要です")
        if req.generation_type.value in ("PHOTO_TO_VIDEO", "VIDEO_TO_VIDEO") and len(req.assets) != 1:
            raise ApiError(422, f"{req.generation_type.value} はasset 1件のみ対応です")
        if req.generation_type.value == "VIDEO_TO_VIDEO" and req.assets[0].kind != AssetKind.VIDEO:
            raise ApiError(422, "VIDEO_TO_VIDEO には kind=video の asset が必要です")
        if req.generation_type.value != "VIDEO_TO_VIDEO" and any(a.kind != AssetKind.IMAGE for a in req.assets):
            raise ApiError(422, "写真系の生成には kind=image の asset のみ使用できます")

        local_paths: list[Path] = []
        for asset in sorted(req.assets, key=lambda a: a.order):
            local_paths.append(_resolve_asset_to_local_path(asset.uri))

        job = create_job(req, local_paths)
        _USAGE[req.user_id] = used + 1

        resp = VideoCreateResponseV2(
            project_id=job.project_id,
            video_id=job.video_id,
            status=job.status,
            message="動画作成を受け付けました",
            usage=UsageInfo(
                plan=req.plan,
                used_this_month=_USAGE[req.user_id],
                quota_this_month=quota,
                remaining=max(0, quota - _USAGE[req.user_id]),
            ),
        )
        return JSONResponse(resp.model_dump(mode="json"), status_code=200)
    except ApiError as e:
        return JSONResponse({"detail": e.detail}, status_code=e.status_code)


async def get_status(request: Request) -> JSONResponse:
    video_id = request.path_params["video_id"]
    job = get_job(video_id)
    if job is None:
        return JSONResponse({"detail": "video_id が見つかりません"}, status_code=404)

    message = {
        JobStatus.PREPARING: "準備中です",
        JobStatus.PROCESSING: "生成処理中です",
        JobStatus.COMPLETED: "動画ができました",
        JobStatus.FAILED: "動画生成に失敗しました",
    }[job.status]

    resp = StatusResponseV2(
        video_id=job.video_id,
        status=job.status,
        message=message,
        error_detail=job.error_detail,
        next_action="結果を見る" if job.status == JobStatus.COMPLETED else None,
    )
    return JSONResponse(resp.model_dump(mode="json"), status_code=200)


async def get_result(request: Request) -> JSONResponse:
    video_id = request.path_params["video_id"]
    job = get_job(video_id)
    if job is None:
        return JSONResponse({"detail": "video_id が見つかりません"}, status_code=404)

    video_url = None
    duration = None
    if job.status == JobStatus.COMPLETED and job.output_path is not None:
        video_url = f"/static/videos/{job.output_path.name}"

    resp = ResultResponseV2(
        video_id=job.video_id,
        project_id=job.project_id,
        status=job.status,
        generated_video_url=video_url,
        duration_seconds=duration,
        output_size="1080x1920",
        strict_photo_applied=job.request.strict_photo,
        assets_count=len(job.request.assets),
        save_enabled=job.status == JobStatus.COMPLETED,
        share_enabled=job.status == JobStatus.COMPLETED,
    )
    return JSONResponse(resp.model_dump(mode="json"), status_code=200)


async def healthcheck(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _resolve_asset_to_local_path(uri: str) -> Path:
    if uri.startswith("data:"):
        header, _, b64data = uri.partition(",")
        try:
            raw = base64.b64decode(b64data)
        except (binascii.Error, ValueError) as e:
            raise ApiError(422, f"asset uri のdecodeに失敗しました: {e}")
        ext = ".jpg" if "jpeg" in header or "jpg" in header else ".png" if "png" in header else ".bin"
        path = INBOUND_DIR / f"{uuid.uuid4().hex}{ext}"
        path.write_bytes(raw)
        return path

    if uri.startswith("http://") or uri.startswith("https://"):
        # 実際にアプリから送られてくるのはCloudinary等のHTTPS URLであるため、
        # 【修正・2026-09-23】このダウンロード処理が無いと、本番アプリからの
        # リクエストは全て422で弾かれてしまう(=参照実装のままでは実運用不可能
        # だったギャップ)。実際にHTTPでダウンロードして検証する。
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(uri)
                resp.raise_for_status()
                raw = resp.content
        except httpx.HTTPError as e:
            raise ApiError(422, f"asset uri のダウンロードに失敗しました: {uri} ({e})")
        if not raw:
            raise ApiError(422, f"asset uri のダウンロード結果が空でした: {uri}")

        content_type = resp.headers.get("content-type", "")
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
        if not ext:
            suffix = Path(uri.split("?")[0]).suffix
            ext = suffix if suffix else ".jpg"
        path = INBOUND_DIR / f"{uuid.uuid4().hex}{ext}"
        path.write_bytes(raw)
        return path

    local_candidate = Path(uri)
    if local_candidate.exists():
        return local_candidate

    raise ApiError(
        422,
        f"asset uri を解決できませんでした: {uri} "
        "(data: URI、http(s):// URL、またはサーバー上のローカルパスのみ対応しています)",
    )


routes = [
    Route("/healthz", healthcheck, methods=["GET"]),
    Route("/api/videos", create_video, methods=["POST"]),
    Route("/api/videos/{video_id}/status", get_status, methods=["GET"]),
    Route("/api/videos/{video_id}", get_result, methods=["GET"]),
    Mount("/static/videos", app=StaticFiles(directory=str(STORAGE_DIR)), name="videos"),
]

app = Starlette(routes=routes)
