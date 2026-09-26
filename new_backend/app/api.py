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
import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from . import auth, db
from .jobs import STORAGE_DIR, create_job, get_job
from .models import (
    AssetKind,
    CreateVideoRequestV2,
    GoogleAuthRequest,
    GoogleAuthResponse,
    JobStatus,
    ResultResponseV2,
    StatusResponseV2,
    UsageInfo,
    VideoCreateResponseV2,
)

logger = logging.getLogger("videogen.api")

INBOUND_DIR = Path("/tmp/videogen_v2_inbound")
INBOUND_DIR.mkdir(parents=True, exist_ok=True)

# 【2026-09-24修正】以前はこの _PLAN_QUOTA / _USAGE (メモリ上の辞書)だけで
# quotaを判定しており、(1) サーバー再起動で消える (2) クライアントが送ってくる
# plan フィールドをそのまま信用する、という2つの問題があった。
# db.AVAILABLE (DATABASE_URL設定済み)かつユーザーがGoogleサインイン済みの
# 場合は、DBに記録された「本当の」プランを見て判定する。
# 未サインインの旧クライアント(匿名ID)は、このメモリ管理に安全側でフォールバック
# する(既存ユーザーを即座に壊さないための移行措置。db.user_exists()がFalseの
# 場合がこれに該当)。
#
# standardモード(ffmpegのズーム/パンのみ、1本あたりの実費はほぼゼロ)と
# ai_premiumモード(Runway/Kling連携。1本ごとに実費が発生)とで、上限を
# 完全に分ける。ai_premiumの具体的な本数は monetization_plan.md の原価試算に
# 基づく暫定値で、価格確定後に見直すこと。
_STANDARD_PLAN_QUOTA = {"FREE": 5, "LITE": 50, "PREMIUM": 1000}
_AI_PREMIUM_PLAN_QUOTA = {"FREE": 0, "LITE": 5, "PREMIUM": 20}

# レガシー(未サインイン・DB未設定時)フォールバック用のメモリ管理。
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
        mode = req.render_mode.value  # "standard" | "ai_premium"
        quota_table = _STANDARD_PLAN_QUOTA if mode == "standard" else _AI_PREMIUM_PLAN_QUOTA
        use_db = db.AVAILABLE and db.user_exists(req.user_id)

        if use_db:
            # サーバー側DBに記録された「本当の」プランのみを信用する。
            # req.plan(クライアント自己申告)はここでは一切使わない。
            real_plan = db.get_plan(req.user_id)["plan"]
            year_month = _current_year_month()
            used = db.get_usage(req.user_id, mode, year_month)
        else:
            # 未サインイン(匿名ID)またはDB未設定時のレガシーフォールバック。
            # これまで通りクライアント申告のplanを使うため、ai_premiumはFREE扱いで
            # 常に0本(=常に拒否)になる=未サインインでは有料AI機能を使わせない。
            real_plan = req.plan if mode == "standard" else "FREE"
            used = _USAGE.get(req.user_id, 0) if mode == "standard" else 0

        quota = quota_table.get(real_plan, quota_table["FREE"])
        if used >= quota:
            if mode == "ai_premium" and quota == 0:
                raise ApiError(
                    402,
                    "AI動画モードは現在のプランでは利用できません。プランをアップグレードしてください。",
                )
            raise ApiError(402, f"今月の生成枠({quota}本)を使い切りました。プランをアップグレードしてください。")

        if mode == "ai_premium" and len(req.assets) > 5:
            # 【2026-09-26追加】ai_premiumは写真1枚ごとにRunway/Klingの実費が発生するため、
            # 1回のリクエストで課金が意図せず大きくなりすぎないよう上限を設ける。
            raise ApiError(422, "AI動画モード(ai_premium)は1回につき最大5枚までです")

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

        if use_db:
            new_used = db.increment_usage(req.user_id, mode, year_month)
        else:
            new_used = used + 1
            if mode == "standard":
                _USAGE[req.user_id] = new_used

        resp = VideoCreateResponseV2(
            project_id=job.project_id,
            video_id=job.video_id,
            status=job.status,
            message="動画作成を受け付けました",
            usage=UsageInfo(
                plan=real_plan,
                used_this_month=new_used,
                quota_this_month=quota,
                remaining=max(0, quota - new_used),
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


def _current_year_month() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


async def google_auth(request: Request) -> JSONResponse:
    """【2026-09-24追加】GoogleサインインのIDトークンを検証し、サーバー側の
    「本当の」ユーザーID(g:<google_sub>)を発行する。クライアントは以後、
    匿名IDの代わりにこのuser_idを動画生成リクエストで使う。
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "リクエストボディがJSONとして解析できません"}, status_code=422)

    try:
        req = GoogleAuthRequest.model_validate(body)
    except ValidationError as e:
        return JSONResponse({"detail": e.errors()}, status_code=422)

    try:
        claims = await auth.verify_google_id_token(req.id_token)
    except auth.GoogleAuthError as e:
        return JSONResponse({"detail": str(e)}, status_code=401)

    if not db.AVAILABLE:
        return JSONResponse(
            {"detail": "サーバー側のデータベースが未設定のため、現在サインインを完了できません。"},
            status_code=503,
        )

    user_id = db.get_or_create_user(claims["sub"], claims.get("email"))
    plan = db.get_plan(user_id)["plan"]

    resp = GoogleAuthResponse(user_id=user_id, email=claims.get("email"), plan=plan)
    return JSONResponse(resp.model_dump(mode="json"), status_code=200)


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
    Route("/api/auth/google", google_auth, methods=["POST"]),
    Route("/api/videos", create_video, methods=["POST"]),
    Route("/api/videos/{video_id}/status", get_status, methods=["GET"]),
    Route("/api/videos/{video_id}", get_result, methods=["GET"]),
    Mount("/static/videos", app=StaticFiles(directory=str(STORAGE_DIR)), name="videos"),
]


@asynccontextmanager
async def _lifespan(app: Starlette):
    # 【2026-09-24追加】DATABASE_URLが設定されていればテーブルを用意する。
    # 未設定でもここで例外を出して全体を落とすことはしない(db.py側の設計方針
    # に合わせ、可用性を優先しレガシーモードで起動を続ける)。
    # 【修正】Starlette 1.0でon_startup/on_shutdown引数が廃止されたため、
    # 後方互換性の高いlifespanコンテキストマネージャ方式に変更した
    # (このセッションの検証環境ではstarlette==1.0.0が実際にインストールされて
    # おり、on_startup指定だとTypeErrorで起動不能になることを実際に確認済み)。
    if db.AVAILABLE:
        db.init_schema()
        logger.info("DB接続を確認し、スキーマを初期化しました。")
    else:
        logger.warning(
            "DATABASE_URL未設定のため、ユーザー永続化なしのレガシーモードで起動します。"
        )
    yield


app = Starlette(routes=routes, lifespan=_lifespan)
