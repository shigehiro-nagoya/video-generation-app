import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="Video Generation App API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UTC = timezone.utc
BASE_DIR = Path(__file__).resolve().parent
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
STATIC_DIR = BASE_DIR / "static"
VIDEO_OUTPUT_DIR = STATIC_DIR / "videos"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PlanType(str, Enum):
    FREE = "FREE"
    LITE = "LITE"
    PREMIUM = "PREMIUM"


class GenerationType(str, Enum):
    PHOTO_TO_VIDEO = "PHOTO_TO_VIDEO"
    PHOTO_SLIDESHOW = "PHOTO_SLIDESHOW"
    VIDEO_TO_VIDEO = "VIDEO_TO_VIDEO"


class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    OUTDATED = "OUTDATED"


class PlatformPreset(BaseModel):
    name: str
    aspect_ratio: str
    output_size: str


class OutputPolicy(BaseModel):
    container: str
    video_codec: str
    audio_codec: str
    target_video_bitrate: str
    max_file_size_mb: int
    safe_area: str
    notes: List[str] = []


class TransformPlan(BaseModel):
    source_size: str
    target_size: str
    source_aspect_ratio: str
    target_aspect_ratio: str
    resize_mode: str
    safe_area_inset: str
    ffmpeg_filter: str
    rationale: str


PLATFORM_PRESETS: Dict[str, PlatformPreset] = {
    "TikTok": PlatformPreset(name="TikTok", aspect_ratio="9:16", output_size="1080x1920"),
    "Lemon8": PlatformPreset(name="Lemon8", aspect_ratio="3:4", output_size="1080x1440"),
    "X": PlatformPreset(name="X", aspect_ratio="1:1", output_size="1200x1200"),
    "Facebook投稿": PlatformPreset(name="Facebook投稿", aspect_ratio="4:5", output_size="1080x1350"),
    "Facebookリール": PlatformPreset(name="Facebookリール", aspect_ratio="9:16", output_size="1080x1920"),
    "YouTube Shorts": PlatformPreset(name="YouTube Shorts", aspect_ratio="9:16", output_size="1080x1920"),
    "YouTube動画": PlatformPreset(name="YouTube動画", aspect_ratio="16:9", output_size="1920x1080"),
    "その他SNS": PlatformPreset(name="その他SNS", aspect_ratio="1:1 / 4:5 / 9:16", output_size="1080x1080"),
}

OUTPUT_POLICIES: Dict[str, OutputPolicy] = {
    "TikTok": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="8M",
        max_file_size_mb=72,
        safe_area="中央寄せ / 上下のUI領域を避ける",
        notes=["冒頭1秒で主題が見える構成", "字幕は中央安全域に収める"],
    ),
    "Lemon8": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="8M",
        max_file_size_mb=80,
        safe_area="上下に余白を残して説明文を逃がす",
        notes=["3:4 前提でクロップ優先", "説明テキストを下端に寄せすぎない"],
    ),
    "X": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="6M",
        max_file_size_mb=40,
        safe_area="中央正方形を優先",
        notes=["1:1 を基本に短尺重視", "細い文字を避けて視認性を確保"],
    ),
    "Facebook投稿": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="8M",
        max_file_size_mb=100,
        safe_area="上下の説明・操作UIを考慮",
        notes=["4:5 の表示占有率を優先", "サムネイルでも見える主題配置"],
    ),
    "Facebookリール": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="8M",
        max_file_size_mb=120,
        safe_area="縦動画の中央帯を優先",
        notes=["リール向けに人物や主題を中央寄せ", "上下のUIかぶりを避ける"],
    ),
    "YouTube Shorts": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="10M",
        max_file_size_mb=150,
        safe_area="縦動画の中央帯を優先",
        notes=["短い導入で離脱を防ぐ", "高コントラストの文字を推奨"],
    ),
    "YouTube動画": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="12M",
        max_file_size_mb=300,
        safe_area="16:9 の左右端まで使える",
        notes=["横長レイアウトを優先", "サムネイル映えする一枚目を意識"],
    ),
    "その他SNS": OutputPolicy(
        container="mp4",
        video_codec="H.264",
        audio_codec="AAC-LC",
        target_video_bitrate="8M",
        max_file_size_mb=80,
        safe_area="中央安全域を広めに確保",
        notes=["未知のSNS向けに汎用設定", "1:1 ベースで破綻しにくさを優先"],
    ),
}


class AssetInput(BaseModel):
    uri: str
    kind: str = Field(pattern="^(image|video)$")


class CreateVideoRequest(BaseModel):
    user_id: str
    plan: PlanType
    generation_type: GenerationType
    platform: str
    style: str
    orientation: str
    length: str
    quality: str
    assets: List[AssetInput]


class VideoCreateResponse(BaseModel):
    project_id: str
    video_id: str
    status: JobStatus
    message: str


class VideoRecord(BaseModel):
    project_id: str
    video_id: str
    user_id: str
    plan: PlanType
    generation_type: GenerationType
    platform: str
    style: str
    orientation: str
    length: str
    quality: str
    assets: List[AssetInput]
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error_code: Optional[str] = None
    generated_video_path: Optional[str] = None


class StatusResponse(BaseModel):
    video_id: str
    status: JobStatus
    message: str
    next_action: str


class ResultResponse(BaseModel):
    video_id: str
    project_id: str
    status: JobStatus
    title: str
    platform: PlatformPreset
    save_enabled: bool
    share_enabled: bool
    playable: bool
    created_at: datetime
    assets_count: int
    generated_video_url: Optional[str] = None
    output_policy: OutputPolicy
    transform_plan: TransformPlan


class RecentVideosResponse(BaseModel):
    items: List[ResultResponse]


class PlanInfo(BaseModel):
    name: PlanType
    description: str
    features: List[str]


class LogItem(BaseModel):
    event: str
    user_id: str
    project_id: Optional[str] = None
    video_id: Optional[str] = None
    status: Optional[JobStatus] = None
    error_code: Optional[str] = None
    timestamp: datetime


class LogSummaryResponse(BaseModel):
    user_id: str
    created_count: int
    completed_count: int
    failed_count: int
    save_ready_count: int
    share_ready_count: int


class ClientEventRequest(BaseModel):
    event: str
    user_id: str
    project_id: Optional[str] = None
    video_id: Optional[str] = None
    status: Optional[JobStatus] = None
    error_code: Optional[str] = None


VIDEOS: Dict[str, VideoRecord] = {}
USER_LAST_SETTINGS: Dict[str, dict] = {}
USER_PREFERENCES: Dict[str, dict] = {}
EVENT_LOGS: List[LogItem] = []

ALLOWED_LENGTHS = {
    PlanType.FREE: {"短め", "ふつう"},
    PlanType.LITE: {"短め", "ふつう", "長め"},
    PlanType.PREMIUM: {"短め", "ふつう", "長め", "しっかり長め"},
}

ALLOWED_QUALITIES = {
    PlanType.FREE: {"標準"},
    PlanType.LITE: {"標準", "高め"},
    PlanType.PREMIUM: {"標準", "高め", "高画質 / HD"},
}

LENGTH_PRIORITY = ["短め", "ふつう", "長め", "しっかり長め"]
QUALITY_PRIORITY = ["標準", "高め", "高画質 / HD"]


def now_utc() -> datetime:
    return datetime.now(UTC)



def add_log(
    event: str,
    user_id: str,
    project_id: Optional[str] = None,
    video_id: Optional[str] = None,
    status: Optional[JobStatus] = None,
    error_code: Optional[str] = None,
) -> None:
    EVENT_LOGS.append(
        LogItem(
            event=event,
            user_id=user_id,
            project_id=project_id,
            video_id=video_id,
            status=status,
            error_code=error_code,
            timestamp=now_utc(),
        )
    )



def status_message(status: JobStatus) -> str:
    return {
        JobStatus.PREPARING: "作る準備をしています",
        JobStatus.PROCESSING: "動画を作っています",
        JobStatus.COMPLETED: "動画ができました",
        JobStatus.FAILED: "動画を作れませんでした",
        JobStatus.OUTDATED: "この結果は古くなっています",
    }[status]



def next_upgrade_plan(plan: PlanType) -> Optional[PlanType]:
    if plan == PlanType.FREE:
        return PlanType.LITE
    if plan == PlanType.LITE:
        return PlanType.PREMIUM
    return None



def build_plan_restriction_detail(
    plan: PlanType,
    field: str,
    reason: str,
    suggested_value: str,
) -> dict:
    source_values = ALLOWED_LENGTHS[plan] if field == "length" else ALLOWED_QUALITIES[plan]
    priority = LENGTH_PRIORITY if field == "length" else QUALITY_PRIORITY
    allowed_values = [value for value in priority if value in source_values]
    current_plan_label = {
        PlanType.FREE: "無料版",
        PlanType.LITE: "Lite",
        PlanType.PREMIUM: "Premium",
    }[plan]
    upgrade_plan = next_upgrade_plan(plan)
    if upgrade_plan == PlanType.LITE:
        upgrade_message = "Lite にすると、前回設定の利用と長め動画・高め画質が使えます"
    elif upgrade_plan == PlanType.PREMIUM:
        upgrade_message = "Premium にすると、高画質 / HD とおすすめ機能が使えます"
    else:
        upgrade_message = "今のプランで利用できる設定をご利用ください"
    return {
        "type": "PLAN_RESTRICTION",
        "field": field,
        "reason": reason,
        "current_plan": plan.value,
        "current_plan_label": current_plan_label,
        "allowed_now": f"{current_plan_label}では「{' / '.join(allowed_values)}」まで使えます",
        "suggested_value": suggested_value,
        "upgrade_plan": upgrade_plan.value if upgrade_plan else None,
        "upgrade_message": upgrade_message,
    }



def validate_plan_rules(payload: CreateVideoRequest) -> None:
    if payload.length not in ALLOWED_LENGTHS[payload.plan]:
        raise HTTPException(
            status_code=403,
            detail=build_plan_restriction_detail(
                plan=payload.plan,
                field="length",
                reason="この長さは今のプランでは使えません",
                suggested_value=[value for value in LENGTH_PRIORITY if value in ALLOWED_LENGTHS[payload.plan]][-1],
            ),
        )
    if payload.quality not in ALLOWED_QUALITIES[payload.plan]:
        raise HTTPException(
            status_code=403,
            detail=build_plan_restriction_detail(
                plan=payload.plan,
                field="quality",
                reason="この画質は今のプランでは使えません",
                suggested_value=[value for value in QUALITY_PRIORITY if value in ALLOWED_QUALITIES[payload.plan]][-1],
            ),
        )



def parse_output_size(platform_name: str) -> tuple[int, int]:
    output_size = PLATFORM_PRESETS[platform_name].output_size
    try:
        width_str, height_str = output_size.split("x")
        return int(width_str), int(height_str)
    except ValueError:
        return 1080, 1080


def get_output_policy(platform_name: str) -> OutputPolicy:
    return OUTPUT_POLICIES.get(
        platform_name,
        OutputPolicy(
            container="mp4",
            video_codec="H.264",
            audio_codec="AAC-LC",
            target_video_bitrate="8M",
            max_file_size_mb=80,
            safe_area="中央安全域を優先",
            notes=["汎用出力設定"],
        ),
    )



def aspect_ratio_value(ratio_label: str) -> float:
    left, right = ratio_label.split(":")
    return float(left) / float(right)



def format_ratio(width: int, height: int) -> str:
    ratio_map = {
        (1080, 1920): "9:16",
        (1920, 1080): "16:9",
        (1200, 1200): "1:1",
        (1080, 1080): "1:1",
        (1080, 1350): "4:5",
        (1080, 1440): "3:4",
        (1440, 1080): "4:3",
    }
    return ratio_map.get((width, height), f"{width}:{height}")



def infer_source_size(video: VideoRecord) -> tuple[int, int]:
    if video.orientation == "たて":
        return 1080, 1920
    if video.orientation == "よこ":
        return 1920, 1080
    if video.orientation == "正方形":
        return 1080, 1080
    if video.generation_type == GenerationType.VIDEO_TO_VIDEO:
        return 1920, 1080
    if video.platform == "Lemon8":
        return 1440, 1080
    return 1440, 1080



def build_transform_plan(video: VideoRecord) -> TransformPlan:
    source_width, source_height = infer_source_size(video)
    target_width, target_height = parse_output_size(video.platform)
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    safe_area_inset = "上下 12% / 左右 8%"

    if abs(source_ratio - target_ratio) < 0.01:
        resize_mode = "passthrough"
        filter_chain = f"scale={target_width}:{target_height},setsar=1"
        rationale = "元の比率を保ったまま投稿先サイズへ合わせます"
    elif video.generation_type == GenerationType.VIDEO_TO_VIDEO:
        resize_mode = "letterbox"
        fit_scale = min(target_width / source_width, target_height / source_height)
        scaled_width = max(2, int(round(source_width * fit_scale / 2) * 2))
        scaled_height = max(2, int(round(source_height * fit_scale / 2) * 2))
        pad_x = max(0, (target_width - scaled_width) // 2)
        pad_y = max(0, (target_height - scaled_height) // 2)
        filter_chain = (
            f"scale={scaled_width}:{scaled_height},"
            f"pad={target_width}:{target_height}:{pad_x}:{pad_y}:black,setsar=1"
        )
        rationale = "元映像を切り落とさず、必要な余白だけ追加して投稿先サイズへ合わせます"
    else:
        resize_mode = "crop"
        fill_scale = max(target_width / source_width, target_height / source_height)
        scaled_width = max(2, int(round(source_width * fill_scale / 2) * 2))
        scaled_height = max(2, int(round(source_height * fill_scale / 2) * 2))
        crop_x = max(0, (scaled_width - target_width) // 2)
        crop_y = max(0, (scaled_height - target_height) // 2)
        filter_chain = (
            f"scale={scaled_width}:{scaled_height},"
            f"crop={target_width}:{target_height}:{crop_x}:{crop_y},setsar=1"
        )
        rationale = "主題を中央に残しながら、投稿先サイズに合わせてクロップします"

    return TransformPlan(
        source_size=f"{source_width}x{source_height}",
        target_size=f"{target_width}x{target_height}",
        source_aspect_ratio=format_ratio(source_width, source_height),
        target_aspect_ratio=PLATFORM_PRESETS[video.platform].aspect_ratio.split(" /")[0],
        resize_mode=resize_mode,
        safe_area_inset=safe_area_inset,
        ffmpeg_filter=filter_chain,
        rationale=rationale,
    )



def ensure_generated_video(video: VideoRecord) -> None:
    output_file = VIDEO_OUTPUT_DIR / f"{video.video_id}.mp4"
    if output_file.exists():
        video.generated_video_path = f"/static/videos/{output_file.name}"
        return

    source_width, source_height = infer_source_size(video)
    policy = get_output_policy(video.platform)
    transform = build_transform_plan(video)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={source_width}x{source_height}:rate=30",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf",
        transform.ffmpeg_filter,
        "-shortest",
        "-t",
        "4",
        "-c:v",
        "libx264",
        "-b:v",
        policy.target_video_bitrate,
        "-maxrate",
        policy.target_video_bitrate,
        "-bufsize",
        policy.target_video_bitrate,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_file),
    ]

    try:
        run(ffmpeg_cmd, check=True, capture_output=True)
        video.generated_video_path = f"/static/videos/{output_file.name}"
    except CalledProcessError as exc:
        video.status = JobStatus.FAILED
        video.error_code = "VIDEO_GENERATION_FAILED"
        add_log(
            event="video_generation_failed",
            user_id=video.user_id,
            project_id=video.project_id,
            video_id=video.video_id,
            status=video.status,
            error_code=video.error_code,
        )
        raise HTTPException(status_code=500, detail="動画ファイルの作成に失敗しました") from exc



def sync_status(video: VideoRecord) -> VideoRecord:
    elapsed = now_utc() - video.created_at
    if video.status == JobStatus.FAILED:
        return video
    if video.status == JobStatus.OUTDATED:
        return video

    previous = video.status
    if elapsed >= timedelta(seconds=5):
        video.status = JobStatus.COMPLETED
        if video.generated_video_path is None:
            ensure_generated_video(video)
    elif elapsed >= timedelta(seconds=2):
        video.status = JobStatus.PROCESSING
    else:
        video.status = JobStatus.PREPARING

    if previous != video.status:
        add_log(
            event="status_changed",
            user_id=video.user_id,
            project_id=video.project_id,
            video_id=video.video_id,
            status=video.status,
        )
    video.updated_at = now_utc()
    return video



def build_absolute_url(request: Request, path: str) -> str:
    base_url = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return f"{base_url}{path}"



def to_result(video: VideoRecord, request: Request) -> ResultResponse:
    preset = PLATFORM_PRESETS.get(video.platform)
    if not preset:
        raise HTTPException(status_code=400, detail="Unknown platform")
    generated_video_url = build_absolute_url(request, video.generated_video_path) if video.generated_video_path else None
    playable = video.status == JobStatus.COMPLETED and generated_video_url is not None
    return ResultResponse(
        video_id=video.video_id,
        project_id=video.project_id,
        status=video.status,
        title=f"{video.platform}向け動画",
        platform=preset,
        save_enabled=playable,
        share_enabled=playable,
        playable=playable,
        created_at=video.created_at,
        assets_count=len(video.assets),
        generated_video_url=generated_video_url,
        output_policy=get_output_policy(video.platform),
        transform_plan=build_transform_plan(video),
    )


@app.get("/")
def root() -> dict:
    return {"message": "Video Generation App backend is running"}


@app.get("/api/health")
def api_health() -> dict:
    return {"status": "ok", "service": "video-generation-app-api"}


@app.get("/api/platforms")
def get_platforms() -> List[PlatformPreset]:
    return list(PLATFORM_PRESETS.values())


@app.post("/api/videos", response_model=VideoCreateResponse)
def create_video(payload: CreateVideoRequest) -> VideoCreateResponse:
    if not payload.assets:
        raise HTTPException(status_code=400, detail="素材が必要です")
    if payload.platform not in PLATFORM_PRESETS:
        raise HTTPException(status_code=400, detail="投稿先が正しくありません")
    validate_plan_rules(payload)

    project_id = str(uuid4())
    video_id = str(uuid4())
    created = now_utc()

    VIDEOS[video_id] = VideoRecord(
        project_id=project_id,
        video_id=video_id,
        user_id=payload.user_id,
        plan=payload.plan,
        generation_type=payload.generation_type,
        platform=payload.platform,
        style=payload.style,
        orientation=payload.orientation,
        length=payload.length,
        quality=payload.quality,
        assets=payload.assets,
        status=JobStatus.PREPARING,
        created_at=created,
        updated_at=created,
    )

    USER_LAST_SETTINGS[payload.user_id] = {
        "platform": payload.platform,
        "style": payload.style,
        "orientation": payload.orientation,
        "length": payload.length,
        "quality": payload.quality,
    }

    USER_PREFERENCES[payload.user_id] = {
        "favorite_platform": payload.platform,
        "favorite_style": payload.style,
        "favorite_length": payload.length,
    }

    add_log(
        event="create_video",
        user_id=payload.user_id,
        project_id=project_id,
        video_id=video_id,
        status=JobStatus.PREPARING,
    )

    return VideoCreateResponse(
        project_id=project_id,
        video_id=video_id,
        status=JobStatus.PREPARING,
        message="動画作成を受け付けました",
    )


@app.get("/api/videos/{video_id}/status", response_model=StatusResponse)
def get_video_status(video_id: str) -> StatusResponse:
    video = VIDEOS.get(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    video = sync_status(video)
    next_action = "結果を見る" if video.status == JobStatus.COMPLETED else "そのまま待つ"
    if video.status == JobStatus.FAILED:
        next_action = "もう一度作る"
    add_log(
        event="check_status",
        user_id=video.user_id,
        project_id=video.project_id,
        video_id=video.video_id,
        status=video.status,
        error_code=video.error_code,
    )
    return StatusResponse(
        video_id=video_id,
        status=video.status,
        message=status_message(video.status),
        next_action=next_action,
    )


@app.get("/api/videos/{video_id}", response_model=ResultResponse)
def get_video_result(video_id: str, request: Request) -> ResultResponse:
    video = VIDEOS.get(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    video = sync_status(video)
    add_log(
        event="get_result",
        user_id=video.user_id,
        project_id=video.project_id,
        video_id=video.video_id,
        status=video.status,
        error_code=video.error_code,
    )
    return to_result(video, request)


@app.get("/api/videos", response_model=RecentVideosResponse)
def get_recent_videos(user_id: str, request: Request) -> RecentVideosResponse:
    user_videos = [sync_status(v) for v in VIDEOS.values() if v.user_id == user_id]
    user_videos.sort(key=lambda x: x.created_at, reverse=True)
    add_log(event="get_recent_videos", user_id=user_id)
    return RecentVideosResponse(items=[to_result(v, request) for v in user_videos])


@app.get("/api/plans", response_model=List[PlanInfo])
def get_plans() -> List[PlanInfo]:
    return [
        PlanInfo(name=PlanType.FREE, description="まずは無料で試せます", features=["基本の動画作成", "保存", "共有"]),
        PlanInfo(name=PlanType.LITE, description="前回の設定ですぐ作れます", features=["長め動画", "高め画質", "前回設定の利用"]),
        PlanInfo(name=PlanType.PREMIUM, description="あなた向けに使いやすくなります", features=["高画質 / HD", "おすすめ機能", "個人向け提案"]),
    ]


@app.get("/api/users/{user_id}/last-settings")
def get_last_settings(user_id: str) -> dict:
    add_log(event="get_last_settings", user_id=user_id)
    return USER_LAST_SETTINGS.get(user_id, {})


@app.get("/api/users/{user_id}/recommendation")
def get_recommendation(user_id: str) -> dict:
    prefs = USER_PREFERENCES.get(user_id)
    add_log(event="get_recommendation", user_id=user_id)
    if not prefs:
        return {
            "title": "はじめて向けおすすめ",
            "platform": "TikTok",
            "style": "かんたん",
            "length": "短め",
            "reason": "まずは作りやすい設定です",
        }
    return {
        "title": "あなた向けおすすめ",
        "platform": prefs["favorite_platform"],
        "style": prefs["favorite_style"],
        "length": prefs["favorite_length"],
        "reason": "よく使う設定に近づけています",
    }


@app.post("/api/logs/client-event")
def create_client_event(payload: ClientEventRequest) -> dict:
    add_log(
        event=payload.event,
        user_id=payload.user_id,
        project_id=payload.project_id,
        video_id=payload.video_id,
        status=payload.status,
        error_code=payload.error_code,
    )
    return {"ok": True}


@app.get("/api/logs/summary", response_model=LogSummaryResponse)
def get_log_summary(user_id: str) -> LogSummaryResponse:
    user_videos = [sync_status(v) for v in VIDEOS.values() if v.user_id == user_id]
    completed_results = []
    for video in user_videos:
        if video.status == JobStatus.COMPLETED and video.generated_video_path:
            completed_results.append(video)
    return LogSummaryResponse(
        user_id=user_id,
        created_count=len(user_videos),
        completed_count=sum(1 for v in user_videos if v.status == JobStatus.COMPLETED),
        failed_count=sum(1 for v in user_videos if v.status == JobStatus.FAILED),
        save_ready_count=len(completed_results),
        share_ready_count=len(completed_results),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
