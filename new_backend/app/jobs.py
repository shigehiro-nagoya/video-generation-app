"""ジョブ状態管理(参照実装・インメモリ)

重要な設計ポイント: レンダリングが成功して renderer.verify_output() を通るまでは
絶対に COMPLETED にしない。例外(RenderVerificationFailed / StrictPhotoViolation /
その他)が起きたら必ず FAILED にし、error_detail に具体的な理由を残す。
「よく分からないが動いたことにする」を許さない状態遷移にする。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import CreateVideoRequestV2, JobStatus
from .renderer import (
    PhotoAsset,
    RenderVerificationFailed,
    StrictPhotoViolation,
    UserCrop,
    render_photo_safe_video,
)

STORAGE_DIR = Path("/tmp/videogen_v2_storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    video_id: str
    project_id: str
    request: CreateVideoRequestV2
    status: JobStatus = JobStatus.PREPARING
    error_detail: Optional[str] = None
    output_path: Optional[Path] = None
    duration_seconds: Optional[float] = None
    created_at: float = field(default_factory=time.time)


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def create_job(request: CreateVideoRequestV2, local_asset_paths: list[Path]) -> Job:
    video_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    job = Job(video_id=video_id, project_id=project_id, request=request)
    with _LOCK:
        _JOBS[video_id] = job

    thread = threading.Thread(target=_run_render, args=(job, local_asset_paths), daemon=True)
    thread.start()
    return job


def get_job(video_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(video_id)


def _run_render(job: Job, local_asset_paths: list[Path]) -> None:
    job.status = JobStatus.PROCESSING

    if job.request.render_mode.value == "ai_premium":
        # 「動くふりをしない」: 未実装の高画質生成モードは、黒画面を返すのではなく
        # 明示的に FAILED にする。
        job.status = JobStatus.FAILED
        job.error_detail = (
            "render_mode=ai_premium は現時点で未実装です(参照実装の対象外)。"
            "standard を指定してください。"
        )
        return

    try:
        photo_assets = []
        by_order = sorted(job.request.assets, key=lambda a: a.order)
        for asset_spec, local_path in zip(by_order, local_asset_paths):
            user_crop = None
            if asset_spec.user_crop is not None:
                c = asset_spec.user_crop
                user_crop = UserCrop(x=c.x, y=c.y, w=c.w, h=c.h)
            photo_assets.append(PhotoAsset(path=local_path, user_crop=user_crop))

        out_path = STORAGE_DIR / f"{job.video_id}.mp4"
        render_photo_safe_video(photo_assets, out_path, strict_photo=job.request.strict_photo)

        job.output_path = out_path
        job.status = JobStatus.COMPLETED
    except (RenderVerificationFailed, StrictPhotoViolation) as e:
        job.status = JobStatus.FAILED
        job.error_detail = str(e)
    except Exception as e:  # noqa: BLE001 — 参照実装なので広く捕捉して必ずFAILEDにする
        job.status = JobStatus.FAILED
        job.error_detail = f"予期しないエラー: {e}"
