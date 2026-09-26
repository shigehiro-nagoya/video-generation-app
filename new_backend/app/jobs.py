"""ジョブ状態管理(参照実装・インメモリ)

重要な設計ポイント: レンダリングが成功して renderer.verify_output() を通るまでは
絶対に COMPLETED にしない。例外(RenderVerificationFailed / StrictPhotoViolation /
その他)が起きたら必ず FAILED にし、error_detail に具体的な理由を残す。
「よく分からないが動いたことにする」を許さない状態遷移にする。
"""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import ai_provider
from .models import CreateVideoRequestV2, JobStatus
from .renderer import (
    PhotoAsset,
    RenderVerificationFailed,
    StrictPhotoViolation,
    UserCrop,
    concat_mp4s,
    render_photo_safe_video,
    verify_output,
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
        _run_ai_premium_render(job, local_asset_paths)
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


def _run_ai_premium_render(job: Job, local_asset_paths: list[Path]) -> None:
    """render_mode=ai_premium の実処理(Task #6)。Runway/Klingで写真1枚ごとに
    5秒のAI生成クリップを作り、複数枚あればconcatで結合する。standardモードと
    同様、verify_output()を必ず通してからCOMPLETEDにする(「動くふりをしない」)。
    実際の生成AI呼び出しはai_provider.pyに切り出してあり、失敗理由は
    ai_provider.AiProviderError のメッセージとしてerror_detailにそのまま残す。
    """
    work_dir = STORAGE_DIR / f"_ai_work_{job.video_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    try:
        by_order = sorted(job.request.assets, key=lambda a: a.order)
        for i, (asset_spec, local_path) in enumerate(zip(by_order, local_asset_paths)):
            clip_path = work_dir / f"ai_clip_{i:03d}.mp4"
            ai_provider.generate_ai_clip(job.request.ai_provider, local_path, clip_path)
            clip_paths.append(clip_path)

        out_path = STORAGE_DIR / f"{job.video_id}.mp4"
        concat_mp4s(clip_paths, out_path)
        verify_output(out_path, expected_min_clips=len(clip_paths))

        job.output_path = out_path
        job.status = JobStatus.COMPLETED
    except (ai_provider.AiProviderError, RenderVerificationFailed) as e:
        job.status = JobStatus.FAILED
        job.error_detail = str(e)
    except Exception as e:  # noqa: BLE001
        job.status = JobStatus.FAILED
        job.error_detail = f"予期しないエラー(ai_premium): {e}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
