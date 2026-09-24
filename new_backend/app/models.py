"""バックエンドAPI契約 v2 (再設計案)

現行(v0.3.0)からの主な変更点:

1. assets が「順序を持つ複数枚」を前提にした構造になった(現行も型としてはList
   だったが、実際に複数枚渡した場合の挙動が未定義・未検証だった)。
2. 各 asset に、ユーザーが明示的に選んだ場合のみ有効になる crop(UserCropSpec)を
   持たせた。省略時は「安全パディング(クロップ無し)」が既定になる。
3. strict_photo フラグを必須にし、既定 True。false にする場合も、現時点では
   「クロップ無し既定」の実装ロジック自体は変えない(将来、生成的な背景補完等を
   許可する拡張ポイントとして予約するのみ)。
4. quality を実際に選べる形にし、render_mode の意味を明文化(standard=ffmpeg
   ズーム/パンのみ、ai_premium=将来の生成モデル統合用の予約値。現時点で
   ai_premium を指定した場合は明示的に 501 Not Implemented を返す=「動くふりを
   しない」)。
5. ステータスは COMPLETED になる前に renderer.verify_output() を必ず通す
   (黒画面/0バイトのまま COMPLETED を返すことを構造的に禁止)。
6. usage/plan(課金の土台)を status 応答に含め、無料枠超過時は 402 相当の
   エラーを返せるようにした(実際の決済連携は対象外・フックのみ)。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class GenerationType(str, Enum):
    PHOTO_TO_VIDEO = "PHOTO_TO_VIDEO"          # 写真1枚
    PHOTO_SLIDESHOW = "PHOTO_SLIDESHOW"        # 写真複数枚(順序あり)
    VIDEO_TO_VIDEO = "VIDEO_TO_VIDEO"          # 動画1本


class RenderMode(str, Enum):
    STANDARD = "standard"        # ffmpeg ズーム/パンのみ。今すぐ動く。
    AI_PREMIUM = "ai_premium"    # 生成モデル統合用の予約値。未実装なら501を返す。


class AssetKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class UserCropSpec(BaseModel):
    """ユーザー本人がその場で選んだクロップ範囲(0-1の相対座標)。
    このフィールドが無い asset は、既定で「安全パディング(クロップ無し)」になる。
    """
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @field_validator("w")
    @classmethod
    def _w_in_bounds(cls, v, info):
        return v

    def model_post_init(self, __context) -> None:  # pydantic v2
        if self.x + self.w > 1.0001 or self.y + self.h > 1.0001:
            raise ValueError("crop範囲が画像の範囲外です(x+w<=1, y+h<=1である必要があります)")


class AssetInputV2(BaseModel):
    uri: str
    kind: AssetKind
    order: int = Field(ge=0, description="複数枚時の表示・再生順。0始まり。")
    user_crop: Optional[UserCropSpec] = None
    mime_type: Optional[str] = None


class CreateVideoRequestV2(BaseModel):
    user_id: str
    plan: str = "FREE"
    generation_type: GenerationType
    platform: str
    style: str
    orientation: str = "vertical"
    quality: str = "standard"
    render_mode: RenderMode = RenderMode.STANDARD
    strict_photo: bool = True
    assets: list[AssetInputV2]
    narration_enabled: bool = False
    narration_text: Optional[str] = None
    narration_voice: Optional[str] = None

    @field_validator("assets")
    @classmethod
    def _assets_non_empty_and_ordered(cls, v: list[AssetInputV2]):
        if not v:
            raise ValueError("assets は最低1件必要です")
        orders = sorted(a.order for a in v)
        if orders != list(range(len(v))):
            raise ValueError("assets[].order は 0 から始まる連番である必要があります(欠番/重複不可)")
        return v

    @field_validator("generation_type")
    @classmethod
    def _generation_type_matches_assets(cls, v, info):
        # pydantic v2: assets はまだ検証順序上ここで見えない場合があるため、
        # 実際の整合性チェックは api.py 側(全フィールド確定後)でも二重に行う。
        return v


class GoogleAuthRequest(BaseModel):
    """【2026-09-24追加】Googleサインインで得たIDトークンをサーバーに送るためのリクエスト。"""
    id_token: str


class GoogleAuthResponse(BaseModel):
    """【2026-09-24追加】サーバー側で検証・発行した「本当の」ユーザーIDとプランを返す。
    クライアントはこの user_id を以後の動画生成リクエストで使う(匿名IDから移行)。
    """
    user_id: str
    email: Optional[str] = None
    plan: str


class JobStatus(str, Enum):
    PREPARING = "PREPARING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class UsageInfo(BaseModel):
    plan: str
    used_this_month: int
    quota_this_month: int
    remaining: int


class VideoCreateResponseV2(BaseModel):
    project_id: str
    video_id: str
    status: JobStatus
    message: str
    usage: UsageInfo


class StatusResponseV2(BaseModel):
    video_id: str
    status: JobStatus
    message: Optional[str] = None
    error_detail: Optional[str] = None  # FAILEDの場合、黒画面検知等の具体的理由
    next_action: Optional[str] = None


class ResultResponseV2(BaseModel):
    video_id: str
    project_id: str
    status: JobStatus
    generated_video_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    output_size: Optional[str] = None
    strict_photo_applied: bool
    assets_count: int
    save_enabled: bool = False
    share_enabled: bool = False
