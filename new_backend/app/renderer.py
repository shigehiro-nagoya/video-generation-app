"""Strict-Photo / Photo-safe レンダラー(参照実装)

設計方針(監査報告書 D-3/D-4 で指摘した違反を構造的に防ぐための実装):

1. 既定(strict_photo=True, user_crop 未指定)では、写真を「絶対にクロップしない」。
   実際の画像サイズを ffprobe/Pillow で読み取り、target サイズに contain(レターボックス)
   方式でフィットさせる。はみ出す余白は、同じ画像をぼかして拡大した背景で埋める
   (よくある「縦動画に横写真を入れる」際の見せ方で、実際の写真ピクセルは一切失われない)。
   これにより「本人の関与なしにシステムが写真の一部を切り落とす」ことを構造的に防ぐ。

2. ユーザーが明示的にクロップ範囲(user_crop: x/y/w/h、0-1の相対値)を指定した場合のみ、
   その範囲でクロップする。これは「人間が意図して選ぶ加工」であり、監査報告書で合意した
   定義上、違反にはあたらない。

3. ズーム/パン(Ken Burns)は、常に「安全キャンバス(レターボックス適用後の全体)」に対して
   行う。安全キャンバスの外側(=実際には存在しない余白部分)にズームインしていって、
   まるで写真の外側に何かがあるかのように見せることはしない。

4. 複数写真は順番通りに等尺(または指定尺)のクリップとして生成し、素朴な concat で
   結合する(合成・生成的な補間は行わない=「勝手に足す」の余地をなくす)。

5. レンダリング後、実際に非0バイト・非ブラックフレームであることをこのモジュール自身の
   `verify_output()` で確認してから呼び出し側に返す。ここで検証に失敗した場合は
   例外を送出し、呼び出し側(API層)はステータスを COMPLETED にしてはならない
   (監査報告書で指摘した「黒画面/0バイトでもCOMPLETEDを返す」不具合の再発防止)。
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class UserCrop:
    """0-1 の相対座標。ユーザー本人が明示的に選んだ場合のみ渡される。"""
    x: float
    y: float
    w: float
    h: float


@dataclass
class PhotoAsset:
    path: Path
    user_crop: UserCrop | None = None  # None = 安全パディング(既定・クロップ無し)


class StrictPhotoViolation(RuntimeError):
    """写真保全ポリシーに反する入力・状態を検知した場合に送出する。"""


class RenderVerificationFailed(RuntimeError):
    """生成結果が壊れている(黒画面/0バイト等)と判定した場合に送出する。
    呼び出し側はこれを catch して status=FAILED として扱うこと。COMPLETED を返してはならない。
    """


TARGET_W = 1080
TARGET_H = 1920
CLIP_SECONDS = 3.0
FPS = 30


def _probe_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


# 【2026-09-24 追加】これまではどのクリップも「中央に向かってゆっくりズームインする」
# 動きしか無く、ユーザーから「動きがズームだけしかない」と指摘された。パン(左右移動)や
# ズームアウトも選べるようにし、1本の動画の中でクリップごとに動きが変わるようにする。
_CAMERA_MOVES = ("zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left")


def _pick_camera_moves(n: int) -> list[str]:
    """クリップ数ぶんのカメラワークを決める。動きのリストをシャッフルしてから順番に
    (足りなければ繰り返して)割り当てることで、1本の動画内でできるだけ同じ動きが
    連続しないようにする(写真1枚だけの動画でも、実行のたびに4種類からランダムに
    選ばれるので「毎回ズームだけ」にはならない)。"""
    shuffled = list(_CAMERA_MOVES)
    random.shuffle(shuffled)
    return [shuffled[i % len(shuffled)] for i in range(n)]


def _build_single_clip_filter(src_w: int, src_h: int, crop: UserCrop | None, move_type: str) -> str:
    """1枚の写真からKen Burns風のズーム/パン付きのクリップを作るffmpegフィルタ文字列を組み立てる。
    move_typeで"zoom_in"/"zoom_out"/"pan_left_to_right"/"pan_right_to_left"の
    いずれかの動きを選べる(_CAMERA_MOVES参照)。"""

    if crop is not None:
        # 人間が明示的に選んだクロップ範囲のみ適用(構想合意済みの「加工の範囲内」)。
        cw = max(1, round(src_w * crop.w))
        ch = max(1, round(src_h * crop.h))
        cx = max(0, min(src_w - cw, round(src_w * crop.x)))
        cy = max(0, min(src_h - ch, round(src_h * crop.y)))
        pre = f"crop={cw}:{ch}:{cx}:{cy},"
        eff_w, eff_h = cw, ch
    else:
        # 既定: クロップ一切なし。安全キャンバス(レターボックス)に contain で収める。
        pre = ""
        eff_w, eff_h = src_w, src_h

    # 1) ぼかした背景(拡大+crop で画面いっぱいに引き伸ばし、写真本体ではなく背景演出用)
    # 2) 実写真は contain(scale=-1 系ではなく decrease で必ず収まるように)して重ねる
    # 3) 最後にわずかなズーム(Ken Burns)を全体キャンバスにかける
    # 【修正】以前の実装は canvas を先に scale=iw*1.08:ih*1.08 で拡大したうえで
    # zoompan に x/y を指定していなかった。zoompan は x/y 省略時に既定で
    # 左上(0,0)を基準に切り出すため、frame=0(zoom=1.0)の時点で既に右端・下端が
    # 数%分見切れて灰色の背景帯だけになる、という「システムが気づかぬうちに写真の
    # 外周を切り落とす」不具合が実際に発生していた(四隅サンプリングで検証し発見)。
    # これは本プロジェクトが除去対象としている「写真保全違反」そのものであるため、
    # 別途の pre-scale は行わず、zoompan 自身の z 引数でキャンバス全体に対して
    # ズームしつつ、x/y を常に中央基準の式にして切り出し位置を明示的に固定した。
    # 【2026-09-24 修正・パフォーマンス】Renderの無料プラン(CPU 0.15コア相当)で
    # 実際にデプロイして検証したところ、ぼかし背景の生成(1080x1920のフルサイズに
    # 対してgblur sigma=30を直接かける処理)がCPUを長時間占有し、その間サーバーの
    # 応答が止まってプロセスが再起動してしまう(=ジョブが永久にPROCESSINGのまま
    # 消える)という実障害を実際に確認した。そこで、ぼかし背景は「大幅に縮小して
    # からぼかし、最後に拡大し直す」という標準的な高速化手法に変更し、ぼかしの
    # 計算量を約1/16に削減した(見た目は縮小・拡大されるため、ぼかし背景としては
    # 実用上ほぼ同じに見える)。
    bg_w, bg_h = TARGET_W // 4, TARGET_H // 4
    d = int(CLIP_SECONDS * FPS)

    # 【2026-09-24追加】move_typeごとにz(ズーム)/x/y(切り出し位置)の式を切り替える。
    # ズーム系は以前と同じくonly-increasing/decreasingの滑らかな変化、パン系は
    # zoomを1.15固定にして横方向にずらせる余地を確保し、x を時間(on/出力フレーム番号)
    # に応じて左端↔右端まで動かす。yは常に中央固定(縦方向にはズレさせない=
    # 写真の上下が不自然に見切れるのを防ぐ)。
    if move_type == "zoom_out":
        z_expr = "max(1.08-0.0007*on,1.0)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif move_type == "pan_left_to_right":
        z_expr = "1.15"
        x_expr = f"(iw-iw/zoom)*on/{max(d - 1, 1)}"
        y_expr = "ih/2-(ih/zoom/2)"
    elif move_type == "pan_right_to_left":
        z_expr = "1.15"
        x_expr = f"(iw-iw/zoom)*(1-on/{max(d - 1, 1)})"
        y_expr = "ih/2-(ih/zoom/2)"
    else:  # "zoom_in"(既定)
        z_expr = "min(1.0+0.0007*on,1.08)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    filter_complex = (
        f"[0:v]{pre}scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},scale={bg_w}:{bg_h},gblur=sigma=8,"
        f"scale={TARGET_W}:{TARGET_H},eq=brightness=-0.08[bg];"
        f"[0:v]{pre}scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[canvas];"
        f"[canvas]zoompan="
        f"z='{z_expr}':"
        f"x='{x_expr}':y='{y_expr}':"
        f"d={d}:s={TARGET_W}x{TARGET_H}:fps={FPS}"
    )
    return filter_complex


def render_photo_safe_video(
    assets: list[PhotoAsset],
    out_path: Path,
    strict_photo: bool = True,
) -> Path:
    if not assets:
        raise StrictPhotoViolation("assets が空です。最低1枚の写真が必要です。")

    if strict_photo:
        for a in assets:
            if a.user_crop is None:
                continue  # OK: 明示的なユーザー指定のみ許可されるクロップ
    # strict_photo=False は将来の「ユーザーが保全を明示的にオフにした」場合の拡張ポイント。
    # 現時点ではクロップ判定ロジック自体は user_crop の有無だけで一貫しているため、
    # ここでは特別分岐は設けず、常に同じ安全パスを通す(=「勝手にオンオフで挙動を変えて
    # 見えない差異を作らない」)。

    work_dir = out_path.parent / f"_work_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []

    camera_moves = _pick_camera_moves(len(assets))

    try:
        for i, asset in enumerate(assets):
            src_w, src_h = _probe_size(asset.path)
            filter_complex = _build_single_clip_filter(src_w, src_h, asset.user_crop, camera_moves[i])
            clip_path = work_dir / f"clip_{i:03d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", str(asset.path),
                "-filter_complex", filter_complex,
                "-t", str(CLIP_SECONDS),
                # 【2026-09-24 修正・パフォーマンス】Render無料プラン(CPU 0.15コア相当)
                # では既定のpreset(medium)だとエンコードがCPUを長時間占有し、
                # プロセスが再起動してしまう不具合を実際に確認した。
                # preset=ultrafastに変更してエンコードのCPU負荷を大幅に下げた。
                "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
                "-pix_fmt", "yuv420p", "-r", str(FPS),
                str(clip_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for asset {i}: {result.stderr[-2000:]}")
            clip_paths.append(clip_path)

        concat_mp4s(clip_paths, out_path)

        verify_output(out_path, expected_min_clips=len(assets))
        return out_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def concat_mp4s(clip_paths: list[Path], out_path: Path) -> None:
    """複数のmp4クリップを素朴な(再エンコード無しの)concatで1本に結合する。
    1本しかない場合は単純コピー。standardモード(render_photo_safe_video)と
    ai_premiumモード(jobs.py)の両方から使う共通処理として切り出した。
    """
    if not clip_paths:
        raise RuntimeError("concat_mp4s: clip_paths が空です")

    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], out_path)
        return

    work_dir = out_path.parent / f"_concat_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        concat_list = work_dir / "concat.txt"
        concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths))
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-2000:]}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def verify_output(path: Path, expected_min_clips: int = 1) -> None:
    """黒画面/0バイトを検知して、壊れた出力を「成功扱い」にしないための検証。
    監査報告書で確認した「実際のバックエンドがCOMPLETEDを返しても中身が黒画面/0バイト」
    という不具合の再発を、この参照実装では構造的に防ぐ。
    """
    if not path.exists() or path.stat().st_size == 0:
        raise RenderVerificationFailed(f"出力ファイルが存在しないか0バイトです: {path}")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RenderVerificationFailed(f"ffprobeが失敗しました(壊れたファイルの疑い): {probe.stderr}")
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if duration < 0.5:
        raise RenderVerificationFailed(f"再生時間が異常に短いです: {duration}s")

    # 複数点のフレームを実際にrawピクセルとして抽出し、平均輝度が閾値以下(=ほぼ黒)で
    # ないか確認する(ffmpegのログ出力に頼らず、生ピクセル値を直接計算する方式)。
    sample_points = [duration * f for f in (0.05, 0.5, 0.95)]
    for t in sample_points:
        frame_check = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "gray", "-vf", "scale=16:16",
                "-",
            ],
            capture_output=True,
        )
        pixels = frame_check.stdout
        if not pixels:
            raise RenderVerificationFailed(f"t={t:.2f}s のフレームを抽出できませんでした。")
        avg_luma = sum(pixels) / len(pixels)
        if avg_luma < 4.0:
            raise RenderVerificationFailed(
                f"t={t:.2f}s のフレームがほぼ完全に黒です(平均輝度={avg_luma:.2f}/255)。レンダリング失敗と判定します。"
            )
