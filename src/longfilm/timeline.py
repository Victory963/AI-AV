"""分轨工业化与交付时间线 —— 把分镜与渲染产物变成能进 NLE 的时间线。

为什么要有这一层：AI 出的片子最后一定要过人工精修，而剪辑师不会去读我们的
Storyboard JSON。只有导出成 EDL / OTIO / ASS / concat 清单，成片才能进
DaVinci、Premiere 做套底和精剪，字幕才能烧录，无缝拼接才能落地。

五条轨的分工（对齐 schema.AudioPlan）：
    画面轨 video    —— 每镜一个 clip，转场造成的时间重叠在这里如实记录
    对白轨 dialogue —— 音频先行的实测结果，每句一个 clip
    Foley 轨 foley  —— shot.sfx 与环境声
    音乐轨 music    —— 全片铺底
    字幕轨 subtitle —— 只有文本，没有媒体，导出 .ass 用

外部事实查证日期 2026-09-19：
    OTIO JSON 结构由 opentimelineio 0.18.1 实际序列化产物比对而来（见 _selftest），
    当前版本的 Clip 是 **Clip.2**（media_references 字典 + active_media_reference_key），
    早期文档里的 Clip.1/media_reference 已经不是现行写法。
    CMX3600 EDL 的双行溶解写法（先 C 零长事件，再 D nnn 事件）是行业通用格式。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ._fonts import family_name, require_cjk_font
from .schema import Storyboard, Transition

if TYPE_CHECKING:  # 只为类型标注，避免 timeline 硬依赖 audio_first 才能 import
    from .audio_first import DialogueTrack

log = logging.getLogger(__name__)


# ============================================================ 数据结构


class TrackKind(str, Enum):
    VIDEO = "video"
    DIALOGUE = "dialogue"
    FOLEY = "foley"
    MUSIC = "music"
    SUBTITLE = "subtitle"

    @property
    def is_audio(self) -> bool:
        return self in {TrackKind.DIALOGUE, TrackKind.FOLEY, TrackKind.MUSIC}


# 会造成时间重叠的转场。CUT / MATCH_CUT 是硬切，不占时间。
# 注意 Transition.FADE_IN 与 FADE_OUT 在 schema 里同值（"fadeblack"），是同一个枚举成员。
_BLENDING: frozenset[Transition] = frozenset({
    Transition.DISSOLVE, Transition.FADE_IN, Transition.WIPE_L,
    Transition.WIPE_R, Transition.SLIDE_L, Transition.OVERLAP_BLEND,
})


@dataclass
class Clip:
    """时间线上的一段。时间一律用秒，导出时才量化到帧。"""

    id: str
    kind: TrackKind
    start_s: float
    duration_s: float
    source_uri: str | None = None
    source_in_s: float = 0.0
    name: str = ""
    shot_id: str = ""
    scene_id: str = ""
    speaker_id: str = ""
    text: str = ""                     # 字幕轨专用
    transition_in: Transition = Transition.CUT
    transition_s: float = 0.0          # >0 表示与前一段重叠这么久
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def overlaps(self, other: Clip) -> float:
        return max(0.0, min(self.end_s, other.end_s) - max(self.start_s, other.start_s))


@dataclass
class Track:
    name: str
    kind: TrackKind
    clips: list[Clip] = field(default_factory=list)

    def add(self, clip: Clip) -> Clip:
        self.clips.append(clip)
        return clip

    def sorted_clips(self) -> list[Clip]:
        return sorted(self.clips, key=lambda c: (c.start_s, c.end_s))

    def duration_s(self) -> float:
        return max((c.end_s for c in self.clips), default=0.0)

    def gaps(self, *, until_s: float | None = None, min_s: float = 0.04) -> list[tuple[float, float]]:
        """返回轨道上的空洞 [(start, end)]。默认阈值 ~1 帧，忽略浮点毛刺。"""
        out: list[tuple[float, float]] = []
        cursor = 0.0
        for c in self.sorted_clips():
            if c.start_s - cursor > min_s:
                out.append((cursor, c.start_s))
            cursor = max(cursor, c.end_s)
        end = until_s if until_s is not None else cursor
        if end - cursor > min_s:
            out.append((cursor, end))
        return out

    def overlaps(self) -> list[tuple[Clip, Clip, float]]:
        out: list[tuple[Clip, Clip, float]] = []
        cs = self.sorted_clips()
        for a, b in zip(cs, cs[1:]):
            ov = a.overlaps(b)
            if ov > 1e-6:
                out.append((a, b, ov))
        return out


@dataclass
class Timeline:
    name: str
    fps: int = 24
    resolution: tuple[int, int] = (1920, 1080)
    sample_rate: int = 48000
    tracks: list[Track] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def track(self, kind: TrackKind) -> Track:
        for t in self.tracks:
            if t.kind is kind:
                return t
        t = Track(name=kind.value.upper(), kind=kind)
        self.tracks.append(t)
        return t

    def clips_of(self, kind: TrackKind) -> list[Clip]:
        return self.track(kind).sorted_clips()

    def duration_s(self) -> float:
        return max((t.duration_s() for t in self.tracks), default=0.0)

    def duration_frames(self) -> int:
        return round(self.duration_s() * self.fps)


# ============================================================ 构建


def build_timeline(
    storyboard: Storyboard,
    renders: Mapping[str, str] | None = None,
    *,
    dialogue: DialogueTrack | None = None,
    default_transition_s: float = 0.5,
    fps: int | None = None,
) -> Timeline:
    """从分镜与渲染产物构建时间线。

    转场重叠的处理：溶解/划像这类转场会**吃掉**两镜各一半的时间 ——
    B 镜的入点要提前 ``ov`` 秒落在 A 镜的尾巴上，成片总长因此短于各镜时长之和。
    这里如实记录重叠（B.start < A.end），让 sync_report 和各导出器都拿到真相；
    如果在这一步就把重叠抹平，剪辑师在 NLE 里看到的切点就会和我们烧出来的片子对不上。

    ``renders`` 给 shot_id -> 视频 URI；缺省时回退到 ``Shot.render_uri``。
    ``dialogue`` 给音频先行的产出，有它才能把每句对白挂上真实音频文件。
    """
    # 延迟导入：timeline 只需要 stitch 里那条纯计算的裁剪规则，
    # 不想为了导出 EDL 就把整个 ffmpeg 封装层拉进来（也避免将来反向依赖成环）。
    from .stitch import clamp_transition_s

    fps = fps or storyboard.delivery.fps
    tl = Timeline(
        name=f"{storyboard.project}_{storyboard.episode}",
        fps=fps,
        resolution=storyboard.delivery.upscale_to or storyboard.delivery.resolution,
        metadata={"logline": storyboard.logline, "style_bible": storyboard.style_bible},
    )
    v = tl.track(TrackKind.VIDEO)
    dlg = tl.track(TrackKind.DIALOGUE)
    foley = tl.track(TrackKind.FOLEY)
    music = tl.track(TrackKind.MUSIC)
    sub = tl.track(TrackKind.SUBTITLE)
    renders = renders or {}

    cursor = 0.0
    prev: Clip | None = None
    for scene in storyboard.scenes:
        for shot in scene.shots:
            ov = 0.0
            if prev is not None and shot.transition_in in _BLENDING:
                # overlap_frames 是按**生成片段自己的帧率**数出来的，不是交付帧率；
                # 未声明时（=0）退回默认转场长度 —— 这两条都必须和
                # stitch.resolve_clips 一字不差，否则 EDL 和成片对不上。
                if (shot.transition_in is Transition.OVERLAP_BLEND
                        and shot.continuity.overlap_frames > 0):
                    want = shot.continuity.overlap_frames / max(shot.fps, 1)
                else:
                    want = default_transition_s
                # 裁剪规则由 stitch 持有：烧片的那一份才是真相，这里只能照抄
                ov = clamp_transition_s(want, prev.duration_s, shot.duration_s)

            start = max(0.0, cursor - ov)
            uri = renders.get(shot.id) or shot.render_uri
            clip = v.add(Clip(
                id=f"v_{shot.id}", kind=TrackKind.VIDEO, start_s=start,
                duration_s=shot.duration_s, source_uri=uri, name=shot.id,
                shot_id=shot.id, scene_id=scene.id,
                transition_in=shot.transition_in, transition_s=ov,
                metadata={
                    "shot_size": shot.shot_size.value,
                    "camera_move": shot.camera_move.value,
                    "provider": shot.provider_used,
                    "qc_score": shot.qc_score,
                },
            ))

            # 对白：line.start_s 是镜内相对秒（见 audio_first.DialogueTrack 的基准约定）
            cues = list(dialogue.of_shot(shot.id)) if dialogue is not None else []
            for i, line in enumerate(shot.dialogue):
                if line.start_s is None or line.end_s is None:
                    log.warning("%s 第 %d 句未回填时间，先跑 audio_first.synthesize_dialogue", shot.id, i)
                    continue
                a_start = start + line.start_s
                a_dur = line.end_s - line.start_s
                cue = next((c for c in cues if c.line_index == i), None)
                dlg.add(Clip(
                    id=f"d_{shot.id}_{i:02d}", kind=TrackKind.DIALOGUE,
                    start_s=a_start, duration_s=a_dur,
                    source_uri=cue.audio_uri if cue else None,
                    name=f"{line.speaker_id}: {line.text[:12]}",
                    shot_id=shot.id, scene_id=scene.id, speaker_id=line.speaker_id,
                    text=line.text,
                ))
                sub.add(Clip(
                    id=f"s_{shot.id}_{i:02d}", kind=TrackKind.SUBTITLE,
                    start_s=a_start, duration_s=a_dur, name=line.speaker_id,
                    shot_id=shot.id, scene_id=scene.id, speaker_id=line.speaker_id,
                    text=line.text,
                ))

            for j, name in enumerate(shot.sfx):
                foley.add(Clip(
                    id=f"f_{shot.id}_{j:02d}", kind=TrackKind.FOLEY,
                    start_s=start, duration_s=shot.duration_s,
                    name=name, shot_id=shot.id, scene_id=scene.id,
                    metadata={"cue": name, "todo": "待 Foley 库匹配"},
                ))

            cursor = start + shot.duration_s
            prev = clip

    total = cursor
    if storyboard.audio.music_track:
        music.add(Clip(id="m_bed", kind=TrackKind.MUSIC, start_s=0.0, duration_s=total,
                       source_uri=storyboard.audio.music_track, name="music bed"))
    if storyboard.audio.ambience_track:
        # 环境声按 Foley 处理：它和 sfx 一样是"空间感"层，和音乐的响度规则不同
        foley.add(Clip(id="f_ambience", kind=TrackKind.FOLEY, start_s=0.0, duration_s=total,
                       source_uri=storyboard.audio.ambience_track, name="ambience"))

    log.info(
        "时间线构建完成：%s，%d 镜，%.2fs（各镜之和 %.2fs，转场吃掉 %.2fs）",
        tl.name, len(v.clips), total, storyboard.duration_s(), storyboard.duration_s() - total,
    )
    return tl


# ============================================================ 时间码


def seconds_to_frames(s: float, fps: int) -> int:
    return int(round(s * fps))


# 丢帧时间码只在 NTSC 的 29.97 / 59.94 上有定义（名义 30/60）。
# 24/25fps 上打开它算出来的是一串没人认识的数字，而且不会报错 ——
# 送去电视台就是整条节目时间码对不上，所以宁可在这里炸。
_DROP_FRAME_RATES: frozenset[int] = frozenset({30, 60})


def _check_drop_frame(fps: int) -> None:
    if round(fps) not in _DROP_FRAME_RATES:
        raise ValueError(
            f"丢帧时间码只适用于 29.97/59.94（名义 {sorted(_DROP_FRAME_RATES)}），"
            f"收到 fps={fps}；{fps}fps 请用 drop_frame=False"
        )


def frames_to_timecode(frames: int, fps: int, *, drop_frame: bool = False) -> str:
    """帧号 -> SMPTE 时间码。

    丢帧只对 29.97/59.94 有意义（每分钟丢 2/4 个**编号**，不丢画面），
    24fps 交付用不上，但导出给北美电视台时缺了它时间码会一路漂。
    """
    frames = max(0, frames)
    if drop_frame:
        _check_drop_frame(fps)
        drop = round(fps * 0.066666)  # 29.97 -> 2, 59.94 -> 4
        nominal = round(fps)
        fpm = nominal * 60 - drop
        fp10m = nominal * 600 - drop * 9
        d, m = divmod(frames, fp10m)
        if m >= drop:
            frames += drop * 9 * d + drop * ((m - drop) // fpm)
        else:
            frames += drop * 9 * d
        fps_i = nominal
        sep = ";"
    else:
        fps_i = int(round(fps))
        sep = ":"
    f = frames % fps_i
    total_s = frames // fps_i
    return (f"{total_s // 3600:02d}:{(total_s // 60) % 60:02d}:"
            f"{total_s % 60:02d}{sep}{f:02d}")


def seconds_to_timecode(s: float, fps: int, *, drop_frame: bool = False, offset_s: float = 0.0) -> str:
    return frames_to_timecode(seconds_to_frames(s + offset_s, fps), fps, drop_frame=drop_frame)


def _timecode_to_frames(tc: str, fps: int, *, drop_frame: bool = False) -> int:
    """时间码 -> 帧号。frames_to_timecode 的精确逆运算。

    丢帧必须显式还原：丢的是**编号**不是画面，所以 00:01:00;02 的真实帧号是
    1800 而不是 1802。不做这步换算，drop_frame 的 EDL 起始时间码就会整体偏移。
    """
    parts = re.split(r"[:;]", tc.strip())
    if len(parts) != 4:
        raise ValueError(f"时间码格式应为 HH:MM:SS:FF，收到 {tc!r}")
    h, m, s, f = (int(p) for p in parts)
    nominal = int(round(fps))
    frames = ((h * 60 + m) * 60 + s) * nominal + f
    if drop_frame:
        _check_drop_frame(fps)
        drop = round(fps * 0.066666)
        total_min = h * 60 + m
        frames -= drop * (total_min - total_min // 10)   # 每分钟丢 drop 个，逢 10 分钟不丢
    return frames


# ============================================================ EDL (CMX3600)


# EDL 的音频通道码：CMX3600 原本只认 A/A2/AA，A3 以上属于扩展，
# Resolve/Premiere 能读，老式线性机房设备不一定。
_EDL_AUDIO_CHANNEL = {TrackKind.DIALOGUE: "A", TrackKind.FOLEY: "A2", TrackKind.MUSIC: "A3"}


def _reel(clip: Clip, fallback: str = "AX") -> str:
    """卷名 8 字符，大写字母数字。AX = auto-conform，靠 FROM CLIP NAME 关联素材。"""
    raw = re.sub(r"[^A-Za-z0-9]", "", clip.shot_id or clip.id).upper()
    return (raw[:8] or fallback).ljust(8)


def _edl_event(
    n: int, clip: Clip, *, chan: str, rec_in_f: int, rec_out_f: int,
    fps: int, drop_frame: bool, trans: str = "C", trans_dur_f: int = 0,
    src_in_override_f: int | None = None,
) -> list[str]:
    src_in = (src_in_override_f if src_in_override_f is not None
              else seconds_to_frames(clip.source_in_s, fps))
    src_out = src_in + (rec_out_f - rec_in_f)
    dur = f"{trans_dur_f:03d}" if trans.startswith("D") else ""
    return [
        f"{n:03d}  {_reel(clip)} {chan:<4} {trans:<4} {dur:<3} "
        f"{frames_to_timecode(src_in, fps, drop_frame=drop_frame)} "
        f"{frames_to_timecode(src_out, fps, drop_frame=drop_frame)} "
        f"{frames_to_timecode(rec_in_f, fps, drop_frame=drop_frame)} "
        f"{frames_to_timecode(rec_out_f, fps, drop_frame=drop_frame)}"
    ]


def export_edl(
    timeline: Timeline,
    *,
    title: str | None = None,
    record_start_tc: str = "01:00:00:00",
    drop_frame: bool = False,
    include_audio: bool = True,
) -> str:
    """导出 CMX3600 EDL。

    溶解用行业通用的双行事件写法：同一事件号先出一条零长的 ``C`` 行交代出画素材，
    再出一条 ``D nnn`` 行把新素材溶进来，nnn 是转场帧数。单行写法很多机器读不出转场。

    EDL 表达不了字幕，字幕走 export_ass_subtitles；
    也表达不了多于 3 条的音轨，超出的轨会被跳过并在注释里说明。
    """
    fps = timeline.fps
    off = _timecode_to_frames(record_start_tc, fps, drop_frame=drop_frame)
    lines = [
        f"TITLE: {title or timeline.name}",
        "FCM: DROP FRAME" if drop_frame else "FCM: NON-DROP FRAME",
    ]
    n = 0

    vclips = timeline.clips_of(TrackKind.VIDEO)
    for i, c in enumerate(vclips):
        prev = vclips[i - 1] if i else None
        nxt = vclips[i + 1] if i + 1 < len(vclips) else None
        rec_in = seconds_to_frames(c.start_s, fps) + off
        rec_out = seconds_to_frames(c.end_s, fps) + off
        # 后一镜要溶进来时，本镜的事件必须在溶解起点让位 ——
        # 记录时间在 EDL 里不允许重叠，重叠区归下一个 D 事件管。
        if nxt is not None and nxt.transition_s > 1e-6:
            # 转场长过本镜时让位点会跑到本镜入点之前，EDL 里 rec_out < rec_in 是非法事件。
            # clamp_transition_s 已经挡住这种情况，这里只做最后一道兜底并留痕。
            rec_out = max(rec_in, seconds_to_frames(nxt.start_s, fps) + off)
            if rec_out < seconds_to_frames(nxt.start_s, fps) + off:
                log.warning("%s 的转场吃穿了本镜，EDL 事件被压成零长", c.id)
        ov_f = seconds_to_frames(c.transition_s, fps) if prev is not None else 0

        if ov_f > 0 and prev is not None:
            n += 1
            # 零长的 C 行交代出画素材，紧跟的 D 行把新素材溶进来 —— 单行写法很多机器读不出转场
            prev_out_f = (seconds_to_frames(prev.source_in_s, fps)
                          + seconds_to_frames(c.start_s, fps) - seconds_to_frames(prev.start_s, fps))
            lines += _edl_event(n, prev, chan="V", rec_in_f=rec_in, rec_out_f=rec_in,
                                fps=fps, drop_frame=drop_frame, trans="C",
                                src_in_override_f=prev_out_f)
            lines += _edl_event(n, c, chan="V", rec_in_f=rec_in, rec_out_f=rec_out,
                                fps=fps, drop_frame=drop_frame, trans="D", trans_dur_f=ov_f)
        else:
            n += 1
            lines += _edl_event(n, c, chan="V", rec_in_f=rec_in, rec_out_f=rec_out,
                                fps=fps, drop_frame=drop_frame, trans="C")
        lines.append(f"* FROM CLIP NAME: {c.name or c.id}")
        if c.source_uri:
            lines.append(f"* SOURCE FILE: {c.source_uri}")
        if ov_f > 0:
            lines.append(f"* TRANSITION: {c.transition_in.value} {ov_f} FRAMES")

    if include_audio:
        for kind in (TrackKind.DIALOGUE, TrackKind.FOLEY, TrackKind.MUSIC):
            chan = _EDL_AUDIO_CHANNEL[kind]
            for c in timeline.clips_of(kind):
                n += 1
                lines += _edl_event(
                    n, c, chan=chan,
                    rec_in_f=seconds_to_frames(c.start_s, fps) + off,
                    rec_out_f=seconds_to_frames(c.end_s, fps) + off,
                    fps=fps, drop_frame=drop_frame, trans="C",
                )
                lines.append(f"* FROM CLIP NAME: {c.name or c.id}")
                if c.source_uri:
                    lines.append(f"* SOURCE FILE: {c.source_uri}")

    subs = timeline.clips_of(TrackKind.SUBTITLE)
    if subs:
        lines.append(f"* NOTE: {len(subs)} 条字幕未写入 EDL（格式不支持），见同名 .ass 文件")
    return "\n".join(lines) + "\n"


# ============================================================ OpenTimelineIO


def _rt(frames: int, fps: int) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": float(fps), "value": float(frames)}


def _tr(start_f: int, dur_f: int, fps: int) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "duration": _rt(dur_f, fps),
        "start_time": _rt(start_f, fps),
    }


def _otio_gap(dur_f: int, fps: int) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "Gap.1", "metadata": {}, "name": "",
        "source_range": _tr(0, dur_f, fps),
        "effects": [], "markers": [], "enabled": True, "color": None,
    }


def _otio_clip(clip: Clip, *, start_f: int, dur_f: int, fps: int) -> dict[str, Any]:
    if clip.source_uri:
        ref: dict[str, Any] = {
            "OTIO_SCHEMA": "ExternalReference.1", "metadata": {}, "name": "",
            "available_range": None, "available_image_bounds": None,
            "target_url": _as_url(clip.source_uri),
        }
    else:
        ref = {
            "OTIO_SCHEMA": "MissingReference.1", "metadata": {}, "name": "",
            "available_range": None, "available_image_bounds": None,
        }
    meta = {k: v for k, v in clip.metadata.items() if v is not None}
    meta["longfilm"] = {
        "shot_id": clip.shot_id, "scene_id": clip.scene_id,
        "speaker_id": clip.speaker_id, "text": clip.text,
    }
    return {
        "OTIO_SCHEMA": "Clip.2", "metadata": meta, "name": clip.name or clip.id,
        "source_range": _tr(start_f, dur_f, fps),
        "effects": [], "markers": [], "enabled": True, "color": None,
        "media_references": {"DEFAULT_MEDIA": ref},
        "active_media_reference_key": "DEFAULT_MEDIA",
    }


def _otio_transition(clip: Clip, ov_f: int, fps: int) -> dict[str, Any]:
    half = ov_f / 2.0
    return {
        "OTIO_SCHEMA": "Transition.1", "metadata": {"longfilm": {"kind": clip.transition_in.value}},
        "name": clip.transition_in.value,
        "in_offset": _rt(int(math.floor(half)), fps),
        "out_offset": _rt(int(math.ceil(half)), fps),
        "transition_type": "SMPTE_Dissolve",
    }


def _as_url(uri: str) -> str:
    if "://" in uri:
        return uri
    return Path(uri).resolve().as_uri()


def _lanes(clips: Sequence[Clip]) -> list[list[Clip]]:
    """把可能重叠的 clip 摊到多条不重叠的泳道上。

    OTIO 的 Track 是严格顺序容器，装不下重叠 —— 真实 NLE 遇到同时响的两个音效
    也是多开一条轨，这里照做，而不是把重叠的那条丢掉。
    """
    lanes: list[list[Clip]] = []
    for c in sorted(clips, key=lambda x: (x.start_s, x.end_s)):
        for lane in lanes:
            if lane[-1].end_s <= c.start_s + 1e-6:
                lane.append(c)
                break
        else:
            lanes.append([c])
    return lanes


def _otio_video_children(clips: Sequence[Clip], fps: int) -> list[dict[str, Any]]:
    """画面轨：把重叠折成 Transition，clip 对接成首尾相接。

    OTIO 与 NLE 表达溶解的方式是「两段素材在切点处对接 + 一个跨切点的 Transition」，
    切点取重叠区的中点。直接把重叠的 clip 塞进同一条 Track 会得到非法 OTIO。
    """
    out: list[dict[str, Any]] = []
    if not clips:
        return out
    ovs = [0] + [seconds_to_frames(c.transition_s, fps) for c in clips[1:]]
    bounds: list[int] = []   # 每段的切入点（帧）
    for i, c in enumerate(clips):
        f = seconds_to_frames(c.start_s, fps)
        bounds.append(f + (ovs[i] + 1) // 2 if i else f)

    first = seconds_to_frames(clips[0].start_s, fps)
    if first > 0:
        out.append(_otio_gap(first, fps))

    for i, c in enumerate(clips):
        seg_start = bounds[i]
        seg_end = bounds[i + 1] if i + 1 < len(clips) else seconds_to_frames(c.end_s, fps)
        if seg_end <= seg_start:
            log.warning("转场吃穿了 %s（%d 帧），跳过该段", c.id, seg_end - seg_start)
            continue
        if i and ovs[i] == 0:
            hole = seg_start - seconds_to_frames(clips[i - 1].end_s, fps)
            if hole > 0:
                out.append(_otio_gap(hole, fps))
        if i and ovs[i] > 0:
            out.append(_otio_transition(c, ovs[i], fps))
        src_in = seconds_to_frames(c.source_in_s, fps) + (seg_start - seconds_to_frames(c.start_s, fps))
        out.append(_otio_clip(c, start_f=src_in, dur_f=seg_end - seg_start, fps=fps))
    return out


def _otio_track(name: str, kind: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "Track.1", "metadata": {}, "name": name,
        "source_range": None, "effects": [], "markers": [],
        "enabled": True, "color": None, "children": children, "kind": kind,
    }


def export_otio(timeline: Timeline) -> dict[str, Any]:
    """导出 OpenTimelineIO JSON（dict）。

    手写而不依赖 opentimelineio 包：产线机器不一定装得上它（C++ 扩展），
    但导出时间线是交付的必经环节，不能被一个可选依赖卡住。
    结构按 opentimelineio 0.18.1 的实际序列化产物 1:1 对齐，装了包的机器会在自测里校验。
    """
    fps = timeline.fps
    children: list[dict[str, Any]] = [
        _otio_track("V1", "Video", _otio_video_children(timeline.clips_of(TrackKind.VIDEO), fps))
    ]

    audio_no = 0
    for kind in (TrackKind.DIALOGUE, TrackKind.FOLEY, TrackKind.MUSIC):
        for lane in _lanes(timeline.clips_of(kind)):
            audio_no += 1
            kids: list[dict[str, Any]] = []
            cursor = 0
            for c in lane:
                s, e = seconds_to_frames(c.start_s, fps), seconds_to_frames(c.end_s, fps)
                if s > cursor:
                    kids.append(_otio_gap(s - cursor, fps))
                kids.append(_otio_clip(
                    c, start_f=seconds_to_frames(c.source_in_s, fps), dur_f=max(1, e - s), fps=fps
                ))
                cursor = max(cursor, e)
            children.append(_otio_track(f"A{audio_no} {kind.value}", "Audio", kids))

    for lane_i, lane in enumerate(_lanes(timeline.clips_of(TrackKind.SUBTITLE)), start=1):
        kids = []
        cursor = 0
        for c in lane:
            s, e = seconds_to_frames(c.start_s, fps), seconds_to_frames(c.end_s, fps)
            if s > cursor:
                kids.append(_otio_gap(s - cursor, fps))
            kids.append(_otio_clip(c, start_f=0, dur_f=max(1, e - s), fps=fps))
            cursor = max(cursor, e)
        children.append(_otio_track(f"S{lane_i} subtitle", "Subtitle", kids))

    return {
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {"longfilm": dict(timeline.metadata),
                     "resolution": list(timeline.resolution)},
        "name": timeline.name,
        "global_start_time": _rt(0, fps),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1", "metadata": {}, "name": "tracks",
            "source_range": None, "effects": [], "markers": [],
            "enabled": True, "color": None, "children": children,
        },
    }


# ============================================================ ASS 字幕


# 断句优先级：句末标点最优，其次逗号顿号。断行处这些标点直接吃掉（中文字幕通行做法），
# 但问号感叹号要留着 —— 它们带语气，删了意思就变了。
_BREAK_EAT = "，、；,;"
_BREAK_KEEP = "。！？…!?."
_ASS_ESCAPE = {"{": r"\{", "}": r"\}"}


@dataclass
class SubtitleStyle:
    """.ass 样式。默认值按 1080p 中文剧情片的通行规格。"""

    font_name: str = "Microsoft YaHei"
    font_size: int = 54
    primary: str = "&H00FFFFFF"        # ASS 颜色是 &HAABBGGRR，AA=00 为不透明
    outline_colour: str = "&H00101010"
    back_colour: str = "&H80000000"
    outline: float = 2.4
    shadow: float = 1.2
    bold: int = 0
    alignment: int = 2                 # 2 = 底部居中
    # 安全区：横向留 10%（题名安全区），纵向底部留 8%（介于动作安全 5% 与题名安全 10% 之间）
    safe_x: float = 0.10
    safe_y: float = 0.08
    max_chars_per_line: int = 18       # 中文字幕一行超过 ~18 字，观众扫不完
    max_lines: int = 2

    def margins(self, w: int, h: int) -> tuple[int, int, int]:
        return round(w * self.safe_x), round(w * self.safe_x), round(h * self.safe_y)


def _ass_time(s: float) -> str:
    """ASS 时间码 H:MM:SS.cc（厘秒，小时一位）。"""
    s = max(0.0, s)
    cs = int(round(s * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    sec, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{sec:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    for k, v in _ASS_ESCAPE.items():
        text = text.replace(k, v)
    return text.replace("\r", "").replace("\n", r"\N")


def wrap_cjk(text: str, max_chars: int) -> list[str]:
    """中文断行。中文没有词间空格，不自己断就只能交给 libass 按字硬断在词中间。

    策略：优先在最靠近行尾的标点处断；实在没标点才按字数硬断。
    """
    # max_chars<=0 时硬断点会落在 0，text 一个字都吃不掉 —— 循环永远不结束。
    # 这不是理论情况：SubtitleStyle.max_chars_per_line 是可配字段，配成 0 就会挂死整条产线。
    max_chars = max(1, int(max_chars))
    text = text.strip()
    lines: list[str] = []
    while len(text) > max_chars:
        window = text[:max_chars]
        cut = -1
        for i in range(len(window) - 1, max(0, max_chars // 3) - 1, -1):
            if window[i] in _BREAK_EAT or window[i] in _BREAK_KEEP:
                cut = i + 1
                break
        if cut <= 0:
            cut = max_chars
        head, text = text[:cut], text[cut:].lstrip()
        if head and head[-1] in _BREAK_EAT:
            head = head[:-1]
        lines.append(head)
    if text:
        lines.append(text)
    return lines or [""]


def export_ass_subtitles(
    timeline: Timeline,
    *,
    font: str = "Microsoft YaHei",
    style: SubtitleStyle | None = None,
    show_speaker: bool = False,
    min_event_s: float = 0.6,
) -> str:
    """生成 .ass 字幕，供 ffmpeg 的 subtitles/ass 滤镜烧录（本机 ffmpeg 无 drawtext，只能走 libass）。

    一条对白若超过 ``max_lines`` 行装不下，会按字数比例拆成多条事件依次出现，
    而不是硬塞成四行糊住半个画面。
    """
    # 不能原地改 style：调用方往往拿一份 SubtitleStyle 导好几集，
    # 就地写 font_name 会让第二集悄悄继承第一集的字体。
    st = replace(style or SubtitleStyle(), **({"font_name": font} if font else {}))
    w, h = timeline.resolution
    ml, mr, mv = st.margins(w, h)
    # 一行放得下多少字，同时受版面宽度与可读性上限约束；中文方块字宽 ≈ 字号
    fit = max(6, int((w - ml - mr) / max(st.font_size, 1)))
    per_line = max(1, min(st.max_chars_per_line, fit))

    head = [
        "[Script Info]",
        "; 由 longfilm.timeline.export_ass_subtitles 生成",
        f"Title: {timeline.name}",
        "ScriptType: v4.00+",
        "WrapStyle: 2",            # 2 = 只认 \N，不自动折行（折行已在上面算好）
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{st.font_name},{st.font_size},{st.primary},&H000000FF,"
        f"{st.outline_colour},{st.back_colour},{st.bold},0,0,0,100,100,0,0,1,"
        f"{st.outline},{st.shadow},{st.alignment},{ml},{mr},{mv},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    body: list[str] = []
    for c in timeline.clips_of(TrackKind.SUBTITLE):
        lines = wrap_cjk(c.text, per_line)
        chunks = [lines[i:i + st.max_lines] for i in range(0, len(lines), st.max_lines)] or [[""]]
        total_chars = sum(len("".join(ch)) for ch in chunks) or 1
        t = c.start_s
        for ci, chunk in enumerate(chunks):
            share = len("".join(chunk)) / total_chars
            dur = c.duration_s if len(chunks) == 1 else max(min_event_s, c.duration_s * share)
            end = min(c.end_s, t + dur) if ci < len(chunks) - 1 else c.end_s
            body.append(
                f"Dialogue: 0,{_ass_time(t)},{_ass_time(max(end, t + 0.04))},Default,"
                f"{c.speaker_id if show_speaker else ''},0,0,0,,"
                + _escape_ass(r"\N".join(chunk))
            )
            t = end
    return "\n".join(head + body) + "\n"


def ass_burn_filter(ass_path: str | Path, *, fonts_dir: str | Path | None = None) -> str:
    """拼出 ffmpeg 的字幕烧录滤镜串。路径里的特殊字符要转义，否则 filtergraph 会解析错。"""
    p = str(Path(ass_path).resolve())
    esc = p.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    arg = f"subtitles='{esc}'"
    if fonts_dir:
        d = str(Path(fonts_dir).resolve()).replace("\\", "\\\\").replace(":", r"\:")
        arg += f":fontsdir='{d}'"
    return arg


# ============================================================ concat 清单


def export_ffmpeg_concat(timeline: Timeline, *, kind: TrackKind = TrackKind.VIDEO) -> str:
    """导出 concat demuxer 清单。用法：``ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp4``。

    concat demuxer 只能硬切，做不了溶解 —— 有转场的地方会在注释里点名，
    那些切点需要由 xfade 单独处理，不能默默当成硬切拼出去。
    """
    clips = timeline.clips_of(kind)
    out = ["ffconcat version 1.0",
           f"# {timeline.name} / {kind.value} / {len(clips)} 段 / {timeline.duration_s():.2f}s",
           "# 用法: ffmpeg -f concat -safe 0 -i <本文件> -c copy out.mp4"]
    blended = [c for c in clips if c.transition_s > 1e-6]
    if blended:
        out.append(
            f"# 警告: {len(blended)} 处转场（"
            + "、".join(f"{c.name}:{c.transition_in.value}" for c in blended[:6])
            + "）无法用 concat 表达，将被拼成硬切；需要溶解请改走 xfade"
        )
    missing = [c for c in clips if not c.source_uri]
    if missing:
        out.append(f"# 警告: {len(missing)} 段尚无渲染产物，已跳过：" +
                   "、".join(c.name or c.id for c in missing[:6]))
    for c in clips:
        if not c.source_uri:
            continue
        # concat demuxer 的转义规则：单引号写成 '\''
        path = str(c.source_uri).replace("'", "'\\''")
        out.append(f"file '{path}'")
        if c.source_in_s > 1e-6:
            out.append(f"inpoint {c.source_in_s:.3f}")
        out.append(f"outpoint {c.source_in_s + c.duration_s:.3f}")
        out.append(f"duration {c.duration_s:.3f}")
    return "\n".join(out) + "\n"


# ============================================================ 音画对齐检查


def sync_report(timeline: Timeline, *, tolerance_s: float = 0.04) -> str:
    """音画对齐检查。容差默认一帧左右（24fps 下 ~41ms），小于它的偏差人耳察觉不到。"""
    fps = timeline.fps
    total = timeline.duration_s()
    vclips = timeline.clips_of(TrackKind.VIDEO)
    out = [
        f"音画对齐报告 —— {timeline.name}",
        f"总长 {total:.2f}s / {timeline.duration_frames()} 帧 @ {fps}fps，"
        f"分辨率 {timeline.resolution[0]}x{timeline.resolution[1]}",
        "轨道：" + "、".join(f"{t.kind.value}({len(t.clips)})" for t in timeline.tracks),
        "",
    ]
    problems = 0

    # 1) 对白是否溢出所在镜头 —— 音频先行之后这里本该全绿，飘红说明有人手改过镜长
    out.append("[1] 对白溢出镜头")
    v_by_shot = {c.shot_id: c for c in vclips}
    hit = False
    for d in timeline.clips_of(TrackKind.DIALOGUE):
        v = v_by_shot.get(d.shot_id)
        if v is None:
            out.append(f"    ✗ {d.id} 找不到对应画面镜头 {d.shot_id}")
            hit, problems = True, problems + 1
            continue
        head = v.start_s - d.start_s
        tail = d.end_s - v.end_s
        if head > tolerance_s:
            out.append(f"    ✗ {d.id} 比镜头 {v.name} 早起 {head:.3f}s")
            hit, problems = True, problems + 1
        if tail > tolerance_s:
            out.append(f"    ✗ {d.id} 溢出镜头 {v.name} 尾部 {tail:.3f}s"
                       f"（建议把 {v.name} 延到 {d.end_s - v.start_s:.2f}s 或拆镜）")
            hit, problems = True, problems + 1
    if not hit:
        out.append("    ✓ 全部对白落在各自镜头内")

    # 2) 轨道空洞
    out.append("")
    out.append("[2] 轨道空洞")
    hit = False
    for t in timeline.tracks:
        if t.kind is TrackKind.SUBTITLE or not t.clips:
            continue   # 字幕本来就是稀疏的，不算空洞
        for a, b in t.gaps(until_s=total):
            if t.kind is TrackKind.VIDEO:
                out.append(f"    ✗ 画面轨 {a:.2f}s-{b:.2f}s 无素材，成片会是黑场（{b - a:.2f}s）")
                hit, problems = True, problems + 1
            elif t.kind is TrackKind.MUSIC:
                out.append(f"    · 音乐轨 {a:.2f}s-{b:.2f}s 无铺底（{b - a:.2f}s），确认是否有意留白")
                hit = True
    if not hit:
        out.append("    ✓ 无异常空洞")

    # 3) 同轨重叠：画面轨的重叠必须来自已声明的转场，否则就是排错了
    out.append("")
    out.append("[3] 同轨重叠")
    hit = False
    for a, b, ov in timeline.track(TrackKind.VIDEO).overlaps():
        if abs(ov - b.transition_s) > tolerance_s:
            out.append(f"    ✗ {a.name} 与 {b.name} 重叠 {ov:.3f}s，但声明的转场只有 {b.transition_s:.3f}s")
            hit, problems = True, problems + 1
        else:
            out.append(f"    · {a.name} -> {b.name} 溶解 {ov:.2f}s（{seconds_to_frames(ov, fps)} 帧），符合声明")
            hit = True
    for a, b, ov in timeline.track(TrackKind.DIALOGUE).overlaps():
        out.append(f"    ✗ 对白抢白：{a.name} 与 {b.name} 重叠 {ov:.3f}s")
        hit, problems = True, problems + 1
    if not hit:
        out.append("    ✓ 无重叠")

    # 4) 缺素材
    out.append("")
    out.append("[4] 缺失素材")
    miss = [c for t in timeline.tracks if t.kind is not TrackKind.SUBTITLE
            for c in t.sorted_clips() if not c.source_uri]
    if miss:
        for c in miss[:12]:
            out.append(f"    ✗ {c.kind.value} / {c.name or c.id} 尚无文件")
        if len(miss) > 12:
            out.append(f"    … 另有 {len(miss) - 12} 段")
        problems += len(miss)
    else:
        out.append("    ✓ 所有 clip 都已关联文件")

    out.append("")
    out.append(f"结论：{'通过 ✓' if problems == 0 else f'发现 {problems} 处问题 ✗'}")
    return "\n".join(out)


# ============================================================ 自测


def _selftest() -> None:
    import json
    import tempfile

    from .audio_first import (
        SilentTTSBackend, _demo_storyboard, build_audio_refs,
        retime_shots_to_audio, synthesize_dialogue,
    )
    from .schema import Transition as T

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sb = _demo_storyboard()
        sb.audio.music_track = str(tmp / "music.wav")
        sb.audio.ambience_track = str(tmp / "room_tone.wav")

        # 走一遍音频先行，拿到实测时间
        track = synthesize_dialogue(sb, SilentTTSBackend(), tmp / "dlg")
        retime_shots_to_audio(sb, track, min_s=4.0, max_s=10.0)
        build_audio_refs(track, sb, stem_dir=tmp / "stems")

        shots = sb.all_shots()
        shots[1].transition_in = T.DISSOLVE          # 制造一个真实的转场重叠
        shots[-1].transition_in = T.OVERLAP_BLEND
        shots[-1].continuity.overlap_frames = 12
        renders = {s.id: str(tmp / f"{s.id}.mp4") for s in shots}

        tl = build_timeline(sb, renders, dialogue=track, default_transition_s=0.5)
        sum_shots = sum(s.duration_s for s in shots)
        print(f"[1] 时间线 {tl.name}：{len(shots)} 镜，镜长之和 {sum_shots:.2f}s "
              f"-> 成片 {tl.duration_s():.2f}s")
        assert tl.duration_s() < sum_shots - 0.4, "溶解应当吃掉时间，总长必须短于镜长之和"
        assert len(tl.clips_of(TrackKind.VIDEO)) == len(shots)
        assert len(tl.clips_of(TrackKind.SUBTITLE)) == len(track.cues)
        v = tl.clips_of(TrackKind.VIDEO)
        assert abs(v[1].transition_s - 0.5) < 1e-6, v[1].transition_s
        assert abs(v[-1].transition_s - 12 / tl.fps) < 1e-6, v[-1].transition_s
        assert v[1].start_s < v[0].end_s - 1e-6, "溶解处 B 必须提前入点"

        # 对白绝对时间要和画面镜头对得上
        for d in tl.clips_of(TrackKind.DIALOGUE):
            vc = next(c for c in v if c.shot_id == d.shot_id)
            assert vc.start_s - 1e-6 <= d.start_s and d.end_s <= vc.end_s + 0.05, (d.id, d.start_s)
        print(f"[1] 对白 {len(tl.clips_of(TrackKind.DIALOGUE))} 条，全部落在所属镜头内")

        # 2) 时间码往返
        for sec in (0.0, 1.0, 3.7083, 61.5, 3723.25):
            tc = seconds_to_timecode(sec, 24)
            assert _timecode_to_frames(tc, 24) == seconds_to_frames(sec, 24), (sec, tc)
        assert frames_to_timecode(seconds_to_frames(3723.25, 24), 24) == "01:02:03:06"
        # 29.97 丢帧：第 1 分钟的 00 和 01 帧编号被跳过
        assert frames_to_timecode(1800, 30, drop_frame=True) == "00:01:00;02", \
            frames_to_timecode(1800, 30, drop_frame=True)
        # 丢帧也必须能原样还原：丢的是编号不是画面，00:01:00;02 的真实帧号是 1800
        for fr in (0, 1799, 1800, 17982, 107892):
            tc = frames_to_timecode(fr, 30, drop_frame=True)
            assert _timecode_to_frames(tc, 30, drop_frame=True) == fr, (fr, tc)
        # 10 分钟处不丢：17982 帧 = 00:10:00;00
        assert frames_to_timecode(17982, 30, drop_frame=True) == "00:10:00;00"
        # 24/25fps 上没有丢帧时间码这回事，必须炸而不是算出一串没人认识的数
        for bad_fps in (24, 25):
            try:
                frames_to_timecode(100, bad_fps, drop_frame=True)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{bad_fps}fps 的丢帧时间码应当报错")
        print("[2] 时间码往返一致（含丢帧），非 NTSC 帧率拒绝丢帧")

        # 3) EDL
        edl = export_edl(tl, record_start_tc="01:00:00:00")
        assert edl.startswith("TITLE:") and "FCM: NON-DROP FRAME" in edl
        d_lines = [ln for ln in edl.splitlines() if re.match(r"^\d{3}  \S+\s+V\s+D\s+\d{3}", ln)]
        assert len(d_lines) == 2, f"两处转场应各出一条 D 事件，实得 {len(d_lines)}\n" + edl
        ev = [ln for ln in edl.splitlines() if re.match(r"^\d{3}  ", ln)]
        assert all(len(re.split(r"\s+", ln.strip())) >= 8 for ln in ev)
        first_v = next(ln for ln in ev if re.match(r"^\d{3}  \S+\s+V ", ln))
        assert "01:00:00:00" in first_v, first_v
        # 画面事件的记录时间必须首尾相接、不得重叠：重叠的 EDL 套底会直接错位
        v_ev = [ln for ln in ev if re.match(r"^\d{3}  \S+\s+V\s", ln)]
        cur = _timecode_to_frames("01:00:00:00", tl.fps)
        for ln in v_ev:
            f = re.split(r"\s+", ln.strip())
            r_in, r_out = _timecode_to_frames(f[-2], tl.fps), _timecode_to_frames(f[-1], tl.fps)
            assert r_in == cur, f"记录时间不连续：期望 {cur} 实得 {r_in}\n{ln}"
            assert r_out >= r_in, ln
            cur = r_out
        assert cur - _timecode_to_frames("01:00:00:00", tl.fps) == tl.duration_frames(), cur
        print(f"[3] EDL {len(ev)} 个事件行，含 {len(d_lines)} 处溶解；"
              f"画面记录时间首尾相接且总长 {tl.duration_frames()} 帧")
        for ln in v_ev[:4]:
            print("     " + ln)

        # 4) OTIO：手写结构必须能被官方库原样解析回来
        doc = export_otio(tl)
        assert doc["OTIO_SCHEMA"] == "Timeline.1"
        blob = json.dumps(doc, ensure_ascii=False)
        try:
            import opentimelineio as otio
        except ImportError:
            print("[4] 未装 opentimelineio，跳过官方库校验（手写结构仍已导出）")
        else:
            parsed = otio.adapters.read_from_string(blob, "otio_json")
            n_tracks = len(parsed.tracks)
            got = parsed.duration().to_seconds()
            assert abs(got - tl.duration_s()) < 1.5 / tl.fps, (got, tl.duration_s())
            kinds = {t.kind for t in parsed.tracks}
            assert {"Video", "Audio", "Subtitle"} <= kinds, kinds
            n_trans = sum(
                1 for t in parsed.tracks for c in t
                if isinstance(c, otio.schema.Transition)
            )
            assert n_trans == 2, n_trans
            print(f"[4] OTIO 经 opentimelineio {otio.__version__} 解析通过："
                  f"{n_tracks} 轨 / {n_trans} 转场 / 时长 {got:.2f}s（手写结构合法）")

        # 5) ASS 字幕
        font = require_cjk_font()
        fonts = font.parent
        ass = export_ass_subtitles(tl, font=family_name(font))
        assert "[V4+ Styles]" in ass and "ScriptType: v4.00+" in ass
        evs = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
        assert len(evs) >= len(track.cues), (len(evs), len(track.cues))
        assert f"PlayResX: {tl.resolution[0]}" in ass
        # 安全区：MarginV 必须 > 0 且为画高的 ~8%
        style_line = next(ln for ln in ass.splitlines() if ln.startswith("Style: Default"))
        mv = int(style_line.split(",")[21])
        assert abs(mv - round(tl.resolution[1] * 0.08)) <= 1, mv
        long = "当年那场火烧掉的不只是仓库还有名单上二十七个人的全部记录以及所有的证词"
        wrapped = wrap_cjk(long, 18)
        assert all(len(x) <= 18 for x in wrapped) and "".join(wrapped).replace("，", "") \
            .startswith("当年那场火"), wrapped
        # 断行不能吃字：除了被吃掉的断句标点，正文必须一字不少
        assert "".join(wrapped) == long.replace("，", "").replace("、", ""), wrapped
        # max_chars<=0 曾经让 wrap_cjk 死循环（一个字都吃不掉），必须能收敛
        for bad in (0, -5):
            assert wrap_cjk("中文测试", bad) == list("中文测试"), bad
        assert wrap_cjk("", 18) == [""]
        # 传进来的样式不能被就地改写：同一份 style 要能连着导好几集
        shared = SubtitleStyle(font_name="SimHei")
        export_ass_subtitles(tl, font="Microsoft YaHei", style=shared)
        assert shared.font_name == "SimHei", f"style 被改写成了 {shared.font_name}"
        print(f"[5] ASS {len(evs)} 条事件，MarginV={mv}（安全区），断行示例 {wrapped}")

        # 真烧一次，证明 libass 吃得下这份 .ass（本机 ffmpeg 无 drawtext，只能走这条路）
        ass_path = tmp / "sub.ass"
        ass_path.write_text(ass, encoding="utf-8")
        from .audio_first import _run, ffmpeg_bin
        burn = tmp / "burn.mp4"
        r = _run([
            ffmpeg_bin(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:s=640x360:r={tl.fps}:d=3",
            "-vf", ass_burn_filter(ass_path, fonts_dir=fonts),
            "-frames:v", str(tl.fps * 2), str(burn),
        ])
        assert r.returncode == 0 and burn.exists() and burn.stat().st_size > 0, \
            r.stderr.decode("utf-8", "replace")[-600:]
        print(f"[5] libass 实烧通过：{burn.stat().st_size} 字节，字体目录 {fonts}")

        # 6) concat 清单
        cat = export_ffmpeg_concat(tl)
        files = [ln for ln in cat.splitlines() if ln.startswith("file ")]
        assert cat.startswith("ffconcat version 1.0")
        assert len(files) == len(v), (len(files), len(v))
        assert "警告" in cat and "xfade" in cat, "有转场时必须警告 concat 表达不了"
        print(f"[6] concat 清单 {len(files)} 条 file 指令，已警告 2 处转场")

        # 7) 对齐检查。先验证它确实会抓 Foley 缺素材（sfx "纸张翻动" 还没配音效文件）
        rep0 = sync_report(tl)
        assert "foley / 纸张翻动 尚无文件" in rep0 and "发现 1 处问题" in rep0, rep0
        print("[7] 未配 Foley 素材时 -> 抓到 1 处：" +
              next(ln.strip() for ln in rep0.splitlines() if "纸张翻动" in ln))
        for c in tl.track(TrackKind.FOLEY).clips:
            c.source_uri = c.source_uri or str(tmp / "paper.wav")
        rep = sync_report(tl)
        assert "结论：通过" in rep, rep
        print("[7] 补齐素材后 -> 通过")
        v[0].duration_s = 1.0
        bad = sync_report(tl)
        assert "溢出镜头" in bad and "发现" in bad, bad
        print("[7] 人为改短 sh01 后 -> 抓到溢出：")
        for ln in bad.splitlines():
            if "溢出镜头" in ln or "黑场" in ln:
                print("     " + ln.strip())

        # 8) 跨模块：转场长度必须和 stitch 烧片时算出来的一模一样。
        # 两边各写一份裁剪规则时不会报错，只会让 EDL 的切点和成片错位。
        from .schema import Continuity, DeliverySpec, Scene as _Scene, Shot as _Shot
        from .stitch import _clamp_transitions, resolve_clips
        probe_sb = Storyboard(project="x", scenes=[_Scene(id="s1", shots=[
            _Shot(id="a", scene_id="s1", index=0, duration_s=4.0, fps=25, render_uri="a.mp4"),
            _Shot(id="b", scene_id="s1", index=1, duration_s=2.0, fps=25, render_uri="b.mp4",
                  transition_in=T.CUT),
            _Shot(id="c", scene_id="s1", index=2, duration_s=5.0, fps=25, render_uri="c.mp4",
                  transition_in=T.DISSOLVE),
            _Shot(id="d", scene_id="s1", index=3, duration_s=6.0, fps=25, render_uri="d.mp4",
                  transition_in=T.OVERLAP_BLEND, continuity=Continuity(overlap_frames=10)),
        ])], delivery=DeliverySpec(resolution=(320, 240), upscale_to=None, fps=24))
        for want in (0.5, 3.0):
            ptl = build_timeline(probe_sb, default_transition_s=want)
            mine = [c.transition_s for c in ptl.clips_of(TrackKind.VIDEO)]
            theirs = _clamp_transitions(
                resolve_clips(probe_sb, default_transition_s=want),
                [s.duration_s for s in probe_sb.all_shots()],
            )
            assert all(abs(a - b) < 1e-9 for a, b in zip(mine, theirs)), (want, mine, theirs)
        # overlap_frames 按镜头自己的 fps（25）折算，不是交付帧率（24）
        last = ptl.clips_of(TrackKind.VIDEO)[-1]
        assert abs(last.transition_s - 10 / 25) < 1e-9, last.transition_s
        print(f"[8] 转场长度与 stitch 逐项一致：{[round(x, 3) for x in mine]}"
              f"；overlap_frames 按镜头 fps 折算 = {last.transition_s:.3f}s")

        # 9) 空输入：五个导出器都不该炸，也不该产出半截文件
        empty = build_timeline(Storyboard(project="empty"))
        assert empty.duration_s() == 0.0
        assert export_edl(empty).startswith("TITLE: empty_ep01")
        assert export_otio(empty)["OTIO_SCHEMA"] == "Timeline.1"
        assert not [ln for ln in export_ffmpeg_concat(empty).splitlines() if ln.startswith("file ")]
        assert "[Events]" in export_ass_subtitles(empty)
        assert "结论：通过" in sync_report(empty)
        # 单镜无对白：EDL 只有一条事件，且记录时间对得上
        one = build_timeline(Storyboard(project="one", scenes=[_Scene(id="s1", shots=[
            _Shot(id="only", scene_id="s1", index=0, duration_s=1.0, render_uri="x.mp4")])]))
        ev1 = [ln for ln in export_edl(one).splitlines() if re.match(r"^\d{3}  ", ln)]
        assert len(ev1) == 1 and ev1[0].endswith("01:00:00:00 01:00:01:00"), ev1
        print("[9] 空分镜 / 单镜无对白：五个导出器均正常，EDL 事件 " + str(len(ev1)) + " 条")

    print("\ntimeline 自测全部通过 ✓")


if __name__ == "__main__":
    _selftest()
