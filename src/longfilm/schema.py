"""分镜数据模型 —— 整条产线的唯一契约。

所有模块（参考位编排、提示词 OS、引擎路由、拼接、质检）都只认这里的类型。
一份 Storyboard JSON 就是一集片的完整可复现描述。
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------- 镜头语法枚举


class ShotSize(str, Enum):
    """景别。值直接用于提示词拼装，故写成英文术语。"""

    ECU = "extreme close-up"
    CU = "close-up"
    MCU = "medium close-up"
    MS = "medium shot"
    MLS = "medium long shot"
    FS = "full shot"
    LS = "long shot"
    ELS = "extreme long shot"
    OTS = "over-the-shoulder shot"
    POV = "point-of-view shot"
    INSERT = "insert shot"
    TWO = "two shot"


class CameraMove(str, Enum):
    STATIC = "static locked-off camera"
    PAN_L = "camera pans left"
    PAN_R = "camera pans right"
    TILT_U = "camera tilts up"
    TILT_D = "camera tilts down"
    DOLLY_IN = "camera dollies in"
    DOLLY_OUT = "camera dollies out"
    TRUCK_L = "camera trucks left"
    TRUCK_R = "camera trucks right"
    PEDESTAL_U = "camera pedestals up"
    PEDESTAL_D = "camera pedestals down"
    CRANE = "crane shot rising"
    ORBIT_L = "camera orbits left around subject"
    ORBIT_R = "camera orbits right around subject"
    HANDHELD = "handheld camera, subtle organic shake"
    STEADICAM = "steadicam follow"
    ZOOM_IN = "slow zoom in"
    ZOOM_OUT = "slow zoom out"
    PUSH_PULL = "dolly zoom (vertigo effect)"
    WHIP = "whip pan"


class Transition(str, Enum):
    """入点转场。拼接器按这个选 xfade 类型或硬切。"""

    CUT = "cut"
    DISSOLVE = "dissolve"
    FADE_IN = "fadeblack"
    FADE_OUT = "fadeblack"
    WIPE_L = "wipeleft"
    WIPE_R = "wiperight"
    SLIDE_L = "slideleft"
    MATCH_CUT = "match_cut"      # 靠首尾帧对齐做的无缝切，拼接器按 CUT 处理但要求 QC 更严
    OVERLAP_BLEND = "overlap"    # 重叠帧混接，A 尾 N 帧与 B 首 N 帧光流/线性混合


class ContentRating(str, Enum):
    """内容分级。路由器据此决定走官方 API 还是自建开源引擎。"""

    G = "g"            # 全年龄
    PG13 = "pg13"      # 轻度冲突/情绪张力
    R_VIOLENCE = "r_violence"
    R_SUGGESTIVE = "r_suggestive"   # 暗示性但非露骨
    BLOCKED = "blocked"             # 策略禁止，产线直接拒单


class EngineHint(str, Enum):
    AUTO = "auto"
    OFFICIAL = "official"   # 闭源 API（画质天花板，有审核）
    OPEN = "open"           # 自建开源（可控、可微调）
    HERO = "hero"           # 英雄镜：不计成本走最贵的
    DRAFT = "draft"         # 预演：走最便宜最快的


# ---------------------------------------------------------------- 参考位


ImageRole = Literal[
    "identity",      # 定妆/身份锚（最高权重）
    "wardrobe",      # 服装
    "environment",   # 场景
    "style",         # 画风/调色参考
    "prop",          # 道具
    "lighting",      # 光位参考
    "composition",   # 构图/分镜草图
    "first_frame",   # 首帧锁戏
    "last_frame",    # 尾帧锁戏
]

VideoRole = Literal["motion", "camera", "style", "previz"]
AudioRole = Literal["dialogue", "voice_timbre", "ambience", "music"]


class ImageRef(BaseModel):
    role: ImageRole
    uri: str
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    note: str = ""
    subject_id: str | None = None   # 这张图锁的是哪个角色


class VideoRef(BaseModel):
    role: VideoRole
    uri: str
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    note: str = ""


class AudioRef(BaseModel):
    role: AudioRole
    uri: str
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    note: str = ""


class RefPack(BaseModel):
    """一个镜头的多模态参考包。上限对齐官方 API：9 图 + 3 视频 + 3 音频。

    超限时由 refpack.py 的优先级裁剪器按 role 权重砍掉低价值位，
    不在这里静默截断 —— 静默截断会让身份锚丢失而查不出原因。
    """

    images: list[ImageRef] = Field(default_factory=list)
    videos: list[VideoRef] = Field(default_factory=list)
    audios: list[AudioRef] = Field(default_factory=list)

    MAX_IMAGES: int = 9
    MAX_VIDEOS: int = 3
    MAX_AUDIOS: int = 3

    @model_validator(mode="after")
    def _check_limits(self) -> RefPack:
        if len(self.images) > self.MAX_IMAGES:
            raise ValueError(
                f"参考图 {len(self.images)} 张超过上限 {self.MAX_IMAGES}；"
                "请先调用 refpack.fit_to_budget() 做优先级裁剪"
            )
        if len(self.videos) > self.MAX_VIDEOS:
            raise ValueError(f"参考视频 {len(self.videos)} 段超过上限 {self.MAX_VIDEOS}")
        if len(self.audios) > self.MAX_AUDIOS:
            raise ValueError(f"参考音频 {len(self.audios)} 段超过上限 {self.MAX_AUDIOS}")
        return self

    def by_role(self, role: str) -> list[ImageRef | VideoRef | AudioRef]:
        return [r for r in (*self.images, *self.videos, *self.audios) if r.role == role]

    def slots_used(self) -> dict[str, int]:
        return {"images": len(self.images), "videos": len(self.videos), "audios": len(self.audios)}


# ---------------------------------------------------------------- 角色圣经


class Appearance(BaseModel):
    face: str = ""
    hair: str = ""
    body: str = ""
    skin: str = ""
    wardrobe: str = ""
    distinguishing: str = ""   # 疤/痣/纹身等强锚点，跨镜一致性的抓手

    def to_prompt(self) -> str:
        parts = [self.face, self.hair, self.body, self.skin, self.wardrobe, self.distinguishing]
        return ", ".join(p.strip() for p in parts if p.strip())


class VoiceProfile(BaseModel):
    timbre_ref_uri: str | None = None   # 音色克隆参考（3-10s 干声）
    tts_engine: str = "indextts2"
    tts_voice_id: str | None = None
    pitch_shift: float = 0.0
    speed: float = 1.0
    emotion_default: str = "neutral"
    language: str = "zh"


class LoRASpec(BaseModel):
    base_model: str                     # 如 "wan2.2-i2v-a14b"
    path: str                           # 本地或对象存储路径
    trigger_word: str
    strength: float = Field(default=0.85, ge=0.0, le=2.0)
    trained_steps: int | None = None
    dataset_hash: str | None = None     # 训练集指纹，用于复现


class CharacterBible(BaseModel):
    """角色圣经。所有镜头的身份一致性都从这里取锚。

    age_statement 是强制字段：本产线只做虚构成年角色，
    该声明会写进产物 manifest，也会被内容门禁读取。
    """

    id: str
    name: str
    is_fictional: Literal[True] = True
    age_statement: str = Field(description="必须显式写明为虚构成年角色，例如 '虚构角色，设定年龄 27 岁'")
    persona: str = ""
    appearance: Appearance = Field(default_factory=Appearance)
    voice: VoiceProfile = Field(default_factory=VoiceProfile)

    # 定妆资产：视角 × 光位 × 表情 的矩阵，refpack 从中挑 identity 锚
    portraits: list[ImageRef] = Field(default_factory=list)
    turnaround: list[ImageRef] = Field(default_factory=list)
    motion_refs: list[VideoRef] = Field(default_factory=list)

    lora: LoRASpec | None = None
    negative_prompt: str = ""

    @field_validator("age_statement")
    @classmethod
    def _must_state_adult(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("age_statement 不可为空：本产线要求每个角色显式声明为虚构成年角色")
        return v

    def identity_prompt(self) -> str:
        bits = [self.appearance.to_prompt()]
        if self.lora:
            bits.insert(0, self.lora.trigger_word)
        return ", ".join(b for b in bits if b)


# ---------------------------------------------------------------- 镜头


class DialogueLine(BaseModel):
    speaker_id: str
    text: str
    emotion: str = "neutral"
    start_s: float | None = None    # 音频先行时由 TTS 实测时长回填
    end_s: float | None = None


class Continuity(BaseModel):
    """跨镜连续性约束。拼接器和续写链读这里。"""

    prev_shot_id: str | None = None
    next_shot_id: str | None = None
    inherit_last_frame: bool = False    # 本镜首帧 = 上一镜尾帧
    emit_last_frame: bool = True        # 导出尾帧供下一镜用
    extend_from_job: str | None = None  # 官方 extend：接这个 job 往下续
    overlap_frames: int = 0             # >0 时拼接器做重叠混接
    screen_direction: Literal["l2r", "r2l", "neutral"] = "neutral"  # 轴线，防跳轴
    match_on: str = ""                  # match cut 的匹配元素（形状/动作/颜色）


class Grade(BaseModel):
    """该镜的目标调色，供 grade.py 统一到场景基准。"""

    palette: str = ""
    contrast: float = 1.0
    saturation: float = 1.0
    temperature: float = 0.0    # -1 冷 .. +1 暖
    lut: str | None = None


class Shot(BaseModel):
    id: str
    scene_id: str
    index: int
    duration_s: float = Field(default=8.0, ge=1.0, le=60.0)

    shot_size: ShotSize = ShotSize.MS
    camera_move: CameraMove = CameraMove.STATIC
    lens_mm: int = 35
    aperture: str = "f/2.8"
    fps: int = 24

    subject_ids: list[str] = Field(default_factory=list)
    action: str = ""
    environment: str = ""
    lighting: str = ""
    mood: str = ""
    style: str = ""

    dialogue: list[DialogueLine] = Field(default_factory=list)
    sfx: list[str] = Field(default_factory=list)

    refs: RefPack = Field(default_factory=RefPack)
    continuity: Continuity = Field(default_factory=Continuity)
    grade: Grade = Field(default_factory=Grade)
    transition_in: Transition = Transition.CUT

    content_rating: ContentRating = ContentRating.G
    engine_hint: EngineHint = EngineHint.AUTO
    seed: int | None = None
    negative_prompt: str = ""

    # 产线回填
    provider_used: str | None = None
    job_id: str | None = None
    render_uri: str | None = None
    qc_score: float | None = None
    takes: int = 0

    @field_validator("duration_s")
    @classmethod
    def _atomic_window(cls, v: float) -> float:
        # 不在这里硬卡 15s：不同 provider 上限不同，由 router 按 capabilities 裁。
        # 但低于 1s 的镜没有生成意义。
        return v

    def fingerprint(self) -> str:
        """内容指纹：同指纹的镜可以命中缓存，不重复烧 GPU/API 额度。"""
        payload = self.model_dump(
            exclude={"provider_used", "job_id", "render_uri", "qc_score", "takes"}
        )
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Scene(BaseModel):
    id: str
    title: str = ""
    synopsis: str = ""
    location: str = ""
    time_of_day: str = "day"
    base_grade: Grade = Field(default_factory=Grade)
    shots: list[Shot] = Field(default_factory=list)

    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.shots)


# ---------------------------------------------------------------- 全片


class AudioPlan(BaseModel):
    """分轨工业化：画面 / 对白 / Foley / 音乐 四条独立轨，最后对齐。"""

    dialogue_track: str | None = None
    foley_track: str | None = None
    music_track: str | None = None
    ambience_track: str | None = None
    audio_first: bool = True       # 先出对白轨再出画（口型更稳）
    loudness_lufs: float = -16.0   # 流媒体交付标准


class DeliverySpec(BaseModel):
    resolution: tuple[int, int] = (1280, 720)
    upscale_to: tuple[int, int] | None = (1920, 1080)
    fps: int = 24
    interpolate_to_fps: int | None = None
    codec: Literal["h264", "h265", "av1"] = "h264"
    crf: int = 18
    aspect: str = "16:9"
    burn_subtitles: bool = False
    ai_disclosure: bool = True     # 合规：显式标识 AI 生成
    c2pa: bool = True


class Storyboard(BaseModel):
    project: str
    episode: str = "ep01"
    logline: str = ""
    target_duration_s: float = 300.0

    characters: list[CharacterBible] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    audio: AudioPlan = Field(default_factory=AudioPlan)
    delivery: DeliverySpec = Field(default_factory=DeliverySpec)

    style_bible: str = ""          # 全片统一画风描述，拼进每条提示词
    global_negative: str = ""

    def all_shots(self) -> list[Shot]:
        return [s for sc in self.scenes for s in sc.shots]

    def character(self, cid: str) -> CharacterBible | None:
        return next((c for c in self.characters if c.id == cid), None)

    def duration_s(self) -> float:
        return sum(sc.duration_s() for sc in self.scenes)

    def validate_continuity(self) -> list[str]:
        """返回连续性问题清单（不抛异常，交给 CLI 决定是否放行）。"""
        problems: list[str] = []
        shots = self.all_shots()
        ids = {s.id for s in shots}
        for s in shots:
            for ref_id, label in (
                (s.continuity.prev_shot_id, "prev_shot_id"),
                (s.continuity.next_shot_id, "next_shot_id"),
            ):
                if ref_id and ref_id not in ids:
                    problems.append(f"{s.id}: {label} 指向不存在的镜头 {ref_id}")
            for sid in s.subject_ids:
                if not self.character(sid):
                    problems.append(f"{s.id}: 角色 {sid} 不在角色圣经里")
            if s.continuity.inherit_last_frame and not s.continuity.prev_shot_id:
                problems.append(f"{s.id}: 声明继承尾帧但没有 prev_shot_id")
            if s.content_rating is ContentRating.BLOCKED:
                problems.append(f"{s.id}: 内容分级为 blocked，产线拒单")
        # 轴线检查。判据是「同一个角色的视线/运动方向在后续镜头里翻转」——
        # 正反打里两个角色方向相反是正确的对切，不是跳轴，所以必须按角色追踪，
        # 而不是简单比较相邻镜头的方向。
        for sc in self.scenes:
            last_dir: dict[str, tuple[str, str]] = {}   # char_id -> (direction, shot_id)
            for shot in sc.shots:
                d = shot.continuity.screen_direction
                if d == "neutral":
                    continue
                for cid in shot.subject_ids:
                    prev = last_dir.get(cid)
                    if prev and prev[0] != d and not shot.continuity.match_on:
                        problems.append(
                            f"{shot.id}: 角色 {cid} 相对 {prev[1]} 跳轴（{prev[0]} -> {d}）"
                            "；若为有意翻转请在 continuity.match_on 声明匹配元素"
                        )
                    last_dir[cid] = (d, shot.id)
        return problems

    @classmethod
    def load(cls, path: str | Path) -> Storyboard:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p


def json_schema() -> dict[str, Any]:
    """导出 JSON Schema，供 LLM 生成分镜时做结构化输出约束。"""
    return Storyboard.model_json_schema()
