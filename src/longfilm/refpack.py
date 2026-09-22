"""参考位编排器 —— 9 图 + 3 视频 + 3 音频的价值排序与预算裁剪。

一个镜头**可用**的参考素材几乎总是多于**可提交**的参考位：
角色定妆、服装、场景、全片风格、道具、光位、构图草图、首帧、尾帧……
随便一个两人对戏的中景就能凑出十几张。所以这一层要回答两个问题：

1. 该塞哪些？—— `build_refpack()` 从角色圣经 / 场景 / 镜头自动组装；
2. 塞不下砍谁？—— `fit_to_budget()` 按价值排序裁剪，并**如实返回被砍清单**。

裁剪必须是可审计的。静默截断是这条产线上最贵的 bug：身份锚被 list 切片
悄悄丢掉，产出的脸漂了，排查时却完全看不出是「参考位没进去」还是
「模型本身不稳」，只能盲目重抽 —— 每次重抽都是真金白银的 GPU/API 额度。
所以 `fit_to_budget` 返回 `list[DroppedRef]`，`slot_manifest()` 把最终占位
打成人类可读报告，专门用来回答「为什么这一镜脸漂了」。

接口设计参考了 GitHub 上的 `Emily2040/seedance-2.0`（查证日期 2026-09-19，
仓库真实存在，7.3k stars，默认分支 main，最后推送 2026-09-08）。借鉴的点：
- 参考位按**上传序**绑定标签 `@Image1..9` / `@Video1..3` / `@Audio1..3`
  （中文界面 `@图片N` / `@视频N` / `@音频N`），标签必须逐字保留、不可改写成
  `[Image1]`，否则平台的 @ 解析器静默失配 —— 见其 references/reference-workflow.md；
- 「transfer / ignore 子句」：每个参考位都要显式写明*控制什么*和*不继承什么*，
  否则 identity / motion / environment 角色会互相串味 —— 见其
  references/reference-transfer-contract.md；
- 「配额模型」：身份保真、动作幅度、场景密度抢同一份预算，参考图承担的保真
  就是提示词不必再花的预算 —— 见其 references/allocation-model.md。
本模块只借鉴接口形态，数据结构一律复用 longfilm.schema。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

from .providers.base import Capabilities
from .schema import (
    AudioRef,
    CharacterBible,
    ImageRef,
    RefPack,
    Shot,
    ShotSize,
    Storyboard,
    VideoRef,
)

logger = logging.getLogger(__name__)

RefKind = Literal["image", "video", "audio"]


# ---------------------------------------------------------------- 优先级表

# 图参考位的基础优先级。数值越大越不可裁。排序理由（从不可裁到可裁）：
#
#   first_frame(100) 直接决定第 0 帧的像素。掉了它，本镜起点与上一镜尾帧对不上，
#                    整条续写链断裂 —— 这是唯一一个「丢了就必须重做整段」的位。
#   identity(95)     身份锚。脸崩是后期**修不回来**的缺陷（调色能救光，插帧能救
#                    抖动，没有任何后期能把 A 的脸换回 B），所以排在一切可修项之上。
#   last_frame(85)   首尾帧模式下锁定出点，决定下一镜能不能接。比 identity 低一档，
#                    因为尾帧对不上最多是本镜转场难看，脸漂是整集废掉。
#   wardrobe(60)     服装跨镜跳变观众一眼看出，但至少可以靠提示词文本兜底。
#   environment(50)  场景同理，且 environment 常常已经被 first_frame 隐含携带。
#   style(45)        画风跳变是跨镜可见缺陷，影响的是整集观感；虽有 style_bible 文本
#                    兜底，但文本描述画风的还原度远低于一张图。
#   composition(35)  构图草图只影响单镜、不影响一致性，缺了模型自己也能编一个合理构图。
#   prop(20)         道具错了是穿帮，但通常只影响单镜，且可以重抽。
#   lighting(10)     光位最容易被后期 colorbalance/lut3d 救回来，因此最先砍。
#
# 视频位和音频位单独建表：role 名字与图位有重叠（style 两边都有）但价值完全不同，
# 合并成一张表会让「视频 style 参考」继承图位的权重，排序就错了。
ROLE_PRIORITY: dict[str, int] = {
    "first_frame": 100,
    "identity": 95,
    "last_frame": 85,
    "wardrobe": 60,
    "environment": 50,
    "style": 45,
    "composition": 35,
    "prop": 20,
    "lighting": 10,
}

# 视频位：motion 决定动作是否可信，是视频参考的主要价值；camera 次之（相机运动
# 也能用文本描述兜底）；previz 是预演片，只做节奏参考；style 最弱，图位就能做。
VIDEO_ROLE_PRIORITY: dict[str, int] = {
    "motion": 70,
    "camera": 60,
    "previz": 40,
    "style": 30,
}

# 音频位：dialogue 是音频先行产线的驱动源（口型对不上整镜作废），必须保；
# voice_timbre 决定音色一致性，跨集可听出来；ambience/music 属于后期可替换轨。
AUDIO_ROLE_PRIORITY: dict[str, int] = {
    "dialogue": 100,
    "voice_timbre": 80,
    "ambience": 40,
    "music": 30,
}

# RefPack 契约里的硬上限，从 schema 读而不是写死 9/3/3，避免两处不同步。
_HARD_IMAGE_CAP: int = RefPack.model_fields["MAX_IMAGES"].default
_HARD_VIDEO_CAP: int = RefPack.model_fields["MAX_VIDEOS"].default
_HARD_AUDIO_CAP: int = RefPack.model_fields["MAX_AUDIOS"].default


def role_priority(kind: RefKind, role: str) -> int:
    """取某个参考位的基础优先级。未知 role 给 0 —— 最先被裁，但不抛异常，
    因为 role 是 Literal，新增角色时不该让整条产线挂掉。"""
    table = {
        "image": ROLE_PRIORITY,
        "video": VIDEO_ROLE_PRIORITY,
        "audio": AUDIO_ROLE_PRIORITY,
    }[kind]
    return table.get(role, 0)


# ---------------------------------------------------------------- 预算


@dataclass(frozen=True)
class RefBudget:
    """某个 provider 能吃下的参考位预算。

    不同引擎差别极大：闭源 API 通常 9 图 / 3 视频 / 3 音频，开源 i2v 常常只有
    1 张首帧，纯 t2v 一张都不吃。把预算做成显式对象而不是散在 router 里的
    if，换供应商时只改 Capabilities 声明。
    """

    max_images: int = _HARD_IMAGE_CAP
    max_videos: int = _HARD_VIDEO_CAP
    max_audios: int = _HARD_AUDIO_CAP
    allowed_image_roles: frozenset[str] = frozenset()   # 空 = 不限制
    supports_first_frame: bool = True
    supports_last_frame: bool = True
    provider: str = "default"

    @classmethod
    def from_capabilities(cls, caps: Capabilities) -> RefBudget:
        """从 provider 能力声明生成预算，并夹到 RefPack 契约上限内。

        夹上限是因为 Capabilities 由各 provider 自己填，填大了会让 RefPack
        的 model_validator 直接抛异常 —— 那个异常发生在提交前一刻，比在这里
        夹住难排查得多。
        """
        return cls(
            max_images=max(0, min(caps.max_ref_images, _HARD_IMAGE_CAP)),
            max_videos=max(0, min(caps.max_ref_videos, _HARD_VIDEO_CAP)),
            max_audios=max(0, min(caps.max_ref_audios, _HARD_AUDIO_CAP)),
            allowed_image_roles=frozenset(caps.supported_image_roles),
            supports_first_frame=caps.supports_first_frame,
            supports_last_frame=caps.supports_last_frame,
            provider=caps.name,
        )

    def accepts_image_role(self, role: str) -> bool:
        if role == "first_frame" and not self.supports_first_frame:
            return False
        if role == "last_frame" and not self.supports_last_frame:
            return False
        if self.allowed_image_roles and role not in self.allowed_image_roles:
            return False
        return True


@dataclass(frozen=True)
class DroppedRef:
    """一条被裁掉的参考。调用方据此决定是降级引擎还是改工单。"""

    kind: RefKind
    role: str
    uri: str
    reason: str
    subject_id: str | None = None
    critical: bool = False      # True = 砍掉了身份锚/首帧，画面一致性已无保障

    def __str__(self) -> str:
        mark = "!! " if self.critical else "   "
        who = f" [{self.subject_id}]" if self.subject_id else ""
        return f"{mark}{self.kind}:{self.role}{who} {_short_uri(self.uri)} — {self.reason}"


# ---------------------------------------------------------------- 标签绑定

_TAG_WORDS: dict[str, dict[RefKind, str]] = {
    "en": {"image": "Image", "video": "Video", "audio": "Audio"},
    "zh": {"image": "图片", "video": "视频", "audio": "音频"},
}

# 每个 role「控制什么 / 不要继承什么」。写死 ignore 子句是因为角色串味是
# 多参考位最常见的失败模式：一张场景图会顺手把它自己的光线和人也带进来。
_TRANSFER_CLAUSE: dict[str, tuple[str, str]] = {
    "identity": ("角色身份：面部结构、发型、体型、专属标记", "该图的环境、光线、构图"),
    "wardrobe": ("服装款式、材质、配色", "该图的人脸、姿态、背景"),
    "environment": ("场景空间、陈设、材质", "该图里出现的人物与其身份"),
    "style": ("画风、色调、质感", "该图的具体内容与人物"),
    "prop": ("道具的形制与材质", "该图的人物、场景、光线"),
    "lighting": ("光位、光比、色温", "该图的人物、场景、构图"),
    "composition": ("构图、机位、主体在画幅中的占比", "该图的画风与人物身份"),
    "first_frame": ("本镜起始画面的全部像素", "任何与之冲突的文本描述"),
    "last_frame": ("本镜结束画面的目标状态", "中间过程的动作设计"),
    "motion": ("动作编排、节奏、肢体时序", "该片段的人物身份、服装、场景、logo"),
    "camera": ("运镜轨迹与速度", "该片段的人物、场景、画风"),
    "previz": ("段落节奏与镜头长度", "该片段的画质与身份"),
    "dialogue": ("对白内容与时长（口型对齐基准）", "该音轨的环境噪声"),
    "voice_timbre": ("音色", "该音轨的语速与内容"),
    "ambience": ("环境声底噪", "任何人声"),
    "music": ("配乐情绪与节奏", "任何人声与歌词"),
}


def slot_tag(kind: RefKind, index0: int, lang: str = "en") -> str:
    """参考位标签。index0 从 0 起，标签从 1 起（平台按上传序编号）。

    标签必须逐字出现在提示词里，不能改成 `[Image1]` 或翻译 —— 平台的 @ 解析器
    只认这一种写法，写错了参考不绑定却不报错，是最难查的一类失败。
    """
    return f"@{_TAG_WORDS[lang][kind]}{index0 + 1}"


def binding_directives(pack: RefPack, *, lang: str = "en") -> list[str]:
    """为每个已占用的参考位生成一条「控制什么 / 不继承什么」指令。

    prompt_os 直接引用这些句子，不自己另造一套标签逻辑 —— 标签编号必须与
    实际提交顺序严格一致，两处各算一次迟早会错位。
    """
    out: list[str] = []
    for kind, refs in (("image", pack.images), ("video", pack.videos), ("audio", pack.audios)):
        for i, r in enumerate(refs):
            controls, ignores = _TRANSFER_CLAUSE.get(r.role, (r.role, "其它一切"))
            who = f"（{r.subject_id}）" if getattr(r, "subject_id", None) else ""
            out.append(
                f"{slot_tag(kind, i, lang)} 只控制{controls}{who}；不要继承{ignores}。"
            )
    return out


# ---------------------------------------------------------------- 身份锚挑选

# 景别 → 期望的定妆视角。近景要正脸特写（观众盯着看的就是五官），
# 远景要全身 turnaround（这时脸只有几十个像素，体型和服装剪影才是识别线索）。
_VIEW_KEYWORDS: dict[str, tuple[str, ...]] = {
    "face": ("正脸", "脸", "front", "face", "portrait", "特写", "closeup", "close-up", "头像"),
    "bust": ("半身", "bust", "three-quarter", "3/4", "四分之三", "侧脸", "profile"),
    "full": ("全身", "full", "turnaround", "转身", "t-pose", "a-pose", "全景", "body"),
}

_CLOSE_SIZES = frozenset({ShotSize.ECU, ShotSize.CU, ShotSize.MCU, ShotSize.INSERT, ShotSize.OTS})
_MID_SIZES = frozenset({ShotSize.MS, ShotSize.MLS, ShotSize.TWO, ShotSize.POV})


def preferred_views(shot_size: ShotSize) -> tuple[str, ...]:
    """按景别给出视角偏好序。返回的是**完整降级序**而不是单个选择，
    因为角色圣经里未必备齐所有视角，缺了要能平滑退到次优。"""
    if shot_size in _CLOSE_SIZES:
        return ("face", "bust", "full")
    if shot_size in _MID_SIZES:
        return ("bust", "face", "full")
    return ("full", "bust", "face")


def _view_of(ref: ImageRef) -> str:
    """从 note/uri 里识别这张定妆图是什么视角。识别不出按 bust 处理 ——
    半身是最不会错得离谱的默认值。"""
    hay = f"{ref.note} {ref.uri}".lower()
    for view in ("face", "full", "bust"):
        if any(k.lower() in hay for k in _VIEW_KEYWORDS[view]):
            return view
    return "bust"


def pick_identity_anchors(char: CharacterBible, shot_size: ShotSize, k: int) -> list[ImageRef]:
    """按景别为一个角色挑 k 张身份锚，并回填 subject_id。

    排序键是 (视角偏好名次, -weight, 原始下标)：全部是确定性量，
    同一份角色圣经两次调用必须挑出同一批图，否则 Shot.fingerprint 缓存失效。
    """
    if k <= 0:
        return []
    pool = [r for r in (*char.portraits, *char.turnaround) if r.role == "identity"]
    if not pool:
        # 圣经里没标 identity 的，把所有定妆图都当身份锚用：宁可用一张服装图
        # 锁脸，也不要让这个角色完全没有视觉锚点。
        pool = list(char.portraits) + list(char.turnaround)
    order = preferred_views(shot_size)
    ranked = sorted(
        enumerate(pool),
        key=lambda t: (order.index(_view_of(t[1])) if _view_of(t[1]) in order else 9,
                       -t[1].weight, t[0]),
    )
    picked: list[ImageRef] = []
    seen: set[str] = set()
    for _, ref in ranked:
        if ref.uri in seen:
            continue
        seen.add(ref.uri)
        picked.append(ref.model_copy(update={"role": "identity", "subject_id": char.id}))
        if len(picked) == k:
            break
    return picked


# ---------------------------------------------------------------- 多角色配额


def allocate_identity_quota(subject_ids: Sequence[str], slots: int) -> dict[str, int]:
    """多角色同框时 identity 位怎么分。

    规则：先保底后轮转 —— 每个角色先拿 1 张，剩余名额按 subject_ids 的书写顺序
    轮流发放（分镜约定主角写在前面，所以靠前的角色先拿第 2 张）。

    保底是硬要求。如果按「权重高的多拿」来分，两人对戏时很容易出现主角 5 张、
    配角 0 张：配角的脸会直接漂成随机路人，而这在双人镜里是最刺眼的穿帮。
    名额不够给所有人保底时（极小预算的 provider），只能前 N 个角色各 1 张，
    其余角色靠提示词文本描述兜底，并打 warning 让调用方知道该降级换引擎。
    """
    uniq: list[str] = []
    for sid in subject_ids:
        if sid not in uniq:
            uniq.append(sid)
    if not uniq or slots <= 0:
        return {}
    if slots < len(uniq):
        logger.warning(
            "identity 位只有 %d 个但同框 %d 个角色：%s 将没有身份锚，只能靠文本描述兜底",
            slots, len(uniq), ", ".join(uniq[slots:]),
        )
        return {sid: (1 if i < slots else 0) for i, sid in enumerate(uniq)}
    quota = {sid: 1 for sid in uniq}
    for i in range(slots - len(uniq)):
        quota[uniq[i % len(uniq)]] += 1
    return quota


def _identity_slot_budget(n_subjects: int, max_images: int, has_frame_lock: bool) -> int:
    """identity 位总额：给服装/场景/风格留出底线，再把剩下的全给身份。

    留 3 个非身份位是经验值：服装 1 + 场景 1 + 风格 1 正好覆盖「观众能一眼看出
    不对」的三类跨镜跳变。预算小于 6 时只留 1 个，小于 3 时一个不留 —— 那种
    预算下身份锚本身都不够，再谈风格没有意义。
    """
    if max_images <= 0:
        return 0
    reserve = 3 if max_images >= 6 else (1 if max_images >= 3 else 0)
    if has_frame_lock:
        reserve += 1   # 首帧位不能被身份位挤掉
    cap = max(0, max_images - reserve)
    return max(cap, min(n_subjects, max_images))


# ---------------------------------------------------------------- 组装


def build_refpack(
    shot: Shot,
    storyboard: Storyboard,
    *,
    extra_images: Sequence[ImageRef] = (),
    extra_videos: Sequence[VideoRef] = (),
    extra_audios: Sequence[AudioRef] = (),
    budget: RefBudget | None = None,
    prev_last_frame_uri: str | None = None,
    drops_out: list[DroppedRef] | None = None,
) -> RefPack:
    """从角色圣经 + 场景 + 镜头自动组装参考包，并裁到预算内。

    组装顺序即价值顺序，后面的候选只填前面剩下的空位：

    1. 首帧锁戏：`continuity.inherit_last_frame` 时把上一镜尾帧放进 first_frame 位。
       尾帧 uri 优先用调用方传入的 `prev_last_frame_uri`（真实渲染产物），
       其次在上一镜的 refs 里找 role="last_frame" —— Shot 契约里没有
       last_frame_uri 字段，尾帧是产线运行期的产物，不该反写进分镜。
    2. 每个 subject 按 `allocate_identity_quota` 的配额取身份锚，按景别选视角。
    3. 角色圣经里的 wardrobe 图。
    4. 镜头自带的 refs（分镜作者手工指定的场景/道具/构图，权威高于自动推导）。
    5. extra_images：调用方补的场景图、全片 style 图等。

    注：Storyboard 契约里只有 `style_bible` 文本，没有全片 style **图**字段，
    所以风格参考图必须由调用方经 extra_images 传入（role="style"）。这里不去
    猜一个不存在的字段。

    被裁清单写进 `drops_out`（若提供）并同时打日志：返回类型必须是 RefPack
    才能直接喂给 GenRequest，所以裁剪信息走出参走 —— 但绝不静默丢弃。
    """
    bud = budget or RefBudget()
    drops: list[DroppedRef] = []
    images: list[ImageRef] = []

    # 1) 首帧继承
    frame_uri = prev_last_frame_uri
    if shot.continuity.inherit_last_frame and frame_uri is None:
        frame_uri = _prev_last_frame(shot, storyboard)
    has_frame_lock = bool(shot.continuity.inherit_last_frame and frame_uri)
    if has_frame_lock:
        images.append(
            ImageRef(
                role="first_frame",
                uri=frame_uri or "",
                weight=1.0,
                note=f"继承自 {shot.continuity.prev_shot_id} 的尾帧",
            )
        )
    elif shot.continuity.inherit_last_frame:
        logger.warning(
            "%s 声明 inherit_last_frame 但拿不到上一镜尾帧，本镜起点无锚", shot.id
        )

    # 2) 身份锚
    # 每个 subject 至少生成一个身份锚**候选**，哪怕预算根本放不下。
    # 让 fit_to_budget 去砍并标 critical，比在这里悄悄不生成好：
    # 前者 router 能看见「这家 provider 装不下身份锚」并降级，后者查无此事。
    n_subjects = len(set(shot.subject_ids))
    id_slots = max(
        _identity_slot_budget(n_subjects, bud.max_images, has_frame_lock), n_subjects
    )
    quota = allocate_identity_quota(shot.subject_ids, id_slots)
    for sid, n in quota.items():
        char = storyboard.character(sid)
        if char is None:
            logger.warning("%s: 角色 %s 不在角色圣经里，无法取身份锚", shot.id, sid)
            continue
        images.extend(pick_identity_anchors(char, shot.shot_size, n))

    # 3) 服装
    for sid in quota:
        char = storyboard.character(sid)
        if char is None:
            continue
        for r in (*char.portraits, *char.turnaround):
            if r.role == "wardrobe":
                images.append(r.model_copy(update={"subject_id": char.id}))

    # 4) 镜头自带 + 5) 调用方补充
    images.extend(shot.refs.images)
    images.extend(extra_images)

    videos: list[VideoRef] = list(shot.refs.videos) + list(extra_videos)
    for sid in quota:
        char = storyboard.character(sid)
        if char is not None:
            videos.extend(char.motion_refs)

    audios: list[AudioRef] = list(shot.refs.audios) + list(extra_audios)
    for sid in quota:
        char = storyboard.character(sid)
        if char is not None and char.voice.timbre_ref_uri:
            audios.append(
                AudioRef(role="voice_timbre", uri=char.voice.timbre_ref_uri, note=f"{sid} 音色")
            )

    kept_i, kept_v, kept_a, drops = fit_to_budget(images, videos, audios, bud)
    if drops_out is not None:
        drops_out.extend(drops)
    for d in drops:
        (logger.error if d.critical else logger.info)("%s 裁剪参考位: %s", shot.id, d)

    return RefPack(images=kept_i, videos=kept_v, audios=kept_a)


def _prev_last_frame(shot: Shot, storyboard: Storyboard) -> str | None:
    pid = shot.continuity.prev_shot_id
    if not pid:
        return None
    prev = next((s for s in storyboard.all_shots() if s.id == pid), None)
    if prev is None:
        return None
    for r in prev.refs.images:
        if r.role == "last_frame":
            return r.uri
    return None


# ---------------------------------------------------------------- 裁剪


def fit_to_budget(
    images: Sequence[ImageRef],
    videos: Sequence[VideoRef],
    audios: Sequence[AudioRef],
    budget: RefBudget,
) -> tuple[list[ImageRef], list[VideoRef], list[AudioRef], list[DroppedRef]]:
    """优先级裁剪。返回 (保留图, 保留视频, 保留音频, 被裁清单)。

    三道工序：
    1. 去重 —— 同一 uri 占两个位是纯浪费，先合并（保留先出现的那条）；
    2. 过滤 —— provider 不支持的 role 直接剔除（比如纯 t2v 不吃 first_frame）；
    3. 排序裁剪 —— 排序键 (-优先级, -weight, 原始下标)，全确定性，可复现。

    身份锚保护：排序之前先为**每个 subject** 锁定一张 identity 图（该角色里
    优先级/权重最高的那张）。这批图在裁剪时先落座，之后才轮到其它 role 抢位。
    这样即使预算只剩 2 张，双人镜也是「每人一张脸」而不是「主角两张脸 + 配角没脸」。
    只有当保护位本身就超过预算时才会砍身份锚，此时对应 DroppedRef.critical=True，
    调用方应当据此降级换引擎，而不是照常提交。
    """
    drops: list[DroppedRef] = []

    imgs = _dedup(images, "image", drops)
    vids = _dedup(videos, "video", drops)
    auds = _dedup(audios, "audio", drops)

    allowed: list[ImageRef] = []
    for r in imgs:
        if budget.accepts_image_role(r.role):
            allowed.append(r)
        else:
            drops.append(
                DroppedRef("image", r.role, r.uri, f"{budget.provider} 不支持该 role",
                           r.subject_id)
            )
    imgs = allowed

    ranked = sorted(
        range(len(imgs)),
        key=lambda i: (-role_priority("image", imgs[i].role), -imgs[i].weight, i),
    )

    protected: list[int] = []
    seen_subjects: set[str | None] = set()
    for i in ranked:
        r = imgs[i]
        if r.role == "identity" and r.subject_id not in seen_subjects:
            seen_subjects.add(r.subject_id)
            protected.append(i)
        elif r.role == "first_frame":
            protected.append(i)
    protected_set = set(protected)

    keep_idx: list[int] = []
    # 保护位按 ranked 次序落座，超预算的部分标 critical。
    for i in sorted(protected, key=ranked.index):
        if len(keep_idx) < budget.max_images:
            keep_idx.append(i)
        else:
            r = imgs[i]
            drops.append(
                DroppedRef("image", r.role, r.uri,
                           f"预算仅 {budget.max_images} 张，身份/首帧锚位不足", r.subject_id,
                           critical=True)
            )
    for i in ranked:
        if i in protected_set:
            continue
        if len(keep_idx) < budget.max_images:
            keep_idx.append(i)
        else:
            r = imgs[i]
            drops.append(
                DroppedRef("image", r.role, r.uri,
                           f"超出 {budget.max_images} 图预算（优先级 "
                           f"{role_priority('image', r.role)}）", r.subject_id)
            )

    kept_images = [imgs[i] for i in sorted(keep_idx, key=ranked.index)]
    kept_videos = _rank_cut(vids, "video", budget.max_videos, drops)
    kept_audios = _rank_cut(auds, "audio", budget.max_audios, drops)
    return kept_images, kept_videos, kept_audios, drops


def _dedup(refs: Sequence, kind: RefKind, drops: list[DroppedRef]) -> list:
    out, seen = [], set()
    for r in refs:
        key = (r.role, r.uri)
        if key in seen:
            drops.append(
                DroppedRef(kind, r.role, r.uri, "与已有参考位重复",
                           getattr(r, "subject_id", None))
            )
            continue
        seen.add(key)
        out.append(r)
    return out


def _rank_cut(refs: Sequence, kind: RefKind, cap: int, drops: list[DroppedRef]) -> list:
    ranked = sorted(
        range(len(refs)),
        key=lambda i: (-role_priority(kind, refs[i].role), -refs[i].weight, i),
    )
    keep = ranked[:cap]
    for i in ranked[cap:]:
        r = refs[i]
        drops.append(DroppedRef(kind, r.role, r.uri, f"超出 {cap} 个{kind}预算"))
    return [refs[i] for i in sorted(keep, key=ranked.index)]


# ---------------------------------------------------------------- 报告


def _short_uri(uri: str, n: int = 30) -> str:
    return uri if len(uri) <= n else "…" + uri[-(n - 1):]


def slot_manifest(pack: RefPack, *, lang: str = "en") -> str:
    """人类可读的参考位占用报告。

    这份报告的唯一用途是回答「为什么这一镜脸漂了」：一眼看出身份锚到底进没进去、
    进去了几张、是不是被一堆 prop/lighting 图挤到了后面。
    """
    lines: list[str] = []
    used = pack.slots_used()
    lines.append(
        f"参考位占用 images {used['images']}/{_HARD_IMAGE_CAP} "
        f"videos {used['videos']}/{_HARD_VIDEO_CAP} "
        f"audios {used['audios']}/{_HARD_AUDIO_CAP}"
    )
    lines.append(f"{'SLOT':<9}{'ROLE':<13}{'SUBJ':<10}{'W':<6}URI / NOTE")
    lines.append("-" * 72)
    for kind, refs, cap in (
        ("image", pack.images, _HARD_IMAGE_CAP),
        ("video", pack.videos, _HARD_VIDEO_CAP),
        ("audio", pack.audios, _HARD_AUDIO_CAP),
    ):
        for i, r in enumerate(refs):
            subj = getattr(r, "subject_id", None) or "-"
            lines.append(
                f"{slot_tag(kind, i, lang):<9}{r.role:<13}{subj:<10}"
                f"{r.weight:<6.2f}{_short_uri(r.uri)}"
                + (f"  // {r.note}" if r.note else "")
            )
        for i in range(len(refs), cap):
            lines.append(f"{slot_tag(kind, i, lang):<9}{'(空)':<13}")

    anchored = sorted({r.subject_id for r in pack.images
                       if r.role == "identity" and r.subject_id})
    lines.append("-" * 72)
    lines.append("身份锚覆盖角色: " + (", ".join(anchored) if anchored else "无 —— 脸大概率会漂"))
    if not any(r.role == "first_frame" for r in pack.images):
        lines.append("提示: 无 first_frame 锚，本镜起点由模型自由发挥")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    from .schema import Appearance, Continuity, Scene, VoiceProfile

    def portrait(cid: str, view: str, w: float = 1.0) -> ImageRef:
        return ImageRef(role="identity", uri=f"s3://bible/{cid}/{view}.png",
                        weight=w, note=view, subject_id=cid)

    lin = CharacterBible(
        id="lin", name="林越", age_statement="虚构角色，设定年龄 27 岁",
        appearance=Appearance(face="鹅蛋脸", distinguishing="左眉一道疤"),
        voice=VoiceProfile(timbre_ref_uri="s3://voice/lin.wav"),
        portraits=[portrait("lin", "正脸特写"), portrait("lin", "半身 bust"),
                   ImageRef(role="wardrobe", uri="s3://bible/lin/coat.png", note="风衣")],
        turnaround=[portrait("lin", "全身 turnaround")],
        motion_refs=[VideoRef(role="motion", uri="s3://bible/lin/walk.mp4")],
    )
    qi = CharacterBible(
        id="qi", name="祁川", age_statement="虚构角色，设定年龄 34 岁",
        portraits=[portrait("qi", "正脸 face"), portrait("qi", "侧脸 profile")],
        turnaround=[portrait("qi", "全身 full body")],
    )

    noise = [
        ImageRef(role="lighting", uri=f"s3://lit/{i}.png") for i in range(4)
    ] + [
        ImageRef(role="prop", uri=f"s3://prop/{i}.png") for i in range(3)
    ] + [
        ImageRef(role="environment", uri="s3://env/bar.png"),
        ImageRef(role="style", uri="s3://style/noir.png"),
        ImageRef(role="composition", uri="s3://comp/sketch.png"),
    ]

    s1 = Shot(id="sh01", scene_id="sc01", index=0, shot_size=ShotSize.FS,
              refs=RefPack(images=[ImageRef(role="last_frame", uri="s3://out/sh01_last.png")]))
    s2 = Shot(id="sh02", scene_id="sc01", index=1, shot_size=ShotSize.CU,
              subject_ids=["lin", "qi"],
              continuity=Continuity(prev_shot_id="sh01", inherit_last_frame=True))
    sb = Storyboard(project="t", characters=[lin, qi],
                    scenes=[Scene(id="sc01", shots=[s1, s2])])

    drops: list[DroppedRef] = []
    pack = build_refpack(s2, sb, extra_images=noise, drops_out=drops)

    assert len(pack.images) == 9, pack.slots_used()
    assert pack.images[0].role == "first_frame"
    assert pack.images[0].uri == "s3://out/sh01_last.png"
    anchored = {r.subject_id for r in pack.images if r.role == "identity"}
    assert anchored == {"lin", "qi"}, anchored          # 两个角色都有脸
    # 近景优先取正脸
    lin_first = next(r for r in pack.images if r.role == "identity" and r.subject_id == "lin")
    assert "正脸" in lin_first.note, lin_first
    assert drops, "9 图预算下 13+ 张候选必须产生被裁清单"
    assert not any(d.critical for d in drops)
    dropped_roles = {d.role for d in drops}
    assert "lighting" in dropped_roles and "prop" in dropped_roles, dropped_roles
    assert "identity" not in dropped_roles, "身份锚被裁了"
    assert pack.audios and pack.audios[0].role == "voice_timbre"

    # 确定性：同输入两次必须完全一致
    pack2 = build_refpack(s2, sb, extra_images=noise)
    assert pack.model_dump() == pack2.model_dump()

    # 小预算 provider：2 图也要每人一张脸
    tiny = RefBudget(max_images=2, max_videos=0, max_audios=0, provider="tiny",
                     supports_first_frame=False, supports_last_frame=False)
    p_tiny = build_refpack(s2, sb, extra_images=noise, budget=tiny)
    assert len(p_tiny.images) == 2
    assert {r.subject_id for r in p_tiny.images} == {"lin", "qi"}, p_tiny.images

    # 纯 t2v：一个位都不给，全部进被裁清单且身份锚标 critical
    d0: list[DroppedRef] = []
    caps0 = Capabilities(name="t2v-only", kind="open", max_ref_images=0)
    p0 = build_refpack(s2, sb, extra_images=noise,
                       budget=RefBudget.from_capabilities(caps0), drops_out=d0)
    assert p0.slots_used() == {"images": 0, "videos": 0, "audios": 0}
    assert any(d.critical for d in d0), "0 图预算必须报 critical"

    # 能力声明 → 预算，且夹在 RefPack 契约上限内
    caps = Capabilities(name="big", kind="official", max_ref_images=99, max_ref_videos=9,
                        max_ref_audios=9, supports_first_frame=True,
                        supported_image_roles=frozenset({"identity", "first_frame"}))
    b = RefBudget.from_capabilities(caps)
    assert (b.max_images, b.max_videos, b.max_audios) == (9, 3, 3)
    assert b.accepts_image_role("identity") and not b.accepts_image_role("prop")

    # 配额公平性
    assert allocate_identity_quota(["a", "b"], 5) == {"a": 3, "b": 2}
    assert allocate_identity_quota(["a", "b", "c"], 2) == {"a": 1, "b": 1, "c": 0}
    assert allocate_identity_quota(["a", "a", "b"], 4) == {"a": 2, "b": 2}

    man = slot_manifest(pack)
    assert "@Image1" in man and "@Audio1" in man and "身份锚覆盖角色" in man
    assert "@图片1" in slot_manifest(pack, lang="zh")
    dirs = binding_directives(pack)
    assert dirs[0].startswith("@Image1 只控制")

    print(man)
    print("\n被裁参考位:")
    for d in drops:
        print(" ", d)
    print("\nrefpack 自测通过")


if __name__ == "__main__":
    _selftest()
