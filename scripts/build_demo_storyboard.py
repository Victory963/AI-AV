"""生成 demo 用的完整剧集分镜（5 分钟 × 30 个原子镜）。

这份分镜是 demo 视频的真实输入：路由、渲染、质检、拼接全都跑它，
而不是在 demo 里写死几个数字糊弄人。

角色全部虚构、显式设定为成年。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from longfilm.schema import (  # noqa: E402
    Appearance,
    AudioPlan,
    CameraMove,
    CharacterBible,
    Continuity,
    ContentRating,
    DeliverySpec,
    DialogueLine,
    EngineHint,
    Grade,
    ImageRef,
    LoRASpec,
    Scene,
    Shot,
    ShotSize,
    Storyboard,
    Transition,
    VoiceProfile,
)

STYLE = (
    "cinematic sci-fi, anamorphic lens flare, volumetric haze, "
    "teal and amber palette, 35mm film grain, shallow depth of field"
)
NEG = (
    "extra fingers, deformed hands, warped face, flickering, text, watermark, "
    "logo, low resolution, jpeg artifacts, duplicate limbs, morphing identity"
)


def cast() -> list[CharacterBible]:
    lin = CharacterBible(
        id="lin_lan",
        name="林岚",
        age_statement="虚构角色，设定年龄 32 岁，成年深空考古学家",
        persona="冷静、好奇、对失落文明有近乎固执的执着；说话简短，习惯自言自语记录观察。",
        appearance=Appearance(
            face="东亚女性面孔，高颧骨，单眼皮，左眉尾有一道浅疤",
            hair="黑色短发，右侧削短，发梢略乱",
            body="中等身材，肩线利落，站姿微微前倾",
            skin="冷调中性肤色，鼻梁有细小雀斑",
            wardrobe="深灰色考古服，左胸有橙色任务徽标，袖口卷起，腕上戴多功能终端",
            distinguishing="左眉尾浅疤 + 橙色任务徽标 + 腕部终端蓝光",
        ),
        voice=VoiceProfile(tts_engine="indextts2", emotion_default="calm", language="zh", speed=0.97),
        lora=LoRASpec(
            base_model="wan2.2-i2v-a14b",
            path="loras/lin_lan_v3.safetensors",
            trigger_word="linlan_char",
            strength=0.82,
            trained_steps=2400,
        ),
        negative_prompt="双眼皮, 长发, 白大褂",
        portraits=[
            ImageRef(role="identity", uri="assets/cast/lin_lan/front_neutral.png", weight=1.0,
                     subject_id="lin_lan", note="source=ai_generated 正面 平光 中性"),
            ImageRef(role="identity", uri="assets/cast/lin_lan/three_quarter_key.png", weight=0.9,
                     subject_id="lin_lan", note="source=ai_generated 3/4 侧 主光右 微笑"),
            ImageRef(role="identity", uri="assets/cast/lin_lan/profile_rim.png", weight=0.8,
                     subject_id="lin_lan", note="source=ai_generated 正侧 轮廓光"),
            ImageRef(role="identity", uri="assets/cast/lin_lan/closeup_eyes.png", weight=0.85,
                     subject_id="lin_lan", note="source=ai_generated 特写 眼部细节 用于近景镜"),
            ImageRef(role="wardrobe", uri="assets/cast/lin_lan/wardrobe_full.png", weight=0.7,
                     subject_id="lin_lan", note="source=ai_generated 全身服装"),
        ],
        turnaround=[
            ImageRef(role="identity", uri=f"assets/cast/lin_lan/turn_{a}.png", weight=0.6,
                     subject_id="lin_lan", note=f"source=ai_generated 转身 {a}°")
            for a in (0, 45, 90, 135, 180)
        ],
    )

    kai = CharacterBible(
        id="kai_k7",
        name="凯 K-7",
        age_statement="虚构角色，非人类仿生助理，外观设定为成年男性形象（相当于 40 岁）",
        persona="礼貌、精确、偶尔生硬的幽默；句尾习惯补一句数据置信度。",
        appearance=Appearance(
            face="中性成年男性面孔，哑光合成皮肤，虹膜为环形发光结构",
            hair="极短银灰色，发际线处可见接缝",
            body="偏高瘦，动作精确无冗余",
            skin="哑光浅灰合成材质，颈侧有散热格栅",
            wardrobe="白色分体外骨骼，关节处露出深色纤维束，胸口有青色指示环",
            distinguishing="环形发光虹膜 + 颈侧散热格栅 + 胸口青色指示环",
        ),
        voice=VoiceProfile(tts_engine="indextts2", emotion_default="neutral", language="zh", speed=1.03),
        negative_prompt="人类皮肤纹理, 胡须, 眼白",
        portraits=[
            ImageRef(role="identity", uri="assets/cast/kai_k7/front_neutral.png", weight=1.0,
                     subject_id="kai_k7", note="source=ai_generated 正面 平光"),
            ImageRef(role="identity", uri="assets/cast/kai_k7/three_quarter.png", weight=0.9,
                     subject_id="kai_k7", note="source=ai_generated 3/4 侧"),
            ImageRef(role="identity", uri="assets/cast/kai_k7/closeup_iris.png", weight=0.85,
                     subject_id="kai_k7", note="source=ai_generated 虹膜特写"),
            ImageRef(role="wardrobe", uri="assets/cast/kai_k7/exo_full.png", weight=0.7,
                     subject_id="kai_k7", note="source=ai_generated 外骨骼全身"),
        ],
    )
    return [lin, kai]


# (景别, 运镜, 时长, 焦段, 主体, 动作, 对白, 转场, 引擎倾向, 继承尾帧)
SC1 = [
    (ShotSize.ELS, CameraMove.DOLLY_IN, 12, 24, [], "废弃空间站残骸在气态巨行星光晕里缓慢自转", None, Transition.FADE_IN, EngineHint.HERO, False),
    (ShotSize.LS, CameraMove.PAN_R, 10, 24, ["lin_lan"], "登陆舱靠泊，舱门泄压喷出白雾，一个身影踏入通道", None, Transition.CUT, EngineHint.OFFICIAL, False),
    (ShotSize.MS, CameraMove.STEADICAM, 9, 35, ["lin_lan"], "她举着手电沿着失重的走廊前进，漂浮的碎屑从镜头前掠过", ("lin_lan", "站体自转周期 4 小时 17 分。比记录慢了三分之一。", "calm"), Transition.CUT, EngineHint.AUTO, True),
    (ShotSize.MCU, CameraMove.STATIC, 8, 50, ["lin_lan"], "她停下，抬头看向一面刻满未知符号的舱壁，手电光扫过", None, Transition.CUT, EngineHint.AUTO, True),
    (ShotSize.INSERT, CameraMove.DOLLY_IN, 6, 85, [], "舱壁符号特写，符号在光下泛起微弱磷光", None, Transition.MATCH_CUT, EngineHint.OPEN, False),
    (ShotSize.CU, CameraMove.STATIC, 7, 85, ["lin_lan"], "她的眼睛映着磷光，瞳孔收缩", ("lin_lan", "凯，看这个。它在回应光。", "curious"), Transition.CUT, EngineHint.AUTO, False),
    (ShotSize.MS, CameraMove.TRUCK_L, 9, 35, ["kai_k7"], "仿生助理沿走廊滑行而来，胸口指示环由青转黄", ("kai_k7", "检测到非随机磷光响应。置信度 0.87。", "neutral"), Transition.CUT, EngineHint.AUTO, False),
    (ShotSize.TWO, CameraMove.ORBIT_R, 11, 35, ["lin_lan", "kai_k7"], "两人并肩站在符号墙前，镜头绕行半圈", ("kai_k7", "建议不要触碰。上一支勘探队的记录在这里中断。", "neutral"), Transition.CUT, EngineHint.OFFICIAL, False),
    (ShotSize.OTS, CameraMove.STATIC, 8, 50, ["lin_lan", "kai_k7"], "越过凯的肩看林岚，她伸出戴着手套的手", ("lin_lan", "上一支队没带你。", "dry"), Transition.CUT, EngineHint.AUTO, True),
    (ShotSize.ECU, CameraMove.DOLLY_IN, 6, 100, ["lin_lan"], "手套指尖逼近符号，磷光骤然加强", None, Transition.CUT, EngineHint.HERO, False),
]

SC2 = [
    (ShotSize.FS, CameraMove.HANDHELD, 10, 28, ["lin_lan", "kai_k7"], "整面舱壁亮起，符号像活物一样重排，两人后退", None, Transition.CUT, EngineHint.HERO, True),
    (ShotSize.MCU, CameraMove.HANDHELD, 7, 50, ["kai_k7"], "凯的虹膜环快速旋转，指示环转红", ("kai_k7", "它在读取我。这不是刻痕，是接口。", "alarmed"), Transition.CUT, EngineHint.AUTO, False),
    (ShotSize.MS, CameraMove.WHIP, 5, 35, ["lin_lan"], "林岚猛转头看向走廊深处", None, Transition.CUT, EngineHint.OPEN, False),
    (ShotSize.LS, CameraMove.STATIC, 9, 24, [], "走廊尽头的隔离门正在逐层开启，蓝光从缝隙涌出", None, Transition.CUT, EngineHint.OFFICIAL, False),
    (ShotSize.MCU, CameraMove.DOLLY_OUT, 8, 50, ["lin_lan"], "她后退半步，手电熄灭，脸被蓝光照亮", ("lin_lan", "它不是在回应光。", "quiet"), Transition.CUT, EngineHint.AUTO, True),
    (ShotSize.CU, CameraMove.STATIC, 6, 85, ["lin_lan"], "特写：她轻声补完", ("lin_lan", "它是在回应我们。", "quiet"), Transition.CUT, EngineHint.HERO, True),
    (ShotSize.ELS, CameraMove.CRANE, 12, 24, [], "镜头拉升穿过残骸，整座空间站的外壳纹路依次点亮", None, Transition.DISSOLVE, EngineHint.HERO, False),
    (ShotSize.MS, CameraMove.STEADICAM, 10, 35, ["lin_lan", "kai_k7"], "两人穿过开启的隔离门，进入一个巨大的环形舱", None, Transition.CUT, EngineHint.OFFICIAL, False),
    (ShotSize.LS, CameraMove.TILT_U, 11, 24, ["lin_lan", "kai_k7"], "镜头上摇，环形舱顶部悬浮着一个缓慢旋转的几何体", ("kai_k7", "质量读数为零。它不在这里。", "neutral"), Transition.CUT, EngineHint.HERO, True),
    (ShotSize.MCU, CameraMove.STATIC, 8, 50, ["lin_lan"], "她仰头，橙色徽标被顶部的光染成白色", ("lin_lan", "记录时间。我们找到了。", "awed"), Transition.CUT, EngineHint.AUTO, False),
]

SC3 = [
    (ShotSize.INSERT, CameraMove.STATIC, 6, 85, [], "腕部终端屏幕：坐标与时间戳在跳动", None, Transition.CUT, EngineHint.OPEN, False),
    (ShotSize.MS, CameraMove.ORBIT_L, 10, 35, ["kai_k7"], "凯绕着几何体走，指示环在青红之间闪烁", ("kai_k7", "我需要提醒你，我的电源还有 41 分钟。", "dry"), Transition.CUT, EngineHint.AUTO, False),
    (ShotSize.MCU, CameraMove.STATIC, 7, 50, ["lin_lan"], "她没回头，只是笑了一下", ("lin_lan", "那就 41 分钟。", "warm"), Transition.CUT, EngineHint.AUTO, True),
    (ShotSize.TWO, CameraMove.DOLLY_IN, 11, 35, ["lin_lan", "kai_k7"], "两人站在几何体下方，镜头缓缓推进", None, Transition.CUT, EngineHint.OFFICIAL, False),
    (ShotSize.CU, CameraMove.STATIC, 7, 85, ["lin_lan"], "她伸手，几何体的一角朝她的方向缓缓转动", None, Transition.CUT, EngineHint.HERO, True),
    (ShotSize.ECU, CameraMove.DOLLY_IN, 5, 100, [], "几何体表面浮现出与舱壁相同的符号", None, Transition.MATCH_CUT, EngineHint.OPEN, False),
    (ShotSize.FS, CameraMove.PEDESTAL_U, 10, 28, ["lin_lan", "kai_k7"], "整个环形舱的光暗下去，只剩几何体与两人", None, Transition.CUT, EngineHint.HERO, False),
    (ShotSize.LS, CameraMove.STATIC, 9, 24, ["lin_lan"], "她独自走向几何体，凯留在原地", None, Transition.CUT, EngineHint.OFFICIAL, True),
    (ShotSize.MCU, CameraMove.DOLLY_IN, 8, 50, ["lin_lan"], "她闭上眼，光覆盖她的脸", None, Transition.CUT, EngineHint.HERO, True),
    (ShotSize.ELS, CameraMove.STATIC, 12, 24, [], "空间站在巨行星光晕中完全点亮，随后归于黑暗", None, Transition.FADE_OUT, EngineHint.HERO, False),
]

SCENES = [
    ("sc01", "接舷", "考古学家林岚与仿生助理凯登上失联的空间站，发现会回应光的符号墙。",
     "废弃空间站 · 失重走廊", "night", SC1,
     Grade(palette="teal shadows, amber practicals", contrast=1.08, saturation=0.92, temperature=-0.25)),
    ("sc02", "唤醒", "符号被激活，整座站体苏醒，隔离门开启，露出环形舱与悬浮几何体。",
     "废弃空间站 · 隔离门与环形舱", "night", SC2,
     Grade(palette="cyan bloom, deep blue shadows", contrast=1.15, saturation=1.05, temperature=-0.35)),
    ("sc03", "接触", "倒计时中的最后接触。",
     "环形舱 · 悬浮几何体下方", "night", SC3,
     Grade(palette="white key, near-monochrome", contrast=1.22, saturation=0.78, temperature=0.10)),
]


def build() -> Storyboard:
    chars = cast()
    scenes: list[Scene] = []
    prev_id: str | None = None

    for sc_id, title, synopsis, location, tod, rows, base_grade in SCENES:
        shots: list[Shot] = []
        for i, (size, move, dur, lens, subs, action, dlg, trans, hint, inherit) in enumerate(rows, 1):
            sid = f"{sc_id}_s{i:02d}"
            dialogue = []
            if dlg:
                spk, text, emo = dlg
                dialogue = [DialogueLine(speaker_id=spk, text=text, emotion=emo)]
            # 轴线：每个角色在本集里有固定的银幕方向（林岚朝右、凯朝左），
            # 正反打靠两人方向相反成立；同一角色的方向全片不翻转，否则观众会失去空间感。
            direction = "neutral"
            if len(subs) == 1:
                direction = {"lin_lan": "l2r", "kai_k7": "r2l"}.get(subs[0], "neutral")
            shots.append(
                Shot(
                    id=sid,
                    scene_id=sc_id,
                    index=i,
                    duration_s=float(dur),
                    shot_size=size,
                    camera_move=move,
                    lens_mm=lens,
                    aperture="f/2.0" if lens >= 50 else "f/2.8",
                    subject_ids=subs,
                    action=action,
                    environment=location,
                    lighting="practical rim light, volumetric haze, cold key from corridor",
                    mood="tense curiosity",
                    style=STYLE,
                    dialogue=dialogue,
                    transition_in=trans,
                    engine_hint=hint,
                    content_rating=ContentRating.PG13,
                    negative_prompt=NEG,
                    seed=1000 + len(shots) + len(scenes) * 100,
                    grade=base_grade.model_copy(),
                    continuity=Continuity(
                        prev_shot_id=prev_id,
                        inherit_last_frame=inherit and prev_id is not None,
                        emit_last_frame=True,
                        overlap_frames=6 if trans is Transition.OVERLAP_BLEND else 0,
                        screen_direction=direction,
                        match_on="符号磷光" if trans is Transition.MATCH_CUT else "",
                    ),
                )
            )
            prev_id = sid

        # 回填 next 指针
        for a, b in zip(shots, shots[1:]):
            a.continuity.next_shot_id = b.id

        scenes.append(
            Scene(id=sc_id, title=title, synopsis=synopsis, location=location,
                  time_of_day=tod, base_grade=base_grade, shots=shots)
        )

    sb = Storyboard(
        project="deep-archive",
        episode="ep01",
        logline="两名探索者登上失联的深空考古站，发现那里的符号不是留给后人看的，而是在等人回应。",
        target_duration_s=260.0,
        characters=chars,
        scenes=scenes,
        style_bible=STYLE,
        global_negative=NEG,
        audio=AudioPlan(audio_first=True, loudness_lufs=-16.0),
        delivery=DeliverySpec(
            resolution=(1280, 720),
            upscale_to=(1920, 1080),
            fps=24,
            codec="h264",
            crf=18,
            burn_subtitles=True,
            ai_disclosure=True,
            c2pa=True,
        ),
    )
    return sb


def main() -> None:
    sb = build()
    problems = sb.validate_continuity()
    out = ROOT / "configs" / "demo_episode.json"
    sb.save(out)

    shots = sb.all_shots()
    print(f"分镜已生成: {out}")
    print(f"  项目 {sb.project}/{sb.episode}  角色 {len(sb.characters)}  场景 {len(sb.scenes)}  镜头 {len(shots)}")
    print(f"  总时长 {sb.duration_s():.1f}s = {sb.duration_s()/60:.2f} 分钟  (目标 {sb.target_duration_s:.0f}s)")
    print(f"  平均镜长 {sb.duration_s()/len(shots):.1f}s  最长 {max(s.duration_s for s in shots):.0f}s")
    dlg = sum(len(s.dialogue) for s in shots)
    inh = sum(1 for s in shots if s.continuity.inherit_last_frame)
    print(f"  对白 {dlg} 条  继承尾帧 {inh} 镜  指纹样例 {shots[0].fingerprint()}")
    hints: dict[str, int] = {}
    for s in shots:
        hints[s.engine_hint.value] = hints.get(s.engine_hint.value, 0) + 1
    print(f"  引擎倾向分布 {hints}")
    if problems:
        print(f"  连续性问题 {len(problems)} 条:")
        for p in problems[:10]:
            print(f"    - {p}")
    else:
        print("  连续性检查: 通过")
    assert len(shots) == 30, f"期望 30 个原子镜，实际 {len(shots)}"
    assert not problems, f"连续性检查未通过: {problems[:3]}"


if __name__ == "__main__":
    main()
