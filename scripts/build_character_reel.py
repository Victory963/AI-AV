"""生成「一个虚构角色的 60 秒全身动作样片」分镜。

为什么是 18 个镜头而不是一条 60 秒：**没有任何引擎能一次出 60 秒**。
TI2V-5B 单条实用上限约 3.4 秒，商用闭源 API 也就 15~30 秒。60 秒必须靠
分镜系统接起来 —— 这正是续接链和身份锚存在的理由，也是这份分镜要证明的事。

镜头编排的三条规矩：
  1. 每 4 镜回一次锚（DriftBudget 的标定值）。回锚镜不继承上一镜尾帧，
     改从角色定妆图起，防止脸一路漂下去。
  2. 景别和运镜交替，不能 18 镜全是全景静止 —— 那是幻灯片不是片子。
  3. 动作按「能不能看出是同一个人在动」排：先站定再走动，
     大动作（转圈、跳）放在回锚镜之后，那时身份最稳。

    python scripts/build_character_reel.py            # 写出 configs/su_qing_reel.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from longfilm.charbible import CharacterBibleBuilder  # noqa: E402
from longfilm.schema import (  # noqa: E402
    AudioPlan,
    CameraMove,
    ContentRating,
    Continuity,
    DeliverySpec,
    EngineHint,
    Grade,
    Scene,
    Shot,
    ShotSize,
    Storyboard,
    Transition,
)

SHOT_S = 3.375          # 81 帧 @24fps，Wan 的 4k+1 约束
REANCHOR_EVERY = 4      # 与 chain.DriftBudget 的标定值一致
ROOM = ("明亮的家居摄影棚，浅木地板，米白色墙面，身后是大面积落地窗，"
        "窗外是虚化的绿色庭院，暖白色自然光从左前方斜射进来")


def character():
    return (CharacterBibleBuilder(
        "su_qing", "苏晴", age=24, occupation="生活方式频道主持人",
        persona="活泼直率，爱笑，说话带点跳脱的节奏；遇到喜欢的东西会忍不住原地转一圈。")
        .describe(
            face="圆润杏眼，双眼皮，眼尾微微下垂显得亲和；鼻梁小巧，笑起来左颊有一个梨涡",
            hair="蜜茶色及腰长卷发，中分，发梢自然外翘，右耳上方别一枚小星星发夹",
            body="身高一米六二，体型匀称偏纤细，站姿轻快，走路时手臂摆幅偏大",
            skin="暖调白皙，脸颊有淡淡的自然红晕，锁骨下方有一颗小痣",
            wardrobe="米白色针织开衫配浅樱粉连衣裙，裙长过膝，脚踩白色小皮鞋，手腕戴一条细银链",
            distinguishing="星星发夹 + 左颊梨涡 + 蜜茶色外翘发梢")
        .with_voice(tts_engine="indextts2", tts_voice_id="zh_female_sweet_01",
                    speed=1.04, emotion_default="happy", language="zh")
        .with_negative("暴露服装, 成人内容, 过度妆容, 未成年外观, 儿童, 多余的手指, 畸形的手")
        .build())


# (景别, 运镜, 动作, 这一镜要证明什么)
BEATS: tuple[tuple[ShotSize, CameraMove, str, str], ...] = (
    (ShotSize.FS, CameraMove.STATIC,
     "站在窗前，双手自然垂放，看向镜头轻轻挥手打招呼，随后露出微笑", "身份基准镜：全身、正面、平光"),
    (ShotSize.FS, CameraMove.TRUCK_L,
     "从画面右侧向左走过整个房间，步伐轻快，裙摆随走动摆动，手臂自然摆动", "走动时的身形与步态"),
    (ShotSize.MS, CameraMove.STATIC,
     "停下脚步转身回头看向镜头，长发随转身甩动，抬手把耳边的头发别到耳后", "转身甩发：最容易崩的中间帧"),
    (ShotSize.MCU, CameraMove.DOLLY_IN,
     "面向镜头说话，表情从平静转为开心，眨眼，微微点头", "面部细节与表情过渡"),
    (ShotSize.FS, CameraMove.STATIC,
     "原地张开双臂转一整圈，裙摆飞起，转完停稳后笑着看向镜头", "大动作：整圈旋转，回锚后做"),
    (ShotSize.MLS, CameraMove.ORBIT_L,
     "双手在身前交握站立，身体随镜头环绕微微转动，视线跟着镜头走", "环绕镜头下的身形一致性"),
    (ShotSize.FS, CameraMove.PEDESTAL_D,
     "蹲下身去捡起地上的一本书，翻看一眼，再站起来抱在胸前", "蹲下起身：全身关节的连续运动"),
    (ShotSize.MS, CameraMove.STATIC,
     "抱着书走到窗边，侧身靠着窗框，低头翻了一页，抬头看向窗外", "走位加静态姿势的衔接"),
    (ShotSize.FS, CameraMove.STATIC,
     "站在窗前伸了个懒腰，双臂举过头顶再放下，肩膀放松地转了转", "伸展：躯干与四肢的大幅动作，回锚后做"),
    (ShotSize.MLS, CameraMove.TRUCK_R,
     "沿着窗边向右缓步走，一边走一边转头看向镜头，头发随步伐轻晃", "横移跟拍时的脸部稳定性"),
    (ShotSize.MCU, CameraMove.STATIC,
     "停下来对着镜头认真说话，双手在胸前比划了一个小小的手势", "手部动作：最容易画坏的部位"),
    (ShotSize.FS, CameraMove.DOLLY_OUT,
     "向后退两步，双手背在身后，歪头看向镜头笑", "后退运动 + 镜头拉开的透视变化"),
    (ShotSize.FS, CameraMove.STATIC,
     "原地轻轻跳了一下，落地后稳住，双手在身侧张开，开心地笑", "跳跃：重心变化，回锚后做"),
    (ShotSize.MS, CameraMove.PAN_R,
     "坐到窗边的浅色单人沙发上，双腿并拢侧放，双手放在膝上，转头看镜头", "坐下：从站姿到坐姿的过渡"),
    (ShotSize.MCU, CameraMove.STATIC,
     "坐姿，双手撑着下巴，眨眼后露出笑容，轻轻点头", "坐姿特写的表情细节"),
    (ShotSize.FS, CameraMove.PEDESTAL_U,
     "从沙发上站起来，整理了一下裙摆，向镜头走近两步", "起身走近：纵深方向的运动", ),
    (ShotSize.MLS, CameraMove.STEADICAM,
     "跟着她在房间里走动，她边走边回头说话，最后停在房间中央", "跟拍：镜头与人物同时运动"),
    (ShotSize.FS, CameraMove.STATIC,
     "站定在房间中央，双手在身前交叠，对镜头挥手告别，笑着定格", "收尾镜：回到与开场同构的站姿"),
)


def build() -> Storyboard:
    char = character()
    shots: list[Shot] = []
    for i, (size, move, action, why) in enumerate(BEATS):
        reanchor = i % REANCHOR_EVERY == 0
        shots.append(Shot(
            id=f"sq_s{i + 1:02d}",
            scene_id="sc01",
            index=i,
            duration_s=SHOT_S,
            shot_size=size,
            camera_move=move,
            lens_mm=35 if size in (ShotSize.FS, ShotSize.MLS) else 50 if size is ShotSize.MS else 85,
            aperture="f/2.8",
            fps=24,
            subject_ids=["su_qing"],
            action=action,
            environment=ROOM,
            lighting="暖白色自然光从左前方斜射，右侧有柔和补光，整体高调、低对比",
            mood="轻松、明亮、亲切",
            style="真实电影感，浅景深，皮肤质感自然，不做磨皮",
            # 每 4 镜回一次锚：不继承上一镜尾帧，改从定妆图起
            continuity=Continuity(
                prev_shot_id=None if reanchor or i == 0 else f"sq_s{i:02d}",
                inherit_last_frame=not reanchor,
                emit_last_frame=True,
                screen_direction="l2r",
                match_on=why,
            ),
            grade=Grade(look="暖白高调", lut_uri=None),
            transition_in=Transition.CUT if i == 0 else Transition.DISSOLVE,
            content_rating=ContentRating.G,
            engine_hint=EngineHint.OPEN,      # 只走自建通道
            seed=90_000 + i,
            negative_prompt="暴露服装, 成人内容, 多余的手指, 畸形的手, 面部闪烁, 画面抖动",
        ))

    scene = Scene(
        id="sc01", title="全身动作样片", synopsis="一个虚构角色在明亮家居摄影棚里的一分钟活动",
        location="家居摄影棚", time_of_day="day",
        base_grade=Grade(look="暖白高调"), shots=shots,
    )
    return Storyboard(
        project="character-reel", episode="su_qing_60s",
        logline="虚构角色苏晴的 60 秒全身动作样片：走动、转身、旋转、蹲起、跳跃、坐下、跟拍。",
        target_duration_s=len(BEATS) * SHOT_S,
        characters=[char], scenes=[scene],
        style_bible="明亮家居摄影棚，暖白高调，真实电影感，浅景深，35–85mm",
        global_negative="暴露服装, 成人内容, 未成年外观, 真人肖像, 多余的手指, 畸形的手",
        audio=AudioPlan(audio_first=False, loudness_lufs=-16.0),
        delivery=DeliverySpec(resolution=(1280, 704), fps=24, upscale_to=None,
                              burn_subtitles=False, ai_disclosure=True, c2pa=True),
    )


def main() -> int:
    sb = build()
    out = ROOT / "configs" / "su_qing_reel.json"
    sb.save(out)
    total = sum(s.duration_s for s in sb.all_shots())
    print(f"角色：{sb.characters[0].name}（{sb.characters[0].age_statement}）")
    print(f"分镜：{len(sb.all_shots())} 镜 × {SHOT_S}s = {total:.1f}s")
    anchors = [s.id for s in sb.all_shots() if not s.continuity.inherit_last_frame]
    print(f"回锚镜（不继承尾帧）：{', '.join(anchors)}")
    problems = sb.validate_continuity()
    print("连续性检查：", problems or "通过")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
