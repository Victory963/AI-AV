

====================================================================================================
# [audio-lipsync-tts] 数字人声音与口型完整技术栈（TTS/声音克隆 → 音频驱动数字人视频 → 后置唇形对齐 → Foley/音乐），面向 2-8 分钟长片产线

## IndexTTS-2 / IndexTTS-2.5  (bilibili)
- 类别/成熟度: open-weights / **production** | 许可: bilibili Model Use License Agreement（商用需单独授权） | 成本: 单卡消费级即可（4090 实测 RTF≈0.21）；具体显存未在 README 标注
- 是什么: 自回归零样本 TTS，唯一同时做到「情感与音色解耦」+「精确时长控制」的开源中文 TTS；T2S(Transformer) + S2M(非自回归) 两段式，2.5 把 U-DiT 主干换成 Zipformer。
- 关键事实:
    * IndexTTS-2 发布 2025-09-08（arXiv 2506.21619）；IndexTTS-2.5 发布 2026-08-10（arXiv 2601.03888，v1 2026-01-07 / v5 2026-08-11）
    * 2.5 参数量 0.8B
    * RTF 在 RTX 4090 + kv_cache：BF16 0.2065 / FP32 0.2060（约 5 倍实时）
    * 2.5 相对 2.0 的 RTF 提升 2.28 倍，WER 与说话人相似度持平
    * 语言：中/英/日/西/阿（2.5）；arXiv 摘要只列中英日西，README 多列阿拉伯语
    * 时长控制 duration_factor 0.5x–2.0x；情感控制三通道：情感参考音频 / 8 维情感向量 / 文本情感（Qwen3 微调的软指令）+ emo_alpha 0.0–1.0
    * 许可证：bilibili Model Use License Agreement，商用需邮件 indexspeech@bilibili.com 申请（非标准开源许可）
- 长片作用: 长片对白轨的首选工位。duration_factor 精确时长控制是长片的刚需——分镜表先定每句话占几秒，TTS 直接按秒数出音频，再驱动画面，避免「音频长度和镜头长度对不上」这个长片最大的返工源。中文情感表现力在开源里第一梯队，但许可证不是 Apache/MIT，商用前必须走 bilibili 授权，这是排产前要先解决的法务项。
- 来源: https://github.com/index-tts/index-tts https://arxiv.org/abs/2601.03888 https://arxiv.org/pdf/2506.21619

## CosyVoice 3 (Fun-CosyVoice3-0.5B-2512)  (阿里通义 FunAudioLLM)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: 0.5B 级别，单张 8–12GB 消费卡可跑；官方未给显存表
- 是什么: 0.5B 的 LLM-based 流式 TTS，文本流入 + 音频流出双向流式，支持自然语言 instruct 控制（方言/情感/语速/音量）与细粒度呼吸标记、拼音/CMU 音素纠音。
- 关键事实:
    * 版本 Fun-CosyVoice3-0.5B-2512，2025 年 12 月发布
    * 参数量 0.5B
    * 首包延迟低至 150ms
    * 9 种主要语言 + 18 种以上中文方言/口音（粤语、闽南、四川等）
    * 许可证 Apache 2.0（可直接商用，无附加条件）
- 长片作用: 长片产线的「可商用兜底音轨」。IndexTTS-2.5 如果授权谈不下来，CosyVoice 3 是唯一 Apache 2.0 且中文情感/方言能打的替代品。150ms 首包让它也能当交互式预演（导演口播临时改词即时听）。缺点是没有 IndexTTS 那种显式 duration_factor，镜头时长对齐要靠后期拉伸或重采样。
- 来源: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 https://github.com/QwenAudio/CosyVoice

## GPT-SoVITS (v4 / v2Pro / v2ProPlus)  (RVC-Boss 社区)
- 类别/成熟度: open-weights / **production** | 许可: MIT | 成本: 消费级 6–12GB 可训可推，整合包一键部署
- 是什么: 少样本微调 TTS+VC，1 分钟音频即可训练出专属角色音色；v3 有金属声，v4 修复并输出原生 48kHz。
- 关键事实:
    * 最新 release 20250606v2pro（2025-06-06）；20250422v4（2025-04-22）；20250228v3（2025-02-28）
    * v4 输出原生 48kHz
    * 声音克隆样本：约 1 分钟即可微调出稳定角色音色（零样本亦支持，质量低于微调）
    * 语言：中、英、日、韩、粤
    * 许可证 MIT（代码与模型均可商用）
    * 截至检索未见 v5
- 长片作用: 角色音色库工位。长片有固定主角，最稳的做法不是零样本克隆而是每个角色微调一份 GPT-SoVITS 权重，保证 2-8 分钟全片音色不漂移。MIT 许可让它是唯一「训练 + 商用」全无障碍的路线。情感表现力弱于 IndexTTS-2.5，实践中常用组合：主角长对白用微调 GPT-SoVITS 保一致性，强情绪镜头用 IndexTTS-2.5 情感参考音频补。
- 来源: https://github.com/RVC-Boss/GPT-SoVITS/releases https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE https://huggingface.co/lj1995/GPT-SoVITS

## Fish-Speech / OpenAudio S2 Pro  (Fish Audio)
- 类别/成熟度: open-weights / **usable** | 许可: Fish Audio Research License（非商用） | 成本: 4B+0.4B，推理需 ~16GB 级别显存（官方未给表）
- 是什么: 双自回归架构（4B 慢解码器 + 400M 快解码器）+ RVQ 10 码本，80+ 语言，TTFA ~100ms。
- 关键事实:
    * S2 Pro：4B（Slow AR）+ 400M（Fast AR），RVQ 10 codebooks
    * 训练数据 1000 万小时以上，覆盖 80+ 语言（Tier1：中/英/日）
    * RTF 0.195（单张 H200）；TTFA ≈100ms；RTF<0.5 时吞吐 3000+ acoustic tokens/s
    * 声音克隆参考样本 10–30 秒
    * 主仓 LICENSE = FISH AUDIO RESEARCH LICENSE：明文禁止商用，商用需另签书面协议；分发须标注 "Built with Fish Audio"；禁止用其输出改进其他基础生成模型
- 长片作用: 技术指标好看但对商业长片产线是死路——许可证明文禁商用。只能当技术参照或内部评测基线。注意：多篇第三方 2026 年文章仍称「S1/S2 Pro 为 MIT，可自托管」，与仓库 LICENSE 原文直接冲突，以 LICENSE 为准。
- 来源: https://github.com/fishaudio/fish-speech https://raw.githubusercontent.com/fishaudio/fish-speech/main/LICENSE https://fish.audio/blog/introducing-s1/

## F5-TTS  (上海交大 / SWivid)
- 类别/成熟度: open-weights / **usable** | 许可: 代码 MIT / 权重 CC-BY-NC 4.0（非商用） | 成本: 小模型，8GB 级消费卡可跑；RTF 0.04 意味着 25 倍实时
- 是什么: Flow-matching 非自回归 TTS，长文本韵律与句间停顿自然，推理极快。
- 关键事实:
    * v1 base 模型发布 2025-03-12；初版 2024-10-08 上 HF
    * RTF 0.0394（L20，16 NFE，并发 2）/ 0.0402（batch=1）；TensorRT-LLM 离线 PyTorch 模式 0.1467
    * 训练数据 Emilia + WenetSpeech4TTS + LibriTTS + LJSpeech
    * 代码 MIT，但预训练权重 CC-BY-NC（因 Emilia 数据集为野采数据）
    * 截至检索未见 F5-TTS v2 或后继
- 长片作用: 速度是全场最快之一（RTF 0.04），批量生成 8 分钟对白几乎瞬时——但权重 CC-BY-NC 卡死商用。只有自行用可商用数据重训权重才能进产线，那是几万卡时的工程量，不建议。
- 来源: https://github.com/SWivid/F5-TTS https://arxiv.org/pdf/2410.06885

## MaskGCT  (香港中文大学(深圳) / Amphion (open-mmlab))
- 类别/成熟度: open-weights / **research** | 许可: CC-BY-NC-4.0（非商用） | 成本: 研究级，未给生产显存表
- 是什么: 完全非自回归的掩码生成码本 Transformer TTS，不需要文本-语音显式对齐，也不需要音素级时长预测。
- 关键事实:
    * 发布 2024-10-19
    * 训练数据 Emilia，10 万小时（英文 5 万 + 中文 5 万）
    * HF 权重许可证 CC-BY-NC-4.0（非商用）
- 长片作用: 已被 2025-2026 的 IndexTTS-2.5 / CosyVoice 3 在中文表现力和时长可控性上全面超越，且非商用许可。在 2-8 分钟产线里不占任何工位，仅作为学术基线保留。
- 来源: https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct https://arxiv.org/pdf/2409.00750 https://huggingface.co/amphion/MaskGCT

## ChatTTS  (2noise 社区)
- 类别/成熟度: open-weights / **usable** | 许可: 未核实（见 uncertainClaims） | 成本: 小模型，6GB 级可跑
- 是什么: 面向对话场景的中英 TTS，主打口语化韵律与笑声/停顿等副语言标记。
- 关键事实:
    * 主流行期为 2024 中–2025 中；2026 年第三方评测仍把它列为「对话式中文」选项
    * 提供说话风格与情感的细粒度控制标记
    * 本次检索未能核实其 2026 年维护状态与权重许可证原文
- 长片作用: 在长片产线里最多顶「群演/画外闲聊」这种不要求音色一致性的工位。它的定位是对话口语化而非角色音色复刻，做不了「全片主角同一把嗓子」这件事，因此不适合当主对白引擎。
- 来源: https://github.com/2noise/ChatTTS https://www.codesota.com/guides/tts-models

## XTTS-v2 (Coqui)  (Coqui Inc.（已关停）)
- 类别/成熟度: open-weights / **research** | 许可: CPML 1.0.0（非商用，且无法再购买商用授权） | 成本: 8GB 级可跑
- 是什么: 自回归多语言零样本克隆 TTS，2023-2024 年的社区默认选择。
- 关键事实:
    * 权重许可证 Coqui Public Model License (CPML) 1.0.0：仅限非商用
    * Coqui Inc. 已于 2024 年 1 月关停，商用授权渠道不复存在（历史价约 365 USD/年，适用营收/融资 <100 万美元的公司）
    * 社区分叉 idiap/coqui-ai-TTS 为 MPL-2.0，但那是代码许可，不改变权重的 CPML 约束
    * 架构为自回归，2026 年同样几秒参考音频下的克隆稳定性已落后
- 长片作用: 技术与法务双重死路：许可证禁商用且卖方主体已消失，无任何合规化路径。不要进产线，历史工程里如有依赖应迁移到 CosyVoice 3 或 GPT-SoVITS。
- 来源: https://huggingface.co/coqui/XTTS-v2 https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt https://github.com/coqui-ai/TTS/discussions/4304

## ElevenLabs Eleven v3  (ElevenLabs)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: $100/百万字符（v3）；8 分钟中文对白约 1500-2500 字，单集成本 <$0.3，成本不是问题
- 是什么: 闭源表现力 TTS，行内音频标签（[whispers]/[laughs]/[excited]）直接控制演绎。
- 关键事实:
    * API 定价 $0.10 / 1000 字符 = $100 / 百万字符（v3 与 Multilingual v2）；Flash / Turbo / v3 Conversational 为 $0.05 / 1000 字符 = $50 / 百万字符
    * 支持 70+ 语言（含中文）
    * 官方定位：叙事、有声书、角色配音；明确不为实时场景设计
    * 音频标签为英文语法标记
- 长片作用: 闭源对照组的表现力上限。在长片里适合做「情绪高点镜头」的少量补拍配音，但中文音色库深度与中文语气词处理弱于 MiniMax，且不能本地部署 = 角色音色资产不在自己手里。作为产线主引擎不划算（音色托管在别人服务器上，断供即全片报废）。
- 来源: https://elevenlabs.io/docs/overview/models https://elevenlabs.io/pricing

## MiniMax Speech 2.8 / 2.6  (MiniMax（稀宇科技）)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: $60–100/百万字符 + $1.50/克隆音色
- 是什么: 闭源中文优先 TTS，2.8 加入原生拟声标签（笑/咳/叹/喷嚏）与降噪流水线重构；2.6 引入 Fluent LoRA 克隆。
- 关键事实:
    * Speech 2.8 发布 2026-01-23；Speech 2.6 发布 2025-10-30（官方已标 legacy）
    * 定价：HD $100 / 百万字符；Turbo $60 / 百万字符
    * 端到端延迟 <250ms（Turbo）
    * 声音克隆：约 10 秒音频；快速克隆 $1.50/音色；Voice Design 文本造音 $3.00/音色（另 $30/百万字符预览音频）
    * 40+ 语言（language_boost 参数）
    * 情感参数：happy/sad/angry/fearful/disgusted/surprised/calm/fluent/whisper；音高、强度、音色 −100~100 可调
- 长片作用: 中文闭源对照的性价比标杆：价格是 ElevenLabs 的 0.6–1.0 倍但中文语气与拟声标签更贴国内耳朵，10 秒即可克隆。适合长片前期快速试音、对白 demo；正式排产仍建议落到本地 IndexTTS-2.5/GPT-SoVITS，因为 2-8 分钟成片会反复改词重出，闭源 API 的返工成本与审核风险都在外部。注意平台无同意验证机制，克隆授权责任在使用方——本项目角色虚构、不碰真人肖像的策略正好规避这一点。
- 来源: https://platform.minimax.io/docs/guides/speech-voice-clone https://invideo.io/blog/minimax-ai-voice-models/ https://www.together.ai/models/minimax-speech-2-6-turbo

## OmniHuman-1.5  (字节跳动 / BytePlus)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: $0.12–0.16 / 秒；8 分钟全片若全走 OmniHuman ≈ $58–77（还需分 16–32 段）
- 是什么: 闭源电影级数字人 API：单图 + 语音 + 可选文本提示 → 原生 1080p 说话视频，支持多人场景分轨配音、镜头运动与手势控制。
- 关键事实:
    * 720p 支持最长 60 秒音频；1080p 支持最长 30 秒
    * 定价：BytePlus 官方页标 $0.12/秒；fal 文档标 $0.16/秒（30 秒 = $4.80）
    * 输入：图片 JPEG/PNG + 音频 MP3/WAV/M4A，均需公网可访问 URL
    * 权重不开放，仅 API；无本地部署、无 LoRA
    * 部分第三方接入方把上限压到 35 秒，并提示超过 15 秒质量下降
- 长片作用: 当前闭源人物镜头质量天花板，但 30–60 秒硬上限意味着 2-8 分钟必须切段，切段处的身份/光照/机位连续性得自己缝——这正是你已有分镜系统要解决的问题。和 Seedance 2.0 的约束同构：可以当「单镜画面发生器」，不能当长片引擎。$0.12 与 $0.16 两个价格口径不一致，签约前需向 BytePlus 核实。
- 来源: https://www.byteplus.com/en/product/OmniHuman https://fal.ai/learn/devs/omnihuman-15-user-guide https://omnihuman-lab.github.io/v1_5/

## LongCat-Video-Avatar-1.5  (美团 LongCat（meituan-longcat），项目页挂在 MeiGen-AI)
- 类别/成熟度: open-weights / **production** | 许可: MIT | 成本: INT8 量化可降；具体数值官方未公布（需实测）
- 是什么: 基于 LongCat-Video 基础模型的音频驱动人物视频框架，原生支持 音频-文本→视频(AT2V)、音频-文本-图像→视频(ATI2V)、视频续写(Video Continuation)，单流与多流音频通吃。
- 关键事实:
    * 发布 2026-05-21（arXiv 2605.26486）；前代 LongCat-Video-Avatar 2025-12-16
    * 许可证 MIT（权重明确 MIT）
    * 分辨率 480P / 720P，--resolution 可控
    * 音频编码器由 Wav2Vec2 换成 Whisper-Large，唇形同步更准
    * DMD2 步蒸馏，8 步推理
    * 支持 INT8 量化 DiT 降显存
    * 双音频模式支持 merge 与 concat；泛化到风格化域（动漫、动物）
    * 官方 README 未给具体显存数值与最长时长数值
- 长片作用: 开源侧目前最该押注的人物镜头主力工位。三点决定性：①MIT，商用零障碍；②8 步蒸馏，单镜成本落到可批量的水平；③原生「视频续写」+ 多流音频，这是长片拼接与多人对白的原生能力，而不是外挂补丁。在你的 2-8 分钟产线里，它应该占「有对白的人物镜头」80% 的产能，Seedance 只用于它做不好的大场面/复杂运镜。
- 来源: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5 https://github.com/meituan-longcat/LongCat-Video https://arxiv.org/pdf/2605.26486

## InfiniteTalk  (MeiGen-AI)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: FP8 + 低显存模式下社区实测 12GB 可跑（来自 ComfyUI 整合包教程，非官方数值）
- 是什么: 稀疏帧视频配音框架：给一段已有视频 + 新音频，重新合成唇形、头部运动、体态、表情全对齐的新视频（V2V）；也可退化为 图+音→视频（I2V）。
- 关键事实:
    * 发布 2025-08-19（技术报告 + 权重 + 代码 + Gradio + ComfyUI）
    * 基座 Wan2.1-I2V-14B-480P，音频编码 chinese-wav2vec2-base
    * 分辨率 480P / 720P
    * 默认最长 1000 帧 ≈ 40 秒（--max_frame_num 可调），并支持分段流式实现无限长
    * 许可证 Apache 2.0
    * 低显存模式 --num_persistent_param_in_dit 0；FP8 量化；TeaCache / APG 加速；FusionX LoRA 8 步 / Lightx2v 4 步
    * 提供多人专用权重
- 长片作用: 整条音频先行链路的关键胶水工位，价值被严重低估。它的 V2V 模式让你可以：先用 Seedance 2.0 / LTX / Wan 出「画面好但对白声不是自己音色」的镜头，再用 InfiniteTalk 把自有 TTS 对白轨贴回去，同时重写体态和表情。这一步绕开了 Seedance「自带对白声 + 音频参考 ≤15 秒」的硬约束，并让全片音色由你的 TTS 统一控制。Apache 2.0 + 无限长分段是长片刚需。
- 来源: https://github.com/MeiGen-AI/InfiniteTalk https://huggingface.co/MeiGen-AI/InfiniteTalk https://raw.githubusercontent.com/MeiGen-AI/InfiniteTalk/main/README.md

## MultiTalk (MeiGen)  (MeiGen-AI)
- 类别/成熟度: open-weights / **usable** | 许可: 未在本次检索中核实（仓库为 MeiGen-AI 系，同组 InfiniteTalk 为 Apache 2.0） | 成本: Wan2.1-14B 级别
- 是什么: 多流音频驱动的多人对话视频生成：多路音频 + 参考图 + 提示词 → 多人交互视频，各自唇形对上各自音轨。
- 关键事实:
    * NeurIPS 2025 收录
    * 长视频能力被第三方描述为约 15 秒（未在官方 README 核实）
    * 已被同组 InfiniteTalk（2025-08-19）与 LongCat-Video-Avatar-1.5（2026-05-21）在长度与稳定性上超越
- 长片作用: 多人同框对话这个工位现在应该直接用 LongCat-Video-Avatar-1.5 的多流音频或 InfiniteTalk 的多人权重，MultiTalk 本身只剩历史价值。15 秒级上限对 2-8 分钟长片的对话戏（往往一场戏 30-90 秒）不够用。
- 来源: https://github.com/MeiGen-AI/MultiTalk

## HunyuanVideo-Avatar  (腾讯混元 + 腾讯音乐)
- 类别/成熟度: open-weights / **usable** | 许可: Tencent Hunyuan Community License（地域排除 EU / UK / 韩国） | 成本: 社区优化后 10GB 可跑 15 秒片段
- 是什么: 基于 HunyuanVideo 的音频驱动人物动画，自动识别场景与情绪，生成说话/唱歌视频，支持肖像/半身/全身多尺度。
- 关键事实:
    * 发布 2025-05-28（arXiv 2505.20156）
    * 开源的是单角色模式；官方说明支持最长 14 秒音频
    * 社区优化（DeepBeepMeep）后 10GB 显存即可生成 15 秒语音/歌曲驱动视频（原需 80GB 或 24GB），支持 TeaCache
    * checkpoint 路径出现 704 / 720p；单帧采样示例 --sample-n-frames 129
    * 许可证为 Tencent Hunyuan Community License 体系，其 Territory 定义排除欧盟、英国、韩国
    * 本次检索未见 2026 年更新
- 长片作用: 14 秒上限 + 2026 年停更，使它在长片产线里只适合做「短镜头补拍」或低显存备份路线。地域许可排除欧盟英国韩国，如果成片要在这些地区发行需要法务复核。相比 LongCat-1.5（MIT、8 步、多流音频）没有保留理由，除非你的机器只有 10-12GB 显存。
- 来源: https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar https://huggingface.co/tencent/HunyuanVideo-Avatar https://arxiv.org/html/2505.20156v1

## EchoMimicV3 / EchoMimicV3-Flash-Pro  (蚂蚁集团 (antgroup))
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 12GB（Flash-Pro）～16GB；8 步推理
- 是什么: 1.3B 参数统一多模态多任务人体动画框架，音频 + 文本 + 姿态多条件统一，AAAI 2026。
- 关键事实:
    * 论文 2025-07-08（arXiv 2507.03905），代码与权重 2025-08-08，AAAI 2026 录用 2025-11-09
    * 最新更新 2026-01-22：Flash 版本支持 8 步高质量生成、无需人脸 mask
    * 参数量 1.3B
    * Flash-Pro 显存需求 12GB；实测 GPU 覆盖 V100(16G)、RTX4090D(24G)、A100(80G)；ComfyUI 下 16GB 可跑
    * 最高分辨率 768×768
    * 默认分段 138 帧，超过需分段拼接（partial_video_length 可调降显存）
    * 许可证 Apache 2.0，作者声明不对生成内容主张权利
- 长片作用: 低显存备份工位与半身镜头的性价比选择。768×768 上限和 138 帧分段对 2-8 分钟成片意味着大量拼接，画质上限也低于 LongCat-1.5 / InfiniteTalk 的 720P。在单机 12-16GB 的开发/预览环境里做分镜草稿（先看口型和节奏对不对），正式渲染再换 LongCat，是合理的两档策略。
- 来源: https://github.com/antgroup/echomimic_v3 https://arxiv.org/abs/2507.03905 https://huggingface.co/BadToBest/EchoMimicV3

## EchoMimicV2  (蚂蚁集团 (antgroup))
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0（同 antgroup EchoMimic 系列） | 成本: 消费级可跑
- 是什么: 半身（semi-body）音频驱动人体动画，CVPR 2025，强调手势与上半身动作的简化控制。
- 关键事实:
    * CVPR 2025 录用
    * 定位为「半身」动画，含手部动作控制
    * 已被同组 EchoMimicV3（1.3B 统一框架，2025-08 权重、2026-01 Flash 更新）取代
- 长片作用: 已被 V3 全面覆盖，在新产线里不应再占工位。仅当你已有针对 V2 的姿态数据流水线时才值得保留。
- 来源: https://github.com/antgroup/echomimic_v2 https://antgroup.github.io/ai/echomimic_v2/

## Wan2.2-S2V-14B  (阿里通义万相)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 单卡 80GB 起（A100/H100 级），消费卡需 FP8/社区量化
- 是什么: 14B 音频驱动「电影级」视频生成：参考图 + 音频 + 可选文本 + 可选姿态引导，不止说话头，目标是人物互动、身体运动和运镜。
- 关键事实:
    * 发布 2025-08-26（arXiv 2508.18621）
    * 许可证 Apache 2.0
    * 分辨率 480P / 720P，输出宽高比跟随参考图
    * 扩散 3D VAE 架构，Wav2Vec 注入音频，FramePack 压缩保持运动一致性
    * 单卡本地推理至少需 80GB 显存（官方 README）
    * 视频长度自动跟随输入音频长度，官方未给硬上限
    * 支持与 CosyVoice 串联做 TTS→S2V 任务链
- 长片作用: 「带运镜的人物镜头」工位——它比纯说话头模型更愿意动镜头和动身体，这正是接近 Seedance 单镜观感所需要的。但 80GB 单卡门槛把它挡在消费级产线外，实际要靠社区 FP8/GGUF 量化。与 LongCat-1.5 相比，优势是运镜与全身表现，劣势是显存与速度。适合做少量「关键镜头」而不是全片主力。
- 来源: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B https://github.com/Wan-Video/Wan2.2 https://arxiv.org/pdf/2508.18621

## LTX-2 / LTX-2.3 / LTX-2.5  (Lightricks)
- 类别/成熟度: open-weights / **production** | 许可: LTX-2 Community License Agreement（开放权重，商用有条件） | 成本: distilled/fp8 可下探消费卡；22B full 建议 32GB+
- 是什么: 22B DiT 音视频联合基础模型，单次前向同时生成画面与同步音频（对白唇形、Foley、环境声），并原生支持 audio-to-video 与 video-to-audio 两个方向。
- 关键事实:
    * LTX-2 开源 2026-01-06（官方新闻稿）；模型发布 2025-10-23；LTX-2.3 发布 2026-03-05；LTX-2.5 发布 2026-08-11
    * 参数量：LTX-2 为 14B 视频 + 5B 音频；LTX-2.3 / 2.5 为 22B DiT
    * 原生 4K、最高 50fps；单次生成约 10–20 秒
    * LTX-2.3 模式全集：text-to-video、image-to-video、audio-to-video、video-to-audio、音视频联合生成、视频续写、关键帧插值
    * checkpoint：ltx-2.3-22b-dev（可训练）、fp8 量化版、ltx-2.3-22b-distilled（8 步）
    * 支持 LoRA 微调，官方称多数设置下 <1 小时
    * 许可证 LTX-2 Community License Agreement（允许商用但附条件；非 OSI 认证，属「开放权重」）；第三方称 LTX-2.5 对 ARR <1000 万美元组织免费且无强制品牌标识
    * LTX-2.5 速度：2×GB200 上 10 秒 720p 用时 6.8 秒；消费端建议 32GB（5090）起
- 长片作用: 这是「音频先行」在开源侧唯一原生成立的引擎——audio-to-video 是它的一等公民模式，你可以把已经定好的对白轨直接当条件送进去出画，而不是先出画再补口型。更关键的是它支持 LoRA 微调（<1 小时），这恰好补上 Seedance 2.0「权重不开放、不能本机 LoRA」的最大短板：角色一致性 LoRA 只能在 LTX 这条线上训。在 2-8 分钟产线里，它应占「需要强角色一致性 + 对白先行」的镜头，与 LongCat（人物特写/中景）形成互补。
- 来源: https://huggingface.co/Lightricks/LTX-2.3 https://github.com/Lightricks/LTX-Video https://www.globenewswire.com/news-release/2026/01/06/3213304/0/en/lightricks-open-sources-ltx-2-the-first-production-ready-audio-and-video-generation-model-with-truly-open-weights.html

## FantasyTalking  (高德 / 阿里 (Amap))
- 类别/成熟度: open-weights / **usable** | 许可: 未在本次检索中核实 | 成本: Wan2.1-14B 级别，需量化下探消费卡
- 是什么: 基于 Wan2.1 DiT 的说话肖像生成，通过跨模态注意力把音频条件注入大视频模型，侧脸与遮挡下仍能保持唇形，可驱动非人物体。
- 关键事实:
    * ACM MM 2025（第 33 届 ACM Multimedia）录用
    * GitHub / HuggingFace / ModelScope 三处全开源
    * 在 FID、FVD、IDC、ES、Aesthetic 五项上优于 AniPortrait、EchoMimic、Sonic、Hallo3
    * 已知短板：第三方评测指出它「只动嘴，其他区域运动有限」
- 长片作用: 「只需要嘴动、身体别乱动」的稳态镜头工位——比如角色静止听别人说话时的反打镜头，或者需要极高身份稳定性的大特写。长对白主镜头不适合（身体太静会露馅）。许可证未核实前不要进商用产线。
- 来源: https://github.com/Fantasy-AMAP/fantasy-talking https://dl.acm.org/doi/10.1145/3746027.3755217

## Sonic  (腾讯 + 浙江大学 (jixiaozhong))
- 类别/成熟度: open-weights / **usable** | 许可: 非商用（商用需腾讯云授权） | 成本: 消费级可跑
- 是什么: 把注意力从局部唇形转向全局音频感知的肖像动画方法，CVPR 2025。
- 关键事实:
    * CVPR 2025 录用
    * 许可证为非商用；商用需走腾讯云「视频创作大模型」
    * 第三方评测指出其面部表情丰富度有限
    * 本次检索未见 2026 年更新
- 长片作用: 非商用许可直接排除在产线之外。技术上也已被 LongCat-1.5 / InfiniteTalk 超越。仅作为对比基线出现在各家论文表格里。
- 来源: https://github.com/jixiaozhong/Sonic https://github.com/jixiaozhong/Sonic/blob/main/README.md

## Hallo3 / Hallo4  (复旦大学视觉实验室 + 百度)
- 类别/成熟度: open-weights / **research** | 许可: 未在本次检索中核实 | 成本: CogVideoX 级别，需 24GB+
- 是什么: Hallo3 是基于视频 DiT（CogVideoX 系）的高动态肖像动画；Hallo4 加入 DPO 直接偏好优化与时序运动调制。
- 关键事实:
    * Hallo4 论文 arXiv 2505.23525（2025-05）
    * Hallo3 为开源模型，Fudan Vision Lab 出品
    * 第三方多人场景评测指出 Hallo3 在角色一致性上会失守
    * 与 FantasyTalking 同属「把音频条件注入预训练 DiT」的思路
- 长片作用: 研究线产物，角色一致性在长片里不达标（多人场景会掉身份）。在 2-8 分钟产线里不占工位，Hallo4 的 DPO 思路值得关注但没有生产级交付。
- 来源: https://arxiv.org/html/2505.23525v1 https://github.com/fudan-generative-vision/hallo3

## DICE-Talk  (toto222 / 学术团队)
- 类别/成熟度: open-weights / **research** | 许可: 未在本次检索中核实 | 成本: 消费级可跑
- 是什么: 扩散式情感说话头生成，把身份与情感解耦，用跨模态注意力联合建模音视情感线索，情感表示为身份无关的高斯分布。
- 关键事实:
    * 开源 2025-05-06
    * 定位：肖像（头部）级别，非半身非全身
    * 本次检索未见 2026 年更新或长视频能力
- 长片作用: 情感解耦的思路对长片有价值（同一角色在不同情绪镜头间保持身份），但交付形态只有头部肖像、无长视频机制。不进产线，可作为情感控制方案的参考实现。
- 来源: https://github.com/toto222/DICE-Talk

## LatentSync 1.6  (字节跳动)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: 推理 18GB（1.6，512²）/ 8GB（1.5，256²）
- 是什么: 音频条件潜空间扩散的后置唇形替换：改写已有视频里的嘴部区域去对齐新音频，端到端不依赖中间 3D 表示。
- 关键事实:
    * 1.5 发布 2025-03-14（加时序层、改善中文视频、stage2 训练显存降到 20GB，256×256）
    * 1.6 发布 2025-06-11（改用 512×512 训练，缓解模糊问题）
    * 1.6 推理显存 18GB；stage2 训练 55GB。1.5 推理 8GB
    * inference_steps 可调 20–50，步数越高画质越好越慢
    * 许可证 Apache 2.0
    * 截至检索未见 1.7 / 2.0
- 长片作用: 成片前的最后一道唇形保险工位。所有上游路线（Seedance、LTX、LongCat）都可能在某些镜头唇形失准，LatentSync 1.6 的 512×512 是当前开源后置对齐的画质上限，Apache 2.0 无授权风险。代价是嘴部区域重绘会引入与原画面的纹理差异，在大特写下仍看得出——所以它该是「抢救工具」而不是「标准流程」，全片默认走原生音频驱动，只对失败镜头开刀。
- 来源: https://github.com/bytedance/LatentSync https://arxiv.org/html/2412.09262v1

## MuseTalk 1.5  (腾讯音乐天琴实验室 (TMElyralab))
- 类别/成熟度: open-weights / **production** | 许可: MIT | 成本: 4GB 起可跑；V100 30fps+
- 是什么: 潜空间 inpainting 的实时唇形同步，1.5 引入感知损失 + GAN 损失 + 同步损失的两阶段训练与时空数据采样。
- 关键事实:
    * 1.5 发布 2025-03-28；训练代码 2025-04-05 开源；1.0 为 2024-04
    * 人脸区域工作分辨率 256×256
    * V100 上 30fps+ 实时
    * 最低实测：RTX 3050 Ti Laptop 4GB 显存 fp16，8 秒视频约 5 分钟
    * stage2 训练：batch 2 + 梯度累积 8，H20 上约 85GB/卡
    * 许可证 MIT（代码与训练模型均可商用）
    * 官方自承三大短板：分辨率上限、身份保持不完美（胡须与唇形特征）、单帧生成导致的抖动
- 长片作用: 256×256 的人脸工作分辨率对 2-8 分钟成片（1080p 及以上）是硬伤，放大后必然糊，且官方承认有帧间抖动。它在你的产线里的正确工位是「预览/实时草稿」：分镜阶段用它以 30fps 秒级出口型草样确认节奏和断句，确认后再走 LongCat/LTX 正式渲染。不要用于成片。MIT 许可是它相对 LatentSync 的唯一优势（后者也是 Apache 2.0，所以这优势不成立）。
- 来源: https://github.com/TMElyralab/MuseTalk

## VideoReTalking / Wav2Lip 系  (西安电子科技大学 + 腾讯 AI Lab（VideoReTalking）；IIIT Hyderabad（Wav2Lip）)
- 类别/成熟度: open-weights / **research** | 许可: 各仓库不一（Wav2Lip 权重历史上为非商用研究许可） | 成本: 极低（4GB 以下）
- 是什么: 上一代基于 GAN 的口型替换：Wav2Lip 以 SyncNet 判别器驱动嘴部重绘，VideoReTalking 加表情编辑与身份感知增强。
- 关键事实:
    * Wav2Lip 嘴部工作分辨率仅 96×96，VideoReTalking 为 SIGGRAPH Asia 2022 工作
    * 已被 LatentSync（512×512，Apache 2.0）与 MuseTalk 1.5（256×256，MIT）在画质与时序一致性上全面取代
    * 本次检索未见 2026 年的活跃后继版本
- 长片作用: 只剩历史价值。96×96 的嘴部分辨率在 1080p 成片上是肉眼可见的糊块。唯一残余用途是超低算力的批量粗对齐（比如给几百条分镜快速打样），但 MuseTalk 在同等速度下画质更好，没有保留理由。
- 来源: https://github.com/OpenTalker/video-retalking https://github.com/Rudrabha/Wav2Lip

## MMAudio (V2)  (伊利诺伊大学 UIUC + Sony AI)
- 类别/成熟度: open-weights / **usable** | 许可: 代码开源，但作者不保证权重可商用（训练数据许可传染风险） | 成本: 消费级可跑（大模型约 8–12GB）
- 是什么: 多模态联合训练的视频到音频生成，从画面 + 短文本提示生成帧级对齐的音轨，CVPR 2025。
- 关键事实:
    * CVPR 2025；V2 为更新后的大模型 44.1kHz checkpoint
    * 变体从 16kHz 小模型到 44kHz 大模型
    * VGGSound 上 FAD 9.01（对照：ThinkSound 9.92，HunyuanVideo-Foley 6.07）
    * 训练数据 AudioSet / Freesound / VGGSound / AudioCaps / WavCaps，作者明确声明不保证预训练模型适合商用
- 长片作用: Foley 工位的老牌基线，ComfyUI 生态成熟、接入成本最低。但 FAD 9.01 已被 HunyuanVideo-Foley（6.07）和 PrismAudio 拉开，且商用许可存在数据集传染风险。在长片产线里适合做「快速铺底音效草稿」，终混阶段应换更强的模型或真人 Foley 素材。
- 来源: https://github.com/hkchengrex/MMAudio https://github.com/hkchengrex/MMAudio/blob/main/docs/MODELS.md https://openaccess.thecvf.com/content/CVPR2025/papers/Cheng_MMAudio_Taming_Multimodal_Joint_Training_for_High-Quality_Video-to-Audio_Synthesis_CVPR_2025_paper.pdf

## ThinkSound → PrismAudio  (浙江大学 + 阿里通义 (FunAudioLLM / QwenAudio))
- 类别/成熟度: open-weights / **usable** | 许可: 随 ThinkSound 仓库（需在分支内核实具体条款） | 成本: 518M 参数，消费卡轻松跑
- 是什么: ThinkSound 用思维链推理做分步可交互的音频生成与编辑（NeurIPS 2025）；后继 PrismAudio 把单一推理拆成语义/时序/美学/空间四个专门 CoT 模块，各配对应奖励函数，做多维强化学习优化，是首个把 RL 引入视频到音频生成的框架。
- 关键事实:
    * ThinkSound：NeurIPS 2025，含 AudioCoT 数据集，权重在 HF/ModelScope
    * PrismAudio：ICLR 2026 录用，发布 2026-03-24，518M 参数
    * PrismAudio 权重开放在 HuggingFace 与 ModelScope；代码在 ThinkSound 仓库的 prismaudio 分支
    * PrismAudio 在 VGGSound 域内测试集上全感知维度 SOTA，显著超过 ThinkSound
    * 支持空间立体声输出
- 长片作用: Foley 工位的 2026 年新首选之一。518M 的小体量 + 四维 CoT（尤其是「时序」模块）正好对应长片最痛的点：脚步、开门、碰撞这些必须踩在帧上的点音效。空间立体声输出对 2-8 分钟成片的听感层次是实打实的加分。建议它与 HunyuanVideo-Foley 做 A/B，按镜头类型分流。
- 来源: https://github.com/FunAudioLLM/ThinkSound/tree/prismaudio https://openreview.net/pdf?id=cIfDKEbAky https://thinksound-project.github.io/

## HunyuanVideo-Foley  (腾讯混元)
- 类别/成熟度: open-weights / **production** | 许可: 未在仓库内明示（推定为 Tencent Hunyuan Community License，需核实） | 成本: 8–20GB 显存按变体与 offload 而定
- 是什么: 带表征对齐的多模态扩散 Foley 生成，48kHz 高保真输出。
- 关键事实:
    * XXL 开源 2025-08-28；XL 版 2025-09-29（含 offload 推理支持）
    * 48kHz Hi-Fi 输出
    * 显存：XXL 常规 20GB / offload 12GB；XL 常规 16GB / offload 8GB
    * VGGSound FAD 6.07，全评测指标最优（对照 MMAudio 9.01、ThinkSound 9.92）
    * 权重公开在 HuggingFace tencent/HunyuanVideo-Foley
    * 仓库未明示许可证（腾讯混元系通常为 Community License，地域排除 EU/UK/韩国）
- 长片作用: 当前客观指标最强的开源 V2A，48kHz 直接够成片规格（对比很多模型只出 16k/44.1k）。在 2-8 分钟产线里它顶「环境声 + Foley 铺底」这个工位，和对白轨、音乐轨分三层混。落地前必须先把许可证条款钉死——腾讯系的地域排除条款会影响海外发行。
- 来源: https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley https://huggingface.co/tencent/HunyuanVideo-Foley

## ACE-Step 1.5 (含 XL)  (ACE Studio + 阶跃星辰 (StepFun))
- 类别/成熟度: open-weights / **production** | 许可: MIT（1.5 系列）；v1-3.5B 为 Apache 2.0 | 成本: ≤6GB（量化）～20GB（XL 全量）；单曲生成秒级
- 是什么: 扩散式音乐生成基础模型，歌词提示控制结构与风格，非自回归因此迭代快、风格可直接用提示词引导。
- 关键事实:
    * ACE-Step v1 为 3.5B（Apache 2.0）；ACE-Step 1.5 系列 DiT 2B（base）/ XL DiT 4B，LM 0.6B / 1.7B / 4B；XL 变体 2026-04-02 公布
    * 许可证 MIT
    * 时长：10 秒 到 10 分钟（600 秒）
    * 速度：A100 上整首 <2 秒，RTX 3090 上 <10 秒
    * 显存：量化+offload 最低 ≤6GB；XL 建议 ≥12GB（带 offload）/ ≥20GB（不带）
    * 支持 50+ 语言歌词
    * HF 变体：base / sft / turbo / xl-base / xl-sft / xl-turbo
- 长片作用: 配乐工位的唯一合理选择。600 秒上限直接覆盖你 2-8 分钟成片的全长——意味着可以一次性生成整片配乐床而不是拼段，这对长片的情绪连贯性是决定性的。MIT 许可、生成秒级、消费卡可跑，三项全绿。工作流：分镜定稿后按情绪段落写歌词/风格提示，一次出整条 BGM，再按镜头点做音量自动化。
- 来源: https://github.com/ace-step/ACE-Step-1.5 https://github.com/ace-step/ACE-Step https://huggingface.co/ACE-Step/acestep-v15-base

## YuE  (香港科技大学 HKUST + M-A-P)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0（带署名要求） | 成本: 7B 自回归，24GB 级显存；生成耗时远高于 ACE-Step
- 是什么: 7B 自回归 Transformer 全曲音乐生成（含人声），面向完整歌曲而非片段。
- 关键事实:
    * 参数量 7B
    * 许可证 Apache 2.0，需署名 'YuE by HKUST/M-A-P'
    * 自回归架构，生成速度显著慢于扩散式的 ACE-Step
    * 2026 年被第三方列为仍值得部署的四个开源音乐模型之一
- 长片作用: 人声歌曲（片头/片尾主题曲）这一个具体工位上值得保留，ACE-Step 虽也支持歌词但 YuE 的人声完整度在某些风格上更可控。纯配乐床不要用它——自回归的速度劣势在 8 分钟长度上会被放大到不可接受。
- 来源: https://github.com/multimodal-art-projection/YuE https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/

## Stable Audio Open  (Stability AI)
- 类别/成熟度: open-weights / **usable** | 许可: Stability AI Community License（年营收 <$1M 免费商用） | 成本: 消费级可跑
- 是什么: 面向音效、采样和短循环的开放音频生成模型，不是完整歌曲模型。
- 关键事实:
    * 商用免费仅限年收入 100 万美元以下的实体（Stability AI Community License）
    * 定位为 SFX / samples / short loops，不做整曲
    * 第三方 2026 年文章提到 'Stable Audio Open 1.5' 版本，未在 Stability 官方渠道核实
- 长片作用: 音效素材库工位——生成可循环的环境声垫（雨声、人群、机械嗡鸣）。它不做画面对齐，所以不能替代 MMAudio/PrismAudio/HunyuanVideo-Foley 这类 V2A；定位是「素材生成器」而非「配音器」。营收门槛对早期项目无影响，规模化后要重新谈许可。
- 来源: https://huggingface.co/stabilityai/stable-audio-open-1.0 https://www.spheron.network/blog/deploy-open-source-ai-music-generation-gpu-cloud-2026/

## Seedance 2.0 / 2.5（约束核实）  (字节跳动 Seed / 火山引擎 · BytePlus)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: 按秒计费，需查火山引擎/BytePlus 当期价目
- 是什么: 统一多模态音视频联合生成架构，文本/图像/音频/视频四模态输入，原生自带音轨。
- 关键事实:
    * Seedance 2.0：单次请求最多 12 个参考文件；音频输入 MP3，最多 3 个，总时长不超过 15 秒，可指定背景音乐、节奏同步点、音效
    * 提供内容审核 API 对生成内容做合规审计（官方通道有审核，与用户已知约束一致）
    * Seedance 2.5 在 2026 火山引擎 FORCE 大会发布：单视频最长 30 秒，最多 50 个多模态参考素材，原生 4K，支持更灵活的局部视频编辑
    * 接入渠道：火山引擎（中国）/ BytePlus（国际），以及 fal.ai、PiAPI 等第三方
    * 权重不开放，无本地部署与 LoRA 通道（本次检索未见任何开放权重迹象）
- 长片作用: 用户给的硬约束全部核实通过，且 2.5 把单段放宽到 30 秒、参考素材放宽到 50 个。对本调研维度的关键结论：Seedance「自带对白声」+「音频参考 ≤15 秒」意味着它无法承载 2-8 分钟的统一音色对白轨。正确用法是把它当画面发生器，音轨由自有 TTS 独立生产，再用 InfiniteTalk V2V 或 LatentSync 1.6 把自有对白贴回 Seedance 出的画面上。
- 来源: https://seed.bytedance.com/en/seedance2_0 https://www.volcengine.com/article/40588 https://segmentfault.com/a/1190000047897908

### 建议
音频先行在开源侧的最佳实现链路（按工位排，2026-09 状态）

第 0 层 · 剧本与时间预算
分镜表先落「每句台词 → 目标秒数」。这一步决定了后面所有模型的调用参数，不要跳过。

第 1 层 · 对白轨（这是全片的时间主骨架）
- 主引擎：IndexTTS-2.5（0.8B，2026-08-10，RTF 0.21@4090）。选它唯一的理由是 duration_factor 0.5x–2.0x 的显式时长控制——你在分镜表写「这句 3.2 秒」，它就出 3.2 秒，画面段落长度随即确定。中文情感走「情感参考音频 + 8 维情感向量」双通道。法务前置动作：发邮件到 indexspeech@bilibili.com 谈商用授权，这是整条链路唯一的许可证阻塞点。
- 角色音色库：每个主角用 GPT-SoVITS（MIT，20250606v2pro，v4 原生 48kHz）微调一份专属权重，1 分钟素材即可。理由是 2-8 分钟成片要反复改词重出，零样本克隆会在几十次调用里漂移，微调权重不会。
- 可商用兜底 / 流式预演：CosyVoice 3（Apache 2.0，0.5B，150ms 首包）。如果 bilibili 授权谈不下来，它直接顶上主引擎，代价是失去 duration_factor（需后期做时间拉伸）。
- 明确排除：F5-TTS（权重 CC-BY-NC）、MaskGCT（CC-BY-NC）、XTTS-v2（CPML 非商用且 Coqui 已关停无法补授权）、Fish-Speech/OpenAudio S2 Pro（仓库 LICENSE 明文禁商用，别信第三方说的 MIT）。
- 闭源对照只用于试音阶段：MiniMax Speech 2.8（$60-100/百万字符，10 秒克隆，<250ms）中文更贴国内耳朵；ElevenLabs v3（$100/百万字符，70+ 语言，[audio tags]）英文与强演绎更好。两者都不建议当产线主引擎——音色资产不在自己手里。

第 2 层 · 强制对齐，把音频切成镜头
TTS 出整段后跑 WhisperX 或 MFA 拿词级时间戳，按分镜切成 8–30 秒的音频片段，每片带入点/出点/情绪标签。这一层是「音频先行」和「先出画再配音」的真正分水岭，它把音频变成了排产调度表。

第 3 层 · 出画（三条并行路线，按镜头类型分流）
- A 路 · 人物对白镜头主力（占 70-80% 产能）：LongCat-Video-Avatar-1.5（美团，MIT，2026-05-21，480P/720P，Whisper-Large 音频编码，DMD2 蒸馏 8 步，原生多流音频 + 视频续写，INT8 量化）。MIT 许可 + 8 步推理 + 原生续写，三项同时成立的只有它，这是开源侧当下最该押注的一个模型。
- B 路 · 角色一致性镜头：LTX-2.3 / LTX-2.5（22B，2026-03-05 / 2026-08-11，开放权重，原生 audio-to-video 模式，LoRA 微调 <1 小时）。它是唯一把「音频当一等输入条件出画」的开源基础模型，也是唯一能训角色 LoRA 的一条线——正好补 Seedance 2.0 权重不开放、不能本机 LoRA 的坑。
- C 路 · 把闭源画面接回自有对白（关键工位，最容易被忽略）：InfiniteTalk（Apache 2.0，Wan2.1-I2V-14B 基座，默认 1000 帧/40 秒、分段可无限长，FP8 + 低显存模式，多人权重）的 video-to-video 重配音。流程是：Seedance 2.0/2.5 出「画面好但自带的是它的对白声」的镜头 → InfiniteTalk 用你的 TTS 轨重写唇形、头部运动、体态。这一步同时绕开了 Seedance 的三条约束（音频参考 ≤15 秒、自带对白声、单段 4-15 秒），并让全片音色由你的 TTS 统一。
- 低显存草稿档：EchoMimicV3-Flash-Pro（Apache 2.0，1.3B，12GB，8 步，768×768，138 帧/段）。12-16GB 开发机上先看口型和节奏对不对，确认后再上 A/B 路正式渲染。
- 带运镜的关键镜头：Wan2.2-S2V-14B（Apache 2.0，80GB 单卡门槛，需社区 FP8 量化下探）。少量使用。

第 4 层 · 后置唇形抢救（不做标准流程，只对失败镜头开刀）
LatentSync 1.6（Apache 2.0，512×512，推理 18GB）。它是开源后置对齐的画质上限，但嘴部重绘会留下纹理差异，大特写下仍可见。MuseTalk 1.5（MIT，256×256，V100 30fps+，4GB 起）只用于分镜阶段的实时草稿预览，绝不进成片——官方自承分辨率上限、身份保持不完美、单帧生成抖动三大短板。VideoReTalking / Wav2Lip 系（嘴部 96×96）已无保留理由。

第 5 层 · 三轨混音
- Foley / 环境声：HunyuanVideo-Foley（48kHz，VGGSound FAD 6.07，全场最优；显存 8-20GB 按变体）与 PrismAudio（ICLR 2026，2026-03-24，518M，四维 CoT + RL，空间立体声）做 A/B 分流——点音效踩帧用 PrismAudio 的时序 CoT，环境声垫用 Hunyuan 的 48kHz。MMAudio（FAD 9.01）只当快速草稿，且作者声明不保证权重商用。
- 配乐：ACE-Step 1.5（MIT，2B/4B DiT，10 秒–600 秒，A100 <2 秒/首，≤6GB 量化可跑）。600 秒上限刚好一次出完整片 BGM，不用拼段，这对长片情绪连贯是决定性的。片头/片尾人声主题曲可用 YuE（Apache 2.0，7B）。Stable Audio Open 只当循环素材生成器（年营收 <$1M 免费商用）。
- 混音纪律：对白轨来自第 1 层且全片唯一；视频模型自带的音轨（LTX 的原生音频、Seedance 的自带对白声）在长片里一律静音丢弃，只保留画面。否则每 15 秒换一次音色，成片必废。

关于 Seedance 2.0 的定位结论
核实通过：闭源、权重不开放、不能本机 LoRA、单段 4-15 秒（2.5 放宽到 30 秒，参考素材放宽到 50 个）、音频参考最多 3 个共 15 秒、官方通道有内容审核 API。这些约束合起来说明一件事：Seedance 只能是「单镜画面发生器」，不能是长片引擎，更不能是音轨来源。你的系统平台真正要建的是第 2 层（对齐调度）和第 3 层 C 路（InfiniteTalk 重配音回贴），这两个工位把闭源画面质量和开源音轨控制权缝在一起，是整个架构的承重墙。

法务前置清单（排产前必须闭环）
1) IndexTTS-2.5 商用授权（bilibili）；2) HunyuanVideo-Foley 与 HunyuanVideo-Avatar 的腾讯混元 Community License 地域条款（排除欧盟/英国/韩国），影响海外发行；3) LTX-2 Community License 的商用附加条件原文；4) MMAudio 权重的数据集传染风险；5) FantasyTalking / MultiTalk / Hallo3 / DICE-Talk 许可证本次未核实，入库前逐个确认。内容策略上角色虚构、设定成年、不使用真人肖像，正好规避各家 TTS 平台「克隆授权责任在使用方」的主要风险敞口。

### 存疑
- ChatTTS 的权重许可证原文与 2026 年维护状态本次未核实；社区普遍说法是它依赖随机 speaker 采样、没有稳定的指定音色克隆能力，此说法也未在本次检索中从官方仓库确认。入库前需读其 LICENSE。
- Fish Audio 早期版本（fish-speech 1.x、OpenAudio S1-mini）的许可证是否仍为 Apache / CC-BY-NC-SA 未核实。已确认的是：当前 fishaudio/fish-speech 主仓 LICENSE 为 FISH AUDIO RESEARCH LICENSE，明文禁止商用。多篇 2026 年第三方文章（tryspeakeasy、aipedia）称「S1 与 S2 Pro 为 MIT，可自托管」，与仓库 LICENSE 原文直接冲突，不采信。
- OmniHuman-1.5 单价存在两个口径：BytePlus 官方产品页标 $0.12/秒，fal 的用户指南标 $0.16/秒（30 秒 = $4.80）。签约前需向 BytePlus 直接核实。同时第三方接入方给出的时长上限也不一致（35 秒 / <60 秒 / 建议不超 15 秒），官方 720p≤60s、1080p≤30s 的口径来自 fal 文档而非 BytePlus 官方页。
- LongCat-Video-Avatar-1.5 的具体显存需求、最长视频时长、参数量均未在 HuggingFace 卡片或 GitHub README 中公布，只确认了 MIT 许可、480P/720P、8 步推理、INT8 量化选项。需实测。另外该模型的归属存在表述差异：权重与代码在 meituan-longcat 名下（美团），但项目页挂在 meigen-ai.github.io 且由 MeiGen-AI 的 InfiniteTalk README 发布，两家的合作关系未核实。
- LatentSync 1.7 与 MuseTalk 2.0 在本次检索中均未找到任何官方发布记录。第三方 2026 年评测文章（sync.so、reviewnexa）讨论的仍是 1.6 / 1.5。若有传闻版本，不可证实。
- Wan 2.5 与 Wan 2.6 的权重开放状态说法矛盾：一方（opencreator.io）称 Wan 2.6 为 open-source release，另一方（videoai.me）明确称 Wan 2.5 权重不公开供本地部署。已确认为 Apache 2.0 开放权重的只有 Wan2.2-S2V-14B（2025-08-26）。2.5/2.6 的音视频同步生成能力（1080p 24fps 最长 15 秒、原生对白唇形）来源均为第三方站点，未见阿里官方仓库佐证。
- MultiTalk 的「最长 15 秒」上限来自第三方摘要，未在 MeiGen-AI/MultiTalk 官方 README 中核实；该仓库的许可证本次也未确认。
- LTX-2.5 「年 ARR <1000 万美元的组织免费使用、无强制品牌标识」来自第三方博客（orcarouter.ai / datanorth.ai），未读 LTX-2 Community License Agreement 原文。LTX-2.3 HuggingFace 卡片只写了许可证名称 ltx-2-community-license-agreement，未展开条款。同时 LTX-2 的参数量存在两个口径：官方新闻稿系列说 14B 视频 + 5B 音频，而 LTX-2.3/2.5 被描述为 22B 单体 DiT，二者关系未澄清。
- HunyuanVideo-Foley 仓库未明示许可证；本调研按腾讯混元系惯例推定为 Tencent Hunyuan Community License，但这是推断不是事实。HunyuanVideo-Avatar 的权重许可证同样未在其 HuggingFace 卡片确认，只确认了 HunyuanVideo 主仓 LICENSE 中 Territory 定义排除欧盟、英国、韩国；「1 亿 MAU 上限」条款本次未找到佐证。
- MMAudio 的商用可行性：官方文档声明训练数据（AudioSet / Freesound / VGGSound / AudioCaps / WavCaps）各有许可，作者不保证预训练模型适合商用。这意味着没有明确的商用许可，也没有明确的禁止——属于法务灰区，不应按「可商用」入库。
- FantasyTalking、Hallo3 / Hallo4、DICE-Talk、Sonic 的 2026 年维护状态与具体许可证条款均未核实。Sonic 已确认为非商用（商用需走腾讯云视频创作大模型），其余三个的许可证为空白。
- 「Stable Audio Open 1.5」这个版本号仅见于第三方博客（Spheron），未在 Stability AI 官方渠道或 HuggingFace 核实。已确认的是 Stable Audio Open 的商用免费门槛为年营收 100 万美元以下。
- IndexTTS-2.5 的语言支持存在口径差异：GitHub README 列中/英/日/西/阿拉伯语五种，arXiv 2601.03888 摘要只列中/英/日/西四种。参数量 0.8B 来自 README，arXiv 摘要未给。显存需求官方未公布。
- InfiniteTalk 在 FP8 + 低显存模式下「12GB 可跑」的数值来自中文社区 ComfyUI 整合包教程（bilibili / CSDN），不是官方 README 数值，官方 README 未提供显存表。
- 第三方文章称 InfiniteTalk 有「V2 版」（CSDN 整合包页面），但 MeiGen-AI/InfiniteTalk 官方 README 与 releases 页均未见 V2 发布记录，判定为整合包发行方自行编号，不是官方版本。

### 事实核查修正
- [WRONG] [ChatTTS] 提供说话风格与情感的细粒度控制标记
  → 官方仓库 README/FAQ 明确否认存在情感控制标记。ChatTTS 的 token 级控制单元只有三类：[laugh_0]–[laugh_2]（笑声）、[oral_0]–[oral_9]（口语化程度）、[break_0]–[break_7] 及 [uv_break]/[lbreak]（停顿）。FAQ 原文表述为『目前唯一的 token 级控制单元』就是笑声、口语化和停顿，并把情感控制列为『未来版本可能加入』。因此『说话风格与情感的细粒度控制』是把口语化/停顿控制误读成了情感控制，入库时不应把 ChatTTS 归入可控情感 TTS。 https://github.com/2noise/ChatTTS
- [WRONG] [ChatTTS] 本次检索未能核实其 2026 年维护状态与权重许可证原文（研究员自标存疑项）
  → 该项可核实，且结论对选型有决定性影响：ChatTTS 代码许可证为 AGPLv3+，模型权重为 CC BY-NC 4.0，官方明确限定为教育与研究用途，禁止商用。这意味着 ChatTTS 不具备商用可行性，应与 MaskGCT / F5-TTS 权重 / XTTS-v2 归为同一非商用档，而不是作为『对话式中文』候选入库。另外社区关于『只有随机 speaker 采样、无稳定指定音色克隆』的说法已从官方仓库确认：官方接口为 chat.sample_random_speaker()，文档只提供采样并保存 speaker embedding 以复现音色，未提供 https://github.com/2noise/ChatTTS
- [WRONG] [F5-TTS] 训练数据 Emilia + WenetSpeech4TTS + LibriTTS + LJSpeech
  → 混淆了『仓库提供数据准备脚本的数据集』与『已发布预训练权重的实际训练数据』。官方 README 只声明预训练模型训练于 Emilia（原文：权重采用 CC-BY-NC 许可正是 due to the training data Emilia, which is an in-the-wild dataset）；HuggingFace SWivid/F5-TTS 模型页关联的 dataset 也只有 amphion/Emilia-Dataset 一个。WenetSpeech4TTS / LibriTTS / LJSpeech 是仓库 Training 章节支 https://github.com/SWivid/F5-TTS
- [OUTDATED] [Fish-Speech / OpenAudio S2 Pro] 产品命名为 OpenAudio S2 Pro
  → 当前官方命名已改回 Fish Audio S2 Pro，不再用 OpenAudio 前缀（OpenAudio 是 S1/S1-mini 时期的品牌）。权重仓库路径为 huggingface.co/fishaudio/s2-pro，独立技术报告为 arXiv 2603.08823（2026-03-09），HF collection 更新于 2026-03-10。规格数字本身核实无误（4B Slow AR + 400M Fast AR、10 个 RVQ codebook ~21Hz、10M+ 小时、80+ 语言、RTF 0.195、TTFA ~100ms、3 https://huggingface.co/fishaudio/s2-pro
- [UNVERIFIABLE] [GPT-SoVITS] 许可证 MIT（代码与模型均可商用）
  → 前半句成立、后半句是推断。官方 README 的 MIT 徽章与仓库 LICENSE 文件覆盖的是代码；仓库内并未对预训练权重单独作出 MIT 授权声明。GPT-SoVITS 的预训练模型链路依赖第三方上游组件（中文 RoBERTa-wwm-ext 类 BERT、HuBERT 类自监督特征提取器等），这些上游各有自己的许可证。因此『模型也可商用』属于未经官方确认的外推，入库前应逐个核对权重包内各子模型的来源与许可证，不要按 MIT 一刀切。 https://github.com/RVC-Boss/GPT-SoVITS
- [WRONG] [IndexTTS-2.5] 情感控制三通道：情感参考音频 / 8 维情感向量 / 文本情感（Qwen3 微调的软指令）+ emo_alpha 0.0–1.0
  → 官方 README 列出的是四种方式而非三种：(1) 情感参考音频；(2) 8 维情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]；(3) use_emo_text=True 从待合成文本自动抽取情感；(4) 通过独立的 emo_text 参数传入与正文解耦的情感描述文本。(3) 和 (4) 是两条不同的通路（前者复用正文、后者独立描述），研究员把它们合并成了一条。emo_alpha 0.0–1.0 与 8 维向量的具体维度语义核实无误。另『Qwen3 微调 https://github.com/index-tts/index-tts
- [UNVERIFIABLE] [MaskGCT] 发布 2024-10-19
  → 找不到 2024-10-19 的一手依据，两个可考的日期都不是这一天：arXiv 2409.00750 的提交日为 2024-09-01；HuggingFace amphion/MaskGCT 权重仓库的 initial commit 为 2024-10-13。建议改为『论文 2024-09-01，开源权重 2024-10-13 上 HF』。训练数据（Emilia，英文 5 万 + 中文 5 万 = 10 万小时）与权重许可证 CC-BY-NC-4.0 均核实无误。 https://huggingface.co/amphion/MaskGCT/commits/main
- [UNVERIFIABLE] [XTTS-v2 (Coqui)] 历史商用授权价约 365 USD/年，适用营收/融资 <100 万美元的公司
  → 无法从任何仍存活的一手来源核实。Coqui 官网与 Coqui Studio 已随公司 2024 年 1 月关停而下线，HuggingFace coqui/XTTS-v2 模型卡只写明许可证名称 Coqui Public Model License (CPML) 并链向一篇讲 CPML 由来的博客，页面本身不含任何价格或营收门槛条款，且页面文案仍停留在公司关停前的状态（仍在推广 Coqui Studio 与 Coqui API）。这两个具体数字应从声明中删除或明确标注为无法验证的历史传闻——它们对选型不产生实际影响，因为授权渠道已确定不存在。CPML 本 https://huggingface.co/coqui/XTTS-v2
- [UNVERIFIABLE] [F5-TTS] RTF ... 0.0402（batch=1）；TensorRT-LLM 离线 PyTorch 模式 0.1467
  → 官方 README 的基准表中可确认的只有一个数字：F5-TTS Base (Vocos) 在单张 L20、16 NFE、并发 2、26 组 prompt-audio/target-text 配对下 RTF = 0.0394。batch=1 的 0.0402 与 TensorRT-LLM 离线 PyTorch 模式的 0.1467 未在本次抓取的 README 主文中出现（可能位于 src/f5_tts/runtime/triton_trtllm 子目录的独立文档中，未验证）。引用这两个数字前需定位到具体的子目录 README 并注明测试条件，否则会与主 https://github.com/SWivid/F5-TTS
- [OUTDATED] [IndexTTS-2] 发布 2025-09-08（arXiv 2506.21619）
  → arXiv ID 正确但日期需要拆分标注：arXiv 2506.21619《IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech》v1 提交于 2025-06-23，v2 修订于 2025-09-03。2025-09-08 既非论文提交日也非修订日，应指开源权重发布日。建议写成『论文 2025-06-23（v2 修订 2025-09-03），开源权重 2025-09-08』，否 https://arxiv.org/abs/2506.21619

### 遗漏补充
- MOSS-TTSD（OpenMOSS/复旦）——专为『对白』设计的对话式语音合成模型，原生支持双说话人交替、中英双语长音频（播客级），开源权重。这条调研线主题就是对白 TTS，却完全没有覆盖专做对话的模型，是最核心的遗漏。
- VibeVoice（微软）——长对话多说话人 TTS，1.5B 与 7B 两档，官方宣称可生成最长 90 分钟、最多 4 个说话人的连续对话音频，MIT 许可。直接命中多角色对白场景，且许可证友好，必须入库。
- Dia-1.6B（Nari Labs）——Apache 2.0，单次前向生成整段双人对话脚本（支持 [S1]/[S2] 说话人标记与笑声、咳嗽等非语言音），是开源侧对白 TTS 的标杆之一。
- FireRedTTS-2（小红书）——面向长篇多说话人对话的开源中文 TTS，流式 + 对话上下文建模，中文对白质感在开源阵营第一梯队。
- Higgs Audio V2（Boson AI）——基于 LLM 的统一音频生成模型，支持多说话人对话零样本克隆与背景音，开源权重，对白场景表现突出。
- Chatterbox（Resemble AI）——MIT 许可的开源 TTS，独有 exaggeration（情绪夸张度）连续控制参数，是少数『既能商用又能调情绪强度』的选项，与 IndexTTS-2.5 的情感控制正面竞争但许可证宽松得多。
- Kokoro-82M（hexgrad）——Apache 2.0、仅 82M 参数的超轻量 TTS，CPU 可实时，适合做大批量对白预览稿或低成本兜底；开源商用友好度上限最高的一档，整条调研线缺少轻量级方案。
- Step-Audio 2 / Step-Audio-EditX（阶跃星辰）——Apache 2.0，EditX 支持对已生成语音做情感/语速/口音的后期编辑（而非重新合成），对『配音改一句不想重录全段』的实际制作流程价值很大。
- Spark-TTS（SparkAudio，基于 Qwen2.5-0.5B）与 Orpheus TTS（Canopy Labs，基于 Llama-3B，Apache 2.0）——两条主流的 LLM-backbone 开源 TTS 路线，与 CosyVoice/Fish 架构对照时的重要参照系。
- MiniMax Speech 2.x（海螺）与 Qwen3-TTS（阿里）、豆包/Seed-TTS（字节）——三家国内闭源 API 的中文对白质感目前普遍优于开源方案，调研只覆盖了 ElevenLabs 一家闭源 API，中文项目的实际选型里这三家才是主要对手。
- Seed-VC 与 RVC（音色转换 / 变声）——对白制作里常用的『先用任意 TTS 出稿、再用 VC 转成目标角色音色』两段式工作流，可绕开单一 TTS 的克隆稳定性瓶颈，整条线完全没提 VC 路线。
- LivePortrait（快手，MIT）——表情与姿态迁移的开源标杆，许可证友好，常与口型模型组合使用；调研的口型/数字人部分列了大量论文级项目却漏了这个工程落地率最高的。
- 实时口型方案缺位：Ditto-TalkingHead（蚂蚁，实时扩散口型）、MuseTalk 1.5（腾讯，已确认存在的稳定版本，而非未经证实的 2.0）、Wav2Lip / Wav2Lip-384 与 SadTalker（基线对照）。
- 商用口型 API：Sync.so 的 lipsync-2 / lipsync-2-pro（零样本口型，无需按人训练）、Hedra Character-3、HeyGen、Synthesia、D-ID——调研在数字人侧几乎只覆盖开源权重，缺少可直接签约交付的商用 API 档位。
- Runway Act-Two（表演捕捉驱动）——用真人表演视频驱动角色的口型与表情，与『音频驱动口型』是两条不同的技术路线，对有真人参考表演的片子质量上限更高。
- 端到端原生对白视频路线：Veo 3.x（Google）、Sora 2（OpenAI）、可灵 Kling 2.x 对口型——这些模型在生成视频时直接输出同步对白音频与唇形，可能从根本上替代『TTS + 口型驱动』两段式管线，作为竞争路线必须在决策里出现。
- EchoMimic V2 / V3（蚂蚁）——音频驱动的半身/全身数字人，开源，在 InfiniteTalk / MultiTalk 之外的另一条主流开源路线。


====================================================================================================
# [character-consistency] 跨镜头角色身份一致性技术全景（虚构角色 / 2-8 分钟长片产线）

## Wan2.1-VACE (1.3B / 14B)  (阿里巴巴 通义万相 (ali-vilab))
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: 1.3B ≈8-12GB；14B ≈40GB+（fp8/block-swap 可降至 24GB）
- 是什么: All-in-One 视频创作编辑模型，统一 R2V（参考图到视频）/ V2V / MV2V（掩码编辑），可自由组合任务。ICCV 2025。
- 关键事实:
    * 两档权重：1.3B（480P）与 14B（480P/720P）
    * 支持多参考图输入（人物+物体+背景分离参考）
    * Apache 2.0（随 Wan2.1 主仓）
    * 1.3B 档可在 ~8-12GB 显存跑通，14B 档需 40GB+ 或 fp8/block-swap
    * GitHub ali-vilab/VACE 提供 UserGuide.md 与 vace_wan_inference.py
- 长片作用: 长片产线的「镜头级参考注入 + 局部重绘」工位。分镜拼接时，VACE 的 R2V 用来把角色定妆图打进每一镜；MV2V 用来在成片后单独修脸/修服装而不重渲整镜。是目前唯一开源、可本机 LoRA 叠加、又能同时吃「参考图 + 姿态 + 掩码」的多条件入口。
- 来源: https://github.com/ali-vilab/VACE https://github.com/ali-vilab/VACE/blob/main/UserGuide.md https://github.com/Wan-Video/Wan2.1

## Wan2.2-VACE-Fun-A14B  (阿里巴巴)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0（随 Wan2.2） | 成本: A14B 双专家，fp8 下约 24-32GB；bf16 需 60GB+
- 是什么: VACE 能力迁移到 Wan2.2 MoE 架构的版本，分 High-noise / Low-noise 双专家权重，支持 inpainting / pose / depth / reframe 四类控制。
- 关键事实:
    * A14B MoE：总参 ~27B，单步激活 ~14B
    * High noise + Low noise 两份权重必须配对加载
    * 控制模式：inpainting、pose、depth、reframe
    * 已被 ComfyUI 原生工作流与多家 API（AI/ML API 等）接入
- 长片作用: 接替 Wan2.1-VACE 作为 720P 镜头主力。双专家结构意味着角色 LoRA 也要训两份（high/low），这是产线上最容易踩的坑——只加载一份会导致「构图像、细节不像」或反之。
- 来源: https://docs.aimlapi.com/api-references/video-models/alibaba-cloud/wan2.2-vace-fun-a14b-inpainting-image-to-video https://github.com/Wan-Video/Wan2.2 https://www.stablediffusiontutorials.com/2025/09/wan2.2-vace-fun.html

## Wan2.2-Animate-14B  (阿里巴巴 通义万相)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: fp8 下 ~24GB 可跑 480P；720P 建议 48GB+
- 是什么: 统一的角色动画 + 角色替换模型：给一张角色图 + 一段驱动视频，复刻表情与动作；或把角色换进原视频场景（自动匹配光照色调，带 Relighting LoRA）。
- 关键事实:
    * 发布 2025-09-19，HF: Wan-AI/Wan2.2-Animate-14B
    * MoE ~27B 总参 / ~14B 激活
    * 720P @ 24fps
    * Apache 2.0，权重+推理代码全开
    * 附带 Relighting LoRA 做场景融合
    * ComfyUI 原生工作流已支持
- 长片作用: 「表演工位」。长片里对白镜、情绪镜的最稳路线不是纯 T2V，而是先用真人或 3D 参考演一遍，再用 Animate 把虚构角色贴上去——动作与嘴型的时域抖动被驱动视频锁死，身份漂移只剩下静态外观一个自由度。
- 来源: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B https://wan.video/blog/wan2.2-animate https://arxiv.org/pdf/2509.14055

## Wan-Animate-2  (阿里巴巴 通义万相)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: 720P 推荐 8×A800；480P 2×A800；蒸馏版 10 步可显著降本
- 是什么: Animate 的第二代：端到端直吃驱动视频（去掉中间姿态提取器），重新设计 DiT；新增文本可控视角调整，把镜头视角与驱动素材解耦。
- 关键事实:
    * 发布 2026-08-07（GitHub Wan-Video/Wan-Animate-2，仓库 8 月 8 日更新）
    * 14B 基础权重 + 蒸馏权重（10 步 vs 基础 40 步）
    * 默认 720P；480P 可在 2×A800 跑
    * 官方调优目标 8×A800
    * Apache-2.0
    * 已有 Diffusers 集成与 Wan2.2-Animate-2-14B-Diffusers / -Distilled-Diffusers（HF，2026-08-13）
- 长片作用: 取代 Wan2.2-Animate 做表演工位，且「文本视角控制」直接解决长片痛点：同一段驱动表演可以出正面/侧面/过肩三个镜头，而角色外观来自同一次注入——这是把「一次表演摊销到多个分镜」的关键，能压低跨镜漂移的源头方差。
- 来源: https://github.com/Wan-Video/Wan-Animate-2 https://huggingface.co/Wan-AI

## Wan2.2-S2V-14B  (阿里巴巴 通义万相)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: 官方 ≥80GB（A100 80G / H100）；fp8 社区量化可压到 ~32-40GB
- 是什么: 语音驱动视频：参考图 + 音频（+ 可选文本 / pose_video）生成口型同步的表演镜头，Wav2Vec2 音频编码器。
- 关键事实:
    * HF: Wan-AI/Wan2.2-S2V-14B，2025-09 发布
    * 同时支持 480P / 720P
    * 本地推理官方标注需 ≥80GB 显存
    * --pose_video 支持姿态序列引导
    * Apache-2.0
    * 支持长片段扩展与精确唇形编辑
- 长片作用: 「对白工位」的开源替代。用户方案里 Seedance 自带对白声但闭源；S2V 是唯一能在本机把「角色 LoRA + 自有 TTS 音轨」合成对白镜的开源路径，且能叠 pose_video 控制身体表演。
- 来源: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B https://github.com/Wan-Video/Wan2.2

## Phantom / Phantom-Data  (字节跳动 智能创作团队)
- 类别/成熟度: open-weights / **usable** | 许可: 以仓库 LICENSE 为准（基于 Wan 系列基座） | 成本: 与 Wan 1.3B/14B 基座同级
- 是什么: 主体一致视频生成框架：重设计文本-图像联合注入模块，用 text-image-video 三元组学跨模态对齐，统一支持单主体与多主体参考。ICCV 2025。
- 关键事实:
    * arXiv 2502.11079（2025-02）
    * GitHub Phantom-video/Phantom 开源代码与权重
    * Phantom-Data：约 100 万条跨配对身份一致数据（首个通用 S2V 一致性数据集）
    * 显式处理两个失效模式：image content leakage（参考图背景泄漏）与 multi-subject confusion（多主体串味）
- 长片作用: 多角色同框镜的解法。长片里「双人对戏」镜头最常见的崩法是 A 的脸长到 B 身上，Phantom 的 ID router 思路是目前开源侧对这个问题最直接的处理。但它本质是单镜方案，跨镜一致性仍靠外部锚点。
- 来源: https://arxiv.org/abs/2502.11079 https://github.com/Phantom-video/Phantom https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Phantom_Subject-Consistent_Video_Generation_via_Cross-Modal_Alignment_ICCV_2025_paper.pdf

## SkyReels-A2  (Skywork AI（昆仑万维）)
- 类别/成熟度: open-weights / **usable** | 许可: 以仓库 LICENSE 为准 | 成本: 基于 Wan/HunyuanVideo 级基座，14B 档 40GB+
- 是什么: Elements-to-Video（E2V）：把任意视觉元素（角色、道具、背景）按文本指令组装进视频，对每个元素保持与参考图的严格一致。附 A2-Bench 评测集。
- 关键事实:
    * arXiv 2504.02436（2025-04）
    * GitHub SkyworkAI/SkyReels-A2，代码+权重公开，仓库最后更新 2025-06-03
    * 自称首个开源商用级 E2V 模型
    * 图文联合嵌入模块注入多元素表征
    * A2-Bench 系统评测基准
- 长片作用: 「元素装配工位」：角色 + 固定服装 + 固定道具（法杖、耳环、外套）作为独立元素分别锚定。对服装漂移特别有用——把服装当成独立 element 而非角色外观的一部分，能显著降低换镜后的服饰变形。缺点是 2025-06 后已近一年未更新，被 SkyReels-V3 的 R2V 取代。
- 来源: https://arxiv.org/html/2504.02436 https://github.com/SkyworkAI/SkyReels-A2 https://skyworkai.github.io/skyreels-a2.github.io/

## SkyReels-V3（R2V-14B / V2V-14B / A2V-19B）  (Skywork AI)
- 类别/成熟度: open-weights / **usable** | 许可: Skywork License（需核对商用条款） | 成本: 标准 24GB+；--low_vram FP8 可在 24GB 以下跑 540P
- 是什么: 多模态视频生成模型族，三个任务三份权重：参考图到视频、视频续写（含转场/切镜）、语音驱动数字人。
- 关键事实:
    * 发布 2026-01-29，GitHub SkyworkAI/SkyReels-V3
    * R2V-14B：接受 1-4 张参考图，720P，支持 1:1 / 3:4 / 4:3 / 16:9 / 9:16
    * V2V-14B：单镜续写 5-30 秒；shot-switching 模式含 5 种转场类型；720P 最长 30 秒输出
    * A2V-19B：单张肖像 + 音频，最长 200 秒说话人视频
    * 官方自评 Reference Consistency 0.6698（族内最高），Visual Quality 0.8119（高于 Kling 1.6 的 0.8034），A/V Sync 8.18（对标 OmniHuman 1.5 的 8.25）
    * 低显存模式 --low_vram + FP8 量化，可退到 540P/480P
    * Skywork License（非标准 Apache）
- 长片作用: 2026 年开源侧唯一同时覆盖「参考注入 + 跨镜切换续写 + 200 秒说话人」三个工位的模型族。V2V-14B 的 shot-switching 是长片拼接的直接工具：它在模型内部做切镜，比外部硬剪的跨镜漂移低。A2V-19B 的 200 秒上限意味着单个长对白镜不必切碎。注意许可证不是 Apache，商用前必须读条款。
- 来源: https://github.com/SkyworkAI/SkyReels-V3 https://huggingface.co/Skywork/SkyReels-V3-A2V-19B

## Stand-In (+ Stand-In-V2 预告)  (微信视觉团队 WeChatCV)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: 在 Wan 14B 基座上额外 153M，显存开销可忽略；跟随基座
- 是什么: 轻量即插即用身份控制：条件图像分支 + 受限自注意力 + 位置映射，只训 1% 额外参数就把身份锁进视频生成。CVPR 2026。
- 关键事实:
    * arXiv 2508.07901（2025-08-11 首版，v4 2026-03-20），CVPR 2026 接收
    * 仅 153M 可训练参数（约基座 1%），仅用 2000 对训练数据
    * 支持 Wan2.1-14B-T2V（主）与 Wan2.2-T2V-A14B（2025-12 加入）
    * Apache-2.0
    * 兼容社区 LoRA（风格化）、VACE（姿态引导）、实验性 face swap（infer_face_swap.py）
    * 官方 ComfyUI 预处理节点
    * 依赖 AntelopeV2 人脸识别模型做身份编码
    * Stand-In-V2 于 2026-08-10 公告 coming soon
- 长片作用: 最划算的「脸部锚」。153M 的 adapter 可以和角色 LoRA 并联：LoRA 负责发型/服装/体型等整体外观，Stand-In 负责人脸几何。它是本项目里「双锚推理」中人脸那一锚的最佳开源实现，且官方明确与 VACE 兼容，可以三层叠（VACE 参考图 + 角色 LoRA + Stand-In 人脸）。
- 来源: https://arxiv.org/abs/2508.07901 https://github.com/WeChatCV/Stand-In https://huggingface.co/papers/2508.07901

## ConsisID  (北大袁粒组 PKU-YuanGroup)
- 类别/成熟度: open-weights / **research** | 许可: Apache-2.0 | 成本: 44GB 标准；5GB（极限 offload，速度极慢）
- 是什么: 频率分解式身份保持 T2V：把人脸拆成低频（整体轮廓）与高频（五官细节）分别注入 DiT。CVPR 2025 Highlight。
- 关键事实:
    * 基座 CogVideoX-5B
    * 输出 720×480，49 帧 @ 8fps（约 6 秒）
    * 推理约 44GB 显存；开 VAE tiling + sequential CPU offload 可降到 ~5GB
    * Apache-2.0（CogVideoX 组件另循其许可）
    * 代码 / 数据 / 权重全开（HF / ModelScope / WiseModel）
- 长片作用: 分辨率和时长（720×480 / 6 秒）已远落后于 2026 产线标准，不建议进产线。价值在方法论：「低频管身份骨架、高频管五官纹理」这个拆法，正是长片里脸漂的诊断框架——低频漂=骨相变了（需要更强参考锚），高频漂=五官糊了（需要 face refiner 二次修）。
- 来源: https://github.com/PKU-YuanGroup/ConsisID https://arxiv.org/abs/2411.17440

## Ingredients  (feizc 等（社区/学术）)
- 类别/成熟度: open-weights / **research** | 许可: 以仓库为准 | 成本: CogVideoX-5B 级，~40GB
- 是什么: 多 ID 定制视频：人脸提取器（全局+局部）+ 多尺度投影器 + ID router（动态把多个 ID embedding 分配到对应时空区域）。
- 关键事实:
    * arXiv 2501.01790（2025-01）
    * GitHub feizc/Ingredients，数据/代码/权重全开
    * 基于 CogVideoX 系 DiT
    * 三模块结构：facial extractor / multi-scale projector / ID router
    * 多阶段训练协议
- 长片作用: ID router 是多角色同框的核心机制参考。产线上不直接用（基座太旧），但如果自研多角色注入层，这是最清晰的开源实现范本。
- 来源: https://arxiv.org/abs/2501.01790 https://github.com/feizc/Ingredients https://huggingface.co/papers/2501.01790

## HunyuanCustom  (腾讯混元)
- 类别/成熟度: open-weights / **usable** | 许可: 腾讯混元社区许可（非 Apache，需核对商用条款） | 成本: HunyuanVideo 13B 级基座，60GB+；社区量化版可降
- 是什么: 多模态驱动的定制视频生成：LLaVA 文图融合 + 时序拼接的 ID 增强模块 + AudioNet（空间交叉注意力分层对齐）+ 视频驱动注入。
- 关键事实:
    * arXiv 2505.04512，权重与推理代码 2025-05-08 发布
    * GitHub Tencent-Hunyuan/HunyuanCustom
    * 关键限制：仅开源了单主体（Single-Subject）权重，多参考图理论支持但未放权重
    * 条件模态：图像 / 音频 / 视频 / 文本
- 长片作用: 「时序拼接式 ID 增强」是值得抄的工程技巧：把参考图当作视频第 -1 帧拼进 latent 序列，让自注意力天然看到身份。但多主体权重未开放，对双人戏镜头没用。腾讯侧更新的路线已转向 HunyuanVideo-1.5。
- 来源: https://arxiv.org/abs/2505.04512 https://github.com/Tencent-Hunyuan/HunyuanCustom https://hunyuancustom.github.io/

## HunyuanVideo-1.5  (腾讯混元)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: 推理 14GB；LoRA 训练 24GB 级
- 是什么: 轻量化视频基座（8.3B），SSTA（选择性滑动分块注意力）+ 字形感知双语文本编码 + 视频超分网络。
- 关键事实:
    * 2025-11-21 开源；2025-12-05 放出训练代码与 LoRA 微调脚本
    * 8.3B 参数
    * 消费级 GPU 14GB 显存可推理
    * Apache-2.0，权重/推理代码/训练配方全公开
    * 技术报告 arXiv 2511.18870
- 长片作用: 本地角色视频 LoRA 的最低成本基座。14GB 推理 + 官方 LoRA 脚本意味着可以在 4090 级卡上做「一个角色一个 LoRA」的批量训练，适合做分镜预览（proxy render），终片再上 Wan 14B / 闭源 API。
- 来源: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 https://huggingface.co/tencent/HunyuanVideo-1.5 https://arxiv.org/html/2511.18870v1

## Helios  (北大袁粒组 PKU-YuanGroup)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: ~6GB（group offload）到 80GB（四模型并置）；H100 实时
- 是什么: 实时长视频生成模型：分块自回归（33 帧/块）生成分钟级视频。
- 关键事实:
    * 2026-03-04 放出代码与权重
    * 14B，三档：Base / Mid / Distilled
    * 单卡 H100 19.5 FPS（社区报告最高 20.89 FPS）；单张昇腾 NPU ~10 FPS
    * 标准分辨率 640×384
    * 1452 帧（44 块）≈ 24fps 下 60 秒 / 16fps 下 90 秒
    * group offloading 下 ~6GB 显存；80GB 卡可同时装四个 14B 模型
    * Apache-2.0
- 长片作用: 分镜预览工位的首选：640×384 实时出分钟级样片，导演可以先看整段节奏再决定哪几镜值得上 720P/闭源 API 渲。但官方文档未强调身份一致性机制，块间角色漂移需要外部锚点补，不能当终片管线。
- 来源: https://github.com/PKU-YuanGroup/Helios

## Gloria: Consistent Character Video Generation via Content Anchors  (中科大（Yuhang Yang, Wei Zhai, Yang Cao, Zheng-Jun Zha 等）)
- 类别/成熟度: technique / **research** | 许可: 未公开 | 成本: 未披露
- 是什么: 用一组紧凑的「锚帧」（全局锚 / 视角锚 / 表情锚）表征角色外观，配 Superset Content Anchoring（防复制粘贴）与 RoPE as Weak Condition（区分多个锚参考）。CVPR 2026 主会。
- 关键事实:
    * arXiv 2603.29931，2026-03-31 提交
    * CVPR 2026 Main Conference 接收
    * 声称可生成超过 10 分钟无明显身份漂移的角色视频
    * 三类锚帧：global anchor / viewpoint anchor / expression anchor
    * 项目页给出主观用户研究对比图，未在页面公开 ArcFace/CLIP-I/DINO 数值
    * 代码/权重发布状态未在摘要与项目页确认
- 长片作用: 这是本次调研里与「2-8 分钟长片」最对口的一篇。它的「锚帧集合」在工程上就是用户说的『定妆图矩阵』——论文给出了矩阵该怎么切分（全局/视角/表情三类）以及为什么需要 superset（否则模型直接复制粘贴参考图，导致表演僵硬）。即使权重不放，这个分类法应当直接搬进角色圣经的资产结构。
- 来源: https://arxiv.org/abs/2603.29931 https://yyvhang.github.io/Gloria_Page/

## GroundShot: Entity-Grounded Shot Scheduling  (未在摘要中披露)
- 类别/成熟度: technique / **research** | 许可: 未公开 | 成本: 训练自由，开销等同基座推理 + 检索
- 是什么: 训练自由（training-free）的多镜头长视频一致性框架：在线从已接受的镜头里建「实体级视觉记忆」，生成下一镜前先检索合适的参考。附 GroundBench。
- 关键事实:
    * arXiv 2606.20799，2026-06-18 提交，2026-07-20 修订
    * 模型无关（model-agnostic），无需改模型、无需训练
    * 核心论点：首次出现的实体外观质量决定了后续所有镜头的一致性上限
    * GroundBench：实体级一致性评测，带受控挑战维度
    * 代码可用性未在摘要中说明
- 长片作用: 直接就是本项目分镜系统的调度层设计图。两条可落地结论：(1) 第一镜的角色外观要花最多算力抠到最好，它是整片一致性的天花板；(2) 不要每镜都用同一张定妆图，而要维护一个随生成累积的实体记忆库，按当前镜的视角/光位检索最接近的一帧做参考——这正是『双锚推理』里『每镜参考图』该怎么选的答案。
- 来源: https://arxiv.org/abs/2606.20799 https://arxiv.org/pdf/2606.20799

## FilmWeaver / STAGE / DreamShot（多镜头叙事生成三件套）  (分别为不同学术组；STAGE 与 DreamShot 均 CVPR 2026)
- 类别/成熟度: technique / **research** | 许可: 未公开 | 成本: 未披露
- 是什么: 三条互补的多镜头一致性路线：FilmWeaver 用双级缓存（Shot Cache 长期概念记忆 + Temporal Cache 镜内连贯）的自回归扩散；STAGE 显式预测「起止帧对」组成的结构化 storyboard 当视觉锚；DreamShot 用视频扩散先验直接生成多镜头分镜图。
- 关键事实:
    * FilmWeaver arXiv 2512.11274
    * STAGE: CVPR 2026 论文（Zhang et al., Storyboard-Anchored Generation for Cinematic Multi-shot Narrative）
    * DreamShot: CVPR 2026, Personalized Storyboard Synthesis with Video Diffusion Prior
    * 三者均为方法论文，未见生产级权重发布
- 长片作用: STAGE 的「起止帧对」是长片拼接最实用的工程抽象：每镜先定死首帧和尾帧，中间交给视频模型插值，跨镜衔接处用上一镜尾帧当下一镜首帧参考——这条在 Wan I2V / VACE 上今天就能手工实现，不需要等权重。FilmWeaver 的 Shot Cache 概念等价于 GroundShot 的实体记忆库。
- 来源: https://arxiv.org/pdf/2512.11274 https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_STAGE_Storyboard-Anchored_Generation_for_Cinematic_Multi-shot_Narrative_CVPR_2026_paper.pdf https://en.papernotes.org/CVPR2026/video_generation/dreamshot_storyboard_synthesis/

## ContextAnyone  (Ziyang Mai, Yu-Wing Tai)
- 类别/成熟度: technique / **research** | 许可: 未公开 | 成本: 未披露
- 是什么: 把参考图当「显式保留的外观锚」而非条件信号：参考图与目标视频在同一 DiT 里联合重建，非对称信息流防止噪声视频特征污染参考分支，Gap-RoPE 分离参考与视频 token。
- 关键事实:
    * arXiv 2512.07328，2025-12-08 提交，2026-09-01 修订
    * 评测基准基于 OpenVid-HD 构建
    * 评测维度：身份一致性、细粒度外观一致性、运动特性保持
    * 代码/权重未在页面确认
- 长片作用: 「非对称信息流」这一点直接回答了双锚冲突：当角色 LoRA 和每镜参考图同时在场时，最大的失败模式是去噪中期参考分支被视频 latent 反向污染，导致后半段镜头外观塌回基座先验。ContextAnyone 的做法是单向阻断——工程上对应在 ComfyUI 里把参考分支的 attention 设为只读（或对参考 token 做 detach），值得在自建管线里试。
- 来源: https://arxiv.org/abs/2512.07328 https://arxiv.org/pdf/2512.07328

## AnyID: Ultra-Fidelity Universal Identity-Preserving Video Generation  (未在摘要中披露)
- 类别/成熟度: technique / **research** | 许可: 未公开 | 成本: 未披露
- 是什么: 可扩展的全参考架构 + 主参考生成范式（differential prompts），异构输入（人脸、半身照、视频）统一处理；大规模数据 + 基于人类偏好标注（身份保真度与提示可控性）的 RL 微调。
- 关键事实:
    * arXiv 2603.25188，2026-03-26 提交
    * 输入支持 faces / portraits / videos 混合异构参考
    * 训练含 RLHF 阶段，奖励维度为 identity fidelity + prompt controllability
    * 基座模型、指标数值、代码权重均未在摘要页披露
- 长片作用: 值得关注的是「differential prompts + primary reference」范式：多张参考图不是平权平均（那会导致外观塌成均值脸），而是选一张主参考、其余用差分提示描述差异。这解决了定妆图矩阵直接全塞进去导致的「平均脸」问题。但无权重无数值，暂列观察名单。
- 来源: https://arxiv.org/abs/2603.25188 https://arxiv.org/pdf/2603.25188

## EverAnimate: Minute-Scale Human Animation via Latent Flow Restoration  (未披露)
- 类别/成熟度: technique / **research** | 许可: 未公开 | 成本: 未披露
- 是什么: Persistent Latent Propagation + Restorative Flow Matching，配轻量 LoRA 微调，做分钟级人物动画。
- 关键事实:
    * arXiv 2605.15042，2026-05-14 提交
    * 实测展示到 90 秒
    * 10 秒：PSNR +8%、SSIM +7%、LPIPS -22%、FID -11%
    * 90 秒：PSNR/SSIM +15%、LPIPS -32%、FID -27%
    * 项目页 everanimate.github.io，代码/权重未确认
    * 基座模型、分辨率未披露
- 长片作用: 数据点价值：它量化了长时长退化曲线——10 秒时 baseline 还能用，90 秒时 LPIPS 退化到需要 32% 的修复幅度。这为产线设定了硬指标：单镜超过 ~15 秒后，纯生成路线的外观退化会超过人眼容忍，必须切镜或引入锚点刷新。
- 来源: https://arxiv.org/pdf/2605.15042 https://arxiv.org/abs/2605.15042

## DreamActor-M1 / M2.0  (字节跳动)
- 类别/成熟度: closed-api / **vaporware** | 许可: 无开放权重 | 成本: 无本机部署路径；API 计费
- 是什么: 混合引导的人物图像动画 DiT：驱动帧提取骨架 + 头球（head sphere）编码为姿态 latent，3D VAE 编码视频 latent，人脸表情另走 face motion encoder。M2.0 为免姿态（pose-free）运动迁移。
- 关键事实:
    * arXiv 2504.01724（M1，2025-04）
    * M1 与 M2.0 均未发布权重——论文 + API 模式
    * 仅通过字节官方 API 与接入平台访问
    * 主打单张照片驱动面部、身体、微表情（眨眼、唇颤）
- 长片作用: 对本项目只有方法论价值（head sphere 做头部姿态解耦这个技巧可以抄到 Wan-Animate 的预处理里）。作为产线组件必须判定为不可用：无权重 = 不能本机 LoRA = 不满足角色定制需求。任何宣称「接入 DreamActor」的服务都是套壳 API。
- 来源: https://arxiv.org/html/2504.01724v1 https://grisoon.github.io/DreamActor-M1/ https://huggingface.co/papers/2504.01724

## StableAnimator / StableAnimator++ / UniAnimate(-DiT) / Animate-X(++)  (分别为复旦+微软等、阿里达摩院、阿里)
- 类别/成熟度: open-weights / **usable** | 许可: 各仓库不同，多为研究许可，商用需逐个核对 | 成本: SVD / DiT 基座级，16-40GB 不等
- 是什么: 姿态驱动人物动画的三条开源路线：StableAnimator 走专用人脸编码器 + ID 感知适配；UniAnimate(-DiT) 走精确人体骨架对齐；Animate-X(++) 走增强运动表征以支持非人形角色，++ 版加动态背景。
- 关键事实:
    * StableAnimator arXiv 2411.17697；StableAnimator++ arXiv 2507.15064（解决姿态错位与面部畸变）
    * Animate-X arXiv 2410.10306；Animate-X++ arXiv 2508.09454（通用角色 + 动态背景）
    * StableAnimator 在两个数据集上 CSIM 超过 UniAnimate 36.9% / 45.8%
    * UniAnimate-DiT 强依赖与人体关节精确对齐的骨架，对拟人化/非人形角色失效
    * Animate-X 用户研究：身份保持 98.5%、时序一致 93.4%、视觉质量 95.8%（主观，非客观指标）
- 长片作用: 工位分流：真人比例的虚构角色 → UniAnimate-DiT（骨架对齐最准）；风格化/非人比例角色（Q 版、兽人、高挑二次元）→ Animate-X++（骨架不匹配时唯一能用的）；对脸部 ID 要求最高的近景 → StableAnimator++（CSIM 领先明显）。但这一整代已被 Wan-Animate-2 在画质上全面超过，建议仅在 Wan-Animate 失效的角色类型上作备选。
- 来源: https://arxiv.org/pdf/2411.17697 https://arxiv.org/pdf/2507.15064 https://arxiv.org/pdf/2410.10306

## Meta Movie Gen / PT2V  (Meta AI)
- 类别/成熟度: closed-api / **vaporware** | 许可: 无开放权重 | 成本: 无部署路径
- 是什么: 30B 参数视频基座，PT2V（Personalized Text-to-Video）分支以单张参考图条件化，生成保持该人物身份的视频。
- 关键事实:
    * 论文 2024-10 发布（arXiv 2410.13720）
    * Movie Gen Video 30B 参数；PT2V 在其上加身份条件层
    * 权重从未公开发布
    * 2026 年以产品形态落在 Instagram / Facebook / Messenger 内
    * 无独立 API 供第三方产线调用
- 长片作用: 对本项目零可用性。论文里的 PT2V 训练配方（身份条件层如何与文本条件解耦）仍是值得参考的设计文档，但作为产线工位不存在。「Movie Gen 的角色注入」目前只能当学术引用，不能当技术选型。
- 来源: https://ai.meta.com/research/movie-gen/ https://huggingface.co/papers/2410.13720 https://ai.meta.com/static-resource/movie-gen-research-paper

## Seedance 2.0 / 2.5（字节，闭源 API）  (字节跳动 Seed)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API | 成本: 按 API 计费（2.5 定价官方公告未披露）
- 是什么: 多模态视频生成：文本 + 图 + 视频 + 音频统一输入，单次生成内含多镜头切分。2.5 为 2026-07-31 发布的新版。
- 关键事实:
    * Seedance 2.0：2026-02 发布，单次最长 15 秒，多模态参考上限 9 图 + 3 视频 + 3 音频（合计 12 文件）
    * Seedance 2.0 后续更新支持原生 4K 输出 + 10-bit 色深
    * Seedance 2.5：2026-07-31 发布，单次一镜到底 30 秒（不拼接），支持多轮续接成分钟级
    * Seedance 2.5 官方口径参考上限：30 图 + 10 视频 + 10 音频；CineD 口径为「最多 50 个多模态参考素材（含图/视频/音频/剧本/风格指南）」
    * 2.5 新增 3D 白模 blockout 锁镜头位与构图（previs 工位）
    * 2.5 声称提示遵循度较 2.0 提升约 20%，新增局部编辑（保留表演与光照重绘选区）
    * 2.5 先在即梦 / 豆包 Pro 上线，API 经 BytePlus ModelArk
    * 内容策略：first_frame_url 禁止含真实人脸图像
    * 权重不开放，无法本机 LoRA
- 长片作用: 用户给的硬约束需要更新两条：单段上限已从 15 秒放宽到 30 秒（2.5），参考上限已从 9+3+3 放宽到 30+10+10。这把 2-8 分钟成片从「至少 16-32 镜」压到「8-16 镜」，正好落在用户设定的 8-12 镜区间。3D 白模 blockout 是长片最被低估的一致性工具——锁死机位后，跨镜差异只剩角色外观一个变量。但「无本机 LoRA」这条硬约束依然成立，角色一致性只能靠参考素材包 + 多轮续接的内置一致性，没有权重级锚点。
- 来源: https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5 https://seed.bytedance.com/en/seedance2_0 https://www.cined.com/bytedance-seedance-2-5-api-goes-live-30-second-single-shot-clips-50-reference-inputs-and-3d-camera-blockouts/

## Kling 3.0 / Elements 多主体参考  (快手 可灵)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API | 成本: 按 API 计费
- 是什么: 从纯视频模型转为统一多模态系统。Elements 功能做参考驱动生成：上传参考图（或参考视频）锁定角色，模型抽取外观（3.0 Omni 起还抽声音）在新场景复现。
- 关键事实:
    * Kling 3.0 于 2026-02-05 发布（快手官方 IR 新闻稿）
    * 单次时长上限 15 秒，原生多语种/方言音频生成
    * Elements 每次生成接受最多 4 张参考图
    * Elements 3.0（Kling Video O1 起）支持上传角色短参考视频，同时抽取外观与声音
    * 早前 Kling 已全球上线「多图参考」（multi-image reference）功能，专门针对一致性问题
    * Motion Control 3.0 针对复杂多角度运动下的面部身份稳定性优化
    * 权重不开放
- 长片作用: Elements 的「参考视频抽取外观+声音」是闭源侧对长片最对口的能力：一次录制角色 bible 视频，全片复用，声画同源。但 4 图上限意味着无法塞完整定妆图矩阵（通常需 12-24 张），跨镜多角度时仍会漂。相比 Seedance 2.5 的 30 图上限，Kling 在一致性锚点容量上明显落后，建议作为风格备选而非主力。
- 来源: https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-launches-30-model-ushering-era-where-everyone-can-be https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-kling-ai-unveils-multi-image-reference-feature-further/ https://kling.art/model

## Vidu Q2 Reference-to-Video  (生数科技 Vidu)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API | 成本: 按 API 计费（各转售商定价不一）
- 是什么: 参考到视频：最多 7 张参考图（人脸、手势、场景、道具分别锚定），Multiple-Entity Consistency 保证各元素互不串味。
- 关键事实:
    * Vidu Q2 Reference Pro 于 2026-01-27 全球上线
    * 参考图上限 7 张
    * 同族还有 reference-to-image（Q2）做定妆图/分镜图生成
    * 可经 WaveSpeedAI / Atlas Cloud / Eachlabs / ComfyUI 官方 API 节点调用
    * 权重不开放
- 长片作用: 「多实体分别锚定」这一点对本项目的服装漂移最有价值：把角色脸、服装、道具当三个独立实体分别给参考图，而不是一张全身定妆图打包。Vidu 的 reference-to-image 还能反过来用作定妆图矩阵的批量生产工具（同角色多视角/多光位）。7 图上限居中，不如 Seedance 2.5，优于 Kling。
- 来源: https://www.prnewswire.com/news-releases/vidu-launches-q2-reference-to-video-pioneering-a-new-era-of-high-consistency-and-creative-control-302590002.html https://www.vidu.com/ai-reference-to-video https://comfy.org/workflows/api_vidu_q2_r2v-c1956ef24421/

## OmniHuman 1.5  (字节跳动（经 BytePlus）)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API | 成本: $0.16/秒 ≈ 每 15 秒镜 $2.4；2-8 分钟全对白片约 $19-77（仅此工位）
- 是什么: 单图 + 音频 + 文本提示生成电影级数字人表演视频。
- 关键事实:
    * BytePlus 官方 API $0.16/秒
    * 音频时长须 < 60 秒；官方建议 ≤ 15 秒，超出会质量退化
    * 可经 Replicate / fal / WaveSpeedAI / Kie.ai 等多家转售
    * SkyReels-V3 自评把 OmniHuman 1.5 的 A/V Sync 记为 8.18 对 8.25（SkyReels 略低）
    * 权重不开放
- 长片作用: 对白工位的闭源标杆。定价可算：8 分钟全片纯 OmniHuman 约 $77，成本可接受，但「建议 ≤15 秒」意味着 8-12 镜仍需分段，跨段身份靠同一张参考图维持——与开源 Wan2.2-S2V 的差距主要在唇形精度而非一致性。注意：单图输入意味着没有定妆图矩阵的位置，侧面/背面镜会硬漂。
- 来源: https://www.byteplus.com/en/product/OmniHuman https://replicate.com/bytedance/omni-human-1.5 https://fal.ai/models/fal-ai/bytedance/omnihuman/v1.5

## musubi-tuner（角色 LoRA 训练器）  (kohya-ss)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0（混合） | 成本: 12GB（图）/ 24GB（视频）起；4090 可全流程
- 是什么: 2026 年视频/图像 LoRA 训练事实标准，Wan 系专门优化，带 FP8 与 block-swap 省显存。
- 关键事实:
    * 支持架构：HunyuanVideo、HunyuanVideo 1.5、Wan2.1/2.2、FramePack、MiniMax-H3；图像侧 FLUX.1 Kontext、FLUX.2、Qwen-Image、Z-Image、HiDream-O1、Kandinsky 5、Ideogram 4、Krea 2
    * 显存：图像训练建议 ≥12GB，视频训练建议 ≥24GB；12GB 卡需降到 960×544 以下 + block swap + fp8
    * Apache-2.0 为主（HunyuanVideo 相关代码循其原许可）
    * 训练前需用脚本预缓存 latent 与 text-encoder 输出
    * dataset.toml 支持多 dataset block，用 num_repeats 混合图像集与视频集
- 长片作用: 两段式流程的执行器。关键能力是「一个 config 里同时挂图像数据集和视频数据集，用 num_repeats 配平」——这正是两段式（图 LoRA 打底 → 视频 LoRA）可以合并成单阶段联合训练的技术前提，省一半算力。
- 来源: https://github.com/kohya-ss/musubi-tuner https://huggingface.co/datasets/John6666/forum1/blob/main/wan22_lora_training.md

## Wan 2.2 角色 LoRA 实操超参（社区收敛值）  (社区（musubi-tuner / ai-toolkit 生态）)
- 类别/成熟度: technique / **production** | 许可: N/A | 成本: 12-24GB 显存；单角色训练数小时级
- 是什么: 经社区反复验证的角色 LoRA 训练配置区间。
- 关键事实:
    * 数据集：最低可用 15 张；推荐 25-40 张。低于 15 张时 LoRA 无法区分「人物特征」与「单张照片的光照/机位特征」，会过拟合到拍摄条件
    * 分辨率：训练常打到 544×960（显存效率），基座原生 720P@24fps
    * 视频帧桶：target_frames = [17, 33, 65]（24fps 下约 0.7 / 1.4 / 2.7 秒）；视频片段常用 ~2 秒 @ 544×960
    * rank/alpha：起手 r=32 / alpha=16；复杂身份（纹身、特殊瞳色、复杂发型）推 r=64 / alpha=32；另有 r=16 管身份 / r=32 管风格的口径
    * 学习率：2e-4 为强基线；偏运动的训练可到 3e-4
    * discrete_flow_shift ≈ 2-3 收紧身份，调高则增加泛化
    * batch size：16GB 卡用 1，24GB 用 1-2
    * Wan2.2 MoE 必须 high-noise 与 low-noise 两份 LoRA 都训都挂（角色/风格 LoRA 两者都关键）
    * 显存底线：图像 LoRA ~12GB，视频 LoRA ~24GB；16GB 上做 I2V LoRA 靠 fp8 + block-swap 可行
    * LoKr 在小/中型角色数据集上比 LoRA 更快到达可用相似度并保留更多细节，代价是算力与显存更高
- 长片作用: 这是角色圣经资产化的量化落点：25-40 张定妆图、544×960 起训、r=32/a=16 起步、flow_shift 2-3 收身份。低于 15 张必翻车这条是最重要的红线——很多团队用 8 张定妆图训完发现「换个光位就不像了」，根因就在这里。
- 来源: https://huggingface.co/datasets/John6666/forum1/blob/main/wan22_lora_training.md https://github.com/kohya-ss/musubi-tuner https://wavespeed.ai/blog/posts/blog-wan-2-2-lora-training-settings/

## 图像侧角色打底基座：FLUX.2 [dev] / Z-Image-Turbo / Qwen-Image  (Black Forest Labs / 阿里 Tongyi-MAI / 阿里)
- 类别/成熟度: open-weights / **production** | 许可: FLUX.2-dev 非商用 / Z-Image Apache-2.0 / Qwen-Image Apache-2.0 | 成本: Z-Image 16GB 推理、24GB 训练；FLUX.2 训练 80GB；Qwen-Image 24-32GB 训练
- 是什么: 两段式流程第一段（定妆图矩阵生产 + 图像角色 LoRA）的候选基座。
- 关键事实:
    * FLUX.2 [dev]：2025-11-25 发布，32B flow-matching transformer，FLUX.2-dev Non-Commercial License（商用受限，这是选型红线）；训练建议 80GB 显存
    * Z-Image-Turbo：Tongyi-MAI，6B，Apache-2.0（可商用、可微调、可分发衍生），8 NFE 蒸馏，16GB 消费卡可跑，H800 亚秒级
    * Qwen-Image：20B，社区报告 rank 16 / lr 1e-4 / batch 1 / 2000 步 / 量化基座 可在单张 24GB 卡训；有报告称需 ≥32GB
    * 训练器：ostris/ai-toolkit（MIT）与 musubi-tuner（Apache-2.0）均已覆盖这三者
    * 角色 LoRA 数据量口径：Z-Image 20-50 张多角度；FLUX.2 官方口径 20-1000 张
- 长片作用: 选型结论明确：本项目要商用，FLUX.2 [dev] 的非商用许可直接出局（只能用作内部研发/对比基线）。打底基座应选 Z-Image-Turbo（Apache-2.0 + 16GB + 6B 训得快）或 Qwen-Image（Apache-2.0 + 20B 质量更高）。这一段负责把「角色长什么样」固化成可复用资产，再迁到 Wan 视频 LoRA。
- 来源: https://github.com/black-forest-labs/flux2 https://bfl.ai/blog/flux-2 https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

## PuLID / InstantID / IP-Adapter FaceID（图像侧身份适配器）  (字节（PuLID）/ InstantX（InstantID）/ Tencent AI Lab（IP-Adapter）)
- 类别/成熟度: framework / **production** | 许可: 各项目不同（PuLID Apache-2.0；InstantID Apache-2.0 但依赖 InsightFace 非商用模型，商用需核） | 成本: 图像侧适配器，显存开销 <2GB
- 是什么: 免训练（tuning-free）人脸身份注入适配器，视频侧的对应物是 Stand-In / ConsisID / Phantom 这一类。
- 关键事实:
    * 2026 年社区共识口径：InstantID 更稳、PuLID 保真度更高；PuLID 在 FLUX/SDXL 写实流程上人脸保真优于 InstantID
    * InstantID 到 2026 年仍是 SDXL-only，未迁到 FLUX 系
    * PuLID 有官方 FLUX.1 [dev] 集成版本
    * 人脸 LoRA 通常保留嵌入式方法会抹平的纹理与微观细节——两者是互补而非替代
    * 视频侧等价物参数量参考：Stand-In 在 Wan2.1 上用 rank 128 LoRA，153M 可训练参数（基座 1%）
- 长片作用: 回答「IP-Adapter/PuLID/InstantID 在视频侧的等价物」：一一对应关系是 IP-Adapter→VACE/SkyReels R2V 的参考图分支（通用主体）、PuLID/InstantID→Stand-In（专门人脸、1% 参数、免重训）、FaceID LoRA→角色视频 LoRA。产线里图像段用 PuLID 生成定妆图矩阵，视频段用 Stand-In 把同一张脸锁进每镜，两侧都用 AntelopeV2/ArcFace 系特征，嵌入空间一致，衔接损耗最小。
- 来源: https://arxiv.org/pdf/2401.07519 https://myaiforce.com/flux-pulid-vs-ecomid-vs-instantid/ https://github.com/WeChatCV/Stand-In

## OpenS2V-Nexus / OpenS2V-Eval（主体到视频评测基准）  (PKU-YuanGroup 等)
- 类别/成熟度: service / **usable** | 许可: 数据集与代码开放（详见仓库） | 成本: 评测开销，无训练成本
- 是什么: 主体到视频（S2V）的细粒度评测基准与百万级数据集。
- 关键事实:
    * arXiv 2505.20292（2025-05-26 提交，2025-06-03 定稿）
    * OpenS2V-5M：500 万条 720P 主体-文本-视频三元组
    * OpenS2V-Eval：7 大类共 180 条提示，含真实与合成测试数据，已评测 18 个代表性 S2V 模型
    * 三个自动指标：NexusScore（主体一致性）、NaturalScore（自然度）、GmeScore（文本相关性）
    * 论文本身未把这三个指标与 ArcFace / CLIP-I / DINO 做直接对照
- 长片作用: 产线的回归测试台。建议把自己的 8-12 镜分镜固定成一套私有 prompt set，套用 NexusScore/NaturalScore/GmeScore 三元组做每次模型/LoRA 升级的 A/B 门禁，而不是靠肉眼。NaturalScore 尤其重要——它能抓到「身份很像但表演僵硬」这个 copy-paste 失效模式（Gloria 论文里的 superset anchoring 要解决的正是它）。
- 来源: https://arxiv.org/abs/2505.20292 https://arxiv.org/pdf/2505.20292

## 身份一致性量化评测协议（ArcFace / CLIP-I / DINOv2 三专家）  (综合自 HunyuanCustom、Mixture of Facial Experts、ConsID-Gen 等论文的通行做法)
- 类别/成熟度: technique / **production** | 许可: N/A | 成本: 评测成本可忽略（CPU/单卡）
- 是什么: 三个指标测三件不同的事，不可互相替代：ArcFace 余弦测姿态不变的结构身份（眼鼻口几何）；CLIP 测高层语义属性（发型、表情、服装类别）；DINOv2 测细粒度纹理与结构边缘。
- 关键事实:
    * ArcFace 用法：对参考人脸与生成视频每一帧分别抽 embedding，逐帧余弦相似度再取均值（论文中常记作 CSIM / Face-Sim）
    * CLIP-I 在不同论文里有两种定义：帧 embedding 与参考图 embedding 的余弦（图像一致性），以及帧 embedding 与文本 prompt 的余弦（提示遵循度）——报数时必须写清是哪一种
    * DINO-I 同时用于「帧 vs 参考图」与「两段视频的全帧对」两种比较，后者用于跨镜一致性
    * 实践阈值示例：有工作在身份连贯性分析中丢弃平均相似度低于 0.6 的帧嵌入
    * StableAnimator 在两个数据集上 CSIM 超过 UniAnimate 36.9% / 45.8%，说明同一维度下不同方法差距可达数十个百分点，指标是有判别力的
- 长片作用: 直接对应用户问的三类漂移：脸漂用 ArcFace 余弦（逐帧 + 跨镜两套）；发型漂用 CLIP-I（发型是高层语义属性，ArcFace 对它几乎不敏感）；服装漂用 DINOv2（纹理与结构边缘，CLIP 只能分辨「还是夹克」但分不出「口袋从两个变三个」）。三者必须分开报，合成一个总分会掩盖问题工位。
- 来源: https://arxiv.org/html/2508.09476 https://arxiv.org/pdf/2505.04512 https://arxiv.org/html/2602.10113v1

## Garments2Look（服装一致性的现实上限）  (ArtmeScienceLab 等，CVPR 2026)
- 类别/成熟度: open-weights / **research** | 许可: 以仓库为准 | 成本: 数据集与评测，非生成模型
- 是什么: 整套服装（含配饰）级别的多参考虚拟试穿数据集与评测。
- 关键事实:
    * CVPR 2026，arXiv 2603.14153，GitHub ArtmeScienceLab/Garments2Look
    * 每对样本含 3-12 张参考服装图
    * 论文明确结论：现有方法在「整套服装试穿」与「推断正确的层次与搭配」上仍然失败，出现错位与伪影
    * 开源基线参考：IDM-VTON 被社区列为自建虚拟试穿的最佳开源起点
    * 视频侧相关：Dynamic Try-On（arXiv 2412.09822）、From Mannequin to Human（arXiv 2510.16833）
- 长片作用: 给服装漂移设定现实预期：到 2026 年 9 月，「多层次整套服装跨镜保持」学术上仍未解决。产线对策不是等模型，而是降低难度——角色服装设计阶段就避免复杂层次与小面积重复纹样，把服装拆成可独立锚定的 element（SkyReels-A2 / Vidu Q2 多实体思路），并接受服装是三类漂移里最需要后期 MV2V 修补的一类。
- 来源: https://github.com/ArtmeScienceLab/Garments2Look https://arxiv.org/pdf/2603.14153 https://arxiv.org/pdf/2412.09822

### 建议
## 一、先修正两条「硬约束」（核实结果）

用户给的约束里有两条已经过期，会直接影响分镜系统的镜数规划：

1. **Seedance 单段上限已放宽**。Seedance 2.5（2026-07-31 发布）支持**单次一镜到底 30 秒、不拼接**，并支持多轮续接到分钟级且官方声明在续接中保持主角/环境/节奏一致性。2.0 仍是 15 秒，但已支持原生 4K + 10-bit。
2. **参考素材上限已放宽**。2.0 是 9 图 + 3 视频 + 3 音频（合计 12 文件）；**2.5 官方口径为 30 图 + 10 视频 + 10 音频**（CineD 口径「最多 50 个多模态参考素材，含剧本与风格指南」）。
3. 新增一条对长片极有价值的能力：**2.5 支持 3D 白模 blockout 锁机位与构图**（previs 工位），以及局部编辑（保留表演与光照重绘选区）。

其余约束核实无误：权重不开放、不能本机 LoRA、自带对白声、官方通道有内容审核（且明确禁止 first_frame_url 含真实人脸——这一条与你的「不碰真人肖像」策略天然吻合）。

**规划含义**：8 分钟成片，用 2.5 的 30 秒单段，**16 镜**即可覆盖；配合多轮续接，8-12 镜是可达的。你原来按 15 秒规划的镜数可以砍半，而镜数减半本身就是最有效的一致性改善手段。

---

## 二、工位映射表（谁顶哪个工位）

| 工位 | 首选 | 备选 | 判据 |
|---|---|---|---|
| 角色定妆图矩阵生产 | Z-Image-Turbo (Apache-2.0, 6B, 16GB) + PuLID | Qwen-Image；Vidu Q2 reference-to-image | FLUX.2 [dev] 是**非商用许可**，直接出局 |
| 图像角色 LoRA（打底） | musubi-tuner / ai-toolkit on Z-Image 或 Qwen-Image | — | 12GB 起，几小时收敛 |
| 分镜预览（proxy） | Helios 14B（H100 实时 19.5FPS，640×384，分钟级） | HunyuanVideo-1.5（8.3B / 14GB） | 只看节奏不看脸 |
| 单镜主力渲染（开源） | Wan2.2-VACE-Fun-A14B + 角色 LoRA + Stand-In | SkyReels-V3-R2V-14B（1-4 参考图） | 720P，需 24-48GB |
| 单镜主力渲染（闭源上限） | Seedance 2.5（30 秒 / 30 图参考 / 3D blockout） | Vidu Q2 R2V（7 图多实体）、Kling 3.0 Elements（4 图） | 参考容量：Seedance > Vidu > Kling |
| 表演/对白镜 | Wan-Animate-2（2026-08，文本视角控制） | Wan2.2-S2V-14B（音频驱动，80GB）、OmniHuman 1.5（$0.16/s） | 用驱动视频锁死时域，只剩外观一个自由度 |
| 跨镜调度 | GroundShot 式实体记忆库（training-free，可自建） | STAGE 式首尾帧对锚定 | 两者今天就能在 ComfyUI 手工实现 |
| 成片修补 | VACE MV2V 掩码重绘（只修脸/只修服装，不重渲整镜） | — | 唯一能局部修而不破坏时域的开源手段 |
| 回归门禁 | ArcFace CSIM + CLIP-I + DINOv2 三指标分报 | OpenS2V-Eval 的 NexusScore/NaturalScore/GmeScore | 见第五节 |

---

## 三、脸漂 / 发型漂 / 服装漂：三类问题三个解，不要用一个方案硬扛

这是本次调研最重要的结论：**这三类漂移的物理机制不同，最优解也不同，合在一个 LoRA 里训会互相拖后腿。**

### 1. 脸漂 —— 最优解：轻量身份适配器（Stand-In），不是 LoRA

- **为什么不是 LoRA**：人脸是低维强结构信号，LoRA 要同时学脸+发+衣，容量被稀释；且 LoRA 强度调高会连带锁死表情与光照，造成「像但死」。
- **Stand-In**（CVPR 2026，微信视觉，Apache-2.0）只训 **153M 参数（基座 1%）、仅用 2000 对数据**，支持 Wan2.1-14B-T2V 与 Wan2.2-T2V-A14B，且**官方明确兼容 VACE 与社区 LoRA**，有 ComfyUI 节点。这是目前唯一能「与角色 LoRA 并联而不打架」的人脸锚。
- 理论依据见 ConsisID 的频率分解：脸的低频（骨相）与高频（五官纹理）要分开注入。诊断法：**低频漂 = 骨相变了 → 参考锚不够强；高频漂 = 五官糊了 → 需要 face refiner 二次修**。
- 闭源侧对应物：Kling 的 Motion Control 3.0、Seedance 2.5 的多轮续接一致性。但它们不可调，漂了只能重摇。

### 2. 发型漂 —— 最优解：角色 LoRA 主力 + 定妆图矩阵覆盖视角

- 发型是**中频语义属性**：ArcFace 对它几乎不敏感（所以「脸很像但换了个发型」不会被 CSIM 抓到），身份适配器也管不住。
- 它只能靠 LoRA 把「这个角色的头发在各个角度长什么样」写进权重。这就是**定妆图矩阵必须覆盖多视角**的根本原因——后脑勺和侧后方 3/4 是发型漂的重灾区，而这两个角度在常规定妆图里最容易缺。
- Gloria（CVPR 2026）的**视角锚（viewpoint anchor）**正是为此设计。它宣称可生成超 10 分钟无明显漂移的角色视频，其锚帧分三类：全局锚 / 视角锚 / 表情锚 —— 这个三分法应当**直接搬进你的角色圣经资产结构**。
- 量化用 **CLIP-I**（帧 vs 参考图），不是 ArcFace。

### 3. 服装漂 —— 最优解：把服装当独立实体锚定 + 接受后期修补；同时在设计阶段降难度

- **坦率结论**：到 2026 年 9 月，「多层次整套服装跨镜保持」**学术上仍未解决**。CVPR 2026 的 Garments2Look 明确写：现有方法在整套试穿与推断正确层次/搭配上仍然失败，出现错位与伪影。
- 三条可落地对策：
  - **实体分离**：不要给一张全身定妆图，而是把脸 / 服装 / 关键道具作为**独立 element 分别给参考图**。SkyReels-A2 的 E2V 与 Vidu Q2 的 Multiple-Entity Consistency 都是这个范式。
  - **设计端降难度**：角色服装避免复杂层次、小面积重复纹样（格纹、细密刺绣、排扣）、以及依赖精确计数的元素（三个口袋、五颗纽扣）。这些是生成模型的已知失效点。
  - **接受修补工位**：服装是三类漂移里唯一必须留后期预算的。用 **VACE 的 MV2V 掩码重绘**只修服装区域，不重渲整镜，时域不被破坏。
- 量化用 **DINOv2**，不是 CLIP —— CLIP 只能判断「还是夹克」，DINOv2 才能抓到纹理与结构边缘的变化。

---

## 四、角色圣经资产化 + 两段式 LoRA：具体参数

### 4.1 定妆图矩阵的最小规格

结合 Gloria 的三类锚 + Wan LoRA 的数据量红线：

- **数量红线：最低 15 张，推荐 25-40 张**。低于 15 张时，LoRA 在数学上无法把「人物特征」和「单张照片的光照/机位特征」分离，必然过拟合到拍摄条件——这是「换个光位就不像了」的根因，也是最常见的翻车点。
- **视角**（对应 viewpoint anchor）：正面 / 侧面左右 / 3/4 左右 / 后侧 3/4 / 正后 ≈ **7 个视角**。正后与后侧 3/4 不能省，它们是发型漂的唯一训练信号。
- **光位**（对应泛化）：顺光 / 侧逆光 / 顶光 **3 个光位**。光位的作用不是增加信息，而是**防止 LoRA 把某个光位烧进身份**。
- **表情**（对应 expression anchor）：中性 / 微笑 / 皱眉或惊讶 **3 档**。
- 7 视角 × 不必全交叉，实际取 **7 视角（顺光中性） + 5 个补充光位/表情组合 × 若干视角 ≈ 28-35 张**，正落在推荐区间。
- **关键警告（Gloria 的 Superset Content Anchoring）**：如果矩阵图之间差异太小、覆盖太窄，模型会退化成**直接复制粘贴参考图**，表现为「身份极像但表演僵硬、眨眼与口型消失」。必须刻意保留姿态与表情的多样性。这个失效模式用 ArcFace 抓不到，要用 OpenS2V-Eval 的 **NaturalScore** 或肉眼查。
- **AnyID 的补充范式**：多张参考不要平权平均（会塌成「均值脸」），而应**选一张主参考 + 其余用差分提示描述差异**（primary reference + differential prompts）。

### 4.2 两段式流程（图 LoRA 打底 → 视频 LoRA）

**第一段（图像）**：
- 基座：Z-Image-Turbo（Apache-2.0, 6B, 16GB）或 Qwen-Image（Apache-2.0, 20B）。**不要用 FLUX.2 [dev]，它是非商用许可。**
- 训练器：ostris/ai-toolkit（MIT）或 musubi-tuner（Apache-2.0），两者 2026 年都已覆盖 FLUX.2 / Z-Image / Qwen-Image。
- 参考超参：rank 16 / lr 1e-4 / batch 1 / ~2000 步 / 量化基座，单张 24GB 卡可跑（Qwen-Image 20B）。
- 产出：一个稳定的图像角色 LoRA，用它**批量扩产定妆图矩阵到 100+ 张**（含更多视角/光位/服装状态），作为第二段的数据源。这才是两段式的真正收益——不是「迁移权重」，而是**用第一段解决第二段的数据瓶颈**。

**第二段（视频）**：
- 基座：Wan2.2（720P@24fps）或 HunyuanVideo-1.5（8.3B / 14GB，成本敏感时）。
- 训练器：musubi-tuner，需先预缓存 latent 与 text-encoder 输出。
- 超参收敛值：
  - **rank/alpha：起手 r=32 / alpha=16；复杂身份（纹身、异色瞳、复杂发型）推 r=64 / alpha=32**
  - **lr = 2e-4**（偏运动可到 3e-4）
  - **训练分辨率 544×960**（显存效率），基座原生 720P
  - **视频帧桶 target_frames = [17, 33, 65]**（24fps 下 0.7 / 1.4 / 2.7 秒），片段取 ~2 秒
  - **discrete_flow_shift ≈ 2-3 收紧身份**，调高增加泛化
  - batch：16GB 用 1，24GB 用 1-2
- **Wan2.2 MoE 必须训两份 LoRA（high-noise + low-noise）并配对加载**。只挂一份是产线最高频的坑：只挂 high 会「构图像、细节不像」，只挂 low 会反过来。
- **省算力做法**：musubi-tuner 的 `dataset.toml` 支持在**同一配置里挂图像数据集与视频数据集，用 `num_repeats` 配平**（图像管细节、视频管运动）。这使两段可以合并为单阶段联合训练，省掉一半算力。若你的角色外观已稳定，建议直接走这条。
- 显存底线：图像 LoRA ~12GB，视频 LoRA ~24GB；16GB 卡靠 fp8 + block-swap 也能做 I2V LoRA。
- **LoKr 替代**：在小/中型角色数据集上比 LoRA 更快到达可用相似度并保留更多细节，代价是算力与显存更高。若你只有 15-25 张图，LoKr 值得试。

### 4.3 双锚推理（角色 LoRA + 每镜参考图）：收益与冲突

**实际收益（分工清晰时才成立）**：
- 角色 LoRA 负责**跨镜不变量**：骨相、发型结构、体型、服装设计。
- 每镜参考图负责**当镜变量**：视角、光位、景别、服装状态（湿、破损、外套脱掉）。
- 再叠 Stand-In 负责**人脸几何**。三层各管一件事，互不重叠。

**三个真实冲突及对策**：

1. **参考分支被污染**。ContextAnyone（arXiv 2512.07328，2026-09 修订）指出的核心失效：去噪中期，视频 latent 会**反向污染参考分支**，导致镜头后半段外观塌回基座先验（表现为「前 3 秒像、后 5 秒不像」）。它的解法是**非对称信息流（单向阻断）+ Gap-RoPE 分离参考与视频 token**。工程近似：在 ComfyUI 里把参考分支 attention 设为只读，或对参考 token 做 detach。这是长镜头最值得优先解决的一条。

2. **LoRA 强度与参考图权重互相压制**。LoRA 强度调高会覆盖参考图带来的当镜变化（光位/表情被 LoRA 的训练条件反噬）。对策：LoRA 强度按用途分档——**近景对白镜降到 0.6-0.7 让表情自由，中远景升到 0.9-1.0 锁外观**；并在训练时用 3 个光位做去光照烧录。

3. **每镜参考图选错，反而加速漂移**。**不要每镜都用同一张原始定妆图**。GroundShot（arXiv 2606.20799，training-free，模型无关）的做法是：**在线从已接受的镜头里建实体级视觉记忆，生成下一镜前先按当前镜的视角/光位检索最接近的一帧做参考**。它还给出一条硬结论：**首次出现的实体外观质量决定了整片一致性的上限**——所以第一镜要花最多算力抠到最好。这条今天就能实现，不需要等权重。

---

## 五、量化评测方法（可直接做成 CI 门禁）

**三指标分开报，绝不合成总分**（合成会掩盖具体是哪个工位坏了）：

| 指标 | 测什么 | 怎么算 | 对应漂移 |
|---|---|---|---|
| **ArcFace 余弦 / CSIM** | 姿态不变的结构身份（眼鼻口几何） | 参考人脸与生成视频**逐帧**抽 embedding，逐帧余弦再取均值 | 脸漂 |
| **CLIP-I** | 高层语义属性（发型、表情、服装类别） | 帧 embedding vs 参考图 embedding 余弦 | 发型漂 |
| **DINOv2 / DINO-I** | 细粒度纹理与结构边缘 | 帧 vs 参考图；以及**两段视频的全帧对**（跨镜用） | 服装漂 |

**关键实施细节**：
- **CLIP-I 在文献里有两种定义**（帧 vs 参考图 = 图像一致性；帧 vs 文本 prompt = 提示遵循度）。内部报数必须写清是哪一种，否则无法跨实验比较。
- **两套尺度都要测**：(a) 镜内逐帧（抓时域退化）；(b) **跨镜全帧对**（抓分镜拼接处的跳变）。DINO-I 与 ArcFace 都支持「两段视频全帧对」比较，这是 8-12 镜产线真正该看的数。
- **阈值参考**：有工作在身份连贯性分析中**丢弃平均相似度低于 0.6 的帧嵌入**。可以把 0.6 当作「该帧不可用」的下界，但建议先在你自己的角色上标定基线——不同角色（写实 vs 风格化）的绝对值差异很大，**看相对退化曲线比看绝对值可靠**。
- **判别力已验证**：StableAnimator 在两个数据集上 CSIM 超过 UniAnimate 36.9% / 45.8%，说明同维度下不同方法差距可达数十个百分点，这些指标是有区分度的，不是噪声。
- **时长退化基线**：EverAnimate（arXiv 2605.15042）给出量化曲线——10 秒时 LPIPS 需修复 22%，**90 秒时升到 32%**。据此可设产线硬规则：**单镜超过约 15 秒后必须切镜或刷新锚点**，纯生成路线的外观退化会超过人眼容忍。
- **补充 NaturalScore**：OpenS2V-Eval（180 prompts / 7 类 / 已评 18 个模型）的 NexusScore（主体一致性）+ NaturalScore（自然度）+ GmeScore（文本相关性）三元组。**NaturalScore 专门抓「身份很像但表演僵硬」的 copy-paste 失效**，这是前三个指标结构性看不见的盲区。
- **落地建议**：把你的 8-12 镜固定成一套私有 prompt set，每次模型/LoRA 升级跑一遍，三指标 + NaturalScore 做 A/B 门禁。

---

## 六、最终架构建议：双轨 + 一个共享角色圣经

鉴于 Seedance 2.0/2.5 权重不开放、不能本机 LoRA，而它的单镜观感是你的目标，唯一自洽的架构是**双轨并行，共享同一套角色资产**：

- **轨 A（闭源，出终片）**：Seedance 2.5。用 30 图参考上限塞入定妆图矩阵的主子集 + 3D 白模 blockout 锁机位（锁死机位后跨镜差异只剩外观一个变量，这是最被低估的一致性工具）+ 30 秒单段 + 多轮续接。8 分钟片 ≈ 16 镜，可压到 8-12 镜。
- **轨 B（开源，出可控性）**：Wan2.2-VACE-Fun + 角色 LoRA + Stand-In。用途有三：(1) **生产定妆图矩阵与参考素材包**喂给轨 A；(2) **轨 A 审核拒绝或效果崩掉时的兜底渲染**；(3) **VACE MV2V 做终片局部修补**——这是轨 A 完全不具备的能力，服装漂必须靠它收尾。
- **共享层**：一套角色圣经（Gloria 三类锚结构：全局/视角/表情）+ 一个 GroundShot 式实体记忆库（随生成累积，按视角光位检索）。两轨都从这里取参考图，保证身份定义只有一个真相源。

**不要做的事**：不要指望 DreamActor-M1/M2.0（无权重，论文+API，任何宣称接入的都是套壳）、Movie Gen PT2V（权重从未发布，只活在 Meta 自家产品里）、以及各类宣称 Wan 2.5/2.6/2.7/3.0 开源的 SEO 站点（官方 Wan-Video GitHub 组织里只有 Wan2.1、Wan2.2、Wan-Animate-2、Wan-Dancer 四个生成模型仓库）。

### 存疑
- 【Wan 2.5/2.6/2.7/3.0 开源状态】多个 SEO 站点（wan27.org、kingy.ai、flowith.io、localaimaster.com）对 Wan 2.5/2.6/2.7/3.0 是否开源权重给出**互相矛盾**的说法（有说 2.7 于 2026-03 回归开源权重、有说 3.0 于 2026 初开源、有说 2.6 于 2025 末开源）。核实结果：官方 GitHub 组织 Wan-Video 下只有 Wan2.1、Wan2.2、Wan-Animate-2、Wan-Dancer、Wan-skills 五个仓库，**不存在 Wan2.5/2.6/2.7/3.0 仓库**；HF Wan-AI 组织最新模型为 Wan2.2-Animate-2-14B 系列。倾向判断：Wan 2.5 及以上为闭源商用 API，上述开源说法为 SEO 内容农场生成，不可采信。建议直接以 GitHub/HF 官方组织为准。
- 【SkyReels V4】多篇文章（wavespeed.ai blog、其他聚合站）称 SkyReels V4 于 2026-02-25 发布，为「首个开源的视频+音频双流 1080p/32fps 模型」。核实结果：SkyworkAI GitHub 组织下**无 SkyReels-V4 仓库**（最新为 SkyReels-V3，2026-01-30 更新），且 SkyReels-V3 仓库下存在标题为 'release SkyReels-V4？' 的 Issue #11（说明社区在催但未发布）。HF Skywork 组织下亦未见 V4 权重。倾向判断：**vaporware 或 API-only**，所谓「开源」表述无据。不要写进选型。
- 【Sora 2 Pro 的 persistent character_id】搜索结果（cinevva.com 等第三方指南站）称 Sora 2 Pro 有 persistent character_id 系统，可从短视频提取角色一次、后续数十个片段引用同一 ID 且身份不随时间退化（存为持久 embedding）。未找到 OpenAI 官方文档佐证该 API 参数名与行为。倾向判断：可能是对 Sora 'cameo' 功能的第三方转述或杜撰。若该机制属实，它是最接近本项目需求的闭源方案，值得单独核实 OpenAI 官方 API 文档。
- 【Gloria 的代码与权重】Gloria（CVPR 2026，arXiv 2603.29931）声称可生成超 10 分钟无明显身份漂移的角色视频，但 arXiv 摘要页与项目页（yyvhang.github.io/Gloria_Page/）**均未确认代码或权重发布**，也未公开 ArcFace/CLIP-I/DINO 的具体数值（只有主观用户研究图）。其「超 10 分钟」的声明与基座模型、分辨率、GPU 需求均未披露。方法论（三类锚帧分类）可采信并借鉴，但**不能当作可用组件**。
- 【AnyID 的基座与权重】AnyID（arXiv 2603.25188）摘要页未披露基座模型、指标数值、代码/权重可用性、时长与分辨率。其 RLHF 训练与 primary-reference + differential prompts 范式值得借鉴，但无法验证效果或可用性。
- 【Nano Banana Pro 的 14+ 参考图能力】第三方站点（selfielabstudio.com、nanobananapro.photo、aibanana.net）称 Nano Banana Pro 原生支持 14+ 张参考图、带 seed 控制与角度特定提示、并在 2026 基准中登顶。未找到 Google 官方文档佐证「14+ 参考图」这一具体数字。定妆图矩阵生产若考虑该工具，需先核实官方 API 文档的参考图上限。
- 【Kling 3.0 Elements 的精确规格】Kuaishou 官方 IR 新闻稿确认 Kling 3.0 于 2026-02-05 发布及「多图参考」功能全球上线，但「Elements 每次最多 4 张参考图」「Elements 3.0 可从参考视频同时抽取外观与声音」「Motion Control 3.0 优化面部身份稳定性」这几条来自第三方指南站（atlascloud.ai、oakgen.ai、vidmuse.ai），未在官方文档核实。选型前需查可灵官方 API 文档确认参考图上限。
- 【Seedance 2.5 参考素材上限的两个口径】字节官方 blog 写「up to 30 images, 10 video clips, and 10 audio clips」；CineD 写「最多 50 个多模态参考素材（图/视频/音频/剧本/风格指南）」。两者可能是「媒体文件上限 50 vs 分模态上限 30/10/10」的不同切法，也可能其一有误。实际配额需以 BytePlus ModelArk API 文档为准。Seedance 2.5 定价官方公告亦未披露。
- 【Stand-In-V2】GitHub 仓库于 2026-08-10 公告 Stand-In-V2 'coming soon'，截至 2026-09-19 检索未见权重发布。规划时应按 V1（153M / Wan2.1-14B-T2V + Wan2.2-T2V-A14B）能力做，不要押注 V2。
- 【GroundShot / FilmWeaver / STAGE / DreamShot / ContextAnyone / EverAnimate 的代码可用性】这六项多镜头/长时长一致性工作均未在摘要或项目页确认代码与权重发布。其中 GroundShot 自述为 training-free 且模型无关，理论上可按论文自行复现；其余需等待官方发布。均应按「方法论可借鉴、组件不可用」处理。
- 【Wan2.2-VACE-Fun-A14B 的官方来源】该模型的规格主要来自第三方 API 文档（docs.aimlapi.com）与教程站（stablediffusiontutorials.com），未从 ali-vilab/VACE 或 Wan-Video/Wan2.2 官方 README 直接核实其发布日期、双专家权重结构与四种控制模式的完整清单。上产线前需拉取 HF 官方模型卡确认。
- 【SkyReels-V3 的许可证条款】仓库标注为 'Skywork License'（非 Apache-2.0），具体商用限制（是否限制用户规模、是否要求署名、是否禁止特定用途）未在检索中获取。SkyReels-V3 是本次调研中长片能力最对口的开源模型族，**上产线前必须完整阅读 LICENSE.txt**。
- 【SkyReels-V3 自评基准数值】Reference Consistency 0.6698 / Visual Quality 0.8119 / A-V Sync 8.18 这组数字来自官方 README 自评，未见第三方复现。与 Kling 1.6（0.8034）、OmniHuman 1.5（8.25）的对比属厂商自测，应作参考而非定论。
- 【Animate-X 的 98.5% 身份保持等数字】这组数字来自用户主观研究（user study），不是 ArcFace/CSIM 等客观指标，不可与 StableAnimator 的 CSIM 数值直接比较。
- 【Wan2.2-Animate 的 MoE 参数口径】'~27B 总参 / ~14B 激活' 来自第三方报道（comfyui-wiki、wink.ai），HF 模型卡标注为 17B。两个口径可能分别指 MoE 总参与单专家参数，需以 HF 官方模型卡为准。
- 【InstantID 商用可用性】InstantID 代码为 Apache-2.0，但其人脸特征提取依赖 InsightFace 系模型（AntelopeV2 等），该系列模型多为非商用许可。Stand-In 同样依赖 AntelopeV2。**本项目若商用，必须单独核实人脸识别模型的许可链**，这是整条管线里最容易被忽略的法务风险点。

### 事实核查修正
- [WRONG] [Wan2.2-Animate-14B] MoE ~27B 总参 / ~14B 激活
  → Wan2.2 官方 README 明确 Animate-14B 是 dense 架构，不是 MoE；MoE（高噪/低噪双专家，27B 总参 / 14B 激活）只适用于 T2V-A14B 与 I2V-A14B。HF 模型卡 metadata 标注为 17B params（Wan2.2-Animate-14B 与 Wan2.2-Animate-14B-Diffusers 均为 17B）。调研员自己在存疑项里怀疑过这条，结论应直接定为错：既不是 MoE，也没有 27B/14B 的激活口径，正确写法是「单体 dense，HF 标注 17B（命名 14B）」。 https://github.com/Wan-Video/Wan2.2
- [WRONG] [Wan2.2-Animate-14B] 720P @ 24fps
  → 分辨率部分对（官方支持 480P & 720P），fps 部分错。官方推理示例导出为 30 fps（--segment_frame_length=77 / --num_inference_steps=20 的 animation 模式示例）。24 FPS 是 Wan2.2-TI2V-5B 的规格（模型库表格里只有 TI2V-5B 写 'supports 720P at 24 FPS'），被串到 Animate 头上了。 https://github.com/Wan-Video/Wan2.2
- [WRONG] [Wan2.2-S2V-14B] HF: Wan-AI/Wan2.2-S2V-14B，2025-09 发布
  → 仓库地址对，日期错。Wan2.2 官方 changelog：「Aug 26, 2025: 🎵 We introduce Wan2.2-S2V-14B」。HF 页面同样记为 2025-08-26 发布（模型文件最后更新 2025-09-17，可能是这个日期被误当成发布日）。对应论文 Wan-S2V arXiv:2508.18621。 https://github.com/Wan-Video/Wan2.2
- [WRONG] [Wan2.2-S2V-14B] 参数量 14B
  → 命名是 14B，但 HF 模型卡 metadata 实测标注为 16B params（BF16）。Wan-AI 组织页列表同样显示 Wan2.2-S2V-14B = 16B。写规格表时要注明「命名 14B / 实际权重 16B」，否则显存与下载量估算会偏低。 https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- [UNVERIFIABLE] [Wan2.2-S2V-14B] 支持长片段扩展与精确唇形编辑
  → 前半句成立：官方模型卡说明不设 --num_clip 时会按音频长度自动扩展生成长视频。后半句「精确唇形编辑」在官方 README 与 HF 模型卡中找不到任何对应表述——S2V 是音频驱动生成（audio-driven cinematic video generation），不是对已有视频做唇形替换/编辑。这条疑似把闭源 Wan2.5 的能力或第三方产品描述套了过来，不要写进选型。 https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- [WRONG] [Wan2.2-VACE-Fun-A14B] 归属于 Wan2.2 官方家族 / 可从 ali-vilab/VACE 获取
  → 归属错。官方权重在 HF 的 alibaba-pai 组织下（阿里 PAI 的 VideoX-Fun 系列，base model 标注为 finetune 自 Wan-AI/Wan2.2-T2V-A14B），不在 Wan-Video/Wan-AI 官方主线，也不在 ali-vilab/VACE 仓库里——拉取 ali-vilab/VACE README 可见它只收录 VACE-Wan2.1-1.3B-Preview、VACE-LTX-Video-0.9、Wan2.1-VACE-1.3B、Wan2.1-VACE-14B 四个模型，全文无任何 Wan2.2  https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B
- [WRONG] [Wan2.2-VACE-Fun-A14B] 控制模式：inpainting、pose、depth、reframe
  → 与官方模型卡不符。alibaba-pai 的模型卡列的控制条件是 Canny、Depth、Pose、MLSD、trajectory control（轨迹控制），外加参考图（reference image）注入；训练规格为 81 帧 @ 16 fps，多分辨率 512/768/1024。卡上并未列出 'inpainting' 与 'reframe' 这两个名字（它们是 Wan2.1-VACE 的任务命名 Expand-Anything / MV2V），第三方 API 文档的四模式清单不能当作官方规格。 https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B
- [UNVERIFIABLE] [Wan2.2-VACE-Fun-A14B] 已被 ComfyUI 原生工作流与多家 API 接入
  → ComfyUI 官方文档站没有对应的原生工作流教程页（docs.comfy.org/tutorials/video/wan/wan2-2-fun-vace 返回 404），与 Wan2.2-Animate 形成对比——后者有官方教程页且确认为原生 Mix/Move 两模式工作流。HF 上能查到的是社区转换件（QuantStack 的 GGUF、linoyts 的 diffusers 版、fal 的 FlashPack 等），属于社区生态而非官方原生节点。「原生工作流」这个说法目前无一手来源。 https://docs.comfy.org/tutorials/video/wan/wan2-2-animate
- [UNVERIFIABLE] [Wan2.1-VACE] 1.3B 档可在 ~8-12GB 显存跑通，14B 档需 40GB+ 或 fp8/block-swap
  → ali-vilab/VACE 的 README 与 UserGuide.md 均未给出任何显存表，只给了 Python 3.10.13 / CUDA 12.4 / PyTorch ≥2.5.1 的环境要求。这组数字来自社区实测，不是官方标注（8.19GB 那个常被引用的数字是 Wan2.1 主仓对 T2V-1.3B 基座的标注，不是 VACE 控制权重）。排产能规划时需自行实测，不要当官方指标引用。 https://github.com/ali-vilab/VACE
- [WRONG] [SkyReels-A2] GitHub SkyworkAI/SkyReels-A2，代码+权重公开
  → 需限定为预览版。README 原文是「We release pre-view version of checkpoints, code of model inference and gradio demo」——即 preview 版权重 + 推理代码 + gradio demo，不是完整发布。另外「仓库最后更新 2025-06-03」与 README 最新 news 条目 2025-06-01（SkyReels-Audio 技术报告）对不上；无论取哪个日期，该仓库已停更一年以上，作为产线组件风险高。论文本身的三条声明（首个开源商用级 E2V、图文联合嵌入 https://github.com/SkyworkAI/SkyReels-A2

### 遗漏补充
- 角色 LoRA 微调（musubi-tuner / diffusion-pipe / ai-toolkit + Wan2.1/2.2 基座）——整条调研线几乎没提，但这是目前身份锁定最可靠、最可控、也是实拍级项目最常用的手段：20-50 张定妆图训一个角色 LoRA，可跨镜头、跨模型（T2V/I2V/VACE/Animate）复用，且能与 VACE 参考图叠加。相比押注某个参考图注入模型，这条路线的可复现性和法务清洁度都更高，应该作为主方案之一而不是附注。
- HunyuanCustom（腾讯混元，开源）——主体一致性定制视频生成，支持单主体/多主体，以及 image / audio / video 三种驱动条件，是 Phantom、VACE-R2V 的直接同赛道竞品，调研完全缺席。同系还有 HunyuanVideo-Avatar（音频驱动数字人）与 HunyuanPortrait。
- ConsisID（CVPR 2025，开源，Identity-Preserving T2V via frequency decomposition）——专攻人脸 ID 频域分解注入，是「文本生成即锁脸」这一路线的代表作，并自带 ConsisID-Bench 评测集。
- MAGREF（字节，开源）——masked guidance 的多主体参考视频生成，明确针对 multi-subject confusion（多主体串味）做设计，与 Phantom 的失效模式正好对偶，做多角色同框时应一并评测。
- MultiTalk（MeiGen-AI）与 InfiniteTalk——前者做多人对话场景的音频驱动 + 身份区分，后者专攻无限时长下的身份漂移抑制，是 Wan2.2-S2V 单人短片能力之外的开源补充，也比 Gloria 这类不可用组件更现实。
- Wan-Dancer-14B（Wan-AI 官方组织下，HF 更新 2026-07-17）——官方 Wan-Video 组织的五仓之一，是角色动作驱动/舞蹈生成方向的官方权重，调研在列举 Wan-Video 组织仓库时提到了名字却没有任何规格核查，属于已知存在但未展开的空白。
- 定妆图矩阵的开源替代：Qwen-Image-Edit-2509（多图参考，人物一致性强）与 FLUX.1 Kontext（角色三视图/表情表/多角度一致性）——与其等 Nano Banana Pro 的「14+ 参考图」被证实，不如直接上这两个权重可下载、可批处理、可脚本化的开源方案做定妆图矩阵生产。同类还有 PuLID、PhotoMaker、IP-Adapter FaceID 系人脸注入（注意与 InstantID 相同的 InsightFace 许可链风险）。
- Vidu Q1/Q2「参考生视频」（生数科技）——最多 7 张参考图锁定主体与场景，闭源 API 但国内可直接调用，是 Kling Elements 的直接竞品；调研的闭源候选只覆盖了 Sora 2 / Kling / Seedance，漏掉了这一家。
- Runway Gen-4 References——闭源，产品定位就是跨镜头保持同一角色与场景，是本维度最成熟的商业对照组，应与 Sora 2 cameo、Kling Elements 并列进闭源候选表。
- 身份一致性的客观验收指标栈缺失——整条调研线引用了大量厂商自评数字（SkyReels-V3 的 0.6698/0.8119/8.18、Animate-X 的 98.5%）却没有定义自己的验收口径。应补：ArcFace/AdaFace CSIM（人脸）、DINOv2/DINOv3 特征相似度（整体主体）、CLIP-I，以及逐帧方差作为漂移度量，这样才能把厂商自测和自测结果解耦。
- OpenS2V-Nexus / OpenS2V-Eval / OpenS2V-5M——主体到视频（S2V）方向的大规模数据集与公开评测基准，是 Phantom-Data 之外的第二个一手数据与评测来源，可用来做第三方复现，直接回应「SkyReels-V3 自评数值无人复现」这个存疑项。
- 工程侧的身份锁定手段（非模型方案）——首帧锚定 + 逐镜头 I2V 链式续接、跨镜头 face swap 后处理（FaceFusion/InsightFace，法务风险同 InstantID）、以及「角色一次 3D/Gaussian 建模 → 多镜头渲染 → v2v 风格化」的资产路线。这三条不依赖任何未发布权重，是 Gloria / AnyID / GroundShot 等不可用组件的现实替代。


====================================================================================================
# [distill-align] 用闭源 API 产出当教师（蒸馏）+ 人工偏好对齐视频模型 + 运镜可控性 —— 面向 2-8 分钟 AI 数字人长片产线（核实日期 2026-09-19）

## Seedance 2.5 / 2.0 系列（火山方舟 API）  (字节跳动 / 火山引擎)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API，按 token 计费 | 成本: 720p 5 秒约 2.5-7.6 元/条；1080p 5 秒约 12-19 元/条；2-8 分钟成片按 5 秒/镜估算需 24-96 镜，单次全片 1080p 成本约 300-1800 元（不含重出）
- 是什么: 闭源视频生成 API，权重不开放。截至 2026-09-19 Seedance 2.5 已「全面公开」，2.0 系列含 2.0 / 2.0-fast / 2.0-mini 三档。用户给的硬约束需要修正。
- 关键事实:
    * Seedance 2.5 输出时长 duration=[4,30] 秒或 -1（自动），支持「30 秒视频连贯直出」；2.0 系列 [4,15]；1.5 pro [4,12]；1.0 系列 [2,12]
    * 全模态参考：2.5 支持 参考图 0-30 张 + 参考视频 0-10 个（总时长≤30s）+ 参考音频 0-10 段（总≤30s），可单独传音频；2.0 系列才是 0-9 图 + 0-3 视频 + 0-3 音频，且不可单独传音频
    * 输出分辨率：2.5 支持 480p/720p/1080p(10bit)；2.0 支持 480p/720p/1080p/4K(10bit)；2.0-fast / 2.0-mini 仅 480p/720p；全系 24 fps；比例 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16
    * 刊例价（元/百万 token，在线推理）：2.5 480p/720p 输入不含视频 70.00、含视频 42.00；1080p 77.00 / 46.00（限时 72 折）。2.0：46/28（480p-720p）、51/31（1080p）、26/16（4K）。2.0-fast 37/22（75 折）。2.0-mini 23/14（4 折）
    * 单条价格示例（16:9、5 秒、输入不含视频）：2.5 480p 3.36 元、720p 7.56 元、1080p 18.71 元；2.0 480p 2.31 / 720p 4.97 / 1080p 12.39 / 4K 25.27 元；2.0-fast 720p 4.00 元；2.0-mini 720p 2.48 元（折后约 1.0 元）
    * token 用量公式：(输入视频时长+输出视频时长) × 输出宽 × 输出高 × 帧率(24) / 1024；含输入视频时有「最低 token 用量」保底
    * 关键限制：2.5 与 2.0 系列「不支持直接上传含有真人人脸的参考图/视频」；但平台信任本账号近 30 天内由这些模型生成的含人脸原始视频，可作为输入二次创作
    * watermark 参数默认示例为 true（右下角 AI 生成水印标识），可设 false；2.5/2.0 不支持离线推理（flex）打折，1.5 pro 及以下可 5 折
    * 1.5 pro 独有 Draft 样片模式：仅 480p，token 折算系数 无声 0.7 / 有声 0.6，用于低成本验证后再出正片；2.0/2.5 暂不支持
- 长片作用: 顶「单镜生成」工位的天花板档。但 30 秒上限 + 24fps + 真人脸禁传，决定了 2-8 分钟必须靠分镜拼接；30 张参考图 + 10 段参考视频是跨镜角色一致性的主力抓手，比自己训 LoRA 更省事。真人脸限制对「数字人」是硬伤：只能用虚构角色 / 平台预置虚拟人像 / 本账号自产人脸视频滚动复用。
- 来源: https://docs.volcengine.com/docs/82379/1520757 https://www.volcengine.com/docs/82379/2298881

## OpenAI 服务条款 / OpenAI Services Agreement（Sora 所属）  (OpenAI)
- 类别/成熟度: service / **production** | 许可: 合同条款，非许可证 | 成本: —
- 是什么: 消费端 Terms of Use 与企业/API 端 Services Agreement，均对「用输出训练模型」有明确禁止条款。
- 关键事实:
    * Terms of Use 生效日 2026-01-01。禁止项原文：「Use Output to develop models that compete with OpenAI.」
    * 同一列表还禁止：「Automatically or programmatically extract data or Output」（禁止程序化批量抓输出）、禁止逆向工程模型/算法/系统、禁止绕过速率限制与安全缓解
    * 所有权条款对用户有利：「you (a) retain your ownership rights in Input and (b) own the Output. We hereby assign to you all our right, title, and interest, if any, in and to Output.」——输出归你，但用途受上面禁止条款约束
    * Services Agreement（企业/API，生效 2026-01-01）3.3(e)：「except for a Permitted Exception, use Output to develop artificial intelligence models that compete with OpenAI's products and services」；3.3(f) 禁止「extract data from the Services other than as permitted」
    * 注意 3.3(e) 的限定词是「compete with OpenAI's products and services」，不是「任何模型」——存在解释空间，但 OpenAI 保留单方更新条款的权利（16.13，重大变更提前 30 天通知）
- 长片作用: 如果把 Sora 输出当教师去训一个「做视频生成的」模型，即便只是 LoRA，也落在「develop models that compete」的字面射程内。Sora 是这四家里措辞最直接把你钉死的一家。
- 来源: https://openai.com/policies/terms-of-use/ https://openai.com/policies/services-agreement/

## Google 服务条款 / Gemini API 附加条款（Veo 所属）  (Google)
- 类别/成熟度: service / **production** | 许可: 合同条款 | 成本: —
- 是什么: Veo 通过 Gemini App / Flow / Gemini API / Vertex AI 分发，受 Google 主 ToS 约束。
- 关键事实:
    * Google Terms of Service 生效日 2026-07-30，「Don't abuse our services」一节原文：「You must not use AI-generated content from our services to develop machine learning models or related AI technology.」——这是四家里最宽、最绝对的一条：不限于「竞争性」模型，任何 ML/AI 技术都不行
    * 同节还有：「You must not reverse engineer our services or underlying technology, such as our machine learning models, to extract trade secrets」
    * 旧版 Generative AI Additional Terms（生效 2023-08-09）原文为「You may not use the Services to develop machine learning models or related technology」，已于 2024-05-22 被主 ToS 吸收取代
    * Gemini API Terms 版本日 2026-03-23：免费档 Google 会用你的 prompt 和 response「to provide, improve, and develop Google products and services and machine learning technologies」；付费档「Google doesn't use your prompts...or responses to improve our products」。输出所有权 Google 不主张，但承认可能给别人生成相同内容
- 长片作用: 用 Veo 输出蒸馏任何 LoRA，在 Google 条款下是字面违约，没有「非竞争」的回旋余地。要么放弃 Veo 当教师，要么走 Vertex AI 企业合同单独谈（Service Specific Terms 可能不同，本次未取得该文本）。
- 来源: https://policies.google.com/terms https://ai.google.dev/gemini-api/terms https://policies.google.com/terms/generative-ai

## Veo 3.1 / 3.1 Fast / 3.1 Lite  (Google DeepMind)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API | 成本: 8 秒 1080p 标准档 $3.20/条；Lite 8 秒 1080p $0.64/条
- 是什么: Google 当前主力闭源视频 API（Gemini API 侧）。
- 关键事实:
    * 模型 id：veo-3.1-generate-preview / veo-3.1-fast-generate-preview / veo-3.1-lite-generate-preview；veo-3.0-generate-001 与 veo-3.0-fast-generate-001 已 deprecated
    * 时长仅支持 4 / 6 / 8 秒；1080p、4K 或使用参考图时强制 8 秒；帧率 24fps
    * 分辨率：3.1 与 3.1 Fast 支持 720p/1080p/4K；3.1 Lite 支持 720p/1080p；视频延长（extension）仅 720p
    * 价格：Veo 3.1 $0.40/秒（720p 与 1080p）、$0.60/秒（4K）；Fast $0.10/秒（720p）、$0.12（1080p）、$0.30（4K）；Lite $0.05（720p）、$0.08（1080p）。无免费层
    * 自带原生音频
- 长片作用: 8 秒上限比 Seedance 2.5 的 30 秒短得多，做 2-8 分钟片需要更碎的分镜。Lite 档 $0.05/秒 是目前主流闭源里最便宜的合规出片通道之一，可顶「粗剪 / 动态分镜预览」工位。但其 ToS 禁止拿输出训模型，不能当教师。
- 来源: https://ai.google.dev/gemini-api/docs/veo https://ai.google.dev/gemini-api/docs/pricing

## RAPID: Real-Time Defense Against Unauthorized Model Distillation for T2I Services  (学术（arXiv 2609.15799，2026-09-14）)
- 类别/成熟度: technique / **research** | 许可: 论文，未见权重 | 成本: —
- 是什么: 专门防御「黑盒输出蒸馏」的论文：攻击者查询闭源服务、收集 prompt-image 对、训练替代模型。这是「拿闭源 API 输出做蒸馏」这件事真实存在、且厂商正在投入对抗的直接证据。
- 关键事实:
    * 明确定义威胁模型：「black-box output-based distillation, where an adversary queries the service, collects prompt-image pairs, and trains an unauthorized substitute model that mimics its generation behavior」
    * 防御手段是把对抗扰动烘进 VAE decoder，让服务直接输出「带毒」图像——即对所有用户生效、无需逐样本优化
    * 实验覆盖 4 个 T2I 模型、4 个数据集、5 个代表性 baseline，结论是能持续降低替代模型生成质量同时保持服务视觉保真度
    * 2026-09-14 提交，说明这是当下活跃的攻防前沿，而非历史问题
- 长片作用: 对产线的实际含义：今天能蒸的，明天可能蒸不动。如果厂商把这类 decoder 级扰动上线（Seedance / Veo 都有能力做），你囤的教师数据会贬值，且难以事先检测。把「蒸馏」当长期护城河是危险的，只能当一次性加速手段。
- 来源: https://arxiv.org/abs/2609.15799

## JourneyDB  (学术（arXiv 2307.00716，2023-07-03）)
- 类别/成熟度: technique / **production** | 许可: 数据集自带许可，与 Midjourney ToS 的关系从未被法院检验 | 成本: —
- 是什么: 400 万张 Midjourney 生成图像的公开数据集。是「用闭源生成服务输出构建训练数据集」在学术界被公开做、被引用、未被下架的先例。
- 关键事实:
    * 4,000,000 张 Midjourney 生成图像 + prompt 配对
    * NeurIPS 2023 Datasets & Benchmarks track
    * 定位是「生成图像理解」基准，不是直接的生成模型蒸馏——这是它能存活的关键差别
- 长片作用: 可引用的先例，但引用时要诚实：它做的是「理解」不是「生成蒸馏」，法律风险等级不同。对你的项目参考价值在于：学术圈默认这类数据可收集可发布，实际执法几乎为零；但商用产线的风险敞口和学术数据集完全不同。
- 来源: https://arxiv.org/abs/2307.00716

## Self-Forcing  (Adobe Research + UT Austin（arXiv 2506.08009）)
- 类别/成熟度: technique / **usable** | 许可: Apache-2.0 | 成本: 推理单卡 24GB；训练 64×H100×2h（约 128 H100 卡时，按 $2.5/h 约 $320）
- 是什么: 自回归视频扩散的少步蒸馏 + 训练测试对齐方法，官方实现直接基于 Wan2.1-T2V-1.3B。
- 关键事实:
    * 2025-06-09 提交，v2 2025-11-10；NeurIPS 2025 Spotlight
    * 训练时用自己生成的上下文（autoregressive rollout + KV caching）而非 ground truth，配合少步扩散模型和随机梯度截断
    * 官方 repo 许可 Apache-2.0，基座 Wan2.1-T2V-1.3B
    * 推理要求：Linux、64GB 内存、单卡 ≥24GB 显存（RTX 4090 / A100 / H100 实测），单张 4090 可实时流式生成
    * 训练成本：64 张 H100、约 600 iterations、2 小时以内
- 长片作用: 顶「实时预览 / 低成本铺量」工位。数字人对口型和走位的迭代次数极大，实时预览能把导演-调整循环从分钟级压到秒级。但 1.3B 基座的画质离 Seedance 2.0 还有明显差距，成片还得回大模型。
- 来源: https://arxiv.org/abs/2506.08009 https://github.com/guandeh17/Self-Forcing

## CausVid（From Slow Bidirectional to Fast Autoregressive Video Diffusion Models）  (MIT + Adobe（arXiv 2412.07772）)
- 类别/成熟度: technique / **usable** | 许可: 以论文与社区复现为主 | 成本: —
- 是什么: 把 DMD（分布匹配蒸馏）扩展到视频，把双向扩散 Transformer 改造成自回归、把 50 步蒸成 4 步。
- 关键事实:
    * 2024-12-10 提交，v4
    * 50 步扩散模型 → 4 步生成器
    * 关键技术：基于教师 ODE 轨迹的学生初始化 + 非对称蒸馏（双向教师监督因果学生）
    * 目标是缓解自回归生成的误差累积
- 长片作用: 是当前所有「Wan 少步加速」路线（Lightning / Turbo LoRA）的方法论源头。在产线里顶「出片加速」工位，不改画风只改速度。
- 来源: https://arxiv.org/abs/2412.07772

## DMD2（Improved Distribution Matching Distillation）  (MIT + Adobe（arXiv 2405.14867）)
- 类别/成熟度: technique / **production** | 许可: 论文 + 官方实现 | 成本: —
- 是什么: DMD 的改进版：去掉回归损失和昂贵的噪声-图像对数据集构建，让学生不再被教师采样路径绑死。
- 关键事实:
    * 2024-05-23 提交，v2
    * 核心改动：消除 regression loss，从而摆脱「用教师多步确定性采样器造大规模 noise-image 对」的成本
    * 代价是训练不稳定，论文给出稳定化技术组合
- 长片作用: 工业界少步模型的事实标准。但注意下面 Mask Forcing / CrossDistill 两条：DMD 的 reverse-KL 是 mode-seeking，会主动杀多样性——这和你后面「人工打分只回收过检镜」的数据飞轮是同向叠加的坏效应。
- 来源: https://arxiv.org/abs/2405.14867

## LCM / VideoLCM（Latent Consistency Models）  (清华 / 阿里（arXiv 2310.04378 / 2312.09109）)
- 类别/成熟度: technique / **usable** | 许可: 开源实现 | 成本: LCM 图像侧 32 A100 卡时（约 $80）
- 是什么: 一致性蒸馏路线，视频侧的最早系统性尝试。
- 关键事实:
    * LCM 2023-10-06；768×768 的 2-4 步 LCM 训练只需 32 张 A100 卡时
    * VideoLCM 2023-12-14，把一致性蒸馏搬到潜空间视频扩散
- 长片作用: 历史价值大于现实价值。2026 年在视频上已被 DMD2 / Self-Forcing / PDD 系全面超越，只在极低算力场景还值得考虑。标注它是为了说明「step distillation ≠ 学风格」——LCM 这条线根本不改模型的内容分布，学不到运镜和质感。
- 来源: https://arxiv.org/abs/2310.04378 https://arxiv.org/abs/2312.09109

## Wan2.2-Lightning（lightx2v）  (lightx2v 社区)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: 与 Wan2.2 A14B 推理同级；4 步使单镜成本降到原来约 1/20
- 是什么: Wan2.2 官方之外最广泛使用的 4 步蒸馏 LoRA，生产环境事实标准。
- 关键事实:
    * 基座 Wan2.2-T2V-A14B 与 Wan2.2-I2V-A14B
    * 4 步推理（NFE4），无需 CFG，约 20× 加速
    * LoRA rank 64
    * 许可 Apache-2.0
    * 版本节点：T2V-A14B-NFE4-V1 与 I2V-A14B-NFE4-V1 2025-08-07，V1.1 同日改进；ComfyUI workflow 2025-08-08
    * 支持 480P 与 720P
- 长片作用: 顶「出片加速」工位，2-8 分钟片 24-96 镜的规模下这是成本能不能收敛的关键。但和你要训的风格/运镜 LoRA 会打架，见下一条 DART。
- 来源: https://huggingface.co/lightx2v/Wan2.2-Lightning https://huggingface.co/lightx2v

## DART: Distillation-Aware Reparameterization for Training-Free LoRA Reuse in Few-Step Video Diffusion Models  (学术（arXiv 2609.20051，2026-09-17）)
- 类别/成熟度: technique / **research** | 许可: 论文，未见权重 | 成本: 免训练，只需前向评估
- 是什么: 直接研究「为长轨迹训的 LoRA 塞进少步蒸馏模型后会走样」这个问题，并给出免训练修复。对你的方案是最要命的一条。
- 关键事实:
    * 明确结论：「reusing a LoRA trained for a longer trajectory can alter its functional effect or degrade target quality」——即你在满步 Wan2.2 上训的风格/运镜 LoRA，挂到 4 步 Lightning 上效果会变
    * 在 four-step Wan2.2 目标上，DART-F 把 joint quality score 从 0.9029 提到 0.9227
    * macro functional retention 从 -0.4644 提到 +0.1349（负值即 LoRA 的功能效果被反转/抵消）
    * 免训练：低秩坐标输运 + 目标调度响应校准，只需前向评估，不需要源训练视频
    * 论文自述：不同 adapter 表现分化，有的恢复有的强衰减，「without assuming recovery for every adapter」
- 长片作用: 直接回答「Wan 上 LoRA 学运镜/质感」的工程可行性：可行，但你必须在「满步 + 自训 LoRA」和「4 步 Lightning 加速」之间二选一，或者接受一次 DART 式校准。产线排期上意味着要么单镜成本 ×20，要么 LoRA 效果打折。这一条是该路线最大的隐藏成本。
- 来源: https://arxiv.org/abs/2609.20051

## Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout  (学术（arXiv 2609.09123，2026-09-08）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 直指 DMD 视频蒸馏的 mode collapse：reverse-KL 的 mode-seeking 让学生分布坍到教师分布的少数几个模式上。
- 关键事实:
    * 症状描述明确：过饱和（over-saturation）、过平滑（over-smoothing），视觉质量与真实感受限
    * 归因：「the mode-seeking behavior of the reverse KL objective in DMD, which can cause the student distribution to collapse onto only a few modes of the teacher distribution」
    * 方案：自回归 rollout 时沿空间与时间轴随机 mask 注入更干净的信号，逼学生探索教师分布更多区域
    * 不需要真实视频数据，也不需要额外后训练阶段
- 长片作用: 解释了一个产线上会真实撞到的现象：用蒸馏加速后画面「塑料感 / 一个味儿」。如果你再叠一层「只回收过检镜」的数据飞轮，两个坍缩会叠乘。
- 来源: https://arxiv.org/abs/2609.09123

## CrossDistill: Balancing Quality and Diversity via Trajectory-Level Hybrid Few-Step Distillation  (学术（arXiv 2609.14725，2026-09-13）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 把 trajectory distillation 与 distribution matching 按噪声区间分工：高噪声段用轨迹保持（保模式覆盖），低噪声段用分布匹配（锐化细节）。
- 关键事实:
    * 核心观察：high-noise steps 决定全局模式，low-noise steps 细化局部细节
    * 在轨迹上设一个 crossover point 切分两种目标，并通过 crossover state 耦合
    * PCM 与 DMD 都是可插拔实例；关键设计是噪声分区、交叉耦合与目标顺序
    * 实验在文生视频扩散模型上，结论是扩展了少步的质量-多样性前沿，保留 seed 级变化
- 长片作用: 这是「trajectory distillation」在 2026 年的正确打开方式：不是替代 DMD，而是按噪声区间分工。对长片产线意义在于「同一 prompt 换 seed 要能出不同镜头」，否则 24-96 个镜头会互相撞脸。
- 来源: https://arxiv.org/abs/2609.14725

## Parallel Decoding Distillation (PDD)  (学术（arXiv 2607.26004，2026-07-28）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 纯 trajectory-based 的少步蒸馏，绕开 VSD 与对抗损失，直接在 Wan 14B 上验证。
- 关键事实:
    * 明确批评现有 SOTA：「these training losses are notoriously hard to optimize and suffer from mode collapse, leading to loss of video diversity and lack of motion」
    * 做法：一次网络评估预测多个去噪步；学习 mean velocity 的表示，不需要 JVP 或有限差分
    * 在 LTX-2.3 文生视频/音频、Wan 14B 文生视频、Qwen-Image 上达到 4-8 NFE 的 SOTA
    * 兼容任何预训练模型，支持可变 NFE 采样
    * 明确报告「significant improvement in generated video diversity」
- 长片作用: 如果你担心加速蒸馏吃掉动态和多样性（长片最怕「每个镜头都在慢慢推」），PDD 是目前公开方法里对 motion 和 diversity 交代最明确的一条，且直接在 Wan 14B 上验过。
- 来源: https://arxiv.org/abs/2607.26004

## Reward Lightning: Fast Video Generation via Homologous Preference Distillation  (学术（arXiv 2607.03960，2026-07-04）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 把偏好对齐和少步蒸馏放进同一个共享表示里做，避免两个目标在不同表示空间上互相拆台。
- 关键事实:
    * 提出 latent reward model (LRM)，直接在潜空间打分，不解码回像素
    * LRM 偏好准确率超过像素级与潜空间 reward baseline 分别 11.0% 与 14.7%
    * 最终生成器 1 到 4 步，VBench 平均分 +2.1%，在文本对齐、运动质量、视觉质量上领先
    * 项目页 reward-lightning.github.io
- 长片作用: 对产线的直接价值：你如果既要做人工偏好对齐、又要做 4 步加速，别分两步串行做（会互相回退），这篇给出了合并的正确姿势。LRM 不解码即可打分，也意味着自动质检可以便宜几倍。
- 来源: https://arxiv.org/abs/2607.03960

## VideoReward / VideoAlign（Improving Video Generation with Human Feedback）  (快手 KwaiVGI（arXiv 2501.13918）)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: 2B VLM 推理，单卡 8-16GB 足够；打分成本远低于生成成本
- 是什么: 多维视频奖励模型 + 三种流模型对齐算法，是视频侧偏好对齐目前最完整的开源一套。
- 关键事实:
    * 2025-01-23 提交，v2
    * VideoReward 基于 Qwen2-VL-2B-Instruct，2B 参数，许可 Apache-2.0
    * 三个打分维度：Visual Quality (VQ)、Motion Quality (MQ)、Text Alignment (TA)
    * 配套三个算法：Flow-DPO 与 Flow-RWR 用于训练，Flow-NRG 用于推理时引导
    * 配套基准 VideoGen-RewardBench
    * 官方说明可用于 data filtering、guidance、reject sampling、DPO 及其他 RL 方法
    * 论文结论：Flow-DPO 优于 Flow-RWR 与标准 SFT
- 长片作用: 顶「自动质检 / 过检筛选」工位，是你那个「人工打分回收过检镜」飞轮的自动化前置——人只看机器打分后 top-K，能把人工标注量压一个量级。三维打分也让你能分别控「运动质量」和「画面质量」，避免只优化一个指标。
- 来源: https://arxiv.org/abs/2501.13918 https://huggingface.co/KwaiVGI/VideoReward

## VisionReward  (清华 / 智谱（arXiv 2412.21059）)
- 类别/成熟度: open-weights / **usable** | 许可: 开源（见官方 repo） | 成本: —
- 是什么: 图像与视频统一的细粒度多维人类偏好模型，分层视觉评估 + 线性加权，可解释。
- 关键事实:
    * 2024-12-30 提交，v4
    * 偏好预测准确率超过 VideoScore 17.2%
    * 用 VisionReward 做对齐的文生视频模型，相对用 VideoScore 的同款模型 pairwise win rate 高 31.6%
    * 设计成可解释：分层评估 + 线性加权，每一维可单独看
- 长片作用: 可解释性对产线有实操价值：当某个镜头被判不合格，你要知道是「运动抖」还是「构图差」还是「不对文本」，才能定向改 prompt 而不是盲目重抽。
- 来源: https://arxiv.org/abs/2412.21059

## LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment  (学术（arXiv 2412.04814）)
- 类别/成熟度: technique / **research** | 许可: 论文 + 数据集 | 成本: 约 1 万条人工标注的规模——这是你自建偏好数据的量级参考
- 是什么: 带理由的人工标注 + reward model（LiFT-Critic）+ 微调，证明小模型靠对齐能追上大两倍的模型。
- 关键事实:
    * 2024-12-06 提交，v3
    * LiFT-HRA 数据集：约 10k 条人工标注，每条含一个分数和对应的理由（rationale）
    * 在 CogVideoX-2B 上微调后，「outperforms the CogVideoX-5B across all 16 metrics」
- 长片作用: 给你一个可信的标注预算锚点：1 万条带理由的人工标注，就能让 2B 模型在全部 16 个指标上超过同族 5B。对你的产线意味着人工打分不是无底洞，万级标注量即可见效。
- 来源: https://arxiv.org/abs/2412.04814

## VideoDPO  (学术（arXiv 2412.14167）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 把 DPO 迁移到视频扩散的代表作，提出 OmniScore 同时覆盖视觉质量与语义对齐，自动构造偏好对。
- 关键事实:
    * 2024-12-18 提交
    * OmniScore 综合评估视觉质量与语义对齐两个维度
    * 偏好对自动采集 + 基于分数的重加权（score-based re-weighting）
    * 定位是 omni-preference：避免只优化单一维度导致另一维度退化
- 长片作用: 「自动构造偏好对」这一点对产线最实用：你不需要一开始就上人工，先用 OmniScore 自动配对跑通管线，人工只用在模型分不清的边界样本上。
- 来源: https://arxiv.org/abs/2412.14167

## Diffusion-DPO  (Salesforce + 斯坦福（arXiv 2311.12908）)
- 类别/成熟度: technique / **production** | 许可: 论文 + 社区实现广泛 | 成本: —
- 是什么: 把 DPO 重新推导到扩散模型似然记法上的原始论文，视频侧所有 DPO 变体的共同祖先。
- 关键事实:
    * 2023-11-21 提交
    * 核心贡献是重新表述 DPO 以适配扩散模型的似然概念，用分类目标直接优化策略，不需要显式 reward model
- 长片作用: 迁移到视频的门槛低（VideoDPO / Flow-DPO / DenseDPO 都是它的直系），是你自建偏好对齐管线最稳妥的起点。
- 来源: https://arxiv.org/abs/2311.12908

## 视频 DPO 的工程化变体：Reg-DPO / RealDPO / DenseDPO  (多家（arXiv 2511.01450 / 2510.14955 / 2506.03517）)
- 类别/成熟度: technique / **usable** | 许可: 论文 | 成本: —
- 是什么: 三条解决 DPO 在视频上实际踩坑的路线，都发生在 2025 下半年。
- 关键事实:
    * Reg-DPO（2511.01450 v3，2025-11-03）：把 SFT loss 作为正则加进 DPO，配合自动构造的 GT-Pair（真实视频做正样本）
    * RealDPO（2510.14955 v2，2025-10-16）：直接用真实世界视频作为 DPO 的正样本，改善运动合成与自纠错
    * DenseDPO（2506.03517 v2，2025-06-04）：构造时间对齐的偏好对，做 segment-level DPO，专门中和「偏好静态画面」的 motion bias
    * 另有 Beyond Reward Margin（2511.19049，2025-11-24）专治 DPO 的 likelihood displacement，提出 Policy-Guided DPO
- 长片作用: DenseDPO 的 motion bias 问题对长片是致命的：朴素 DPO 会系统性偏好「几乎不动的画面」，因为静态画面没有运动伪影。你的数字人如果越训越僵，先查这个。Reg-DPO / RealDPO 的共同处方是「掺真实视频」——这正是下面数据飞轮那一组结论的工程版。
- 来源: https://arxiv.org/abs/2511.01450 https://arxiv.org/abs/2510.14955 https://arxiv.org/abs/2506.03517

## Flow-GRPO  (学术（arXiv 2505.05470）)
- 类别/成熟度: technique / **production** | 许可: 论文 + 开源实现 | 成本: 在线 RL，需要边训边采样，成本约为 DPO 的数倍
- 是什么: 把在线策略梯度 RL 接进流匹配模型：ODE 转 SDE + Denoising Reduction。是 2026 年视频 RL 对齐的主干方法。
- 关键事实:
    * 2025-05-08 提交，v5
    * SD3.5-M 经 RL 微调后 GenEval 准确率从 63% 提到 95%
    * 视觉文字渲染准确率从 59% 提到 92%
    * 论文自述 reward hacking 极小（minimal）
    * 衍生极多：截至 2026-09 已有 OP-GRPO（2604.04142，首个 off-policy 变体）、Mix-GRPO、SDE-GRPO、SAGE-GRPO（2603.21872）、KVPO（2605.14278，自回归视频）、World-R1（2604.24764）、VGGRPO（2603.26599，4D latent reward）等
- 长片作用: 顶「模型级对齐」工位。相对 DPO 的优势是能优化不可微、规则化的目标（比如「镜头必须是推镜」「角色必须在画面左三分之一」）。对分镜系统尤其有用：构图与运镜是可验证目标。
- 来源: https://arxiv.org/abs/2505.05470 https://arxiv.org/abs/2604.04142

## DanceGRPO  (字节跳动 + 港大（arXiv 2505.07818）)
- 类别/成熟度: technique / **usable** | 许可: 论文 + 开源实现 | 成本: 在线 RL
- 是什么: 统一覆盖扩散模型与整流流、图像与视频的 GRPO 框架，解决此前 RL 方法的稳定性问题。
- 关键事实:
    * 2025-05-12 提交，v4
    * 报告在多个既定基准上比 baseline 高出「up to 181%」
    * 评测涵盖 HPS-v2.1、CLIP Score、VideoAlign、GenEval
    * 明确覆盖 diffusion models 与 rectified flows 两类
- 长片作用: 注意它用 VideoAlign（即上面的 VideoReward）当奖励之一——DanceGRPO + VideoReward 是一套可以直接拼起来的开源 RLHF 栈，这是你自建偏好对齐管线最短的路径。181% 这个数字要谨慎看，是相对特定 baseline 的相对提升。
- 来源: https://arxiv.org/abs/2505.07818

## Advances in GRPO for Generation Models: A Survey  (学术（arXiv 2603.06623，2026-02-21）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: Flow-GRPO 系方法的综述，覆盖方法论、奖励设计、credit assignment 与跨模态应用。
- 关键事实:
    * 2026-02-21 提交
    * 明确指出 Flow-GRPO 自提出以来触发了快速的研究增长，跨文生图、视频、3D、语音
- 长片作用: 做技术选型时先读这篇，能省掉自己在 30 篇 GRPO 变体里试错的时间。
- 来源: https://arxiv.org/abs/2603.06623

## 偏好模式坍缩（Preference Mode Collapse）与多样性退化：D2-Align / TMPO / DDRL / GARDO  (多家（arXiv 2512.24146 / 2605.10983 / 2512.04332 / 2512.24138）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 四篇 2025-12 至 2026-05 的论文，共同结论是：奖励分数涨、真实质量和多样性掉。这是你那个「只回收过检镜」飞轮的直接理论对应物。
- 关键事实:
    * D2-Align（2512.24146 v2，2025-12-30）正式命名 Preference Mode Collapse (PMC)：「models converge on narrow, high-scoring outputs (e.g., images with monolithic styles or pervasive overexposure), severely degrading generative diversity」，并建了 DivGenBench 专门量化
    * TMPO（2605.10983 v3，2026-05-09）归因：mode-seeking 本质导致概率集中在少数高奖励路径；用轨迹级奖励分布匹配（Softmax Trajectory Balance）继承 forward KL 的 mode-covering 性质；相对 SOTA 多样性提升 9.1%
    * DDRL（2512.04332 v3，2025-12-03）归因：现有正则化给出的惩罚不可靠；处方是用数据本身做正则（Data-regularized Diffusion RL）
    * GARDO（2512.24138，2025-12-30）：代理奖励只部分刻画真实目标，导致「proxy scores increase while real image quality deteriorates and generation diversity collapses」
    * HyperAlign（2601.15968 v2）也重复了同一权衡：微调类方法「risk reward over-optimization and loss of generation diversity」
- 长片作用: 对你的飞轮设计：这四篇一致指向同一处方——别让优化目标只是「过检率」。具体到产线：(1) 必须保留一个不参与训练的固定 holdout prompt 集，每轮测多样性而非只测过检率；(2) 用 forward-KL/mode-covering 类目标（TMPO）而非纯 reverse-KL；(3) 用数据做正则（DDRL）而非只靠 KL 惩罚系数。注意这些论文没有一篇给出「掺通用数据的百分比经验值」——那个数字见下一条。
- 来源: https://arxiv.org/abs/2512.24146 https://arxiv.org/abs/2605.10983 https://arxiv.org/abs/2512.04332

## 自消费循环与模型坍缩的混合比理论：Fisher-Rao 界 / 人类策展反噬 / 选择偏差  (多家（arXiv 2609.18878 / 2605.29267 / 2606.13732）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 三篇直接回答「合成数据里要掺多少真实数据才不坍缩」以及「人工筛选本身会不会反而害事」。
- 关键事实:
    * Fisher-Rao（2609.18878，2026-09-16）：明确把问题表述为「what is the exact minimum required ratio of human-to-synthetic data」；指出此前工作给的欧氏度量下界在高维下会退化为 vacuous（无意义）；改用概率单纯形上的 Fisher-Rao 度量给出维度稳定的收缩与不变性界，结论是「effective required data ratio 与此前暗示的不同」——但论文本身没有给出一个可直接套用的百分比
    * 人类策展反噬（2605.29267，2026-05-28）：Ferbach et al. 2024 证明单模型自消费下人工策展总能提升对齐；这篇把它推广到多模型交互场景，结论是「cross-model interactions can dampen or even invert this effect, ultimately degrading long-term alignment」
    * 选择偏差（2606.13732 v2，2026-06-11）：当验证者只看到目标流形的一小块、碎片化且有偏的切片时，「selection itself becomes biased」——数据筛选本身就成了坍缩来源
    * Reviewing Model Collapse and Countermeasures（2608.21366，2026-06-17）是配套综述
- 长片作用: 对你的飞轮最重要的一条负面结论：「人工打分只回收过检镜」正好命中 2606.13732 描述的失败模式——你的审核员就是那个只看到流形一小块的有偏验证者。而 2605.29267 说明，当你同时用多个闭源模型产数据时，人工策展的正面效果可能被跨模型交互抵消甚至反转。必须掺真实视频/通用数据，但学界目前没有给出可引用的百分比经验值，这个数字只能你自己做消融。
- 来源: https://arxiv.org/abs/2609.18878 https://arxiv.org/abs/2605.29267 https://arxiv.org/abs/2606.13732

## CameraCtrl  (学术（arXiv 2404.02101）)
- 类别/成熟度: technique / **usable** | 许可: 开源 | 成本: —
- 是什么: 最早把相机位姿作为即插即用控制模块加到视频扩散上的工作，其余模块不动。
- 关键事实:
    * 2024-04-02 提交，v2
    * plug-and-play 相机位姿控制模块，训练时基座其他模块冻结
    * 论文专门做了训练数据消融，结论：「videos with diverse camera distributions and similar appearance to the base model」最有利于可控性与泛化
- 长片作用: 那条数据消融结论对你的蒸馏方案是个具体指导：教师数据不光要运镜多样，外观还得贴近 Wan 自己的分布。用 Seedance 输出（外观分布离 Wan 很远）去训 Wan 的运镜 LoRA，正好违反这条——这是该路线在技术上最可能失败的点。
- 来源: https://arxiv.org/abs/2404.02101

## MotionCtrl  (腾讯 ARC（arXiv 2312.03641）)
- 类别/成熟度: technique / **usable** | 许可: 开源 | 成本: —
- 是什么: 把相机运动与物体运动解耦独立控制的统一控制器。
- 关键事实:
    * 2023-12-06 提交，v2
    * 三个声称优势：相机与物体运动独立控制；运动条件由相机位姿与轨迹决定，是 appearance-free 的，对物体外观/形状影响极小；一次训练可泛化到广泛的位姿与轨迹
    * appearance-free 是它与后来方法的关键区分点
- 长片作用: 「运镜控制应该 appearance-free」这个设计原则直接否定了你的方案前提：如果运镜控制本身不该带外观信息，那「用 Seedance 输出做运镜 LoRA」学到的很可能是 Seedance 的外观而非运镜逻辑。运镜该用几何条件注入，不该用 LoRA 学。
- 来源: https://arxiv.org/abs/2312.03641

## Uni3C  (阿里（arXiv 2504.14899）)
- 类别/成熟度: technique / **usable** | 许可: 开源 | 成本: —
- 是什么: 统一相机控制与人体动作控制的 3D 增强框架，用冻结的视频生成基座训 PCDController。
- 关键事实:
    * 2025-04-21 提交，v2
    * PCDController：用单目深度反投影得到的点云做相机控制，可在冻结基座上训练
    * 论文强调泛化性：「performing well regardless of whether the inference backbone is frozen or fine-tuned」
    * 推理阶段联合对齐的 3D world guidance：场景点云 + SMPL-X 角色，把相机与人体动作两路控制信号统一
    * 设计动机就是解决「同时有高质量相机与人体标注的数据太少」
- 长片作用: 对「AI 数字人 + 运镜」这个具体组合，Uni3C 是命中率最高的一条：SMPL-X 控人体、点云控相机，两路可分别在各自领域训练。这正是数字人拍片需要的分工。顶「运镜 + 表演控制」工位。
- 来源: https://arxiv.org/abs/2504.14899

## ReCamMaster  (快手 KwaiVGI（arXiv 2503.11647）)
- 类别/成熟度: technique / **research** | 许可: 开源代码与数据集 | 成本: —
- 是什么: 给定一段已有视频，用新的相机轨迹重新渲染同一动态场景。
- 关键事实:
    * 2025-03-14 提交，v2
    * 核心机制是一个简单但强力的 video conditioning 机制，激活预训练 T2V 的生成能力
    * 为解决训练数据稀缺，用 Unreal Engine 5 构建了多机位同步视频数据集，刻意按真实拍摄特性设计，覆盖多样场景与运镜
    * 目标是泛化到 in-the-wild 视频
- 长片作用: 对长片产线是一个被低估的工位：同一场表演生成一次，用 ReCamMaster 重渲多个机位，直接产出「正反打 / 过肩 / 全景」这组分镜，且表演与光照天然一致。这比每个机位重新生成再想办法对齐要可靠得多。UE5 合成多机位数据也是你自建训练集的可复制路径（且完全无版权风险）。
- 来源: https://arxiv.org/abs/2503.11647

## Go-with-the-Flow（Warped Noise）  (Netflix Eyeline Studios 等（arXiv 2501.08331）)
- 类别/成熟度: technique / **usable** | 许可: 开源 | 成本: 微调开销极小；噪声 warping 实时
- 是什么: 只改数据不改架构的运动控制：把训练视频预处理成结构化噪声，用光流场做噪声 warping。
- 关键事实:
    * 2025-01-14 提交，v5
    * 关键卖点：「This is achieved by just a change in data」——与扩散模型设计无关，不改模型架构也不改训练管线
    * 噪声 warping 算法可实时运行，用相关的 warped noise 替换时间维的随机高斯性，同时保持空间高斯性
    * 一站式覆盖三类控制：局部物体运动控制、全局相机运动控制、运动迁移
    * 微调现代视频扩散基座的开销极小（minimal overhead）
- 长片作用: 对你的产线是性价比最高的运镜方案：不改 Wan 架构、不和 Lightning 少步 LoRA 冲突（它动的是噪声不是权重）、能同时做运镜和运动迁移。「运动迁移」这一项还能直接顶数字人的表演复用工位——一段好表演迁到多个镜头。出品方是 Netflix 的特效部门，工程可靠性有背书。
- 来源: https://arxiv.org/abs/2501.08331

## Wan2.1-Fun / Wan2.2-Fun Control-Camera（VideoX-Fun）  (阿里 PAI（aigc-apps）)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: A14B 权重 64GB，推理需 offload 才能上消费卡
- 是什么: 官方生态里唯一直接可用的 Wan 相机控制权重，不是论文是可下载的模型。
- 关键事实:
    * 提供 Wan2.2-Fun（A14B、5B，含 InP / Control / Control-Camera）与 Wan2.1-Fun V1.1（1.3B、14B，含 InP / Control / Control-Camera）
    * 相机镜头控制权重支持 pan 与 rotation 操作，覆盖 512 / 768 / 1024 三档分辨率
    * 许可 Apache-2.0（CogVideoX-5B 部分另用 CogVideoX 许可）
    * 权重体积 13GB（CogVideoX-Fun V1.1-2b）到 64GB（Wan2.2-Fun A14B）
    * 支持 LoRA 训练，含 Reward Lora 与蒸馏变体
    * 最近一次大更新 2025-10-16：新增 Wan 2.2 系列、Wan-VACE 控制模型、Fantasy Talking 数字人模型
- 长片作用: 顶「运镜控制」工位的落地选项——今天就能装。注意它的相机控制只覆盖 pan 与 rotation，推拉（dolly/zoom）、升降（crane）、跟拍（tracking）不在明确支持列表里，复杂运镜还得靠 Uni3C 或 Go-with-the-Flow 补。
- 来源: https://github.com/aigc-apps/VideoX-Fun

## Probing into Camera Control of Video Models  (学术（arXiv 2605.14815，2026-05-14）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: 免训练
- 是什么: 免训练的相机控制方法，同时被当成探针来测量各个基座模型自身的运镜能力与偏差。这是最接近直接回答「提示词镜头语法在开源模型上到底管不管用」的论文。
- 关键事实:
    * 2026-05-14 提交
    * 对现有方法的批评很硬：额外相机模块 + 配对数据的路线，数据「limited in scale, diversity, and scene dynamics」，会「bias the model toward a narrow output distribution and compromise the strong prior learned by the base model」
    * 方法：把相机控制重述为一组位移场，在去噪过程中对潜特征做可微重采样施加
    * 效果：相比微调基线，在各类质量指标上「minimal degradation」
    * 作为探针的发现：识别出各代表性视频模型「universal biases shared by representative video models」，以及它们对相机控制响应的差异
    * 无需训练，适用于大多数视频扩散模型
- 长片作用: 两条对产线致命的结论：(1) 用配对数据微调运镜会破坏基座的先验——这正是你打算做的事；(2) 所有主流视频模型共享某些运镜偏差，意味着靠提示词写「推拉摇移跟升降」在开源模型上存在系统性的天花板，不是写法问题而是模型问题。结论：运镜要靠几何条件注入（位移场 / 点云 / warped noise），不要靠提示词，也不要靠 LoRA。
- 来源: https://arxiv.org/abs/2605.14815

## Auteur: Language-Driven Cinematographic Framing  (学术（arXiv 2606.01900，2026-06-01）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: —
- 是什么: 用 DSL 把景别、角度、构图编码为人体姿态的函数，再由微调的 MLLM 当「虚拟导演」生成关键帧。直接对应「镜头语法」这一维。
- 关键事实:
    * 2026-06-01 提交，v3
    * 核心洞察：专业摄影师构思镜头不是世界坐标系的轨迹，而是相对演员的取景——「encoding shot size, angle, and composition as functions of human pose and motion」
    * 提出 human-centric camera parameterization 与一套可转换为标准 6-DoF 相机参数的 DSL
    * 微调的多模态大模型把自然语言 + 粗略人体动作映射为稀疏 DSL 关键帧，再确定性插值为连续相机轨迹，喂给视频生成器
    * 数据集：34K 条对齐的文本、人体动作与 DSL 标注相机轨迹，来自程序化合成 + CondensedMovies 真实电影素材
    * 提出了新的取景专用指标，声称一致优于现有方法
    * 项目页 cyberiada.github.io/Auteur/
- 长片作用: 直接顶「分镜系统」工位。你的平台需要把「中景、过肩、45 度俯角」这类导演语言变成可执行参数——Auteur 的 DSL 正是这个转换层，而且它是 human-centric 的，天然适配数字人。这比在 prompt 里堆镜头术语靠谱一个数量级。
- 来源: https://arxiv.org/abs/2606.01900

## CamPilot: Multi-Agent Cinematic Assistant for Camera-Controlled Movie Generation  (学术（arXiv 2609.10943，2026-09-10）)
- 类别/成熟度: technique / **research** | 许可: 论文，未见权重 | 成本: —
- 是什么: 多智能体电影摄影助手，用 GRPO 从真实电影学运镜规划，专门处理多镜连续性。发表于 9 天前。
- 关键事实:
    * 2026-09-10 提交
    * 从 14K 部真实专业电影学习 camera work planning
    * GRPO 学习范式，内化运动模式与构图原则，支持对拍摄手法（机位角度、运动、焦点行为）与跨镜关系的推理
    * 明确针对「multi-shot continuity remains challenging」这一长片核心痛点
    * 配套基准 CamEval，评估运镜质量与电影感投入度
    * 多个 agent 协作并演化以提升整体输出质量
- 长片作用: 这是本次检索里与「2-8 分钟靠分镜系统拼接」最贴合的一篇，直接顶「分镜规划 + 跨镜连续性」工位。GRPO 学 14K 部电影的运镜——这个数据来源（真实电影）也比用闭源 API 输出当教师在法律上干净得多，值得优先考虑作为替代路线。
- 来源: https://arxiv.org/abs/2609.10943

## CamWorldQA  (学术（arXiv 2608.18710，2026-08-19）)
- 类别/成熟度: technique / **research** | 许可: 论文 + 基准 | 成本: —
- 是什么: 首个相机控制视频生成的感知质量评估基准。
- 关键事实:
    * 2026-08-19 提交
    * 720 段生成视频，来自 6 种代表性生成方法、20 段多样源视频、6 种相机轨迹，每段有主观实验得到的人评质量分
    * 指出通用 VQA 无法捕捉相机控制生成的特有特征：viewpoint consistency、motion coherence、content preservation
    * 配套 CWQA：无参考质量评估网络，三分支（空间特征、时序运动特征、光流特征）
- 长片作用: 顶「运镜质检」工位。你要验证自己的运镜方案到底有没有效，需要一个不是 VBench 的度量——CamWorldQA 的三个维度（视角一致性、运动连贯性、内容保持）就是你的验收标准。
- 来源: https://arxiv.org/abs/2608.18710

## Wan 2.2 开源现状（截至 2026-09-19 核实）  (阿里通义万相)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: 14B 推理单卡需 80GB（或 offload）；Animate-2 720P 训练/推理默认 8×A800
- 是什么: 你打算做 LoRA 的基座。需要注意：Wan 官方组织下目前没有 2.5 / 2.6 的开放权重。
- 关键事实:
    * GitHub Wan-Video 组织下仓库：Wan2.2（更新 2026-03-17）、Wan2.1（2026-03-05）、Wan-Dancer（2026-07-17）、Wan-Animate-2（2026-08-08）、Wan-skills（2026-04-17）。没有 Wan2.5 或 Wan2.6 仓库
    * HuggingFace Wan-AI 组织下最新权重：Wan2.2-Animate-2-14B（Aug 9）与其 Distilled 版（Aug 13）、Wan-Dancer-14B（Jul 17）；T2V-A14B / I2V-A14B / TI2V-5B 仍是 2025-08-09
    * Wan2.2 架构：T2V-A14B 与 I2V-A14B 是 MoE，总参 27B、每步激活 14B；TI2V-5B 是 5B 稠密模型
    * 许可 Apache-2.0
    * 显存：14B 单卡「at least 80GB VRAM」；5B 单卡「at least 24GB VRAM（如 RTX 4090）」；可用 --offload_model 降低
    * 分辨率：14B 支持 480P/720P；TI2V-5B 支持 720P@24fps，消费级 GPU 上 9 分钟内出 5 秒视频
    * Wan2.2-Animate-2-14B：2026-08-07 发布，14B，Apache-2.0，默认配置为 8×A800 跑 720P，480P 需 2×A800；蒸馏版仅需 10 步推理且不用 CFG；带文本驱动的视角控制，可把相机视角与驱动视频解耦；有面向实时流的 Lite 变体
    * 官方警告（Animate 相关）：「we do not recommend using LoRA models trained on Wan2.2, since weight changes during training may lead to unexpected behavior」
- 长片作用: Wan2.2-Animate-2 才是数字人产线该用的入口，不是 T2V-A14B：它带身份保持 + 文本驱动视角控制 + 10 步蒸馏版 + 实时 Lite 变体，一条线覆盖了你的角色一致性、运镜和加速三个需求。而且它自带的「文本驱动视角控制」在一定程度上已经替代了你想训的运镜 LoRA。官方那条「不推荐在 Animate 上用 Wan2.2 训的 LoRA」的警告，是你方案的又一个工程约束。
- 来源: https://github.com/Wan-Video/Wan2.2 https://github.com/Wan-Video https://huggingface.co/Wan-AI

## MiniMax-H3  (MiniMax)
- 类别/成熟度: open-weights / **production** | 许可: MiniMax H3 Community License Agreement（非标准开源许可，商用需核对） | 成本: 33B，推荐 4 GPU 部署
- 是什么: 2026 年新出现的开放权重视频基座，规模与能力都超过 Wan2.2 单体，值得在选型时与 Wan 一起评估。
- 关键事实:
    * 33B 参数稠密单流 Transformer（其中约 13B 在 AdaLN 分支），三维多模态 RoPE
    * HuggingFace 上 4.45M 下载、5.47k likes，模型页最近更新 8 月 13 日
    * 许可：MiniMax H3 Community License Agreement（不是 Apache-2.0，需逐条核对商用条款）
    * 输出 4-15 秒，默认 768p，最高支持到 2K；比例支持 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16
    * 三段式结构：H3-Context-IR 预处理 + H3-Base 生成 + H3-Regenerate-2K 放大
    * SGLang 部署推荐 4 GPU
    * 生态已跟上：musubi-tuner 与 diffusion-pipe 都已支持其 LoRA 训练；lightx2v 有 Minimax-h3-Turbo（1.53M 下载）少步加速版
    * H3 还有配套的 Prompt-Rewriter LoRA（8B 版本等）
- 长片作用: 如果你的目标是「单镜观感接近 Seedance 2.0」，33B 的 H3 比 14B 的 Wan2.2 起点更高，15 秒上限也比 Wan 长，2K 输出对成片有实际价值。选型时不应该默认锁 Wan。代价是许可不是 Apache-2.0，商用前必须逐条读。
- 来源: https://huggingface.co/MiniMaxAI/MiniMax-H3 https://huggingface.co/MiniMaxAI https://huggingface.co/lightx2v

## musubi-tuner  (kohya-ss)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: 视频 LoRA 训练 24GB 起步（低分辨率）；14B 720p 实际需 48-80GB 或 block swap
- 是什么: 当前最活跃的视频模型 LoRA 训练框架，是你这条路线的实际施工工具。
- 关键事实:
    * 支持 LoRA 训练的视频模型：HunyuanVideo、Wan2.1/2.2、FramePack、MiniMax-H3
    * 同时支持 FLUX.1 Kontext、FLUX.2、Qwen-Image、Z-Image、HiDream-O1、Kandinsky 5、Ideogram 4 等图像模型
    * 显存：「12GB or more recommended for image training, 24GB or more for video training」，实际需求随分辨率与设置变化
    * 低显存建议：分辨率降到 960×544 或更低，配合 --blocks_to_swap、--fp8_llm 等省显存选项
    * 最新更新 2026-09-16（新增 MiniMax-H3 支持、Krea 2 的 ConvRot int8 量化、数据集配置改进）
    * 许可：Apache-2.0（部分代码沿用 Diffusers 等原许可）
    * 未给出 Wan 14B 的具体显存数字
- 长片作用: 你的「训 LoRA」工位就落在这里。三天前还在更新，且已支持 MiniMax-H3，说明生态跟得上模型迭代。
- 来源: https://github.com/kohya-ss/musubi-tuner

## diffusion-pipe  (tdrussell)
- 类别/成熟度: framework / **production** | 许可: GPL-3.0 | 成本: 提供 14B 最小显存配置模板，但未给出具体 GB 数
- 是什么: 另一个主流视频 LoRA 训练框架，偏多卡/大模型，许可需注意。
- 关键事实:
    * 支持 SDXL、Flux、LTX-Video、HunyuanVideo (t2v)、Cosmos、Lumina Image 2.0、Wan2.1 (t2v 与 i2v)、Wan2.2、Chroma、HiDream、SD3 等
    * 可直接在量化模型上训 LoRA
    * 仓库自带 wan_14b_min_vram.toml 示例配置，用 AdamW8BitKahan 优化器 + block swapping + activation checkpointing 压显存
    * 许可 GPL-3.0（copyleft，衍生作品需同许可）——与 musubi-tuner 的 Apache-2.0 不同，商业闭源产线要注意
    * 更新节奏：2026-01 加 Flux 2，2026-08 加 MiniMax H3
- 长片作用: GPL-3.0 这一条对商业平台是实际风险点：如果你的训练代码与它深度耦合并分发，会触发 copyleft。产线用 musubi-tuner（Apache-2.0）更安全。
- 来源: https://github.com/tdrussell/diffusion-pipe

## AesRM / Aligning Human Sense / Human detectors as reward models  (多家（arXiv 2604.28078 / 2608.21425 / 2601.14037）)
- 类别/成熟度: technique / **research** | 许可: 论文 | 成本: 人体检测器方案成本极低（现成检测器推理）
- 是什么: 三条 2026 年的奖励模型新思路，对人工打分体系的设计有直接参考价值。
- 关键事实:
    * AesRM（2604.28078，2026-04-30）：分层美学奖励模型，用专家标注，在 visual、fidelity、plausibility 三个维度上用 GRPO 训练——「专家级反馈」而非众包
    * Aligning Human Sense（2608.21425，2026-08-16）：基于 Wasserstein 的分布对齐，把校准后的奖励整合进 GRPO，做偏好分布匹配而非点估计
    * Human detectors are surprisingly powerful reward models（2601.14037 v2，2026-01-15）：直接用人体检测器的置信度当奖励 + GRPO，显著改善人体运动合成——极其便宜的奖励信号
    * 另有 Shell-LCC（2606.30248 v2）：「Your Data Manifold is Secretly a Reward Model」，用数据流形结构产生稠密奖励，省掉奖励模型的计算开销
- 长片作用: 2601.14037 对数字人产线是个便宜的立竿见影方案：用人体检测器置信度当奖励，专治「人体畸变/多手多脚」这个数字人最高频的废片原因，不需要建人工标注队伍。AesRM 的「专家标注而非众包」也提醒你：人工打分的质量比数量更重要（呼应 LiFT 只用 1 万条就见效）。
- 来源: https://arxiv.org/abs/2604.28078 https://arxiv.org/abs/2608.21425 https://arxiv.org/abs/2601.14037

### 建议
# 直接回答：用过审非露骨镜头当教师、在 Wan 上 LoRA 学运动/运镜/质感

## 一、技术上成立吗？—— 分三段答，两段不成立、一段成立

**学「质感」：成立，但收益被高估。** 风格/色彩/质感 LoRA 从生成数据学，这是已验证的做法（JourneyDB 400 万张 Midjourney 图是公开先例）。50-200 个片段就能起效。问题在于你学到的是 Seedance 的调色和纹理，不是它的「单镜观感」。Seedance 2.0 好看的主因是 30 秒级的时序一致性和物理合理性，这些不在风格 LoRA 的能力范围内。

**学「运镜」：技术上不成立，应该放弃。** 三条独立证据指向同一结论：
1. MotionCtrl（2312.03641）确立的设计原则是运镜控制应当 **appearance-free** ——运镜条件不该携带外观信息。而 LoRA 恰恰是把外观和运动混在一起学。
2. CameraCtrl（2404.02101）的数据消融结论是：教师视频的外观必须贴近基座分布。Seedance 输出的外观分布离 Wan 很远，正好违反。
3. 2026 年 5 月的 Probing into Camera Control（2605.14815）直接说：用配对数据微调运镜会 "bias the model toward a narrow output distribution and compromise the strong prior learned by the base model"，且发现所有主流视频模型共享系统性的运镜偏差——这不是提示词写法问题，是模型能力问题。

**运镜应该走几何条件注入，不走 LoRA。** 按落地难度排序的三个替代方案：
- **Go-with-the-Flow**（2501.08331，Netflix Eyeline Studios）：只改噪声不改权重，同时覆盖运镜、局部运动控制、运动迁移，微调开销极小，且**不会与 Lightning 少步 LoRA 冲突**。优先做这个。
- **Wan2.2-Fun Control-Camera**（阿里 PAI，Apache-2.0）：今天就能下载，但只明确支持 pan 与 rotation，推拉升降跟拍不在列。
- **Uni3C**（2504.14899）：点云控相机 + SMPL-X 控人体，两路解耦，最贴合数字人场景。

**学「运动」：成立，但换个对象。** 你真正该学的不是 Seedance 的运动，而是你自己角色的表演风格——这用真人动捕或 UE5 合成数据更干净、更可控、零法律风险（ReCamMaster 的 UE5 多机位数据集就是可复制的范式）。

## 二、一个你必须知道的隐藏成本：LoRA 与少步加速二选一

DART（2609.20051，2026-09-17，两天前）实测：为满步轨迹训的 LoRA 挂到 4 步 Wan2.2 上，**macro functional retention 是 -0.4644**（负值＝功能效果被反转），修复后也只到 +0.1349。

这意味着：你自训的 LoRA + Wan2.2-Lightning 4 步加速，**不能简单叠加**。2-8 分钟片按 5 秒/镜算是 24-96 个镜头，满步推理的成本是 4 步的约 20 倍。这个二选一决定了整条产线的单位经济性，必须在立项时就算清，不能等训完 LoRA 才发现。

叠加一条官方警告：Wan2.2-Animate 的 README 明确写「不推荐在 Animate 上使用 Wan2.2 训练的 LoRA」——而 Animate-2 恰恰是数字人该用的入口。

## 三、成本

**教师数据采集**（以 1000 个 5 秒片段计）：
| 教师 | 单条价 | 1000 条 |
|---|---|---|
| Seedance 2.0-mini 720p | 2.48 元（4折后约 1.0） | 约 1,000-2,500 元 |
| Seedance 2.0 720p | 4.97 元 | 约 5,000 元 |
| Seedance 2.5 1080p | 18.71 元 | 约 18,700 元 |
| Veo 3.1 Lite 720p 8s | $0.40 | 约 2,900 元 |

**训练**：musubi-tuner（Apache-2.0，2026-09-16 刚更新），视频 LoRA 训练 24GB 起步，Wan 14B 720p 实际需 48-80GB 或开 block swap。租 H100 按 $2-3/h、单个 LoRA 20-40 卡时估算，约 **300-900 元**。

**单个 LoRA 全流程：约 2,000 元（mini 档）到 2.5 万元（1080p 档）**，不含人工筛选工时。

**这笔钱的对照组**：同样的钱，Seedance 2.5 出 1080p 成片约 3.74 元/秒，2-8 分钟全片一次过约 **450-1,800 元**。也就是说，**训一个 LoRA 的钱够你直接用 Seedance 出 10-40 条成片**。这条路线只有在出片量级到数百条以上时才可能回本，且前提是 LoRA 真的有效（上面已论证运镜部分无效）。

## 四、法律上必须注意的条款（已逐条取得原文）

| 厂商 | 条款与生效日 | 原文 | 严重度 |
|---|---|---|---|
| **Google / Veo** | Google ToS，2026-07-30 | "You must not use AI-generated content from our services to develop machine learning models or related AI technology." | **最重**。不限「竞争性」，任何 ML/AI 技术都禁止。没有回旋空间 |
| **OpenAI / Sora** | Terms of Use，2026-01-01 | "Use Output to develop models that compete with OpenAI."；同时禁止 "Automatically or programmatically extract data or Output" | **重**。批量程序化拉输出这个动作本身就违约，不用等到训练那一步 |
| **OpenAI 企业/API** | Services Agreement 3.3(e)(f)，2026-01-01 | "except for a Permitted Exception, use Output to develop artificial intelligence models that compete with OpenAI's products and services" | 限定词是"compete with"，有解释空间，但 OpenAI 可单方更新（16.13） |
| **Google Gemini API** | 2026-03-23 | 免费档 Google 会用你的输入输出训练自己的模型；付费档不会 | 反向风险：免费档等于把你的 prompt 工程送给 Google |
| **字节 / Seedance** | 未取得原文，见 uncertainClaims | — | 未知 |
| **快手 / 可灵** | 未取得原文，见 uncertainClaims | — | 未知 |

**另外两条和内容策略直接相关的硬约束**（来自方舟 API 文档，已核实）：
- Seedance 2.5 与 2.0 系列「**不支持直接上传含有真人人脸的参考图 / 视频**」。对「数字人」产品是结构性限制——只能用虚构角色、平台预置虚拟人像、或本账号近 30 天内由这些模型生成的含人脸视频滚动复用。这一条与你「角色虚构、不碰真人肖像」的策略方向一致，但也意味着你的角色资产被锁在平台 30 天窗口内。
- `watermark` 参数默认示例为 `true`（右下角 AI 生成水印）。**如果用带水印的输出做训练数据，水印会被学进 LoRA**。必须显式设 `false`，而这又可能与平台的 AIGC 标识合规要求冲突（中国《人工智能生成合成内容标识办法》）。

**一个正在逼近的技术性法律风险**：RAPID（2609.15799，2026-09-14）是专门防御「黑盒输出蒸馏」的论文，把对抗扰动烘进 VAE decoder，对全体用户生效且难以事先检测。厂商有能力随时上线这类防御。把蒸馏当长期护城河是错的，最多是一次性加速手段。

## 五、偏好对齐：可以做，但你设想的飞轮设计有致命缺陷

**技术栈是现成的**：DanceGRPO（2505.07818）+ VideoReward/VideoAlign（Apache-2.0，Qwen2-VL-2B）是可以直接拼起来的开源 RLHF 栈——DanceGRPO 的评测本来就用 VideoAlign 当奖励之一。标注量级参考 LiFT：**1 万条带理由的人工标注**就让 CogVideoX-2B 在全部 16 个指标上超过 5B。

**但「人工打分只回收过检镜进下一轮」这个设计会失败**，五篇论文从不同角度指向同一结论：
1. **D2-Align**（2512.24146）正式命名 Preference Mode Collapse：模型收敛到「窄的、高分的输出」——单一画风、普遍过曝，多样性严重退化。配套 DivGenBench 可量化。
2. **GARDO**（2512.24138）：代理奖励分数上涨的同时，真实质量下降、多样性坍缩。
3. **DenseDPO**（2506.03517）：朴素 DPO 有系统性 **motion bias**——偏好几乎不动的画面（静态没有运动伪影）。**你的数字人会越训越僵，这是机制性的，不是调参问题。**
4. **选择偏差致坍缩**（2606.13732）：当验证者只看到目标流形的一小块、碎片化且有偏的切片时，"selection itself becomes biased"。**你的审核员就是这个有偏验证者。**
5. **人类策展反噬**（2605.29267）：单模型自消费下人工策展总是有益（Ferbach 2024），但在多模型交互场景下，跨模型影响可以削弱甚至**反转**这个效果。你同时用 Seedance + Veo + 自训模型，正好落在这个场景里。

**关于「混通用数据的比例经验值」——学界目前没有可引用的数字。** 最新的 Fisher-Rao 论文（2609.18878，2026-09-16，三天前）明确把这当作未解问题，指出此前欧氏度量下的界在高维会退化为 vacuous，并给出了维度稳定的新界，但**没有给出可套用的百分比**。任何声称「掺 X% 通用数据」的说法都是经验直觉，不是已验证结论。你只能自己做消融。

**可落地的缓解措施**（都有论文支撑）：
- 保留一个**不参与训练的固定 holdout prompt 集**，每轮同时测过检率和多样性（DivGenBench 式指标），而不是只测过检率。过检率单调上升而多样性下降，就是 PMC 的信号。
- 用 **mode-covering** 而非 mode-seeking 目标：TMPO（2605.10983）的轨迹级奖励分布匹配，多样性相对 SOTA +9.1%。
- 用**数据做正则**而非只调 KL 系数：DDRL（2512.04332）。工程版处方就是 Reg-DPO（把 SFT loss 加进 DPO）和 RealDPO（用真实视频做正样本）。
- **人工标注重质不重量**：AesRM（2604.28078）用专家级标注而非众包；LiFT 用 1 万条带理由的标注就够。
- **便宜的第一步**：用人体检测器置信度当奖励（2601.14037），专治数字人最高频的废片原因（人体畸变、多手多脚），零标注成本。

## 六、结论与建议路线

**不要做的**：用 Seedance/Veo 输出训 Wan 的运镜 LoRA。技术上违反 appearance-free 与分布匹配两条原则，法律上 Google 条款直接封死、OpenAI 条款字面覆盖，经济上训一个 LoRA 的钱够直接出 10-40 条成片，还要面对 DART 揭示的「LoRA 与 4 步加速二选一」。

**建议的替代路线，按优先级**：
1. **运镜**：Go-with-the-Flow（改噪声不改权重，不冲突少步加速）打底，Uni3C（点云+SMPL-X）补复杂运镜。**分镜规划层**用 Auteur 的 human-centric DSL（景别/角度/构图 → 6-DoF），这是把导演语言变成参数的正确抽象层。
2. **多机位与跨镜一致性**：ReCamMaster——同一场表演生成一次、重渲多机位，正反打天然一致。UE5 合成多机位数据完全无版权风险。
3. **基座选型不要默认锁 Wan**：MiniMax-H3（33B、4-15 秒、最高 2K、已被 musubi-tuner 与 diffusion-pipe 支持、有 Turbo 少步版）起点比 Wan2.2 14B 高。代价是 MiniMax H3 Community License 不是 Apache-2.0，商用前要逐条读。数字人入口用 Wan2.2-Animate-2-14B（2026-08-07，Apache-2.0，带身份保持 + 文本驱动视角控制 + 10 步蒸馏版 + 实时 Lite）。
4. **合法的「学电影运镜」路线**：CamPilot（2609.10943，9 天前）用 GRPO 从 **14K 部真实专业电影**学运镜规划，专攻多镜连续性。数据来源是真实电影而非闭源 API 输出，法律上干净得多。虽然暂无权重，但这是方法论上该抄的方向。
5. **对齐**：DanceGRPO + VideoReward 起步，先跑 VideoDPO 的自动偏好对构造，人工只投在模型分不清的边界样本上。质检用 Reward Lightning 的 latent reward model 思路（不解码即打分，便宜几倍）+ CamWorldQA 的三维运镜指标。
6. **如果一定要用闭源输出**：只用 Seedance（中国实体、条款未取得、且你已是付费客户），不碰 Veo 和 Sora；只学质感不学运镜；watermark 显式设 false；并接受这是一次性加速而非护城河（RAPID 类防御随时可能上线）。**在投入前先让法务取得火山方舟服务协议原文。**

**建议的下一步**：先花 2,500 元（Seedance 2.0-mini 1000 条）+ 900 元训练，做一个**质感 LoRA 的对照实验**，用 CamWorldQA 三维指标和 DivGenBench 多样性指标做前后对比。这个实验的成本远低于一个月的工程投入，但能把「成立/不成立」从推测变成数据。运镜部分直接跳过 LoRA，从 Go-with-the-Flow 开始。

### 存疑
- 【未取得原文】火山引擎/火山方舟（Seedance）对「用模型输出再训练模型」的服务协议条款。尝试了 volcengine.com/docs/6257/64963、6269/64921、82379/1263272 等路径，返回的都是费用中心/SDK 安装等无关文档；Bing 检索也未定位到协议原文页。这是你唯一在用的付费教师通道，**建议直接让法务向火山引擎商务索取《火山方舟大模型服务协议》全文**，重点看输出内容的知识产权归属与再训练限制。在取得原文前，不要假设它比 OpenAI/Google 宽松。
- 【未取得原文】可灵 AI（快手）的服务协议中关于输出再训练的条款。klingai.com/legal/terms-of-service 与 kling.ai/legal/terms-of-service 均为 JS 渲染，jina reader 只拿到首页营销内容；app.klingai.com/global/dev/.../legal/termsOfService 返回空白；中文 API 文档托管在 docs.qingque.cn 需登录。无法给出任何 Kling 条款判断。
- 【用户给的硬约束需要修正，已核实】(1)「官方单段 4-15 秒（2.5 约 30 秒）」——2.0 系列确为 duration [4,15]；Seedance 2.5 为 [4,30] 且文档明确写「支持 30 秒视频连贯直出」，不是「约 30 秒」。(2)「多模态参考最多 9 图 + 3 视频 + 3 音频」——这只适用于 2.0 系列；**Seedance 2.5 已放宽到 0-30 张图 + 0-10 个视频 + 0-10 段音频**，且 2.5 支持仅传音频。(3) 2.5 已「全面公开」，不再是受限内测。这三条会实质改变你的分镜系统设计（30 张参考图对跨镜角色一致性是量变到质变）。
- 【未核实】Wan 2.5 / 2.6 是否存在闭源 API 版本。已核实的是：截至 2026-09-19，GitHub Wan-Video 组织与 HuggingFace Wan-AI 组织下**都没有** Wan2.5 或 Wan2.6 的仓库或权重，最新开放权重是 Wan2.2-Animate-2-14B（2026-08）与 Wan-Dancer-14B（2026-07）。但阿里云 Model Studio 的 Wan 文档页返回 404，无法确认 API 侧是否有更高版本。
- 【无官方数字】Wan2.2 A14B（14B 激活）LoRA 训练的具体显存与 GPU 小时数。musubi-tuner 只给「视频训练 24GB 以上」的通用建议，diffusion-pipe 提供 wan_14b_min_vram.toml 模板但不列 GB 数。正文里的「20-40 H100 卡时 / 300-900 元」是我基于 1000 条片段、5000-10000 步的**外推估算**，不是实测或官方数字，实际可能偏差 2-3 倍。
- 【无公开实验】Seedance 的 AI 生成水印（watermark=true 时右下角标识）对下游 LoRA 训练的污染程度。「水印会被学进 LoRA」是从图像 LoRA 的已知行为外推的合理推断，但没有针对 Seedance 水印的公开消融实验。
- 【学界未给出】「混通用数据的比例经验值」。这是任务明确要求的一项，但检索结果是：最新的 Fisher-Rao 论文（2609.18878，2026-09-16）把「最小人类/合成数据比」明确列为 open question，并指出此前的欧氏度量下界在高维下 vacuous。**任何具体百分比（10%、30%、50%）目前都没有可引用的来源**，社区流传的数字属于经验直觉。
- 【无判例】「用过审的非露骨镜头当教师」这一特定行为的法律判例或执法先例。ToS 违约的实际后果（封号、索赔、禁令）在生成式视频领域尚无公开案例可援引。RAPID 论文证明厂商在技术上防御，但不等于法律上已追诉。
- 【相对数字需谨慎】DanceGRPO 的「outperforms baseline methods by up to 181%」。这是相对特定 baseline 在特定指标上的相对提升，不是绝对质量提升 181%。同理 VisionReward 的「31.6% higher pairwise win rate」也是相对 VideoScore 的对照结果。
- 【未验证】MiniMax H3 Community License Agreement 的具体商用条款（是否有月活/收入门槛、是否要求署名、是否禁止用于特定用途）。只确认了许可名称，未取得全文。在把 H3 作为商业产线基座前必须逐条读完。
- 【未取得】Google Cloud Vertex AI 的 Service Specific Terms 中对 Veo 输出的处理是否与消费端 Google ToS 不同。企业合同通常有不同措辞，如果确实想用 Veo 当教师，这是唯一可能的合规路径，但本次未能验证。
- 【方法有效性未在你的场景验证】CamPilot、Auteur、Probing into Camera Control、TMPO、DART 等 2026 年论文均未见公开权重或可复现实现，maturity 标为 research。它们的结论方向可信（多篇独立论文交叉印证），但具体数字能否在 Wan/H3 + 数字人场景复现，需要自己验证。


====================================================================================================
# [evaluation-qc] AI 生成视频的自动化质检与评测体系（面向 2-8 分钟多镜头数字人长片产线）

## VBench (v1)  (上海AI Lab / Vchitect (南洋理工 S-Lab))
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: 多维度可 torchrun 分卡；单卡 8-12GB 可跑 consistency/flicker/aesthetic 这几个轻量维度
- 是什么: 最广泛使用的文生视频评测套件，把「视频好不好」拆成 16 个可分别计算的维度，每个维度都有独立可调用的 Python 实现，可以脱离官方 prompt suite 对自己的视频跑单维度打分。
- 关键事实:
    * 2024-02 发布，CVPR 2024 Highlight
    * 16 维度：subject_consistency / background_consistency / temporal_flickering / motion_smoothness / dynamic_degree / aesthetic_quality / imaging_quality / object_class / multiple_objects / human_action / color / spatial_relationship / scene / temporal_style / appearance_style / overall_consistency
    * License: Apache-2.0，pip install vbench，权重缓存在 ~/.cache/vbench
    * PyPI v0.1.5（2025-01）修了预处理并支持 torch>=2.0
    * HuggingFace leaderboard 已收录 40+ 模型
    * 依赖 Detectron2 + CUDA<=12.1，安装链较脆
- 长片作用: 不要整套跑。真正能进产线门禁的只有 4 个：temporal_flickering（纯 CPU）、subject_consistency（DINO）、background_consistency（CLIP）、motion_smoothness。这 4 个直接对应「镜头级废片筛选」工位，剩下 12 个是论文对比用的，产线没价值。
- 来源: https://github.com/Vchitect/VBench https://arxiv.org/abs/2311.17982

## VBench temporal_flickering 的确切算法  (Vchitect)
- 类别/成熟度: technique / **production** | 许可: Apache-2.0 | 成本: 0 GB VRAM，纯 CPU，成本可忽略
- 是什么: 时序闪烁指标的可复现实现，本质是相邻帧 MAE 取反归一化。是整个质检体系里唯一零模型依赖、纯 OpenCV、可在 CPU 上跑满吞吐的指标。
- 关键事实:
    * 公式：score = (255.0 - mean(MAE(frame_i, frame_i+1))) / 255.0，输出 0-1，越高越不闪
    * MAE 实现：np.mean(cv2.absdiff(img1.astype(f32), img2.astype(f32)))
    * 官方要求「please ensure the video is static」——有静态过滤器，因为大运动镜头会被误判为闪烁
    * 零 GPU、零模型权重，1080p 5秒片段 CPU 上 < 1s
- 长片作用: L0 层门禁的第一道闸。但必须做运动补偿改造：直接用原版公式，一个推镜头会被判成重度闪烁。产线做法是先用 DIS/Farnebäck 光流把 frame_i+1 warp 回 frame_i，再算 MAE（即 warping error E_warp），才能把「真闪烁」和「真运动」分开。
- 来源: https://github.com/Vchitect/VBench/blob/master/vbench/temporal_flickering.py

## VBench subject_consistency 的确切算法（可直接改造成身份漂移检测）  (Vchitect)
- 类别/成熟度: technique / **production** | 许可: Apache-2.0 | 成本: DINO ViT-B/16 约 86M 参数，FP16 下 <1GB VRAM；CPU 上 16 帧约 2-4s
- 是什么: 用 DINO 特征做帧间余弦相似度，同时对「首帧」和「前一帧」算两路相似度再平均。这是跨帧主体一致性的事实标准实现。
- 关键事实:
    * 模型：torch.hub 加载的 DINO（ViT-B/16）
    * 特征 L2 归一化：F.normalize(feat, dim=-1, p=2)
    * 双路：sim_fir = cos(first_frame_feat, feat_i)，sim_pre = cos(prev_frame_feat, feat_i)，cur_sim = (sim_pre + sim_fir)/2
    * 聚合：sim_per_video = sum(cur_sim) / (N_frames - 1)
    * 主流模型在 VBench 上该项落在 0.90-0.98 区间
- 长片作用: 关键改造：VBench 只在单片段内算。2-8 分钟片要的是「跨镜头」一致性——把每个镜头抽 3 帧（首/中/尾）算 DINO 特征，与该角色的 anchor 特征库比，得到 cross-shot drift。这是分镜拼接产线里最重要的一个数，直接决定观众会不会觉得「换人了」。
- 来源: https://github.com/Vchitect/VBench/blob/master/vbench/subject_consistency.py

## VBench-2.0  (Vchitect / 上海AI Lab)
- 类别/成熟度: framework / **usable** | 许可: Apache-2.0 | 成本: 全量跑需 8×A100 级别；单维度（如 Human Anatomy）单卡 24GB 可跑
- 是什么: VBench 的第二代，不再评「画面漂不漂亮」，改评「内在忠实度」——物理定律、常识、人体解剖学正确性、构图完整性。含一条专门的人体异常检测（Human Anomaly Detection）管线。
- 关键事实:
    * arXiv 2503.21755，2025-03-27 首次提交，2025-08-20 修订
    * 18 维度分 5 大类：Human Fidelity（Human Anatomy / Human Identity / Human Clothes）、Controllability（Dynamic Spatial Relationship / Dynamic Attribute / Motion Order Understanding / Human Interaction / Complex Landscape / Complex Plot / Camera Motion）、Creativity（Diversity / Composition）、Commonsense（Motion Rationality / Instance Preservation）、Physics（Mechanics / Thermotics / Material / Multi-View Consistency）
    * 依赖模型链：LLaVA-Video-7B-Qwen2、Qwen2.5-7B-Instruct、CLIP、CoTracker、YOLO-World、InsightFace
    * 官方建议 18 维度分 18 张卡跑（上限 8 卡），单卡串行「not recommended」
    * 2025-04 单独释出 Human Anomaly Detection pipeline（vbench2/third_party/ViTDetector），含人工标注的真实/AIGC 异常数据集与训练代码
    * 2025-05 支持对自有视频跑单维度；2026-03 上线 VBench-I2V Arena
- 长片作用: 只有 Human Anatomy + Human Identity + Human Clothes 这三维直接对口数字人产线，分别顶「手指/肢体崩坏检测」「身份漂移检测」「服装连贯性检测」三个工位。Human Anomaly Detection 那条独立管线是目前唯一有公开训练数据和代码的「AIGC 人体崩坏检测器」，比自己拿 MediaPipe 手工搓规则靠谱得多，建议直接蒸馏成一个轻量 ViT 分类头挂在 L1 层。其余 15 维不要进产线。
- 来源: https://arxiv.org/abs/2503.21755 https://github.com/Vchitect/VBench/tree/master/VBench-2.0

## VBench-Long (vbench2_beta_long)  (Vchitect)
- 类别/成熟度: framework / **usable** | 许可: Apache-2.0 | 成本: 与 VBench v1 同级，单卡 12GB 够跑这 6 维
- 是什么: VBench 的长视频扩展，2024-06 发布。核心是「先切再评 + 慢快双分支」：用 PySceneDetect 把长片按语义切成短 clip，慢分支在 clip 内按全帧率评，快分支低帧率抽样跨全片评长程一致性。
- 关键事实:
    * 2024-06 发布
    * 切分：PySceneDetect 做场景检测，再按 clip_length_mix.yaml 里每维度不同的固定长度二次切
    * 慢分支 = 标准 VBench 短 clip 评估；快分支 = 低帧率采样评「long-range consistency of background scenes and foreground subjects」
    * 支持自定义长视频的 6 维：subject_consistency / background_consistency / motion_smoothness / dynamic_degree / aesthetic_quality / imaging_quality
    * temporal_flickering 需配静态过滤预处理才支持长视频
- 长片作用: 这是目前唯一开源的、直接对口「2-8 分钟成片」的评测协议。但它的假设和你反过来了：它假设你有一条长视频要切开评；你是有一堆镜头要拼。所以正确用法是把它的「快分支」逻辑单独抄出来——用你自己的分镜表代替 PySceneDetect（你本来就知道镜头边界），只保留跨 clip 的低帧率一致性计算。这直接就是成片级门禁。
- 来源: https://github.com/Vchitect/VBench/tree/master/vbench2_beta_long

## VBench++ (TPAMI 版)  (Vchitect)
- 类别/成熟度: framework / **usable** | 许可: Apache-2.0 | 成本: 同 VBench 主线
- 是什么: VBench 系列的期刊整合版，把 I2V 套件、Trustworthiness（文化/公平/偏见/安全）模块合并进主框架，并加了「adaptive Image Suite」做 I2V 公平评测。
- 关键事实:
    * 2025-11 作为 TPAMI 期刊论文发布
    * 并入 VBench-I2V（2024-03）与 VBench-Trustworthiness（2024-03）
    * Trustworthiness 单独评 culture / fairness / bias / safety
- 长片作用: I2V 套件对你有用——你的分镜产线本质是 keyframe→I2V，评测协议应该用 I2V 而不是 T2V 的那套 prompt。Trustworthiness 模块可以当作官方通道内容审核之外的一层自检，提前发现会被审核拦的镜头，省 API 费。
- 来源: https://github.com/Vchitect/VBench

## VideoScore / VideoScore-v1.1  (TIGER-AI-Lab (滑铁卢大学))
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 8B 模型，bf16 约 17-18GB VRAM；建议 24GB 卡（4090/A5000）
- 是什么: 第一个真正可用的「视频质量回归模型」：给一段视频 + prompt，直接回归出 5 个 1-5 分的子分，替代一堆手工指标。
- 关键事实:
    * 基座 Mantis-8B-Idefics2（8B）
    * 训练集 VideoFeedback：37.6K 条文生视频人工标注，覆盖 11 个生成模型
    * 5 个维度：Visual Quality / Temporal Consistency / Dynamic Degree / Text-to-Video Alignment / Factual Consistency
    * VideoFeedback-test 上 Spearman 相关 77.1，比此前指标高约 50 点
    * v1.1 于 2024-11-28 发布，支持 48 帧推理
    * HF: TIGER-Lab/VideoScore，License MIT
- 长片作用: 已被 VideoScore2 取代，除非你需要 5 个细分维度（VideoScore2 只给 3 个）。它的 Dynamic Degree 子分在产线里有个独特用途：识别「假动态」——画面在动但主体没动的空转镜头，这类片在长片里最拖节奏。
- 来源: https://github.com/TIGER-AI-Lab/VideoScore https://huggingface.co/TIGER-Lab/VideoScore

## VideoScore2  (TIGER-AI-Lab)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 / MIT（两处标注不一致，商用前需确认） | 成本: 8B，bf16 约 18GB；AWQ/FP8 量化后可压到 ~10GB 跑在单张 4080
- 是什么: VideoScore 的二代，换成 Qwen2.5-VL-7B-Instruct 基座，输出「分数 + 思维链理由」，用 SFT + GRPO 两阶段训练。目前开源自动评分器里最值得上产线的一个。
- 关键事实:
    * 2025-09-26 发布（HF 模型页），GitHub README 标 2025-10-01
    * 基座 Qwen2.5-VL-7B-Instruct，8B 参数
    * 3 维度：visual quality（清晰度/流畅度/artifacts）、text-to-video alignment、physical/common-sense consistency
    * 输出 chain-of-thought rationale，不是黑盒分数
    * VideoScore-Bench-v2 准确率 44.35%（+5.94）；4 个域外 benchmark（含 VideoGenReward-Bench、VideoPhy2）平均 50.37%
    * 训练集 VideoFeedback2：27,168 条人工标注视频，带分数 + 推理轨迹
    * License Apache-2.0（HF 页）/ MIT（GitHub 页），两处不一致
    * 推理：python vs2_inference.py --video_path=<path> --t2v_prompt=<prompt>，需 torch 2.6.0 + transformers 4.53.2
- 长片作用: 顶 L2 层「精筛打分」工位。它输出的 CoT 理由是产线金矿——不是只给你一个分，而是告诉你「第 3 秒手部出现 6 指」，这条理由可以直接作为重生成 prompt 的负向约束，也可以直接喂进人工评审界面当预填意见，把人工标注速度提 3-5 倍。不要在 L0/L1 用它，8B VLM 单镜头推理 10-30s，批量筛废片扛不住。
- 来源: https://huggingface.co/TIGER-Lab/VideoScore2 https://github.com/TIGER-AI-Lab/VideoScore2 https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2

## EvalCrafter  (腾讯 AI Lab / EvalCrafter team)
- 类别/成熟度: framework / **usable** | 许可: 未明示（仓库未声明自身许可证，商用有风险） | 成本: RAFT + deepface + CLIP 全链，单卡 12-16GB
- 是什么: 最早的一套「17 个客观指标全家桶」评测框架，把 RAFT 光流、deepface 人脸一致性、CLIP 语义对齐等现成模型打包成一条流水线。
- 关键事实:
    * 代码与 Docker 2024-01-10 发布
    * 700 条 prompt（prompt700.txt）
    * 17 个客观指标 + 主观用户打分
    * 含 RAFT 光流指标、deepface 人脸一致性、CLIP 系列语义指标
    * ECTV 数据集 2024-01-24 发布，约 10,000 条 AI 生成视频
    * 视频需按 0000.mp4 - 0699.mp4 顺序组织才能跑
    * 仓库未明确声明自身 License，只声明依赖了多个开源项目
- 长片作用: 框架本身别用（prompt 编号硬耦合、License 不明），但它的指标实现是现成抄的参考：RAFT 光流异常检测和 deepface 人脸一致性这两块可以直接摘出来做 L1 层。它是全套 benchmark 里唯一把「光流异常」当一等公民的。
- 来源: https://github.com/evalcrafter/EvalCrafter https://arxiv.org/abs/2310.11440

## T2V-CompBench  (香港大学等)
- 类别/成熟度: framework / **usable** | 许可: 未在官网明示 | 成本: MLLM 分支依赖外部 VLM（GPT-4o 级或本地 7B），Detection/Tracking 分支单卡 8GB 够
- 是什么: 专评「组合性」——多物体、多属性、时空动态关系。三类指标设计：MLLM-based、Detection-based、Tracking-based。
- 关键事实:
    * arXiv 2407.14505（2024）
    * 1,400 条 prompt，从 167 万条真实用户 query 分析得出
    * 7 个类别，覆盖 attributes / quantities / spatio-temporal dynamics
    * MLLM-based 评一致性与动态属性绑定、动作绑定、物体交互
    * Detection-based 评空间关系与物体交互；Tracking-based 评 motion binding
    * 评测了 23 个模型（17 开源 + 6 商用）
    * 官网提到 V2，但页面未给出 V2 的具体变更；页面未标注 License
- 长片作用: 对数字人对白戏用处不大（单人单主体场景组合性不是瓶颈）。真正顶用的工位是「多角色同框镜头」——两个角色对戏时谁在左谁在右、谁递了东西给谁，这类错误单靠 DINO 一致性查不出来。它的 Tracking-based motion binding 指标可以拿来查「动作绑错人」。
- 来源: https://t2v-compbench-2025.github.io/ https://arxiv.org/abs/2407.14505

## MovieBench  (Show Lab (新加坡国立) / 浙江大学)
- 类别/成熟度: framework / **research** | 许可: CC BY-SA 4.0 | 成本: 数据集，无推理成本；但 CC BY-SA 的传染性对商用产线是坑
- 是什么: 电影级长视频生成数据集/基准，三层层级结构（movie / scene / shot），带角色姓名、角色图像、角色音频标注，专门针对「多场景、连贯叙事、角色一致」这三个长视频难点。
- 关键事实:
    * arXiv 2411.15262，2024-11-22 提交，2025-03-31 修订，CVPR 2025
    * 三层数据层级：movie level（全片概览）/ scene level（中层场景一致性信息）/ shot level（具体镜头细节描述）
    * 标注含：电影剧本、角色信息（姓名 + 图像 + 音频）
    * License: CC BY-SA 4.0
    * 关键风险：官网仍写「currently organizing the corresponding data and plan to release it within the next three months」，截至核查时数据未确认实际开放
    * 论文与官网均未给出确切的电影数、时长、镜头数、角色数
- 长片作用: 它的价值不在 benchmark，在于它的「角色卡 = 姓名 + 参考图 + 参考音频」这个数据结构，正好就是你的分镜系统该有的角色档案 schema（且正好对齐 Seedance 的多模态参考位）。把它的 hierarchical schema 抄过来做你的分镜元数据格式，比自己发明一套强。但不要依赖它的数据——数据可能还没放出来，且 ShareAlike 会污染你的商用数据集。
- 来源: https://arxiv.org/abs/2411.15262 https://weijiawu.github.io/MovieBench/

## StoryEval  (华盛顿大学 / UCSC / Microsoft)
- 类别/成熟度: framework / **usable** | 许可: Apache-2.0 | 成本: 本地 72B 判官需 4×48GB；改用 API 判官（GPT-4o / Gemini）或本地 Qwen3-VL-8B 成本可降两个数量级
- 是什么: 叙事连贯性的最小可行度量：不评画质，只问「prompt 里写的 2-4 个连续事件，视频里到底演完了几个」。用 VLM 做事件完成度判定，多判官一致投票。
- 关键事实:
    * arXiv 2412.16211，2024-12-17 提交
    * 423 条 prompt，7 个事件类别，每条含 2-4 个连续事件
    * 判官：GPT-4V（GPT-4o）+ LLaVA-OV-Chat-72B，采用 unanimous voting（一致投票）
    * 评测 11 个模型，最好的平均故事完成率不超过 50%
    * License: Apache-2.0
    * 本地跑 LLaVA-OV-Chat-72B 需「至少 4 张 49G GPU」
- 长片作用: 这是整个清单里对你最直接的一条。「单镜观感接近 Seedance 2.0」是画质问题，「2-8 分钟成片不散架」是叙事问题——StoryEval 的事件完成率就是叙事工位的唯一硬指标。产线做法：分镜表里每个镜头本来就写了要演什么事件，直接拿 Qwen3-VL-8B 当判官问「这段视频里，X 有没有完成 Y 动作？是/否」，得到 per-shot event completion rate。低于阈值的镜头直接重生成。不需要用它的 423 条 prompt，只需要它的协议。
- 来源: https://arxiv.org/abs/2412.16211 https://github.com/ypwang61/StoryEval

## LoCoT2V-Bench  (哈尔滨工业大学（深圳）/ 港大)
- 类别/成熟度: framework / **usable** | 许可: 未在 arXiv 页明示 | 成本: 未披露
- 是什么: 专门为「长时长 + 复杂多场景」文生视频设计的基准，引入 HERD（Human Expectation Realization Degree）指标，并明确把 character consistency 当核心维度。
- 关键事实:
    * arXiv 2510.26412v3，2025-10-30 首次提交，Accepted by ICML 2026 (Regular)
    * 多场景 prompt 带层级元数据（character settings、camera behaviors），从真实视频构建
    * LoCoT2V-Eval 五大维度：perceptual quality / text-video alignment / temporal quality / dynamic quality / HERD
    * 核心结论：模型普遍「感知质量和背景一致性强，但细粒度文视对齐和角色一致性显著弱」
    * 代码与数据已声明释出
    * 论文页未给出 prompt 条数与视频时长的具体数字
- 长片作用: ICML 2026 中稿、专攻长复杂视频，是目前学术侧最贴近你场景的基准。它的结论直接验证了你的产线设计假设：单镜画质不是瓶颈，角色一致性和细粒度对齐才是。HERD 指标值得研究——它试图量化「人的期望被实现到什么程度」，比纯像素指标更接近你要的「成片能不能看」。
- 来源: https://arxiv.org/abs/2510.26412

## PersonaShot  (（arXiv 2608.16717，机构未从摘要页确认）)
- 类别/成熟度: framework / **research** | 许可: 未知 | 成本: 未披露
- 是什么: 2026 年 8 月的多镜头角色连贯性基准，明确针对「视频生成从单镜片段走向多镜叙事、人物是叙事锚点」这个转变，评「跨剪辑点后物理与情绪状态是否连贯」。
- 关键事实:
    * arXiv 2608.16717v1，2026-08-17 提交
    * 约 1,000 条多镜头片段
    * 16 个指标，三大类：physical continuity（物理连续性）/ affective dynamics（情绪动态）/ cinematic grammar（影视语法）
    * 三个时间层级评角色连贯：within-shot states（镜内状态）、cross-shot transitions（跨镜转场）、sequence-level trajectories（序列级轨迹）
    * 用 criterion-specific evaluators，由大型多模态 teacher 指导，对齐专家人工判断
    * 代码/数据可用性未在摘要页说明
- 长片作用: 这是唯一把「跨剪辑点连贯」正式建模成三个时间层级的基准，恰好是分镜拼接产线的核心失败模式。它的 affective dynamics（情绪是否在剪辑点突变）是其他所有基准都没有的维度——数字人对白戏里，上一镜在哭下一镜在笑是最致命的穿帮，DINO 一致性和 ArcFace 都查不出来，只有情绪连贯性指标能抓。即使代码没开源，这 16 个指标的定义本身就值得抄成你自己的评审 rubric。
- 来源: https://arxiv.org/abs/2608.16717

## DramaChain Bench  (腾讯混元 + 北京电影学院 + 北大 + 深大)
- 类别/成熟度: framework / **research** | 许可: 未知 | 成本: 未披露
- 是什么: 2026 年 9 月发布的短剧全产业链基准，评的不是单个模型而是整条流水线：剧本 → 分镜 → 关键帧图 → 镜头视频 → 成片，并专门测「上游缺陷是否级联传导到下游」。
- 关键事实:
    * arXiv 2609.00646，2026-09-01 提交
    * 五阶段链路：script / storyboard / keyframe imagery / shot-level video / finished short drama
    * 5 条评估轴（DramaChain Dimensions）在每阶段实例化，共 63 个叶子维度
    * 数据规模：5,785 条目、17,488 个专业标注员有效评分、255,925 条可溯源归因记录
    * 自动化 agentic judge 与人工标注的平均 PLCC = 0.918
    * 一作 Haoyuan Shi 与 Mingtao Chen 同等贡献；有北京电影学院参与，评审 rubric 带影视专业视角
    * 代码可用性未在摘要页说明；视频时长未披露
- 长片作用: 这是和你的系统架构最同构的一篇——它评的就是你要建的东西。最关键的一条结论方向是「缺陷级联」：如果分镜阶段角色设定写歪了，后面所有镜头重生成多少次都救不回来。这直接推导出产线设计原则——质检门禁必须前移到分镜/关键帧阶段，不能等到视频生成完再筛，否则每次废片都要烧一次 Seedance API 的钱。它的 agentic judge 对人工 PLCC 0.918 也证明了「用 VLM agent 做全链路自动评审」在 2026 年已经是可行工程，不是研究玩具。
- 来源: https://arxiv.org/abs/2609.00646

## CineForge / CineScope、VGA-BenchV2、CutCraft、StreamAV-Bench、VWG-Bench（2026 新一批）  (多家)
- 类别/成熟度: framework / **research** | 许可: 未知 | 成本: 未披露
- 是什么: 2026 年 8-9 月集中涌现的一批新基准，共同趋势是从「评单镜画质」转向「评长时程叙事、剪辑语法、多阶段链路」。
- 关键事实:
    * CineForge / CineScope（arXiv 2608.29621，2026-08-30）：多尺度指标覆盖 causal state、directorial orchestration、pacing、resource allocation，针对长时程故事驱动视频
    * VGA-BenchV2（arXiv 2608.25452，2026-08-26）：52 个子维度，联合评生成质量与美学价值，测了 12 个主流模型
    * CutCraft（arXiv 2609.08275v2，2026-09-08）：首个评「多镜头音视频生成中剪辑技法执行」的基准，含 shot-structure alignment + 多模态判断的层级评估
    * StreamAV-Bench（arXiv 2608.26336，2026-08-26）：流式音视频，含 instruction adherence 与 long-horizon stability 渐进赛道
    * VWG-Bench（arXiv 2609.11242，2026-09-10）：9 个推理维度、38 个细粒度任务，评生成器能否执行逻辑推理与规则遵守
    * 这批全部为 2026 年新论文，代码/数据成熟度均未验证
- 长片作用: CutCraft 的 shot-structure alignment 是拼接产线唯一对口的外部标准——它评的就是「这几个镜头剪在一起符不符合剪辑语法」，CineScope 的 pacing 指标评节奏。这两个正好补上所有传统指标的盲区：每个镜头单看都满分，拼起来是流水账。但都太新，2026-09 的东西不要押注，当作 rubric 设计的灵感来源，不要当作依赖项。
- 来源: https://arxiv.org/abs/2608.29621 https://arxiv.org/abs/2608.25452 https://arxiv.org/abs/2609.08275

## DOVER / DOVER-Mobile  (南洋理工 S-Lab (VQAssessment))
- 类别/成熟度: open-weights / **production** | 许可: S-Lab License（需确认商用条款） | 成本: DOVER-Mobile：CPU 1.4s/视频，<1.9GB 内存，零 GPU。批量 1000 镜头单核约 23 分钟，8 进程约 3 分钟
- 是什么: 双分支无参考视频质量评估：一支评美学（aesthetic），一支评技术质量（technical），两者解耦。DOVER-Mobile 是它的轻量版，是整个清单里唯一能在 CPU 上做到秒级、且质量相关性还过得去的视频级质检模型。
- 关键事实:
    * ICCV 2023，2023-07-17 接收；权重 2023-11-22 上 HuggingFace
    * DOVER 标准版：模型约 200MB，CPU 单视频 3.6 秒，推理峰值显存 ~1.9GB
    * DOVER-Mobile：9.86M 参数（少 5.7 倍），52.3 GFLOPs（少 5.4 倍），显存需求降 3.1 倍，CPU 单视频 1.4 秒
    * PLCC：KoNViD-1k 0.883 / 0.853（标准 / Mobile），LIVE-VQC 0.854 / 0.835，LSVQ_test 0.889 / 0.867，LSVQ_1080p 0.830 / 0.802
    * License 含 S-Lab-LICENSE 选项（S-Lab 系许可证通常限非商用，商用前必须确认）
- 长片作用: L0 层批量废片筛选的主力。1.4s/视频的 CPU 成本意味着你可以对每次生成的全部候选（包括同 prompt 多次采样）无差别打分，而不用先挑。它的美学/技术双分支拆分在产线上有实际意义：技术分低 = 编码/生成 artifact，该重生成；美学分低 = 构图/光影差，该改 prompt 或改参考图。两种处置完全不同。注意它训练于 UGC 真实视频（KoNViD/LIVE-VQC/LSVQ），对 AIGC 特有崩坏（多指、面部扭曲）的敏感度未经验证，必须和 VBench-2.0 Human Anatomy 那条线配合，不能单用。
- 来源: https://github.com/VQAssessment/DOVER https://arxiv.org/abs/2211.04894

## LAION Aesthetic Predictor V2  (LAION (Christoph Schuhmann))
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0（回归头）；CLIP ViT-L/14 为 MIT | 成本: CLIP ViT-L/14 约 428M 参数，FP16 ~0.9GB VRAM；CPU 单帧约 0.2-0.5s。回归头本身成本可忽略
- 是什么: 业界事实标准的美学打分器：CLIP ViT-L/14 图像 embedding 接一个极小的线性/MLP 头，输出 1-10 分。整个训练数据来自「你有多喜欢这张图，1-10 分」这一个问题。
- 关键事实:
    * V1：5,000 条 Simulacra Aesthetic Captions 图-评分对训练的线性模型
    * V2：SAC 扩到 176,000 对 + LAION-Logos 15,000 对 + AVA 250,000 张照片
    * backbone：OpenAI CLIP ViT-L/14 embedding（不吃原图，吃 embedding）
    * 分数范围 1-10
    * LAION-Aesthetics V2 官方子集阈值与规模：4.5+ = 12 亿对、4.75+ = 9.39 亿、5+ = 6 亿、6+ = 1200 万、6.25+ = 300 万、6.5+ = 62.5 万
    * 权重文件如 sac+logos+ava1-l14-linearMSE.pth，回归头本身只有几百 KB
    * License: Apache-2.0
- 长片作用: L0 层的第二个闸，也是唯一 License 干净（Apache-2.0 + MIT）可放心商用的美学指标。关键实践：不要对全片算一个平均分，要算逐帧分的均值和最小值——长片最怕的是「大部分帧好看，中间两秒崩了」，均值会把它掩盖掉，min 值才抓得住。阈值参考：6+ 这档在 LAION 5B 里只占 1%（12M/1.2B），说明 6.0 是一个相当严的线；5.0-5.5 更适合做产线的及格线，6.0 以上做「可直接上片」的优选线。
- 来源: https://laion.ai/blog/laion-aesthetics/ https://github.com/christophschuhmann/improved-aesthetic-predictor

## Q-Align / OneAlign  (Q-Future (南洋理工 S-Lab))
- 类别/成熟度: open-weights / **usable** | 许可: S-Lab License（非商用倾向，商用需确认） | 成本: mPLUG-Owl2 约 8B 级，推理 bf16 约 17GB；不是轻量指标
- 是什么: 把视觉打分当成「让 LMM 学人类的离散评级词（excellent/good/fair/poor/bad）」的多模态打分基座，一个模型同时做 IQA（图质）、IAA（图美学）、VQA（视频质量）。
- 关键事实:
    * ICML 2024，2024-07-24
    * 基座 mPLUG-Owl2，需 transformers >= 4.36.1
    * HF 模型 ID：q-future/one-align
    * 训练硬件：单任务至少 4×A6000 或 2×A100；OneAlign 全量训练需 8×A6000 或 4×A100
    * 视频是按帧分析再聚合，不是原生视频输入
    * 仓库双许可：LICENSE + S-Lab-LICENSE
    * 也可通过 IQA-PyTorch 调用：pyiqa.create_metric('qalign')
- 长片作用: 顶 L2 层「美学精评」，但性价比不如 VideoScore2——同样吃 ~18GB 显存，VideoScore2 给的是三维带理由的打分，Q-Align 只给一个质量分。除非你特别需要它的 IQA/IAA 分离能力（关键帧选图阶段用 IAA 挑 keyframe 确实合适），否则 L2 用 VideoScore2 即可。License 也比 LAION 那条路脏。
- 来源: https://github.com/Q-Future/Q-Align https://huggingface.co/q-future/one-align

## IQA-PyTorch（pyiqa）许可证陷阱  (Chaofeng Chen)
- 类别/成熟度: framework / **production** | 许可: PolyForm Noncommercial 1.0.0 —— 禁止商用 | 成本: brisque/niqe 纯 CPU 毫秒级；深度指标 CPU 上 0.5-3s/帧
- 是什么: 把 brisque / niqe / nima / dbcnn / musiq / maniqa / topiq / clipiqa / laion_aes / qalign 等十几个无参考质量指标统一封装的工具箱，一行 pyiqa.create_metric(name, device='cpu') 就能调，所有列出的 NR 指标都支持 CPU。
- 关键事实:
    * 支持的 NR 指标：qalign、clipiqa、musiq、maniqa、topiq、laion_aes、nima、dbcnn、brisque、niqe
    * 全部可指定 device='cpu' 运行
    * brisque / niqe 是传统方法，无深度模型，计算开销极低
    * 调用：iqa_metric = pyiqa.create_metric('metric_name', device=device); score = iqa_metric(image_path)；或 CLI：pyiqa [metric_name] -t [image_path]
    * 关键：工具箱采用 PolyForm Noncommercial License 1.0.0，部分组件为 NTU S-Lab License
- 长片作用: 工程上它是最省事的「一把梭」方案，但 PolyForm Noncommercial 对商用拍片平台是硬红线。正确做法：开发/验证期用 pyiqa 快速对比各指标哪个和你的人工打分相关性最高，选定后在产线上用 Apache-2.0 的原始实现（LAION 那条线）重写，不要把 pyiqa 打进生产镜像。brisque/niqe 本身是公开算法（OpenCV contrib 也有 BRISQUE），可以绕开这个许可证单独实现。
- 来源: https://github.com/chaofengc/IQA-PyTorch

## VQAScore / t2v_metrics  (CMU (Zhiqiu Lin) 等)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: qwen3-vl-2b 约 5GB VRAM 可跑；主流配置需 40GB+
- 是什么: 用「向 VLM 提一个是非问句并取 Yes 的概率」来度量文-视对齐度的指标框架。已被 Google DeepMind（Imagen3/Imagen4）、字节 Seed、NVIDIA 采用做内部评测。
- 关键事实:
    * License: Apache-2.0
    * v3.1 支持：GPT-4o / GPT-4.1、Gemini 2.5（需 Vertex AI）、Gemma 3、PaliGemma、Qwen2.5-VL / Qwen3-VL / Qwen3.5、Qwen3-Omni（图+视频+音频）
    * v3.0 保留 CLIP-FlanT5、LLaVA-1.5、InstructBLIP 用于论文复现
    * 官方明言「Most models require 40GB+ GPUs」，小显存建议用 qwen3-vl-2b
    * 模型规模跨度 2B - 235B
    * 视频处理靠抽帧；Qwen 系支持 Dynamic FPS 采样
    * 配套 GenAI-Bench 评组合式生成
- 长片作用: 顶「文-视对齐」工位，是 CLIPScore 的正经替代品（CLIPScore 对否定、计数、空间关系基本失效）。产线用法很具体：每个镜头的分镜描述拆成 3-5 个是非问句（「画面里只有一个人吗？」「她穿的是红色外套吗？」「她在开门吗？」），逐个用 qwen3-vl-2b 算 Yes 概率，得到一个可解释的对齐向量而不是一个糊涂分。字节 Seed 自己在用这套，意味着它和 Seedance 的训练目标可能同源，作为验收指标有一致性优势。Qwen3-Omni 支持音频这点对你的带对白成片尤其有用——可以直接问「说话人的口型和音频匹配吗」。
- 来源: https://github.com/linzhiqiu/t2v_metrics

## InsightFace / ArcFace（buffalo_l）—— 身份漂移检测，但有商用禁令  (DeepInsight)
- 类别/成熟度: open-weights / **production** | 许可: 非商用研究用途（商用必须单独授权） | 成本: buffalo_s（159MB）在 ONNX CPU 上单帧约 30-80ms；GPU <1GB
- 是什么: 人脸识别事实标准，ArcFace 系列 embedding 是「跨帧身份漂移」检测的默认选择。VBench-2.0 的 Human Identity 维度就是用它。
- 关键事实:
    * buffalo_l：SCRFD-10GF 检测 + ResNet50@WebFace600K 识别，326MB，含 2d106 & 3d68 对齐、性别年龄属性
    * buffalo_s：SCRFD-500MF + MBF@WebFace600K，159MB（轻量版）
    * antelopev2：SCRFD-10GF + ResNet100@Glint360K，407MB
    * buffalo_l LFW 准确率 99.83%
    * pip install -U insightface，FaceAnalysis 默认用 buffalo_l，ONNX Runtime 可纯 CPU 跑
    * 致命约束：Model Zoo 明确写「ALL models are available for non-commercial research purposes only」；开源人脸识别模型商用需联系 recognition-oss-pack@insightface.ai 授权
- 长片作用: 顶「身份漂移」工位，技术上是最优解，但商用许可是硬伤。产线做法：给每个虚构角色建一个 anchor embedding 库（从通过审核的关键帧里取 5-10 个），每个镜头抽 3-5 帧算 embedding，与 anchor 的余弦相似度做两件事——(1) 帧内 min 值 < 阈值 = 该镜头身份崩了；(2) 跨镜头 anchor 相似度方差过大 = 全片角色不统一。注意你的角色是虚构生成的，不是真人，ArcFace 在 AIGC 人脸上的判别阈值必须用自己的数据重新标定，不能直接套真人人脸验证的经验阈值。商用必须换 facenet-pytorch 或去拿授权。
- 来源: https://github.com/deepinsight/insightface https://github.com/deepinsight/insightface/tree/master/model_zoo

## facenet-pytorch（InsightFace 的 MIT 替代路线）  (Tim Esler)
- 类别/成熟度: open-weights / **production** | 许可: MIT | 成本: <1GB VRAM；CPU 上 512×512 人脸 crop 约 50-150ms/帧
- 是什么: MIT 许可的人脸 embedding 方案：MTCNN 检测 + InceptionResnetV1 识别，是商用产线里 ArcFace 的合法替身。
- 关键事实:
    * 两个预训练权重：VGGFace2（LFW 0.9965，107MB）、CASIA-Webface（LFW 0.9905，111MB）
    * embedding 维度 512
    * License: MIT（可商用）
    * MTCNN 检测吞吐（GPU）：1080×1920 12.97 FPS、720×1280 20.32 FPS、540×960 25.50 FPS
    * 官方明确不给固定的余弦/L2 阈值，建议自己用聚类或距离度量标定
    * 文档未明确 CPU 性能，但模型 107MB 量级，CPU 可行
- 长片作用: 如果这个平台要商业化，身份漂移检测就该建在这条线上而不是 InsightFace。精度差距（99.65% vs 99.83% LFW）在你的用途里完全不重要——你不是在做百万人底库的 1:N 识别，是在做「同一个虚构角色跨 200 个镜头有没有变脸」的 1:1 判别，两者都远超需求。关键是自己标阈值：拿 100 个已知同角色镜头对 + 100 个不同角色镜头对，画 ROC 找工作点。
- 来源: https://github.com/timesler/facenet-pytorch

## SyncNet + LSE-C / LSE-D（口型同步指标，含真实视频基线数字）  (Joon Son Chung / Oxford VGG；Wav2Lip (IIIT-H))
- 类别/成熟度: technique / **production** | 许可: MIT（syncnet_python） | 成本: 提供官方 CPU 环境；CPU 上 5 秒 clip 约 3-10s（含人脸检测），GPU <2GB
- 是什么: 音视频同步判别网络，以及由它导出的两个事实标准口型同步指标。LSE-D = 唇部与音频表征的平均距离（越低越同步），LSE-C = 平均置信度（越高越同步）。
- 关键事实:
    * SyncNet 原始论文 Chung & Zisserman 2016；syncnet_python 仓库 MIT License
    * 提供 environment.yml（GPU）与 environment-cpu.yml（纯 CPU），自动检测 CUDA 否则回落 CPU
    * Wav2Lip 论文（arXiv 2008.10010）Table 1 的真实视频基线：LRW LSE-D 7.012 / LSE-C 6.931；LRS2 LSE-D 6.736 / LSE-C 7.838；LRS3 LSE-D 6.956 / LSE-C 7.592
    * Wav2Lip+GAN 输出：LRW 6.774 / 7.263；LRS2 6.469 / 7.781；LRS3 6.986 / 7.574
    * 注意：Wav2Lip 的 LSE-D 比真实视频还低（6.469 < 6.736），说明该指标可被过拟合，不能单独作为唯一验收标准
    * 原文定义：「A lower LSE-D denotes a higher audio-visual match」「Higher the confidence, the better the audio-video correlation」
- 长片作用: 顶「对白镜头口型验收」工位，对数字人拍片是不可跳过的一关。有了 Wav2Lip Table 1 这组数字，阈值可以定得有据可依：真实视频落在 LSE-D 6.7-7.0 / LSE-C 6.9-7.8 区间，所以产线的硬性 fail 线设 LSE-D > 8.5 或 LSE-C < 5.0 是合理的（明显差于真人），及格线 LSE-D <= 7.5 且 LSE-C >= 6.0。但必须配一条反向校验——因为 Wav2Lip 能刷到比真人还好的 LSE-D，说明高分不等于好，只能用它做「筛掉明显不同步的」，不能用它排序优选。Seedance 自带对白声这点意味着音画本来就是联合生成的，LSE 主要用来抓偶发性的崩口型镜头。
- 来源: https://github.com/joonson/syncnet_python https://arxiv.org/abs/2008.10010 https://ar5iv.labs.arxiv.org/html/2008.10010

## LatentSync / StableSyncNet（SyncNet 的 2025 改进与对 LSE 的批评）  (字节跳动)
- 类别/成熟度: open-weights / **usable** | 许可: 未在检索中确认（字节开源项目通常为 Apache-2.0，需核实） | 成本: 未披露
- 是什么: 音频条件潜空间扩散口型同步模型，附带重新设计的 StableSyncNet 架构，并公开批评了此前 SyncNet 类方法的 shortcut learning 问题。
- 关键事实:
    * arXiv 2412.09262v2，v2 于 2025-03-13
    * StableSyncNet 在 HDTF 测试集上达到 94% 准确率
    * 论文明确指出先前方法存在 shortcut learning（学到了捷径而非真正的音画对应）
    * 相关线索：Lip Forcing（arXiv 2606.11180，2026-06-09）用 SyncNet-based reward 做实时口型同步；EAD-Net（arXiv 2604.23325，2026-04-25）用 SyncNet supervision 做情感感知说话头
    * AVSyncNet（arXiv 2307.09368v3）提出稳定化同步损失，解决 SyncNet 训练不稳定
- 长片作用: 两层价值：(1) 如果 Seedance 输出的口型不够好，LatentSync 可以作为后处理修补工位，对已通过其他质检的镜头单独修口型，比整镜重生成便宜得多；(2) 更重要的是它对 shortcut learning 的批评直接告诉你：不要把 LSE-C/LSE-D 当唯一口型 KPI，更不要拿它去优化。StableSyncNet 的判别器比原版 SyncNet 更可信，做验收器优先用它。
- 来源: https://arxiv.org/abs/2412.09262

## MediaPipe Hand Landmarker（手指/肢体结构崩坏的轻量探针）  (Google)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: CPU 17ms/帧（移动 SoC），x86 服务器 CPU 更快；零 GPU
- 是什么: 21 点手部关键点检测，两阶段（手掌检测 + 关键点回归），移动端优化。检测「手指崩坏」最便宜的可落地手段。
- 关键事实:
    * 21 个手部关节坐标；输出左右手分类（handedness）、图像坐标、世界坐标
    * HandLandmarker (full) 输入 192×192 或 224×224，float16 量化
    * 延迟：Pixel 6 上 CPU 17.12ms、GPU 12.27ms
    * 训练数据：约 30,000 张真实图 + 合成数据
    * 关键参数：num_hands 默认 1、min_hand_detection_confidence 默认 0.5、running_mode（IMAGE/VIDEO/LIVE_STREAM）
    * VIDEO 模式会复用上一帧关键点框，跳过昂贵的 palm detection
- 长片作用: 顶「结构崩坏筛查」工位的低成本版。17ms/帧的 CPU 成本意味着你可以对每个镜头抽 10-20 帧全检。但注意它不是崩坏检测器，是关键点检测器——用法是反向的：检测置信度突然掉、检测到的手数在帧间跳变（1→2→0）、21 点的骨长比例超出人体解剖范围、或相邻帧关键点位置跳跃过大，这些异常才是崩坏信号。它抓不到「6 根手指但每根都正常」这种情况——那种要靠 VBench-2.0 的 Human Anatomy 异常检测器或 VLM 判官。肢体层面同理需要配 pose landmarker 或 DWPose。
- 来源: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

## 光流 warping error（E_warp）—— 时序一致性的正确度量  (学界通用（Lai et al. ECCV 2018 起）)
- 类别/成熟度: technique / **production** | 许可: 算法无许可证问题；RAFT 为 BSD-3，OpenCV DIS 为 Apache-2.0 | 成本: RAFT-small GPU <2GB，约 20-50ms/帧对；OpenCV DIS 光流纯 CPU 720p 约 10-30ms/帧对，完全可批量
- 是什么: 用光流把第 t+1 帧 warp 回第 t 帧，再算两者的遮挡掩码加权误差。这是把「真闪烁」和「真运动」区分开的唯一正确方法，比裸帧差（VBench temporal_flickering）严谨得多。
- 关键事实:
    * 定义：E_warp = 遮挡掩码加权的 warp 后帧与原帧之间的误差，逐相邻帧计算后平均
    * ChordVideo（arXiv 2608.00769，2026-08-01）推导出「warping-error bound that separates motion bias from stochastic flicker」——明确把运动偏差与随机闪烁分离，并报告 warping error 降低 78%
    * Consistent Video Editing as Flow-Driven I2V（arXiv 2506.07713v2，2025-06-13）报告 warping error 改善 50.66%
    * Flow-Guided Diffusion for Video Inpainting（arXiv 2311.15368v2）以 E_warp 为主要时序一致性指标，报告改善 10%
    * Physics-Guided Motion Loss（arXiv 2506.02244v2）报告 warping error 降低 22-37%
    * EvalCrafter 用 RAFT 做光流；CPU 侧可用 OpenCV DIS 光流（DISOpticalFlow）或 Farnebäck 作为廉价替代
- 长片作用: 这是 L0/L1 层最该自己实现的一个指标，也是唯一同时覆盖「时序闪烁」和「光流异常」两个需求的指标。用 OpenCV DIS 光流（纯 CPU、Apache-2.0）算三个数就够：(1) E_warp 均值 = 时序一致性；(2) 光流幅值的时间序列方差 = 运动突变/跳帧；(3) 光流场的散度异常 = 结构撕裂/融化。这三个数零模型依赖、零许可证风险、CPU 可跑，是批量筛废片性价比最高的一组。ChordVideo 那条「把 motion bias 和 stochastic flicker 分离」的界线尤其关键——不做这个分离，推镜头和摇镜头会被系统性误杀。
- 来源: https://arxiv.org/abs/2608.00769 https://arxiv.org/abs/2506.07713 https://github.com/evalcrafter/EvalCrafter

## VideoReward / VideoAlign（Flow-DPO 配套奖励模型）  (快手 KwaiVGI)
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 2B 模型，bf16 约 5GB VRAM，单张 8GB 消费卡可跑；INT8 后约 3GB
- 是什么: 三维视频奖励模型，MIT 许可、仅 2B 参数，是开源奖励模型里性价比最高的一个，专门设计用来做数据过滤、guidance、reject sampling 和 DPO。
- 关键事实:
    * 基座 QWen2-VL-2B-Instruct（2B）
    * HF: KwaiVGI/VideoReward，License MIT
    * 2025-02-08 发布；NeurIPS 2025
    * 三维：Visual Quality（清晰度/美学/单帧合理性）、Motion Quality（动态稳定性/动态合理性/自然度/动态程度）、Text Alignment（文视相关性）
    * 官方定位：支持 data filtering、guidance、reject sampling、DPO 及其他 RL 方法
    * 配套 VideoGen-RewardBench 评测集
    * 配套论文 Improving Video Generation with Human Feedback（arXiv 2501.13918v2）提出 Flow-DPO 与 Flow-RWR
- 长片作用: 这是清单里唯一同时满足「小显存 + MIT 商用 + 三维打分 + 明确为 DPO 设计」的模型，应该作为 L1 层的默认打分器。2B/5GB 的体量意味着它可以和 DINO、光流一起塞进同一张 12GB 卡上跑批。它的 Motion Quality 维度拆分（动态稳定性 vs 动态程度）对长片尤其重要——能区分「稳但死板」和「动但抖」，这两类废片的处置完全不同。reject sampling 用法最直接：同一个分镜 prompt 用 Seedance 生成 4 条，用它排序取 Top-1，几乎不增加人工成本就能显著提升成片率。
- 来源: https://github.com/KwaiVGI/VideoAlign https://huggingface.co/KwaiVGI/VideoReward https://arxiv.org/abs/2501.13918

## VisionReward  (清华 THUDM)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: CogVLM2-Video 约 12B 级（视频版），bf16 约 25GB+；图像版 19B 更重
- 是什么: 把视频质量拆成 64 个二元判断题（checklist）再线性加权成分数的可解释奖励模型，配套多目标偏好优化 MPO。
- 关键事实:
    * VisionReward-Video 基座 cogvlm2-video-llama3-chat；VisionReward-Image 基座 cogvlm2-llama3-chat-19B
    * 视频侧 64 个维度，具体判断题在 VisionReward_Video/VisionReward_video_qa.txt
    * License: Apache-2.0，HF: THUDM/VisionReward-Video 与 THUDM/VisionReward-Image
    * arXiv 2412.21059（2024），AAAI 2026 接收
    * 视频偏好测试：64.0 Tau / 72.1 Diff，声称超过 VideoScore 17.2%
    * 支持 MPO 多目标偏好优化，实现「stable and controllable RLHF」
- 长片作用: 最大价值不是它的分数，是那个 64 题的 checklist 文件——它是现成的、经过验证的人工评审 rubric。直接把 VisionReward_video_qa.txt 拿来做你 Label Studio 评审界面的题目模板，省掉自己设计评审维度的几周。MPO 那条线对你意义有限（权重不开放的模型没法优化），但如果产线里有本地开源兜底模型，它是唯一给出多目标（不止两两偏好）优化方案的。显存 25GB+ 让它进不了 L1，只能做离线全量评估。
- 来源: https://github.com/THUDM/VisionReward https://arxiv.org/abs/2412.21059

## Label Studio（人工评审工作流主力）  (HumanSignal (原 Heartex))
- 类别/成熟度: framework / **production** | 许可: Apache-2.0（Community 版） | 成本: 自托管，CPU 即可；单机 docker 起步
- 是什么: Apache-2.0 的开源标注平台，原生支持视频时间轴标注、帧级标注、视频分类与转写。是把人工打分结构化回流的最省事选择。
- 关键事实:
    * License: Apache 2.0（© Heartex 2020-2025）
    * 视频格式硬要求：MP4 容器 + H.264(AVC) 视频 + AAC 音频 + 恒定帧率（理想 30fps），音视频轨时长必须一致，否则总时长检测会出错
    * Video 标签参数：frameRate 默认 24、height 默认 600px、timelineHeight 默认 64px、defaultPlaybackSpeed 1、muted false
    * 支持：视频分类、转写、帧级标注、时间轴区域分割
    * 多用户标注：登录后每条标注绑定账号
    * ML Backend SDK 可接模型做预标注、在线学习、主动学习；有 webhook
    * 标注导出有标准格式，label-studio-converter 可转常见 ML 库格式
    * 高级项目分析、质量控制 UI、内置自动标注属于付费 Cloud/Enterprise
- 长片作用: 顶「人工评审」工位。三个落地要点：(1) 必须先用 ffmpeg 统一转成 MP4/H.264/AAC/CFR 30fps，否则时间轴会错位，评审员标的帧号对不上你的镜头；(2) frameRate 默认是 24，要显式设成你实际的帧率，不然帧号全错；(3) 最关键——用 ML Backend 把 L2 层 VideoScore2 的分数和 CoT 理由作为预标注推进去，评审员从「从零打分」变成「确认或修正 AI 的判断」，这是把人工吞吐提上去的唯一办法。OSS 版没有质量控制 UI（标注员一致性统计），需要自己用 webhook 导出后算 Krippendorff alpha。
- 来源: https://labelstud.io/tags/video https://github.com/HumanSignal/label-studio

## CVAT  (CVAT.ai)
- 类别/成熟度: framework / **production** | 许可: MIT（Community） | 成本: 自托管 CPU 即可
- 是什么: MIT 许可的视频/图像/3D 标注平台，工具更偏几何标注（框、多边形、掩码、关键点、cuboid、tag）。
- 关键事实:
    * CVAT Community 为 MIT License；/serverless 下代码也是 MIT，但可能使用第三方非商用资产
    * 支持 image、video、3D 标注；工具含 bounding boxes、polygons、masks、keypoints、cuboids、tags
    * 高级项目分析、质量控制 UI、内置自动标注仅在付费 CVAT Online / Enterprise 提供
    * 自托管依赖 Docker Engine + Docker Compose + Git，docker compose up -d 起步
    * 官网页面未显示当前版本号与发布日期
- 长片作用: 对你这个场景不如 Label Studio。CVAT 强在几何标注（画框画掩码），而你的人工评审主要是「打分 + 选理由」，是分类/评分任务，CVAT 的 tag 功能能做但界面不顺手。唯一适合用 CVAT 的工位是「崩坏区域标注」——要训练自己的崩坏检测器时，需要人工框出手部/面部崩坏的具体区域，这时 CVAT 的掩码工具比 Label Studio 好用。两者都装、按任务分流是合理的。
- 来源: https://github.com/cvat-ai/cvat

## 视频 DPO 技术路线（2025-2026 主要方法）  (多家)
- 类别/成熟度: technique / **research** | 许可: 各异 | 成本: DPO 训练视频扩散模型是 8×A100 起步量级
- 是什么: 把评分/偏好回流成训练信号的一批方法，核心分歧在于「偏好对怎么造」和「DPO 在扩散模型上为何不稳」。
- 关键事实:
    * DenseDPO（arXiv 2506.03517v2，2025-06-04）：用「对同一视频做损坏副本」造对齐的视频对，做 segment-level DPO，并用 VLM 自动标注——解决了传统 DPO 视频对之间内容不同导致信号被内容差异污染的问题
    * Reg-DPO（arXiv 2511.01450v3，2025-11-03）：把 SFT 当正则项加进 DPO，配自动构造的 GT-Pair，提升训练稳定性
    * RealDPO（arXiv 2510.14955v2，2025-10-16）：直接用真实世界视频当正样本，实现迭代自我纠错
    * PG-DPO（arXiv 2511.19049，2025-11-24）：针对扩散模型 likelihood displacement 问题，用自适应拒绝缩放 + 隐式偏好正则
    * LocalDPO（arXiv 2601.04068v4，2026-01-07）：region-aware DPO loss，只在损坏的时空区域上学习
    * Step-Video-T2V（arXiv 2502.10248v3）：30B 模型上实践 Video-DPO 降 artifact
    * UnifiedReward（arXiv 2503.05236v2）：支持成对排序，为 DPO 自动构造偏好对
    * cIPO（arXiv 2607.28058，2026-07-30）：从重建误差导出隐式偏好，不需要显式偏好对
- 长片作用: 必须把话说死：Seedance 2.0 权重不开放，人工评分回流成 DPO 训练对这件事对 Seedance 本身完全无效——没有任何办法把偏好数据灌进闭源 API。评分回流在你的产线里只有三个真实出口：(1) 作为 reject sampling 的排序器，同 prompt 多采样取最优（这是投入产出比最高的，立刻可用）；(2) 作为 prompt/参数自动搜索的 reward，用评分反推哪类 prompt 写法、哪套参考图组合成功率高；(3) 如果产线有本地开源兜底模型（Wan / HunyuanVideo 系），才能真正跑 DPO。DenseDPO 的「同源损坏副本造对」方法在你这里特别适用——同一个镜头的不同采样天然就是内容对齐的视频对，人工只需判断哪条更好，不用跨内容比较。
- 来源: https://arxiv.org/abs/2506.03517 https://arxiv.org/abs/2511.01450 https://arxiv.org/abs/2510.14955

## Seedance 2.0（核实结果：部分可证，部分未能核实）  (字节跳动 Seed)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: 按 API 计费，具体单价未能核实
- 是什么: 字节 Seed 的旗舰多模态视频生成模型，官网定位「director-level control」。
- 关键事实:
    * seed.bytedance.com/en/seedance2_0 页面确认：支持 text、image、audio、video 四类输入，明确「support images, audios and videos as references」
    * 确认支持 audio-video joint generation（音视频联合生成），即自带对白声属实
    * 页面提供 Try Now / Get API / Compare Now 入口，指向 API 形式提供
    * Seedance 1.0：原生支持多镜头生成，1080p，「natively supports the generation of narrative videos with multiple cohesive shots」
    * seed.bytedance.com 导航中出现 Seedance 2.5
    * 权重开放与否、最大时长、9图+3视频+3音频 的具体上限数字，均未能从官网或火山引擎文档页核实（docs.volcengine.com/docs/82379/1520757 返回空内容，为 JS 渲染页）
- 长片作用: 核实结论：「多模态参考（图+视频+音频）」和「自带对白声」两条官网可直接佐证。「权重不开放」在官网无反证（ByteDance Seed 的 Seedance 系列从未在 HuggingFace 放权重），可认为成立。「官方单段 4-15 秒 / 2.5 约 30 秒」「9图+3视频+3音频」这两条数字未能从一手文档核实，已移入 uncertainClaims。对质检体系的直接含义：既然是闭源 API 且自带音频，你的质检必须是纯黑盒后验的——所有指标都只能从输出视频/音频反推，没有任何中间态（latent、attention map）可用。这反过来抬高了 L0/L1 那套像素级+光流级指标的价值。
- 来源: https://seed.bytedance.com/en/seedance2_0 https://seed.bytedance.com/seedance

## Qwen3-VL（自建 VLM 判官的底座）  (阿里通义)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: 2B 约 5GB / 4B 约 9GB / 8B 约 17GB（bf16）；FP8 约减半，8B FP8 可进 12GB 卡
- 是什么: Apache-2.0 的多模态大模型系列，覆盖 2B 到 235B，是自建「镜头合格判官」最合适的开源底座。
- 关键事实:
    * 尺寸：2B、4B、8B、32B，以及 MoE 变体 30B-A3B、235B-A22B；均有 Instruct 与 Thinking 版
    * 发布：2025-10-04 起，大模型先发，小模型到 2025-10-21 陆续发布
    * License: Apache-2.0
    * 视频输入支持灵活抽帧，可设 num_frames 或 fps，默认 fps=2；像素预算通过 total_pixels 控制
    * 官方提供 FP8 版本（HuggingFace collection）；Qwen2.5-VL 有 AWQ 量化版（3B/7B/72B）
    * 仓库未给出各尺寸的确切显存需求
- 长片作用: 这是把所有「需要理解语义才能判」的质检项统一收口的方案。VQAScore 的是非问句、StoryEval 的事件完成度判定、PersonaShot 的情绪连贯性、VisionReward 的 64 题 checklist——这些本质上都是「给 VLM 看视频问问题」，用一个 Qwen3-VL-8B-FP8 全包比部署四套模型现实得多。Apache-2.0 + FP8 + 12GB 卡可跑，是唯一在商用许可和硬件成本上都没坑的语义判官方案。默认 fps=2 对 5-15 秒镜头正好是 10-30 帧，不用调。
- 来源: https://github.com/QwenLM/Qwen3-VL

### 建议
## 一、先把结论说死

1. **评分回流做 DPO，对 Seedance 完全无效**。权重不开放 = 没有任何路径把偏好数据灌进去。人工评分在你的产线里只有三个真出口：① reject sampling 排序器（同 prompt 多采样取最优，立刻可用、ROI 最高）；② prompt/参考图组合的自动搜索 reward；③ 如果有本地开源兜底模型才谈得上真 DPO。任何说"打分回流训练 Seedance"的方案都是假的。

2. **质检门禁必须前移到分镜/关键帧阶段**。DramaChain Bench（2026-09，腾讯混元+北电，5785 条目 / 63 叶子维度 / agentic judge 对人工 PLCC 0.918）的核心发现是缺陷沿产业链级联：分镜阶段角色设定写歪，下游重生成多少次都救不回。等视频出来再筛 = 每次废片烧一次 API 钱。

3. **License 是这个领域最大的隐形地雷**，比技术选型更容易翻车。三个必须避开：InsightFace 全系「non-commercial research purposes only」、IQA-PyTorch 是 PolyForm Noncommercial 1.0.0、Q-Align/DOVER 带 S-Lab License。干净可商用的只有：LAION aesthetic（Apache-2.0）、facenet-pytorch（MIT）、VideoReward（MIT）、syncnet_python（MIT）、MediaPipe（Apache-2.0）、Qwen3-VL（Apache-2.0）、VBench（Apache-2.0）、Label Studio（Apache-2.0）、OpenCV DIS 光流（Apache-2.0）。

---

## 二、四层镜头合格门禁（可直接实施）

### L0 — 纯 CPU 批量粗筛，目标砍掉 30-50% 废片
单镜头预算 < 3 秒 CPU，零 GPU，零许可证风险。

| 指标 | 实现 | 建议起点阈值（hard fail） |
|---|---|---|
| 解码健全性 | ffprobe：帧数/时长/音轨一致 | 任一不符 = fail |
| 黑帧/静帧 | 逐帧均值亮度 + 帧差 | 连续 >0.5s 帧差≈0 = fail |
| 运动补偿时序闪烁 E_warp | OpenCV DISOpticalFlow warp + 遮挡掩码加权 MAE | E_warp 归一化分 < 0.90 = fail |
| 色彩漂移 | 逐帧转 CIELab，取 mean(L,a,b) 时间序列；算首尾 ΔE00 与逐帧一阶差分最大值 | 首尾 ΔE00 > 8 或单帧跳变 ΔE00 > 4 = fail |
| 光流异常 | DIS 光流幅值时间序列方差 + 散度异常率 | 幅值方差 z-score > 3 = fail（抓跳帧/撕裂） |
| 美学 | LAION aesthetic v2（CLIP ViT-L/14 + 线性头，Apache-2.0），逐帧算，取 mean 和 **min** | mean < 5.0 或 **min < 4.3** = fail |
| 整体质量 | DOVER-Mobile（9.86M 参数，CPU 1.4s/视频） | 技术分位 < 25% 分位 = fail |

**关键工程点**：裸帧差版 temporal_flickering（VBench 原版公式 `(255-mean(MAE))/255`）会把推镜摇镜系统性误杀，必须先做光流 warp。ChordVideo（arXiv 2608.00769）推导的 warping-error bound 明确把 motion bias 与 stochastic flicker 分离，这条界线不做，L0 直接废掉。
**美学取 min 不取 mean**：长片最怕"大部分帧好看、中间两秒崩了"，均值会掩盖掉。
DOVER-Mobile 的 S-Lab License 需确认商用条款；不通过就只留 LAION 那条。

### L1 — 单卡 8-12GB，语义前的结构化检查
单镜头预算 10-20 秒。所有模型可共驻一张 12GB 卡。

| 工位 | 模型 | 建议起点阈值 |
|---|---|---|
| 镜内主体一致性 | DINO ViT-B/16，VBench 双路公式 `(cos(f_first,f_i)+cos(f_prev,f_i))/2` | 镜内均值 < 0.93 = fail；单帧 min < 0.88 = fail |
| **跨镜身份漂移** | facenet-pytorch InceptionResnetV1 (MIT, 512-d)，每角色建 5-10 个 anchor embedding | 与 anchor 余弦 min < 阈值 = fail；跨全片 anchor 相似度方差超标 = 全片 fail |
| 手部/肢体结构 | MediaPipe Hand Landmarker（21 点，CPU 17ms/帧）+ pose landmarker | 手数帧间跳变、置信度骤降、骨长比超解剖范围、关键点帧间跳跃 = fail |
| 口型同步 | syncnet_python（MIT，官方有 CPU 环境） | **LSE-D > 8.5 或 LSE-C < 5.0 = hard fail；及格线 LSE-D ≤ 7.5 且 LSE-C ≥ 6.0** |
| 三维打分 | VideoReward（Qwen2-VL-2B，MIT，bf16 ~5GB） | VQ/MQ/TA 任一 < 阈值 = fail |

**LSE 阈值是全清单唯一有一手基线数字支撑的**：Wav2Lip 论文 Table 1 实测真实视频 LRW 7.012/6.931、LRS2 6.736/7.838、LRS3 6.956/7.592。所以"明显差于真人"的线就落在 LSE-D>8.5 / LSE-C<5.0。
**但绝不能拿 LSE 做排序优选**：Wav2Lip+GAN 在 LRS2 刷到 LSE-D 6.469，比真人 6.736 还"好"，LatentSync（arXiv 2412.09262）明确指出这是 shortcut learning。LSE 只配做筛废，不配做排名。
**ArcFace 阈值不能套真人经验值**：你的角色是虚构生成的，必须拿 100 对同角色 + 100 对异角色镜头画 ROC 自己标工作点。

### L2 — 24GB 卡或 API，只对 L1 通过的跑
单镜头预算 30-120 秒。

- **VideoScore2**（Qwen2.5-VL-7B 基座，8B，2025-09-26，VideoScore-Bench-v2 44.35%）：三维打分 + **chain-of-thought 理由**。这个理由是产线金矿——不是给一个糊涂分，而是告诉你"第 3 秒出现 6 指"，可直接转成重生成的负向约束，也可作为人工评审界面的预填意见。
- **VQAScore / t2v_metrics**（Apache-2.0）：把每条分镜描述拆成 3-5 个是非问句逐个算 Yes 概率，得到可解释的对齐向量。字节 Seed 自己在用这套，与 Seedance 训练目标可能同源。
- **叙事层（StoryEval 协议）**：不用它的 423 条 prompt，只用协议——分镜表里本来就写了每镜要演什么事件，直接问判官"X 有没有完成 Y？"。参考基线：11 个模型平均故事完成率**无一超过 50%**，这是你的现实预期锚点。

**统一收口建议**：L2 的所有语义判定（VQAScore 问句、StoryEval 事件、PersonaShot 情绪连贯、VisionReward 64 题）本质都是"给 VLM 看视频问问题"。部署一个 **Qwen3-VL-8B-FP8**（Apache-2.0，FP8 约 12GB 卡可跑，默认 fps=2 对 5-15 秒镜头正好 10-30 帧）全包，比部署四套模型现实得多。

### L3 — 人工评审，只看分数边界带
Label Studio（Apache-2.0）。三个必做工程项：
1. ffmpeg 统一转 **MP4/H.264/AAC/CFR 30fps**，音视频轨时长必须一致——否则时间轴错位，评审员标的帧号对不上你的镜头。
2. `frameRate` 显式设成实际帧率（**默认是 24，不改帧号全错**）。
3. **用 ML Backend 把 VideoScore2 的分数和 CoT 理由推成预标注**，评审员从"从零打分"变成"确认或修正 AI 判断"，这是把人工吞吐提 3-5 倍的唯一办法。

评审 rubric 不要自己发明：直接用 VisionReward 的 `VisionReward_Video/VisionReward_video_qa.txt`（64 道二元判断题，Apache-2.0，已验证），省掉几周设计时间。
OSS 版无质量控制 UI，标注员一致性（Krippendorff α）需自己 webhook 导出后算。
崩坏区域的掩码标注用 CVAT（MIT）更顺手，按任务分流。

---

## 三、门禁判定公式

```
hard_fail = OR(任一 L0/L1 硬指标越线)          → 直接重生成，不进人工
soft_score = Σ w_i · normalize(metric_i)       → 排序用
gate_pass  = NOT hard_fail AND soft_score ≥ τ
```
- τ 用人工标注的 200 条镜头做校准，目标：**人工判"可用"的召回 ≥ 0.95**（宁可放过，不可错杀——错杀的代价是白烧一次 API）。
- 同 prompt 生成 N=4 条，全部过 L0/L1，用 VideoReward + VideoScore2 排序取 Top-1。这是不增加人工成本就能显著提升成片率的最直接手段。
- 所有阈值都是工程起点，**必须用自己的 100-200 条样本重新标定**。跨模型、跨分辨率、跨题材阈值都不通用。

---

## 四、成片级（2-8 分钟）门禁，单镜通过不等于成片能看

单镜全部满分、拼起来是流水账，是这条产线最可能踩的坑。必须额外加三道：

1. **跨镜身份漂移**：抄 VBench-Long（`vbench2_beta_long`）的"快分支"——低帧率抽样跨全片评前景主体与背景一致性。但用你的分镜表代替 PySceneDetect（你本来就知道镜头边界）。
2. **跨剪辑点连贯**：PersonaShot（arXiv 2608.16717，~1000 多镜片段 / 16 指标）的三个时间层级——within-shot states、cross-shot transitions、sequence-level trajectories。它的 **affective dynamics** 是其他所有基准都没有的维度：上一镜在哭下一镜在笑，DINO 和 ArcFace 都查不出来，只有情绪连贯性能抓。即使代码未开源，这 16 个指标定义本身就值得抄成 rubric。
3. **剪辑语法与节奏**：CutCraft（arXiv 2609.08275）的 shot-structure alignment、CineScope（arXiv 2608.29621）的 pacing。这两个是 2026-08/09 的新东西，不要押注做依赖，当 rubric 灵感来源。

---

## 五、不要做的事

- 不要整套跑 VBench 16 维或 VBench-2.0 18 维。后者官方建议 18 维分 18 张卡（上限 8 卡），单卡串行官方自己说 "not recommended"。产线只需要 4+3 个维度。
- 不要把 IQA-PyTorch 打进生产镜像（PolyForm Noncommercial）。开发期用它快速对比哪个指标与你的人工打分相关性最高，选定后用 Apache-2.0 原始实现重写。BRISQUE/NIQE 是公开算法，OpenCV contrib 有实现，可单独绕开。
- 不要依赖 MovieBench 的数据（官网仍写"plan to release within the next three months"，且 CC BY-SA 4.0 的 ShareAlike 会污染商用数据集）。但它的"角色卡 = 姓名+参考图+参考音频"schema 正好对齐 Seedance 的多模态参考位，直接抄这个数据结构。
- 不要用 EvalCrafter 框架本身（prompt 编号硬耦合 0000.mp4-0699.mp4、仓库未声明自身 License）。只摘它的 RAFT 光流 + deepface 一致性实现做参考。
- 不要单用 DOVER：它训练于 UGC 真实视频（KoNViD/LIVE-VQC/LSVQ），对 AIGC 特有崩坏（多指、面部扭曲）的敏感度未经验证，必须和 VBench-2.0 Human Anatomy 那条线配合。

---

## 六、落地优先级

| 优先级 | 动作 | 理由 |
|---|---|---|
| P0 | L0 全套（DIS 光流 E_warp + Lab 色彩漂移 + LAION 美学 min/mean） | 纯 CPU、零许可证风险、立刻砍 30-50% API 浪费 |
| P0 | reject sampling：N=4 采样 + VideoReward 排序 | MIT、5GB 显存、不增加人工成本直接提成片率 |
| P1 | facenet-pytorch 跨镜身份 anchor 库 + ROC 标阈值 | 分镜拼接产线的核心失败模式 |
| P1 | syncnet_python LSE 门禁（阈值已有一手基线） | 数字人对白戏不可跳过 |
| P1 | Label Studio + VisionReward 64 题 rubric + VideoScore2 预标注 | 人工吞吐的瓶颈解法 |
| P2 | Qwen3-VL-8B-FP8 统一语义判官（VQAScore 问句 + StoryEval 事件） | 收口 L2，一个模型替四套 |
| P2 | VBench-2.0 Human Anomaly Detection 蒸馏成轻量 ViT 头 | 唯一有公开训练数据和代码的 AIGC 人体崩坏检测器 |
| P3 | 成片级：VBench-Long 快分支 + PersonaShot rubric | 单镜合格 ≠ 成片能看 |

**检索方式说明**：本轮 WebSearch 配额已在会话早期耗尽（200/200），全部结果通过 WebFetch 直取 GitHub / arXiv abs / arXiv API / 官方文档获得，并逐条交叉核对。因此可能遗漏仅见于新闻稿或博客、无 arXiv/GitHub 落点的 2026 年最新条目。

### 存疑
- Seedance 2.0 官方单段时长上限 4-15 秒、Seedance 2.5 约 30 秒：未能从一手文档核实。火山引擎文档页 docs.volcengine.com/docs/82379/1520757 为 JS 渲染，WebFetch 返回空内容；seed.bytedance.com 的 Seedance 2.0 页面不披露时长。
- Seedance 多模态参考上限「最多 9 图 + 3 视频 + 3 音频」：官网仅确认支持 images/audios/videos 作为参考，具体数量上限未能核实。
- Seedance 2.0 权重不开放：官网无明确声明，但 ByteDance Seed 的 Seedance 系列从未在 HuggingFace 发布权重，推断成立，非直接证据。
- Seedance 官方通道有内容审核：未能从检索到的官方页面直接佐证（火山引擎文档未取到）。
- 本文所有阈值（E_warp<0.90、ΔE00>8、LAION mean<5.0/min<4.3、DINO<0.93/0.88、LSE-D>8.5/LSE-C<5.0）均为基于指标定义与已知基线推导的工程起点值，不是任何论文或产线实测的公开经验值。必须用自有 100-200 条样本重新标定，跨模型/分辨率/题材不通用。
- VideoScore2 的确切推理显存：GitHub 与 HF 页面均未披露，文中 ~18GB（bf16）是按 8B 参数量推算，非实测。
- VideoScore2 许可证：HuggingFace 模型页标 Apache-2.0，GitHub 仓库标 MIT，两处不一致，商用前需向作者确认。
- VideoReward、VisionReward、LoCoT2V-Bench 的确切显存需求均未在官方仓库披露，文中数字为按参数量推算。
- DOVER / DOVER-Mobile 的 S-Lab License 具体商用条款未逐条核实，仅知仓库含 S-Lab-LICENSE 文件。S-Lab 系许可证通常限非商用。
- Q-Align 的推理（非训练）显存未披露；官方只给了训练所需的 4×A6000 / 2×A100。
- EvalCrafter 仓库自身许可证未声明，只声明依赖多个开源项目，商用有法律风险。
- T2V-CompBench 的许可证、以及官网提到的 V2 相对 V1 的具体变更，均未在官网披露。
- LoCoT2V-Bench 的 prompt 条数、视频时长、HERD 指标的确切计算公式，arXiv 摘要页未给出。
- MovieBench 的确切数据规模（电影数、总时长、场景/镜头数、角色数）以及数据是否已实际开放，论文页与项目官网均未给出；官网仍写「计划三个月内释出」。
- PersonaShot（2608.16717）、CineForge/CineScope（2608.29621）、DramaChain Bench（2609.00646）、VGA-BenchV2（2608.25452）、CutCraft（2609.08275）、StreamAV-Bench（2608.26336）、VWG-Bench（2609.11242）的代码/数据是否开源、许可证、可复现性均未验证。这批均为 2026 年 8-9 月论文，maturity 一律标 research。
- PersonaShot 的作者机构未从摘要页确认。
- LatentSync / StableSyncNet 的许可证未在本轮检索中确认。
- LSE-C / LSE-D 在 AIGC 虚构角色人脸（非真人、非 LRS2/LRS3 域）上的判别可靠性未经任何公开验证。Wav2Lip 的基线数字来自真人数据集，直接外推到 AIGC 数字人存在域偏移风险。
- DOVER 与 LAION aesthetic 在 AIGC 视频上与人工打分的相关性未见公开报告。两者训练域分别是 UGC 真实视频与真实照片/插画，对 AIGC 特有崩坏（多指、面部扭曲、材质融化）的敏感度未知。
- VBench-2.0 各维度与外部模型的逐一对应关系仓库未系统披露，文中只能列出它推荐预下载的模型集合（LLaVA-Video-7B-Qwen2、Qwen2.5-7B-Instruct、CLIP、CoTracker、YOLO-World、InsightFace）。
- VBench-2.0 的 prompt suite 条数、期望的视频分辨率/时长/帧数规格，仓库页面未说明。
- InsightFace 商用授权的实际价格与可获得性未核实（需联系 recognition-oss-pack@insightface.ai）。
- facenet-pytorch 的 CPU 推理实测速度官方未给出；文中 FPS 数字是 MTCNN 检测部分的 GPU 数据，不是 CPU、也不含 embedding 计算。
- CVAT 当前版本号与发布日期未能从仓库页面取得。
- Label Studio 当前版本号与发布日期未能从仓库页面取得；OSS 版与 Enterprise 版在多标注员评审工作流上的确切功能分界未逐条核实。
- MediaPipe Hand Landmarker 的 17.12ms CPU 延迟是 Pixel 6 移动 SoC 数据，x86 服务器 CPU 的实际吞吐未验证。
- 本轮 WebSearch 配额已耗尽（200/200），全部结果经 WebFetch 直取 GitHub / arXiv / 官方文档获得。可能遗漏仅见于新闻稿、博客或无 arXiv/GitHub 落点的 2026 年最新条目。

### 事实核查修正
- [WRONG] [VBench (v1)] 2024-02 发布，CVPR 2024 Highlight
  → 2024-02 是「被 CVPR 2024 接收为 Highlight」的时间，不是发布时间。官方 README Updates 原文为 [02/2024] VBench accepted to CVPR 2024 as Highlight。实际发布链路：[11/2023] Prompt Suites 释出、arXiv 2311.17982 于 2023-11-29 提交、[12/2023] 16 维评估代码释出、[01/2024] PyPI 包 v0.1.0（2024-01-14）。把 02/2024 写成「发布」会误导时间线。CVPR 2024 Hi https://github.com/Vchitect/VBench/blob/master/README.md
- [WRONG] [VideoScore / VideoScore-v1.1] HF: TIGER-Lab/VideoScore，License MIT
  → 两个模型仓库许可证不同，声明把它们混为一谈。HF API 实测：TIGER-Lab/VideoScore 的 license 是 apache-2.0（createdAt 2024-06-19，lastModified 2025-01-08）；只有 TIGER-Lab/VideoScore-v1.1 的 YAML frontmatter 是 license: mit（createdAt 2024-11-28）。GitHub TIGER-AI-Lab/VideoScore 的 LICENSE 文件是 MIT（Copyright (c) 2024 TIGER https://huggingface.co/api/models/TIGER-Lab/VideoScore
- [WRONG] [VBench++ (TPAMI 版)] 2025-11 作为 TPAMI 期刊论文发布
  → 2025-11 是「被 TPAMI 接收」，不是期刊发表。README 原文：[11/2025] VBench++ accepted to TPAMI。VBench++ 本身早在 2024-11-20 就以 arXiv 2411.13503 预印本形式公开（comments 注明与 2311.17982 大量文本重叠）。截至 2026-09 未检索到正式的 TPAMI 卷期/页码/DOI，引用时应写 arXiv 或 accepted to TPAMI 2025。 https://arxiv.org/abs/2411.13503
- [WRONG] [VBench-2.0] 依赖模型链：LLaVA-Video-7B-Qwen2、Qwen2.5-7B-Instruct、CLIP、CoTracker、YOLO-World、InsightFace
  → 按 VBench-2.0/vbench2/utils.py 源码实测，列表既有错项也有漏项。实际引用的权重：LLaVA-Video-7B-Qwen2（CACHE_DIR/lmms-lab/LLaVA-Video-7B-Qwen2）、Qwen2.5-7B-Instruct（CACHE_DIR/Qwen/Qwen2.5-7B-Instruct）、CoTracker2（torch.hub facebookresearch/co-tracker）、RAFT（models.zip dropbox 下载，光流，声明里漏了）、YOLO-World v2-XL（yolo https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/utils.py
- [UNVERIFIABLE] [VBench-2.0] 官方建议 18 维度分 18 张卡跑（上限 8 卡），单卡串行「not recommended」
  → 能从 VBench-2.0 README 证实的只有两点：支持最多 8 张 GPU 并行（一维度一卡），以及单卡串行跑全部 18 维「not recommended」。「官方建议分 18 张卡」未在 README 中找到对应表述，且与「上限 8 卡」自相矛盾。请按「最多 8 卡并行」写。 https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md
- [UNVERIFIABLE] [VBench temporal_flickering] 零 GPU、零模型权重，1080p 5秒片段 CPU 上 < 1s
  → 「零模型权重、纯 numpy+cv2」CONFIRMED（temporal_flickering.py 只有 cv2.absdiff + np.mean，无任何权重加载）。但「1080p 5 秒 CPU < 1s」无任何一手来源，官方仓库未给性能数据；且该实现内部仍走 VBench 的分布式/device 框架。该数字应标为自测待定，不要当成官方指标。另注：真正的耗时瓶颈通常是解码而非 MAE 计算。 https://github.com/Vchitect/VBench/blob/master/vbench/temporal_flickering.py
- [UNVERIFIABLE] [VBench subject_consistency] 主流模型在 VBench 上该项落在 0.90-0.98 区间
  → 未能从一手来源核实。HuggingFace Vchitect/VBench_Leaderboard 是 Gradio 动态 Space，WebFetch 只返回页面框架（Like 362 / Running），拿不到榜单数值表；本轮 WebSearch 配额已耗尽无法交叉验证。若要把 0.93/0.88 之类阈值挂在这个区间上，必须自己从 leaderboard 的 CSV/JSON 后端拉数再引用。 https://huggingface.co/spaces/Vchitect/VBench_Leaderboard

### 遗漏补充
- pyiqa（chaofengc/IQA-PyTorch）：统一封装 MUSIQ / CLIP-IQA / MANIQA / TOPIQ / NIQE / BRISQUE / LAION-Aesthetic 等几十个 IQA 指标，一行切换。做废片率产线的第一优先级工具，比自己拼 DOVER+LAION 省事得多，且能一次性给出多指标交叉投票，显著降低单指标误杀率。
- VQAScore（CLIP-FlanT5 / GenAI-Bench 系）与 TIFA / DSG：用 VQA 方式做「文生视频是否照着 prompt 生成」的细粒度判定，比 CLIPScore 对组合性 prompt（多主体、属性绑定、否定）敏感得多。本条调研线在 text-to-video alignment 上只覆盖了 VideoScore 系的黑盒打分，缺了可解释的问答式对齐检查。
- TransNetV2（镜头边界检测）：AI 生成长视频最典型的废片形态之一是「模型自己插了非预期的硬切/跳变」。PySceneDetect 只做像素阈值，对缓慢溶解和 AIGC 式突变漏检严重；TransNetV2 是学术界标准的 shot boundary detector，应作为 flicker/连贯性之外的独立闸门。
- FFmpeg 原生质检滤镜链（blackdetect / freezedetect / blackframe / signalstats / silencedetect / astats）：零模型、零 GPU、毫秒级，能在任何深度学习指标之前先把全黑帧、冻帧、纯静音、爆音、色域越界等硬废片直接筛掉。工业 QC（Baton / QCTools / Netflix Photon）走的都是这条路，成本收益比远高于跑 VBench-2.0。
- FVD 的替代指标 JEDi（JEPA Embedding Distance）与 FVMD（Fréchet Video Motion Distance）：FVD 已被多篇工作证明对 I3D 特征的纹理偏置敏感、样本效率低、与人类判断相关性弱。若产线还在用 FVD 做模型选型会得到误导性结论，JEDi/FVMD 是 2024-2025 的公认改进方向。
- VideoPhy / VideoPhy-2 与 PhyGenBench：专门评「物理常识违反」（物体穿模、液体反重力、刚体形变）。这是 AI 短片最刺眼的废片类型之一，VBench-2.0 的 Physics 大类覆盖但粒度粗，VideoPhy 系提供了独立的、可单跑的判别器。
- ChronoMagic-Bench / TC-Bench：评「时序上的状态变化是否真实发生」（metamorphic amplitude、属性随时间的正确转变）。对于需要叙事推进的 AI 短片，这比单纯的 temporal_flickering/subject_consistency 更接近「这条片子能不能用」的判据。
- 人评基础设施与统计方法：Bradley-Terry / Elo 成对比较（而非绝对打分）、Krippendorff's alpha 衡量标注员一致性、bootstrap 置信区间。调研线列了 CVAT / Label Studio 这类标注工具，但没有给出把多标注员分数聚合成可决策的废片率指标的统计方法，而阈值标定恰恰依赖这一层。
- MEt3R 及基于 VGGT/DUSt3R 的多视角一致性度量：用 3D 重建一致性来判定同一场景跨镜头/跨帧的几何是否自洽，对「换个角度房间布局就变了」这类废片非常有效，且不依赖 VBench-2.0 的重型模型链。
- AIGC 特有崩坏的专项检测：多指/肢体异常（除 VBench-2.0 的 ViTDetector 外，可用 MediaPipe Hand Landmarker + 手指数/拓扑规则做零成本兜底）、画面内文字乱码（PaddleOCR / EasyOCR 检出非法字形）。这两类是观众最先察觉的废片，却不在任何通用 VQA 指标的敏感区内。


====================================================================================================
# [legal-compliance] AI 数字人长片生产的合规边界（中国标识办法 / EU AI Act 第50条 / 美国州法与联邦立法 / 闭源 API ToS / 开源许可证 / C2PA 与水印）——检索日期 2026-09-19，仅陈述条文与出处，不构成法律意见

## 《人工智能生成合成内容标识办法》（国信办通字〔2025〕2号）  (国家网信办 / 工信部 / 公安部 / 广电总局（中国大陆）)
- 类别/成熟度: framework / **production** | 许可: 部门规章 | 成本: 合规成本，非算力成本
- 是什么: 中国对 AI 生成合成内容的强制标识规章，全文 14 条。第三条把标识拆成「显式标识」（用户可明显感知的文字/声音/图形）与「隐式标识」（文件数据中不易感知的技术措施）。第四条规定视频的显式标识必须加在「视频起始画面和视频播放周边的适当位置」，末尾和中间为可选；第四条第二款要求提供下载、复制、导出功能时，必须确保文件中含有满足要求的显式标识。第五条要求在文件元数据中添加隐式标识，内容为「生成合成内容属性信息、服务提供者名称或者编码、内容编号等制作要素信息」，并「鼓励」加数字水印。
- 关键事实:
    * 成文日期 2025-03-07，施行日期 2025-09-01（第十四条）
    * 管辖区：中华人民共和国境内（第二条，经《深度合成规定》《生成式AI暂行办法》口径引入）
    * 第九条：用户可申请「不含显式标识」的输出，服务提供者可在用户协议明确标识义务与使用责任后提供，并留存提供对象信息等日志「不少于六个月」——这是商业成片去除角标的唯一合法通道
    * 第六条：传播平台须核验元数据隐式标识；无隐式标识但用户声明的标「可能为」；均无但检测到痕迹的标「疑似」；并须在元数据追加「传播平台名称或者编码、内容编号」等传播要素
    * 第七条：应用商店上架审核时须核验生成合成内容标识材料
    * 第十条第二款：任何组织和个人不得恶意删除、篡改、伪造、隐匿标识，也不得为他人实施上述行为提供工具或服务
    * 第八条：服务提供者须在用户服务协议中写明标识的方法与样式
- 长片作用: 直接决定 2-8 分钟成片的母版交付形态。产线必须同时产出两条链路：(1) 元数据隐式标识——在最终 mux 后写入「生成合成属性 + 服务提供者名称/编码 + 内容编号」；(2) 显式标识——起始画面角标必须有，中/尾可选。若成片走「无角标母版」，必须走第九条路径：与用户/客户签协议把标识义务转移，并建立≥6个月的发放日志表（谁、何时、哪条内容编号）。第十条第二款还意味着：产线里任何「去水印/去角标」工具节点本身就是违规工具的提供行为，不能做成产品功能。
- 来源: https://www.gov.cn/zhengce/zhengceku/202503/content_7014286.htm

## GB 45438-2025《网络安全技术 人工智能生成合成内容标识方法》  (国家市场监督管理总局 / 国家标准化管理委员会；主管与归口：中央网信办)
- 类别/成熟度: framework / **production** | 许可: 强制性国家标准 | 成本: 标准全文需通过 openstd 在线预览或购买获取
- 是什么: 与标识办法配套的国家标准，规定标识的具体技术实现方法。英文名 Cybersecurity technology—Labeling method for content generated by artificial intelligence。标识办法第十一条把它变成硬约束：「服务提供者开展标识活动的，还应当符合……强制性国家标准的要求」。
- 关键事实:
    * 标准号 GB 45438-2025（GB 无 /T，全国标准信息公共服务平台检索页分类标注为「强标」，筛选项归入「强制性国家标准」）
    * 发布日期 2025-02-28，实施日期 2025-09-01（与标识办法同日）
    * CCS 分类 L80，ICS 分类 35.030
    * 管辖区：中华人民共和国
- 长片作用: 这是产线里元数据字段名、取值格式、角标最小尺寸/位置的唯一权威来源。标识办法只说「属性信息、服务提供者名称或者编码、内容编号」，具体 key 名与编码规则在 GB 45438-2025 正文里。落地前必须拿到标准正文再写 FFmpeg/metadata 注入脚本——不要凭二手博客里的字段名实现。
- 来源: https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=F32EA2A561F1886CD8D606513512D547 https://openstd.samr.gov.cn/bzgk/gb/std_list?p.p1=0&p.p2=GB%2045438

## 《互联网信息服务深度合成管理规定》  (国家网信办 / 工信部 / 公安部（中国大陆）)
- 类别/成熟度: framework / **production** | 许可: 部门规章 | 成本: —
- 是什么: 标识办法的上位规章，25 条。第十六条是隐式标识的法源（「应当采取技术措施添加不影响用户使用的标识，并依照法律、行政法规和国家有关规定保存日志信息」）；第十七条第一款列举必须做显著标识的五类服务，标识办法第四条的显式标识义务正是挂在这一款上。
- 关键事实:
    * 2022-11-03 网信办室务会审议通过，2022-12-11 公布，2023-01-10 起施行（第二十五条）
    * 第十七条第一款（三）：人脸生成、人脸替换、人脸操控、姿态操控等人物图像、视频生成，或显著改变个人身份特征的编辑服务 —— 数字人视频生成明确落入此款
    * 第十七条第一款（二）：合成人声、仿声等语音生成或显著改变个人身份特征的编辑服务
    * 第十四条第二款：提供人脸、人声等生物识别信息编辑功能的，应当提示使用者依法告知被编辑的个人，并取得其「单独同意」
    * 第十五条第二款：提供「生成或者编辑人脸、人声等生物识别信息」功能的模型、模板等工具，应当自行或委托专业机构开展安全评估
    * 第九条：基于手机号、身份证号、统一社会信用代码或国家网络身份认证进行真实身份信息认证，未认证者不得提供信息发布服务
    * 第十条：须对使用者的输入数据和合成结果进行技术或人工审核
    * 第十九条：具有舆论属性或社会动员能力的，须履行算法备案
    * 第十八条：不得采用技术手段删除、篡改、隐匿第十六、十七条规定的标识
- 长片作用: 决定了「角色虚构、设定成年、不碰真人肖像」这条内容策略在中国法下的分量：一旦碰真人脸/真人声，第十四条第二款的「单独同意」和第十五条第二款的安全评估立刻触发，合规成本从「加角标」跳到「逐人取得书面单独同意 + 安全评估备案」。虚构角色路线可以完全绕开这两条，只剩第十六/十七条的标识义务。另外第九条意味着面向公众的平台形态必须做实名认证。
- 来源: https://www.gov.cn/zhengce/zhengceku/2022-12/12/content_5731431.htm

## 《生成式人工智能服务管理暂行办法》  (网信办等七部门（中国大陆）)
- 类别/成熟度: framework / **production** | 许可: 部门规章 | 成本: —
- 是什么: 面向境内公众提供生成式 AI 服务的基础规章，24 条。第十二条把标识义务直接转致《深度合成规定》。
- 关键事实:
    * 2023-07-10 公布，2023-08-15 起施行（第二十四条）
    * 第二条第三款：未向境内公众提供服务的研发、应用不适用本办法 —— 纯内部/自用的自部署管线不落入
    * 第二条第二款：国家对利用生成式AI从事新闻出版、影视制作、文艺创作等活动另有规定的，从其规定
    * 第四条（四）：不得侵害他人肖像权、名誉权、荣誉权、隐私权和个人信息权益
    * 第七条（二）（三）：训练数据不得侵害知识产权；涉及个人信息应取得个人同意
    * 第十七条：具有舆论属性或社会动员能力的，须开展安全评估并履行算法备案
    * 第二十二条（二）：通过提供可编程接口等方式提供服务的组织、个人，同样是「服务提供者」
- 长片作用: 划出两条完全不同的合规路径：把长片系统做成「对境内公众开放的服务」→ 落入本办法，要算法备案 + 安全评估 + 标识 + 实名；只做内部产片、成片再上架到第三方平台 → 第二条第三款可能不落入本办法，但成片上传时平台会按第六条给你打「AI生成」标。第二条第二款还提示：走影视/网络视听发行时另有广电口径，需单独核。
- 来源: https://www.gov.cn/zhengce/zhengceku/202307/content_6891752.htm

## 《中华人民共和国民法典》第1018—1020、1023条（肖像权与声音）  (全国人民代表大会（中国大陆）)
- 类别/成熟度: framework / **production** | 许可: 法律 | 成本: —
- 是什么: 人格权编中肖像权的实体法依据。第1019条是「真人脸一律不碰」在中国法下最直接的条文：明确把「利用信息技术手段伪造」列为侵害肖像权的方式之一，且不以营利为要件。第1023条把声音准用肖像权保护规则。
- 关键事实:
    * 2020-05-28 通过，2021-01-01 起施行
    * 第1018条：自然人享有肖像权，有权依法制作、使用、公开或者许可他人使用自己的肖像
    * 第1019条：任何组织或者个人不得以丑化、污损，或者「利用信息技术手段伪造」等方式侵害他人的肖像权
    * 第1020条：合理实施的五种情形（个人学习/新闻报道/国家机关履职/公共环境展示/维护公共利益）可不经同意
    * 第1023条第二款：对自然人声音的保护，参照适用肖像权保护的有关规定
    * 管辖区：中华人民共和国
- 长片作用: 这是「角色虚构、不碰真人肖像」这条项目铁律的法律底座，且它比标识办法更硬：标识违规是行政处罚，肖像权侵权是民事诉权，任何被识别出的自然人都能单独起诉。实操上，参考图（9 图）与参考视频（3 视频）环节是最高风险点——只要上传的参考素材里含可识别真人面孔，即使输出是「虚构角色」，也可能被认定为「利用信息技术手段伪造」。第1023条同理约束音色参考。产线应在参考素材入库环节做人脸检测拦截，而不是只在输出端审。
- 来源: https://en.wikisource.org/wiki/Civil_Code_of_the_People%27s_Republic_of_China/Book_Four

## EU AI Act 第 50 条（透明度义务）  (欧盟（Regulation (EU) 2024/1689）)
- 类别/成熟度: framework / **production** | 许可: 欧盟条例 | 成本: —
- 是什么: 欧盟对 AI 交互与合成内容的透明度义务。第50(2)条要求生成合成音/图/视频/文本的「提供者」以机器可读格式标记输出并使其可被检测为人工生成或操纵；第50(4)条要求「部署者」对深度伪造内容披露其为人工生成或操纵。
- 关键事实:
    * 法规 2024-08-01 生效；第 50 条自 2026-08-02 起适用（已生效，检索日 2026-09-19）
    * 2026-08-02 前已投放市场的系统，第50(2)条的合规截止日为 2026-12-02（4个月过渡期）
    * 第50(4)条对「艺术、创作、讽刺、虚构或类似作品」有减损：只需以不妨碍作品展示或欣赏的适当方式披露该生成内容的存在
    * 第50(1)条：与人交互的 AI 须告知用户在与 AI 交互，除非对合理知情的自然人而言「显而易见」
    * 第50(2)条明确要求考虑技术可行性与 state of the art 标准（这是 C2PA / 水印被引入的法条入口）
    * 第50(7)条：委员会可制定实践守则（codes of practice），必要时可通过实施法案
    * AI Omnibus 于 2026-07-27 生效，高风险系统（Annex III）延至 2027-12-02，但未推迟第 50 条的 2026-08-02 适用日
- 长片作用: 如果成片在欧盟发行：第50(4)条的「虚构/创作作品」减损对 2-8 分钟虚构短片是适用的——不需要满屏警告，但仍要以适当方式披露内容为生成的。真正的硬义务落在第50(2)条，且它压在「提供者」身上：自部署 Wan/LTX/Hunyuan 产片时，你既是提供者也是部署者，机器可读标记（C2PA manifest 或水印）要自己做。走 Veo/Sora/Kling API 时第50(2)条由 API 方承担，你只承担第50(4)条的披露。注意 Hunyuan 许可证把 EU 排除在授权地域外，欧盟路径下 Hunyuan 不可用。
- 来源: https://artificialintelligenceact.eu/article/50/ https://artificialintelligenceact.eu/implementation-timeline/ https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai

## California AB 853 — California AI Transparency Act（扩展版）  (加利福尼亚州（美国）)
- 类别/成熟度: framework / **production** | 许可: 州法 | 成本: —
- 是什么: 把加州 AI 透明度法从「生成式 AI 提供者」扩展到大型在线平台、生成式 AI 托管平台与拍摄设备制造商，核心是 provenance data（来源数据）的保留与展示。
- 关键事实:
    * 2025-10-13 经州长签署并送交州务卿存档（chaptered）
    * 覆盖的生成式 AI 提供者义务推迟至 2026-08-02（须提供免费的 AI 检测工具并输出检测到的 system provenance data）
    * 大型在线平台义务自 2027-01-01 起：须检测并展示符合标准的来源数据，提供可供用户查验来源数据的界面，且「在技术可行范围内不得故意剥离任何 system provenance data」
    * 生成式 AI 托管平台义务自 2027-01-01 起：不得明知而提供缺少所需披露的系统
    * 拍摄设备制造商义务自 2028-01-01 起：默认嵌入潜在披露（厂商名、设备版本、时间戳）
    * 管辖区：美国加利福尼亚州
- 长片作用: 「不得故意剥离来源数据」这一条直接约束剪辑产线：如果成片将在加州大型平台分发，中间任何 FFmpeg 重编码步骤把 C2PA manifest 打掉，平台侧有义务不剥离，但你作为上游把它弄丢了就失去了可证明性。实践结论：C2PA manifest 必须在最终 mux 之后、以不再重编码的方式写入；同时保留 sidecar manifest 以便被剥离后仍可恢复。
- 来源: https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202520260AB853

## California Civil Code § 1708.86（AB 602 起，经 2025 年修订）  (加利福尼亚州（美国）)
- 类别/成熟度: framework / **production** | 许可: 州法 | 成本: —
- 是什么: 针对非自愿数字化色情内容（digitized sexually explicit material）的民事诉权。AB 602 于 2019 年加入，2025 年再修订。
- 关键事实:
    * 最新修订版自 2026-01-01 生效（Stats. 2025, Ch. 673）
    * 被描绘者可起诉：制作并明知披露、未经同意披露、以及明知协助或教唆上述行为的人
    * 法定赔偿：个人原告每次违法 1,500—50,000 美元；有恶意的最高 250,000 美元
    * 公诉人提起的：每次违法 25,000 美元，有恶意 50,000 美元
    * 同意必须是「以平实语言书面、明知且自愿签署」，并须包含对该素材及其将被并入的作品的一般性描述；被描绘者在三个工作日内可撤回同意（除非给了 72 小时审阅期或经授权代表批准）
    * 明确规定：服务方设有「禁止生成此类内容」的免责声明不构成抗辩
    * 管辖区：美国加利福尼亚州
- 长片作用: 「免责声明不构成抗辩」这一条是对平台型产品的直接警告：在 ToS 里写一句「禁止生成真人色情」不能免责。对本项目的意义是双重的：(1) 角色虚构 + 不碰真人脸这条策略同时也是这条法的规避路径，因为诉权主体是「被描绘的可识别自然人」；(2) 如果做成对外开放的平台，需要的是输入端人脸拦截与输出端审核的技术措施，而不是条款免责。
- 来源: https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=CIV&sectionNum=1708.86

## California AB 1836（已故名人数字复制品）  (加利福尼亚州（美国）)
- 类别/成熟度: framework / **production** | 许可: 州法 | 成本: —
- 是什么: 扩展加州已故名人权利，禁止在影音作品或录音中未经授权使用已故名人声音或肖像的计算机生成复制品。
- 关键事实:
    * 2024-09-17 经州长批准并送交州务卿存档；按加州惯例自 2025-01-01 生效
    * 赔偿：一万美元（$10,000）与实际损失中的较高者（传统 right of publicity 的下限为 750 美元）
    * 豁免：新闻/公共事务/体育报道、评论、批评、学术、讽刺、戏仿、纪录片或历史传记语境（除非造成真实性的虚假印象）、一闪而过或附带性使用、以及被豁免作品的广告
    * 管辖区：美国加利福尼亚州
- 长片作用: 「已故」这个维度常被产线忽略：历史人物、已故演员的面孔在参考图里同样构成风险，且加州对已故名人的保护期长。与 ELVIS Act 一起构成美国侧「真人脸一律不碰」的第二层理由——不只是在世者。
- 来源: https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240AB1836

## Tennessee ELVIS Act（Tenn. Code Ann. § 47-25-1105）  (田纳西州（美国）)
- 类别/成熟度: framework / **production** | 许可: 州法 | 成本: —
- 是什么: 把「voice」加入田纳西 right of publicity 保护客体，并创设了对「工具分发者」的直接责任——这在美国州法中是最罕见也对本项目最危险的一条。
- 关键事实:
    * §47-25-1105(a)(1)：明知使用或侵害他人姓名、照片、声音或肖像，用于广告、募捐、劝募或商品服务推销，未经事先同意者，构成 unauthorized use
    * §47-25-1105(a)(3)：分发、传输或以其他方式提供某种算法、软件、工具或其他技术、服务、设备，而该算法/软件/工具的「主要目的或功能是生成某一特定可识别个人的照片、声音或肖像」，且明知此种提供未经授权的，同样承担责任
    * ELVIS Act 于 2024-03-21 由州长签署（Public Chapter 588），自 2024-07-01 起施行
    * 管辖区：美国田纳西州
- 长片作用: (a)(3) 直接命中「本机 LoRA」这个工位：训练一个专门复刻某个真实个人面孔或音色的 LoRA 并把它分发出去（哪怕只是内部团队之间传、或挂到模型市场），在田纳西就是独立的可诉行为，不需要等到出片。项目的「不碰真人肖像」策略要向下贯彻到 LoRA / 参考音色资产库层，而不只是成片层。对虚构角色训练的 LoRA，(a)(3) 要求的「特定可识别个人」要件不成立，因此虚构角色路线是安全的。
- 来源: https://codes.findlaw.com/tn/title-47-commercial-instruments-and-transactions/tn-code-sect-47-25-1105/

## NO FAKES Act（S. 1367 / H.R. 2794，119th Congress）  (美国联邦（参议院 Coons 等 / 众议院 Salazar 等）)
- 类别/成熟度: framework / **vaporware** | 许可: 未通过的法案 | 成本: —
- 是什么: 拟创设联邦层面的 digital replica 权利，覆盖声音与视觉肖像。截至检索日仍未成法。
- 关键事实:
    * S. 1367 于 2025-04-09 提出（Coons 主提案，Blackburn、Klobuchar、Tillis 共同提案），13 名共同提案人（7 共和 6 民主）
    * H.R. 2794 于 2025-04-09 提出（Salazar 主提案），10 名共同提案人
    * 截至 2026-09-19，两案均停留在 Introduced 阶段，未出委员会；GovTrack 给出的成法概率为 S.1367 5%、H.R.2794 2%
    * 法案内容：数字复制品权可在死后延续最长 70 年；生前许可不超过 10 年（未成年人 5 年）且须书面；法定赔偿每作品 5,000—750,000 美元，视被告类型与合规努力而定；新闻、纪录片、评论、讽刺与一闪而过的使用有豁免
    * 前身：S. 4875（118th, 2024-07-31）、H.R. 9551（118th, 2024-09-12），均已失效
    * 管辖区：美国联邦（若通过）
- 长片作用: 按本次调研的判定口径：提出 17 个月、零委员会动作、双院成法概率 5%/2%，作为产线规划依据等同 vaporware——不应为它预留任何工程投入。美国侧的真实约束仍然来自州法（加州、田纳西等）。但它的存在意味着「真人脸」这条线的联邦风险随时可能一次性抬升，虚构角色路线是对这种立法不确定性的天然对冲。
- 来源: https://www.govtrack.us/congress/bills/119/s1367 https://www.govtrack.us/congress/bills/119/hr2794 https://www.congress.gov/119/bills/s1367/BILLS-119s1367is.xml

## Google Gemini API Additional Terms of Service（含 Veo 系列）  (Google（美国，全球适用）)
- 类别/成熟度: closed-api / **production** | 许可: 商业 API 条款 | 成本: 付费层是数据保护的前提，不是可选项
- 是什么: Veo 通过 Gemini API 提供，受本附加条款约束。付费与免费两档在训练数据使用上有根本差异。
- 关键事实:
    * 条款生效日 2026-03-23
    * 输出归属：「Google won't claim ownership over that content.」同时声明 Google 可能为他人生成相同或相似内容（即不保证独占性）
    * Unpaid Services（AI Studio、免费 API 配额）：Google 会使用你的 prompts、system instructions、cached content 与文件来改进和开发 Google 产品服务（含机器学习技术），人工审阅者可阅读并标注输入输出；条款明确要求「不要提交敏感、机密或个人信息」
    * Paid Services（已开通结算的付费配额）：Google 明确不使用 prompts 或 responses 改进产品；按 DPA 以数据处理者身份处理；日志仅为检测违规与维护安全而保留
    * 地域例外：EEA、瑞士、英国用户即使在免费层也享受付费层的数据保护；且这些地区用户必须使用付费服务
    * 检索类接地（grounding）数据保留 30 天
    * 禁止事项：开发竞争模型或逆向工程组件、绕过安全机制、在需监管批准的临床/医疗场景使用、自动化处理输出以构建训练数据集
    * 最低年龄 18 岁
- 长片作用: 对长片产线的直接结论：任何进入正式产片的 Veo 调用必须走付费配额，免费层等于把分镜脚本、角色设定和参考图交给 Google 训练并可能被人工审阅。「不得自动化处理输出以构建训练数据集」这一条还封死了「用 Veo 输出蒸馏本地模型」的路径——这与「不能本机 LoRA」的硬约束同源。输出所有权归你但不独占，意味着不能对单镜画面主张排他性。
- 来源: https://ai.google.dev/gemini-api/terms https://policies.google.com/terms/generative-ai

## Runway Terms of Use  (Runway（美国）)
- 类别/成熟度: closed-api / **production** | 许可: 商业 API / SaaS 条款 | 成本: —
- 是什么: Runway 的通用服务条款，覆盖 Gen 系列视频模型的输入与输出。
- 关键事实:
    * Last Updated: 2026-09-15（检索日前 4 天，是本次调研中最新的一份厂商条款）
    * 输出归属与商用：「The Company does not claim ownership of any of your Inputs or Outputs. Subject to your compliance with the Agreement, the Company does not restrict your commercial use of your Outputs.」
    * 训练：「Inputs and Outputs may be used by the Company to train and improve its AI models, algorithms and related technology.」用户授予 non-exclusive, irrevocable, perpetual, worldwide, royalty-free 许可
    * 本次检索未在条款正文中找到按套餐分级的训练退出（opt-out）条款——条款文本对所有用户一体适用
    * 真人素材：不得未经他人许可提交其照片；内容不得含裸露、暴力、色情
    * 封禁：可「immediately and without notice」暂停或终止服务；被终止用户不得以其他用户名重新注册或访问
    * 条款正文未规定正式的申诉流程
    * 管辖区：美国
- 长片作用: Runway 在本次对比中是训练条款最激进的一家：输出可商用、但输入输出一律可被用于训练且是不可撤销永久许可。对一个要建自有角色 IP 的长片系统，这意味着角色设定图、分镜参考、甚至成片片段都会进入 Runway 的训练集。若角色形象一致性是核心资产，Runway 只适合放在不涉及核心角色的工位（空镜、环境、转场素材），不适合做角色主镜。「不得以其他用户名重新注册」这一条还意味着封号是账号级不可恢复的，产线不能把单一 Runway 账号做成单点依赖。
- 来源: https://runway.com/terms-of-use https://runway.com/safety/usage-policy

## 可灵 AI（Kling）用户协议  (快手（中国）)
- 类别/成熟度: closed-api / **production** | 许可: 商业 SaaS 条款 | 成本: 去水印需会员
- 是什么: Kling 的用户协议 / user policy，规定输出归属、平台许可、水印与封禁。
- 关键事实:
    * 生效日期 2025-12-05
    * 输出归属：「输出内容的知识产权及相关权益归属于您」
    * 平台许可：快手可对用户输入与输出「无偿使用」，范围覆盖快手短视频平台及关联应用，用途包括训练、演绎与推广，且不区分会员身份
    * 水印：非会员不得去除可灵 AI 可见水印，或须「在生成内容的使用场景中显著提示」；会员可去除水印
    * 禁止内容：未经授权的真人肖像、未成年人素材（协议提示「慎重发布包含未成年人素材的内容」）、色情、暴力、仇恨言论、未加披露标识的深度伪造
    * 封禁：平台保留不经事先通知「暂停或终止」账号的权利；申诉渠道 support.cn@klingai.com
    * 管辖区：中国大陆
- 长片作用: 「会员才能去水印」与中国标识办法第九条是同一件事的两面：去掉的是平台商业水印，但标识办法第四条的显式标识义务不会因为买了会员而消失——两者不要混淆，产线仍需自行在起始画面加合规角标。「无偿使用、不区分会员」的训练条款与 Runway 同级，同样不适合承载核心角色资产。
- 来源: https://klingai.com/docs/user-policy https://klingai.com/docs/privacy-policy

## Seedance 2.5（ByteDance Seed，闭源 API）  (字节跳动 Seed（中国）)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API（具体条款未取得） | 成本: 按 API 计费，本次未取得价目原文
- 是什么: 字节的音画联合生成视频模型，仅通过 API 提供，无开放权重。官网自述为 audio-video joint generation model，支持白模控制与绿幕编辑、专业运镜控制。
- 关键事实:
    * 官网明示单次生成最长 30 秒，并可「extend twice」（两次续接）——这与项目已知硬约束中「2.5 约 30 秒」一致
    * 访问方式：官网仅提供 Get API / Try now，无权重下载入口，无法本机 LoRA
    * Seedance 1.0 官网描述为支持文/图到多镜头视频生成、1080p
    * 本次检索未能在官网页面上找到关于输出水印或内容标识的任何声明
    * docs.volcengine.com 在本次检索的网络环境下仅解析到 IPv6（NAT64）且不可达，火山方舟侧的 API 文档与服务条款原文未能取得
- 长片作用: 在 2-8 分钟长片产线里，Seedance 2.5 顶的是「角色主镜 + 对白镜头」这一最核心工位：30 秒单段 + 两次续接意味着单个场景可以做到接近 90 秒而不必靠外部拼接，显著降低分镜系统的接缝数量。合规上它有一个结构性优势：作为中国境内的服务提供者，标识办法第五条的隐式标识义务由火山方舟承担，输出文件里应当已带元数据标识；但显式标识（起始画面角标）与第九条的「无显式标识版本」申请，仍需在你的产线和用户协议里落实。ToS 未取得是本条最大的空白，见 uncertainClaims。
- 来源: https://seed.bytedance.com/en/seedance2_5 https://seed.bytedance.com/en/seedance

## Wan 2.2 系列（Apache-2.0 开放权重）  (阿里巴巴通义万相（中国）)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: TI2V-5B 24GB+；A14B 80GB+
- 是什么: 开放权重的视频生成模型族，是本次调研中商用边界最宽松的一家：真正的 Apache-2.0，无地域限制、无 MAU 门槛、无收入门槛。
- 关键事实:
    * 许可证：Apache 2.0；README 明确「We claim no rights over your generated contents」
    * T2V-A14B / I2V-A14B：2025-07-28 发布，27B 总参数 / 14B 激活（MoE），480P 与 720P，约 5 秒，需 80GB+ 显存
    * TI2V-5B：2025-07-28 发布，5B，720P，约 5 秒，24GB+ 显存（消费级 4090 可跑的唯一一档）
    * S2V-14B：2025-08-26 发布，14B，音频驱动，480P/720P
    * Animate-14B：2025-09-19 发布，14B
    * 使用约束以 README 的软性声明形式存在，非许可证条款：不得分享违法内容、不得伤害个人或群体、不得散布用于伤害的个人信息、不得传播虚假信息、不得针对弱势群体
    * Wan-Video GitHub 组织下另有 Wan-Animate-2（2026-08-08 有提交）、Wan-Dancer（2026-07-17 有提交），均标注 Apache-2.0；本次未取得其模型卡与规格
    * 截至检索日，Wan-Video 官方 GitHub 组织下未见 Wan 2.5 / 2.6 的开放权重仓库
- 长片作用: Apache-2.0 的意义在于：它是唯一一个可以合法承载「本机 LoRA + 自有角色 IP + 商业发行 + 输出再训练」全链路的选项。Seedance 顶主镜，Wan 顶的是需要角色一致性可控、需要本地微调、且输出要作为自有资产再利用的工位——例如固定角色的中近景补镜、转场、以及为分镜系统生成大量可丢弃的候选镜头。其 README 的使用约束不是许可证条件，因此自部署时的实际执行力接近于零：无远程 kill switch、无授权服务器、无地域校验，唯一的执行手段是上游社区与法律。这意味着合规责任 100% 落在你这一侧。
- 来源: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B https://github.com/Wan-Video/Wan2.2

## HunyuanVideo-1.5 与 Tencent Hunyuan Community License  (腾讯混元（中国）)
- 类别/成熟度: open-weights / **production** | 许可: Tencent Hunyuan Community License（非 OSI 开源） | 成本: 最低 14GB 显存（开 offload），单卡消费级可跑
- 是什么: 开放权重视频模型，但许可证是自定义社区许可，带明确的地域排除与 MAU 门槛，不是 OSI 开源许可。
- 关键事实:
    * HunyuanVideo-1.5：2025-11-20 发布，8.3B 参数，480p/720p 生成 + 超分至 1080p，最长约 10 秒（121 帧 @ 24fps），开启 offload 后最低 14GB 显存
    * 许可证名称：tencent-hunyuan-community
    * 地域排除（逐字）：「'Territory' shall mean the worldwide territory, excluding the territory of the European Union, United Kingdom and South Korea.」
    * MAU 门槛（逐字）：版本发布日当月，被许可方所有产品服务的月活超过 1 亿的，必须另行向腾讯申请许可
    * Acceptable Use Policy 明文禁止：以剥削或伤害未成年人为目的使用；未经同意、授权或合法权利冒充他人；生成机器生成内容而不明确且显著标明其为机器生成；散布可验证的虚假信息以伤害他人；影响安全、权利或福祉的高风险自动化决策（执法、医疗、就业、住房、信贷）；基于受保护特征的歧视
    * 再分发须附带许可证并保留 NOTICE：「Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2024 Tencent.」
- 长片作用: 14GB 显存 + 10 秒这个组合，让它在产线里适合顶「大批量廉价候选镜头 / 预演分镜（animatic）」的工位——单卡就能并发跑，用来在正式调用 Seedance 之前把分镜拍板。但它有两条硬红线必须写进部署清单：(1) 欧盟、英国、韩国不在授权地域内，任何面向这三地的使用或分发都超出许可；(2) AUP 明文要求「不得在不标明机器生成的情况下生成或散布内容」——这条把中国标识办法的义务变成了许可证条件，违反它同时是违约和违规。自部署时该许可证同样无技术强制力，但 MAU 与地域条款是可被腾讯事后主张的合同义务，不是空文。
- 来源: https://huggingface.co/tencent/HunyuanVideo-1.5 https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE https://huggingface.co/tencent/HunyuanVideo/blob/main/LICENSE

## LTX-2.5 与 LTX-2.x Community License  (Lightricks（以色列）)
- 类别/成熟度: open-weights / **production** | 许可: LTX-2.x Community License（2026-08-11；非 OSI 开源，非 OpenRAIL） | 成本: 权重约 66 GiB；年收入 ≥$10M 须付费许可
- 是什么: 开放权重的音画联合生成模型，但 2026-08-11 起换用新的 LTX-2.x Community License，引入了收入门槛型商用限制。它不是 OpenRAIL——许可证文本中无 OpenRAIL 字样，是自定义协议加 OpenRAIL 风格的 Attachment A 使用限制。
- 关键事实:
    * LTX-2.x Community License Agreement 日期 2026-08-11，适用于该日之后发布的所有 LTX-2.5 版本及未来 LTX-2.x；此前的 LTX-2 Community License 日期为 2026-01-05，适用于 LTX-2 及 LTX-2.3（至 2026-08-11）
    * 收入门槛（Section 2.1 逐字）：年收入达到或超过 10,000,000 美元的实体（Commercial Entities）必须另行取得付费许可（Commercial Use Agreement），联系 ltxv-licensing@lightricks.com；关联公司按合并口径计算
    * Section 2.2：上述实体仅可在「非商业目的」下免费使用——明确排除任何直接或间接创收活动、任何与终端用户直接交互或对其产生影响的使用、以及为商业用途训练/微调/蒸馏任何模型
    * Section 1.5：Derivatives 的定义极宽，包含 LoRA、微调权重、以及「用 LTX-2.x 生成的合成数据训练其他模型」；所有 Derivatives 必须仅以本协议条款分发
    * Section 3.5：LoRA 等微调产物转让给年收入 ≥1000 万美元的实体前，受让方必须先取得 Lightricks 付费许可
    * Attachment A 使用限制（共 16 项）：第 2 项禁止以剥削或伤害未成年人为目的使用；第 5 项禁止在不明确且可理解地声明内容为机器生成的情况下生成或散布内容；第 7 项禁止未经同意冒充他人（明文括注 e.g. deepfakes）；另并入 https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf，且 Lightricks 可随时更新（更新不溯及既往）
    * LTX-2.5 规格：22B transformer（dev 与 distilled 两档），Gemma 4 12B 文本编码器，DFR 管线默认 1024×1536 @ 24fps，UHD 4K 为 3840×2176，121 帧起步；完整权重下载约 66 GiB，HuggingFace 上为 gated repo，需接受条款
    * 显存优化：--quantization fp8-cast 搭配 --offload {cpu,disk}
    * 仓库另有 arXiv 论文 2601.03233
- 长片作用: 在产线里 LTX-2.5 是唯一一个开放权重且原生音画同出、原生支持 4K 的选项，顶的工位是「需要本地可控、且需要同步对白口型的角色镜头」——这是 Wan 做不到、Seedance 只能闭源做的一格。但许可证是三家开源里最紧的：Section 1.5 + 2.2 的组合意味着，用 LTX 输出去训练你自己的角色模型、或做角色 LoRA，一旦项目年收入过千万美元就必须买断；Section 3.5 还把这个义务传染给任何接收你 LoRA 的合作方。自部署时这些条款同样无技术强制力（本地推理不回连、不校验），但 gated repo 的接受记录和 Lightricks 的合同主张权是真实的追溯入口。规划建议：在收入跨过 1000 万美元门槛之前就把商务许可谈清楚，而不是事后补——Section 2.1 明确规定违约后需按标准费率补缴自使用起全期费用，且须在书面要求后 30 日内支付。
- 来源: https://github.com/Lightricks/LTX-2 https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x https://huggingface.co/Lightricks/LTX-2.5

## C2PA / Content Credentials 规范 2.4 与 c2patool  (C2PA（Linux Foundation 项目）/ Content Authenticity Initiative)
- 类别/成熟度: technique / **usable** | 许可: 规范开放；c2pa-rs 为开源（Apache-2.0/MIT 双许可） | 成本: CPU 侧开销可忽略；需要签名证书
- 是什么: 内容来源与编辑历史的密码学清单（manifest）标准，以及其 Rust 参考实现与命令行工具。在 MP4 等 BMFF 格式中，manifest 以 JUMBF 容器装入一个标准 uuid box。
- 关键事实:
    * 规范版本 2.4，发布于 2026-04；文档集包含 Content Credentials、crJSON、Attestations、Soft Binding API，以及面向 AI/ML 的实施指引
    * c2patool 最新发布版 v0.27.22（2026-09-10），同期 c2pa Rust 库 v0.90.22（2026-09-10）；v0.28.0-rc.1 于 2026-09-14 进入 RC；c2pa-rc-v0.91.0-rc.3 于 2026-09-17
    * 可读写 manifest 的视频/音频格式：mp4（video/mp4，分片 MP4/DASH 仅支持 Rust 库的基于文件操作）、mov（video/quicktime）、avi、m4a、mp3、wav、flac
    * 图像格式：jpg/jpeg、png、webp、avif、heic/heif、tif/tiff、gif、svg、dng、jxl；pdf 为只读
    * c2pa-rs 仍处于 0.x beta，破坏性变更约每两个月发一次
    * Soft binding = 不具统计唯一性的内容标识符（指纹或不可见水印），用于在比特已变化时仍能匹配；Durable Content Credentials = 带 soft binding、可在 manifest 仓库中被发现与恢复的凭证
- 长片作用: 这是 EU AI Act 第50(2)条「machine-readable format」和加州 AB 853「system provenance data」在工程上的落地件，也是中国标识办法第五条隐式标识的国际对应物（但两者字段体系不同，不能互相替代——中国口径要按 GB 45438-2025 写）。产线位置很明确：c2patool 必须是 FFmpeg 之后的最后一道工序，因为任何重编码都会破坏 manifest 的 hard binding。0.x beta 且每两个月有破坏性变更，意味着必须在产线里锁定 c2patool 版本号并纳入回归测试，不能用 latest。
- 来源: https://spec.c2pa.org/specifications/specifications/2.4/index.html https://github.com/contentauth/c2pa-rs https://github.com/contentauth/c2pa-rs/blob/main/docs/supported-formats.md

## FFmpeg 缺少原生 C2PA 支持（产线结构性事实）  (FFmpeg 项目)
- 类别/成熟度: framework / **production** | 许可: LGPL/GPL | 成本: —
- 是什么: 对 FFmpeg 主线源码的直接核查结果：configure 脚本中不存在任何 c2pa / libc2pa 相关的启用项，9.0 版 Changelog 与 <next> 版 Changelog 中均无 C2PA 相关条目。
- 关键事实:
    * 检索方式：直接 grep FFmpeg/FFmpeg master 分支的 configure 脚本，匹配 'c2pa' 与 'libc2pa'，零命中
    * 当前发布版本 n9.0.2 / n9.0.1 / n9.0；开发分支为 n9.1-dev
    * FFmpeg 9.0 Changelog 中新增项包括 LCEVC track muxing、APV Vulkan、ONNX Runtime DNN backend 等，无 C2PA
    * 检索日期 2026-09-19
- 长片作用: 这条否定性事实直接决定了长片产线的工序顺序，且经常被误设计：不存在「在 FFmpeg 里写入 C2PA manifest」这个选项。正确的管线是——分段生成 → FFmpeg 拼接/转码/混音 → 最终 mux 完成 → 调用 c2patool 对成片 MP4 写入并签名 manifest → 之后不得再经过任何 FFmpeg 重编码。与此并行，中国标识办法第五条要求的元数据隐式标识（按 GB 45438-2025 字段）可以在 FFmpeg 阶段通过 -metadata 写入，但要验证它与随后 c2patool 的 uuid box 写入不冲突；起始画面的显式角标则必须在 FFmpeg 的滤镜链里烧录，是渲染步骤不是元数据步骤。
- 来源: https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/configure https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/Changelog

## SynthID（Google 不可见水印）  (Google DeepMind)
- 类别/成熟度: technique / **usable** | 许可: 专有（SynthID-Text 另有独立的开源发布，本次未核实其现状） | 成本: 随 Google 服务提供，无单独计费信息
- 是什么: Google 自有的不可见水印体系，覆盖图像、视频、文本与音频，附带一个验证门户。
- 关键事实:
    * 覆盖的产品：图像（Nano Banana / Gemini 图像模型）、视频（Veo）、文本（Gemini app 与网页端）、音频（Lyria 与 NotebookLM 播客生成）
    * SynthID Detector 验证门户已上线，可上传图像、视频、音频文件检测；目前与记者和媒体从业者合作测试，对外为早期测试者候补名单制
    * 鲁棒性声明（官方措辞，无量化指标）：图像与视频水印「设计为可抵御裁剪、加滤镜、改变帧率、有损压缩等修改」；音频水印「不会被加噪、MP3 压缩或变速等常见修改改变」
    * 官方页面未给出第三方接入、开源发布、检测准确率或具体日期
- 长片作用: 对本项目它是一个「已经在你片子里、但你控制不了」的事实：只要某个镜头出自 Veo，成片里就带有 SynthID 视频水印，且它按官方说法能扛住帧率变化和有损压缩——也就是说经过 FFmpeg 拼接和转码后大概率仍然存在。这有两个后果：(1) 合规上是好事，它天然满足 EU AI Act 第50(2)条的机器可读标记，且无法被「误删」；(2) 但它也意味着「Veo 镜头 + 自部署模型镜头」混剪的成片，检测结果会是部分带 SynthID 部分不带，无法给整片一个统一的来源声明——统一的声明层只能由你自己写入的 C2PA manifest 提供。SynthID 不是可由你主动调用的工具，因此不能作为满足中国标识办法第五条「鼓励加数字水印」的自主手段。
- 来源: https://deepmind.google/science/synthid/

### 建议
合规上可以给出一个明确且不含立场的事实结构，供产线设计直接落位：

一、中国境内是三层叠加，缺一不可。第一层《生成式人工智能服务管理暂行办法》（2023-08-15 施行）决定你是否是「服务提供者」——只做内部产片、不向境内公众提供服务的，第二条第三款可能不落入；一旦对外开放接口或产品，算法备案与安全评估就是前置手续。第二层《深度合成管理规定》（2023-01-10 施行）第十七条第一款（二）（三）把人物图像视频生成和合成人声明确列入必须显著标识的范围，数字人长片百分之百落在里面。第三层《标识办法》（2025-09-01 施行）+ GB 45438-2025（同日实施，强制性国标）给出具体做法。工程上必须同时实现三件事：起始画面烧录显式角标（第四条第四项）、最终 mux 后写入含「属性信息 + 服务提供者名称或编码 + 内容编号」的元数据隐式标识（第五条，字段格式以 GB 45438-2025 正文为准，本次未能取得该正文，务必先买/在线预览后再写脚本）、以及若要交付无角标母版，走第九条的用户协议 + ≥6 个月发放日志路径。第十条第二款同时封死了在产品里内置「去标识」功能这条路。

二、「真人脸一律不碰」在四个法域同时有独立依据，且最危险的一条不在成片层而在资产层。中国是《民法典》第1019条（利用信息技术手段伪造即构成侵害肖像权，不以营利为要件）与第1023条（声音准用），加上《深度合成规定》第十四条第二款要求对被编辑个人取得「单独同意」、第十五条第二款要求安全评估。美国侧，加州 Civil Code §1708.86（最新修订 2026-01-01 生效，法定赔偿最高 25 万美元，且明确「ToS 里写禁止」不构成抗辩）、加州 AB 1836（已故名人数字复制品，2024-09-17 签署）、以及田纳西 ELVIS Act §47-25-1105(a)(3)——这一条最关键：分发「主要目的是生成某一特定可识别个人照片/声音/肖像」的算法、软件或工具本身即构成可诉行为，不需要等到出片。因此「不碰真人肖像」必须向下贯彻到 LoRA 与音色资产库，并在参考素材入库环节做人脸检测拦截（9 图 + 3 视频 + 3 音频的多模态参考通道是最高风险点），而不是只在输出端审核。反过来说，虚构角色路线使 (a)(3) 的「特定可识别个人」要件不成立，是对这一整组法律的结构性规避。NO FAKES Act（S.1367 / H.R.2794，2025-04-09 提出）截至 2026-09-19 仍停在 Introduced 阶段、成法概率 5%/2%，按本次判定口径等同 vaporware，不应为其预留工程投入。

三、欧盟第 50 条已经生效（2026-08-02，AI Omnibus 于 2026-07-27 生效但未推迟该日期；2026-08-02 前上市的系统合规截止 2026-12-02）。对 2-8 分钟虚构短片，第50(4)条的「艺术、创作、讽刺、虚构或类似作品」减损适用，只需以不妨碍观赏的适当方式披露，不需要满屏警告。真正的硬义务是第50(2)条的机器可读标记，且压在「提供者」身上——走 Veo/Sora/Kling API 时由对方承担，自部署 Wan/LTX/Hunyuan 时由你承担。注意 Hunyuan 许可证把欧盟、英国、韩国排除在授权地域外，欧盟发行路径下 Hunyuan 不可用。

四、闭源 API 的分工建议由 ToS 而非画质决定。Gemini API（条款 2026-03-23 生效）付费层明确不用你的数据训练、免费层会训练且有人工审阅，所以任何正式产片的 Veo 调用必须走付费配额；同时「不得自动化处理输出构建训练数据集」封死了蒸馏路径。Runway（2026-09-15 更新）与可灵（2025-12-05 生效）都是输出可商用、但输入输出一律可被用于训练且无分级退出，因此两者只适合放在不涉及核心角色的工位（空镜、环境、转场），不适合承载角色 IP 主镜。Seedance 2.5 官网确认单次 30 秒 + 两次续接，是主镜与对白工位的最佳选择，但火山方舟侧的 API 文档与服务条款在本次网络环境下不可达（docs.volcengine.com 仅解析到 IPv6），商用权、再训练、封禁与申诉条款全部未核实，这是本轮调研最大的空白，必须在签约前单独补齐。

五、开源三家的商用边界差异极大，直接决定它们能顶哪个工位。Wan 2.2 全系 Apache-2.0，无地域限制、无 MAU 门槛、无收入门槛，README 的使用约束是软性声明不是许可证条件，因此它是唯一能合法承载「本机 LoRA + 自有角色 IP + 商业发行 + 输出再训练」全链路的选项（TI2V-5B 24GB 显存可在消费级单卡跑）。HunyuanVideo-1.5（2025-11-20，8.3B，最低 14GB 显存，10 秒/121帧@24fps）显存最友好，适合大批量预演分镜，但许可证排除欧盟/英国/韩国、1 亿 MAU 门槛，且 AUP 把「须标明机器生成」变成了许可证条件。LTX-2.5（22B，1024×1536@24fps，UHD 3840×2176，权重约 66 GiB）是唯一开放权重且原生音画同出的，但 LTX-2.x Community License（2026-08-11）规定年收入 ≥1000 万美元须付费许可，且 Section 1.5 把「用其输出训练其他模型」也算作 Derivative、Section 3.5 把这个义务传染给 LoRA 的受让方——建议在跨过门槛前就把商务许可谈定，因为 Section 2.1 规定违约需按标准费率补缴全期费用并在书面要求后 30 日内支付。三家的共同点是：自部署时这些条款没有任何技术强制力（无回连、无授权校验、无 kill switch），实际执行力完全来自合同追索与 HuggingFace gated repo 的接受记录，因此合规责任 100% 在你这一侧。

六、C2PA 的工序位置是硬约束。经直接核查 FFmpeg master 分支的 configure 脚本与 Changelog，FFmpeg 没有任何原生 C2PA 支持，「在 FFmpeg 里写 manifest」这个选项不存在。正确管线是：分段生成 → FFmpeg 拼接/转码/混音/烧录角标 → 最终 mux → c2patool（锁定版本，当前稳定版 v0.27.22 / 2026-09-10，规范 2.4 / 2026-04）写入并签名 manifest → 之后不得再经过任何重编码。c2pa-rs 仍是 0.x beta、约每两个月一次破坏性变更，必须在产线里锁版本并做回归测试。中国的元数据隐式标识（GB 45438-2025 字段）与 C2PA manifest 是两套不同体系，不能互相替代，需并行写入并验证不冲突。SynthID 是 Veo 输出自带、不可主动调用的，混剪片会出现部分镜头带部分不带，统一的来源声明层只能靠你自己写入的 C2PA manifest。

### 存疑
- 火山方舟 / 火山引擎关于 Seedance 的 API 文档与服务条款原文本次未能取得：docs.volcengine.com 在本次网络环境下仅解析到 IPv6（NAT64，64:ff9b::657e:3138）且 curl 返回 000，www.volcengine.com/docs/* 全部 301 重定向到该不可达域名，/terms 与 /legal 为 JS 渲染的 SPA 无法提取文本。因此以下全部未核实，需另行补查：Seedance 输出物的商用权归属、是否允许用输出再训练模型、账号封禁的触发条件与申诉流程、企业版与个人版的差异、按秒或按 token 的价目、以及输出文件是否已由火山方舟按标识办法第五条写入元数据隐式标识。项目已知硬约束中「多模态参考最多 9 图 + 3 视频 + 3 音频」「官方单段 4-15 秒」「官方通道有内容审核」三项本次均未能从官方文档核实，仅 Seedance 2.5 的「单次 30 秒 + 可续接两次」在 seed.bytedance.com 官网得到确认。
- OpenAI Sora 的条款本次完全未能取得：openai.com、help.openai.com、sora.chatgpt.com 均返回 403（Cloudflare 拦截），congress.gov 同样 403，web.archive.org 在本环境不可用。因此 Sora 的输出所有权、商用权、cameo/肖像同意机制、对真人与未成年人的限制、是否写入 C2PA 元数据与可见水印、封禁与申诉条款，全部无法陈述。不要把任何关于 Sora 条款的说法写进决策依据，需换网络环境或由人工直接访问确认。
- GB 45438-2025 的正文字段名与取值格式未取得。全国标准信息公共服务平台的在线预览需 JS 查看器，本次仅确认了标准号、中英文名称、发布日期 2025-02-28、实施日期 2025-09-01、CCS L80、ICS 35.030、主管与归口为中央网信办。网络上流传的字段结构（例如一个名为 AIGC 的元数据扩展字段，内含 Label / ContentProducer / ProduceID / ReservedCode / ContentPropagator / PropagateID 等键）本次无法从任何权威来源验证，不应据此实现元数据注入脚本。
- 标准的强制性属性存在来源冲突：全国标准信息公共服务平台的检索列表页把 GB 45438-2025 归类为「强标」并列在「强制性国家标准」筛选项下（这与 GB 而非 GB/T 的编号惯例一致），但对其详情页的自动摘要一度读作「推荐性国家标准」。本报告按列表页与编号惯例采信为强制性，但若该属性对合规结论有决定性影响，应以标准正文封面的标注为准。
- EU AI Act 第 50(7) 条项下的「AI 生成内容标记与标注实践守则」（Code of Practice on marking and labelling AI-generated content / Transparent Generative AI Systems）的当前状态、发布日期、以及它是否点名 C2PA 或 SynthID，本次未能核实：digital-strategy.ec.europa.eu 对相关 slug 返回软 404（HTTP 200 但内容为错误页），artificialintelligenceact.eu 的对应页面返回 404。
- Tennessee ELVIS Act 的条文内容（§47-25-1105(a)(1) 与 (a)(3)）已通过 FindLaw 核实，但其签署日 2024-03-21 与施行日 2024-07-01 未能从一手来源确认：publications.tnsosfiles.com 的 Public Chapter 588 PDF 返回 403，tn.gov 与 capitol.tn.gov 的相关页面分别返回 404 与 ECONNRESET。该两个日期属广为引用的公开事实，但本轮未取得原始出处。
- Wan-Animate-2（GitHub 最后提交 2026-08-08）与 Wan-Dancer（2026-07-17）两个仓库仅确认存在且标注 Apache-2.0，其模型卡、参数量、分辨率、时长、显存需求与发布日期均未取得。同时，「Wan-Video 官方组织下不存在 Wan 2.5 / 2.6 开放权重」这一结论基于该组织 GitHub 仓库列表（本次仅列出 6 个仓库），并非穷尽性核查——Wan 2.5 是否以 API-only 形式存在、或权重是否发布在 HuggingFace 而未建 GitHub 仓库，未验证。
- LTX-2.x 所并入的 Lightricks Acceptable Use Policy（https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf）正文本次未下载，其中关于 NSFW、未成年人与真人肖像的具体措辞未核实。许可证正文明确该 AUP 可由 Lightricks 随时更新且「使用时生效的版本」有效，因此其内容属于动态约束，需在部署时取当期版本。
- Runway 的「无按套餐分级的训练退出」这一结论来自对其 Terms of Use 正文的检索，未覆盖其 Enterprise / 企业协议——企业版另行签署的 MSA 中是否包含训练退出条款，未核实。同理，可灵、Veo 的企业版与个人版差异本次只核实了 Gemini API 的付费/免费分层，其余厂商的企业条款均未取得。
- SynthID 的第三方可用性、是否有开源发布（SynthID-Text 在 Hugging Face 的现状）、检测准确率与误报率、以及 SynthID Detector 是否已对公众开放，官方页面未提供，本次未能核实。官方仅给出定性的鲁棒性措辞，无任何量化指标。
- GitHub REST API 在本次调研后段触发速率限制（未认证请求），因此 Tencent-Hunyuan 组织下是否存在比 HunyuanVideo-1.5 更新的视频模型（例如 2026 年的版本）未能核查；HunyuanVideo-1.5 为本次可确认的最新版本，但不保证是当前最新。


====================================================================================================
# [longform-extension] 把短片段延长到分钟级的技术路线全景（截至 2026-09-19 实检）

## Stable Video Infinity (SVI / SVI 2.0 / SVI 2.0 Pro)  (EPFL VITA Lab (vita-epfl))
- 类别/成熟度: open-weights / **usable** | 许可: MIT（LoRA 权重）；基座 Wan 2.2 为 Apache 2.0 | 成本: 显存恒定，等于跑一次 Wan 2.2 I2V A14B 81 帧（24GB 卡需量化，实测推荐 fp16 + 40GB 以上；本地 4090 社区方案靠 block-swap）
- 是什么: 在 Wan I2V 基座上叠一层 LoRA 的「误差回收微调」(Error-Recycling Fine-Tuning)：训练时把 DiT 自己生成的错误注入到干净输入上，模拟误差累积轨迹，让模型学会自我纠偏。推理端是无状态滚动生成——每个 clip 用上一 clip 末帧做锚，显存恒定，与总长无关。
- 关键事实:
    * arXiv 2510.09212，2025-10-10；ICLR 2026 Oral
    * SVI 1.0: 2025-10 发布，基座 Wan 2.1 I2V-14B 480P
    * SVI 2.0: 2025-12-04；SVI 2.0 Pro: 2025-12-26，基座换成 Wan 2.2 I2V-A14B（HIGH/LOW 双 LoRA）
    * 只训 LoRA，约 1k 样本即可；训练环境 A100 80GB / CUDA 12.0 / Torch 2.5.0
    * 官方 demo：SVI-Shot 20 分钟压测、Tom&Jerry 10 分钟、SVI-Talk 10 分钟
    * 社区实测稳定区间：480p (480×832) + 81 帧/clip + 24fps + 每 clip 换 seed → 1 分钟无可见色偏；40 秒无色偏已被多人复现
    * 已知崩点：121 帧/clip 必然色偏；必须 fp16（LoRA 以 fp16 训练）；量化/蒸馏掉质量；LightX2V 开太猛会慢动作
    * 不能用标准 Wan workflow——padding 设置不同，这是社区最常见的报错来源
    * License: MIT
- 长片作用: 2-8 分钟产线里顶「B-roll 长空镜 / 氛围镜 / 单场景长镜」工位。它是目前开源侧唯一被大量第三方独立复现到「1 分钟不崩」的方案，且是 LoRA 形态——可以直接挂在你已有的 Wan 2.2 数字人 LoRA 旁边，不用换基座。但 480p 训练域是硬伤，720p 出图漂移风险明显上升，做不了接近 Seedance 2.0 的单镜观感。
- 来源: https://arxiv.org/abs/2510.09212 https://github.com/vita-epfl/Stable-Video-Infinity https://github.com/vita-epfl/Stable-Video-Infinity/issues/51

## Helios / Helios-Base / Helios-Mid / Helios-Distilled  (北大 PKU-YuanGroup（与字节合作）)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: H100 单卡 ~24GB 常规；group offloading 降至 ~6GB（速度代价大）
- 是什么: 14B 自回归扩散模型，专门为「实时 + 分钟级」重训。不用 KV-cache、不用因果掩码、不用稀疏/线性注意力、不用 TinyVAE、不用量化，而是用三级金字塔把历史帧重度压缩，token 预算恒定。抗漂移不是后处理，是训练时主动模拟漂移：显式建模 position shift / color shift / restoration shift 三种漂移形态，加 Relative RoPE（治重复）、First-Frame Anchor（治色跳）、Frame-Aware Corrupt（让模型提前适应不完美历史帧）。
- 关键事实:
    * arXiv 2603.04379，2026-03-04；权重 2026-07-28 仍在更新
    * 14B 参数，safetensors，Apache 2.0，HF: BestWishYsh/Helios-Base
    * 单张 H100 端到端 19.5 FPS；昇腾 NPU 约 10 FPS
    * 官方示例 1452 帧 ≈ 24fps 下 60 秒；标准分辨率 640×384
    * chunk 粒度 33 帧，num_frames 建议取 33 的倍数
    * 显存：常规 ~24GB；开 group offloading 可压到 ~6GB；训练时 80GB 卡内可塞 4 个 14B
    * 第三方 H200 实测吞吐几乎与长度无关：240 帧 24s / 480 帧 42s / 960 帧 82s（约 11 FPS 恒定）
    * 三档：Base（质量最好）/ Mid / Distilled（效率最好）；官方承认 I2V、V2V 略弱于 T2V
    * 支持 Diffusers / vLLM-Omni / SGLang-Diffusion 三条推理路径
- 长片作用: 目前 2026 年开源侧「单模型长时序」最强的一张牌，而且权重真开了（有第三方博客写「no released weights」是过期信息）。在你的产线里顶「长镜引擎 / 实时预演」工位——因为吞吐与长度无关，可以拿它做分镜草稿的分钟级快速预览，定稿再交给 Seedance。硬伤是原生 640×384，要出片得走超分链路，单镜观感距 Seedance 2.0 有代差。
- 来源: https://arxiv.org/abs/2603.04379 https://github.com/PKU-YuanGroup/Helios https://huggingface.co/BestWishYsh/Helios-Base

## LongCat-Video + LongCat-Video-Avatar 1.0 / 1.5  (美团 LongCat 团队)
- 类别/成熟度: open-weights / **usable** | 许可: MIT（权重） | 成本: 官方 README 未给 VRAM 数字；示例为 2 卡推理，13.6B 稠密推 720p 实际需 ≥48GB 或多卡/offload
- 是什么: 13.6B 稠密 DiT，关键点是「Video-Continuation 是原生预训练任务之一」——不是事后拼的外挂，而是和 T2V/I2V 在同一框架里联合训练，所以续写时分布不失配。时空双轴 coarse-to-fine + block sparse attention 控成本。Avatar 分支是音频驱动数字人，1.5 把 Wav2Vec2 换成 Whisper-Large，并做了 8 步蒸馏。
- 关键事实:
    * LongCat-Video: arXiv 2510.22200，2025-10-25 发布；13.6B 参数；MIT License
    * 输出 720p / 30fps；官方表述为 minutes-long，无色偏与质量衰减
    * Avatar 1.0: 2025-12-16；Avatar 1.5: 2026-05-21 开源（美团技术博客 2026-05-25）
    * Avatar 1.5：50 步 → 8 步蒸馏，约 15× 提速，10 秒视频约 1 分钟出片
    * Avatar 1.5 指标：跳帧率 0.8%，唇音误差率 29.8%（对比组内最低）；帧级 GRPO 对齐
    * Avatar 1.5 支持 480P 与 720P；共享基座 + 多 LoRA 适配器降低显存
    * 权重 MIT，GitHub + HF + ModelScope 三处同步
- 长片作用: 在你的产线里同时顶两个工位：（a）通用长镜续写引擎，（b）长口播数字人。MIT 许可 + 原生续写预训练，是唯一一个「长时序不是补丁」的开源基座，适合做你自研 LoRA 的落脚点。Avatar 1.5 的 8 步蒸馏把成本压到可批量，是 2-8 分钟对白段落的现实选项。
- 来源: https://arxiv.org/abs/2510.22200 https://github.com/meituan-longcat/LongCat-Video https://huggingface.co/meituan-longcat/LongCat-Video

## SkyReels-V3（R2V-14B / V2V-14B / A2V-19B）  (Skywork AI（昆仑万维）)
- 类别/成熟度: open-weights / **usable** | 许可: 自定义 LICENSE.txt（未确认是否允许商用，需实读） | 成本: 未公布；14B 720p 推理经验值 ≥40GB，FP8 低显存模式可降
- 是什么: 统一多模态 in-context 学习框架，把参考图生视频、音频驱动数字人、视频续写放进一个体系。长视频靠 keyframe-constrained generation（关键帧约束）+ history enhancement（历史增强）两个机制，而不是纯自回归滚动。
- 关键事实:
    * 2026-01-29 同时放出推理代码与权重，API 上 apifree.ai
    * 三个变体：R2V-14B（参考图生视频）、V2V-14B（视频续写）、A2V-19B（口播数字人）
    * 明确的时长上限：R2V 5 秒 / V2V 续写 30 秒 / A2V 数字人 200 秒
    * 720P 默认，低显存可走 540P / 480P；24 fps
    * --low_vram 走 FP8 量化；官方未公布具体 VRAM 阈值
    * 支持 1:1 / 3:4 / 4:3 / 16:9 / 9:16
    * 社区已有 GGUF 量化（vantagewithai/SkyReels-V3-14B-GGUF）
    * License 为仓库自带 LICENSE.txt（非标准 Apache/MIT，需逐条核）
- 长片作用: A2V-19B 的 200 秒是本次调研里开源侧最硬的「单次 >60 秒」数字。原因很直接：音频把运动先验锁死了，误差没有自由度可累积——这恰恰揭示了分界线的本质。在你的产线里它顶「长段口播/对白数字人」工位，一次 200 秒能覆盖一个完整对白段落。R2V 只有 5 秒、V2V 只有 30 秒，说明同一团队在「自由运动」场景下也不敢承诺更长。
- 来源: https://github.com/SkyworkAI/SkyReels-V3 https://huggingface.co/Skywork/SkyReels-V3-A2V-19B https://huggingface.co/Skywork/SkyReels-V3-R2V-14B

## SkyReels-V2 Diffusion Forcing (DF)  (Skywork AI)
- 类别/成熟度: open-weights / **usable** | 许可: 自定义 LICENSE.txt | 成本: 1.3B@540P 14.7GB（4090 可跑）；14B@540P 43–51GB（需 A100/H100）
- 是什么: Diffusion Forcing 框架：每个 token 独立噪声等级 + 因果注意力，配合 FoPP（Frame-oriented Probability Propagation）和非递减噪声调度，把组合搜索空间从 O(1e48) 压到 O(1e32)。续写靠条件化上一段末帧，理论无限长。
- 关键事实:
    * arXiv 2504.13074，2025-04-18
    * 1.3B 与 14B 两档，各有 DF / T2V / I2V 变体
    * 540P: 97–737+ 帧（约 4s–30s+）；720P: 121–1457+ 帧（约 5s–60s+），24fps
    * 峰值显存：1.3B@540P ≈ 14.7GB；14B@540P ≈ 43.4–51.2GB
    * VBench1.0 总分 83.9%；SkyReels-Bench T2V 3.14 / I2V 3.29
    * --addnoise_condition 建议 20–50，用来压跨段不一致
    * 异步推理（asynchronous inference）指令跟随与一致性更好但更慢
    * License 为自定义 LICENSE.txt
- 长片作用: 是「无限长」这个说法最早被工程化的开源实现，1.3B 版本是唯一一个 24GB 消费卡能本地跑 DF 长视频的选项。但 2026 年已经被 SVI / Helios / LongCat 在稳定性上全面超过，V3 出来后官方重心也转移了。在你的产线里现在只适合做低成本草稿 / 动态分镜预览，不建议做成片工位。
- 来源: https://arxiv.org/abs/2504.13074 https://github.com/SkyworkAI/SkyReels-V2 https://huggingface.co/docs/diffusers/en/api/pipelines/skyreels_v2

## FramePack / FramePack-F1 / FramePack-P1  (lllyasviel（Stanford，ControlNet 作者）)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 6GB 笔记本 GPU 即可跑 13B，这是它唯一无可替代的点
- 是什么: 下一帧（下一段）预测架构：把输入上下文按时间重要性压缩到恒定长度，使生成负载与视频总长无关（O(1) 显存）。F1 是纯前向单向版（动态更大但更容易崩）。P1 加了两个抗漂移设计：Planned Anti-Drifting（先生成远端锚点段再补中间，锚点之间不漂）和 History Discretization（对整个数据集做 K-Means，把历史转成离散 token，消除训练/推理表示 gap，锚点本身也不漂）。
- 关键事实:
    * FramePack 主版：2025-04；F1: 2025-05-03；P1 结果: 2025-06-26 与 2025-07-14 陆续公布
    * 13B 模型（基于 HunyuanVideo 改造）在 6GB 笔记本显存上可扩散数千帧 @ fps-30
    * 官方宣称可到 60 秒，社区实测 60–120 秒可出但质量不保
    * 已知崩点非常明确：超过 8–10 秒即出现皮肤/高光闪烁、色偏、过饱和、解剖结构崩坏；60 秒时人物外观、光照、颜色大幅改变
    * P1 是论文/结果发布形态，主仓库集成度低于 F1
    * License: Apache 2.0
- 长片作用: 2025 年的「显存平民化」里程碑，但在 2-8 分钟成片产线里已经不该进主链路——8-10 秒后画质塌方是结构性的，不是调参能救的。它的价值现在是（a）超低显存离线批量出草稿，（b）P1 的 Planned Anti-Drifting「先打远端锚点再补中间」这个思路值得你的分镜系统借鉴：先定关键帧再补运动，正是拼接式产线的正确形态。
- 来源: https://github.com/lllyasviel/FramePack https://lllyasviel.github.io/frame_pack_gitpage/ https://lllyasviel.github.io/frame_pack_gitpage/p1/

## Self-Forcing（原版）  (Xun Huang 等（Adobe / CMU 等），NeurIPS 2025 Spotlight)
- 类别/成熟度: technique / **usable** | 许可: 见仓库（研究用途为主） | 成本: 4090 可跑短片；但 10 秒即 ~129GB，长视频需要 KV cache 压缩才可用
- 是什么: 训练时就模拟推理过程：带 KV cache 做自回归 rollout，让模型在「看自己生成的历史」上训练，消除 train-test 分布错配（exposure bias）。是后续所有 Forcing 系方法的祖宗。
- 关键事实:
    * arXiv 2506.08009，2025-06-09
    * 单张 RTX 4090 即可实时流式生成，质量对标当时 SOTA 双向扩散
    * 基座 Wan2.1-T2V-1.3B
    * 第三方实测（H200）：5s/81 帧 ≈70s；10s/165 帧 168s 且吃 ~129GB 显存；20s/321 帧 287s，KV cache 被迫截断到 42 帧
    * 显存瓶颈是它的结构性上限——KV cache 随长度线性涨，10 秒就把 H200 打满
    * 代码开源：guandeh17/Self-Forcing
- 长片作用: 它本身不是长视频方案，是「让自回归视频不崩」的基础训练范式。在你的产线里不直接占工位，但它决定了你选的任何自回归长视频模型的底层质量。它暴露的 129GB@10s 显存墙，正是 2026 年一堆 KV cache 压缩论文（VideoMLA / Forcing-KV / ARL2）存在的理由。
- 来源: https://arxiv.org/abs/2506.08009 https://github.com/guandeh17/Self-Forcing https://self-forcing.github.io/

## Self-Forcing++  (Justin Cui 等（UCLA / 字节）)
- 类别/成熟度: research / **research** | 许可: 见仓库 | 成本: 恒定显存（rolling KV cache），但 1.3B 基座画质天花板低
- 是什么: 把 Self-Forcing 的自 rollout 扩展到分钟级：backward noise initialization（消除 chunk 边界断裂）+ extended distribution matching（用短片段教师对齐长自生成序列）+ rolling KV cache + GRPO 微调。关键是不需要长视频真值标注，全靠教师的短段知识 + 学生自生成动态。
- 关键事实:
    * arXiv 2510.02283，2025-10-02
    * 最长 4 分 15 秒，达到基座位置编码理论上限的 99.9%，是 CausVid 基线的 50×+
    * 基座 Wan2.1 1.3B
    * 代码仓库 justincui03/Self-Forcing-Plus-Plus 存在，但第三方产线评估（Atlas Cloud）判定「基础设施不足以生产部署」
    * 公认短板：demo 内容偏静态，动态幅度上不去
- 长片作用: 它证明了「位置编码外推上限 = 单模型单次长度的物理天花板」这件事——4分15秒就是 Wan2.1 1.3B 的 PE 极限。这个结论对你很重要：任何单模型超长方案都会撞到这堵墙，除非改 PE（RIFLEx / LoL 走的路）。但 1.3B + 静态内容的画质，离 Seedance 2.0 差两个代际，不能进成片链路。
- 来源: https://arxiv.org/abs/2510.02283 https://github.com/justincui03/Self-Forcing-Plus-Plus https://huggingface.co/papers/2510.02283

## CausVid  (MIT / Adobe（Tianwei Yin 等）)
- 类别/成熟度: technique / **research** | 许可: 见仓库 | 成本: 1.3B 级别，消费卡可跑
- 是什么: 把双向视频 DiT 用 DMD 式少步蒸馏成因果学生模型，靠 rolling KV cache 逐 chunk 流式生成，前面的 K/V 复用而不重算。是「双向教师 → 因果学生」这条线的开山之作。
- 关键事实:
    * arXiv 2412.07772，2024-12-10
    * 1.3B 量级，延迟约 0.78（与 Self-Forcing 同级）
    * 2 分钟视频测试中 Quality Drift 分数 3.35（越低越好），对比 Self-Forcing 1.66、Rolling Forcing 0.01
    * 误差累积是它最被诟病的点，长视频明显退化
- 长片作用: 历史坐标，2026 年不该再进产线。它的 Quality Drift 3.35 是整个领域的「差基线」，你看到任何方案报数字都会拿它当对照组。唯一用途是理解为什么纯因果逐帧预测必然漂移——这是 Rolling Forcing 用「放松严格因果」来解决的问题。
- 来源: https://arxiv.org/abs/2412.07772 https://arxiv.org/html/2412.07772v1

## Rolling Forcing  (TencentARC + NTU（Kunhao Liu 等），ICLR 2026)
- 类别/成熟度: research / **research** | 许可: 见 TencentARC 仓库 | 成本: 单卡实时（未公布具体 VRAM 数字）
- 是什么: 三件事：(1) rolling window 联合去噪——同时对多帧以递增噪声等级去噪，放松相邻帧的严格因果性，从根上压住误差增长；(2) attention sink——永久保留最初若干帧的 KV 作为全局锚，锁住长程一致性；(3) 在大幅拉长的去噪窗口上做少步蒸馏的高效训练算法，非重叠窗口，缓解自生成历史的 exposure bias。
- 关键事实:
    * arXiv 2509.25161，2025-09-30；ICLR 2026 收录
    * 单张 GPU 上 16 FPS 实时，multi-minute（多分钟）文生视频
    * 2 分钟视频 Quality Drift = 0.01，对比 Self-Forcing 1.66、CausVid 3.35——几乎零可感退化
    * 权重在 HF: TencentARC/RollingForcing
    * 人工评测在「整体质量」和「更少误差累积」两项上都压过 5 个主流基线
- 长片作用: Quality Drift 0.01 @ 2 分钟是本次调研里最漂亮的抗漂移数字，而且权重真放出来了。attention sink（首帧 KV 永久保留做全局锚）这个机制直接可以抄到你的分镜拼接系统里——跨镜头保持角色外观，本质就是给每个新镜头喂一个固定的锚 KV。目前仍是研究形态（无 ComfyUI 生态、无量化、无 LoRA 训练链路），不能当成片引擎，但值得作为你自研长镜模块的技术底座。
- 来源: https://arxiv.org/abs/2509.25161 https://huggingface.co/TencentARC/RollingForcing https://arxiv.org/html/2509.25161v1

## MAGI-1  (Sand AI)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 24B: 8×H100（不现实）；4.5B: 单张 4090（画质代价大）
- 是什么: chunk 级自回归世界模型：把视频切成固定 24 帧的 chunk，逐 chunk 去噪预测。流水线设计允许同时处理最多 4 个 chunk，峰值推理成本与视频总长无关，天然支持流式与无缝续写。支持 chunk-wise prompting（每段给不同提示词）。
- 关键事实:
    * arXiv 2505.13211，2025-05-19；开源时间 2025-04-21
    * 两档：24B 与 4.5B；Apache 2.0
    * chunk = 24 帧；pipeline 并发 4 chunk
    * 硬件：24B 需 8×H100/H800；24B-distill+fp8_quant 可跑 4×H100 或 8×RTX 4090；4.5B 单张 RTX 4090
    * Physics-IQ：V2V 56.02 / I2V 30.23，当时超过 VideoPoet、Kling 1.6、Sora
    * 恒定峰值显存，理论无限长
- 长片作用: chunk-wise prompting 是它对你最有用的特性——一次生成里分段给不同提示词，等于在单模型内部做了微型分镜。但 24B 的 8×H100 门槛让它在国内自建产线基本不可行，4.5B 画质又不够。2026 年已被 Helios / LongCat 在「质量×成本」上全面超越。定位：技术参考，非生产工位。
- 来源: https://arxiv.org/abs/2505.13211 https://github.com/SandAI-org/MAGI-1 https://huggingface.co/sand-ai/MAGI-1

## StreamingT2V / StreamingSVD  (Picsart AI Research，CVPR 2025)
- 类别/成熟度: research / **research** | 许可: 见仓库 | 成本: 2024 代基座，消费卡可跑但画质不达 2026 标准
- 是什么: 自回归续写式长视频：短模型 + 条件注意力模块（CAM，短期记忆）+ 外观保持模块（APM，长期记忆，锁住首帧外观）+ 随机混合的视频增强器做无缝拼接。
- 关键事实:
    * CVPR 2025 正式收录
    * StreamingSVD 可在 720×1280 自回归生成任意长度
    * 论文主张 2 分钟级别，且质量不随长度衰减
    * 基座是 SVD / ModelScope 等 2024 年代模型，画质已明显落后
- 长片作用: 历史坐标。它的 CAM（短期记忆）+ APM（长期外观记忆）双记忆结构，正是后来所有方案（attention sink / first-frame anchor / gated recall）的原型。在 2026 年产线里没有工位，但它的架构划分（短期运动记忆 vs 长期外观记忆）可以直接指导你的分镜系统设计：跨镜保外观用参考图，镜内保运动用末帧锚。
- 来源: https://openaccess.thecvf.com/content/CVPR2025/papers/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.pdf

## Video-Infinity  (NUS 等)
- 类别/成熟度: research / **research** | 许可: 见仓库 | 成本: 需要多卡（8 卡级别）才有意义
- 是什么: 把一个长视频任务切分到多张 GPU 并行，用 dual-scope attention（双尺度注意力）调制时序自注意力，在设备间平衡局部与全局上下文。本质是分布式并行推理，不是抗漂移方案。
- 关键事实:
    * 训练型方案（需要适配），非训练无关
    * 核心卖点是多 GPU 并行加速长视频生成，不解决误差累积
    * 基座为 2024 代模型（VideoCrafter2 等）
    * 2026 年无活跃更新，生态基本停滞
- 长片作用: 在你的产线里没有工位。它解决的是「长视频推理太慢」而不是「长视频会崩」，而 2026 年 Helios 这类恒定吞吐方案已经把速度问题在单卡上解决了。列在这里只为闭合你的必查清单。
- 来源: https://github.com/huggingface/diffusers/discussions/8925

## FIFO-Diffusion  (首尔大学等，NeurIPS 2024)
- 类别/成熟度: technique / **research** | 许可: 见仓库 | 成本: 取决于队列长度与基座模型
- 是什么: 训练无关的推理技巧：维护一个帧队列，队列里每帧噪声等级递增，做「对角去噪」——同时处理一串连续帧，最干净的从队头出队，新噪声帧从队尾入队。理论上可无限长。
- 关键事实:
    * arXiv 2405.11473，2024-05-19
    * 完全训练无关，可套在任意预训练文生视频模型上
    * 已知硬伤：帧内/帧间一致性先验太强，导致运动幅度极小，背景近乎静止
    * 显存与队列长度相关，不与总长相关
- 长片作用: 「训练无关无限长」这条路的典型反例：它确实不崩，但代价是几乎不动。这个 trade-off 是整个领域的铁律——运动幅度和长时序稳定性是对立的。对你的判断有价值：任何声称「训练无关 + 无限长 + 不崩」的方案，先去看它 demo 里的运动幅度，静态就是作弊。
- 来源: https://jjihwan.github.io/projects/FIFO-Diffusion/

## RIFLEx  (清华 TSAIL（朱军组），ICML 2025)
- 类别/成熟度: technique / **production** | 许可: 见仓库（开放） | 成本: 零额外成本，改 RoPE 参数即可
- 是什么: 纯位置编码外推：系统分析 RoPE 各频率分量在外推时的作用，找出主导外推行为的「本征频率」(intrinsic frequency)，把该频率降下去，就能抑制时序重复而不牺牲运动一致性。改一行代码级别的改动。
- 关键事实:
    * arXiv 2502.15894，2025-02-21；ICML 2025 Poster
    * 完全训练无关可做 2× 时长外推；极少量微调可做 3×
    * 也能用于空间分辨率外推，以及时空联合外推
    * 对比 Position Extrapolation (PE)：RIFLEx 解决了重复问题，PE 在时空联合外推上直接失败
    * 已被集成进大量 ComfyUI / Wan 生态节点
- 长片作用: 这是全清单里唯一「免费、零成本、已经在生态里跑了一年、可以今天就加进产线」的技术。在你的产线里它顶「把 5 秒基座拉到 10 秒、10 秒拉到 20 秒」这个工位——把每个原子镜的可用长度翻倍，直接减少一半拼接点。2× 是训练无关的安全区，3× 要微调。不要指望它做分钟级：位置编码外推到 3× 以上必然重复或减速。
- 来源: https://arxiv.org/abs/2502.15894 https://riflex-video.github.io/ https://icml.cc/virtual/2025/poster/43698

## Wan 生态：Wan 2.2（开源天花板）/ VACE 时序扩展 / Wan-Animate-2 / Wan 2.6-2.7（闭源 API）  (阿里巴巴通义)
- 类别/成熟度: open-weights / **production** | 许可: Wan 2.2 / Wan-Animate-2 为 Apache 2.0；Wan 2.6/2.7 为闭源 API | 成本: Animate-2: 8×A800 @720P，2×A800 @480P；Wan 2.6 API 约 $0.10/秒起（720p I2V）
- 是什么: VACE 是 Wan 的全能视频创作编辑框架，用 Video Condition Unit 统一 R2V / V2V / MV2V，其中 Temporal Extension（时序扩展）支持首尾帧约束、自定义关键帧序列续写。Wan-Animate-2 是角色动画分支，把参考视频的身体动作与面部表情迁移到静态角色图。
- 关键事实:
    * 【关键核实】HF 上 Wan-AI 组织最新可下载权重仍是 Wan 2.2 系列；不存在 Wan 2.5 / 2.6 / 2.7 / 3.0 开放权重
    * Wan 2.7 于 2026-04-07 发布，API-only，2–15 秒片段，720p/1080p，无公开权重
    * VACE: arXiv 2503.07598，ICCV 2025；1.3B 与 14B 两档；时序扩展要求 length-1 可被 4 整除
    * Wan2.2-Animate-2-14B: 2026-08-07 放出推理脚本与 Base/Distill 权重，Apache 2.0
    * Animate-2 默认 8×A800 跑 720P，2×A800 可跑 480P；Distill 版 10 步（Base 40 步），达到可流式的实时推理
    * 第三方资料称 Animate-2 支持至 120 秒、720p——官方 README 未写时长上限（见 uncertainClaims）
- 长片作用: Wan 2.2 是整个开源长视频生态的事实底座——SVI 2.0 Pro、InfiniteTalk、绝大多数长时序 LoRA 都挂在它上面。VACE 的时序扩展顶「关键帧约束续写」工位，配合你的分镜系统非常自然：分镜给关键帧，VACE 补中间。Wan-Animate-2 顶「动作/表情驱动数字人」工位，是 Seedance 无法本机 LoRA 之外你能做角色一致性的主要抓手。务必注意：想要 Wan 2.7 的画质就只能走 API，本机 LoRA 路线的上限被钉死在 Wan 2.2。
- 来源: https://huggingface.co/Wan-AI https://arxiv.org/pdf/2503.07598 https://github.com/Wan-Video/Wan-Animate-2

## LTX-Video 0.9.8 / LTX-2 / LTX-2.5  (Lightricks)
- 类别/成熟度: open-weights / **production** | 许可: 0.9.8 为 OpenRail-M（可商用）；LTX-2.5 开放权重，商用条款见 Lightricks 官方（收入阈值说法待核，见 uncertainClaims） | 成本: 2B distilled 消费卡可跑；22B LTX-2.5 面向 GB200 级硬件，本地部署成本高
- 是什么: LTX-Video 0.9.8 支持前向与后向视频扩展（extension），输入片段帧数须为 8 的倍数加 1（9/17/25…）。LTX-2 换成 Gemma 4 12B 文本编码器 + 统一音视频生成 + 扩散解码器。LTX-2.5 主打推理速度与 autoregressive 支持，面向实时与机器人世界模型。
- 关键事实:
    * v0.9.8: 2025-07-16，13B dev / 13B distilled / 2B distilled 三档，支持最长 60 秒，OpenRail-M 许可
    * LTX-2: 2025-10-23 公布，2026-01 全量开源（代码+权重+工具链），4K / 最高 50fps，单次 10 秒带同步音频
    * LTX-2.5: 2026-08-11 开源权重，22B DiT，2×GB200 上 6.8 秒出 10 秒 720p
    * LTX-2.5 新增原生 multishot（跨剪辑保持角色、环境、声音一致）+ 扩散视频解码器 + 更强 autoregressive 支持
    * ComfyUI 首日支持（官方合作）
    * 2025-07 曾宣布 autoregressive 更新可连续生成约 60 秒
- 长片作用: LTX-2.5 的 native multishot（单次生成里跨剪辑保持角色/环境/声音一致）是 2026 年最贴合你需求的特性——它把「分镜拼接」的一部分工作搬进了模型内部。在你的产线里顶「多镜头连贯段落生成」工位。但注意它的单次仍是 10 秒级，multishot 是在 10 秒内切几个镜头，不是把单次拉到分钟级。0.9.8 的 60 秒是老基座的画质，不达标。
- 来源: https://github.com/Lightricks/LTX-Video https://en.wikipedia.org/wiki/LTX-2 https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights

## InfiniteTalk  (MeiGen-AI（美团系）)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: 恒定显存，约等于跑 Wan2.1 81 帧；24GB 卡量化后可跑
- 是什么: 稀疏帧视频配音框架：从音频驱动生成无限长口播视频，同时控制唇形、头部动作、身体姿态与表情。核心是 context window 机制（默认 81 帧窗口）+ 参考图做长期外观锚，通过自适应约束权重保持长程外观稳定。支持 I2V 与 V2V 两种模式。
- 关键事实:
    * 2025-08-20 开源；基座 Wan2.1；Apache 2.0
    * context window 默认 81 帧，这是实现「无限」的关键机制
    * 显存恒定（逐 chunk 生成），与总长无关
    * 明确局限：不泛化到通用场景，只在口播/配音域有效
    * 2026 年被 LongCat-Video-Avatar 1.5（2026-05-21）在指标上超过
- 长片作用: 在你的「数字人拍片」产线里，这是最成熟、最被验证、部署成本最低的长时序工位——口播段落可以直接一次出几分钟。它和 SkyReels-V3 A2V-19B（200秒）、LongCat-Avatar 1.5 构成「音频锁定运动先验 → 长时序免疫漂移」这一类的三个可选实现。但它做不了叙事运镜镜头，那部分仍然必须走原子镜拼接。
- 来源: https://github.com/MeiGen-AI/InfiniteTalk https://comfyui-wiki.com/en/news/2025-08-20-infinitetalk-audio-driven-video-generation

## NOVA  (BAAI 北京智源，ICLR 2025)
- 类别/成熟度: open-weights / **research** | 许可: 见仓库 | 成本: 0.6B，消费卡轻松跑
- 是什么: 无向量量化的自回归视频生成：时间轴上逐帧因果预测，空间上帧内 set-by-set 双向预测。保留 GPT 式因果性以获得 in-context 能力，同时靠帧内双向建模提效率。
- 关键事实:
    * arXiv 2412.14169，2024-12-18；ICLR 2025
    * 仅 0.6B 参数，在数据效率、推理速度、视觉保真度上超越当时更大的自回归视频模型
    * 论文声称在「延长时长」上泛化良好，但未给出具体秒数上限
    * 权重与代码开源：baaivision/NOVA
- 长片作用: 在 2-8 分钟产线里没有工位——0.6B 的画质离成片标准太远。它的价值是架构启发：「时间因果 + 空间双向」这个分解在 2026 年被 Helios、ARL2 等继承。列入只为闭合清单。
- 来源: https://arxiv.org/abs/2412.14169 https://github.com/baaivision/nova https://bitterdhg.github.io/NOVA_page/

## Pyramid Flow  (北大 / 快手，ICLR 2025)
- 类别/成熟度: open-weights / **research** | 许可: miniFLUX 版 Apache 2.0；SD3 版 Stability AI Community License（商用受限） | 成本: 消费卡可跑
- 是什么: 金字塔流匹配（Pyramidal Flow Matching）：在不同分辨率的金字塔层级上做流匹配，训练高效的自回归视频生成。
- 关键事实:
    * arXiv 2410.05954，2024-10-08；ICLR 2025
    * 768p checkpoint 最长 10 秒 @ 24fps
    * 仅用开源数据集训练
    * miniFLUX 变体 Apache 2.0；SD3 变体为 Stability AI Community License
    * 原生支持图生视频
- 长片作用: 最长 10 秒，且是 2024 代画质，在你的产线里没有工位。它被列进「长视频清单」是历史误置——它解决的是训练效率不是时长。明确排除。
- 来源: https://github.com/jy0205/Pyramid-Flow https://huggingface.co/rain1011/pyramid-flow-miniflux

## Causal Forcing / Causal Forcing++  (清华 thu-ml（朱军组），ICML 2026)
- 类别/成熟度: research / **research** | 许可: CC-BY 4.0 | 成本: 未公布
- 是什么: 修正自回归扩散蒸馏的初始化问题：用自回归教师做 ODE 初始化，弥合双向注意力教师与因果学生之间的架构 gap，再叠 DMD 流程。针对的是实时交互式视频生成。
- 关键事实:
    * arXiv 2602.02214，2026-02-02 提交，2026-06-01 终版；ICML 2026
    * 相对前法提升：Dynamic Degree +19.3%、VisionReward +8.7%、Instruction Following +16.7%
    * 代码开源：thu-ml/Causal-Forcing（含 Causal Forcing++）
    * CC-BY 4.0
    * 论文未给明确的最长时长数字
- 长片作用: Dynamic Degree +19.3% 是关键数字——它专门在「动态幅度」这个长时序方案普遍塌方的指标上做提升。前面提过 FIFO 的铁律（稳定 vs 运动幅度对立），Causal Forcing 是 2026 年正面攻这条铁律的工作。目前研究形态，无生产链路，但如果你要自研长镜模块，这是蒸馏环节该抄的配方。
- 来源: https://arxiv.org/abs/2602.02214 https://github.com/thu-ml/Causal-Forcing

## VideoMLA  (Virginia Tech)
- 类别/成熟度: research / **research** | 许可: 未确认 | 成本: KV cache 显存降 92.7%，这是让长视频在单卡可行的关键
- 是什么: 把 Self-Forcing 系方法的 KV cache 显存墙用 MLA（多头潜在注意力）思路打掉：用共享低秩内容潜变量 + 头间共享的解耦 3D-RoPE 位置键，替代每头独立的 K/V。
- 关键事实:
    * arXiv 2605.30351，2026-05-28
    * 每 token 每层 KV 显存从 3072 标量压到 224 标量，降 92.7%
    * 基座 Wan2.1-T2V-1.3B（30 层，hidden 1536）；只在 5 秒片段上训练
    * 60 秒生成时 VBench 总分 0.859，在对比的流式方法里最高
    * 项目页 videomla.github.io；文中未明确说明代码/权重发布
- 长片作用: 直接回应 Self-Forcing 那个「10 秒吃 129GB」的结构性瓶颈。在你的产线里暂时没有工位（研究形态、1.3B 基座），但它标定了一个重要事实：2026 年「60 秒单次生成」在技术上已经不是显存问题了，剩下的是画质与运动幅度问题。
- 来源: https://arxiv.org/html/2605.30351 https://arxiv.org/pdf/2605.30351

## LoL: Longer than Longer, Scaling Video Generation to Hour  (UCLA / 字节（Justin Cui 等，Self-Forcing++ 同一作者）)
- 类别/成熟度: research / **research** | 许可: CC BY 4.0（论文） | 成本: 恒定显存，零训练成本
- 是什么: 训练无关的轻量方法。诊断出长视频的失效模式叫 sink-collapse——生成内容反复塌回锚帧；根因是 RoPE 周期性与多头注意力的冲突。解法是 Multi-Head RoPE Jitter：给每个 head 加随机相位扰动，打破头间同质化。
- 关键事实:
    * arXiv 2601.16914，2026-01-23
    * 声称可连续生成长达 12 小时的视频
    * 完全训练无关，显存恒定
    * CC BY 4.0
    * 第三方产线评估的直接批评：「所有 demo 都是偏静态的场景，不知道它在跳舞、运动这种内容上能不能活」
    * 论文未明确基座模型，代码发布情况未确认
- 长片作用: 12 小时这个数字要用 FIFO 铁律去读：训练无关 + 无限长 + 不崩，几乎必然以运动幅度为代价，而第三方评估恰好印证了「demo 偏静态」。在你的产线里最多顶「固定机位长空镜 / 背景板」这种工位。不要因为 12 小时这个数字改变分镜策略。
- 来源: https://arxiv.org/abs/2601.16914 https://arxiv.org/pdf/2601.16914

## 2026 上半年 KV/漂移治理论文群（Forcing-KV / ARL2 / TetherCache / Towards Error-Free / OPSD-V / BAgger）  (多机构)
- 类别/成熟度: research / **research** | 许可: 论文多为 CC-BY，权重基本未放 | 成本: 不适用（无可部署制品）
- 是什么: 围绕自回归长视频的两个瓶颈（KV cache 显存、误差/属性漂移）的一批 2026 年新工作：Forcing-KV 做混合 KV cache 压缩；ARL2 用门控线性注意力做跨帧记忆替代二次注意力；TetherCache 用 gated recall + trusted alignment 稳住长片；Towards Error-Free 用 clip 间因果注意力 + 恒定 KV 缓存 + truncation-rectified flow；BAgger 用反向聚合缓解漂移。
- 关键事实:
    * Forcing-KV: arXiv 2605.09681（2026-05）
    * ARL2: arXiv 2605.16579（2026-05-15，修订 05-20）；75% 层换成混合线性注意力时 2.26× 加速、显存降 54%，质量持平且时序一致性更好；代码发布未提及
    * TetherCache: arXiv 2606.13035（2026-06-12），项目页 my4f175.github.io/TetherCache
    * Towards Error-Free Long Video Generation: arXiv 2606.22370（2026-06-21），阿里系作者，minute-level，KV 缓存恒定 + T-RFlow；未提代码/权重
    * BAgger: arXiv 2512.12080（2025-12）
    * OPSD-V: arXiv 2607.08766（2026-07）
    * 共同点：全部无生产级权重发布，全部基于 1.3B–14B 学术基座
- 长片作用: 这一群论文对你的实际意义只有一条：2026 年学术界在「分钟级不崩」上已经形成密集共识和多条可行路径，但没有一个落成可部署制品。你的产线不能等它们。它们的价值是预告——2027 年开源侧「单次 2 分钟且画质可用」大概率会成立，所以你的分镜系统应该把「单镜最大长度」设计成可配置参数，而不是硬编码 15 秒。
- 来源: https://arxiv.org/html/2605.09681v1 https://arxiv.org/pdf/2605.16579 https://arxiv.org/pdf/2606.13035

## Seedance 2.0 / 2.5（硬约束核实结果）  (字节跳动 Seed)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: Seedance 2.0 约 $0.0849/秒起；2.5 定价未公开（第三方转售约 $0.10–0.30/秒）
- 是什么: 闭源多模态音视频统一架构，接受文本、图、音频、视频输入，自带同步对白声。你的既有约束基本正确，但 2.5 已经把两个上限抬高了。
- 关键事实:
    * Seedance 2.0: 2026-04-09 上线；单次 4–15 秒；最高 720p（US 区 480p/720p）；参考上限 9 图 + 3 视频 + 3 音频 — 与你的约束完全一致，核实通过
    * Seedance 2.5: 2026-07-31 官宣，2026-08-07 开发者 API 上线；单次 4–30 秒；最高 1080p（无 4K，4K 仍需回 2.0）；参考上限抬到 30 图 + 10 视频 + 10 音频 — 你的「约 30 秒」正确，但参考上限 9/3/3 是 2.0 的数字，2.5 已是 30/10/10
    * 权重不开放、不能本机 LoRA — 核实通过，字节从未发布 Seedance 权重
    * 内容审核确实存在：提示词里点名具体影片、片厂、艺术家会被拒，用纯视觉描述同样风格则通过
    * Seedance 2.0 第三方计价约 $0.0849/秒起；2.5 无第一方价目表
    * 对照：Sora 2 单次 15 秒（Pro 25 秒）；Veo 3.1 单次 8 秒硬顶，靠 extend 链到约 148 秒
- 长片作用: 它定义了你产线的画质天花板和原子镜长度。2.5 的 30 秒 + 30 张参考图是对你架构的实质性利好：原子镜从 15 秒翻倍到 30 秒，2-8 分钟成片的拼接点从约 20-30 个降到 10-16 个；30 张参考图让角色一致性在不能本机 LoRA 的前提下有了更大操作空间。建议把主力 A-roll 迁到 2.5，需要 4K 的镜头才回落 2.0。
- 来源: https://fal.ai/seedance-2.0 https://replicate.com/bytedance/seedance-2.0 https://www.krea.ai/blog/seedance-2-5-api-access-guide-features-code-examples-for-long-form-video

### 建议
【直接回答：2026-09 的真实分界线在 20–30 秒，不在 60 秒；而 60 秒以上单次不崩，只在「运动先验被外部信号锁死」的垂类里成立】

一、三段式分界线（按你的单镜观感要求划）
- 0–30 秒：任何主流方案都稳。闭源侧 Seedance 2.5 单次 30 秒 1080p 带同步音，这是你能拿到的最高画质原子镜。这一段不需要任何长时序技术，直接调 API。
- 30–60 秒：只有专用长时序方案扛得住，且必须降画质档。开源侧可用的只有三个：SVI 2.0 Pro（480p/81帧一clip/每clip换seed，社区多人复现 1 分钟无色偏）、Helios（14B Apache 2.0，1452 帧≈60秒，原生 640×384）、LongCat-Video（13.6B MIT，720p/30fps，原生续写预训练）。它们都能跑到 60 秒不崩，但单镜观感距 Seedance 2.0 有明显代差——分辨率低一档、运动幅度受限、细节不够。
- 60 秒以上单次：通用叙事镜头，目前没有任何方案做得到「稳定且画质可用」。唯一真正成立的是音频/姿态驱动垂类——SkyReels-V3 A2V-19B 明确标 200 秒、InfiniteTalk 恒定显存理论无限、LongCat-Avatar 1.5、Wan-Animate-2。原因很干净：音频或参考视频把运动先验锁死了，误差没有自由度可累积。同一个 Skywork 团队，A2V 敢标 200 秒，R2V 只标 5 秒、V2V 只标 30 秒——这个自我打脸最说明问题。

二、有没有方案真能单次稳定出 60 秒以上且不崩？
有，但要分清三类：
1. 真能，且可生产：仅限数字人口播/配音域。SkyReels-V3 A2V-19B（200秒，权重已开）、InfiniteTalk（Apache 2.0，恒定显存，已在生产用了一年）、LongCat-Video-Avatar 1.5（MIT，8步蒸馏，10秒约1分钟出片）。这正好命中你的「AI 数字人拍片」主线。
2. 真能，但研究形态：Rolling Forcing（2分钟 Quality Drift 0.01 vs CausVid 3.35，权重在 TencentARC），Helios（分钟级，吞吐与长度无关）。缺 ComfyUI 生态、缺量化、缺 LoRA 训练链路，不能直接进成片链路。
3. 数字好看但要打折读：Self-Forcing++ 4分15秒（1.3B 基座 + demo 偏静态）、LoL 12 小时（训练无关 + demo 偏静态）、SVI 官方 20 分钟压测。FIFO-Diffusion 早就证明了铁律——训练无关的超长稳定，代价永远是运动幅度趋近于零。看到超长数字，先去看 demo 里东西动不动。

三、给你的架构建议（2-8 分钟，Seedance 2.0 级观感，靠分镜拼接）
主链路保持拼接，但把原子镜从 15 秒改成 30 秒：
- A-roll 叙事镜（有运镜、有场景变化）：Seedance 2.5，单次 4–30 秒，30 图 + 10 视频 + 10 音频参考做角色一致性。8 分钟片子拼接点从约 30 个降到 16 个左右。需要 4K 的镜头回落 Seedance 2.0。
- 长对白/口播段落：这是唯一可以打破拼接规则的地方。走 SkyReels-V3 A2V-19B（一次 200 秒）或 LongCat-Avatar 1.5。一个完整对白段落一次出完，不拼。这能显著降低你的分镜复杂度。
- B-roll 长空镜/氛围镜：SVI 2.0 Pro（480p、81帧/clip、fp16、每clip换seed，严格照社区配置，121帧必崩）。这是开源侧唯一被大量第三方独立复现的 1 分钟方案，而且是 LoRA 形态，能和你已有的 Wan 2.2 LoRA 共存。
- 动作/表情驱动补拍：Wan2.2-Animate-2-14B（Apache 2.0，2026-08-07 放权重，Distill 版 10 步）。Seedance 不能本机 LoRA，角色一致性的兜底必须放在开源侧，这是主要抓手。
- 免费提效：RIFLEx 现在就加进所有 Wan 链路，训练无关 2× 时长外推，把每个开源原子镜的可用长度直接翻倍。这是全清单里唯一零成本、零风险、今天就能上的东西。

四、两个必须踩住的工程约束
- 本机 LoRA 路线的画质上限被钉死在 Wan 2.2。HF 上 Wan-AI 组织最新可下载权重仍是 Wan 2.2 系列，不存在 Wan 2.5/2.6/2.7/3.0 开放权重，Wan 2.7 是 API-only（2026-04-07，2–15秒）。凡是告诉你 Wan 2.7 有 Apache 2.0 权重的，都是错的。你的开源侧天花板就是 Wan 2.2 + 其上的全部 LoRA 生态。
- 把「单镜最大长度」做成可配置参数，不要硬编码。2026 上半年 Forcing-KV / ARL2 / VideoMLA / TetherCache / Towards Error-Free 这批论文已经把 KV 显存墙（Self-Forcing 10秒吃 129GB）打掉了 92.7%，学术侧共识密集，2027 年开源侧「单次 2 分钟画质可用」大概率成立。届时你只改一个参数，不重构分镜系统。

五、一句话结论
2026-09，单模型超长和原子镜拼接的分界线是 30 秒——不是因为技术做不到 60 秒，而是因为能做到 60 秒的方案画质全部低一档。唯一的例外是音频驱动数字人，那里 200 秒是今天就能用的现实。所以你的系统应该是：叙事镜拼接（30秒原子镜）+ 对白段落单次长出（200秒）+ 空镜用开源长时序兜底，三条链路并行，而不是赌任何一个单模型能一次出完 2-8 分钟。

### 存疑
- Wan 2.7 以 Apache 2.0 开放 1.3B/14B 权重于 2026-04 — 无法证实且有反证。HF 上 Wan-AI 组织最新可下载权重仍为 Wan 2.2 系列（Wan2.2-Animate-2-14B, 2026-08-13 更新）。Runpod 指南明确称 Wan 2.7 为 API-only。多个高排名 SEO 站点的 Apache 2.0 说法可追溯到单一 LinkedIn 帖子，无第一方渠道佐证。建议按「Wan 2.2 是开源天花板」规划。
- Wan2.2-Animate-2 支持最长 120 秒、720p — 该数字来自第三方聚合页，官方 GitHub README 与 HF 模型卡均未写时长上限。需自行压测确认。
- LongCat-Video「720P/30fps 连续播放 5 分钟后色彩一致性仍保持初始值 95% 以上」「1 分钟视频单卡约 4 分钟计算」 — 来自中文二手博客，官方 README 与技术报告只写 minutes-long 与 within minutes，无具体分钟数与百分比。另有来源称 LongCat 可达 15 分钟，同样无一手出处。
- Seedance 2.5 定价 $0.0214/1000 tokens ≈ 720p $0.47/秒、480p $0.22/秒，含参考视频按 0.6× 计费 — 火山引擎与 BytePlus 官方文档至 2026-08 仍无 2.5 价目表，该数字全部来自第三方转售商，不可作为成本模型基准。
- SVI 官方 demo「20 分钟压测无漂移」「10 分钟 Tom&Jerry」 — 这些是作者自选样本，社区独立复现的可靠上限是 480p / 1 分钟。不要按 20 分钟做产能规划。
- Helios 支持最高 4K — 出现在 README 摘要中，但同页标准分辨率写 640×384，两者存在矛盾，4K 可能指超分后处理而非原生生成。
- LoL 声称可生成长达 12 小时连续视频 — 论文未公布基座模型与代码发布情况，且第三方评估明确指出所有 demo 均为偏静态场景，在舞蹈、运动等高动态内容上的表现完全未知。
- Self-Forcing++ 的代码发布状态 — GitHub 仓库 justincui03/Self-Forcing-Plus-Plus 存在，但 Atlas Cloud 的产线评估称「no released code / 基础设施不足以生产部署」，两者矛盾，需实际 clone 验证权重是否齐全。
- SkyReels-V3 的许可条款 — 仓库为自定义 LICENSE.txt，未能确认是否允许商用及是否有用户规模限制。A2V-19B 200 秒若要用于商业产线，必须先逐条读 LICENSE.txt。
- ARL2（2605.16579）、TetherCache（2606.13035）、Towards Error-Free（2606.22370）、VideoMLA（2605.30351）的代码/权重发布状态均未在论文中明确说明，仅有项目页链接。在验证前一律按「仅论文」对待。
- LTX-2.5 的「年收入低于 1000 万美元可免费商用」条款 — 来自第三方报道（datanorth.ai），Lightricks 官方许可文本未核实。LTX-Video 0.9.8 的 OpenRail-M 是确认的。
- 「没有任何单模型能一次性稳定生成 10 分钟高质量视频」这一判断 — 来自第三方综述（hyperstack.cloud），与本次调研的所有一手证据一致，但属于综述性断言而非可直接引用的实验结果。

### 事实核查修正
- [WRONG] SVI 2.0（2025-12-04）/ SVI 2.0 Pro（2025-12-26）基座换成 Wan 2.2 I2V-A14B（HIGH/LOW 双 LoRA）
  → 日期正确，但基座说法错误。官方 README 的 News 写的是「SVI-2.0 released for Wan 2.1 and Wan 2.2」——不是「换成」。主分支（main）基座仍是 Wan 2.1 I2V-14B，官方发布的权重文件名直接写明：vita-video-gen/svi-model → version-2.0/SVI_Wan2.1-I2V-14B_lora_v2.0.safetensors。Wan 2.2 实现放在独立的 svi_wan22 分支。SVI-2.0 Pro 在 README 中描述为基于改进的 Wan 2.1 14B https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/README.md
- [WRONG] SVI 官方 demo：SVI-Shot 20 分钟压测、Tom&Jerry 10 分钟、SVI-Talk 10 分钟
  → Tom&Jerry 的数字记错了。README 里 SVI-Tom 条目写的是「This will never drift or forget in our 20 min test」，而对外放出的 demo 视频是「8-minute crazy Tom & Jerry video made with SVI-Tom」——既不是 10 分钟。SVI-Shot 的 20 min test 与 SVI-Talk 的 10 min test 确认无误。注意 20 min 是作者自测时长，不是可复现产能指标。 https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/README.md
- [OUTDATED] SVI 训练环境 A100 80GB / CUDA 12.0 / Torch 2.5.0
  → A100 80G + CUDA 12.0 正确，Torch 版本过时。README 现写的测试环境是 PyTorch 2.8.0；torch==2.5.0 是安装脚本自动回装的旧版本，仓库同时声明兼容 torch 2.4.1。把 2.5.0 当作「官方训练环境」会误导环境搭建。 https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/README.md
- [UNVERIFIABLE] SVI 只训 LoRA，约 1k 样本即可
  → 「只训 LoRA」确认无误。但「1k 样本」不是 SVI 的通用结论：README 中该数字只出现在 Wan 2.2 Animate 的微调语境（「tuning with only 1k samples is sufficient to unlock infinite-length generation」），对 SVI-Shot / Film / Talk 等 1.0 系列，README 只有定性表述「very little training data」，无具体样本量。不要把 1k 当成跨变体的预算基线。 https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/README.md
- [UNVERIFIABLE] SVI 社区实测稳定区间：480p+81帧/clip+24fps+每clip换seed → 1 分钟无可见色偏；40 秒无色偏已被多人复现；已知崩点 121 帧/clip 必然色偏、必须 fp16、量化/蒸馏掉质量、LightX2V 开太猛会慢动作
  → 其中只有三条能在官方 README 找到一手依据：(a)「Use different seeds for different clips, which is very important!」(b) 建议 480p 为最优分辨率 (c) 建议少用 LightX2V。其余全部无一手来源：81 帧/clip、24fps、「1 分钟无可见色偏」「40 秒被多人复现」「121 帧必然色偏」「必须 fp16」「量化掉质量」均未出现在 README、HF 模型卡或论文中，也未找到可追溯的复现帖。这类数字不应写进技术选型文档，至少要标注为未验证的社区口径。 https://raw.githubusercontent.com/vita-epfl/Stable-Video-Infinity/main/README.md
- [WRONG] Helios 权重 2026-07-28 仍在更新
  → 措辞把 README 改动说成了权重更新。HF BestWishYsh/Helios-Base 的 commit 历史显示：2026-07-28 的最后一次提交是「Update README.md」；权重（safetensors）批量上传集中在 2026-02 下旬（初始 commit cb75bfd，2026-02-23），3 月主要是 README 更新与 2026-03-15 的 modular 功能。所以权重自 2026-02/03 起基本冻结，不是「7 月仍在更新」。 https://huggingface.co/BestWishYsh/Helios-Base/commits/main
- [UNVERIFIABLE] Helios 第三方 H200 实测吞吐几乎与长度无关：240 帧 24s / 480 帧 42s / 960 帧 82s（约 11 FPS 恒定）
  → 找不到任何一手来源。HF 模型页 discussions 区只有一条 2026 年初的「gguf version」请求，无 H200 benchmark 讨论；官方 README 只给 H100 19.5 FPS 与昇腾 NPU ~10 FPS（这两项已核实为 CONFIRMED）。这组 240/480/960 帧的数字无法追溯到具体作者或复现脚本，不能作为「吞吐与长度无关」的论据——恰恰这是该模型最关键的卖点，需要自己压测。 https://huggingface.co/BestWishYsh/Helios-Base
- [WRONG] Helios 支持最高 4K（研究员已自标存疑）
  → 存疑判断成立，可以定性了：HF 官方模型卡的标准输出分辨率明确是 640×384，4K 的说法只出现在第三方/社区的「消费级 PC 也能跑」类文档里，属于第三方实现而非官方规格。按 640×384 原生 + 外接超分规划，不要按原生 4K 规划。 https://huggingface.co/BestWishYsh/Helios-Base
- [UNVERIFIABLE] LongCat-Video-Avatar 1.5：50 步 → 8 步蒸馏，约 15× 提速，10 秒视频约 1 分钟出片；跳帧率 0.8%，唇音误差率 29.8%（对比组内最低）；帧级 GRPO 对齐；共享基座 + 多 LoRA 适配器降低显存
  → HF meituan-longcat/LongCat-Video-Avatar-1.5 模型卡（最后更新 2026-06-04）只确认三件事：DMD2-based step distillation 的「8-Step Inference」、支持 480P 与 720P、MIT License。卡上没有 50→8 的原步数、没有 15× 提速倍数、没有 10 秒/1 分钟的出片时间、没有 0.8% 跳帧率与 29.8% 唇音误差率、没有 GRPO、也没有共享基座+多 LoRA 的表述——这些只被引到一份未公开链接的 Technical Report PDF https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5
- [UNVERIFIABLE] LongCat-Video-Avatar 权重 MIT，GitHub + HF + ModelScope 三处同步
  → MIT 对 LongCat-Video 主仓与 Avatar-1.5 模型卡都已确认。但「GitHub 三处同步」对 Avatar 不成立：github.com/meituan-longcat/LongCat-Video-Avatar 与 .../LongCat-Video-Avatar-1.5 均返回 404，Avatar 的发布信息只以 News 条目形式挂在 LongCat-Video 主仓 README 里，独立代码仓未找到。若产线依赖 Avatar 的训练/推理代码，需先确认代码到底在哪。 https://github.com/meituan-longcat/LongCat-Video
- [WRONG] SkyReels-V3 的 --low_vram 走 FP8 量化；官方未公布具体 VRAM 阈值
  → 前半句对，后半句错。官方 README 原文就写了阈值：「For GPUs with lower VRAM (e.g., under 24GB), use these options:」，随后才是 --low_vram（FP8 weight-only quantization + block offload）与降分辨率到 540P/480P。24GB 是官方给出的门限，不是未公布。 https://github.com/SkyworkAI/SkyReels-V3/blob/main/README.md
- [WRONG] SkyReels-V3 明确的时长上限：R2V 5 秒 / V2V 续写 30 秒 / A2V 数字人 200 秒
  → V2V 一档被合并成了单一数字，丢了关键约束。README 把视频扩展拆成两种模式：Single-shot Extension 为 5–30 秒，Shot Switching Extension（切镜头续写）上限只有 5 秒。另外 A2V 的 200 秒是「支持最长 200 秒音频输入」，是输入侧口径。做分镜级长片续写时，真正受限的是切镜头那条 5 秒路径。 https://raw.githubusercontent.com/SkyworkAI/SkyReels-V3/main/README.md
- [WRONG] SkyReels-V2 Diffusion Forcing：arXiv 2504.13074，2025-04-18
  → arXiv ID 正确（SkyReels-V2: Infinite-length Film Generative Model），但日期差一天：v1 提交于 2025-04-17，最后修订 v3 为 2025-04-21。 https://arxiv.org/abs/2504.13074
- [WRONG] SkyReels-V2 --addnoise_condition 建议 20–50，用来压跨段不一致
  → 把「推荐值 + 硬上限」误读成了推荐区间。README 的表述是长视频生成推荐设为 20，并警告不要超过 50（超过会牺牲一致性）。写成「建议 20–50」会让人默认取中值 35，而官方推荐点就是 20。 https://raw.githubusercontent.com/SkyworkAI/SkyReels-V2/main/README.md
- [UNVERIFIABLE] SkyReels-V2 540P: 97–737+ 帧（约 4s–30s+）；720P: 121–1457+ 帧（约 5s–60s+），24fps
  → 下界与分辨率确认：540P=544×960 基线 97 帧，720P=720×1280 基线 121 帧，24fps 正确。但 737 / 1457 这两个上界数字在我核到的 README 文本中未出现，无法定位其一手出处（可能来自某版 DF 长视频示例脚本）。峰值显存那条反而是对的：1.3B@540P ≈14.7GB、14B@540P ≈43.4GB（T2V/I2V）到 ≈51.2GB（DF），VBench 总分 83.9%、SkyReels-Bench T2V 3.14 / I2V 3.29 亦全部确认。 https://raw.githubusercontent.com/SkyworkAI/SkyReels-V2/main/README.md

### 遗漏补充
- SkyReels-V3 技术报告 arXiv 2601.17323（2026-01-24 提交，2026-01-29 修订，22 位作者）——调研员给了发布日期和三个变体却完全没给 arXiv ID，这是该模型唯一的一手技术文档。顺带：其 LICENSE.txt 已核实为 Skywork Community License（中英双语），正文明确允许商用、可见文本中无 MAU/用户规模门槛，但完整条款在单独链接的 PDF 中，商用前仍需读那份 PDF。
- FramePack / FramePack-F1（lllyasviel）——恒定上下文长度的帧打包方案，把长视频生成的计算与长度解耦，13B 基座在 6GB 显存笔记本上可跑分钟级生成。这是「超长生成」维度里最主要的低门槛开源路线，调研完全没覆盖。
- MAGI-1（Sand AI）——24B autoregressive chunk-by-chunk 视频扩散，Apache 2.0 开源权重，原生按 chunk 流式续写，是 SVI / Helios 之外最直接的自回归长视频对照组。
- CausVid / Self-Forcing 原始工作——调研只提了 Self-Forcing++，但因果蒸馏 + KV cache 的实时自回归长视频这条线的源头是 CausVid 与 Self-Forcing，缺了源头无法判断 ++ 的增量。
- RIFLEx 与 Ouroboros-Diffusion——位置编码外推 / 无训练长度外推路线，零训练成本把现有 Wan、HunyuanVideo 拉长，是「不换基座先试一把」的最低成本选项，与 LoRA 微调路线互补。
- StreamingT2V / FIFO-Diffusion——免训练的滑窗长视频基线。做色偏与漂移评测时它们是必需的 baseline，否则无法量化 SVI 的 error-recycling 到底贡献了多少。
- HunyuanVideo-I2V / HunyuanVideo 1.5 的分段续写实践——腾讯这条开源线在国内算力上的部署材料比 Wan 系更多，调研里完全没有出现。
- LongCat-Video 的原生 Video-Continuation 预训练任务本身——调研把 LongCat 当成「13.6B + MIT」的一个条目，但它区别于其他模型的核心正是把视频续写作为预训练任务之一（而非后接的外推技巧），这一点在做长片方案选型时是决定性的架构差异。
- 评测侧缺口：VBench-Long / VBench-2.0 与 LOVE-Bench 这类专门面向长视频漂移、身份一致性、色彩稳定性的基准，调研引用的 VBench1.0 83.9% 是短视频口径，不能用来论证分钟级稳定性。


====================================================================================================
# [lora-training-stack] 开源视频模型 LoRA / 全参微调工程栈 + 云 GPU 选型（截至 2026-09-19 实检）

## musubi-tuner (kohya-ss)  (kohya-ss)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0（hunyuan_model 目录沿用 HunyuanVideo 许可） | 成本: 16GB 可跑小分辨率角色 LoRA；24GB 舒适；80GB 才能上 720p + 长帧数
- 是什么: kohya 系的视频/图像扩散模型训练器，目前社区训 Wan LoRA 的事实标准。基于 HF Accelerate，用 TOML 描述数据集，先 cache latents + text-encoder 输出再训练。
- 关键事实:
    * 支持视频架构：HunyuanVideo、HunyuanVideo 1.5、Wan2.1/2.2、FramePack、MiniMax-H3、HiDream-O1；图像侧 FLUX.1 Kontext/FLUX.2、Qwen-Image、Z-Image、Kandinsky 5、Krea 2、Ideogram 4
    * 官方显存口径：图像训练 ≥12GB，视频训练 ≥24GB；12GB 卡需 960x544 以下 + --blocks_to_swap + --fp8_llm；建议系统内存 ≥64GB
    * Wan2.2 双模型训练：--dit 指低噪模型、--dit_high_noise 指高噪模型，--timestep_boundary 默认 I2V 0.9 / T2V 0.875；推荐 timestep 区间 I2V 低噪 0-900 / 高噪 900-1000，T2V 低噪 0-875 / 高噪 875-1000
    * 显存开关：--blocks_to_swap、--fp8 / --fp8_scaled、--gradient_checkpointing、--offload_inactive_dit
    * 多卡：仅 DDP（accelerate），官方未确认支持 FSDP/DeepSpeed，因此全参微调在多卡上基本不可行——每张卡仍需装下整个模型
    * 数据集 TOML：resolution 默认 [960,544]；enable_bucket / bucket_no_upscale 做分辨率桶；target_frames 必须满足 N*4+1（1,5,9,13,25,45,65…）；frame_extraction 五种 head/chunk/slide/uniform/full；frame_stride、frame_sample、max_frames 默认 129、source_fps 必须写小数（30.0）
    * caption 两种：同名 .txt（caption_extension）或 JSONL（image_jsonl_file / video_jsonl_file，字段 image_path/caption），JSONL 必须显式给 cache_directory
    * 2026-09-16 更新：MiniMax-H3 音视频联合训练、Krea 2 ConvRot int8、JSONL 增加 audio path
    * 社区实测：RTX 4070 Ti Super 16GB、rank16/alpha16、1600 步、lr 3e-5，高噪+低噪两个模型背靠背约 12 小时；分辨率组合 360x360x65 / 512x512x33 / 640x640x21
- 长片作用: 长片产线的「角色一致性工位」。2-8 分钟片子靠分镜拼接，最大的崩点是跨镜头角色漂移——musubi 训出的 Wan2.2 角色 LoRA 是本地唯一能把同一张脸锁死在几十个分镜里的手段。注意 A14B 是 MoE 双模型，必须训两个 LoRA（高噪管运动与构图、低噪管细节与脸），推理时两段都要挂，否则脸在前半段会飘。
- 来源: https://github.com/kohya-ss/musubi-tuner https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md

## diffusion-pipe (tdrussell)  (tdrussell)
- 类别/成熟度: framework / **production** | 许可: 未在检索中确认（仓库为公开 MIT/Apache 系，需以仓库 LICENSE 为准） | 成本: 2×24GB 起可做 14B 级视频 LoRA；Wan2.2 全参微调现实门槛 8×80GB
- 是什么: 基于 DeepSpeed 的流水线并行训练脚本，唯一在消费级多卡上能把大视频模型切开训的开源方案。
- 关键事实:
    * 视频模型支持矩阵：LTX-Video(LoRA)、HunyuanVideo(LoRA+fp8)、Wan2.1(LoRA+全参+fp8)、Wan2.2(LoRA+全参+fp8，高/低噪分开)、HunyuanVideo-1.5(LoRA+全参)、Cosmos(仅 LoRA，官方注明「不适合消费级硬件微调」)、LTX 2.3(LoRA，24GB 需 blocks_to_swap=46)、MiniMax H3(LoRA+全参，支持在量化模型上直接训)
    * 混合并行：pipeline_stages 配流水线并行，数据并行自动铺满剩余卡（4 卡 + pipeline_stages=2 = 两份模型各跨两卡）
    * 启动方式 deepspeed --num_gpus=N train.py --deepspeed --config xxx.toml；RTX 40 系必须设 NCCL_P2P_DISABLE=1
    * HunyuanVideo 无 block swap 需 48GB 或 2×24GB 流水线并行；Flux 2 Dev LoRA 需 ≥48GB
    * 数据集格式最简单：媒体文件 + 同名 .txt caption（image1.png / image1.txt）；支持 Pillow 全图像格式与 ImageIO 视频格式，不支持 WebP 视频
    * 预缓存三开关：--cache_only、--regenerate_cache、--trust_cache
    * 环境要求 Python 3.12 + PyTorch ≥2.9.0 + nvcc；Windows 原生不支持（DeepSpeed），需 WSL2
    * 2026 新增：MiniMax H3 + CFG 增强训练、Krea 2、Ideogram4、量化 ComfyUI 模型上直接训 LoRA、LTX 2.3
- 长片作用: 「多卡/全参工位」。musubi 训不动全参，diffusion-pipe 是唯一能把 Wan2.2 全参微调铺到 8×H100 的开源栈。如果你要的是一整套自有画风/自有数字人形象体系（而不是单个角色 LoRA），这是唯一路径。日常角色 LoRA 用它反而不划算——配置更重、生态教程比 musubi 少。
- 来源: https://github.com/tdrussell/diffusion-pipe https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md https://github.com/tdrussell/diffusion-pipe/blob/main/README.md

## DiffSynth-Studio (ModelScope / 阿里达摩)  (ModelScope (Alibaba))
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: 文档未给统一显存表；ZeRO-3 + gradient checkpointing offload 下 Wan2.2 A14B LoRA 可压到单卡 24-48GB 区间（需实测）
- 是什么: 阿里自家的 Wan 全家桶训练+推理框架，覆盖面最广、和 Wan 官方权重贴得最近。
- 关键事实:
    * Wan 训练覆盖最全：Wan2.1 T2V(1.3B/14B)/I2V/FLF2V/VACE；Wan2.2 T2V/I2V/TI2V/Animate/S2V/VACE-Fun/Fun；外加 Wan-Dancer、MOVA、LongCat、Video-As-Prompt
    * 统一入口 examples/wanvideo/model_training/train.py，每个模型配 full 训练 .sh 与 lora 训练 .sh
    * 多卡：DeepSpeed ZeRO Stage 3 分片（accelerate_config_zero3.yaml + --initialize_model_on_cpu）；推理侧用 xfuser + flash_attn 的 unified sequence parallel
    * 可训练模块可选 dit / vae / text_encoder；--use_gradient_checkpointing_offload 把激活换到 CPU
    * 支持动态分辨率（height/width 留空）、LoRA rank / target_modules / 预置 LoRA 差分训练
- 长片作用: 「Animate / S2V 数字人工位」。这是唯一同时能训 Wan2.2-Animate 和 Wan2.2-S2V 的开源栈。数字人拍片真正的刚需不是 T2V，是「给一张角色图 + 一段驱动视频/一段语音 → 出对口型的镜头」，Animate/S2V 就是干这个的，DiffSynth 是它们唯一的官方训练路径。缺点：文档偏散，显存/吞吐数字要自己 benchmark。
- 来源: https://github.com/modelscope/DiffSynth-Studio https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/lora/Wan2.2-I2V-A14B.sh

## ai-toolkit (ostris)  (ostris)
- 类别/成熟度: framework / **production** | 许可: MIT | 成本: 24GB 消费卡（4090/5090）即可训 Wan2.2 14B LoRA；官方文档未明确 DeepSpeed/FSDP 多卡分片能力
- 是什么: 带 Web UI 的训练套件，上手成本最低，MIT 许可，2026 年是最流行的 LoRA trainer 之一。
- 关键事实:
    * 视频模型：Wan 2.1(1.3B/14B)、Wan 2.2(14B A14B / 5B TI2V)、LTX-2、LTX-2.3、MiniMax-H3
    * Wan 2.2 T2V/I2V 14B LoRA 从 24GB 消费卡一路支持到 H100/H200；提供 4-bit ARA 量化与高/低噪分阶段训练
    * Wan 2.2 T2I 支持于 2025-08-16 加入（ostris 官方公告），I2V 14B 教程 2025-08-21 发布
    * 数据集：图片目录 + 同名 .txt caption，仅收 jpg/jpeg/png；caption 内 [trigger] 占位符自动替换为触发词
    * 配置示例按显存分档（如 train_lora_flux_24gb.yaml）
    * 2026 新增实验性 AI Toolkit Manager：自动探测硬件并装对应 PyTorch 构建
- 长片作用: 「快速出第一版角色 LoRA 的工位」。有 Web UI，非工程同事也能挂数据跑训练，适合做角色形象的快速试错迭代（一个角色试 5 个 caption 策略）。但它偏图像路线（大量 Wan-as-T2I 用法），要训「运动风格 LoRA」还是 musubi / diffusion-pipe 更稳。多卡能力是它的短板。
- 来源: https://github.com/ostris/ai-toolkit https://x.com/ostrisai/status/1956819166830199215 https://x.com/ostrisai/status/1958228670583316973

## SimpleTuner (bghira)  (bghira)
- 类别/成熟度: framework / **production** | 许可: AGPL-3.0（重要：闭源商用产线要评估传染性） | 成本: 12-16GB 起（小模型），Wan 14B LoRA 24GB 起
- 是什么: 通用图像/视频/音频扩散微调套件，是这批里唯一明确同时给出 DeepSpeed 和 FSDP2 的。
- 关键事实:
    * 视频模型与参数量：Wan Video 1.3B-14B(Apache-2.0)、LTX Video ~2.5B(Apache-2.0)、LTX Video 2 19B(Apache-2.0)、Hunyuan Video 8.3B(AGPL-3.0)、LongCat Video 13.6B(MIT)、Kandinsky 5.0 Video 2B-19B(MIT)
    * Wan 2.x I2V 支持高/低噪 stage preset，外加 2.1 time-embedding fallback
    * 显存：Wan2.1 1.3B rank-16 LoRA batch 4 略超 12GB，现实下限 16GB / 单张 3090；Wan2.1 14B「塞得进 24GB 但要调参」，理想是多张 4090/A6000/L40S；Wan2.2 I2V 现实下限 16GB
    * 分档口径：12B+ 大模型全参需 A100-80G，LoRA ≥24G；2B-8B LoRA ≥16G、全参 ≥40G；<2B ≥12G
    * 多卡：DeepSpeed + FSDP2 集成，支持 optimizer state offload 与 gradient checkpointing
    * 数据集：dataloader 配置，caption_strategy="textfile"；桶策略 aspect_ratio（默认）或 resolution_frames；Wan2.1 示例 832x480、75 帧 = 5 秒
    * 量化：INT8-Quanto 可用（Apple/NVIDIA 已测），不能与 NF4/INT4 同用
    * 2026-01 在做 LTX-2 audio-only 训练；PyPI 最新见 simpletuner 4.9.x
- 长片作用: 「多模型横跳 + 合规要留意的工位」。它的价值是一套配置覆盖 Wan / LTX-2 / LongCat（13.6B MIT，长视频向）三条线，方便你在「哪个开源底模单镜观感更接近 Seedance 2.0」上做 A/B。但 AGPL-3.0 对要做闭源 SaaS 平台的项目是真实法务风险——只用它产出权重一般没问题，把它的代码嵌进你的服务就要小心。
- 来源: https://github.com/bghira/SimpleTuner https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/WAN.md https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/LTXVIDEO.md

## finetrainers (Hugging Face)  (Hugging Face)
- 类别/成熟度: framework / **usable** | 许可: Apache-2.0 | 成本: LTX LoRA 5GB / Hunyuan LoRA 32GB；Wan2.2 不支持
- 是什么: HF 官方的视频模型训练库，架构干净（DDP/FSDP-2/HSDP/CP），但视频线已实质停更。
- 关键事实:
    * 视频模型：LTX-Video、HunyuanVideo、CogVideoX-5b、Wan（文档中 Wan 仅列到 2.1：T2V-1.3B/14B-Diffusers、I2V-14B 480P/720P、FLF2V-14B-720P），Wan 2.2 无任何文档
    * 训练类型：--training_type lora 或 --training_type full-finetune
    * 显存表：LTX-Video LoRA 最低 5GB、CogVideoX-5b LoRA 18GB、HunyuanVideo LoRA 32GB；全参 21-53GB
    * 并行后端：DDP、FSDP-2、HSDP、CP（Context Parallel，2025-05-13 加入）
    * 维护状态：2026 年内主分支只有 bot/chore 提交（2026-04-08 pin actions、2026-09-17 Dependabot 配置），最后一次实质功能提交在 2025 年中；1.4k star、62 open issues、无归档标记
- 长片作用: 在你这条产线上基本不占工位。它唯一的相对优势是 Context Parallel（训长序列/长帧数），但既然 Wan 2.2 都没接，就别指望它接 Wan2.2-Animate。列在这里是为了明确「不要在它上面投工程时间」——选它等于自己写 Wan2.2 的 MoE 双模型适配。
- 来源: https://github.com/huggingface/finetrainers https://github.com/huggingface/finetrainers/blob/main/docs/models/wan.md https://github.com/huggingface/finetrainers/commits/main

## OneTrainer (Nerogar)  (Nerogar)
- 类别/成熟度: framework / **usable** | 许可: AGPL-3.0 | 成本: 未公布视频训练显存基线
- 是什么: GUI 优先的一站式扩散训练器，图像线很强，视频线只到 HunyuanVideo。
- 关键事实:
    * 支持模型：Ernie Image、Z-Image、Qwen Image、FLUX.1、Flux.2 Dev/Klein、Chroma、SD1.5/2.x/3.0/3.5、SDXL、Würstchen-v2、Stable Cascade、PixArt-Alpha/Sigma、Sana、Hunyuan Video + inpainting
    * 视频模型：仅 Hunyuan Video，README 中无 Wan 任何版本
    * EMA 权重可放 CPU 内存降显存；README 未给显存基线表，也未说明多卡方案
- 长片作用: 对 Wan 路线无用。如果产线里有「静帧概念图 / 关键帧美术」的图像 LoRA 需求（比如先训一套角色三视图 LoRA 再喂 I2V），OneTrainer 的 GUI 值得给美术用；视频工位请忽略它。
- 来源: https://github.com/Nerogar/OneTrainer

## Wan 官方仓库（Wan-Video/Wan2.2）  (Alibaba Tongyi Wanxiang)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0 | 成本: A14B 推理单卡 80GB；TI2V-5B 24GB
- 是什么: Wan2.2 官方开源仓库——只有推理，没有训练。这是最常被误解的一点。
- 关键事实:
    * 官方仓库不含任何训练/微调脚本，只有 generate.py 推理
    * 三个开源模型：T2V-A14B（MoE，27B 总参 / 14B 激活）、I2V-A14B（同上）、TI2V-5B（dense）
    * 分辨率：A14B 支持 480P + 720P；TI2V-5B 仅 720P
    * 显存：TI2V-5B 可跑 4090（≥24GB）；A14B 单卡推理需 ≥80GB
    * 多卡推理：FSDP + DeepSpeed Ulysses
    * 仓库最后更新 2026-03-17，18k star；README 最新 news 停在 2025-11-13（Wan2.2-Animate-14B 接入 Diffusers）
    * Wan-Video org 下共 6 个仓库，无 Wan2.5 / 2.6 / 2.7 / 3.0 任何一个
- 长片作用: 这是你整条本地产线的底模。硬结论：Wan2.2（2025-07 开源，Apache-2.0）到今天仍是阿里最后一个开权重的视频旗舰，Wan2.5/2.6/2.7 全是百炼闭源 API，只能云端调用。所以「本机 LoRA + 自有角色」这条路只能建在 Wan2.2 上。训练必须用第三方栈（musubi / diffusion-pipe / DiffSynth），官方不给。
- 来源: https://github.com/Wan-Video/Wan2.2 https://github.com/orgs/Wan-Video/repositories https://huggingface.co/Wan-AI

## Wan-Animate-2 / Wan2.2-Animate-2-14B  (Alibaba Tongyi Wanxiang)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: 2×A800 (480P) / 8×A800 (720P)；按 RunPod H100 Community $2.69/h 折算，8 卡 720P 推理约 $21.5/小时
- 是什么: 开源角色动画/角色替换模型：参考图 + 驱动视频 → 保身份的动画视频，且支持文字控制视角。
- 关键事实:
    * 发布 2026-08-07，仓库 2026-08-08 更新，319 star
    * 14B 参数 DiT；直接吃驱动视频，去掉中间动作提取器（作者称因此 identity preservation 更强）
    * 默认 720P；480P 已在缩减硬件上验证
    * 硬件：720P 面向 8×A800；480P 可 2×A800
    * 有蒸馏 Lite 版，作者称推理延迟降到「实时门槛」，用于流式角色动画
    * 权重：huggingface.co/Wan-AI/Wan2.2-Animate-2-14B
    * 文档未提及长视频/超长序列生成
- 长片作用: 数字人长片里最实用的一块开源拼图：它把「表演」和「形象」解耦——你只要拍/合成一段驱动视频（真人替身或 3D 预演），Animate-2 负责换成你的虚拟角色，身份漂移问题从「靠 LoRA 碰运气」变成「靠参考图锁定」。对 2-8 分钟片子，这比纯 T2V 拼分镜可控得多。代价是 8×A800 级别的推理成本，且 Lite 版实时性宣称未经第三方复现。
- 来源: https://github.com/Wan-Video/Wan-Animate-2 https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B

## 阿里百炼 万相视频模型 LoRA 定制化训练 API  (阿里云 Model Studio（百炼）)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API，权重不出云 | 成本: 零本地显存；wan2.2-i2v-flash 训练单价是 wan2.7 的 1/33，是做风格/特效 LoRA 最便宜的官方入口
- 是什么: 真实存在、文档齐全、可自助提交的官方 LoRA 微调服务。产出的 LoRA 不能下载，只能在百炼上部署成私有端点调用。
- 关键事实:
    * 支持微调的基础模型（首帧 I2V）：wan2.7-i2v、wan2.6-i2v、wan2.5-i2v-preview、wan2.2-i2v-flash；首尾帧模式：wan2.7-i2v、wan2.2-kf2v-flash。全部为 SFT-LoRA（efficient_sft）
    * 数据量：最少 10 条，推荐 20-100 条视频
    * 视频时长上限随模型不同：wan2.2 系 2-5 秒；wan2.5 / wan2.7 系 2-10 秒
    * 素材规格：图片 BMP/JPEG/PNG/WEBP，视频 MP4/MOV，均 ≤4096×4096；zip 包 ≤1GB；data.jsonl ≤20MB
    * data.jsonl 字段：{"prompt": ..., "first_frame_path": "image_1.jpg", "video_path": "video_1.mp4"}；首尾帧模式加 "last_frame_path"
    * prompt 结构要求：主体描述 + 环境描述 + 触发词 + 动作描述；官方建议用无意义罕见词做触发词（文档示例 s86b5p）
    * 默认超参：n_epochs=50、learning_rate=2e-5、lora_rank=32、lora_alpha=32、eval_epochs=20、split=0.9；batch_size wan2.7=1 / wan2.5=4 / wan2.2 系=4；max_pixels wan2.7=102400 / wan2.5=36864 / wan2.2 系=262144
    * 步数公式 steps = n_epochs × ⌈数据集大小 / batch_size⌉，官方建议总步数 ≥800
    * 训练价格（按训练 Token，元/千 Token）：wan2.7-i2v ¥2.0、wan2.5-i2v-preview ¥0.32、wan2.2-i2v-flash ¥0.06。官方计费示例：一条 10 秒视频、wan2.7-i2v、max_pixels=36864、n_epochs=800，预估 ¥576
    * 训练耗时官方表述为「数小时」；部署耗时 5-10 分钟；部署免费；推理按所微调基础模型的标准调用价计费
    * 产出模型不可下载到本地，只能部署为百炼在线服务（状态需为 RUNNING 才能调用）；微调模型命名形如 wan2.5-i2v-preview-ft-[timestamp]
    * 文本微调 API 端点（国际站新加坡）POST https://dashscope-intl.aliyuncs.com/api/v1/fine-tunes；文本微调功能标注仅华北2（北京）可用
- 长片作用: 占「高画质成片渲染工位」，但它锁住你。真实用途：把某个特效/转场/运镜风格固化成 LoRA，然后用 wan2.6/2.7 的 15 秒 1080p + 原生音频出高质量分镜——这比 Wan2.2 开源版单镜观感强得多，更接近 Seedance 2.0 的档位。致命约束有三个：(1) LoRA 不能下载，你在百炼上训的角色资产永远搬不走，平台化后是单点依赖；(2) 训练素材上限 2-10 秒，学不到长镜头节奏；(3) 走官方通道就有内容审核。建议把它当「渲染后端之一」而不是资产库，角色身份资产仍要在本地 Wan2.2 上留一份可导出的 LoRA。
- 来源: https://help.aliyun.com/zh/model-studio/wan-video-generation-finetune-guide https://www.alibabacloud.com/help/zh/model-studio/wan-video-generation-finetune-guide https://help.aliyun.com/zh/model-studio/model-training-and-deployment-billing

## fal.ai Wan-2.2 LoRA Trainer（托管训练）  (fal.ai)
- 类别/成熟度: service / **production** | 许可: 服务条款下商用；产出权重可下载 | 成本: 2000 步 ≈ $8；A14B 高噪+低噪各 2000 步 ≈ $16
- 是什么: 按步计费的托管 Wan2.2 LoRA 训练端点，输出可下载的 LoRA 权重 URL——和百炼相反，产物是你的。
- 关键事实:
    * fal-ai/wan-22-trainer/t2v-a14b：$0.004/步，最低计 100 步；$4 = 1000 步
    * fal-ai/wan-22-image-trainer（T2I 方向，练主体/风格）：$0.0045/步，$4.5 = 1000 步
    * 输出为 LoRA 权重文件 URL + config 文件，可下载后本地/自有推理端点使用
    * 同系列还有 fal-ai/wan-trainer/t2v（Wan2.1）
- 长片作用: 角色 LoRA 的「无运维快车道」。对比自建：一次 2000 步 ×2 模型在 fal 是 ~$16 且零配置，在 RunPod H100 自己跑是 ~$10-25 但要花半天搭环境、调 blocks_to_swap、处理 OOM。第一个角色建议直接上 fal 验证数据集质量，等你要跑 20+ 个角色、要控 caption 策略、要训 Animate/S2V 时再自建——托管端点不会给你 Wan2.2-Animate 的训练入口。
- 来源: https://fal.ai/models/fal-ai/wan-22-trainer/t2v-a14b/api https://fal.ai/models/fal-ai/wan-22-image-trainer https://fal.ai/models/fal-ai/wan-trainer/t2v

## RunPod  (RunPod)
- 类别/成熟度: service / **production** | 许可: — | 成本: 训 Wan2.2 A14B LoRA 推荐 H100 SXM Community $2.69/h；预算档 RTX 5090 $0.69/h
- 是什么: 容器化 GPU 出租，Community Cloud（散户机器，便宜）/ Secure Cloud（托管机房）双轨。训 LoRA 的默认选择。
- 关键事实:
    * 2026-09 官方价（Community / Secure，$/GPU/h）：H100 PCIe 1.99 / 2.89；H100 SXM 2.69 / 3.49；H100 NVL 2.59 / 3.19；H200 3.59 / 4.59；B200 5.98 / 6.79
    * A100 80GB PCIe 1.19 / 1.59；A100 80GB SXM 1.39 / 1.59
    * RTX 4090 0.34 / 0.74；RTX 5090 0.69 / 0.99；RTX 6000 Ada 0.74 / 0.84；RTX Pro 6000 1.69 / 2.09；L40S 0.79 / 1.09
    * 定价页未列 spot/可抢占价
    * 第三方索引交叉核对（2026-09-19）：RunPod H100 NVL $2.59、A100 SXM $1.00、RTX 4090 $0.34、RTX 5090 $0.69、H200 $3.59
- 长片作用: 训练工位首选。理由：按秒计费、镜像生态成熟（musubi/diffusion-pipe 一键模板多）、Network Volume 能把 cache 的 latents 留下来跨次复用——视频训练 cache latents 很贵，这一点直接省掉每次重跑的 1-2 小时。批量出片则不建议用 Community（节点会掉）。
- 来源: https://www.runpod.io/pricing https://getdeploying.com/gpus/nvidia-h100

## Vast.ai  (Vast.ai)
- 类别/成熟度: service / **production** | 许可: — | 成本: 同规格通常比 RunPod 便宜 30-50%
- 是什么: 开放算力市场，价格由宿主竞价决定，最便宜但可靠性参差。
- 关键事实:
    * 2026-09-19 实价：H100 SXM on-demand $1.74/GPU/h（全网第二便宜，Lium $1.25 之后）；H200 on-demand $3.60、spot 低至 $0.02
    * A100 80GB：on-demand $0.33（4×SXM 机型）、reserved $0.55（12 月）、spot $0.18——全网最低且标注有货
    * RTX 4090 $0.42/h；RTX 5090 $0.44/h（全网第二便宜，HyperAI $0.35 之后）
    * 官方口径：Interruptible 比 On-Demand 便宜 50%+；Reserved 最多再省 50%；40+ 数据中心供给
    * 全网 H100 on-demand 中位价 $3.38（39 家）、H200 中位 $4.44（36 家）、A100 80GB 中位 $1.86（43 家）、RTX 4090 中位 $0.48（19 家，90 天跌 11%）、RTX 5090 中位 $0.68（22 家）
- 长片作用: 成本压缩工位。适合：(1) 长时间的 cache latents 与数据预处理（纯 CPU/IO 密集，掉机重来损失小）；(2) 批量出片的可中断渲染队列——2-8 分钟片子按 5 秒一镜算有 24-96 个镜头，天然适合做成幂等任务队列丢 interruptible 上跑，掉了重跑单镜。不适合：一次连跑 12 小时不存 checkpoint 的训练任务。用它必须先把 checkpoint 存频调高并挂外部存储。
- 来源: https://vast.ai/pricing https://getdeploying.com/gpus/nvidia-h100 https://getdeploying.com/gpus/nvidia-a100

## Lambda Labs（Lambda）  (Lambda)
- 类别/成熟度: service / **production** | 许可: — | 成本: H100 SXM $3.99-4.29/h，比 RunPod Community 贵约 50%
- 是什么: 面向 AI 的老牌 GPU 云，价格透明但明显高于市场，库存紧张。
- 关键事实:
    * 2026-09 官方按需价（$/GPU/h）：B200 SXM6 8× $6.69 / 4× $6.79 / 2× $6.89 / 1× $6.99
    * H100 SXM 8× $3.99 / 4× $4.09 / 2× $4.19 / 1× $4.29
    * A100 SXM 80GB 8× $2.79；A100 40GB 全配置 $1.99；A10 1× $1.29；GH200 1× $2.29；Quadro RTX 6000 1× $0.69
    * 价格不含税；定价页未列 H200 / GB200 实例
    * 第三方索引标注 Lambda H100 $3.29（PCIe）且 out of stock
- 长片作用: 在这条产线上性价比不占优。唯一值得考虑的场景是你要做 Wan2.2 全参微调、需要稳定的 8×H100 多节点（Lambda 的 8 卡整机 + Infiniband 比散户市场靠谱）。单个角色 LoRA 用 Lambda 是纯浪费钱。
- 来源: https://lambda.ai/pricing https://getdeploying.com/gpus/nvidia-h100

## CoreWeave  (CoreWeave)
- 类别/成熟度: service / **production** | 许可: — | 成本: H100 $6.16/GPU/h 按需，是 RunPod Community 的 2.3 倍
- 是什么: 企业级 GPU 云，整机/集群售卖，按需单价是全场最贵之一。
- 关键事实:
    * 整机按需价：HGX H100 8 卡节点 $49.24/h（≈$6.16/GPU/h）；HGX H200 8 卡 $50.44/h（≈$6.31/GPU/h）；HGX B200 8 卡 $68.80/h（≈$8.60/GPU/h）；GB200 NVL72 $42.00/h
    * Spot 最多比按需便宜 60%；第三方索引给出 CoreWeave H200 spot $2.62/h
    * 承诺用量（reserved）最多 60% 折扣
    * per-GPU 计价仅对其 inference platform 客户开放，需联系销售
- 长片作用: 个人/小团队阶段完全不用考虑。只有当平台跑到「每天出几百条 2-8 分钟成片、需要预留百卡级常驻推理集群」时，它的 reserved 折扣和 InfiniBand 才开始有意义。现在用它训 LoRA 是烧钱。
- 来源: https://www.coreweave.com/pricing https://getdeploying.com/gpus/nvidia-h200 https://www.thundercompute.com/blog/coreweave-gpu-pricing-review

## Together AI  (Together AI)
- 类别/成熟度: service / **production** | 许可: — | 成本: H100 $1.99-3.99/h（口径差异大，需询价确认）
- 是什么: 以 Serverless 推理为主，同时卖 GPU 集群；视频生成有按条计费的 serverless 端点。
- 关键事实:
    * GPU 集群按需：H100 $3.99/h（标注促销价）、B200 $8.19/h；提供 H100/H200/B200/B300；第三方索引给 Together H100 $1.99、H200 $2.99（口径可能为预留/合约价）
    * Serverless 视频生成：$0.14-$2.50 每条视频
    * Serverless 图像：$0.015-$0.134 每张
    * 微调（文本模型）$0.34-$40.00 每百万 token，每个 job 最低 $4-$60
    * 存储 $0.16/GiB/月
- 长片作用: 不是训练首选，但它的 serverless 视频端点可以当「成片渲染的溢出通道」——当你自建 GPU 池被占满时，把部分分镜按 $0.14-2.50/条丢过去。注意它上面的视频模型是别家的托管版本，不能挂你自己的 LoRA，所以只能用于不需要角色一致性的空镜/转场/氛围镜头。
- 来源: https://www.together.ai/pricing https://getdeploying.com/gpus/nvidia-h100 https://getdeploying.com/gpus/nvidia-h200

## Novita AI  (Novita AI)
- 类别/成熟度: service / **usable** | 许可: — | 成本: H100 $3.39/h，比 RunPod Community 贵 26%
- 是什么: 中资背景的海外 GPU + API 混合平台，GPU 实例价格中等偏高，API 侧视频按秒计费。
- 关键事实:
    * 第三方索引 2026-09-19：Novita H100 SXM on-demand $3.39/GPU/h；A100 SXM $1.60/h（标注缺货）
    * 官方价格页未把 GPU 实例逐卡型列价（/pricing 只列 API），GPU 逐卡价需进控制台
    * API 侧：视频生成 $0.084-$0.168/秒（随模型与分辨率）；图像 $0.02/张；TTS $15/百万字符；语音克隆 $0.1-$1.50/个音色
    * LLM 示例价：DeepSeek V4 Flash $0.14/Mt in、$0.28/Mt out
- 长片作用: 训练侧没有理由选它（比 RunPod/Vast 贵且缺货）。有价值的是 API 侧：$0.084-0.168/秒的视频生成，折算 2-8 分钟成片纯渲染 $10-80——可作为成本上界参考，用来判断「自建 GPU 池是否划算」的盈亏平衡点。另外它的 TTS/语音克隆是数字人对白的备选（注意：Wan2.5+ 自带对白声，Wan2.2 开源版不带，需要外接 TTS + S2V 对口型）。
- 来源: https://novita.ai/pricing https://getdeploying.com/gpus/nvidia-h100 https://getdeploying.com/gpus/nvidia-a100

## AutoDL（国内）  (AutoDL 算力云)
- 类别/成熟度: service / **production** | 许可: — | 成本: H800 ¥9.98/h ≈ $1.40/h（按 7.1 汇率），比 RunPod H100 Community $2.69 便宜近一半
- 是什么: 国内最主流的零售 GPU 租赁，按秒计费无合同，是国内个人/小团队训 LoRA 的默认选择。
- 关键事实:
    * 2026 报价（元/卡/小时）：RTX 4090 24GB ¥2.1-3.45（部分渠道报低至 ¥2.68 或更低）；A100 80GB ¥5.98；A100 40GB ¥3.28-3.45
    * RTX 5090 32GB ¥2.78（会员 95 折）；H800 80GB ¥9.98；H20 96GB ¥9.98
    * 2026-03 时点 RTX 5090 32G 无库存记录；国内高端卡 2026 上半年整体涨价约 40%
    * 同类平台对比：算家云 4090 青春版 ¥1.24-1.3、A100 80GB ¥6.68；润云 5090 ¥2.29/卡/时
- 长片作用: 如果你人在国内，这是最省事的训练工位——不用翻墙、不用信用卡、镜像里 PyTorch/CUDA 现成、数据盘可跨实例复用（同样能留住 cache 的 latents）。但两个坑：(1) 没有 8 卡整机，做不了 Wan2.2 全参微调；(2) 高端卡（H800/H20）经常排队。国内跑训练 + 海外跑批量推理是个常见的务实组合。
- 来源: https://www.autodl.com/ https://www.autodl.com/docs/gpu/ https://aieii.com/posts/2026-08-20-gpu-cloud-price-comparison-2026/

## 阿里云 PAI / EGS GPU 实例  (阿里云)
- 类别/成熟度: service / **production** | 许可: — | 成本: A100 按量 ¥34.7/h ≈ $4.9/h，约为 Vast.ai A100 $0.33 的 15 倍
- 是什么: 阿里云的 AI 平台（PAI-DSW/DLC）与弹性 GPU 服务（2026 更名 EGS），公开价明显高于零售市场。
- 关键事实:
    * gn7e 实例（NVIDIA A100，16 核 125GB 内存）按量 ¥34.742/小时——第三方对比文章直接把它列为「阿里云 A100 ¥34.7/小时（按量）」
    * PAI-DSW 按量付费按分钟计费，账单按小时推送
    * 支持包年包月 / 按量付费 / 抢占式实例三种计费模式
    * 公开 GPU 卡型覆盖 L20、A10、V100、T4、P100、P4、GRID 虚拟化及机密计算实例；A100/H20/H800 等高端卡的精确小时价不在公开页面，需控制台或询价
- 长片作用: 纯算力性价比极差，不要用它训 LoRA。它唯一的结构性优势是和百炼在同一云内：如果你走百炼 wan2.6/2.7 渲染路线，把数据处理/分镜编排放 PAI 上能省掉 OSS 跨云流量和鉴权复杂度。做资质合规（ICP/算法备案）也只有国内云能接。把它定位成「合规与编排层」，不是「训练层」。
- 来源: https://help.aliyun.com/zh/pai/product-overview/dsw-billing-description https://developer.aliyun.com/article/1714433 https://aieii.com/posts/2026-08-20-gpu-cloud-price-comparison-2026/

## 腾讯云 GPU / 火山引擎 GPU  (腾讯云 / 字节火山引擎)
- 类别/成熟度: service / **production** | 许可: — | 成本: 无公开挂牌价；需询价
- 是什么: 两家国内大厂 GPU 云，共同特点是 H100/H800/A100 的小时价不公开挂牌。
- 关键事实:
    * 火山引擎 GPU 云服务器计费分实例规格 + 云盘 + 公网 IP 三个维度，支持按量计费 / 包年包月 / 抢占式实例
    * 火山引擎公开卡型覆盖 T4、V100、A100、A800、H100、L40S；精确小时价需登录控制台或走询价
    * 腾讯云公开实例族包含 GN10Xp（V100 NVLink 32GB）、计算型 PNV5b（48GB GDDR6）等；GPU 定价入口 buy.cloud.tencent.com/price/gpu 需登录看实时价
    * 第三方观察：国内 A100 按小时普遍 ¥10 以上（对照 AutoDL ¥5.98，说明大厂溢价明显）
- 长片作用: 现阶段不占工位。火山引擎的真实吸引力不在 GPU 出租，而在 Seedance 2.0 本身就是字节的模型——如果未来要谈 Seedance 的企业通道/更高配额/更宽审核策略，火山引擎是唯一入口。把它当商务关系而非算力供应商来看。腾讯云在这条链路上没有不可替代性。
- 来源: https://www.volcengine.com/docs/6419/69805 https://cloud.tencent.com/document/product/560/8025 https://buy.cloud.tencent.com/price/gpu

## Wan2.2 A14B 双模型（高噪/低噪）LoRA 的工程含义  (—)
- 类别/成熟度: technique / **production** | 许可: — | 成本: 同样步数，成本 ×2（两个模型各训一遍）
- 是什么: Wan2.2 A14B 是 MoE 双专家结构，训练和推理都必须成对处理，这是所有成本估算翻倍的根因。
- 关键事实:
    * A14B = 27B 总参 / 14B 激活，分 high-noise 与 low-noise 两个 DiT
    * musubi：--dit（低噪）+ --dit_high_noise（高噪），切换边界 --timestep_boundary 默认 I2V 0.9 / T2V 0.875
    * 社区共识：分开训两个 LoRA 比联合训效果更好；两者用不同 flow shift（低噪 1.0-2.0，高噪 3.0-5.0）
    * 社区共识：大模型用 rank 16 或更低，而不是 rank 32
    * 实测参考：16GB 卡 rank16 1600 步，高噪+低噪背靠背约 12 小时；24GB 卡 batch 4 / 768×768 / rank32 / 16 epoch 可跑
    * 角色 LoRA 的一个关键经验：用单帧静图训练即可泛化到运动，静图的采集与打标成本远低于视频片段（ComfyUI 版 Musubi Wan LoRA Trainer 即按单帧训练设计）
- 长片作用: 直接决定你的排期和预算：任何「训一个 Wan2.2 角色 LoRA 要多久多少钱」的报价，如果没有 ×2，就是错的。另外「静图可训角色 LoRA」这条对长片产线意义重大——你可以先用图像模型批量生成角色的多角度/多表情静图（几十张），直接训出角色 LoRA，跳过「先要有该角色的视频素材」这个鸡生蛋问题。
- 来源: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md https://github.com/kohya-ss/musubi-tuner/discussions/455 https://www.runcomfy.com/comfyui-nodes/comfyUI-Realtime-Lora/musubi-wan-lora-trainer

### 建议
【硬约束核实结论】
1. 你列的 Seedance 2.0 约束我无法在本轮检索中找到官方文档逐条核对，均维持原样不推翻，但已把「未能独立证实」的条目放进 uncertainClaims。
2. 一个必须先说的事实修正方向：阿里的开源和闭源在 Wan 2.2 之后彻底分叉。Wan-Video GitHub org 至今只有 6 个仓库（Wan2.1、Wan2.2、Wan-Animate-2、Wan-Dancer、Wan-skills、diffusers fork），没有 Wan2.5/2.6/2.7/3.0 任何一个；HuggingFace Wan-AI 组织同样最高只到 Wan2.2 系。而百炼的微调文档里 wan2.5-i2v-preview / wan2.6-i2v / wan2.7-i2v 都在支持列表里。结论：Wan2.5 及以后全是闭源 API，和 Seedance 2.0 是同一种关系。所以你的「本机 LoRA」只能建在 Wan2.2（Apache-2.0）之上。

【训练栈选型（直接可执行）】
- 主栈 musubi-tuner：角色 LoRA 的默认工位。Apache-2.0、生态最厚、Wan2.2 双模型支持最成熟、24GB 可跑。明确限制：多卡只有 DDP，不支持 FSDP/DeepSpeed，所以它只能训 LoRA，全参微调别指望。
- 副栈 diffusion-pipe：需要多卡/全参时切过去。DeepSpeed + 流水线并行，2×24GB 能把 14B 切开。Windows 原生不可用，走 WSL2。
- 数字人专用栈 DiffSynth-Studio：唯一能训 Wan2.2-Animate 和 Wan2.2-S2V 的开源栈，ZeRO-3 分片。你要做「数字人对口型说 2-8 分钟台词」，这条线躲不开。
- 快速试错 ai-toolkit：MIT，有 Web UI，24GB 消费卡直接跑，适合让非工程同事迭代角色数据集。
- SimpleTuner：横向比 Wan / LTX-2 / LongCat 时用。警告：AGPL-3.0，你要做闭源商业平台，代码不要嵌进服务。
- 直接排除：finetrainers（Wan 只到 2.1，2026 年只有 bot 提交，实质停更）、OneTrainer（根本不支持 Wan）。

【百炼官方 LoRA 训练 API：真实存在，但别把角色资产押在上面】
可自助提交，wan2.2-i2v-flash / wan2.5-i2v-preview / wan2.6-i2v / wan2.7-i2v 都能训。数据 20-100 条视频、单条 2-10 秒、zip ≤1GB、data.jsonl 三字段（prompt / first_frame_path / video_path），触发词用无意义罕见词。默认 lora_rank=32、lr=2e-5、n_epochs=50，建议总步 ≥800。训练价按 Token：wan2.2-i2v-flash ¥0.06/千Token、wan2.5-i2v-preview ¥0.32、wan2.7-i2v ¥2.0；官方例子 10 秒视频 + wan2.7-i2v 预估 ¥576。部署免费、5-10 分钟起服务、推理按基础模型标准价。
致命点：产出 LoRA 不可下载，只能在百炼云端推理。所以定位是「渲染后端」，不是「资产库」。角色身份这种你最核心的资产，必须在本地 Wan2.2 上再训一份可导出的 LoRA，否则平台哪天调价/下线/改审核策略，你整个角色库归零。

【云 GPU 选型（2026-09-19 实价）】
- 训 LoRA：RunPod H100 SXM Community $2.69/h（或 Secure $3.49）为基准盘；省钱用 Vast.ai H100 $1.74/h、A100 80GB $0.33/h。预算档 RTX 5090 RunPod $0.69 / Vast $0.44。国内用 AutoDL：4090 ¥2.1-3.45、5090 ¥2.78、A100 80G ¥5.98、H800 ¥9.98。
- 批量出片：Vast.ai interruptible（官方口径比按需便宜 50%+，H200 spot 见过 $0.02）+ 幂等分镜任务队列，掉机只重跑单镜。2-8 分钟片子按 5 秒一镜有 24-96 个镜头，天然适配。
- 别用：CoreWeave（H100 $6.16/GPU/h 按需）、Lambda（H100 SXM $3.99-4.29 且常缺货）、阿里云 EGS（A100 按量 ¥34.7/h ≈ $4.9，是 Vast 的 15 倍）。这三家只在「要 8 卡 InfiniBand 整机做全参微调」或「要国内合规落地」时才有理由。
- 腾讯云 / 火山引擎不挂牌高端卡小时价，需询价。火山的价值不在算力，在它是 Seedance 的娘家——企业通道要谈就从这里谈。

【训一个 Wan 2.2 角色 LoRA 的真实成本】
基准配置（推荐）：RunPod 或 Vast 单张 H100 80GB，musubi-tuner，30-60 张角色静图 + 5-15 条 3-5 秒短片，rank 16 / alpha 16，分辨率桶 512×512×33 与 640×640×21，target_frames 遵守 N*4+1，fp8_scaled 开，blocks_to_swap 视情况，高噪与低噪各训 1500-2500 步。
- cache latents + text encoder：0.5-1.5 GPU-小时
- 高噪模型 2000 步：1.5-3.5 GPU-小时
- 低噪模型 2000 步：1.5-3.5 GPU-小时
- 单次成功跑合计：4-9 GPU-小时
- 单次成本：Vast H100 $1.74/h → $7-16；RunPod H100 Community $2.69/h → $11-24；RunPod Secure $3.49/h → $14-31
现实预算（含调参迭代，第一个角色通常要 3-5 轮 caption/lr/rank 调整）：
- 首个角色：$60-150，日历时间 2-4 天（其中 GPU 实跑 20-45 小时）
- 流程跑顺后每个新角色：$15-40，8-18 小时
- 24GB 消费卡路线（RTX 5090 $0.69/h）：单次跑 12-20 小时 → $8-14，绝对便宜但要忍受 OOM 调试和低分辨率上限，日历时间会拉长一倍以上
- 完全不想运维：fal.ai fal-ai/wan-22-trainer/t2v-a14b $0.004/步，2000 步 ×2 模型 ≈ $16，权重可下载。建议第一个角色直接用它验数据集，再决定要不要自建。
参考锚点：社区 16GB 卡实测 1600 步高噪+低噪背靠背约 12 小时，和上面 H100 的 3-7 小时区间在量级上一致。

【给长片系统的落位建议】
1. 角色身份 = 本地 Wan2.2 LoRA（musubi，可导出，永久资产）。
2. 表演与口型 = Wan2.2-Animate-2 + S2V（DiffSynth 训，720P 需 8×A800 档位推理，成本高但可控性最强）。
3. 高观感成片镜头 = 百炼 wan2.6/2.7 闭源 API（15 秒 1080p、原生音频），或 Seedance 2.0 通道。这一层只当渲染器，不放资产。
4. 训练用 RunPod/AutoDL 按需，出片用 Vast interruptible 队列，合规落地走阿里云/火山。
5. 关键工程前置：把 cache 的 latents 放在可跨实例挂载的网络盘（RunPod Network Volume / AutoDL 数据盘）。视频训练的 cache 阶段能占掉 20-30% 的总 GPU 时间，复用它是最大的单项省钱手段。

### 存疑
- Seedance 2.0 的具体参数（单段 4-15 秒、2.5 约 30 秒、多模态参考 9 图 + 3 视频 + 3 音频、自带对白声、内容审核）——本轮检索未命中字节官方文档逐条核对，按用户给定硬约束原样保留，未独立证实。
- 多个 SEO 站点（wan27.org、localaimaster.com、atlascloud.ai、flowith.io）声称「Wan 3.0 已 Apache-2.0 开源，2026-04 放出 1.3B 与 14B 检查点」。GitHub Wan-Video org 只有 6 个仓库、无 Wan3.0；HuggingFace Wan-AI 组织也没有。该说法高度可疑，判定为内容农场编造，不写入 findings。
- 「Wan 2.6 是国内首个支持角色扮演的视频模型 / 单次 15 秒 / 720P+1080P / Wan2.7 用 resolution 参数替代 size 且默认出声」——来源为阿里云开发者社区文章与第三方站点，未从 help.aliyun.com API 参考页逐字核对。
- 阿里百炼计费页的示例「一条 10 秒视频、wan2.7-i2v、max_pixels=36864、n_epochs=800，预估 ¥576」中 n_epochs=800 与微调指南的 n_epochs 默认 50 冲突，疑为「总步数 800」笔误。由此推算 wan2.2-i2v-flash 的等价训练费用（单价是 wan2.7 的 1/33，但 max_pixels 默认高 7 倍）会落在 ¥100-200 量级，属推导值，非官方数字。
- Together AI 的 H100 价格两个口径冲突：官网 $3.99/h（标注促销），第三方索引 $1.99/h；H200 官网未列，索引 $2.99/h。疑为按需 vs 预留/合约价混淆，需询价确认。
- 腾讯云与火山引擎的 H100/H800/A100 按量小时价未能取得任何官方公开数字，两家均需登录控制台或走询价流程。文中所有国内大厂 GPU 价格结论仅基于第三方观察（「国内 A100 按小时普遍 ¥10 以上」），不可用于报价。
- musubi-tuner 在 Wan2.2 A14B 上的实际 秒/步 吞吐未找到 H100 或 4090 的权威 benchmark。成本区间中的 1.5-3.5 GPU-小时/2000步 是从 16GB 卡 1600 步 12 小时（双模型）的社区实测按算力比外推，误差可能达 ±50%。
- Vast.ai 的 H200 spot $0.02/h、A100 spot $0.18/h 属市场瞬时报价，来自聚合索引快照，不代表可持续可得的价格。
- Wan-Animate-2 蒸馏 Lite 版「推理延迟降到实时门槛，可用于流式角色动画」为仓库自述，无第三方复现数据。Wan-Animate-2 的最大生成时长/帧数上限官方文档未给出。
- diffusion-pipe 的 LICENSE 文件本轮未直接读取，许可证类型未确认。
- AutoDL 的 RTX 5090 ¥2.78/h、H800 ¥9.98/h、H20 ¥9.98/h 来自第三方对比文章而非 AutoDL 官网实时页面（官网首页抓取未返回价格表），且文中提到 2026-03 时点 5090 无库存。
- 「RunPod A100 SXM $1.00/h」（getdeploying 索引）与 RunPod 官方定价页 $1.39/$1.59 不一致，疑为索引抓到了 Community 特价或历史价。

### 事实核查修正
- [WRONG] [musubi-tuner] 支持视频架构包含 HiDream-O1
  → HiDream-O1 在 musubi-tuner 里是图像模型，不是视频架构。证据：docs 目录为 hidream_o1.md，README 文档索引写作「HiDream-O1-Image」，源码为 src/musubi_tuner/hidream_o1_generate_image.py（generate_image 而非 generate_video）。分类错误会导致误判「musubi 能训 HiDream 视频 LoRA」。 https://github.com/kohya-ss/musubi-tuner
- [WRONG] [musubi-tuner] 图像侧支持 …Kandinsky 5…
  → Kandinsky 5 在 musubi-tuner 里是视频模型，不是图像模型。证据：src/musubi_tuner/kandinsky5_generate_video.py、docs/kandinsky5.md。SimpleTuner 也把它列为「Kandinsky 5.0 Video（2B lite / 19B pro）」。原文把视频/图像两类互换了（与 HiDream-O1 那条正好错反）。 https://github.com/kohya-ss/musubi-tuner/tree/main/src/musubi_tuner
- [WRONG] [musubi-tuner] 显存开关：--blocks_to_swap、--fp8 / --fp8_scaled、--gradient_checkpointing、--offload_inactive_dit
  → 训练侧的 DiT 量化开关是 --fp8_base（可叠加 --fp8_scaled），文本编码器侧是 --fp8_llm / --fp8_t5；裸 --fp8 只出现在推理脚本（wan_generate_video.py 等）。直接在训练命令里写 --fp8 会报未知参数。--blocks_to_swap / --gradient_checkpointing / --offload_inactive_dit 三个确认无误。 https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md
- [WRONG] [musubi-tuner] JSONL（image_jsonl_file / video_jsonl_file，字段 image_path/caption）
  → 视频 JSONL 的字段是 video_path（不是 image_path），共享 schema 为 video_path / caption / control_path / audio_path；图像 JSONL 才是 image_path / caption（Qwen-Image-Layered 另有 image_path_0/1/2）。「JSONL 必须显式给 cache_directory」确认无误。 https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md
- [WRONG] [musubi-tuner] 仅 DDP，官方未确认支持 FSDP/DeepSpeed，因此全参微调在多卡上基本不可行
  → 前半句可以从「未确认」升级为「已确认不支持」：README 只写 Multi-GPU training (using Accelerate)、文档待补，accelerate config 示例明确让用户对 DeepSpeed 回答 NO，仓库内无任何 FSDP 代码或文档。但后半句的前提站不住：musubi-tuner 并非只有 LoRA，src/musubi_tuner 下存在 hv_train.py 与 hidream_o1_train.py 这类全参微调脚本（仅个别架构有，Wan 系列确实只有 *_train_network.py 即 LoRA）。 https://github.com/kohya-ss/musubi-tuner/tree/main/src/musubi_tuner
- [UNVERIFIABLE] [musubi-tuner] 社区实测：RTX 4070 Ti Super 16GB、rank16/alpha16、1600 步、lr 3e-5，高噪+低噪两个模型背靠背约 12 小时；分辨率组合 360x360x65 / 512x512x33 / 640x640x21
  → 未找到一手来源（非官方 issue/discussion/可追溯帖子）。该数字同时是后续「1.5-3.5 GPU-小时/2000 步」成本外推的唯一锚点，属于单点未证实数据支撑整条成本链，建议在报告里降级为「某社区口径，未复现」或直接删掉成本区间。 
- [WRONG] [diffusion-pipe] HunyuanVideo 无 block swap 需 48GB 或 2×24GB 流水线并行
  → 误植。supported_models.md 里这句话（「Without block swapping, you will need 48GB VRAM, or 2x24GB with pipeline parallelism」）位于 ## HiDream 小节（第 239 行），## HunyuanVideo 小节（94-113 行）根本没有任何显存数字。另一处 48GB/2×24GB 出现在 SDXL 全参微调（第 52 行）。同段落的「Flux 2 Dev LoRA 需 ≥48GB」则确认无误（第 536 行，Flux 2 小节：Dev needs https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md
- [WRONG] [diffusion-pipe] 视频模型支持矩阵（各模型 LoRA/全参/fp8 能力）
  → LoRA/全参/fp8 三列与官方 Summary 表一致，但矩阵漏掉了关键的任务维度限制，会让人误以为「支持」＝全任务可训：MiniMax H3 目前只支持 T2I 与 T2VA；LTX 2.3 只支持 T2I 与 T2V，无音频、无 I2V；HunyuanVideo-1.5 只支持 T2I 与 T2V；HunyuanVideo 只有 t2v。另 LTX 2.3 的 blocks_to_swap=46 原文是「该模型的最大值，且只有在分辨率/时长/rank 都足够低时才『可能勉强』塞进 24GB」，不是「24GB 需设 46」这种确定性配置。 https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md
- [WRONG] [diffusion-pipe] 环境要求 Python 3.12 + PyTorch ≥2.9.0 + nvcc
  → README 没有规定 PyTorch 最低版本。原文是：PyTorch 刻意不写进 requirements（不同 GPU 需要不同版本），「As of this writing (October 26, 2025), PyTorch 2.9.0 with CUDA 12.8 works on my 4090」——这是作者单机实测记录，不是下限。把它写成硬性 ≥2.9.0 会让老卡用户误以为必须升级。Python 3.12、nvcc、WSL2（DeepSpeed 无原生 Windows 支持）三项确认无误。 https://github.com/tdrussell/diffusion-pipe
- [WRONG] [DiffSynth-Studio] Wan 训练覆盖…外加 Wan-Dancer、MOVA、LongCat、Video-As-Prompt
  → MOVA 不属于 Wan 训练矩阵。examples/wanvideo/model_training/{lora,full} 下只有 Wan-Dancer-14B-global/local.sh、LongCat-Video.sh、Video-As-Prompt-Wan2.1-14B.sh（以及未被提及的 krea-realtime-video.sh、Wan2.2-Animate-2-14B / -Distilled.sh、Wan2.1-1.3b-speedcontrol-v1.sh）；MOVA 是并列的独立目录 examples/mova。其余 Wan2 https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo/model_training
- [WRONG] [ai-toolkit] Wan 2.2 T2I 支持于 2025-08-16 加入（ostris 官方公告），I2V 14B 教程 2025-08-21 发布
  → 2025-08-16 合并的是 PR #377（分支名 wan22_14b），即 Wan 2.2 14B 整体支持，不是「T2I 支持」；Wan2.2 5B 支持更早，2025-07-29。24GB 的图像训练示例配置 config/examples/train_lora_wan22_14b_24gb.yaml 是 2025-08-28 才加入的（commit message: Added example config for training wan22 14b 24GB on images）。「I2V 14B 教程 2025-08-21」未能从仓库或 https://github.com/ostris/ai-toolkit/commits/main/config/examples/train_lora_wan22_14b_24gb.yaml
- [WRONG] [SimpleTuner] LTX Video ~2.5B(Apache-2.0)、LTX Video 2 19B(Apache-2.0)、Hunyuan Video 8.3B(AGPL-3.0)
  → 三个许可证全错（属于「把受限/自有许可说成开源许可」）。HuggingFace 一手模型页：Lightricks/LTX-Video 为 license: other（LTXV 自有许可，非 Apache-2.0）；Lightricks/LTX-2 为 license: other，license_name: ltx-2-community-license-agreement（社区许可，有使用限制，非 Apache-2.0）；tencent/HunyuanVideo 为 license: other，license_name: tencent-hunyu https://huggingface.co/Lightricks/LTX-2
- [WRONG] [SimpleTuner] Wan 2.x I2V 支持高/低噪 stage preset
  → preset 本身存在（model_flavour=i2v-14b-2.2-high / i2v-14b-2.2-low、wan_validation_load_other_stage、wan_force_2_1_time_embedding 均确认），但漏掉了 WAN.md 开头的决定性限制：「Currently, image-to-video training is not supported for Wan, but T2V LoRA and Lycoris will run on the I2V models.」——即 SimpleTuner 在 https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/WAN.md
- [OUTDATED] [SimpleTuner] 2026-01 在做 LTX-2 audio-only 训练
  → 已经不是「在做」：audio-only 训练 2026-01-20 就合并了（PR #2461 feature/audio-only-ltx2），随后 01-31 加 --validation_audio_only、02-01 修 audio fps、02-13 修 s2v/ltx-2 音频自动切分，2026-08-15 还并了 audio-dataset-fake-video。当前 LTXVIDEO2.md 已文档化 audio 数据块与 allow_zero_audio，并已出现 LTX-2.5 的 AV 双 CFG 验证（video CFG 3.0 https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/LTXVIDEO2.md
- [WRONG] （存疑项）Together AI H100 两个口径冲突 $3.99 vs $1.99，H200 索引 $2.99；疑为按需 vs 预留/合约价混淆
  → 假设错了，不是「按需 vs 预留」，是「按需 vs 可抢占（preemptible）」。官网定价页同时列：HGX H100 on-demand $3.99/GPU·h（带 Promotion valid until 09/30/26 标注）、preemptible $1.99；HGX H200 on-demand $5.99、preemptible $2.99。第三方索引抓到的 $1.99/$2.99 正是 preemptible 档。Reserved 是另一套（7-30 天到 181+ 天阶梯，最长档需 Contact us）。所以 H200 官网是有 https://www.together.ai/pricing
- [WRONG] （存疑项）「RunPod A100 SXM $1.00/h」与官方 $1.39/$1.59 不一致，疑为索引抓到 Community 特价
  → $1.39 本身就是 Community Cloud 价、$1.59 是 Secure Cloud 价，所以 $1.00 不是 Community 档，只是 getdeploying 的过期快照。官网当前口径：A100 SXM 80GB $1.39(Community)/$1.59(Secure)；H100 SXM $2.69/$3.49；H200 141GB $3.59/$4.59。报价时应按 Community/Secure 两档分别给，不要用第三方索引单值。 https://www.runpod.io/pricing

### 遗漏补充
- VideoX-Fun (aigc-apps/VideoX-Fun, 2.2k stars, 2026-09-17 仍在更新)：阿里 PAI 侧的 Wan-Fun/CogVideoX-Fun 官方训练框架，支持任意分辨率生成与 LoRA/全参训练，是 Wan2.1/2.2 Fun 系列控制模型（Control/InP/Camera）的上游出处。本轮调研把 Fun 系列只记在 DiffSynth 名下，漏了原厂训练栈。
- Lightricks/LTX-Video-Trainer（469 stars）：LTX 官方组织下的 LoRA / IC-LoRA 训练器，是 LTX 系列最接近一手的训练路径；调研里 LTX 训练只经由 diffusion-pipe / SimpleTuner / ai-toolkit 三个第三方栈。
- LoRA 权重格式互通问题完全缺席：diffusion-pipe 输出 ComfyUI 格式、finetrainers/部分 SimpleTuner 输出 Diffusers 格式、musubi 部分架构输出 Kohya 风格，HunyuanImage-2.1 的 key 名甚至与原模型结构不同。跨框架/跨推理端迁移是实际落地的首要卡点，应单列一节。
- musubi-tuner 的 torch.compile 与 LoHa/LoKr 支持（docs/torch_compile.md、docs/loha_lokr.md）以及 --log_grad_metrics 梯度诊断：分别关系到吞吐和「训崩了怎么查」，对 2000 步级别的成本估算比外推的 GPU-小时更有用。
- SimpleTuner 的 TREAD（token dropout，WAN.md 实测 1.3B 从 10 s/step 降到 7.7 s/step @ bs=2 480p）与 CREPA（视频 DiT 跨帧表征对齐正则）：这是本维度唯一有一手 sec/step 数字的加速手段，恰好能替代那个被标注为 ±50% 误差的吞吐外推。
- diffusion-pipe 的 eval set / held-out metrics 与 TensorBoard 指标：几个框架里只有它内建泛化评估，选型时是区分点。
- 云 GPU 侧漏掉的按需供应商：Ostris Cloud（ai-toolkit 作者自营，README 内直接推荐，与 ai-toolkit 集成最紧）、Modal、fal.ai（有现成 Wan/LTX LoRA 训练 API，免运维）、Lambda、Nebius、DataCrunch、Crusoe、Salad，以及 RunPod Serverless 这种按秒计费的训练/推理形态。
- 国内侧漏掉的 GPU 渠道：阿里云 PAI-DSW / 灵骏（DiffSynth 的原生落地环境，按量价格公开）、腾讯云 HAI、以及共绩算力/蓝耘/趋动云等二级算力平台——比只给「国内 A100 普遍 ¥10 以上」这种第三方观察更可报价。


====================================================================================================
# [open-t2v-sota] 2026年9月可用的开源视频生成基座模型全景 — 面向2-8分钟AI数字人长片产线的基座选型

## MiniMax H3 (Hailuo 3.0)  (MiniMax (稀宇科技))
- 类别/成熟度: open-weights / **production** | 许可: MiniMax H3 Community License Agreement。硬性限制：(1) 本地部署权利明确排除美国、欧盟、英国、韩国四地；(2) 年营收超 $20M 需另行书面授权；(3) 产品 UI 必须显著标注 "MiniMax H3"；(4) 禁止蒸馏（不得用其输出训练更小模型）；(5) 禁止违法/色情/侵权内容。 | 成本: pruned INT8 约 21GB，刚好塞进 24GB 卡（4090）全驻显存；5 秒 768p 级片段在 32GB RTX 5090 约 3 分钟，4090 上 INT8 约 3–10 分钟（社区实测区间，随步数/分辨率浮动）；12GB RTX 3060 需 6–25 分钟。NVFP4 只有 Blackwell（50 系 / RTX PRO 6000）原生加速，Ada（4090）上是模拟执行无提速且掉质量，4090 应选 pruned int8。
- 是什么: 33B 稠密单流 omni-modal Transformer（H3-Omni-Transformer），文/图/视频/音频 token 用 3D MM-RoPE 统一到一个序列里，原生同步出 32kHz 立体声。2026-08-03 在 HuggingFace 放出权重，是目前开源权重里画质天花板。
- 关键事实:
    * 2026-07-30 预告，2026-08-03 正式开放权重（MiniMaxAI/MiniMax-H3）
    * 33B 参数稠密（其中 13B 在 AdaLN 分支）
    * 原生输出 4–15 秒，24fps，本地权重原生短边 768p；2K 需要 H3-Regenerate-2K，该模块未开源，只能走官方 API
    * 原生同步音频 32kHz 立体声
    * BF16 原始 checkpoint 123.6GB；社区量化 pruned INT8 约 21GB / 42.5GB 多个版本；NVFP4 版本（Blackwell 原生）
    * HF 月下载量 4,449,605，点赞 5.47k（截至 2026-09）
    * Artificial Analysis Text-to-Video Arena：Elo 1220，全部开源权重模型第一；总榜第 4，仅次于 Gemini Omni Flash(1233)/Wan 3.0(1229)/MiniMax H3 Max(1227)，且高于 Dreamina Seedance 2.0 720p(1210)
    * Image-to-Video 开源权重榜同样第一：带音频 1181，不带音频 1354
    * 未开源组件：H3-Context-IR（多模态指令精炼层，只能托管调用）、稀疏注意力推理优化、H3-Regenerate-2K
- 长片作用: 这是 2-8 分钟长片产线里的「主镜头基座」工位——每个分镜 4–15 秒的成片镜头由它出。它是唯一一个单镜观感真正逼近 Seedance 2.0 且能本机 LoRA 的开源权重模型（Arena 上甚至略高于 Seedance 2.0 720p）。自带同步对白音频，直接顶掉了「视频生成 + 单独 TTS + 对口型」三段式里的后两段。缺点：单段上限 15 秒，2-8 分钟必须靠分镜拼接；2K 要走 API，本地只能 768p 再自己超分。
- 来源: https://huggingface.co/MiniMaxAI/MiniMax-H3 https://artificialanalysis.ai/video/leaderboard/text-to-video https://explainx.ai/blog/minimax-h3-open-video-model-hailuo-july-2026

## MiniMax H3 LoRA 生态（AI-Toolkit / ComfyUI / Civitai）  (社区 + Ostris + Civitai)
- 类别/成熟度: technique / **usable** | 许可: 各 LoRA 独立授权，但底模的 MiniMax H3 Community License 约束会向下传导（地域 + 营收 + UI 署名 + 禁蒸馏） | 成本: 消费级卡可训（NVFP4 量化 + VAE 梯度 checkpointing），Turbo LoRA 推理 4–6 步
- 是什么: 围绕 H3 的微调与加速生态。Ostris AI-Toolkit 在权重放出当天（2026-08-03）就加入 minimax_h3 扩展，支持 T2V/I2V LoRA 训练；Civitai 为它单开了 "Hailuo H3 by MiniMax LoRA" 模型类别并办官方训练赛。
- 关键事实:
    * AI-Toolkit minimax_h3 扩展 2026-08-03 上线，含 packed-token transformer / video+audio VAE / text encoder / NVFP4 量化工具链
    * 支持两个 checkpoint（FL2VA 和 Ref2VA）的 T2V 与 I2V LoRA 训练，VAE 梯度 checkpointing + NVFP4 量化降显存，可在消费级卡训
    * Civitai 已有独立模型类别，官方 H3 训练赛（2 Million Buzz 奖池）2026-09 进行中，上线数周社区已发布 670+ 个基于 H3 的模型
    * Turbo LoRA（lightx2v / drbaph 等多个版本）：把 20 步压到 4–6 步出可用画面，文件 <1GB
    * 已有融合微调产物如 Minimax-h3_Singularity（HDR 清晰度、远景人脸修复、皮肤、动作与 VFX 强化），以 ComfyUI diffusion model 形式加载，与官方 INT8 同槽位
    * 对比：Civitai wan2.2 标签约 368 个模型（另有社区整理的 475 个 Wan2.2 LoRA 合集，2026-03）
- 长片作用: 这是「角色一致性」工位的关键。2-8 分钟片子里同一个虚构数字人要跨几十个分镜保持同一张脸/同一套造型，靠的就是角色 LoRA。H3 生态虽然只有 6 周，但增速（670+/6周 vs Wan2.2 累积 368–475）和官方工具链完备度已经反超。Turbo LoRA 把单镜成本压到 4–6 步，是长片批量出镜头的经济性前提。风险：太新，兼容性没有 Wan 生态那么无脑。
- 来源: https://comfyui-wiki.com/en/news/2026-08-03-ai-toolkit-minimax-h3-training https://comfyui-wiki.com/en/news/2026-08-06-minimax-h3-turbo-lora https://civitai.com/articles/35316/the-civitai-h3-training-contest

## Wan 2.2 (T2V-A14B / I2V-A14B / TI2V-5B / S2V-14B / Animate-14B)  (阿里巴巴 通义实验室)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 — 全系列可商用，无营收门槛、无地域限制、无 UI 署名要求。目前画质第一梯队里唯一真正干净的商用许可。 | 成本: TI2V-5B 24GB（4090 可跑，5s/720P <9min）；A14B 需 80GB（A100/H100 80G 单卡，或 4090 上走 GGUF/fp8 分块 offload，社区方案成熟）
- 是什么: 阿里最后一代开放权重的视频基座，Apache 2.0。A14B 是 MoE 双专家结构（高噪专家管早期布局、低噪专家管后期细节，按 SNR 阈值切换），总参 27B / 激活 14B；TI2V-5B 是稠密小模型，靠 Wan2.2-VAE 的 16×16×4 压缩比 + patchify 做到 64× 总压缩。
- 关键事实:
    * T2V-A14B / I2V-A14B：27B 总参、14B 激活 MoE，480P/720P，24fps
    * TI2V-5B：5B 稠密，720P，24fps
    * S2V-14B：2025-08-26 发布，音频驱动语音到视频
    * Animate-14B：2025-09-19 发布，角色动画/角色替换
    * TI2V-5B 在单张 RTX 4090 出 5 秒 720P 片需 <9 分钟
    * VRAM：TI2V-5B 至少 24GB；T2V/I2V-A14B 至少 80GB
    * Wan2.2 GitHub 仓库最后更新 2026-03-17；HF 上 Wan2.2-Animate-2-14B 系列更新到 2026-08
    * 训练数据比 Wan2.1 多 +65.6% 图像 / +83.2% 视频
- 长片作用: 「兜底基座 + 合规保险」工位。画质已经落后 H3 一代（Wan 2.2 未进 Arena 前 25），但它是全场唯一 Apache 2.0 的一线基座：如果你的长片产线要出海、要卖给企业客户、或要做二次分发，H3 的地域+营收+禁蒸馏条款会卡死你，Wan 2.2 不会。另外 Wan2.2-Animate-14B 和 S2V-14B 在数字人工位仍有具体用处（角色替换、音驱口型），可作为 H3 主镜的补充模块。
- 来源: https://github.com/Wan-Video/Wan2.2 https://huggingface.co/Wan-AI

## Wan 2.5 / 2.6 / 2.7 / 3.0  (阿里巴巴 通义实验室)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API，无权重许可 | 成本: API 计费，不适用本地显存
- 是什么: 阿里 Wan 2.2 之后的全部后续版本，全部闭源 API，无权重发布。市面上大量「Wan 2.5/2.6 开源」的中文/英文博文是 SEO 农场内容，属于事实错误。
- 关键事实:
    * Wan 2.5-Preview：2025-09 上线，仅 Alibaba Cloud API，权重从未发布到 HuggingFace / GitHub / ModelScope
    * Wan 2.6、Wan 2.7 同样闭源 API-only
    * Wan 3.0：2026-08-06 公测、2026-08-24 全量上线，仍是闭源 API；支持 30 秒单镜、文档到视频
    * Wan 3.0 在 Artificial Analysis T2V Arena Elo 1229，总榜第 2（高于 Seedance 2.0 720p 的 1210）
    * Wan2.2 官方仓库 issue #291「Wan 2.5 weights? Will be open-sourced?」至今无正式开源承诺
    * 截至 2026-09 中旬，Wan-AI HF 组织下最新开放模型是 Wan2.2-Animate-2-14B，无任何 3.0 权重
- 长片作用: 不能进本地产线（不能 LoRA、不能私有化）。它的意义有两个：(1) 证伪——不要被「Wan 2.5/2.6 开源」的软文骗去做技术选型；(2) Wan 3.0 的 30 秒单镜 + Elo 1229 说明闭源侧已经把单镜上限推到 30 秒，这是你分镜系统在做「单镜多长」决策时的行业参照值。
- 来源: https://github.com/Wan-Video/Wan2.2/issues/291 https://github.com/Wan-Video https://www.atlascloud.ai/blog/tips/is-wan-3.0-open-source

## LTX-2.5 (22B)  (Lightricks)
- 类别/成熟度: open-weights / **production** | 许可: LTX-2.x Community License：年营收 <$10M 免费使用（含商用）；>$10M 需付费 Commercial Use Agreement。微调产物的转让也按营收档位判定是否需授权。无地域限制。 | 成本: NVFP4 蒸馏包 16GB 起（Blackwell）；INT8 配置对应 20–24GB（4090）；完整 BF16 建议 48–80GB（RTX 6000 Ada / A100 / H100）。速度是全场最快的一线模型。
- 是什么: 22B 开放权重世界模型，自带音频 VAE + vocoder 做同步音视频。Lightricks 是唯一一家把官方 LoRA / IC-LoRA 工具箱系统化产品化的厂商，HF 上有 66 个模型，其中大量是官方维护的控制型 IC-LoRA。
- 关键事实:
    * LTX-2.5 主模型 2026-01-06 发布；HF Lightricks 组织共 66 个模型
    * 22B 参数，蒸馏版另有
    * 分辨率宽高需整除 32，帧数 num_frames % 8 == 1，上限 121 帧；stage1 默认 544×960，可上 1920×1088 或 4K(3840×2176) 超分
    * 原生同步音频（独立 audio VAE + vocoder）
    * 量化：bf16 原生 / fp8-cast 动态 / NVFP4(Blackwell + ltx-kernels) / ComfyUI int8 + convrot
    * 实测速度：RTX 5090(32GB) 出 4 秒 720p 约 25 秒；RTX 4090 跑 5 秒片实测显存 22.67 GiB；2×GB200 上 6.8 秒是官方 headline
    * Artificial Analysis T2V Arena：LTX-2.5 Fast Elo 1055 / Pro 1053，开源权重第 2（落后 H3 约 165 分）
    * 官方 IC-LoRA 阵容（2026-09 更新）：Pixel-Spatial-Upscaler、Ingredients、Water-Simulation、Cinemagraph、Decompression、Colorization、Clean-Plate、Deblur、Day-To-Night、Slow-Motion-Control 等
- 长片作用: 「产能工位」和「后期工位」。2-8 分钟片子 = 几十到上百个镜头，如果每镜 3–10 分钟（H3 在 4090 的水平），一部 5 分钟片可能要跑一整天。LTX-2.5 在 5090 上 25 秒出 4 秒 720p，是唯一能支撑「快速预演整片节奏」的基座——先用它把全片分镜粗排一遍看节奏，再把关键镜头交给 H3 精出。另外它的官方 IC-LoRA（Clean-Plate / Deblur / Colorization / Day-To-Night / Upscaler / Slow-Motion）本质是一套视频后期工具链，在长片产线里顶「调色-修复-补帧-超分」工位，这是别家没有的。画质单论比 H3 差一档。
- 来源: https://huggingface.co/Lightricks/LTX-2.5 https://huggingface.co/Lightricks https://www.runpod.io/blog/ltx-2-5-the-open-weights-world-model-built-for-speed-and-how-to-run-it-on-runpod

## SkyReels-V3 (R2V-14B / V2V-14B / A2V-19B)  (Skywork AI (昆仑万维))
- 类别/成熟度: open-weights / **usable** | 许可: skywork-license（自有许可证，非标准开源协议）。具体商用条款需逐条核实，不是 Apache/MIT。 | 成本: A2V-19B 推荐 24GB+，低显存模式 FP8 + block offload；V2 系列 540P/1.3B 仅 14.7GB
- 是什么: 统一多模态 in-context learning 框架，一套权重同时做「多主体参考图生视频」「音频驱动生视频」「视频到视频」。A2V-19B 是其中最关键的一个——音频驱动的数字人分支。
- 关键事实:
    * 2026-01-29 同时放出推理代码与权重（GitHub SkyworkAI/SkyReels-V3 + HF Skywork/）
    * 三个变体：R2V-14B（参考图到视频）、V2V-14B（视频到视频）、A2V-19B（音频到视频）
    * A2V-19B：19B 参数，720P（可降 540P/480P 省显存），24fps
    * A2V-19B 明确支持 talking avatar 最长 200 秒，「minute-long coherent videos」
    * 视频续接（video extension）：单镜续 5–30 秒
    * 多分辨率联合训练，支持 1:1 / 3:4 / 4:3 / 16:9 / 9:16
    * 显存：推荐 24GB+；--low_vram 走 FP8 weight-only 量化 + block offload
    * 前代 SkyReels-V2（2025-04-21）：1.3B/5B/14B，Diffusion Forcing 自回归，理论无限长，实例做到 60 秒(1457帧)；540P+1.3B 约 14.7GB，540P+14B 约 43.4–51.2GB
    * SkyReels V4 已在 Arena 上（Elo 1095），但未见开源权重
- 长片作用: 这是「长时段数字人说话」工位上目前唯一的开源权重答案。A2V-19B 单次 200 秒 talking avatar 的能力，在所有开源模型里独一份——H3 是 15 秒、LTX-2.5 是 121 帧、MAGI-2 是 10 秒。如果你的 2-8 分钟片里有大段口播/对白镜头，这个模型可以直接一镜到底吃掉 2-3 分钟，不需要分镜拼。SkyReels-V2 的 Diffusion Forcing 自回归（理论无限长、实测 60 秒）则是长镜头续接的备选路线。画质不如 H3，但它顶的工位 H3 顶不了。
- 来源: https://huggingface.co/Skywork/SkyReels-V3-A2V-19B https://github.com/SkyworkAI/SkyReels-V3 https://github.com/SkyworkAI/SkyReels-V2

## MAGI-2-preview / MAGI-1.1  (Sand.ai)
- 类别/成熟度: open-weights / **research** | 许可: Apache 2.0（MAGI-2-preview HF 卡面标注）。注意 Artificial Analysis 榜单把 MAGI-2 Preview 标为非开源权重，与 HF 实际情况矛盾——以 HF 为准，但这是个需要自己下载验证的点。 | 成本: MAGI-2：8×H100/H800 起步，单机 80GB 卡跑不动 1080p 全流程。MAGI-1 4.5B 蒸馏量化版 12GB 可跑，是唯一消费级可用分支。
- 是什么: MAGI-2-preview 是 114B 总参 / 每 token 激活约 6B 的 MagiMoE 统一音视频模型，单流设计（文本+视频+音频 token 拼成一条序列过同一个 Transformer backbone）。MAGI-1 系是 chunk-by-chunk 自回归（每 24 帧一块并发去噪），理论无限长。
- 关键事实:
    * MAGI-1 首发 2025-04-21；MAGI-1.1 24B 权重（含蒸馏前/蒸馏后/量化）2026-06-17 开源
    * MAGI-2-preview 2026-08-05 发布，HF sand-ai/MAGI-2-preview 权重可下载，约 307GB
    * MAGI-2：114B 总参，每 token 激活约 6B；只支持 10 秒一种时长；1088×1920（1080p 档）；音频用 Stable Audio Open VAE 生成并 mux 进输出
    * MAGI-2 硬件门槛：需要 8 张 NVIDIA Hopper；1080p 下 preview + refiner 两阶段在 80GB 卡上也放不下，必须 offload
    * MAGI-1 变体：24B（8×H100/H800）、4.5B（单张 4090，至少 24GB；蒸馏+量化版 12GB 可跑）
    * MAGI-2 Preview 在 Artificial Analysis T2V Arena Elo 1156（总榜第 6），I2V 带音频榜 Elo 1096（开源侧第 2）
    * MAGI-1 系 chunk 自回归支持精确时序控制与可扩展序列长度
- 长片作用: MAGI-2 现在是「研究性能力上限」而非产线工位——Elo 1156 排开源第 3，但 8×Hopper 的门槛让它进不了任何非超算产线，而且只支持 10 秒一档。真正有产线价值的是 MAGI-1 系的 chunk-by-chunk 自回归范式：它是长片续镜（把一个镜头无缝延长、或跨镜头保持运动连续）的技术路线参照，4.5B 蒸馏量化版 12GB 可实验。建议只做技术储备，不进主产线。
- 来源: https://huggingface.co/sand-ai/MAGI-2-preview https://github.com/SandAI-org/MAGI-1 https://sand.ai/blog/magi-2-preview

## HunyuanVideo-1.5 (8.3B)  (腾讯混元)
- 类别/成熟度: open-weights / **usable** | 许可: Tencent Community License（腾讯社区许可）。可商用，但有 MAU 门槛限制（社区普遍引用「1 亿 MAU 以下可商用」，需自行核对 LICENSE 原文）。 | 成本: 14GB（offload）/ 13.6GB(BF16 满载) / 6–8GB(GGUF)。4090 蒸馏版 75 秒出 480p I2V 片，是 24GB 以下卡上性价比最高的选择之一。
- 是什么: 腾讯把 HunyuanVideo 1.0 从 13B 砍到 8.3B 的轻量版 DiT，主打消费级显卡可跑。引入 SSTA（Selective and Sliding Tile Attention）把推理速度接近翻倍。
- 关键事实:
    * 2025-11-20 放出推理代码与权重（GitHub Tencent-Hunyuan/HunyuanVideo-1.5）
    * 8.3B 参数（1.0 版是 13B）
    * 480p / 720p 原生，默认 121 帧（可配 --video_length），支持超分到 1080p
    * 最低显存 14GB（开 model offloading）；BF16 全模型约 13.6GB；GGUF 量化版社区做到 6–8GB
    * 2025-12-05 放出 step-distilled 模型 + 训练代码 + LoRA 微调脚本
    * 2025-12-23 加入 FP8 GEMM 推理支持
    * 蒸馏后 480p I2V 在 RTX 4090 上端到端提速 75%，75 秒内出片
    * 720p 10 秒视频相对 FlashAttention-3 有 1.87× 提速
    * 截至 2026-09，腾讯 HF 主页最近更新的模型是 WeVisDoc / EVIE / AuK / Ex-Omni，未见 HunyuanVideo 2.0 或 1.5 之后的新视频基座
- 长片作用: 「低成本批量工位」。8.3B + 14GB 显存 + 75 秒出片，意味着你可以在一台 4090 甚至多台便宜卡上并行跑几十路，适合出大量 B-roll、空镜、转场镜头这些不需要顶级画质的素材。官方在 2025-12 就放了 LoRA 微调脚本，生态成熟度中等偏上。但注意：它已经快一年没有新版本（1.0→1.5 之后停更），画质相对 H3/LTX-2.5 明显落后一代，不建议做主镜基座。
- 来源: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 https://x.com/TencentHunyuan/status/1991721236855156984 https://huggingface.co/tencent

## HunyuanVideo-Avatar  (腾讯混元 + 腾讯音乐天琴实验室)
- 类别/成熟度: open-weights / **usable** | 许可: Tencent 系许可（需核对仓库 LICENSE，非标准开源协议） | 成本: 与 HunyuanVideo 主干同级，需核实具体配置；社区普遍在 24GB 卡上跑单角色模式
- 是什么: MM-DiT 架构的语音驱动数字人模型：一张人像照 + 一段音频 → 会说会唱的视频，自动识别场景语境与情绪，支持写实/卡通/3D渲染/拟人多风格，支持多角色对话。
- 关键事实:
    * 2025-05-28 放出推理代码与权重（GitHub Tencent-Hunyuan/HunyuanVideo-Avatar，HF tencent/HunyuanVideo-Avatar）
    * 单角色模式已开源；多角色模式官方称「即将开源」，截至 2026-09 仍未见发布
    * 官方站点单次支持最长 14 秒音频输入
    * arXiv 论文 2505.20156
    * 已在腾讯音乐娱乐集团多个 App 内部署
- 长片作用: 数字人口型工位的老牌方案，但 14 秒的音频上限使它在 2-8 分钟长片里只能做碎片化对白镜头，不能一镜到底。已被 SkyReels-V3-A2V-19B（200 秒）全面超越。现在的价值主要是多风格支持（卡通/3D/拟人角色）——如果你的虚构角色不是写实人类，它的风格覆盖面仍有用。另外「多角色对话」承诺跳票一年多，不要押注。
- 来源: https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar https://huggingface.co/tencent/HunyuanVideo-Avatar https://arxiv.org/html/2505.20156v2

## Wan-Dancer-14B  (阿里巴巴 通义实验室)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 8×A800 80GB 官方测试配置；单卡最低配未公布，属于集群级模型
- 是什么: 音乐驱动的长时长舞蹈视频生成框架，用「分层关键帧规划 + 局部时序精修」解决超过 20 秒后的时序漂移、身份不一致、动作重复三大问题。
- 关键事实:
    * HF 上 Wan-AI/Wan-Dancer-14B 更新于 2026-07-17，GitHub Wan-Video/Wan-Dancer 同期
    * 14B 参数
    * 生成 720p / 30fps、时长超过 1 分钟的视频
    * 明确针对「常规方法在 20 秒后崩坏」这一问题设计
    * 在 8×NVIDIA A800 80GB 上测试，未公布单卡最低显存
- 长片作用: 两层价值。第一层是直接用：如果长片里有舞蹈/律动/表演段落，它能一镜出 1 分钟以上 720p/30fps。第二层更重要——它的「分层关键帧规划 + 局部时序精修」正是你要建的分镜拼接系统该抄的架构：先规划全局关键帧保证结构与身份一致，再局部精修填充。这是 Apache 2.0 开源的、已经跑通的长视频一致性工程方案，比论文级的 GroundShot/MultiShotMaster 更可直接参考。
- 来源: https://github.com/Wan-Video/Wan-Dancer https://huggingface.co/Wan-AI

## Wan-Animate-2 / Wan2.2-Animate-2-14B  (阿里巴巴 通义实验室)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: 720P 需 8×A800；480P 需 2×A800。蒸馏版 10 步、无 CFG，是唯一接近实时的分支。
- 是什么: 角色动画/角色替换框架。相比 Wan2.2-Animate，新版重新设计 DiT 直接消费驱动视频，去掉了中间的动作提取器（motion extractor），从而拿到更高保真的动作生成和更强的身份保持，并加入文本驱动的视角控制。
- 关键事实:
    * 2026-08-07 发布（GitHub Wan-Video/Wan-Animate-2）；HF Wan2.2-Animate-2-14B 更新 2026-08-09，Diffusers 版 2026-08-13
    * 14B 参数
    * 面向 8×A800 做 720P；2×A800 可跑 480P
    * 两个版本：base（40 步）与 distillation（10 步，无需 CFG）
    * 蒸馏版达到「streaming character animation 的实时阈值」
    * 提供原生推理 / Diffusers / Gradio demo 三种部署
- 长片作用: 「角色一致性替身」工位，是分镜拼接系统里最容易被低估的一环。长片的核心痛点是同一个虚构数字人在几十个分镜里长得不一样。Wan-Animate-2 的做法是反过来：先用常规手段（甚至真人替身动作参考或 3D 动画）拍出运动，再用它把角色形象贴上去，身份保持由模型的 identity preservation 负责而不是靠 LoRA 碰运气。Apache 2.0 + 蒸馏版准实时，是目前最工程化可靠的跨镜头身份一致方案。
- 来源: https://github.com/Wan-Video/Wan-Animate-2 https://huggingface.co/Wan-AI

## NVIDIA Cosmos 3 (Super 64B / Nano 16B / Edge 4B)  (NVIDIA)
- 类别/成熟度: open-weights / **production** | 许可: OpenMDW-1.1 License（NVIDIA 开放模型许可），另可联系 cosmos-license@nvidia.com 定制授权。无地域/营收限制条款。 | 成本: Super 64B 需 H200/B200/GB200 级；Nano 16B 可在 H100 或 RTX Pro 6000 上跑；Edge 4B 可在 Jetson / RTX Pro 上跑
- 是什么: NVIDIA 的世界模型系列，面向物理仿真、机器人策略学习、自动驾驶与具身智能的合成数据生成，不是影视向的美学生成模型。
- 关键事实:
    * Cosmos 3 于 2026-05-31 发布
    * 三档：Cosmos3-Super 64B（H200/B200/GB200 数据中心）、Cosmos3-Nano 16B（RTX Pro 6000 / H100 / B200）、Cosmos3-Edge 4B（Jetson AGX Orin / RTX Pro）
    * 分辨率 256p / 480p / 720p；帧率 10/16/24/30 fps；长度 5–300 帧（Edge 限 50–150 帧）
    * 宽高比 16:9 / 4:3 / 1:1 / 3:4 / 9:16
    * Artificial Analysis I2V 开源权重榜（无音频）：Cosmos3-Super-Image2Video-4Step Elo 1274、Cosmos3-Super-Image2Video Elo 1253，仅次于 MiniMax H3 的 1354
- 长片作用: 不是数字人拍片的主基座——它的训练目标是物理正确而不是电影感，没有对白音频，也几乎没有角色向 LoRA 生态。但 I2V 榜上 Elo 1253–1274 说明它的图生视频运动质量很硬。在 2-8 分钟产线里它适合的是「物理真实的环境/道具/载具空镜」工位：车流、水、布料、机械运动这类靠美学模型容易穿帮的镜头。许可证干净（OpenMDW-1.1，无地域限制），可作为 H3 地域受限时某些工位的替代。
- 来源: https://github.com/NVIDIA/Cosmos https://artificialanalysis.ai/video/leaderboard/image-to-video/open-weights

## Open-Sora 2.0 (11B)  (HPC-AI Tech (潞晨科技))
- 类别/成熟度: open-weights / **research** | 许可: Apache 2.0 | 成本: 单卡 256×256 就要 52.5GB 峰值，显存效率极差；768×768 需 8 卡
- 是什么: 号称 $200K 训练成本复现商业级视频模型的开源项目，11B 参数，对标当时的 HunyuanVideo 11B 与 Step-Video 30B。
- 关键事实:
    * Open-Sora 2.0 发布于 2025-03-12，此后 18 个月未见新主版本
    * 11B 参数
    * 分辨率最高 768px，支持 16:9 / 9:16 / 1:1 / 2.39:1；帧数 4k+1 到 129 帧
    * 训练成本 $200K
    * 显存：256×256 单卡峰值 52.5GB；768×768 在 8 卡上每卡 44.3GB
    * 未出现在 Artificial Analysis 前 25 名榜单
- 长片作用: 不要用。2025-03 之后停更，显存效率被同期所有模型吊打（256×256 要 52.5GB，而 HunyuanVideo-1.5 出 720p 只要 14GB），没有 LoRA 生态，没上任何 2026 榜单。它的历史价值是证明了低成本训练路线，对你的产线没有工位。
- 来源: https://github.com/hpcaitech/Open-Sora

## CogVideoX / CogVideoX1.5-5B  (智谱 AI (THUDM))
- 类别/成熟度: open-weights / **research** | 许可: 2B Apache 2.0；5B 自有 CogVideoX LICENSE；代码 Apache 2.0 | 成本: INT8 最低 7GB，BF16 最低 10GB — 显存门槛全场最低
- 是什么: 清华/智谱的早期开源视频 DiT，曾是 2024 年消费级可跑的代表方案。
- 关键事实:
    * 最新开源版本 CogVideoX1.5-5B 发布于 2024-11-08，此后近两年无新开源版本
    * 三档：2B / 5B / 1.5-5B
    * CogVideoX1.5-5B：1360×768，帧数 16N+1(N≤10)，默认 81 帧，16fps，支持 5 秒与 10 秒
    * 显存：diffusers + 优化 BF16 最低 10GB；INT8 量化最低 7GB；SAT 版 BF16 需 76GB
    * 许可：CogVideoX-2B 为 Apache 2.0；5B 系列为自有 CogVideoX LICENSE；代码 Apache 2.0
    * 16fps 是硬伤（同代模型已普遍 24fps）
- 长片作用: 已出局。16fps、2024-11 停更、画质落后两代。唯一残余价值是 7GB 显存门槛（全场最低），可在极低配机器上做流程验证/占位素材，但不值得为它建产线工位。
- 来源: https://github.com/THUDM/CogVideo

## Step-Video-T2V (30B) / Step-Video-TI2V  (StepFun (阶跃星辰))
- 类别/成熟度: open-weights / **research** | 许可: MIT — 全场最宽松，无任何商用/地域/营收限制 | 成本: 80GB+ 单卡（A100-80G / H100-80G），单片 860 秒（约 14 分钟），性价比极低
- 是什么: 30B 参数的大体量开源 T2V，MIT 许可。TI2V 是其图生视频分支。
- 关键事实:
    * Step-Video-T2V 发布 2025-02-17；Step-Video-TI2V 发布 2025-03-17；此后未见新版本
    * 30B 参数
    * 分辨率 768×768 与 544×992；最长 204 帧
    * 显存：768×768×204f 峰值 78.55GB；544×992×204f 峰值 77.64GB；建议 80GB+
    * 推理耗时：768×768×204f，50 步带 flash-attention 约 860 秒；不带约 1437 秒；Turbo 版建议 10–15 步
    * MIT License — 全场最宽松许可
    * 未出现在 Artificial Analysis 前 25 名榜单
- 长片作用: 不进产线。80GB 卡上 14 分钟出一个 8.5 秒片，在 2-8 分钟长片（几十上百镜）的量级下经济性完全不成立，且已停更一年半。MIT 许可是它唯一的独特优势——如果有极端严格的法务要求（连 Apache 2.0 的专利条款都不能接受），它是理论后备，但实际不推荐。
- 来源: https://github.com/stepfun-ai/Step-Video-T2V

## Genmo Mochi-1  (Genmo)
- 类别/成熟度: open-weights / **research** | 许可: Apache 2.0（480p 预览版） | 成本: 社区方案多卡或 4090 + 量化，但已无实际意义
- 是什么: 2024-10 发布的 Apache 2.0 开源 T2V 研究预览版，当时是开源 SOTA。
- 关键事实:
    * 研究预览版发布于 2024-10，Apache 2.0，可个人与商用
    * 当前公开版本仅 480p
    * 承诺于 2024-10 的 720p Mochi 1 HD 变体，截至 2026 年初仍未以开放权重发布
    * GitHub genmoai/mochi 仓库此后无重大更新
    * 未出现在任何 2026 年榜单
- 长片作用: 事实上已废弃。480p、两年停更、HD 版跳票近两年。不要为它分配任何工位。它在 2024 年的历史地位不等于 2026 年的可用性——这是选型时最容易犯的错。
- 来源: https://github.com/genmoai/mochi https://www.genmo.ai/blog/mochi-1-a-new-sota-in-open-text-to-video

## Rhymes AI Allegro / PixArt 系  (Rhymes AI / PixArt (华为诺亚等))
- 类别/成熟度: open-weights / **research** | 许可: 需逐一核实；Allegro 原始为 Apache 2.0 系 | 成本: 不适用（已无产线价值）
- 是什么: Allegro 是 2024-10 与 Mochi-1 同期发布的小体量开源 T2V。PixArt 系本质是图像生成模型（PixArt-α/Σ），不是视频基座。
- 关键事实:
    * Allegro 发布于 2024-10，仅 6 秒、15 FPS、720p
    * 此后无重大版本更新，未出现在任何 2026 年榜单
    * PixArt 系（α / Σ）是文生图 DiT，不属于视频生成基座，把它列入视频模型对比是范畴错误
- 长片作用: 两者都不进产线。Allegro 6秒/15fps 的规格在 2026 年完全不可用。PixArt 是图像模型——如果你需要的是分镜首帧/关键帧的图像生成，那是另一条技术线（且 2026 年该工位应看 Qwen-Image / FLUX 系而非 PixArt），不要和视频基座混在一张表里比。
- 来源: https://aibrews.substack.com/p/claudes-computer-use-mochi-1-and

## 多镜头一致性框架（GroundShot / MultiShotMaster / Memento）  (学界 + KlingAI Research)
- 类别/成熟度: framework / **research** | 许可: 各项目独立，MultiShotMaster 有官方实现仓库 | 成本: GroundShot 为 training-free，额外成本主要是调度与多轮推理；其余需核实
- 是什么: 专门解决「多镜头长视频里实体/角色跨镜不一致」的框架层方案，架在基座模型之上，不替代基座。
- 关键事实:
    * GroundShot (arXiv 2606.20799)：training-free、model-agnostic 的 agentic 框架，做 entity-grounded 的分镜调度，无需额外训练或改模型
    * MultiShotMaster (KlingAIResearch/MultiShotMaster, CVPR 2026 收录)：支持文本驱动的镜头间一致性、可变镜头数与镜头时长、定制主体+运动控制、背景驱动的定制场景
    * Memento (arXiv 2606.14667)：Reconstruct to Remember，做长视频一致性
    * EntityBench (arXiv 2605.15199)：面向实体一致的长程多镜头视频生成的评测基准
- 长片作用: 这正是你要建的「分镜系统」本身的工位。基座只管 4–15 秒单镜，2-8 分钟的连贯性 100% 由这一层负责。GroundShot 的 training-free + model-agnostic 特性最值得优先试——它不绑定基座，理论上可以套在 MiniMax H3 上。MultiShotMaster 有 CVPR 2026 背书和官方实现，且直接支持可变镜头数/时长，与你的分镜拼接需求最对口。EntityBench 应该作为你系统的内部验收指标。注意：这些都是 research 级，需要自己做工程化。
- 来源: https://arxiv.org/abs/2606.20799 https://github.com/KlingAIResearch/MultiShotMaster https://arxiv.org/pdf/2606.14667

## Seedance 2.0 / 2.5（硬约束核实结果）  (字节跳动 Seed)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API | 成本: 按 API 调用计费（fal / BytePlus / 火山引擎各有价目，需按实际通道核价）
- 是什么: 用户给定的对标目标。核实结论：用户列出的硬约束基本成立，但有两处需要更新。
- 关键事实:
    * 确认闭源：Seedance 2.0 为 ByteDance 闭源 API，无权重发布，不能本机 LoRA — 用户表述正确
    * 确认时长：Seedance 2.0 单次生成上限 15 秒，720p，多宽高比 — 用户「4-15 秒」表述正确
    * 需更新：Seedance 2.0 在 15 秒内可产生多镜头带自然切换与转场（官方能力），不是纯单镜
    * 确认多模态参考：接受文本 + 参考图 + 音频 + 视频输入组合 — 与用户「9图+3视频+3音频」量级一致（具体上限数字未在公开文档逐条核实到）
    * 确认自带对白声：统一多模态音视频架构，原生出声
    * 分发通道：fal.ai（2026-04-09 上线）、BytePlus（国际）、火山引擎（中国）；官方通道有内容审核 — 用户表述正确
    * Artificial Analysis T2V Arena：Dreamina Seedance 2.0 720p Elo 1210，总榜第 5
    * 关键对比：开源权重的 MiniMax H3（Elo 1220）在该榜上已略高于 Seedance 2.0 720p
- 长片作用: 作为对标基准它依然有效，但结论要改：2026-09 的开源侧已经追上来了。你不需要在「接近 Seedance 观感」和「能本机 LoRA」之间二选一——MiniMax H3 在盲测 Elo 上已经与 Seedance 2.0 720p 持平甚至略高，且权重开放。这改变了整个平台的架构决策：主产线应该建在本地 H3 上而不是 Seedance API 上，后者只在需要 2K 或特定镜头补拍时作为外挂。
- 来源: https://fal.ai/seedance-2.0 https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5 https://artificialanalysis.ai/video/leaderboard/text-to-video

### 建议
【直接结论】2026-09 的长片主力基座是 MiniMax H3（33B，2026-08-03 开放权重），条件是你的部署地不在美/欧/英/韩、年营收在 $20M 以下、且能接受在 UI 标注「MiniMax H3」并放弃蒸馏权利。它同时满足你的两个核心要求：(1) 画质——Artificial Analysis 盲测 T2V Arena Elo 1220，开源权重第一，总榜第 4，已经略高于 Dreamina Seedance 2.0 720p 的 1210，与榜首 Gemini Omni Flash(1233) 只差 13 分；(2) LoRA 生态——权重发布当天 Ostris AI-Toolkit 就上了完整 T2V/I2V 训练链路，Civitai 为它单开模型类别并办官方训练赛，6 周内社区发布 670+ 个衍生模型（对比 Wan2.2 累积 368–475 个），Turbo LoRA 已把 20 步压到 4–6 步。

【但不要建单基座产线】2-8 分钟片子是多工位流水线，建议这样分配：
1. 主镜头（有台词、有角色、观感要求最高）→ MiniMax H3 pruned INT8，24GB 卡可跑，4090 约 3–10 分钟/5秒，5090 约 3 分钟。自带 32kHz 立体声，省掉 TTS + 对口型两道工序。注意本地权重原生只有 768p，2K 必须走官方 API 或自己接超分。
2. 长段口播/独白（一镜 1–3 分钟）→ SkyReels-V3-A2V-19B（2026-01-29 开源）。它是全场唯一支持 200 秒 talking avatar 的开源权重模型，24GB+ 可跑（--low_vram 走 FP8）。这类镜头不要用 H3 硬拼，15 秒一段拼 2 分钟必崩。
3. 整片节奏预演 + 后期工具链 → LTX-2.5（22B，2026-01-06）。RTX 5090 上 25 秒出 4 秒 720p，是唯一能支撑「一天内把整片粗排跑三遍」的基座；它官方维护的 IC-LoRA 套件（Clean-Plate / Deblur / Colorization / Day-To-Night / Pixel-Spatial-Upscaler / Slow-Motion-Control）直接就是一条视频后期产线，这是别家没有的。
4. 跨镜角色身份一致 → Wan-Animate-2（2026-08-07，Apache 2.0）。它去掉了中间 motion extractor，identity preservation 由模型负责而不是靠 LoRA 碰运气；蒸馏版 10 步无 CFG 接近实时。这比「每个镜头都祈祷 LoRA 出同一张脸」可靠得多。
5. 分镜拼接系统本体 → 抄 Wan-Dancer 的「分层关键帧规划 + 局部时序精修」架构（Apache 2.0，已跑通 720p/30fps 超 1 分钟），配合 GroundShot（training-free、model-agnostic，可直接套在 H3 上）和 MultiShotMaster（CVPR 2026，支持可变镜头数/时长），用 EntityBench 做内部验收。
6. 物理真实的环境/载具/流体空镜 → Cosmos 3（2026-05-31，OpenMDW-1.1，无地域限制），I2V 开源榜 Elo 1253–1274。

【合规分叉：如果你要出海】MiniMax H3 的许可证明确排除美/欧/英/韩的本地部署权，还禁止蒸馏、要求 UI 署名。如果你的平台要卖给这些地区的客户或做全球分发，主基座必须换成 Wan 2.2 A14B（Apache 2.0，零附加条件，但画质落后一代，A14B 需 80GB 或 4090 上走 GGUF/fp8 分块 offload）或 LTX-2.5（<$10M 营收免费，无地域限制）。这个决策必须在动工前定，不要先按 H3 建完再发现法务卡死。

【明确排除，不要浪费时间】Wan 2.5/2.6/2.7/3.0 全部闭源无权重（市面上「Wan 2.5 开源」的中英文博文是 SEO 农场错误内容，Wan 3.0 已于 2026-08-24 全量上线但仍是 API）；CogVideoX（2024-11 停更，16fps）、Mochi-1（480p，HD 版跳票两年）、Open-Sora 2.0（2025-03 停更，256×256 就要 52.5GB）、Allegro（6秒/15fps）已全部出局；Step-Video-T2V 虽是全场最宽松的 MIT，但 80GB 卡上 14 分钟出 8.5 秒片，长片经济性不成立；MAGI-2-preview 虽是 Apache 2.0 且 Elo 1156，但需要 8×Hopper 且只支持 10 秒一档，只做技术储备；HunyuanVideo-1.5 可作为低成本 B-roll/空镜的并行批量工位（14GB，4090 蒸馏版 75 秒出片），但已近一年未更新，不做主镜；PixArt 是图像模型，属范畴错误，分镜首帧那条线应另行调研。

【对你原始假设的一处修正】你把 Seedance 2.0 当成不可企及的画质上限、只能靠 API。这个前提在 2026-09 已经不成立了：开源权重的 MiniMax H3 在盲测 Elo 上已经与它持平甚至略高。所以架构应该反过来——主产线建在本地 H3 + LoRA 上，Seedance API 只在需要 2K 或临时补拍时作为外挂通道。这同时绕开了官方通道内容审核对虚构成年角色设定的误判风险。

### 存疑
- MiniMax H3 在 4090 上出 5 秒 768p 片的「3–10 分钟」是社区实测区间的推断值，官方未发布 A100/H100/4090 的标准化耗时表。32GB RTX 5090 约 3 分钟这个数字来自第三方博客而非官方 benchmark。
- HunyuanVideo-1.5 的 Tencent Community License「1 亿 MAU 以下可商用」这一门槛来自第三方报道（WinBuzzer 等），未从 GitHub 仓库 LICENSE 原文逐条核实。做法务决策前必须自己读原文。
- SkyReels-V3 使用的 skywork-license 具体商用条款（是否有营收/MAU/地域限制）未核实到原文，HF 卡面只显示许可证名称。
- Artificial Analysis 榜单把 MAGI-2 Preview 标为「非开源权重」，但 HuggingFace sand-ai/MAGI-2-preview 卡面标注 Apache 2.0 且有约 307GB 权重可下载。两处信息矛盾，需实际下载验证。同理 Artificial Analysis 也未把 Cosmos3/MAGI-2 在 T2V 榜上标为开源权重，其开源标记口径可能与权重实际可得性不一致。
- LTX-2.5 的「2026-01-06 发布」来自 HF 模型卡摘要，未在 Lightricks 官方发布公告中二次确认。另外 LTX-2.5 与 LTX-2 / LTX-2.3 的版本关系、以及 LTX-2 原始权重是否最终开放，链路未完全理清（GitHub Lightricks/LTX-Video 的 README 仍停留在 v0.9.8 时代，明显未同步）。
- 用户给出的 Seedance 2.0 多模态参考上限「最多 9 图 + 3 视频 + 3 音频」，我只核实到它接受文图音视频组合输入，未在官方文档中逐条核实到这三个具体数字。建议直接查 BytePlus/火山引擎 API 文档确认。
- Seedance 2.5 的「约 30 秒」时长：搜索结果提到 Seedance 2.5 存在（seed.bytedance.com 有 one-take creation 博文，EvoLink 称 2026-08-07 上线），但未核实到其官方单段时长上限确为 30 秒。
- 「SkyReels V4」在 Artificial Analysis T2V 榜上 Elo 1095，但榜单标为非开源权重，且未找到 V4 的 GitHub/HF 权重发布，其开源状态未定。
- 「HappyHorse-1.0/1.1」(Elo 1119/1147) 和 "Agnes-Video-2.5"(1076) 出现在榜单前 25 但来源不明，未能确认其厂商与开源状态，因此未写入 findings。
- Wan-Animate-2 与 Wan-Dancer 的单卡最低显存未公布，官方只给了 8×A800 / 2×A800 的测试配置。社区是否有 4090 可跑的量化方案未核实。
- Open-Sora 项目在 2025-03 之后是否仍有实质性维护，WebFetch 返回的「持续活跃至 2026」与「最新 release 为 2025-03」自相矛盾，我按后者（有明确日期的事实）判定为停滞。
- MiniMax H3 是否存在官方 GGUF 量化发布未核实；搜索只见到社区 pruned INT8 与 NVFP4 版本。

### 事实核查修正
- [WRONG] [LTX-2.5 (22B)] LTX-2.5 主模型 2026-01-06 发布；HF Lightricks 组织共 66 个模型
  → 版本张冠李戴。HF API 显示 Lightricks/LTX-2.5 的 createdAt = 2026-07-23T07:55:24Z（lastModified 2026-09-01），LTX-2.5-Diffusers createdAt = 2026-07-26。2026-01-06 / arXiv 2601.03233 对应的是 LTX-2：GitHub Lightricks/LTX-2 仓库 created_at = 2026-01-03T13:16:29Z，HF Lightricks/LTX-2 createdAt = 2026-01-0 https://huggingface.co/api/models/Lightricks/LTX-2.5
- [WRONG] [MiniMax H3] BF16 原始 checkpoint 123.6GB；社区量化 pruned INT8 约 21GB / 42.5GB 多个版本
  → 123.6GB 在仓库里找不到对应物。按 HF blobs API 实测 MiniMaxAI/MiniMax-H3：单个 transformer 目录 = 66.28 GB（33B × BF16，数学自洽）；text_encoder = 66.73 GB；video_vae = 10.42 GB；audio_vae = 0.61 GB；即一个完整可跑 checkpoint（如 FL2VA）≈ 144.0 GB，整仓（FL2VA + Ref2VA + diffusers 版 transformer/transformer_ref + 共享组件）= 498. https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3?blobs=true
- [UNVERIFIABLE] [MiniMax H3] Artificial Analysis Text-to-Video Arena：Elo 1220，全部开源权重模型第一；总榜第 4，仅次于 Gemini Omni Flash(1233)/Wan 3.0(1229)/MiniMax H3 Max(1227)，且高于 Dreamina Seeda
  → 无法取得一手来源。artificialanalysis.ai/text-to-video/arena 308 重定向到 /video/arena，该页 395KB HTML 中不含任何 Elo 数值或视频模型榜单（全部由 JS 客户端拉取），/video/arena/text-to-video、/text-to-video/arena/leaderboard、/api/v2/data/text-to-video/arena 等端点均返回 404 或 SPA 外壳。本次核实中 grep 整页只能搜到「Hailuo AI (MiniMax)」「LTX」两个裸 https://artificialanalysis.ai/video/arena
- [UNVERIFIABLE] [MiniMax H3] Image-to-Video 开源权重榜同样第一：带音频 1181，不带音频 1354
  → 同上，AA 榜单数据无法从页面取得。另外这两个数字内部不自洽：同一模型两个模式差 173 分（不带音频 1354 反而比 T2V 榜首 1220 还高 134 分），而 AA 的 Arena Elo 是同一池内配对比较，跨模式出现这种量级的跳变需要额外解释。建议直接从 AA 官方导出或截图取数，不要二手引用。 https://artificialanalysis.ai/video/arena
- [UNVERIFIABLE] [LTX-2.5] Artificial Analysis T2V Arena：LTX-2.5 Fast Elo 1055 / Pro 1053，开源权重第 2（落后 H3 约 165 分）
  → 同上，AA 榜单不可取数。另需注意「LTX-2.5 Fast / Pro」是 Lightricks 托管 API 的档位命名，与开放权重档（LTX-2.5 dev / distilled，HF 上 ltx-2.5-22b-dev / ltx-2.5-22b-distilled）不是同一实体——拿 API 档位的 Elo 去代表开放权重的能力，本身就是「把闭源档位说成开源」的典型口径错误，即便分数拿到了也要分开标注。 https://huggingface.co/Lightricks/LTX-2.5
- [UNVERIFIABLE] [Wan 2.5/2.6/2.7/3.0] Wan 3.0 在 Artificial Analysis T2V Arena Elo 1229，总榜第 2（高于 Seedance 2.0 720p 的 1210）
  → Elo 数值同样无法核实（见上）。不过「Wan 3.0 是闭源 API-only」这半句已独立证实：阿里云百炼模型列表里存在 wan3.0-video，而 HF Wan-AI 组织下最新开放权重是 Wan2.2-Animate-2-14B-Diffusers（2026-08-13），无任何 3.0 权重。 https://help.aliyun.com/zh/model-studio/models
- [UNVERIFIABLE] [Wan 2.5/2.6/2.7/3.0] Wan 3.0：2026-08-06 公测、2026-08-24 全量上线，仍是闭源 API；支持 30 秒单镜、文档到视频
  → 「闭源 API」部分 CONFIRMED（阿里云百炼列出 wan3.0-video；HF/GitHub/ModelScope 无权重）。但 2026-08-06 公测、2026-08-24 全量、30 秒单镜、文档到视频这四项都没能从阿里云官方文档核实到。反证是：阿里云文生视频 API 参考页当前主文档给出的模型是 wan2.7-t2v / wan2.7-t2v-2026-06-12，时长参数取值范围明确写着 [2, 15] 秒整数、默认 5 秒，分辨率 720P/1080P——没有 30 秒档。如果 30 秒确实存在，也应在 wan3.0-video  https://help.aliyun.com/zh/model-studio/text-to-video-api-reference
- [UNVERIFIABLE] [Wan 2.5/2.6/2.7/3.0] Wan 2.6、Wan 2.7 同样闭源 API-only
  → Wan 2.7 部分 CONFIRMED：阿里云百炼文生视频 API 文档列出 wan2.7-t2v 与快照版 wan2.7-t2v-2026-06-12，HF Wan-AI 组织下无 2.7 权重，确为 API-only。Wan 2.6 本次在阿里云文档中没有检索到任何对应模型名（列表里只出现 wan2.7-image-pro 与 wan3.0-video），Wan 2.6 是否真实存在过这一公开版本号未能证实，建议单独确认后再写入，否则有凭空补版本号的风险。 https://help.aliyun.com/zh/model-studio/text-to-video-api-reference
- [WRONG] [MiniMax H3 LoRA 生态] Civitai 已有独立模型类别，官方 H3 训练赛（2 Million Buzz 奖池）2026-09 进行中，上线数周社区已发布 670+ 个基于 H3 的模型
  → 「独立模型类别」CONFIRMED：Civitai API 的 baseModels 枚举里确实有官方分类 "MiniMax H3"。但 670+ 这个量级复现不出来：用 baseModels=MiniMax%20H3 游标翻页穷举，总计 319 个已发布模型；用 tag=minimax h3 穷举只有 123 个。即便加上其他 H3 相关 baseModel 变体也远达不到 670。「2 Million Buzz 官方训练赛」本次未能从 Civitai 一手页面核实，判 UNVERIFIABLE。 https://civitai.com/api/v1/models?limit=100&baseModels=MiniMax%20H3
- [WRONG] [MiniMax H3 LoRA 生态] 对比：Civitai wan2.2 标签约 368 个模型（另有社区整理的 475 个 Wan2.2 LoRA 合集，2026-03）
  → 368 不对，且没说清口径。实测：tag=wan2.2 穷举 = 253 个；baseModels="Wan Video 2.2 T2V-A14B" 穷举 = 523 个。两个口径都不等于 368。由于 Civitai 的 tag 与 baseModel 是两套体系（同一模型可能只挂其中之一），任何「H3 vs Wan2.2 生态规模对比」都必须用同一口径，否则结论方向都可能反转：按 baseModel 口径是 Wan2.2 (523) > H3 (319)，按 tag 口径也是 Wan2.2 (253) > H3 (123)——与原文暗示的「H3 生态 https://civitai.com/api/v1/models?limit=100&baseModels=Wan%20Video%202.2%20T2V-A14B
- [WRONG] [MiniMax H3 LoRA 生态] Turbo LoRA（lightx2v / drbaph 等多个版本）：把 20 步压到 4–6 步出可用画面，文件 <1GB
  → 仓库存在性 CONFIRMED（lightx2v/Minimax-h3-Turbo 1,531,171 下载、drbaph/MiniMax-H3-Turbo-Lora-ComfyUI、larryvrh/MiniMax-H3-Turbo-Lora 各 20 万级下载），但「文件 <1GB」和「4–6 步」两处都不准。实测文件体积：lightx2v 的 ComfyUI bf16 版 minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors = 1.96 GB（fp8 全量融合版 34.02  https://huggingface.co/api/models/lightx2v/Minimax-h3-Turbo?blobs=true
- [WRONG] [MiniMax H3 LoRA 生态] 已有融合微调产物如 Minimax-h3_Singularity ……以 ComfyUI diffusion model 形式加载，与官方 INT8 同槽位
  → 模型本身 CONFIRMED（WarmBloodAban/Minimax-h3_Singularity，217,900 下载、499 likes，另有 Abiray/MiniMax-H3-Singularity-GGUF、TenStrip/Minimax-h3_Singularity-Lora 衍生）。错在「官方 INT8」：MiniMaxAI 组织下只发布了 MiniMax-H3 一个仓库，498.47 GB 全部为 BF16 safetensors，不存在任何官方 INT8 权重。H3 的 INT8/convrot 版本全部来自社区（MATLOWAI https://huggingface.co/MiniMaxAI
- [UNVERIFIABLE] [LTX-2.5] 实测速度：RTX 5090(32GB) 出 4 秒 720p 约 25 秒；RTX 4090 跑 5 秒片实测显存 22.67 GiB；2×GB200 上 6.8 秒是官方 headline
  → GitHub Lightricks/LTX-2 官方仓库与 HF LTX-2.5 模型卡中均未给出任何 RTX 5090 / RTX 4090 / GB200 的速度或显存基准，只有量化档位（fp8-cast、fp8-scaled-mm for Hopper+、NVFP4、int8+convrot）与「降低显存占用」的定性描述。这三个数字应标注为第三方/社区实测并给出具体来源，否则不能作为官方 headline 引用。 https://github.com/Lightricks/LTX-2
- [UNVERIFIABLE] [LTX-2.5] stage1 默认 544×960，可上 1920×1088 或 4K(3840×2176) 超分
  → 4K = 3840×2176（注意不是 2160）以及 --spatial-upscalings 2 的用法在官方仓库中可证实；num_frames % 8 == 1、宽高整除 32、上限 121 帧也可证实。但「stage1 默认 544×960」与 Lightricks/LTX-2 仓库 README 给出的默认 1024×1536 @24fps 冲突，两处默认值口径不一致（可能分属 dev / distilled 管线或不同版本文档）。写进文档前需要指明是哪条管线的默认值。 https://github.com/Lightricks/LTX-2
- [WRONG] [SkyReels-V3] 三个变体：R2V-14B（参考图到视频）、V2V-14B（视频到视频）、A2V-19B（音频到视频）
  → 数量、参数量、日期都对（GitHub SkyworkAI/SkyReels-V3 News 明确写 2026-01-29 放出推理代码与权重；HF Skywork/SkyReels-V3-A2V-19B / -R2V-14B / -V2V-14B 均 2026-01-28 更新，A2V-19B createdAt 2026-01-19），但两个变体的能力描述错位：官方自述是 Reference-to-Video 14B-720P、Video Extension 14B-720P（视频延展/续写，单镜 5–30 秒、转场 5 秒上限），以及 Talking https://github.com/SkyworkAI/SkyReels-V3
- [WRONG] [调研员存疑项] 用户给出的 Seedance 2.0 多模态参考上限「最多 9 图 + 3 视频 + 3 音频」未能核实
  → 这组数字很可能被错归到了 Seedance 头上——它其实是 MiniMax H3 的规格。MiniMaxAI/MiniMax-H3 模型卡对 Ref2VA（Omni-Reference）checkpoint 的原文描述就是「supports up to 9 images, 3 video clips, 3 audio clips, or 12 mixed files」，与「9 图 + 3 视频 + 3 音频」逐项吻合。所以这条不是「Seedance 待查」，而是需要先排除串台；Seedance 2.0 的真实上限仍应去 BytePlus/火山引擎 AP https://huggingface.co/MiniMaxAI/MiniMax-H3

### 遗漏补充
- tencent/HunyuanVideo-1.5（2025-11-18 开源，HF 1036 likes）——调研员只在存疑项里提了它的 License 门槛，主 findings 完全没有这个条目。它是当前唯一定位「消费级显卡可跑」的一线开源视频基座，在 H3(144GB/checkpoint) 与 LTX-2.5(42GB transformer) 之间填了最关键的一档，做选型表不能漏。
- sand-ai/MAGI-2-preview（2026-08-03，Apache 2.0，306.72GB 真实权重，MoE + 原生音视频同步）——本次已验证权重可下载且许可证是全场最宽松的 Apache 2.0。H3 是自家 Community License、LTX-2.5 是 other、SkyReels-V3 是 NOASSERTION，MAGI-2 是这一维度里商用法务风险最低的选项，价值被 AA 的错误开源标记掩盖了。
- meituan-longcat/LongCat-Video（2025-10，MIT，HF 568 likes）——MIT 许可 + 原生视频续写/长视频方向，是除 SkyReels-V3 Video Extension 外唯一的开放权重长时长路线。
- Wan-AI/Wan-Dancer-14B（2026-07-17，85.67GB，global_model + local_model 双 34.49GB 结构）——Wan 系 2026 年最新开放权重之一，调研员只在存疑项里带了一句显存，没有列为方案。
- Wan-AI/Wan2.2-Animate-2-14B（2026-08-09，82.54GB，同时提供 wan_animate_2_bf16 与 wan_animate_2_bf16_distillation 两版，32.79GB/个）——这才是 Wan 组织当前最新的开放权重，且官方直接给了蒸馏版，落地成本远低于调研员讨论的 A14B。
- Lightricks/LTX-2.3 全家桶——HF 1,126,972 下载 / 1899 likes，官方还单独发了 LTX-2.3-fp8、LTX-2.3-nvfp4 量化仓库和 15 个 IC-LoRA（HDR、Relight、Union-Control、Motion-Track-Control、In-Outpainting、DubIt、Foley-V2A 等，其中 Relight/Union-Control/Motion-Track/Foley 在 2.5 线上还没有对应版本）。实际部署主力和功能覆盖面都强于 LTX-2.5，调研员把它当成被取代的旧版是误判。
- FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree（279,777 下载）——用 VSA 稀疏注意力 + 4 步蒸馏，正好补上 MiniMax 官方 README 里明说「初版开源只提供 full attention，稀疏注意力实现将来再发」的那块缺口。这是社区替官方补齐未开源组件的关键一环。
- 真实部署入口被漏：Comfy-Org/MiniMax-H3 的下载量是 20,277,946 次，比官方 MiniMaxAI/MiniMax-H3 的 4,449,605 高 4.5 倍；Comfy-Org/Wan_2.2_ComfyUI_Repackaged 是 5,791,673。用官方仓库下载量衡量生态热度会系统性低估，选型时也应直接指向 repackaged 版本。
- zai-org/SCAIL-2（2026-06-09，MIT，pose-driven 角色动画）——与 Wan2.2-Animate-2 正面对位的开放权重方案，许可证更干净，调研员的角色动画一节只有 Wan 一家。
- 单卡可跑的 Wan2.2 量化/蒸馏路线整条缺失：Phr00t/WAN2.2-14B-Rapid-AllInOne（1661 likes）、lightx2v/Wan2.2-Lightning、lightx2v/Wan2.2-Distill-Loras、QuantStack/Wan2.2-T2V-A14B-GGUF（728,697 下载）。调研员按官方 README 写「T2V/I2V-A14B 至少 80GB」是对的，但据此得出 A14B 消费级不可用则是错的。
- Skywork 组织页已挂出 SkyReels-V4（Multi-modal Video-Audio Generation, Inpainting and Editing），但 HF 上没有任何 V4 权重实例、GitHub 也无发布——正好坐实调研员「V4 开源状态未定」的怀疑，应明确写成「已预告、未放权重」而不是留空。
- 对照组基座缺失：stepfun-ai/stepvideo-t2v、nvidia/Cosmos-1.0-Diffusion-7B、genmo/mochi-1-preview、zai-org/CogVideoX-5b。做「开源视频基座 SOTA」的时间线时需要它们作为 2024–2025 基准，否则无法说明 2026 年这一代的提升幅度。


====================================================================================================
# [orchestration-infra] AI 视频生产的编排层工程实践（ComfyUI 服务化 / 任务队列 / 多供应商容灾 / 分镜结构化 / Emily2040 仓库查证）

## Emily2040/seedance-2.0  (Emily2040（个人开发者）)
- 类别/成熟度: technique / **usable** | 许可: MIT | 成本: 0（纯文本 skill，无推理成本）
- 是什么: 真实存在的 GitHub 仓库。它不是代码库也不是 SDK，而是一个面向 Agent 的「提示词操作系统」：一套 Markdown 规则 + 5 个 JSON Schema，教 Agent 把创意拆成分镜、把参考素材绑定到 9 个语义槽位、再编译成 Seedance 的 T2V/I2V/V2V/R2V/FLF2V/edit/extend 提示词。它本身不调用任何 API、不生成视频。
- 关键事实:
    * 仓库确实存在：github.com/Emily2040/seedance-2.0，7,362 stars / 1,068 forks / 8 open issues（2026-09-19 查）
    * MIT 许可证；created_at 2026-02-25；pushed_at 2026-09-08；updated_at 2026-09-19；主语言 Python；仓库体积 33,536 KB
    * 文档体量极大：README.md 69,098 bytes、CHANGELOG.md 66,322 bytes、SKILL.md 27,253 bytes、SECURITY.md 12,974 bytes；版本号 v6.7.0，593 commits
    * schemas/ 目录下 5 个真实 JSON Schema：clip-contract.schema.json (10,614 B)、project-state.schema.json (15,883 B)、prompt-spec.schema.json (1,387 B)、take-review.schema.json (1,987 B)、generation-run.schema.json (936 B)
    * clip-contract 有 19 个 required 字段：project_id / clip_id / scene_id / sequence_index / narrative_job / felt_intent / target_duration_sec / generation_mode / shot_structure / already_happened / this_clip_only / reserved_for_later / planned_start_state / planned_end_state / continuity_locks / allowed_changes / status 等
    * project-state 有 17 个 required 字段：schema_version / state_revision / canon_revision / project_id / project_mode / clip_budget_sec / prompt_budget / story / world_bible / surface / reference_registry / scenes / beats / clips / take_history / current_clip_id / updated_at
    * 9 参考槽位的实际含义是「角色语义标签」：identity / first frame / last frame / product / environment / motion / camera / timing / audio / style，用 @Image1 @Video1 @Audio1 字节级保留标签绑定
    * 镜头语法术语：shot contract、visible beat、motivated light、continuity anchor、non-transferable detail、suppressed behavior；决策门：Intake→Source→Professional→Sequence→Mode→Capability→Reference Authority→Multilingual→Safety→Direction→Prompt Build→Quality Pass→Repair Loop
    * 安装方式：python scripts/install_codex_skill.py --client codex --scope user；声称兼容 Codex / Claude Code / Gemini CLI / Cursor / Windsurf / Copilot / Goose 等 15 个 Agent 宿主
    * 它自己声明的时长口径是 5–15 秒（与 fal 官方 4–15 秒口径有 1 秒出入），且明确说「duration 归 provider 控制面管，提示词里不写时长」
    * 仓库自陈限制（原文）："Repository checks cover routing, state, packaging, source integrity and declared document structure. Passing them is not a rendered-quality verdict." —— 即它的 CI 只校验文档结构，没有任何渲染质量验证
- 长片作用: 顶「分镜编译 + 连续性台账」这个工位，而且是目前唯一一个把 Seedance 2.0 的 9 图/3 视频/3 音频槽位、镜头语法、跨镜连续性锁写成可校验 JSON Schema 的公开实现。对你的 2-8 分钟片子，最有价值的不是那堆英文提示词规则（那部分你可以自己写），而是 clip-contract.schema.json 里的 planned_start_state / observed_end_state / continuity_locks / allowed_changes / parent_clip_id 这套「镜与镜之间状态交接」的字段设计，以及 take_history 的 accept / accept_with_deviation / repair / reject 四态评审记录 —— 直接可以当你编排层数据库表结构的起点。警告：schema 自己承认 JSON Schema 无法校验 clip-id 唯一性、parent 存在性、自环、拓扑序，这些必须你自己写 validator。
- 来源: https://github.com/Emily2040/seedance-2.0 https://api.github.com/repos/Emily2040/seedance-2.0 https://raw.githubusercontent.com/Emily2040/seedance-2.0/main/SKILL.md

## Seedance 2.0 / 2.5（fal.ai 托管通道）  (ByteDance，经 fal.ai 分发)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API，权重不开放 | 成本: 720p 约 $0.30/秒 → 一条 8 秒镜头 ≈ $2.42；2-8 分钟成片按 8 秒/镜算需 15–60 镜，单次全片 720p 出片 ≈ $36–145，含重拍 3× 则 ≈ $110–436。1080p 直接翻 2.25 倍。
- 是什么: 字节 Seedance 的第三方托管入口。用户给的硬约束在这里被逐条证实，且拿到了精确单价。
- 关键事实:
    * Seedance 2.0 reference-to-video：duration 为 auto 或 4–15 的任意整数秒（证实「官方单段 4-15 秒」）
    * 参考输入上限证实：图片最多 9 张（单张 ≤30 MB）、视频最多 3 段（合计 2–15 秒、总计 <50 MB）、音频最多 3 条（合计 ≤15 秒、单条 ≤15 MB），跨模态总文件数上限 12
    * 分辨率：480p / 720p / 1080p；宽高比 auto / 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16
    * Seedance 2.0 R2V 价格：720p $0.3024/秒；带视频输入时享 0.6× 折扣 = $0.1814/秒；1080p $0.682/秒；另有 $0.014 / 1K tokens 的计价口径
    * Seedance 2.0 text-to-video 价格：720p $0.3034/秒、720p Fast $0.2419/秒、1080p $0.682/秒
    * 原生音频证实："Native audio generation: music, SFX, and lip-synced dialogue, all in a single pass at no extra cost"，generate_audio 默认 true
    * Seedance 2.5 image-to-video：duration 为 auto 或 4–30 秒（证实「2.5 约 30 秒」）；480p ≈$0.2205/秒、720p ≈$0.4730/秒、1080p ≈$1.164/秒
    * API 字段名（可直接写进你的抽象层）：prompt / image_urls / video_urls / audio_urls / image_url / end_image_url / resolution / duration / aspect_ratio / generate_audio / seed / end_user_id
    * Replicate 同时上架 seedance-2.5（156K runs）、seedance-2.0（1.4M runs）、seedance-2.0-fast（684.3K runs）、seedance-2.0-mini（96K runs）、seedance-1.5-pro（4.3M runs）、seedance-1-pro（2.4M runs）、seedance-1-lite（3.8M runs）
    * Replicate 侧 duration 支持 -1（模型自选时长）、aspect_ratio 支持 adaptive；720p 具体像素为 1280×720 / 1112×834 / 960×960 / 834×1112 / 720×1280 / 1470×630，480p 为 864×496 / 752×560 / 640×640 / 560×752 / 496×864 / 992×432
- 长片作用: 顶「单镜画面生成」这个唯一工位。成本核算的硬结论：这是全产线最贵的一环，占比会超过 90%，所以编排层最重要的经济学设计不是并发而是「避免重复生成」—— 每一次 API 调用必须被内容寻址缓存住（prompt+refs+seed+params 的哈希），否则重跑一次流水线就是几百美元。另外 4–15 秒上限意味着 2-8 分钟片子至少 15 个镜头接口，镜与镜之间的状态交接（末帧→首帧、@Image 身份锚）是你系统的真正难点。
- 来源: https://fal.ai/models/bytedance/seedance-2.0/reference-to-video https://fal.ai/models/bytedance/seedance-2.0/text-to-video https://fal.ai/models/bytedance/seedance-2.5/image-to-video

## 多供应商视频路由器：开源生态缺位  (—)
- 类别/成熟度: service / **vaporware** | 许可: — | 成本: 自建成本：一个覆盖 fal + Replicate + Volcengine 三通道、带重试/降级/成本核算的路由层，工作量约 2-3 人周。
- 是什么: 结论先行：截至 2026-09-19，不存在「视频生成版的 LiteLLM」。我做了三路交叉验证，全部为负。你必须自建统一抽象层。
- 关键事实:
    * LiteLLM 支持 fal.ai，但只挂在 /images/generations 端点下，provider 前缀 fal_ai/，覆盖 FLUX Pro v1.1 / Imagen 4 / SD 3.5 / Recraft v3 / Ideogram v3 / Seedream / Bria 3.2 等 10+ 个图像模型。视频：文档中零提及
    * LiteLLM v1.100.1 → v1.103.0-dev.2 的 release notes 中无 /videos 端点、无 Sora / Veo / Kling / fal video 支持
    * docs.litellm.ai/docs/video_generation 返回 HTTP 404（该页面不存在）
    * OpenRouter 的 video 输出模态筛选下只有 1 个模型：black-forest-labs/flux-video-edit，$0.03/秒，最大输入 15 秒 720p —— 这是视频编辑不是视频生成，够不上路由器
    * GitHub 搜索 "unified API video generation fal replicate" 和 "video generation api router multi-provider" 均返回 total_count: 0
    * 星数前列的开源 AI 网关全部是纯 LLM 的：QuantumNous/new-api（48,431★ AGPL-3.0）、tensorzero（11,721★ Apache-2.0）、coaidev/coai（9,311★）、theagentrouter/agent-router（2,120★）、astaxie/TokenHub（1,315★）、ferro-labs/ai-gateway（256★）、mcowger/plexus（235★）—— 无一支持视频异步任务语义
    * 聚合平台侧：Segmind 页面确实列出 "Seedance 2.0 API"、"Kling 2.6 Pro Motion Control API"、Runway / Luma / PixVerse / LTX Studio / Veo，但价格页不公开单价，只提供 Model Price Explorer；aimlapi.com/video-models 与 /pricing 两个页面均 404
    * fal.ai 已上架的视频模型 id（可作为你抽象层的路由目标）：bytedance/seedance-2.5/{text-to-video,image-to-video,reference-to-video}、bytedance/seedance-2.0/{text-to-video,reference-to-video}、fal-ai/kling-video/v3/pro/image-to-video、minimax/h3/{text-to-video,image-to-video,reference-to-video}（含 H3 Max）、blackforestlabs/flux-3/image-to-video、lightricks/ltx-2.5/image-to-video/fast、xai/grok-imagine-video/v1.5/*
- 长片作用: 顶「多供应商容灾 + 成本核算」工位，但这个工位现在是空的，没有现成件可用。为什么 LLM 有 LiteLLM 而视频没有：LLM 是同步流式、请求体同构（messages）；视频是异步长任务（提交→轮询/webhook→产物 URL），各家的 job 状态机、产物过期时间、参考图上传方式全不一样，无法像 messages 那样一层薄壳抹平。自建时的最小抽象面：submit(spec) → job_id、poll(job_id) → {status, progress, artifact_url}、cancel(job_id)，加上 provider capability matrix（谁支持 9 图参考、谁支持原生音频、谁支持 30 秒），降级策略按「能力等价类」而不是按「模型排名」写 —— Seedance 2.0 R2V 降级到 Kling v3 I2V 会直接丢掉 9 图身份锚，那不是降级是换片。
- 来源: https://docs.litellm.ai/docs/providers/fal_ai https://github.com/BerriAI/litellm/releases https://openrouter.ai/models?fmt=table&output_modalities=video

## ComfyUI 原生 HTTP API + WebSocket  (Comfy Org)
- 类别/成熟度: framework / **production** | 许可: ComfyUI 本体 GPL-3.0 | 成本: 自托管，成本=GPU 租金
- 是什么: ComfyUI 自带的无鉴权 REST + WS 接口，是所有服务化封装的底座。你要自建编排层就直接对着这套协议写客户端。
- 关键事实:
    * HTTP 路由：POST /prompt（提交工作流，返回 prompt_id 与队列位置或校验错误）、GET /prompt、GET /queue、POST /queue（清空 pending/running）、POST /interrupt、GET /history、GET /history/{prompt_id}、POST /history、POST /upload/image、GET /view、GET /models、GET /models/{folder}、GET /system_stats（返回 Python 版本、设备、VRAM）
    * WebSocket /ws 消息类型：status、execution_start、executing、progress、executed、execution_cached —— progress 是你做长任务进度条的唯一来源，execution_cached 是你判断「这一步被 ComfyUI 内部缓存命中了」的信号
    * client_id 是 WS 与 /prompt 关联的键，必须自己生成并在两边保持一致，否则收不到自己任务的事件
    * 自定义路由扩展方式：@routes.post('/path') / @routes.get('/path') 装饰器 + aiohttp async handler
    * 最新版本 v0.36.0（2026-09-15）：新增视频拼接节点、Marigold v2 深度、Comfy Compiler 优化、结构化资产事件日志；partner nodes 新增 Flux Video Edit / Gemini 3.8 Flash / Tripo P2
    * v0.35.0（2026-09-09）：3D mesh 解析（GLB/GLTF/OBJ/STL）、视频 trim/crop 节点、HDR 视频编解码支持
    * v0.34.0（2026-08-26）：视频 HDR 保存、AV1/WebM/MKV 编解码、色彩空间选项；partner nodes 新增 FishAudio 与 Seedance 2.5 video extensions
    * ComfyUI API Nodes（Partner Nodes）是预付费信用点制（Stripe 充值，月度信用点账期末过期、充值信用点 1 年过期，不可退款不可转移），且默认只允许 127.0.0.1/localhost 登录，非白名单站点需 API Key —— 这条会卡住你的服务端部署
- 长片作用: 顶「本地/自托管的后处理与合成工位」：换脸、补帧、超分、口型对齐、字幕烧录、音画对齐、片尾拼接。注意它不顶主生成工位——Seedance 权重不开放，ComfyUI 里的 Seedance 只能走 partner node 转发到字节 API，而 partner node 的信用点制 + localhost 限制让它不适合做你的生产入口。正确做法：Seedance 走你自己的 fal/Volcengine 客户端，ComfyUI 只做产物落地后的图像/视频处理链路。GPL-3.0 是你产品化时的许可证风险点（如果你要闭源分发）。
- 来源: https://docs.comfy.org/development/comfyui-server/comms_routes https://api.github.com/repos/comfyanonymous/ComfyUI/releases https://docs.comfy.org/tutorials/api-nodes/overview

## SaladTechnologies/comfyui-api  (Salad Technologies)
- 类别/成熟度: framework / **production** | 许可: MIT（但运行时仍依赖 GPL-3.0 的 ComfyUI） | 成本: 自托管，按 GPU 计；无软件许可费
- 是什么: 把 ComfyUI 包装成无状态 HTTP API 的薄壳，是目前维护最勤、最贴近生产的 ComfyUI 服务化封装。
- 关键事实:
    * 定位原文："a simple wrapper that facilitates using ComfyUI as a stateless API"，去掉请求间状态以支持水平扩容
    * 最新版本 v1.19.2，发布于 2026-09-18（三天前）；v1.19.1 于 2026-09-16；维护非常活跃
    * 450 stars / 78 forks；MIT 许可证（依赖均为 MIT 或 Apache-2.0，ComfyUI 本体仍是 GPL-3.0）
    * 构建目标版本：ComfyUI 0.35.0 + PyTorch 2.13.0 + CUDA 13.0
    * 同步模式返回 base64；异步模式支持 webhook 回调，或直传 S3 兼容存储 / HTTP endpoint / HuggingFace repo / Azure Blob
    * 支持 warmup workflow：启动时跑一遍初始化工作流预载模型，解决冷启动
    * 输出格式 PNG（默认）/ JPEG / WebP，可配压缩参数
    * v1.19.2 修的是安全问题：下载文件名路径穿越、防止下载覆盖已有文件、限制出站 HTTP 请求以保护内网服务 —— 说明它被真实部署在多租户环境里
- 长片作用: 顶「ComfyUI 后处理集群的接入层」。相比自己对着 /prompt + /ws 写客户端，它现成给了 webhook + S3 直传 + warmup，正好是长片产线最需要的三件事（异步、产物不过手、冷启动摊薄）。选它而不是 RunPod worker 的理由是 MIT 许可证和可以部署在任意云；选 RunPod 的理由是不想自己管 autoscaling。
- 来源: https://github.com/SaladTechnologies/comfyui-api https://api.github.com/repos/SaladTechnologies/comfyui-api/releases

## runpod-workers/worker-comfyui + RunPod Serverless  (RunPod)
- 类别/成熟度: service / **production** | 许可: AGPL-3.0 | 成本: L40S $1.75/hr ≈ $0.00049/秒；一条 8 秒镜头的超分+补帧后处理若耗时 60 秒 GPU，约 $0.029 —— 相比 Seedance 生成的 $2.42 可以忽略不计
- 是什么: 官方维护的 ComfyUI serverless worker，按秒计费、自动扩缩到零。
- 关键事实:
    * 742 stars / 708 forks；AGPL-3.0 许可证（注意：比 comfyui-api 的 MIT 更具传染性）；216 commits
    * 两种端点：/runsync 同步等结果，/run 异步返 job id 后轮询 /status；支持 webhook 完成通知
    * 预构建镜像 runpod/worker-comfyui:<version>-[model]，含 base（无模型）/ FLUX.1 schnell / FLUX.1 dev / SDXL / SD3 medium
    * 输出默认 base64 字符串，可配 S3 URL；响应结构为 output.images 数组，含 filename / type / data
    * RunPod GPU 小时价（2026-09 官网）：RTX 4090 24GB $1.10/hr、A40 48GB $1.22/hr、RTX 5090 32GB $1.58/hr、L40/L40S 48GB $1.75/hr、A100 80GB $2.72/hr、H100 80GB $4.79/hr、H200 141GB $5.93/hr
    * 官网宣称 flex worker 比其他 serverless 云省 25%，但定价页未区分 flex/active 单价
- 长片作用: 顶「后处理弹性算力」工位。关键经济学判断：后处理的 GPU 成本比生成成本低两个数量级，所以不要为省后处理算力而牺牲流程可靠性，该重跑就重跑。AGPL-3.0 是真实风险：如果你的平台对外提供网络服务且集成了这个 worker 的代码，AGPL 的网络分发条款会触发源码开放义务，建议用 comfyui-api（MIT）替代或严格隔离进程边界。
- 来源: https://github.com/runpod-workers/worker-comfyui https://www.runpod.io/pricing

## ComfyDeploy（comfy-deploy/comfydeploy）  (ComfyDeploy)
- 类别/成熟度: framework / **usable** | 许可: GPL-3.0 | 成本: 自托管免费；SaaS 定价未在公开页面获取到
- 是什么: 工作流版本化 + 机器管理 + API 暴露的 ComfyUI 部署平台，开源仓库已接近停更。
- 关键事实:
    * 主仓库 comfy-deploy/comfydeploy：455 stars，GPL-3.0，last push 2025-09-19 —— 整整一年无更新
    * 周边仓库同样停滞：comfyui-deploy-gradio-demo（55★，2024-08-10）、comfydeploy-fullstack-demo（25★，2025-04-20）、comfyui-api-comfydeploy（20★，2025-08-29）、comfydeploy.js SDK（4★，MIT，2024-08-13）、comfydeploy-mcp（2★，2025-08-06）
    * comfy-deploy/models（16★，GPL-3.0，2024-11-05）：基于 ComfyDeploy + Modal 的云推理端点集合，也已停更
    * 原 github.com/comfy-deploy/comfyui-deploy 路径返回 404（仓库已改名为 comfydeploy）
- 长片作用: 原本应该顶「工作流版本化」工位（这正是长片产线最需要的：第 37 号镜头用的是哪个 workflow 的哪个版本），但开源侧一年没动、GPL-3.0、生态周边全部停滞，我不建议把它放进关键路径。这个工位更稳妥的做法是：把 ComfyUI workflow JSON 当普通代码资产提交进 Git，用 commit sha 作为 workflow_version 写进你的 generation-run 记录。
- 来源: https://api.github.com/search/repositories?q=comfydeploy&sort=stars&order=desc

## HuggingFace Diffusers（视频 pipeline 服务化）  (Hugging Face)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0（库本身；各模型权重许可证各异） | 成本: 取决于模型；14B 级视频模型典型需 40–80GB VRAM（A100/H100 级），消费级 24GB 需量化或分块
- 是什么: 开源扩散模型推理库，是自托管视频生成（非 Seedance）的事实标准入口。
- 关键事实:
    * 最新版 v0.40.0，发布于 2026-08-20；v0.39.0（2026-07-03）、v0.38.0（2026-05-01）、v0.37.1（2026-03-25）、v0.37.0（2026-03-05）
    * v0.40.0 新增视频 pipeline：LTX-2.5、Wan-Animate-2、MiniMax-H3（视频+音频联合合成）；同时加入 tensor-parallel 支持与 CLI 改进
    * v0.39.0 新增：Motif-Video（T2V + I2V）、AnyFlow（any-step 视频生成）、Cosmos 3 扩展（V2V 与动作条件生成）
    * v0.37.0 新增：LTX-2（音频条件 T2V）、Helios（14B，分钟级生成，17 FPS）、Modular Diffusers 架构
    * v0.36.0 新增 HunyuanVideo 1.5、Sana-Video
    * Wan-Animate-2 专攻「从参考图做角色动画」—— 这是与你的数字人诉求最相关的一条线
- 长片作用: 顶「Seedance 之外的本地备份产能 + 可 LoRA 的角色一致性工位」。这是你唯一能绕开「Seedance 权重不开放、不能本机 LoRA」这条硬约束的路径：用 Wan-Animate-2 或 LTX-2.5 在本地训角色 LoRA 做身份锚，产出的参考帧再喂给 Seedance 的 9 图槽位。注意 Modular Diffusers（v0.37.0 起）是服务化的关键——它让你能把 pipeline 拆成可独立缓存的模块，而不是每次整条重跑。服务化本身 Diffusers 不提供，需自己套 FastAPI/Ray Serve。
- 来源: https://api.github.com/repos/huggingface/diffusers/releases https://github.com/huggingface/diffusers/releases

## Celery + Redis（长耗时 GPU 任务）  (Celery 项目)
- 类别/成熟度: framework / **production** | 许可: BSD-3-Clause | 成本: 免费
- 是什么: 最普及的 Python 任务队列。对长耗时 GPU 任务有明确的、文档记载的坑。
- 关键事实:
    * acks_late=False（默认）：worker 执行前就 ack，worker 崩溃则任务永久丢失 —— 对 10 分钟级 GPU 任务不可接受
    * acks_late=True：执行完才 ack，worker 死了会重投，但官方原文要求 "Make sure your tasks are idempotent"
    * acks_late=True 的破例条款：若子进程被 sys.exit() 或信号终止（含内核 OOM killer、段错误），消息仍然被 ack —— 这正是 GPU OOM 时最常见的死法，意味着 OOM 导致的失败不会自动重投
    * task_reject_on_worker_lost=True 可强制重投，但官方警告会造成高频消息循环
    * Redis 作为 broker 没有 AMQP 级的消息持久化，超过 visibility window 消息会过期；也没有 Dead Letter Exchange 等价物；官方建议生产用 RabbitMQ
    * 重试配置：autoretry_for=(Exc,) + max_retries=5；retry_backoff=True 走 1s→2s→4s→8s 指数退避，retry_backoff_max 默认上限 600 秒；retry_jitter 默认开启
    * 时限：soft_time_limit 抛 SoftTimeLimitExceeded 可捕获清理；time_limit 硬杀进程。官方原文 "only use them to detect cases where you haven't used manual timeouts yet"，IO 操作应自己加 timeout 而非依赖 time limit
- 长片作用: 顶「短任务调度」工位，不顶「长片编排」工位。结论：Celery+Redis 可以拿来调度单条镜头的 API 提交与轮询（每步都是秒级），但不要用它来编排「一部 40 镜的片子」这个跨小时的长流程——它没有持久化的工作流状态机，崩溃后无法从第 23 镜续跑。正确分层：Celery 做 worker 池执行单步，上层用 Temporal/Prefect 管流程状态。
- 来源: https://docs.celeryq.dev/en/stable/userguide/tasks.html

## Temporal（持久化执行）  (Temporal Technologies)
- 类别/成熟度: framework / **production** | 许可: MIT（Temporal Server），Temporal Cloud 为商业服务 | 成本: 自托管免费；Temporal Cloud 按 action 计费
- 是什么: 持久化工作流引擎。把「一部片子的生产流程」写成可崩溃可续跑的代码，是长耗时 GPU 编排最贴合的模型。
- 关键事实:
    * 三种 activity 超时：start_to_close_timeout、schedule_to_close_timeout、heartbeat_timeout；官方明确 Temporal 依赖 Start-To-Close timeout 来发现「task 在投递中丢失或 worker 崩溃后」的情况
    * 心跳："Activities must heartbeat to receive cancellations from a Temporal Service."；服务端在超时窗内未收到心跳则判定 activity 失败，返回 Cancelled Failure 且 message: 'TIMED_OUT'
    * 心跳可携带 details，重试时 activity 能读回上次心跳的 details 从断点继续 —— 这就是 GPU 长任务「断点续跑」的落地机制
    * RetryPolicy 默认值：initial_interval=1 秒、backoff_coefficient=2.0、maximum_interval=100× initial_interval、maximum_attempts=无限、non_retryable_error_types=空
    * 错误语义：ApplicationError 支持 type 分类与 non_retryable 标志；ActivityError 的 cause 字段保留原始异常供 workflow 检查；抛普通 Python 异常（ValueError/TypeError）会变成 Workflow Task failure 并自动重试，等你改代码重新部署后继续，状态不丢
    * 幂等键官方建议：Workflow Run ID + Activity ID 组合，传给外部服务做去重
    * 若 retry policy 设为 1 且发生超时，activity 不会被重试
- 长片作用: 顶「全片生产流程编排」这个核心工位，我认为这是你该选的。理由具体到你的场景：(1) 一部 8 分钟片 = 60 个镜头 × 每镜 2-5 分钟 API 等待 = 跨数小时的流程，Temporal 的 workflow 状态存在服务端，进程重启、机器换台、代码热更都不丢进度；(2) heartbeat details 让「轮询 fal job 状态」这件事可以在 worker 崩溃后从同一个 job_id 续上，不会重新提交一次 $2.42 的生成；(3) Workflow Run ID + Activity ID 做幂等键，天然对应「第 N 镜第 M 次重拍」；(4) non_retryable 标志正好用来区分「内容审核拒绝」（不该重试，要改提示词）和「503 限流」（该指数退避重试）。
- 来源: https://docs.temporal.io/activity-execution https://docs.temporal.io/develop/python/failure-detection

## Prefect 3（任务缓存与幂等）  (Prefect)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: OSS 免费；Prefect Cloud 按 workspace 计费
- 是什么: Python 原生编排框架，缓存策略是它相对 Celery 的核心优势。
- 关键事实:
    * 内置 cache_policy：DEFAULT（inputs + code + flow run id）、INPUTS（仅参数值）、TASK_SOURCE（仅任务代码，不含嵌套任务）、FLOW_PARAMETERS（仅父 flow 参数）、NO_CACHE
    * 策略可用运算符组合：TASK_SOURCE + INPUTS 复合；INPUTS - 'debug' 排除指定参数 —— 排除 debug 这类不影响产物的参数，正是「同样的分镜不重复烧钱」所需
    * cache_key_fn 接收 TaskRunContext 与入参字典，可完全自定义缓存键
    * cache_expiration 接收 datetime.timedelta 使缓存过期
    * 缓存依赖结果持久化：需 PREFECT_RESULTS_PERSIST_BY_DEFAULT=true；结果默认落在 ~/.prefect/storage/，文件名即缓存键
    * flow 级支持 retries（可配延迟与次数上限）与 timeouts
    * 官方表述：缓存使流水线在失败重试时保持幂等，相同入参的重复调用直接返回缓存值不重新执行
- 长片作用: 顶「内容寻址式重跑」工位。它的 cache_policy 代数（INPUTS - 'debug'、TASK_SOURCE + INPUTS）是我见过最直接表达「什么变了才该重新烧钱」的 API：把 seed、prompt、reference URLs 纳入缓存键，把 debug 开关、日志级别、输出路径排除掉。但 Prefect 的持久化状态模型没有 Temporal 强，跨小时长流程中途崩溃的恢复保证弱一档。务实组合：Temporal 管流程骨架，缓存逻辑自己用 content hash 实现（其实就是把 Prefect 的 cache_policy 思路抄成 20 行代码），不必引入两个编排框架。
- 来源: https://docs.prefect.io/v3/concepts/caching https://docs.prefect.io/v3/concepts/flows

## Dagster（资产 + 分区）  (Dagster Labs)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: OSS 免费；Dagster+ 按 materialization credits 计费
- 是什么: 以「数据资产」而非「任务」为中心的编排器，分区模型天然映射到分镜。
- 关键事实:
    * 核心概念是 asset definition：声明「什么数据应该存在」以及如何计算它；官方对比原文 "asset definitions know about their dependencies, while ops do not"
    * materialization 定义为「运行其函数并把结果存入持久化存储」；可从 UI 或 Python API 触发
    * 分区（Partitions）支持按时间/类别切分资产，支持逐分区独立物化、分区依赖、backfill 回填
    * 支持只重跑失败的分区而不重跑整个资产 —— 对应「只重拍第 37 号镜头」
    * Dagster+ 计费口径：materialization 消耗 credits，asset observation 不消耗
- 长片作用: 顶「产物血缘与选择性重跑」工位。概念契合度其实最高：一部片子 = 一个 partitioned asset，每个镜头 = 一个 partition，backfill = 批量重拍，asset lineage = 分镜→参考帧→生成片段→后处理→成片的血缘图。但它是为数据工程（批处理、按天分区）设计的，跑 GPU 长任务与人工评审回环（take_history 的 accept/repair/reject）会别扭。建议只借鉴它的分区与血缘建模思想，不引入运行时。
- 来源: https://docs.dagster.io/guides/build/assets https://docs.dagster.io/guides/build/partitions-and-backfills

## DVC + MinIO（产物寻址与版本化）  (Iterative.ai / MinIO)
- 类别/成熟度: framework / **production** | 许可: DVC Apache-2.0；MinIO AGPL-3.0（社区版）/ 商业版 AIStor | 成本: 软件免费；存储成本=对象存储单价×副本数
- 是什么: 内容寻址的大二进制版本化（DVC）+ S3 兼容的版本化对象存储（MinIO）。
- 关键事实:
    * DVC 用 MD5 内容哈希做寻址，缓存布局形如 .dvc/cache/files/md5/22/a1a2931c8370d3aeedd7183606fd7f（前 2 字符分桶）
    * dvc add 生成轻量 .dvc YAML 指针文件进 Git，实际二进制放 remote；git checkout .dvc 文件 + dvc checkout 即可秒级切版本（官方称 100GB 文件切版本 <1 秒）
    * DVC remote 支持 Amazon S3、NFS、SSH、Google Drive、Azure Blob、HDFS；dvc push / dvc pull 同步
    * MinIO 版本化 S3 API 原生兼容：每个版本一个 UUIDv4 version ID，最新版标记 latest；删除未指定 version ID 时写入零字节 DeleteMarker 作为新 latest，旧版本保留；指定 version ID 删除不可逆
    * MinIO 明确不做差分版本化，保留完整副本。官方举例警示：1GB 对象每天增量 100MB 更新，10 次迭代后占用超过 14GB
    * object lock / retention 是版本化的前置条件；可用生命周期规则自动清理 noncurrent 版本，或 mc rm --versions 手动删
- 长片作用: 顶「产物寻址与版本化」工位，但我要给一个反直觉的建议：不要用 DVC 管生成的视频。DVC 的 MD5 内容寻址设计前提是「数据集相对稳定、迭代次数有限」，而你的产线是「每个镜头重拍 3-10 次、每次几十 MB mp4」，MinIO 那条「不做差分版本化」的警告会直接命中你——一部 8 分钟片子做 5 轮迭代就是几十 GB 全量副本。更合适的做法：自己做一层极薄的内容寻址，键 = sha256(provider + model_id + prompt + sorted(reference_hashes) + seed + params) → 产物直接 PUT 到 S3/MinIO 的 cas/<sha256前2>/<sha256> 路径，天然去重且天然幂等；版本化交给 MinIO 的 bucket versioning + lifecycle 规则自动淘汰 noncurrent 版本。DVC 只用来管「模型权重与 LoRA」这类真正稳定的大文件。
- 来源: https://doc.dvc.org/start https://docs.min.io/enterprise/aistor-object-store/administration/objects-and-versioning/versioning/

## OpenTimelineIO (OTIO)  (Academy Software Foundation)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 | 成本: 免费
- 是什么: 电影工业的剪辑时间线交换格式与 API，是「分镜 JSON schema 标准」这个问题唯一有工业背书的答案。
- 关键事实:
    * 定位原文："Open Source API and interchange format for editorial timeline information"
    * Apache-2.0；2,000+ stars / 355 forks
    * 最新版本 v0.18.1（2025-11-09，修 Linux Python wheel，cibuildwheel 升到 3.2.1）；v0.18.0（2025-11-06）为主要版本
    * v0.18.0 schema 变更：Effects 新增 enabled 标志、新增 per-track 与 per-clip 颜色支持、新增 Color 原语结构；移除 Imath 2 支持、核心库移出 OTIOView、新增 C++ 符号可见性与 Doxygen
    * v0.17.0（2024-06-24）起 adapter 从核心库剥离到独立 PyPI 包 OpenTimelineIO-Plugins；核心包只保留原生 .otio / .otioz / .otiod
    * 语言绑定：C++ 核心 + Python 3.9–3.12 官方绑定与插件系统；EDL / FCPXML / AAF 适配器已迁到 OpenTimelineIO 组织下的独立仓库
    * schema 覆盖 track / clip / media reference，编码的是「剪辑点的顺序与时长 + 外部媒体引用」
- 长片作用: 顶「成片时间线的最终交付与外部剪辑软件对接」工位，不顶「分镜创作」工位。关键区分：OTIO 描述的是「已经存在的素材如何排布」（in/out 点、轨道、转场），它没有任何字段描述「这个镜头该怎么拍」（镜头景别、运镜、光线、角色身份锚）。所以你的 2-8 分钟产线需要两层：上游用自定义的 shot-spec JSON（可直接抄 Emily2040 的 clip-contract.schema.json）做创作意图，下游生成完毕后导出 OTIO 让 Premiere/Resolve 能接。OTIO 是「拼接」这一步的正解，别指望它管分镜。
- 来源: https://github.com/AcademySoftwareFoundation/OpenTimelineIO https://api.github.com/repos/AcademySoftwareFoundation/OpenTimelineIO/releases

## Fountain / FDX / OpenUSD（剧本与场景格式的适用性）  (Fountain 社区 / Final Draft / Pixar+AOUSD)
- 类别/成熟度: technique / **production** | 许可: Fountain 公共规范；OpenUSD 开源（Pixar 维护） | 成本: 免费
- 是什么: 三个常被提议但实际不适配 AI 分镜的既有标准。逐个证伪。
- 关键事实:
    * Fountain 支持的元素：Scene Headings（INT./EXT./EST. 或 . 强制）、Action、Character（大写）、Dialogue、Parentheticals、Dual Dialogue（^）、Transitions（TO: 结尾或 > 强制）、Lyrics（~）、Centered Text（><）、Emphasis、Title Page 键值对、Page Breaks（===）、Notes（[[ ]]）、Boneyard（/* */）、Sections（#）、Synopses（=）
    * Fountain 的致命缺口（已核实）：不编码镜头规格、不编码摄影机指令、不编码时长/时间码数据。它的设计目标原文是保持 "a look like a screenplay" 的纯文本可读性
    * OpenUSD 当前版本 26.08；定位原文 "a high-performance extensible software platform for collaboratively constructing animated 3D scenes, designed to meet the needs of large-scale film and visual effects production"；核心能力是 composition、layers、variants，schema 覆盖 geometry / shading / lighting / physics
    * OpenUSD 面向 3D 场景互换与多人协同，对「文生视频的提示词化分镜」没有任何对应 schema
- 长片作用: 结论：不存在「成熟的分镜 JSON schema 标准」。Fountain 顶「剧本文本」工位（你可以用它做 LLM 生成对白的中间格式，parser 生态成熟），但它连镜头都不表达，接不上 Seedance 的参数面。FDX 是 Final Draft 的私有 XML，同样只到剧本层。OpenUSD 是 3D 管线格式，与你的 2D 生成式产线完全不同赛道，引入它只会增加复杂度。务实路线：Fountain（剧本层，可选）→ 你自己的 shot-spec JSON（分镜层，参考 clip-contract.schema.json）→ OTIO（时间线层，交付用）。中间那层必须自建，这是业界空白。
- 来源: https://fountain.io/syntax/ https://openusd.org/release/index.html

## VideoClaw（原 HITsz-TMG/FilmAgent，已改名）  (哈工大深圳 HITsz-TMG)
- 类别/成熟度: framework / **usable** | 许可: MIT | 成本: 无 GPU 需求，成本=所调用 API 的费用
- 是什么: 重要更正：github.com/HITsz-TMG/FilmAgent 现在 301 重定向到 HITsz-TMG/VideoClaw。SIGGRAPH Asia 2024 的 FilmAgent（Unity 虚拟片场）已演进成一个真正调用商业视频 API 的端到端产线工具。
- 关键事实:
    * 仓库 HITsz-TMG/VideoClaw：1,809 stars / 270 forks；MIT 许可证；created_at 2024-08-29；pushed_at 2026-08-26
    * 自述："AI 全自动化视频生成员工 | Your First AIGC Coworker. Chat an Idea. Get a Film."
    * 流程链路：剧本策划 → 角色/场景设计 → 分镜规划 → 参考图生成 → 视频生成 → 后期剪辑
    * 调用的 LLM：GPT-4o、GPT-5、DeepSeek、Gemini、通义千问；视觉理解：Gemini、Qwen、Kimi
    * 调用的图像模型：阿里 Wan、豆包 Seedream、DALL-E；视频模型：阿里 Wan（i2v / r2v 模式）、Kling、豆包 Seedance；含 TTS 旁白
    * 分镜以 JSON 落盘在 results 目录，含摄影机角度、动作描述、参考内容的结构化描述
    * 依赖：Python 3.9+、Node.js 18+、npm 9+、ffmpeg；Linux/macOS/Windows 均有自动安装脚本；无本地 GPU 要求（全走 API）
    * 示例作品为 8 集短剧系列；单集时长未标注
    * FilmAgent 原论文（SIGGRAPH Asia 2024，Zhenran Xu / Jifang Wang 等）在致谢中被列为概念来源
- 长片作用: 这是全网最接近你目标的开源参照物，顶「端到端产线原型」工位。它已经踩过你要踩的坑：LLM 分镜 → 参考图先行 → i2v/r2v 生成 → ffmpeg 后期，且分镜以 JSON 落盘。差距在于：(1) 它是脚本式而非服务化，没有任务队列、没有断点续跑、没有幂等；(2) 视频通道走 Wan/Kling/Seedance 混用但没有统一抽象层与降级；(3) 它的样片是短剧集而非连续 2-8 分钟单片，镜间连续性保障弱于 Emily2040 的 continuity_locks 设计。建议：读它的 prompt 与 JSON 结构，不要直接拿它当生产系统。
- 来源: https://api.github.com/repos/HITsz-TMG/FilmAgent https://raw.githubusercontent.com/HITsz-TMG/VideoClaw/main/README.md

## MoneyPrinterTurbo  (harry0703)
- 类别/成熟度: framework / **usable** | 许可: MIT | 成本: 接近零（Edge TTS 免费 + 免费素材库）
- 是什么: 星数最高的自动短视频工具，但主要是素材拼接不是生成。
- 关键事实:
    * 约 125k stars（124.6k）/ 19.3k forks；MIT 许可证
    * 输出时长实测样例：14 / 23 / 24 / 27 / 44 / 59 秒 —— 全部在 1 分钟以内
    * 主要工作方式是从 Pexels / Pixabay 抓库存素材拼接，AI 生成视频只是可选通道（接 MiniMax H3、火山引擎 Seedance、WaveSpeed AI）
    * LLM 接入：Kimi/Moonshot、OpenAI、Anthropic Claude、Gemini、DeepSeek、通义千问、Azure OpenAI、火山引擎、xAI Grok、MiniMax、小米 MiMo
    * TTS 接入：Edge TTS（免费无需 key）、Azure Speech、SiliconFlow、Gemini TTS、小米 MiMo TTS、MiniMax TTS、ElevenLabs、Chatterbox、Kokoro、Fish Audio、面壁 VoxCPM
- 长片作用: 对你的 2-8 分钟数字人片子几乎不顶任何工位。它的 14-59 秒输出上限、库存素材拼接的本质、没有角色一致性概念，都与你的诉求正交。唯一可借鉴的是它那份极全的 TTS provider 适配代码 —— 但 Seedance 2.0 自带对白声，你连这个都不需要。125k stars 是短视频自动化赛道的热度，不是技术深度，不要被星数误导。
- 来源: https://github.com/harry0703/MoneyPrinterTurbo

## ShortGPT  (RayVentura)
- 类别/成熟度: framework / **usable** | 许可: MIT | 成本: 接近零
- 是什么: 自述为 "an experimental AI framework for youtube shorts / tiktok channel automation"。
- 关键事实:
    * 约 8,000 stars；MIT 许可证；stable 分支 298 commits；75 个 open issues、11 个 PR
    * 工作方式：脚本生成 + TTS 配音 + 字幕 + MoviePy 剪辑，素材来自 Pexels API 与 Bing Image
    * 本质是素材拼装（assembles content），不生成原创视频
- 长片作用: 不顶工位。定位是 Shorts/TikTok 竖屏自动化，与「单镜观感接近 Seedance 2.0 的 2-8 分钟叙事片」不在同一问题域。列在这里是为了明确排除：不要因为它出现在「AI 视频开源项目」清单里就去评估它。
- 来源: https://github.com/RayVentura/ShortGPT

## 研究型分镜项目群：MovieAgent / VideoGen-of-Thought / StoryDiffusion / AutoStudio / Anim-Director  (showlab / 多所高校)
- 类别/成熟度: technique / **research** | 许可: 混杂：Apache-2.0 / BSD-3-Clause / 未声明 | 成本: 多数需 A100 级显卡跑 SVD/HunyuanVideo；权重多不完整
- 是什么: 五个被反复引用的学术项目。逐个核实后：思路可借鉴，代码全部不可生产。
- 关键事实:
    * MovieAgent（arXiv:2503.07314，2025-03）：多 Agent CoT 规划，模拟导演/编剧/分镜师/场景经理角色；363 stars / 43 forks / 17 commits；最后更新 2025-03-18（停更 18 个月）；依赖 GPT-4o + ROICtrl + SVD + HunyuanVideo_I2V；致命限制：需要预训练角色权重，官方只给了《冰雪奇缘 2》示例，换角色必须自己训 EDLora
    * VideoGen-of-Thought（arXiv:2412.02259，2024-12）：单条 prompt 生成多镜头视频，做动态故事线建模与身份感知的跨镜传播；仅 65 stars；BSD-3-Clause；NeurIPS 2025 NextVid Workshop oral（workshop 不是主会）
    * StoryDiffusion（arXiv:2405.01434，NeurIPS 2024 Spotlight）：Consistent Self-Attention 可热插拔到任意 SD1.5/SDXL 模型 + Motion Predictor 做长视频；6.5k stars / 644 forks；Apache-2.0；致命问题：视频生成模型的预训练权重至今仍在 TODO 列表，从未发布，只有漫画生成的 HF Space demo
    * AutoStudio（arXiv:2406.01388，2024-06）：3 个 LLM agent + SD 做多轮交互式一致主体生成；452 stars；最后 commit 2024-06（停更 27 个月）；纯图像不是视频；需自备 SD + IP-Adapter checkpoint
    * Anim-Director（HITsz-TMG/Anim-Director）："Controllable Animation Video Generation with Large Models-based Multimodal Agents"；258 stars；无许可证声明（这意味着默认保留全部权利，法律上不可商用）；pushed_at 2026-01-07
- 长片作用: 没有一个能顶生产工位。但有两条设计思想值得抄进你的编排层：(1) MovieAgent 的角色分工（导演/编剧/分镜师/场景经理各自一个 agent，用 CoT 串联）比单个大 prompt 更可控，可直接映射成你的多阶段 LLM 调用链；(2) VideoGen-of-Thought 的「身份感知跨镜传播」正是 Seedance 9 图参考槽位要解决的同一问题，它的 identity propagation 建模可以指导你决定「第 N 镜该把哪几张图塞进 image_urls」。StoryDiffusion 必须标记为 vaporware 级的可用性——6.5k stars 但视频权重两年半没放出来，只有漫画 demo；Anim-Director 无许可证不可商用，这一条比技术问题更硬。
- 来源: https://github.com/showlab/MovieAgent https://arxiv.org/abs/2503.07314 https://github.com/DuNGEOnmassster/VideoGen-of-Thought

### 建议
【调研条件说明】本次 WebSearch 配额（200/200）在第一次调用时已耗尽，全部事实均通过 WebFetch 直接拉取 GitHub API / 官方文档 / 厂商页面获得。好处是每条都有一手来源，代价是少数页面（火山引擎 Ark 文档、AIMLAPI）因重定向或 404 未能取证，已列入 uncertainClaims。

【第一问的直接回答】Emily2040/seedance-2.0 真实存在且活跃：7,362 stars、1,068 forks、MIT、2026-02-25 创建、2026-09-08 最后推送、v6.7.0。但要把它的性质讲清楚：它是纯 Markdown 的 Agent skill（SKILL.md 27KB + README 69KB），零代码零 API 调用，不生成任何视频。它真正的价值在 schemas/ 下那 5 个 JSON Schema —— clip-contract（19 个必填字段）与 project-state（17 个必填字段）把「镜与镜之间的状态交接」形式化了：planned_start_state / observed_end_state / continuity_locks / allowed_changes / parent_clip_id / take_history 四态评审。这套字段设计可以直接当你数据库表结构的起点。它的 9 槽位定义也与 fal 官方文档完全对上（9 图 / 3 视频 / 3 音频）。风险提示：它自己的 CI 只校验文档结构，README 原文承认 "Passing them is not a rendered-quality verdict"，且 schema 自陈无法校验 clip-id 唯一性、parent 存在性、自环与拓扑序——这些 validator 你必须自己写。

【硬约束核实结果】用户给的五条硬约束，四条被 fal.ai 官方模型页精确证实：单段 4–15 秒（duration 为 auto 或 4–15 整数）、2.5 为 4–30 秒、9 图 + 3 视频 + 3 音频（且跨模态总文件数上限 12、图单张 ≤30MB、视频合计 2–15 秒 <50MB、音频合计 ≤15 秒）、自带对白声（原生音乐/音效/唇同步对白单次生成且不额外收费）、权重不开放。第五条「官方通道有内容审核」我没能从火山引擎文档取证（页面重定向后返回空），fal 页面也只有 Trust & Safety 链接无具体条款——这条我不敢替你确认，列进了 uncertainClaims。

【最关键的负面结论】不存在「视频生成版的 LiteLLM」。三路交叉验证全部为负：LiteLLM 的 fal_ai provider 只挂在 /images/generations 下（10+ 图像模型），v1.100–v1.103 release notes 零视频提及，docs.litellm.ai/docs/video_generation 是 404；OpenRouter 的 video 输出筛选下只有 1 个模型（flux-video-edit，$0.03/秒，还是编辑不是生成）；GitHub 两次针对性搜索 total_count 均为 0，星数前列的开源 AI 网关（new-api 48k★、tensorzero 11.7k★、coai 9.3k★）全是纯 LLM。根因是架构性的：LLM 是同步流式 + messages 同构，视频是异步长任务 + 各家 job 状态机/产物过期/参考图上传方式全不同，抹不平。所以多供应商容灾这一层你必须自建，最小抽象面是 submit(spec)→job_id / poll(job_id)→{status,progress,artifact_url} / cancel(job_id)，外加一张 provider capability matrix。降级策略要按「能力等价类」写而不是按模型排名——Seedance 2.0 R2V 降级到 Kling v3 I2V 会丢掉 9 图身份锚，那不叫降级叫换片。

【推荐的产线分层，逐个工位定人】
1. 分镜层（创作意图）：自建 shot-spec JSON，直接抄 Emily2040 的 clip-contract.schema.json。业界确无标准——Fountain 已证实不编码镜头/摄影机/时长，FDX 同样只到剧本层，OpenUSD 是 3D 场景互换格式与本赛道正交。
2. 流程编排层：Temporal。理由是 heartbeat details 能让「轮询 fal job」在 worker 崩溃后从同一 job_id 续上，不会重投一次 $2.42 的生成；Workflow Run ID + Activity ID 做幂等键天然对应「第 N 镜第 M 次重拍」；non_retryable 标志正好区分「审核拒绝」（改提示词）与「503 限流」（指数退避）。不要用 Celery 编排全片——官方文档白纸黑字写明 acks_late=True 在 OOM killer / 段错误时仍会 ack，而 GPU OOM 正是最常见死法。Celery 只配做 worker 池执行单步。
3. 缓存与去重层：不要上 DVC 管生成的 mp4。MinIO 官方警告不做差分版本化（1GB 对象每天 +100MB、10 次迭代占 14GB+），而你是每镜重拍 3-10 次。正解是自建 20 行内容寻址：key = sha256(provider + model_id + prompt + sorted(reference_hashes) + seed + params)，产物 PUT 到 cas/<前2>/<sha256>，天然去重天然幂等；版本化交给 MinIO bucket versioning + lifecycle 淘汰 noncurrent。DVC 只留给模型权重与 LoRA。Prefect 的 cache_policy 代数（INPUTS - 'debug'、TASK_SOURCE + INPUTS）是这套哈希键该纳入/排除什么的最佳参考，但不必为它引第二个编排框架。
4. 生成层：Seedance 2.0 走 fal（$0.3024/秒@720p，带视频输入 0.6× 折扣降到 $0.1814/秒）或 Replicate（seedance-2.0 已 1.4M runs，另有 -fast / -mini 两档）。成本硬账：8 秒镜头 720p ≈ $2.42，2-8 分钟片按 8 秒/镜需 15–60 镜，单次全片 ≈ $36–145，含 3× 重拍 ≈ $110–436，1080p 再乘 2.25。这一环占总成本 90%+，所以编排层最重要的经济学设计不是并发是「避免重复生成」。
5. 本地补位层：Diffusers 0.40.0（2026-08-20）的 Wan-Animate-2 是你唯一绕开「Seedance 不能本机 LoRA」的路径——本地训角色 LoRA 产出身份锚参考帧，再喂进 Seedance 的 image_urls 9 槽位。
6. 后处理层：ComfyUI + SaladTechnologies/comfyui-api（v1.19.2，2026-09-18，MIT，自带 webhook + S3 直传 + warmup 预载）。不要用 runpod-workers/worker-comfyui 做产品集成——AGPL-3.0 的网络分发条款会触发源码开放义务。ComfyUI 的 Partner Nodes 也别用作生产入口：预付费信用点制 + 默认只允许 localhost 登录。后处理算力便宜到可忽略（L40S $1.75/hr ≈ $0.00049/秒，一镜后处理约 $0.029，比生成低两个数量级），该重跑就重跑。
7. 交付层：OTIO（Apache-2.0，v0.18.1，2025-11-09）导出时间线给 Premiere/Resolve。注意 v0.17.0 起 EDL/FCPXML/AAF 适配器已剥离到独立的 OpenTimelineIO-Plugins 包。OTIO 只管「素材如何排布」，不管「镜头该怎么拍」，别指望它替代分镜层。
8. 参照物：HITsz-TMG/FilmAgent 已改名为 VideoClaw（1,809★，MIT，2026-08-26 推送），是全网最接近你目标的开源实现，已跑通 剧本→角色/场景→分镜 JSON→参考图→i2v/r2v→ffmpeg 全链路。读它的 prompt 与 JSON 结构，但别当生产系统用——它没有队列、没有断点续跑、没有统一抽象层。

【合规】本次调研仅覆盖技术与许可证事实。许可证层面有两个真实的商用风险点必须点名：Anim-Director 无任何 LICENSE 文件（默认保留全部权利，法律上不可商用）、ComfyUI 本体 GPL-3.0 与 worker-comfyui 的 AGPL-3.0 在闭源分发时会触发义务。角色虚构、设定成年、不涉真人肖像的内容策略与本调研的技术选型不冲突，但要注意 Seedance 官方通道的审核策略我未能取证，上线前需自行向火山引擎确认书面条款。

### 存疑
- Seedance 官方通道（火山引擎 Ark）的内容审核策略：docs.volcengine.com/docs/82379/1520757 与 /1824157 两个文档页在重定向后返回空内容，未能取证官方 CNY 定价、模型 id 与审核条款原文。用户给的『官方通道有内容审核』这条硬约束我无法独立证实，只能说 fal.ai 的模型页上也没有具体审核条款（只有 Trust & Safety 与 Verify fal-Generated Content 两个链接）。上线前需向火山引擎索要书面条款。
- seed.bytedance.com/en/seedance 官方页面只展示 Seedance 1.0 的能力描述（多镜头、1080p），在模型列表里提到 Seedance 2.5 但不给任何规格；Seedance 2.0 的官方一手规格页我没找到。本报告中 Seedance 2.0/2.5 的全部数字均来自 fal.ai 与 Replicate 的第三方托管文档，可能与字节官方口径有出入。
- Replicate 上 seedance-2.0 系列的精确单价未取得：replicate.com/bytedance/seedance-2.0 与 /api/pricing 页面均未渲染价格。Replicate 定价页只公开了 wavespeedai/wan-2.1-i2v-480p $0.09/秒、720p $0.25/秒，以及 GPU 秒价 A100-80GB $0.0014/s、H100 $0.001525/s、L40S $0.000975/s。fal 与 Replicate 的 Seedance 价差无法对比。
- AIMLAPI 的视频模型清单与定价完全未取证：aimlapi.com/video-models 与 aimlapi.com/pricing 均返回 HTTP 404。它是否真的聚合了 Seedance 我不能确认。
- Segmind 页面确实列出『Seedance 2.0 API』与『Kling 2.6 Pro Motion Control API』，但定价页不公开任何视频单价，只提供 Model Price Explorer 工具。它是真接入还是仅挂名，以及单价是否有加价，无法判断。
- Diffusers v0.37.0 release note 中提到的 Helios『14B 模型、分钟级生成、17 FPS』来自 release notes 的二手摘要，我没有读到 Helios 的原始模型卡或论文，参数量与帧率待核。同理 v0.40.0 的 MiniMax-H3『视频+音频联合合成』也未经模型卡验证。
- OpenTimelineIO-Plugins 具体打包了哪些 adapter（EDL/CMX3600、FCPX XML、AAF、ALE、RV、Maya、Premiere、XGES）未能列全：PyPI 页面加载失败，GitHub 仓库页只写了『a convenience includes both OpenTimelineIO and a set of plugins maintained by the OpenTimelineIO community』而不枚举。EDL 与 AAF 的支持程度需查各自独立仓库确认。
- RunPod Serverless 的 flex 与 active 两种模式的分档秒价未取得：定价页只给小时价且未区分 worker 类型，官网仅宣称 flex worker 比同行省 25%。本报告中的秒价是我按小时价除以 3600 换算的，不是官方公布值。
- Emily2040/seedance-2.0 的 7,362 stars 与 1,068 forks 数字来自 GitHub API 实时读取，可信；但这个 star 数对一个 2026-02 才创建的纯 Markdown skill 仓库而言增长异常快（7 个月 7.3k star、1k fork，fork/star 比高达 14.5%，远高于文档类仓库常见的 3-5%）。我无法判断这是真实热度还是推广结果，建议你自己看 star 曲线再决定投入程度。
- Anim-Director（HITsz-TMG/Anim-Director，258★）的 arXiv 编号、发表会议与具体架构未能取证——github.com/HITsz-TMG/Anim-Director 的 WebFetch 调用失败无输出，仅从 GitHub 搜索 API 拿到仓库名、描述、星数、无许可证、pushed_at 2026-01-07。它是否真的可跑、依赖哪些模型，未验证。
- Prefect 的 result persistence 写入 S3 的具体配置（ResultStorage / S3Bucket block）未在取到的文档片段中出现，只确认了默认落在 ~/.prefect/storage/。若要用 S3 做结果存储需另行查证。
- Dagster 的 partition/IO manager/retry 的具体 API 名称未取得：docs.dagster.io 的两个页面返回的主要是导航菜单而非正文。本报告中关于 Dagster 分区重跑的描述是概念层面的，未经 API 级验证。


====================================================================================================
# [postprocess-vsr] 出片后处理工业化链路：视频超分(VSR) / 插帧(VFI) / 镜头接缝与色彩一致 / 成片封装与 AI 标识合规

## SeedVR2 (3B / 7B)  (ByteDance Seed)
- 类别/成熟度: open-weights / **production** | 许可: Apache-2.0（代码+权重） | 成本: 1×H100-80G = 720p/100帧；1080p/2K 需 4×H100-80G。消费级 4090 需 FP8/GGUF + BlockSwap，速度掉到 ~20s/帧（3B）
- 是什么: 一步扩散（diffusion adversarial post-training）视频修复/超分模型，SeedVR 的二代。输入任意分辨率，单步采样出高分辨率视频。
- 关键事实:
    * arXiv 2506.05301，2025-06-05 提交；代码与权重 2025 年 6 月released；2026-01-27 被 ICLR2026 接收
    * 两档：3B（由 7B 蒸馏）与 7B（generator 实测 8,239.6 M 参数）
    * 论文实测：720p / 100 帧，SeedVR2-7B 单步 269.0–299.4 秒；对比 SeedVR-7B 50 步 1284.8 秒、STAR 50 步 2326.0 秒、Upscale-A-Video 50 步 1284.5 秒
    * 官方显存口径：1×H100-80G 可处理 100×720×1280；4×H100-80G（sp_size=4）才上 1080p / 2K
    * 许可证 Apache 2.0（代码与权重同）
    * ComfyUI 社区版（numz/RedbeardNZ SeedVR2_comfyUI，Apache-2.0）提供 FP16/FP8/GGUF：3B-FP8 约 6GB 权重、实测仍需 ≥18GB 显存；4090 上 768px→2K 单图 30–60 秒，社区口径 3B ≈20 秒/帧、7B ≈60 秒/帧
    * 作者自陈瓶颈：causal video VAE 的编解码比普通 VAE 慢 4 倍以上，720p/100 帧时 VAE 占总耗时约 95%
    * 作者自陈局限：对「极轻度退化」的输入会过度生成细节；repo 自称 prototype models，重度退化与时序一致性仍有失败例
- 长片作用: 顶「离线终稿超分」工位：分镜切片（4–15s）逐段送进去，不做实时。按论文速度，7B 处理 8 分钟 24fps 成片（11520 帧，720p）约需 8.5–9.6 小时单 H100；3B 约折半。对 Seedance 这类扩散生成片的「塑料感 / 压缩块」修复效果是这批开源模型里最好的一档，但绝不能放在预览环节。注意它会「加细节」——数字人脸部易被改写，必须锁定 3B 或降低增强强度做人脸一致性回归测试。
- 来源: https://github.com/ByteDance-Seed/SeedVR https://arxiv.org/abs/2506.05301 https://huggingface.co/ByteDance-Seed/SeedVR2-3B

## FlashVSR / FlashVSR v1.1  (OpenImagingLab（上海 AI Lab 系）/ 清华)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: A100 80G 为官方最优路径；低显存需 FlashVSR-Pro 的 tiling 分块
- 是什么: 首个面向实时的一步扩散「流式」VSR 框架：三阶段蒸馏 + locality-constrained sparse attention + tiny conditional decoder。
- 关键事实:
    * CVPR 2026 论文；v1 权重 2025-10 发布，v1.1（稳定性与保真度增强）2025-11 发布
    * 官方实测：768×1408 视频在单张 A100 上 ~17 FPS
    * 相对前代一步扩散 VSR 最高约 12× 加速
    * 许可证 Apache-2.0；权重在 HuggingFace JunhaoZhuang/FlashVSR
    * 仅针对 4× SR 训练与优化，官方明确建议用 4× 档位以保证稳定
    * 依赖 Block-Sparse Attention 后端，Python 3.11.13；官方说明：A100/A800 最优，H200 加速有限，其它 GPU 兼容性未知
    * 同时开源了训练集 VSR-120K（120k 视频 + 180k 图像）
    * 生态：ComfyUI 节点（1038lab/ComfyUI-FlashVSR、smthemex/ComfyUI_FlashVSR）、FlashVSR-Pro（Docker + NVENC + 低显存 tiling）
- 长片作用: 顶「产线级批量超分」工位——它是这份清单里唯一能把 2–8 分钟片子的超分从「小时级」压到「分钟级」的开源选项。按 17fps 算，11520 帧（8 分钟 24fps）约 11.3 分钟单 A100，比 SeedVR2-7B 快约 50 倍。建议分工：日常迭代/预览全部走 FlashVSR，只有最终交付镜头（或人脸大特写）才切 SeedVR2。风险点是它只保证 4× 且 GPU 兼容面窄，本机若是 4090 需要先做 Block-Sparse Attention 编译验证，不能当作已验证能力写进排期。
- 来源: https://github.com/OpenImagingLab/FlashVSR https://zhuang2002.github.io/FlashVSR/ https://huggingface.co/JunhaoZhuang/FlashVSR

## STAR (Spatial-Temporal Augmentation with T2V)  (南京大学 PCA Lab)
- 类别/成熟度: open-weights / **usable** | 许可: MIT（I2VGen-XL 变体）/ CogVideoX License（CogVideoX 变体） | 成本: 39GB 才跑得动 72 帧 426×240 的 4×；1080p 基本不可行
- 是什么: 把 T2V 扩散先验（I2VGen-XL / CogVideoX-5B）搬进真实世界 VSR，用局部信息增强模块抑制生成过强导致的保真度损失。
- 关键事实:
    * ICCV 2025；arXiv 2501.02976；2025-01-07 放出预训练权重与推理代码，2025-07-01 放出 I2VGen-XL 训练代码
    * 两档权重：light_deg.pt（轻度退化）、heavy_deg.pt（重度退化）；另有 CogVideoX-5B 版
    * 显存实测（官方 README）：4× 放大 72 帧 426×240 需要约 39GB 显存；建议最低 24GB 并调小 frame_length
    * CogVideoX 变体被锁死在 720×480
    * 许可证：I2VGen-XL 变体 MIT；CogVideoX-5B 变体走 CogVideoX 自家许可
    * 论文实测（SeedVR2 论文对照表）：720p/100 帧 50 步需 2326.0 秒，参数量 2041 M
- 长片作用: 不建议进产线。39GB 只换来 72 帧 240p 输入，折算到 2–8 分钟成片是灾难性的吞吐；720p/100 帧 2326 秒意味着 8 分钟片要 74 小时。它的价值是「学术对照组」——用来验证 SeedVR2/FlashVSR 在你自己素材上的相对优势，或者对个别难修复的镜头做单段抢救。
- 来源: https://github.com/NJU-PCALab/STAR https://arxiv.org/abs/2501.02976 https://nju-pcalab.github.io/projects/STAR/

## Upscale-A-Video (UAV)  (NTU S-Lab（周尚辰）)
- 类别/成熟度: open-weights / **research** | 许可: NTU S-Lab License 1.0（非商业） | 成本: 未公布明确显存口径；速度上 49 帧 720p ≈ 8.5 分钟
- 是什么: CVPR 2024 Highlight，基于文本引导的时序一致扩散 VSR，局部用 3D 卷积+时序注意力，全局用带光流的递归潜变量传播。
- 关键事实:
    * CVPR 2024 Highlight；仓库 sczhou/Upscale-A-Video
    * 许可证 NTU S-Lab License 1.0 —— 明确的非商业许可
    * 速度：49 帧 720p 需要 510 秒推理（论文口径）；SeedVR2 论文对照表记为 720p/100帧 50 步 1284.5 秒，参数量 691 M
    * 支持 LLaVA 辅助生成 prompt；官方示例覆盖 AIGC 视频、老片、电影、动画
- 长片作用: 对商业「AI 数字人拍片平台」来说是法务红线：S-Lab 1.0 非商业条款直接排除它进产线的可能，不管效果如何。只能作为内部效果基线对比使用，输出物不得进入交付成片。技术上它也已被 SeedVR2 在速度与质量上双超。
- 来源: https://github.com/sczhou/Upscale-A-Video https://shangchenzhou.com/projects/upscale-a-video/

## VEnhancer  (Vchitect（上海 AI Lab）/ 南洋理工)
- 类别/成熟度: open-weights / **research** | 许可: 仓库未在检索到的页面明示许可证（需人工核对 LICENSE 文件） | 成本: ≥60GB VRAM；6 秒片段 40–50 分钟/A100
- 是什么: 一体化生成式时空增强：同时做空间超分、时间超分（插帧）与视频精修，专门针对 AI 生成视频去除空间伪影与时序闪烁。
- 关键事实:
    * arXiv 2407.07667（2024-07）；2024-08 开源；2024-09 放出 v2 checkpoint（纹理细节更多、身份保持更好）
    * 硬性显存：官方要求单卡 ≥60GB VRAM，推荐 H100 / A100
    * 实测：单卡 A100 默认参数增强一段 6 秒 CogVideoX 视频，占用 60GB 显存、耗时 40–50 分钟
    * 可任意倍率同时放大空间与时间分辨率（这是它区别于纯 VSR 的点）
- 长片作用: 设计目标（专治 AI 生成片伪影 + 同时插帧）恰好对口你的场景，但吞吐完全不可产线化：6 秒 45 分钟意味着 8 分钟成片约 60 小时单 A100，且 60GB 门槛直接排除 4090/5090。结论是「概念上正确、工程上作废」，它的职能已经被 FlashVSR（空间）+ RIFE（时间）这对组合以数百倍的成本优势取代。
- 来源: https://github.com/Vchitect/VEnhancer https://arxiv.org/abs/2407.07667 https://vchitect.github.io/VEnhancer-project/

## Real-ESRGAN（含 realesr-animevideov3）  (腾讯 ARC / xinntao)
- 类别/成熟度: open-weights / **production** | 许可: BSD-3-Clause | 成本: 消费级显卡即可，成本近似为零
- 是什么: 纯 GAN 单帧超分，逐帧跑视频。有专门的动画视频小模型 realesr-animevideov3 与 x4plus_anime_6B。
- 关键事实:
    * 许可证 BSD-3-Clause，36.8k stars
    * 无任何时序模块：逐帧独立推理，AI 生成片上必然出现帧间闪烁与纹理游走
    * 官方文档自承：ncnn 可执行文件走 tile 分块，会引入 block inconsistency，且与 PyTorch 实现结果略有差异
    * 模型极小，可在 6–8GB 显存卡上跑；ncnn-vulkan 版本不挑 N 卡
- 长片作用: 只顶两个边角工位：(1) 静态素材/参考图（Seedance 的 9 张参考图）预放大；(2) 极限降本的草稿预览超分。绝不能用于交付成片——逐帧 GAN 在 2–8 分钟连续画面上的闪烁会被观众立刻察觉，这正是「单镜观感接近 Seedance 2.0」目标的反面。
- 来源: https://github.com/xinntao/Real-ESRGAN

## Topaz Video（原 Video AI）+ Starlight  (Topaz Labs（2026-06-25 起被 Adobe 收购）)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业订阅 | 成本: $299/年（Personal）/ $699/年（Pro）/ 云端 $39 月起 + credits
- 是什么: 闭源商业视频增强套件；Starlight 是其扩散式（生成式）超分模型，分本地与云端版本。
- 关键事实:
    * Adobe 于 2026-06-25 宣布达成收购 Topaz Labs 的最终协议，预计 2026 下半年完成交割（待监管批准）；Adobe 声明 Topaz 产品将继续作为独立产品销售，Eric Yang 继续带队
    * 定价（2026）：Personal 订阅 $299/年；Pro $699/年（含 Starlight）；云端计划另起 $39/月含一定额度 credits
    * credit 口径：1 分钟 1080p30 升 4K 用若干核心模型约消耗 34 credits；部分 Starlight 任务消耗更高
    * 永久授权版 Video AI 已停售，存量用户不再获得新模型更新
    * Personal 授权下部分 Starlight 模型强制走云端处理
    * Adobe 同时看中 Topaz 的 Neurostream（端侧跑大模型）技术
- 长片作用: 作为闭源对照组与「交付兜底」有价值，但两个硬伤与你的平台冲突：(1) 云端 Starlight 意味着素材出境 + 二次审核风险，跟你已有的 Seedance 官方审核叠加；(2) credit 计费在 2–8 分钟 × 大量重拍的产线上会线性爆炸。被 Adobe 收购后中长期还有产品线并入 Firefly、API 政策变动的不确定性。建议只用作质量标尺：拿同一批镜头跑 Topaz 与 SeedVR2/FlashVSR 做盲评，确定开源档位是否够用，而不是把它焊进产线。
- 来源: https://news.adobe.com/news/2026/06/adobe-to-acquire-topaz-labs https://www.topazlabs.com/ https://pcai.nero.com/blog/topaz-labs-alternative

## TSD-SR  (Microtreei 等（CVPR 2025）)
- 类别/成熟度: open-weights / **research** | 许可: 以仓库 LICENSE 为准（检索页面未明示） | 成本: 单步图像 SR，显存需求显著低于视频扩散模型
- 是什么: 一步扩散的真实世界「图像」超分蒸馏框架（Target Score Distillation + Distribution-Aware Sampling Module）。
- 关键事实:
    * CVPR 2025，论文页 pp.23174–23184；arXiv 2411.18263
    * 定位是 Real-ISR（image super-resolution），不是视频 SR：无任何时序模块
    * 宣称在扩散类方法中速度最快（单步）
    * 仓库 github.com/Microtreei/TSD-SR
- 长片作用: 被点名要覆盖，但必须澄清工位错配：它是图像模型，逐帧套用在视频上和 Real-ESRGAN 一样会闪烁，且扩散生成的随机性会让闪烁比 GAN 更严重。在你的产线里唯一合理用途是「分镜关键帧/参考图增强」这一前置工位，不进视频链路。
- 来源: https://arxiv.org/abs/2411.18263 https://github.com/Microtreei/TSD-SR https://openaccess.thecvf.com/content/CVPR2025/html/Dong_TSD-SR_One-Step_Diffusion_with_Target_Score_Distillation_for_Real-World_Image_CVPR_2025_paper.html

## STCDiT  (学术（项目页 jychen9811.github.io/STCDiT_page）)
- 类别/成熟度: technique / **research** | 许可: 未知 | 成本: 未公布
- 是什么: 时空一致的 Diffusion Transformer VSR：motion-aware VAE 重建（按运动一致性切 clip）+ anchor-frame 引导（用每个 clip 首帧的结构信息约束生成）。
- 关键事实:
    * arXiv 2511.18786，2025-11-24 提交
    * 基于预训练视频扩散模型，宣称在结构保真与时序一致性上超过 SOTA
    * 检索到的信息中未给出权重/代码发布状态、GPU 需求、与 SeedVR2/FlashVSR 的直接对照数字
- 长片作用: 不要排进 2026 Q4 产线。但它的 anchor-frame 思路与你的「分镜拼接」架构天然契合——每个分镜段用首帧做结构锚定，正是跨段一致性的解法之一，值得作为自研方向的参考，而不是现成组件。
- 来源: https://arxiv.org/abs/2511.18786

## NTIRE 2026 短视频 UGC 修复挑战赛 / KwaiVIR 基准  (中科大 + 快手（CVPR 2026 Workshop）)
- 类别/成熟度: technique / **research** | 许可: 以数据集发布方条款为准 | 成本: ?
- 是什么: 面向真实世界短视频（S-UGC）的生成式视频修复公开挑战赛与配套基准数据集。
- 关键事实:
    * arXiv 2604.10551，2026-04-12；CVPR 2026 workshop 录用
    * 95 支队伍注册，12 支提交有效最终方案
    * KwaiVIR 数据集：200 段合成训练视频 + 48 段真实野外训练视频 + 11 段验证 + 20 段测试
    * 综述结论仅为「在 KwaiVIR 上取得强性能，生成式 S-UGC 修复进展令人鼓舞」，检索到的摘要未点名冠军方案
- 长片作用: 这是 2026 年评估 VSR 的最新公开标尺。在你的产线里的用途是「选型依据与回归集」：把 KwaiVIR 的测试集加上你自己的 Seedance 输出样张，建一个内部盲评 harness，用同一套指标（DOVER/E_warp/LPIPS + 人眼盲评）定期重测 SeedVR2 / FlashVSR，而不是每半年凭传言换模型。
- 来源: https://arxiv.org/abs/2604.10551

## Practical-RIFE v4.25 / v4.26 / v4.26.heavy  (hzwer（旷视系作者）)
- 类别/成熟度: open-weights / **production** | 许可: MIT | 成本: 消费级显卡（4090 上 1080p 插帧为实时量级），显存需求个位数 GB
- 是什么: 实时光流插帧（IFNet 架构），任意时刻 t 插帧，目前工业界事实标准。
- 关键事实:
    * v4.26 发布于 2024-09-21；v4.25 发布于 2024-09-19（官方推荐大多数场景默认用 4.25）；4.25.lite 发布于 2024-10-20；截至检索日未见 2025–2026 年新版本
    * v4.26 是 v4.25 训练的完成版；v4.26.heavy 为最高质量档，算力需求更高
    * v4.17（2024-05-24）起引入了 FILM 的 gram loss
    * 仓库声明 MIT 许可证，模型下载链接内容适用同一 MIT 许可
    * 仓库无官方 fps benchmark；作者明确该仓库面向工程师，普通用户建议用 SVFI / RIFE-App / FlowFrames 等外壳
    * 生态成熟：vs-mlrt、vsrife（PyPI）、video2x 均已支持 4.25/4.26
- 长片作用: 顶「24→48/60fps 提帧」主力工位，也是唯一在 2–8 分钟长片上成本可接受的选项。产线要点：(1) 必须先做镜头切分再插帧——跨分镜硬切处插帧会产生融化态过渡帧，这在你的拼接架构里是高频故障；(2) Seedance 输出若已带运动模糊，2× 插帧质量通常稳，4× 以上在快速动作上会拖影；(3) 数字人口型段落建议按音频节奏验证，插帧后要重新核对唇音同步。
- 来源: https://github.com/hzwer/Practical-RIFE https://pypi.org/project/vsrife/ https://deepwiki.com/hzwer/Practical-RIFE/5.1-rife-model-versions

## FILM (Frame Interpolation for Large Motion)  (Google Research)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: 消费级显卡可跑，但比 RIFE 慢一个量级
- 是什么: ECCV 2022 的大位移插帧网络：多尺度金字塔特征提取 + 双向运动估计 + 融合模块，无需额外光流/深度网络。
- 关键事实:
    * arXiv 2202.04901，ECCV 2022；许可证 Apache-2.0
    * 仓库 google-research/frame-interpolation 已处于归档/无 CI 状态，元数据显示最后更新 2026-02-18
    * 原始实现为 TensorFlow；社区有 PyTorch 移植（jkawamoto/frame-interpolation-pytorch）与 Replicate API
    * 其 gram loss 已被 RIFE v4.17+ 吸收
- 长片作用: 在你的链路里只顶一个窄工位：两个分镜之间的「大位移过渡帧生成」。FILM 的设计初衷是近似重复照片之间补运动，正好适合把 A 镜末帧与 B 镜首帧强行插出几帧做 morph 过渡（比 xfade 溶解自然）。常规 24→60fps 提帧不要用它，RIFE 更快更稳且已经吸收了它的损失函数。仓库归档意味着不会再有新模型，视为冻结依赖。
- 来源: https://github.com/google-research/frame-interpolation https://arxiv.org/abs/2202.04901 https://film-net.github.io/

## GIMM-VFI (GIMM-VFI-R / GIMM-VFI-F)  (NeurIPS 2024（GSeanCDAT）)
- 类别/成熟度: open-weights / **usable** | 许可: 需核对仓库 LICENSE（未在检索页面明示） | 成本: 显存需求未公开；速度约为 RIFE 的 1/3–1/6
- 是什么: 可泛化隐式运动建模：从预训练光流器（RAFT / FlowFormer）抽双向流，编码时空运动隐变量，再用坐标神经网络隐式预测任意时刻光流。
- 关键事实:
    * NeurIPS 2024；arXiv 2407.08680；仓库 GSeanCDAT/GIMM-VFI
    * 两个变体：GIMM-VFI-R（RAFT 基）、GIMM-VFI-F（FlowFormer 基，更重更准）
    * 支持任意时刻 t 插帧（连续运动建模），不限于 2×/4× 整数倍
    * ComfyUI 侧（kijai/ComfyUI-GIMM-VFI）依赖 cupy，目前是 NVIDIA-only
    * 社区匹配片段计时：RIFE 约 9 秒时，GIMM 两个变体约 25–53 秒（约 3–6× 慢）
    * 主仓库许可证在检索到的页面未明示，需人工核对 LICENSE
- 长片作用: 顶「RIFE 失效镜头的返工工位」。具体触发条件：快速横摇、物体穿越遮挡、混乱动作——这些场景 RIFE 会鬼影和涂抹，GIMM 因为显式建模了连续光流能稳住。产线做法是 RIFE 全量跑 + 自动检测（帧差/光流残差异常）挑出问题段落，只对这些段落用 GIMM 重跑，而不是全片用 GIMM（成本 3–6 倍）。商用前必须先确认许可证。
- 来源: https://github.com/GSeanCDAT/GIMM-VFI https://arxiv.org/html/2407.08680v4 https://gseancdat.github.io/projects/GIMMVFI

## OpenColorIO 2.5.2 + 内置 ACES 2.0 config  (Academy Software Foundation)
- 类别/成熟度: framework / **production** | 许可: BSD-3-Clause（ASWF 标准） | 成本: CPU 库，零边际成本
- 是什么: 电影工业标准色彩管理库；从 v2.5.0 起内置 ACES 2 配置，无需外挂 OCIO config 文件。
- 关键事实:
    * 最新发布 v2.5.2，2025-05-13
    * v2.5.0 的头条特性即「built-in ACES 2 configs」
    * ASWF 项目，采用 BSD-3-Clause（以仓库 LICENSE 为准）
    * 有 Python 绑定（PyOpenColorIO），可脚本化做 colorspace 转换与 LUT 烘焙
    * Nuke / Resolve / Blender / OIIO 均原生集成（OpenImageIO 最新 v3.1.17.0，2026-09-01）
- 长片作用: 顶「色彩管理基线」工位，是解决跨分镜色彩漂移的正确底座而非补丁。做法：Seedance 输出的 8bit Rec.709 片段统一 IDT 进 ACEScct 工作空间 → 在统一空间里做跨段匹配与调色 → 最后一次性 ODT 出 Rec.709/PQ。这样每段的色彩操作是可叠加、可回滚、数学一致的；相比直接在 sRGB 上用 colorbalance 逐段硬调，跨段一致性提升是结构性的。用 PyOpenColorIO 可以把这一步完全脚本化进你的批处理平台。
- 来源: https://github.com/AcademySoftwareFoundation/OpenColorIO/releases https://opencolorio.org/

## DaVinci Resolve 21 脚本化批量套色（Python API）  (Blackmagic Design)
- 类别/成熟度: service / **production** | 许可: 闭源商业（Studio $295 买断，非订阅） | 成本: $295 一次性（Studio）；免费版是否支持外部脚本需自行核实
- 是什么: Resolve 内置 Lua/Python 脚本 API，可程序化导入素材、建时间线、逐 clip 套 LUT / CDL / 已存 grade，并触发批量渲染。
- 关键事实:
    * 当前版本 DaVinci Resolve 21（2026）；Studio 买断 $295
    * Console 支持 Python 2.7 / Python 3.6 / Lua；外部脚本需设置 RESOLVE_SCRIPT_API / RESOLVE_SCRIPT_LIB 等环境变量
    * 对象层级：Resolve → ProjectManager → Project → Timeline → TimelineItem / MediaPool / MediaStorage
    * 调色相关方法：TimelineItem.SetLUT()（按节点套 LUT）、ApplyGradeFromDRX()（从 .drx 文件套已存 grade）、SetCDL()（套 slope/offset/power/saturation 的 ASC-CDL）
    * 默认脚本只允许在 Fusion Console 内运行，需在 Preferences 里开放命令行/网络调用权限
    * 支持 DaVinci Wide Gamut / Intermediate log 调色环境与 Dolby Vision
- 长片作用: 顶「成片统一套色 + 人工兜底」工位，是整条链路里唯一有专业监看与人工介入能力的环节。产线设计：平台把所有分镜片段 + 一份基准 CDL/LUT 通过 Python API 灌进一条时间线，用 SetCDL() 批量套基准值，再由调色师只对少数偏色段落手动修正，最后脚本触发 Deliver 渲染。$295 买断对商业产线是可忽略成本。注意 Python 3.6 的版本约束——你的平台主进程若跑在 3.11+，需要用子进程/RPC 隔离调用，不要试图同进程 import。
- 来源: https://www.blackmagicdesign.com/products/davinciresolve/whatsnew https://deric.github.io/DaVinciResolve-API-Docs/

## 跨片段色彩匹配算法（直方图匹配 / Reinhard / MKL-Monge-Kantorovich）  (经典算法（Reinhard et al. 2001）)
- 类别/成熟度: technique / **production** | 许可: 算法本身无许可限制；具体实现库各自（scikit-image BSD-3） | 成本: CPU 毫秒级/帧，成本可忽略
- 是什么: 把 B 片段的色彩统计量对齐到 A 片段（或对齐到全片基准帧）的一族算法：逐通道直方图规定化；Reinhard 在 Lab 空间做均值/标准差线性迁移；MKL 做全协方差矩阵的线性色彩迁移。
- 关键事实:
    * Reinhard 方法：转 Lαβ（去相关色彩空间），对每通道做 (x−μ_src)/σ_src·σ_ref+μ_ref，三通道各 2 个参数，共 6 参数，计算量近似为零
    * 直方图匹配：逐通道 CDF 映射，对色调分布差异大的片段更强，但容易产生色带与块状伪影（8bit 素材尤甚）
    * MKL / Monge-Kantorovich 线性迁移：用 3×3 协方差矩阵 + 平移，保留通道相关性，比 Reinhard 更强、比直方图匹配更平滑
    * Python 生态：scikit-image 的 match_histograms()、OpenCV 手写 Reinhard、color-matcher 包（含 MKL / Reinhard / HM 多种）
    * 关键失效模式：若两段画面内容差异大（一段有大面积天空、一段没有），基于全局统计的匹配会过校正——必须按「同场景同光照」分组匹配，或只用中间帧的人脸/主体 ROI 做统计
- 长片作用: 顶「分镜接缝色彩一致」的第一道自动工位。推荐产线策略：选一段作 anchor（通常是主角正脸中景），对每段取中间帧、按人脸/主体 ROI 提统计量，用 MKL 或 Reinhard 求出 3×3+平移 的线性变换，然后把这个变换固化成一个 .cube LUT，用 FFmpeg lut3d 全段烘焙——这样整段是同一个确定性变换，不会出现逐帧统计导致的时序抖动。绝对不要逐帧做直方图匹配，那会引入新的闪烁。
- 来源: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html https://github.com/hahnec/color-matcher

## overlap 混接 + 光流引导过渡（分镜拼接接缝方案）  (工程实践（RIFE/FILM + FFmpeg 组合）)
- 类别/成熟度: technique / **usable** | 许可: N/A（FFmpeg LGPL/GPL，RIFE MIT，FILM Apache-2.0） | 成本: xfade 近乎免费；插帧过渡按 RIFE/FILM 成本计
- 是什么: 在两个分镜片段之间不做硬切，而是让生成阶段多出 N 帧重叠区，在重叠区做溶解、或用插帧网络在 A 末帧与 B 首帧之间生成光流引导的中间帧。
- 关键事实:
    * 三档方案，成本递增：(1) FFmpeg xfade 纯像素溶解；(2) 重叠区取 A/B 各若干帧交叉溶解 + 匹配后的色彩变换；(3) 用 FILM / RIFE 在 A 末帧–B 首帧之间生成 4–12 帧 morph 过渡
    * xfade 的硬性前提：两路输入必须分辨率、像素格式、时基一致，且 offset 以第一路输入起点为基准的秒数计；transition 类型数十种（fade/wipe*/slide*/circle*/dissolve/pixelize 等），duration 默认 1 秒
    * 光流插帧做过渡的适用边界：仅当 A 末帧与 B 首帧构图接近（同机位、同主体）时有效；跨机位强行 morph 会产生明显的液化伪影，此时应退回溶解或硬切
    * Seedance 官方单段 4–15 秒的限制意味着 2–8 分钟成片需要约 8–120 个接缝，接缝策略必须自动化且可回退
- 长片作用: 这是你整个「靠分镜系统拼接」架构最核心的风险点，也是「单镜观感接近 Seedance 2.0」最容易崩的地方。建议做成分级决策器：先算 A 末帧与 B 首帧的全局运动/构图相似度，高相似度走 RIFE morph（观感最接近连续单镜），中等走重叠溶解，低相似度直接硬切（硬切比失败的 morph 好得多）。同时在生成阶段就要求每段多生成 0.5 秒余量，给接缝留操作空间——这个决策必须前置到分镜/生成模块，事后补救成本高得多。
- 来源: https://ffmpeg.org/ffmpeg-filters.html#xfade https://github.com/hzwer/Practical-RIFE https://github.com/google-research/frame-interpolation

## FFmpeg 后处理滤镜链（xfade / minterpolate / colorbalance / lut3d）  (FFmpeg)
- 类别/成熟度: framework / **production** | 许可: LGPL-2.1+ / GPL-2+（取决于编译开关，--enable-gpl 后为 GPL） | 成本: CPU 为主，零许可成本；注意 GPL 编译分发义务
- 是什么: 整条后处理链路的胶水层：转场、兜底插帧、一级色彩校正、LUT 烘焙、拼接、封装全部可在一条命令里完成。
- 关键事实:
    * xfade：参数 transition / duration / offset / expr；要求两路输入同分辨率、同像素格式、同时基
    * minterpolate：参数 fps、mi_mode（dup/blend/mci）、mc_mode（obmc/aobmc）、me_mode（bidir/bilat）、me、mb_size、search_param、vsbmc、scd；典型高质量组合 mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1
    * minterpolate 是纯 CPU 块匹配，速度比 RIFE 慢若干数量级，且在 AI 生成片的非刚体运动上块状伪影明显——只应作为无 GPU 环境的兜底
    * lut3d：吃 .cube / .3dl 格式 3D LUT，可把前面算出的色彩变换一次性烘焙进片段，是把「色彩匹配算法」落地成确定性操作的标准手段
    * colorbalance：rs/gs/bs（阴影）、rm/gm/bm（中间调）、rh/gh/bh（高光）九参数 + pl（保亮度），适合快速一级校正，但不如 lut3d 可复现可审计
    * concat demuxer 要求所有片段编码参数一致，否则必须走 concat filter 重编码
- 长片作用: 顶「产线执行层」。所有模型（VSR/VFI/色彩）的产出最终都要由 FFmpeg 落地成帧序列或成片。两个必须踩死的工程细节：(1) 中间环节一律走无损或近无损（ProRes / FFV1 / x264 CRF≤12），绝不用有损中间格式串联，否则 8 段串下来压缩伪影会被 VSR 放大；(2) 分发用 GPL 编译的 FFmpeg（x264/x265 需要 --enable-gpl）有开源义务，商业平台需要法务确认部署形态。
- 来源: https://ffmpeg.org/ffmpeg-filters.html https://ffmpeg.org/legal.html

## 成片编码参数（H.264 / H.265 / AV1）  (x264 / x265 / SVT-AV1 / NVENC)
- 类别/成熟度: technique / **production** | 许可: x264/x265 GPL（商业需买商业授权或用 LGPL 替代）；SVT-AV1 BSD-3-Clause + AOM 专利许可 | 成本: x265 商业授权需向 MulticoreWare 购买；SVT-AV1 无授权费
- 是什么: 最终封装环节的编码器选型与参数，决定交付体积、兼容性与观感。
- 关键事实:
    * H.264（libx264）：兼容性最好，CRF 18–20 + preset slow + profile high + level 4.2 可覆盖 1080p 交付；yuv420p 是全平台兼容的唯一安全像素格式
    * H.265（libx265）：同画质比 H.264 省约 30–50% 码率，CRF 20–23 对应 x264 的 18–20；Apple 生态需 hvc1 tag（-tag:v hvc1）否则 QuickTime/iOS 不认
    * AV1（SVT-AV1，libsvtav1）：preset 数值越小越慢越好；preset 4–6 是质量/速度平衡区，preset 8+ 用于快速预览；同画质比 H.265 再省约 20–30%，但老设备解码支持差
    * 硬件编码：NVENC 的 AV1 编码需要 Ada（RTX 40 系）及以上；硬件编码速度快一个量级但同码率画质低于 x264/x265 的 slow 档，只建议用于预览
    * 关键：AI 生成的数字人画面有大面积低纹理平面 + 高频面部细节，容易在中低码率下出现 banding，交付建议加 -pix_fmt yuv420p10le（10bit）或在编码前加轻微 dither/grain
    * faststart：-movflags +faststart 是 Web 交付必选，否则 moov atom 在文件尾导致无法边下边播
- 长片作用: 顶「交付封装」工位。对 2–8 分钟成片的具体建议：主交付 H.264 High@L4.2 CRF 18 + faststart（兼容性兜底），同时出一份 H.265 或 AV1 作为高效版本。特别注意 x264/x265 的 GPL 授权——商业平台如果把 FFmpeg 打进产品分发，x265 需要商业授权，这是容易被忽略的法务坑；SVT-AV1 走 BSD 没有这个问题，这是 AV1 在商业产线里的一个隐性优势。
- 来源: https://trac.ffmpeg.org/wiki/Encode/H.264 https://trac.ffmpeg.org/wiki/Encode/H.265 https://gitlab.com/AOMediaCodec/SVT-AV1

## 音画同步与字幕烧录  (FFmpeg 生态)
- 类别/成熟度: technique / **production** | 许可: libass ISC；FFmpeg 视编译开关 | 成本: CPU，成本可忽略
- 是什么: 多段拼接后的音视频对齐、以及硬字幕（burn-in）/软字幕（mov_text / WebVTT）的处理。
- 关键事实:
    * Seedance 自带对白声意味着每个分镜段是「带音轨的独立片段」，拼接时音视频必须同轨切割，不能分别处理
    * concat demuxer 要求每段的音频采样率、声道布局、编码完全一致，否则出现累积漂移；安全做法是先把每段音频统一重采样（aresample=async=1:first_pts=0）再拼
    * 硬字幕：subtitles 滤镜（需 libass 编译）吃 .ass/.srt，可控字体/描边/位置；烧录后不可关闭，但保证任何播放器一致呈现
    * 软字幕：mp4 容器用 mov_text（-c:s mov_text），mkv 用 ass/srt；软字幕不增加画质损失但部分平台不显示
    * VSR 与插帧必须在字幕烧录之前完成——超分会把字幕边缘一起放大导致锯齿，插帧会让字幕出现半透明重影
    * 音画同步验证：拼接后用 ffprobe 比对 video/audio stream 的 duration 与 start_time，差值 >40ms（1 帧 @24fps）即需要修正
- 长片作用: 顶「交付前最后一道装配」工位。在你的架构里最大的具体风险是：Seedance 的自带对白声 + 插帧到 48/60fps 的组合。插帧只改视频时间轴的采样密度、不改时长，理论上不影响同步，但如果你的实现是「抽帧重排」而非「时长保持插帧」就会漂。建议在管线里加一个强制断言：每段插帧前后的 video duration 必须逐段比对，任何段漂移 >1 帧即中断产出。字幕一定放最后一步。
- 来源: https://trac.ffmpeg.org/wiki/HowToBurnSubtitlesIntoVideo https://ffmpeg.org/ffmpeg-filters.html#subtitles

## C2PA Content Credentials（c2patool / c2pa-rs）  (C2PA / Content Authenticity Initiative（Adobe 牵头）)
- 类别/成熟度: framework / **production** | 许可: Apache-2.0 + MIT（c2pa-rs / c2patool） | 成本: 库免费；证书需向 CA 采购（年费）
- 是什么: 内容来源与编辑历史的密码学签名标准；把「谁生成、用什么模型生成、经过哪些编辑」写进文件的 manifest 并用证书签名。
- 关键事实:
    * 规范当前为 C2PA Specifications 2.3（spec.c2pa.org），含技术规范、实现者指南、AI/ML 专项指南、CDDL/JSON Schema
    * 开源实现 c2pa-rs / c2patool，双许可 Apache-2.0 + MIT
    * 视频写入支持：MP4（video/mp4、application/mp4，分片 MP4/DASH 仅支持基于文件的操作）、MOV（video/quicktime）、AVI；音频支持 MP3/M4A/WAV/FLAC；PDF 只读
    * AI 生成内容通过 assertion 中的 digitalSourceType 标注（IPTC 词表 trainedAlgorithmicMedia 表示完全由 AI 生成）
    * 致命工程约束：C2PA manifest 附着在文件上，一旦平台（微信/抖音/YouTube）二次转码，manifest 通常被剥离——必须配合 soft binding（不可见水印/指纹）才能在分发后仍可溯源
    * 签名需要有效证书链；生产环境要走 C2PA conformance program 认可的签发方，不能用自签证书交付
- 长片作用: 顶「合规封装」工位，且必须紧贴最终编码之后——任何后续重编码都会毁掉签名，所以 C2PA 签名是整条产线的最后一个动作。对你的平台是刚需而非可选项：它是同时满足中国「隐式标识（元数据）」与 EU AI Act 第 50 条「machine-readable marking」的最省力载体，一次实现覆盖两个法域。但要清醒：manifest 不抗转码，若你的分发路径经过第三方平台，必须另外叠加不可见水印。
- 来源: https://spec.c2pa.org/specifications/specifications/2.3/index.html https://github.com/contentauth/c2patool https://github.com/contentauth/c2pa-rs/blob/main/docs/supported-formats.md

## 中国《人工智能生成合成内容标识办法》  (国家网信办等四部门)
- 类别/成熟度: technique / **production** | 许可: N/A（法规） | 成本: 合规成本：元数据写入 + 片头片尾显著标识的渲染
- 是什么: 中国境内 AI 生成合成内容的强制标识规章，要求显式标识 + 隐式标识双轨。
- 关键事实:
    * 2025-03-14 发布，2025-09-01 起施行（截至 2026-09-19 已实施满一年）
    * 显式标识：文本加文字提示或通用符号；音频加语音/音频节奏提示；图片视频在起始画面、结束或适当位置加显著标识；虚拟场景在起始画面加提示
    * 隐式标识：文件元数据中必须包含「生成合成内容属性信息、服务提供者名称或者编码、内容编号」等制作要素信息；鼓励（非强制）加数字水印
    * 传播平台义务：核验文件元数据标识、在分发时为已标识内容加显著提示、对疑似 AI 内容基于检测结果加标、并向用户提供主动声明标识的功能
    * 用户义务：使用 AI 生成内容时应主动声明并使用平台提供的标识功能
    * 明令禁止：恶意删除、篡改、伪造、隐匿标识，或提供帮助他人实施上述行为的工具与服务
    * 配套有强制性国家标准规定具体标识方法（标准编号与技术细节需另行核实）
- 长片作用: 对「长时间 AI 数字人拍片系统平台」是产线必备工位，不是事后贴标。两条要落进封装环节：(1) 隐式标识——用 C2PA manifest 或 mp4 自定义 metadata 写入服务提供者名称/编码 + 内容编号，建议内容编号与你的分镜任务 ID 绑定以便追溯；(2) 显式标识——在成片起始画面自动烧录 AI 生成提示（可用 FFmpeg drawtext/overlay 做成模板化的片头层），这一步必须在字幕烧录同一环节完成。另注意：你走 Seedance 官方通道时字节作为服务提供者已有其标识义务，但你作为二次加工与分发方有独立义务，不能依赖上游。
- 来源: https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm

## EU AI Act 第 50 条（透明度义务）  (European Union)
- 类别/成熟度: technique / **production** | 许可: N/A（法规） | 成本: 合规成本：机器可读标记 + 披露文案
- 是什么: 欧盟 AI 法案对 AI 交互告知、合成内容机器可读标记、深度伪造披露的透明度义务条款。
- 关键事实:
    * 按 Article 113，第 50 条自 2026-08-02 起适用——截至 2026-09-19 已经生效
    * 第 50(2) 条：生成合成音频/图像/视频/文本的提供者必须以机器可读格式标记输出为人工生成或操纵，解决方案须「有效、可互操作、稳健、可靠」（在技术可行范围内）；纯辅助性轻微编辑除外
    * 第 50(4) 条：部署者（deployer）使用深度伪造须披露；AI 生成新闻类内容须披露，除非经负责人编辑审核；艺术、讽刺、虚构作品只需以不妨碍观赏的方式表明内容存在
    * 第 50(1) 条：须让人知道自己在与 AI 系统交互，除非对合理知情的自然人而言显而易见
    * 处罚适用第 99 条行政罚款框架（具体金额需核实条款）
    * 注意 provider 与 deployer 义务不同：你既生成（provider 侧标记义务）又分发（deployer 侧披露义务）
- 长片作用: 若成片有任何面向欧盟受众的分发，这是已生效的硬约束。工程上和中国办法高度可合并实现：C2PA manifest（digitalSourceType=trainedAlgorithmicMedia）一次性满足「机器可读标记」；片头显式提示同时满足中国显式标识与第 50(4) 条披露。第 50(4) 条对「虚构作品」的宽免值得注意——你的内容策略是角色虚构，理论上可适用「以不妨碍观赏的方式表明」的较轻档位，但这不豁免第 50(2) 条的机器可读标记义务（那条是针对 provider 的，无虚构例外）。这一点要请法务确认，不要自行套用宽免。
- 来源: https://artificialintelligenceact.eu/article/50/ https://artificialintelligenceact.eu/article/113/

### 建议
【结论先行】后处理链路不存在选型困难，存在的是吞吐量陷阱。这个维度真正会杀死项目的不是「哪个 VSR 效果好」，而是「8 分钟成片一次超分要不要跑 9 小时」。

【推荐产线，按工位排】
1. 生成余量（前置到分镜模块）：每段 Seedance 输出多要 0.5 秒余量，给接缝留料。这是事后无法补救的决策。
2. 中间格式：全程 ProRes 422 HQ 或 x264 CRF≤12，绝不用有损中间格式串联。
3. 色彩匹配（第一道）：OCIO 2.5.2 把所有段 IDT 进 ACEScct → 选 anchor 段 → 按人脸 ROI 用 MKL/Reinhard 求线性变换 → 固化成 .cube → FFmpeg lut3d 整段烘焙。关键：每段一个确定性变换，禁止逐帧统计匹配（会引入新闪烁）。
4. 接缝：分级决策器。A 末帧/B 首帧构图相似度高 → RIFE morph 过渡（最接近单镜）；中 → 重叠溶解 + 色彩变换；低 → 硬切。失败的 morph 比硬切难看得多。
5. 超分：双档制。日常迭代与预览全量走 FlashVSR v1.1（A100 上 17fps@768×1408，8 分钟片约 11 分钟）；最终交付或人脸大特写切 SeedVR2-3B。SeedVR2-7B 只用于单镜抢救——按论文 720p/100 帧 269–299 秒折算，全片要 8.5–9.6 小时单 H100，不可能进日常循环。
6. 插帧：Practical-RIFE v4.25/v4.26（MIT，可商用）全量；用帧差/光流残差自动挑出鬼影段落，只对这些段落用 GIMM-VFI 重跑。必须先切镜头再插帧。
7. 统一套色与人工兜底：DaVinci Resolve 21 Studio（$295 买断），Python API 批量 SetCDL()/SetLUT()，调色师只修少数偏色段。注意 Resolve 的 Python 3.6 约束，需子进程隔离。
8. 封装：主交付 H.264 High@L4.2 CRF 18 + faststart；同出 H.265(hvc1) 或 SVT-AV1 preset 4–6。10bit 输出以抑制数字人大面积肤色区的 banding。
9. 字幕：最后一步，在 VSR 与插帧之后。同时做音画同步断言（逐段 duration 漂移 >1 帧即中断）。
10. 合规标识：最末一个动作。C2PA（c2patool，Apache+MIT）写 manifest 含 digitalSourceType=trainedAlgorithmicMedia + 服务提供者编码 + 内容编号（绑定分镜任务 ID）；片头 drawtext 烧录 AI 生成显式提示。一次实现同时覆盖中国《标识办法》（2025-09-01 已施行）与 EU AI Act 第 50 条（2026-08-02 已生效）。

【必须排除的】
- Upscale-A-Video：NTU S-Lab License 1.0 非商业，法务红线，无论效果。
- VEnhancer：60GB 显存 + 6 秒片段 40–50 分钟，8 分钟片约 60 小时，工程上作废。
- STAR：39GB 只换 72 帧 240p，吞吐灾难。
- Real-ESRGAN / TSD-SR 进视频链路：无时序模块，必然闪烁，只能用于参考图预处理。
- FFmpeg minterpolate 作为主力插帧：CPU 块匹配，在非刚体运动上块状伪影明显，仅作无 GPU 兜底。

【三件需要你在排期前自己验证的事】
1. 本机 GPU 是什么。FlashVSR 官方只保证 A100/A800，H200 加速有限，其它卡「兼容性未知」，而它是整条链路的吞吐命脉。若只有 4090，必须先做 Block-Sparse Attention 编译验证，否则整个成本模型作废。
2. GIMM-VFI 的许可证（检索页面未明示 LICENSE），商用前必须核对。
3. x265 的商业授权。若把 FFmpeg 打进产品分发，x265 需向 MulticoreWare 购买；SVT-AV1 走 BSD 无此问题——这是 AV1 在商业产线里被低估的优势。

【对「单镜观感接近 Seedance 2.0」目标的直接判断】
后处理能修分辨率、闪烁、色彩漂移，修不了接缝处的主体不连续。你的架构里最脆的环节是第 4 步而不是第 5 步。建议把工程投入的优先级倒过来：接缝决策器 > 色彩一致 > 超分选型。超分是已解决问题（FlashVSR + SeedVR2 双档即可收工），接缝是没有现成方案的自研区——STCDiT 的 anchor-frame 思路（每个 clip 用首帧结构锚定）是目前最值得参考的方向。

### 存疑
- Practical-RIFE 检索到的最新版本仍是 v4.26（2024-09-21）与 4.25.lite（2024-10-20），未见 2025–2026 年新版本。这可能是 GitHub 页面抓取不完整，也可能项目确实已停更近两年——排期前请直接核对 repo 的 commit 历史与 releases。
- GIMM-VFI 主仓库（GSeanCDAT/GIMM-VFI）的许可证在所有检索到的页面中均未明示。ComfyUI 封装（kijai/ComfyUI-GIMM-VFI）有自己的 LICENSE，但不代表上游权重许可。商用前必须人工核对上游 LICENSE 文件。
- VEnhancer 仓库（Vchitect/VEnhancer）的许可证未在检索页面中明示，需人工核对。
- DaVinci Resolve 免费版是否支持「外部」Python 脚本 API（而非仅 Fusion Console 内脚本）未能证实。历史上外部脚本被认为是 Studio 独占，但近几个大版本存在相反说法。若你计划用免费版跑批量套色，必须先实测验证，不要按 Studio 文档假设。
- GIMM-VFI 与 RIFE 的速度对比（同一片段 RIFE ~9 秒 vs GIMM 25–53 秒）来自社区测评转述，非官方 benchmark，缺少 GPU 型号、分辨率、帧数等前提条件。量级（3–6 倍慢）可信，绝对数字不可信。
- SeedVR2 论文中 720p/100 帧 269.0–299.4 秒的测试 GPU 型号未在检索到的摘要中明确（推测为 H100，因 repo 显存口径以 H100-80G 为准）。所有基于此数字的小时数折算都带这一前提。
- C2PA Specifications 2.3 的确切发布日期、状态（是否为当前最新版）、以及 ISO 标准化进展未能从官方页面证实。规范版本号 2.3 本身已确认存在。
- 中国《标识办法》配套强制性国家标准的编号与技术细节（隐式标识的具体元数据字段名、显式标识的位置与面积量化要求）未能证实——openstd 国标公开平台与 tc260 页面均未返回有效内容。合规实现前必须取得标准原文，不能按《办法》正文的原则性表述直接落地。
- EU AI Act 第 50 条违规的具体罚款上限（常被引述为 1500 万欧元或全球年营业额 3%，取高者，依第 99 条）未从条文页面直接证实。条文仅指向第 99 条行政罚款框架。
- EU AI Act 第 50(4) 条对「艺术、讽刺、虚构作品」的较轻披露档位是否适用于你的虚构角色数字人内容，是法律判断而非技术判断。我的理解是该宽免不豁免第 50(2) 条针对 provider 的机器可读标记义务，但这需要法务确认。
- STCDiT（arXiv 2511.18786）是否已发布代码与权重、GPU 需求、与 SeedVR2/FlashVSR 的量化对照，均未证实。
- NTIRE 2026 短视频 UGC 修复挑战赛的获胜方案与所用基础模型未从摘要中获得。若要以此作选型依据，需读正文的排行榜表格。
- FlashVSR 的显存需求（VRAM 数值）在官方 repo 与项目页均未给出明确数字，只给了 GPU 型号建议。低显存路径依赖第三方 FlashVSR-Pro 的 tiling 实现，其质量影响未验证。
- Topaz Starlight 的 credit 消耗数字（1 分钟 1080p30 升 4K 约 34 credits）来自第三方评测站转述 Topaz 的估算，且该数字明确标注为「若干核心模型」而非 Starlight 本身，Starlight 消耗更高但具体倍数未知。
- 本次调研 WebSearch 配额（200 次）已在检索过程中耗尽，后续仅能用 WebFetch 定点抓取已知 URL。因此对「2026 年是否出现了更新的 VSR/VFI 模型」的覆盖可能不完整——尤其 2026 年 5 月之后的新发布，除 NTIRE 2026、STCDiT、FlashVSR(CVPR2026) 外可能有遗漏。

### 事实核查修正
- [WRONG] 论文实测：720p / 100 帧，SeedVR2-7B 单步 269.0–299.4 秒
  → 数字对但归属错了。SeedVR2 论文附录 B Table 4（720p/100帧）：Ours-3B = 3391.5 M 参数 / 269.0 s；Ours-7B = 8239.6 M 参数 / 299.4 s。即 269.0 s 是 3B 档，不是 7B 的下限。同表基线：SeedVR-7B 8239.6 M / 1284.8 s、STAR 2041.0 M / 2326.0 s、UAV 691.0 M / 1284.5 s、VEnhancer 2044.8 M / 2029.2 s、MGLD-VSR 1430.8 M / 1181.0 s。另注：表 https://arxiv.org/html/2506.05301v2
- [WRONG] ComfyUI 社区版 numz/RedbeardNZ SeedVR2_comfyUI
  → 仓库名不对。官方社区实现是 numz/ComfyUI-SeedVR2_VideoUpscaler（作者 NumZ + AInVFX/Adrien Toupet），Apache-2.0，非 ByteDance 官方维护。『RedbeardNZ SeedVR2_comfyUI』是镜像/派生写法，不是主仓库。 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- [WRONG] 3B-FP8 约 6GB 权重、实测仍需 ≥18GB 显存
  → 与官方 README 的显存分档矛盾。该 repo 明示：≤8GB 用 GGUF Q4_K_M + BlockSwap + VAE tiling；12–16GB 用 FP8（按需开 BlockSwap/tiling）；24GB+ 用 FP16 免优化。所以 FP8 的门槛是 12–16GB 而非 ≥18GB，低显存路径（GGUF 4bit）可下探到 8GB 以内。README 未给权重文件体积，『约 6GB』无来源。 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- [UNVERIFIABLE] 4090 上 768px→2K 单图 30–60 秒；社区口径 3B ≈20 秒/帧、7B ≈60 秒/帧
  → 该 repo 只给相对加速比（torch.compile 使 DiT 快 20–40%、VAE 快 15–25%，张量操作比 einops 快 2–5×），没有任何 秒/帧 或具体 GPU 的绝对 benchmark。这组数字查不到一手来源，排期不要直接用。 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- [WRONG] FlashVSR 同时开源了训练集 VSR-120K（120k 视频 + 180k 图像）
  → 数据集规模对，但尚未开源。官方 repo 与 HF 模型页都把 “Dataset release (VSR-120K) for large-scale training” 列在 TODO / coming soon 下。只有推理代码与权重已放出，训练集没有。 https://huggingface.co/JunhaoZhuang/FlashVSR
- [OUTDATED] FlashVSR 权重在 HuggingFace JunhaoZhuang/FlashVSR
  → 不完整。v1 在 JunhaoZhuang/FlashVSR（LQ_proj_in.ckpt、TCDecoder.ckpt、Wan2.1_VAE.pth、diffusion_pytorch_model_streaming_dmd.safetensors，2025-10）；v1.1 在独立仓库 JunhaoZhuang/FlashVSR-v1.1（2025-11）。想用 v1.1 必须换 repo，不是同一路径的更新。 https://huggingface.co/JunhaoZhuang/FlashVSR
- [UNVERIFIABLE] FlashVSR 是 CVPR 2026 论文
  → 仅有 GitHub README 的自述支持。arXiv 2510.12747（2025-10-14 提交，Junhao Zhuang 等，清华/CUHK）的 comments 字段没有任何会议接收标注。作为选型依据可以接受（作者自述），但不要当作已核实的同行评议状态引用。 https://arxiv.org/abs/2510.12747
- [WRONG] VEnhancer 硬性显存：官方要求单卡 ≥60GB VRAM，推荐 H100 / A100
  → 官方 README 原话是 “at least A100 80G is required”（单卡推理）。是 A100-80G 起步，不是『≥60GB』——60GB 这个数不出现在官方口径里，按 60GB 选卡会选错（如 A6000 48G / L40S 48G 都不够，而 60GB 档位在 NVIDIA 产品线里本就不存在）。 https://github.com/Vchitect/VEnhancer
- [UNVERIFIABLE] 实测：单卡 A100 默认参数增强一段 6 秒 CogVideoX 视频，占用 60GB 显存、耗时 40–50 分钟
  → 官方 repo 没有给出任何推理耗时数据。唯一可引的第三方量化口径是 SeedVR2 论文 Table 4：VEnhancer 2044.8 M 参数，720p/100 帧 2029.2 秒（约 34 分钟）——但那是 720p/100 帧，不是 6 秒 CogVideoX 片段，两者不可互换。 https://github.com/Vchitect/VEnhancer
- [WRONG] VEnhancer 2024-08 开源
  → 早一个月。官方 News：[2024.07.28] 推理代码与预训练模型发布；[2024.08.18] 支持任意长视频 + 15 步快速采样；[2024.09.10] 多卡推理与 tiled VAE；[2024.09.12] v2 checkpoint（venhancer_v2.pt）。arXiv 2407.07667 / 2024-07 正确。 https://github.com/Vchitect/VEnhancer
- [WRONG] VEnhancer 可任意倍率同时放大空间与时间分辨率
  → 不是任意倍率。官方给的空间放大范围是 up_scale 1~8，且明确推荐 ×3、×4；时间侧是设 target_fps（默认 24）而非任意倍数。『空间+时间统一模型』这一点成立，『任意倍率』夸大了。 https://github.com/Vchitect/VEnhancer
- [UNVERIFIABLE] Upscale-A-Video 速度：49 帧 720p 需要 510 秒推理（论文口径）
  → sczhou/Upscale-A-Video 的 README 没有任何推理速度或显存数字。可核实的只有 SeedVR2 论文 Table 4 的 691.0 M / 1284.5 s（720p、100 帧）。『49 帧 510 秒』找不到一手出处。（CVPR 2024 Highlight、NTU S-Lab License 1.0 非商业、--use_llava 支持均已核实无误。） https://github.com/sczhou/Upscale-A-Video
- [WRONG] Topaz 定价（2026）：Personal 订阅 $299/年；Pro $699/年（含 Starlight）；云端计划另起 $39/月含一定额度 credits
  → 产品线已重组为 Topaz Studio 套件，定价表对不上。当前官网：Topaz Studio（Personal）$399/年 或 $45/月年付 / $69/月；Topaz Studio Pro（商用）$799/年 或 $79/月年付。单品档：Topaz Video Personal $299/年 或 $39/月（$59/月 单月）——$299/年 对应的是单品 Topaz Video 而非套件 Personal。不存在 $699/年 档（Pro 是 $799/年）。$39/月是 Topaz Video 桌面版月付，不是云端计划；云端 Topaz f https://www.topazlabs.com/pricing
- [UNVERIFIABLE] credit 口径：1 分钟 1080p30 升 4K 用若干核心模型约消耗 34 credits
  → 官网定价页只给『Topaz Video: 25 credits』这类按任务的粗口径与 $0.10/credit 的加购价，没有『1 分钟 1080p30→4K ≈ 34 credits』这个换算。该数字来自第三方转述，不能用来做成本模型。 https://www.topazlabs.com/pricing
- [UNVERIFIABLE] 永久授权版 Video AI 已停售，存量用户不再获得新模型更新
  → 前半句成立：官网定价页已全部为订阅制，不提供永久授权。后半句（存量永久授权用户不再获得新模型）在定价页上没有任何表述，无一手来源，且在 Adobe 收购交割前后可能变化，不要写进结论。 https://www.topazlabs.com/pricing
- [UNVERIFIABLE] Personal 授权下部分 Starlight 模型强制走云端处理
  → 定价页区分的是『unlimited local rendering』+『video credits』两套额度，但没有说明哪些 Starlight 模型必须云端。这条是落地阻塞项（决定能否离线批处理），建议直接在 Topaz Video 试用版里实测确认，不要按转述排期。 https://www.topazlabs.com/pricing
- [WRONG] （存疑项）GIMM-VFI 主仓库 GSeanCDAT/GIMM-VFI 的许可证未明示，商用前必须人工核对
  → 已查明，可以结案：仓库根目录 LICENSE 是 S-Lab License 1.0，明确『Redistribution and use for non-commercial purpose』，商用需联系作者授权。即 GIMM-VFI（NeurIPS 2024，权重在 HF GSean/GIMM-VFI）与 Upscale-A-Video 同属 NTU S-Lab 非商业许可族，商用项目应直接排除或走授权谈判。kijai/ComfyUI-GIMM-VFI 的 LICENSE 只覆盖封装代码，不改变上游权重许可。 https://raw.githubusercontent.com/GSeanCDAT/GIMM-VFI/main/LICENSE

### 遗漏补充
- DLoRAL（yjsunnn/DLoRAL，NeurIPS 2025，不是 ICCV 2025）——One-Step Diffusion for Detail-Rich and Temporally Consistent VSR，MIT 许可，权重已放出（2025-10 有改进版 + 复现论文用的原版），带 Colab。这是本维度最该补的一条：一步扩散 VSR 里唯一 MIT 的，商用许可比 FlashVSR/SeedVR2 的 Apache 还省事，且没有 GIMM-VFI/UAV 的非商业陷阱。缺点是 repo 未给显存与速度数字。
- MGLD-VSR：SeedVR2 论文 Table 4 里的第四条基线（1430.8 M 参数，720p/100 帧 1181.0 s），是该表中除 SeedVR2 外最快的扩散 VSR。原调研五个方案里完全没有它，做横向对照表会缺一行。
- VEnhancer 的可引用量化档位：2044.8 M 参数 / 720p·100 帧 2029.2 s（SeedVR2 Table 4）。原调研只给了显存和一个无出处的『40–50 分钟』，没有可对齐的量化数字。
- STCDiT（arXiv 2511.18786，2025-11-24，Junyang Chen / Jinshan Pan 等，南京理工）：segment-wise reconstruction 处理复杂运镜 + anchor-frame guidance。已确认有 project page https://jychen9811.github.io/STCDiT_page，但 arXiv comments 无会议接收标注，代码/权重是否发布仍未证实。作为 2026 年潜在替代项列入观察即可，不能进选型短名单。
- FILM（google-research/frame-interpolation，ECCV 2022，Apache-2.0，L1/Style/VGG 三个预训练模型）：插帧维度被完全漏掉。在 Practical-RIFE 已停更（最新仍是 4.26 / 2024-09-21 与 4.25.lite / 2024-10-20，经核实确无 2025–2026 新模型）、GIMM-VFI 为非商业许可的前提下，FILM 是唯一『可商用 + 大位移鲁棒』的插帧兜底，必须进候选。
- Video2X 6.x（k4yt3x/video2x，AGPL-3.0，已用 C/C++ 重写）：一站式批量超分+插帧工程化方案，内置 Real-ESRGAN、Real-CUGAN、RIFE（全部模型）、Anime4K v4 与自定义 MPV GLSL shader，Vulkan 后端，Windows/Linux/AppImage/容器/Colab 全平台。原调研把 Real-ESRGAN 和 RIFE 当作两个孤立组件，漏了把它们串成生产流水线的这一层。注意 AGPL-3.0 对分发型产品有传染性，内部批处理使用无碍。
- 动漫/AIGC 专用超分路线：Real-CUGAN 与 Anime4K v4（均在 Video2X 内置）。原调研只提了 realesr-animevideov3，但对二次元/赛璐珞风格的 AI 生成片，Real-CUGAN 的线条保持通常优于 Real-ESRGAN anime 分支，值得做 A/B。
- 低显存落地的真实路径被写错方向：SeedVR2 在 ComfyUI 侧的可行下限是 GGUF Q4_K_M + BlockSwap + VAE tiling ≤8GB、FP8 12–16GB、FP16 24GB+。这条比『SeedVR2 要 H100』更决定项目能否在消费级卡上跑，原调研用了一个偏高且无来源的 ≥18GB。
- Topaz 已不是单品而是 Topaz Studio 套件（Video / Photo / Gigapixel / Image Web / Mobile / Astra / Bloom 七个 app），且引入了月度 credits 配额制（Personal 300、Pro 600、$0.10/credit 加购、按月清零）。做成本模型时要按 credits 而不是按『年费买断式使用』来算，原调研的定价框架整体过时。
- 去闪烁 / 时序一致性的专用后处理层（ProPainter、All-In-One Deflicker 一类）在本维度完全缺席。逐帧超分（Real-ESRGAN）之后必然出现的帧间闪烁，除了换成 VSR 模型外还有独立的 deflicker 后处理路线可选。注意：这一条我未做一手核实，仅作为需补调研的方向提出，其许可证（S-Lab 系的可能性较高）必须单独核对。


====================================================================================================
# [seedance-official] 字节 Seedance 2.0/2.5 官方 API 能力边界 + 闭源竞品同位对比（截至 2026-09-19 实检）

## Seedance 2.0（标准版）— 火山方舟 Ark 正式 API  (字节跳动 / 火山引擎)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API，权重不开放，不可本地部署，不可 LoRA | 成本: 国内 Ark：46 元/百万 token（无视频输入）、28 元/百万 token（有视频输入）。15 秒 720p 24fps ≈ 3088 万 token ≈ 8.65 元（含视频输入口径）。BytePlus 资源包：标准 $4.30/1M token、Fast $3.30/1M token
- 是什么: 多模态音视频联合生成模型，文/图/视频/音频四模态混合输入，原生出声（含对白），异步任务制 API。国内站 Ark 模型 ID `doubao-seedance-2-0-260128` 与 `doubao-seedance-2-0-fast-260128`；国际站 BytePlus ModelArk 前缀为 `dreamina-seedance-2-0-260128`。
- 关键事实:
    * 模型发布：2026-02（先上火山方舟体验中心），API 全面开放 2026-04-14（科技日报报道）
    * 单次生成时长：4–15 秒，或 duration=-1 交给模型自动判定
    * 分辨率：480p / 720p / 1080p，标准版最高 2K（2048×1080）；mini 档封顶 720p
    * 帧率：24fps
    * 参考位上限：12 个文件 = 9 图 + 3 视频 + 3 音频；图 ≤30MB/张，视频 2–15s 且 ≤50MB，音频 MP3 ≤15MB
    * 端点：POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks，Bearer API Key，轮询 GET .../tasks/{id}
    * content 数组 role 枚举：first_frame / last_frame / reference_image / reference_video / reference_audio；last_frame 必须配 first_frame
    * 首尾帧模式与多模态参考模式互斥（同时传会报参数错误）
    * generate_audio 布尔开关控制原生音轨；watermark 控制「AI 生成」水印
    * 产物 video_url 为火山对象存储直链，成功后 24 小时过期，必须立即转存
- 长片作用: 是 2–8 分钟长片的「单镜主力渲染机」。它只负责 4–15 秒的一个镜头，镜头之间的连贯必须靠你自己的分镜系统 + 参考图池（9 图槽位锁角色脸/服装/场景/画质）来保证。长片时长 100% 由拼接解决，模型本身不给你任何超过 15 秒的东西。
- 来源: https://apidog.com/blog/seedance-2-0-api/ https://www.wapi.cn/api_detail/ai_models_25.html https://www.datacamp.com/tutorial/seedance-2-0-api-guide

## Seedance 2.5 — 单段 30 秒 + 50 路参考（当前旗舰）  (字节跳动 Seed / 火山引擎)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商用 API，权重不开放 | 成本: 42–70 元/百万 token；30 秒 1080p 单镜粗估 ¥10–20 量级
- 是什么: Seedance 2.0 架构的放大版，把单次一镜到底从 15 秒拉到 30 秒，参考位从 12 个扩到 50 个，并加入时间戳级精确编辑、绿幕编辑、镜头视角编辑。Ark 模型 ID `doubao-seedance-2-5-260628`。
- 关键事实:
    * 模型正式发布：2026-07-31；火山方舟 API 全面开放：2026-08-07
    * 单次生成上限：30 秒（原生直出），官方表述支持 multi-round extensions（多轮续写）
    * 参考素材：单次最多约 50 路 = 最多 30 图 + 10 视频 + 10 音频（官方博客口径）
    * 分辨率：480p / 720p / 1080p（AIHubMix 模型页口径）
    * 计费：42 元/百万 token（有视频输入）、70 元/百万 token（无视频输入）
    * 折算约：无视频输入 ≈ $0.10/秒@480p、$0.23/秒@720p、$0.53/秒@1080p；有视频输入 ≈ $0.06 / $0.14 / $0.32
    * 支持 10+ 语种对白
    * 官方入口：即梦 Jimeng / 豆包 Pro / 火山方舟 Ark / BytePlus ModelArk
- 长片作用: 把「一个镜头」的物理上限从 15s 抬到 30s，对 2–8 分钟成片意味着分镜数量直接腰斩（8 分钟从 ~32 刀降到 ~16 刀），拼接接缝一半消失。30 图参考位足够把「主角正脸/侧脸/半身/全身/服装/道具/场景 A/B/C」全塞进去，是当前做角色一致性最强的闭源工位。这是你这套系统应当作为默认渲染档的模型，而不是 2.0。
- 来源: https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5 https://news.qq.com/rain/a/20260807A0A9H300 https://aihubmix.com/model/doubao-seedance-2-5-260628

## Seedance 官方「视频延长 / 续写」能力  (字节跳动 / 火山引擎)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: 按生成秒数计费，与正常生成同价；带视频输入走较低单价档（28 元/百万 token @2.0、42 @2.5）
- 是什么: 不是独立 endpoint，而是同一个 contents/generations/tasks 接口的一种任务类型：把已有视频（或之前的生成任务）作为 content 传入，用 @video N 指代并写延长指令，模型从末帧继续拍。官方提示词指南明确区分「编辑/延长任务」与「参考任务」。
- 关键事实:
    * Seedance 2.0：一次可延长 4–15 秒，从输入视频最后一帧开始，保留原音频，返回一条合并后的完整视频
    * 支持双向：既能生成「之后发生什么」，也能生成「之前发生什么」
    * Seedance 2.5：延长区间 4–30 秒，支持多轮续写
    * 提示词写法：编辑/延长任务直接写 `@video N`，不要写 `参考 @video N`，否则会被判成参考任务
    * BytePlus 官方 API 文档列出的 content 输入类型含「sample task ID」，即可用先前生成任务的 ID 作为续写锚点
    * 异步同款：返回 task_id 轮询，产物链接 24 小时有效
- 长片作用: 这是长片产线里唯一的官方「接缝修复工位」。策略：主镜头用 2.5 直出 30s，需要 45–60s 的连续长镜时用 extend 续 1–2 轮，而不是硬剪。但注意 extend 每轮都会有画质/一致性漂移，实践上不建议超过 2 轮，剩下的长度仍然靠分镜拼。
- 来源: https://docs.byteplus.com/en/docs/ModelArk/2222480 https://docs.volcengine.com/docs/82379/2222480 https://evolink.ai/docs/cn/api-manual/video-series/seedance2.5/seedance-2.5-video-extend

## Seedance 内容审核行为与返回码  (字节跳动 / 火山引擎 + BytePlus)
- 类别/成熟度: closed-api / **production** | 许可: 官方通道强制审核，无法关闭 | 成本: 失败不计费，但占用排队时间与并发额度
- 是什么: 输入前置过滤 + 输出后置过滤的双层审核，默认开启且不可在官方通道关闭。审核失败以任务 status=failed + 具名 error code 返回。
- 关键事实:
    * 输入真人脸直接拒：Dreamina Seedance 2.0 系列不接受含真实人脸的参考图/视频；即使是 AI 生成的写实人脸也会被判为真人而拒绝（实测报错「input image may contain real person」）
    * 常见错误名：OutputVideoSensitiveContentDetected（最高频）、OutputAudioSensitiveContentDetected、OutputVideoSensitiveContentDetected.PolicyViolation（版权相关）、InputTextSensitive
    * 输出审核发生在生成之后：模型已经算完，中间帧出现歧义画面也会被拦
    * 失败不计费（生成失败/审核拦截均不扣费）
    * BytePlus 侧默认开启 content pre-filter，明确阻止生成公众人物相似形象以防深伪
    * 可用规避路径（合规范围内）：卡通/风格化角色图通过率显著高于写实人像
- 长片作用: 决定你的角色美术方向。你的策略是「角色虚构、成年设定、不碰真人肖像」——正好和官方审核对齐，但必须注意：写实风格的虚构角色图仍可能被误判为真人而整条被拒。产线上必须做三件事：①参考图统一走可控的风格化/半写实风格并预检；②对 OutputVideoSensitiveContentDetected 做自动重试+换 seed+微调提示词的退避策略；③把审核失败率当作成本项纳入排期（它是长片产线最大的隐性延误源）。
- 来源: https://blog.segmind.com/seedance-2-0-error-guide-every-error-explained-with-fixes/ https://www.datacamp.com/tutorial/seedance-2-0-api-guide https://docs.byteplus.com/en/docs/modelark/2291680

## Seedance 2.0 mini / fast / pro 三档  (字节跳动 / 火山引擎)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: mini ≈0.2 元/秒@720P（促销价）、fast ≈0.6 元/秒@720P（促销价）
- 是什么: 同一能力的画质/速度/价格分档。Pro 最贵最好，标准版次之，Fast 走量，Mini 最便宜。
- 关键事实:
    * Mini：封顶 720P，约为标准版一半的每秒成本；不提供标准版的「通用参考」12 文件多模态控制
    * 标准版：原生 1080P，最高 2K（2048×1080）
    * Fast：`doubao-seedance-2-0-fast-260128`，速度优先
    * Mini 相对 Fast：速度约 2×、成本约 62%、运动质量更好、画质持平
    * 三档时长均为 4–15 秒、24fps
    * 2026-08-07～09-07 促销：2.0 mini 6 折 ≈ 0.2 元/秒@720P；2.0 fast 75 折 ≈ 0.6 元/秒@720P
    * 官方资源包活动明确「仅适用 Doubao-Seedance-2.0，不含 fast、mini 模型」
- 长片作用: 长片产线的成本杠杆。正确用法：分镜草稿/走位验证全部用 Mini（0.2 元/秒），定稿镜头才用 2.5 或 2.0 标准版渲染。按 8 分钟片 ≈ 480 秒成片、抽卡比 3:1 算，草稿档能把总成本压掉 ~40%。
- 来源: https://news.qq.com/rain/a/20260807A0A9H300 https://www.atlascloud.ai/blog/tips/seedance-2-0-mini-vs-seedance-2-0 https://www.volcengine.com/activity/seedance2

## Seedance 1.x 谱系（1.0 pro / lite / 1.5 pro）  (字节跳动 Seed)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API，权重不开放 | 成本: 显著低于 2.x，但官方未公开统一价表
- 是什么: 2.x 之前的两代。1.0 系列无原生音频，是纯视频模型；1.5 Pro 首次引入音视频联合生成（双分支 DiT）。
- 关键事实:
    * `doubao-seedance-1-0-pro-250528`（2025-05-28）：文生/图生视频，1080P，多镜头无缝切换，支持首尾帧
    * `doubao-seedance-1-0-lite-t2v-250428` / `doubao-seedance-1-0-lite-i2v-250428`（2025-04-28）：lite 档，i2v 支持首帧+尾帧+文本
    * `doubao-seedance-1-0-pro-fast`：提速版本
    * 时长：2–12 秒，默认 5 秒；分辨率 480p/720p/1080p
    * Seedance 1.5 Pro 论文：arXiv 2512.13507，2025-12-15，dual-branch Diffusion Transformer + 跨模态联合模块，音视频同步生成；参数量未公开；明确只在火山引擎可用，未开源
- 长片作用: 在 2.5 已上线的今天，1.x 只剩两个用途：①首尾帧插值（lite-i2v 的首尾帧模式仍是最省钱的「两张定帧之间补动画」工位）；②低成本批量空镜/转场素材。主叙事镜头不要用 1.x，因为它没有原生对白。
- 来源: https://www.emergentmind.com/papers/2512.13507 https://arxiv.org/abs/2512.13507 https://302.ai/product/detail/1856

## Seedance 官方微调 / LoRA 入口：不存在（已证实为否）  (火山引擎 火山方舟)
- 类别/成熟度: closed-api / **production** | 许可: 权重闭源，无精调入口 | 成本: N/A（无此产品）
- 是什么: 火山方舟确实提供「模型精调」产品（SFT / DPO / 强化学习），但覆盖范围是文本/多模态理解类的 Doubao 系列大模型。检索火山方舟精调概述、模型列表、Seedance 各版本文档与 BytePlus ModelArk fine-tuning 目录，均未出现任何 Seedance 视频生成模型可精调的条目。
- 关键事实:
    * 火山方舟精调支持的方式：有监督微调(SFT)、直接偏好优化(DPO)、强化学习；对象是 Doubao 文本/理解类模型
    * Seedance 全系（1.0/1.5/2.0/2.5）在官方文档中只有推理 API，没有训练/精调/自定义角色权重条目
    * BytePlus ModelArk 侧边栏有 Model fine-tuning 目录，但视频生成 API 与之并列、不交叉
    * 替代官方口径的「角色一致性」手段只有 reference_image 槽位（2.0 最多 9 张、2.5 最多 30 张），属于推理期条件注入，不是权重训练
    * 结论：不能在 Seedance 上做角色 LoRA，本机更不可能——权重从未发布
- 长片作用: 这条决定了你整个数字人平台的架构走向：角色身份不能烧进权重，只能每次调用都靠「参考图池 + 提示词模板」重建。因此你必须自建一个『角色资产库』（每个角色 20–30 张多角度/多服装/多光照的定妆图，经过审核预检），把它当成事实上的 LoRA。这是 2–8 分钟长片里角色不漂移的唯一可行路径。
- 来源: https://www.volcengine.com/docs/82379/1099459 https://docs.volcengine.com/docs/82379/1587798 https://docs.byteplus.com/en/docs/ModelArk/1520757

## 第三方「无审核 Seedance API」的真实性质  (各类中转站 / 聚合平台)
- 类别/成熟度: service / **vaporware** | 许可: 违反火山引擎服务条款；存在刑事风险 | 成本: 报价常见 $0.02–0.25/秒，明显低于官方成本价的即为假模型信号
- 是什么: 市面上以 `doubao-seedance-2-0` 模型名、OpenAI 兼容格式对外售卖的中转端点。实际供给来自三类：企业白名单账号转售额度、即梦/豆包网页端逆向、以及纯假模型（用便宜模型冒充）。没有任何一家能真正关闭字节侧的输出审核。
- 关键事实:
    * 典型中转：兔子中转、TopMix（需申请白名单）、nemovideo、Lovart.ai，均宣称支持 `doubao-seedance-2-0`
    * 审核位置在字节侧推理管线内（输入前置 + 输出后置），中转站处于字节下游，物理上无法绕过 OutputVideoSensitiveContentDetected
    * 风险 1：代充/盗刷卡账号，被封通常在 3–5 天内，额度不退
    * 风险 2：中转站被曝存在大比例「假模型」（用廉价模型冒充高价模型）、劫持、投毒案例
    * 风险 3：有中转站运营者自述因「通过非法技术手段获取 API」被立案（该自述真伪存疑，见 uncertainClaims）
    * 官方正规通道要求：Seedance 2.0 API 面向完成企业认证的企业用户开放
- 长片作用: 对你的产线只有一个结论：不要把任何一条中转链路放进生产。理由不是道德而是工程——①审核绕不过去，宣称「无审核」的必然是假模型或换了别的模型；②账号随时被封会让你的排期直接断档；③24 小时过期的产物链接 + 中转站转存 = 你的素材所有权不可控。企业认证走官方 Ark 是唯一可运营的选择。
- 来源: https://openairouter.net/blog/seedance-2-0-sd2-api-proxy-guide https://zhuanlan.zhihu.com/p/2032951488624977427 https://www.v2ex.com/t/1212688

## 火山方舟限流与并发（RPM / TPM / 并发数）  (火山引擎 火山方舟)
- 类别/成熟度: closed-api / **production** | 许可: N/A | 成本: TPM 保障包另行计费；免费额度 50 万 token/模型/30 天
- 是什么: Ark 对所有模型施加 RPM（每分钟请求）、TPM（每分钟 token）、并发数三重限流，超限返回 HTTP 429。具体数值随账号等级/开通状态动态分配，官方不公开统一表，需在控制台「开通管理」页查看当前额度。
- 关键事实:
    * 超限返回 429
    * 提供「TPM 保障包」作为付费提额产品（包天预付费 / 按小时后付费）
    * 新用户免费额度：每个模型 50 万 token，有效期 30 天，需实名认证
    * 已知具体数字：Seedance 2.0 mini 体验期内，控制台体验中心并发数限制为 1；体验期结束恢复默认限流
    * 视频任务为异步长任务（单条 15s 1080p 通常分钟级），实际吞吐由并发任务数而非 RPM 决定
- 长片作用: 这是长片产线的真实产能瓶颈。8 分钟成片按 30s/镜 = 16 镜，抽卡 3× = 48 次生成，单次数分钟；若并发只有个位数，一条片子的渲染墙钟时间是数小时。上线前必须：①向火山申请提高视频任务并发额度或买 TPM 保障包；②把调度器设计成「队列 + 并发上限可配 + 429 指数退避」；③不要用固定 sleep 轮询，用 Webhook 回调（Ark 支持视频任务 Webhook）。
- 来源: https://www.volcengine.com/docs/82379/1510762 https://www.volcengine.com/docs/82379/1159200 https://www.huasheng.ai/insights/volcengine-ark-api-guide/

## OmniHuman-1.5（即梦 / Dreamina 数字人 API）  (字节跳动 / 火山引擎 即梦AI 开放平台)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: 按音频秒数计费（per_second），官方未公开统一单价
- 是什么: 与 Seedance 并列的另一条官方产品线：一张人像图 + 一段驱动音频 → 口型对齐的数字人说话视频。走的是 volcengine.com/docs/85621（即梦AI）而非 Ark 82379。
- 关键事实:
    * 输入图片：jpg/jpeg/png/webp，单次仅 1 张，≤10MB
    * 输入音频：MP3 / WAV，最长 35 秒
    * 输出时长 = 音频时长，上限 35 秒；按音频秒数向上取整计费（per_second）
    * 产物链接有效期 24 小时
    * 可选文本提示语种：中/英/日/韩/西/印尼
    * OmniHuman-1.5 新增双人音频驱动（首次支持多人对话/辩论/共舞场景）
    * 论文口径：通过帧衔接策略可生成 1 分钟以上连贯视频，身份一致性误差 <3%（官方 API 单次仍限 35 秒）
- 长片作用: 如果你的「数字人拍片」里有大量正脸口播/主持/旁白段落，这个工位比 Seedance 便宜得多且口型更稳——Seedance 是「演戏」，OmniHuman 是「说话」。合理分工：叙事镜头/运动镜头走 Seedance 2.5，固定机位口播段走 OmniHuman 1.5（35s 一段，按台词切分），两条流在剪辑层合并。注意它同样吃真人肖像合规限制，且单次 35 秒同样需要拼接。
- 来源: https://www.volcengine.com/docs/85621/1810471 https://evolink.ai/docs/cn/api-manual/video-series/omnihuman/omnihuman-1.5-video-generate https://www.oschina.net/news/368897

## Google Veo 3.1 / 3.1 Fast / 3.1 Lite（Gemini API + Vertex AI）  (Google DeepMind)
- 类别/成熟度: closed-api / **production** | 许可: 闭源，preview 阶段模型 ID 可能变更 | 成本: 按秒计费，Gemini API 付费预览档；Lite 为最低成本档
- 是什么: 带原生音频的视频生成模型，Gemini API 与 Vertex AI 双通道。截至 2026-09 仍是 3.1 代，无 Veo 4。
- 关键事实:
    * 模型 ID：`veo-3.1-generate-preview` / `veo-3.1-fast-generate-preview` / `veo-3.1-lite-generate-preview`
    * 单段时长：4 / 6 / 8 秒；使用 1080p、4K 或参考图时仅支持 8 秒
    * 分辨率：3.1 与 Fast 支持 720p(默认)/1080p/4K；Lite 仅 720p/1080p
    * 帧率：24fps
    * 参考图上限：3 张（Ingredients-to-Video 在产品端为 4 张）
    * 首尾帧插值：仅 3.1 与 3.1 Fast 支持，Lite 不支持
    * 视频延长：仅 3.1 与 3.1 Fast，每次 +7 秒、最多 20 次，原+延总长上限 148 秒，延长仅 720p
    * 无官方微调/LoRA
- 长片作用: 在长片产线里 Veo 3.1 是唯一一个官方给出「148 秒连续可延长」上限的闭源 API——这是目前所有闭源模型里最长的单条连贯时长。但代价是延长段只能 720p，且基础单段只有 8 秒。它适合做「一镜到底的长转场/长运动镜头」这个特殊工位，或作为 Seedance 审核卡死时的备用渲染通道；不适合当主力，因为 3 张参考图撑不住多角色一致性。
- 来源: https://ai.google.dev/gemini-api/docs/veo https://ai.google.dev/gemini-api/docs/video https://developers.googleblog.com/introducing-veo-3-1-and-new-creative-capabilities-in-the-gemini-api/

## OpenAI Sora 2 / Sora 2 Pro —— API 于 2026-09-24 停止服务  (OpenAI)
- 类别/成熟度: closed-api / **vaporware** | 许可: 闭源，即将下线 | 成本: 即将归零
- 是什么: 曾经的旗舰视频 API，现已进入下线倒计时，且 OpenAI 未发布任何继任产品。
- 关键事实:
    * Sora App 已于 2026-04-26 关闭
    * Sora API 计划于 2026-09-24 停止接受新请求（距今 5 天）
    * OpenAI 未公布 Sora 3 或任何继任视频 API
    * Sora 2 时长档：4 / 8 / 12 秒；Sora 2 Pro：10 / 15 / 25 秒，单次上限 25 秒
    * 定价：Sora 2 $0.10/秒(720p)，Batch $0.05/秒；Pro $0.30(720p)/$0.50(1024p)/$0.70(1080p)，Batch 约半价
    * 无微调
- 长片作用: 对你的项目：直接排除，不要写进任何架构图。5 天后就是死链接。列在这里只为一件事——它的退场说明闭源视频 API 的政策风险是真实的，你的分镜系统必须做成「渲染后端可插拔」，Seedance 只是当前默认后端，而不是写死的依赖。
- 来源: https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation https://en.wikipedia.org/wiki/Sora_(text-to-video_model) https://openrouter.ai/openai/sora-2-pro

## 快手 Kling 3.0 / 3.0 Omni  (快手 Kuaishou)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: 720p 带视频输入约 12 credits/秒 ≈ $0.18/秒
- 是什么: 国产闭源视频模型，3.0 代加入原生音频（Omni Audio）与多镜头（Multi Shot）变体。
- 关键事实:
    * 单段时长：3–15 秒自由选择；原生上限约 10 秒
    * 延长能力：通过 extend 可累积到约 3 分钟——在闭源阵营里属于最长的可延长总长之一
    * Kling 3.0 Omni：多镜头变体，接受最多 7 张参考图
    * 计费：按秒 + 按分辨率；带视频输入时 12 credits/秒(720p) 或 16 credits/秒(1080p)；$1 = 66 credits（约 $0.01515/credit）
    * 无官方微调/LoRA 入口
- 长片作用: 是 Seedance 之外最实用的第二渲染后端：审核策略与字节不同，Seedance 被拦的镜头有相当概率能在 Kling 过。3 分钟的累积延长上限让它可以顶「长镜头/长口播」这个工位。但 7 张参考图不如 Seedance 2.5 的 30 张，多角色同框的一致性会弱一档。
- 来源: https://invideo.io/blog/kling-3-0-complete-guide/ https://kling.ai/blog/kling-video-3-0-credit-cost-guide https://piapi.ai/kling-3-0

## Runway Gen-4.5  (Runway)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: $0.12/秒（HDR、升格、专业格式另有加价）
- 是什么: 面向影视工业的闭源视频模型，主打单张参考图的跨场景角色一致性。
- 关键事实:
    * API 定价：$0.12 / 生成秒
    * 官方明确：单张参考图即可在不同环境复现同一角色，no fine-tuning required
    * 仅限短片段，长叙事需多次生成后剪辑
    * 订阅档：Standard $12/月 625 credits（≈52 秒 Gen-4.5）；Max 9500 credits/月（≈791 秒 Gen-4.5，未用额度可结转）
    * 无开放微调
- 长片作用: 性价比高但单段太短，在 2–8 分钟产线里只适合当「角色定妆/一致性校验」的工具位：用它快速验证某张角色参考图能不能稳定复现，再把通过验证的图喂给 Seedance 2.5 的参考池。不建议作为成片渲染主力——Max 档一个月的额度也就够 13 分钟成片，抽卡后剩不下多少。
- 来源: https://www.therundown.ai/tools/runway-gen-4-5 https://meetcody.ai/models/runway-gen-4-5/ https://www.eesel.ai/blog/runway-ai-pricing

## Luma Ray3.2  (Luma AI)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: 1080p 10 秒约 $3.60（SDR 基准，HDR 2×）
- 是什么: 面向 VFX/调色管线的视频模型，特点是原生 16-bit HDR 与 EXR(ACES2065-1) 导出、以及多关键帧控制。
- 关键事实:
    * Ray3.2 发布于 2026-06
    * 视频修改/输出上限约 20 秒 @1080p HDR
    * Multi-Keyframe：单条片段最多 16 个关键帧
    * 原生 HDR 16-bit 色深；EXR 导出 ACES2065-1 色彩空间
    * 计费：HDR 输出 2× SDR 价，HDR+EXR 3×；720p 5 秒约 $0.30，1080p 10 秒约 $3.60
    * 订阅：Plus $30/月、Pro $90/月、Ultra $300/月，月度 credits 不结转
    * 无开放微调
- 长片作用: 唯一一个把「16 个关键帧」暴露给你的闭源模型——这对分镜系统是有价值的：你可以用关键帧把一个 20 秒镜头的运动轨迹钉死，而不是靠提示词碰运气。如果你的成片要进专业调色（EXR/ACES），这是唯一选项。但 20 秒上限 + 高价 + 弱角色一致性，决定了它只能顶「需要精确运镜的特殊镜头 + 需要调色的片头片尾」这两个工位。
- 来源: https://lumalabs.ai/llm-info https://www.therundown.ai/tools/luma-ai https://www.eesel.ai/blog/luma-ai-pricing

## MiniMax Hailuo 2.3  (MiniMax 稀宇科技)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: 各托管平台按秒计价，普遍低于 Seedance/Kling 同档
- 是什么: 国产闭源视频模型，主打性价比与运动质量，有 standard 与 fast 两档，广泛上架 fal / Novita / Runware / Cloudflare Workers AI 等第三方托管。
- 关键事实:
    * 单段时长：6 秒 或 10 秒（固定两档）
    * 分辨率：768p / 1080p
    * 图生视频：first_frame_image 必填，prompt 可选
    * 多家第三方托管可直接调用（fal.ai、Novita、Runware、WaveSpeed、Cloudflare AI）
    * 无开放微调
- 长片作用: 在长片产线里是「便宜的填充镜头工位」：空镜、环境镜、物件特写这类不需要角色一致性的 B-roll，用它批量出比用 Seedance 便宜。它只有 first_frame 一个控制位，做不了多参考角色锁定，主叙事镜头不要用。
- 来源: https://novita.ai/docs/api-reference/model-apis-minimax-hailuo-2.3-i2v https://runware.ai/docs/models/minimax-hailuo-2-3 https://developers.cloudflare.com/ai/models/minimax/hailuo-2.3-fast/

## Vidu Q2 / Q2-Pro（生数科技）  (生数科技 Shengshu)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API | 成本: 第三方托管按次/按秒计价，属中低价位
- 是什么: 以「参考生视频」（Reference-to-Video）为核心卖点的国产闭源模型，多主体一致性是其主打能力。
- 关键事实:
    * Reference-to-Video：单次最多上传 7 张参考图作为视觉锚点
    * 单段时长上限 10 秒
    * 分辨率最高 1080p
    * 有 Q2 与 Q2-Pro 两档；ComfyUI 官方工作流、fal / WaveSpeed / Runware 均已上架
    * 无开放微调
- 长片作用: 7 图参考位是 Seedance 2.5（30 图）之外最接近「多主体一致性」的闭源方案，而且它比 Seedance 便宜、审核口径不同。在产线里的位置：当某个镜头需要「角色 + 道具 + 场景」三者同时锁定，而 Seedance 那一镜反复被审核拦时，Vidu Q2 是第一备胎。10 秒上限意味着它撑不起主叙事长镜。
- 来源: https://www.atlascloud.ai/models/vidu/q2-pro/reference-to-video https://fal.ai/models/fal-ai/vidu/q2/reference-to-video https://comfy.org/workflows/api_vidu_q2_r2v-c1956ef24421/

## PixVerse V5 系列（含官方 Extend 端点）  (爱诗科技 PixVerse)
- 类别/成熟度: closed-api / **production** | 许可: 闭源 API（MCP Server 开源） | 成本: 各托管平台中最低价档之一
- 是什么: 国产闭源视频模型，提供独立的 Extend（视频延长）端点与官方 MCP Server。
- 关键事实:
    * V5 时长：5 或 8 秒（1080p 同为 5/8 秒）
    * V5.5 / V5.6 增加 10 秒档；V6 与 C1 支持任意 1–15 秒
    * Extend 端点：分析现有视频结尾并续接 5–8 秒
    * 官方开放平台文档：docs.platform.pixverse.ai；官方 MCP Server 开源在 GitHub（PixVerseAI/PixVerse-MCP）
    * fal.ai / Together.ai 等均已托管
    * 无开放微调
- 长片作用: 价格档最低的一层，加上官方 MCP Server 让它最容易接进 Agent 化的分镜系统。在你的产线里适合做「大批量草稿预览」和「转场/贴片」。V6/C1 的 1–15 秒任意时长比固定档位更适合按分镜表精确出秒数。主叙事不推荐——角色一致性能力明显弱于 Seedance/Vidu。
- 来源: https://docs.platform.pixverse.ai/extend-1268531m0 https://github.com/PixVerseAI/PixVerse-MCP https://fal.ai/models/fal-ai/pixverse/extend/api

## Seedance 产物链接 24 小时过期（产线级坑）  (火山引擎 / BytePlus)
- 类别/成熟度: technique / **production** | 许可: N/A | 成本: N/A（但丢失即等于全额重烧）
- 是什么: 所有 Seedance / OmniHuman 任务成功后返回的 video_url 指向火山对象存储临时直链，成功后 24 小时失效。
- 关键事实:
    * video_url 在任务 succeeded 后 24 小时过期（Ark 与 BytePlus 双侧一致）
    * OmniHuman 1.5 同样 24 小时
    * 任务状态流：queued → running → succeeded / failed / expired / cancelled
    * 支持 Webhook 回调任务状态变更，避免长轮询
    * 推荐轮询退避：初始 10 秒，上限 60 秒
- 长片作用: 对 2–8 分钟长片这是硬性架构约束：一条 8 分钟片子要渲染几十次、跨几小时甚至几天，如果不在任务成功的瞬间自动转存到你自己的对象存储，隔夜素材全部作废、必须重烧钱重渲。你的分镜系统必须把「Webhook 回调 → 立即下载 → 落盘到自有 OSS → 写入素材库并记录 seed/prompt/参考图指纹」做成一个原子步骤。
- 来源: https://apidog.com/blog/seedance-2-0-api/ https://evolink.ai/docs/cn/api-manual/video-series/omnihuman/omnihuman-1.5-video-generate https://www.volcengine.com/docs/82379/1521309

### 建议
对你的「长时间 AI 数字人拍片系统平台」，实检结论如下。

**一、你的硬约束需要修正一条。** 9 图 + 3 视频 + 3 音频（共 12 个参考文件）是 **Seedance 2.0** 的规格，不是当前上限。Seedance 2.5 已于 2026-07-31 发布、2026-08-07 在火山方舟全面开放 API，模型 ID `doubao-seedance-2-5-260628`，**单段 30 秒、最多约 50 路参考（30 图 + 10 视频 + 10 音频）**，并支持多轮续写。其余三条（闭源、不可本机 LoRA、自带对白、官方强审核）全部核实无误。你的默认渲染档应当立刻从 2.0 切到 2.5——8 分钟成片的分镜数直接从 ~32 降到 ~16，接缝减半。

**二、角色一致性只能靠「参考图池」，这点已证实为终局。** 火山方舟的模型精调产品（SFT/DPO/RL）只覆盖 Doubao 文本类模型，Seedance 全系没有任何精调/自定义权重入口，权重也从未发布。所以你的平台里必须有一个一等公民模块：**角色资产库**——每个虚构角色维护 20–30 张多角度/多服装/多光照定妆图，全部通过审核预检并打指纹，每次生成时按镜头需要从池中选 5–10 张填进 reference_image 槽位。这就是你事实上的 LoRA，是长片角色不漂移的唯一可行路径。

**三、推荐的工位分配（多后端可插拔，不要写死 Seedance）：**
- 主叙事镜头（带对白、多角色同框）→ **Seedance 2.5**，30s/镜
- 需要超过 30s 的连续长镜 → Seedance 官方 extend（4–30s/轮，建议不超过 2 轮）
- 固定机位口播/旁白 → **OmniHuman 1.5**（即梦 API，图+音频 ≤35s，比 Seedance 便宜得多且口型更稳）
- 分镜草稿/走位验证 → **Seedance 2.0 mini**（促销价 ≈0.2 元/秒@720P），定稿才上 2.5，能砍掉约 40% 成本
- 空镜/B-roll → **Hailuo 2.3** 或 **PixVerse V6/C1**
- Seedance 审核反复拦截的镜头 → 备胎按序切 **Kling 3.0 Omni**（7 图参考、可累积延长至 ~3 分钟）→ **Vidu Q2**（7 图参考）
- 需要精确运镜的特殊镜头 / 要进 ACES 调色的片头片尾 → **Luma Ray3.2**（16 关键帧、EXR 导出）
- 需要 148 秒级连贯长镜（720p 可接受）→ **Veo 3.1**（+7s × 最多 20 次）

**四、三条必须写进架构的工程约束：**
1. **24 小时产物过期**：必须用 Webhook 回调 →（成功瞬间）下载 → 落自有 OSS → 连同 seed/prompt/参考图指纹入库，做成原子步骤。隔夜不转存 = 全额重烧。
2. **审核失败是最大隐性成本**：输出审核发生在生成之后，`OutputVideoSensitiveContentDetected` 是最高频错误；失败不计费但吃并发和墙钟时间。必须做「换 seed + 微调提示词 + 切备用后端」的三级自动退避。另外注意：**写实风格的虚构角色图仍可能被误判为真人而整条被拒**，美术风格应向可控的风格化/半写实收敛。
3. **并发是真瓶颈**：Ark 按 RPM/TPM/并发三重限流、超限 429，具体额度不公开、需在控制台查看并申请提额或购买 TPM 保障包。8 分钟片 ≈ 16 镜 × 3 次抽卡 = 48 次分钟级任务，并发若是个位数，单片渲染墙钟就是数小时。调度器必须队列化 + 并发上限可配 + 指数退避。

**五、明确不要做的事：** 不要接任何宣称「无审核 Seedance」的中转站。审核在字节侧推理管线内，中转站处于下游，物理上绕不过去；宣称能绕过的，要么是假模型冒充，要么是即梦网页逆向/盗刷号池——封号通常 3–5 天内发生，且存在刑事风险。官方 Ark 企业认证通道是唯一可运营路径。同理，Sora 2 API 将于 **2026-09-24（5 天后）** 永久停服且无继任产品——这正是你必须把渲染后端做成可插拔的理由。

### 存疑
- Seedance 2.0 Pro 的确切模型 ID `doubao-seedance-2-0-pro-260215`：仅出现在搜索引擎的 AI 摘要中，未在火山方舟官方文档或任何一手文档页核实到；Pro 档存在且支持 2K 有多方旁证，但 ID 字符串不可信。
- Seedance 2.0 按万 token 计价表（480p/720p 无视频参考 ¥0.782/万token、1080p ¥0.867/万token 等，来自 wapi.cn）与官方口径 46 元/百万 token（无视频）/ 28 元/百万 token（有视频）在数量级上不自洽，前者存疑。
- Seedance 2.5 是否原生 4K 输出：CometAPI 页面称 native 4K，但 AIHubMix 模型页与多数来源只列 480p/720p/1080p。官方文档未直接核实到，暂以 1080p 为准。
- Seedance 2.5 API 开放的确切日期：有 7 月 16 日全面开放、7 月 13 日大客户预先接入、8 月 7 日开发者 API 上线三种说法。8 月 7 日为多源一致的公开 API 日期，7 月的说法可能指大客户/预告阶段。
- wan27.org 的《Seedance 2.5 API 开发者指南》整篇具有 SEO 生成内容特征，其中「Ark 视频生成用 HMAC-SHA256 AK/SK 签名」（实际 Ark 视频 API 用 Bearer API Key）、「inference steps 25–50」、「参考可含 Blender/Maya 3D 白模」、「新账号 2 并发 / 10 RPM / 每日 50 次」等均无旁证，判为不可采信。
- Segmind 的 Seedance 2.0 错误指南中「duration 合法值为 4,5,6,8,10,12,15」「resolution 只能 480p 或 720p」：可能是 Segmind 自家代理层的限制，而非火山方舟原生限制（官方口径为 4–15 秒 + 480p/720p/1080p/2K）。错误码名称本身与其他来源一致，可信度较高。
- EvoLink 文档中的 `seedance-2.5-video-extend` 是该聚合平台自定义的模型名与参数名（video_urls / image_urls / audio_urls / quality / content_filter），不是火山方舟原生 API 的字段名；其反映的能力边界（4–30s 延长、30 图/10 视频/10 音频）与官方博客一致，但字段命名不可直接照抄。
- V2EX 上「中转站运营者因非法获取 API 被行政拘留 37 天」一案：原帖社区本身质疑真伪（中国行政拘留上限为 20 日），且有人指出原文转自 X 且来源不明。中转站的封号与合规风险有多方旁证，但这个具体案例不应作为事实引用。
- Luma「Ray3.14」：therundown.ai 有此条目，但 Luma 官方 llm-info 页只提到 Ray3.2。是否存在 3.14 版本未核实。
- Google I/O 2026 发布「Gemini Omni Flash」多模态视频模型：仅见于一篇第三方博客，Gemini API 官方文档与变更日志中未见该模型；Veo 4 确认尚未发布。
- OmniHuman-1.5 论文所称「帧衔接策略可生成 1 分钟以上连贯视频、身份一致性误差 <3%」：这是论文能力口径，官方 API 实际单次仍限 35 秒音频，二者不可混为一谈。
- 火山方舟对 Seedance 视频任务的具体并发数 / RPM / TPM 数值：官方文档只说明存在三重限流并返回 429，未公开数值表，仅核实到「Seedance 2.0 mini 体验期内控制台体验中心并发为 1」这一个具体数字。实际额度需登录控制台「开通管理」查看。
- Seedance 2.0 API 面向企业认证用户开放的准确时间点：一篇 CSDN 文章称「2024 年 4 月 2 日开放公测」，日期明显错误（模型 2026 年才发布），该来源整体不可信；科技日报的 2026-04-14 全面开放为较可靠日期。

### 事实核查修正
- [WRONG] Seedance 2.0 标准版分辨率 480p/720p/1080p，最高 2K（2048×1080）
  → 官方《模型列表》对 doubao-seedance-2-0-260128 明确列出：480p（8bit）/720p（8bit）/1080p（8bit）/4k（10bit）。根本没有「2K / 2048×1080」这一档。4k 档限流极严：企业与个人均为 最大 RPM 15、最大并发 1；非 4k 档为 企业 RPM 600/并发 10、个人 RPM 180/并发 3。价格上 4k 也单列（无视频输入 26 元/百万 token，含视频 16 元）。 https://www.volcengine.com/docs/82379/1330310
- [WRONG] Seedance 2.5 可能原生 4K 输出（CometAPI 口径），暂以 1080p 为准
  → 可以定论：doubao-seedance-2-5-260628 官方规格为 480p（8bit）/720p（8bit）/1080p（10bit），无 4k。反而是 Seedance 2.0 才有 4k（10bit）。CometAPI 的「native 4K」说法应判为错误。 https://www.volcengine.com/docs/82379/1330310
- [WRONG] Seedance 2.0 参考位上限：12 个文件 = 9 图 + 3 视频 + 3 音频
  → 数字自相矛盾（9+3+3=15）。官方《Doubao Seedance 2.5 教程》规格对比表写明：Seedance 2.0 系列「参考素材数量上限 15（9张图+3个视频+3个音频）」，Seedance 2.5 为「50（30张图+10个视频+10个音频）」。《Seedance 2.0 系列教程》同样写「图片：0~9 张；视频：0~3 个；音频：0~3 个」，并注明不支持「文本+音频」与「纯音频」输入（纯音频参考是 2.5 新增能力）。 https://www.volcengine.com/docs/82379/2607688
- [WRONG] 参考视频 2–15s 且 ≤50MB；音频 MP3 ≤15MB
  → 官方创建任务 API：单个参考视频 ≤200 MB（不是 50MB），时长 Seedance 2.0 系列 [2,15]s、最多 3 个且总时长 ≤15s；视频还支持 480p/720p/1080p/4k、帧率 [24,60]、格式 mp4/mov。音频格式为 wav 与 mp3 两种（不止 MP3），单个 ≤15 MB，请求体整体 ≤64 MB。单张图片 <30 MB 这条正确。 https://www.volcengine.com/docs/82379/1520757
- [UNVERIFIABLE] 首尾帧模式与多模态参考模式互斥（同时传会报参数错误）
  → 「互斥」部分属实：官方原文为「图生视频-首帧、图生视频-首尾帧、全模态参考生视频（包括参考图、视频、音频）为 3 种互斥场景，不可混用」。但紧接的一句是「全模态参考生视频优先使用图生视频-首尾帧」，属于优先级/降级语义，并非文档明示会「报参数错误」。该报错行为未在一手文档核实。 https://www.volcengine.com/docs/82379/1520757
- [UNVERIFIABLE] Seedance 2.0 模型发布 2026-02（先上火山方舟体验中心），API 全面开放 2026-04-14（科技日报报道）
  → 一手渠道核实不到这两个日期。官方 Model ID 为 doubao-seedance-2-0-260128，快照日指向 2026-01-28；方舟《模型发布公告》页中 doubao-seedance-2-0-260128 与 doubao-seedance-2-0-fast-260128 当前标记为「更新」而非「新发布」，月份分区（202601/202602/202604…）无法从页面结构稳定归属。建议降级为「约 2026 年初」，不要引用具体日。 https://www.volcengine.com/docs/82379/1159178
- [UNVERIFIABLE] Seedance 2.5 火山方舟 API 全面开放：2026-08-07
  → 官方口径只能确认两点：(1) 发布日 2026-07-31，当日「API 服务也将在近期上线火山方舟」，BytePlus 英文博客写 ModelArk API「coming soon」；(2) 当前 API 文档写「Seedance 2.5 已全面公开」。8-07 这个具体日期无一手来源。可确证的邻近官方日期是折扣活动起止：Seedance 2.5（1080p，72 折）2026-08-14 14:00 至 2026-09-17 14:00。 https://seed.bytedance.com/zh/blog/%E4%B8%80%E9%95%9C%E6%88%90%E7%89%87-%E9%9A%8F%E5%BF%83%E5%8F%82%E8%80%83-seedance-2-5-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83
- [OUTDATED] Seedance 2.5 计费：42 元/百万 token（有视频输入）、70 元/百万 token（无视频输入）
  → 只说对了一半档位。官方《模型价格》doubao-seedance-2.5「按输出视频分辨率和输入是否包含视频区分定价」：480p/720p → 无视频 70.00、含视频 42.00；1080p → 无视频 原价 77.00、含视频 原价 46.00（且 2026-08-14~09-17 1080p 限时 72 折）。引用时必须带分辨率档位。 https://www.volcengine.com/docs/82379/1544106
- [UNVERIFIABLE] 折算约：无视频输入 ≈ $0.10/秒@480p、$0.23/秒@720p、$0.53/秒@1080p；有视频输入 ≈ $0.06/$0.14/$0.32
  → 官方不按秒计价，且明确「视频价格 = token 单价 × token 用量」，同时对 Seedance 2.0 系列与 2.5「当输入包含视频时存在最低 token 用量限制」（低于则按最低量计），最低量随分辨率/宽高比/输出时长变化。因此任何固定的 $/秒 折算都只是估算，官方提供的是《Seedance 2.0 系列价格计算器》《Seedance 2.5 系列价格计算器》与最低 token 用量表。 https://www.volcengine.com/docs/82379/1544106
- [WRONG] Seedance 2.5 支持 10+ 语种对白
  → 官方文档说的是「提示词语言支持」，不是对白语种。原文：所有模型支持中英文提示词；Seedance 2.5 额外支持西班牙语、印度尼西亚语、葡萄牙语、日语、马来语、泰语、阿拉伯语、越南语、韩语（合计 11 种）；Seedance 2.0 系列额外支持西班牙语、印尼语、葡萄牙语、日语（合计 6 种）。把它写成「对白语种」是换概念。 https://www.volcengine.com/docs/82379/1520757
- [UNVERIFIABLE] Seedance 2.0 视频延长：一次可延长 4–15 秒，从输入视频最后一帧开始，保留原音频，返回一条合并后的完整视频
  → 可确认的只有：方舟模型列表中 2.0 / 2.0 fast / 2.0 mini 均标注「延长视频」能力，且 2.0 系列 duration 取值 [4,15] 或 -1。「从最后一帧开始」「保留原音频」「返回合并后的完整视频」三点均无一手来源；而且官方对 2.5 明确写「对原视频进行向前或向后延长」，说明「从最后一帧开始」并非通则。 https://www.volcengine.com/docs/82379/1330310
- [UNVERIFIABLE] 提示词写法：编辑/延长任务直接写 `@video N`，不要写 `参考 @video N`，否则会被判成参考任务
  → 官方没有这条「不要写参考」的表述。官方给出的是显式参数 omni_reference_task_type（auto/reference/edit/extend）来前置声明子任务类型，并给出示例提示词：延长「向后延长 @video1，@image1 的角色从天而降…」「续写 @video1 前 5s…」；编辑「@video1 中加一些小动物」「把 @video1 的人物修改为 @image1」「删掉 @video1 的背景音乐」。误判时的官方错误码是 InvalidParameter.TaskTypeConstraint（同步）与 InvalidParam https://www.volcengine.com/docs/82379/2607688
- [WRONG] BytePlus 官方 API 文档列出的 content 输入类型含「sample task ID」，即可用先前生成任务的 ID 作为续写锚点
  → 概念误解。该输入类型对应火山方舟的 content.type = draft_task（「样片任务 ID」）。它属于「样片模式」：先用 draft=true 以 480p 生成低成本 Draft 预览视频，确认后再用 Draft 的 task id 生成正式视频，平台自动复用 model/text/image_url/generate_audio/seed/ratio/duration/camera_fixed。且该能力官方标注「模型支持：Seedance 1.5 pro」，与续写/延长锚点无关（延长靠 role=reference_video + om https://www.volcengine.com/docs/82379/1520757
- [WRONG] 输入真人脸直接拒：Seedance 2.0 系列不接受含真实人脸的参考图/视频
  → 对火山方舟而言过度概括。真人脸拦截确实存在（错误码 InputImageSensitiveContentDetected.PrivacyInformation / InputVideoSensitiveContentDetected.PrivacyInformation，报错文案正是「The request failed because the input image/video may contain real person」），但官方同时给出三条合规通路：①本账号近 30 天内由 Seedance 2.5 / 2.0 系列生成的含人脸原始产物可直接作 https://www.volcengine.com/docs/82379/1299023
- [WRONG] 常见错误名含 InputTextSensitive
  → 错误码全名为 InputTextSensitiveContentDetected（另有 .PolicyViolation 变体）。同组另三个写法正确：OutputVideoSensitiveContentDetected、OutputAudioSensitiveContentDetected、OutputVideoSensitiveContentDetected.PolicyViolation。官方错误码表还包含 InputImage/InputVideo/InputAudio 各自的 SensitiveContentDetected 与 .Policy https://www.volcengine.com/docs/82379/1299023
- [UNVERIFIABLE] 输出审核发生在生成之后：模型已经算完，中间帧出现歧义画面也会被拦
  → 「存在生成后输出审核」有旁证（Output*SensitiveContentDetected 系列错误码确实存在，且计费文档写「因审核等原因导致生成失败的，不收取费用」——失败不计费这条 CONFIRMED）。但「中间帧歧义画面也会被拦」这一机制细节无官方表述。另注意 Seedance 2.5 特定任务类型「需排队等待到任务被消费时才会返回报错信息」。 https://www.volcengine.com/docs/82379/1544106
- [UNVERIFIABLE] BytePlus 侧默认开启 content pre-filter，明确阻止生成公众人物相似形象以防深伪
  → 未在 BytePlus/火山一手文档核实到这段表述。可核实的最接近条目是火山方舟错误码表中的 OutputImageSensitiveContentDetected.DeepFake，以及 ContentSecurityDetection（CSD）审核链路会在报错中返回 CSDRequestId/Label/SubLabel。 https://www.volcengine.com/docs/82379/1299023
- [WRONG] Mini：不提供标准版的「通用参考」12 文件多模态控制
  → 官方《Seedance 2.0 系列教程》原文：「三者支持的功能基本一致，主要区别在于生成品质与成本的取舍」。模型列表中 doubao-seedance-2-0-mini-260615 同样标注全模态参考生视频 / 参考生视频 / 编辑视频 / 延长视频 / 首尾帧生视频 / 首帧生视频 / 文生视频，参考素材上限同为 15（9图+3视频+3音频）。Mini 的真实差异只在分辨率（仅 480p/720p）与价格。「封顶 720p」「约为标准版一半成本」（23 vs 46 元/百万 token）两条正确。 https://www.volcengine.com/docs/82379/2291680
- [WRONG] 2026-08-07～09-07 促销：2.0 mini 6 折 ≈ 0.2 元/秒@720P；2.0 fast 75 折 ≈ 0.6 元/秒@720P
  → 折扣力度与活动期都错。官方：Seedance 2.0 mini 与 2.0 fast 的活动期均为北京时间 2026-08-07 14:00 至 2026-10-07 14:00（不是 09-07）；mini 为 480p/720p 按刊例价 4 折（不是 6 折），fast 为 75 折（这条对）。两者优惠仅面向企业用户，且各自有折后 token 用量上限（系统自 2026-09-18 14:00 起累计），达上限后恢复原价。刊例价：mini 无视频 23.00 / 含视频 14.00；fast 无视频 37.00 / 含视频 22.00（元/百万 t https://www.volcengine.com/docs/82379/2630943
- [WRONG] 官方资源包活动明确「仅适用 Doubao-Seedance-2.0，不含 fast、mini 模型」
  → 官方《Seedance 2.0-2.5 模型资源包使用规则》列出四种资源包：Seedance 2.5、Seedance 2.0、Seedance 2.0 Fast、Seedance 2.0 Mini，且四者任一有余量均可开通模型，资源包「仅支持抵扣对应的模型」。真实限制是另一回事：资源包属预付费商品，不参与后付费限时折扣活动；有效期 90 天；7 天内未使用可全额退款；抵扣以单价较低的「含视频输入」场景为 1:1 基准折算。 https://www.volcengine.com/docs/82379/2191775
- [WRONG] Seedance 2.0 Pro 档存在且支持 2K，仅模型 ID 字符串 doubao-seedance-2-0-pro-260215 不可信
  → Pro 档本身就不存在。官方教程原文：「Seedance 2.0 系列模型目前包括 Doubao Seedance 2.0、Doubao Seedance 2.0 Fast 和 Doubao Seedance 2.0 Mini」，模型列表中也只有 doubao-seedance-2-0-260128 / -fast-260128 / -mini-260615 三个。搜索引擎 AI 摘要里的 260215 几乎可以确定是把文本大模型 doubao-seed-2-0-pro-260215（豆包 Seed 2.0 Pro，深度思考模型）误当成了 seedanc https://www.volcengine.com/docs/82379/2291680
- [WRONG] 火山方舟对 Seedance 视频任务的并发 / RPM 未公开数值表，只能登录控制台查看
  → 官方《模型列表》公开了具体数值。doubao-seedance-2-5-260628 与 doubao-seedance-2-0-mini-260615、-fast-260128：最大 RPM 企业 600 / 个人 180，最大并发 企业 10 / 个人 3。doubao-seedance-2-0-260128：非 4k 同上；4k 分辨率单独限流 RPM 15、并发 1（企业与个人相同）。doubao-seedance-1-0-pro-250528 / -pro-fast-251015 / -1-5-pro-251215：最大 TPD 5000 亿、 https://www.volcengine.com/docs/82379/1330310
- [OUTDATED] doubao-seedance-1-0-lite-t2v-250428 / doubao-seedance-1-0-lite-i2v-250428 为现行 lite 档
  → 现行火山方舟《模型列表》中已检索不到任何 seedance-1-0-lite 条目。当前在列的 Seedance 1.x 只有 doubao-seedance-1-0-pro-250528、doubao-seedance-1-0-pro-fast-251015、doubao-seedance-1-5-pro-251215（且已标「即将下线」）。写成现行可用模型会误导。 https://www.volcengine.com/docs/82379/1330310
- [OUTDATED] doubao-seedance-1-0-pro-fast：提速版本
  → 完整 Model ID 为 doubao-seedance-1-0-pro-fast-251015。规格：480p/720p/1080p，24fps，时长 [2,12]s，支持首帧生视频与文生视频（不支持首尾帧），独有 frames / seed / camera_fixed 参数。 https://www.volcengine.com/docs/82379/1330310

### 遗漏补充
- omni_reference_task_type（auto / reference / edit / extend）：Seedance 2.5 专有的子任务类型前置声明参数，决定 ratio 与 duration 的硬约束（edit 必须 ratio=adaptive、duration=-1、参考视频 4–30s；extend 必须 ratio=adaptive）。配套两套报错机制：同步报错 InvalidParameter.TaskTypeConstraint（提交即拒）与异步报错 InvalidParameter.TaskTypeMismatch（排队后才拒）。这是 2.5 接入的头号踩坑点，调研完全没覆盖。
- output_format = mp4 / mov（Seedance 2.5 专有）：mov 为 H.264 + yuv444p 色度采样 + PCM 音频，官方明确推荐在视频编辑与视频延长场景作为输入与输出格式以保持色彩与声画一致性；代价是部分播放器不兼容。
- tools 参数中的 web_search 联网搜索工具：Seedance 2.5 与 Seedance 2.0 系列均支持，模型自主判断是否联网（商品、天气等时效内容），实际搜索次数在查询任务 API 的 usage.tool_usage.web_search 返回。这是一个视频生成模型罕见的能力，调研未提及。
- priority 执行优先级 [0,9]：同一 Endpoint 队列内插队，默认 FIFO，不打断 running 任务。配合 callback_url 回调（状态 queued/running/succeeded/failed/expired，失败会重试 3 次）与 execution_expires_after 超时设置，构成生产级排队调度方案。
- service_tier = flex 离线推理模式：价格为在线推理的 50%、TPD 配额更高——但官方明确「Seedance 2.5、Seedance 2.0 系列暂不支持」。做成本方案时这是必须先排除的选项。
- 素材 & 虚拟人像库 + 素材 ID 输入：image_url / video_url / audio_url 除公网 URL 与 Base64 外，可直接填「素材 ID」，取自官方预置素材库、私有虚拟人像库、私有真人人像库。这是真人形象合规落地的官方正路，调研只写了「真人脸被拒」的负面结论。
- 版权 IP 视频生成：官方支持基于特定授权 IP（如周星驰电影《功夫女足》版权）生成视频，费用 = 视频生成费用 × 1.5，当前仅限体验中心，对应文档「生成版权 IP 视频 / 使用视频生成 API 创作生成 IP 视频 / 创作真人 IP 视频」。
- draft 样片模式（draft=true）+ content.type=draft_task 两步式工作流：先出 480p 低成本 Draft 校验镜头调度与 Prompt 意图，再凭 Draft task id 生成正式视频；token 折算系数无声 0.7 / 有声 0.6。官方标注仅 Seedance 1.5 pro 支持——是成本控制的重要手段，也是「sample task ID」这条声明的真正出处。
- frames 参数（取值 [29,289] 且必须满足 25+4n）：用于生成小数秒视频，优先级高于 duration，仅 Seedance 1.0 pro / 1.0 pro fast 支持。同组还有 seed [-1, 2147483647] 与 camera_fixed（参考图场景不支持），均只在 1.x 档可用——2.0/2.5 没有 seed，复现性方案要据此调整。
- Seedance 1.5 pro（doubao-seedance-1-5-pro-251215，已标『即将下线』）：时长 [4,12]s，480p/720p/1080p，是唯一支持 draft 样片模式与 seed 的中间代，价格档独立。做长期方案时要把它排除在外。
- 模型开通门槛：调用 Seedance 2.5 / 2.0 系列前必须满足任一条件——账户余额 > 200 元、或购买 200 元档位及以上专属节省计划、或持有对应模型资源包且有余量。未满足则根本无法开通，这是接入第一道卡点。
- 产物留存与转存：video_url 有效期 24 小时，且 Seedance 2.5 生成的视频 URL 下载次数上限 100 次；任务记录仅支持查询最近 7 天 [T-7d, T)。官方推荐用火山引擎 TOS 数据订阅把推理产物自动转存到自有 TOS 桶，比手写轮询下载更稳。
- 最低 token 用量兜底规则：Seedance 2.0 系列与 2.5 在『输入包含视频』时存在最低 token 用量，估算低于该值按最低值计费，具体随分辨率/宽高比/输出时长变化。官方提供 Seedance 2.0 / 2.5 价格计算器与最低 token 用量表——任何按秒估价的模型都必须叠加这一条。
- safety_identifier（终端用户唯一标识，≤64 字符，建议对用户名/ID/邮箱做哈希）：面向 C 端产品接入时用于平台侧识别违规用户，属于合规必填项之一。
- 隐形水印：除 watermark=true 的可见『AI 生成』右下角水印外，火山另有独立文档《为 AI 生成产物添加隐式水印》，是符合国内 AIGC 标识要求的另一条路径。
- Seedance 2.5 新增纯音频参考生成视频（无需搭配图片或视频），而 Seedance 2.0 系列明确不支持『文本+音频』与『纯音频』输入——这是 2.0 与 2.5 之间一条硬能力分界。
- 参考视频输入本身的完整约束：支持 480p/720p/1080p/4k、帧率 [24,60]、宽高比 [0.4,2.5]、边长 300–6000 px、总像素 [407696, 8295044]、≤200MB；Seedance 2.5 非编辑任务单视频 [2,30]s、编辑任务 [4,30]s、10 个总计 ≤30s。做素材预处理管线必须按这套数值裁剪。


====================================================================================================
# [threed-mocap] 3D / 动捕底稿 + I2V 贴皮产线（角色资产、单目动捕、Blender/UE/VTuber 集成、视频侧控制、动作迁移）

## GVHMR (World-Grounded Human Motion Recovery via Gravity-View Coordinates)  (浙江大学 ZJU3DV)
- 类别/成熟度: open-weights / **usable** | 许可: 仓库含 LICENSE 文件，但 README 未标明具体类型（需人工确认，SMPL 模型本身另需注册） | 成本: 推理显存未公布；训练 2×4090。单机可跑，无 API 费用
- 是什么: 单目视频 → 世界坐标系下的 SMPL 人体运动恢复，输出带重力对齐的全局轨迹而非仅相机系姿态。
- 关键事实:
    * SIGGRAPH Asia 2024 论文，README 标注已被 TPAMI 2026 收录
    * 2025-03-08 更新：用自研 SimpleVO 替换 DPVO，效率更高且与 GVHMR 兼容
    * 新增 f_mm 参数可直接指定毫米焦距，便于与 Blender 相机对齐
    * 发布 checkpoint 在 2×RTX4090 上训练 420 epoch
    * 提供 Google Colab、HuggingFace Space 和命令行批处理三种入口
    * 在 3DPW / RICH / EMDB 三个基准上评测
- 长片作用: 顶「动作底稿采集」工位：把参考表演视频（自拍/素材）转成带全局位移的 SMPL 动作，直接导进 Blender 驱动角色走位。世界坐标输出是关键——2-8 分钟叙事片需要角色在场景里真实位移，相机系 mocap 会导致角色「原地踏步」，剪不进有空间关系的分镜。
- 来源: https://github.com/zju3dv/GVHMR

## WHAM (Reconstructing World-grounded Humans with Accurate 3D Motion)  (CMU / Meta)
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 官方未标 VRAM；单卡消费级可跑，无授权费
- 是什么: 视频 → 世界坐标 SMPL 姿态与体型估计，含 Temporal SMPLify 可选精修。
- 关键事实:
    * CVPR 2024，arXiv 2312.07531（2023-12 提交）
    * MIT license
    * 依赖 ViTPose 做 2D 关键点，相机运动依赖外部 SLAM（DPVO 或 DROID-SLAM）
    * 无相机标定时可退化为 local-only 模式
    * README 的 TODO 仍列有数据预处理与训练实现待完善项
- 长片作用: 与 GVHMR 同工位的备选/交叉验证方案。MIT 许可比 GVHMR 明确，商业产线优先选它。缺点是外挂 SLAM 链路长（ViTPose + DPVO），批量跑几十个镜头时工程复杂度高于 GVHMR。
- 来源: https://github.com/yohanshin/WHAM https://arxiv.org/abs/2312.07531

## NLF (Neural Localizer Fields)  (Istvan Sarandi / MPI)
- 类别/成熟度: open-weights / **usable** | 许可: 代码 MIT，但预训练权重仅限非商业研究用途 —— 商业产线的实质阻断点 | 成本: 未公布 VRAM 与速度
- 是什么: 图像 → 连续 3D 人体姿态与形状，单模型覆盖多种关键点/网格定义。
- 关键事实:
    * NeurIPS 2024，arXiv 2407.07532
    * 代码仓库标注 MIT license
    * 权重明确限定 noncommercial research use（PyTorch 与 TensorFlow 双版本在 Releases 下发）
    * 提供 demo.ipynb 使用示例，训练代码双框架均开放
- 长片作用: 精度上是这一代单帧人体估计的强基线，但权重非商用条款让它在「长时间 AI 数字人拍片系统平台」这种要卖的产品里不能直接用。只能作为内部质量标尺（比对 GVHMR/WHAM 输出），不能进产线。
- 来源: https://github.com/isarandi/nlf https://arxiv.org/abs/2407.07532

## SMPLest-X  (SMPLCap（南洋理工 S-Lab 等）)
- 类别/成熟度: open-weights / **usable** | 许可: 仓库含 LICENSE.txt，具体条款未在 README 说明（需人工确认） | 成本: 8.2GB 权重，单卡推理；24GB 卡可跑
- 是什么: 表达式全身（SMPL-X，含手部与表情）姿态形状估计，SMPLer-X 的重构升级版，强调易安装与可扩展。
- 关键事实:
    * 论文 2025-01-20 发布；新代码库 2025-02-14；预训练模型 2025-02-17
    * SMPLest-X-Huge 权重体积 8.2GB
    * 已被 TPAMI 接收（2025）
    * 测试默认 NUM_GPU=1；训练示例用 16 GPU
- 长片作用: 顶「全身+手+脸一体化底稿」工位。叙事片里角色要拿道具、打手势、说台词，SMPL-X 的手和脸参数是 GVHMR/WHAM（纯 SMPL 身体）拿不到的。代价是无全局世界坐标轨迹，需与 GVHMR 组合：GVHMR 出位移，SMPLest-X 出手部/表情。
- 来源: https://github.com/SMPLCap/SMPLest-X

## 4D-Humans / HMR 2.0  (UC Berkeley (Goel, Pavlakos, Kanazawa, Malik))
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 推理显存未标，batch_size 可调；MIT 无费用
- 是什么: Transformer 做单帧人体重建 + 跨帧 tracking，输出 tracklet 级 3D 姿态与形状。
- 关键事实:
    * ICCV 2023，「Humans in 4D」
    * MIT license
    * 两个 checkpoint：HMR2.0b（默认）与 HMR2.0a
    * 输出：渲染网格图、可选 .obj 网格、含 3D pose/shape 的 .pkl tracklet
    * 训练用 8×A100 跑 7 天
- 长片作用: 最成熟稳定的多人 tracking 底座（.pkl tracklet 天然按人分轨）。但它是 2023 年的东西，精度已被 GVHMR/SMPLest-X 超过，且无世界坐标。在长片产线里只顶「多人镜头的人物分轨与初始化」这一个窄工位，不建议当主力。
- 来源: https://github.com/shubham-goel/4D-Humans

## HY-Motion-1.0  (腾讯混元)
- 类别/成熟度: open-weights / **usable** | 许可: 仓库含 License.txt，条款未在 README 明示（腾讯系通常带 MAU 与地域限制，需核实） | 成本: 24–26GB VRAM，单张 4090/A6000 可跑；开源无 API 费
- 是什么: 文本 → 3D 人体骨骼动作生成（DiT 架构），号称首个 billion 级文生动作模型。
- 关键事实:
    * 2025-12-30 发布
    * HY-Motion-1.0 = 1.0B 参数，HY-Motion-1.0-Lite = 0.46B
    * 显存：1.0B 需 26GB，Lite 需 24GB
    * 文档建议动作长度 < 5 秒以降低显存
    * 三阶段训练，预训练用 3000 小时动作数据
    * SMPL 骨骼格式，提及 FBX-SDK 兼容
    * GitHub 2558 stars，2026-07-18 仍在更新
- 长片作用: 顶「无参考素材时的动作底稿」工位。当分镜里需要一个你没法自己表演也找不到素材的动作（挥剑、跌倒、特定步态），用文本生成 5 秒动作片段接进 Blender，比找素材快。但 <5 秒的建议长度意味着 2-8 分钟片子里它只能做「动作砖块」，不能做整段表演——长表演仍需 GVHMR 从真人参考视频抽。
- 来源: https://github.com/Tencent-Hunyuan/HY-Motion-1.0

## Blender 5.2 LTS  (Blender Foundation)
- 类别/成熟度: framework / **production** | 许可: GPL | 成本: 免费；EEVEE Next 灰模渲染 1080p 基本是秒级/帧，一台带 GPU 的工作站足够
- 是什么: 开源 DCC，本产线里承担绑定、重定向、灰模渲染与 depth/normal AOV 导出。
- 关键事实:
    * Blender 5.2 LTS 发布于 2026-07-14（当前稳定版）
    * Blender 5.1：2026-03-17；Blender 5.0：2025-11-18
    * 4.x 线：4.5 LTS 2025-07-15、4.4 2025-03-18、4.3 2024-11-19、4.2 LTS 2024-07-16
    * GPL，完全免费商用
- 长片作用: 注意调研任务里写的「Blender 4.x」已经落后两个大版本：2026-09 应该按 5.2 LTS 建产线。选 LTS（5.2 或 4.5）而非 5.1，因为插件生态（Auto-Rig Pro、各类 mocap 导入器）在大版本跳跃后常滞后 1-3 个月。这是长片产线的稳定性底座，不要追新。
- 来源: https://www.blender.org/download/releases/

## BlenderMCP (ahujasid/blender-mcp)  (Siddharth Ahuja（第三方，非 Blender 官方）)
- 类别/成熟度: framework / **usable** | 许可: MIT | 成本: 免费；但生 3D 走 Rodin 时按 Rodin 计费
- 是什么: MCP 服务器，让 LLM 通过 socket 双向操控 Blender：建场景、改材质、查场景、执行 Python。
- 关键事实:
    * 29.0k stars / 2.7k forks，212 commits，持续维护
    * MIT license
    * 支持 Blender 3.0 及以上
    * 内置资产源：Poly Haven（模型/HDRI）、Sketchfab、Poly Pizza 低模
    * 内置 AI 生 3D：Hyper3D Rodin 与 Hunyuan3D
    * 支持 GLB / FBX 导出，含节点 schema 与 API 参考查询
- 长片作用: 顶「分镜 → 3D 场景搭建」的自动化工位。2-8 分钟片子有几十个镜头，手搭场景是最大人力黑洞；BlenderMCP 让 agent 按分镜脚本批量摆位、打光、设相机、导 GLB。这是把 3D 底稿路线从「手工奢侈品」变成「可规模化产线」的关键件。风险：它执行任意 Python，生产环境要沙箱化，且非官方项目无 SLA。
- 来源: https://github.com/ahujasid/blender-mcp

## Hunyuan3D 2.1  (腾讯混元)
- 类别/成熟度: open-weights / **production** | 许可: 腾讯混元社区许可：月活超 100 万需单独向腾讯申请授权；明确写明「不适用于欧盟、英国和韩国」；分发须署名「Tencent Hunyuan 3D 2.1 is licensed under...」并向终端用户披露实际提供方；禁止用其输出去训练/改进其他 AI 模型 | 成本: 29GB VRAM 跑全流程（A6000/A100 级），单卡 4090 24GB 需分步或降配
- 是什么: 图像 → 高保真 3D 资产，带生产级 PBR 材质合成（金属反射、次表面散射）。
- 关键事实:
    * 2025-06-13 发布
    * 形状模型 3.3B 参数，纹理模型 2B 参数
    * 显存：仅形状 10GB，仅纹理 21GB，形状+纹理合计 29GB
    * 含 Delight 模型（去除输入图内置光照）
    * GitHub 4000 stars
    * 许可为 Tencent Hunyuan 3D 2.1 Community License
- 长片作用: 顶「角色/道具资产生成」工位，是本地化方案里唯一带 production-ready PBR 的。但那条「禁止用输出改进其他 AI 模型」在 AI 拍片平台里是真陷阱：你用它生成的角色渲染图去训 LoRA 或做角色一致性微调，就踩线了。做灰模底稿和最终渲染可以，做训练数据不行。EU/UK/韩国的地域排除也要写进合规文档。
- 来源: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE https://huggingface.co/tencent/Hunyuan3D-2.1

## TRELLIS  (Microsoft Research)
- 类别/成熟度: open-weights / **production** | 许可: MIT（子模块需单独核查） | 成本: 16GB VRAM 起，4090 可跑；无授权费
- 是什么: 结构化 3D latent 生成，单模型同时输出 3D Gaussians、辐射场和网格三种表示。
- 关键事实:
    * TRELLIS-image-large 1.2B；TRELLIS-text-base 342M、text-large 1.1B、text-xlarge 2.0B
    * MIT license（模型与主体代码；子模块 diffoctreerast、Modified Flexicubes 另有许可）
    * 2024-12-18 图像模型首发；2025-03-25 放出训练代码与文本模型
    * 要求 NVIDIA GPU 至少 16GB 显存，在 A100 / A6000 上测试
- 长片作用: MIT 许可让它成为商业产线里最干净的 3D 资产来源——没有 Hunyuan3D 的 MAU/地域/禁训条款。质量略逊于 Hunyuan3D 2.1 的 PBR，但本产线只要灰模底稿，PBR 本来就不是刚需。建议：底稿几何用 TRELLIS（干净），需要最终高质量贴图时才上 Hunyuan3D 并接受其条款。
- 来源: https://github.com/microsoft/TRELLIS

## TripoSG  (VAST AI Research)
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 8GB VRAM —— 本批 3D 生成里门槛最低
- 是什么: 1.5B rectified-flow transformer，图像 → 高保真 3D 网格。
- 关键事实:
    * 1.5B 参数，VAE 用 2048 latent tokens
    * MIT license
    * 2025-03 发布 v1.0；2025-04 发布 TripoSG-scribble（512 token，草图快速出型）
    * 最低 8GB VRAM
    * 训练数据 200 万条精筛 Image-SDF 对
    * 输出 GLB
- 长片作用: 顶「快速出型/道具批量生成」工位。8GB 显存意味着它能和视频生成模型共存在一台机器上排队，不用独占卡。scribble 版对分镜阶段「画个草图就出块几何」特别合适——2-8 分钟片子的背景道具量大，不值得每个都上 3.3B 模型。
- 来源: https://github.com/VAST-AI-Research/TripoSG

## UniRig  (VAST AI Research / 清华)
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 8GB VRAM；免费
- 是什么: 自回归大模型做自动绑定：输入静态 3D 模型，输出骨架层级 + 逐顶点蒙皮权重 + 可选骨骼属性。
- 关键事实:
    * SIGGRAPH 2025（TOG）
    * MIT license
    * 最低 8GB VRAM
    * 输入格式：.obj / .fbx / .glb / .vrm
    * 训练集 Rig-XL：14000+ 已绑定 3D 模型
    * 论文报告绑定精度提升 215%、动作精度提升 194%
    * 支持人、动物及各类物体，含精细二次元角色
    * 注意：完整的 Rig-XL/VRoid 训练版 checkpoint 仍标为「计划发布」，当前公开权重基于 Articulation-XL2.0
- 长片作用: 这是整条 3D 底稿链路的瓶颈解药。3D 生成模型出的是死网格，从死网格到能被 mocap 驱动，绑定过去是纯手工（每角色数小时到数天）。UniRig 把它压到分钟级，且支持 .vrm（直通 VTuber 工具链）。MIT 许可干净。但要清楚当前公开 checkpoint 不是论文里那个 Rig-XL 版，二次元角色上的实际表现可能低于论文数字。
- 来源: https://github.com/VAST-AI-Research/UniRig

## Rodin / Hyper3D  (Deemos)
- 类别/成熟度: closed-api / **production** | 许可: 闭源商业 API，权重不开放；付费档含商用导出权 | 成本: $30/月约 60 模型 ≈ $0.50/模型；$120/月约 416 模型 ≈ $0.29/模型
- 是什么: 闭源商业 3D 生成 API/网页服务，图像/文本/多图 → 3D 资产，含 HD 贴图。
- 关键事实:
    * Free 档 $0/月，按结果付费，单次直购 credit $1.50
    * Creator 档 $30/月（年付 $24/月 = $288/年），约 60 个模型
    * Business 档 $120/月（年付 $96/月 = $1152/年），约 416 个模型，含完整 API 访问
    * Business 档速率 120–240 RPM
    * Enterprise 档：私有部署、定制微调、批量折扣
    * 付费档提供「unlimited export and any use」权利，ChatAvatar 资产商用许可含在 Business 及以上
- 长片作用: 顶「不想自己维护 3D 生成算力时的外包工位」。算笔账：一部 2-8 分钟片子若需 30 个角色/道具资产，Creator 档 $30 就够了——比买卡跑 Hunyuan3D 便宜得多。且它已被 BlenderMCP 原生集成，能被 agent 直接调用。缺点：闭源、无法定制角色风格一致性，且资产要出境（若有数据合规要求需注意）。
- 来源: https://www.hyper3d.ai/pricing https://github.com/ahujasid/blender-mcp

## Wan2.2-Fun-A14B-Control  (阿里 AIGC Apps（VideoX-Fun）)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: 64GB 权重，A14B MoE 官方建议 80GB 级显存；社区 FP8/GGUF 量化可下探到 24GB 级
- 是什么: Wan2.2 A14B 的控制版权重，接受 Canny / Depth / Pose / MLSD 等控制条件加轨迹控制。
- 关键事实:
    * 权重体积 64.0 GB
    * 支持多分辨率 512 / 768 / 1024
    * 81 帧 @ 16 FPS（约 5 秒/段）
    * Apache 2.0
    * 同系列还有 Wan2.2-VACE-Fun-A14B（64.0GB，VACE 方案训练，额外支持指定主体做参考生成）与 Wan2.2-Fun-A14B-Control-Camera（64.0GB，镜头运动控制）
- 长片作用: 这是「3D 灰模底稿 → 成片画面」贴皮链路的主力工位，也是本次调研里最重要的一条。Blender 渲出 depth + normal + 骨架，喂进 Fun-Control，Wan2.2 负责贴皮和打光。Apache 2.0 意味着可本地部署、可商用、可叠自训 LoRA —— 这正是 Seedance 2.0 闭源 API 给不了的那一半能力。81 帧/段（5s）决定了分镜粒度：2-8 分钟片子拆成 24–96 段，每段一次控制生成。
- 来源: https://github.com/aigc-apps/VideoX-Fun

## VACE (Video All-in-one Creation and Editing)  (阿里 ali-vilab)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0（LTX 变体为 RAIL-M，商用需另看） | 成本: 1.3B 版是消费级卡（12–16GB）唯一能跑的完整控制方案；14B 版需 40GB+
- 是什么: 统一的视频生成与编辑控制框架，覆盖 Move-Anything / Swap-Anything / Reference-Anything / Expand-Anything / Animate-Anything。
- 关键事实:
    * VACE-Wan2.1-1.3B-Preview：81 × 480 × 832，Apache 2.0
    * Wan2.1-VACE-14B：81 × 720 × 1280，Apache 2.0
    * VACE-LTX-Video-0.9：97 × 512 × 768，RAIL-M 许可
    * 2025-03-31 首发；2025-05-14 放出 1.3B 与 14B；2025-06-26 被 ICCV 2025 接收
    * 官方仓库仅覆盖到 Wan2.1，Wan2.2 版本由 VideoX-Fun 的 Wan2.2-VACE-Fun-A14B 提供
- 长片作用: 顶「镜头内局部改写」工位——角色换装、道具替换、画面扩展，不用整段重生成。在 2-8 分钟片子的返修阶段价值极高：导演说「这个镜头角色衣服不对」，VACE 能只改那一处而不动构图和运动，纯 prompt 路线做不到。1.3B 版本 Apache 2.0 + 低显存，是小团队起步的现实选择。
- 来源: https://github.com/ali-vilab/VACE

## Go-with-the-Flow（噪声形变运动控制）  (Eyeline Research / Netflix)
- 类别/成熟度: technique / **research** | 许可: 未能从仓库确认 | 成本: 声称不增加架构开销、噪声预处理可实时；微调成本未公布
- 是什么: 不改模型结构，只改训练/推理时的噪声采样：用光流场把随机时间维高斯噪声替换成相关的 warped noise，保持空间高斯性。
- 关键事实:
    * arXiv 2501.08331，2025-01-14 提交，最新版 2025-08-06
    * CVPR 2025 Oral
    * 噪声形变算法可实时运行
    * 一次微调即同时获得局部物体运动控制、全局相机运动控制、运动迁移三种能力
    * 论文未指明具体微调的 base 模型；摘要只称适用于「现代视频扩散基座模型」
    * 注：GitHub 仓库 Eyeline-Research/Go-with-the-Flow 本次抓取返回空内容，代码可用性存疑
- 长片作用: 理论上顶「用 3D 渲染的光流直接驱动视频生成」工位——比 depth/pose ControlNet 更轻，因为不需要额外控制分支。Blender 能原生输出精确光流 AOV（Vector pass），这是 3D 底稿路线的天然优势。但本次未能验证代码与权重可用性，也没有针对 Wan2.2 的公开实现，暂不能进主线产线，只能当技术储备。
- 来源: https://arxiv.org/abs/2501.08331

## CogVideoX ControlNet (TheDenk)  (社区第三方)
- 类别/成熟度: open-weights / **research** | 许可: Apache 2.0 | 成本: 48GB（2B）/ 80GB（5B）—— 性价比极差
- 是什么: 给 CogVideoX 加 ControlNet 控制分支。
- 关键事实:
    * 仅支持 Canny 与 Hed 两种控制类型；不支持 pose
    * 基座支持 CogVideoX-5b 与 CogVideoX-2b
    * 显存：2B 版需 48GB（如 A6000），5B 版需 80GB
    * Apache 2.0
    * 分辨率与帧数未在文档给出
- 长片作用: 明确不推荐进产线。48GB 显存只换来 Canny/Hed 两种控制、没有 depth 也没有 pose，而 Wan2.2-Fun-Control 在相近显存下给出 Depth+Pose+Canny+MLSD+轨迹。这一条列出来是为了排除选项：2026 年做 3D 底稿贴皮，CogVideoX 线路已无理由。
- 来源: https://github.com/TheDenk/cogvideox-controlnet

## Wan2.2-Animate-14B  (阿里通义实验室)
- 类别/成熟度: open-weights / **production** | 许可: Apache 2.0 | 成本: FP8 版可在 24GB 级卡运行；BF16 需 40GB+。开源无 API 费
- 是什么: 角色图 + 驱动视频 → 角色动画；两种模式：Animation（角色复刻驱动视频的表情动作）与 Replacement（把角色塞回原视频替换原角色，复刻场景光照色调）。
- 关键事实:
    * 2025-09-19 发布，14B 参数，Apache 2.0
    * arXiv 2509.14055
    * 基于 Wan-I2V 改造：骨架信号空间对齐注入身体动作，源图隐式面部特征注入表情
    * 配套 Relighting LoRA，用于在保持角色外观一致的同时套用环境光色
    * ComfyUI 官方工作流用 DWPose Estimator 自动把输入视频预处理成 pose 与 face 控制视频
    * ComfyUI 的 Video Extend 每段增加约 77 帧（约 4.8 秒），可复制模块串接，通过 batch 与 offset 参数链接
    * 要求视频宽高为 16 的倍数
    * 提供 FP8 scaled（Kijai 版）与 BF16 两种权重
- 长片作用: 顶「角色表演贴皮」的核心工位，是目前开源里最成熟的一条。与 3D 底稿的接法：Blender 渲出灰模角色的动作视频当驱动视频 → Wan-Animate 贴上最终角色皮。77 帧/段的 Video Extend 是长片的关键——2-8 分钟意味着要串 25–100 段，这条链路工程上跑得通但累积漂移是真问题（详见 recommendation）。
- 来源: https://github.com/Wan-Video/Wan2.2 https://humanaigc.github.io/wan-animate/ https://arxiv.org/abs/2509.14055

## Wan-Animate-2  (阿里通义实验室)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 720P 需 8×A800（约 640GB 显存池）；480P 需 2×A800。这是本批里硬件门槛最高的一个
- 是什么: 端到端角色动画框架，在重新设计的 DiT 里直接消费驱动视频，去掉中间运动提取器。
- 关键事实:
    * 2026-08-07 发布（仓库最后更新 2026-08-08）
    * 14B 参数（Wan2.2-Animate-2-14B）
    * Apache 2.0
    * 支持 720P 生成，调优目标为 8×A800；480P 生成为 2×A800
    * 明确「eliminating intermediate motion extractors」以提升动作保真与身份保持
    * 新增 text-driven viewpoint control（文本驱动视角控制）
    * 提供蒸馏版用于实时推理
    * README 中 arXiv 号为占位符「arXiv:TODO.06009」，论文尚未正式挂出
    * GitHub 319 stars（相对 Wan2.2 的 17551 仍属早期）
- 长片作用: 这是本次调研对产线战略影响最大的一条。它取消中间运动提取器，意味着不再需要「渲染骨架图 / DWPose」这一层——直接把 Blender 渲的灰模视频当驱动视频灌进去。这既简化了 3D 底稿链路（少一个预处理环节、少一次信息损失），也削弱了「必须精确渲 pose」的论点。text-driven viewpoint control 则部分替代了 3D 相机规划。但 8×A800 的门槛和无论文/早期 star 数说明它还没到能压产线的程度：现阶段当 R&D 跟踪，主线仍押 Wan2.2-Animate + Fun-Control。
- 来源: https://github.com/Wan-Video/Wan-Animate-2 https://github.com/Wan-Video

## Wan-Dancer-14B  (阿里通义实验室)
- 类别/成熟度: open-weights / **usable** | 许可: Apache 2.0 | 成本: 8×A800 80GB；开源无授权费但算力门槛高
- 是什么: 音乐驱动的长时长舞蹈视频生成：参考图 + 音频 + 文本提示 + seed → 带全局结构与时间连续性的长视频。
- 关键事实:
    * arXiv 2607.09581，v1 提交 2026-07-10，v3 修订 2026-07-17；仓库更新至 2026-07-17
    * 14B 参数，Apache 2.0
    * 生成 720p / 30fps、时长超过 1 分钟的稳定视频
    * 论文明确指出现有扩散模型「典型在 20 秒后失效」
    * 两阶段：全局关键帧规划 + 局部时间精修，利用全曲音乐上下文
    * 技术点：time-mapped RoPE 实现动态帧率自适应、基于光流的损失函数增强运动连续性、运动速度控制
    * 测试硬件 8 × NVIDIA A800 80GB
- 长片作用: 这是公开权重里唯一明确做到「单次生成 >1 分钟连续人物视频、720p/30fps」的模型，直接冲击「靠分镜拼接」这个前提。对 2-8 分钟叙事片的意义：它证明了「全局关键帧规划 + 局部精修」这个两阶段范式能把连续时长从 20s 推到 60s+。但它是音乐/舞蹈专用，不是叙事片通用模型——不能直接拿来拍对白戏。价值在于范式借鉴：你的分镜系统应该照这个结构设计（先全局规划关键帧，再逐段精修），而不是纯线性串 5 秒片段。
- 来源: https://github.com/Wan-Video/Wan-Dancer https://arxiv.org/abs/2607.09581

## UniAnimate-DiT  (阿里 ali-vilab)
- 类别/成熟度: open-weights / **usable** | 许可: 声明「面向学术研究」，并附免责声明称用户自负责任 —— 商业使用条款不明确，是实质风险 | 成本: 480P 优化后 14GB（消费级 4080/4090 可跑）；5s 480p ≈ 3 分钟/A800
- 是什么: 在 Wan2.1-14B-I2V 上用 LoRA 微调做人物动画，姿态驱动。
- 关键事实:
    * 基座 Wan2.1-14B-I2V；LoRA rank 64/alpha 64（单卡训练），多卡用 rank 128/alpha 128
    * 480P 默认约 23GB 显存，开优化后 14GB；720P 默认约 36GB，开优化后 26GB
    * 输出 81 帧、832×480（480P）或 1280×720（720P）
    * 训练于 832×480，但「直接在 1280×720 推理通常是允许的」
    * 提供独立的长视频推理脚本（inference_unianimate_wan_long_video_480p.py 及 720p 版）
    * teacache 可带来约 4 倍推理加速
    * 单张 A800 生成 5 秒 480p 约 3 分钟
    * 最新 checkpoint 为 2025-04
- 长片作用: 14GB 显存这个数字是它唯一的护城河：在没有 A800 的团队里，它是能真跑起来的人物动画方案。3 分钟/5 秒 的速度意味着 2-8 分钟成片（假设 30 段×5s）在单卡上约 1.5 小时一轮——可接受。但「面向学术研究」的许可措辞对商业平台是红线，需要法务判断或找作者授权。相比之下 Wan2.2-Animate 的 Apache 2.0 干净得多。
- 来源: https://github.com/ali-vilab/UniAnimate-DiT

## Animate-X  (蚂蚁集团 / 阿里)
- 类别/成熟度: open-weights / **usable** | 许可: Apache-2.0 | 成本: SD2.1 级基座，消费级卡可跑；未公布具体 VRAM
- 是什么: 面向各类角色（含拟人化非人形角色）的通用动画框架，用 Pose Indicator 强化姿态表征。
- 关键事实:
    * ICLR 2025 接收（2025-02-11 公告）
    * Apache-2.0
    * checkpoint 2024-12-10 发布，代码 2024-12-20
    * 基座为 latent diffusion（SD 2.1，v2-1_512-ema-pruned.ckpt）
    * 姿态检测用 DWPose（dw-ll_ucoco_384.onnx）
    * 默认输出 768×512、32 帧 @ 8fps，可配置到 96+ 帧
- 长片作用: 768×512 / 32 帧 @ 8fps 是 2024 年的规格，画质与时长在 2026 年已不具竞争力，唯一剩余价值是非人形角色（拟人动物、玩偶、非标准体型）——这类角色 SMPL 骨架不适用，Wan-Animate 的人形骨架也覆盖不好。若你的叙事片里有非人形角色，它顶那一个窄工位；否则不要用。
- 来源: https://github.com/antgroup/animate-x

## MimicMotion  (腾讯)
- 类别/成熟度: open-weights / **usable** | 许可: 仓库含 LICENSE 文件，类型未在 README 说明 | 成本: 16GB VRAM（4060ti）；35s 视频 = 20 分钟/4090
- 是什么: 基于 SVD 的姿态引导人物动画，带置信度感知的姿态引导。
- 关键事实:
    * ICML 2025，arXiv 2406.19680
    * 基座 stabilityai/stable-video-diffusion-img2vid-xt-1-1
    * 最大 72 帧 @ 576×1024
    * 显存：72 帧模型 16GB（4060ti 可跑）；16 帧 U-Net 最低 8GB，但 VAE 解码需 16GB
    * 35 秒 demo 在 4090 上需 20 分钟
    * 代码与首版 checkpoint 2024-07-01，v1.1 checkpoint 2024-07-08
- 长片作用: 16GB 显存 + 576×1024 竖屏是它剩下的全部价值，适合超低成本竖屏短内容。对 2-8 分钟叙事片：72 帧上限（约 4.8s @15fps）与 SVD 代画质都不够，20 分钟/35 秒的速度换算到 8 分钟成片约 4.5 小时/轮，且质量不如 Wan 系。建议排除。
- 来源: https://github.com/Tencent/MimicMotion https://arxiv.org/abs/2406.19680

## Champ  (复旦生成视觉实验室 / 南京大学)
- 类别/成熟度: open-weights / **usable** | 许可: MIT | 成本: 约 20GB VRAM 跑 250 帧；MIT 免费
- 是什么: 用 SMPL 渲染出的多路条件图驱动人物动画——这正是「3D 底稿贴皮」的经典范式原型。
- 关键事实:
    * ECCV 2024，代码 2024-03-24 首发，MIT license
    * 基座 Stable Diffusion v1.5，含 denoising UNet、guidance encoders、Reference UNet、motion module
    * 需要四路引导序列：depth/、normal/、semantic_map/、dwpose/（外加 mask/）
    * 默认 motion-02 约 250 帧，需约 20GB 显存
    * 测试 GPU：A100、RTX3090；Ubuntu 20.04 / Windows 11，CUDA 12.1
    * 显存不足时需换更短的动作序列
- 长片作用: 它对本次调研的价值不在于「用它拍片」（SD1.5 画质已过时），而在于它给出了「3D 底稿需要渲什么」的权威答案：depth + normal + semantic segmentation + 2D 骨架，四路并用。这直接回答了「灰模够不够」——Champ 的消融说明单靠骨架不够，normal 提供表面朝向、semantic 提供部位归属，二者缺一则手脚穿插与身份混淆明显上升。把这四路映射到 2026 的 Wan2.2-Fun-Control（原生吃 Depth + Pose），你至少要补齐 depth 与 normal。
- 来源: https://github.com/fudan-generative-vision/champ

## R-DMesh  (腾讯混元)
- 类别/成熟度: open-weights / **research** | 许可: 未标明（需人工确认） | 成本: 未公布
- 是什么: 视频引导的 3D 动画：给一个静态网格和一段参考视频，自动把网格对齐到视频起始姿态并生成时序一致的动画。支持姿态重定向、动作重定向与 4D 生成。
- 关键事实:
    * SIGGRAPH 2026 接收（2026-03-28 公告）
    * 代码 2026-05-12 发布，checkpoint 2026-05-13 发布
    * 输入：静态网格 .glb 或 .fbx + 参考视频 .mp4
    * 输出：动态网格 .fbx + 渲染视频
    * GitHub 60 stars / 9 forks（极早期）
    * 许可与显存需求未在仓库页面标明
- 长片作用: 这是「3D 底稿」路线在 2026 年的新解法，也是对传统 mocap→retarget 链路的直接替代：跳过 SMPL 中间层，参考视频直接驱动你自己的角色网格，输出 .fbx 可进 Blender/UE 做最终渲染。若可用，它把「GVHMR 抽 SMPL → Blender 重定向到角色骨架 → 修穿模」三步压成一步。但 60 stars / 4 个月内无生态说明尚未被验证，且许可未标明。列为高优先级跟踪项，不进当前产线。
- 来源: https://github.com/Tencent-Hunyuan/R-DMesh https://github.com/Tencent-Hunyuan

## ComfyUI-WanVideoWrapper（长上下文窗口调度）  (Kijai（社区）)
- 类别/成熟度: framework / **production** | 许可: 社区项目，需核对仓库 LICENSE | 成本: 1.3B + 滑窗：<5GB 跑 1025 帧；14B + block swap：约 16GB 起
- 是什么: ComfyUI 下的 Wan 系模型统一封装，含长视频滑窗调度与显存卸载。
- 关键事实:
    * 支持 WanVideo T2V（1.3B / 14B）、WanVideoFun 控制模式、WanAnimate、VACE、ReCamMaster（相机运动）、MoCha、S2V
    * 长视频：文档演示 1025 帧，用 81 帧窗口 + 16 帧重叠
    * 1025 帧长视频在 1.3B T2V 模型上「用了不到 5GB 显存」
    * Block swap 可配置：512×512×81 在 20/40 blocks 卸载下约 16GB 显存
    * 还集成免训练方法 SteadyDancer 与 One-to-All-Animation
- 长片作用: 这是把 81 帧上限捅破的工程答案：81 帧窗口 + 16 帧重叠的滑窗调度，1025 帧 ≈ 64 秒 @16fps 单次生成。2-8 分钟片子按这个调度是 2–8 次长段而非 24–96 次短段，段间接缝数量下降一个数量级——这对一致性的实际贡献可能比换更大的模型还大。「<5GB 跑 1025 帧」说明瓶颈在模型规模不在序列长度。产线必须基于这类滑窗调度器构建，不要自己写朴素的首尾帧串接。
- 来源: https://github.com/kijai/ComfyUI-WanVideoWrapper

## MetaHuman + Live Link（UE5 路线）  (Epic Games)
- 类别/成熟度: service / **production** | 许可: Epic 的 MetaHuman 许可条款本次未能抓取验证（unrealengine.com 返回 403，Epic 文档为 JS 渲染）——见 uncertainClaims | 成本: UE5 本体免费（年收入超阈值后按 Epic 条款分成）；需一台 iPhone（TrueDepth）或普通网络摄像头
- 是什么: UE 内的高保真数字人创建（MetaHuman Creator）与表演捕捉动画生成（MetaHuman Animator）。
- 关键事实:
    * MetaHuman Creator：「Create high-fidelity, digital human characters directly in Unreal Engine」
    * MetaHuman Animator 支持三条采集路径：iOS 设备 TrueDepth 摄像头的深度数据（离线）、立体头戴相机 HMC（离线）、以及「real time from any mono video camera (including webcam)」实时路径
    * Live Link Face 应用同时提供 iOS 和 Android 版本
    * MetaHuman 文档已独立于 UE 文档主站（dev.epicgames.com/documentation/en-us/metahuman/）
- 长片作用: 顶「面部表演与口型」工位，是开源方案最弱的一环。Wan-Animate 的隐式面部特征做表情迁移，但对白戏的精确口型（尤其中文）仍是短板。MetaHuman Animator 的 mono webcam 实时路径意味着导演可以自己对着摄像头念台词，生成高质量面部动画曲线，再渲成灰模底稿喂给 Wan2.2-Fun-Control 贴皮。最大未决风险是许可：MetaHuman 资产能否用于非 UE 渲染/线性内容/AI 训练素材，必须先做法务确认。
- 来源: https://dev.epicgames.com/documentation/en-us/metahuman/metahuman-documentation https://dev.epicgames.com/documentation/en-us/metahuman/metahuman-animator-in-unreal-engine

## Warudo  (HakuyaLabs)
- 类别/成熟度: service / **production** | 许可: 个人/非直播免费；企业用途需 Warudo Pro（价格未在文档页列出） | 成本: 个人免费；Pro 定价需询价
- 是什么: Unity 引擎的 VTuber / 虚拟制作软件，节点式蓝图，聚合极多 mocap 输入源。
- 关键事实:
    * 非直播用途免费；直播时若独占持有自己的 VTuber IP 且无合约直播时长要求（主流平台除外）也免费；企业 VTuber 须购买 Warudo Pro
    * 支持的动捕源：MediaPipe、OpenSeeFace、iFacialMocap、SteamVR、Leap Motion、Sony mocopi、Rokoko、Xsens MVN、Virdyn Studio、Noitom Axis、StretchSense 手套、VMC、MotionBuilder、Vicon Shogun、OptiTrack Motive、Chingmu Avatar
    * 支持「任何 Unity 兼容的 3D 资产」
    * 文档最后更新 2026-01-13
- 长片作用: 顶「低成本实时表演采集台」工位。它的真实价值是那份动捕源清单：MediaPipe（纯摄像头零成本）到 OptiTrack（专业光学）在同一套软件里统一成 Unity 场景，输出可走 VMC 协议转出。对 2-8 分钟片子：用它实时预演走位和表演（导演即时看到角色在演什么），比 Blender 里 offline 烘焙迭代快一个量级。但它是实时预览工具，不是渲染农场——最终底稿仍应回 Blender 出 depth/normal AOV。
- 来源: https://www.warudo.app/ https://docs.warudo.app/docs/

## VSeeFace  (Emiliana（独立开发）)
- 类别/成熟度: service / **production** | 许可: 免费软件（非开源，具体条款见官网） | 成本: 免费；一个网络摄像头即可，iPhone 可选
- 是什么: 免费的 VRM 虚拟形象实时面部/动作追踪软件，支持 VMC 协议双向收发。
- 关键事实:
    * 最新版本 v1.13.38c5
    * 完全免费
    * 模型格式：仅支持 VRM0，「VSeeFace only supports the VRM0 standard, not VRM 1.0」；另支持 VSFAvatar（Unity asset bundle，可带自定义动画/着色器/组件）
    * 不支持 Live2D
    * 追踪：网络摄像头面部（含眼动、眨眼、眉毛、口型）、Leap Motion 手指、iPhone 经 iFacialMocap/FaceMotion3D 提供 52 个 ARKit blendshape、ThreeDPoseTracker 全身
    * VMC 协议「both supports sending and receiving」人形骨骼旋转、根节点偏移与 blendshape 值
    * 支持网络追踪（把追踪负载分到另一台 PC）
- 长片作用: 顶「零预算面部动作采集」工位，且 VMC 双向收发是关键——它能把面部数据喂给 Warudo/Unity，也能接收外部数据。52 个 ARKit blendshape 是与 MetaHuman Live Link Face 同一套标准，意味着两条路线的面部数据可互换。限制：VRM0 only 会卡住新资产（UniRig 输出 .vrm 需确认是 VRM0 还是 1.0），且它输出的是实时流不是可编辑曲线，做精修要走 VMC 录制到 Blender。
- 来源: https://www.vseeface.icu/

## VTube Studio  (Denchi / DenchiSoft)
- 类别/成熟度: service / **production** | 许可: 商业软件，一次性买断 | 成本: 一次性买断（价格需查 Steam）
- 是什么: Live2D 虚拟形象追踪软件，Steam 一次性买断。
- 关键事实:
    * Steam 上为「a non-recurring one-time payment」一次性付费（官网未列具体价格）
    * 仅支持 Live2D 模型，官网未提及 3D 模型支持
    * 追踪：经 OpenSeeFace 的网络摄像头追踪、iPhone/Android 设备作为面部追踪器、新增手部追踪、眼动与眨眼
    * 最低要求：iOS 需 FaceID 或 A12 及以上芯片；Android 需支持 Google ARCore
    * 提供插件 API（如 Twitch 打赏、手柄联动）
    * 平台：Steam（PC/Mac）、iOS、Android
- 长片作用: 对本产线基本无用，列出是为了明确排除。它是 2D Live2D 工具，不产出任何 3D 底稿、depth 或 normal，也没有可导出的 3D 动作数据。VTuber 工具链里只有 Warudo（Unity 3D）和 VSeeFace（VRM + VMC）对「3D 底稿 → I2V 贴皮」有实际贡献。
- 来源: https://denchisoft.com/

## Hunyuan3D-Buffalo1.0 / Hunyuan3D-WorldClaw  (腾讯混元)
- 类别/成熟度: open-weights / **research** | 许可: 未标明 | 成本: 未公布
- 是什么: Buffalo 1.0 为统一多模态 3D 生成/理解/编辑模型；WorldClaw 为 agentic 大规模 3D 开放世界生成。
- 关键事实:
    * Buffalo1.0：2026-08-05 发布，arXiv 2608.02711，仅 248 stars / 7 commits / 10 forks
    * WorldClaw：仓库更新至 2026-08-13，1289 stars
    * 相关同期项目：HY-World-2.0（2656 stars，2026-08-12）、HunyuanWorld-Mirror（ICML 2026，1205 stars）、HunyuanWorld-Voyager（1596 stars）、Hunyuan3D-Part（541 stars，2025-12-05）
    * Buffalo1.0 的参数量、显存、许可、是否开放权重均未在仓库页面标明
- 长片作用: WorldClaw / HY-World-2.0 这条线顶的是「场景底稿」工位（不是角色）——2-8 分钟叙事片的场景一致性问题和角色一致性同等重要，纯 prompt 生成的场景每镜都在变。但 7 个 commit、许可未标、参数未公布，这批 2026-08 的东西全部只能算 research，不能排进产线。明确标注：这些是技术报告 + 仓库占位阶段，不是可用工具。
- 来源: https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0 https://github.com/Tencent-Hunyuan

### 建议
## 一、直接回答核心问题：真实增益、成本增量、何时值得上

### 增益不是「画质更好」，而是三件纯 prompt 结构性做不到的事

**1. 跨镜身份一致性从「概率」变成「约束」。** 纯 prompt / 纯参考图路线下，角色一致性是靠模型的参考图理解能力维持的概率事件，误差在镜头间累积且不可回滚。3D 底稿路线把角色固化成一个网格资产（TRELLIS 1.2B / Hunyuan3D 2.1 3.3B），每个镜头都从同一个网格渲出，几何与比例是硬约束；漂移只可能发生在最后一道贴皮（Wan2.2-Fun-Control / Wan2.2-Animate），且是单镜独立的、不累积的。

**2. 走位、道具交互、相机运动变成可复现可返修的数据。** 这是最被低估的增益。导演说「这个镜头角色应该从左边进画、绕过桌子」——纯 prompt 只能重新抽奖；3D 底稿里改一条曲线重渲即可，且改完角色长相完全不变。返修成本从 O(重新生成整镜并祈祷) 降到 O(改关键帧)。对 2-8 分钟片子（几十个镜头、必然多轮返修）这是决定项目能否收敛的因素。

**3. 打通了 Seedance 2.0 硬约束下的一致性瓶颈。** 既然权重不开放、不能本机 LoRA、参考最多 9 图 + 3 视频，3D 底稿正好是「制造那 9 张参考图」的最优解：同一个网格渲 9 个角度/表情的 turnaround，加 3 段灰模驱动视频占满 video 槽。这是在不能微调的前提下，把一致性预算用到极致的唯一现实打法。

### 成本增量（以下为我的工程估算，非厂商数据）

一次性投入（整片摊销）：
- 角色资产：TripoSG（8GB VRAM）或 Hunyuan3D 2.1（29GB）生成 + UniRig（8GB）自动绑定。主角约 2–6 人时/个（含手工修型修权重），配角/道具 15–40 分钟/个。或走 Rodin Creator 档 $30/月约 60 个模型（≈$0.50/模型），外包掉算力。
- 场景搭建：BlenderMCP（MIT，29k stars）让 agent 按分镜批量摆位打光，把过去最烧人力的环节自动化。

每镜增量：
- 动作获取：GVHMR / WHAM 从参考表演视频抽 SMPL，分钟级/镜；无素材时用 HY-Motion-1.0（1.0B，26GB VRAM，建议 <5 秒）文生动作块。
- Blender 重定向 + 灰模渲染 + AOV 导出：EEVEE Next 秒级/帧，1080p 一个 5 秒镜头约 1–3 分钟含导出。
- 人工修正（穿模、脚滑、手部）：这是真正的成本，约 10–30 分钟/镜，取决于动作复杂度。

**净增量：每镜约 20–45 分钟人工 + 数分钟算力。** 对 8 分钟片（按滑窗调度约 8–30 个长段）总增量约 5–20 人时。对比：纯 prompt 路线在同等镜头数下的「重抽 + 挑片 + 一致性返工」通常吃掉同一个量级甚至更多的时间，只是分布得更碎、更不可预测。

### 什么时候值得上 —— 明确阈值

**值得上（三条命中任意两条即上）：**
- 同一角色出现在 ≥ 6–8 个镜头（资产复用摊薄一次性成本）
- 有明确的空间关系需求：角色与道具交互、多角色同框走位、连贯的相机运动
- 片长 > 90 秒，或预期 ≥ 2 轮导演返修

**不值得上（直接走纯 prompt / I2V 首帧控制）：**
- 单镜 < 15 秒的空镜、环境镜、情绪特写、抽象转场
- 远景群演、背景人物
- 一次性出现的角色
- 探索阶段的风格试错（3D 底稿会锁死构图，反而妨碍试）

**现实建议：混合分层。** 按镜头打标签，主角叙事镜走 3D 底稿（占全片约 40–60%），氛围镜走纯 prompt。不要全片一刀切——那是这条路线最常见的失败方式。

## 二、渲染底稿的最低精度要求（直接回答「灰模够不够、要不要打光」）

**灰模够，但必须满足三个条件，否则不够：**

1. **必须出 depth + normal，不能只出骨架。** Champ（ECCV 2024，MIT）的范式给了权威答案：它同时用 depth/、normal/、semantic_map/、dwpose/ 四路 SMPL 渲染图。单靠 2D 骨架时，手脚穿插与前后遮挡关系会崩。Wan2.2-Fun-Control 原生吃 Depth + Pose + Canny + MLSD，至少把 Depth 和 Pose 一起给。

2. **必须打光，但不是为了好看，是为了让 normal/灰度有梯度。** 纯 flat 无光照灰模在转成 Canny/MLSD 时边缘信息几乎为零，模型只能靠 depth 单通道推断表面朝向，肩颈、衣褶、面部结构会被抹平。做法：EEVEE Next 三点光（key + fill + rim），不需要 PBR 材质、不需要 GI、不需要最终光照质量——要的是「表面朝向可被分辨」。渲染成本仍是秒级/帧。

3. **不需要的东西要明确砍掉：** 不需要贴图、不需要头发/布料解算、不需要 4K、不需要 Cycles。这些在贴皮环节会被完全覆写，渲了就是纯浪费。底稿分辨率与目标一致即可（Wan2.2-Fun-Control 支持 512/768/1024，且 Wan-Animate 要求宽高为 16 的倍数）。

**例外：Wan-Animate-2（2026-08-07）明确去掉了中间运动提取器，直接消费驱动视频。** 若它成熟，「渲 pose 图」这一层可以省掉，直接把打了光的灰模视频当驱动视频灌进去。但它需要 720P 8×A800 / 480P 2×A800，现阶段不能压产线。

## 三、推荐产线（按工位分配，2026-09 可落地版）

| 工位 | 主选 | 备选 / 说明 |
|---|---|---|
| 角色资产 | TRELLIS（MIT，1.2B，16GB） | Hunyuan3D 2.1 质量更好但许可有坑；Rodin $30/月外包 |
| 自动绑定 | UniRig（MIT，8GB，SIGGRAPH 2025） | 注意公开 checkpoint 非论文的 Rig-XL 版 |
| 身体动捕 | GVHMR（世界坐标）+ WHAM（MIT 交叉校验） | 手/脸补 SMPLest-X；无素材用 HY-Motion-1.0 |
| 面部/口型 | MetaHuman Animator（mono webcam 实时路径） | 零预算走 VSeeFace（52 ARKit blendshape + VMC） |
| 实时预演 | Warudo（非直播免费，16+ 种 mocap 源） | 只做预演，最终底稿回 Blender |
| 场景/渲染 | Blender 5.2 LTS + EEVEE Next + AOV(depth/normal/vector) | 注意不是 4.x；BlenderMCP 做 agent 自动化 |
| 主贴皮 | Wan2.2-Fun-Control A14B（Apache 2.0，64GB 权重） | 这是核心工位 |
| 角色表演贴皮 | Wan2.2-Animate-14B（Apache 2.0，FP8 可 24GB） | ComfyUI Video Extend 每段 ~77 帧/4.8s |
| 长序列调度 | ComfyUI-WanVideoWrapper 滑窗（81 帧窗 + 16 帧重叠，实测 1025 帧） | **这条比换更大模型更重要** |
| 局部返修 | VACE 1.3B / Wan2.2-VACE-Fun-A14B | 改衣服/道具不动构图 |
| 闭源出口 | Seedance 2.0（9 图 turnaround + 3 段灰模驱动视频） | 受审核约束，作为高质量单镜渠道 |

**排除清单（明确不要用）：** CogVideoX-ControlNet（48–80GB 换来只有 Canny/Hed）、MimicMotion（72 帧 / SVD 代画质）、Animate-X（768×512 @8fps，仅非人形角色还有价值）、VTube Studio（纯 2D Live2D，对 3D 底稿零贡献）。

**许可红线（必须进法务清单）：** ① Hunyuan3D 2.1 禁止用其输出改进其他 AI 模型，且不适用于欧盟/英国/韩国，>100 万 MAU 需另行授权；② NLF 权重仅限非商业研究；③ UniAnimate-DiT 声明「面向学术研究」；④ MetaHuman 能否用于 UE 之外的渲染与线性内容，本次未能验证，必须先确认。

## 四、最重要的一条战略判断

**2026 年真正的变量不是「3D 底稿 vs 纯 prompt」，而是长序列调度。** Wan-Dancer（arXiv 2607.09581，2026-07）已经做到 720p/30fps 超过 1 分钟单次连续生成，论文直接点出现有扩散模型「典型在 20 秒后失效」；ComfyUI-WanVideoWrapper 用 81 帧窗 + 16 帧重叠跑到 1025 帧且 1.3B 模型下显存不到 5GB。这意味着 2-8 分钟片子的接缝数量可以从几十降到几个。

**把预算排序定为：① 长序列滑窗调度 → ② 3D 角色资产复用 → ③ 更大的贴皮模型。** 先把接缝数量降一个数量级，再谈底稿精度；反过来做，会在一个每 5 秒断一次的管线上花大钱修单镜质量，而观众感知到的不一致主要来自接缝，不是单镜。同时按 Wan-Dancer 的两阶段范式设计分镜系统（全局关键帧规划 + 局部时间精修），而不是线性串短片段。

### 存疑
- 【检索能力受限声明】本次任务开始时 WebSearch 配额已耗尽（200/200），全部结果来自 WebFetch 定点抓取已知 URL。这意味着：2026 年 5 月之后新出现、且不在我已知 URL 列表里的方法/模型很可能被遗漏。以下所有 findings 的覆盖面应视为「已知项的核实」而非「全领域扫描」。
- 【Seedance 2.0 硬约束未能核实】任务给出的「官方单段 4-15 秒、2.5 约 30 秒、多模态参考最多 9 图 + 3 视频 + 3 音频、自带对白声、官方通道有内容审核」本次未能验证：seed.bytedance.com/en/seedance 只呈现 Seedance 1.0（1080p、多镜头生成）；volcengine.com/product/seedance 与 docs.volcengine.com/docs/82379/1520757 抓取返回空内容（JS 渲染）。按指令未推翻这些约束，但也未获得任何官方证据，recommendation 中基于它们的推论需要你自行复核。
- 【MetaHuman 许可】MetaHuman 资产能否用于 Unreal Engine 之外的渲染器/引擎/线性内容，以及 2025-2026 是否有许可变更，本次完全未能验证：unrealengine.com 的 EULA 与新闻页返回 403，dev.epicgames.com 的 FAQ 与 licensing 页为 JS 渲染只返回目录标题。这是 UE5 路线的最大未决法律风险。
- 【Auto-Rig Pro】价格、当前版本号、Blender 5.x 兼容性未能获取：blendermarket.com 已 301 跳转至 superhivemarket.com，而后者返回 403。任务要求覆盖此项但无法给出任何可引用数字，故未写入 findings。
- 【Rigify 具体规格】docs.blender.org 的 rigify 文档页（index/introduction/basics 三个路径）全部返回 404，未能核实其随 Blender 默认附带状态、metarig 种类与是否内建 mocap 重定向。Rigify 随 Blender 分发属常识，但本次无检索证据支撑任何具体条目。
- 【Hunyuan3D 3.0 / 2.5 是否有开放权重】Tencent-Hunyuan 组织的 3D 仓库列表中不存在 Hunyuan3D-3.0 或 Hunyuan3D-2.5 仓库；huggingface.co/tencent/Hunyuan3D-3.0 返回 401。搜索结果显示 2.5 仅有技术报告（2025-06）。倾向判断：3.0 若存在则可能是闭源 API/技术报告形态，但无法确证，故未写入 findings。
- 【Go-with-the-Flow 代码可用性】github.com/Eyeline-Research/Go-with-the-Flow 抓取返回空仓库（无文件无 README）。arXiv 2501.08331 与 CVPR 2025 Oral 身份可确认，但权重与代码是否实际可用无法核实，故 maturity 标为 research。
- 【SteadyDancer 与 One-to-All-Animation】二者仅在 ComfyUI-WanVideoWrapper 文档中被提及为「免训练技术」，尝试抓取 github.com/Kunbyte-AI/SteadyDancer 返回 404，未找到独立仓库或论文。无法给出任何规格数字。
- 【Wan-Animate-2 论文】仓库 README 中 arXiv 号为占位符「arXiv:TODO.06009」，论文尚未正式发布。其「eliminating intermediate motion extractors」与「distilled variant for real-time inference」的实际效果无第三方验证，320 stars 属极早期。
- 【多个开源项目的许可证类型未确认】GVHMR、SMPLest-X、MimicMotion、HY-Motion-1.0、R-DMesh 的仓库均含 LICENSE 文件但 README 未标明类型，本次未逐个抓取 LICENSE 原文。商业部署前必须逐一核实。
- 【NLF 许可组合存疑】代码标 MIT 但权重标 noncommercial research use，这种组合的法律边界（能否用 MIT 代码配自训权重商用）需项目方或法务确认。
- 【Hunyuan3D 2.1 之外的腾讯系许可】HY-Motion-1.0、Hunyuan3D-Buffalo1.0、R-DMesh 是否沿用同一套「>100万 MAU 需授权 + 排除 EU/UK/韩国 + 禁止用输出训练其他 AI 模型」的社区许可，未核实。若沿用，HY-Motion 的动作输出用于训练你自己的模型会踩线。
- 【成本估算全部为推算】recommendation 中的人时（2-6 人时/主角、20-45 分钟/镜等）、摊销阈值（6-8 镜、90 秒）、分层比例（40-60% 走 3D 底稿）均为我的工程估算，无任何厂商数据或第三方测评支撑，应作为起始假设而非结论。
- 【「灰模要打光」的结论是推理而非实测】该结论由 Champ 使用 depth+normal+semantic+dwpose 四路条件这一事实、以及 Wan2.2-Fun-Control 支持 Canny/MLSD 这一事实推导得出，未找到任何直接比较「flat 灰模 vs 打光灰模」贴皮质量的公开消融实验。建议上产线前自行做一组 A/B。
- 【Animate-X / MimicMotion / Champ 已过时】这是基于分辨率、帧数与基座代际（SD1.5/SD2.1/SVD vs Wan2.2 DiT）的判断，非检索到的对比评测结论。
- 【Wan2.2-Fun-Control 与 VACE-Fun 的实际显存】VideoX-Fun 文档只给出 64.0GB 权重体积，未给出推理显存需求；「社区 FP8/GGUF 可下探到 24GB 级」是基于 Wan2.2-Animate 有 FP8 版本的类推，未核实 Fun-Control 是否有同等量化版本。
- 【ComfyUI-WanVideoWrapper 的许可证】未核实仓库 LICENSE，作为产线核心调度器，商用前必须确认。
- 【Warudo Pro 定价】文档页未列出企业版具体价格，需询价。VTube Studio 的 Steam 具体售价同样未获取（官网只说是一次性付费）。

### 事实核查修正
- [WRONG] [WHAM] README 的 TODO 仍列有数据预处理与训练实现待完善项
  → WHAM README 的 TODO 区块（第 96-104 行）实际为：`- [ ] Data preprocessing`（未完成）、`- [x] Training implementation`（已完成）、`- [x] Colab demo release`（已完成）、`- [x] Demo for custom videos`（已完成）。只有「数据预处理」一项未打勾，训练实现已经标记完成并随仓库发布。把训练实现说成「待完善」会误导选型——WHAM 是可自训练的，缺的只是预处理文档。 https://raw.githubusercontent.com/yohanshin/WHAM/main/README.md
- [WRONG] [研究员自标存疑项] GVHMR、SMPLest-X 的仓库含 LICENSE 文件但类型未确认，商用前需核实
  → 已核实，两者都是禁止商用的许可，不是待确认状态：(1) GVHMR 的 LICENSE 是浙大 CAD&CG 自定义学术许可，原文明确写「Permission to use, copy, modify and distribute this software and its documentation for educational, research and non-profit purposes only. Any modification based on this work must be open-source and prohibited f https://raw.githubusercontent.com/zju3dv/GVHMR/main/LICENSE
- [UNVERIFIABLE] [HY-Motion-1.0] GitHub 2558 stars，2026-07-18 仍在更新
  → 量级方向正确但精确数字无法坐实：仓库页当前显示 2.6k stars、36 commits，commits/master 页显示最近提交确实在 2026 年 7 月（内容为 merge PR、bare except 修复、README/显存说明调整等维护性提交）。但 2558 这个精确星数与 2026-07-18 这个精确日期本次未能从 GitHub API 取到（api.github.com 对本环境 403 / rate limit），只能确认「约 2.6k stars、2026 年 7 月仍有维护性提交」。另注意：默认分支是 master 不是 m https://github.com/Tencent-Hunyuan/HY-Motion-1.0/commits/master
- [WRONG] [研究员自标存疑项] HY-Motion-1.0 是否沿用腾讯系社区许可（>100万 MAU 需授权 + 排除 EU/UK/韩国 + 禁止用输出训练其他 AI 模型）未核实
  → 已核实为「沿用」，不是未知：仓库根目录 License.txt 即《Tencent HY-MOTION 1.0 Community License Agreement》，三条限制全部存在——第 4 节 MAU 超过 1 million 须向腾讯申请许可；前言与 1(l) 明确「THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA」；第 5(b) 禁止用 Works 或任何 Output「to improve any other https://raw.githubusercontent.com/Tencent-Hunyuan/HY-Motion-1.0/master/License.txt

### 遗漏补充
- TRAM（Global Trajectory and Motion of 3D Humans from in-the-wild Videos, ECCV 2024, arXiv 2403.17346）——与 WHAM/GVHMR 同赛道的世界坐标系人体+相机联合恢复方法，调研三选手里完全缺席，做对比基线时应补上
- CameraHMR / TokenHMR / PromptHMR（CVPR-3DV 2025 一系）——单帧 SMPL 回归的新一代方法，直接影响底稿的体型与相机内参精度，比继续用 HMR2.0b 更值得评测
- Multi-HMR（ECCV 2024, Naver Labs, 单次前向多人全身 SMPL-X 含手脸）——多人同框镜头的底稿方案，当前调研的 GVHMR/WHAM 路线对多人场景支持很弱
- Meshcapade / SMPL-X Blender add-on 与 SMPL-to-FBX 转换链——从 SMPL 参数落到 Blender/UE 可用骨骼的关键一环，调研只覆盖了估计端没覆盖转换端
- Mixamo（Adobe，免费自动绑定 + 动作库）与 AccuRIG / ActorCore（Reallusion，免费自动绑定）——作为 Auto-Rig Pro（价格与 Blender 5.x 兼容性未能核实）的零成本替代，应作为 fallback 写入
- Cascadeur（物理感知 + AI 自动补姿的关键帧工具）——在「动捕底稿不够用、需要人工二次雕」的环节比纯 Blender 手 K 效率高得多
- Move.ai / Rokoko Video / DeepMotion / Plask 等商业单目动捕 SaaS——按分钟计费、输出直接是 FBX，是开源估计器在「人时成本」维度的真正对照组，调研的成本推算缺这一侧参照
- MoMask / MDM / MotionLCM / OmniControl 等开源文生动作模型——HY-Motion-1.0 是唯一被评估的文生动作方案，而它带腾讯社区许可的商用与训练限制，需要 MIT/Apache 侧的替代
- MegaSaM / MonST3R / VGGT 等新一代动态场景相机轨迹估计——GVHMR 用 SimpleVO、WHAM 用 DPVO/DROID-SLAM，这批新方法在动态前景大面积遮挡时的相机轨迹更稳，直接决定世界坐标底稿是否漂
- Unreal Engine MetaHuman Animator + Live Link Face——面部底稿链路完全缺席（调研只讨论了身体动捕），而混合产线里脸部往往是贴皮质量的瓶颈
- Grease Pencil / Freestyle / Blender Compositor 的分层输出（depth / normal / cryptomatte / ID pass）——作为喂给 ControlNet-类条件的标准产线输出方式，比「渲个灰模」更可控，且能直接回应研究员未实测的「灰模要不要打光」问题
- Viggle AI 与 Runway Act-Two 等「3D/视频驱动角色」的闭源对照——判断自建 3D 底稿产线是否值得时，必须有这类一键方案作为 make-or-buy 的分母
