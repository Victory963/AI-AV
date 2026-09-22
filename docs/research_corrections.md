# 事实核查修正（推翻或存疑的说法）

## 闭源 API 能力边界
- **[WRONG]** Seedance 2.0 标准版分辨率 480p/720p/1080p，最高 2K（2048×1080）
  → 官方《模型列表》对 doubao-seedance-2-0-260128 明确列出：480p（8bit）/720p（8bit）/1080p（8bit）/4k（10bit）。根本没有「2K / 2048×1080」这一档。4k 档限流极严：企业与个人均为 最大 RPM 15、最大并发 1；非 4k 档为 企业 RPM 600/并发 10、个人 RPM 180/并发 3。价格上 4k 也单列（无视频输入 26 元/百万 token，含视频 
- **[WRONG]** Seedance 2.5 可能原生 4K 输出（CometAPI 口径），暂以 1080p 为准
  → 可以定论：doubao-seedance-2-5-260628 官方规格为 480p（8bit）/720p（8bit）/1080p（10bit），无 4k。反而是 Seedance 2.0 才有 4k（10bit）。CometAPI 的「native 4K」说法应判为错误。
- **[WRONG]** Seedance 2.0 参考位上限：12 个文件 = 9 图 + 3 视频 + 3 音频
  → 数字自相矛盾（9+3+3=15）。官方《Doubao Seedance 2.5 教程》规格对比表写明：Seedance 2.0 系列「参考素材数量上限 15（9张图+3个视频+3个音频）」，Seedance 2.5 为「50（30张图+10个视频+10个音频）」。《Seedance 2.0 系列教程》同样写「图片：0~9 张；视频：0~3 个；音频：0~3 个」，并注明不支持「文本+音频」与「纯音频」输入（纯音频参考是 2.5 新增能力
- **[WRONG]** 参考视频 2–15s 且 ≤50MB；音频 MP3 ≤15MB
  → 官方创建任务 API：单个参考视频 ≤200 MB（不是 50MB），时长 Seedance 2.0 系列 [2,15]s、最多 3 个且总时长 ≤15s；视频还支持 480p/720p/1080p/4k、帧率 [24,60]、格式 mp4/mov。音频格式为 wav 与 mp3 两种（不止 MP3），单个 ≤15 MB，请求体整体 ≤64 MB。单张图片 <30 MB 这条正确。
- **[UNVERIFIABLE]** 首尾帧模式与多模态参考模式互斥（同时传会报参数错误）
  → 「互斥」部分属实：官方原文为「图生视频-首帧、图生视频-首尾帧、全模态参考生视频（包括参考图、视频、音频）为 3 种互斥场景，不可混用」。但紧接的一句是「全模态参考生视频优先使用图生视频-首尾帧」，属于优先级/降级语义，并非文档明示会「报参数错误」。该报错行为未在一手文档核实。
- **[UNVERIFIABLE]** Seedance 2.0 模型发布 2026-02（先上火山方舟体验中心），API 全面开放 2026-04-14（科技日报报道）
  → 一手渠道核实不到这两个日期。官方 Model ID 为 doubao-seedance-2-0-260128，快照日指向 2026-01-28；方舟《模型发布公告》页中 doubao-seedance-2-0-260128 与 doubao-seedance-2-0-fast-260128 当前标记为「更新」而非「新发布」，月份分区（202601/202602/202604…）无法从页面结构稳定归属。建议降级为「约 2026 年初」，不
- **[UNVERIFIABLE]** Seedance 2.5 火山方舟 API 全面开放：2026-08-07
  → 官方口径只能确认两点：(1) 发布日 2026-07-31，当日「API 服务也将在近期上线火山方舟」，BytePlus 英文博客写 ModelArk API「coming soon」；(2) 当前 API 文档写「Seedance 2.5 已全面公开」。8-07 这个具体日期无一手来源。可确证的邻近官方日期是折扣活动起止：Seedance 2.5（1080p，72 折）2026-08-14 14:00 至 2026-09-17 14:
- **[OUTDATED]** Seedance 2.5 计费：42 元/百万 token（有视频输入）、70 元/百万 token（无视频输入）
  → 只说对了一半档位。官方《模型价格》doubao-seedance-2.5「按输出视频分辨率和输入是否包含视频区分定价」：480p/720p → 无视频 70.00、含视频 42.00；1080p → 无视频 原价 77.00、含视频 原价 46.00（且 2026-08-14~09-17 1080p 限时 72 折）。引用时必须带分辨率档位。
- **[UNVERIFIABLE]** 折算约：无视频输入 ≈ $0.10/秒@480p、$0.23/秒@720p、$0.53/秒@1080p；有视频输入 ≈ $0.06/$0.14/$0.32
  → 官方不按秒计价，且明确「视频价格 = token 单价 × token 用量」，同时对 Seedance 2.0 系列与 2.5「当输入包含视频时存在最低 token 用量限制」（低于则按最低量计），最低量随分辨率/宽高比/输出时长变化。因此任何固定的 $/秒 折算都只是估算，官方提供的是《Seedance 2.0 系列价格计算器》《Seedance 2.5 系列价格计算器》与最低 token 用量表。
- **[WRONG]** Seedance 2.5 支持 10+ 语种对白
  → 官方文档说的是「提示词语言支持」，不是对白语种。原文：所有模型支持中英文提示词；Seedance 2.5 额外支持西班牙语、印度尼西亚语、葡萄牙语、日语、马来语、泰语、阿拉伯语、越南语、韩语（合计 11 种）；Seedance 2.0 系列额外支持西班牙语、印尼语、葡萄牙语、日语（合计 6 种）。把它写成「对白语种」是换概念。
- **[UNVERIFIABLE]** Seedance 2.0 视频延长：一次可延长 4–15 秒，从输入视频最后一帧开始，保留原音频，返回一条合并后的完整视频
  → 可确认的只有：方舟模型列表中 2.0 / 2.0 fast / 2.0 mini 均标注「延长视频」能力，且 2.0 系列 duration 取值 [4,15] 或 -1。「从最后一帧开始」「保留原音频」「返回合并后的完整视频」三点均无一手来源；而且官方对 2.5 明确写「对原视频进行向前或向后延长」，说明「从最后一帧开始」并非通则。
- **[UNVERIFIABLE]** 提示词写法：编辑/延长任务直接写 `@video N`，不要写 `参考 @video N`，否则会被判成参考任务
  → 官方没有这条「不要写参考」的表述。官方给出的是显式参数 omni_reference_task_type（auto/reference/edit/extend）来前置声明子任务类型，并给出示例提示词：延长「向后延长 @video1，@image1 的角色从天而降…」「续写 @video1 前 5s…」；编辑「@video1 中加一些小动物」「把 @video1 的人物修改为 @image1」「删掉 @video1 的背景音乐」。误判时的
- **[WRONG]** BytePlus 官方 API 文档列出的 content 输入类型含「sample task ID」，即可用先前生成任务的 ID 作为续写锚点
  → 概念误解。该输入类型对应火山方舟的 content.type = draft_task（「样片任务 ID」）。它属于「样片模式」：先用 draft=true 以 480p 生成低成本 Draft 预览视频，确认后再用 Draft 的 task id 生成正式视频，平台自动复用 model/text/image_url/generate_audio/seed/ratio/duration/camera_fixed。且该能力官方标注「模型支
- **[WRONG]** 输入真人脸直接拒：Seedance 2.0 系列不接受含真实人脸的参考图/视频
  → 对火山方舟而言过度概括。真人脸拦截确实存在（错误码 InputImageSensitiveContentDetected.PrivacyInformation / InputVideoSensitiveContentDetected.PrivacyInformation，报错文案正是「The request failed because the input image/video may contain real person」），但官
- **[WRONG]** 常见错误名含 InputTextSensitive
  → 错误码全名为 InputTextSensitiveContentDetected（另有 .PolicyViolation 变体）。同组另三个写法正确：OutputVideoSensitiveContentDetected、OutputAudioSensitiveContentDetected、OutputVideoSensitiveContentDetected.PolicyViolation。官方错误码表还包含 InputImage/
- **[UNVERIFIABLE]** 输出审核发生在生成之后：模型已经算完，中间帧出现歧义画面也会被拦
  → 「存在生成后输出审核」有旁证（Output*SensitiveContentDetected 系列错误码确实存在，且计费文档写「因审核等原因导致生成失败的，不收取费用」——失败不计费这条 CONFIRMED）。但「中间帧歧义画面也会被拦」这一机制细节无官方表述。另注意 Seedance 2.5 特定任务类型「需排队等待到任务被消费时才会返回报错信息」。
- **[UNVERIFIABLE]** BytePlus 侧默认开启 content pre-filter，明确阻止生成公众人物相似形象以防深伪
  → 未在 BytePlus/火山一手文档核实到这段表述。可核实的最接近条目是火山方舟错误码表中的 OutputImageSensitiveContentDetected.DeepFake，以及 ContentSecurityDetection（CSD）审核链路会在报错中返回 CSDRequestId/Label/SubLabel。
- **[WRONG]** Mini：不提供标准版的「通用参考」12 文件多模态控制
  → 官方《Seedance 2.0 系列教程》原文：「三者支持的功能基本一致，主要区别在于生成品质与成本的取舍」。模型列表中 doubao-seedance-2-0-mini-260615 同样标注全模态参考生视频 / 参考生视频 / 编辑视频 / 延长视频 / 首尾帧生视频 / 首帧生视频 / 文生视频，参考素材上限同为 15（9图+3视频+3音频）。Mini 的真实差异只在分辨率（仅 480p/720p）与价格。「封顶 720p」「约为
- **[WRONG]** 2026-08-07～09-07 促销：2.0 mini 6 折 ≈ 0.2 元/秒@720P；2.0 fast 75 折 ≈ 0.6 元/秒@720P
  → 折扣力度与活动期都错。官方：Seedance 2.0 mini 与 2.0 fast 的活动期均为北京时间 2026-08-07 14:00 至 2026-10-07 14:00（不是 09-07）；mini 为 480p/720p 按刊例价 4 折（不是 6 折），fast 为 75 折（这条对）。两者优惠仅面向企业用户，且各自有折后 token 用量上限（系统自 2026-09-18 14:00 起累计），达上限后恢复原价。刊例价：
- **[WRONG]** 官方资源包活动明确「仅适用 Doubao-Seedance-2.0，不含 fast、mini 模型」
  → 官方《Seedance 2.0-2.5 模型资源包使用规则》列出四种资源包：Seedance 2.5、Seedance 2.0、Seedance 2.0 Fast、Seedance 2.0 Mini，且四者任一有余量均可开通模型，资源包「仅支持抵扣对应的模型」。真实限制是另一回事：资源包属预付费商品，不参与后付费限时折扣活动；有效期 90 天；7 天内未使用可全额退款；抵扣以单价较低的「含视频输入」场景为 1:1 基准折算。
- **[WRONG]** Seedance 2.0 Pro 档存在且支持 2K，仅模型 ID 字符串 doubao-seedance-2-0-pro-260215 不可信
  → Pro 档本身就不存在。官方教程原文：「Seedance 2.0 系列模型目前包括 Doubao Seedance 2.0、Doubao Seedance 2.0 Fast 和 Doubao Seedance 2.0 Mini」，模型列表中也只有 doubao-seedance-2-0-260128 / -fast-260128 / -mini-260615 三个。搜索引擎 AI 摘要里的 260215 几乎可以确定是把文本大模型 do
- **[WRONG]** 火山方舟对 Seedance 视频任务的并发 / RPM 未公开数值表，只能登录控制台查看
  → 官方《模型列表》公开了具体数值。doubao-seedance-2-5-260628 与 doubao-seedance-2-0-mini-260615、-fast-260128：最大 RPM 企业 600 / 个人 180，最大并发 企业 10 / 个人 3。doubao-seedance-2-0-260128：非 4k 同上；4k 分辨率单独限流 RPM 15、并发 1（企业与个人相同）。doubao-seedance-1-0-pr
- **[OUTDATED]** doubao-seedance-1-0-lite-t2v-250428 / doubao-seedance-1-0-lite-i2v-250428 为现行 lite 档
  → 现行火山方舟《模型列表》中已检索不到任何 seedance-1-0-lite 条目。当前在列的 Seedance 1.x 只有 doubao-seedance-1-0-pro-250528、doubao-seedance-1-0-pro-fast-251015、doubao-seedance-1-5-pro-251215（且已标「即将下线」）。写成现行可用模型会误导。
- **[OUTDATED]** doubao-seedance-1-0-pro-fast：提速版本
  → 完整 Model ID 为 doubao-seedance-1-0-pro-fast-251015。规格：480p/720p/1080p，24fps，时长 [2,12]s，支持首帧生视频与文生视频（不支持首尾帧），独有 frames / seed / camera_fixed 参数。
- 遗漏补充：omni_reference_task_type（auto / reference / edit / extend）：S, output_format = mp4 / mov（Seedance 2.5 专有）：mov 为 H.264 + yuv, tools 参数中的 web_search 联网搜索工具：Seedance 2.5 与 Seedance 2.0 系列均, priority 执行优先级 [0,9]：同一 Endpoint 队列内插队，默认 FIFO，不打断 running 任, service_tier = flex 离线推理模式：价格为在线推理的 50%、TPD 配额更高——但官方明确「Seed, 素材 & 虚拟人像库 + 素材 ID 输入：image_url / video_url / audio_url 除公网 , 版权 IP 视频生成：官方支持基于特定授权 IP（如周星驰电影《功夫女足》版权）生成视频，费用 = 视频生成费用 × 1, draft 样片模式（draft=true）+ content.type=draft_task 两步式工作流：先出 48, frames 参数（取值 [29,289] 且必须满足 25+4n）：用于生成小数秒视频，优先级高于 duration，, Seedance 1.5 pro（doubao-seedance-1-5-pro-251215，已标『即将下线』）：时长, 模型开通门槛：调用 Seedance 2.5 / 2.0 系列前必须满足任一条件——账户余额 > 200 元、或购买 2, 产物留存与转存：video_url 有效期 24 小时，且 Seedance 2.5 生成的视频 URL 下载次数上限 

## 开源基座模型
- **[WRONG]** [LTX-2.5 (22B)] LTX-2.5 主模型 2026-01-06 发布；HF Lightricks 组织共 66 个模型
  → 版本张冠李戴。HF API 显示 Lightricks/LTX-2.5 的 createdAt = 2026-07-23T07:55:24Z（lastModified 2026-09-01），LTX-2.5-Diffusers createdAt = 2026-07-26。2026-01-06 / arXiv 2601.03233 对应的是 LTX-2：GitHub Lightricks/LTX-2 仓库 created_at = 20
- **[WRONG]** [MiniMax H3] BF16 原始 checkpoint 123.6GB；社区量化 pruned INT8 约 21GB / 42.5GB 多个版本
  → 123.6GB 在仓库里找不到对应物。按 HF blobs API 实测 MiniMaxAI/MiniMax-H3：单个 transformer 目录 = 66.28 GB（33B × BF16，数学自洽）；text_encoder = 66.73 GB；video_vae = 10.42 GB；audio_vae = 0.61 GB；即一个完整可跑 checkpoint（如 FL2VA）≈ 144.0 GB，整仓（FL2VA + Re
- **[UNVERIFIABLE]** [MiniMax H3] Artificial Analysis Text-to-Video Arena：Elo 1220，全部开源权重模型第一；总榜第 4，仅次于 Gemini Omni Flash(1233)/Wan 3.0(1229)/MiniMax H
  → 无法取得一手来源。artificialanalysis.ai/text-to-video/arena 308 重定向到 /video/arena，该页 395KB HTML 中不含任何 Elo 数值或视频模型榜单（全部由 JS 客户端拉取），/video/arena/text-to-video、/text-to-video/arena/leaderboard、/api/v2/data/text-to-video/arena 等端点均返回
- **[UNVERIFIABLE]** [MiniMax H3] Image-to-Video 开源权重榜同样第一：带音频 1181，不带音频 1354
  → 同上，AA 榜单数据无法从页面取得。另外这两个数字内部不自洽：同一模型两个模式差 173 分（不带音频 1354 反而比 T2V 榜首 1220 还高 134 分），而 AA 的 Arena Elo 是同一池内配对比较，跨模式出现这种量级的跳变需要额外解释。建议直接从 AA 官方导出或截图取数，不要二手引用。
- **[UNVERIFIABLE]** [LTX-2.5] Artificial Analysis T2V Arena：LTX-2.5 Fast Elo 1055 / Pro 1053，开源权重第 2（落后 H3 约 165 分）
  → 同上，AA 榜单不可取数。另需注意「LTX-2.5 Fast / Pro」是 Lightricks 托管 API 的档位命名，与开放权重档（LTX-2.5 dev / distilled，HF 上 ltx-2.5-22b-dev / ltx-2.5-22b-distilled）不是同一实体——拿 API 档位的 Elo 去代表开放权重的能力，本身就是「把闭源档位说成开源」的典型口径错误，即便分数拿到了也要分开标注。
- **[UNVERIFIABLE]** [Wan 2.5/2.6/2.7/3.0] Wan 3.0 在 Artificial Analysis T2V Arena Elo 1229，总榜第 2（高于 Seedance 2.0 720p 的 1210）
  → Elo 数值同样无法核实（见上）。不过「Wan 3.0 是闭源 API-only」这半句已独立证实：阿里云百炼模型列表里存在 wan3.0-video，而 HF Wan-AI 组织下最新开放权重是 Wan2.2-Animate-2-14B-Diffusers（2026-08-13），无任何 3.0 权重。
- **[UNVERIFIABLE]** [Wan 2.5/2.6/2.7/3.0] Wan 3.0：2026-08-06 公测、2026-08-24 全量上线，仍是闭源 API；支持 30 秒单镜、文档到视频
  → 「闭源 API」部分 CONFIRMED（阿里云百炼列出 wan3.0-video；HF/GitHub/ModelScope 无权重）。但 2026-08-06 公测、2026-08-24 全量、30 秒单镜、文档到视频这四项都没能从阿里云官方文档核实到。反证是：阿里云文生视频 API 参考页当前主文档给出的模型是 wan2.7-t2v / wan2.7-t2v-2026-06-12，时长参数取值范围明确写着 [2, 15] 秒整数、默
- **[UNVERIFIABLE]** [Wan 2.5/2.6/2.7/3.0] Wan 2.6、Wan 2.7 同样闭源 API-only
  → Wan 2.7 部分 CONFIRMED：阿里云百炼文生视频 API 文档列出 wan2.7-t2v 与快照版 wan2.7-t2v-2026-06-12，HF Wan-AI 组织下无 2.7 权重，确为 API-only。Wan 2.6 本次在阿里云文档中没有检索到任何对应模型名（列表里只出现 wan2.7-image-pro 与 wan3.0-video），Wan 2.6 是否真实存在过这一公开版本号未能证实，建议单独确认后再写入，
- **[WRONG]** [MiniMax H3 LoRA 生态] Civitai 已有独立模型类别，官方 H3 训练赛（2 Million Buzz 奖池）2026-09 进行中，上线数周社区已发布 670+ 个基于 H3 的模型
  → 「独立模型类别」CONFIRMED：Civitai API 的 baseModels 枚举里确实有官方分类 "MiniMax H3"。但 670+ 这个量级复现不出来：用 baseModels=MiniMax%20H3 游标翻页穷举，总计 319 个已发布模型；用 tag=minimax h3 穷举只有 123 个。即便加上其他 H3 相关 baseModel 变体也远达不到 670。「2 Million Buzz 官方训练赛」本次未能
- **[WRONG]** [MiniMax H3 LoRA 生态] 对比：Civitai wan2.2 标签约 368 个模型（另有社区整理的 475 个 Wan2.2 LoRA 合集，2026-03）
  → 368 不对，且没说清口径。实测：tag=wan2.2 穷举 = 253 个；baseModels="Wan Video 2.2 T2V-A14B" 穷举 = 523 个。两个口径都不等于 368。由于 Civitai 的 tag 与 baseModel 是两套体系（同一模型可能只挂其中之一），任何「H3 vs Wan2.2 生态规模对比」都必须用同一口径，否则结论方向都可能反转：按 baseModel 口径是 Wan2.2 (523)
- **[WRONG]** [MiniMax H3 LoRA 生态] Turbo LoRA（lightx2v / drbaph 等多个版本）：把 20 步压到 4–6 步出可用画面，文件 <1GB
  → 仓库存在性 CONFIRMED（lightx2v/Minimax-h3-Turbo 1,531,171 下载、drbaph/MiniMax-H3-Turbo-Lora-ComfyUI、larryvrh/MiniMax-H3-Turbo-Lora 各 20 万级下载），但「文件 <1GB」和「4–6 步」两处都不准。实测文件体积：lightx2v 的 ComfyUI bf16 版 minimax_h3_fl2v_turbo_4step_v
- **[WRONG]** [MiniMax H3 LoRA 生态] 已有融合微调产物如 Minimax-h3_Singularity ……以 ComfyUI diffusion model 形式加载，与官方 INT8 同槽位
  → 模型本身 CONFIRMED（WarmBloodAban/Minimax-h3_Singularity，217,900 下载、499 likes，另有 Abiray/MiniMax-H3-Singularity-GGUF、TenStrip/Minimax-h3_Singularity-Lora 衍生）。错在「官方 INT8」：MiniMaxAI 组织下只发布了 MiniMax-H3 一个仓库，498.47 GB 全部为 BF16 saf
- **[UNVERIFIABLE]** [LTX-2.5] 实测速度：RTX 5090(32GB) 出 4 秒 720p 约 25 秒；RTX 4090 跑 5 秒片实测显存 22.67 GiB；2×GB200 上 6.8 秒是官方 headline
  → GitHub Lightricks/LTX-2 官方仓库与 HF LTX-2.5 模型卡中均未给出任何 RTX 5090 / RTX 4090 / GB200 的速度或显存基准，只有量化档位（fp8-cast、fp8-scaled-mm for Hopper+、NVFP4、int8+convrot）与「降低显存占用」的定性描述。这三个数字应标注为第三方/社区实测并给出具体来源，否则不能作为官方 headline 引用。
- **[UNVERIFIABLE]** [LTX-2.5] stage1 默认 544×960，可上 1920×1088 或 4K(3840×2176) 超分
  → 4K = 3840×2176（注意不是 2160）以及 --spatial-upscalings 2 的用法在官方仓库中可证实；num_frames % 8 == 1、宽高整除 32、上限 121 帧也可证实。但「stage1 默认 544×960」与 Lightricks/LTX-2 仓库 README 给出的默认 1024×1536 @24fps 冲突，两处默认值口径不一致（可能分属 dev / distilled 管线或不同版本文
- **[WRONG]** [SkyReels-V3] 三个变体：R2V-14B（参考图到视频）、V2V-14B（视频到视频）、A2V-19B（音频到视频）
  → 数量、参数量、日期都对（GitHub SkyworkAI/SkyReels-V3 News 明确写 2026-01-29 放出推理代码与权重；HF Skywork/SkyReels-V3-A2V-19B / -R2V-14B / -V2V-14B 均 2026-01-28 更新，A2V-19B createdAt 2026-01-19），但两个变体的能力描述错位：官方自述是 Reference-to-Video 14B-720P、Vid
- **[WRONG]** [调研员存疑项] 用户给出的 Seedance 2.0 多模态参考上限「最多 9 图 + 3 视频 + 3 音频」未能核实
  → 这组数字很可能被错归到了 Seedance 头上——它其实是 MiniMax H3 的规格。MiniMaxAI/MiniMax-H3 模型卡对 Ref2VA（Omni-Reference）checkpoint 的原文描述就是「supports up to 9 images, 3 video clips, 3 audio clips, or 12 mixed files」，与「9 图 + 3 视频 + 3 音频」逐项吻合。所以这条不是「S
- 遗漏补充：tencent/HunyuanVideo-1.5（2025-11-18 开源，HF 1036 likes）——调研员只在, sand-ai/MAGI-2-preview（2026-08-03，Apache 2.0，306.72GB 真实权重，M, meituan-longcat/LongCat-Video（2025-10，MIT，HF 568 likes）——MIT, Wan-AI/Wan-Dancer-14B（2026-07-17，85.67GB，global_model + loca, Wan-AI/Wan2.2-Animate-2-14B（2026-08-09，82.54GB，同时提供 wan_anim, Lightricks/LTX-2.3 全家桶——HF 1,126,972 下载 / 1899 likes，官方还单独发了, FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree（27, 真实部署入口被漏：Comfy-Org/MiniMax-H3 的下载量是 20,277,946 次，比官方 MiniMax, zai-org/SCAIL-2（2026-06-09，MIT，pose-driven 角色动画）——与 Wan2.2-A, 单卡可跑的 Wan2.2 量化/蒸馏路线整条缺失：Phr00t/WAN2.2-14B-Rapid-AllInOne（16, Skywork 组织页已挂出 SkyReels-V4（Multi-modal Video-Audio Generatio, 对照组基座缺失：stepfun-ai/stepvideo-t2v、nvidia/Cosmos-1.0-Diffusion

## 长视频续写
- **[WRONG]** SVI 2.0（2025-12-04）/ SVI 2.0 Pro（2025-12-26）基座换成 Wan 2.2 I2V-A14B（HIGH/LOW 双 LoRA）
  → 日期正确，但基座说法错误。官方 README 的 News 写的是「SVI-2.0 released for Wan 2.1 and Wan 2.2」——不是「换成」。主分支（main）基座仍是 Wan 2.1 I2V-14B，官方发布的权重文件名直接写明：vita-video-gen/svi-model → version-2.0/SVI_Wan2.1-I2V-14B_lora_v2.0.safetensors。Wan 2.2 实现放
- **[WRONG]** SVI 官方 demo：SVI-Shot 20 分钟压测、Tom&Jerry 10 分钟、SVI-Talk 10 分钟
  → Tom&Jerry 的数字记错了。README 里 SVI-Tom 条目写的是「This will never drift or forget in our 20 min test」，而对外放出的 demo 视频是「8-minute crazy Tom & Jerry video made with SVI-Tom」——既不是 10 分钟。SVI-Shot 的 20 min test 与 SVI-Talk 的 10 min test 确
- **[OUTDATED]** SVI 训练环境 A100 80GB / CUDA 12.0 / Torch 2.5.0
  → A100 80G + CUDA 12.0 正确，Torch 版本过时。README 现写的测试环境是 PyTorch 2.8.0；torch==2.5.0 是安装脚本自动回装的旧版本，仓库同时声明兼容 torch 2.4.1。把 2.5.0 当作「官方训练环境」会误导环境搭建。
- **[UNVERIFIABLE]** SVI 只训 LoRA，约 1k 样本即可
  → 「只训 LoRA」确认无误。但「1k 样本」不是 SVI 的通用结论：README 中该数字只出现在 Wan 2.2 Animate 的微调语境（「tuning with only 1k samples is sufficient to unlock infinite-length generation」），对 SVI-Shot / Film / Talk 等 1.0 系列，README 只有定性表述「very little train
- **[UNVERIFIABLE]** SVI 社区实测稳定区间：480p+81帧/clip+24fps+每clip换seed → 1 分钟无可见色偏；40 秒无色偏已被多人复现；已知崩点 121 帧/clip 必然色偏、必须 fp16、量化/蒸馏掉质量、LightX2V 开太猛会慢动作
  → 其中只有三条能在官方 README 找到一手依据：(a)「Use different seeds for different clips, which is very important!」(b) 建议 480p 为最优分辨率 (c) 建议少用 LightX2V。其余全部无一手来源：81 帧/clip、24fps、「1 分钟无可见色偏」「40 秒被多人复现」「121 帧必然色偏」「必须 fp16」「量化掉质量」均未出现在 README、
- **[WRONG]** Helios 权重 2026-07-28 仍在更新
  → 措辞把 README 改动说成了权重更新。HF BestWishYsh/Helios-Base 的 commit 历史显示：2026-07-28 的最后一次提交是「Update README.md」；权重（safetensors）批量上传集中在 2026-02 下旬（初始 commit cb75bfd，2026-02-23），3 月主要是 README 更新与 2026-03-15 的 modular 功能。所以权重自 2026-02/
- **[UNVERIFIABLE]** Helios 第三方 H200 实测吞吐几乎与长度无关：240 帧 24s / 480 帧 42s / 960 帧 82s（约 11 FPS 恒定）
  → 找不到任何一手来源。HF 模型页 discussions 区只有一条 2026 年初的「gguf version」请求，无 H200 benchmark 讨论；官方 README 只给 H100 19.5 FPS 与昇腾 NPU ~10 FPS（这两项已核实为 CONFIRMED）。这组 240/480/960 帧的数字无法追溯到具体作者或复现脚本，不能作为「吞吐与长度无关」的论据——恰恰这是该模型最关键的卖点，需要自己压测。
- **[WRONG]** Helios 支持最高 4K（研究员已自标存疑）
  → 存疑判断成立，可以定性了：HF 官方模型卡的标准输出分辨率明确是 640×384，4K 的说法只出现在第三方/社区的「消费级 PC 也能跑」类文档里，属于第三方实现而非官方规格。按 640×384 原生 + 外接超分规划，不要按原生 4K 规划。
- **[UNVERIFIABLE]** LongCat-Video-Avatar 1.5：50 步 → 8 步蒸馏，约 15× 提速，10 秒视频约 1 分钟出片；跳帧率 0.8%，唇音误差率 29.8%（对比组内最低）；帧级 GRPO 对齐；共享基座 + 多 LoRA 适配器降低显存
  → HF meituan-longcat/LongCat-Video-Avatar-1.5 模型卡（最后更新 2026-06-04）只确认三件事：DMD2-based step distillation 的「8-Step Inference」、支持 480P 与 720P、MIT License。卡上没有 50→8 的原步数、没有 15× 提速倍数、没有 10 秒/1 分钟的出片时间、没有 0.8% 跳帧率与 29.8% 唇音误差率、没有 
- **[UNVERIFIABLE]** LongCat-Video-Avatar 权重 MIT，GitHub + HF + ModelScope 三处同步
  → MIT 对 LongCat-Video 主仓与 Avatar-1.5 模型卡都已确认。但「GitHub 三处同步」对 Avatar 不成立：github.com/meituan-longcat/LongCat-Video-Avatar 与 .../LongCat-Video-Avatar-1.5 均返回 404，Avatar 的发布信息只以 News 条目形式挂在 LongCat-Video 主仓 README 里，独立代码仓未找到。若
- **[WRONG]** SkyReels-V3 的 --low_vram 走 FP8 量化；官方未公布具体 VRAM 阈值
  → 前半句对，后半句错。官方 README 原文就写了阈值：「For GPUs with lower VRAM (e.g., under 24GB), use these options:」，随后才是 --low_vram（FP8 weight-only quantization + block offload）与降分辨率到 540P/480P。24GB 是官方给出的门限，不是未公布。
- **[WRONG]** SkyReels-V3 明确的时长上限：R2V 5 秒 / V2V 续写 30 秒 / A2V 数字人 200 秒
  → V2V 一档被合并成了单一数字，丢了关键约束。README 把视频扩展拆成两种模式：Single-shot Extension 为 5–30 秒，Shot Switching Extension（切镜头续写）上限只有 5 秒。另外 A2V 的 200 秒是「支持最长 200 秒音频输入」，是输入侧口径。做分镜级长片续写时，真正受限的是切镜头那条 5 秒路径。
- **[WRONG]** SkyReels-V2 Diffusion Forcing：arXiv 2504.13074，2025-04-18
  → arXiv ID 正确（SkyReels-V2: Infinite-length Film Generative Model），但日期差一天：v1 提交于 2025-04-17，最后修订 v3 为 2025-04-21。
- **[WRONG]** SkyReels-V2 --addnoise_condition 建议 20–50，用来压跨段不一致
  → 把「推荐值 + 硬上限」误读成了推荐区间。README 的表述是长视频生成推荐设为 20，并警告不要超过 50（超过会牺牲一致性）。写成「建议 20–50」会让人默认取中值 35，而官方推荐点就是 20。
- **[UNVERIFIABLE]** SkyReels-V2 540P: 97–737+ 帧（约 4s–30s+）；720P: 121–1457+ 帧（约 5s–60s+），24fps
  → 下界与分辨率确认：540P=544×960 基线 97 帧，720P=720×1280 基线 121 帧，24fps 正确。但 737 / 1457 这两个上界数字在我核到的 README 文本中未出现，无法定位其一手出处（可能来自某版 DF 长视频示例脚本）。峰值显存那条反而是对的：1.3B@540P ≈14.7GB、14B@540P ≈43.4GB（T2V/I2V）到 ≈51.2GB（DF），VBench 总分 83.9%、SkyR
- 遗漏补充：SkyReels-V3 技术报告 arXiv 2601.17323（2026-01-24 提交，2026-01-29 修, FramePack / FramePack-F1（lllyasviel）——恒定上下文长度的帧打包方案，把长视频生成的计, MAGI-1（Sand AI）——24B autoregressive chunk-by-chunk 视频扩散，Apac, CausVid / Self-Forcing 原始工作——调研只提了 Self-Forcing++，但因果蒸馏 + KV, RIFLEx 与 Ouroboros-Diffusion——位置编码外推 / 无训练长度外推路线，零训练成本把现有 Wa, StreamingT2V / FIFO-Diffusion——免训练的滑窗长视频基线。做色偏与漂移评测时它们是必需的 b, HunyuanVideo-I2V / HunyuanVideo 1.5 的分段续写实践——腾讯这条开源线在国内算力上的部, LongCat-Video 的原生 Video-Continuation 预训练任务本身——调研把 LongCat 当成, 评测侧缺口：VBench-Long / VBench-2.0 与 LOVE-Bench 这类专门面向长视频漂移、身份一致

## 角色一致性
- **[WRONG]** [Wan2.2-Animate-14B] MoE ~27B 总参 / ~14B 激活
  → Wan2.2 官方 README 明确 Animate-14B 是 dense 架构，不是 MoE；MoE（高噪/低噪双专家，27B 总参 / 14B 激活）只适用于 T2V-A14B 与 I2V-A14B。HF 模型卡 metadata 标注为 17B params（Wan2.2-Animate-14B 与 Wan2.2-Animate-14B-Diffusers 均为 17B）。调研员自己在存疑项里怀疑过这条，结论应直接定为错：既不
- **[WRONG]** [Wan2.2-Animate-14B] 720P @ 24fps
  → 分辨率部分对（官方支持 480P & 720P），fps 部分错。官方推理示例导出为 30 fps（--segment_frame_length=77 / --num_inference_steps=20 的 animation 模式示例）。24 FPS 是 Wan2.2-TI2V-5B 的规格（模型库表格里只有 TI2V-5B 写 'supports 720P at 24 FPS'），被串到 Animate 头上了。
- **[WRONG]** [Wan2.2-S2V-14B] HF: Wan-AI/Wan2.2-S2V-14B，2025-09 发布
  → 仓库地址对，日期错。Wan2.2 官方 changelog：「Aug 26, 2025: 🎵 We introduce Wan2.2-S2V-14B」。HF 页面同样记为 2025-08-26 发布（模型文件最后更新 2025-09-17，可能是这个日期被误当成发布日）。对应论文 Wan-S2V arXiv:2508.18621。
- **[WRONG]** [Wan2.2-S2V-14B] 参数量 14B
  → 命名是 14B，但 HF 模型卡 metadata 实测标注为 16B params（BF16）。Wan-AI 组织页列表同样显示 Wan2.2-S2V-14B = 16B。写规格表时要注明「命名 14B / 实际权重 16B」，否则显存与下载量估算会偏低。
- **[UNVERIFIABLE]** [Wan2.2-S2V-14B] 支持长片段扩展与精确唇形编辑
  → 前半句成立：官方模型卡说明不设 --num_clip 时会按音频长度自动扩展生成长视频。后半句「精确唇形编辑」在官方 README 与 HF 模型卡中找不到任何对应表述——S2V 是音频驱动生成（audio-driven cinematic video generation），不是对已有视频做唇形替换/编辑。这条疑似把闭源 Wan2.5 的能力或第三方产品描述套了过来，不要写进选型。
- **[WRONG]** [Wan2.2-VACE-Fun-A14B] 归属于 Wan2.2 官方家族 / 可从 ali-vilab/VACE 获取
  → 归属错。官方权重在 HF 的 alibaba-pai 组织下（阿里 PAI 的 VideoX-Fun 系列，base model 标注为 finetune 自 Wan-AI/Wan2.2-T2V-A14B），不在 Wan-Video/Wan-AI 官方主线，也不在 ali-vilab/VACE 仓库里——拉取 ali-vilab/VACE README 可见它只收录 VACE-Wan2.1-1.3B-Preview、VACE-LTX-V
- **[WRONG]** [Wan2.2-VACE-Fun-A14B] 控制模式：inpainting、pose、depth、reframe
  → 与官方模型卡不符。alibaba-pai 的模型卡列的控制条件是 Canny、Depth、Pose、MLSD、trajectory control（轨迹控制），外加参考图（reference image）注入；训练规格为 81 帧 @ 16 fps，多分辨率 512/768/1024。卡上并未列出 'inpainting' 与 'reframe' 这两个名字（它们是 Wan2.1-VACE 的任务命名 Expand-Anything /
- **[UNVERIFIABLE]** [Wan2.2-VACE-Fun-A14B] 已被 ComfyUI 原生工作流与多家 API 接入
  → ComfyUI 官方文档站没有对应的原生工作流教程页（docs.comfy.org/tutorials/video/wan/wan2-2-fun-vace 返回 404），与 Wan2.2-Animate 形成对比——后者有官方教程页且确认为原生 Mix/Move 两模式工作流。HF 上能查到的是社区转换件（QuantStack 的 GGUF、linoyts 的 diffusers 版、fal 的 FlashPack 等），属于社区生态
- **[UNVERIFIABLE]** [Wan2.1-VACE] 1.3B 档可在 ~8-12GB 显存跑通，14B 档需 40GB+ 或 fp8/block-swap
  → ali-vilab/VACE 的 README 与 UserGuide.md 均未给出任何显存表，只给了 Python 3.10.13 / CUDA 12.4 / PyTorch ≥2.5.1 的环境要求。这组数字来自社区实测，不是官方标注（8.19GB 那个常被引用的数字是 Wan2.1 主仓对 T2V-1.3B 基座的标注，不是 VACE 控制权重）。排产能规划时需自行实测，不要当官方指标引用。
- **[WRONG]** [SkyReels-A2] GitHub SkyworkAI/SkyReels-A2，代码+权重公开
  → 需限定为预览版。README 原文是「We release pre-view version of checkpoints, code of model inference and gradio demo」——即 preview 版权重 + 推理代码 + gradio demo，不是完整发布。另外「仓库最后更新 2025-06-03」与 README 最新 news 条目 2025-06-01（SkyReels-Audio 技术报告）对
- 遗漏补充：角色 LoRA 微调（musubi-tuner / diffusion-pipe / ai-toolkit + Wan2, HunyuanCustom（腾讯混元，开源）——主体一致性定制视频生成，支持单主体/多主体，以及 image / aud, ConsisID（CVPR 2025，开源，Identity-Preserving T2V via frequency , MAGREF（字节，开源）——masked guidance 的多主体参考视频生成，明确针对 multi-subject, MultiTalk（MeiGen-AI）与 InfiniteTalk——前者做多人对话场景的音频驱动 + 身份区分，后者, Wan-Dancer-14B（Wan-AI 官方组织下，HF 更新 2026-07-17）——官方 Wan-Video , 定妆图矩阵的开源替代：Qwen-Image-Edit-2509（多图参考，人物一致性强）与 FLUX.1 Kontext, Vidu Q1/Q2「参考生视频」（生数科技）——最多 7 张参考图锁定主体与场景，闭源 API 但国内可直接调用，是 , Runway Gen-4 References——闭源，产品定位就是跨镜头保持同一角色与场景，是本维度最成熟的商业对照组, 身份一致性的客观验收指标栈缺失——整条调研线引用了大量厂商自评数字（SkyReels-V3 的 0.6698/0.811, OpenS2V-Nexus / OpenS2V-Eval / OpenS2V-5M——主体到视频（S2V）方向的大规模数, 工程侧的身份锁定手段（非模型方案）——首帧锚定 + 逐镜头 I2V 链式续接、跨镜头 face swap 后处理（Fac

## LoRA 训练栈与云GPU
- **[WRONG]** [musubi-tuner] 支持视频架构包含 HiDream-O1
  → HiDream-O1 在 musubi-tuner 里是图像模型，不是视频架构。证据：docs 目录为 hidream_o1.md，README 文档索引写作「HiDream-O1-Image」，源码为 src/musubi_tuner/hidream_o1_generate_image.py（generate_image 而非 generate_video）。分类错误会导致误判「musubi 能训 HiDream 视频 LoRA」。
- **[WRONG]** [musubi-tuner] 图像侧支持 …Kandinsky 5…
  → Kandinsky 5 在 musubi-tuner 里是视频模型，不是图像模型。证据：src/musubi_tuner/kandinsky5_generate_video.py、docs/kandinsky5.md。SimpleTuner 也把它列为「Kandinsky 5.0 Video（2B lite / 19B pro）」。原文把视频/图像两类互换了（与 HiDream-O1 那条正好错反）。
- **[WRONG]** [musubi-tuner] 显存开关：--blocks_to_swap、--fp8 / --fp8_scaled、--gradient_checkpointing、--offload_inactive_dit
  → 训练侧的 DiT 量化开关是 --fp8_base（可叠加 --fp8_scaled），文本编码器侧是 --fp8_llm / --fp8_t5；裸 --fp8 只出现在推理脚本（wan_generate_video.py 等）。直接在训练命令里写 --fp8 会报未知参数。--blocks_to_swap / --gradient_checkpointing / --offload_inactive_dit 三个确认无误。
- **[WRONG]** [musubi-tuner] JSONL（image_jsonl_file / video_jsonl_file，字段 image_path/caption）
  → 视频 JSONL 的字段是 video_path（不是 image_path），共享 schema 为 video_path / caption / control_path / audio_path；图像 JSONL 才是 image_path / caption（Qwen-Image-Layered 另有 image_path_0/1/2）。「JSONL 必须显式给 cache_directory」确认无误。
- **[WRONG]** [musubi-tuner] 仅 DDP，官方未确认支持 FSDP/DeepSpeed，因此全参微调在多卡上基本不可行
  → 前半句可以从「未确认」升级为「已确认不支持」：README 只写 Multi-GPU training (using Accelerate)、文档待补，accelerate config 示例明确让用户对 DeepSpeed 回答 NO，仓库内无任何 FSDP 代码或文档。但后半句的前提站不住：musubi-tuner 并非只有 LoRA，src/musubi_tuner 下存在 hv_train.py 与 hidream_o1_tra
- **[UNVERIFIABLE]** [musubi-tuner] 社区实测：RTX 4070 Ti Super 16GB、rank16/alpha16、1600 步、lr 3e-5，高噪+低噪两个模型背靠背约 12 小时；分辨率组合 360x360x65 / 512x512x33 / 640x6
  → 未找到一手来源（非官方 issue/discussion/可追溯帖子）。该数字同时是后续「1.5-3.5 GPU-小时/2000 步」成本外推的唯一锚点，属于单点未证实数据支撑整条成本链，建议在报告里降级为「某社区口径，未复现」或直接删掉成本区间。
- **[WRONG]** [diffusion-pipe] HunyuanVideo 无 block swap 需 48GB 或 2×24GB 流水线并行
  → 误植。supported_models.md 里这句话（「Without block swapping, you will need 48GB VRAM, or 2x24GB with pipeline parallelism」）位于 ## HiDream 小节（第 239 行），## HunyuanVideo 小节（94-113 行）根本没有任何显存数字。另一处 48GB/2×24GB 出现在 SDXL 全参微调（第 52 行）。同段
- **[WRONG]** [diffusion-pipe] 视频模型支持矩阵（各模型 LoRA/全参/fp8 能力）
  → LoRA/全参/fp8 三列与官方 Summary 表一致，但矩阵漏掉了关键的任务维度限制，会让人误以为「支持」＝全任务可训：MiniMax H3 目前只支持 T2I 与 T2VA；LTX 2.3 只支持 T2I 与 T2V，无音频、无 I2V；HunyuanVideo-1.5 只支持 T2I 与 T2V；HunyuanVideo 只有 t2v。另 LTX 2.3 的 blocks_to_swap=46 原文是「该模型的最大值，且只有在
- **[WRONG]** [diffusion-pipe] 环境要求 Python 3.12 + PyTorch ≥2.9.0 + nvcc
  → README 没有规定 PyTorch 最低版本。原文是：PyTorch 刻意不写进 requirements（不同 GPU 需要不同版本），「As of this writing (October 26, 2025), PyTorch 2.9.0 with CUDA 12.8 works on my 4090」——这是作者单机实测记录，不是下限。把它写成硬性 ≥2.9.0 会让老卡用户误以为必须升级。Python 3.12、nvcc、
- **[WRONG]** [DiffSynth-Studio] Wan 训练覆盖…外加 Wan-Dancer、MOVA、LongCat、Video-As-Prompt
  → MOVA 不属于 Wan 训练矩阵。examples/wanvideo/model_training/{lora,full} 下只有 Wan-Dancer-14B-global/local.sh、LongCat-Video.sh、Video-As-Prompt-Wan2.1-14B.sh（以及未被提及的 krea-realtime-video.sh、Wan2.2-Animate-2-14B / -Distilled.sh、Wan2.1-
- **[WRONG]** [ai-toolkit] Wan 2.2 T2I 支持于 2025-08-16 加入（ostris 官方公告），I2V 14B 教程 2025-08-21 发布
  → 2025-08-16 合并的是 PR #377（分支名 wan22_14b），即 Wan 2.2 14B 整体支持，不是「T2I 支持」；Wan2.2 5B 支持更早，2025-07-29。24GB 的图像训练示例配置 config/examples/train_lora_wan22_14b_24gb.yaml 是 2025-08-28 才加入的（commit message: Added example config for trai
- **[WRONG]** [SimpleTuner] LTX Video ~2.5B(Apache-2.0)、LTX Video 2 19B(Apache-2.0)、Hunyuan Video 8.3B(AGPL-3.0)
  → 三个许可证全错（属于「把受限/自有许可说成开源许可」）。HuggingFace 一手模型页：Lightricks/LTX-Video 为 license: other（LTXV 自有许可，非 Apache-2.0）；Lightricks/LTX-2 为 license: other，license_name: ltx-2-community-license-agreement（社区许可，有使用限制，非 Apache-2.0）；tence
- **[WRONG]** [SimpleTuner] Wan 2.x I2V 支持高/低噪 stage preset
  → preset 本身存在（model_flavour=i2v-14b-2.2-high / i2v-14b-2.2-low、wan_validation_load_other_stage、wan_force_2_1_time_embedding 均确认），但漏掉了 WAN.md 开头的决定性限制：「Currently, image-to-video training is not supported for Wan, but T2V Lo
- **[OUTDATED]** [SimpleTuner] 2026-01 在做 LTX-2 audio-only 训练
  → 已经不是「在做」：audio-only 训练 2026-01-20 就合并了（PR #2461 feature/audio-only-ltx2），随后 01-31 加 --validation_audio_only、02-01 修 audio fps、02-13 修 s2v/ltx-2 音频自动切分，2026-08-15 还并了 audio-dataset-fake-video。当前 LTXVIDEO2.md 已文档化 audio 数据
- **[WRONG]** （存疑项）Together AI H100 两个口径冲突 $3.99 vs $1.99，H200 索引 $2.99；疑为按需 vs 预留/合约价混淆
  → 假设错了，不是「按需 vs 预留」，是「按需 vs 可抢占（preemptible）」。官网定价页同时列：HGX H100 on-demand $3.99/GPU·h（带 Promotion valid until 09/30/26 标注）、preemptible $1.99；HGX H200 on-demand $5.99、preemptible $2.99。第三方索引抓到的 $1.99/$2.99 正是 preemptible 档。
- **[WRONG]** （存疑项）「RunPod A100 SXM $1.00/h」与官方 $1.39/$1.59 不一致，疑为索引抓到 Community 特价
  → $1.39 本身就是 Community Cloud 价、$1.59 是 Secure Cloud 价，所以 $1.00 不是 Community 档，只是 getdeploying 的过期快照。官网当前口径：A100 SXM 80GB $1.39(Community)/$1.59(Secure)；H100 SXM $2.69/$3.49；H200 141GB $3.59/$4.59。报价时应按 Community/Secure 两档分
- 遗漏补充：VideoX-Fun (aigc-apps/VideoX-Fun, 2.2k stars, 2026-09-17 仍在更, Lightricks/LTX-Video-Trainer（469 stars）：LTX 官方组织下的 LoRA / IC, LoRA 权重格式互通问题完全缺席：diffusion-pipe 输出 ComfyUI 格式、finetrainers/, musubi-tuner 的 torch.compile 与 LoHa/LoKr 支持（docs/torch_compi, SimpleTuner 的 TREAD（token dropout，WAN.md 实测 1.3B 从 10 s/step, diffusion-pipe 的 eval set / held-out metrics 与 TensorBoard 指, 云 GPU 侧漏掉的按需供应商：Ostris Cloud（ai-toolkit 作者自营，README 内直接推荐，与 , 国内侧漏掉的 GPU 渠道：阿里云 PAI-DSW / 灵骏（DiffSynth 的原生落地环境，按量价格公开）、腾讯云

## TTS/口型/数字人驱动
- **[WRONG]** [ChatTTS] 提供说话风格与情感的细粒度控制标记
  → 官方仓库 README/FAQ 明确否认存在情感控制标记。ChatTTS 的 token 级控制单元只有三类：[laugh_0]–[laugh_2]（笑声）、[oral_0]–[oral_9]（口语化程度）、[break_0]–[break_7] 及 [uv_break]/[lbreak]（停顿）。FAQ 原文表述为『目前唯一的 token 级控制单元』就是笑声、口语化和停顿，并把情感控制列为『未来版本可能加入』。因此『说话风格与情感的
- **[WRONG]** [ChatTTS] 本次检索未能核实其 2026 年维护状态与权重许可证原文（研究员自标存疑项）
  → 该项可核实，且结论对选型有决定性影响：ChatTTS 代码许可证为 AGPLv3+，模型权重为 CC BY-NC 4.0，官方明确限定为教育与研究用途，禁止商用。这意味着 ChatTTS 不具备商用可行性，应与 MaskGCT / F5-TTS 权重 / XTTS-v2 归为同一非商用档，而不是作为『对话式中文』候选入库。另外社区关于『只有随机 speaker 采样、无稳定指定音色克隆』的说法已从官方仓库确认：官方接口为 chat.sa
- **[WRONG]** [F5-TTS] 训练数据 Emilia + WenetSpeech4TTS + LibriTTS + LJSpeech
  → 混淆了『仓库提供数据准备脚本的数据集』与『已发布预训练权重的实际训练数据』。官方 README 只声明预训练模型训练于 Emilia（原文：权重采用 CC-BY-NC 许可正是 due to the training data Emilia, which is an in-the-wild dataset）；HuggingFace SWivid/F5-TTS 模型页关联的 dataset 也只有 amphion/Emilia-Datas
- **[OUTDATED]** [Fish-Speech / OpenAudio S2 Pro] 产品命名为 OpenAudio S2 Pro
  → 当前官方命名已改回 Fish Audio S2 Pro，不再用 OpenAudio 前缀（OpenAudio 是 S1/S1-mini 时期的品牌）。权重仓库路径为 huggingface.co/fishaudio/s2-pro，独立技术报告为 arXiv 2603.08823（2026-03-09），HF collection 更新于 2026-03-10。规格数字本身核实无误（4B Slow AR + 400M Fast AR、10
- **[UNVERIFIABLE]** [GPT-SoVITS] 许可证 MIT（代码与模型均可商用）
  → 前半句成立、后半句是推断。官方 README 的 MIT 徽章与仓库 LICENSE 文件覆盖的是代码；仓库内并未对预训练权重单独作出 MIT 授权声明。GPT-SoVITS 的预训练模型链路依赖第三方上游组件（中文 RoBERTa-wwm-ext 类 BERT、HuBERT 类自监督特征提取器等），这些上游各有自己的许可证。因此『模型也可商用』属于未经官方确认的外推，入库前应逐个核对权重包内各子模型的来源与许可证，不要按 MIT 一刀
- **[WRONG]** [IndexTTS-2.5] 情感控制三通道：情感参考音频 / 8 维情感向量 / 文本情感（Qwen3 微调的软指令）+ emo_alpha 0.0–1.0
  → 官方 README 列出的是四种方式而非三种：(1) 情感参考音频；(2) 8 维情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]；(3) use_emo_text=True 从待合成文本自动抽取情感；(4) 通过独立的 emo_text 参数传入与正文解耦的情感描述文本。(3) 和 (4) 是两条不同的通路（前者复用正文、后者独立描述），
- **[UNVERIFIABLE]** [MaskGCT] 发布 2024-10-19
  → 找不到 2024-10-19 的一手依据，两个可考的日期都不是这一天：arXiv 2409.00750 的提交日为 2024-09-01；HuggingFace amphion/MaskGCT 权重仓库的 initial commit 为 2024-10-13。建议改为『论文 2024-09-01，开源权重 2024-10-13 上 HF』。训练数据（Emilia，英文 5 万 + 中文 5 万 = 10 万小时）与权重许可证 CC-B
- **[UNVERIFIABLE]** [XTTS-v2 (Coqui)] 历史商用授权价约 365 USD/年，适用营收/融资 <100 万美元的公司
  → 无法从任何仍存活的一手来源核实。Coqui 官网与 Coqui Studio 已随公司 2024 年 1 月关停而下线，HuggingFace coqui/XTTS-v2 模型卡只写明许可证名称 Coqui Public Model License (CPML) 并链向一篇讲 CPML 由来的博客，页面本身不含任何价格或营收门槛条款，且页面文案仍停留在公司关停前的状态（仍在推广 Coqui Studio 与 Coqui API）。这两个
- **[UNVERIFIABLE]** [F5-TTS] RTF ... 0.0402（batch=1）；TensorRT-LLM 离线 PyTorch 模式 0.1467
  → 官方 README 的基准表中可确认的只有一个数字：F5-TTS Base (Vocos) 在单张 L20、16 NFE、并发 2、26 组 prompt-audio/target-text 配对下 RTF = 0.0394。batch=1 的 0.0402 与 TensorRT-LLM 离线 PyTorch 模式的 0.1467 未在本次抓取的 README 主文中出现（可能位于 src/f5_tts/runtime/triton_t
- **[OUTDATED]** [IndexTTS-2] 发布 2025-09-08（arXiv 2506.21619）
  → arXiv ID 正确但日期需要拆分标注：arXiv 2506.21619《IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech》v1 提交于 2025-06-23，v2 修订于 2025-09-03。2025-09-08 既非论文提交日也非修订日，应指开源权
- 遗漏补充：MOSS-TTSD（OpenMOSS/复旦）——专为『对白』设计的对话式语音合成模型，原生支持双说话人交替、中英双语长音, VibeVoice（微软）——长对话多说话人 TTS，1.5B 与 7B 两档，官方宣称可生成最长 90 分钟、最多 4, Dia-1.6B（Nari Labs）——Apache 2.0，单次前向生成整段双人对话脚本（支持 [S1]/[S2] , FireRedTTS-2（小红书）——面向长篇多说话人对话的开源中文 TTS，流式 + 对话上下文建模，中文对白质感在开, Higgs Audio V2（Boson AI）——基于 LLM 的统一音频生成模型，支持多说话人对话零样本克隆与背景音, Chatterbox（Resemble AI）——MIT 许可的开源 TTS，独有 exaggeration（情绪夸张度, Kokoro-82M（hexgrad）——Apache 2.0、仅 82M 参数的超轻量 TTS，CPU 可实时，适合做, Step-Audio 2 / Step-Audio-EditX（阶跃星辰）——Apache 2.0，EditX 支持对已, Spark-TTS（SparkAudio，基于 Qwen2.5-0.5B）与 Orpheus TTS（Canopy La, MiniMax Speech 2.x（海螺）与 Qwen3-TTS（阿里）、豆包/Seed-TTS（字节）——三家国内闭, Seed-VC 与 RVC（音色转换 / 变声）——对白制作里常用的『先用任意 TTS 出稿、再用 VC 转成目标角色音, LivePortrait（快手，MIT）——表情与姿态迁移的开源标杆，许可证友好，常与口型模型组合使用；调研的口型/数字

## 超分/插帧/调色
- **[WRONG]** 论文实测：720p / 100 帧，SeedVR2-7B 单步 269.0–299.4 秒
  → 数字对但归属错了。SeedVR2 论文附录 B Table 4（720p/100帧）：Ours-3B = 3391.5 M 参数 / 269.0 s；Ours-7B = 8239.6 M 参数 / 299.4 s。即 269.0 s 是 3B 档，不是 7B 的下限。同表基线：SeedVR-7B 8239.6 M / 1284.8 s、STAR 2041.0 M / 2326.0 s、UAV 691.0 M / 1284.5 s、VEn
- **[WRONG]** ComfyUI 社区版 numz/RedbeardNZ SeedVR2_comfyUI
  → 仓库名不对。官方社区实现是 numz/ComfyUI-SeedVR2_VideoUpscaler（作者 NumZ + AInVFX/Adrien Toupet），Apache-2.0，非 ByteDance 官方维护。『RedbeardNZ SeedVR2_comfyUI』是镜像/派生写法，不是主仓库。
- **[WRONG]** 3B-FP8 约 6GB 权重、实测仍需 ≥18GB 显存
  → 与官方 README 的显存分档矛盾。该 repo 明示：≤8GB 用 GGUF Q4_K_M + BlockSwap + VAE tiling；12–16GB 用 FP8（按需开 BlockSwap/tiling）；24GB+ 用 FP16 免优化。所以 FP8 的门槛是 12–16GB 而非 ≥18GB，低显存路径（GGUF 4bit）可下探到 8GB 以内。README 未给权重文件体积，『约 6GB』无来源。
- **[UNVERIFIABLE]** 4090 上 768px→2K 单图 30–60 秒；社区口径 3B ≈20 秒/帧、7B ≈60 秒/帧
  → 该 repo 只给相对加速比（torch.compile 使 DiT 快 20–40%、VAE 快 15–25%，张量操作比 einops 快 2–5×），没有任何 秒/帧 或具体 GPU 的绝对 benchmark。这组数字查不到一手来源，排期不要直接用。
- **[WRONG]** FlashVSR 同时开源了训练集 VSR-120K（120k 视频 + 180k 图像）
  → 数据集规模对，但尚未开源。官方 repo 与 HF 模型页都把 “Dataset release (VSR-120K) for large-scale training” 列在 TODO / coming soon 下。只有推理代码与权重已放出，训练集没有。
- **[OUTDATED]** FlashVSR 权重在 HuggingFace JunhaoZhuang/FlashVSR
  → 不完整。v1 在 JunhaoZhuang/FlashVSR（LQ_proj_in.ckpt、TCDecoder.ckpt、Wan2.1_VAE.pth、diffusion_pytorch_model_streaming_dmd.safetensors，2025-10）；v1.1 在独立仓库 JunhaoZhuang/FlashVSR-v1.1（2025-11）。想用 v1.1 必须换 repo，不是同一路径的更新。
- **[UNVERIFIABLE]** FlashVSR 是 CVPR 2026 论文
  → 仅有 GitHub README 的自述支持。arXiv 2510.12747（2025-10-14 提交，Junhao Zhuang 等，清华/CUHK）的 comments 字段没有任何会议接收标注。作为选型依据可以接受（作者自述），但不要当作已核实的同行评议状态引用。
- **[WRONG]** VEnhancer 硬性显存：官方要求单卡 ≥60GB VRAM，推荐 H100 / A100
  → 官方 README 原话是 “at least A100 80G is required”（单卡推理）。是 A100-80G 起步，不是『≥60GB』——60GB 这个数不出现在官方口径里，按 60GB 选卡会选错（如 A6000 48G / L40S 48G 都不够，而 60GB 档位在 NVIDIA 产品线里本就不存在）。
- **[UNVERIFIABLE]** 实测：单卡 A100 默认参数增强一段 6 秒 CogVideoX 视频，占用 60GB 显存、耗时 40–50 分钟
  → 官方 repo 没有给出任何推理耗时数据。唯一可引的第三方量化口径是 SeedVR2 论文 Table 4：VEnhancer 2044.8 M 参数，720p/100 帧 2029.2 秒（约 34 分钟）——但那是 720p/100 帧，不是 6 秒 CogVideoX 片段，两者不可互换。
- **[WRONG]** VEnhancer 2024-08 开源
  → 早一个月。官方 News：[2024.07.28] 推理代码与预训练模型发布；[2024.08.18] 支持任意长视频 + 15 步快速采样；[2024.09.10] 多卡推理与 tiled VAE；[2024.09.12] v2 checkpoint（venhancer_v2.pt）。arXiv 2407.07667 / 2024-07 正确。
- **[WRONG]** VEnhancer 可任意倍率同时放大空间与时间分辨率
  → 不是任意倍率。官方给的空间放大范围是 up_scale 1~8，且明确推荐 ×3、×4；时间侧是设 target_fps（默认 24）而非任意倍数。『空间+时间统一模型』这一点成立，『任意倍率』夸大了。
- **[UNVERIFIABLE]** Upscale-A-Video 速度：49 帧 720p 需要 510 秒推理（论文口径）
  → sczhou/Upscale-A-Video 的 README 没有任何推理速度或显存数字。可核实的只有 SeedVR2 论文 Table 4 的 691.0 M / 1284.5 s（720p、100 帧）。『49 帧 510 秒』找不到一手出处。（CVPR 2024 Highlight、NTU S-Lab License 1.0 非商业、--use_llava 支持均已核实无误。）
- **[WRONG]** Topaz 定价（2026）：Personal 订阅 $299/年；Pro $699/年（含 Starlight）；云端计划另起 $39/月含一定额度 credits
  → 产品线已重组为 Topaz Studio 套件，定价表对不上。当前官网：Topaz Studio（Personal）$399/年 或 $45/月年付 / $69/月；Topaz Studio Pro（商用）$799/年 或 $79/月年付。单品档：Topaz Video Personal $299/年 或 $39/月（$59/月 单月）——$299/年 对应的是单品 Topaz Video 而非套件 Personal。不存在 $699/
- **[UNVERIFIABLE]** credit 口径：1 分钟 1080p30 升 4K 用若干核心模型约消耗 34 credits
  → 官网定价页只给『Topaz Video: 25 credits』这类按任务的粗口径与 $0.10/credit 的加购价，没有『1 分钟 1080p30→4K ≈ 34 credits』这个换算。该数字来自第三方转述，不能用来做成本模型。
- **[UNVERIFIABLE]** 永久授权版 Video AI 已停售，存量用户不再获得新模型更新
  → 前半句成立：官网定价页已全部为订阅制，不提供永久授权。后半句（存量永久授权用户不再获得新模型）在定价页上没有任何表述，无一手来源，且在 Adobe 收购交割前后可能变化，不要写进结论。
- **[UNVERIFIABLE]** Personal 授权下部分 Starlight 模型强制走云端处理
  → 定价页区分的是『unlimited local rendering』+『video credits』两套额度，但没有说明哪些 Starlight 模型必须云端。这条是落地阻塞项（决定能否离线批处理），建议直接在 Topaz Video 试用版里实测确认，不要按转述排期。
- **[WRONG]** （存疑项）GIMM-VFI 主仓库 GSeanCDAT/GIMM-VFI 的许可证未明示，商用前必须人工核对
  → 已查明，可以结案：仓库根目录 LICENSE 是 S-Lab License 1.0，明确『Redistribution and use for non-commercial purpose』，商用需联系作者授权。即 GIMM-VFI（NeurIPS 2024，权重在 HF GSean/GIMM-VFI）与 Upscale-A-Video 同属 NTU S-Lab 非商业许可族，商用项目应直接排除或走授权谈判。kijai/ComfyUI
- 遗漏补充：DLoRAL（yjsunnn/DLoRAL，NeurIPS 2025，不是 ICCV 2025）——One-Step D, MGLD-VSR：SeedVR2 论文 Table 4 里的第四条基线（1430.8 M 参数，720p/100 帧 1, VEnhancer 的可引用量化档位：2044.8 M 参数 / 720p·100 帧 2029.2 s（SeedVR2, STCDiT（arXiv 2511.18786，2025-11-24，Junyang Chen / Jinshan Pa, FILM（google-research/frame-interpolation，ECCV 2022，Apache-2., Video2X 6.x（k4yt3x/video2x，AGPL-3.0，已用 C/C++ 重写）：一站式批量超分+插帧工, 动漫/AIGC 专用超分路线：Real-CUGAN 与 Anime4K v4（均在 Video2X 内置）。原调研只提了, 低显存落地的真实路径被写错方向：SeedVR2 在 ComfyUI 侧的可行下限是 GGUF Q4_K_M + Bloc, Topaz 已不是单品而是 Topaz Studio 套件（Video / Photo / Gigapixel / Im, 去闪烁 / 时序一致性的专用后处理层（ProPainter、All-In-One Deflicker 一类）在本维度完全

## 3D/动捕底稿
- **[WRONG]** [WHAM] README 的 TODO 仍列有数据预处理与训练实现待完善项
  → WHAM README 的 TODO 区块（第 96-104 行）实际为：`- [ ] Data preprocessing`（未完成）、`- [x] Training implementation`（已完成）、`- [x] Colab demo release`（已完成）、`- [x] Demo for custom videos`（已完成）。只有「数据预处理」一项未打勾，训练实现已经标记完成并随仓库发布。把训练实现说成「待完善」会误
- **[WRONG]** [研究员自标存疑项] GVHMR、SMPLest-X 的仓库含 LICENSE 文件但类型未确认，商用前需核实
  → 已核实，两者都是禁止商用的许可，不是待确认状态：(1) GVHMR 的 LICENSE 是浙大 CAD&CG 自定义学术许可，原文明确写「Permission to use, copy, modify and distribute this software and its documentation for educational, research and non-profit purposes only. Any modifica
- **[UNVERIFIABLE]** [HY-Motion-1.0] GitHub 2558 stars，2026-07-18 仍在更新
  → 量级方向正确但精确数字无法坐实：仓库页当前显示 2.6k stars、36 commits，commits/master 页显示最近提交确实在 2026 年 7 月（内容为 merge PR、bare except 修复、README/显存说明调整等维护性提交）。但 2558 这个精确星数与 2026-07-18 这个精确日期本次未能从 GitHub API 取到（api.github.com 对本环境 403 / rate limit
- **[WRONG]** [研究员自标存疑项] HY-Motion-1.0 是否沿用腾讯系社区许可（>100万 MAU 需授权 + 排除 EU/UK/韩国 + 禁止用输出训练其他 AI 模型）未核实
  → 已核实为「沿用」，不是未知：仓库根目录 License.txt 即《Tencent HY-MOTION 1.0 Community License Agreement》，三条限制全部存在——第 4 节 MAU 超过 1 million 须向腾讯申请许可；前言与 1(l) 明确「THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND S
- 遗漏补充：TRAM（Global Trajectory and Motion of 3D Humans from in-the-w, CameraHMR / TokenHMR / PromptHMR（CVPR-3DV 2025 一系）——单帧 SMPL , Multi-HMR（ECCV 2024, Naver Labs, 单次前向多人全身 SMPL-X 含手脸）——多人同框镜, Meshcapade / SMPL-X Blender add-on 与 SMPL-to-FBX 转换链——从 SMPL, Mixamo（Adobe，免费自动绑定 + 动作库）与 AccuRIG / ActorCore（Reallusion，免, Cascadeur（物理感知 + AI 自动补姿的关键帧工具）——在「动捕底稿不够用、需要人工二次雕」的环节比纯 Ble, Move.ai / Rokoko Video / DeepMotion / Plask 等商业单目动捕 SaaS——按分, MoMask / MDM / MotionLCM / OmniControl 等开源文生动作模型——HY-Motion-, MegaSaM / MonST3R / VGGT 等新一代动态场景相机轨迹估计——GVHMR 用 SimpleVO、WH, Unreal Engine MetaHuman Animator + Live Link Face——面部底稿链路完全缺, Grease Pencil / Freestyle / Blender Compositor 的分层输出（depth /, Viggle AI 与 Runway Act-Two 等「3D/视频驱动角色」的闭源对照——判断自建 3D 底稿产线是否

## 评测质检
- **[WRONG]** [VBench (v1)] 2024-02 发布，CVPR 2024 Highlight
  → 2024-02 是「被 CVPR 2024 接收为 Highlight」的时间，不是发布时间。官方 README Updates 原文为 [02/2024] VBench accepted to CVPR 2024 as Highlight。实际发布链路：[11/2023] Prompt Suites 释出、arXiv 2311.17982 于 2023-11-29 提交、[12/2023] 16 维评估代码释出、[01/2024] P
- **[WRONG]** [VideoScore / VideoScore-v1.1] HF: TIGER-Lab/VideoScore，License MIT
  → 两个模型仓库许可证不同，声明把它们混为一谈。HF API 实测：TIGER-Lab/VideoScore 的 license 是 apache-2.0（createdAt 2024-06-19，lastModified 2025-01-08）；只有 TIGER-Lab/VideoScore-v1.1 的 YAML frontmatter 是 license: mit（createdAt 2024-11-28）。GitHub TIGER-
- **[WRONG]** [VBench++ (TPAMI 版)] 2025-11 作为 TPAMI 期刊论文发布
  → 2025-11 是「被 TPAMI 接收」，不是期刊发表。README 原文：[11/2025] VBench++ accepted to TPAMI。VBench++ 本身早在 2024-11-20 就以 arXiv 2411.13503 预印本形式公开（comments 注明与 2311.17982 大量文本重叠）。截至 2026-09 未检索到正式的 TPAMI 卷期/页码/DOI，引用时应写 arXiv 或 accepted t
- **[WRONG]** [VBench-2.0] 依赖模型链：LLaVA-Video-7B-Qwen2、Qwen2.5-7B-Instruct、CLIP、CoTracker、YOLO-World、InsightFace
  → 按 VBench-2.0/vbench2/utils.py 源码实测，列表既有错项也有漏项。实际引用的权重：LLaVA-Video-7B-Qwen2（CACHE_DIR/lmms-lab/LLaVA-Video-7B-Qwen2）、Qwen2.5-7B-Instruct（CACHE_DIR/Qwen/Qwen2.5-7B-Instruct）、CoTracker2（torch.hub facebookresearch/co-tracker
- **[UNVERIFIABLE]** [VBench-2.0] 官方建议 18 维度分 18 张卡跑（上限 8 卡），单卡串行「not recommended」
  → 能从 VBench-2.0 README 证实的只有两点：支持最多 8 张 GPU 并行（一维度一卡），以及单卡串行跑全部 18 维「not recommended」。「官方建议分 18 张卡」未在 README 中找到对应表述，且与「上限 8 卡」自相矛盾。请按「最多 8 卡并行」写。
- **[UNVERIFIABLE]** [VBench temporal_flickering] 零 GPU、零模型权重，1080p 5秒片段 CPU 上 < 1s
  → 「零模型权重、纯 numpy+cv2」CONFIRMED（temporal_flickering.py 只有 cv2.absdiff + np.mean，无任何权重加载）。但「1080p 5 秒 CPU < 1s」无任何一手来源，官方仓库未给性能数据；且该实现内部仍走 VBench 的分布式/device 框架。该数字应标为自测待定，不要当成官方指标。另注：真正的耗时瓶颈通常是解码而非 MAE 计算。
- **[UNVERIFIABLE]** [VBench subject_consistency] 主流模型在 VBench 上该项落在 0.90-0.98 区间
  → 未能从一手来源核实。HuggingFace Vchitect/VBench_Leaderboard 是 Gradio 动态 Space，WebFetch 只返回页面框架（Like 362 / Running），拿不到榜单数值表；本轮 WebSearch 配额已耗尽无法交叉验证。若要把 0.93/0.88 之类阈值挂在这个区间上，必须自己从 leaderboard 的 CSV/JSON 后端拉数再引用。
- 遗漏补充：pyiqa（chaofengc/IQA-PyTorch）：统一封装 MUSIQ / CLIP-IQA / MANIQA , VQAScore（CLIP-FlanT5 / GenAI-Bench 系）与 TIFA / DSG：用 VQA 方式做「, TransNetV2（镜头边界检测）：AI 生成长视频最典型的废片形态之一是「模型自己插了非预期的硬切/跳变」。PySc, FFmpeg 原生质检滤镜链（blackdetect / freezedetect / blackframe / sig, FVD 的替代指标 JEDi（JEPA Embedding Distance）与 FVMD（Fréchet Video , VideoPhy / VideoPhy-2 与 PhyGenBench：专门评「物理常识违反」（物体穿模、液体反重力、刚, ChronoMagic-Bench / TC-Bench：评「时序上的状态变化是否真实发生」（metamorphic a, 人评基础设施与统计方法：Bradley-Terry / Elo 成对比较（而非绝对打分）、Krippendorff's , MEt3R 及基于 VGGT/DUSt3R 的多视角一致性度量：用 3D 重建一致性来判定同一场景跨镜头/跨帧的几何是否, AIGC 特有崩坏的专项检测：多指/肢体异常（除 VBench-2.0 的 ViTDetector 外，可用 Media
