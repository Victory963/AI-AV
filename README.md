# longfilm —— 长时间 AI 数字人拍片系统

把一份分镜 JSON 变成一集 2–8 分钟的成片：参考位编排 → 多引擎路由 → 续接链 → 质检重拍 → 调色拼接 → 超分 → 合规交付。

**定位**：单镜观感靠引擎（闭源 API 或自建开源），成片长度靠这套分镜系统。不是「微调出一台无审查 Seedance」—— 官方权重不开放，这条路已被证实是终局，详见 [docs/PLAN.md](docs/PLAN.md) §0。

---

## 现在就能跑（不需要 GPU，不需要任何密钥）

```bash
git clone https://github.com/Victory963/AI-AV.git && cd AI-AV
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install pydantic numpy pillow edge-tts imageio-ffmpeg
sudo apt install -y ffmpeg fonts-noto-cjk      # 系统 ffmpeg（带 libass）+ 开源中文字体

export PYTHONPATH=$PWD/src

# 0) 全部模块自测（26 个，无 GPU 无密钥）
.venv/bin/python -m longfilm.cli selftest

# 1) 生成一份 30 镜的示例分镜（2 个虚构成年角色，4.3 分钟）
.venv/bin/python scripts/build_demo_storyboard.py

# 2) 端到端跑完整条产线（mock 引擎出真 mp4，约 15 分钟）
.venv/bin/python scripts/run_pipeline.py --work out/work/full

# 3) 把这次运行渲成 demo 视频
.venv/bin/python scripts/make_demo.py
```

`run_pipeline.py` 支持 `--resume`（捡回已渲染产物）、`--stages`（只跑某几段）、`--limit` / `--scale`（缩小规模快速验证）。每个阶段跑完就落盘，进程被杀也能接上。

接真引擎只是换环境变量：`ARK_API_KEY`（火山方舟 Seedance）、`DASHSCOPE_API_KEY`（阿里百炼万相）、`COMFY_URL`（自建 ComfyUI）、`FAL_KEY` / `REPLICATE_API_TOKEN`（聚合容灾）。路由只读 `Capabilities`，不认名字，所以加一家供应商不用改队列。

上云见 [deploy/README.md](deploy/README.md)：选卡、实价、一键开机、LoRA 训练、成本计算器。

**仓库里不含的东西**（授权或体积原因），代码都有自动查找与兜底：

| 缺的 | 为什么不放 | 怎么补 |
|---|---|---|
| `bin/ffmpeg` `bin/ffprobe` | 第三方二进制；ffprobe 超过 GitHub 单文件 100MB 上限 | 自动按 `LONGFILM_FFMPEG` → `bin/` → `PATH` → imageio-ffmpeg 查找；缺 ffprobe 时退回解析 `ffmpeg -i` |
| 中文字体 | 微软雅黑 / 黑体不允许再分发 | 装 `fonts-noto-cjk`，或放进 `assets/fonts/`，或设 `LONGFILM_FONT`。见 [assets/fonts/README.md](assets/fonts/README.md) |
| 角色参考图 | 每张都要声明来源；仓库不带任何真人肖像 | 见 [assets/cast/README.md](assets/cast/README.md)。mock 引擎不需要它们 |
| 密钥 | 只从环境变量读 | 复制 [.env.example](.env.example) 为 `.env` 按需填写 |

---

## 产线结构

| 模块 | 职责 |
|---|---|
| `schema.py` | **唯一契约**。Storyboard / Scene / Shot / CharacterBible / RefPack / Continuity，所有模块只认它 |
| `storyboard_gen.py` `charbible.py` | 节拍表 → 场景镜头；角色圣经与定妆矩阵 |
| `audio_first.py` | TTS 合成 → 实测时长回填 → 按对白反推镜长（装不下就在换气处拆镜） |
| `refpack.py` | 9/3/3 参考位编排。身份锚按角色配额分配，裁剪不静默 |
| `prompt_os.py` | 确定性提示词编译，分层拼装 + 方言适配 |
| `router.py` | 双引擎路由。只读 Capabilities 打分，内容门禁 + 降级记录 + 成本账本 |
| `chain.py` | 续接链：官方 extend / 首尾帧 / job 续写 / 重叠混接，**带显式漂移预算与强制重锚** |
| `qc.py` | 质检门禁。7 项 CPU 指标，三态裁决，给可执行的重拍建议 |
| `grade.py` `stitch.py` | 跨镜色彩统一；原子镜拼接、混音、响度归一 |
| `upscale.py` | 后处理计划器。按可用硬件选链路，落选原因全部记录 |
| `timeline.py` | 分轨时间线 → EDL / OTIO / ASS（成片能进 DaVinci 精修） |
| `compliance.py` | 显式标识 + 元数据 + 溯源清单（每张参考图的哈希与来源类型） |
| `queue.py` `cache.py` | SQLite 工单队列（租约防卡死）+ 内容寻址缓存（同指纹不重复扣费） |
| `distill.py` `lora.py` `dpo.py` | 教师数据集 → 训练配置 → 偏好对齐与飞轮多样性监控 |
| `animate.py` `previz3d.py` | 动作迁移；Blender 相机按景别/运镜/焦段自动 K 帧 |
| `providers/` | ark_seedance · dashscope_wan · comfy_local · aggregator · mock |
| `_proc.py` `_fonts.py` `cli.py` | 子进程父死子亡加固；中文字体定位；命令行入口 |

每个模块都能单独跑自测：`python -m longfilm.<模块名>`。逐模块的公开 API 与易错点见 [docs/MODULES.md](docs/MODULES.md)。

---

## 三条贯穿设计的约束

1. **路由只读 Capabilities，不写 `if provider == "..."`。** 换供应商时队列和工单不动。
2. **参考位裁剪绝不静默。** 每条被裁的都进 `DroppedRef`；裁到身份锚会置 `critical`，路由据此换引擎。静默截断会让「这镜为什么脸漂了」永远查不出来。
3. **对白时间戳存镜内相对秒。** retime / 拆镜 / 重排会不断改绝对时间，却不改一句话在它所属镜头里的位置。绝对值交给 timeline 现算。

---

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) —— **架构详解**：分层、唯一契约、六阶段数据流、单镜渲染时序、路由打分与容灾、续接漂移预算、质检重拍、可靠性机制、provider 矩阵、部署拓扑、扩展点
- [docs/MODULES.md](docs/MODULES.md) —— **代码详解**：逐模块的职责、公开 API、设计要点与自测命令
- [docs/PLAN.md](docs/PLAN.md) —— 调研结论与方案。**12 条技术线、321 条方案、119 条事实修正**；含硬约束的三条过期更正、工位全景表、成本模型、落地路线、20 条增量创新
- [docs/research_index.md](docs/research_index.md) —— 方案索引（按工位分组，带成熟度与许可证）
- [docs/research_corrections.md](docs/research_corrections.md) —— 事实核查修正（推翻或存疑的说法）
- [deploy/README.md](deploy/README.md) —— 云 GPU 开机到出片手册

---

## 边界

- **本机（无 GPU / 3GB 内存）能跑的是调度，不是画质。** demo 里的镜头由 mock 引擎合成。画质那一半要接真引擎。
- **本产线只做虚构成年角色。** `CharacterBible.age_statement` 是 schema 强制字段，参考图必须标注来源类型，合规检查会拦下未标注的。不实现成人露骨内容功能，不提供规避平台审核的手段 —— 审核在厂商推理管线内，下游绕不过去。
- **未经真实密钥验证的部分**：各 provider 的具体 endpoint 与计费按官方文档实现，但没有真实 key 跑过。代码里标了 `TODO(日期)` 的地方是查证未果的字段。
