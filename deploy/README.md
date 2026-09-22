# 云 GPU 开机到出片手册

本机没有 GPU 也能跑通整条产线（用 mock 引擎），但真出片必须上云。这份手册讲**开一台机器到出第一条片**要做什么、花多少钱、会踩哪些坑。

价格查证日期 **2026-09-20**，来源见 `prices.yaml` 每条的 `source` 字段。带 `[二手]` 标记的价格来自测评/聚合站，**排期前必须复核**。

---

## 1. 选卡：训练和推理选的不是同一张

这是最容易一开始就选错的决定。两种负载的瓶颈完全不同：

| 负载 | 瓶颈 | 推荐 | 为什么 |
|---|---|---|---|
| **训角色 LoRA** | 显存（放得下 14B + 优化器状态） | **RTX 4090 24GB** 或 **A6000 48GB** | musubi-tuner 24GB 可跑；多卡只有 DDP，加卡只加吞吐不降显存门槛 |
| **批量出片** | 吞吐（每秒成品烧多少 GPU 秒） | **RTX 5090 32GB**（性价比）或 **H100 SXM**（赶工期） | 出片是长时间连跑，机时单价 × 吞吐才是真成本 |
| **全参微调** | 显存 + 多卡通信 | **2×A100 80GB** 起，走 diffusion-pipe（DeepSpeed） | musubi 不支持 FSDP/DeepSpeed，全参别指望它 |
| **后处理超分** | 显存 + 特定架构 | **A100/A800**（FlashVSR 官方只保证这两个） | 见 §4 的坑 |

### 实价对比（2026-09-20）

| GPU | RunPod Community | RunPod Secure | Vast.ai |
|---|---|---|---|
| RTX 4090 24GB | $0.34/h | — | $0.39/h |
| RTX 5090 32GB | $0.69/h | — | $0.45/h |
| RTX A6000 48GB | $0.33/h | — | $0.30/h |
| L40S 48GB | $0.79/h | — | $0.60/h |
| A100 SXM 80GB | $1.39/h | — | $0.90/h |
| H100 SXM 80GB | $2.69/h | $3.49/h | $1.89/h |
| H200 141GB | $3.59/h | — | — |

国内可用 AutoDL（4090 约 ¥2.1/h）。**Vast.ai 是竞价实例，会被抢占** —— 训练要开 checkpoint 续跑，出片要让队列能重新领取任务（`queue.py` 的租约机制就是干这个的）。

### 预算优先：最便宜的可行组合（含 Paperspace 实测对比）

先说结论：**Paperspace 对训练和批量出片都不划算**，差距不是百分之几，是一个数量级。

同一批工作量（一集 5 分钟，30 镜 × 8.6s，重拍率 35%），跑 `cost_calculator.py` 三个档位横比：

| 档位 | 最优卡 | 单集成本 | 折合每秒 |
|---|---|---|---|
| **Vast.ai** | RTX 3090 24GB $0.10/h | **$2.42 – 4.08** | $0.009 – 0.016 |
| Vast.ai | RTX 5090 32GB $0.45/h | $4.04 – 7.02 | $0.016 – 0.027 |
| RunPod Community | RTX 4090 24GB $0.34/h | $4.53 – 7.91 | $0.018 – 0.031 |
| **Paperspace** | RTX A5000 24GB $1.38/h | **$68 – 91** | $0.26 – 0.35 |

Paperspace 比 Vast 3090 贵 **约 22 倍**，比 RunPod 4090 贵 **约 11 倍**。贵在两处：

1. **机时价本身高**：A5000 $1.38/h vs RunPod 4090 $0.34/h —— 而 4090 还比 A5000 快。A6000 差距更夸张：Paperspace $1.89/h vs RunPod $0.33/h。
2. **$39/月的 Growth 订阅是硬门槛**：A5000 及以上的卡不订阅根本开不了。只训一两个 LoRA 时，这笔固定费能占总成本一半以上。（A4000/RTX5000 只要 Pro $8/月，但 A4000 是 16GB，**训不动 Wan 2.2 的 14B**——musubi 的门槛是 24GB。）

**那 Paperspace 什么时候值得用？** 三种情况：

- **试水**：免费层的 RTX4000（8GB）够跑小模型推理、验证工作流，不花钱。
- **需要绝对不被抢占的长训练**：Vast 的竞价实例会被回收。如果你的训练不能断（见下），稳定性值这个钱。
- **已经在付订阅**：团队多人共用时 $39/月摊薄了，边际成本只剩机时。

### 用竞价实例训练的前提：断点续训

最便宜的卡都是可抢占的，所以**没有断点续训就不能用它们**。这套脚本已经处理好了：

```bash
# 第一次跑
OUT_REMOTE=s3://your-bucket/loras bash deploy/train_lora_pod.sh --character anchor

# 被抢占后，重开一台机器，跑同一条命令 —— 自动从最新 state 接着训
OUT_REMOTE=s3://your-bucket/loras bash deploy/train_lora_pod.sh --character anchor
```

机制：
- 训练开着 `--save_state`，每个 epoch 存一次完整状态（优化器动量、lr 调度、随机数），不只是 LoRA 权重
- 重跑时 `plan_resume()` 扫 `output_dir` 里的 `*-state` 目录，**按解析出的训练进度取最大**（不按文件时间——同一轮的多个 state 可能落在同一秒，mtime 排序等于随机挑一个）
- 训练中每 10 分钟把 state 同步到 `OUT_REMOTE`。**不设这个变量，state 只在本地盘，机器被回收就全丢**

一个必须知道的坑：**musubi 的 `--resume` 会再跑满 `max_train_epochs` 个 epoch，而不是补到原定总数**。不调这个值，续一次就多训一倍。`plan_resume()` 返回的是「剩余」epoch 数，脚本已经替换进去了。（来源：kohya-ss/musubi-tuner issue #424 / #927，查证 2026-09-20；另有 issue #667 报告 Wan 续训的 loss/lr 曲线与原始运行对不齐，所以续训产出的 LoRA 建议跟一次完整训练的产物做效果对比再上产线。）

### 给「便宜训练」的推荐配置

| 目标 | 选择 | 理由 |
|---|---|---|
| **最省** | Vast.ai RTX 3090 $0.10/h + 断点续训 + S3 同步 | 24GB 够训 14B；无 fp8，步时约 4090 的 1.6 倍，但单价只有 1/3 |
| **省且稳** | RunPod Community RTX 4090 $0.34/h | 有 fp8，步时短；Community 也可能被抢占，续训照样要开 |
| **不能断** | RunPod Secure 4090 $0.74/h 或 Paperspace A5000 $1.38/h | 贵 2–4 倍买确定性 |
| **别选** | Paperspace A4000 $0.76/h | 16GB，训不动 A14B。便宜但跑不了等于没用 |

自己算一遍：

```bash
.venv/bin/python deploy/cost_calculator.py --episodes 1 --shots 30 --seconds 8.6 --tier vast
.venv/bin/python deploy/cost_calculator.py --episodes 1 --shots 30 --seconds 8.6 --tier paperspace
```

### 先算钱再开机

```bash
# 一集 5 分钟（30 镜 × 8.6s，重拍率 35%），用 Vast 竞价档
.venv/bin/python deploy/cost_calculator.py \
    --episodes 1 --shots 30 --seconds 8.6 --retake 0.35 --tier vast

# 一季 12 集，看 API 路线和自建路线的交叉点
.venv/bin/python deploy/cost_calculator.py --episodes 12 --shots 80 --seconds 6
```

计算器会把**每条假设的可信度**标出来（已核 / 二手 / 估计）。其中吞吐是按硬件规格外推的量级估计，**不是实测**。跑完第一次真实渲染后，用实测值覆盖：

```bash
# 实测「H100 每出 1 秒成品烧多少 GPU 秒」的乐观/悲观端
.venv/bin/python deploy/cost_calculator.py --calibrate 28 45
```

---

## 2. 开机

```bash
# 1) 机器开好后，先体检（不下任何东西，30 秒出结果）
bash deploy/bootstrap_gpu.sh --check

# 2) 冒烟：只下 5B ti2v + VAE + fp8 文本编码器（约 17GB）
bash deploy/bootstrap_gpu.sh --profile minimal --smoke

# 3) 产线主力：Wan2.2 I2V A14B 双专家
bash deploy/bootstrap_gpu.sh --profile i2v

# 4) 训练环境（注意 VAE/T5 用的是 .pth 版本，见 §4）
bash deploy/bootstrap_gpu.sh --profile train
```

脚本是**幂等**的：权重按「本地字节数 == 清单字节数」判断，不按「文件存在」—— 断在 13GB safetensors 中途是最常见的失败，文件在但没法用。重跑只补缺的部分。

退出码有语义（见脚本头部 `EXIT_*` 常量），CI 和 RunPod 模板脚本靠它分流。

磁盘准备：`minimal` 约 20GB，`i2v` 约 70GB，`full` 约 150GB。**RunPod 的容器盘默认很小，权重必须落在 Network Volume 或 `/workspace`**，否则重启就没了。

---

## 3. 出片与训练

### 出片（把 longfilm 指向云上的 ComfyUI）

```bash
export COMFY_URL=http://<pod-ip>:8188
export PYTHONPATH=$PWD/src
.venv/bin/python scripts/run_pipeline.py --work out/work/ep01
```

`providers/comfy_local.py` 会按 `deploy/comfy_workflows/*.json` 的模板注入参数。模板里的模型文件名是占位符，**必须改成你实际下载的文件名**（见 `comfy_workflows/README.md`）。

### 训 LoRA

训练配置由 `src/longfilm/lora.py` 生成，不要手写 toml：

```bash
# 生成 musubi 配置 + 可直接在 pod 上跑的脚本
.venv/bin/python -m longfilm.lora --character lin_lan --emit runpod > /tmp/train.sh
# 或用包装脚本一步到位（同步数据 → 训练 → 回传产物）
bash deploy/train_lora_pod.sh --character lin_lan --steps 2400 --rank 32
```

角色 LoRA 的推荐配方（素材量、rank、步数）见 `lora.recommend_character_lora()`，它会按角色特征的独特性给出不同建议。

---

## 4. 已知的坑

按踩中概率排序。

1. **musubi 训练和 ComfyUI 推理用的不是同一套权重文件。**
   推理用 `diffusion_models/` 下的 fp8_scaled safetensors + umt5 safetensors + `wan2.2_vae.safetensors`；
   训练的 DiT 用同一批 safetensors，但 **VAE 必须用 `Wan2.1_VAE.pth`**（musubi `docs/wan.md` 明写 Wan2.2_VAE.pth 与 14B 不兼容），**T5 必须用 `models_t5_umt5-xxl-enc-bf16.pth`**（.safetensors 版本 musubi 不吃）。
   `models.yaml` 把两者分成不同 profile，不共用，就是因为这条。

2. **FlashVSR 官方只保证 A100/A800。** 它是整条后处理链的吞吐命脉（8 分钟片约 11 分钟 vs SeedVR2-7B 的 8.5~9.6 小时）。若你只有 4090，**必须先做 Block-Sparse Attention 编译验证**，否则整个成本模型作废。

3. **自训 LoRA 与少步加速不能简单叠加。** DART（arXiv 2609.20051）实测：为满步轨迹训的 LoRA 挂到 4 步 Wan2.2 上，macro functional retention 为 **−0.4644**（功能效果被反转）。另外 Wan2.2-Animate 的 README 明确写「不推荐使用 Wan2.2 训练的 LoRA」。满步推理成本约为 4 步的 20 倍 —— **这个二选一必须在立项时定，不能等训完才发现**。

4. **闭源 API 的产物 24 小时过期。** 火山方舟返回的 `video_url` 是对象存储直链，成功后 24 小时失效。必须做成原子步骤：回调 → 立刻下载 → 落自有 OSS → 连同 seed/prompt/参考图指纹入库。隔夜不转存 = 全额重烧。

5. **竞价实例会被抢占。** Vast.ai 便宜一半，代价是随时被收回。训练要开 checkpoint，出片要让队列能重新领取（`queue.py` 的租约超时机制）。

6. **许可证会卡死商用。** 几个高频踩中的：MiniMax H3 排除美/欧/英/韩本地部署且禁蒸馏；FLUX.2 [dev] 非商用；F5-TTS / MaskGCT / XTTS-v2 / Fish-Speech 非商用；Upscale-A-Video 非商用；SimpleTuner 是 AGPL-3.0（不要嵌进闭源服务）。详见 [../docs/PLAN.md](../docs/PLAN.md) §4。

---

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `bootstrap_gpu.sh` | 一键开机：体检 → 装 ComfyUI → 下权重 → 起服务 → 冒烟 |
| `models.yaml` | 权重清单。size 来自 HF API 实测，幂等下载的依据 |
| `prices.yaml` | GPU 与 API 价目表，每条带来源与可信度 |
| `cost_calculator.py` | 成本对比计算器，支持 `--calibrate` 用实测吞吐覆盖估计 |
| `train_lora_pod.sh` | LoRA 训练包装：同步数据 → 调 `lora.py` 生成配置 → 训练 → 回传 |
| `runpod_serverless_handler.py` | RunPod serverless worker，按需扩缩容出片 |
| `Dockerfile` / `docker-compose.yml` | 容器化部署（ComfyUI + longfilm API） |
| `comfy_workflows/` | ComfyUI API 格式的 workflow 模板 |
