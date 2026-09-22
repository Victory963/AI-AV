"""LoRA 训练工单生成器 —— 把「要训一个什么」写成机器可执行的东西。

产线里训练是唯一一件**不在本机发生**的事：本机没有 GPU，训练必然发生在
租来的云卡上。所以这个模块的产物不是「训练代码」，而是一张**工单**：
一份配置 + 一段能原样贴进云机器的 bash + 一个成本预算。
工单可序列化、可 diff、可回填进 LoRASpec，训完之后能凭 dataset_hash 复现。

支持两套开源训练栈，它们的配置形态**不一样**，不要互相套：

1) musubi-tuner（kohya-ss）
   - 训练超参走**命令行参数**，数据集走一份 TOML（docs/dataset_config.md）。
   - 查证 2026-09-19，https://github.com/kohya-ss/musubi-tuner
     README 原文：“VRAM: 12GB or more recommended for image training, 24GB or
     more for video training … For 12GB, use a resolution of 960x544 or lower
     and use memory-saving options such as `--blocks_to_swap`, `--fp8_llm`,
     etc.” / “Main Memory: 64GB or more recommended, 32GB + swap may work”。
   - docs/wan.md 的官方示例命令（原样摘录，查证 2026-09-19）：
       accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \\
         src/musubi_tuner/wan_train_network.py \\
         --task t2v-14B --dit path/to/wan2.1_xxx_bf16.safetensors \\
         --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base \\
         --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \\
         --max_data_loader_n_workers 2 --persistent_data_loader_workers \\
         --network_module networks.lora_wan --network_dim 32 \\
         --timestep_sampling shift --discrete_flow_shift 3.0 \\
         --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \\
         --output_dir path/to/output_dir --output_name name-of-lora
   - Wan 任务名（docs/wan.md，查证 2026-09-19）：
       Wan2.1: t2v-1.3B, t2v-14B, i2v-14B, t2i-14B, t2v-1.3B-FC, t2v-14B-FC, i2v-14B-FC
       Wan2.2: t2v-A14B, i2v-A14B
   - 训练前必须先跑两步预缓存：wan_cache_latents.py（--vae，I2V 加 --i2v，
     Wan2.1 I2V 还要 --clip）与 wan_cache_text_encoder_outputs.py（--t5，
     显存紧可加 --fp8_t5）。
   - docs/advanced_config.md（查证 2026-09-19）确认支持 --network_alpha
     （示例 network_dim = 32 / network_alpha = 16）与 --network_args
     （原文 “specify the arguments in the form of key=value in --network_args”，
     示例 --network_args "loraplus_lr_ratio=4"）。

2) diffusion-pipe（tdrussell）
   - 训练超参和数据集各一份 TOML。字段以仓库 examples/ 为准。
   - examples/wan_14b_min_vram.toml 原文首行（查证 2026-09-19）：
     “This configuration should allow you to train Wan 14b t2v on 512x512x81
     sized videos (or varying aspect ratios of the same size), with 24GB VRAM.”
     其关键取值：blocks_to_swap = 32、transformer_dtype = 'float8'、
     activation_checkpointing = 'unsloth'、[adapter] rank = 32、
     [optimizer] type = 'AdamW8bitKahan' / lr = 2e-5 / betas = [0.9, 0.99] /
     weight_decay = 0.01。
   - 数据集侧字段（examples/dataset.toml，查证 2026-09-19）：resolutions、
     enable_ar_bucket、min_ar、max_ar、num_ar_buckets、frame_buckets、
     [[directory]] path / num_repeats。原文提醒：“Videos will be assigned to
     the longest frame bucket possible, such that the video is still greater
     than or equal to the frame bucket length.”

⚠ 两套栈的学习率差一个数量级：musubi 官方示例 2e-4（adamw8bit），
diffusion-pipe 官方示例 2e-5（AdamW8bitKahan）。这不是笔误，是优化器与
参数化差异导致的，**照搬另一边的 lr 会直接训崩或训不动**。
本模块的 default_lr_for() 按栈给不同默认值，改的时候请连带改对应栈。
"""

from __future__ import annotations

import logging
import re
import math
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .schema import CharacterBible, LoRASpec, Shot, ShotSize

log = logging.getLogger(__name__)

Stack = Literal["musubi", "diffusion_pipe"]
Platform = Literal["runpod", "vast", "local"]
GpuName = Literal["H100", "H200", "A100", "4090", "5090", "L40S"]


# ---------------------------------------------------------------- 基座模型


class ModelFamily(str, Enum):
    WAN21 = "wan2.1"
    WAN22 = "wan2.2"
    HUNYUAN = "hunyuanvideo"


@dataclass(frozen=True)
class BaseModelSpec:
    """一个可训基座。musubi 用 task 名，diffusion-pipe 用 [model].type。"""

    key: str
    family: ModelFamily
    musubi_task: str                  # --task 的取值，见 docs/wan.md
    dpipe_type: str                   # [model] type 的取值
    params_b: float                   # 参数量（B），只用于成本/显存的粗量级估计
    needs_clip: bool = False          # Wan2.1 I2V 需要 CLIP（docs/wan.md）
    moe_two_stage: bool = False       # Wan2.2 A14B 是双专家，musubi 有 --offload_inactive_dit
    note: str = ""


BASE_MODELS: dict[str, BaseModelSpec] = {
    "wan2.1-t2v-1.3b": BaseModelSpec("wan2.1-t2v-1.3b", ModelFamily.WAN21, "t2v-1.3B",
                                     "wan", 1.3, note="小基座，单卡 12-16GB 可训，适合先跑通流程"),
    "wan2.1-t2v-14b": BaseModelSpec("wan2.1-t2v-14b", ModelFamily.WAN21, "t2v-14B", "wan", 14.0),
    "wan2.1-i2v-14b": BaseModelSpec("wan2.1-i2v-14b", ModelFamily.WAN21, "i2v-14B", "wan", 14.0,
                                    needs_clip=True,
                                    note="Wan2.1 I2V 的 latent 预缓存需要 --clip"),
    "wan2.2-t2v-a14b": BaseModelSpec("wan2.2-t2v-a14b", ModelFamily.WAN22, "t2v-A14B", "wan", 14.0,
                                     moe_two_stage=True),
    "wan2.2-i2v-a14b": BaseModelSpec("wan2.2-i2v-a14b", ModelFamily.WAN22, "i2v-A14B", "wan", 14.0,
                                     moe_two_stage=True,
                                     note="产线主力：I2V 才能吃首帧，接得上续写链"),
    "hunyuanvideo-t2v": BaseModelSpec("hunyuanvideo-t2v", ModelFamily.HUNYUAN, "",
                                      "hunyuan-video", 13.0,
                                      note="musubi 有独立的 hv_train_network.py，不用 --task"),
}


def default_lr_for(stack: Stack) -> float:
    """两套栈的官方示例 lr 差一个数量级，默认值必须跟着栈走。"""
    return 2e-4 if stack == "musubi" else 2e-5


def default_optimizer_for(stack: Stack) -> str:
    return "adamw8bit" if stack == "musubi" else "AdamW8bitKahan"


# ---------------------------------------------------------------- 训练工单


class MemoryOpts(BaseModel):
    """显存优化开关。三件事换显存：重算、低精度、块交换。

    代价各不相同，所以不合成一个「省显存等级」——
    gradient checkpointing 换的是算力（约 +30% 步时），
    fp8 换的是数值精度，block swap 换的是 PCIe 带宽与主存。
    """

    gradient_checkpointing: bool = True
    fp8_base: bool = True               # musubi --fp8_base / dpipe transformer_dtype='float8'
    fp8_t5: bool = False                # musubi --fp8_t5，只在文本编码器缓存阶段省
    blocks_to_swap: int = Field(default=0, ge=0, le=40)
    offload_inactive_dit: bool = False  # 仅 Wan2.2 双专家有意义
    activation_checkpointing: str = "unsloth"   # diffusion-pipe 侧取值
    transformer_dtype: str = "float8"
    save_dtype: str = "bfloat16"

    def describe(self) -> str:
        on = [n for n, v in (
            ("gradient_checkpointing", self.gradient_checkpointing),
            ("fp8_base", self.fp8_base), ("fp8_t5", self.fp8_t5),
            ("offload_inactive_dit", self.offload_inactive_dit),
        ) if v]
        if self.blocks_to_swap:
            on.append(f"blocks_to_swap={self.blocks_to_swap}")
        return ", ".join(on) or "无（全精度全驻留）"


class LoRATrainingJob(BaseModel):
    """一张完整的训练工单。序列化后就是这次训练的全部可复现信息。"""

    name: str
    stack: Stack = "musubi"
    base_model: str = "wan2.2-i2v-a14b"

    # 数据
    dataset_dir: str                      # 切片后的 videos/ 目录
    dataset_config: str = ""              # 由 emit_* 写出的 toml 路径
    cache_dir: str = ""
    dataset_hash: str = ""                # 来自 distill.DistillDataset.dataset_hash()
    dataset_remote: str = ""              # 云机器上拉数据的地址（s3:// / https://）
    n_clips: int = Field(default=0, ge=0)
    num_repeats: int = Field(default=1, ge=1)

    # 权重路径（云机器上的绝对路径，由脚本下载后填）
    dit_path: str = "/workspace/models/dit.safetensors"
    vae_path: str = "/workspace/models/vae.safetensors"
    t5_path: str = "/workspace/models/t5.pth"
    clip_path: str = ""
    dpipe_ckpt_path: str = "/workspace/models/Wan2.2-I2V-A14B"

    # LoRA 结构
    rank: int = Field(default=32, ge=1, le=256)
    alpha: float = Field(default=16.0, gt=0)
    network_module: str = "networks.lora_wan"
    network_args: list[str] = Field(default_factory=list)

    # 优化
    lr: float = Field(default=0.0, ge=0.0)     # 0 = 按 stack 取默认
    optimizer: str = ""
    micro_batch_size: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    n_gpus: int = Field(default=1, ge=1)
    epochs: int = Field(default=16, ge=1)
    max_train_steps: int | None = None          # 给定则覆盖 epochs 推导
    warmup_steps: int = Field(default=10, ge=0)
    gradient_clipping: float = 1.0
    timestep_sampling: str = "shift"
    discrete_flow_shift: float = 3.0
    seed: int = 42

    # 桶
    resolution: tuple[int, int] = (960, 544)
    target_frames: list[int] = Field(default_factory=lambda: [49])

    # 保存
    output_dir: str = "/workspace/output"
    output_name: str = ""
    save_every_n_epochs: int = Field(default=1, ge=1)
    # 存完整训练状态（优化器动量、lr 调度器、随机数状态），不只是 LoRA 权重。
    # 只存权重的话，被抢占后只能从头训 —— 而最便宜的卡恰恰都是可抢占的，
    # 所以这个开关默认开着：多占几 GB 盘，换的是「能用 $0.1/h 的卡」。
    save_state: bool = True
    resume_from: str = ""        # musubi 的 --resume 指向 state 目录
    checkpoint_every_n_minutes: int = Field(default=60, ge=1)
    upload_to: str = ""                         # 产物回传地址

    memory: MemoryOpts = Field(default_factory=MemoryOpts)
    trigger_word: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _fill_defaults(self) -> LoRATrainingJob:
        if self.base_model not in BASE_MODELS:
            raise ValueError(
                f"未知基座 {self.base_model!r}；已登记 {sorted(BASE_MODELS)}"
            )
        if not self.lr:
            self.lr = default_lr_for(self.stack)
        if not self.optimizer:
            self.optimizer = default_optimizer_for(self.stack)
        if not self.output_name:
            self.output_name = self.name
        # alpha > rank 会把等效 lr 放大，rank 很大时极易训崩；这里只告警不强改，
        # 因为「alpha=rank」在某些配方里是故意的。
        if self.alpha > self.rank:
            log.warning("job %s: alpha(%.1f) > rank(%d)，等效学习率被放大 %.1fx，确认这是有意的",
                        self.name, self.alpha, self.rank, self.alpha / self.rank)
        return self

    @property
    def spec(self) -> BaseModelSpec:
        return BASE_MODELS[self.base_model]

    def steps_per_epoch(self) -> int:
        eff = max(1, self.n_clips * self.num_repeats)
        per = self.micro_batch_size * self.grad_accum * self.n_gpus
        return max(1, math.ceil(eff / per))

    def total_steps(self) -> int:
        return self.max_train_steps or self.steps_per_epoch() * self.epochs

    def to_lora_spec(self, path: str) -> LoRASpec:
        """训完回填进 CharacterBible.lora，供 router 挂载。"""
        return LoRASpec(
            base_model=self.base_model, path=path,
            trigger_word=self.trigger_word or self.name,
            trained_steps=self.total_steps(), dataset_hash=self.dataset_hash or None,
        )


# ---------------------------------------------------------------- 配置生成


def emit_musubi_config(job: LoRATrainingJob) -> str:
    """musubi-tuner 的**数据集** TOML。

    musubi 的训练超参不走 toml 而是走命令行（见 docs/wan.md 示例），
    所以这里把等效命令原样写进文件头注释 —— 配置和命令分家最容易踩的坑，
    就是改了 toml 以为改了训练参数。
    """
    frames = ", ".join(str(f) for f in job.target_frames)
    cmd = " \\\n#   ".join(_wrap_argv(musubi_train_argv(job)))
    return f"""# musubi-tuner dataset config — job: {job.name}
# 生成者 longfilm.lora.emit_musubi_config；字段依据 docs/dataset_config.md（查证 2026-09-19）
#
# ⚠ musubi 的训练超参不在本文件里，在命令行。本工单对应的训练命令是：
#   {cmd}
#
[general]
resolution = [{job.resolution[0]}, {job.resolution[1]}]
caption_extension = ".txt"
batch_size = {job.micro_batch_size}
num_repeats = {job.num_repeats}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "{Path(job.dataset_dir).as_posix()}"
cache_directory = "{Path(job.cache_dir or (Path(job.dataset_dir).parent / 'cache')).as_posix()}"
target_frames = [{frames}]
frame_extraction = "chunk"
"""


def musubi_cache_argv(job: LoRATrainingJob) -> tuple[list[str], list[str]]:
    """返回 (latent 预缓存命令, 文本编码器预缓存命令)。

    这两步必须在训练前跑完，否则 wan_train_network.py 找不到 cache 直接退出。
    """
    lat = ["python", "src/musubi_tuner/wan_cache_latents.py",
           "--dataset_config", job.dataset_config or "dataset.toml",
           "--vae", job.vae_path]
    if "i2v" in job.spec.musubi_task.lower():
        lat.append("--i2v")
        if job.spec.needs_clip:
            if not job.clip_path:
                raise ValueError(
                    f"{job.base_model} 的 latent 预缓存需要 --clip，但 job.clip_path 为空"
                )
            lat += ["--clip", job.clip_path]
    lat.append("--vae_cache_cpu")   # 显存换主存，云机器上主存一般够

    txt = ["python", "src/musubi_tuner/wan_cache_text_encoder_outputs.py",
           "--dataset_config", job.dataset_config or "dataset.toml",
           "--t5", job.t5_path, "--batch_size", "16"]
    if job.memory.fp8_t5:
        txt.append("--fp8_t5")
    return lat, txt


def musubi_train_argv(job: LoRATrainingJob) -> list[str]:
    """拼 accelerate launch 命令。参数名对齐 docs/wan.md（查证 2026-09-19）。"""
    m = job.memory
    argv = [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "src/musubi_tuner/wan_train_network.py",
        "--task", job.spec.musubi_task,
        "--dit", job.dit_path,
        "--vae", job.vae_path,
        "--t5", job.t5_path,
        "--dataset_config", job.dataset_config or "dataset.toml",
        "--sdpa", "--mixed_precision", "bf16",
        "--optimizer_type", job.optimizer,
        "--learning_rate", f"{job.lr:g}",
        "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
        "--network_module", job.network_module,
        "--network_dim", str(job.rank),
        "--network_alpha", f"{job.alpha:g}",
        "--timestep_sampling", job.timestep_sampling,
        "--discrete_flow_shift", f"{job.discrete_flow_shift:g}",
        "--save_every_n_epochs", str(job.save_every_n_epochs),
        *(["--save_state"] if job.save_state else []),
        *(["--resume", job.resume_from] if job.resume_from else []),
        "--seed", str(job.seed),
        "--output_dir", job.output_dir,
        "--output_name", job.output_name,
    ]
    if job.max_train_steps:
        argv += ["--max_train_steps", str(job.max_train_steps)]
    else:
        argv += ["--max_train_epochs", str(job.epochs)]
    if m.fp8_base:
        argv.append("--fp8_base")
    if m.gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if m.blocks_to_swap:
        argv += ["--blocks_to_swap", str(m.blocks_to_swap)]
    if m.offload_inactive_dit:
        if not job.spec.moe_two_stage:
            raise ValueError(
                f"--offload_inactive_dit 只对 Wan2.2 双专家有意义，{job.base_model} 不是"
            )
        argv.append("--offload_inactive_dit")
    for a in job.network_args:
        argv += ["--network_args", a]
    return argv



def plan_resume(job: LoRATrainingJob) -> tuple[str, int]:
    """扫 output_dir 找最新的 state，返回 (state 路径, 还需要训几个 epoch)。

    为什么要算「还需要几个」而不是直接续：musubi 的 --resume 有个反直觉行为 ——
    它从 state 接着训，但会**再跑满 max_train_epochs 个 epoch**，而不是补到
    原定总数。不调这个值，续一次就多训一倍。（kohya-ss/musubi-tuner
    issue #424 / #927，查证 2026-09-20）

    另两个已知坑，写在这里省得再踩：
      - 续训后日志里的 epoch 从 1 重新计数，内部进度是对的，别被吓到；
      - 有用户报告 Wan 续训的 loss/lr 曲线与原始运行对不齐（issue #667），
        所以续训产出的 LoRA 建议跟一次完整训练的产物做效果对比再上产线。

    没找到 state 就返回 ("", 原定 epochs)，调用方照常从头训。
    """
    out = Path(job.output_dir)
    if not out.is_dir():
        return "", job.epochs
    # musubi 的 state 目录形如 <name>-000004-state / <name>-step00001200-state。
    # 按**解析出的进度**取最大，不按 mtime —— 同一轮里多个 state 可能落在同一秒，
    # mtime 相同时排序结果取决于 iterdir 的返回顺序，等于随机挑一个续。
    spe = max(job.steps_per_epoch(), 1)
    cands: list[tuple[int, Path]] = []
    for d in out.iterdir():
        if not (d.is_dir() and d.name.endswith("-state")):
            continue
        m = re.search(r"-(?:step)?0*(\d+)-state$", d.name)
        if not m:
            continue
        n = int(m.group(1))
        # 目录名里可能是 epoch 序号，也可能是 step 数；按量级区分
        done = n if n <= job.epochs * 2 else max(1, n // spe)
        cands.append((done, d))
    if not cands:
        return "", job.epochs
    done_epochs, latest = max(cands, key=lambda t: (t[0], t[1].name))
    remaining = max(1, job.epochs - done_epochs)
    log.info("发现训练状态 %s：已完成约 %d epoch，还需 %d", latest.name, done_epochs, remaining)
    return str(latest), remaining

def emit_diffusion_pipe_config(job: LoRATrainingJob) -> str:
    """diffusion-pipe 的训练 TOML。字段依据 examples/wan_14b_min_vram.toml（查证 2026-09-19）。"""
    m = job.memory
    ds_path = job.dataset_config or str(Path(job.output_dir) / "dataset.toml")
    model_block = (
        f"type = '{job.spec.dpipe_type}'\n"
        f"ckpt_path = '{job.dpipe_ckpt_path}'\n"
        if job.spec.dpipe_type == "wan" else
        f"type = '{job.spec.dpipe_type}'\n"
        f"transformer_path = '{job.dit_path}'\n"
        f"vae_path = '{job.vae_path}'\n"
    )
    return f"""# diffusion-pipe config — job: {job.name}
# 生成者 longfilm.lora.emit_diffusion_pipe_config
# 字段依据 examples/wan_14b_min_vram.toml / examples/main_example.toml（查证 2026-09-19）
output_dir = '{job.output_dir}'
dataset = '{Path(ds_path).as_posix()}'

epochs = {job.epochs}
micro_batch_size_per_gpu = {job.micro_batch_size}
pipeline_stages = {job.n_gpus}
gradient_accumulation_steps = {job.grad_accum}
gradient_clipping = {job.gradient_clipping:g}
warmup_steps = {job.warmup_steps}

eval_every_n_epochs = 1
eval_before_first_step = true
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1

save_every_n_epochs = {job.save_every_n_epochs}
checkpoint_every_n_minutes = {job.checkpoint_every_n_minutes}
activation_checkpointing = '{m.activation_checkpointing}'
partition_method = 'parameters'
save_dtype = '{m.save_dtype}'
caching_batch_size = 1
steps_per_print = 1
# single_beginning：每条素材只取开头一段。我们在 distill.to_training_set 里
# 已经按 chunk 切好片了，这里再随机取会把切片逻辑做两遍。
video_clip_mode = 'single_beginning'
blocks_to_swap = {m.blocks_to_swap}

[model]
{model_block}dtype = 'bfloat16'
transformer_dtype = '{m.transformer_dtype}'
timestep_sample_method = 'logit_normal'

[adapter]
type = 'lora'
rank = {job.rank}
dtype = 'bfloat16'

[optimizer]
type = '{job.optimizer}'
lr = {job.lr:g}
betas = [0.9, 0.99]
weight_decay = 0.01
stabilize = false

[monitoring]
enable_wandb = false
"""


def emit_diffusion_pipe_dataset(job: LoRATrainingJob) -> str:
    """diffusion-pipe 的 dataset.toml。

    frame_buckets 必须包含切片实际帧数，否则「视频落到不超过自身长度的最长桶」
    这条规则会把所有切片都丢掉（官方 dataset.toml 原文提醒过这一点）。
    """
    fb = ", ".join(str(f) for f in sorted({1, *job.target_frames}))
    return f"""# diffusion-pipe dataset — job: {job.name}
resolutions = [[{job.resolution[0]}, {job.resolution[1]}]]
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7
frame_buckets = [{fb}]

[[directory]]
path = '{Path(job.dataset_dir).as_posix()}'
num_repeats = {job.num_repeats}
"""


def _wrap_argv(argv: list[str], per_line: int = 4) -> list[str]:
    """把 argv 折成人能读的多行。只用于注释和脚本排版。"""
    q = [shlex.quote(a) for a in argv]
    lines, buf = [], []
    for tok in q:
        buf.append(tok)
        if tok.startswith("--") is False and len(buf) >= per_line:
            lines.append(" ".join(buf))
            buf = []
    if buf:
        lines.append(" ".join(buf))
    return lines


# ---------------------------------------------------------------- 训练脚本


_PREAMBLE = """#!/usr/bin/env bash
# {name} —— 由 longfilm.lora.emit_train_script 生成，平台 {platform}
# 生成日期 2026-09-19
set -euo pipefail

# 密钥一律从环境变量读，脚本里不落任何明文，也不 echo 出来。
# 缺失时立刻失败，而不是训到一半上传失败白烧几十美元。
: "${{HF_TOKEN:?请先 export HF_TOKEN（下载基座权重用）}}"
{secret_checks}
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
WORK={workdir}
mkdir -p "$WORK/models" "$WORK/data" "{output_dir}"
cd "$WORK"
"""


def emit_train_script(job: LoRATrainingJob, *, platform: Platform = "runpod") -> str:
    """生成能直接贴进云 GPU 机器跑的 bash：环境 → 数据 → 训练 → 回传。

    三个平台的差别只在「盘在哪、怎么进程守护」，训练段完全相同 ——
    所以平台差异集中在 workdir 与收尾，不要在训练段里写 if platform。
    """
    needs_s3 = job.dataset_remote.startswith("s3://") or job.upload_to.startswith("s3://")
    secret_checks = ""
    if needs_s3:
        secret_checks = (
            ': "${AWS_ACCESS_KEY_ID:?请先 export AWS_ACCESS_KEY_ID}"\n'
            ': "${AWS_SECRET_ACCESS_KEY:?请先 export AWS_SECRET_ACCESS_KEY}"\n'
        )
    workdir = {"runpod": "/workspace", "vast": "/workspace", "local": "$HOME/longfilm-train"}[platform]

    head = _PREAMBLE.format(name=job.name, platform=platform, secret_checks=secret_checks,
                            workdir=workdir, output_dir=job.output_dir)

    if platform == "local":
        env = """
# --- 环境（本地：假定已有 CUDA 与 python3.10+）
python3 -m venv .venv && . .venv/bin/activate
"""
    else:
        env = f"""
# --- 环境（{platform}：容器里通常已有 torch，只补训练栈依赖）
apt-get update -qq && apt-get install -y -qq git git-lfs aria2 >/dev/null
python -m pip install -q --upgrade pip
"""

    if job.stack == "musubi":
        env += """
if [ ! -d musubi-tuner ]; then
  git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git
fi
cd musubi-tuner
python -m pip install -q -e .
python -m pip install -q accelerate bitsandbytes hf_transfer
accelerate config default --mixed_precision bf16
"""
    else:
        env += """
if [ ! -d diffusion-pipe ]; then
  git clone --depth 1 --recurse-submodules https://github.com/tdrussell/diffusion-pipe.git
fi
cd diffusion-pipe
python -m pip install -q -r requirements.txt
python -m pip install -q hf_transfer
"""

    # --- 权重与数据
    dl = ["\n# --- 基座权重", "# TODO(2026-09-19): 确认仓库名与文件名后再跑；",
          "#   本模块不猜 HuggingFace repo id，猜错会静默下载到错误的权重。",
          f"#   目标路径：dit={job.dit_path} vae={job.vae_path} t5={job.t5_path}"]
    if job.clip_path:
        dl.append(f"#   clip={job.clip_path}（Wan2.1 I2V 的 latent 预缓存必需）")
    dl.append('if [ ! -f "%s" ]; then echo "缺少基座权重，请先按上面的 TODO 下载" >&2; exit 2; fi'
              % job.dit_path)

    dl.append("\n# --- 训练数据")
    if job.dataset_remote.startswith("s3://"):
        dl.append(f'aws s3 sync "{job.dataset_remote}" "{job.dataset_dir}" --only-show-errors')
    elif job.dataset_remote:
        dl.append(f'aria2c -x8 -d "$WORK/data" -o dataset.tar.zst "{job.dataset_remote}"')
        dl.append(f'mkdir -p "{job.dataset_dir}" && '
                  f'tar --zstd -xf "$WORK/data/dataset.tar.zst" -C "{job.dataset_dir}"')
    else:
        dl.append(f'# 无 dataset_remote：假定 {job.dataset_dir} 已由外部挂载')
    dl.append(f'test -n "$(ls -A "{job.dataset_dir}" 2>/dev/null)" || '
              f'{{ echo "数据目录为空: {job.dataset_dir}" >&2; exit 3; }}')
    n_clips = job.n_clips or 0
    dl.append(f'echo "切片数：$(ls -1 "{job.dataset_dir}"/*.mp4 2>/dev/null | wc -l)'
              f'（工单预期 {n_clips}）"')

    # --- 训练
    if job.stack == "musubi":
        lat, txt = musubi_cache_argv(job)
        train = "\n".join([
            "\n# --- 预缓存（必须，训练脚本不会自动做）",
            " ".join(shlex.quote(a) for a in lat),
            " ".join(shlex.quote(a) for a in txt),
            "\n# --- 训练",
            " \\\n  ".join(_wrap_argv(musubi_train_argv(job))),
        ])
    else:
        train = "\n".join([
            "\n# --- 训练",
            f"NCCL_P2P_DISABLE=1 deepspeed --num_gpus={job.n_gpus} train.py "
            f"--deepspeed --config {shlex.quote(job.dataset_config or 'config.toml')}",
        ])

    tail = [f'\n# --- 产物回传（{job.total_steps()} 步，预计 {job.save_every_n_epochs} epoch 存一次）']
    if job.upload_to.startswith("s3://"):
        tail.append(f'aws s3 sync "{job.output_dir}" "{job.upload_to}" --only-show-errors')
    elif job.upload_to:
        tail.append(f'# TODO：upload_to={job.upload_to} 不是 s3://，请自行补上传命令')
    else:
        tail.append('echo "未配置 upload_to：产物只留在本机 %s" >&2' % job.output_dir)
    tail.append(f'ls -lh "{job.output_dir}"')
    if platform in ("runpod", "vast"):
        tail.append('# 训练结束后建议立刻停机；按秒计费，忘关机比训练本身还贵。')
        tail.append('# runpodctl stop pod "$RUNPOD_POD_ID"   # 按需打开')

    return "\n".join([head, env, *dl, train, *tail, ""])


# ---------------------------------------------------------------- 成本估算


#: RunPod 公开价目（USD/小时），抓取自 https://www.runpod.io/pricing，查证日期 2026-09-19。
#: (Community Cloud, Secure Cloud)。Community 便宜但可用性与盘速不保证。
GPU_HOURLY_USD: dict[str, tuple[float, float]] = {
    "H100": (2.69, 3.49),      # H100 SXM
    "H200": (3.59, 4.59),
    "A100": (1.39, 1.59),      # A100 SXM 80GB
    "4090": (0.34, 0.74),
    "5090": (0.69, 0.99),
    "L40S": (0.79, 1.09),
}
GPU_VRAM_GB: dict[str, int] = {"H100": 80, "H200": 141, "A100": 80,
                               "4090": 24, "5090": 32, "L40S": 48}

#: 相对吞吐（以 H100 = 1.0）。只用于把一个基准步时换算到别的卡，
#: 不是官方 benchmark。依据是各卡 bf16/fp8 张量算力与显存带宽的公开规格量级。
#: TODO(2026-09-19)：未检索到 Wan 14B LoRA 在这几张卡上的公开步时基准，
#: 请用 --max_train_steps 50 跑一次真实计时，再用 measured_s_per_step 覆盖。
GPU_RELATIVE_THROUGHPUT: dict[str, float] = {
    "H100": 1.0, "H200": 1.15, "A100": 0.45, "5090": 0.45, "L40S": 0.33, "4090": 0.30,
}

#: H100 上、14B 基座、batch=1 的**参考步时**（秒/步），按 (帧数 × 像素) 线性外推。
#: 基准点：49 帧 @ 960x544。这是量级估计，不是实测。
_REF_S_PER_STEP_14B = 3.2
_REF_FRAMES = 49
_REF_PIXELS = 960 * 544


@dataclass
class CostEstimate:
    """成本估算。一律给区间，不给单点 —— 单点数字会被当成承诺。"""

    gpu: str
    tier: str
    hourly_usd: float
    total_steps: int
    s_per_step_low: float
    s_per_step_high: float
    hours_low: float
    hours_high: float
    cost_usd_low: float
    cost_usd_high: float
    vram_gb: int
    vram_warning: str = ""
    assumptions: list[str] = field(default_factory=list)

    def render(self) -> str:
        L = [
            f"训练成本估算 · {self.gpu}（{self.tier}，${self.hourly_usd:.2f}/h）",
            f"  步数 {self.total_steps}",
            f"  步时 {self.s_per_step_low:.1f}–{self.s_per_step_high:.1f} s/step",
            f"  耗时 {self.hours_low:.1f}–{self.hours_high:.1f} 小时",
            f"  花费 ${self.cost_usd_low:.1f}–${self.cost_usd_high:.1f}",
            f"  显存 {self.vram_gb}GB{'（' + self.vram_warning + '）' if self.vram_warning else ''}",
        ]
        L += ["  假设："] + [f"    - {a}" for a in self.assumptions]
        return "\n".join(L)


def estimate_cost(
    job: LoRATrainingJob,
    gpu: GpuName = "H100",
    *,
    tier: Literal["community", "secure"] = "community",
    measured_s_per_step: float | None = None,
    uncertainty: float = 0.6,
) -> CostEstimate:
    """按 RunPod 实价估训练成本与耗时，给区间。

    measured_s_per_step 给了就以实测为准，不确定度收窄到 ±20% ——
    实测过的东西不该继续按 ±60% 报，那等于没估。
    """
    if gpu not in GPU_HOURLY_USD:
        raise ValueError(f"未登记的 GPU {gpu!r}；已登记 {sorted(GPU_HOURLY_USD)}")
    hourly = GPU_HOURLY_USD[gpu][0 if tier == "community" else 1]
    steps = job.total_steps()

    if measured_s_per_step:
        mid = measured_s_per_step
        band = 0.2
        basis = f"实测步时 {mid:.2f}s/step"
    else:
        frames = max(job.target_frames)
        pixels = job.resolution[0] * job.resolution[1]
        scale = (frames / _REF_FRAMES) * (pixels / _REF_PIXELS)
        scale *= job.spec.params_b / 14.0
        scale *= job.micro_batch_size * job.grad_accum
        mid = _REF_S_PER_STEP_14B * scale / GPU_RELATIVE_THROUGHPUT[gpu]
        if job.memory.gradient_checkpointing:
            mid *= 1.3        # 重算换显存，典型 +30% 步时
        if job.memory.blocks_to_swap:
            # block swap 走 PCIe，换进换出随块数近似线性增加开销
            mid *= 1.0 + 0.02 * job.memory.blocks_to_swap
        mid /= max(1, job.n_gpus) ** 0.85   # 多卡不是线性加速
        band = uncertainty
        basis = (f"外推自参考点「14B / {_REF_FRAMES}帧 / 960x544 / H100 ≈ "
                 f"{_REF_S_PER_STEP_14B}s/step」，未实测")

    lo, hi = mid * (1 - band), mid * (1 + band)
    h_lo, h_hi = steps * lo / 3600, steps * hi / 3600

    vram = GPU_VRAM_GB[gpu]
    # 大基座 + 小卡永远给一句话：没开优化要说"会 OOM"，开了也要说"这是紧配置"。
    # 只在没开优化时才提示，等于让已经按建议配好的人以为自己有余量 ——
    # 而真正会咬人的是「配好了，然后把分辨率从 544 提到 720」。
    warn = ""
    big = job.spec.params_b >= 13
    if big and vram < 40:
        swap = job.memory.blocks_to_swap
        if not swap:
            warn = ("14B 基座在 24-32GB 卡上必须开 blocks_to_swap + fp8，"
                    "否则 OOM；diffusion-pipe 官方 24GB 配置用的是 blocks_to_swap=32")
        else:
            warn = (f"已开 blocks_to_swap={swap}；{vram}GB 上训 14B 仍是紧配置，"
                    "再提分辨率或帧数要同步加 swap 块数或降 batch")
    elif vram < 24:
        warn = "musubi README 建议视频训练 ≥24GB"

    return CostEstimate(
        gpu=gpu, tier=tier, hourly_usd=hourly, total_steps=steps,
        s_per_step_low=round(lo, 2), s_per_step_high=round(hi, 2),
        hours_low=round(h_lo, 2), hours_high=round(h_hi, 2),
        cost_usd_low=round(h_lo * hourly, 2), cost_usd_high=round(h_hi * hourly, 2),
        vram_gb=vram, vram_warning=warn,
        assumptions=[
            basis,
            f"不确定度 ±{band:.0%}",
            f"价格 RunPod {tier} ${hourly:.2f}/h，抓取日期 2026-09-19（价格会变）",
            f"步数 = ceil({job.n_clips}×{job.num_repeats} / "
            f"({job.micro_batch_size}×{job.grad_accum}×{job.n_gpus})) × {job.epochs} epoch",
            "不含数据上传/权重下载时间（14B 权重约 30-60GB，首次通常另花 10-30 分钟）",
            "不含失败重跑；首次跑通一个新基座建议预留 1.5 倍预算",
        ],
    )


def cheapest_viable_gpu(job: LoRATrainingJob,
                        *, tier: Literal["community", "secure"] = "community"
                        ) -> tuple[str, CostEstimate]:
    """在显存装得下的卡里挑总花费最低的。

    注意最便宜的卡不一定总价最低：4090 每小时便宜 8 倍，但步时慢 3 倍多，
    再叠加 block swap 的 PCIe 开销，14B 上常常反而更贵 —— 所以按**总价**排序。
    """
    need = 24 if job.spec.params_b >= 13 else 16
    cands = [g for g, v in GPU_VRAM_GB.items() if v >= need]
    est = {g: estimate_cost(job, g, tier=tier) for g in cands}  # type: ignore[arg-type]
    best = min(est, key=lambda g: est[g].cost_usd_high)
    return best, est[best]


# ---------------------------------------------------------------- 角色 LoRA 配方


@dataclass
class CharacterLoRARecipe:
    """角色 LoRA 的配方：要多少素材、训多少、拿什么当触发词。

    数值分两类，务必分清：
    - 「有出处」的：rank/alpha/lr/optimizer/显存开关，来自两套栈的官方示例配置；
    - 「经验值」的：定妆图张数、动态素材秒数、步数区间 —— 见 evidence 里的标注，
      这些**没有**找到可引用的权威消融，请按 first_run_protocol 自己标定。
    """

    character_id: str
    trigger_word: str
    n_portraits: int
    n_motion_clips: int
    motion_seconds_total: float
    rank: int
    alpha: float
    lr: float
    epochs: int
    target_steps: tuple[int, int]
    resolution: tuple[int, int]
    target_frames: list[int]
    evidence: list[str] = field(default_factory=list)
    first_run_protocol: list[str] = field(default_factory=list)

    def render(self) -> str:
        L = [
            f"角色 LoRA 配方 · {self.character_id} · 触发词「{self.trigger_word}」",
            f"  素材：定妆图 {self.n_portraits} 张 + 动态 {self.n_motion_clips} 段"
            f"（合计约 {self.motion_seconds_total:.0f}s）",
            f"  结构：rank {self.rank} / alpha {self.alpha:g}（alpha/rank = "
            f"{self.alpha / self.rank:.2f}）",
            f"  优化：lr {self.lr:g}，{self.epochs} epoch，目标步数 "
            f"{self.target_steps[0]}–{self.target_steps[1]}",
            f"  桶：{self.resolution[0]}x{self.resolution[1]} × {self.target_frames} 帧",
            "  依据：",
        ]
        L += [f"    - {e}" for e in self.evidence]
        L += ["  首次标定流程："] + [f"    {i+1}. {s}" for i, s in enumerate(self.first_run_protocol)]
        return "\n".join(L)


def recommend_character_lora(
    character: CharacterBible,
    *,
    stack: Stack = "musubi",
    has_motion_refs: bool = True,
    resolution: tuple[int, int] = (960, 544),
    frames: int = 49,
) -> CharacterLoRARecipe:
    """给一个角色出 LoRA 配方。

    角色 LoRA 和「画风 LoRA」的取值逻辑相反：角色要**记住一张脸**，
    需要高 rank + 少步数容易过拟合；所以这里压 rank 抬素材量，
    宁可多喂角度也不要靠加 rank 硬记。
    """
    trigger = (character.lora.trigger_word if character.lora
               else f"{character.id}_person")
    # 定妆图按「视角 × 光位 × 表情」的矩阵估：至少 3 视角 × 2 光位 × 3 表情。
    n_portraits = max(18, len(character.portraits) + len(character.turnaround))
    n_motion = 12 if has_motion_refs else 0
    motion_s = n_motion * 3.0

    rank, alpha = 32, 16.0
    lr = default_lr_for(stack)
    # 一条经验：视频角色 LoRA 的有效步数窗口比图片 LoRA 宽但更钝，
    # 1200-2500 步之间取最好的检查点，而不是训完用最后一个。
    steps = (1200, 2500)
    epochs = 16

    evidence = [
        "rank=32 / alpha=16：musubi docs/advanced_config.md 示例即 network_dim=32、"
        "network_alpha=16；diffusion-pipe examples/wan_14b_min_vram.toml 的 [adapter] rank=32。"
        "（查证 2026-09-19）",
        f"lr={lr:g}：{'musubi docs/wan.md 官方示例 --learning_rate 2e-4（adamw8bit）'
                      if stack == 'musubi' else
                      'diffusion-pipe examples/wan_14b_min_vram.toml [optimizer] lr = 2e-5'
                      '（AdamW8bitKahan）'}。（查证 2026-09-19）",
        f"epochs={epochs}：musubi docs/wan.md 示例 --max_train_epochs 16。（查证 2026-09-19）",
        f"分辨率 {resolution[0]}x{resolution[1]}：musubi README 对 12GB 显存给的建议上限"
        "「use a resolution of 960x544 or lower」。（查证 2026-09-19）",
        f"帧数 {frames}：必须是 N*4+1（musubi docs/dataset_config.md）；"
        "diffusion-pipe 官方 24GB 配置训的是 512x512x81。（查证 2026-09-19）",
        f"定妆图 {n_portraits} 张 / 动态 {n_motion} 段 / 步数 {steps[0]}-{steps[1]}："
        "**经验值，未找到可引用的消融实验**。"
        "TODO(2026-09-19)：按 first_run_protocol 自行标定后回来改这三个数。",
    ]
    protocol = [
        "先用 wan2.1-t2v-1.3b 小基座跑通全流程（几美元），确认数据管线没问题再上 14B。",
        f"每 {max(1, steps[0] // 6)} 步存一个检查点，训到 {steps[1]} 步。",
        "用同一组 8 条测试提示词（含未见过的场景/服装）逐个检查点出图，"
        "人工看三件事：像不像、换场景会不会带出训练集背景、动作会不会僵。",
        "「带出训练集背景」= 过拟合信号，回退到更早的检查点或降 mix ratio 里的教师占比。",
        "选定检查点后用 longfilm.dpo.FlywheelMonitor 记一轮，把质检通过率和多样性一起存档。",
    ]
    return CharacterLoRARecipe(
        character_id=character.id, trigger_word=trigger,
        n_portraits=n_portraits, n_motion_clips=n_motion, motion_seconds_total=motion_s,
        rank=rank, alpha=alpha, lr=lr, epochs=epochs, target_steps=steps,
        resolution=resolution, target_frames=[frames],
        evidence=evidence, first_run_protocol=protocol,
    )


# ---------------------------------------------------------------- 双锚推理


#: 近景看脸，远景看轮廓。景别决定身份锚该压在哪一侧。
_TIGHT = {ShotSize.ECU, ShotSize.CU, ShotSize.MCU}
_WIDE = {ShotSize.LS, ShotSize.ELS, ShotSize.FS}

#: 双锚总预算。LoRA 强度与参考图权重同时拉满时，两个锚会互相拉扯，
#: 表现为脸部抖动/五官漂移；把两者之和约束在一个预算内是最直接的止血法。
#: 经验值，非实测常数 —— TODO(2026-09-19)：未找到可引用的公开实验，
#: 建议在自家基座上用 3×3 网格（lora 0.6/0.8/1.0 × ref 0.6/0.9/1.2）扫一次。
DUAL_ANCHOR_BUDGET = 1.75


@dataclass
class DualAnchorPlan:
    """角色 LoRA 强度 + 参考图权重的搭配。"""

    character_id: str
    shot_id: str
    lora_strength: float
    identity_weight: float
    wardrobe_weight: float
    budget_used: float
    reason: list[str] = field(default_factory=list)
    conflict_playbook: list[str] = field(default_factory=list)

    def apply(self, shot: Shot) -> Shot:
        """把权重写回 Shot 的 refs。就地改，返回同一个对象方便链式调用。"""
        for r in shot.refs.images:
            if r.subject_id != self.character_id:
                continue
            if r.role == "identity":
                r.weight = self.identity_weight
            elif r.role == "wardrobe":
                r.weight = self.wardrobe_weight
        return shot

    def render(self) -> str:
        L = [
            f"双锚配置 · {self.character_id} @ {self.shot_id}",
            f"  LoRA 强度 {self.lora_strength:.2f} ｜ identity 参考图权重 "
            f"{self.identity_weight:.2f} ｜ wardrobe {self.wardrobe_weight:.2f}"
            f"（预算占用 {self.budget_used:.2f}/{DUAL_ANCHOR_BUDGET:.2f}）",
            "  理由：",
        ]
        L += [f"    - {r}" for r in self.reason]
        L += ["  冲突排查："] + [f"    - {c}" for c in self.conflict_playbook]
        return "\n".join(L)


def dual_anchor_config(character: CharacterBible, shot: Shot,
                       *, budget: float = DUAL_ANCHOR_BUDGET) -> DualAnchorPlan:
    """角色 LoRA 与参考图的搭配建议。

    两者锚的是**同一件事**（这张脸），但来源不同：LoRA 锚的是训练分布的平均脸，
    参考图锚的是具体这一张。它们不一致时模型会在两者之间摇摆，
    表现为同一镜内五官漂移 —— 所以要按镜头情境把权重往其中一侧倾斜，
    而不是两边都拉满。
    """
    has_lora = character.lora is not None
    base_lora = character.lora.strength if has_lora else 0.0
    reason: list[str] = []

    if not has_lora:
        # 没有 LoRA 时参考图是唯一的锚，可以给到接近上限
        ident = min(2.0, budget * 0.8)
        reason.append("角色无 LoRA：参考图是唯一身份锚，权重给到预算的 80%")
        plan_lora = 0.0
    else:
        plan_lora = base_lora
        ident = max(0.0, budget - plan_lora)
        reason.append(f"角色有 LoRA（trigger={character.lora.trigger_word}，"
                      f"基准强度 {base_lora:.2f}）：参考图分走预算剩余部分")

    if shot.shot_size in _TIGHT:
        # 近景脸占画面大，参考图的具体五官比 LoRA 的平均脸更可靠
        ident = min(2.0, ident * 1.15)
        plan_lora *= 0.9
        reason.append(f"近景 {shot.shot_size.name}：脸占比大，抬参考图、压 LoRA，"
                      "避免平均脸把特征磨平")
    elif shot.shot_size in _WIDE:
        # 远景脸只有几十像素，高权重参考图反而会把脸不成比例地放大
        ident *= 0.6
        plan_lora = min(1.0, plan_lora * 1.05)
        reason.append(f"远景 {shot.shot_size.name}：脸只有几十像素，压参考图权重，"
                      "否则模型会为了对齐参考图而把头画大")

    if shot.continuity.inherit_last_frame:
        # 首帧已经把脸钉死了，再加高权重身份图等于三个锚打架，运动会僵
        ident *= 0.5
        reason.append("本镜继承上一镜尾帧作首帧：首帧已锁脸，identity 图减半，"
                      "否则三锚互锁会让运动幅度塌成静止")

    wardrobe = min(1.0, ident * 0.55)
    ident = round(min(2.0, ident), 2)
    plan_lora = round(min(2.0, plan_lora), 2)

    playbook = [
        "症状：同一镜内五官漂移/脸像在两张脸之间切换 → 双锚互斥。"
        "先把 identity 权重降 0.2，若仍抖再降 LoRA 0.1，一次只动一个。",
        "症状：脸很像但表情僵、动作幅度小 → 锚太强压住了运动。"
        "优先降 identity（参考图是静态的，它压运动比 LoRA 更狠）。",
        "症状：脸不像但动作好 → 锚太弱。近景优先抬 identity，远景优先抬 LoRA。",
        "症状：换了场景却带出训练集的背景/服装 → LoRA 过拟合，不是权重问题，"
        "回退检查点或重训（见 recommend_character_lora 的 first_run_protocol）。",
        f"两者之和保持在 {budget:.2f} 以内；要同时抬必须先确认参考图就是 LoRA 训练集里的那张脸。",
    ]
    return DualAnchorPlan(
        character_id=character.id, shot_id=shot.id,
        lora_strength=plan_lora, identity_weight=ident,
        wardrobe_weight=round(wardrobe, 2),
        budget_used=round(plan_lora + ident, 2),
        reason=reason, conflict_playbook=playbook,
    )


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    from .schema import Appearance, Continuity, ImageRef, RefPack

    job = LoRATrainingJob(
        name="lin-identity-v1", stack="musubi", base_model="wan2.2-i2v-a14b",
        dataset_dir="/workspace/data/videos", dataset_config="/workspace/data/dataset.toml",
        cache_dir="/workspace/data/cache", dataset_hash="7330bfbd41c1c4d9",
        dataset_remote="s3://lf/ep01/trainset/", upload_to="s3://lf/lora/lin-v1/",
        n_clips=420, num_repeats=2, epochs=16, trigger_word="linyue_person",
        resolution=(960, 544), target_frames=[49],
        memory=MemoryOpts(blocks_to_swap=16, offload_inactive_dit=True),
    )

    # ---- [1] 工单默认值按栈走
    assert job.lr == 2e-4 and job.optimizer == "adamw8bit", (job.lr, job.optimizer)
    dp = job.model_copy(update={"stack": "diffusion_pipe", "lr": 0.0, "optimizer": ""})
    dp = LoRATrainingJob(**dp.model_dump())
    assert dp.lr == 2e-5 and dp.optimizer == "AdamW8bitKahan", (dp.lr, dp.optimizer)
    print(f"[1] 栈相关默认值：musubi lr={job.lr:g}/{job.optimizer}，"
          f"diffusion-pipe lr={dp.lr:g}/{dp.optimizer}（官方示例差一个数量级）")

    # ---- [2] 步数推导
    assert job.steps_per_epoch() == 840, job.steps_per_epoch()
    assert job.total_steps() == 840 * 16
    print(f"[2] 步数：{job.n_clips} 片 × {job.num_repeats} repeats / "
          f"(bs{job.micro_batch_size}×ga{job.grad_accum}) = {job.steps_per_epoch()} 步/epoch "
          f"× {job.epochs} = {job.total_steps()} 步")

    # ---- [3] musubi 配置与命令
    toml = emit_musubi_config(job)
    assert "target_frames = [49]" in toml and 'frame_extraction = "chunk"' in toml
    assert "resolution = [960, 544]" in toml
    argv = musubi_train_argv(job)
    for flag, val in (("--task", "i2v-A14B"), ("--network_module", "networks.lora_wan"),
                      ("--network_dim", "32"), ("--network_alpha", "16"),
                      ("--timestep_sampling", "shift"), ("--discrete_flow_shift", "3"),
                      ("--max_train_epochs", "16")):
        assert flag in argv and argv[argv.index(flag) + 1] == val, (flag, val)
    assert "--fp8_base" in argv and "--gradient_checkpointing" in argv
    assert "--offload_inactive_dit" in argv
    assert argv[argv.index("--blocks_to_swap") + 1] == "16"
    lat, txt = musubi_cache_argv(job)
    assert "--i2v" in lat and "--vae_cache_cpu" in lat and "--t5" in txt
    print(f"[3] musubi：dataset toml {len(toml)}B；训练命令 {len(argv)} 个 token，关键项：")
    print("      --task i2v-A14B --network_dim 32 --network_alpha 16 "
          "--fp8_base --blocks_to_swap 16 --offload_inactive_dit")

    # ---- [4] 误用 offload_inactive_dit / 缺 clip 要报错
    try:
        bad = LoRATrainingJob(name="x", base_model="wan2.1-t2v-14b",
                              dataset_dir="/d", memory=MemoryOpts(offload_inactive_dit=True))
        musubi_train_argv(bad)
        raise AssertionError("Wan2.1 上用 --offload_inactive_dit 居然没报错")
    except ValueError as e:
        assert "双专家" in str(e), e
    try:
        bad2 = LoRATrainingJob(name="y", base_model="wan2.1-i2v-14b", dataset_dir="/d")
        musubi_cache_argv(bad2)
        raise AssertionError("Wan2.1-I2V 缺 clip 居然没报错")
    except ValueError as e:
        assert "--clip" in str(e), e
    print("[4] 参数误用拦截：Wan2.1 用 --offload_inactive_dit、Wan2.1-I2V 缺 --clip 均抛 ValueError")

    # ---- [5] diffusion-pipe 配置
    dpj = LoRATrainingJob(**{**job.model_dump(), "stack": "diffusion_pipe",
                             "lr": 0.0, "optimizer": "",
                             "dataset_config": "/workspace/data/dataset_dp.toml"})
    cfg = emit_diffusion_pipe_config(dpj)
    dsc = emit_diffusion_pipe_dataset(dpj)
    for k in ("output_dir =", "micro_batch_size_per_gpu =", "blocks_to_swap = 16",
              "activation_checkpointing = 'unsloth'", "[adapter]", "rank = 32",
              "[optimizer]", "type = 'AdamW8bitKahan'", "lr = 2e-05",
              "transformer_dtype = 'float8'", "type = 'wan'"):
        assert k in cfg, k
    assert "frame_buckets = [1, 49]" in dsc and "resolutions = [[960, 544]]" in dsc
    assert "num_repeats = 2" in dsc
    print(f"[5] diffusion-pipe：config {len(cfg.splitlines())} 行 + dataset "
          f"{len(dsc.splitlines())} 行，字段全部命中官方示例")

    # ---- [6] 训练脚本
    for plat in ("runpod", "vast", "local"):
        sh = emit_train_script(job, platform=plat)  # type: ignore[arg-type]
        assert sh.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in sh
        assert 'HF_TOKEN:?' in sh and "AWS_ACCESS_KEY_ID:?" in sh
        assert "wan_cache_latents.py" in sh and "wan_cache_text_encoder_outputs.py" in sh
        assert "wan_train_network.py" in sh
        assert "aws s3 sync" in sh
        # 绝不能有明文密钥
        assert "AKIA" not in sh and "sk-" not in sh
    sh = emit_train_script(job, platform="runpod")
    print(f"[6] emit_train_script：3 个平台均生成，runpod 版 {len(sh.splitlines())} 行；"
          f"密钥校验行：{[l for l in sh.splitlines() if ':?' in l][0][:46]}…")

    # ---- [7] 成本估算
    est_h100 = estimate_cost(job, "H100")
    est_4090 = estimate_cost(job, "4090")
    assert est_h100.cost_usd_low < est_h100.cost_usd_high
    assert est_h100.hours_low > 0 and est_4090.vram_warning, est_4090
    assert est_h100.hourly_usd == 2.69   # RunPod community，2026-09-19 抓取
    measured = estimate_cost(job, "H100", measured_s_per_step=4.0)
    assert measured.hours_high / measured.hours_low < est_h100.hours_high / est_h100.hours_low
    print("[7] 成本估算：")
    for ln in est_h100.render().splitlines()[:6]:
        print("      " + ln)
    print(f"      4090 对照：${est_4090.cost_usd_low:.0f}–${est_4090.cost_usd_high:.0f} / "
          f"{est_4090.hours_low:.0f}–{est_4090.hours_high:.0f}h（{est_4090.vram_warning[:24]}…）")
    best, best_est = cheapest_viable_gpu(job)
    print(f"      总价最低可行卡：{best}（${best_est.cost_usd_low:.0f}–"
          f"${best_est.cost_usd_high:.0f}），注意不是时薪最低的那张")

    # ---- [8] 角色 LoRA 配方
    lin = CharacterBible(
        id="lin", name="林越", age_statement="虚构角色，设定年龄 27 岁",
        appearance=Appearance(face="鹅蛋脸", distinguishing="左眉一道疤"),
        portraits=[ImageRef(role="identity", uri=f"s3://b/{i}.png", subject_id="lin")
                   for i in range(9)],
        lora=LoRASpec(base_model="wan2.2-i2v-a14b", path="s3://lora/lin.safetensors",
                      trigger_word="linyue_person", strength=0.85),
    )
    rec = recommend_character_lora(lin)
    assert rec.rank == 32 and rec.alpha == 16 and rec.lr == 2e-4
    assert any("经验值" in e for e in rec.evidence), "经验值必须显式标注"
    assert any("musubi" in e for e in rec.evidence)
    assert rec.first_run_protocol
    print("[8] 角色 LoRA 配方：")
    for ln in rec.render().splitlines()[:5]:
        print("      " + ln)

    # ---- [9] 双锚
    tight = Shot(id="sh10", scene_id="sc01", index=0, shot_size=ShotSize.CU,
                 subject_ids=["lin"],
                 refs=RefPack(images=[
                     ImageRef(role="identity", uri="s3://b/0.png", subject_id="lin"),
                     ImageRef(role="wardrobe", uri="s3://b/w.png", subject_id="lin"),
                 ]))
    wide = tight.model_copy(deep=True)
    wide.id, wide.shot_size = "sh11", ShotSize.LS
    inherit = tight.model_copy(deep=True)
    inherit.id = "sh12"
    inherit.continuity = Continuity(prev_shot_id="sh10", inherit_last_frame=True)

    p_t = dual_anchor_config(lin, tight)
    p_w = dual_anchor_config(lin, wide)
    p_i = dual_anchor_config(lin, inherit)
    assert p_t.identity_weight > p_w.identity_weight, (p_t.identity_weight, p_w.identity_weight)
    assert p_w.lora_strength > p_t.lora_strength
    assert p_i.identity_weight < p_t.identity_weight
    assert p_t.budget_used <= DUAL_ANCHOR_BUDGET + 0.3
    p_t.apply(tight)
    assert tight.refs.images[0].weight == p_t.identity_weight
    assert tight.refs.images[1].weight == p_t.wardrobe_weight
    noloras = lin.model_copy(deep=True)
    noloras.lora = None
    p_n = dual_anchor_config(noloras, tight)
    assert p_n.lora_strength == 0.0 and p_n.identity_weight > p_t.identity_weight
    print(f"[9] 双锚：近景 lora {p_t.lora_strength:.2f}/ref {p_t.identity_weight:.2f}｜"
          f"远景 lora {p_w.lora_strength:.2f}/ref {p_w.identity_weight:.2f}｜"
          f"继承尾帧 ref {p_i.identity_weight:.2f}｜无 LoRA ref {p_n.identity_weight:.2f}")
    print("      " + p_t.conflict_playbook[0])

    # ---- [10] 回填 LoRASpec
    spec = job.to_lora_spec("s3://lora/lin-v1.safetensors")
    assert spec.dataset_hash == "7330bfbd41c1c4d9" and spec.trained_steps == job.total_steps()
    assert spec.trigger_word == "linyue_person"
    print(f"[10] 回填 LoRASpec：steps={spec.trained_steps} dataset_hash={spec.dataset_hash}")

    print("\nlora 自测通过")


if __name__ == "__main__":
    _selftest()
