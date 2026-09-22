#!/usr/bin/env bash
# 在云 GPU 上跑一次角色 LoRA 训练：同步数据 → 生成配置 → 训练 → 回传产物。
#
# 这个脚本**不重写训练逻辑**。配置与命令行全部由 src/longfilm/lora.py 生成
# （emit_musubi_config / musubi_cache_argv / musubi_train_argv），
# 这样「训练怎么配」只有一个真相来源，不会出现脚本和代码各配一套、
# 训出来的 LoRA 跟产线推理时的假设对不上的经典事故。
#
# 用法：
#   bash deploy/train_lora_pod.sh --character lin_lan
#   bash deploy/train_lora_pod.sh --character lin_lan --rank 48 --steps 3000 --dry-run
#
# 断点续训：直接重跑同一条命令即可。脚本会扫 output_dir 里的 *-state 目录，
# 找到就自动 --resume 并把 epoch 数换成「剩余」。这是能用 $0.1/h 竞价卡的前提。
#   DATASET_REMOTE=s3://bucket/lin_lan OUT_REMOTE=s3://bucket/loras \
#       bash deploy/train_lora_pod.sh --character lin_lan
#
# 退出码：0 成功 / 10 参数错 / 20 环境缺失 / 30 数据同步失败 / 40 训练失败 / 50 回传失败

set -Eeuo pipefail

EXIT_ARGS=10; EXIT_ENV=20; EXIT_SYNC=30; EXIT_TRAIN=40; EXIT_UPLOAD=50

CHARACTER=""
RANK=32
STEPS=0
EPOCHS=16
STACK="musubi"
BASE="wan2.2-i2v-a14b"
WORKROOT="${WORKROOT:-/workspace}"
DATASET_REMOTE="${DATASET_REMOTE:-}"
OUT_REMOTE="${OUT_REMOTE:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --character) CHARACTER="$2"; shift 2 ;;
    --rank)      RANK="$2";      shift 2 ;;
    --steps)     STEPS="$2";     shift 2 ;;
    --epochs)    EPOCHS="$2";    shift 2 ;;
    --stack)     STACK="$2";     shift 2 ;;
    --base)      BASE="$2";      shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift ;;
    -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数 $1" >&2; exit $EXIT_ARGS ;;
  esac
done

[[ -n "$CHARACTER" ]] || { echo "缺 --character" >&2; exit $EXIT_ARGS; }

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$2" >&2; exit "$1"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
command -v "$PY" >/dev/null || die $EXIT_ENV "找不到 python"

DATA_DIR="$WORKROOT/data/$CHARACTER"
OUT_DIR="$WORKROOT/loras/$CHARACTER"
CFG_DIR="$WORKROOT/cfg/$CHARACTER"
mkdir -p "$DATA_DIR" "$OUT_DIR" "$CFG_DIR"

# ---------------------------------------------------------------- 1. 环境体检
log "环境体检"
if [[ $DRY_RUN -eq 0 ]]; then
  command -v nvidia-smi >/dev/null || die $EXIT_ENV "没有 nvidia-smi：这个脚本要在有 GPU 的机器上跑"
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ${VRAM}MiB"
  # musubi 训 14B 的实测门槛是 24GB；低于这个数先说清楚，别训到一半 OOM
  if [[ "$VRAM" -lt 23000 ]]; then
    log "  [警告] 显存 ${VRAM}MiB 低于 musubi 训 14B 的 24GB 门槛，"
    log "         要么换 5B 基座，要么开 block_swap / fp8，见 lora.MemoryOpts"
  fi
fi

# ---------------------------------------------------------------- 2. 同步数据
if [[ -n "$DATASET_REMOTE" ]]; then
  log "同步训练数据 $DATASET_REMOTE -> $DATA_DIR"
  if [[ $DRY_RUN -eq 0 ]]; then
    case "$DATASET_REMOTE" in
      s3://*)    aws s3 sync "$DATASET_REMOTE" "$DATA_DIR" || die $EXIT_SYNC "s3 同步失败" ;;
      hf://*)    "$PY" -m huggingface_hub.commands.huggingface_cli download \
                    "${DATASET_REMOTE#hf://}" --repo-type dataset --local-dir "$DATA_DIR" \
                    || die $EXIT_SYNC "hf 下载失败" ;;
      http*)     curl -fL --retry 3 -C - -o "$DATA_DIR/dataset.zip" "$DATASET_REMOTE" \
                    && unzip -q -o "$DATA_DIR/dataset.zip" -d "$DATA_DIR" || die $EXIT_SYNC "下载失败" ;;
      *)         rsync -a --info=progress2 "$DATASET_REMOTE/" "$DATA_DIR/" || die $EXIT_SYNC "rsync 失败" ;;
    esac
  fi
fi

N_CLIPS=$(find "$DATA_DIR" -type f \( -name '*.mp4' -o -name '*.mov' -o -name '*.webm' \) 2>/dev/null | wc -l)
log "训练集：$N_CLIPS 个片段"
if [[ $DRY_RUN -eq 0 && "$N_CLIPS" -lt 10 ]]; then
  # 低于 10 条基本训不出稳定身份，早说比训完再发现强
  die $EXIT_SYNC "片段数 $N_CLIPS 太少（建议 ≥20），检查 DATASET_REMOTE 是否同步成功"
fi

# ---------------------------------------------------------------- 3. 生成配置
log "生成训练配置（真相来源：src/longfilm/lora.py）"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - "$CHARACTER" "$STACK" "$BASE" "$DATA_DIR" "$OUT_DIR" "$CFG_DIR" "$RANK" "$EPOCHS" "$STEPS" "$N_CLIPS" <<'PYEOF' || die $EXIT_ENV "配置生成失败"
import sys, json, shlex
from pathlib import Path
from longfilm.lora import (
    LoRATrainingJob, emit_musubi_config, emit_diffusion_pipe_config,
    emit_diffusion_pipe_dataset, musubi_cache_argv, musubi_train_argv,
)
name, stack, base, data, out, cfg, rank, epochs, steps, n_clips = sys.argv[1:11]
cfgd = Path(cfg)
job = LoRATrainingJob(
    name=name, stack=stack, base_model=base, dataset_dir=data,
    output_dir=out, cache_dir=str(Path(data) / "_cache"),
    n_clips=int(n_clips), rank=int(rank), alpha=int(rank) / 2,
    epochs=int(epochs), max_train_steps=int(steps) or None,
)
# 断点续训：被抢占的实例重开后，从最新 state 接着训，而不是从头来。
# 注意 musubi 的 --resume 会再跑满 max_train_epochs 个 epoch，所以这里要把
# epochs 换成「剩余」——plan_resume 已经算好了。
from longfilm.lora import plan_resume
resume_path, remaining = plan_resume(job)
if resume_path:
    job.resume_from = resume_path
    job.epochs = remaining
    print(f"  [续训] 从 {Path(resume_path).name} 接着训，剩余 {remaining} epoch")
else:
    print("  [全新] 没找到训练状态，从头开始")

if stack == "musubi":
    (cfgd / "dataset.toml").write_text(emit_musubi_config(job), encoding="utf-8")
    job.dataset_config = str(cfgd / "dataset.toml")
    cache, _ = musubi_cache_argv(job)
    (cfgd / "cmd_cache.txt").write_text(" ".join(shlex.quote(a) for a in cache), encoding="utf-8")
    (cfgd / "cmd_train.txt").write_text(
        " ".join(shlex.quote(a) for a in musubi_train_argv(job)), encoding="utf-8")
else:
    (cfgd / "dataset.toml").write_text(emit_diffusion_pipe_dataset(job), encoding="utf-8")
    job.dataset_config = str(cfgd / "dataset.toml")
    (cfgd / "train.toml").write_text(emit_diffusion_pipe_config(job), encoding="utf-8")
(cfgd / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
print(f"  配置写入 {cfgd}")
PYEOF

# ---------------------------------------------------------------- 4. 训练
if [[ $DRY_RUN -eq 1 ]]; then
  log "--dry-run：配置已生成，不执行训练"
  [[ -f "$CFG_DIR/cmd_train.txt" ]] && { log "  将执行："; sed 's/^/    /' "$CFG_DIR/cmd_train.txt"; }
  exit 0
fi

if [[ "$STACK" == "musubi" ]]; then
  [[ -d "$WORKROOT/musubi-tuner" ]] || die $EXIT_ENV "$WORKROOT/musubi-tuner 不存在，先跑 bootstrap_gpu.sh --profile train"
  cd "$WORKROOT/musubi-tuner"
  log "预缓存 latent 与文本嵌入"
  bash -c "$(cat "$CFG_DIR/cmd_cache.txt")" || die $EXIT_TRAIN "缓存阶段失败"
  log "开始训练"
  bash -c "$(cat "$CFG_DIR/cmd_train.txt")" || die $EXIT_TRAIN "训练失败"
else
  [[ -d "$WORKROOT/diffusion-pipe" ]] || die $EXIT_ENV "$WORKROOT/diffusion-pipe 不存在"
  cd "$WORKROOT/diffusion-pipe"
  deepspeed --num_gpus="$(nvidia-smi -L | wc -l)" train.py --deepspeed \
      --config "$CFG_DIR/train.toml" || die $EXIT_TRAIN "训练失败"
fi

# ---------------------------------------------------------------- 5. 回传
SAFETENSORS=$(find "$OUT_DIR" -name '*.safetensors' -newer "$CFG_DIR/job.json" 2>/dev/null | sort | tail -1)
[[ -n "$SAFETENSORS" ]] || die $EXIT_TRAIN "训练结束但没找到 .safetensors 产物"
log "产物：$SAFETENSORS ($(du -h "$SAFETENSORS" | cut -f1))"

# 训练配置随权重一起回传 —— 只有权重没有配置的 LoRA 是不可复现的
cp "$CFG_DIR/job.json" "$OUT_DIR/" 2>/dev/null || true

if [[ -n "$OUT_REMOTE" ]]; then
  log "回传 $OUT_DIR -> $OUT_REMOTE"
  case "$OUT_REMOTE" in
    s3://*) aws s3 sync "$OUT_DIR" "$OUT_REMOTE/$CHARACTER" || die $EXIT_UPLOAD "s3 回传失败" ;;
    hf://*) "$PY" -m huggingface_hub.commands.huggingface_cli upload \
                "${OUT_REMOTE#hf://}" "$OUT_DIR" "$CHARACTER" || die $EXIT_UPLOAD "hf 上传失败" ;;
    *)      rsync -a "$OUT_DIR/" "$OUT_REMOTE/$CHARACTER/" || die $EXIT_UPLOAD "rsync 回传失败" ;;
  esac
fi

log "完成。把 LoRA 挂进产线：在 CharacterBible.lora 里填 path=$SAFETENSORS"
