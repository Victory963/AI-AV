#!/usr/bin/env bash
# longfilm 云 GPU 一键开机脚本
#
# 目标：一台刚开的裸 GPU 机器 → 能出片的 ComfyUI 服务，一条命令，重跑不重复干活。
#
# 设计原则（踩过的坑，别改）：
#  1. **幂等**。每一步先问「已经做完了吗」。权重按「本地字节数 == 清单字节数」判断，
#     不按「文件存在」—— 断在 13GB 的 safetensors 是最常见的失败，文件存在但没法用。
#  2. **早失败**。磁盘、显存、驱动在下任何一个字节之前就检查完。
#     下了 40 分钟才发现盘不够，是最贵的一种错误。
#  3. **退出码有语义**。见下面 EXIT_* 常量；CI 与 runpod 模板脚本靠它分流。
#  4. 只在真正需要时联网。--offline 可以在无网机器上做一次完整的自检。
#
# 用法：
#   bash deploy/bootstrap_gpu.sh --profile i2v
#   bash deploy/bootstrap_gpu.sh --profile minimal --smoke
#   bash deploy/bootstrap_gpu.sh --dry-run          # 只打印计划，不动盘
#   bash deploy/bootstrap_gpu.sh --check            # 只做环境体检
#
# 查证日期 2026-09-20：
#   ComfyUI 手动安装 = git clone + pip install -r requirements.txt + torch(cu130)
#     来源 https://github.com/comfyanonymous/ComfyUI README（README 现推荐 CUDA 13.0+）
#   ComfyUI-Manager 已迁到 Comfy-Org，且必须落在 custom_nodes/comfyui-manager 这个目录名
#     来源 https://github.com/Comfy-Org/ComfyUI-Manager
#   HuggingFace CLI 已由 huggingface-cli 更名为 hf（2025-07 起），hf download 自带断点续传
#     来源 https://huggingface.co/blog/hf-cli

set -Eeuo pipefail

# ---------------------------------------------------------------- 退出码

readonly EXIT_OK=0
readonly EXIT_USAGE=2
readonly EXIT_NO_GPU=10
readonly EXIT_DRIVER_OLD=11
readonly EXIT_LOW_VRAM=12
readonly EXIT_PYTHON=20
readonly EXIT_COMFY_INSTALL=21
readonly EXIT_NODE_INSTALL=22
readonly EXIT_DISK=30
readonly EXIT_DOWNLOAD=31
readonly EXIT_SIZE_MISMATCH=32
readonly EXIT_START=40
readonly EXIT_HEALTH=41
readonly EXIT_SMOKE=42

# ---------------------------------------------------------------- 默认值

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LONGFILM_ROOT:-/workspace}"
PROFILE="i2v"
PORT="8188"
LISTEN="0.0.0.0"
MODELS_FILE="$SCRIPT_DIR/models.yaml"
DRY_RUN=0
DO_CHECK_ONLY=0
DO_DOWNLOAD=1
DO_START=1
DO_UPDATE=0
DO_SMOKE=0
OFFLINE=0
MIN_VRAM_GB=16
# torch 轮子索引。ComfyUI README（2026-09-20）推荐 cu130；老驱动回落 cu128。
TORCH_INDEX_DEFAULT="https://download.pytorch.org/whl/cu130"
TORCH_INDEX_FALLBACK="https://download.pytorch.org/whl/cu128"

STEP_NO=0
T0=$(date +%s)

# ---------------------------------------------------------------- 日志

log()  { printf '%s | %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
step() { STEP_NO=$((STEP_NO + 1)); printf '\n\033[1m==> [%d] %s\033[0m\n' "$STEP_NO" "$*" >&2; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*" >&2; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*" >&2; }
die()  { local code=$1; shift; printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit "$code"; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '    [dry-run] %s\n' "$*" >&2; else "$@"; fi; }

on_err() {
  local code=$?
  printf '\n\033[31m✗ 第 %d 步失败（退出码 %d），行 %s\033[0m\n' \
    "$STEP_NO" "$code" "${BASH_LINENO[0]}" >&2
  exit "$code"
}
trap on_err ERR

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

选项：
  --profile NAME     权重档位：minimal|i2v|t2v|train|full（默认 i2v）
  --root DIR         工作根目录（默认 $LONGFILM_ROOT 或 /workspace）
  --port N           ComfyUI 端口（默认 8188）
  --listen ADDR      监听地址（默认 0.0.0.0）
  --models FILE      权重清单（默认 deploy/models.yaml）
  --min-vram N       最低显存 GB，低于则拒绝启动（默认 16）
  --check            只做环境体检，不安装不下载
  --skip-download    跳过权重下载
  --no-start         装完不起服务
  --update           已存在的 ComfyUI / 自定义节点执行 git pull
  --smoke            起服务后跑一次最小工作流冒烟
  --offline          不联网（只做本地校验，用于离线自检）
  --dry-run          只打印将要执行的动作
  -h, --help         本帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)       PROFILE="${2:?--profile 需要参数}"; shift 2 ;;
    --root)          ROOT="${2:?--root 需要参数}"; shift 2 ;;
    --port)          PORT="${2:?--port 需要参数}"; shift 2 ;;
    --listen)        LISTEN="${2:?--listen 需要参数}"; shift 2 ;;
    --models)        MODELS_FILE="${2:?--models 需要参数}"; shift 2 ;;
    --min-vram)      MIN_VRAM_GB="${2:?--min-vram 需要参数}"; shift 2 ;;
    --check)         DO_CHECK_ONLY=1; shift ;;
    --skip-download) DO_DOWNLOAD=0; shift ;;
    --no-start)      DO_START=0; shift ;;
    --update)        DO_UPDATE=1; shift ;;
    --smoke)         DO_SMOKE=1; shift ;;
    --offline)       OFFLINE=1; DO_DOWNLOAD=0; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit "$EXIT_OK" ;;
    *)               usage; die "$EXIT_USAGE" "未知参数：$1" ;;
  esac
done

COMFY_DIR="$ROOT/ComfyUI"
STAGE_DIR="$ROOT/hf_stage"       # hf download 的落地目录，权重真身在这里
TRAIN_DIR="$ROOT/models"         # 训练侧权重根（与 ComfyUI/models 分开，见 models.yaml 注释）
VENV_DIR="$ROOT/venv"
STATE_DIR="$ROOT/.longfilm_state"
LOG_DIR="$ROOT/logs"

# ---------------------------------------------------------------- 1 环境体检

step "环境体检"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "$EXIT_NO_GPU" "找不到 nvidia-smi。这台机器没有 NVIDIA GPU 或驱动未装。"
fi
if ! SMI_OUT="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1)"; then
  die "$EXIT_NO_GPU" "nvidia-smi 调不通：$SMI_OUT"
fi
log "检出 GPU："
printf '%s\n' "$SMI_OUT" | sed 's/^/      /' >&2

GPU_COUNT=$(printf '%s\n' "$SMI_OUT" | grep -c . || true)
VRAM_MB=$(printf '%s\n' "$SMI_OUT" | head -1 | awk -F', *' '{print $2}')
DRIVER=$(printf '%s\n' "$SMI_OUT" | head -1 | awk -F', *' '{print $3}')
VRAM_GB=$((VRAM_MB / 1024))
ok "GPU x${GPU_COUNT}，首卡显存 ${VRAM_GB}GB，驱动 ${DRIVER}"

if [ "$VRAM_GB" -lt "$MIN_VRAM_GB" ]; then
  die "$EXIT_LOW_VRAM" "显存 ${VRAM_GB}GB < 要求 ${MIN_VRAM_GB}GB。Wan2.2 A14B fp8 实测最低 24GB；16GB 只能跑 5B。"
fi

# 驱动主版本号决定能装哪个 CUDA 轮子。
# 为什么用驱动号而不是 nvidia-smi 报的 CUDA Version：后者是「驱动支持的最高版本」，
# 跟实际装了什么 toolkit 无关，拿它做判断会误判。
DRIVER_MAJOR="${DRIVER%%.*}"
TORCH_INDEX="$TORCH_INDEX_DEFAULT"
if [ -z "$DRIVER_MAJOR" ] || ! [ "$DRIVER_MAJOR" -ge 525 ] 2>/dev/null; then
  die "$EXIT_DRIVER_OLD" "驱动 ${DRIVER} 过旧（需 >= 525）。换机器，别试着在容器里升驱动。"
fi
if [ "$DRIVER_MAJOR" -lt 580 ]; then
  TORCH_INDEX="$TORCH_INDEX_FALLBACK"
  warn "驱动 ${DRIVER} < 580，CUDA 13 轮子可能起不来，回落到 cu128"
fi
ok "torch 轮子索引：$TORCH_INDEX"

# 磁盘：权重才是大头。i2v ≈ 36GB，full ≈ 130GB，再留 40GB 给中间产物。
NEED_GB=$(case "$PROFILE" in
  minimal) echo 60 ;;
  i2v|t2v) echo 90 ;;
  train)   echo 120 ;;
  full)    echo 220 ;;
  *)       echo 90 ;;
esac)
mkdir -p "$ROOT"
AVAIL_GB=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
AVAIL_GB="${AVAIL_GB:-0}"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  die "$EXIT_DISK" "$ROOT 可用 ${AVAIL_GB}GB < profile=${PROFILE} 需要的 ${NEED_GB}GB。挂 network volume 或换 profile。"
fi
ok "磁盘 ${AVAIL_GB}GB 可用（profile=${PROFILE} 需 ${NEED_GB}GB）"

PY_BIN="$(command -v python3 || true)"
[ -n "$PY_BIN" ] || die "$EXIT_PYTHON" "找不到 python3"
PY_VER="$("$PY_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
case "$PY_VER" in
  3.10|3.11|3.12) ok "python $PY_VER" ;;
  *) warn "python $PY_VER 不在 ComfyUI/musubi 验证过的 3.10-3.12 范围内，可能装不上依赖" ;;
esac

[ -f "$MODELS_FILE" ] || die "$EXIT_USAGE" "权重清单不存在：$MODELS_FILE"
[ -f "$SCRIPT_DIR/_miniyaml.py" ] || die "$EXIT_USAGE" "缺少 $SCRIPT_DIR/_miniyaml.py"
ok "权重清单 $MODELS_FILE"

if [ "$DO_CHECK_ONLY" = 1 ]; then
  log "体检通过（--check，到此为止）"
  exit "$EXIT_OK"
fi

run mkdir -p "$ROOT" "$STAGE_DIR" "$TRAIN_DIR" "$STATE_DIR" "$LOG_DIR"

# ---------------------------------------------------------------- 2 ComfyUI

step "安装 / 更新 ComfyUI"

if [ -d "$COMFY_DIR/.git" ]; then
  ok "已存在 $COMFY_DIR"
  if [ "$DO_UPDATE" = 1 ] && [ "$OFFLINE" = 0 ]; then
    run git -C "$COMFY_DIR" pull --ff-only
    ok "已 git pull"
  fi
else
  [ "$OFFLINE" = 0 ] || die "$EXIT_COMFY_INSTALL" "--offline 但 $COMFY_DIR 不存在"
  run git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" \
    || die "$EXIT_COMFY_INSTALL" "clone ComfyUI 失败"
  ok "已克隆 ComfyUI"
fi

# venv：容器里直接装系统 python 也能跑，但一旦 runpod 模板自带的 torch 与
# ComfyUI requirements 打架，没有 venv 就只能重开机器。代价是几百 MB，值。
if [ ! -x "$VENV_DIR/bin/python" ]; then
  run "$PY_BIN" -m venv --system-site-packages "$VENV_DIR" \
    || die "$EXIT_PYTHON" "创建 venv 失败"
  ok "已创建 venv（继承系统 site-packages，复用容器自带 torch）"
else
  ok "venv 已存在"
fi
VPY="$VENV_DIR/bin/python"
# dry-run 下 venv 还不存在，拿系统 python 顶着，好让后面的命令能被打印出来
if [ "$DRY_RUN" = 1 ]; then VPY="$PY_BIN"; fi

DEPS_MARK="$STATE_DIR/deps.ok"
if [ -f "$DEPS_MARK" ] && [ "$DO_UPDATE" = 0 ]; then
  ok "依赖已装（$DEPS_MARK 存在，--update 可强制重装）"
elif [ "$OFFLINE" = 1 ]; then
  warn "--offline：跳过 pip 安装"
else
  run "$VPY" -m pip install -q --upgrade pip
  # torch 单独一条：它必须来自 CUDA 专用索引，混进 requirements 会被 PyPI 的 CPU 轮子顶掉
  if ! "$VPY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    run "$VPY" -m pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX" \
      || die "$EXIT_COMFY_INSTALL" "安装 torch 失败（索引 $TORCH_INDEX）"
    ok "已装 torch（$TORCH_INDEX）"
  else
    ok "torch 已可用且能看到 CUDA，不动它"
  fi
  run "$VPY" -m pip install -q -r "$COMFY_DIR/requirements.txt" \
    || die "$EXIT_COMFY_INSTALL" "安装 ComfyUI 依赖失败"
  # hf_transfer 能把 HF 下载拉满带宽；14GB 的文件上它和不上它差几倍
  run "$VPY" -m pip install -q "huggingface_hub[hf_transfer]>=0.34" aiohttp \
    || warn "安装 huggingface_hub 失败，下载会退回 curl"
  run touch "$DEPS_MARK"
  ok "依赖安装完成"
fi

# ---------------------------------------------------------------- 3 自定义节点

step "安装自定义节点"

# 目录名固定为 comfyui-manager —— Manager 自己的更新逻辑按这个路径找自己，改名会静默失效。
# TODO(2026-09-20)：这四个仓库都没有可信的版本 tag 可钉，只能跟随默认分支。
#   上游一次破坏性提交就能把工作流搞崩；生产环境请 fork 后钉自己的 commit。
NODE_REPOS=(
  "comfyui-manager|https://github.com/Comfy-Org/ComfyUI-Manager.git"
  "ComfyUI-VideoHelperSuite|https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
  "ComfyUI-KJNodes|https://github.com/kijai/ComfyUI-KJNodes.git"
  "ComfyUI-WanVideoWrapper|https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
)

run mkdir -p "$COMFY_DIR/custom_nodes"
for entry in "${NODE_REPOS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  dest="$COMFY_DIR/custom_nodes/$name"
  if [ -d "$dest/.git" ]; then
    if [ "$DO_UPDATE" = 1 ] && [ "$OFFLINE" = 0 ]; then
      run git -C "$dest" pull --ff-only || warn "$name 更新失败，保留旧版本"
    fi
    ok "$name 已在位"
  elif [ "$OFFLINE" = 1 ]; then
    warn "--offline：跳过 $name"
    continue
  else
    run git clone --depth 1 "$url" "$dest" || die "$EXIT_NODE_INSTALL" "克隆 $name 失败"
    ok "$name 已克隆"
  fi
  if [ -f "$dest/requirements.txt" ] && [ "$OFFLINE" = 0 ]; then
    run "$VPY" -m pip install -q -r "$dest/requirements.txt" \
      || warn "$name 的依赖没装全，该节点可能 import 失败（ComfyUI 会在启动日志里报）"
  fi
done

# ---------------------------------------------------------------- 4 权重下载

step "按清单下载权重（profile=$PROFILE）"

PLAN_FILE="$STATE_DIR/plan-$PROFILE.tsv"
# 用 python 解析 YAML 再吐 TSV，而不是在 bash 里 grep/awk 硬解：
# 清单里有引号、中文、URL 里的冒号，正则解 YAML 是纯粹的自找麻烦。
export LONGFILM_DEPLOY_DIR="$SCRIPT_DIR"
if ! "$PY_BIN" - "$MODELS_FILE" "$PROFILE" "$COMFY_DIR" "$TRAIN_DIR" "$STAGE_DIR" \
      > "$PLAN_FILE.tmp" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["LONGFILM_DEPLOY_DIR"])
import _miniyaml

models_file, profile, comfy_dir, train_dir, stage_dir = sys.argv[1:6]
doc = _miniyaml.load(models_file)
rows = []
for g in doc.get("groups") or []:
    if profile not in (g.get("profiles") or []):
        continue
    for f in g.get("files") or []:
        if f.get("optional"):
            continue
        dest = f["dest"]
        if dest.startswith("@train:"):
            abs_dest = os.path.join(train_dir, dest[len("@train:"):])
        else:
            abs_dest = os.path.join(comfy_dir, dest)
        rows.append("\t".join([
            g["id"],
            f.get("repo", ""),
            f.get("file", ""),
            f.get("url", ""),
            abs_dest,
            str(int(f.get("size_bytes") or 0)),
            "1" if f.get("verified") else "0",
        ]))
if not rows:
    print("__EMPTY__", file=sys.stderr)
print("\n".join(rows))
PYEOF
then
  die "$EXIT_USAGE" "解析 $MODELS_FILE 失败（profile=$PROFILE）"
fi
mv "$PLAN_FILE.tmp" "$PLAN_FILE"

PLAN_N=$(grep -c . "$PLAN_FILE" || true)
if [ "${PLAN_N:-0}" -eq 0 ]; then
  die "$EXIT_USAGE" "profile=$PROFILE 在清单里没有任何条目；可选 profile 见 models.yaml meta.profiles"
fi
TOTAL_BYTES=$(awk -F'\t' '{s+=$6} END{print s+0}' "$PLAN_FILE")
log "计划下载 $PLAN_N 个文件，合计 $((TOTAL_BYTES / 1024 / 1024 / 1024))GB"

have_hf() { [ -x "$VENV_DIR/bin/hf" ] || command -v hf >/dev/null 2>&1; }
hf_bin()  { if [ -x "$VENV_DIR/bin/hf" ]; then echo "$VENV_DIR/bin/hf"; else command -v hf; fi; }

download_one() {
  local group="$1" repo="$2" relfile="$3" url="$4" dest="$5" want="$6" verified="$7"
  local destdir; destdir="$(dirname "$dest")"

  # 幂等判据：目标已在位且大小对得上就跳过。
  # verified=0 的条目没有可信大小，退化成「存在即跳过」，并且喊一声。
  if [ -e "$dest" ]; then
    local have; have=$(stat -Lc %s "$dest" 2>/dev/null || echo 0)
    if [ "$verified" = "1" ] && [ "$have" = "$want" ]; then
      ok "已在位 $(basename "$dest")（$((have / 1024 / 1024))MB）"
      return 0
    fi
    if [ "$verified" != "1" ] && [ "$have" -gt 0 ]; then
      warn "已在位 $(basename "$dest")（UNVERIFIED，清单无可信大小，未校验）"
      return 0
    fi
    warn "$(basename "$dest") 大小 $have != 期望 $want，重下"
    run rm -f "$dest"
  fi

  run mkdir -p "$destdir"

  if [ -n "$url" ]; then
    # 直链：curl -C - 就是断点续传
    run curl -fL --retry 5 --retry-delay 3 -C - -o "$dest" "$url" \
      || { warn "下载失败：$url"; return 1; }
  else
    # HF：hf download 自带断点续传与本地缓存；落到 STAGE_DIR 后软链到目标位置。
    # 为什么软链不是移动：移动会让 hf 的缓存失效，下次重跑等于重下 14GB。
    local stage_repo="$STAGE_DIR/${repo//\//__}"
    run mkdir -p "$stage_repo"
    local env_prefix=(env HF_HUB_ENABLE_HF_TRANSFER=1)
    if have_hf; then
      run "${env_prefix[@]}" "$(hf_bin)" download "$repo" "$relfile" \
        --local-dir "$stage_repo" \
        || { warn "hf download 失败：$repo/$relfile"; return 1; }
    else
      run curl -fL --retry 5 --retry-delay 3 -C - --create-dirs \
        -o "$stage_repo/$relfile" \
        "https://huggingface.co/$repo/resolve/main/$relfile" \
        || { warn "curl 回退下载失败：$repo/$relfile"; return 1; }
    fi
    run ln -sfn "$stage_repo/$relfile" "$dest"
  fi

  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  local got; got=$(stat -Lc %s "$dest" 2>/dev/null || echo 0)
  if [ "$verified" = "1" ] && [ "$got" != "$want" ]; then
    warn "$(basename "$dest") 下完大小 $got != 清单 $want"
    return 2
  fi
  ok "完成 $(basename "$dest")（$((got / 1024 / 1024))MB）"
  return 0
}

if [ "$DO_DOWNLOAD" = 1 ]; then
  FAILED=0
  MISMATCH=0
  while IFS=$'\t' read -r group repo relfile url dest want verified; do
    [ -n "${group:-}" ] || continue
    log "[$group] $(basename "$dest")"
    set +e
    download_one "$group" "$repo" "$relfile" "$url" "$dest" "$want" "$verified"
    rc=$?
    set -e
    case "$rc" in
      0) ;;
      2) MISMATCH=$((MISMATCH + 1)) ;;
      *) FAILED=$((FAILED + 1)) ;;
    esac
  done < "$PLAN_FILE"
  [ "$FAILED" -eq 0 ]   || die "$EXIT_DOWNLOAD" "$FAILED 个文件下载失败。重跑本脚本会从断点续传。"
  [ "$MISMATCH" -eq 0 ] || die "$EXIT_SIZE_MISMATCH" "$MISMATCH 个文件大小与清单不符。清单可能过期，核对 models.yaml 后再跑。"
  ok "权重全部就位"
else
  warn "跳过下载（--skip-download / --offline）"
fi

# ---------------------------------------------------------------- 5 起服务

step "启动 ComfyUI"

health_url="http://127.0.0.1:$PORT/system_stats"

is_up() { curl -fsS --max-time 3 "$health_url" >/dev/null 2>&1; }

if [ "$DO_START" = 0 ]; then
  warn "--no-start：不启动服务"
elif is_up; then
  ok "端口 $PORT 上已有健康的 ComfyUI，不重复启动（幂等）"
elif [ "$DRY_RUN" = 1 ]; then
  log "    [dry-run] $VPY $COMFY_DIR/main.py --listen $LISTEN --port $PORT"
else
  LOGFILE="$LOG_DIR/comfyui-$(date -u +%Y%m%d-%H%M%S).log"
  # --listen 0.0.0.0 是云机器必须的：默认只绑 127.0.0.1，端口转发出去也连不上。
  nohup "$VPY" "$COMFY_DIR/main.py" \
    --listen "$LISTEN" --port "$PORT" \
    >"$LOGFILE" 2>&1 &
  echo $! > "$STATE_DIR/comfyui.pid"
  ok "已拉起，pid=$(cat "$STATE_DIR/comfyui.pid")，日志 $LOGFILE"

  step "健康检查"
  for i in $(seq 1 60); do
    if is_up; then
      ok "第 ${i} 次探活成功：$health_url"
      break
    fi
    if ! kill -0 "$(cat "$STATE_DIR/comfyui.pid")" 2>/dev/null; then
      tail -40 "$LOGFILE" >&2
      die "$EXIT_START" "ComfyUI 进程已退出，日志见上。"
    fi
    sleep 5
    [ "$i" -lt 60 ] || { tail -40 "$LOGFILE" >&2; die "$EXIT_HEALTH" "5 分钟内未就绪"; }
  done

  # 光能连上不够：节点没加载成功时 /system_stats 照样返回 200。
  # 真正要确认的是 Wan 的核心节点在 object_info 里。
  if curl -fsS --max-time 20 "http://127.0.0.1:$PORT/object_info" \
      | grep -q "WanImageToVideo"; then
    ok "WanImageToVideo 节点已注册"
  else
    warn "object_info 里没有 WanImageToVideo —— ComfyUI 版本可能过旧，i2v 工作流会跑不起来"
  fi
fi

# ---------------------------------------------------------------- 6 冒烟

if [ "$DO_SMOKE" = 1 ] && [ "$DRY_RUN" = 0 ]; then
  step "冒烟：用 longfilm 的 comfy_local provider 提交一镜"
  if [ -d "$SCRIPT_DIR/../src/longfilm" ]; then
    COMFY_URL="http://127.0.0.1:$PORT" \
    PYTHONPATH="$SCRIPT_DIR/../src" \
      "$VPY" -m longfilm.providers.comfy_local \
      || die "$EXIT_SMOKE" "comfy_local 自测未通过"
    ok "冒烟通过"
  else
    warn "找不到 src/longfilm，跳过冒烟（本脚本可以脱离仓库单独用）"
  fi
fi

# ---------------------------------------------------------------- 收尾

ELAPSED=$(( $(date +%s) - T0 ))
printf '\n\033[1m全部完成\033[0m（耗时 %dm%ds）\n' "$((ELAPSED / 60))" "$((ELAPSED % 60))" >&2
cat >&2 <<EOF
  ComfyUI      : http://<公网地址>:$PORT
  权重清单     : $MODELS_FILE (profile=$PROFILE)
  权重实体     : $STAGE_DIR （ComfyUI/models 下是软链）
  训练侧权重   : $TRAIN_DIR
  日志         : $LOG_DIR
  再跑一次本脚本是安全的：已完成的步骤会跳过。

  下一步：
    export COMFY_URL=http://127.0.0.1:$PORT
    PYTHONPATH=src python -m longfilm.providers.comfy_local     # 探活
    bash deploy/train_lora_pod.sh --help                        # 训角色 LoRA
EOF
exit "$EXIT_OK"
