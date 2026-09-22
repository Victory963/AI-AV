# longfilm 出片节点：ComfyUI + longfilm 产线，一个镜像两种用法
#   - 常驻：docker compose up（compose 里起 comfy + api 两个服务）
#   - serverless：容器命令覆盖为 python -u deploy/runpod_serverless_handler.py
#
# 基础镜像选 devel 而不是 runtime：ComfyUI 的若干自定义节点（尤其是
# FlashVSR 的 Block-Sparse Attention）需要在容器内编译 CUDA 扩展，
# runtime 镜像没有 nvcc，装到一半才失败。
#
# 查证 2026-09-20：ComfyUI README 现推荐 CUDA 12.8+/13.0 的 torch 轮子。
# 这里锁 12.8 是因为它同时被 torch 稳定版和绝大多数自定义节点支持；
# 要上 13.0 请同时改 bootstrap_gpu.sh 的 TORCH_INDEX。
ARG CUDA_TAG=12.8.1-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/workspace/ComfyUI \
    LONGFILM_ROOT=/opt/longfilm \
    PYTHONPATH=/opt/longfilm/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git git-lfs curl ca-certificates \
        ffmpeg fonts-noto-cjk build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && git lfs install

# ---- ComfyUI ----------------------------------------------------------------
ARG COMFY_REF=master
RUN git clone --depth 1 --branch ${COMFY_REF} https://github.com/comfyanonymous/ComfyUI ${COMFY_ROOT} \
 && python -m pip install --upgrade pip \
 && python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
 && python -m pip install -r ${COMFY_ROOT}/requirements.txt

# ComfyUI-Manager 必须落在这个目录名，换名字 ComfyUI 就认不出来
RUN git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager \
        ${COMFY_ROOT}/custom_nodes/comfyui-manager \
 && python -m pip install -r ${COMFY_ROOT}/custom_nodes/comfyui-manager/requirements.txt || true

# ---- longfilm ---------------------------------------------------------------
# 先装依赖再拷代码：改一行 Python 不该让 torch 那一层缓存失效
COPY pyproject.toml ${LONGFILM_ROOT}/
RUN python -m pip install pydantic numpy pillow requests edge-tts

COPY src ${LONGFILM_ROOT}/src
COPY deploy ${LONGFILM_ROOT}/deploy
COPY scripts ${LONGFILM_ROOT}/scripts
COPY configs ${LONGFILM_ROOT}/configs
COPY assets ${LONGFILM_ROOT}/assets

WORKDIR ${LONGFILM_ROOT}

# 权重不进镜像（几十 GB，且更新频率完全不同）。
# 挂 volume 到 /workspace/ComfyUI/models，或在容器里跑 bootstrap_gpu.sh 下载。
VOLUME ["/workspace/ComfyUI/models", "/workspace/out"]

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8188/system_stats || exit 1

CMD ["python", "-u", "/workspace/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"]
