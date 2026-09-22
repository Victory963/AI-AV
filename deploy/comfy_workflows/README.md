# ComfyUI workflow 模板

`longfilm.providers.comfy_local.ComfyProvider` 启动时会把本目录下所有 `*.json`
当作 **API 格式**（ComfyUI 菜单里的 *Export (API)*，不是 *Save*）的 workflow 模板加载。

| 文件 | mode | 用途 |
| --- | --- | --- |
| `wan_i2v_lora.json` | `i2v` | Wan 类首/尾帧驱动 + LoRA 链，产线主力 |
| `wan_t2v_placeholder.json` | `t2v` | 纯文生视频占位模板，**结构对、模型名是假的** |

## 上线前必须替换的占位符

模板里所有 `PLACEHOLDER_` 开头的值都是假文件名，照原样提交会被 ComfyUI 以
`node_errors` 打回。按本机 `ComfyUI/models/` 下的实际文件改：

| 占位符 | 放在哪 | 说明 |
| --- | --- | --- |
| `PLACEHOLDER_wan_i2v_14b.safetensors` | `models/diffusion_models/` | I2V 主干 |
| `PLACEHOLDER_wan_t2v_14b.safetensors` | `models/diffusion_models/` | T2V 主干 |
| `PLACEHOLDER_umt5_xxl.safetensors` | `models/text_encoders/` | Wan 的文本编码器 |
| `PLACEHOLDER_wan_vae.safetensors` | `models/vae/` | |
| `PLACEHOLDER_clip_vision_h.safetensors` | `models/clip_vision/` | 首帧视觉编码，仅 i2v 用 |
| `PLACEHOLDER_character.safetensors` / `PLACEHOLDER_style.safetensors` | `models/loras/` | 不挂 LoRA 时 provider 会**自动把该节点旁路**，无需真实存在 |
| `PLACEHOLDER_start.png` / `PLACEHOLDER_end.png` | — | 运行时被 `/upload/image` 返回的真实文件名覆盖 |

`class_type` 也要按本机版本核对一次：`curl $COMFY_URL/object_info | jq keys`。
`WanFirstLastFrameToVideo`、`EmptyHunyuanLatentVideo`、`LoraLoaderModelOnly` 是
ComfyUI 原生节点，**`VHS_VideoCombine` 来自 ComfyUI-VideoHelperSuite 自定义节点包**，
没装就把保存节点换成原生 `SaveWEBM`，并同步改 `_longfilm.bind` 里的
`fps` / `filename_prefix` 路径。

## `_longfilm` 元数据块

每个模板顶层多一个 `_longfilm` 键，provider 提交前会剥掉它（ComfyUI 不认识）。

```jsonc
"_longfilm": {
  "mode": "i2v",                       // provider 按有无首帧选模板
  "bind": {                            // 语义名 -> JSON 节点路径，注入走路径不走字符串替换
    "positive": "5.inputs.text",
    "length":   "11.inputs.length"
  },
  "lora_node": "4",                    // LoRA 链的头节点；串多个时自动克隆并改接下游
  "end_image_node": "8"                // 未提供尾帧时整条支路被摘除
}
```

换 workflow 时只需要改这一个文件：结构和绑定表在一起，不会出现
「模板换了、绑定表没跟上」的错配。`bind` 里声明的路径**必须存在**，
否则 `inject()` 直接抛 `KeyError` —— 静默跳过会让「参数没生效」变成
要肉眼比对两份几百行 JSON 才能发现的 bug。

## 校验

```bash
PYTHONPATH=src .venv/bin/python -m longfilm.providers.comfy_local   # 离线结构自测
COMFY_URL=http://127.0.0.1:8188 PYTHONPATH=src .venv/bin/python -c "
from longfilm.providers.comfy_local import ComfyProvider; print(ComfyProvider().health())"
```
