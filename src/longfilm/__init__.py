"""longfilm — 长时间 AI 数字人拍片系统。

产线分工位：分镜(schema) → 参考位编排(refpack) → 提示词OS(prompt_os)
→ 引擎路由(router/providers) → 续写链(chain) → 质检门禁(qc)
→ 拼接调色(stitch/grade) → 超分(upscale) → 分轨对齐(timeline) → 合规封装(compliance)
"""

__version__ = "0.1.0"
