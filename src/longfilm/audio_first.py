"""音频先行产线 —— 先把对白轨钉死，再用实测时长反推画面。

为什么音频先行：视频引擎的单段上限是 5-15s，而一句对白说多久是**客观事实**，
不是可以拍脑袋填的参数。先出画再配音，只能靠变速或补黑场把声音塞进画里，
口型和呼吸必然崩。反过来先合成对白、用 ffmpeg 实测时长、再把镜头时长对齐到
这个实测值，镜头切点天然落在换气处，口型参考也能直接喂给支持音频驱动的引擎。

本模块的三个产出，按重要性排序：
1. 回填 ``DialogueLine.start_s / end_s``（**镜内相对秒**，见 DialogueTrack 的说明）；
2. 按实测时长重排 ``Shot.duration_s``，超过 provider 单段上限时自动拆镜；
3. 每镜合成一条对白 stem，挂进 ``Shot.refs.audios``（role="dialogue"）。

外部事实查证日期 2026-09-19，来源见 README_TTS 常量。
"""

from __future__ import annotations

import abc
import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from ._proc import popen as _sp_popen  # noqa: F401
from ._proc import run as _sp_run
from .providers.base import FailureKind, ProviderError
from .schema import AudioRef, DialogueLine, Scene, Shot, Storyboard, VoiceProfile

log = logging.getLogger(__name__)


# ============================================================ 开源 TTS 选型备忘
# 查证日期 2026-09-19。许可证差异直接决定能不能商用，写在代码里免得后面踩雷。
README_TTS: dict[str, dict[str, str]] = {
    "index-tts2": {
        "repo": "https://github.com/index-tts/index-tts",
        "license": "bilibili Model Use License Agreement（**不是** Apache/MIT，商用前必须读原文）",
        "serving": "官方只给 WebUI(127.0.0.1:7860) 与 Python API；生产部署官方指向 vLLM recipe。"
                   "没有官方 REST 规格 —— 用 HttpTTSBackend 对接自建包装层。",
        "gpu": "需要 CUDA，基准跑在 RTX 4090。本机无 GPU，跑不了。",
    },
    "cosyvoice2": {
        "repo": "https://github.com/FunAudioLLM/CosyVoice",
        "license": "Apache-2.0",
        "serving": "仓库自带 runtime/python/fastapi/server.py，client.py 默认端口 50000，"
                   "模式 sft|zero_shot|cross_lingual|instruct。"
                   "TODO(2026-09-19)：请求体字段名未在官方 README 列出，"
                   "接入前需读 server.py 确认，勿凭猜测填。",
        "gpu": "官方 Docker 用 --runtime=nvidia，按需 GPU。",
    },
    "f5-tts": {
        "repo": "https://github.com/SWivid/F5-TTS",
        "license": "代码 MIT；**预训练权重 CC-BY-NC（禁止商用）** —— 商业长片产线不能直接用官方权重。",
        "serving": "官方只给 Gradio(f5-tts_infer-gradio) 与 Triton/TensorRT-LLM 部署，无标准 REST。",
        "gpu": "安装说明以 CUDA/ROCm/XPU 为前提。",
    },
    "gpt-sovits": {
        "repo": "https://github.com/RVC-Boss/GPT-SoVITS",
        "license": "MIT",
        "serving": "自带 api_v2.py，GET/POST /tts，字段 text / text_lang / ref_audio_path / "
                   "prompt_text / prompt_lang / text_split_method / media_type / streaming_mode。"
                   "HttpTTSBackend.gpt_sovits() 直接按这套字段发。",
        "gpu": "5s 参考音即可零样本克隆；推理建议 GPU。",
    },
    "edge-tts": {
        "repo": "https://github.com/rany2/edge-tts",
        "license": "GPL-3.0（Python 客户端）；调用的是微软 Edge 在线 TTS 服务，无需 API key",
        "serving": "纯 Python 客户端，走微软云，**需要联网但不需要 GPU**。"
                   "本机可用，是 demo 唯一能出真人声的后端。",
        "gpu": "不需要。",
    },
}


# ============================================================ ffmpeg 封装


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")

# 实测用的解码格式：单声道 s16le，字节数 / (2*SR) 即为精确秒数。
_PROBE_RATE = 48000


def ffmpeg_bin() -> str:
    """定位 ffmpeg。优先用仓库自带的静态二进制，保证各机行为一致。"""
    env = os.environ.get("LONGFILM_FFMPEG")
    if env and Path(env).exists():
        return env
    vendored = Path(__file__).resolve().parents[2] / "bin" / "ffmpeg"
    if vendored.exists():
        return str(vendored)
    found = shutil.which("ffmpeg")
    if found:
        return found
    # imageio-ffmpeg 会随包带一份二进制，作为最后兜底。延迟导入：它不是硬依赖。
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ProviderError(
            FailureKind.BAD_REQUEST,
            "找不到 ffmpeg：请把静态二进制放到 <repo>/bin/ffmpeg，"
            "或设置环境变量 LONGFILM_FFMPEG 指向它",
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: Sequence[str], *, timeout_s: float = 300.0) -> subprocess.CompletedProcess[bytes]:
    return _sp_run(  # noqa: S603 - 参数由本模块构造，不接外部 shell
        list(args), capture_output=True, timeout=timeout_s, check=False
    )


def probe_duration_s(path: str | Path, *, exact: bool = False) -> float:
    """实测音频时长。本产线不估算时长 —— 估算值喂进时间线就是慢性对不齐。

    ``exact=False`` 读容器头里的 Duration（厘秒精度，24fps 下 ~1/4 帧，够用且快）。
    ``exact=True`` 全解码后按采样点数换算，得到**播放器真正会播的长度**；
    mp3 的编码器延迟/补零也会被算进去，做口型对齐时必须用这个。
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise ProviderError(FailureKind.BAD_REQUEST, f"音频文件不存在或为空：{p}")
    ff = ffmpeg_bin()

    if exact:
        r = _run([ff, "-v", "error", "-i", str(p), "-map", "0:a:0",
                  "-f", "s16le", "-ac", "1", "-ar", str(_PROBE_RATE), "-"])
        if r.returncode != 0 or not r.stdout:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"解码失败，无法实测时长：{p}\n{r.stderr.decode('utf-8', 'replace')[-400:]}",
            )
        return len(r.stdout) / (2 * _PROBE_RATE)

    r = _run([ff, "-i", str(p)])
    m = _DURATION_RE.search(r.stderr.decode("utf-8", "replace"))
    if not m:
        raise ProviderError(FailureKind.BAD_REQUEST, f"ffmpeg 未报出 Duration，文件可能损坏：{p}")
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


# ============================================================ TTS 后端抽象


@dataclass(frozen=True)
class TTSTask:
    """一次合成请求。把 VoiceProfile 摊平，后端不必回头认识 schema。"""

    text: str
    voice: VoiceProfile
    emotion: str = "neutral"
    speaker_id: str = ""

    def cache_key(self, backend_name: str) -> str:
        """同文本 + 同音色 + 同后端 = 同结果，可直接命中磁盘缓存少烧一次调用。"""
        blob = json.dumps(
            [backend_name, self.text, self.emotion, self.voice.model_dump(mode="json")],
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TTSBackend(abc.ABC):
    """TTS 后端统一接口。

    和 VideoProvider 一样：后端只负责「把文本变成一个音频文件」，
    不做重试、不做时长估算、不碰 Storyboard。时长一律由 probe_duration_s 实测。
    """

    #: 输出容器后缀，决定落盘文件名
    audio_ext: str = "wav"

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def synthesize(self, task: TTSTask, out_path: Path) -> Path:
        """合成到 out_path 并返回它。失败抛 ProviderError。"""

    def available(self) -> bool:
        """探活。产线在降级到 SilentTTSBackend 之前调用。"""
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} ext={self.audio_ext}>"


# ---------------------------------------------------------------- 离线占位后端


# 中文口播的常见语速区间是 4-6 字/秒；4.8 对应从容的剧情对白。
DEFAULT_CPS_ZH = 4.8
# 标点停顿（秒）。这是占位后端唯一的"表演"，也是拆镜点能落在换气处的原因。
_PAUSE_S: dict[str, float] = {
    "，": 0.18, ",": 0.18, "、": 0.12, "；": 0.25, ";": 0.25,
    "：": 0.20, ":": 0.20, "。": 0.38, ".": 0.38, "！": 0.42, "!": 0.42,
    "？": 0.42, "?": 0.42, "…": 0.45, "—": 0.30, "\n": 0.35,
}
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_DIGIT_RE = re.compile(r"\d")


def estimate_duration_s(
    text: str, *, cps: float = DEFAULT_CPS_ZH, min_s: float = 0.4, head_tail_s: float = 0.12
) -> float:
    """按字数与标点估算朗读时长。**仅供无网无 GPU 的占位后端使用**。

    真实后端合成出来的音频一律走 probe_duration_s 实测，不走这里 ——
    估算值只用来让整条链路在离线环境下仍然能跑出结构正确的时间线。
    """
    cjk = len(_CJK_RE.findall(text))
    # 英文按词计：~2.9 词/秒 对 ~4.8 字/秒，一个词约合 1.65 个汉字的时长
    words = len(_LATIN_WORD_RE.findall(text))
    digits = len(_DIGIT_RE.findall(text))
    units = cjk + words * 1.65 + digits
    pauses = sum(_PAUSE_S.get(ch, 0.0) for ch in text)
    return max(min_s, units / max(cps, 0.1) + pauses + head_tail_s * 2)


class SilentTTSBackend(TTSBackend):
    """离线占位：按文本长度生成等长静音（或蜂鸣），保证无网无 GPU 也能跑通全链路。

    刻意生成**真实音频文件**而不是只返回一个数字：下游的 probe_duration_s、
    stem 混音、时间线对齐走的是和真后端完全相同的代码路径，
    等换上真 TTS 时不会冒出一条"只有占位模式才会走"的分支。
    """

    audio_ext = "wav"

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        cps: float = DEFAULT_CPS_ZH,
        tone_hz: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.cps = cps
        self.tone_hz = tone_hz  # None = 纯静音；给个频率则出蜂鸣，便于人耳定位切点

    @property
    def name(self) -> str:
        return "silent"

    def synthesize(self, task: TTSTask, out_path: Path) -> Path:
        dur = estimate_duration_s(task.text, cps=self.cps / max(task.voice.speed, 0.1))
        if self.tone_hz:
            src = f"sine=frequency={self.tone_hz}:sample_rate={self.sample_rate}:duration={dur:.3f}"
        else:
            src = f"anullsrc=r={self.sample_rate}:cl=mono:d={dur:.3f}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        r = _run([
            ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", src,
            "-t", f"{dur:.3f}", "-ac", "1", "-ar", str(self.sample_rate),
            "-c:a", "pcm_s16le", str(out_path),
        ])
        if r.returncode != 0:
            raise ProviderError(
                FailureKind.SERVER,
                f"占位音频生成失败：{r.stderr.decode('utf-8', 'replace')[-400:]}",
            )
        return out_path


# ---------------------------------------------------------------- edge-tts 后端


# 均已用 `python -m edge_tts --list-voices` 实测存在（2026-09-19）。
_EDGE_DEFAULT_VOICES: dict[str, str] = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AriaNeural",
    "en-us": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
}
# pitch_shift 按半音解释，但 edge-tts 只收 Hz 偏移。用 200Hz 标称基频换算，
# 这是近似值：真实基频随音色而变，需要精确控制时请直接给 tts_voice_id 换音色。
_NOMINAL_F0_HZ = 200.0


class EdgeTTSBackend(TTSBackend):
    """微软 Edge 在线 TTS。纯 Python、无需 GPU、无需 API key，但**需要联网**。

    这是本机唯一能出真人声的后端，demo 价值高；产线上它只适合做预演（previz），
    因为音色不可克隆、情绪不可控（Communicate 不透传 SSML style）。
    """

    audio_ext = "mp3"

    def __init__(self, *, default_voice: str | None = None, timeout_s: float = 60.0) -> None:
        self.default_voice = default_voice
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return "edge-tts"

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False
        return True

    def _voice_for(self, vp: VoiceProfile) -> str:
        if vp.tts_voice_id:
            return vp.tts_voice_id
        if self.default_voice:
            return self.default_voice
        lang = (vp.language or "zh").lower()
        return _EDGE_DEFAULT_VOICES.get(lang) or _EDGE_DEFAULT_VOICES.get(lang.split("-")[0], "") \
            or _EDGE_DEFAULT_VOICES["zh"]

    def synthesize(self, task: TTSTask, out_path: Path) -> Path:
        # 延迟导入：没装 edge-tts 的机器仍然要能 import 本模块。
        try:
            import asyncio

            import edge_tts
        except ImportError as exc:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                "未安装 edge-tts。请执行 `pip install edge-tts`，"
                "或改用 SilentTTSBackend 做离线占位",
            ) from exc

        vp = task.voice
        rate_pct = round((vp.speed - 1.0) * 100)
        pitch_hz = round(_NOMINAL_F0_HZ * (2 ** (vp.pitch_shift / 12.0) - 1.0))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        async def _go() -> None:
            comm = edge_tts.Communicate(
                task.text,
                self._voice_for(vp),
                rate=f"{rate_pct:+d}%",
                pitch=f"{pitch_hz:+d}Hz",
            )
            await comm.save(str(out_path))

        try:
            asyncio.run(asyncio.wait_for(_go(), timeout=self.timeout_s))
        except TimeoutError as exc:
            raise ProviderError(FailureKind.TIMEOUT, f"edge-tts 合成超时（{self.timeout_s}s）") from exc
        except Exception as exc:  # edge-tts 把网络与服务端错误混成多种异常类型
            raise ProviderError(
                FailureKind.SERVER, f"edge-tts 合成失败（通常是网络不可达）：{exc}"
            ) from exc
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise ProviderError(FailureKind.SERVER, f"edge-tts 未产出音频：{out_path}")
        return out_path


# ---------------------------------------------------------------- 通用 HTTP 后端


@dataclass
class HttpTTSSpec:
    """自建 TTS 服务的接线说明。字段名可配 —— 各家开源项目的 API 各不相同。"""

    endpoint: str
    method: Literal["POST", "GET"] = "POST"
    text_field: str = "text"
    static_payload: dict[str, Any] = field(default_factory=dict)
    voice_field: str | None = None
    speed_field: str | None = None
    emotion_field: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    api_key_env: str | None = None            # 密钥只从环境变量读，绝不落代码/日志
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "

    response: Literal["audio", "json"] = "audio"
    json_audio_b64_field: str = ""            # 点分路径，如 "data.audio"
    json_audio_url_field: str = ""
    audio_ext: str = "wav"
    timeout_s: float = 120.0
    encode: Literal["json", "form"] = "json"


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _http_request(
    url: str, *, method: str, headers: dict[str, str], body: bytes | None, timeout_s: float
) -> tuple[int, bytes, str]:
    """优先用 requests，未装则退回 urllib。返回 (status, body, content_type)。"""
    try:
        import requests
    except ImportError:
        pass
    else:
        resp = requests.request(method, url, headers=headers, data=body, timeout=timeout_s)
        return resp.status_code, resp.content, resp.headers.get("Content-Type", "")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
            return r.status, r.read(), r.headers.get("Content-Type", "") or ""
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "") or ""
    except urllib.error.URLError as exc:
        raise ProviderError(FailureKind.SERVER, f"TTS 服务不可达：{url}（{exc.reason}）") from exc


class HttpTTSBackend(TTSBackend):
    """通用 HTTP TTS 后端，对接自建的 IndexTTS-2 / CosyVoice2 / GPT-SoVITS 服务。

    刻意做成「字段可配」而不是给每家写一个子类：开源 TTS 的 API 一年能改三回，
    改一行配置比改一个类省事，也避免在产线里堆一串只差字段名的重复实现。
    """

    def __init__(self, spec: HttpTTSSpec, *, name: str = "http-tts") -> None:
        self.spec = spec
        self._name = name
        self.audio_ext = spec.audio_ext

    @property
    def name(self) -> str:
        return self._name

    # ---- 预设：字段名已查证，见 README_TTS ----

    @classmethod
    def gpt_sovits(
        cls,
        base_url: str = "http://127.0.0.1:9880",
        *,
        ref_audio_path: str,
        prompt_text: str = "",
        prompt_lang: str = "zh",
        text_lang: str = "zh",
        media_type: str = "wav",
        text_split_method: str = "cut5",
    ) -> HttpTTSBackend:
        """GPT-SoVITS api_v2.py 的 /tts。字段名来自官方 api_v2.py（查证 2026-09-19）。"""
        return cls(
            HttpTTSSpec(
                endpoint=f"{base_url.rstrip('/')}/tts",
                static_payload={
                    "text_lang": text_lang,
                    "ref_audio_path": ref_audio_path,
                    "prompt_text": prompt_text,
                    "prompt_lang": prompt_lang,
                    "text_split_method": text_split_method,
                    "media_type": media_type,
                    "streaming_mode": False,
                },
                speed_field="speed_factor",
                audio_ext=media_type,
            ),
            name="gpt-sovits",
        )

    @classmethod
    def openai_compatible(
        cls,
        base_url: str,
        *,
        model: str,
        api_key_env: str = "TTS_API_KEY",
        response_format: str = "wav",
    ) -> HttpTTSBackend:
        """OpenAI /v1/audio/speech 形态的自建网关（很多开源项目提供兼容层）。

        TODO(2026-09-19)：未逐字段核对上游官方文档，接入自建网关时请按实际实现校正。
        """
        return cls(
            HttpTTSSpec(
                endpoint=f"{base_url.rstrip('/')}/v1/audio/speech",
                text_field="input",
                static_payload={"model": model, "response_format": response_format},
                voice_field="voice",
                speed_field="speed",
                api_key_env=api_key_env,
                audio_ext=response_format,
            ),
            name="openai-tts",
        )

    # ---- 主路径 ----

    def _auth_headers(self) -> dict[str, str]:
        s = self.spec
        h = dict(s.headers)
        if s.api_key_env:
            key = os.environ.get(s.api_key_env, "").strip()
            if not key:
                raise ProviderError(
                    FailureKind.AUTH,
                    f"{self.name} 需要环境变量 {s.api_key_env} 提供密钥，当前未设置",
                )
            h[s.api_key_header] = f"{s.api_key_prefix}{key}"
        return h

    def synthesize(self, task: TTSTask, out_path: Path) -> Path:
        s = self.spec
        payload: dict[str, Any] = dict(s.static_payload)
        payload[s.text_field] = task.text
        if s.voice_field and task.voice.tts_voice_id:
            payload[s.voice_field] = task.voice.tts_voice_id
        if s.speed_field:
            payload[s.speed_field] = task.voice.speed
        if s.emotion_field:
            payload[s.emotion_field] = task.emotion

        headers = self._auth_headers()
        if s.method == "GET":
            from urllib.parse import urlencode

            url = f"{s.endpoint}?{urlencode({k: _qs(v) for k, v in payload.items()})}"
            body = None
        else:
            url = s.endpoint
            if s.encode == "form":
                from urllib.parse import urlencode

                body = urlencode({k: _qs(v) for k, v in payload.items()}).encode()
                headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            else:
                body = json.dumps(payload, ensure_ascii=False).encode()
                headers.setdefault("Content-Type", "application/json")

        status, raw, ctype = _http_request(
            url, method=s.method, headers=headers, body=body, timeout_s=s.timeout_s
        )
        # 日志里只记 endpoint 与状态码，绝不记 headers（含密钥）或完整 body。
        log.debug("%s 合成 %d 字 -> HTTP %s", self.name, len(task.text), status)
        if status in (401, 403):
            raise ProviderError(FailureKind.AUTH, f"{self.name} 鉴权失败（HTTP {status}）")
        if status == 429:
            raise ProviderError(FailureKind.RATE_LIMIT, f"{self.name} 限流（HTTP 429）")
        if status >= 500:
            raise ProviderError(FailureKind.SERVER, f"{self.name} 服务端错误（HTTP {status}）")
        if status >= 400:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"{self.name} 拒绝请求（HTTP {status}）：{raw[:300].decode('utf-8', 'replace')}",
            )

        data = self._extract_audio(raw, ctype)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    def _extract_audio(self, raw: bytes, ctype: str) -> bytes:
        s = self.spec
        if s.response == "audio" and "application/json" not in ctype.lower():
            if not raw:
                raise ProviderError(FailureKind.SERVER, f"{self.name} 返回空音频")
            return raw
        try:
            doc = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise ProviderError(FailureKind.SERVER, f"{self.name} 返回既非音频也非 JSON") from exc
        if s.json_audio_b64_field:
            b64 = _dig(doc, s.json_audio_b64_field)
            if isinstance(b64, str):
                return base64.b64decode(b64)
        if s.json_audio_url_field:
            uri = _dig(doc, s.json_audio_url_field)
            if isinstance(uri, str):
                st, blob, _ = _http_request(
                    uri, method="GET", headers={}, body=None, timeout_s=s.timeout_s
                )
                if st == 200 and blob:
                    return blob
        raise ProviderError(
            FailureKind.SERVER,
            f"{self.name} 的 JSON 响应里找不到音频（配置的字段："
            f"{s.json_audio_b64_field or s.json_audio_url_field or '未配置'}）",
        )


def _qs(v: Any) -> str:
    return json.dumps(v) if isinstance(v, bool) else str(v)


# ============================================================ 对白轨


@dataclass
class DialogueCue:
    """一条已合成的对白。同时记镜内相对时间与全片绝对时间。"""

    shot_id: str
    scene_id: str
    line_index: int
    speaker_id: str
    text: str
    audio_uri: str
    duration_s: float
    shot_offset_s: float      # 相对本镜头首帧
    abs_start_s: float        # 全片绝对时间（重排镜头后会失效，需重算）
    emotion: str = "neutral"

    @property
    def shot_end_s(self) -> float:
        return self.shot_offset_s + self.duration_s


@dataclass
class DialogueTrack:
    """整条对白轨。

    **时间基准约定**：``DialogueLine.start_s / end_s`` 存的是**镜内相对秒**，
    不是全片绝对秒。理由：Shot 是原子生成单元，喂给引擎的音频参考是按镜切好的片段；
    而 retime / 拆镜 / 重排会不断改变绝对时间，却不会改变一句话在它所属镜头里的位置。
    把相对值写进 schema，绝对值交给 timeline.py 现算，是唯一不会越改越乱的分工。
    """

    cues: list[DialogueCue] = field(default_factory=list)
    backend: str = ""
    out_dir: str = ""
    lead_in_s: float = 0.35
    gap_s: float = 0.30
    tail_s: float = 0.45

    def of_shot(self, shot_id: str) -> list[DialogueCue]:
        return [c for c in self.cues if c.shot_id == shot_id]

    def shot_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.cues:
            seen.setdefault(c.shot_id, None)
        return list(seen)

    def speech_duration_s(self) -> float:
        return sum(c.duration_s for c in self.cues)

    def required_shot_duration_s(self, shot_id: str) -> float:
        """这一镜要装下它的对白，最少需要多长（含头尾留白）。"""
        cs = self.of_shot(shot_id)
        if not cs:
            return 0.0
        return max(c.shot_end_s for c in cs) + self.tail_s


# ---------------------------------------------------------------- 合成主入口


def synthesize_dialogue(
    storyboard: Storyboard,
    backend: TTSBackend,
    out_dir: str | Path,
    *,
    lead_in_s: float = 0.35,
    gap_s: float = 0.30,
    tail_s: float = 0.45,
    exact_probe: bool = True,
    use_cache: bool = True,
) -> DialogueTrack:
    """逐条合成对白，实测时长，并回填每条 DialogueLine 的 start_s / end_s。

    这是音频先行的关键产出：跑完之后，分镜里每句话说多久、在镜内什么位置，
    都是**实测事实**而非估计值，后面的 retime / 时间线 / 字幕全部以它为准。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    track = DialogueTrack(
        backend=backend.name, out_dir=str(out.resolve()),
        lead_in_s=lead_in_s, gap_s=gap_s, tail_s=tail_s,
    )

    abs_cursor = 0.0
    for scene in storyboard.scenes:
        for shot in scene.shots:
            cursor = lead_in_s if shot.dialogue else 0.0
            for i, line in enumerate(shot.dialogue):
                char = storyboard.character(line.speaker_id)
                voice = char.voice if char else VoiceProfile()
                task = TTSTask(
                    text=line.text, voice=voice,
                    emotion=line.emotion or voice.emotion_default,
                    speaker_id=line.speaker_id,
                )
                fname = f"{shot.id}_{i:02d}_{task.cache_key(backend.name)}.{backend.audio_ext}"
                path = out / fname
                if use_cache and path.exists() and path.stat().st_size > 0:
                    log.debug("命中缓存，跳过合成：%s", fname)
                else:
                    backend.synthesize(task, path)

                dur = probe_duration_s(path, exact=exact_probe)
                # 回填到契约对象 —— 镜内相对时间，见 DialogueTrack 的基准约定
                line.start_s = round(cursor, 3)
                line.end_s = round(cursor + dur, 3)
                track.cues.append(DialogueCue(
                    shot_id=shot.id, scene_id=scene.id, line_index=i,
                    speaker_id=line.speaker_id, text=line.text,
                    audio_uri=str(path.resolve()), duration_s=dur,
                    shot_offset_s=round(cursor, 3),
                    abs_start_s=round(abs_cursor + cursor, 3),
                    emotion=task.emotion,
                ))
                cursor += dur + gap_s
            abs_cursor += shot.duration_s

    log.info(
        "对白合成完成：%d 条 / %.2fs 净语音，后端 %s",
        len(track.cues), track.speech_duration_s(), backend.name,
    )
    return track


# ---------------------------------------------------------------- 按音频反推镜长


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def retime_shots_to_audio(
    storyboard: Storyboard,
    track: DialogueTrack,
    *,
    min_s: float = 4.0,
    max_s: float = 10.0,
    round_to: float = 0.5,
    keep_silent_shots: bool = True,
) -> list[str]:
    """按实测对白时长反推镜头时长；装不下就拆镜。返回中文调整报告。

    ``max_s`` 应取目标 provider 的 ``Capabilities.max_duration_s``。
    拆镜点只落在**句与句之间的停顿**上，不会切断一句话 —— 切在换气处观众察觉不到，
    切在字中间则必然穿帮。拆出来的后半段会自动声明 ``inherit_last_frame``，
    让引擎从前半段的尾帧接着演，这是长镜头唯一能跨 provider 上限的办法。
    """
    report: list[str] = []
    if min_s > max_s:
        raise ValueError(f"min_s({min_s}) 不能大于 max_s({max_s})")

    for scene in storyboard.scenes:
        new_shots: list[Shot] = []
        for shot in scene.shots:
            cues = track.of_shot(shot.id)
            if not cues:
                old = shot.duration_s
                if keep_silent_shots:
                    shot.duration_s = min(max(old, min_s), max_s)
                    if abs(shot.duration_s - old) > 1e-6:
                        report.append(
                            f"[{shot.id}] 无对白，时长 {old:.2f}s -> {shot.duration_s:.2f}s（夹到引擎区间）"
                        )
                new_shots.append(shot)
                continue

            need = track.required_shot_duration_s(shot.id)
            if need <= max_s + 1e-6:
                old = shot.duration_s
                shot.duration_s = min(max(_round_up(need, round_to), min_s), max_s)
                report.append(
                    f"[{shot.id}] 对白实测需 {need:.2f}s（{len(cues)} 句），"
                    f"时长 {old:.2f}s -> {shot.duration_s:.2f}s"
                )
                new_shots.append(shot)
                continue

            parts = _split_shot_by_dialogue(
                shot, cues, track, max_s=max_s, min_s=min_s, round_to=round_to
            )
            report.append(
                f"[{shot.id}] 对白实测需 {need:.2f}s，超过 provider 单段上限 {max_s:.1f}s，"
                f"在换气处拆成 {len(parts)} 段：" + "、".join(
                    f"{p.id}({p.duration_s:.2f}s/{len(p.dialogue)}句)" for p in parts
                )
            )
            new_shots.extend(parts)

        # 拆镜后重排 index 并修复场景内的前后链，否则 validate_continuity 会挂
        for i, s in enumerate(new_shots):
            s.index = i
        scene.shots = new_shots

    _relink_continuity(storyboard)
    _refresh_absolute_times(storyboard, track)
    return report


def _split_shot_by_dialogue(
    shot: Shot,
    cues: list[DialogueCue],
    track: DialogueTrack,
    *,
    max_s: float,
    min_s: float,
    round_to: float,
) -> list[Shot]:
    """把一镜按对白停顿切成多镜，就地更新 cues 与 DialogueLine 的相对时间。"""
    cues = sorted(cues, key=lambda c: c.shot_offset_s)
    groups: list[list[DialogueCue]] = []
    cur: list[DialogueCue] = []
    used = track.lead_in_s
    for c in cues:
        add = c.duration_s + (track.gap_s if cur else 0.0)
        if cur and used + add + track.tail_s > max_s:
            groups.append(cur)
            cur, used = [], track.lead_in_s
            add = c.duration_s
        cur.append(c)
        used += add
    if cur:
        groups.append(cur)

    parts: list[Shot] = []
    suffixes = "abcdefghijklmnopqrstuvwxyz"
    for gi, group in enumerate(groups):
        part = shot.model_copy(deep=True)
        part.id = f"{shot.id}{suffixes[gi]}" if len(groups) <= len(suffixes) else f"{shot.id}p{gi}"
        # 后续段落从前一段的尾帧续演，保证人物姿态/光位不跳
        if gi > 0:
            part.continuity.inherit_last_frame = True
            part.continuity.prev_shot_id = parts[-1].id
            parts[-1].continuity.emit_last_frame = True
            parts[-1].continuity.next_shot_id = part.id
        # 渲染产物属于原镜，拆开后必须作废，否则会把整镜的视频重复挂到每一段上
        part.render_uri = None
        part.job_id = None
        part.provider_used = None
        part.qc_score = None

        cursor = track.lead_in_s
        lines: list[DialogueLine] = []
        for li, c in enumerate(group):
            src = shot.dialogue[c.line_index]
            line = src.model_copy(deep=True)
            line.start_s = round(cursor, 3)
            line.end_s = round(cursor + c.duration_s, 3)
            lines.append(line)
            # 就地改 cue：它现在属于新镜头
            c.shot_id = part.id
            c.line_index = li
            c.shot_offset_s = round(cursor, 3)
            cursor += c.duration_s + track.gap_s
        part.dialogue = lines
        need = cursor - track.gap_s + track.tail_s
        part.duration_s = min(max(_round_up(need, round_to), min_s), max_s)
        parts.append(part)
    return parts


def _relink_continuity(storyboard: Storyboard) -> None:
    """按场景内的实际顺序重建 prev/next 链。拆镜后必须调用。"""
    for scene in storyboard.scenes:
        for a, b in zip(scene.shots, scene.shots[1:]):
            a.continuity.next_shot_id = b.id
            if b.continuity.prev_shot_id is None or b.continuity.inherit_last_frame:
                b.continuity.prev_shot_id = a.id


def _refresh_absolute_times(storyboard: Storyboard, track: DialogueTrack) -> None:
    """镜长变了，cue 的全片绝对时间随之重算（相对时间不动）。"""
    by_shot: dict[str, list[DialogueCue]] = {}
    for c in track.cues:
        by_shot.setdefault(c.shot_id, []).append(c)
    cursor = 0.0
    for shot in storyboard.all_shots():
        for c in by_shot.get(shot.id, []):
            c.abs_start_s = round(cursor + c.shot_offset_s, 3)
        cursor += shot.duration_s


# ---------------------------------------------------------------- 音频参考位


def build_audio_refs(
    track: DialogueTrack,
    storyboard: Storyboard,
    *,
    stem_dir: str | Path | None = None,
    sample_rate: int = 24000,
    weight: float = 1.0,
) -> list[str]:
    """为每镜混出一条对白 stem 并挂进 ``Shot.refs.audios``（role="dialogue"）。

    刻意**每镜只挂一条**而不是每句一条：RefPack 只有 3 个音频位，三句话就满了；
    更要紧的是，支持音频驱动的引擎期望的是一条与镜头等长、时间位置已经对齐的轨，
    而不是几段需要它自己去猜位置的碎片。混音在这里做，引擎那头就不会猜错。
    """
    out = Path(stem_dir) if stem_dir else Path(track.out_dir) / "stems"
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    for shot in storyboard.all_shots():
        cues = sorted(track.of_shot(shot.id), key=lambda c: c.shot_offset_s)
        # 无论有没有对白都先清掉旧的 dialogue 参考：拆镜/重配后留着旧轨是最隐蔽的错位源
        shot.refs.audios = [a for a in shot.refs.audios if a.role != "dialogue"]
        if not cues:
            continue
        stem = out / f"{shot.id}_dialogue.wav"
        _render_shot_stem(cues, stem, duration_s=shot.duration_s, sample_rate=sample_rate)
        # note 里带 source= 标记：compliance.parse_source_type 要求每一份参考素材
        # 都能说清来源，而这条 stem 是本产线自己用 TTS 合成的，属于 ai_generated。
        # 让生成方顺手标上，比让使用者事后逐条补要可靠 —— 漏标的后果是发片前
        # 被合规门禁拦下，那时候已经晚了。
        shot.refs.audios.append(AudioRef(
            role="dialogue", uri=str(stem.resolve()), weight=weight,
            note=(f"source=ai_generated {len(cues)} 句对白（TTS 合成），"
                  f"已按镜内相对时间对齐，全长 {shot.duration_s:.2f}s"),
        ))
        made.append(str(stem.resolve()))
    log.info("生成对白 stem %d 条", len(made))
    return made


def _render_shot_stem(
    cues: Sequence[DialogueCue], out_path: Path, *, duration_s: float, sample_rate: int
) -> Path:
    """把若干句对白按相对时间摆进一条与镜头等长的单声道轨。"""
    if duration_s <= 0:
        raise ValueError(f"镜头时长必须为正，收到 {duration_s}")
    args: list[str] = [ffmpeg_bin(), "-y", "-v", "error",
                       "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono"]
    for c in cues:
        args += ["-i", c.audio_uri]

    chains = [f"[0:a]atrim=duration={duration_s:.3f},asetpts=N/SR/TB[base]"]
    labels = ["[base]"]
    for i, c in enumerate(cues, start=1):
        delay_ms = max(0, int(round(c.shot_offset_s * 1000)))
        chains.append(
            f"[{i}:a]aresample={sample_rate},"
            f"aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={delay_ms}:all=1[d{i}]"
        )
        labels.append(f"[d{i}]")
    # normalize=0：amix 默认会按输入数衰减，那会让对白越多越轻
    chains.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=first[mix]"
    )

    args += ["-filter_complex", ";".join(chains), "-map", "[mix]",
             "-t", f"{duration_s:.3f}", "-ac", "1", "-ar", str(sample_rate),
             "-c:a", "pcm_s16le", str(out_path)]
    r = _run(args)
    if r.returncode != 0:
        raise ProviderError(
            FailureKind.SERVER,
            f"对白 stem 混音失败：{r.stderr.decode('utf-8', 'replace')[-500:]}",
        )
    return out_path


def pick_backend(prefer: str = "auto", **kw: Any) -> TTSBackend:
    """挑一个当前环境真能用的后端。无网无 GPU 时一定能退到 SilentTTSBackend。"""
    if prefer in ("edge", "edge-tts", "auto"):
        edge = EdgeTTSBackend(**kw)
        if edge.available():
            return edge
        if prefer != "auto":
            raise ProviderError(FailureKind.BAD_REQUEST, "edge-tts 未安装")
        log.info("edge-tts 不可用，退回离线占位后端")
    return SilentTTSBackend()


# ============================================================ 自测


def _demo_storyboard() -> Storyboard:
    from .schema import Appearance, CharacterBible, ShotSize

    chars = [
        CharacterBible(
            id="lin", name="林砚", age_statement="虚构角色，设定年龄 29 岁，成年人",
            appearance=Appearance(face="清瘦", hair="黑色短发", distinguishing="左眉一道浅疤"),
            voice=VoiceProfile(language="zh", tts_voice_id="zh-CN-XiaoxiaoNeural", speed=1.0),
        ),
        CharacterBible(
            id="zhou", name="周决", age_statement="虚构角色，设定年龄 34 岁，成年人",
            appearance=Appearance(face="方正", hair="寸头"),
            voice=VoiceProfile(language="zh", tts_voice_id="zh-CN-YunyangNeural", speed=1.0),
        ),
    ]
    s1 = Scene(id="sc01", title="档案室", location="旧档案室", time_of_day="night", shots=[
        Shot(id="sh01", scene_id="sc01", index=0, duration_s=8.0, shot_size=ShotSize.MS,
             subject_ids=["lin"], action="林砚推开积灰的柜门",
             dialogue=[DialogueLine(speaker_id="lin", text="这批卷宗，十年没人动过了。")]),
        Shot(id="sh02", scene_id="sc01", index=1, duration_s=8.0, shot_size=ShotSize.OTS,
             subject_ids=["lin", "zhou"], action="两人对峙",
             dialogue=[
                 DialogueLine(speaker_id="zhou", text="你确定要往下翻？"),
                 DialogueLine(speaker_id="lin", text="我已经翻到这儿了。", emotion="firm"),
             ]),
        # 这一镜的对白量刻意超过 10s 上限，用来验证自动拆镜
        Shot(id="sh03", scene_id="sc01", index=2, duration_s=8.0, shot_size=ShotSize.CU,
             subject_ids=["zhou"], action="周决长篇陈述",
             dialogue=[
                 DialogueLine(speaker_id="zhou", text="当年那场火，烧掉的不只是仓库。"),
                 DialogueLine(speaker_id="zhou", text="名单上二十七个人，活下来的只有九个。"),
                 DialogueLine(speaker_id="zhou", text="剩下的十八个，档案里写的都是意外。"),
                 DialogueLine(speaker_id="zhou", text="你猜，是谁写的这些档案？"),
             ]),
        Shot(id="sh04", scene_id="sc01", index=3, duration_s=6.0, shot_size=ShotSize.INSERT,
             action="特写：泛黄的名单", sfx=["纸张翻动"]),
    ])
    return Storyboard(project="demo", episode="ep01", characters=chars, scenes=[s1],
                      style_bible="冷调、低照度、35mm 胶片颗粒")


def _selftest() -> None:
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 0) ffmpeg 与实测能力
    ff = ffmpeg_bin()
    print(f"[0] ffmpeg = {ff}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        probe = tmp / "probe.wav"
        _run([ff, "-y", "-v", "error", "-f", "lavfi", "-i",
              "anullsrc=r=24000:cl=mono:d=2.5", "-t", "2.5", str(probe)])
        d_head = probe_duration_s(probe, exact=False)
        d_exact = probe_duration_s(probe, exact=True)
        assert abs(d_head - 2.5) < 0.02, d_head
        assert abs(d_exact - 2.5) < 0.001, d_exact
        print(f"[0] 时长实测 header={d_head:.3f}s exact={d_exact:.6f}s")

        # 1) 估算器的单调性（占位后端的时长模型）
        short, long_ = estimate_duration_s("好。"), estimate_duration_s("这批卷宗，十年没人动过了。")
        assert 0.4 <= short < long_, (short, long_)
        print(f"[1] 时长估算 '好。'={short:.2f}s  长句={long_:.2f}s")

        # 2) 离线占位后端跑通合成 + 回填
        sb = _demo_storyboard()
        backend: TTSBackend = SilentTTSBackend()
        # 有网且装了 edge-tts 时，用真 TTS 跑一遍（由环境变量显式开启，默认不依赖网络）
        if os.environ.get("LONGFILM_SELFTEST_EDGE") == "1":
            e = EdgeTTSBackend()
            assert e.available(), "edge-tts 未安装"
            backend = e
        print(f"[2] 使用后端 {backend.name}")

        track = synthesize_dialogue(sb, backend, tmp / "dialogue")
        assert track.cues, "没有合成出任何对白"
        for c in track.cues:
            assert Path(c.audio_uri).exists() and c.duration_s > 0.1, c
        # 回填检查：每镜内相对时间必须单调且与 cue 一致
        for shot in sb.all_shots():
            prev_end = -1.0
            for i, line in enumerate(shot.dialogue):
                assert line.start_s is not None and line.end_s is not None, f"{shot.id} 未回填"
                assert line.start_s > prev_end - 1e-6, f"{shot.id} 第{i}句时间倒退"
                assert line.end_s > line.start_s
                prev_end = line.end_s
        print(f"[2] 合成 {len(track.cues)} 条，净语音 {track.speech_duration_s():.2f}s；"
              f"sh03 需要 {track.required_shot_duration_s('sh03'):.2f}s")

        # 3) 按实测时长反推镜长 + 自动拆镜
        n_before = len(sb.all_shots())
        report = retime_shots_to_audio(sb, track, min_s=4.0, max_s=10.0)
        n_after = len(sb.all_shots())
        print("[3] 重排报告：")
        for line in report:
            print("     " + line)
        assert n_after > n_before, "sh03 的对白超过 10s，应当被拆开"
        for shot in sb.all_shots():
            assert 4.0 - 1e-6 <= shot.duration_s <= 10.0 + 1e-6, (shot.id, shot.duration_s)
            need = track.required_shot_duration_s(shot.id)
            assert need <= shot.duration_s + 1e-6, f"{shot.id} 装不下对白：需{need} 有{shot.duration_s}"
        assert not sb.validate_continuity(), sb.validate_continuity()
        print(f"[3] 镜头 {n_before} -> {n_after}，全部落在 [4,10]s，连续性检查通过")

        # 4) 每镜对白 stem
        stems = build_audio_refs(track, sb, stem_dir=tmp / "stems")
        assert stems, "没有生成 stem"
        for shot in sb.all_shots():
            refs = [a for a in shot.refs.audios if a.role == "dialogue"]
            if track.of_shot(shot.id):
                assert len(refs) == 1, f"{shot.id} 应恰好挂一条对白参考，实得 {len(refs)}"
                got = probe_duration_s(refs[0].uri, exact=True)
                assert abs(got - shot.duration_s) < 0.05, (shot.id, got, shot.duration_s)
            else:
                assert not refs
        print(f"[4] stem {len(stems)} 条，实测长度与镜长一致（误差 <50ms）")

        # 5) 缓存命中不应改变结果
        again = synthesize_dialogue(sb, backend, tmp / "dialogue")
        assert len(again.cues) == len(track.cues)
        print("[5] 缓存复跑一致")

    print("\naudio_first 自测全部通过 ✓")


if __name__ == "__main__":
    _selftest()
