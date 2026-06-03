"""External LLM provider clients for pseudo-report comparison experiments."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderResponse:
    provider: str
    model: str
    text: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ProviderSpec:
    provider: str
    model: str
    api_key_env: tuple[str, ...]
    base_url: str
    kind: str
    json_mode: bool = True
    price_input_per_1m: float | None = None
    price_output_per_1m: float | None = None

    @property
    def api_key(self) -> str:
        for name in self.api_key_env:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    @property
    def missing_key_names(self) -> str:
        return "|".join(self.api_key_env)


def _env_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def default_provider_specs() -> dict[str, ProviderSpec]:
    return {
        "gpt": ProviderSpec(
            provider="gpt",
            model=os.environ.get("GPT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
            api_key_env=("OPENAI_API_KEY", "GPT_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("GPT_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("GPT_PRICE_OUTPUT_PER_1M"),
        ),
        "qwen": ProviderSpec(
            provider="qwen",
            model=os.environ.get("QWEN_MODEL", "qwen-plus"),
            api_key_env=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("QWEN_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("QWEN_PRICE_OUTPUT_PER_1M"),
        ),
        "glm": ProviderSpec(
            provider="glm",
            model=os.environ.get("GLM_MODEL", "glm-4"),
            api_key_env=("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
            base_url=os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("GLM_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("GLM_PRICE_OUTPUT_PER_1M"),
        ),
        "gemini": ProviderSpec(
            provider="gemini",
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            kind="gemini",
            price_input_per_1m=_env_float("GEMINI_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("GEMINI_PRICE_OUTPUT_PER_1M"),
        ),
        "minimax": ProviderSpec(
            provider="minimax",
            model=os.environ.get("MINIMAX_MODEL", "MiniMax-M3"),
            api_key_env=("MINIMAX_API_KEY",),
            base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
            kind="minimax",
            json_mode=False,
            price_input_per_1m=_env_float("MINIMAX_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("MINIMAX_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix": ProviderSpec(
            provider="aihubmix",
            model=os.environ.get("AIHUBMIX_MODEL", "gpt-5.5"),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_gpt": ProviderSpec(
            provider="aihubmix_gpt",
            model=os.environ.get("AIHUBMIX_GPT_MODEL", os.environ.get("AIHUBMIX_MODEL_GPT", "gpt-5.5")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_GPT_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_GPT_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_qwen": ProviderSpec(
            provider="aihubmix_qwen",
            model=os.environ.get("AIHUBMIX_QWEN_MODEL", os.environ.get("AIHUBMIX_MODEL_QWEN", "qwen-plus")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_QWEN_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_QWEN_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_glm": ProviderSpec(
            provider="aihubmix_glm",
            model=os.environ.get("AIHUBMIX_GLM_MODEL", os.environ.get("AIHUBMIX_MODEL_GLM", "glm-4")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_GLM_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_GLM_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_gemini": ProviderSpec(
            provider="aihubmix_gemini",
            model=os.environ.get("AIHUBMIX_GEMINI_MODEL", os.environ.get("AIHUBMIX_MODEL_GEMINI", "gemini-3.1-pro-preview")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_GEMINI_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_GEMINI_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_deepseek": ProviderSpec(
            provider="aihubmix_deepseek",
            model=os.environ.get("AIHUBMIX_DEEPSEEK_MODEL", os.environ.get("AIHUBMIX_MODEL_DEEPSEEK", "DeepSeek-V3-Fast")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_DEEPSEEK_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_DEEPSEEK_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_llama": ProviderSpec(
            provider="aihubmix_llama",
            model=os.environ.get("AIHUBMIX_LLAMA_MODEL", os.environ.get("AIHUBMIX_MODEL_LLAMA", "deepseek-r1-distill-llama-70b")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_LLAMA_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_LLAMA_PRICE_OUTPUT_PER_1M"),
        ),
        "aihubmix_mimo": ProviderSpec(
            provider="aihubmix_mimo",
            model=os.environ.get("AIHUBMIX_MIMO_MODEL", os.environ.get("AIHUBMIX_MODEL_MIMO", "xiaomi-mimo-v2.5-free")),
            api_key_env=("AIHUBMIX_API_KEY",),
            base_url=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
            kind="openai_compatible",
            price_input_per_1m=_env_float("AIHUBMIX_MIMO_PRICE_INPUT_PER_1M"),
            price_output_per_1m=_env_float("AIHUBMIX_MIMO_PRICE_OUTPUT_PER_1M"),
        ),
    }


def estimate_cost_usd(spec: ProviderSpec, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None
    if spec.price_input_per_1m is None or spec.price_output_per_1m is None:
        return None
    return (prompt_tokens / 1_000_000.0) * spec.price_input_per_1m + (
        completion_tokens / 1_000_000.0
    ) * spec.price_output_per_1m


class LLMProviderClient:
    def __init__(self, spec: ProviderSpec, timeout: int = 90, max_retries: int = 2):
        self.spec = spec
        self.timeout = timeout
        self.max_retries = max_retries

    def generate_json_report(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> ProviderResponse:
        if not self.spec.api_key:
            raise RuntimeError(f"missing_api_key:{self.spec.missing_key_names}")
        if self.spec.kind == "gemini":
            return self._call_gemini(system_prompt, user_prompt, temperature)
        if self.spec.kind == "minimax":
            return self._call_minimax(system_prompt, user_prompt, temperature)
        return self._call_openai_compatible(system_prompt, user_prompt, temperature, use_json_mode=self.spec.json_mode)

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"http_{exc.code}:{text[:1000]}")
                if exc.code < 500:
                    break
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(str(last_error))

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        use_json_mode: bool,
    ) -> ProviderResponse:
        url = self.spec.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.spec.api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": _env_int("LCAD_API_MAX_TOKENS", 900),
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        start = time.time()
        try:
            raw = self._post_json(url, headers, payload)
        except RuntimeError as exc:
            if use_json_mode and "response_format" in str(exc):
                payload.pop("response_format", None)
                raw = self._post_json(url, headers, payload)
            else:
                raise
        latency = time.time() - start
        choices = raw.get("choices", [])
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""
        usage = raw.get("usage", {}) or {}
        return ProviderResponse(
            provider=self.spec.provider,
            model=self.spec.model,
            text=text,
            latency_seconds=latency,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            raw=raw,
        )

    def _call_gemini(self, system_prompt: str, user_prompt: str, temperature: float) -> ProviderResponse:
        model = urllib.parse.quote(self.spec.model, safe="")
        key = urllib.parse.quote(self.spec.api_key, safe="")
        url = f"{self.spec.base_url.rstrip('/')}/models/{model}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        start = time.time()
        raw = self._post_json(url, headers, payload)
        latency = time.time() - start
        text = ""
        candidates = raw.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join(str(p.get("text", "")) for p in parts if p.get("text"))
        usage = raw.get("usageMetadata", {}) or {}
        prompt_tokens = _int_or_none(usage.get("promptTokenCount"))
        completion_tokens = _int_or_none(usage.get("candidatesTokenCount"))
        total_tokens = _int_or_none(usage.get("totalTokenCount"))
        return ProviderResponse(
            provider=self.spec.provider,
            model=self.spec.model,
            text=text,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=raw,
        )

    def _call_minimax(self, system_prompt: str, user_prompt: str, temperature: float) -> ProviderResponse:
        url = self.spec.base_url.rstrip("/") + "/text/chatcompletion_v2"
        headers = {"Authorization": f"Bearer {self.spec.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "name": "MiniMax AI", "content": system_prompt},
                {"role": "user", "name": "user", "content": user_prompt},
            ],
            "temperature": max(0.01, min(1.0, temperature if temperature > 0 else 0.1)),
            "max_completion_tokens": 1600,
            "mask_sensitive_info": True,
        }
        start = time.time()
        raw = self._post_json(url, headers, payload)
        latency = time.time() - start
        base_resp = raw.get("base_resp", {}) or {}
        status_code = _int_or_none(base_resp.get("status_code"))
        if status_code not in (None, 0):
            status_msg = str(base_resp.get("status_msg", ""))
            raise RuntimeError(f"minimax_base_resp_{status_code}:{status_msg}")
        choices = raw.get("choices", [])
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""
        usage = raw.get("usage", {}) or {}
        return ProviderResponse(
            provider=self.spec.provider,
            model=str(raw.get("model", self.spec.model)),
            text=text,
            latency_seconds=latency,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            raw=raw,
        )


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None
