"""多模态大模型调用模块.

通过环境变量配置:
- VISION_API_BASE: API 基础 URL
- VISION_API_KEY: API 密钥
- VISION_MODEL: 模型名称
"""

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx
import structlog
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

from foshan_scraper.models import LlmRecognitionResult

log = structlog.get_logger()

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"

# 模型接口超时：连接30s，读取300s（大模型响应慢），写入60s，连接池30s
MODEL_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=300.0,
    write=60.0,
    pool=30.0,
)

# 触发重试的异常类型
_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)

# 触发重试的 HTTP 状态码
_RETRY_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}


class VisionSettings(BaseSettings):
    """多模态模型 API 配置（从环境变量/.env 读取）。"""

    vision_api_base: str = Field(default=DEFAULT_API_BASE, alias="VISION_API_BASE")
    vision_api_key: str = Field(default="", alias="VISION_API_KEY")
    vision_model: str = Field(default=DEFAULT_MODEL, alias="VISION_MODEL")

    model_config = {"env_prefix": "", "extra": "ignore"}


class _SettingsCache:
    """延迟初始化的配置单例（避免 global 语句）。"""

    _instance: VisionSettings | None = None

    @classmethod
    def get(cls) -> VisionSettings:
        """获取配置实例（延迟初始化）。"""
        if cls._instance is None:
            cls._instance = VisionSettings()
        return cls._instance


def _get_settings() -> VisionSettings:
    """获取配置实例（延迟初始化）。"""
    return _SettingsCache.get()


def get_config() -> dict[str, str]:
    """读取模型配置（不含密钥内容）。"""
    s = _get_settings()
    # 不暴露 API key 内容，仅在需要时从 settings 获取
    return {
        "api_base": s.vision_api_base,
        "api_key": s.vision_api_key,
        "model": s.vision_model,
    }


def load_prompt_template(prompt_path: Path) -> str:
    """加载提示词模板文件."""
    return prompt_path.read_text(encoding="utf-8")


def _parse_llm_json(raw: str) -> LlmRecognitionResult | None:
    """从模型原始输出中提取 JSON 并用 Pydantic 校验.

    Args:
        raw: 模型原始输出字符串

    Returns:
        LlmRecognitionResult 或 None（解析失败）
    """
    # 去掉可能的 markdown 代码块标记
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 提取 JSON 对象（可能有前后缀文字）
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        log.warning("no_json_found_in_response", raw=raw[:200])
        return None

    json_str = json_match.group(0)
    try:
        data = json.loads(json_str)
        return LlmRecognitionResult(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning(
            "json_validation_failed",
            error_type=type(e).__name__,
            error_repr=repr(e),
            json_str=json_str[:200],
        )
        return None


async def call_vision_model(
    image_path: Path,
    prompt: str,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> tuple[LlmRecognitionResult | None, str]:
    """调用多模态模型识别截图.

    Args:
        image_path: 截图文件路径
        prompt: 提示词
        model: 模型名称（默认从环境变量）
        api_base: API 基础 URL
        api_key: API 密钥

    Returns:
        (LlmRecognitionResult, raw_response) 元组
    """
    s = _get_settings()
    actual_model = model or s.vision_model
    actual_base = api_base or s.vision_api_base
    actual_key = api_key or s.vision_api_key

    if not actual_key:
        raise ValueError("VISION_API_KEY 环境变量未设置")

    # 读取图片为 base64
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {actual_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": actual_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 2048,
    }

    # kimi-k2.6 专用配置：关闭 thinking、强制 JSON 模式
    if actual_model == "kimi-k2.6":
        payload["thinking"] = {"type": "disabled"}
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
        resp = await client.post(
            f"{actual_base.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        result = resp.json()

    content = result["choices"][0]["message"]["content"]
    parsed = _parse_llm_json(content)

    return parsed, content


async def call_vision_model_with_retry(
    image_path: Path,
    prompt: str,
    max_retries: int = 3,
) -> tuple[LlmRecognitionResult | None, str]:
    """带重试的模型调用.

    Args:
        image_path: 截图文件路径
        prompt: 提示词
        max_retries: 最大调用次数（含首次），默认 3

    Returns:
        (LlmRecognitionResult, raw_response)
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            parsed, raw = await call_vision_model(image_path, prompt)
            if parsed is not None:
                return parsed, raw
            # 解析失败且还有重试次数
            if attempt < max_retries:
                log.info("retrying_parse_failure", attempt=attempt, max_retries=max_retries)
                await asyncio.sleep(3.0 * attempt)
        except httpx.HTTPStatusError as e:
            last_error = e
            status = e.response.status_code
            log.warning(
                "api_http_error",
                attempt=attempt,
                status=status,
                error_type=type(e).__name__,
                error_repr=repr(e),
            )
            if status in _RETRY_STATUS_CODES and attempt < max_retries:
                wait = 3.0 * attempt
                log.info("retrying_after_wait", attempt=attempt, wait_seconds=wait, reason=f"status_{status}")
                await asyncio.sleep(wait)
                continue
            # 非重试状态码直接抛出；已用尽重试次数也抛出
            raise
        except _RETRY_EXCEPTIONS as e:
            last_error = e
            log.warning(
                "api_network_error",
                attempt=attempt,
                error_type=type(e).__name__,
                error_repr=repr(e),
            )
            if attempt < max_retries:
                wait = 3.0 * attempt
                log.info("retrying_after_wait", attempt=attempt, wait_seconds=wait, reason=type(e).__name__)
                await asyncio.sleep(wait)
                continue
            raise
        except Exception:
            # 其他异常不重试，直接抛出
            raise

    # 所有尝试均失败（解析失败耗尽重试次数）
    if last_error is not None:
        raise last_error
    return None, ""


def read_raw_json(raw_path: Path) -> LlmRecognitionResult | None:
    """从 raw_outputs JSON 文件中读取并解析模型结果.

    兼容两种格式：
    1. 直接格式：{"transaction_date": "...", ...}
    2. 包装格式：{"index": 1, "raw_response": "{...}"}

    Args:
        raw_path: raw_outputs/*.json 文件路径

    Returns:
        LlmRecognitionResult 或 None
    """
    try:
        with raw_path.open(encoding="utf-8") as f:
            data = json.load(f)

        # 兼容包装格式 {"index": N, "raw_response": "..."}
        if isinstance(data, dict) and "raw_response" in data:
            raw_str = data["raw_response"]
            # raw_response 本身是 JSON 字符串，需要再解析一层
            data = json.loads(raw_str)

        return LlmRecognitionResult(**data)
    except Exception as e:
        log.warning(
            "raw_json_read_failed",
            path=str(raw_path),
            error_type=type(e).__name__,
            error_repr=repr(e),
        )
        return None
