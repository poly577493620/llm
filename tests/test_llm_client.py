"""测试多模态模型调用模块（不访问真实网络）."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from foshan_scraper.llm_client import (
    _parse_llm_json,
    call_vision_model,
    call_vision_model_with_retry,
    read_raw_json,
)
from foshan_scraper.models import LlmRecognitionResult


def _make_mock_settings(model: str = "kimi-k2.6") -> MagicMock:
    """创建带假密钥的 mock VisionSettings."""
    m = MagicMock()
    m.vision_api_key = "fake-key"
    m.vision_api_base = "http://test"
    m.vision_model = model
    return m


class TestParseLlMJson:
    """_parse_llm_json 测试（纯函数，不访问网络）."""

    def test_strips_markdown_json_block(self) -> None:
        """去掉 ```json 包裹."""
        raw = '```json\n{"transaction_date": "2026/7/27", "winner": "测试公司"}\n```'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27"
        assert result.winner == "测试公司"

    def test_strips_generic_code_block(self) -> None:
        """去掉通用 ``` 包裹."""
        raw = '```\n{"transaction_date": "2026/7/27"}\n```'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27"

    def test_handles_prefix_text(self) -> None:
        """模型输出包含前缀文字时仍能提取 JSON."""
        raw = '以下是识别结果：\n{"transaction_date": "2026/7/27"}\n如有疑问请联系管理员'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27"

    def test_handles_suffix_text(self) -> None:
        """模型输出包含后缀文字时仍能提取 JSON."""
        raw = '{"winner": "测试公司"}\n\n以上是识别结果。'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.winner == "测试公司"

    def test_invalid_json_returns_none(self) -> None:
        """无效 JSON 返回 None."""
        assert _parse_llm_json("not json at all") is None
        assert _parse_llm_json("") is None
        assert _parse_llm_json("{incomplete:") is None

    def test_partial_invalid_json(self) -> None:
        """部分无效 JSON（但包含有效对象）时提取有效部分."""
        raw = '{"transaction_date": "2026/7/27"} extra stuff'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27"

    def test_all_fields(self) -> None:
        """完整字段解析."""
        raw = json.dumps(
            {
                "transaction_date": "2026/7/27 15:00:02",
                "winner": "测试公司",
                "transaction_area": "4840.75 平方米",
                "transaction_price": "地价:15746万元",
                "land_location": "佛山市禅城区",
                "land_usage": "居住用地",
                "uncertain_fields": ["winner"],
                "notes": "测试备注",
            }
        )
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27 15:00:02"
        assert result.winner == "测试公司"
        assert result.transaction_area == "4840.75 平方米"
        assert result.uncertain_fields == ["winner"]
        assert result.notes == "测试备注"

    def test_missing_optional_fields(self) -> None:
        """可选字段缺失时使用默认值."""
        raw = '{"transaction_date": "2026/7/27"}'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27"
        assert result.winner is None
        assert result.uncertain_fields == []

    def test_extra_fields_allowed(self) -> None:
        """允许额外字段（Pydantic extra=allow）."""
        raw = '{"transaction_date": "2026/7/27", "extra_field": "should_be_ignored"}'
        result = _parse_llm_json(raw)
        assert result is not None
        assert result.transaction_date == "2026/7/27"


class TestReadRawJson:
    """read_raw_json 测试（不访问网络）."""

    def test_wrapped_format(self, tmp_path: Path) -> None:
        """包装格式 {index, raw_response} 能正确解析."""
        raw_path = tmp_path / "raw_01.json"
        wrapped = {
            "index": 1,
            "raw_response": json.dumps(
                {
                    "transaction_date": "2026/7/27",
                    "winner": "测试公司",
                    "transaction_area": "1000",
                    "transaction_price": "地价:1000万元",
                    "land_location": "佛山",
                    "land_usage": "居住用地",
                }
            ),
        }
        raw_path.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")

        result = read_raw_json(raw_path)
        assert result is not None
        assert result.transaction_date == "2026/7/27"
        assert result.winner == "测试公司"

    def test_direct_format(self, tmp_path: Path) -> None:
        """直接 JSON 格式仍然能解析."""
        raw_path = tmp_path / "raw_02.json"
        direct = {
            "transaction_date": "2026/7/28",
            "winner": "另一公司",
            "transaction_area": "2000",
            "transaction_price": "地价:2000万元",
            "land_location": "广州",
            "land_usage": "商业用地",
        }
        raw_path.write_text(json.dumps(direct, ensure_ascii=False), encoding="utf-8")

        result = read_raw_json(raw_path)
        assert result is not None
        assert result.transaction_date == "2026/7/28"
        assert result.winner == "另一公司"

    def test_invalid_returns_none(self, tmp_path: Path) -> None:
        """无效文件返回 None."""
        raw_path = tmp_path / "raw_bad.json"
        raw_path.write_text("not json", encoding="utf-8")

        result = read_raw_json(raw_path)
        assert result is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """不存在的文件返回 None（不抛异常）."""
        raw_path = tmp_path / "raw_missing.json"
        result = read_raw_json(raw_path)
        assert result is None


class TestCallVisionModelWithRetry:
    """call_vision_model_with_retry 重试逻辑测试（全部 mock，不访问网络）."""

    @pytest.fixture
    def mock_image(self, tmp_path: Path) -> Path:
        """创建假图片文件供调用路径参数使用."""
        img = tmp_path / "fake.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        return img

    @pytest.mark.anyio
    async def test_success_no_retry(self, mock_image: Path) -> None:
        """首次成功时不重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        with patch(
            "foshan_scraper.llm_client.call_vision_model",
            new_callable=AsyncMock,
            return_value=(parsed, "raw"),
        ) as mock_call:
            result, raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert raw == "raw"
            assert mock_call.call_count == 1

    @pytest.mark.anyio
    async def test_read_timeout_retries_3_times(self, mock_image: Path) -> None:
        """ReadTimeout 最多调用 3 次，等待 3s 和 6s."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        side_effects = [
            httpx.ReadTimeout("read timeout"),
            httpx.ReadTimeout("read timeout"),
            (parsed, "raw"),
        ]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result, raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert raw == "raw"
            assert mock_call.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(3.0)
            mock_sleep.assert_any_call(6.0)

    @pytest.mark.anyio
    async def test_connect_error_retries(self, mock_image: Path) -> None:
        """ConnectError 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        side_effects = [
            httpx.ConnectError("connect failed"),
            (parsed, "raw"),
        ]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as _mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2
            assert _mock_sleep.call_count == 1
            _mock_sleep.assert_called_with(3.0)

    @pytest.mark.anyio
    async def test_429_retries(self, mock_image: Path) -> None:
        """HTTP 429 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        err_429 = httpx.HTTPStatusError(
            "Too Many Requests",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(429),
        )
        side_effects = [err_429, (parsed, "raw")]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2
            assert mock_sleep.call_count == 1
            mock_sleep.assert_called_with(3.0)

    @pytest.mark.anyio
    async def test_500_retries(self, mock_image: Path) -> None:
        """HTTP 500 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        err_500 = httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(500),
        )
        side_effects = [err_500, (parsed, "raw")]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2
            assert mock_sleep.call_count == 1

    @pytest.mark.anyio
    async def test_400_no_retry(self, mock_image: Path) -> None:
        """HTTP 400 不重试，直接抛出."""
        err_400 = httpx.HTTPStatusError(
            "Bad Request",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(400),
        )

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=err_400,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await call_vision_model_with_retry(mock_image, "prompt")
            assert mock_call.call_count == 1
            assert mock_sleep.call_count == 0

    @pytest.mark.anyio
    async def test_exhausted_retries_raises(self, mock_image: Path) -> None:
        """3 次都失败后抛出最后一次异常."""
        side_effects = [
            httpx.ReadTimeout("timeout 1"),
            httpx.ReadTimeout("timeout 2"),
            httpx.ReadTimeout("timeout 3"),
        ]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(httpx.ReadTimeout):
                await call_vision_model_with_retry(mock_image, "prompt")
            assert mock_call.call_count == 3
            assert mock_sleep.call_count == 2

    @pytest.mark.anyio
    async def test_parse_failure_retries(self, mock_image: Path) -> None:
        """解析失败（返回 None）也会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        side_effects = [
            (None, "invalid json"),
            (parsed, "valid json"),
        ]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result, raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert raw == "valid json"
            assert mock_call.call_count == 2
            assert mock_sleep.call_count == 1
            mock_sleep.assert_called_with(3.0)

    @pytest.mark.anyio
    async def test_remote_protocol_error_retries(self, mock_image: Path) -> None:
        """RemoteProtocolError 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        side_effects = [
            httpx.RemoteProtocolError("protocol error"),
            (parsed, "raw"),
        ]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as _mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2
            assert _mock_sleep.call_count == 1

    @pytest.mark.anyio
    async def test_connect_timeout_retries(self, mock_image: Path) -> None:
        """ConnectTimeout 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        side_effects = [
            httpx.ConnectTimeout("connect timeout"),
            (parsed, "raw"),
        ]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as _mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2
            assert _mock_sleep.call_count == 1

    @pytest.mark.anyio
    async def test_502_retries(self, mock_image: Path) -> None:
        """HTTP 502 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        err_502 = httpx.HTTPStatusError(
            "Bad Gateway",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(502),
        )
        side_effects = [err_502, (parsed, "raw")]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as _mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2

    @pytest.mark.anyio
    async def test_503_retries(self, mock_image: Path) -> None:
        """HTTP 503 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        err_503 = httpx.HTTPStatusError(
            "Service Unavailable",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(503),
        )
        side_effects = [err_503, (parsed, "raw")]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as _mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2

    @pytest.mark.anyio
    async def test_504_retries(self, mock_image: Path) -> None:
        """HTTP 504 会重试."""
        parsed = LlmRecognitionResult(transaction_date="2026/7/27")
        err_504 = httpx.HTTPStatusError(
            "Gateway Timeout",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(504),
        )
        side_effects = [err_504, (parsed, "raw")]

        with (
            patch(
                "foshan_scraper.llm_client.call_vision_model",
                new_callable=AsyncMock,
                side_effect=side_effects,
            ) as mock_call,
            patch("foshan_scraper.llm_client.asyncio.sleep", new_callable=AsyncMock) as _mock_sleep,
        ):
            result, _raw = await call_vision_model_with_retry(mock_image, "prompt")
            assert result == parsed
            assert mock_call.call_count == 2


class TestCallVisionModelPayload:
    """call_vision_model 请求 payload 结构测试（mock HTTP，不访问网络）."""

    @pytest.fixture
    def mock_image(self, tmp_path: Path) -> Path:
        """创建假图片文件."""
        img = tmp_path / "fake.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        return img

    def _make_mock_response(self, content: str) -> httpx.Response:
        """构建 fake HTTP 200 响应."""
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
            },
            request=httpx.Request("POST", "http://test"),
        )

    @pytest.mark.anyio
    async def test_kimi_k26_payload_has_thinking_disabled(
        self,
        mock_image: Path,
    ) -> None:
        """kimi-k2.6 请求 payload 包含 thinking.disabled."""
        fake_resp = self._make_mock_response('{"transaction_date": "2026/7/27", "winner": "测试公司"}')
        sent_payload: dict[str, Any] | None = None

        async def _capture_post(
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            nonlocal sent_payload
            sent_payload = kwargs.get("json")
            return fake_resp

        mock_client = AsyncMock()
        mock_client.post = _capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("foshan_scraper.llm_client._get_settings", return_value=_make_mock_settings()),
        ):
            result, _raw = await call_vision_model(
                mock_image,
                "prompt",
                model="kimi-k2.6",
            )

        assert result is not None
        assert sent_payload is not None
        assert sent_payload.get("thinking") == {"type": "disabled"}

    @pytest.mark.anyio
    async def test_kimi_k26_payload_has_json_object_response_format(
        self,
        mock_image: Path,
    ) -> None:
        """kimi-k2.6 请求 payload 包含 response_format.type = json_object."""
        fake_resp = self._make_mock_response('{"transaction_date": "2026/7/27", "winner": "测试公司"}')
        sent_payload: dict[str, Any] | None = None

        async def _capture_post(
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            nonlocal sent_payload
            sent_payload = kwargs.get("json")
            return fake_resp

        mock_client = AsyncMock()
        mock_client.post = _capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("foshan_scraper.llm_client._get_settings", return_value=_make_mock_settings()),
        ):
            result, _raw = await call_vision_model(
                mock_image,
                "prompt",
                model="kimi-k2.6",
            )

        assert result is not None
        assert sent_payload is not None
        assert sent_payload.get("response_format") == {"type": "json_object"}

    @pytest.mark.anyio
    async def test_kimi_k26_payload_no_temperature(
        self,
        mock_image: Path,
    ) -> None:
        """kimi-k2.6 请求 payload 不包含 temperature 字段."""
        fake_resp = self._make_mock_response('{"transaction_date": "2026/7/27", "winner": "测试公司"}')
        sent_payload: dict[str, Any] | None = None

        async def _capture_post(
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            nonlocal sent_payload
            sent_payload = kwargs.get("json")
            return fake_resp

        mock_client = AsyncMock()
        mock_client.post = _capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("foshan_scraper.llm_client._get_settings", return_value=_make_mock_settings()),
        ):
            result, _raw = await call_vision_model(
                mock_image,
                "prompt",
                model="kimi-k2.6",
            )

        assert result is not None
        assert sent_payload is not None
        assert "temperature" not in sent_payload

    @pytest.mark.anyio
    async def test_gpt4o_payload_no_kimi_fields(
        self,
        mock_image: Path,
    ) -> None:
        """非 kimi-k2.6 模型不添加 thinking 和 response_format."""
        fake_resp = self._make_mock_response('{"transaction_date": "2026/7/27", "winner": "测试公司"}')
        sent_payload: dict[str, Any] | None = None

        async def _capture_post(
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            nonlocal sent_payload
            sent_payload = kwargs.get("json")
            return fake_resp

        mock_client = AsyncMock()
        mock_client.post = _capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("foshan_scraper.llm_client._get_settings", return_value=_make_mock_settings()),
        ):
            result, _raw = await call_vision_model(
                mock_image,
                "prompt",
                model="gpt-4o",
            )

        assert result is not None
        assert sent_payload is not None
        assert "thinking" not in sent_payload
        assert "response_format" not in sent_payload
        assert "temperature" not in sent_payload

    @pytest.mark.anyio
    async def test_payload_has_correct_field_names_in_prompt(
        self,
        mock_image: Path,
    ) -> None:
        """提示词中包含正确的英文字段名."""
        fake_resp = self._make_mock_response('{"transaction_date": "2026/7/27", "winner": "测试公司"}')
        sent_payload: dict[str, Any] | None = None

        async def _capture_post(
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            nonlocal sent_payload
            sent_payload = kwargs.get("json")
            return fake_resp

        mock_client = AsyncMock()
        mock_client.post = _capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        prompt = (
            "提取字段：transaction_date, winner, transaction_area, "
            "transaction_price, land_location, land_usage, "
            "uncertain_fields, notes"
        )

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("foshan_scraper.llm_client._get_settings", return_value=_make_mock_settings()),
        ):
            result, _raw = await call_vision_model(
                mock_image,
                prompt,
                model="kimi-k2.6",
            )

        assert result is not None
        assert sent_payload is not None
        messages = sent_payload.get("messages", [])
        assert len(messages) == 1
        content = messages[0]["content"]
        text_parts = [c["text"] for c in content if c.get("type") == "text"]
        assert len(text_parts) == 1
        assert "transaction_date" in text_parts[0]
        assert "uncertain_fields" in text_parts[0]
        assert "land_usage" in text_parts[0]
