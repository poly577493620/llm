"""测试 CLI 参数解析（不访问网络和真实网站）."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from foshan_scraper.__main__ import DEFAULT_SOURCE_JSON, main


class TestDefaultSourceJsonPath:
    """DEFAULT_SOURCE_JSON 路径测试."""

    def test_default_source_json_points_to_project_data(self) -> None:
        """DEFAULT_SOURCE_JSON 应指向项目内部的 data/reference_data.json."""
        assert DEFAULT_SOURCE_JSON.name == "reference_data.json"
        assert DEFAULT_SOURCE_JSON.parent.name == "data"

    def test_default_source_json_uses_parents2(self) -> None:
        """DEFAULT_SOURCE_JSON 应相对于 src/foshan_scraper/__main__.py 的 parents[2]."""
        # __main__.py 在 03_llm_screenshot/src/foshan_scraper/ 下
        # parents[0]=foshan_scraper, parents[1]=src, parents[2]=03_llm_screenshot
        # 因此 DEFAULT_SOURCE_JSON = 03_llm_screenshot/data/reference_data.json
        # 从测试文件看：parents[1]=03_llm_screenshot
        expected = Path(__file__).parents[1] / "data" / "reference_data.json"
        assert DEFAULT_SOURCE_JSON == expected


def _make_mock_json(tmp_path: Path) -> Path:
    """生成含 status 字段的 mock JSON 文件."""
    mock = [
        {
            "row_guid": f"guid-{i}",
            "transaction_date": "2026/7/27",
            "winner": "测试公司",
            "transaction_area": "1000",
            "detail_url": f"http://example.com/{i}",
            "status": "已成交" if i % 10 == 0 else "不成交",
        }
        for i in range(20)
    ]
    p = tmp_path / "mock.json"
    p.write_text(json.dumps(mock, ensure_ascii=False), encoding="utf-8")
    return p


def _make_wrapped_raw(raw_dir: Path, index: int, winner: str = "测试公司") -> None:
    """在 raw_dir 下创建包装格式的 raw_XX.json."""
    raw_data = {
        "index": index,
        "raw_response": json.dumps(
            {
                "transaction_date": "2026/7/27",
                "winner": winner,
                "transaction_area": "1000",
                "transaction_price": "地价:1000万元",
                "land_location": "佛山市禅城区",
                "land_usage": "居住用地",
            },
            ensure_ascii=False,
        ),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"raw_{index:02d}.json").write_text(json.dumps(raw_data, ensure_ascii=False), encoding="utf-8")


class TestCaptureOnly:
    """--capture-only 参数测试."""

    def test_capture_only_skips_model_and_export(self, tmp_path: Path) -> None:
        """--capture-only 在截图完成后退出，不调用模型也不导出 Excel."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        result = runner.invoke(
            main,
            [
                "--source-json",
                str(mock_json),
                "--capture-only",
                "--browser",
                "--no-browser",
            ],
        )
        assert result.exit_code == 0
        # capture-only 跳过步骤的提示
        assert "capture" in result.output.lower() or "screenshot" in result.output.lower()

    def test_capture_only_without_browser_exits_cleanly(self, tmp_path: Path) -> None:
        """--capture-only 配合 --no-browser 时正常退出."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        result = runner.invoke(
            main,
            [
                "--source-json",
                str(mock_json),
                "--capture-only",
                "--no-browser",
            ],
        )
        assert result.exit_code == 0


class TestOffline:
    """--offline 参数测试."""

    def test_offline_requires_existing_source_json(self, tmp_path: Path) -> None:
        """--offline 模式下 source-json 文件不存在时报错."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--offline", "--source-json", "/nonexistent/path.json"],
        )
        assert result.exit_code != 0


class TestBrowserChannel:
    """--browser-channel 参数测试."""

    def test_browser_channel_accepts_chrome(self, tmp_path: Path) -> None:
        """--browser-channel chrome 应接受."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        result = runner.invoke(
            main,
            [
                "--source-json",
                str(mock_json),
                "--browser-channel",
                "chrome",
                "--no-browser",
            ],
        )
        assert result.exit_code == 0

    def test_browser_channel_accepts_msedge(self, tmp_path: Path) -> None:
        """--browser-channel msedge 应接受."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        result = runner.invoke(
            main,
            [
                "--source-json",
                str(mock_json),
                "--browser-channel",
                "msedge",
                "--no-browser",
            ],
        )
        assert result.exit_code == 0

    def test_browser_channel_rejects_invalid(self, tmp_path: Path) -> None:
        """--browser-channel 不接受非 chrome/msedge 值."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        result = runner.invoke(
            main,
            [
                "--source-json",
                str(mock_json),
                "--browser-channel",
                "firefox",
                "--no-browser",
            ],
        )
        assert result.exit_code != 0

    def test_browser_channel_default_is_chrome(self, tmp_path: Path) -> None:
        """不指定 --browser-channel 时默认值为 chrome."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        # 只传必需参数，观察默认行为
        result = runner.invoke(
            main,
            [
                "--source-json",
                str(mock_json),
                "--no-browser",
            ],
        )
        # --no-browser + 无 source-json 错误即为正常解析（未因 channel 报错）
        assert "Invalid value for '--browser-channel'" not in result.output


class TestResume:
    """--resume/--no-resume 参数测试."""

    def test_resume_reuses_valid_raw_file(self, tmp_path: Path) -> None:
        """resume 遇到有效 raw 文件时不调用真实模型."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        raw_dir = tmp_path / "raw_outputs"
        _make_wrapped_raw(raw_dir, 1)

        with patch(
            "foshan_scraper.__main__.call_vision_model_with_retry",
            new_callable=AsyncMock,
        ) as mock_call:
            result = runner.invoke(
                main,
                [
                    "--source-json",
                    str(mock_json),
                    "--no-browser",
                    "--resume",
                    "--raw-dir",
                    str(raw_dir),
                    "--limit",
                    "1",
                ],
            )
            assert result.exit_code == 0
            assert mock_call.call_count == 0
            assert "复用" in result.output

    def test_resume_skips_invalid_raw_file_and_calls_model(self, tmp_path: Path) -> None:
        """resume 遇到无效 raw 文件时继续调用模型."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        raw_dir = tmp_path / "raw_outputs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # 写入无效 JSON
        (raw_dir / "raw_01.json").write_text("not valid json", encoding="utf-8")

        # 创建假截图文件，让流程走到模型调用
        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        (screenshots_dir / "screenshot_01.png").write_bytes(b"\x89PNG")

        with patch(
            "foshan_scraper.__main__.call_vision_model_with_retry",
            new_callable=AsyncMock,
            return_value=(None, ""),
        ) as mock_call:
            result = runner.invoke(
                main,
                [
                    "--source-json",
                    str(mock_json),
                    "--no-browser",
                    "--resume",
                    "--raw-dir",
                    str(raw_dir),
                    "--screenshots-dir",
                    str(screenshots_dir),
                    "--limit",
                    "1",
                ],
            )
            assert result.exit_code == 0
            assert mock_call.call_count == 1
            assert "解析失败" in result.output

    def test_no_resume_calls_model_even_with_existing_raw(self, tmp_path: Path) -> None:
        """--no-resume 时即使有 raw 文件也重新调用模型."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        raw_dir = tmp_path / "raw_outputs"
        _make_wrapped_raw(raw_dir, 1)

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        (screenshots_dir / "screenshot_01.png").write_bytes(b"\x89PNG")

        with patch(
            "foshan_scraper.__main__.call_vision_model_with_retry",
            new_callable=AsyncMock,
            return_value=(None, ""),
        ) as mock_call:
            result = runner.invoke(
                main,
                [
                    "--source-json",
                    str(mock_json),
                    "--no-browser",
                    "--no-resume",
                    "--raw-dir",
                    str(raw_dir),
                    "--screenshots-dir",
                    str(screenshots_dir),
                    "--limit",
                    "1",
                ],
            )
            assert result.exit_code == 0
            assert mock_call.call_count == 1

    def test_resume_default_enabled(self, tmp_path: Path) -> None:
        """默认启用 resume（不传参时生效）."""
        runner = CliRunner()
        mock_json = _make_mock_json(tmp_path)
        raw_dir = tmp_path / "raw_outputs"
        _make_wrapped_raw(raw_dir, 1)

        with patch(
            "foshan_scraper.__main__.call_vision_model_with_retry",
            new_callable=AsyncMock,
        ) as mock_call:
            result = runner.invoke(
                main,
                [
                    "--source-json",
                    str(mock_json),
                    "--no-browser",
                    "--raw-dir",
                    str(raw_dir),
                    "--limit",
                    "1",
                ],
            )
            assert result.exit_code == 0
            assert mock_call.call_count == 0
            assert "复用" in result.output
