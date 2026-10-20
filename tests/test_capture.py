"""测试截图采集模块（不访问网络和浏览器）。"""

from foshan_scraper.capture import BrowserChannelError


class TestBrowserChannelError:
    """BrowserChannelError 异常测试。"""

    def test_exception_message(self) -> None:
        """异常应携带清晰的错误信息。"""
        msg = "未找到系统 Chrome 浏览器，请改用 --browser-channel msedge"
        exc = BrowserChannelError(msg)
        assert str(exc) == msg
        assert "msedge" in str(exc)
