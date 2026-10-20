"""使用 Playwright 截图采集模块.

通过 Playwright 打开详情页，穿越两层 iframe，
只使用 frame_locator 定位功能，不使用 CSS Selector 读取字段。
"""

import asyncio
import os
import re
import shutil
from pathlib import Path

import structlog
from playwright.async_api import Page, async_playwright

log = structlog.get_logger()


class BrowserChannelError(Exception):
    """指定的浏览器渠道不可用时抛出."""

    pass


SELECTOR_OUTER_IFRAME = 'iframe[src*="TDJY_BasicInfoFrame.aspx"]'
SELECTOR_INNER_IFRAME = 'iframe[src*="TDJY_BasicInfoHistory.aspx"]'
PAGE_LOAD_TIMEOUT = 60_000
CLICK_DELAY = 2


def _extract_field_text(soup_text: str, field_name: str) -> str | None:
    """从 HTML 文本中通过正则提取指定字段（仅用于截图中可见内容）.

    Args:
        soup_text: BeautifulSoup.get_text() 后的纯文本
        field_name: 字段名（如"成交时间"、"竞得人"等）

    Returns:
        字段值或 None
    """
    # 模式：字段名 + 分隔符 + 值（跨多行匹配）
    pattern = re.escape(field_name) + r"[:：]\s*(.+?)(?=\n\s*[^\n：:]{0,10}[:：]|$)"
    match = re.search(pattern, soup_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def _capture_detail_screenshot(
    page: Page,
    detail_url: str,
    output_path: Path,
) -> bool:
    """打开详情页，穿过两层 iframe，截图并保存.

    Args:
        page: Playwright Page 对象
        detail_url: 详情页 URL
        output_path: 截图保存路径

    Returns:
        是否成功截图
    """
    try:
        new_page = await page.context.new_page()
        try:
            log.info("navigating_to_detail", url=detail_url)
            await new_page.goto(detail_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
            await new_page.wait_for_load_state("networkidle", timeout=PAGE_LOAD_TIMEOUT)
            await asyncio.sleep(CLICK_DELAY)

            # 定位第一层 iframe
            outer_frame = new_page.frame_locator(SELECTOR_OUTER_IFRAME)
            await outer_frame.locator("body").wait_for(timeout=PAGE_LOAD_TIMEOUT)
            log.info("outer_iframe_loaded")

            # 定位第二层 iframe
            inner_frame = outer_frame.frame_locator(SELECTOR_INNER_IFRAME)
            await inner_frame.locator("body").wait_for(timeout=PAGE_LOAD_TIMEOUT)
            log.info("inner_iframe_loaded")

            # 在第二层 iframe 中截图
            inner_body = inner_frame.locator("body")
            await inner_body.screenshot(path=str(output_path))
            log.info("screenshot_saved", path=str(output_path))
            return True

        finally:
            await new_page.close()

    except Exception as e:
        log.warning("screenshot_failed", url=detail_url, error=str(e))
        return False


async def capture_all_screenshots(
    detail_urls: list[str],
    output_dir: Path,
    headless: bool = True,
    delay: float = 1.0,
    browser_channel: str = "chrome",
) -> list[tuple[str, Path | None]]:
    """批量采集详情页截图.

    Args:
        detail_urls: 详情页 URL 列表
        output_dir: 截图输出目录
        headless: 是否无头模式
        delay: 相邻页面间隔（秒）
        browser_channel: 浏览器渠道，chrome 或 msedge

    Returns:
        [(url, screenshot_path), ...] 列表，失败时 path 为 None

    Raises:
        BrowserChannelError: 指定的浏览器渠道不可用
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查浏览器可执行文件是否存在
    if browser_channel == "chrome":
        chrome_paths = [
            Path(shutil.which("chrome") or ""),
            Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
            Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
            Path(os.environ.get("LOCALAPPDATA", "") + "/Google/Chrome/Application/chrome.exe"),
        ]
        if not any(p.exists() for p in chrome_paths if p):
            raise BrowserChannelError("未找到系统 Chrome 浏览器，请改用 --browser-channel msedge")
    # msedge 在 Windows 上通常已安装，无需预检

    results: list[tuple[str, Path | None]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel=browser_channel, headless=headless)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        try:
            for i, url in enumerate(detail_urls):
                log.info("capturing_screenshot", index=i + 1, total=len(detail_urls), url=url)
                screenshot_path = output_dir / f"screenshot_{i + 1:02d}.png"
                success = await _capture_detail_screenshot(page, url, screenshot_path)

                if success:
                    results.append((url, screenshot_path))
                else:
                    results.append((url, None))

                # 相邻页面间隔
                if i < len(detail_urls) - 1 and delay > 0:
                    await asyncio.sleep(delay)

        finally:
            await browser.close()

    return results
