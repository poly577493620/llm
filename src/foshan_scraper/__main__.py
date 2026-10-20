"""大模型截图识别实验 CLI 入口."""

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog

from foshan_scraper.capture import BrowserChannelError, capture_all_screenshots
from foshan_scraper.compare import compare_result
from foshan_scraper.llm_client import (
    call_vision_model_with_retry,
    get_config,
    load_prompt_template,
    read_raw_json,
)
from foshan_scraper.models import LlmMatchResult, LlmRecognitionResult
from foshan_scraper.output import export_to_excel, save_raw_output
from foshan_scraper.reader import load_reference_data, select_first_per_page

log = structlog.get_logger()

DEFAULT_PAGES = 10
DEFAULT_SOURCE_JSON = Path(__file__).parents[2] / "data" / "reference_data.json"
PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "字段识别提示词.md"


def _read_raw_response_text(raw_path: Path) -> str:
    """从 raw 文件中提取 raw_response 文本（兼容包装格式）.

    Args:
        raw_path: raw_outputs/*.json 文件路径

    Returns:
        raw_response 字符串，失败返回空字符串
    """
    try:
        text = raw_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict) and "raw_response" in data:
            return str(data["raw_response"])
        # 直接格式：把整个 JSON 作为 raw_response
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return ""


@click.command()
@click.option(
    "--limit",
    "-n",
    default=0,
    type=int,
    help="限制处理记录数（0=全部，默认0，用于演示时可设为2）",
    show_default=True,
)
@click.option(
    "--source-json",
    "-s",
    default=str(DEFAULT_SOURCE_JSON),
    type=click.Path(path_type=Path, exists=True),
    help="任务二 JSON 文件路径",
    show_default=True,
)
@click.option(
    "--prompt-file",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    help="提示词模板文件路径",
)
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    type=click.Path(path_type=Path),
    help="输出目录",
    show_default=True,
)
@click.option(
    "--screenshots-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="截图保存目录（默认 output/screenshots）",
)
@click.option(
    "--raw-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="原始响应保存目录（默认 output/raw_outputs）",
)
@click.option(
    "--headless/--no-headless",
    default=True,
    help="Playwright 无头模式",
    show_default=True,
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="离线模式：跳过截图和模型调用，从 raw_outputs 读取已有结果",
)
@click.option(
    "--browser/--no-browser",
    default=True,
    help="是否启动浏览器（--no-browser 跳过截图采集）",
)
@click.option(
    "--browser-channel",
    default="chrome",
    type=click.Choice(["chrome", "msedge"], case_sensitive=False),
    help="浏览器渠道：chrome（默认）或 msedge",
    show_default=True,
)
@click.option(
    "--capture-only",
    is_flag=True,
    default=False,
    help="仅完成截图采集，不调用模型也不导出 Excel（用于分阶段测试和录屏）",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="启用断点续传：跳过已存在的 raw_XX.json（默认启用）",
    show_default=True,
)
def main(
    limit: int,
    source_json: Path,
    prompt_file: Path | None,
    output_dir: Path,
    screenshots_dir: Path | None,
    raw_dir: Path | None,
    headless: bool,
    offline: bool,
    browser: bool,
    capture_only: bool,
    browser_channel: str,
    resume: bool,
) -> None:
    """大模型截图识别实验.

    读取任务二 JSON，从前 10 页各选一条已成交记录，
    通过 Playwright 截图后调用多模态模型识别，与参考值比对后导出 Excel。
    """
    if screenshots_dir is None:
        screenshots_dir = output_dir / "screenshots"
    if raw_dir is None:
        raw_dir = output_dir / "raw_outputs"

    # 加载参考数据
    click.echo(f"▶ 加载参考数据: {source_json}", err=True)
    try:
        records = load_reference_data(source_json)
    except Exception as e:
        click.echo(f"✗ 加载失败: {e}", err=True)
        sys.exit(1)

    # 选择前 10 页各一条
    selected = select_first_per_page(records, pages=DEFAULT_PAGES)
    if limit > 0:
        selected = selected[:limit]

    click.echo(f"  已选择 {len(selected)} 条记录进行截图识别\n", err=True)

    # 检查 API 配置
    cfg = get_config()
    if not offline and not cfg["api_key"]:
        click.echo("⚠  未设置 VISION_API_KEY 环境变量，无法调用模型", err=True)
        click.echo("   提示: 使用 --offline 从 raw_outputs 读取已有结果\n", err=True)

    # 加载提示词
    actual_prompt_path = prompt_file or PROMPT_PATH
    if not offline and actual_prompt_path.exists():
        prompt_template = load_prompt_template(actual_prompt_path)
        click.echo(f"  提示词已加载: {actual_prompt_path}\n", err=True)
    elif not offline:
        click.echo(f"⚠  提示词文件不存在: {actual_prompt_path}\n", err=True)
        prompt_template = ""
    else:
        prompt_template = ""

    results: list[LlmMatchResult] = []

    async def run() -> None:
        nonlocal results

        # 第一步：截图采集（跳过则仅离线模式）
        screenshot_results: list[tuple[str, Path | None]] = []
        if browser and not offline:
            click.echo("▶ 步骤 1/4: 截图采集\n", err=True)
            click.echo("  启动 Chromium 浏览器，穿过两层 iframe 进行截图...", err=True)
            click.echo(f"  模式: {'无头' if headless else '可视'}，间隔 1 秒\n", err=True)

            detail_urls = [r.model_dump().get("detail_url", "") or "" for r in selected]
            try:
                screenshot_results = await capture_all_screenshots(
                    detail_urls,
                    screenshots_dir,
                    headless=headless,
                    delay=1.0,
                    browser_channel=browser_channel,
                )
            except BrowserChannelError as e:
                click.echo(f"✗ 浏览器启动失败: {e}\n", err=True)
                sys.exit(1)
            done = sum(1 for _, p in screenshot_results if p)
            click.echo(f"  截图完成: {done}/{len(screenshot_results)} 张\n", err=True)
        elif offline:
            click.echo("▶ 步骤 1/4: 离线模式（跳过截图）\n", err=True)
            screenshot_results = [(r.model_dump().get("detail_url", "") or "", None) for r in selected]
        else:
            click.echo("▶ 步骤 1/4: 跳过截图（--no-browser）\n", err=True)
            screenshot_results = [(r.model_dump().get("detail_url", "") or "", None) for r in selected]

        # 第二步：离线模式
        if offline:
            click.echo("▶ 步骤 2/4: 离线模式 - 从 raw_outputs 读取已有结果\n", err=True)
            for i, rec in enumerate(selected):
                idx = i + 1
                raw_path = raw_dir / f"raw_{idx:02d}.json"
                if raw_path.exists():
                    parsed = read_raw_json(raw_path)
                    if parsed:
                        raw_text = _read_raw_response_text(raw_path)

                        cmp_result = compare_result(rec, parsed, raw_text, cfg["model"])
                        results.append(cmp_result)
                        guid_short = rec.model_dump().get("row_guid", "")[:8]
                        click.echo(
                            f"  [{idx}] 已读取: {guid_short}... 状态={cmp_result.verification_status}",
                            err=True,
                        )
                    else:
                        click.echo(f"  [{idx}] 解析失败（raw文件存在但格式无效）", err=True)
                else:
                    click.echo(f"  [{idx}] raw_{idx:02d}.json 不存在，跳过", err=True)
            return

        # capture-only 模式：截图完成后直接退出
        if capture_only:
            success_count = sum(1 for _, p in screenshot_results if p)
            click.echo(
                f"\n  截图完成: {success_count}/{len(screenshot_results)} 张",
                err=True,
            )
            click.echo("  --capture-only 模式，跳过模型调用和导出\n", err=True)
            return

        # 第三步：模型调用
        click.echo(f"▶ 步骤 2/4: 调用多模态模型（{cfg['model']}）\n", err=True)
        click.echo(f"  API: {cfg['api_base']}\n", err=True)

        for i, rec in enumerate(selected):
            idx = i + 1
            ref_dump = rec.model_dump()
            screenshot_path = screenshots_dir / f"screenshot_{idx:02d}.png"
            raw_path = raw_dir / f"raw_{idx:02d}.json"

            # resume 模式：检查是否已有有效 raw 结果
            if resume and raw_path.exists():
                parsed = read_raw_json(raw_path)
                if parsed is not None:
                    raw_text = _read_raw_response_text(raw_path)
                    cmp_result = compare_result(rec, parsed, raw_text, cfg["model"])
                    results.append(cmp_result)
                    guid_short = ref_dump.get("row_guid", "")[:8]
                    click.echo(
                        f"  [{idx}/{len(selected)}] 复用已有结果: {guid_short}... "
                        f"状态={cmp_result.verification_status}",
                        err=True,
                    )
                    continue
                else:
                    click.echo(
                        f"  [{idx}/{len(selected)}] 已有 raw 文件但解析失败，重新调用",
                        err=True,
                    )

            if not screenshot_path.exists():
                click.echo(f"  [{idx}/{len(selected)}] 截图不存在: {screenshot_path}", err=True)
                continue

            click.echo(
                f"  [{idx}/{len(selected)}] 识别中: guid={ref_dump.get('row_guid', '')[:8]}...",
                err=True,
            )

            try:
                parsed, raw = await call_vision_model_with_retry(screenshot_path, prompt_template)
            except Exception as e:
                log.warning(
                    "model_call_failed",
                    index=idx,
                    error_type=type(e).__name__,
                    error_repr=repr(e),
                )
                click.echo(f"    模型调用失败: {e}", err=True)
                continue

            if parsed is None:
                click.echo("    解析失败（重试后仍无效），标记待核验", err=True)
                cmp_result = compare_result(rec, LlmRecognitionResult(), raw, cfg["model"])
                cmp_result.parse_status = "parse_error"
                cmp_result.verification_status = "needs_review"
                cmp_result.mismatch_notes = "模型输出无法解析为有效 JSON，请人工核验截图"
                results.append(cmp_result)
                raw_path = save_raw_output(raw, idx, raw_dir)
                click.echo(f"    原始响应已保存: {raw_path}", err=True)
                continue

            # 保存原始响应
            raw_path = save_raw_output(raw, idx, raw_dir)
            click.echo(f"    原始响应已保存: {raw_path}", err=True)

            # 比对结果
            cmp_result = compare_result(rec, parsed, raw, cfg["model"])
            results.append(cmp_result)

            if not cmp_result.mismatch_fields and not cmp_result.semantic_difference_fields:
                click.echo("    ✓ 全部字段匹配", err=True)
            elif not cmp_result.mismatch_fields and cmp_result.semantic_difference_fields:
                click.echo(
                    f"    ⚠ 字段口径差异: {', '.join(cmp_result.semantic_difference_fields)}"
                    f"（{cmp_result.land_usage_verification}）",
                    err=True,
                )
            else:
                parts: list[str] = []
                if cmp_result.mismatch_fields:
                    parts.append(f"差异={', '.join(cmp_result.mismatch_fields)}")
                if cmp_result.semantic_difference_fields:
                    parts.append(f"口径差异={', '.join(cmp_result.semantic_difference_fields)}")
                click.echo(f"    ⚠ {'; '.join(parts)}", err=True)

        # 第四步：导出结果
        click.echo("\n▶ 步骤 3/4: 导出 Excel\n", err=True)
        excel_path = output_dir / "task3_llm_results.xlsx"
        export_to_excel(results, excel_path)
        click.echo(f"  已导出: {excel_path}\n", err=True)

    asyncio.run(run())

    # 打印汇总
    if results:
        match_count = sum(1 for r in results if not r.mismatch_fields and not r.semantic_difference_fields)
        semantic_only_count = sum(1 for r in results if not r.mismatch_fields and r.semantic_difference_fields)
        review_count = sum(1 for r in results if r.mismatch_fields)
        click.echo("\n" + "━" * 50, err=True)
        click.echo("  核验汇总", err=True)
        click.echo(f"  总计: {len(results)} 条", err=True)
        click.echo(f"  全部匹配: {match_count} 条", err=True)
        click.echo(f"  仅字段口径差异: {semantic_only_count} 条", err=True)
        click.echo(f"  需人工核验: {review_count} 条", err=True)
        click.echo("━" * 50, err=True)
        if semantic_only_count > 0:
            click.echo("\n  字段口径差异记录（API分类 vs 详情页实际用途）:", err=True)
            for r in results:
                if not r.mismatch_fields and r.semantic_difference_fields:
                    click.echo(
                        f"  - guid={r.row_guid[:8]}... "
                        f"口径差异={r.semantic_difference_fields} "
                        f"结论={r.land_usage_verification}",
                        err=True,
                    )
        if review_count > 0:
            click.echo("\n  需人工核验记录:", err=True)
            for r in results:
                if r.mismatch_fields:
                    parts: list[str] = []
                    if r.mismatch_fields:
                        parts.append(f"差异={r.mismatch_fields}")
                    if r.semantic_difference_fields:
                        parts.append(f"口径差异={r.semantic_difference_fields}")
                    click.echo(f"  - guid={r.row_guid[:8]}... {'; '.join(parts)}", err=True)
    else:
        click.echo("\n⚠  未生成任何结果", err=True)


if __name__ == "__main__":
    main()
