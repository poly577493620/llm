"""读取任务二 JSON 参考数据，并选择每页第一条已成交记录."""

import json
from pathlib import Path

from foshan_scraper.models import ReferenceRecord


def load_reference_data(json_path: Path) -> list[ReferenceRecord]:
    """加载任务二 JSON 文件并返回 ReferenceRecord 列表.

    文件为标准 UTF-8 编码，直接读取原始字符串，不做任何转码修复。

    Args:
        json_path: 任务二输出的 land_transactions.json 路径

    Returns:
        ReferenceRecord 列表
    """
    with json_path.open(encoding="utf-8") as f:
        raw_list = json.load(f)

    records: list[ReferenceRecord] = []
    for raw_item in raw_list:
        try:
            records.append(ReferenceRecord.model_validate(raw_item))
        except Exception:
            pass

    return records


def _is_yichengjiao(status_str: str | None) -> bool:
    """判断 status 是否为已成交.

    Args:
        status_str: 状态字符串

    Returns:
        True 表示已成交
    """
    if status_str is None:
        return False
    cleaned = status_str.strip()
    return cleaned == "已成交"


def select_first_per_page(
    records: list[ReferenceRecord],
    pages: int = 10,
) -> list[ReferenceRecord]:
    """从任务二 JSON 中选择每页第一条已成交记录.

    Args:
        records: 所有参考记录
        pages: 最大页数

    Returns:
        每页第一条已成交记录列表（共 pages 条）
    """
    selected: list[ReferenceRecord] = []

    for page_num in range(pages):
        page_start = page_num * 10
        page_end = page_start + 10
        page_records = records[page_start:page_end]

        found = None
        for rec in page_records:
            if rec.status and _is_yichengjiao(rec.status):
                found = rec
                break

        if found is None and page_records:
            # 未找到已成交，取第一条
            found = page_records[0]

        if found is not None:
            selected.append(found)

    return selected
