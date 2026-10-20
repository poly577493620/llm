"""模型识别结果与任务二参考值的比较模块."""

import math
import re

from foshan_scraper.models import LlmMatchResult, LlmRecognitionResult, ReferenceRecord


def _normalize(s: str | None) -> str:
    """规范化字段值用于比较.

    - 去除空白字符
    - 统一数字格式（去除逗号）
    - 转小写
    """
    if s is None:
        return ""
    s = s.strip()
    # 去除数字中的逗号分隔符
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = s.lower()
    return s


def _fields_match(llm_val: str | None, ref_val: str | None) -> bool:
    """判断两个字段值是否匹配.

    匹配规则：
    - 都为空 → 匹配
    - 任一为空 → 不匹配（除非都是空字符串）
    - 规范化后完全一致 → 匹配
    - 数值近似（面积字段）→ 匹配（允许舍入差异）
    """
    if llm_val is None and ref_val is None:
        return True
    if llm_val is None or ref_val is None:
        return False

    norm_llm = _normalize(llm_val)
    norm_ref = _normalize(ref_val)

    if norm_llm == norm_ref:
        return True

    # 面积字段：提取数字部分，允许浮点数舍入差异
    # 只要两个值都包含数字，就尝试数值比较
    try:
        llm_str = re.sub(r"[^\d.]", "", norm_llm)
        ref_str = re.sub(r"[^\d.]", "", norm_ref)
        if llm_str and ref_str:  # both contain numeric parts
            llm_num = float(llm_str)
            ref_num = float(ref_str)
            # 使用 math.isclose，容忍 0.1% 相对误差或 0.01 绝对误差
            if math.isclose(llm_num, ref_num, rel_tol=1e-3, abs_tol=0.01):
                return True
    except ValueError:
        pass

    return False


def _is_empty_or_uncertain(val: str | None, uncertain_fields: list[str], field_name: str) -> bool:
    """判断字段值是否为空、待核验或被标记为不确定."""
    if val is None or val.strip() == "" or val.strip() == "待核验":
        return True
    if field_name in uncertain_fields:
        return True
    return False


def compare_result(
    reference: ReferenceRecord,
    llm_result: LlmRecognitionResult,
    raw_response: str,
    model_name: str,
) -> LlmMatchResult:
    """比较单条记录.

    Args:
        reference: 任务二参考值
        llm_result: 模型识别结果
        raw_response: 原始响应
        model_name: 模型名称

    Returns:
        LlmMatchResult
    """
    ref_dump = reference.model_dump()

    # 五个常规字段比对
    date_match = _fields_match(llm_result.transaction_date, ref_dump.get("transaction_date"))
    winner_match = _fields_match(llm_result.winner, ref_dump.get("winner"))
    area_match = _fields_match(llm_result.transaction_area, ref_dump.get("transaction_area"))
    price_match = _fields_match(llm_result.transaction_price, ref_dump.get("transaction_price"))
    loc_match = _fields_match(llm_result.land_location, ref_dump.get("land_location"))

    # land_usage 单独处理：区分口径差异和识别错误/待核验
    ref_usage = ref_dump.get("land_usage")
    llm_usage = llm_result.land_usage
    uncertain = llm_result.uncertain_fields or []

    usage_match = False
    land_usage_verification = ""
    semantic_difference_fields: list[str] = []

    if _is_empty_or_uncertain(llm_usage, uncertain, "land_usage"):
        # 模型无法识别或标注为不确定 → 真正的待核验问题
        usage_match = False
        land_usage_verification = "识别结果待核验"
    elif _fields_match(llm_usage, ref_usage):
        # 完全一致
        usage_match = True
        land_usage_verification = "完全匹配"
    else:
        # 两边都有有效内容但不一致 → 字段口径差异（API分类 vs 详情页实际用途）
        usage_match = False
        land_usage_verification = "字段口径不同，需人工核验"
        semantic_difference_fields.append("land_usage")

    # 收集真正差异字段（OCR 识别错误或待核验）
    mismatch_fields: list[str] = []
    notes_parts: list[str] = []

    if not date_match:
        mismatch_fields.append("transaction_date")
        notes_parts.append(f"成交时间: 模型=[{llm_result.transaction_date}] 参考=[{ref_dump.get('transaction_date')}]")
    if not winner_match:
        mismatch_fields.append("winner")
        notes_parts.append(f"竞得人: 模型=[{llm_result.winner}] 参考=[{ref_dump.get('winner')}]")
    if not area_match:
        mismatch_fields.append("transaction_area")
        notes_parts.append(f"交易面积: 模型=[{llm_result.transaction_area}] 参考=[{ref_dump.get('transaction_area')}]")
    if not price_match:
        mismatch_fields.append("transaction_price")
        notes_parts.append(
            f"成交地价: 模型=[{llm_result.transaction_price}] 参考=[{ref_dump.get('transaction_price')}]"
        )
    if not loc_match:
        mismatch_fields.append("land_location")
        notes_parts.append(f"地块位置: 模型=[{llm_result.land_location}] 参考=[{ref_dump.get('land_location')}]")

    # land_usage 待核验时加入 mismatch_fields
    if land_usage_verification == "识别结果待核验":
        mismatch_fields.append("land_usage")
        notes_parts.append(
            f"土地用途: 模型=[{llm_result.land_usage}] 参考=[{ref_dump.get('land_usage')}]（识别结果待核验）"
        )

    # 不确定字段处理（仅针对非 land_usage 的真正差异字段）
    for field in uncertain:
        if field in mismatch_fields and field != "land_usage":
            notes_parts.append(f"{field}: 模型标注为不确定，需人工核验")

    # 口径差异说明
    if semantic_difference_fields:
        notes_parts.append(f"土地用途口径差异: API分类=[{ref_usage}] 详情页实际用途=[{llm_usage}]")

    mismatch_notes = "; ".join(notes_parts)

    # verification_status: match / needs_review
    has_real_mismatch = bool(mismatch_fields)
    has_semantic_only = bool(semantic_difference_fields) and not has_real_mismatch
    verification_status = "match" if (not has_real_mismatch and not has_semantic_only) else "needs_review"

    return LlmMatchResult(
        row_guid=ref_dump.get("row_guid", ""),
        detail_url=ref_dump.get("detail_url", ""),
        reference=reference,
        llm_result=llm_result,
        model_name=model_name,
        raw_response=raw_response,
        transaction_date_match=date_match,
        winner_match=winner_match,
        transaction_area_match=area_match,
        transaction_price_match=price_match,
        land_location_match=loc_match,
        land_usage_match=usage_match,
        parse_status="success",
        verification_status=verification_status,
        mismatch_fields=mismatch_fields,
        semantic_difference_fields=semantic_difference_fields,
        mismatch_notes=mismatch_notes,
        reference_land_usage_category=ref_usage,
        llm_actual_land_usage=llm_usage,
        land_usage_verification=land_usage_verification,
    )
