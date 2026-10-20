"""数据模型."""

from datetime import datetime

from pydantic import BaseModel, Field


class ReferenceRecord(BaseModel):
    """任务二 JSON 中的一条参考记录."""

    row_guid: str
    transaction_date: str | None = None
    winner: str | None = None
    transaction_area: str | None = None
    transaction_price: str | None = None
    land_location: str | None = None
    land_usage: str | None = None
    detail_url: str | None = None
    crawled_at: datetime | None = None
    status: str | None = None


class LlmRecognitionResult(BaseModel):
    """大模型截图识别结果."""

    transaction_date: str | None = Field(default=None, description="成交时间")
    winner: str | None = Field(default=None, description="竞得人")
    transaction_area: str | None = Field(default=None, description="交易土地面积")
    transaction_price: str | None = Field(default=None, description="成交地价")
    land_location: str | None = Field(default=None, description="地块位置")
    land_usage: str | None = Field(default=None, description="土地实际用途")
    uncertain_fields: list[str] = Field(default_factory=list, description="无法确认的字段列表")
    notes: str | None = Field(default=None, description="备注")


class LlmMatchResult(BaseModel):
    """模型识别结果与参考值的比对结果."""

    row_guid: str
    detail_url: str
    reference: ReferenceRecord
    llm_result: LlmRecognitionResult
    model_name: str
    raw_response: str

    # 比较结果
    transaction_date_match: bool = False
    winner_match: bool = False
    transaction_area_match: bool = False
    transaction_price_match: bool = False
    land_location_match: bool = False
    land_usage_match: bool = False

    # 解析状态
    parse_status: str = "success"  # success | retry_failed | parse_error
    verification_status: str = "match"  # match | needs_review

    # 差异字段列表（真正 OCR 识别错误 or 待核验字段）
    mismatch_fields: list[str] = Field(default_factory=list)

    # 字段口径差异列表（API 分类 vs 详情页实际用途）
    semantic_difference_fields: list[str] = Field(default_factory=list)

    # 差异说明
    mismatch_notes: str = ""

    # 土地用途口径差异专用字段
    reference_land_usage_category: str | None = None
    llm_actual_land_usage: str | None = None
    land_usage_verification: str = ""
