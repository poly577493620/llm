"""测试数据模型."""

from foshan_scraper.models import LlmMatchResult, LlmRecognitionResult, ReferenceRecord


def test_reference_record_creation() -> None:
    """测试 ReferenceRecord 创建."""
    rec = ReferenceRecord(
        row_guid="test-001",
        transaction_date="2026/7/27",
        winner="测试公司",
        transaction_area="4840.75",
    )
    assert rec.row_guid == "test-001"
    assert rec.transaction_date == "2026/7/27"
    assert rec.detail_url is None


def test_llm_recognition_result_creation() -> None:
    """测试 LlmRecognitionResult 创建."""
    result = LlmRecognitionResult(
        transaction_date="2026/7/27",
        winner="测试公司",
        uncertain_fields=["winner"],
        notes="部分文字模糊",
    )
    assert result.transaction_date == "2026/7/27"
    assert result.winner == "测试公司"
    assert result.uncertain_fields == ["winner"]
    assert result.notes == "部分文字模糊"


def test_llm_match_result_creation() -> None:
    """测试 LlmMatchResult 创建."""
    ref = ReferenceRecord(row_guid="test-001", winner="测试公司")
    llm = LlmRecognitionResult(winner="测试公司")
    match = LlmMatchResult(
        row_guid="test-001",
        detail_url="http://example.com",
        reference=ref,
        llm_result=llm,
        model_name="gpt-4o",
        raw_response="{}",
        winner_match=True,
    )
    assert match.row_guid == "test-001"
    assert match.winner_match is True
    assert match.verification_status == "match"
