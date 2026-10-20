"""测试字段比较模块（不访问网络）."""

from foshan_scraper.compare import _fields_match, _normalize, compare_result
from foshan_scraper.models import LlmRecognitionResult, ReferenceRecord


class TestNormalize:
    """_normalize 测试."""

    def test_strips_whitespace(self) -> None:
        """去除首尾空白."""
        assert _normalize("  hello  ") == "hello"

    def test_removes_commas_in_numbers(self) -> None:
        """去除数字中的逗号."""
        assert _normalize("15,746万元") == "15746万元"

    def test_preserves_units(self) -> None:
        """保留单位，只去除逗号和首尾空白."""
        assert _normalize("15,746万元") == "15746万元"
        assert _normalize("4840.75 平方米") == "4840.75 平方米"
        assert _normalize("100㎡") == "100㎡"

    def test_lowercase(self) -> None:
        """转小写."""
        assert _normalize("TEST公司") == "test公司"

    def test_empty_string(self) -> None:
        """空字符串返回空."""
        assert _normalize("") == ""
        assert _normalize(None) == ""


class TestFieldsMatch:
    """_fields_match 测试."""

    def test_both_none_matches(self) -> None:
        """两者都为空时匹配."""
        assert _fields_match(None, None) is True

    def test_both_empty_matches(self) -> None:
        """两者都为空字符串时匹配."""
        assert _fields_match("", "") is True

    def test_one_none_does_not_match(self) -> None:
        """只有一方为空时不匹配."""
        assert _fields_match("value", None) is False
        assert _fields_match(None, "value") is False

    def test_exact_match(self) -> None:
        """完全一致匹配."""
        assert _fields_match("4840.75", "4840.75") is True

    def test_normalized_match(self) -> None:
        """规范化后匹配."""
        assert _fields_match("  4840.75 平方米", "4840.75") is True
        assert _fields_match("TEST公司", "test公司") is True

    def test_area_float_match(self) -> None:
        """面积字段浮点数近似匹配（差异 < 0.01）."""
        assert _fields_match("4840.751", "4840.75") is True
        assert _fields_match("4840.74", "4840.75") is True

    def test_area_no_match_too_different(self) -> None:
        """面积字段差异过大不匹配."""
        assert _fields_match("5000.00", "4840.75") is False


class TestCompareResult:
    """compare_result 测试（mock 数据，不访问网络）."""

    def _make_ref(
        self,
        transaction_date: str = "2026/7/27 15:00:02",
        winner: str = "测试公司",
        transaction_area: str = "4840.75",
        transaction_price: str = "地价:15746万元",
        land_location: str = "佛山市禅城区",
        land_usage: str = "居住用地",
    ) -> ReferenceRecord:
        return ReferenceRecord(
            row_guid="test-guid",
            transaction_date=transaction_date,
            winner=winner,
            transaction_area=transaction_area,
            transaction_price=transaction_price,
            land_location=land_location,
            land_usage=land_usage,
            detail_url="http://example.com",
        )

    def _make_llm(
        self,
        transaction_date: str | None = None,
        winner: str | None = None,
        transaction_area: str | None = None,
        transaction_price: str | None = None,
        land_location: str | None = None,
        land_usage: str | None = None,
        uncertain_fields: list[str] | None = None,
    ) -> LlmRecognitionResult:
        return LlmRecognitionResult(
            transaction_date=transaction_date,
            winner=winner,
            transaction_area=transaction_area,
            transaction_price=transaction_price,
            land_location=land_location,
            land_usage=land_usage,
            uncertain_fields=uncertain_fields or [],
        )

    def test_all_match(self) -> None:
        """全部字段匹配时 verification_status = match."""
        ref = self._make_ref()
        llm = self._make_llm(
            transaction_date="2026/7/27 15:00:02",
            winner="测试公司",
            transaction_area="4840.75 平方米",
            transaction_price="地价:15746万元",
            land_location="佛山市禅城区",
            land_usage="居住用地",
        )
        result = compare_result(ref, llm, "raw text", "gpt-4o")

        assert result.verification_status == "match"
        assert result.mismatch_fields == []
        assert result.semantic_difference_fields == []
        assert result.land_usage_verification == "完全匹配"
        assert result.parse_status == "success"

    def test_partial_mismatch(self) -> None:
        """部分字段不匹配时 verification_status = needs_review."""
        ref = self._make_ref()
        llm = self._make_llm(
            transaction_date="2026/7/27 15:00:02",
            winner="错误公司",  # 不匹配
            transaction_area="4840.75",
            transaction_price="地价:15746万元",
            land_location="佛山市禅城区",
            land_usage="居住用地",
        )
        result = compare_result(ref, llm, "raw text", "gpt-4o")

        assert result.verification_status == "needs_review"
        assert "winner" in result.mismatch_fields
        assert "transaction_area" not in result.mismatch_fields

    def test_uncertain_fields_flagged(self) -> None:
        """模型不确定的字段在备注中标注."""
        ref = self._make_ref()
        # winner 传 None（模型无法识别），其余字段完全匹配
        llm = self._make_llm(
            transaction_date="2026/7/27 15:00:02",
            winner=None,  # 模型不确定
            transaction_area="4840.75",
            transaction_price="地价:15746万元",
            land_location="佛山市禅城区",
            land_usage="居住用地",
            uncertain_fields=["winner"],
        )
        result = compare_result(ref, llm, "raw text", "gpt-4o")

        assert result.verification_status == "needs_review"
        assert "winner" in result.mismatch_fields
        assert "需人工核验" in result.mismatch_notes


class TestLandUsageVerification:
    """land_usage 核验结论专项测试."""

    def _make_ref(
        self,
        land_usage: str = "商业服务业用地·商业用地",
    ) -> ReferenceRecord:
        return ReferenceRecord(
            row_guid="test-guid",
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage=land_usage,
            detail_url="http://example.com",
        )

    def test_land_usage_exact_match(self) -> None:
        """land_usage 完全一致 → 完全匹配，不加入任何差异列表."""
        ref = self._make_ref("居住用地")
        llm = LlmRecognitionResult(
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage="居住用地",
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert result.land_usage_verification == "完全匹配"
        assert result.land_usage_match is True
        assert result.mismatch_fields == []
        assert result.semantic_difference_fields == []
        assert result.verification_status == "match"

    def test_land_usage_semantic_difference(self) -> None:
        """API分类与详情页实际用途不一致 → 口径差异，不放入 mismatch_fields."""
        ref = self._make_ref("商业服务业用地·商业用地")
        llm = LlmRecognitionResult(
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage="商业用地（零售商业用地、餐饮用地、旅馆用地），兼容娱乐用地",
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert result.land_usage_verification == "字段口径不同，需人工核验"
        assert result.land_usage_match is False
        assert "land_usage" not in result.mismatch_fields
        assert "land_usage" in result.semantic_difference_fields
        assert result.verification_status == "needs_review"
        assert "口径差异" in result.mismatch_notes

    def test_land_usage_empty_is_uncertain(self) -> None:
        """模型 land_usage 为空 → 识别结果待核验，放入 mismatch_fields."""
        ref = self._make_ref("商业服务业用地·商业用地")
        llm = LlmRecognitionResult(
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage=None,
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert result.land_usage_verification == "识别结果待核验"
        assert "land_usage" in result.mismatch_fields
        assert result.semantic_difference_fields == []
        assert result.verification_status == "needs_review"

    def test_land_usage_daiheyan_is_uncertain(self) -> None:
        """模型 land_usage 为"待核验" → 识别结果待核验."""
        ref = self._make_ref("商业服务业用地·商业用地")
        llm = LlmRecognitionResult(
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage="待核验",
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert result.land_usage_verification == "识别结果待核验"
        assert "land_usage" in result.mismatch_fields
        assert result.semantic_difference_fields == []

    def test_land_usage_in_uncertain_fields_is_uncertain(self) -> None:
        """land_usage 被列入 uncertain_fields → 识别结果待核验."""
        ref = self._make_ref("商业服务业用地·商业用地")
        llm = LlmRecognitionResult(
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage="看不清",
            uncertain_fields=["land_usage"],
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert result.land_usage_verification == "识别结果待核验"
        assert "land_usage" in result.mismatch_fields
        assert result.semantic_difference_fields == []

    def test_only_land_usage_semantic_other_five_match(self) -> None:
        """仅 land_usage 口径不同，其他五个字段全部匹配 → 仅口径差异."""
        ref = self._make_ref("商业服务业用地·商业用地")
        llm = LlmRecognitionResult(
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage="商业用地（零售商业用地、餐饮用地、旅馆用地），兼容娱乐用地",
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert result.mismatch_fields == []
        assert result.semantic_difference_fields == ["land_usage"]
        assert result.transaction_date_match is True
        assert result.winner_match is True
        assert result.transaction_area_match is True
        assert result.transaction_price_match is True
        assert result.land_location_match is True
        assert result.land_usage_match is False

    def test_land_usage_semantic_plus_other_mismatch(self) -> None:
        """land_usage 口径不同，同时存在其他真正差异 → mismatch_fields 和 semantic_difference_fields 都有值."""
        ref = self._make_ref("商业服务业用地·商业用地")
        llm = LlmRecognitionResult(
            transaction_date="2025/1/1",  # 明显不匹配
            winner="测试公司",
            transaction_area="1000",
            transaction_price="地价:1000万元",
            land_location="佛山",
            land_usage="商业用地（零售商业用地、餐饮用地、旅馆用地），兼容娱乐用地",
        )
        result = compare_result(ref, llm, "raw", "gpt-4o")

        assert "transaction_date" in result.mismatch_fields
        assert "land_usage" not in result.mismatch_fields
        assert "land_usage" in result.semantic_difference_fields
        assert result.verification_status == "needs_review"
        assert "口径差异" in result.mismatch_notes
