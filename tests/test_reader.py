"""测试参考数据读取模块（不访问真实网站）."""

import json
import tempfile
from pathlib import Path

from foshan_scraper.models import ReferenceRecord
from foshan_scraper.reader import (
    _is_yichengjiao,
    load_reference_data,
    select_first_per_page,
)


class TestIsYiChengjiao:
    """_is_yichengjiao 测试."""

    def test_yichengjiao_true(self) -> None:
        """已成交返回 True."""
        assert _is_yichengjiao("已成交") is True

    def test_yichengjiao_with_whitespace(self) -> None:
        """前后有空格的已成交也能识别."""
        assert _is_yichengjiao("  已成交  ") is True

    def test_other_status_false(self) -> None:
        """其他状态返回 False."""
        assert _is_yichengjiao("不成交") is False

    def test_empty_false(self) -> None:
        """空字符串返回 False."""
        assert _is_yichengjiao("") is False

    def test_none_false(self) -> None:
        """None 返回 False."""
        assert _is_yichengjiao(None) is False


class TestLoadReferenceData:
    """load_reference_data 测试（mock 数据，不访问网络）."""

    def test_loads_mock_json(self) -> None:
        """从临时文件加载并解析 mock 数据."""
        mock_data = [
            {
                "row_guid": "test-guid-001",
                "transaction_date": "2026/7/27 15:00:02",
                "winner": "测试公司A",
                "transaction_area": "4840.75",
                "detail_url": "http://example.com/detail1",
            },
            {
                "row_guid": "test-guid-002",
                "transaction_date": "2026/7/28 10:00:00",
                "winner": "测试公司B",
                "transaction_area": "2000.00",
                "detail_url": "http://example.com/detail2",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(mock_data, f)
            tmp_path = Path(f.name)

        try:
            records = load_reference_data(tmp_path)
            assert len(records) == 2
            assert records[0].row_guid == "test-guid-001"
            assert records[1].winner == "测试公司B"
            assert records[0].transaction_area == "4840.75"
        finally:
            tmp_path.unlink()

    def test_ignores_invalid_records(self) -> None:
        """缺少必填字段的记录被跳过."""
        mock_data = [
            {"row_guid": "valid-guid"},
            {"winner": "only-winner"},  # 缺少 row_guid
            {"row_guid": "another-valid-guid"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(mock_data, f)
            tmp_path = Path(f.name)

        try:
            records = load_reference_data(tmp_path)
            assert len(records) == 2
        finally:
            tmp_path.unlink()

    def test_chinese_strings_unchanged(self) -> None:
        """标准 UTF-8 中文读取后必须保持原样，不得出现乱码."""
        mock_data = [
            {
                "row_guid": "guid-utf8",
                "transaction_date": "2026/7/27 15:00:02",
                "winner": "佛山建发城市发展有限公司",
                "transaction_area": "4840.75",
                "transaction_price": "地价:15746万元",
                "land_location": "佛山市三水区云东海街道云东海大道东侧、规划纵十三路西侧地块",
                "land_usage": "居住用地·城镇住宅用地·二类城镇住宅用地",
                "detail_url": "http://example.com/detail",
                "status": "已成交",
            },
            {
                "row_guid": "guid-industry",
                "transaction_date": "2026/7/28 10:00:00",
                "winner": "佛山某某科技有限公司",
                "transaction_area": "10000.00",
                "transaction_price": "地价:5000万元",
                "land_location": "佛山市高明区某某路",
                "land_usage": "工矿用地·工业用地",
                "detail_url": "http://example.com/detail2",
                "status": "已成交",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(mock_data, f, ensure_ascii=False)
            tmp_path = Path(f.name)

        try:
            records = load_reference_data(tmp_path)
            assert len(records) == 2

            rec0 = records[0]
            assert rec0.winner == "佛山建发城市发展有限公司"
            assert rec0.land_location == "佛山市三水区云东海街道云东海大道东侧、规划纵十三路西侧地块"
            assert rec0.land_usage == "居住用地·城镇住宅用地·二类城镇住宅用地"
            assert rec0.status == "已成交"

            rec1 = records[1]
            assert rec1.winner == "佛山某某科技有限公司"
            assert rec1.land_usage == "工矿用地·工业用地"
            assert rec1.status == "已成交"
        finally:
            tmp_path.unlink()

    def test_chinese_no_garbled_characters(self) -> None:
        """读取后字符串中不得出现 GBK 乱码特征字符."""
        # 这些字符串在错误的 GBK 解码后会产生乱码
        original_winner = "佛山建发城市发展有限公司"
        original_location = "佛山市三水区云东海街道云东海大道东侧、规划纵十三路西侧地块"
        original_usage = "居住用地·城镇住宅用地·二类城镇住宅用地"

        mock_data = [
            {
                "row_guid": "guid-check",
                "winner": original_winner,
                "land_location": original_location,
                "land_usage": original_usage,
                "status": "已成交",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(mock_data, f, ensure_ascii=False)
            tmp_path = Path(f.name)

        try:
            records = load_reference_data(tmp_path)
            assert len(records) == 1
            rec = records[0]
            # 验证完全没有变化
            assert rec.winner == original_winner
            assert rec.land_location == original_location
            assert rec.land_usage == original_usage
            # 不会出现乱码常见特征（如替换字符、锟、拷等）
            assert "�" not in (rec.winner or "")
            assert "锟" not in (rec.winner or "")
            assert "拷" not in (rec.winner or "")
        finally:
            tmp_path.unlink()


class TestSelectFirstPerPage:
    """select_first_per_page 测试（mock 数据）."""

    def _make_record(self, guid: str, status: str) -> ReferenceRecord:
        """创建 mock ReferenceRecord."""
        return ReferenceRecord(
            row_guid=guid,
            transaction_date="2026/7/27",
            winner="测试公司",
            transaction_area="1000",
            detail_url=f"http://example.com/{guid}",
            status=status,
        )

    def test_selects_first_yichengjiao_per_page(self) -> None:
        """每页选择第一条已成交记录."""
        # 10 条记录，前 5 条不成交，后 5 条已成交
        records = [self._make_record(f"no-{i}", "不成交") for i in range(5)] + [
            self._make_record(f"yes-{i}", "已成交") for i in range(5)
        ]

        selected = select_first_per_page(records, pages=1)
        assert len(selected) == 1
        assert selected[0].row_guid == "yes-0"

    def test_fallback_to_first_when_no_yichengjiao(self) -> None:
        """某页无已成交记录时 fallback 到第一条."""
        records = [self._make_record(f"no-{i}", "不成交") for i in range(10)]

        selected = select_first_per_page(records, pages=1)
        assert len(selected) == 1
        assert selected[0].row_guid == "no-0"

    def test_multiple_pages(self) -> None:
        """多页测试：每页一条.

        每页 10 条，3 页共 30 条；已成交记录分别在每页的第一条位置。
        """
        # 10 条/页 × 3 页 = 30 条；已成交在各页首条
        records = []
        for page in range(3):
            base = page * 10
            records.append(self._make_record(f"page{page}-yes-{base}", "已成交"))
            for i in range(1, 10):
                records.append(self._make_record(f"page{page}-no-{base + i}", "不成交"))

        selected = select_first_per_page(records, pages=3)
        assert len(selected) == 3
        assert selected[0].row_guid.startswith("page0-yes")
        assert selected[1].row_guid.startswith("page1-yes")
        assert selected[2].row_guid.startswith("page2-yes")

    def test_status_none_falls_back_to_first(self) -> None:
        """status 为 None 时视为无已成交，fallback 到第一条."""
        records = [
            ReferenceRecord(row_guid="none-0", status=None),
            ReferenceRecord(row_guid="none-1", status=None),
        ]

        selected = select_first_per_page(records, pages=1)
        assert len(selected) == 1
        assert selected[0].row_guid == "none-0"

    def test_mixed_status_in_page(self) -> None:
        """一页内混合状态时选第一条已成交."""
        records = [
            ReferenceRecord(row_guid="mix-no", status="不成交"),
            ReferenceRecord(row_guid="mix-yes", status="已成交"),
            ReferenceRecord(row_guid="mix-no2", status="不成交"),
        ]

        selected = select_first_per_page(records, pages=1)
        assert len(selected) == 1
        assert selected[0].row_guid == "mix-yes"
