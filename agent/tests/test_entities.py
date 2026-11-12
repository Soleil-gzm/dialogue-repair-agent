"""测试：实体抽取

要求：领域无关、确定、不依赖 LLM。
"""

from agent.eval.entities import (
    extract_entities, extract_entities_from_messages, extract_tag_values, system_text,
)


class TestExtractEntities:
    def test_finds_order_id(self):
        assert "ORD-123456" in extract_entities("订单号 ORD-123456 还没到")

    def test_finds_customer_id(self):
        assert "CUST-0007" in extract_entities("客户 CUST-0007")

    def test_finds_hotline(self):
        assert "400-000-0000" in extract_entities("请拨 400-000-0000")

    def test_finds_mobile(self):
        assert "13800138000" in extract_entities("联系 13800138000")

    def test_finds_yuan_amount(self):
        assert "¥1,234.56" in extract_entities("应还 ¥1,234.56")

    def test_finds_amount_with_suffix(self):
        assert "88.00元" in extract_entities("共 88.00元")

    def test_finds_date(self):
        assert "2026-11-12" in extract_entities("时间 2026-11-12")

    def test_ignores_plain_numbers(self):
        assert extract_entities("一共 3 个问题") == []

    def test_empty_and_none_safe(self):
        assert extract_entities("") == []
        assert extract_entities(None) == []

    def test_dedup_and_sorted(self):
        out = extract_entities("ORD-111111 和 ORD-111111，再加 ORD-222222")
        assert out == ["ORD-111111", "ORD-222222"]

    def test_whitespace_normalized(self):
        assert extract_entities("¥ 88.00") == ["¥88.00"]


class TestExtractFromMessages:
    def test_excludes_system(self, mini_dialogues):
        """system 由 R1 保证逐字不变，实体比对只看对话正文。"""
        ents = extract_entities_from_messages(mini_dialogues[0])
        assert all("CUST-" not in e for e in ents), "system 中的客户编号不应计入"

    def test_includes_user_entities(self, mini_dialogues):
        ents = extract_entities_from_messages(mini_dialogues[0])
        assert any(e.startswith("ORD-") for e in ents)

    def test_empty_dialogue(self):
        assert extract_entities_from_messages({"messages": []}) == []


class TestTagValues:
    def test_hash_header_format(self, mini_dialogues):
        vals = extract_tag_values(system_text(mini_dialogues[0]))
        assert "订单编号" in vals
        assert vals["订单编号"].startswith("ORD-")

    def test_angle_bracket_format(self):
        vals = extract_tag_values("<姓名>张三\n<金额>88.00")
        assert vals == {"姓名": "张三", "金额": "88.00"}

    def test_both_formats_coexist(self):
        text = "## 订单编号\nORD-123456\n<金额>88.00"
        vals = extract_tag_values(text)
        assert vals["订单编号"] == "ORD-123456"
        assert vals["金额"] == "88.00"

    def test_missing_system_returns_empty(self):
        assert system_text({"messages": []}) == ""
        assert extract_tag_values("") == {}
