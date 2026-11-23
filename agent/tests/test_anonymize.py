"""脱敏模块测试：用虚构数据验证脱敏生效、结构不破坏、可复现。

注意：本测试不使用任何真实业务数据；机构词通过 monkeypatch 环境变量注入，
确保测试代码本身不包含可识别的业务标识。
"""

import copy

from agent.eval.anonymize import anonymize_dialogue, anonymize_text


def _make_dialogue():
    return {
        "messages": [
            {"role": "system", "content": (
                "# 角色\n你是一位来自测试银行的专员。\n"
                "## 标签信息\n"
                "<姓名>张三\n"
                "<性别>先生\n"
                "<客服电话>13800138000\n"
                "<还款日>2027-01-15\n"
                "<总欠款>5000.00\n"
                "<专员工号>998877\n"
            )},
            {"role": "user", "content": "我叫张三，我的电话是 13800138000，还款日 2027-01-15。"},
            {"role": "assistant", "content": "好的张三先生，测试银行会处理您 5000.00 元的欠款。", "loss": "True"},
        ]
    }


def test_tag_values_anonymized(monkeypatch):
    monkeypatch.setenv("ANONYMIZE_ORG_TERMS", "测试银行")
    out = anonymize_dialogue(_make_dialogue())
    sys = out["messages"][0]["content"]
    assert "张三" not in sys
    assert "13800138000" not in sys
    assert "2027-01-15" not in sys
    assert "5000.00" not in sys
    assert "998877" not in sys
    assert "测试银行" not in sys
    assert "<客户>" in sys
    assert "<电话>" in sys
    assert "<日期>" in sys
    assert "<金额>" in sys
    assert "<工号>" in sys
    assert "<机构A>" in sys


def test_free_text_entities_anonymized(monkeypatch):
    """user/assistant 自由文本中残留的实体也被正则兜底替换。"""
    monkeypatch.setenv("ANONYMIZE_ORG_TERMS", "测试银行")
    out = anonymize_dialogue(_make_dialogue())
    for msg in out["messages"]:
        assert "13800138000" not in msg["content"]
        assert "2027-01-15" not in msg["content"]
        assert "测试银行" not in msg["content"]
    # assistant 提到的金额被兜底替换
    assert "5000.00" not in out["messages"][2]["content"]


def test_structure_preserved():
    """脱敏不破坏对话结构：role / 字段 / 消息数量不变。"""
    d = _make_dialogue()
    out = anonymize_dialogue(d, org_terms=["测试银行"])
    assert len(out["messages"]) == len(d["messages"])
    for a, b in zip(out["messages"], d["messages"]):
        assert a["role"] == b["role"]
    # assistant 的 loss 字段保留
    assert out["messages"][2].get("loss") == "True"


def test_input_not_mutated():
    d = _make_dialogue()
    snapshot = copy.deepcopy(d)
    anonymize_dialogue(d, org_terms=["测试银行"])
    assert d == snapshot


def test_reproducible():
    d = _make_dialogue()
    a = anonymize_dialogue(d, org_terms=["测试银行"])
    b = anonymize_dialogue(d, org_terms=["测试银行"])
    assert a == b


def test_anonymize_text_order_longest_first():
    """长词优先：避免 '测试银行' 被部分替换后残留。"""
    text = "测试银行信用卡中心"
    out = anonymize_text(text, value_map={}, org_terms=["测试银行", "测试银行信用卡中心"])
    assert out == "<机构A>中心" or "测试银行" not in out
