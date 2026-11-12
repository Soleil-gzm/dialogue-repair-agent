"""测试：四类硬约束

重点验证"该拦的拦住、不该拦的不拦"。
"""

import copy

from agent.eval.rules import (
    R1_SYSTEM_IMMUTABLE, R2_STRUCTURE_INVARIANT, R3_ENTITY_FIDELITY,
    R4_SPEAKER_SET, check_consistency, is_consistent,
)


def _rules(violations):
    return {v["rule"] for v in violations}


class TestCleanRewritePasses:
    def test_identical_dialogue_passes(self, mini_dialogues):
        d = mini_dialogues[0]
        assert check_consistency(d, copy.deepcopy(d)) == []

    def test_rewrite_keeping_entities_passes(self, mini_dialogues):
        """只改措辞、保留实体与结构 → 应通过。"""
        d = copy.deepcopy(mini_dialogues[0])
        for m in d["messages"]:
            if m["role"] == "assistant":
                m["content"] = "您好，" + m["content"]
        assert is_consistent(mini_dialogues[0], d) is True


class TestR1SystemImmutable:
    def test_system_change_detected(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"][0]["content"] += "\n额外一行"
        assert R1_SYSTEM_IMMUTABLE in _rules(check_consistency(mini_dialogues[0], d))

    def test_system_whitespace_change_detected(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"][0]["content"] = d["messages"][0]["content"].replace("\n", "\r\n")
        assert R1_SYSTEM_IMMUTABLE in _rules(check_consistency(mini_dialogues[0], d))


class TestR2StructureInvariant:
    def test_message_count_change(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"].pop()
        assert R2_STRUCTURE_INVARIANT in _rules(check_consistency(mini_dialogues[0], d))

    def test_role_change(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"][1]["role"] = "assistant"
        assert R2_STRUCTURE_INVARIANT in _rules(check_consistency(mini_dialogues[0], d))

    def test_loss_missing_vs_false_is_different(self, mini_dialogues):
        """loss 三态：缺失 != "False"，归一化会破坏字节级回环。"""
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"][1]["loss"] = "False"  # 原本缺失
        assert R2_STRUCTURE_INVARIANT in _rules(check_consistency(mini_dialogues[0], d))

    def test_loss_true_to_true_passes(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        assert check_consistency(mini_dialogues[0], d) == []


class TestR3EntityFidelity:
    def test_dropped_entity_detected(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        for m in d["messages"]:
            if m["role"] == "user":
                m["content"] = m["content"].replace("订单号 ORD-", "订单号 ")
                break
        violations = check_consistency(mini_dialogues[0], d)
        assert R3_ENTITY_FIDELITY in _rules(violations)
        assert any("实体丢失" in v["detail"] for v in violations)

    def test_added_entity_detected(self, mini_dialogues):
        """凭空编造金额必须拦住——这是最重要的一类错误。"""
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"][-1]["content"] += "本期应还 ¥9999.99。"
        violations = check_consistency(mini_dialogues[0], d)
        assert R3_ENTITY_FIDELITY in _rules(violations)
        assert any("凭空新增" in v["detail"] for v in violations)


class TestR4SpeakerSet:
    def test_speaker_distribution_change(self, mini_dialogues):
        d = copy.deepcopy(mini_dialogues[0])
        d["messages"][1]["role"] = "assistant"
        assert R4_SPEAKER_SET in _rules(check_consistency(mini_dialogues[0], d))


class TestRobustness:
    def test_empty_dialogues(self):
        assert check_consistency({"messages": []}, {"messages": []}) == []

    def test_none_input_safe(self):
        # 不抛异常即可；具体违规内容不在本层负责
        assert isinstance(check_consistency(None, None), list)
