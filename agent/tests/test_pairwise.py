"""测试：E2 双向盲评（解析 + 双向映射 + compare 接入）"""

import pytest

from agent.benchmark.agent_eval import A3, compare_arms
from agent.eval.judge import JudgeError, LLMJudge
from agent.eval.pairwise import PairwiseJudge, compare_blind, parse_winner
from agent.runtime.llm import LLMResponse


class _ScriptClient:
    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(text=self._texts.pop(0), input_tokens=1)


def _dialogue(text):
    return {"messages": [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": text},
    ]}


def _winner(value):
    return '{"winner": "%s", "reason": "ok"}' % value


class TestParseWinner:
    @pytest.mark.parametrize("value", ["left", "right", "tie", "LEFT"])
    def test_accepts_contract_values(self, value):
        assert parse_winner(_winner(value)) == value.lower()

    def test_rejects_empty(self):
        with pytest.raises(JudgeError):
            parse_winner("")

    def test_rejects_missing_winner(self):
        with pytest.raises(JudgeError):
            parse_winner('{"reason": "忘了给结论"}')

    def test_rejects_conflicting_winners(self):
        with pytest.raises(JudgeError):
            parse_winner('{"winner": "left", "winner": "right"}')


class TestPairwiseJudge:
    def test_compare_passes_stage_and_parses(self):
        client = _ScriptClient([_winner("right")])
        judge = PairwiseJudge(client, run_id="run-x")
        assert judge.compare(_dialogue("甲"), _dialogue("乙")) == "right"
        assert client.calls[0]["stage"] == "pairwise"
        assert client.calls[0]["temperature"] == 0.0
        assert judge.calls == 1

    def test_invalid_output_raises(self):
        judge = PairwiseJudge(_ScriptClient(["not json"]))
        with pytest.raises(JudgeError):
            judge.compare(_dialogue("甲"), _dialogue("乙"))


class TestCompareBlind:
    def test_agreement_counts_as_win(self):
        client = _ScriptClient([_winner("left"), _winner("right")])
        result = compare_blind(
            PairwiseJudge(client),
            [("k", _dialogue("甲"), _dialogue("乙"))],
            arm_a="A", arm_b="B",
        )
        assert result["wins_a"] == 1
        assert result["wins_b"] == 0
        assert result["contradictions"] == 0
        assert result["consistency_rate"] == 1.0
        assert result["win_rate_a"]["point"] == 1.0

    def test_disagreement_counts_as_tie(self):
        client = _ScriptClient([_winner("left"), _winner("left")])
        result = compare_blind(
            PairwiseJudge(client),
            [("k", _dialogue("甲"), _dialogue("乙"))],
            arm_a="A", arm_b="B",
        )
        assert result["wins_a"] == 0 and result["wins_b"] == 0
        assert result["ties"] == 1
        assert result["contradictions"] == 1
        assert result["consistency_rate"] == 0.0
        assert result["win_rate_a"]["point"] is None

    def test_second_arm_agreement(self):
        client = _ScriptClient([_winner("right"), _winner("left")])
        result = compare_blind(
            PairwiseJudge(client),
            [("k", _dialogue("甲"), _dialogue("乙"))],
            arm_a="A", arm_b="B",
        )
        assert result["wins_b"] == 1
        assert result["consistency_rate"] == 1.0

    def test_errors_are_isolated_from_consistency(self):
        client = _ScriptClient(["bad output"])
        result = compare_blind(
            PairwiseJudge(client),
            [("k", _dialogue("甲"), _dialogue("乙"))],
            arm_a="A", arm_b="B",
        )
        assert result["errors"] == 1
        assert result["judgments"] == 0
        assert result["consistency_rate"] is None


class TestCompareArmsWiring:
    def test_compare_arms_can_enable_double_blind(self, mini_dialogues):
        # 固定返回 left：A/B 与 B/A 必然矛盾，应全部记 tie
        client = _ScriptClient([_winner("left")] * (2 * len(mini_dialogues)))
        payload = compare_arms(
            A3, A3, mini_dialogues, [], client=client,
            judge=LLMJudge(client=client), metric="clean", double_blind=True,
        )
        blind = payload["double_blind"]
        assert blind["n_pairs"] == len(mini_dialogues)
        assert blind["contradictions"] == len(mini_dialogues)
        assert blind["consistency_rate"] == 0.0
        assert blind["win_rate_a"]["point"] is None

    def test_compare_arms_skips_double_blind_by_default(self, mini_dialogues):
        payload = compare_arms(A3, A3, mini_dialogues, [], metric="clean")
        assert payload["double_blind"] is None

    def test_double_blind_requires_client(self, mini_dialogues):
        with pytest.raises(ValueError):
            compare_arms(A3, A3, mini_dialogues, [], metric="clean", double_blind=True)
