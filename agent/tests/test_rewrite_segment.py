"""测试：rewrite_segment 工具

覆盖三条边界：只改目标轮、失败不把异常抛给循环、token 与 stage 契约。
"""

import copy
import json

from agent.benchmark.fixtures import generate_dialogues
from agent.runtime.llm import LLMResponse
from agent.tools.rewrite_segment import make_rewrite_segment

FIXED = "已为您重新核实，预计两个工作日内送达。"


class _StubClient:
    def __init__(self, text: str = "", tokens: int = 42, error=None):
        self.text = text or json.dumps({"content": FIXED}, ensure_ascii=False)
        self.tokens = tokens
        self.error = error
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error:
            raise self.error
        return LLMResponse(text=self.text, input_tokens=self.tokens, output_tokens=0)


def _dialogue():
    return generate_dialogues(1, seed=5)[0]


def _first_assistant(dialogue: dict) -> int:
    return next(i for i, m in enumerate(dialogue["messages"])
                if m.get("role") == "assistant")


class TestRewriteSegment:
    def test_only_target_turn_changes(self):
        dialogue = _dialogue()
        turn = _first_assistant(dialogue)
        before = copy.deepcopy(dialogue)

        tool = make_rewrite_segment(_StubClient())
        result = tool(None, {"dialogue": dialogue, "turn": turn, "issues": []})

        assert result["ok"] is True
        assert result["status"] == "rewritten"
        assert result["rewritten_turn"] == turn
        assert result["dialogue"]["messages"][turn]["content"] == FIXED
        # 其它消息逐字保留（含 loss 字段与 system）
        for i, (old, new) in enumerate(zip(before["messages"],
                                          result["dialogue"]["messages"])):
            if i != turn:
                assert old == new
        assert result["dialogue"]["messages"][turn]["loss"] == "True"
        # 输入对象不被原地修改
        assert dialogue == before

    def test_tokens_and_stage_contract(self):
        dialogue = _dialogue()
        turn = _first_assistant(dialogue)
        client = _StubClient(tokens=77)

        result = make_rewrite_segment(client, temperature=0.4, max_tokens=128)(
            None, {"dialogue": dialogue, "turn": turn, "issues": []})

        assert result["tokens"] == 77
        call = client.calls[0]
        assert call["stage"] == "generate"
        assert call["temperature"] == 0.4
        assert call["max_tokens"] == 128
        assert "response_format" not in call

    def test_prompt_mentions_json(self):
        dialogue = _dialogue()
        turn = _first_assistant(dialogue)
        client = _StubClient()
        make_rewrite_segment(client)(None, {"dialogue": dialogue, "turn": turn,
                                            "issues": []})
        text = "\n".join(m["content"] for m in client.calls[0]["messages"])
        assert "json" in text.lower()

    def test_invalid_turn(self):
        dialogue = _dialogue()
        tool = make_rewrite_segment(_StubClient())
        for turn in (-1, 999, 0, "1"):
            result = tool(None, {"dialogue": dialogue, "turn": turn, "issues": []})
            assert result["ok"] is False
            assert result["violations"][0]["rule"] == "W0_invalid_turn"

    def test_parse_error(self):
        dialogue = _dialogue()
        result = make_rewrite_segment(_StubClient(text="不是 json"))(
            None, {"dialogue": dialogue, "turn": _first_assistant(dialogue),
                   "issues": []})
        assert result["ok"] is False
        assert result["violations"][0]["rule"] == "W1_parse_error"
        assert result["tokens"] == 42

    def test_empty_content(self):
        dialogue = _dialogue()
        result = make_rewrite_segment(_StubClient(text='{"content": "   "}'))(
            None, {"dialogue": dialogue, "turn": _first_assistant(dialogue),
                   "issues": []})
        assert result["ok"] is False
        assert result["violations"][0]["rule"] == "W2_empty_content"

    def test_unchanged_content(self):
        dialogue = _dialogue()
        turn = _first_assistant(dialogue)
        original = dialogue["messages"][turn]["content"]
        text = json.dumps({"content": original}, ensure_ascii=False)
        result = make_rewrite_segment(_StubClient(text=text))(
            None, {"dialogue": dialogue, "turn": turn, "issues": []})
        assert result["ok"] is False
        assert result["violations"][0]["rule"] == "W3_unchanged"

    def test_llm_error_does_not_raise(self):
        dialogue = _dialogue()
        result = make_rewrite_segment(_StubClient(error=RuntimeError("boom")))(
            None, {"dialogue": dialogue, "turn": _first_assistant(dialogue),
                   "issues": []})
        assert result["ok"] is False
        assert result["violations"][0]["rule"] == "W4_llm_error"

    def test_accepts_fenced_json(self):
        dialogue = _dialogue()
        text = "```json\n" + json.dumps({"content": FIXED}, ensure_ascii=False) + "\n```"
        result = make_rewrite_segment(_StubClient(text=text))(
            None, {"dialogue": dialogue, "turn": _first_assistant(dialogue),
                   "issues": []})
        assert result["ok"] is True
        assert result["dialogue"]["messages"][_first_assistant(dialogue)]["content"] == FIXED
