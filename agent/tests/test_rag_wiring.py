"""RAG 接线测试：检索内容必须真正进入改写 prompt，且检索失败不能阻断改写。"""

import json

from agent.runtime.llm import LLMResponse
from agent.tools.rewrite_segment import make_rewrite_segment

FIXED = "您希望延期几天？我帮您看看政策。"


class _StubClient:
    def __init__(self, text: str = "", tokens: int = 42):
        self.text = text or json.dumps({"content": FIXED}, ensure_ascii=False)
        self.tokens = tokens
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(text=self.text, input_tokens=self.tokens, output_tokens=0)


class _SpyAugmenter:
    """记录检索调用；boom=True 时模拟检索故障。"""

    def __init__(self, boom: bool = False):
        self.calls = []
        self.boom = boom

    def query(self, user_content, assistant_content, top_k=3):
        self.calls.append({
            "user": user_content,
            "assistant": assistant_content,
            "top_k": top_k,
        })
        if self.boom:
            raise RuntimeError("vector store down")
        return [{
            "score": 0.9,
            "meta": {
                "defect_type": "D1_off_topic",
                "is_off_topic": True,
                "user_content": user_content,
                "assistant_content": assistant_content,
                "repair_after": "您希望延期几天？",
            },
        }]

    def render_for_prompt(self, hits):
        return "RAG_FIXTURE_MARKER 历史相似修复案例"


def _dialogue():
    return {"messages": [
        {"role": "system", "content": "你是客服助手。"},
        {"role": "user", "content": "我想延期还款"},
        {"role": "assistant", "content": "好的，请提供您的姓名。"},
    ]}


def _call(tool):
    return tool(None, {"dialogue": _dialogue(), "turn": 2, "issues": []})


class TestRagWiring:
    def test_retrieval_context_enters_prompt(self):
        client = _StubClient()
        aug = _SpyAugmenter()
        res = _call(make_rewrite_segment(client, rag_augmenter=aug))

        assert res["ok"] is True
        assert res["rag_used"] is True
        # 检索真的被调用了一次，且带上上一句 user 与待改写回复
        assert len(aug.calls) == 1
        assert aug.calls[0]["user"] == "我想延期还款"
        assert aug.calls[0]["assistant"] == "好的，请提供您的姓名。"
        # 检索内容确实进了发给模型的 prompt
        joined = "\n".join(m["content"] for m in client.calls[0]["messages"])
        assert "RAG_FIXTURE_MARKER" in joined

    def test_retrieval_failure_degrades_gracefully(self):
        client = _StubClient()
        res = _call(make_rewrite_segment(client, rag_augmenter=_SpyAugmenter(boom=True)))

        assert res["ok"] is True          # 改写仍然成功
        assert res["rag_used"] is False   # 但标记未增强
        joined = "\n".join(m["content"] for m in client.calls[0]["messages"])
        assert "RAG_FIXTURE_MARKER" not in joined

    def test_no_augmenter_keeps_behavior(self):
        res = _call(make_rewrite_segment(_StubClient()))
        assert res["ok"] is True
        assert res["rag_used"] is False
