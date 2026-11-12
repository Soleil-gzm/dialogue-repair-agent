"""测试：check_coherence 工具

工具层只做参数校验与封装；规则正确性见 test_coherence。
"""

from agent.runtime.registry import ToolRegistry
from agent.tools import DEFAULT_TOOLS, CHECK_COHERENCE, register_all
from agent.tools.check_coherence import make_check_coherence


def _dialogue(*contents):
    return {"messages": [{"role": "assistant", "content": c} for c in contents]}


class TestRegistration:
    def test_registered_in_default_tools(self):
        names = [t.name for t in DEFAULT_TOOLS]
        assert "check_coherence" in names

    def test_register_all_includes_it(self):
        names = register_all()
        assert "check_coherence" in names
        assert "check_coherence" in ToolRegistry.list_tools()

    def test_spec_exposes_dialogue_only(self):
        register_all()
        specs = {s["function"]["name"]: s for s in ToolRegistry.specs()}
        params = specs["check_coherence"]["function"]["parameters"]
        assert set(params["required"]) == {"dialogue"}


class TestInvoke:
    def test_clean_dialogue_ok(self):
        register_all()
        out = ToolRegistry.invoke(
            "check_coherence",
            args={"dialogue": _dialogue("已为您查询，订单正在分拣中。")},
        )
        assert out["ok"] is True
        assert out["violation_count"] == 0
        assert out["violations"] == []

    def test_repetition_detected(self):
        register_all()
        out = ToolRegistry.invoke(
            "check_coherence",
            args={"dialogue": _dialogue("同一句回复。", "同一句回复。")},
        )
        assert out["ok"] is False
        assert out["violations"][0]["rule"] == "C1_assistant_repetition"

    def test_bad_input_contract(self):
        register_all()
        out = ToolRegistry.invoke("check_coherence", args={"dialogue": "oops"})
        assert out["ok"] is False
        assert out["violations"][0]["rule"] == "C0_input_contract"


class TestJudgeInjection:
    def test_factory_accepts_judge(self):
        tool = make_check_coherence(
            judge=lambda _d: [{"rule": "C3_off_topic", "detail": "跑题", "turn": 0}]
        )
        out = tool(None, {"dialogue": _dialogue("今天天气不错。")})
        assert out["ok"] is False
        assert out["violations"][0]["rule"] == "C3_off_topic"

    def test_default_tool_has_no_judge(self):
        """开源默认档位必须完全离线，不能偷偷带语义判官。"""
        out = CHECK_COHERENCE(None, {"dialogue": _dialogue("今天天气不错。")})
        assert out["ok"] is True
