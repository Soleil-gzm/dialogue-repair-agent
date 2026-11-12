"""测试：check_consistency 工具

工具层只做参数校验与封装，规则正确性由 test_rules 保证。
"""

import copy

import pytest

from agent.runtime.registry import ToolRegistry
from agent.tools import DEFAULT_TOOLS, CHECK_CONSISTENCY, register_all


class TestRegistration:
    def test_register_all(self):
        names = register_all()
        assert "check_consistency" in names
        assert "check_consistency" in ToolRegistry.list_tools()

    def test_register_all_is_idempotent(self):
        register_all()
        assert register_all() == []  # 第二次无新增

    def test_specs_contain_tool(self):
        register_all()
        specs = {s["function"]["name"]: s for s in ToolRegistry.specs()}
        assert "check_consistency" in specs
        params = specs["check_consistency"]["function"]["parameters"]
        assert set(params["required"]) == {"before", "after"}

    def test_default_tools_non_empty(self):
        assert DEFAULT_TOOLS
        assert all(t.name for t in DEFAULT_TOOLS)


class TestInvoke:
    def test_clean_pair_ok(self, mini_dialogues):
        register_all()
        out = ToolRegistry.invoke(
            "check_consistency", args={"before": mini_dialogues[0],
                                       "after": copy.deepcopy(mini_dialogues[0])},
        )
        assert out["ok"] is True
        assert out["violations"] == []
        assert out["violation_count"] == 0

    def test_entity_injection_blocked(self, mini_dialogues):
        register_all()
        bad = copy.deepcopy(mini_dialogues[0])
        bad["messages"][-1]["content"] += "本期应还 ¥8888.00。"
        out = ToolRegistry.invoke(
            "check_consistency", args={"before": mini_dialogues[0], "after": bad},
        )
        assert out["ok"] is False
        assert out["violation_count"] >= 1
        assert all("rule" in v and "detail" in v for v in out["violations"])

    def test_invalid_args_return_contract_error(self):
        out = CHECK_CONSISTENCY(None, {"before": "不是对象", "after": {}})
        assert out["ok"] is False
        assert out["violations"][0]["rule"] == "R0_input_contract"

    def test_missing_args_return_contract_error(self):
        out = CHECK_CONSISTENCY(None, {})
        assert out["ok"] is False

    def test_result_is_json_serializable(self, mini_dialogues):
        import json
        register_all()
        bad = copy.deepcopy(mini_dialogues[0])
        bad["messages"].pop()
        out = ToolRegistry.invoke("check_consistency",
                                  args={"before": mini_dialogues[0], "after": bad})
        json.dumps(out, ensure_ascii=False)  # 不抛即通过
