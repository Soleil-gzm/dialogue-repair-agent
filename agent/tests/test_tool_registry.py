"""测试：工具注册表契约

工具必须可脱离 LLM 单测，因此注册表本身零依赖、纯内存。
"""

import pytest

from agent.runtime.registry import (
    ToolContractError, ToolRegistry, FunctionTool,
)


def _noop(state, args):
    return {"ok": True, "args": args}


@pytest.fixture(autouse=True)
def isolated_registry():
    snap = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(snap)


def _tool(name="check_x", **overrides):
    kw = dict(
        name=name,
        description="示例工具",
        schema={"type": "object", "properties": {}},
        fn=_noop,
    )
    kw.update(overrides)
    return FunctionTool(**kw)


class TestRegister:
    def test_register_and_list(self):
        ToolRegistry.register(_tool())
        assert "check_x" in ToolRegistry.list_tools()

    def test_rejects_empty_name(self):
        with pytest.raises(ToolContractError):
            ToolRegistry.register(_tool(name=""))

    def test_rejects_missing_description(self):
        with pytest.raises(ToolContractError):
            ToolRegistry.register(_tool(description=None))

    def test_rejects_non_dict_schema(self):
        with pytest.raises(ToolContractError):
            ToolRegistry.register(_tool(schema="not-a-dict"))

    def test_rejects_duplicate_name(self):
        ToolRegistry.register(_tool())
        with pytest.raises(ToolContractError):
            ToolRegistry.register(_tool())

    def test_rejects_non_callable(self):
        class NotCallable:
            name = "x"
            description = "d"
            schema = {}
        with pytest.raises(ToolContractError):
            ToolRegistry.register(NotCallable())


class TestSpecs:
    def test_openai_function_format(self):
        ToolRegistry.register(_tool())
        specs = ToolRegistry.specs()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "check_x"
        assert "parameters" in specs[0]["function"]

    def test_empty_registry_returns_empty_list(self):
        assert ToolRegistry.specs() == []


class TestInvoke:
    def test_invoke_passes_state_and_args(self):
        ToolRegistry.register(_tool())
        out = ToolRegistry.invoke("check_x", state={"k": 1}, args={"a": 2})
        assert out == {"ok": True, "args": {"a": 2}}

    def test_invoke_unregistered_raises(self):
        with pytest.raises(KeyError):
            ToolRegistry.invoke("nope")

    def test_rejects_non_dict_return(self):
        ToolRegistry.register(_tool(name="bad", fn=lambda s, a: "not-a-dict"))
        with pytest.raises(ToolContractError):
            ToolRegistry.invoke("bad")
