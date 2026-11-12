"""工具注册表

与主工程 StepRegistry 同构：注册即启用，按名分发。
工具必须纯函数化（无 I/O、无网络），以保证可脱离 LLM 单测。
"""

from typing import Any, Callable, Dict, List, Optional


class ToolContractError(TypeError):
    """工具不符合契约。"""


class ToolRegistry:
    """类级注册表。测试中用 clear()/snapshot 隔离。"""

    _tools: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 注册
    @classmethod
    def register(cls, tool: Any) -> None:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ToolContractError(f"工具缺少合法 name: {tool!r}")
        if not isinstance(getattr(tool, "description", None), str):
            raise ToolContractError(f"工具 {name} 缺少 description")
        schema = getattr(tool, "schema", None)
        if not isinstance(schema, dict):
            raise ToolContractError(f"工具 {name} 的 schema 必须是 dict")
        if not callable(tool):
            raise ToolContractError(f"工具 {name} 必须可调用")
        if name in cls._tools:
            raise ToolContractError(f"工具重复注册: {name}")
        cls._tools[name] = tool

    # ------------------------------------------------------------------ 查询
    @classmethod
    def get(cls, name: str) -> Any:
        tool = cls._tools.get(name)
        if tool is None:
            raise KeyError(f"未注册的工具: {name}")
        return tool

    @classmethod
    def list_tools(cls) -> List[str]:
        return sorted(cls._tools.keys())

    @classmethod
    def specs(cls) -> List[Dict[str, Any]]:
        """转为 OpenAI function-calling 的 tools 格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in cls._tools.values()
        ]

    # ------------------------------------------------------------------ 调用
    @classmethod
    def invoke(cls, name: str, state: Any = None, args: Optional[Dict] = None) -> Dict:
        tool = cls.get(name)
        result = tool(state, args or {})
        if not isinstance(result, dict):
            raise ToolContractError(f"工具 {name} 必须返回 dict，实际: {type(result)}")
        return result

    # ------------------------------------------------------------------ 测试支持
    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()

    @classmethod
    def snapshot(cls) -> Dict[str, Any]:
        return dict(cls._tools)

    @classmethod
    def restore(cls, snap: Dict[str, Any]) -> None:
        cls._tools = dict(snap)


class FunctionTool:
    """把普通函数包装成工具的最小实现。

    用法：
        tool = FunctionTool(
            name="check_consistency",
            description="校验重写前后结构是否一致",
            schema={"type": "object", "properties": {...}},
            fn=my_func,
        )
    """

    def __init__(self, name: str, description: str, schema: dict,
                 fn: Callable[[Any, Dict], Dict]):
        self.name = name
        self.description = description
        self.schema = schema
        self._fn = fn

    def __call__(self, state: Any, args: Dict) -> Dict:
        return self._fn(state, args)

    def __repr__(self) -> str:
        return f"<FunctionTool {self.name}>"
