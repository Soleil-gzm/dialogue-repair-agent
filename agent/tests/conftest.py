"""Agent 子系统测试的共享 fixtures

独立于根目录 tests/：本子系统可单独开源，不依赖主工程的测试设施。
所有数据由 agent.benchmark.fixtures 程序化生成，中立域，无真实业务信息。
"""

import json
from pathlib import Path

import pytest

from agent.benchmark.canonical import sha256_obj
from agent.benchmark.fixtures import generate_dialogues

SNAPSHOT_DIR = Path(__file__).resolve().parent / "__snapshots__"


def pytest_addoption(parser):
    # 幂等注册：与根目录 tests/conftest.py 同时存在时只注册一次
    try:
        parser.addoption(
            "--update-snapshots", action="store_true", default=False,
            help="更新快照指纹而非比对",
        )
    except ValueError:
        pass


@pytest.fixture
def update_snapshots(request):
    return request.config.getoption("--update-snapshots")


# ---------------------------------------------------------------------------
# 全局隔离
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_tool_registry():
    """每个用例前后恢复工具注册表。

    ToolRegistry 是类级共享状态：某个用例注册的工具会泄漏到后续用例，
    表现为"单跑通过、全量失败"。统一在此隔离，避免各测试文件各自实现。
    """
    from agent.runtime.registry import ToolRegistry

    snap = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(snap)


@pytest.fixture
def mini_dialogues():
    """3 条合成对话（中立域，可公开）。"""
    return generate_dialogues(3, seed=1)


@pytest.fixture
def raw_dialogues_file(tmp_path, mini_dialogues):
    p = tmp_path / "dialogues.json"
    p.write_text(json.dumps(mini_dialogues, ensure_ascii=False), encoding="utf-8")
    return p


def assert_snapshot(name: str, obj, update: bool) -> None:
    """比对（或更新）对象指纹快照。"""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.sha256"
    digest = sha256_obj(obj)
    if update or not path.exists():
        path.write_text(digest + "\n", encoding="utf-8")  # 纯文本，不带引号
        pytest.skip(f"快照已更新: {name}")
    expected = path.read_text(encoding="utf-8").strip()
    assert digest == expected, f"快照不匹配: {name}"
