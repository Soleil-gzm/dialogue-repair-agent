"""开源边界测试（可执行的自证）

三个断言：
1. 清单覆盖全部文件 —— 新增文件必须显式决定公开与否，不能"默认公开"。
2. PUBLIC 文件不含业务标识 —— 防止脱敏遗漏。
3. PUBLIC 的 Python 文件不顶层依赖主工程 —— 保证可脱离主工程安装。
   （A1 臂例外：它在函数体内懒加载 pipeline.*，属对照基线，允许但必须保持懒加载。）
"""

import ast
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = AGENT_ROOT / "oss_manifest.txt"
DENYLIST = AGENT_ROOT / "oss_denylist.txt"

# 主工程模块前缀
FORBIDDEN_IMPORT_ROOTS = {"pipeline", "configs", "steps"}

# 允许懒加载主工程的模块（仅 A1 臂）
LAZY_IMPORT_ALLOWED = {"benchmark/arms.py"}

# 不参与清单比对的内容（清单自身无法自我列举；生成物目录不纳入版本管理）
IGNORED_DIRS = {"__pycache__", "output", ".pytest_cache", ".export_check"}
IGNORED_NAMES = {"oss_manifest.txt"}

# 扫描器自身必然包含黑名单词，需豁免（否则自指误报）
SCANNER_SELF = "tests/test_oss_boundary.py"


def _load_manifest():
    """返回 (public, nonpublic)。BUILD 与 PRIVATE 均不发布。"""
    public, nonpublic, section = [], [], None
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            upper = line.upper()
            if "PUBLIC" in upper:
                section = "public"
            elif "PRIVATE" in upper or "BUILD" in upper:
                section = "nonpublic"
            continue
        (public if section == "public" else nonpublic).append(line)
    return public, nonpublic


def _actual_files():
    out = []
    for p in AGENT_ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(AGENT_ROOT)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if p.name in IGNORED_NAMES:
            continue
        out.append(rel.as_posix())
    return sorted(out)


def test_manifest_covers_all_files():
    public, private = _load_manifest()
    listed = set(public) | set(private)
    actual = set(_actual_files())

    # 新增文件必须登记（防止"默认公开"）
    missing = actual - listed
    assert not missing, f"以下文件未登记到 oss_manifest.txt: {sorted(missing)}"

    # PUBLIC 必须存在；PRIVATE 在已开源环境中本来就被剔除，允许缺失
    absent_public = [rel for rel in public if rel not in actual]
    assert not absent_public, f"清单中的 PUBLIC 文件不存在: {sorted(absent_public)}"


def _load_denylist():
    """黑名单为私有文件，代码中不留明文；缺失时（如已开源仓库）跳过该项校验。"""
    if not DENYLIST.exists():
        return None
    return [
        l.strip() for l in DENYLIST.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


def test_public_files_have_no_business_terms():
    terms = _load_denylist()
    if terms is None:
        pytest.skip("黑名单为私有文件，已开源环境不适用")

    public, _ = _load_manifest()
    hits = []
    for rel in public:
        if rel == SCANNER_SELF:
            continue
        p = AGENT_ROOT / rel
        if p.suffix.lower() not in {".py", ".md", ".json", ".txt", ".sha256"}:
            continue
        text = p.read_text(encoding="utf-8")
        for term in terms:
            if term in text:
                hits.append(f"{rel} 含 {term}")
    assert not hits, "PUBLIC 文件命中业务标识: " + "; ".join(hits)


def test_public_python_does_not_import_main_project_at_top_level():
    public, _ = _load_manifest()
    violations = []
    for rel in public:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((AGENT_ROOT / rel).read_text(encoding="utf-8"))
        for node in tree.body:  # 只看模块顶层
            roots = []
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".")[0]]
            for r in roots:
                if r in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(f"{rel} 顶层 import {r}")
    assert not violations, "; ".join(violations)


@pytest.mark.parametrize("rel", sorted(LAZY_IMPORT_ALLOWED))
def test_lazy_import_module_still_importable(rel):
    """允许懒加载的模块，其顶层仍必须可独立导入。"""
    src = (AGENT_ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS, \
                    f"{rel} 顶层不应 import {n}（应保持函数内懒加载）"
