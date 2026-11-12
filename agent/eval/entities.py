"""实体抽取（领域无关）

只用正则与结构约定，不依赖任何业务词表，因此可随开源仓库分发。

两类抽取：
- extract_tag_values()  从 system 的 `<标签>值` / `## 标签\\n值` 结构取键值
- extract_entities()    从自由文本中取编号、电话、金额、日期
"""

import re
from typing import Dict, List

# 编号：两到六位大写字母 + 短横 + 数字，如 ORD-123456 / CUST-0001
_RE_ID = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")
# 电话：400-xxx-xxxx / 区号-号码 / 手机号
_RE_PHONE = re.compile(r"\b(?:400-\d{3}-\d{4}|0\d{2,3}-\d{7,8}|1[3-9]\d{9})\b")
# 金额：¥123.45 或 123.45 元
_RE_AMOUNT = re.compile(r"(?:¥\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*\.\d{1,2}\s?元)")
# 日期：2026-11-12 / 2026/11/12 / 11月12日
_RE_DATE = re.compile(r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}月\d{1,2}日)")

_PATTERNS = {
    "id": _RE_ID,
    "phone": _RE_PHONE,
    "amount": _RE_AMOUNT,
    "date": _RE_DATE,
}


def extract_entities(text: str) -> List[str]:
    """返回文本中出现的实体字符串（去重后排序，便于稳定比对）。"""
    if not isinstance(text, str) or not text:
        return []
    found = set()
    for pattern in _PATTERNS.values():
        for m in pattern.findall(text):
            found.add(_normalize(m))
    return sorted(found)


def _normalize(s: str) -> str:
    """只做空白归一，不做数字/大小写改写，保证"原样保真"的判定基准。"""
    return re.sub(r"\s+", "", s)


def extract_entities_from_messages(dialogue: dict) -> List[str]:
    """抽取除 system 之外所有消息中的实体（system 由 R1 单独保证逐字不变）。"""
    out = set()
    for msg in dialogue.get("messages", []) or []:
        if msg.get("role") == "system":
            continue
        out.update(extract_entities(msg.get("content", "")))
    return sorted(out)


def extract_tag_values(system_text: str) -> Dict[str, str]:
    """解析 system 中的标签值，兼容两种写法：

        <标签>值
        ## 标签
        值
    """
    values: Dict[str, str] = {}
    if not isinstance(system_text, str):
        return values

    # 写法一：<标签>值
    for tag, val in re.findall(r"<([^<>\n]+)>([^<\n]*)", system_text):
        v = val.strip()
        if v:
            values[tag.strip()] = v

    # 写法二：## 标签 换行 值
    lines = [l.strip() for l in system_text.splitlines()]
    for i, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        tag = line[3:].strip()
        for nxt in lines[i + 1:]:
            if not nxt or nxt.startswith("#"):
                break
            values.setdefault(tag, nxt)
            break
    return values


def system_text(dialogue: dict) -> str:
    """取首条 system 内容；没有则返回空串。"""
    for msg in dialogue.get("messages", []) or []:
        if msg.get("role") == "system":
            return msg.get("content", "") or ""
    return ""
