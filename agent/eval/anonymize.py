"""对话脱敏：把真实业务标识替换为占位符，供人工标注材料导出使用。

设计原则（与 ADR D6 私有轨合规边界一致）：
1. 代码本身不含任何真实业务数据；机构/产品名词表通过环境变量注入，
   不写死在仓库里。
2. 复用 agent/eval/entities.py 的正则作为唯一入口，不重复实现。
3. 脱敏是单向、可复现的：同一输入 + 同一环境配置 → 同一输出，
   不做随机化，便于标注结果与原始样本回溯对应。

脱敏层级（按可靠性从高到低）：
- 标签值映射：从 system 的 `<标签>值` 结构精确取值，按标签语义给占位符。
- 机构词表：环境变量 ANONYMIZE_ORG_TERMS 提供，长词优先替换。
- 正则兜底：电话 / 日期 / 金额 / 编号，覆盖自由文本中残留的实体。
"""

import os
import re
from typing import Dict, List, Optional

from .entities import _PATTERNS, extract_tag_values, system_text

# 纯小数（如 0.00 / 6317.08）兜底判为金额；entities 的 _RE_AMOUNT 要求
# ¥ 前缀或「元」后缀，不适配标签里的裸小数值。
_RE_DECIMAL = re.compile(r"^\d+\.\d+$")
# 纯整数（不含小数点），用于区分「数字」与「小数金额」
_RE_INT = re.compile(r"^\d+$")
# 全文兜底电话：不用 \b——Python3 re 默认为 Unicode 模式，中文字符属于 \w，
# 会让「中文与数字之间」的边界判断失效，导致正文里的手机号漏匹配。
# 改用「前后非数字」lookaround；额外覆盖 95 开头的客服号（entities 未覆盖）。
_RE_SAFE_PHONE = re.compile(
    r"(?<![\d])(?:400-\d{3}-\d{4}|0\d{2,3}-\d{7,8}|95\d{3,}|1[3-9]\d{9})(?![\d])"
)

# 标签名 → 占位符。标签名属于数据 schema（不是业务数据），可入库。
# 未列出的标签走「值类型推断」兜底。
_TAG_PLACEHOLDER: Dict[str, str] = {
    "姓名": "<客户>", "客户姓名": "<客户>", "name": "<客户>",
    "性别": "<性别>",
    "客服电话": "<电话>", "电话": "<电话>", "联系方式": "<电话>",
    "还款日": "<日期>", "今天日期": "<日期>", "日期": "<日期>",
    "查账时间": "<时间>", "当前时间": "<时间>",
    "逾期天数": "<数字>", "逾期笔数": "<数字>",
    "总欠款": "<金额>", "应还金额": "<金额>", "金额": "<金额>",
    "专员工号": "<工号>", "工号": "<工号>",
    "机构": "<机构A>", "机构名称": "<机构A>",
    "业务类型": "<业务类型>",
}


def _placeholder_for_value(tag: str, val: str) -> str:
    """根据标签名或值的实体类型推断占位符。"""
    if tag in _TAG_PLACEHOLDER:
        return _TAG_PLACEHOLDER[tag]
    if _PATTERNS["phone"].search(val):
        return "<电话>"
    if _PATTERNS["date"].search(val):
        return "<日期>"
    if _PATTERNS["amount"].search(val):
        return "<金额>"
    if _RE_DECIMAL.match(val):
        # 标签里的裸小数（0.00 / 6317.08）几乎都是金额，兜底判金额
        return "<金额>"
    if _PATTERNS["id"].search(val):
        return "<编号>"
    if _RE_INT.match(val):
        return "<数字>"
    return "<脱敏值>"


def _load_org_terms() -> List[str]:
    """从环境变量读取机构/产品名词表（逗号分隔），代码不硬编码业务名。"""
    raw = os.environ.get("ANONYMIZE_ORG_TERMS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _is_entity_like(val: str) -> bool:
    """值是否像一个可脱敏的实体（电话/日期/金额/小数/整数/编号）。"""
    return bool(
        _PATTERNS["phone"].search(val)
        or _PATTERNS["date"].search(val)
        or _PATTERNS["amount"].search(val)
        or _RE_DECIMAL.match(val)
        or _PATTERNS["id"].search(val)
        or _RE_INT.match(val)
    )


def _build_value_map(dialogue: dict) -> Dict[str, str]:
    """从 system 标签值构建「敏感值 → 占位符」映射。"""
    sys_text = system_text(dialogue)
    value_map: Dict[str, str] = {}

    # 写法一/二：<标签>值 与 ## 标签\n值（extract_tag_values 已兼容）
    for tag, val in extract_tag_values(sys_text).items():
        v = (val or "").strip()
        # 跳过非原子值：「## 标签信息」分组标题会被解析成标签，其值是嵌套的
        # <标签>值 串（含尖括号），不是可直接替换的原子实体，跳过以免长串
        # 优先覆盖真实标签值。
        if not v or "<" in v or ">" in v:
            continue
        value_map[v] = _placeholder_for_value(tag, v)

    # 写法三：「标签：值」冒号格式（extract_tag_values 不识别）。
    # 仅纳入实体类型值，避免把正文里的「目标：卡人」等非实体键值误伤。
    for line in sys_text.splitlines():
        m = re.match(r"^\s*([^：:\s]{2,12})\s*[:：]\s*(\S+)\s*$", line)
        if not m:
            continue
        tag, v = m.group(1).strip(), m.group(2).strip()
        if not v or "<" in v or ">" in v or not _is_entity_like(v):
            continue
        value_map.setdefault(v, _placeholder_for_value(tag, v))

    return value_map


def anonymize_text(text: str, value_map: Dict[str, str],
                   org_terms: Optional[List[str]] = None) -> str:
    """对单段文本脱敏：机构词 → 标签值 → 正则兜底。

    纯数字值必须用「前后非数字/小数点」的边界匹配，否则短数字（如逾期天数
    56、笔数 0）会作为子串误替换更长的数字（如工号 2344756、金额 6317.08）。
    不用 \b 是因为 Python3 re 默认为 Unicode 模式，中文字符属于 \w，
    会让「中文与数字之间」的边界判断失效。
    """
    if not isinstance(text, str) or not text:
        return text
    if org_terms:
        for term in sorted(org_terms, key=len, reverse=True):
            text = text.replace(term, "<机构A>")
    numeric, textual = [], []
    for val, ph in value_map.items():
        if _RE_INT.match(val) or _RE_DECIMAL.match(val):
            numeric.append((val, ph))
        else:
            textual.append((val, ph))
    # 非数字值：str.replace，长串优先避免部分匹配
    for val, ph in sorted(textual, key=lambda x: len(x[0]), reverse=True):
        text = text.replace(val, ph)
    # 数字值：边界正则，长串优先
    for val, ph in sorted(numeric, key=lambda x: len(x[0]), reverse=True):
        text = re.sub(rf"(?<![\d.]){re.escape(val)}(?![\d.])", ph, text)
    text = _RE_SAFE_PHONE.sub("<电话>", text)
    text = _PATTERNS["date"].sub("<日期>", text)
    text = _PATTERNS["amount"].sub("<金额>", text)
    text = _PATTERNS["id"].sub("<编号>", text)
    return text


def anonymize_dialogue(dialogue: dict, *,
                       org_terms: Optional[List[str]] = None) -> dict:
    """返回脱敏后的对话副本；不修改入参。保留 messages 的 role 与其他字段。"""
    if org_terms is None:
        org_terms = _load_org_terms()
    value_map = _build_value_map(dialogue)
    new_messages: List[dict] = []
    for msg in dialogue.get("messages", []) or []:
        new_msg = dict(msg)
        if isinstance(new_msg.get("content"), str):
            new_msg["content"] = anonymize_text(new_msg["content"], value_map, org_terms)
        new_messages.append(new_msg)
    result = dict(dialogue)
    result["messages"] = new_messages
    return result
