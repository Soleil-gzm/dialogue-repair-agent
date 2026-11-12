"""四类硬约束校验（纯规则，无 LLM）

返回违规列表；空列表表示通过。每条违规都是可 JSON 序列化的 dict，
便于直接写进 trace 与 benchmark 报告。
"""

from collections import Counter
from typing import Dict, List

from .entities import extract_entities_from_messages, system_text

R1_SYSTEM_IMMUTABLE = "R1_system_immutable"
R2_STRUCTURE_INVARIANT = "R2_structure_invariant"
R3_ENTITY_FIDELITY = "R3_entity_fidelity"
R4_SPEAKER_SET = "R4_speaker_set"
R5_ROLE_CONTENT_LEAK = "R5_role_content_leak"

ALL_RULES = (R1_SYSTEM_IMMUTABLE, R2_STRUCTURE_INVARIANT,
             R3_ENTITY_FIDELITY, R4_SPEAKER_SET, R5_ROLE_CONTENT_LEAK)


def _violation(rule: str, detail: str, turn=None) -> Dict:
    return {"rule": rule, "detail": detail, "turn": turn}


def check_consistency(before: dict, after: dict) -> List[Dict]:
    """校验 `after` 相对 `before` 是否满足四类硬约束。"""
    violations: List[Dict] = []
    b_msgs = list((before or {}).get("messages") or [])
    a_msgs = list((after or {}).get("messages") or [])

    # ---------------- R1：system 逐字不变 ----------------
    b_sys, a_sys = system_text(before or {}), system_text(after or {})
    if b_sys != a_sys:
        violations.append(_violation(
            R1_SYSTEM_IMMUTABLE, "system 内容被修改", turn=0,
        ))

    # ---------------- R2：结构不变 ----------------
    if len(b_msgs) != len(a_msgs):
        violations.append(_violation(
            R2_STRUCTURE_INVARIANT,
            f"消息数变化: {len(b_msgs)} -> {len(a_msgs)}",
        ))
    for i in range(min(len(b_msgs), len(a_msgs))):
        b, a = b_msgs[i], a_msgs[i]
        if b.get("role") != a.get("role"):
            violations.append(_violation(
                R2_STRUCTURE_INVARIANT,
                f"role 变化: {b.get('role')!r} -> {a.get('role')!r}", turn=i,
            ))
        # loss 三态严格比对：缺失 != "False"
        if b.get("loss") != a.get("loss"):
            violations.append(_violation(
                R2_STRUCTURE_INVARIANT,
                f"loss 变化: {b.get('loss')!r} -> {a.get('loss')!r}", turn=i,
            ))

    # ---------------- R4：说话人集合 ----------------
    b_roles = Counter(m.get("role") for m in b_msgs)
    a_roles = Counter(m.get("role") for m in a_msgs)
    if b_roles != a_roles:
        violations.append(_violation(
            R4_SPEAKER_SET,
            f"说话人分布变化: {dict(b_roles)} -> {dict(a_roles)}",
        ))

    # ---------------- R3：实体完全保真 ----------------
    b_ents = set(extract_entities_from_messages(before or {}))
    a_ents = set(extract_entities_from_messages(after or {}))
    for e in sorted(b_ents - a_ents):
        violations.append(_violation(
            R3_ENTITY_FIDELITY, f"实体丢失: {e}",
        ))
    for e in sorted(a_ents - b_ents):
        violations.append(_violation(
            R3_ENTITY_FIDELITY, f"凭空新增实体: {e}",
        ))

    # ---------------- R5：跨角色内容泄漏 ----------------
    # 针对"把客户话术塞进 assistant"：实体集合可能不变（同一句话仍在 user 里），
    # 只靠 R3 抓不到，因此单独比较"assistant 是否复读了 user 的内容"。
    a_user_contents = {m.get("content") for m in a_msgs if m.get("role") == "user"}
    for i in range(min(len(b_msgs), len(a_msgs))):
        if a_msgs[i].get("role") != "assistant":
            continue
        content = a_msgs[i].get("content")
        if content in a_user_contents and content != b_msgs[i].get("content"):
            violations.append(_violation(
                R5_ROLE_CONTENT_LEAK,
                "assistant 复读了 user 的内容（角色混淆）", turn=i,
            ))

    return violations


def is_consistent(before: dict, after: dict) -> bool:
    return not check_consistency(before, after)
