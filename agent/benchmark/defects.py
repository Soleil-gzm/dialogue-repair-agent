"""程序化缺陷注入（L1 的 ground truth 来源）

为什么不用人工标注：注入即标注，缺陷位置与类型天然确定，可复算、可复现。
同 seed 必然产出同一批数据。

六类缺陷：
    D1 答非所问   —— 某轮 assistant 换成与当轮诉求无关的跨话题回复
    D2 指代断裂   —— 删掉前文实体，只留"那个/它"
    D3 事实冲突   —— assistant 引入与 system 不符的金额（凭空新增实体）
    D4 角色混淆   —— 把用户话术塞进 assistant
    D5 重复空转   —— 当前 assistant 复制上一条 assistant
    D6 缺失承接   —— 删掉一轮 assistant 回复

D1 的口径：必须"语义跑题"，而不是"回复换了来源"。早先版本从另一条对话里搬
同域通用话术，替换后往往仍然说得通（"催一下"→"我这边同步处理"），
标签本身就带噪声。现在改为替换为跨话题回复池，保证真的答非所问。

用法：
    python -m agent.benchmark.defects --per-type 20 --seed 20261113 --out out.jsonl
"""

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

DEFECT_TYPES = [
    "D1_off_topic",
    "D2_reference_broken",
    "D3_fact_conflict",
    "D4_role_confusion",
    "D5_repetition",
    "D6_missing_response",
]

_ID_PATTERN = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")

# 跨话题回复池：与本域所有客户诉求（物流/换货/催办/破损/时间）都无承接关系。
# 固定内容 + seed 轮转，保证 D1 标签可复现、语义明确跑题。
OFF_TOPIC_REPLIES = [
    "我们的营业时间是每天上午九点到下午六点。",
    "会员积分可以在下次下单时直接抵扣。",
    "您可以在门店出示二维码领取赠品。",
    "这款产品的说明书有中英文两个版本。",
    "请在十五分钟内完成支付，否则订单会自动关闭。",
    "我们的仓库已经通过了安全认证。",
]


def _assistant_indices(msgs: List[dict]) -> List[int]:
    return [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]


def _user_indices(msgs: List[dict]) -> List[int]:
    return [i for i, m in enumerate(msgs) if m.get("role") == "user"]


# --------------------------------------------------------------------------
# 单类注入器：返回 (defective_dialogue, meta)，均不修改原对象
# --------------------------------------------------------------------------

def inject_d1_off_topic(d: dict, rng: random.Random) -> Tuple[dict, dict]:
    """把某轮 assistant 换成跨话题回复：语义上确定不回应当轮诉求。"""
    out = copy.deepcopy(d)
    a_idx = _assistant_indices(out["messages"])
    if not a_idx:
        return out, {"type": "D1_off_topic", "turn": None, "skipped": True}
    turn = a_idx[rng.randrange(len(a_idx))]
    # 排除与原文相同的内容，确保注入不是空操作
    pool = [t for t in OFF_TOPIC_REPLIES if t != out["messages"][turn].get("content")]
    if not pool:
        return out, {"type": "D1_off_topic", "turn": None, "skipped": True}
    content = pool[rng.randrange(len(pool))]
    out["messages"][turn] = {**out["messages"][turn], "content": content}
    return out, {"type": "D1_off_topic", "turn": turn, "field": "content",
                 "injected": content}


def inject_d2_reference_broken(d: dict, rng: random.Random) -> Tuple[dict, dict]:
    out = copy.deepcopy(d)
    candidates = [i for i in _user_indices(out["messages"])
                  if _ID_PATTERN.search(out["messages"][i].get("content", ""))]
    if not candidates:
        return out, {"type": "D2_reference_broken", "turn": None, "skipped": True}
    turn = candidates[rng.randrange(len(candidates))]
    content = out["messages"][turn]["content"]
    out["messages"][turn] = {**out["messages"][turn],
                             "content": _ID_PATTERN.sub("那个", content)}
    return out, {"type": "D2_reference_broken", "turn": turn, "field": "content"}


def inject_d3_fact_conflict(d: dict, rng: random.Random) -> Tuple[dict, dict]:
    """在 assistant 中引入与 system 不符的金额：凭空新增实体。"""
    out = copy.deepcopy(d)
    a_idx = _assistant_indices(out["messages"])
    if not a_idx:
        return out, {"type": "D3_fact_conflict", "turn": None, "skipped": True}
    turn = a_idx[rng.randrange(len(a_idx))]
    fake = f"¥{rng.randint(100, 9999)}.{rng.randint(0, 99):02d}"
    msg = out["messages"][turn]
    out["messages"][turn] = {**msg, "content": f"{msg.get('content', '')}本期应还 {fake}。"}
    return out, {"type": "D3_fact_conflict", "turn": turn,
                 "field": "amount", "injected": fake}


def inject_d4_role_confusion(d: dict, rng: random.Random) -> Tuple[dict, dict]:
    out = copy.deepcopy(d)
    msgs = out["messages"]
    # 取一对 (user, 其后的 assistant)，且内容必须不同，否则注入是空操作
    pairs = []
    for u in _user_indices(msgs):
        later = [i for i in _assistant_indices(msgs) if i > u]
        if not later:
            continue
        a = later[0]
        if msgs[u].get("content") != msgs[a].get("content"):
            pairs.append((u, a))
    if not pairs:
        return out, {"type": "D4_role_confusion", "turn": None, "skipped": True}
    u, turn = pairs[rng.randrange(len(pairs))]
    out["messages"][turn] = {**msgs[turn], "content": msgs[u].get("content", "")}
    return out, {"type": "D4_role_confusion", "turn": turn, "field": "content"}


def inject_d5_repetition(d: dict, rng: random.Random) -> Tuple[dict, dict]:
    out = copy.deepcopy(d)
    a_idx = _assistant_indices(out["messages"])
    # 相邻两条 assistant 内容必须不同，复制才有意义
    pairs = [
        (a_idx[k - 1], a_idx[k]) for k in range(1, len(a_idx))
        if out["messages"][a_idx[k - 1]].get("content")
        != out["messages"][a_idx[k]].get("content")
    ]
    if not pairs:
        return out, {"type": "D5_repetition", "turn": None, "skipped": True}
    prev, cur = pairs[rng.randrange(len(pairs))]
    out["messages"][cur] = {**out["messages"][cur],
                            "content": out["messages"][prev].get("content", "")}
    return out, {"type": "D5_repetition", "turn": cur, "field": "content"}


def inject_d6_missing_response(d: dict, rng: random.Random) -> Tuple[dict, dict]:
    out = copy.deepcopy(d)
    a_idx = _assistant_indices(out["messages"])
    if not a_idx:
        return out, {"type": "D6_missing_response", "turn": None, "skipped": True}
    turn = a_idx[rng.randrange(len(a_idx))]
    removed = out["messages"].pop(turn)
    return out, {"type": "D6_missing_response", "turn": turn,
                 "field": "message", "removed_role": removed.get("role")}


_INJECTORS = {
    "D1_off_topic": inject_d1_off_topic,
    "D2_reference_broken": inject_d2_reference_broken,
    "D3_fact_conflict": inject_d3_fact_conflict,
    "D4_role_confusion": inject_d4_role_confusion,
    "D5_repetition": inject_d5_repetition,
    "D6_missing_response": inject_d6_missing_response,
}


def inject(d: dict, defect_type: str, rng: random.Random) -> Tuple[dict, dict]:
    if defect_type not in _INJECTORS:
        raise ValueError(f"未知缺陷类型: {defect_type}")
    return _INJECTORS[defect_type](d, rng)


def generate(dialogues: List[dict], per_type: int = 20,
             seed: int = 20261113) -> List[dict]:
    """按类型均匀生成缺陷样本，返回带标注的记录列表。"""
    if not dialogues:
        raise ValueError("至少需要 1 条对话")
    rng = random.Random(seed)
    records: List[dict] = []
    for t_i, dtype in enumerate(DEFECT_TYPES):
        for k in range(per_type):
            base_i = (t_i * per_type + k) % len(dialogues)
            d = dialogues[base_i]
            bad, meta = inject(d, dtype, rng)
            records.append({
                "sample_key": f"{dtype}-{k:03d}",
                "dialogue": bad,
                "defect": meta,
                "base_index": base_i,
            })
    return records


def to_jsonl(records: List[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.defects")
    ap.add_argument("--input", default="agent/fixtures/public/base_v1.json")
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="agent/fixtures/public/defects_v1.jsonl")
    args = ap.parse_args(argv)

    src = Path(args.input)
    dialogues = json.loads(src.read_text(encoding="utf-8"))
    records = generate(dialogues, per_type=args.per_type, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_jsonl(records), encoding="utf-8")

    skipped = sum(1 for r in records if r["defect"].get("skipped"))
    print(f"[OK] {len(records)} 条 -> {out}")
    print(f"     其中因样本不满足条件跳过注入: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
