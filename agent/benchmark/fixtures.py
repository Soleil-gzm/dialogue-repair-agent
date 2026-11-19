"""公开轨 fixture 生成器（中立域）

为什么需要它：Benchmark 的公开轨数据不能来自任何真实业务，必须程序化构造。
本模块只依赖标准库 random，同 seed 必然产出同一份数据。

域设定为「订单售后回访」——与真实业务域无关，仅用于演示与计量。
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from ..runtime.state import dialogue_key

SCENARIOS = ["S1_咨询", "S2_催办", "S3_退换", "S4_投诉", "S5_回访"]
PRODUCTS = ["示例产品甲", "示例产品乙", "示例产品丙"]
CUSTOMER_REPLIES = [
    "我上周下的单，现在还没收到。",
    "这个型号和页面写的不一样。",
    "我想问下能不能换货。",
    "已经等很久了，麻烦帮我催一下。",
    "收到的包装有点破损。",
    "能不能给个大概时间？",
]
AGENT_REPLIES = [
    "已为您查询，订单正在分拣中。",
    "您反馈的问题我记录下来了，会转给对应同事。",
    "换货需要保持包装完整，我帮您登记。",
    "我已为您催办，预计两个工作日内有进展。",
    "很抱歉给您带来不便，我这边同步处理。",
    "稍后会有短信通知，请留意。",
]


def _system(idx: int, scenario: str, product: str, oid: str, rng: random.Random) -> str:
    return (
        "# 角色\n售后回访专员\n"
        f"## 客户编号\nCUST-{idx:04d}\n"
        f"## 订单编号\n{oid}\n"
        f"## 产品名称\n{product}\n"
        f"## 场景标签\n{scenario}\n"
        f"## 客服电话\n400-000-0000\n"
        f"## 联系时间\n2026-11-12 10:00:00\n"
        f"## 订单金额\n{rng.uniform(10, 5000):.2f}\n"
    )


def make_dialogue(idx: int, n_turns: int, rng: random.Random) -> dict:
    scenario = SCENARIOS[idx % len(SCENARIOS)]
    product = PRODUCTS[idx % len(PRODUCTS)]
    oid = f"ORD-{rng.randint(100000, 999999)}"
    messages = [{"role": "system", "content": _system(idx, scenario, product, oid, rng)}]
    for t in range(n_turns):
        text = CUSTOMER_REPLIES[(idx + t) % len(CUSTOMER_REPLIES)]
        # 首轮带上订单号：真实用户会引用单号，也让实体保真校验有可测对象
        if t == 0:
            text = f"{text}（订单号 {oid}）"
        messages.append({"role": "user", "content": text})
        messages.append({
            "role": "assistant",
            "content": AGENT_REPLIES[(idx + t) % len(AGENT_REPLIES)],
            "loss": "True",
        })
    return {"messages": messages}


def generate_dialogues(n: int = 12, seed: int = 20261112) -> List[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        n_turns = rng.choice([3, 4, 5, 6, 7])
        out.append(make_dialogue(i, n_turns, rng))
    return out


# ---------------------------------------------------------------------------
# dev / test 分层切分
# ---------------------------------------------------------------------------

# 与 base_v1 的 seed 分开：dev/test 不应与既有基准数据重叠，否则调 prompt 会污染结论
SPLIT_SEED = 20261118
DEV_N = 30
TEST_N = 50

# 轮次桶：短 / 中 / 长。分层时按桶计数，避免某一长度被切分偏置
TURN_BUCKETS = ((3, 4), (5, 6), (7, 7))
SCENE_MARKER = "## 场景标签"


def _scene_of(dialogue: dict) -> str:
    for msg in dialogue.get("messages") or []:
        if msg.get("role") != "system":
            continue
        text = str(msg.get("content") or "")
        if SCENE_MARKER not in text:
            return "unknown"
        tail = text.split(SCENE_MARKER, 1)[1].lstrip()
        return (tail.splitlines()[0].strip() if tail else "") or "unknown"
    return "unknown"


def _turn_bucket(n_turns: int) -> str:
    for lo, hi in TURN_BUCKETS:
        if lo <= n_turns <= hi:
            return f"{lo}-{hi}"
    return "other"


def _has_loss(dialogue: dict) -> bool:
    for msg in dialogue.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        loss = msg.get("loss")
        if loss is True or (isinstance(loss, str) and loss.lower() == "true"):
            return True
    return False


def _stratum(dialogue: dict) -> Tuple[str, str, bool]:
    msgs = dialogue.get("messages") or []
    n_turns = sum(1 for m in msgs if m.get("role") == "assistant")
    return (_scene_of(dialogue), _turn_bucket(n_turns), _has_loss(dialogue))


def split_dialogues(dialogues: List[dict], dev_n: int = DEV_N,
                    test_n: int = TEST_N) -> Tuple[List[dict], List[dict]]:
    """按「场景 × 轮次桶 × 是否有 loss」分层切分，层内按内容哈希稳定排序。

    单例层直接归 test，保证测试集覆盖每层；其余层用最大余数法分配，
    且强制每层至少留 1 条给 test。输出顺序只由内容哈希决定，可复现。
    """
    items = list(dialogues)
    total = len(items)
    if dev_n < 0 or test_n < 0:
        raise ValueError("dev_n / test_n 不能为负数")
    if dev_n + test_n != total:
        raise ValueError(f"dev_n + test_n 必须等于样本数 {total}，实际 {dev_n + test_n}")
    if len({dialogue_key(d) for d in items}) != total:
        raise ValueError("样本中存在重复对话，无法按内容键切分")

    groups: Dict[Tuple[str, str, bool], List[dict]] = {}
    for d in items:
        groups.setdefault(_stratum(d), []).append(d)
    for group in groups.values():
        group.sort(key=dialogue_key)

    quotas: Dict[Tuple[str, str, bool], int] = {}
    remainders: Dict[Tuple[str, str, bool], float] = {}
    assigned = 0
    for key, group in groups.items():
        size = len(group)
        if size == 1:
            quotas[key], remainders[key] = 0, 0.0
            continue
        quota = size * dev_n / total
        base = min(int(quota), size - 1)
        quotas[key], remainders[key] = base, quota - int(quota)
        assigned += base

    remaining = dev_n - assigned
    ranked = sorted(groups, key=lambda k: (-remainders[k], k))
    while remaining > 0:
        progressed = False
        for key in ranked:
            if quotas[key] < len(groups[key]) - 1:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    if remaining > 0:
        raise ValueError("无法在保证每层至少 1 条 test 的前提下完成 dev 分配")

    dev: List[dict] = []
    test: List[dict] = []
    for key in sorted(groups):
        group = groups[key]
        cut = quotas[key]
        dev.extend(group[:cut])
        test.extend(group[cut:])
    return sorted(dev, key=dialogue_key), sorted(test, key=dialogue_key)


def generate_split(dev_n: int = DEV_N, test_n: int = TEST_N,
                   seed: int = SPLIT_SEED) -> Tuple[List[dict], List[dict]]:
    return split_dialogues(generate_dialogues(dev_n + test_n, seed), dev_n, test_n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.fixtures")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", default="agent/fixtures/public/base_v1.json")
    ap.add_argument("--dev-out", default="", help="dev 集输出路径；与 --test-out 一起启用分层切分")
    ap.add_argument("--test-out", default="", help="test 集输出路径")
    ap.add_argument("--dev-n", type=int, default=DEV_N)
    ap.add_argument("--test-n", type=int, default=TEST_N)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args(argv)

    if args.dev_out or args.test_out:
        dev, test = generate_split(args.dev_n, args.test_n, args.split_seed)
        for data, out in ((dev, args.dev_out), (test, args.test_out)):
            if not out:
                continue
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
            print(f"[OK] {len(data)} 条 -> {p}")
        return 0

    data = generate_dialogues(args.n, args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {len(data)} 条 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
