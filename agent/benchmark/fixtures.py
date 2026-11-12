"""公开轨 fixture 生成器（中立域）

为什么需要它：Benchmark 的公开轨数据不能来自任何真实业务，必须程序化构造。
本模块只依赖标准库 random，同 seed 必然产出同一份数据。

域设定为「订单售后回访」——与真实业务域无关，仅用于演示与计量。
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.fixtures")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", default="agent/fixtures/public/base_v1.json")
    args = ap.parse_args(argv)

    data = generate_dialogues(args.n, args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {len(data)} 条 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
