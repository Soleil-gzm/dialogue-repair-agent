"""私有轨数据规范化：屏蔽 system 之后首条 user（业务设计里的随机开场句）。

背景
----
私有业务数据的每条对话里，system 之后的第一条 user 是从对话库里随机抽取的
一句客户话术，仅用于触发通话，不参与后续对话的连贯性判断。但语义判官会拿它
和紧接着的客服开场白（如「下午好，请问是 X 女士吗？」）做连贯性比对，从而把
turn 2 误判为答非所问。实测 200 条真实对话中 197 条命中该伪影，其中 98 条
只有 turn 2 被标——直接污染了 clean_flagged 指标。

处置
----
把首条 user 的 content 替换为中性占位符（默认「【通话开始】」），保留消息
条数与下标不变（缺陷注入、判官轮号均按下标对齐）。原始业务数据不入库。

用法
----
  python -m agent.eval.preprocess \
    --input agent/fixtures/private/private_test.json \
    --inplace
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import List

OPENING_PLACEHOLDER = "【通话开始】"


def mask_opening_user(dialogue: dict, placeholder: str = OPENING_PLACEHOLDER) -> dict:
    """返回副本：system 之后第一条 user 的 content 替换为占位符。

    不改变消息数量与顺序（后续缺陷注入、判官轮号依赖下标稳定）。
    找不到 user 时原样返回副本。
    """
    out = copy.deepcopy(dialogue)
    for m in out.get("messages") or []:
        if m.get("role") == "user":
            m["content"] = placeholder
            break
    return out


def mask_dialogues(dialogues: List[dict],
                   placeholder: str = OPENING_PLACEHOLDER) -> List[dict]:
    return [mask_opening_user(d, placeholder) for d in dialogues]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.eval.preprocess")
    ap.add_argument("--input", required=True, help="对话 JSON（列表）")
    ap.add_argument("--out", default="", help="输出路径；与 --inplace 二选一")
    ap.add_argument("--inplace", action="store_true", help="原地覆盖输入文件")
    ap.add_argument("--placeholder", default=OPENING_PLACEHOLDER)
    args = ap.parse_args(argv)

    if bool(args.out) == bool(args.inplace):
        ap.error("必须且只能指定 --out 或 --inplace 之一")

    src = Path(args.input)
    dialogues = json.loads(src.read_text(encoding="utf-8"))
    masked = mask_dialogues(dialogues, args.placeholder)
    text = json.dumps(masked, ensure_ascii=False)
    dst = src if args.inplace else Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")

    changed = sum(
        1 for a, b in zip(dialogues, masked)
        if (a.get("messages") or []) != (b.get("messages") or [])
    )
    print(f"[OK] {len(masked)} 条 -> {dst}（屏蔽 {changed} 条首条 user）",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
