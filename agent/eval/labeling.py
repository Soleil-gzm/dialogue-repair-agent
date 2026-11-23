"""人工标注模板生成：从 A4 结果抽样真实对话，脱敏后供二分类人工校准。

用途
----
A4 在私有轨 200 条真实对话上检出 197 条（judge 判 14.9% 的 assistant 轮
答非所问）。自动指标无法区分「真实数据本身质量差」与「判官过度敏感」，
需要人工二分类标注来校准：A4 标出的问题轮，是否真的需要干预。

标注口径（用户偏好：二分类，快）
--------------------------------
对每条抽样对话，标注：
  1. 整体：该对话是否存在需要干预的答非所问/连贯性问题？  是 / 否 / 不确定
  2. 逐轮：judge 标出的 off-topic 轮是否准确？            是 / 否 / 不确定

用法
----
  ANONYMIZE_ORG_TERMS="小赢卡贷,京东" \
  python -m agent.eval.labeling \
    --base agent/fixtures/private/private_test.json \
    --ckpt logs/agent/2026-11-23-private/ckpt_A4.jsonl \
    --cassette logs/agent/2026-11-23-private/cassette_A4.jsonl \
    --out logs/agent/2026-11-23-private/labeling_template.md \
    --index-out logs/agent/2026-11-23-private/labeling_index.json

产物全部在 logs/（gitignored），不入库。脚本本身不含任何真实业务数据。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# 允许 `python -m agent.eval.labeling` 从仓库根运行
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.canonical import dialogue_key  # noqa: E402
from agent.eval.anonymize import anonymize_dialogue  # noqa: E402


def load_judge_off_turns(cassette_path: str) -> Dict[str, Dict[int, str]]:
    """从 cassette 提取每个 dialogue 的 judge「判为答非所问」轮。

    返回 {dialogue_key: {turn: detail}}，turn 对应 messages 下标
    （assistant 轮在偶数下标，与 judge verdict.turn 一致）。
    """
    off: Dict[str, Dict[int, str]] = {}
    with open(cassette_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            meta = rec.get("meta", {})
            if not isinstance(meta, dict) or meta.get("stage") != "judge":
                continue
            h = meta.get("dialogue_key")
            if not h:
                continue
            resp = rec.get("response", {})
            text = resp.get("text", "") if isinstance(resp, dict) else ""
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            for v in data.get("verdicts", []) or []:
                if not v.get("related", True):
                    turn = v.get("turn")
                    if isinstance(turn, int):
                        off.setdefault(h, {})[turn] = v.get("detail", "") or ""
    return off


def _render_dialogue(dialogue: dict, off_turns: Dict[int, str]) -> List[str]:
    """渲染脱敏后的对话，judge 标出的 off 轮用 🔴 标记。"""
    out: List[str] = []
    msgs = dialogue.get("messages", [])
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if role == "system":
            # system 折叠，只在首条展示以省版面
            if i == 0:
                out.append(f"> **[system]**（已脱敏，省略）")
            continue
        mark = " 🔴 **judge 标为答非所问**" if i in off_turns else ""
        label = {"user": "用户", "assistant": "客服"}.get(role, role)
        out.append(f"**[{label} · turn {i}]**{mark}")
        out.append(content)
        out.append("")
    return out


def build_template(items: List[dict], off: Dict[str, Dict[int, str]],
                   org_terms: List[str]) -> str:
    lines: List[str] = [
        "# A4 私有轨人工标注模板（二分类）",
        "",
        "- 背景：A4 在 200 条真实对话上检出 197 条，judge 判 14.9% 的客服轮答非所问。",
        "  本清单用于校准：A4 标出的问题，是真问题还是误报？",
        "- 样本：分层抽样 50 条真实对话（已脱敏），按 judge 标出的 off 轮数分层。",
        "- 🔴 = judge 判该轮答非所问（related=false）。",
        "",
        "## 标注说明",
        "",
        "对每条对话填写：",
        "",
        "1. **整体判断**：该对话是否存在需要干预的答非所问 / 连贯性问题？",
        "   `是` / `否` / `不确定`",
        "2. **逐轮准确性**：judge 标出的 🔴 轮，是否真的答非所问？",
        "   每轮 `是` / `否` / `不确定`（若对话无 🔴 轮则跳过）",
        "3. **备注**：可选。",
        "",
        "---",
        "",
    ]
    for s_i, it in enumerate(items, 1):
        dlg = anonymize_dialogue(it["dialogue"])
        off_turns = off.get(it["key"], {})
        off_list = sorted(off_turns)
        lines.append(f"## 样本 {s_i}　`{it['key']}`")
        lines.append("")
        lines.append(f"- 来源：base 真实对话（下标 {it['idx']}）")
        lines.append(f"- A4 状态：{it['status']}　|　judge 标出 off 轮数：{len(off_list)}"
                     + (f"（turn {off_list}）" if off_list else "（无）"))
        lines.append("")
        lines.append("### 对话（已脱敏）")
        lines.append("")
        lines.extend(_render_dialogue(dlg, off_turns))
        lines.append("### 标注")
        lines.append("")
        lines.append("- 整体判断：该对话是否存在需要干预的问题？　"
                     "□ 是　□ 否　□ 不确定")
        if off_list:
            parts = "　".join(f"turn {t}: □是 □否 □不确定" for t in off_list)
            lines.append(f"- judge 标出轮是否准确：{parts}")
        else:
            lines.append("- judge 标出轮是否准确：（无 off 轮，A4 全过，作阴性对照）")
        lines.append("- 备注：")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base 真实对话 JSON")
    ap.add_argument("--ckpt", required=True, help="A4 checkpoint jsonl")
    ap.add_argument("--cassette", required=True, help="A4 cassette jsonl")
    ap.add_argument("--out", required=True, help="标注模板 Markdown 输出路径")
    ap.add_argument("--index-out", required=True,
                    help="抽样索引 JSON（供标注完成后汇总，不入库）")
    ap.add_argument("--n", type=int, default=50, help="抽样总数（默认 50）")
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    base = json.load(open(args.base, encoding="utf-8"))
    status: Dict[str, str] = {}
    for line in open(args.ckpt, encoding="utf-8"):
        r = json.loads(line)
        status[r["key"]] = r["status"]
    off = load_judge_off_turns(args.cassette)

    items = []
    for i, d in enumerate(base):
        h = dialogue_key(d)
        n_off = len(off.get(h, {}))
        items.append({
            "idx": i, "key": h, "n_off": n_off,
            "status": status.get(f"A4:base:{h}", "?"),
            "dialogue": d,
        })

    # 分层配额：off=0 作阴性对照（全抽，上限 neg）；off≥1 按 1/2/3+ 比例分配
    total = args.n
    neg = max(1, total // 10)
    buckets = {
        "0": [it for it in items if it["n_off"] == 0],
        "1": [it for it in items if it["n_off"] == 1],
        "2": [it for it in items if it["n_off"] == 2],
        "3+": [it for it in items if it["n_off"] >= 3],
    }
    sizes = {b: len(buckets[b]) for b in buckets}
    quota = {"0": min(neg, sizes["0"])}
    remaining = total - quota["0"]
    s1 = sum(sizes[b] for b in ["1", "2", "3+"]) or 1
    for b in ["1", "2", "3+"]:
        quota[b] = round(remaining * sizes[b] / s1)
    # 修正取整误差，优先补给 off=1 桶（最大）
    diff = total - sum(quota.values())
    if diff != 0:
        quota["1"] = max(0, quota["1"] + diff)

    rng = random.Random(args.seed)
    sampled: List[dict] = []
    for b in ["0", "1", "2", "3+"]:
        pool = list(buckets[b])
        rng.shuffle(pool)
        sampled.extend(pool[: quota[b]])
    rng.shuffle(sampled)
    sampled = sampled[:total]

    org_terms = [t.strip() for t in os.environ.get("ANONYMIZE_ORG_TERMS", "").split(",")
                 if t.strip()]
    md = build_template(sampled, off, org_terms)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")

    # 索引：供标注完成后回填汇总（存 key / idx / off 轮，不存原文）
    index = [{
        "sample_no": i + 1, "key": it["key"], "base_idx": it["idx"],
        "n_off": it["n_off"], "off_turns": sorted(off.get(it["key"], {})),
        "a4_status": it["status"],
    } for i, it in enumerate(sampled)]
    Path(args.index_out).write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"抽样 {len(sampled)} 条 → {args.out}")
    print(f"分层分布: " + ", ".join(
        f"off={b}:{sum(1 for x in sampled if (x['n_off']==0 if b=='0' else (x['n_off']==int(b) if b.isdigit() else x['n_off']>=3)))} 条"
        for b in ["0", "1", "2", "3+"]))
    print(f"索引 → {args.index_out}")


if __name__ == "__main__":
    main()
