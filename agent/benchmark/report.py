"""作品集报告生成（三件套 + 图）

输入：臂评测产出的 ablation_*.json、可选 trace jsonl
输出：benchmark_report.{json,md,html} + figures/*.svg

三条约束：
- 只依赖标准库（图表手写 SVG），评审机器上不需要装绘图库
- 纯函数式渲染，同输入必然同输出（可直接比对 sha256）
- 报告里的结论由数据算出来，不写死数字
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ARM_ORDER = ["A3", "A3L", "A4"]

DEFECT_LABELS = {
    "D1_off_topic": "D1 答非所问",
    "D2_reference_broken": "D2 指代断裂",
    "D3_fact_conflict": "D3 事实冲突",
    "D4_role_confusion": "D4 角色混淆",
    "D5_repetition": "D5 重复空转",
    "D6_missing_response": "D6 缺失承接",
}

COLORS = {"A3": "#4c6ef5", "A3L": "#f59f00", "A4": "#12b886"}


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

def load_arms(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    arms = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if data.get("arm"):
            arms.append(data)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    arms.sort(key=lambda a: order.get(a["arm"], 99))
    return arms


def load_trace(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]


def _rate(row: Dict[str, Any], key: str = "blocked") -> float:
    total = row.get("total") or 0
    return (row.get(key, 0) / total) if total else 0.0


def summarize(arms: List[Dict[str, Any]], trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    defect_types = []
    for arm in arms:
        for t in (arm.get("per_type") or {}):
            if t not in defect_types:
                defect_types.append(t)
    defect_types.sort()

    recall: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        per = arm.get("per_type") or {}
        recall[arm["arm"]] = {
            t: round(_rate(per.get(t, {})), 4) for t in defect_types
        }

    return {
        "meta": {
            "title": "对话数据修复 Agent：可复现评测",
            "dataset": {"n_defects": 20 * len(defect_types), "per_type": 20,
                        "n_clean": (arms[0].get("clean_total", 0) if arms else 0)},
            "defect_types": defect_types,
            "defect_labels": {t: DEFECT_LABELS.get(t, t) for t in defect_types},
            "trace_events": len(trace),
        },
        "arms": arms,
        "recall": recall,
        "conclusions": _conclusions(arms, recall, defect_types),
    }


def _conclusions(arms: List[Dict[str, Any]], recall: Dict[str, Dict[str, float]],
                 defect_types: List[str]) -> List[str]:
    by_arm = {a["arm"]: a for a in arms}
    out: List[str] = []

    for lo, hi in (("A3", "A3L"), ("A3L", "A4")):
        if lo not in by_arm or hi not in by_arm:
            continue
        m_lo = by_arm[lo].get("macro_recall_blocked", 0.0)
        m_hi = by_arm[hi].get("macro_recall_blocked", 0.0)
        delta = round((m_hi - m_lo) * 100, 1)
        cost = round(by_arm[hi].get("cost_cny", 0.0) - by_arm[lo].get("cost_cny", 0.0), 4)
        if abs(delta) < 0.05:
            out.append(
                f"{lo} → {hi}：宏观阻断率不变（{m_lo:.1%}），成本变化 ¥{cost:.2f} —— "
                "该层因子在这套工具集上没有带来增量。")
        else:
            out.append(
                f"{lo} → {hi}：宏观阻断率 {m_lo:.1%} → {m_hi:.1%}（{delta:+.1f}pp），"
                f"成本变化 ¥{cost:.2f}。")

    # 语义类缺陷的增量和来源
    if "A4" in recall and "A3" in recall:
        gains = [(t, recall["A4"].get(t, 0) - recall["A3"].get(t, 0)) for t in defect_types]
        gains = [(t, g) for t, g in gains if g > 0.01]
        if gains:
            detail = "、".join(f"{DEFECT_LABELS.get(t, t)} +{g * 100:.0f}pp" for t, g in gains)
            out.append(f"语义判官带来的增量集中在：{detail}。")

    for arm in arms:
        if arm.get("clean_total"):
            out.append(
                f"{arm['arm']} 干净样本误报 {arm.get('clean_flagged', 0)}/"
                f"{arm['clean_total']}（{arm.get('clean_fp_rate', 0):.1%}）。")
        if arm.get("stopped"):
            out.append(
                f"{arm['arm']} 有 {arm['stopped']} 条样本因预算/墙钟中止，"
                "已与「未检出」分开计数（环境噪声不计入模型能力）。")
    return out


# --------------------------------------------------------------------------
# 图（手写 SVG，无外部依赖）
# --------------------------------------------------------------------------

def ablation_svg(arms: List[Dict[str, Any]], width: int = 720, height: int = 320) -> str:
    """消融柱状图：宏观阻断率 + 单条成本标注。"""
    pad_l, pad_b, pad_t = 62, 52, 34
    plot_w, plot_h = width - pad_l - 24, height - pad_b - pad_t
    n = max(1, len(arms))
    slot = plot_w / n
    bar_w = min(96, slot * 0.5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Noto Sans SC, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="20" font-size="14" fill="#333">宏观阻断率（越高越好）与成本</text>',
    ]
    # 网格与刻度
    for i in range(0, 101, 25):
        y = pad_t + plot_h * (1 - i / 100)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e9ecef" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#868e96" '
                     f'text-anchor="end">{i}%</text>')

    for i, arm in enumerate(arms):
        v = float(arm.get("macro_recall_blocked", 0.0))
        cx = pad_l + slot * i + slot / 2
        h = plot_h * v
        color = COLORS.get(arm["arm"], "#495057")
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{pad_t + plot_h - h:.1f}" width="{bar_w:.1f}" '
            f'height="{h:.1f}" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t + plot_h - h - 8:.1f}" font-size="12" '
                     f'fill="#333" text-anchor="middle">{v * 100:.1f}%</text>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t + plot_h + 20:.1f}" font-size="12" '
                     f'fill="#495057" text-anchor="middle">{arm["arm"]}</text>')
        cost = float(arm.get("cost_cny", 0.0))
        parts.append(f'<text x="{cx:.1f}" y="{pad_t + plot_h + 38:.1f}" font-size="11" '
                     f'fill="#868e96" text-anchor="middle">¥{cost:.2f}</text>')

    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
                 f'y2="{pad_t + plot_h}" stroke="#adb5bd" stroke-width="1"/>')
    parts.append('</svg>')
    return "\n".join(parts)


def heatmap_svg(arms: List[Dict[str, Any]], width: int = 720, height: int = 300) -> str:
    """缺陷类型 × 臂 的阻断率热力图。"""
    defect_types: List[str] = []
    for arm in arms:
        for t in (arm.get("per_type") or {}):
            if t not in defect_types:
                defect_types.append(t)
    defect_types.sort()

    pad_l, pad_t, cell_h = 140, 44, 34
    cell_w = (width - pad_l - 16) / max(1, len(arms))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Noto Sans SC, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="16" y="24" font-size="14" fill="#333">各类缺陷阻断率</text>',
    ]
    for j, arm in enumerate(arms):
        x = pad_l + cell_w * j + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{pad_t - 12}" font-size="12" fill="#495057" '
                     f'text-anchor="middle">{arm["arm"]}</text>')

    for i, t in enumerate(defect_types):
        y = pad_t + cell_h * i
        label = DEFECT_LABELS.get(t, t)
        parts.append(f'<text x="{pad_l - 10}" y="{y + cell_h / 2 + 4:.1f}" font-size="12" '
                     f'fill="#495057" text-anchor="end">{label}</text>')
        for j, arm in enumerate(arms):
            row = (arm.get("per_type") or {}).get(t, {})
            v = _rate(row)
            x = pad_l + cell_w * j
            # 越低越红、越高越绿
            r = int(245 - 190 * v)
            g = int(90 + 140 * v)
            b = int(120 - 40 * v)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 6:.1f}" '
                         f'height="{cell_h - 6}" fill="rgb({r},{g},{b})" rx="3"/>')
            parts.append(f'<text x="{x + (cell_w - 6) / 2:.1f}" y="{y + cell_h / 2 + 3:.1f}" '
                         f'font-size="11" fill="#ffffff" text-anchor="middle">{v * 100:.0f}%</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def trace_svg(records: List[Dict[str, Any]], width: int = 760, height: int = 240) -> str:
    """决策链时间轴：每一步决策与工具结果。"""
    steps = [r for r in records if r.get("event") in ("decide", "tool_result")]
    if not steps:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="60">'
                f'<text x="16" y="34" font-size="13" fill="#868e96">无 trace 数据</text></svg>')

    pad_l, pad_t = 24, 34
    n = len(steps)
    gap = (width - pad_l * 2) / max(1, n - 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Noto Sans SC, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="20" font-size="14" fill="#333">决策链（观察 → 决策 → 工具 → 反馈）</text>',
        f'<line x1="{pad_l}" y1="{pad_t + 40}" x2="{width - pad_l}" y2="{pad_t + 40}" '
        f'stroke="#dee2e6" stroke-width="2"/>',
    ]
    for i, rec in enumerate(steps):
        x = pad_l + gap * i
        if rec["event"] == "decide":
            label = rec.get("tool") or "stop"
            color = "#4c6ef5"
            sub = "决策"
        else:
            label = f'{rec.get("tool")}({"ok" if rec.get("ok") else "违规"})'
            color = "#12b886" if rec.get("ok") else "#e03131"
            sub = f'{rec.get("latency_ms", 0)}ms'
        parts.append(f'<circle cx="{x:.1f}" cy="{pad_t + 40}" r="7" fill="{color}"/>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t + 22}" font-size="10" fill="#868e96" '
                     f'text-anchor="middle">{sub}</text>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t + 66}" font-size="11" fill="#343a40" '
                     f'text-anchor="middle">{html_escape(label)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _wrap(text: str, width: int) -> List[str]:
    """按字符数硬折行：中文没有空格，宽度按"字"算比按词更稳。"""
    text = str(text or "")
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def before_after_svg(case: Dict[str, Any], width: int = 760, height: int = 320) -> str:
    """Before/After 对照：受损候选 vs Agent 输出，标注判定理由。"""
    rows = [
        ("用户诉求", case.get("user", ""), "#495057", "#f1f3f5"),
        ("受损候选（注入缺陷）", case.get("candidate", ""), "#e03131", "#fff5f5"),
        ("Agent 输出（回退原文）", case.get("original", ""), "#12b886", "#ebfbee"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Noto Sans SC, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="20" y="24" font-size="14" fill="#333">'
        f'{html_escape(case.get("title", "Before / After"))}</text>',
    ]
    y = 44
    for label, text, color, bg in rows:
        lines = _wrap(text, 34)
        h = 26 + 20 * len(lines)
        parts.append(f'<rect x="20" y="{y}" width="{width - 40}" height="{h}" fill="{bg}" rx="6"/>')
        parts.append(f'<text x="34" y="{y + 20}" font-size="12" fill="{color}">'
                     f'{html_escape(label)}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="34" y="{y + 40 + i * 20}" font-size="13" '
                         f'fill="#343a40">{html_escape(line)}</text>')
        y += h + 12
    if case.get("verdict"):
        parts.append(f'<text x="20" y="{height - 16}" font-size="12" fill="#868e96">'
                     f'判定：{html_escape(case["verdict"])}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def html_escape(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# --------------------------------------------------------------------------
# 三件套
# --------------------------------------------------------------------------

def correction_notes(report: Dict[str, Any]) -> List[str]:
    """把人工复核过的修正显式写进报告：改了什么、为什么改、复跑证据在哪。"""
    notes = []
    for arm in report.get("arms") or []:
        for c in arm.get("corrections") or []:
            notes.append(
                f"{arm['arm']} · {c.get('field')}：{c.get('from')} → {c.get('to')}；"
                f"{c.get('reason')}（复核：{c.get('recheck', 'n/a')}）")
    return notes


def render_markdown(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    arms = report["arms"]
    defects = meta["defect_types"]

    head = ["| 臂 | " + " | ".join(DEFECT_LABELS.get(t, t) for t in defects)
            + " | 误报 | macro | tokens | 成本 |",
            "|---|" + "---|" * (len(defects) + 4)]
    rows = []
    for arm in arms:
        per = arm.get("per_type") or {}
        cells = [f"{per.get(t, {}).get('blocked', 0)}/{per.get(t, {}).get('total', 0)}"
                 for t in defects]
        rows.append(
            f"| {arm['arm']} | " + " | ".join(cells)
            + f" | {arm.get('clean_flagged', 0)}/{arm.get('clean_total', 0)}"
            + f" | {arm.get('macro_recall_blocked', 0):.1%}"
            + f" | {arm.get('tokens', 0):,}"
            + f" | ¥{arm.get('cost_cny', 0):.2f} |")

    lines = [
        f"# {meta['title']}",
        "",
        f"- 缺陷集：{meta['dataset']['n_defects']} 条（{len(defects)} 类 × "
        f"{meta['dataset']['per_type']}），干净样本 {meta['dataset']['n_clean']} 条",
        "- 指标：`blocked` = 检出并回退原文；`误报` = 干净样本被误判回退",
        "- 数据与成本均来自 append-only token 账本",
        "",
        *head, *rows,
        "",
        "## 结论",
        "",
        *[f"{i + 1}. {c}" for i, c in enumerate(report["conclusions"])],
        "",
    ]

    notes = correction_notes(report)
    if notes:
        lines += ["> **数据修正说明**", *[f"> - {n}" for n in notes], ""]

    lines += [
        "## 复现",
        "",
        "```bash",
        "# 离线臂（无需密钥）",
        "python -m agent.benchmark.agent_eval --arm A3 --out agent/benchmark/output/ablation_A3.json",
        "# 在线臂（需要 DEEPSEEK_API_KEY，会消耗 token）",
        "python -m agent.benchmark.agent_eval --arm A4 --live",
        "# 重新生成报告",
        "python -m agent.benchmark.report --results agent/benchmark/output --out portfolio",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_html(report: Dict[str, Any], ablation: str, heatmap: str, timeline: str,
                before_after: str = "") -> str:
    meta = report["meta"]
    arms = report["arms"]
    defects = meta["defect_types"]

    head_cells = "".join(f"<th>{html_escape(DEFECT_LABELS.get(t, t))}</th>" for t in defects)
    body_rows = []
    for arm in arms:
        per = arm.get("per_type") or {}
        cells = "".join(
            f"<td>{per.get(t, {}).get('blocked', 0)}/{per.get(t, {}).get('total', 0)}</td>"
            for t in defects)
        body_rows.append(
            f"<tr><td><b>{html_escape(arm['arm'])}</b><br><span class='muted'>"
            f"{html_escape(arm.get('description', ''))}</span></td>{cells}"
            f"<td>{arm.get('clean_flagged', 0)}/{arm.get('clean_total', 0)}</td>"
            f"<td>{arm.get('macro_recall_blocked', 0):.1%}</td>"
            f"<td>{arm.get('tokens', 0):,}</td><td>¥{arm.get('cost_cny', 0):.2f}</td></tr>")

    conclusions = "".join(f"<li>{html_escape(c)}</li>" for c in report["conclusions"])
    before_after_block = ""
    if before_after:
        before_after_block = f'<h2>3.1 Before / After</h2><div class="card">{before_after}</div>'
    notes = correction_notes(report)
    corrections_block = ""
    if notes:
        corrections_block = (
            "<h2>5.1 数据修正说明</h2><div class=\"card\"><ul>"
            + "".join(f"<li>{html_escape(n)}</li>" for n in notes)
            + "</ul></div>")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_escape(meta['title'])}</title>
<style>
body{{margin:0;background:#f8f9fa;color:#212529;font-family:"Segoe UI","Noto Sans SC",sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:17px;margin:32px 0 12px}}
.muted{{color:#868e96;font-size:12px;font-weight:400}}
.lead{{color:#495057;font-size:14px;line-height:1.8}}
.card{{background:#fff;border:1px solid #e9ecef;border-radius:10px;padding:18px;margin-top:14px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #f1f3f5;padding:8px 6px;text-align:center}}
th{{color:#868e96;font-weight:500;font-size:12px}}
td:first-child,th:first-child{{text-align:left}}
svg{{max-width:100%;height:auto}}
ol{{line-height:1.9;font-size:14px;padding-left:20px}}
code{{background:#f1f3f5;padding:1px 5px;border-radius:4px;font-size:12px}}
pre{{background:#212529;color:#f8f9fa;padding:14px;border-radius:8px;overflow:auto;font-size:12px;line-height:1.7}}
</style></head><body><div class="wrap">
<h1>{html_escape(meta['title'])}</h1>
<p class="muted">缺陷集 {meta['dataset']['n_defects']} 条（{len(defects)} 类 × {meta['dataset']['per_type']}）
· 干净样本 {meta['dataset']['n_clean']} 条 · 指标口径：blocked = 检出并回退原文</p>
<p class="lead">目标不是"感觉改得更顺了"，而是在固定缺陷集上量化"能不能发现问题、能不能挡住问题、花多少钱"。</p>

<h2>1. 消融：效果与成本</h2>
<div class="card">{ablation}</div>

<h2>2. 各类缺陷阻断率</h2>
<div class="card">{heatmap}</div>

<h2>3. 决策链</h2>
<div class="card">{timeline}</div>
{before_after_block}

<h2>4. 明细</h2>
<div class="card"><table>
<thead><tr><th>臂</th>{head_cells}<th>误报</th><th>macro</th><th>tokens</th><th>成本</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></div>

<h2>5. 结论</h2>
<div class="card"><ol>{conclusions}</ol></div>
{corrections_block}

<h2>6. 复现</h2>
<div class="card"><pre>python -m agent.benchmark.agent_eval --arm A3 --out agent/benchmark/output/ablation_A3.json
python -m agent.benchmark.agent_eval --arm A4 --live
python -m agent.benchmark.report --results agent/benchmark/output --out portfolio</pre></div>
</div></body></html>
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _collect_results(results: Path) -> List[Path]:
    """目录模式下只取正式产物：*.raw.json 是修正前的证据留档，不参与汇总。"""
    if results.is_dir():
        return sorted(p for p in results.glob("ablation_*.json")
                      if not p.name.endswith(".raw.json"))
    return [results]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.benchmark.report")
    ap.add_argument("--results", required=True, help="ablation_*.json 文件或所在目录")
    ap.add_argument("--trace", default="", help="trace jsonl（可选）")
    ap.add_argument("--case", default="", help="before/after 案例 json（可选）")
    ap.add_argument("--out", required=True, help="报告输出目录")
    args = ap.parse_args(argv)

    arms = load_arms(_collect_results(Path(args.results)))
    if not arms:
        print("[ERROR] 没有找到 ablation_*.json")
        return 2

    trace = load_trace(Path(args.trace) if args.trace else None)
    report = summarize(arms, trace)

    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ablation = ablation_svg(arms)
    heatmap = heatmap_svg(arms)
    timeline = trace_svg(trace)
    before_after = ""
    if args.case:
        case = json.loads(Path(args.case).read_text(encoding="utf-8"))
        before_after = before_after_svg(case)

    (fig_dir / "01_ablation.svg").write_text(ablation + "\n", encoding="utf-8")
    (fig_dir / "02_heatmap.svg").write_text(heatmap + "\n", encoding="utf-8")
    (fig_dir / "03_trace_timeline.svg").write_text(timeline + "\n", encoding="utf-8")
    if before_after:
        (fig_dir / "04_before_after.svg").write_text(before_after + "\n", encoding="utf-8")

    (out / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "benchmark_report.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "benchmark_report.html").write_text(
        render_html(report, ablation, heatmap, timeline, before_after), encoding="utf-8")

    print(f"[OK] 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
