"""测试：作品集报告生成（纯离线，无绘图库依赖）"""

import json

from agent.benchmark.report import (
    ablation_svg, heatmap_svg, load_arms, load_trace, render_html,
    render_markdown, summarize, trace_svg,
)


def _arm(name, blocked=20, total=20, cost=0.0, tokens=0, clean=12, flagged=0, stopped=0):
    per = {t: {"total": total, "detected": blocked, "blocked": blocked}
           for t in ("D1_off_topic", "D2_reference_broken", "D3_fact_conflict",
                     "D4_role_confusion", "D5_repetition", "D6_missing_response")}
    if name in ("A3", "A3L"):
        per["D1_off_topic"] = {"total": total, "detected": 0, "blocked": 0}
    return {
        "arm": name,
        "description": name,
        "per_type": per,
        "macro_recall_blocked": round(
            sum(r["blocked"] / r["total"] for r in per.values()) / len(per), 4),
        "clean_total": clean, "clean_flagged": flagged,
        "clean_fp_rate": round(flagged / clean, 4) if clean else 0.0,
        "failed": 0, "stopped": stopped,
        "tokens": tokens, "calls": 0, "cost_cny": cost, "elapsed_s": 0.0,
    }


def _arms():
    return [_arm("A3"), _arm("A3L", cost=0.4, tokens=251561),
            _arm("A4", cost=0.55, tokens=305910)]


TRACE = [
    {"event": "start", "key": "k"},
    {"event": "decide", "key": "k", "action": "call_tool", "tool": "check_consistency",
     "reason": "先校验"},
    {"event": "tool_result", "key": "k", "tool": "check_consistency", "ok": True,
     "terminal": False, "latency_ms": 12, "tokens": 0},
    {"event": "decide", "key": "k", "action": "stop", "tool": None, "reason": "通过"},
    {"event": "end", "key": "k", "status": "accepted", "termination": "stop"},
]


class TestSummarize:
    def test_recall_table(self):
        report = summarize(_arms(), TRACE)
        assert report["recall"]["A3"]["D1_off_topic"] == 0.0
        assert report["recall"]["A4"]["D1_off_topic"] == 1.0
        assert report["meta"]["defect_types"][0] == "D1_off_topic"
        assert report["meta"]["trace_events"] == 5

    def test_conclusions_are_generated_from_data(self):
        report = summarize(_arms(), [])
        text = " ".join(report["conclusions"])
        assert "A3 → A3L" in text and "没有带来增量" in text
        assert "A3L → A4" in text
        assert "误报" in text

    def test_stopped_is_called_out(self):
        arms = _arms()
        arms[2]["stopped"] = 1
        text = " ".join(summarize(arms, [])["conclusions"])
        assert "预算/墙钟中止" in text

    def test_no_arms_does_not_crash(self):
        report = summarize([], [])
        assert report["arms"] == []
        assert report["meta"]["defect_types"] == []


class TestFigures:
    def test_ablation_svg_contains_bars(self):
        svg = ablation_svg(_arms())
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert svg.count("<rect") >= 3
        assert "A3L" in svg and "83.3%" in svg

    def test_heatmap_svg_has_all_rows(self):
        svg = heatmap_svg(_arms())
        assert "D1 答非所问" in svg and "D6 缺失承接" in svg

    def test_trace_svg_without_data(self):
        assert "无 trace 数据" in trace_svg([])

    def test_trace_svg_renders_steps(self):
        svg = trace_svg(TRACE)
        assert "check_consistency" in svg
        assert svg.count("<circle") == 3

    def test_before_after_svg(self):
        from agent.benchmark.report import before_after_svg

        case = {"title": "答非所问", "user": "催一下", "candidate": "请先付款",
                "original": "已为您催办", "verdict": "C3_off_topic"}
        svg = before_after_svg(case)
        assert "答非所问" in svg and "请先付款" in svg and "已为您催办" in svg
        assert "C3_off_topic" in svg

    def test_before_after_block_in_html(self):
        report = summarize(_arms(), [])
        html = render_html(report, "", "", trace_svg([]), "<svg></svg>")
        assert "Before / After" in html

    def test_figures_are_deterministic(self):
        assert ablation_svg(_arms()) == ablation_svg(_arms())
        assert trace_svg(TRACE) == trace_svg(TRACE)


class TestRender:
    def test_markdown_has_table_and_repro(self):
        report = summarize(_arms(), TRACE)
        md = render_markdown(report)
        assert md.startswith("# ")
        assert "| 臂 |" in md and "A4" in md
        assert "python -m agent.benchmark.agent_eval" in md
        assert "## 结论" in md

    def test_html_is_self_contained(self):
        report = summarize(_arms(), TRACE)
        html = render_html(report, ablation_svg(_arms()),
                           heatmap_svg(_arms()), trace_svg(TRACE))
        assert "<style>" in html and "<svg" in html
        # xmlns 是 SVG 命名空间声明，不是外部资源；其余外链一律不允许
        body = html.replace('xmlns="http://www.w3.org/2000/svg"', "")
        assert "http://" not in body and "https://" not in body
        assert "<link" not in html and "src=" not in html
        assert "83.3%" in html

    def test_html_escapes_text(self):
        report = summarize(_arms(), [])
        report["conclusions"] = ["<script>alert(1)</script>"]
        html = render_html(report, "", "", trace_svg([]))
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestCorrections:
    def test_correction_notes_are_rendered(self):
        arms = _arms()
        arms[2]["corrections"] = [{
            "field": "per_type.D5_repetition.blocked", "from": 19, "to": 20,
            "reason": "宿主机待机导致预算中止", "recheck": "recheck_d5_001",
        }]
        report = summarize(arms, [])
        md = render_markdown(report)
        assert "数据修正说明" in md and "19 → 20" in md and "recheck_d5_001" in md
        html = render_html(report, "", "", trace_svg([]))
        assert "数据修正说明" in html


class TestLoad:
    def test_load_arms_sorted(self, tmp_path):
        p = tmp_path / "ablation_A4.json"
        p.write_text(json.dumps(_arm("A4")), encoding="utf-8")
        p2 = tmp_path / "ablation_A3.json"
        p2.write_text(json.dumps(_arm("A3")), encoding="utf-8")
        arms = load_arms([p, p2])
        assert [a["arm"] for a in arms] == ["A3", "A4"]

    def test_load_trace_missing_file(self, tmp_path):
        assert load_trace(tmp_path / "nope.jsonl") == []
        assert load_trace(None) == []

    def test_raw_evidence_files_are_ignored(self, tmp_path):
        from agent.benchmark.report import _collect_results

        (tmp_path / "ablation_A4.json").write_text("{}", encoding="utf-8")
        (tmp_path / "ablation_A4.raw.json").write_text("{}", encoding="utf-8")
        names = [p.name for p in _collect_results(tmp_path)]
        assert names == ["ablation_A4.json"]
