"""标注模板生成脚本的单元测试（纯函数部分，不依赖真实数据）。"""

import json
import os
import tempfile

from agent.eval.labeling import (aggregate, build_sheet, load_judge_off_turns)


def _write_jsonl(records):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.close()
    return f.name


def test_load_judge_off_turns_extracts_related_false():
    path = _write_jsonl([
        {"meta": {"stage": "judge", "dialogue_key": "aaa"},
         "response": {"text": json.dumps({"verdicts": [
             {"turn": 2, "related": False, "detail": "未回应"},
             {"turn": 4, "related": True, "detail": "ok"},
         ]})}},
        {"meta": {"stage": "judge", "dialogue_key": "bbb"},
         "response": {"text": json.dumps({"verdicts": [
             {"turn": 2, "related": True},
         ]})}},
        {"meta": {"stage": "detect"}, "response": {"text": ""}},  # 非 judge 跳过
    ])
    try:
        off = load_judge_off_turns(path)
        assert off == {"aaa": {2: "未回应"}}, off
        assert "bbb" not in off  # 全 related=true 不收录
    finally:
        os.unlink(path)


def test_load_judge_off_turns_skips_invalid_json():
    path = _write_jsonl([
        {"meta": {"stage": "judge", "dialogue_key": "ccc"},
         "response": {"text": "not json"}},
        "garbage line",
    ])
    try:
        assert load_judge_off_turns(path) == {}
    finally:
        os.unlink(path)


def _write_tsv(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


def test_build_sheet_shape():
    items = [{"key": "aaa", "n_off": 0}, {"key": "bbb", "n_off": 2}]
    off = {"bbb": {2: "x", 4: "y"}}
    lines = build_sheet(items, off).strip().split("\n")
    assert lines[0].split("\t") == [
        "sample_no", "key", "n_off", "off_turns", "overall", "per_turn", "notes"]
    assert lines[1].split("\t")[:4] == ["1", "aaa", "0", ""]
    assert lines[2].split("\t")[:4] == ["2", "bbb", "2", "2;4"]


def test_aggregate_precision_and_negative_control():
    tsv = _write_tsv(
        "sample_no\tkey\tn_off\toff_turns\toverall\tper_turn\tnotes\n"
        "1\taaa\t0\t\t否\t\t\n"          # 阴性对照，确认无问题
        "2\tbbb\t2\t2;4\t是\t是;否\t\n"  # 两轮：一真一误报
        "3\tccc\t1\t2\t\t\t\n"          # 未填
    )
    try:
        s = aggregate(tsv)
        assert s["labeled"] == 2 and s["pending"] == 1
        assert s["overall"] == {"yes": 1, "no": 1, "unsure": 0}
        assert s["turn"]["yes"] == 1 and s["turn"]["no"] == 1
        assert s["turn"]["precision"] == 0.5
        assert s["negative_controls"] == {
            "n": 1, "labeled": 1, "confirmed_no_issue": 1, "false_positive_rate": 0.0}
    finally:
        os.unlink(tsv)


def test_aggregate_per_turn_length_mismatch_raises():
    tsv = _write_tsv(
        "sample_no\tkey\tn_off\toff_turns\toverall\tper_turn\tnotes\n"
        "1\tbbb\t2\t2;4\t是\t是\t\n"
    )
    try:
        try:
            aggregate(tsv)
        except ValueError as e:
            assert "per_turn" in str(e)
        else:
            raise AssertionError("应因 per_turn 数量不一致报错")
    finally:
        os.unlink(tsv)


def test_aggregate_recheck_kappa_identical():
    header = ("sample_no\tkey\tn_off\toff_turns\toverall\tper_turn\tnotes\n")
    body = "1\ta\t1\t2\t是\t是\t\n2\tb\t1\t2\t否\t否\t\n3\tc\t0\t\t不确定\t\t\n"
    p1 = _write_tsv(header + body)
    p2 = _write_tsv(header + body)
    try:
        s = aggregate(p1, p2)
        assert s["recheck"]["n"] == 3
        assert s["recheck"]["raw_agreement"] == 1.0
        assert s["recheck"]["cohen_kappa"] == 1.0
    finally:
        os.unlink(p1)
        os.unlink(p2)
