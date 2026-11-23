"""标注模板生成脚本的单元测试（纯函数部分，不依赖真实数据）。"""

import json
import os
import tempfile

from agent.eval.labeling import load_judge_off_turns


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
