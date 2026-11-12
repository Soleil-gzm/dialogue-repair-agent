"""单元测试：agent.benchmark.fingerprint"""

import json

from agent.benchmark.fingerprint import (
    strip_volatile, config_fingerprint, input_fingerprint,
    evaluator_fingerprint, adapter_fingerprint, build_lock,
)


class TestStripVolatile:
    def test_removes_top_level_volatile(self):
        out = strip_volatile({"seed": 1, "timestamp": "x", "run_id": "y"})
        assert out == {"seed": 1}

    def test_removes_nested_volatile(self):
        out = strip_volatile({"a": {"timestamp": "x", "v": 2}})
        assert out == {"a": {"v": 2}}

    def test_keeps_list_structure(self):
        out = strip_volatile([{"timestamp": "x", "v": 1}])
        assert out == [{"v": 1}]


class TestConfigFingerprint:
    def test_volatile_keys_do_not_change_fingerprint(self):
        a = {"seed": 1, "timestamp": "2026-01-01"}
        b = {"seed": 1, "timestamp": "2026-12-31"}
        assert config_fingerprint(a) == config_fingerprint(b)

    def test_real_change_changes_fingerprint(self):
        assert config_fingerprint({"seed": 1}) != config_fingerprint({"seed": 2})


class TestInputFingerprint:
    def test_counts_json_records(self, raw_dialogues_file, mini_dialogues):
        fp = input_fingerprint(raw_dialogues_file)
        assert fp["n_total"] == len(mini_dialogues)
        assert len(fp["sha256"]) == 64

    def test_path_uses_forward_slashes(self, raw_dialogues_file):
        fp = input_fingerprint(raw_dialogues_file)
        assert "\\" not in fp["path"]

    def test_counts_jsonl_records(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
        assert input_fingerprint(p)["n_total"] == 2

    def test_content_change_changes_hash(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text("[1]", encoding="utf-8")
        h1 = input_fingerprint(p)["sha256"]
        p.write_text("[2]", encoding="utf-8")
        assert input_fingerprint(p)["sha256"] != h1

    def test_sampling_metadata_recorded(self, raw_dialogues_file):
        fp = input_fingerprint(
            raw_dialogues_file, sample_keys=["a", "b"], sample_seed=7, strata=["scenario"],
        )
        assert fp["n_sampled"] == 2
        assert fp["sample_seed"] == 7
        assert fp["strata"] == ["scenario"]
        assert "sample_keys_sha256" in fp


class TestEvaluatorAndAdapterFingerprints:
    def test_evaluator_fingerprint_stable(self):
        a = evaluator_fingerprint(rules_version="1.0.0", judge_prompt_version="v1",
                                  judge_model="m", rubric_version="1.0")
        b = evaluator_fingerprint(rules_version="1.0.0", judge_prompt_version="v1",
                                  judge_model="m", rubric_version="1.0")
        assert a["sha256"] == b["sha256"]

    def test_evaluator_version_change_detected(self):
        a = evaluator_fingerprint(rules_version="1.0.0", judge_prompt_version="v1",
                                  judge_model="m", rubric_version="1.0")
        b = evaluator_fingerprint(rules_version="1.1.0", judge_prompt_version="v1",
                                  judge_model="m", rubric_version="1.0")
        assert a["sha256"] != b["sha256"]

    def test_adapter_fingerprint_detects_profile_change(self):
        a = adapter_fingerprint(adapter_version="1.0.0", canonical_schema_version="1.0",
                                domain_profile="public_after_sales", domain_profile_hash="h1")
        b = adapter_fingerprint(adapter_version="1.0.0", canonical_schema_version="1.0",
                                domain_profile="private_collection", domain_profile_hash="h1")
        assert a["sha256"] != b["sha256"]


class TestBuildLock:
    def test_has_four_fingerprints(self, raw_dialogues_file):
        lock = build_lock(
            input_fp=input_fingerprint(raw_dialogues_file),
            evaluator_fp=evaluator_fingerprint(rules_version="1", judge_prompt_version="v1",
                                               judge_model="m", rubric_version="1"),
            adapter_fp=adapter_fingerprint(adapter_version="1", canonical_schema_version="1",
                                           domain_profile="d", domain_profile_hash="h"),
            arms={"A0": {"config_sha256": "x", "prompt_version": None}},
        )
        for key in ("input", "evaluator", "adapter", "arms"):
            assert key in lock
        assert lock["lock_version"] == "1.0"
