"""单元测试：agent.benchmark.manifest（只溯源，不判等）"""

import json

from agent.benchmark.manifest import build_manifest, write_manifest


class TestBuildManifest:
    def test_records_machine_and_models(self, tmp_path):
        m = build_manifest(
            root=tmp_path, profile="p", arms=["A0"],
            models={"judge": "m", "generate": None},
        )
        assert m["machine"]["python"]
        assert m["machine"]["platform"]
        assert m["profile"] == "p"
        assert m["arms"] == ["A0"]
        assert m["models"]["judge"] == "m"

    def test_deps_lock_hash_from_uv_lock(self, tmp_path):
        (tmp_path / "uv.lock").write_text("x", encoding="utf-8")
        m = build_manifest(root=tmp_path)
        assert m["deps_lock"]["file"] == "uv.lock"
        assert len(m["deps_lock"]["sha256"]) == 64

    def test_no_lock_file_is_explicitly_none(self, tmp_path):
        assert build_manifest(root=tmp_path)["deps_lock"] is None


class TestWriteManifest:
    def test_writes_canonical_json(self, tmp_path):
        p = write_manifest({"b": 1, "a": 2}, tmp_path / "sub" / "m.json")
        assert p.exists()
        assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2, "b": 1}
