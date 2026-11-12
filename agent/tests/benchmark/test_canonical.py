"""单元测试：agent.benchmark.canonical"""

from agent.benchmark.canonical import (
    stable_dump, sha256_bytes, sha256_obj, sha256_file,
)


class TestStableDump:
    def test_key_order_irrelevant(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        assert stable_dump(a) == stable_dump(b)

    def test_chinese_not_escaped(self):
        assert "中文".encode("utf-8") in stable_dump({"k": "中文"})

    def test_trailing_newline(self):
        assert stable_dump({"k": 1}).endswith(b"\n")

    def test_list_order_matters(self):
        assert stable_dump([1, 2]) != stable_dump([2, 1])


class TestHashes:
    def test_sha256_obj_deterministic(self):
        assert sha256_obj({"a": 1}) == sha256_obj({"a": 1})

    def test_sha256_obj_sensitive_to_value(self):
        assert sha256_obj({"a": 1}) != sha256_obj({"a": 2})

    def test_sha256_file_matches_bytes(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello")
        assert sha256_file(p) == sha256_bytes(b"hello")

    def test_sha256_is_64_hex(self):
        h = sha256_obj({"a": 1})
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
