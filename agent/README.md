# agent/

本目录是 `dialogue-repair-agent` 的**包实现**，不是项目首页。

- 项目说明、快速开始、设计取舍 → [仓库根 README](../README.md)
- 文档总索引（架构 / 决策 / 指南 / 归档） → [docs/README.md](docs/README.md)

## 模块地图

| 路径 | 作用 |
|---|---|
| `benchmark/` | 规范化哈希、四元组指纹、不可变基线锁、运行清单、对照臂、CLI |
| `benchmark/baseline/` | 冻结的基线（`baseline_lock.json`、`lock_changelog.md`） |
| `benchmark/output/` | 运行产物（不入库） |
| `fixtures/public/` | 公开轨数据：`base_v1` 12 / `dev_v1` 30 / `test_v1` 50 / `defects_v1` 120 |
| `eval/` | 规则（R1–R5）、连贯性（C1–C3）、语义判官（D1）、实体抽取 |
| `tools/` | `check_consistency`、`check_coherence`、`fallback_keep`、`rewrite_segment` |
| `runtime/` | 工具注册表、状态、预算、规则/LLM 策略、决策循环、Trace、token 账本、客户端 |
| `tests/` | 单元 / 契约 / 集成 / 快照 / 开源边界自证 |
| `docs/` | 架构设计、Benchmark 评测设计、ADR、路线图、部署指南、归档 |
| `scripts/` | `export_oss.py` 按清单导出 |

## 常用命令

```bash
# 公开轨基础数据
python -m agent.benchmark.fixtures --n 12 --seed 20261112

# dev / test 分层切分
python -m agent.benchmark.fixtures \
  --dev-out agent/fixtures/public/dev_v1.json \
  --test-out agent/fixtures/public/test_v1.json

# 缺陷注入
python -m agent.benchmark.defects --per-type 20 --seed 20261113 \
  --out agent/fixtures/public/defects_v1.jsonl

# 复现冻结基线
python -m agent.benchmark.run \
  --input agent/fixtures/public/base_v1.json \
  --arms A0 --verify-arm A0

# Agent 臂评测（A3 离线；A3L/A3R/A4/A4R 需 --live 与密钥）
python -m agent.benchmark.agent_eval --arm A3 --out agent/benchmark/output/ablation_A3.json
python -m agent.benchmark.agent_eval --arm A3R --live --out agent/benchmark/output/ablation_A3R.json

# 测试
python -m pytest
```
