# agent/

本目录是 `dialogue-repair-agent` 的**包实现**，不是项目首页。

- 项目说明、快速开始、设计取舍 → [仓库根 README](../README.md)
- 设计文档（基准方案、路线） → [docs/README.md](docs/README.md)

## 模块地图

| 路径 | 作用 |
|---|---|
| `benchmark/` | 规范化哈希、四元组指纹、不可变基线锁、对照臂、CLI |
| `benchmark/baseline/` | 冻结的基线（`baseline_lock.json`） |
| `fixtures/public/` | 公开轨数据（程序化合成，无业务内容） |
| `tests/` | 测试，含开源边界自证与导出校验 |

## 常用命令

```bash
python -m agent.benchmark.fixtures --n 12 --seed 20261112
python -m agent.benchmark.run --input agent/fixtures/public/base_v1.json --arms A0 --verify-arm A0
python -m pytest
```
