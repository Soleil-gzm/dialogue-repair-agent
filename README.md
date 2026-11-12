# dialogue-repair-agent

**Reproducible agent evaluation for dialogue data repair.**

用 LLM Agent 修复模板拼接式对话的跨轮断裂问题，并用一套**可复现的评测体系**证明它有效。

当前状态：`M3 真实调用（进行中）` —— 已具备四元组基线指纹、不可变基线锁、可复现对照臂、可插拔插件机制、
五条保真约束 + `check_consistency`、连贯性检查 `check_coherence`（重复空转 / 退化轮次）、
6 类程序化缺陷注入集、决策循环（预算 / 死循环保护 / Trace），以及 DeepSeek 客户端
（重试分类 + 版本化价格表 + token 账本 + 响应缓存）。

> 本仓库由主工程的 `agent/` 目录按 `oss_manifest.txt` 导出，不含任何业务数据或内部文档。
> 数据全部程序化合成（中立域：订单售后回访）。

---

## 它解决什么问题

模板拼接生成的对话通常"局部通顺、全局不连贯"：每一句都像人话，但上下轮接不上。
直接上 LLM 重写会遇到三个问题：

1. **改坏了没人知道** —— 金额、日期、承诺被改写，结构被打乱。
2. **改好了没人证明** —— 没有基线、没有对照、没有区间，"感觉更顺了"不是结论。
3. **成本说不清** —— 不知道多花的 token 换来了多少质量增量。

这个仓库先解决第 2、3 条：**在动手改数据之前，先把"怎么算改好了"定义清楚。**

---

## 核心设计：基线是一组指纹，不是一次结果

```
基线 = ⟨输入指纹, 配置指纹, 评估器指纹, 适配指纹⟩
```

| 指纹 | 内容 | 防止什么 |
|---|---|---|
| 输入 | 内容 sha256 + 记录数 + 抽样 seed | 换了数据还叫"复现" |
| 配置 | 剥离易变键后的配置哈希 | 改了参数不自知 |
| 评估器 | 规则/judge/rubric 版本 | 换尺子量新旧结果 |
| 适配 | adapter + schema + 域配置哈希 | 同一文件、不同形态 |

`baseline_lock.json` **一经生成即不可覆盖**。变更必须显式 `--force --reason`，
并自动写入 `lock_changelog.md`。这让"基线偷偷漂移"在流程上不可能发生。

---

## 对照臂

每个相邻臂只差一个变量，保证结论可归因：

| 臂 | 处理 | 回答的问题 | 可独立运行 |
|---|---|---|---|
| **A0** | 不处理 | 下界；任何方法比它差都是失败 | ✅ |
| **A1** | 现有规则增强 | 相对现有能力有无增量 | ⬜ 需上游工程 |
| A2 | 单次 LLM 重写 | "LLM 重写"本身值不值 | ⬜ M3 |
| A3 | Agent 工具循环 | 工具循环比固定重写强多少 | ⬜ M3 |

> A1 复用上游数据工程的增强步骤，因此不在本仓库可运行范围内；
> 它的指纹仍完整记录在 lock 里，作为"测过什么"的凭据。

---

## 快速开始

```bash
git clone https://github.com/<your-account>/dialogue-repair-agent
cd dialogue-repair-agent

python -m pip install -e ".[dev]"

# 生成公开轨数据（同 seed 必然同产物）
python -m agent.benchmark.fixtures --n 12 --seed 20261112

# 跑 A0 并复现冻结基线
python -m agent.benchmark.run \
  --input agent/fixtures/public/base_v1.json \
  --arms A0 --verify-arm A0

# 测试
python -m pytest
```

无任何外部依赖（仅标准库）；Python ≥ 3.10。

---

## 目录

```
agent/
├── benchmark/
│   ├── canonical.py      规范化序列化 + sha256（唯一哈希入口）
│   ├── fingerprint.py    四元组指纹
│   ├── lock.py           不可变 lock + 校验
│   ├── arms.py           A0 恒等臂 / A1 上游增强臂
│   ├── fixtures.py       中立域程序化数据生成
│   ├── defects.py        6 类缺陷注入（答非所问/指代断裂/事实冲突/角色混淆/重复空转/缺失承接）
│   ├── run.py            CLI
│   ├── baseline/         入库：冻结的基线
│   └── output/           不入库：运行产物
├── runtime/
│   ├── llm.py            LLM 客户端协议 + FakeLLMClient
│   ├── deepseek.py       DeepSeek 客户端（重试分类 / 缓存 / 逐次计量）
│   ├── config.py         .env 读取与 LLMSettings（密钥只走环境变量）
│   ├── ledger.py         token 账本（append-only，缓存命中成本归零）
│   ├── registry.py       工具注册表 + 工具契约
│   ├── loop.py           决策循环（预算 / 死循环识别 / 终止原因）
│   ├── policy.py         决策策略（规则型；LLM 策略待接入）
│   └── state.py / budget.py / trace.py / checkpoint.py
├── eval/
│   ├── entities.py       实体抽取（编号/电话/金额/日期，领域无关）
│   ├── rules.py          五条硬约束：system / 结构 / 实体 / 说话人 / 跨角色泄漏
│   └── coherence.py      连贯性：相邻轮重复空转、退化轮次（语义判官可注入）
├── tools/
│   ├── check_consistency.py  规则型工具（可脱离 LLM 运行）
│   ├── check_coherence.py    连贯性工具（默认离线，判官可注入）
│   └── fallback_keep.py      保底回退到原对话
├── fixtures/public/      公开轨数据（base_v1 / defects_v1）
├── docs/                 设计文档（基准方案、路线）
├── tests/                测试，含开源边界自证
├── oss_manifest.txt      发布清单（PUBLIC / BUILD / PRIVATE）
└── scripts/export_oss.py 按清单导出，防误推
```

> 与主工程耦合的 `agent/plugin/`（`AgentAugmentStep` 胶水层）**不在本仓库**：
> 它依赖上游工程的 `PipelineStep`，只在公司内部项目中使用。

---

## 不变量（已用测试固化，不靠承诺）

| 不变量 | 对应测试 |
|---|---|
| 同 seed 生成同数据 | `test_export` / fixtures |
| A0 产物指纹与冻结值一致 | `test_matches_frozen_fingerprint` |
| lock 不可静默覆盖 | `test_refuses_overwrite_on_change` |
| `--force` 必须留痕 | `test_force_with_reason_overwrites_and_logs` |
| 配置指纹剔除易变键 | `test_volatile_keys_do_not_change_fingerprint` |
| 发布文件无业务标识 | `test_public_files_have_no_business_terms` |
| 发布代码不依赖上游工程 | `test_public_python_does_not_import_main_project_at_top_level` |
| 导出不会带出私有文件 | `test_private_files_excluded` |
| 事实冲突 100% 拦截、干净样本 0% 误杀 | `test_fact_conflict_blocked_100pct`、`test_false_positive_rate_zero` |
| 重复空转 100% 拦截、干净样本 0% 误报 | `test_d5_blocked_100pct`、`test_clean_samples_not_flagged` |

---

## 路线

`T0 测试地基 ✅ → T1 基线冻结 ✅ → M0 插件骨架 ✅ → M1 单工具 ✅ → M2 决策循环 ✅ → M3 真实调用 🚧 → M4 开源切片 → M5 作品集`

M3 剩余：语义判官接入（D1 答非所问）、LLM 决策策略与规则策略消融、
全量真实指标测量（各类缺陷拦截率、干净样本误报率、token 成本）。

---

## 许可

[Apache-2.0](LICENSE)
