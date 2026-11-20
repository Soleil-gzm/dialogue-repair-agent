# dialogue-repair-agent

[![CI](https://github.com/Soleil-gzm/dialogue-repair-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Soleil-gzm/dialogue-repair-agent/actions/workflows/ci.yml)

Portfolio: https://soleil-gzm.github.io/portfolio/cases/agent/

**Reproducible agent evaluation for dialogue data repair.**

用 LLM Agent 修复模板拼接式对话的跨轮断裂问题，并用一套**可复现的评测体系**证明它有效。

当前状态：`Sprint 0 ✅ / Sprint 1a ✅ / Sprint 1b 进行中`（M0–M4 完成，M5 作品集目录待补齐）—— 已具备四元组基线指纹、不可变基线锁、可复现对照臂、
五条保真约束 + `check_consistency`、连贯性检查 `check_coherence`（重复空转 / 退化轮次）、
语义判官（D1 答非所问，可注入、prompt 版本化）、6 类程序化缺陷注入集、
决策循环（预算 / 死循环保护 / Trace）与两种可消融的决策策略（规则型 / LLM function calling）、
**可回退修复闭环**（`rewrite_segment` + A3R 多轮策略）、dev/test 分层切分、
`generate` / `judge` 分侧路由，以及 DeepSeek 客户端（重试分类 + 版本化价格表 + token 账本 + 响应缓存）。

Sprint 1b 已完成 A2 单次重写臂与 L3 统计层（Wilson CI + 精确 McNemar）；剩余 replay 与 checkpoint 批量接入。
完整设计见 [Agent 文档中心](agent/docs/README.md)。

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

| 臂 | 处理 | 回答的问题 | 状态 |
|---|---|---|---|
| **A0** | 不处理 | 下界；任何方法比它差都是失败 | ✅ |
| ~~A1~~ | 上游规则增强（加噪） | 已移出主链，仅作鲁棒性噪声源 | ⛔ 2026-11-19 移除 |
| **A2** | 单次 LLM 重写 | "LLM 重写"本身值不值 | ✅ 已实现 |
| **A3** | Agent 工具循环 + 规则决策 | 工具循环比固定重写强多少 | ✅ 见下节消融 |
| **A3L** | A3 的工具集 + LLM 决策 | 只换决策者是否值得 | ✅ |
| **A3R** | A3 + `rewrite_segment` 多轮修复 | 修复工具相对只回退的净收益 | ✅ 已实现，待 L3 复测 |
| **A4** | A3L + 语义判官 | 判官是否值得 | ✅ 需在线密钥 |
| **A4R** | A4 + `rewrite_segment` | LLM 决策拿到完整工具集后还能提多少 | ✅ 已实现，待评测 |

> **A1 已移除**：它做的是上游 `05_augment` 加噪（语气词/同义替换/语序），与"语义检测 + 改写"不是同一个任务；
> 产物是变体列表，`n_out ≠ n_in` 无法配对，且脱离上游环境不可复现。旧指纹归档在
> `agent/benchmark/baseline/deprecated/a1_2026-11-19.json`，原因见
> [A1 臂移除记录](agent/docs/notes/2026-11-19-A1臂移除记录.md)。
> 加噪能力转为鲁棒性实验的噪声源（同一 Agent 跑 clean vs noisy），不浪费。
> A2 已实现；"A3 vs A2"的主结论以 `agent_eval --compare A3,A2` 的全量产物为准。

---

## 快速开始

```bash
git clone https://github.com/<your-account>/dialogue-repair-agent
cd dialogue-repair-agent

# 离线安装（A0/A3 与测试）；在线臂改用 ".[dev,llm]" 或 uv sync --all-extras
python -m pip install -e ".[dev]"

# 生成公开轨数据（同 seed 必然同产物）
python -m agent.benchmark.fixtures --n 12 --seed 20261112

# 生成 dev / test 分层切分（30 + 50，互不重叠）
python -m agent.benchmark.fixtures \
  --dev-out agent/fixtures/public/dev_v1.json \
  --test-out agent/fixtures/public/test_v1.json

# 跑 A0 并复现冻结基线
python -m agent.benchmark.run \
  --input agent/fixtures/public/base_v1.json \
  --arms A0 --verify-arm A0

# 测试
python -m pytest
```

离线路径零第三方依赖（仅标准库）；在线臂需要 `httpx` / `tenacity`。Python ≥ 3.10。
完整命令与排障见 [部署与运行指南](agent/docs/guides/部署与运行指南.md)。

---

## 目录

```
agent/
├── benchmark/
│   ├── canonical.py      规范化序列化 + sha256（唯一哈希入口）
│   ├── fingerprint.py    四元组指纹
│   ├── lock.py           不可变 lock + 校验
│   ├── manifest.py       运行清单（环境 / 命令 / 产物哈希，不参与判等）
│   ├── arms.py           A0 恒等臂
│   ├── fixtures.py       中立域程序化数据生成 + dev/test 分层切分
│   ├── defects.py        6 类缺陷注入（答非所问/指代断裂/事实冲突/角色混淆/重复空转/缺失承接）
│   ├── run.py            基线 CLI（A0 与 lock 校验）
│   ├── agent_eval.py     Agent 臂评测 CLI（A3/A3L/A3R/A4/A4R：召回 / 误报 / 成本）
│   ├── report.py         作品集报告生成（三件套 + SVG，纯标准库）
│   ├── baseline/         入库：冻结的基线
│   └── output/           不入库：运行产物
├── runtime/
│   ├── llm.py            LLM 客户端协议 + FakeLLMClient
│   ├── deepseek.py       DeepSeek 客户端（重试分类 / 缓存 / 逐次计量 / 分侧路由）
│   ├── config.py         .env 读取与 LLMSettings（密钥只走环境变量）
│   ├── pricing.py        版本化价格表
│   ├── ledger.py         token 账本（append-only，缓存命中成本归零）
│   ├── registry.py       工具注册表 + 工具契约
│   ├── loop.py           决策循环（预算 / 死循环识别 / 终止原因）
│   ├── policy.py         决策策略（规则型 / LLM function calling）
│   └── state.py / budget.py / trace.py / checkpoint.py
├── eval/
│   ├── entities.py       实体抽取（编号/电话/金额/日期，领域无关）
│   ├── rules.py          五条硬约束：system / 结构 / 实体 / 说话人 / 跨角色泄漏
│   ├── coherence.py      连贯性：相邻轮重复空转、退化轮次（语义判官可注入）
│   └── judge.py          语义判官：判定答非所问（D1），prompt 版本化
├── tools/
│   ├── check_consistency.py  规则型工具（可脱离 LLM 运行）
│   ├── check_coherence.py    连贯性工具（默认离线，判官可注入）
│   ├── fallback_keep.py      保底回退到原对话
│   └── rewrite_segment.py    定位并重写指定轮次（A3R 专用）
├── fixtures/public/      公开轨数据（base_v1 / dev_v1 / test_v1 / defects_v1）
├── docs/
│   ├── README.md         文档总索引
│   ├── architecture/     架构设计 / 模块与接口 / Benchmark 评测设计
│   ├── decisions/        ADR / 改造路线图
│   ├── guides/           部署与运行指南
│   ├── notes/            过程笔记（术语 / 决策 / 实验复盘）
│   └── archive/          历史版本与版本追溯
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

## 消融结果（`deepseek-flash`，120 条缺陷 + 12 条干净对话）

复现命令：

```bash
python -m agent.benchmark.agent_eval --arm A3 --out agent/benchmark/output/ablation_A3.json
python -m agent.benchmark.agent_eval --arm A3L --live   # 需密钥，会消耗 token
python -m agent.benchmark.agent_eval --arm A3R --live
python -m agent.benchmark.agent_eval --arm A4  --live
python -m agent.benchmark.agent_eval --arm A4R --live
```

> 口径与溯源：下表为 120 条注入缺陷 + 12 条干净对话的单轮全量运行，成本来自 append-only ledger（高峰价）。
> 结果 JSON 目前未入库（`output/` 被 gitignore），因此这些数字还无法按指纹复核；
> 计划将结果快照与环境 manifest 一并提交，届时每个数字都可复算。

| 臂 | 决策 | 工具 | D1 | D2 | D3 | D4 | D5 | D6 | 误报 | macro | tokens | 成本 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | 规则 | 确定性 | 0/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0/12 | 83.3% | 0 | ¥0 |
| A3L | LLM | 确定性 | 0/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0/12 | 83.3% | 251,561 | ¥0.40 |
| A4 | LLM | 确定性 + 判官 | **20/20** | 20/20 | 20/20 | 20/20 | **20/20** | 20/20 | 0/12 | **100%** | 305,910 | ¥0.55 |

\* A4 首轮 D5 为 19/20：评测期间宿主机进入 Modern Standby 39 分钟，
恢复后该条的墙钟预算触发中止（检出但未回退）；单独复跑该条为 blocked。
评测器现在把这类预算中止计入独立的 `stopped` 字段，不会混进"模型没检出"。

> **指标口径说明（2026-11-20 修订）**
>
> - `blocked` 只统计最终回退原文；`rewrite_segment` 修复成功计入 `detected` 与 `repaired`，
>   不冒充"挡住并保住数据"。
> - 干净样本干预率现在包含 `fallback`、`failed`、`stopped` 与"误报后改写成功"，
>   任何动过或没跑完的干净样本都计入分母。
> - A2 的 outcome 按 metric 分流：`blocked` 恒为 0，`no_leak` 走独立判官复检，
>   不再用一致性校验冒充所有指标。
> - 判官故障单列 `judge_error`，不进入 `detected` / `blocked`；故障率 >5% 时该臂结果标记为不可采信。
> - 上表为历史运行结果；修复后的全量结果以 `logs/agent/2026-11-20-full/` 下的产物与 `--compare` 为准。

三个结论：

1. **只换决策者（A3 → A3L）零增益，却多花 ¥0.40**。这套工具集上瓶颈不在"选哪个工具"，而在工具能力本身。
2. **语义判官（A3L → A4）把 D1 从 0 拉到 100%**，宏观阻断 83.3% → 100%，增量成本 ¥0.15。
3. **指标要能区分"没检出"和"被环境打断"**：39 分钟卡顿经查是 Windows 新型待机
   （22:49:06 进入、23:28:27 退出，与两个进程的阻塞区间完全吻合），不是模型或接口问题 ——
   同一请求复跑只需 2–3.5 秒。若不区分，就会把一次待机误判成决策层不遵守流程。

成本口径来自 append-only ledger（高峰价）。单条成本 A4 约 ¥0.0042。
计时数据受上述待机污染，不作为指标；token 与成本不受影响。

---

## 路线

历史里程碑：`T0 测试地基 ✅ → T1 基线冻结 ✅ → M0 插件骨架 ✅ → M1 单工具 ✅ → M2 决策循环 ✅ → M3 真实调用 ✅ → M4 开源切片 ✅ → M5 作品集 ⬜（目录待补齐）`

当前 Sprint（以 [改造路线图](agent/docs/decisions/改造路线图.md) 为准）：

`S0 基线加固 ✅ → S1a 修复闭环 ✅ → S1b 效果层与回放（部分完成）→ S2 服务化 → S3 RAG`

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | `baseline_lock.json` v2.0、运行清单、CI 指纹复现 | ✅ |
| S1a | `rewrite_segment`、A3R 多轮修复、`generate/judge` 分侧路由、dev/test 切分 | ✅ |
| S1b | A2 臂、L3 胜率与统计 | ✅ |
| S1b | replay、checkpoint 批量接入 | ⬜ |
| S2 | MCP Server、Docker、OpenTelemetry | ⬜ |
| S3 | RAG 增强（当前仅 `A5-RAG` 命名占位） | ⬜ |

作品集产物（三件套报告 + 图 + trace 样例）目标位置为 `portfolio/agent-augmentation/`：
`benchmark_report.{html,md,json}`、`figures/*.svg`、`04_trace_sample.jsonl`、`05_before_after.md`。
该目录当前尚未生成，不能作为已交付产物引用。

## 文档

| 文档 | 内容 |
|---|---|
| [文档总索引](agent/docs/README.md) | 阅读路径、权威顺序、文档规范与校验清单 |
| [架构设计](agent/docs/architecture/架构设计.md) | 整体架构、模块划分、核心流程、技术选型、非功能设计 |
| [模块与接口说明](agent/docs/architecture/模块与接口说明.md) | 工具契约、运行时契约、CLI、数据与产物字段 |
| [Benchmark 评测设计](agent/docs/architecture/Benchmark评测设计.md) | 基线指纹、指标口径、统计与报告规格 |
| [决策记录 ADR](agent/docs/decisions/决策记录ADR.md) | 关键取舍与边界 |
| [改造路线图](agent/docs/decisions/改造路线图.md) | Sprint 排序与状态 |
| [项目笔记](agent/docs/notes/README.md) | 统计术语、A1 移除记录、效果验证与鲁棒性实验 |
| [部署与运行指南](agent/docs/guides/部署与运行指南.md) | 安装、数据生成、离线复现、在线评测、排障 |

---

## 许可

[Apache-2.0](LICENSE)
