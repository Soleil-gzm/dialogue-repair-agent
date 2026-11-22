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
| **A3R** | A3 + `rewrite_segment` 多轮修复 | 修复工具相对只回退的净收益 | ✅ 见下节消融 |
| **A4** | A3L + 语义判官 | 判官是否值得 | ✅ 见下节消融 |
| **A4R** | A4 + `rewrite_segment` | LLM 决策拿到完整工具集后还能提多少 | ✅ 见下节消融 |

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

## 消融结果（2026-11-22 公开轨全量重跑：本地 Ollama `qwen3:8b`）

> **模型溯源**：生成与判官均为本地 Ollama `qwen3:8b`
> （`http://127.0.0.1:11434/v1`，`temperature=0`、`max_tokens=2048`），
> 模型文件 `digest 500a1f067a9f…`、`Q4_K_M`、8.2B、5.2GB。
> 公司电脑与本地电脑的 Ollama 模型可能不同，`manifest_*.json` 的
> `models.generate` / `models.judge` 已记录 provider、接入点、digest、量化方式与大小；
> 引用数字前先核对 digest。A3 为离线规则臂，不调用模型，其 `models` 为 `null` 属预期。
>
> 历史 `deepseek-flash` 表已由本表替换；历史数字不可与本表混读。

复现命令（PowerShell）：

```powershell
$env:DEEPSEEK_API_KEY="ollama"
$env:DEEPSEEK_BASE_URL="http://127.0.0.1:11434/v1"
$env:DEEPSEEK_MODEL="qwen3:8b"
$env:JUDGE_MODEL="qwen3:8b"
$env:JUDGE_BASE_URL="http://127.0.0.1:11434/v1"
$env:LLM_PRICE_TABLE="local"
$env:DEEPSEEK_TIMEOUT="300"

$dir = "logs/agent/2026-11-22-public-ollama"
python -m agent.benchmark.agent_eval --arm A3  --out "$dir/ablation_A3.json"
python -m agent.benchmark.agent_eval --arm A3L --live --out "$dir/ablation_A3L.json"
python -m agent.benchmark.agent_eval --arm A3R --live --out "$dir/ablation_A3R.json"
python -m agent.benchmark.agent_eval --arm A4  --live --out "$dir/ablation_A4.json"
python -m agent.benchmark.agent_eval --arm A4R --live --out "$dir/ablation_A4R.json"
python -m agent.benchmark.agent_eval --arm A2  --live --out "$dir/ablation_A2.json"
python -m agent.benchmark.report --results $dir --out "$dir/report"
```

> 口径：120 条注入缺陷（6 类 × 20）+ 12 条干净对话，单轮全量。
> `blocked` = 最终回退原文；`repaired` 是改写后复检通过，不冒充回退。
> 成本为 ¥0 是因为 `LLM_PRICE_TABLE=local`；token 仍来自 append-only ledger。

| 臂 | 决策 | 工具 | D1 | D2 | D3 | D4 | D5 | D6 | 误报 | macro | tokens | 调用 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 | 单次重写 | 无 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 10/12 | 0.0% | 157,507 | 118 |
| A3 | 规则 | 确定性 | 0/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0/12 | 83.3% | 0 | 0 |
| A3L | LLM | 确定性 | 0/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0/12 | 83.3% | 233,376 | 284 |
| A3R | 规则 | 确定性 + rewrite | 0/20 | 20/20 | 20/20 | 6/20 | 10/20 | 20/20 | 0/12 | 63.3% | 40,729 | 54 |
| A4 | LLM | 确定性 + 判官 | **18/20** | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0/12 | **98.3%** | 273,798 | 332 |
| A4R | LLM | 全工具集 | 6/20 | 20/20 | 20/20 | 20/20 | 15/20 | 20/20 | 0/12 | 84.2% | 429,264 | 468 |

**读表注意**

- **A2 的 `blocked` 恒为 0**：它没有回退能力。上表 A2 的 D 列为 0 不代表"没看见"，
  其 `detected` 统计的是"输入侧可被确定性规则看见"的比例；干净样本干预率 83.3%（10/12），
  另有 7 条解析失败。
- **A3R/A4R 的 `blocked` 低于 A3/A4 不是退化**：修复成功的样本计入 `repaired`，不再回退。
  A3R 修复 24 条（D4 14、D5 10，repair_rate 24%）；A4R 修复 16 条（D1 12、D5 4，repair_rate 13.3%）。
  `repaired` 只证明"改写后通过校验"，必须人工抽检后才能当作真修复引用。
- **覆盖缺口**：A3L / A4 / A4R 分别有 2 / 2 / 3 条样本运行时 `failed`（异常结束），
  已计入独立字段；判官故障均为 0，六个臂 `valid=true`，`judge_error_rate=0%`。
- **干净样本**：除 A2 外，五个臂均为 0/12 误报，没有用误伤换召回。

三个结论：

1. **只换决策者（A3 → A3L）零增益**：宏观阻断率同为 83.3%，却多 284 次 LLM 调用。
   这套工具集上瓶颈不在"选哪个工具"，而在工具能力本身。
2. **语义判官（A3L → A4）是唯一能覆盖 D1 的层**：D1 从 0/20 到 18/20，
   宏观阻断 83.3% → 98.3%（+15.0pp）。确定性规则对"答非所问"无能为力。
3. **修复工具改变的是"回退 vs 修复"的分配，不一定提高 blocked**：
   A3 → A3R 把 24 条从回退变为修复，但 blocked 降到 63.3%；
   A4R 检出 100%，修复率却只有 13.3%，D4 仍全部回退。
   因此评价修复臂必须同时看 `repaired` 与人工抽检，不能只看 `blocked`。

产物与报告：`logs/agent/2026-11-22-public-ollama/`
（`ablation_*.json`、`manifest_*.json`、`report/benchmark_report.{json,md,html}`、`report/figures/*.svg`）。
`logs/` 与 `agent/benchmark/output/` 均为本地生成产物，不入库；按上面的命令可逐项复算。

> **指标口径说明（2026-11-20 修订）**
>
> - `blocked` 只统计最终回退原文；`rewrite_segment` 修复成功计入 `detected` 与 `repaired`，
>   不冒充"挡住并保住数据"。
> - 干净样本干预率包含 `fallback`、`failed`、`stopped` 与"误报后改写成功"，
>   任何动过或没跑完的干净样本都计入分母。
> - A2 的 outcome 按 metric 分流：`blocked` 恒为 0，`no_leak` 走独立判官复检，
>   不再用一致性校验冒充所有指标。
> - 判官故障单列 `judge_error`，不进入 `detected` / `blocked`；故障率 >5% 时该臂结果标记为不可采信。
> - 本表未包含 McNemar 显著性检验；跨臂显著性需另跑 `--compare` 产物。

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
