# Agent 增强层 Benchmark 评测设计

> 版本：v1.1  
> 首次版本：v1.0（2026-11-12，原文见 [归档快照](../archive/v1.0-2026-11-12/Benchmark专项设计方案.md)）  
> 最后修订：2026-11-18  
> 归属：`agent/docs/architecture/`  
> 定位：**独立可执行方案**。本文是 Benchmark 指标口径、统计规则与产物规格的唯一权威来源。  
> 本期范围：不包含 RAG（已移出），对照臂命名预留 `A5-RAG`。

---

## 0. 本方案要回答的问题

一个 Benchmark 只有在能回答下面三句话时才有价值：

1. **比谁？** —— 一组单变量对照臂，每个相邻臂只差一个变量（§1）。
2. **凭什么算数？** —— 输入/配置/评估器/适配四元组指纹锁定 + 八条可比性规则 + 统计检验（§2、§6）。
3. **错了怎么办？** —— 硬门禁不可交换、失败样本计入分母、负面结果照报（§4.5、§7.4、§10）。

> 反例（绝大多数作品集的做法）：跑 20 条，觉得"改完更顺了"，写"连贯性提升 30%"。这条路一站不住脚，因为**没有基线、没有分母、没有区间**。

### 0.1 实现状态（截至 Sprint 1a）

| 能力 | 状态 | 落地位置 |
|---|---|---|
| 公开轨 `base/dev/test`、`defects` 生成 | 已实现 | `benchmark/fixtures.py`、`defects.py` |
| A0 恒等臂 | 已实现 | `benchmark/arms.py`、`run.py` |
| A1 规则增强臂 | 已移出主链（2026-11-19） | 归档 `benchmark/baseline/deprecated/a1_2026-11-19.json` |
| A2 单次重写臂 | 已实现（`run_a2_dialogue`） | 全量结果见 `logs/agent/2026-11-20-full/` |
| A3 / A3L / A3R / A4 | 已实现 | `benchmark/agent_eval.py` |
| L1 规则、连贯性、语义判官 | 已实现 | `eval/rules.py`、`coherence.py`、`judge.py` |
| L2 预算、Trace、死循环保护 | 已实现 | `runtime/budget.py`、`trace.py`、`loop.py` |
| L3 胜率区间、McNemar | 已实现（Wilson CI + 精确二项） | 双盲、人工校准仍未实现 |
| 指纹、不可变锁、运行清单、ledger | 已实现 | `benchmark/fingerprint.py`、`lock.py`、`manifest.py`、`runtime/ledger.py` |
| replay / cassette / checkpoint 批量接入 | 未实现（S1b） | — |
| 报告三件套 + SVG | 已实现 | `benchmark/report.py` |
| Docker / MCP / OTel | 未实现（S2） | — |

> 本文后续章节区分“设计要求”与“当前实现”。凡未实现项，不得在报告或 README 中写成已完成。

---

## 1. 基线对比标准（本方案的核心）

### 1.1 什么才算"一个合法的基线"

基线不是"随手跑一次留个结果"，而是**指纹锁定的四元组**：

```
基线 = ⟨输入指纹, 配置指纹, 评估器指纹, 适配指纹⟩
```

| 指纹 | 内容 | 存储 |
|---|---|---|
| 输入指纹 | 输入 JSON 的 `sha256`、样本数、抽样键列表的 `sha256`、抽样 seed | `baseline_lock.json` |
| 配置指纹 | 该臂完整配置的规范化 JSON 的 `sha256`、prompt 版本、policy、模型名、价格表版本 | 同上 |
| 评估器指纹 | 规则评估器版本、judge prompt 版本、judge 模型、rubric 版本 | 同上 |
| 适配指纹 | `adapter_version` + `canonical_schema_version` + `domain_profile` / `domain_profile_hash` | 同上（当前无独立 adapter 模块，见 §3.5） |

实际 lock 为 **v2.0**，字段定义以 [模块与接口说明](模块与接口说明.md#91-baseline_lockjsonv20) 为准。示意：

```json
{
  "lock_version": "2.0",
  "generated_at": "<生成时间；校验时忽略>",
  "code": {"git_commit": "<sha>", "dirty": false},
  "input": {
    "path": "agent/fixtures/public/base_v1.json",
    "sha256": "<64hex>",
    "n_total": 12,
    "n_sampled": 12,
    "sample_keys_sha256": "<64hex>",
    "sample_seed": 42,
    "strata": ["<分层维度>"]
  },
  "evaluator": {
    "rules_version": "1.0.0",
    "rules_sha256": "<64hex>",
    "judge_prompt_version": "v1",
    "judge_prompt_sha256": "<64hex>",
    "judge_model": "deepseek-chat",
    "rubric_version": "1.0",
    "judge_temperature": 0.0,
    "sha256": "<64hex>"
  },
  "adapter": {
    "adapter_version": "1.0.0",
    "canonical_schema_version": "1.0",
    "domain_profile": "public_after_sales",
    "domain_profile_hash": "<64hex>",
    "sha256": "<64hex>"
  },
  "price_table": {"version": "deepseek_2026-11", "unit": "CNY/1M tokens"},
  "arms": {
    "A0": {"config_sha256": "<64hex>", "prompt_version": null, "artifact_sha256": "<64hex>"}
  },
  "profile": "standard"
}
```

**规则：`baseline_lock.json` 一经生成即不可修改。** 任何一项指纹变化都必须新建 lock 文件并说明原因（写入 `lock_changelog.md`）。这是"基线不会偷偷变"的唯一保证。

### 1.2 对照臂

| 臂 | 名称 | 输入 | 处理方式 | 回答的问题 | 实现状态 |
|---|---|---|---|---|---|
| **A0** | 原始基线 | 同一输入（默认 `base_v1.json`） | 不处理 | 下界。任何方法比它差都是失败 | 已实现 |
| **A2** | 单次重写基线 | 同 A0 | 1 次 LLM 重写，无工具、无校验、无回退 | 隔离"工具循环"的贡献 | 已实现 |
| **A3** | Agent-rule（主方案） | 同 A0 | 工具循环 + 确定性策略 + 校验 + 回退 | 主方案是否成立 | 已实现 |
| **A3L** | Agent-rule-LLM | 同 A0 | A3 的工具集，决策者换成 LLM function calling | 只换决策者的净收益 | 已实现 |
| **A3R** | Agent-rule-repair | 同 A0 | A3 + `rewrite_segment` 多轮修复（检出→定位→重写→复检） | 修复工具相对只回退的净收益 | 已实现（S1a） |
| **A4** | Agent-llm（增强） | 同 A0 | A3L + 语义判官 | 判官值不值那份成本 | 已实现，需在线密钥 |
| (A5-RAG) | — | — | — | — | **已移出本期** |

> `A3L` 与 `A4` 是 Sprint 1a 实际使用的臂名；旧文档中的“A4 = LLM 决策”已拆分为 `A3L`（决策）与 `A4`（决策 + 判官），避免一次变动两个变量。

> **A1 为什么移出（2026-11-19）**：A1 是上游 `05_augment` 的加噪增强（插语气词 / 同义替换 / 调整语序），做的是"造噪声"，与本期"语义检测 + 改写"不是同一个任务；其产物是变体列表，`n_out ≠ n_in`，无法按 `dialogue_key` 配对，胜率与 McNemar 都不成立；且产物未入库、当前环境无 `pipeline` 包，无法复现。旧指纹已归档，A1 的用途转为鲁棒性实验的噪声源（见 §3.6）。详细原因见 [A1 臂移除记录](../notes/2026-11-19-A1臂移除记录.md)。

### 1.3 单变量原则（合法性判据）

每个臂只能与"相邻臂"相差一个变量，否则结论无法归因：

| 对比 | 唯一变量 | 可归因的结论 |
|---|---|---|
| A2 vs A0 | 是否引入 LLM 单次重写 | LLM 重写本身的净收益 |
| A3 vs A2 | 是否有"检查→校验→回退"闭环 | **工具循环的净收益（主结论）** |
| A3L vs A3 | 决策由规则还是模型做出 | 只换决策者的净收益 |
| A4 vs A3L | 是否加入语义判官 | 判官的净收益 |
| A3R vs A3 | 是否有"定位→重写→复检"修复闭环 | 修复工具相对只回退的净收益 |
| A3 vs A0 | 全链路 | 端到端价值 |

**违反单变量原则的对比一律不出结论。** 例如"使用 LLM 重写的臂 vs 规则加噪输出"里同时变了任务与流程，只能作为参考，不写入结论。这也是 A1 不能留在主链的结构性原因：它同时改变了数据分布与任务类型。

### 1.4 预期单调性（预注册假设）

```
胜率：    A0 < A2 < A3 ≤ A3L ≤ A4
保真率：  A0 ≥ A3 = A3L = A4 > A2
成本：    A0 << A2 < A3 < A3L < A4
```

- `A3R` 是 A3 的修复分支，不进入上述单调链；只与 A3 做单变量对比。
- 预设"若 A3 胜率高于 A2 但保真率低于 99%"，则方案判定为**不可用**（硬门禁优先，见 §4.5）。
- 预注册的作用：**避免看到结果后改判据**。这是实验设计最基本的诚信要求，也是面试里最容易加分的一句话。

---

## 2. 可比性规则（八条硬约束）

| # | 规则 | 违反后果 | 落地方式 |
|---|---|---|---|
| R1 | 所有臂使用**同一输入文件**（sha256 一致） | 臂间不可比 | lock 校验失败即中止 |
| R2 | 所有臂使用**同一抽样键集合**与同一 seed | 抽样偏差被误读为方法差异 | 只抽一次键，所有臂共用 |
| R3 | 分层抽样：按 `scenario` × `turn_bucket` × `has_loss_flag` 分层 | 场景分布不一致 | 分层后按比例分配 |
| R4 | **同一评估器版本**评所有臂 | 标准漂移 | evaluator 指纹入 lock |
| R5 | 成对盲评：judge 不知道哪份是实验臂，**A/B 与 B/A 各评一次** | 位置偏差 | **设计目标，未实现（S1b）** |
| R6 | 每条样本在每个臂只出现一次 | judge 记忆污染 | 输入去重 + key 唯一性断言 |
| R7 | 失败样本**不回填、不丢弃**，计入分母 | 虚高 | 结果 JSON 单列 `failed` / `stopped`；L3 分母统一处理属 S1b |
| R8 | 所有臂用**同一价格表版本**折算成本 | 成本曲线被污染 | 价格表版本入 lock |

> R7 是最容易被忽略、也最能体现工程素养的一条。**"只统计成功的样本"是数据工作里最常见的自欺。**

---

## 3. 数据集与抽样

### 3.1 双轨 fixture（同时满足合规与开源）

| 轨 | 路径 | 内容 | 用途 | 是否入库/开源 |
|---|---|---|---|---|
| 公开轨 | `agent/fixtures/public/` | 由生成端程序化合成（零业务数据） | 开源版 Benchmark、CI、演示 | ✅ 入库 |
| 私有轨 | `agent/fixtures/private/` | 真实业务抽样（脱敏） | 内部效果验证 | 不入库（仅主工程） |

两轨使用**完全相同的字段契约**，因此同一份 harness 代码可跑两轨。本仓库只分发公开轨。

### 3.2 三个数据集

| 数据集 | 规模 | 构造方式 | 提供的标签 | 服务的指标层 |
|---|---|---|---|---|
| `defects_v1.jsonl` | 120（6 类 × 20） | 程序化注入缺陷 | 注入轮次 + 缺陷类型 | L1 |
| `dev_v1.json` | 30 | 与 test 一起分层切分，不含测试集 | 无（仅用于调 prompt） | 调参 |
| `test_v1.json` | 50 | 分层切分，与 dev 不重叠 | 当前无人工标注；4 维 rubric 标注属 L3（未实现） | L3 |

生成命令：

```bash
python -m agent.benchmark.fixtures --n 12 --seed 20261112
python -m agent.benchmark.fixtures \
  --dev-out agent/fixtures/public/dev_v1.json \
  --test-out agent/fixtures/public/test_v1.json
python -m agent.benchmark.defects --per-type 20 --seed 20261113 \
  --out agent/fixtures/public/defects_v1.jsonl
```

**dev / test 必须分离**，且 test 只在最终结论时用一次。当前切分在生成阶段一次性完成（默认 `split-seed 20261118`），CLI **没有** `--split test` 开关。

### 3.3 六类缺陷注入（L1 的 ground truth 来源）

| 类型 | 注入手法 | 期望工具行为 | 可自动化判定 |
|---|---|---|---|
| D1 答非所问 | 将某轮 assistant 替换为跨话题回复（与当轮诉求无承接） | 定位到该轮 | ✅ |
| D2 指代断裂 | 删除前文实体，保留"他/这个/那笔" | 定位到该轮 | ✅ |
| D3 事实冲突 | 篡改金额/日期，与 `system` 不符 | 定位 + 一致性拦截 | ✅ |
| D4 角色混淆 | 将客户话术移入 assistant | 定位到该轮 | ✅ |
| D5 重复空转 | 复制上一轮同义内容 | 定位到该轮 | ✅ |
| D6 缺失承接 | 删除回应用户诉求的轮次 | 定位到该轮 | ✅ |

定位判定：预测轮次与注入轮次距离 ≤ 1 视为命中（±1 容忍）。

### 3.4 样本量与统计功效（为什么是 50）

配对二分类比较（McNemar，α=0.05，power=0.8）的近似需求：

| 想检出的胜率差 | 所需样本量 n |
|---|---|
| 20 个百分点 | ≈ 50 |
| 15 个百分点 | ≈ 90 |
| 10 个百分点 | ≈ 200 |
| 5 个百分点 | ≈ 800 |

**结论：主线用 n=50，只能支撑"≥15–20 个百分点"级别的结论。** 报告里必须写明这一点，不能拿 n=50 去宣称 5 个点的差异。若后期需要更细结论，需扩到 n=200；`full` 档当前未实现。

### 3.5 数据字段契约与适配边界

两轨共用同一份对话契约：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "loss": "True"}
  ]
}
```

- `role` 固定为 `system` / `user` / `assistant`；`loss` 为可选字段，当前样本为字符串 `"True"`，规则校验按原值严格比较。
- `canonical_schema_version` 记录在 lock 与运行清单中；换域时字段、类型与层级**只增不改**，新增元数据不得改变既有字段语义。
- **当前没有独立 adapter 模块**：读取、抽样与规范化由 harness 直接完成；`adapter` 指纹记录的是 schema 版本与域配置元数据，不代表存在可插拔适配层。独立 adapter 属后续工作。
- 详细字段与产物定义见 [模块与接口说明](模块与接口说明.md#4-数据契约)。

### 3.6 私有轨（效果验证）与鲁棒性实验的边界

二者是两条独立实验轴，不能互相替代：

| | 私有轨效果验证 | 鲁棒性实验 |
|---|---|---|
| 数据 | 真实生产对话（脱敏） | 干净输入 + 同源噪声输入 |
| 变的是什么 | 数据分布 | 输入质量 |
| 回答的问题 | 真实分布上是否仍然有效 | 噪声下性能衰减多少 |
| 对照 | 公开轨 vs 私有轨指标并列 | 同一 Agent 在 clean vs noisy 上配对 |
| 结论用途 | 外部效度 | 稳定性 / 部署风险 |

- 私有轨仍按 §3.1 的双轨设计与 ADR D6 执行，不因 A1 移出而改变。
- 鲁棒性实验用上游 `05_augment` 作为**输入变换**（非对照臂）：同一臂分别跑干净输入与噪声输入，比较召回、误报、实体保真、成本的变化。配对键必须能回溯到原对话，否则退化为语料级聚合并在 `limitations` 声明。
- 详细设计见 [效果验证与鲁棒性实验](../notes/2026-11-19-效果验证与鲁棒性实验.md)。

---

## 4. 指标体系

四层，**每层都必须给出「数据来源」**，否则指标不可复算。

> 实现口径（2026-11-20）：L1/L2 已有对应代码；L3 的 Wilson CI、精确 McNemar 与 `--compare` 已实现，双盲与人工标注未实现；L4 的 token、成本、指纹复现已实现，P95 与缓存命中率尚未统计。

### 4.1 L1 能力层 —— 工具是否准确

| 指标 | 公式 | 数据来源 | 门槛 |
|---|---|---|---|
| 缺陷定位 P / R / F1 | 按 6 类分别算，再宏平均 | `defects_v1` 注入标签 vs 工具输出 | F1 ≥ 0.75 |
| 一致性拦截率 | 被 `check_consistency` 拒绝的 D3 数 / D3 总数 | 工具输出 | ≥ 0.99 |
| 误杀率 | 干净样本被判"有问题"的比例 | `test_v1` 干净集 | ≤ 0.05 |
| 结构不变率 | 结构完全一致的对话数 / 总数（全或无） | 规则比较 | = 1.00 |
| 实体保真率 | 保留实体数 / 原实体总数 | 实体抽取器 | ≥ 0.99 |

### 4.2 L2 决策层 —— 循环是否聪明

| 指标 | 公式 | 门槛 |
|---|---|---|
| 修复成功率 | 通过评估器的样本数 / 被标记样本数 | ≥ 0.70 |
| 平均工具调用数 | 总调用数 / 样本数 | ≤ 6 |
| 无效调用率 | 未改变状态的调用 / 总调用 | ≤ 0.15 |
| 预算遵守率 | 未超限样本 / 总数 | = 1.00 |
| 死循环次数 | 同状态重复决策计数 | = 0 |
| 决策可解释率 | 含 `reason` 字段的决策 / 总决策 | = 1.00 |
| 回退率 | `fallback_keep` 数 / 总数 | ≤ 0.10 |

### 4.3 L3 效果层 —— 数据是否真的变好（基础统计已实现；双盲/人工未实现）

| 指标 | 定义 | 门槛 | 备注 |
|---|---|---|---|
| 成对胜率 WR | `W / (W+L+T)` | ≥ 0.60 | 同时报 `WR_decided = W/(W+L)` 与 tie 率 |
| 连贯性提升 | 1–5 分均值差（同批评分者） | Δ ≥ +0.8 | 人工与 judge 分别报 |
| 实体保真率 | 同 L1 定义 | ≥ 0.99 | **硬门禁** |
| 结构不变率 | 同 L1 定义 | = 1.00 | **硬门禁** |
| 长度比 | 重写后字符数 / 原字符数 | 0.8–1.3 | 防"注水变长" |
| distinct-2 变化 | (后 − 前) / 前 | 降幅 ≤ 5% | 防同质化 |
| 跨样本重复率 | 不同对话间 4-gram 重复比例 | 降幅 ≤ 5% | **防"洗成一个味"** |

> `跨样本重复率` 是为本项目量身加的：原系统最大的病是"味道重复"，LLM 重写如果把它洗得更均匀，等于用更贵的方式犯了同一个错。

### 4.4 L4 工程层 —— 是否可用、可信、可复现（部分实现）

| 指标 | 定义 | 门槛 |
|---|---|---|
| 单对话 token（in/out/total） | 来自 token ledger | 记录，不设上限 |
| P50 / P95 延迟 | ledger 的 latency 分位 | P95 ≤ 20s(A3) / 60s(A4) |
| 千条成本 | 按价格表折算 | 记录 |
| 缓存命中率 | `cached=true` 行数 / 总行数 | ≥ 90%（重复运行时） |
| 重复计费数 | 断点恢复后重复调用数 | = 0 |
| 结果复现一致性 | 同 lock + 同 seed 两次运行的产物 sha256 | 100% 一致 |
| **禁用插件后基线一致** | 关闭插件产物 vs 改造前快照 | **字节级 0 差异** |

### 4.5 门禁与指标的关系（不可交换）

```
硬门禁（任一不过 → 方案不可用，不看胜率）：
  结构不变率 = 1.00
  实体保真率 ≥ 0.99
  死循环 = 0
  预算遵守率 = 1.00
  禁用后基线一致 = true

软指标（用于比较与讲故事）：
  胜率、连贯性分、成本、延迟
```

**硬门禁先于软指标**这条规则必须写进报告首页。它直接对应一个面试问题："如果改得更通顺但改坏了金额，你怎么判？"

---

## 5. 评估方法

### 5.1 三档评估器（成本递增，可信度递增）

> 当前实现：E1 规则已实现；E2 只有 D1 单视角语义判官（`eval/judge.py`），成对双向盲评未实现；E3 人工标注未开展。

| 档 | 方法 | 成本 | 用途 | 可信度 |
|---|---|---|---|---|
| E1 规则 | 结构/实体/长度/禁用词 | 0 | 全量跑，硬门禁 | 高（可解释） |
| E2 LLM judge | 多视角 prompt（连贯性 / 自然度 / 一致性）+ 成对比较 | 中 | 胜率主指标 | 中（需校准） |
| E3 人工 | 50 条 × 2 评分者，4 维 rubric | 高 | 校准 E2、出结论 | 高 |

### 5.2 E2 的偏差控制（不做这些，胜率不可信）

| 偏差 | 控制手段 |
|---|---|
| 位置偏差 | A/B 与 B/A 各评一次，矛盾记 tie |
| 长度偏好 | rubric 明写"不因更长而加分"；报告附 `winner 长度均值差` |
| 自我偏好 | judge 与生成用不同模型（若只能用同款，必须在局限中声明） |
| 提示泄漏 | judge prompt 不含臂标签、不含"这是重写版"字样 |
| 温度 | 0.0，并记录 |

### 5.3 E2 与 E3 的一致性校准

| 指标 | 门槛 | 不达标怎么办 |
|---|---|---|
| Cohen's κ（人工双评之间） | ≥ 0.6 | <0.4 说明 rubric 不清，重写 rubric 后重标 |
| judge 与人工一致率 | ≥ 75% | 不达标则 judge 只用于**排序**，不用于绝对分 |

### 5.4 人工标注 rubric（4 维，1–5 分）

| 维度 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| 跨轮连贯 | 明显接不上 | 能读懂但生硬 | 自然承接 |
| 角色一致 | 说话人错位 | 偶有出戏 | 完全贴合身份 |
| 事实保真 | 金额/日期错 | 有小偏差 | 完全一致 |
| 拟真度 | 明显模板味 | 尚可 | 像真实对话 |

---

## 6. 统计与结论判定规则

### 6.1 必须报告的统计量

| 指标类型 | 点估计 | 区间 | 检验 |
|---|---|---|---|
| 胜率 / 通过率（二分类） | 比例 | Wilson 95% CI | 配对 McNemar |
| 连贯性分（连续） | 均值差 | paired bootstrap 95% CI | Wilcoxon 符号秩 |
| 成本（连续） | 均值 / 中位 | bootstrap 95% CI | — |

### 6.2 结论用语规范（写报告、写简历都按这个来）

| 情况 | 允许的表述 | 禁止的表述 |
|---|---|---|
| CI 下界 > 0.5 且 CI 上界 < 1.0 | "A3 胜率 0.64（95% CI 0.51–0.76），显著优于 A2" | "提升 64%" |
| CI 跨 0.5 | "未观察到显著差异（n=50，功效不足以检出 <15pp 差异）" | "略有提升" |
| 硬门禁未过 | "方法不可用，胜率结论不成立" | 只报胜率 |

### 6.3 结论模板（报告与 README 共用）

```
在 n=50 的分层测试集上（lock: <sha8>）：
  A3 vs A2 胜率 0.64（95% CI 0.51–0.76，McNemar p=0.03）
  实体保真 0.997，结构不变 1.00（硬门禁通过）
  单条成本 A2 1,240 tokens / A3 3,860 tokens（+211%）
结论：工具循环带来约 14 个百分点的胜率增量，代价是 3.1 倍 token。
      在 n=50 下无法判断 <15pp 的差异，需扩样验证。
```

---

## 7. 产物规格

### 7.1 当前实际产物

```text
agent/benchmark/
├── baseline/                        # 入库：冻结的基线（唯一真源）
│   ├── baseline_lock.json           # 四元组指纹（只读，不可覆盖）
│   └── lock_changelog.md            # 指纹变更记录
└── output/                          # 不入库：每次运行的产物
    ├── ablation_A3.json             # agent_eval 结果
    ├── ablation_A3L.json
    ├── ablation_A3R.json
    ├── ablation_A4.json
    ├── manifest_A3.json             # 与结果同目录的运行清单
    └── report/
        ├── arm_results.json         # A0 臂产物指纹与规模
        └── run_manifest.json        # 环境与命令溯源
```

`report.py` 的输出目录（可与 `output/` 相同或独立）：

```text
<out>/
├── benchmark_report.json
├── benchmark_report.md
├── benchmark_report.html
└── figures/
    ├── 01_ablation.svg
    ├── 02_heatmap.svg
    ├── 03_trace_timeline.svg
    └── 04_before_after.svg
```

**目标产物（未实现，S1b/S2）**：`comparison_matrix.csv`、`significance.json`、`token_ledger_summary.json`、`traces/<sample_key>.jsonl`、PNG 资产目录。这些出现在旧版设计中，当前不得当作已生成文件引用。

> **为什么 baseline 单独放在 `output/` 外面**：`.gitignore` 已全局忽略 `output/`，而基线必须入库才能起到"锁"的作用。运行产物可丢弃，基线不可丢。

### 7.2 四张图分别讲什么（作品集就靠这四张）

| 图 | 文件 | X / Y | 一句话结论 |
|---|---|---|---|
| 消融 | `01_ablation.svg` | 单条 token / 阻断率 | 每多花一份 token 换来多少阻断率 |
| 工具能力 | `02_heatmap.svg` | 缺陷类型 / 检出 | 哪类缺陷查不出（诚实展示短板） |
| 决策链 | `03_trace_timeline.svg` | 时间 / 工具调用 | Agent 真的在"看情况决定"，不是固定流水线 |
| 前后对照 | `04_before_after.svg` | — | 具体改了哪几轮、为什么改 |

> fig2 一定要保留表现差的格子。**全是绿色的热力图没人信。**

### 7.3 JSON 顶层结构（目标结构，部分字段未实现）

> 当前 `benchmark_report.json` 只覆盖已实现的指标（各臂 per-type、macro、误报、token、成本、失败/中止）。下列 L3 与 `gates` 字段属设计目标；实际结果字段以 [模块与接口说明](模块与接口说明.md#74-python--m-agentbenchmarkagent_eval) 为准。

```json
{
  "lock": {"ref": "baseline_lock.json", "sha256": "<64hex>"},
  "metadata": {"profile": "standard", "n": 50, "seed": 20261112, "git_commit": "..."},
  "gates": {"structure_invariance": 1.0, "entity_fidelity": 0.997,
             "loop_incidents": 0, "budget_compliance": 1.0,
             "rollback_identical": true, "passed": true},
  "L1_capability": {"defect_f1": {"D1": 0.0, "...": 0.0}, "macro_f1": 0.0,
                     "consistency_block_rate": 0.0, "false_reject_rate": 0.0},
  "L2_decision": {"repair_success_rate": 0.0, "avg_tool_calls": 0.0,
                   "invalid_call_rate": 0.0, "fallback_rate": 0.0,
                   "explainability": 1.0},
  "L3_quality": {"win_rate": 0.0, "win_rate_ci95": [0.0, 0.0],
                  "win_rate_decided": 0.0, "tie_rate": 0.0,
                  "coherence_delta": 0.0, "diversity_delta": 0.0,
                  "length_ratio": 1.0},
  "L4_engineering": {"tokens_per_dialogue": {}, "p50_ms": 0, "p95_ms": 0,
                      "cost_total": 0.0, "cache_hit_rate": 0.0,
                      "reproducible": true},
  "arms": [{"arm": "A0", "...": 0}],
  "stats": {"method": "mcnemar", "p_value": 0.0, "n": 50},
  "conclusions": ["..."],
  "limitations": ["n=50 无法检出 <15pp 差异", "judge 与生成同模型"]
}
```

**`limitations` 是必填字段。** 一份没有局限说明的 benchmark 报告，在懂行的人眼里等于没写。

---

## 8. 运行方式与复现

### 8.1 当前可用的运行方式

| 方式 | 命令 | 联网 | 用途 | 产物 |
|---|---|---|---|---|
| 离线基线 | `python -m agent.benchmark.run --input agent/fixtures/public/base_v1.json --arms A0` | 否 | 冻结 / 复现 A0 | `arm_results.json`、`run_manifest.json`、lock |
| 离线规则臂 | `python -m agent.benchmark.agent_eval --arm A3 --out ...` | 否 | 规则决策 + 确定性工具 | `ablation_A3.json` |
| 在线臂 | `python -m agent.benchmark.agent_eval --arm A3L/A3R/A4 --live --out ...` | 是 | 正式数据 | 结果 JSON + `manifest_<arm>.json` + ledger |

约束：

- `run.py` **只支持 A0**，不能跑 A2/A3；Agent 臂统一走 `agent_eval.py`。传入其他臂会直接报错退出（不会生成空臂锁）。
- `run.py --profile` 只是写入 lock 的标签，**不切换执行档位**。
- `agent_eval` 的在线开关是 `--live`；缺少密钥时退出码 2。
- replay / cassette 未实现（S1b），当前无法离线回放真实响应。

### 8.2 quick / standard / full（设计目标，未实现）

下表为规划中的规模档，当前没有对应 CLI 开关；调试只能用 `--limit N`，正式结论使用固定的 120 条缺陷 + 12 条干净样本。

| 档 | n（defects / test） | 臂 | 用途 |
|---|---|---|---|
| `quick` | 30 / 20 | A0,A2,A3 | 开发自测 |
| `standard` | 120 / 50 | A0–A3 | 出正式结论 |
| `full` | 240 / 200 | A0–A4 | 细粒度差异 |

> 不要用 `--profile full` 之类的参数；该参数在 `run.py` 中只是标签，不会扩样。

### 8.3 复现断言（CI 实际执行）

```bash
python -m pytest
python -m agent.benchmark.run \
  --input agent/fixtures/public/base_v1.json \
  --arms A0 --verify-arm A0
```

CI 额外断言 `baseline_lock.json` 的 `code.dirty == false`。A0 复现失败即中止，不允许用 `--force` 掩盖。

---

## 9. 成本-效果前沿（与 token ledger 对接）

本期**只计量、不管控**。Benchmark 把 ledger 当成一等数据源：

> 注意：成本-效果前沿需要 L3 胜率；L3 尚未实现，当前只能展示“成本 vs 阻断率”，不能宣称胜率-成本前沿。

| 派生指标 | 定义 | 用途 |
|---|---|---|
| 单条全链 token | 该臂所有 LLM 调用 token 之和 / 样本数 | 横轴 |
| 胜率增量 / 千 token | `ΔWR / (Δtoken/1000)` | 边际收益 |
| 每 +1pp 胜率的成本 | `Δcost / (ΔWR×100)` | 讲"值不值" |
| 缓存后成本 | 命中缓存的调用按 0 计 | 重复运行的真实成本 |

前沿图（目标 fig1，需 L3 胜率完成后）直接用各臂的 (单条 token, 胜率) 描点，连成曲线。**这条曲线就是简历上"成本-效果"那句话的出处，没有它就不要写这句话。** 当前 `01_ablation.svg` 画的是 (单条 token, 阻断率)，不能替代胜率前沿。

---

## 10. 反作弊 / 自证清单（跑完逐项勾）

| # | 检查项 | 为什么 |
|---|---|---|
| 1 | 输入 fixture 不含 LLM 生成内容 | 否则自我强化，指标无意义 |
| 2 | dev / test 无重叠（键级断言） | 防止调参污染测试集 |
| 3 | judge prompt 不含臂标签 | 防泄漏 |
| 4 | 位置双向评过 | 防位置偏差 |
| 5 | 失败样本在分母里 | 防虚高 |
| 6 | 所有 seed 已记录 | 可复现 |
| 7 | 硬门禁独立于胜率报告 | 防"用平均分掩盖硬伤" |
| 8 | 报告含 limitations | 防过度声明 |
| 9 | `baseline_lock.json` 未被修改 | 基线不可漂移 |
| 10 | 负面结果（如某类缺陷 F1 低）已展示 | 可信度 |

---

## 11. 本期不做（范围边界）

| 不做 | 原因 |
|---|---|
| RAG 臂（A5-RAG） | 移出本期，先保核心链路跑通 |
| 人工大规模标注（>100 条） | 成本高；n=50 已够支撑主结论 |
| 多 judge 投票 / 多专家 | 先用单模型多视角替代 |
| 训练后模型下游效果验证（A/B 训练） | 周期远超本期；以数据侧指标替代 |
| 生产级并发压测 | 实验定位，只测 P50/P95 |

---

## 12. 执行顺序（与路线图里程碑对齐）

| 步骤 | 交付 | 通过条件 | 状态 |
|---|---|---|---|
| B1 | `agent/fixtures/public/` 三数据集 + 缺陷注入器 | 120 条缺陷，标签可复算 | ✅ 完成 |
| B2 | E1 规则评估器 + 硬门禁 | 结构/实体/长度全部可判定 | ✅ 完成 |
| B3 | run harness + `baseline_lock.json` | 离线 A0 可跑通 | ✅ 完成（A1 已于 2026-11-19 移出） |
| B4 | A2/A3 接入 + ledger 汇总 | standard 档产出全指标 | ✅ A2/A3/A3L/A3R/A4/A4R 均已接入 |
| B5 | E2 judge + 双向盲评 + 一致性校准 | κ ≥ 0.6，一致率 ≥ 75% | ⬜ 部分：单视角 D1 判官已实现；盲评与校准未做 |
| B6 | 三件套报告 + 四张图 | 评审可离线一键复现 | ✅ 完成（SVG 四图） |
| B7 | （可选）A4 Agent-llm | function calling 稳定后 | ✅ 完成 |

## 13. 设计 vs 实现差异（Sprint 1a 基线）

| 设计项 | 当前实现 | 差距 / 后续 |
|---|---|---|
| 四元组指纹 | 已实现，含规则与判官源码哈希 | 独立 adapter 模块未落地 |
| A0/A2/A3/A3L/A3R/A4/A4R | 已实现（A1 已于 2026-11-19 移出主链） | — |
| L1 指标 | 规则、连贯性、D1 判官已实现 | 定位 P/R/F1 的完整统计未汇总 |
| L2 指标 | 预算、死循环、Trace、回退已实现 | 无效调用率、可解释率未在报告汇总 |
| L3 指标 | 基础统计已实现 | 胜率、Wilson CI、精确 McNemar 已实现；双盲、人工标注未实现 |
| L4 指标 | token、成本、指纹复现已实现 | P95、缓存命中率、重复计费数未统计 |
| 报告产物 | 三件套 + 四张 SVG | 长表、显著性、账本汇总、trace 样例未生成 |
| replay 档 | 未实现 | 无 cassette 机制 |
| 档位规模 | 仅 `--limit` | quick/standard/full 未实现 |
| `blocked` 口径 | 只统计 `fallback` | A3R 重写成功未计入，需在 S1b 统一 |
| Docker/MCP/OTel | 未实现 | 路线图 S2 |

> 本表是后续 Sprint 的验收参照：任何一项从“未实现”变为“已实现”，必须同时更新本文 §0.1、[模块与接口说明](模块与接口说明.md) 与根 README。
