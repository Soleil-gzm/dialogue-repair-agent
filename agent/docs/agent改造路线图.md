# dialogue-repair-agent → Agent 岗位作品改造路线图

> 版本：v2.0 ｜ 2026-11-18（依据《agent决策记录ADR.md》D1–D9 修订）  
> 目标：把现有项目从「质检 + 消融评测 Agent」升级为覆盖目标 JD 高频关键词的完整 Agent 作品，**每个关键词都有真实代码支撑，不靠换词硬凑**。

---

## 0. 改造原则（v2 依据决策更新）

1. **一个关键词 = 一个可运行分支/目录 = 一段可防守的面试叙事**，不做装饰性功能
2. 沿用项目已有纪律：先更新设计文档 → 再动代码 → 同 seed 可复现 → 测试固化
3. 诚实边界：RAG、LangGraph、Docker 做到哪一步就说到哪一步，未实现的词不上简历（D9）
4. **主方向是 Harness**（基线/消融/门禁/可复现），runtime 保持薄而稳（D2）
5. **单 Agent 架构不摇摆**：不做 multi-agent，消融证据已证明决策层非瓶颈（D3）

**已明确的非目标（写进 README 也能加分）：**

- ❌ Multi-agent —— 决策层消融显示零增益，触发条件（上下文隔离分解的任务）不满足
- ❌ 本期 RAG 实现 —— 接口先行，材料用红线规则库，实现后移（D5）

---

## 1. 关键词覆盖矩阵（v2）

| JD 关键词 | 现状 | 改造动作 | Sprint |
| ----------------------- | -------- | ---------------------------------------------- | ------ |
| CI/CD / 环境一致性 | ⚠️ 待补 | **Sprint 0**：GitHub Actions + 依赖/模型锁定（跨机开发的地基） | S0 |
| 工具调用 / Function Calling | ✅ 已有 | 补 `rewrite_segment`，决策空间 3 → 5+；stage 路由双模型 | S1 |
| 评测体系 | ✅✅ 强项 | 补 L3 效果层（胜率/CI/McNemar/双向盲评）——设计已写、未落地 | S1 |
| 稳定性 / 断点续跑 | ⚠️ 部分 | checkpoint 接入批量评测 + replay 档位（cassette 回放） | S1 |
| 真实数据验证 | ⚠️ 缺 | 私有轨：公司数据 + Ollama 本地模型，聚合脱敏发布（并行，不依赖新功能） | S1 并行 |
| MCP | ✅ 有（公司侧） | 工具集封装为 MCP server，Claude Code/Cursor 直连 | S2 |
| Docker / 部署 | ⚠️ 待补 | Dockerfile + Compose 一键复现 offline benchmark | S2 |
| 可观测 / Tracing | ✅ 有 | OTel span 导出 + P50/P95/错误率/缓存命中率汇总 | S2 |
| Agent 工作流编排 | ✅ 已有（手写） | LangGraph 对照实现（可选，不替换手写版） | S2 可选 |
| RAG / 向量检索 | ⚠️ 部分 | **只做接口定义**（`Retriever` + A5 臂设计占位），实现后移 | S3 占位 |

---

## 2. Sprint 0：跨环境地基（预计 1–2 天，最优先）

> 依据 D7：在家（4060）/公司（4090）双机开发，先消除"代码能跑但环境漂移"的风险；同时这是 Docker/CI 缺口的第一块。

### 2.1 GitHub Actions（半天）

- pytest + offline 档复现断言（方案 §8.2）+ lock 校验
- **最强保障**：任何在家能跑、公司跑挂的代码进不了 main

### 2.2 依赖与模型锁定（半天）

- `uv.lock` / requirements.lock 提交，两机统一 `uv sync`
- Ollama 模型钉死完整 tag（如 `qwen2.5:7b-instruct-q4_K_M`），两机核对 hash；模型名纳入 lock 指纹

### 2.3 stage 路由双模型配置（半天）

- 客户端工厂按 stage 路由：`generate`（decide + rewrite 共用）与 `judge` 独立模型、跨家族（D4）
- 配置允许指向同一模型，但报告 limitations 必须声明
- `.env` 放本机差异（OLLAMA_BASE_URL、模型名、密钥），提交 `.env.example` 作契约；代码中永不出现"4060/4090"

---

## 3. Sprint 1：修复闭环 + 评测叙事完整（预计 5–7 天）

> 解决核心缺口：Agent 只有"检查 + 回退"，没有"修复"。

### 3.1 `rewrite_segment` 工具（3 天）

- `tools/rewrite_segment.py`：LLM 重写指定轮次，入参由运行时注入（沿用设计取舍：模型只定位 + 下指令，不复述对话）
- 模型走生成侧配置（`generate` stage，temp 0.3–0.7）
- RulePolicy 升级为多轮策略：检出 → 定位 → 重写 → 复检（consistency + coherence）→ 接受/回退
- 死循环保护覆盖"重写→复检→再重写"循环
- **面试叙事**：消融显示瓶颈在工具不在决策，所以先扩工具再谈决策智能化——排序是实验驱动的

### 3.2 L3 效果层落地（2 天）

按 Benchmark 设计方案 §4.3/§6 兑现：

- A2 臂（单次重写）——主结论 A3 vs A2 的前置条件
- 成对胜率 WR + Wilson 95% CI + McNemar
- E2 judge 双向盲评（A/B 与 B/A 各评一次，矛盾记 tie；judge 走 `judge` stage 独立模型）
- 长度比、distinct-2 变化、跨样本 4-gram 重复率（防"洗成一个味"）

### 3.3 replay 档位 + checkpoint（1–2 天）

- `--profile replay --cassette cassettes/v1.jsonl`：真实响应录制回放，报告零成本复核
- checkpoint 接入批量评测 + 零重复计费断言
- **面试叙事**：稳定性 = 断点续跑 + 零重复计费 + 确定性回放

### 3.4 私有轨实验（与上面并行，不依赖新功能）

依据 D6，公司机（4090）执行：

- Ollama 部署 qwen2.5+（必须支持 tool calls），客户端改 base_url 复用；ledger 加本地模型条目（成本 0、token 照记）
- **合规先行**：内部报备；只发布聚合指标 + 缺陷类型分布，绝不展示原始对话/业务线名/字段值
- 产出：公开轨 vs 私有轨指标并列报告；本地模型 vs 线上模型的检出差距照实报（"模型能力下限对 Agent 效果的敏感性"本身就是结论）
- 判官跨家族：本地 qwen 生成 + 线上 deepseek-chat 或本地异家族模型判分（D4）
- 产物记录 machine id + ollama 版本 + 模型 hash；产物不进 git，只合并结论

---

## 4. Sprint 2：工程化关键词清零（预计 5–7 天）

### 4.1 MCP Server（1–2 天）

- fastmcp 封装 `check_consistency` / `check_coherence` / `rewrite_segment` / `fallback_keep`
- Claude Code / Cursor 直连演示，截图/交互日志入 README
- **面试叙事**：工具层一次封装、两种协议消费（function calling 注册表 + MCP server）

### 4.2 Docker（2 天）

- Dockerfile（多阶段、无业务密钥）+ Compose 一键跑 offline benchmark
- CI 已在 S0 建好，此处补容器化交付面

### 4.3 可观测升级（2 天）

- TraceWriter 加 OpenTelemetry span 导出（tool 调用、latency、token 为 attributes）
- report.py 汇总 P50/P95、错误率、缓存命中率（L4 指标已有设计）
- **面试叙事**：自有 JSONL trace 与 OTel 双通道，Grafana/Jaeger 可看

### 4.4 LangGraph 对照实现（可选，2–3 天）

- 同一决策循环用 `StateGraph` 重写为独立 adapter，构成"手写 vs 框架"消融
- 时间紧则降级为独立 mini demo；**未写过不上简历**

---

## 5. Sprint 3：RAG 接口占位（预计 1 天，实现后移）

依据 D5，本期只做三件事：

1. 定义 `Retriever` 协议 + `retrieve_guideline` 工具占位进 registry（契约对齐现有工具）
2. Benchmark 设计文档补 A5-RAG 臂定义（单变量：是否检索）与缺陷-工具矩阵行
3. 检索语料定向：**红线规则 / 业务合规口径 / 话术禁用词 / 承诺上限**——比"优质对话库"更权威、来源更易防守

实现（向量召回 + BM25 兜底 + A5 臂实验）待 S1/S2 落地后评估排期。**实现之前简历不写 RAG。**

---

## 6. 交付物清单（求职材料对齐）

| 材料 | 内容 | 对应 JD 缺口 |
| ------------ | -------------------------- | ----------------- |
| README 顶部三句话 | 修复闭环 + 可复现评测 + 成本-效果前沿 | Agent Demo / 实战项目 |
| 消融报告（含 L3） | 胜率 CI + 检出率 + 成本曲线 fig1 | 评测体系 ✅✅ |
| 双轨报告 | 公开轨 + 私有轨（脱敏聚合）并列 | 真实数据验证 / 端到端交付 |
| MCP 演示截图 | Cursor/Claude Code 直连工具 | MCP |
| 一键 Docker | `docker compose up` 复现全部结果 | Docker / 部署 |
| 架构图 | 观测→决策→工具→评估→回退闭环（单 Agent） | 工作流编排 / 架构设计 |
| 开源仓库 + 徽章全绿 | CI badge + 复现断言 | GitHub 展示 |

---

## 7. 风险与纪律提醒

1. **每完成一个 Sprint 更新 Benchmark 设计文档**（新增臂定义、工具矩阵行），保持"先定义怎么算改好，再动手改"
2. rewrite 引入重写输出形态 = 数据契约变更，lock 指纹会变：走 `--force --reason` 留痕
3. 私有轨实验**报备在先**，未报备不出数据
4. LangGraph/Dify 等框架词只在真正写过之后写上简历
5. 人工标注（κ 校准）成本高，先用 judge 双向盲评顶住 L3，人工校准作 limitation 声明
6. 判断新需求是否接：对照《agent决策记录ADR.md》——凡与 D2（Harness 主方向）、D3（不做多 Agent）冲突的，先进待定池不排期

---

## 8. 执行顺序总览

```
S0 跨环境地基（CI/锁定/双模型配置）      ← 1–2 天，立即可做
S1 修复闭环 + L3 + replay ｜ 私有轨实验并行 ← 5–7 天
S2 MCP + Docker + 可观测（+LangGraph 可选）← 5–7 天
S3 RAG 接口占位（实现后移）             ← 1 天
```

与《agent决策记录ADR.md》联动：本路线图记录"做什么、什么顺序"，ADR 记录"为什么这么定、怎么防守"。两者变更时互相引用条目编号。
