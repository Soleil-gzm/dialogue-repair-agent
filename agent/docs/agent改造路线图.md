# dialogue-repair-agent → Agent 岗位作品改造路线图

> 版本：v1.0 ｜ 2026-11-18
> 目标：把现有项目从「质检 + 消融评测 Agent」升级为覆盖目标 JD 高频关键词的完整 Agent 作品，**每个关键词都有真实代码支撑，不靠换词硬凑**。

---

## 0. 改造原则

1. **一个关键词 = 一个可运行分支/目录 = 一段可防守的面试叙事**，不做装饰性功能
2. 沿用项目已有纪律：先更新设计文档 → 再动代码 → 同 seed 可复现 → 测试固化
3. 诚实边界：RAG、LangGraph、Docker 做到哪一步就说到哪一步，JD 缺口表里的「⚠️ 部分」逐个清零

---

## 1. 关键词覆盖矩阵（改造后）

| JD 关键词 | 现状 | 改造动作 | 对应 Sprint |
|---|---|---|---|
| 工具调用 / Function Calling | ✅ 已有 | 补 `rewrite_segment`，决策空间从 3 个工具扩到 5+ | S1 |
| Agent 工作流编排 | ✅ 已有（手写） | 加 LangGraph 对照实现（同一循环两种编排） | S3 |
| RAG / 向量检索 | ⚠️ 部分 | `retrieve_history` 工具 + A5-RAG 对照臂 | S2 |
| 评测体系 | ✅✅ 强项 | 补 L3 效果层（胜率/CI/McNemar）——设计已写、未落地 | S1 |
| 可观测 / Tracing | ✅ 有 | trace 升级：OpenTelemetry 导出 + 指标汇总端点 | S3 |
| MCP | ✅ 有（公司侧） | 本项目工具集封装为 MCP server，Claude Code/Cursor 可直连 | S3 |
| Docker / CI/CD | ⚠️ 待补 | Dockerfile + Compose + GitHub Actions 全绿 | S3 |
| 稳定性 / 断点续跑 | ⚠️ 部分 | checkpoint 接入批量评测 + replay 档位（cassette 回放） | S1 |
| 端到端交付 | ✅ 已有 | 修-评-测-报告全链在开源仓库内自证 | — |

---

## 2. Sprint 1：修复闭环 + 评测叙事完整（预计 5–7 天）

> 解决上一轮评估的核心缺口：Agent 只有"检查 + 回退"，没有"修复"。

### 2.1 `rewrite_segment` 工具（3 天）
- 新增 `tools/rewrite_segment.py`：LLM 重写指定轮次，入参由运行时注入（沿用 LLMPolicy 的设计取舍：模型只定位 + 下指令，不复述对话）
- RulePolicy 升级为多轮策略：检出 → 定位 → 重写 → 复检（consistency + coherence）→ 接受/回退
- 死循环保护正好覆盖"重写→复检→再重写"循环，是现成卖点
- **面试叙事**：决策空间从 3 个工具扩到 5 个后，A3 vs A3L 的消融重新有意义——"上一轮实验证明瓶颈在工具能力不在决策，所以我先扩工具再谈决策智能化"

### 2.2 L3 效果层落地（2 天）
按 Benchmark 设计方案 §4.3/§6 实现报告里已承诺但未产出的部分：
- A2 臂（单次重写）——主结论 A3 vs A2 的前置条件
- 成对胜率 WR + Wilson 95% CI + McNemar
- E2 judge 双向盲评（A/B 与 B/A 各评一次，矛盾记 tie）
- 长度比、distinct-2 变化、跨样本 4-gram 重复率（防"洗成一个味"）

### 2.3 replay 档位（1–2 天）
- `--profile replay --cassette cassettes/v1.jsonl`：真实响应录制回放，报告可零成本复核
- 顺带把 checkpoint 接入批量评测（`test_checkpoint.py` 已有基础）
- **面试叙事**：稳定性 = 断点续跑 + 零重复计费 + 确定性回放

---

## 3. Sprint 2：RAG 端到端叙事（预计 5–7 天）

> 清零最大的「⚠️ 部分」缺口。RAG 在这里是**重写环节的上下文增强**（与计划书阶段二一致），不是全流程 RAG，面试时表述要准确。

### 3.1 优质对话检索库（2 天）
- 结构化存储：场景标签 + 对话内容 + 质量标注（沿用项目字段契约）
- 向量化：优先 sentence-transformers 本地小模型；开源分发场景保留 BM25 纯标准库兜底（与项目"零依赖"定位兼容，做成可插拔 Retriever 接口）

### 3.2 `retrieve_history` 工具 + A5-RAG 臂（3 天）
- 工具契约与现有 registry 对齐：返回 top-k 相似对话片段
- 新增对照臂 A5：rewrite + retrieve，与 A3/A4 只差"是否检索"这一个变量，沿用单变量原则与 lock 机制
- 产出对比实验：有检索参考 vs 无检索参考的胜率差 + CI
- **面试叙事**："检索 + 重写"两段式增强，A5 vs A4 消融量化 RAG 净收益——多数候选人只能说"我用了 RAG"，你能给出增量的置信区间

---

## 4. Sprint 3：工程化关键词清零（预计 5–7 天）

### 4.1 MCP Server（1–2 天）
- 用 fastmcp 把 `check_consistency` / `check_coherence` / `rewrite_segment` / `retrieve_history` / `fallback_keep` 封装成 MCP server
- Claude Code / Cursor 直连演示，录一张截图/一段交互日志入 README
- **面试叙事**：同一套工具契约，既是 function calling 决策循环的注册表，又是 MCP server——"工具层一次封装、两种协议消费"

### 4.2 Docker + CI/CD（2 天）
- Dockerfile（多阶段、无业务密钥）+ docker-compose 一键跑 offline benchmark
- GitHub Actions：pytest + offline 档复现断言（方案 §8.2 的 `test_benchmark_is_reproducible`）+ lock 校验
- **面试叙事**：CI 里跑"复现断言"而不是只跑单测——基线指纹在流水线上自动验证

### 4.3 可观测升级（2 天）
- TraceWriter 增加标准 OpenTelemetry span 导出（tool 调用、latency、token 为 attributes）
- report.py 汇总 P50/P95、错误率、缓存命中率（L4 指标已有设计）
- **面试叙事**：自有 JSONL trace 与 OTel 双通道，落进 Grafana/Jaeger 可看

### 4.4 LangGraph 对照实现（可选，2–3 天）
- 用 LangGraph `StateGraph` 重写同一决策循环，作为独立 adapter（不替换手写版，两者构成"手写 vs 框架"消融）
- **面试叙事**：能回答"为什么不用 LangChain"且真正用过 LangGraph——比二选一的站队更有说服力；若时间紧可降级为独立 mini demo

---

## 5. 交付物清单（求职材料对齐）

| 材料 | 内容 | 对应 JD 缺口 |
|---|---|---|
| README 顶部三句话 | 修复闭环 + 可复现评测 + 成本-效果前沿 | Agent Demo / 实战项目 |
| 消融报告（含 L3） | 胜率 CI + 检出率 + 成本曲线 fig1 | 评测体系 ✅✅ |
| MCP 演示截图 | Cursor/Claude Code 直连工具 | MCP |
| 一键 Docker | `docker compose up` 复现全部结果 | Docker / 部署 |
| 架构图 | 观测→决策→工具→评估→回退闭环 | 工作流编排 / 架构设计 |
| 开源仓库 + 徽章全绿 | CI badge + 复现断言 | GitHub 展示 |

---

## 6. 风险与纪律提醒

1. **每完成一个 Sprint 更新 Benchmark 设计文档**（新增 A5 臂、rewrite 工具的缺陷-工具矩阵行），保持"先定义怎么算改好，再动手改"
2. rewrite 引入重写输出形态 = 数据契约变更，lock 指纹会变：用 `--force --reason` 留痕，别绕过
3. LangGraph/Dify 等框架词只在真正写过之后写上简历
4. 人工标注（κ 校准）成本高，可先用 judge 双向盲评顶住 L3，人工校准作为 limitation 声明
