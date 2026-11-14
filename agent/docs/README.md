# Agent 方向设计文档索引

本目录统一归档 Agent 相关设计文档。**新文档一律放这里**，不再散落到 `docs/` 根目录。

## 文档清单与权威顺序

| 顺序 | 文档 | 状态 | 定位 |
|---|---|---|---|
| ① 权威 | [Benchmark专项设计方案.md](Benchmark专项设计方案.md) | v1.0 现行 | **基线对比标准、指标体系、统计判定、产物规格的唯一来源** |
| ② 权威 | [数据策略评估与决策.md](数据策略评估与决策.md) | v1.0 现行 | **开源展示数据与 Benchmark 数据集选型的决策依据**（双轨制） |
| ③ 权威 | [数据格式适配与兼容规范.md](数据格式适配与兼容规范.md) | v1.0 现行 | **冻结契约、统一适配层、差异分级、四条可比性门禁** |
| ④ 权威 | [Agent化改造总纲与Benchmark方案.md](Agent化改造总纲与Benchmark方案.md) | v1.1 现行 | 现状评估、可插拔架构、开源边界、token 计量、求职适配 |
| ⑤ 参考 | [Agent化技术评估与落地方案.md](Agent化技术评估与落地方案.md) | v1.0 归档 | 早期事实核验与路线评估；结论已被 ④ 继承 |
| ⑥ 想法池 | [对话数据Agent化改进计划书.md](对话数据Agent化改进计划书.md) | v0.1 归档 | 原始构想与分阶段设想；日期计划已废弃，仅保留思路 |

**冲突处理**：① > ② > ③ > ④ > ⑤ > ⑥。低优先级文档中的冲突表述不再修改，以高优先级为准。

## 已确认的关键决策（速查）

| # | 决策 |
|---|---|
| X1 | RAG 移出本期范围，只留接口与开关 |
| X2 | 主项目整体不开源；agent 模块领域无关，可独立开源（Apache-2.0） |
| X3 | token 全量留存（append-only ledger），只计量不管控；成本策略延后 |
| X4 | 先补测试（T0）与基线（T1），再动 Agent 代码 |
| X5 | Benchmark 基线标准独立成文；agent 文档统一归档本目录 |
| X6 | 数据双轨：公开轨用程序化构造集（换业务域），私有轨用现网脱敏抽样；现网数据仅贡献统计画像与一次性外部效度检验 |
| X7 | 换域后格式/字段/类型完全冻结不变，元数据只增不改；benchmark 与生产共用同一 adapter，基线指纹扩为四元组 |
| X8 | C3（结构降级 + role 推断）仅限私有轨；生产需要放宽时走 §3.4 流程（确定性 RoleMap、100% 准确率、升 major 版本） |

## 里程碑

`T0 测试地基 → T1 基线冻结 → M0 插件骨架 → M1 单工具 → M2 循环 → M3 真实调用 → M4 开源切片 → M5 作品集`

| 阶段 | 状态 | 交付 |
|---|---|---|
| T0 测试地基 | ✅ 完成 | `tests/{unit,contract,integration,snapshot}`，85 项通过 |
| T1 基线冻结 | ✅ 完成 | `agent/benchmark/`：四元组指纹、不可变 lock、A0/A1 对照臂、`baseline/baseline_lock.json` |
| M0 插件骨架 | ✅ 完成 | `pipeline/core/extension_loader.py` + `agent/plugin/`（透传步骤）+ `agent/runtime/`（LLM 协议、工具注册表） |
| M1 单工具 | ✅ 完成 | 五条硬约束（R1–R5）+ `check_consistency` 工具 + 6 类缺陷注入集（120 条）；结构类拦截 100%，误杀 0 |
| M2 决策循环 | ✅ 完成 | 预算 / 状态 / Trace / 规则策略 / 循环 / 条目级断点；死循环与预算保护全部生效 |
| M3 真实调用 | ✅ 完成 | 配置/密钥、token 账本、DeepSeek 客户端、版本化价格表、`check_coherence`、语义判官（D1）、`LLMPolicy`、token 预算覆盖策略与工具内消耗、A3/A3L/A4 臂评测；D1 0→100%，宏观阻断 83.3%→100% |
| M4 开源切片 | ✅ 完成 | 清单三段（PUBLIC/BUILD/PRIVATE）+ `export_oss` + 边界自证测试；开源仓库独立可跑，无业务词与私有文件 |
| M5 作品集 | ✅ 完成 | `portfolio/agent-augmentation/`：三件套报告（HTML/MD/JSON）+ 3 张 SVG + 真实 trace + before/after；离线可复现 |

**M0 门禁结果**

| 门禁 | 结果 |
|---|---|
| 插件模块缺失时主流程可跑 | ✅ |
| 插件关闭时产物与 T1 冻结基线一致 | ✅ 05_augment 产物 sha256 相同 |
| 主工程不 import 插件 | ✅ AST 扫描 |
| 工具注册表契约 | ✅ 注册 / 去重 / 分发 / 返回类型 |
| 透传不改数据 | ✅ 输入输出规范化哈希相同 |

**T1 复现命令**

```powershell
python -m agent.benchmark.fixtures --n 12 --seed 20261112     # 生成公开轨 fixture
python -m agent.benchmark.run --input agent/fixtures/public/base_v1.json --arms A0,A1
python -m agent.benchmark.run --input agent/fixtures/public/base_v1.json --arms A0,A1 --verify
```

门禁细则见总纲 §5.5。
