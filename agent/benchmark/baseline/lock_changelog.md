# 基线指纹变更记录

- 2026-11-12T22:48:34 覆盖 baseline_lock.json 原因：迁移至 agent/ 单文件夹后重建基线
- 2026-11-12T23:08:47 覆盖 baseline_lock.json 原因：lock 补充每臂 artifact_sha256，使 OSS 侧可在无 A1 环境下自证基线
- 2026-11-13T00:12:46 覆盖 baseline_lock.json 原因：M1：fixture 首轮加入订单号以支撑 D2 指代断裂注入，需重建基线
- 2026-11-13T00:23:29 覆盖 baseline_lock.json 原因：M1: 首轮 user 消息加入订单号，输入数据变更
- 2026-11-18T16:50:28 覆盖 baseline_lock.json 原因：S0 锁加固：lock v2.0（抽样键集合 + 评估器源码/prompt 哈希 + 拒绝 dirty）
- 2026-11-19 移除 A1 臂（当前 v2.0 锁本就只含 A0，未重建）原因：A1 为上游 05_augment 加噪增强，输出变体 n_out≠n_in 不可配对；与语义修复任务不同源；产物未入库且当前环境无 pipeline 包，无法复现。旧指纹归档于 `baseline/deprecated/a1_2026-11-19.json`，说明见 `docs/notes/2026-11-19-A1臂移除记录.md`；A1 的用途转为鲁棒性实验的噪声源。
