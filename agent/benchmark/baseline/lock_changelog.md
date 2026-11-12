# 基线指纹变更记录

- 2026-11-12T22:48:34 覆盖 baseline_lock.json 原因：迁移至 agent/ 单文件夹后重建基线
- 2026-11-12T23:08:47 覆盖 baseline_lock.json 原因：lock 补充每臂 artifact_sha256，使 OSS 侧可在无 A1 环境下自证基线
- 2026-11-13T00:12:46 覆盖 baseline_lock.json 原因：M1：fixture 首轮加入订单号以支撑 D2 指代断裂注入，需重建基线
- 2026-11-13T00:23:29 覆盖 baseline_lock.json 原因：M1: 首轮 user 消息加入订单号，输入数据变更
