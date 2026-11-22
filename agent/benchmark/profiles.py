"""公开轨生成档位参数

`v1` 是已冻结的默认档，产物与 `baseline_lock.json` 绑定，不能静默改变。
`realistic` 是新增的分布校准档：参数只来自私有轨的聚合统计，不包含原文、
业务标签、客户信息或任何可还原到单条对话的内容。

私有轨统计口径（1000 条，2026-11-23 复核）：
- 每条对话 3–53 条消息，即 1–26 组 user/assistant
- 轮次中位数 11、均值 10.85
- 去掉数字后的全局唯一率：assistant 49.53%，user 28.51%
- 33/1000 条对话内部存在完全重复的 assistant 消息
- system 长度 348–375 字符，均值 361
- user 长度中位数 13、均值 15.1、p95 31
- assistant 长度中位数 47、均值 56.1、p95 120
- 1000 条 system 均含金额；system 无订单号；assistant 未出现 system 外编号
"""

PROFILE_V1 = "v1"
PROFILE_REALISTIC = "realistic"
PROFILES = (PROFILE_V1, PROFILE_REALISTIC)

# 轮次直方图：键为 assistant 轮数，值为每 1000 条对话中的条数。
# 生成时用最大余数法分配，n=1000 可精确复现该直方图。
REALISTIC_TURN_HISTOGRAM = {
    1: 5, 2: 21, 3: 32, 4: 51, 5: 61, 6: 54, 7: 53, 8: 68, 9: 75,
    10: 60, 11: 60, 12: 84, 13: 75, 14: 59, 15: 62, 16: 45, 17: 41,
    18: 26, 19: 20, 20: 21, 21: 14, 22: 4, 23: 4, 24: 3, 25: 1, 26: 1,
}

REALISTIC_MESSAGE_RANGE = (3, 53)
# 校准档的轮次分层桶，覆盖 1–26 轮；v1 仍使用 fixtures.TURN_BUCKETS，避免改动冻结切分
REALISTIC_TURN_BUCKETS = ((1, 4), (5, 8), (9, 12), (13, 18), (19, 26))
REALISTIC_ASSISTANT_UNIQUE_RATE = 0.4953
REALISTIC_USER_UNIQUE_RATE = 0.2851
REALISTIC_INTERNAL_ASSISTANT_DUP_RATE = 0.033
REALISTIC_SYSTEM_CHARS = (348, 375)
REALISTIC_USER_CHARS = {"median": 13, "mean": 15.1, "p95": 31}
REALISTIC_ASSISTANT_CHARS = {"median": 47, "mean": 56.1, "p95": 120}

# 公开轨校准档的默认规模与种子。与 v1 分开，避免改动冻结基线。
REALISTIC_BASE_N = 120
REALISTIC_BASE_SEED = 20261123
REALISTIC_DEV_N = 40
REALISTIC_TEST_N = 80
REALISTIC_SPLIT_SEED = 20261124
