"""DeepSeek 价格表（版本化）

单价来源：DeepSeek 官方定价页（deepseek-flash，2026-11 观察值，单位：元 / 百万 tokens）

| 时段 | 输入（缓存命中） | 输入（缓存未命中） | 输出 |
|---|---|---|---|
| 空闲 | 0.02 | 1 | 4 |
| 高峰 | 0.04 | 2 | 8 |

空闲时段 = 北京时间 00:30–08:30，其余按高峰计。官方调价或调整时段时，
新增一张 Pricing 并升 version —— 历史账本里的 cost_cny 不会被追溯污染。

代码里只放"价格表"，不放"预算策略"：成本管控按要求延后。
"""

from datetime import datetime, time, timedelta, timezone
from typing import Optional

from .ledger import Pricing

# 价格表族版本：benchmark 的 baseline lock 记录的是这个族（不含时段）
PRICE_TABLE_VERSION = "deepseek_2026-11"

# 北京时间固定 UTC+8（无夏令时），避免依赖 tzdata
CST = timezone(timedelta(hours=8))

OFF_PEAK_START = time(0, 30)
OFF_PEAK_END = time(8, 30)

PEAK = Pricing(
    input_per_1m=2.0,
    cache_hit_input_per_1m=0.04,
    output_per_1m=8.0,
    version=f"{PRICE_TABLE_VERSION}/peak",
)

OFF_PEAK = Pricing(
    input_per_1m=1.0,
    cache_hit_input_per_1m=0.02,
    output_per_1m=4.0,
    version=f"{PRICE_TABLE_VERSION}/offpeak",
)

PRICES = {PEAK.version: PEAK, OFF_PEAK.version: OFF_PEAK}


def select_pricing(at: Optional[datetime] = None) -> Pricing:
    """按调用时刻选档。

    naive datetime 视为 UTC：不做"猜时区"的隐式转换，宁可要求调用方明确。
    """
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    local = at.astimezone(CST).time()
    return OFF_PEAK if OFF_PEAK_START <= local < OFF_PEAK_END else PEAK


__all__ = [
    "CST", "OFF_PEAK", "OFF_PEAK_END", "OFF_PEAK_START",
    "PEAK", "PRICES", "PRICE_TABLE_VERSION", "select_pricing",
]
