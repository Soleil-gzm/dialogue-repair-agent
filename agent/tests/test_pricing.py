"""测试：价格表分档与计费

单价口径（元 / 百万 tokens，deepseek-flash 2026-11）：
输入缓存未命中 1 / 2，输入缓存命中 0.02 / 0.04，输出 4 / 8（空闲 / 高峰）。
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent.runtime.ledger import Pricing
from agent.runtime.pricing import (
    CST, OFF_PEAK, PEAK, PRICE_TABLE_VERSION, PRICES, select_pricing,
)


def _cst(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, 13, hour, minute, tzinfo=CST)


class TestPriceTable:
    def test_peak_rates(self):
        assert PEAK.input_per_1m == 2.0
        assert PEAK.cache_hit_input_per_1m == 0.04
        assert PEAK.output_per_1m == 8.0

    def test_off_peak_rates(self):
        assert OFF_PEAK.input_per_1m == 1.0
        assert OFF_PEAK.cache_hit_input_per_1m == 0.02
        assert OFF_PEAK.output_per_1m == 4.0

    def test_versions_are_distinct_and_traceable(self):
        assert PEAK.version != OFF_PEAK.version
        assert PEAK.version.startswith(PRICE_TABLE_VERSION)
        assert set(PRICES) == {PEAK.version, OFF_PEAK.version}


class TestSelection:
    @pytest.mark.parametrize("hour,minute", [(0, 30), (3, 0), (8, 29)])
    def test_off_peak_window(self, hour, minute):
        assert select_pricing(_cst(hour, minute)) is OFF_PEAK

    @pytest.mark.parametrize("hour,minute", [(0, 29), (8, 30), (12, 0), (23, 59)])
    def test_peak_window(self, hour, minute):
        assert select_pricing(_cst(hour, minute)) is PEAK

    def test_naive_datetime_is_treated_as_utc(self):
        # 01:00 UTC = 09:00 北京时间，属高峰
        naive = datetime(2026, 11, 13, 1, 0)
        assert select_pricing(naive) is PEAK

    def test_utc_input_converted_to_beijing(self):
        at = datetime(2026, 11, 12, 17, 0, tzinfo=timezone.utc)  # = 01:00 CST
        assert select_pricing(at) is OFF_PEAK

    def test_default_is_now(self):
        assert select_pricing() in (PEAK, OFF_PEAK)


class TestCost:
    def test_million_tokens_peak(self):
        assert PEAK.cost(1_000_000, 1_000_000) == 10.0

    def test_million_tokens_off_peak(self):
        assert OFF_PEAK.cost(1_000_000, 1_000_000) == 5.0

    def test_cache_hit_input_discounted(self):
        # 全部命中：1M * 0.04
        assert PEAK.cost(1_000_000, 0, cache_hit_tokens=1_000_000) == 0.04

    def test_partial_cache_hit_split(self):
        # 0.6M 未命中 * 2 + 0.4M 命中 * 0.04
        assert PEAK.cost(1_000_000, 0, cache_hit_tokens=400_000) == pytest.approx(1.216)

    def test_hit_tokens_clamped_to_input(self):
        assert PEAK.cost(1_000, 0, cache_hit_tokens=99_999) == pytest.approx(0.00004)


class TestBackwardCompatibility:
    def test_default_pricing_is_free_and_unset(self):
        p = Pricing()
        assert p.version == "unset"
        assert p.cost(1_000_000, 1_000_000) == 0.0

    def test_legacy_two_arg_cost_still_works(self):
        p = Pricing(input_per_1m=1000, output_per_1m=2000, version="v1")
        assert p.cost(1_000, 1_000) == pytest.approx(3.0)
