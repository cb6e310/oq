"""项目级固定市场阶段划分，供所有策略统一引用。

固定口径（日期边界均包含当日）：
- 熊市：截至 2019-01-31；2021-11-30 至 2024-09-22；2026-07-01 起至今及以后。
- 牛市：其余日期，即 2019-02-01 至 2021-11-29、2024-09-23 至 2026-06-30。

该划分是项目研究约定，不随行情数据自动变化。需要修改时应统一修改本文件，
避免各策略分别维护不同的牛熊市边界。
"""
from __future__ import annotations

import pandas as pd


BEAR_RANGES = (
    (None, pd.Timestamp("2019-01-31")),
    (pd.Timestamp("2021-11-30"), pd.Timestamp("2024-09-22")),
    (pd.Timestamp("2026-07-01"), None),
)


def market_regime(date) -> str:
    """将单个日期分类为 ``熊市`` 或 ``牛市``。"""
    value = pd.Timestamp(date).normalize()
    for start, end in BEAR_RANGES:
        if (start is None or value >= start) and (end is None or value <= end):
            return "熊市"
    return "牛市"


def market_regime_series(dates) -> pd.Series:
    """批量分类日期，返回与输入索引对齐的字符串 Series。"""
    values = pd.to_datetime(dates)
    result = pd.Series("牛市", index=getattr(dates, "index", None), dtype="string")
    result.loc[values <= pd.Timestamp("2019-01-31")] = "熊市"
    result.loc[(values >= pd.Timestamp("2021-11-30")) &
               (values <= pd.Timestamp("2024-09-22"))] = "熊市"
    result.loc[values >= pd.Timestamp("2026-07-01")] = "熊市"
    return result
