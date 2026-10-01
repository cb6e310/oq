"""K-line historical pattern search.

The implementation follows the design in ``kline_similarity_engine_spec_v1``:
fast close-shape recall, multivariate constrained DTW re-ranking, rank fusion,
and non-maximum suppression.  It has no optional scientific dependencies; a
NumPy implementation of MASS-like subsequence distance and DTW is used so the
engine works in the project's stock_quant environment out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping, Protocol

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
FEATURE_NAMES = (
    "r_close", "gap", "body_return", "upper_wick", "lower_wick",
    "range_pct", "volume_log_change",
)
DEFAULT_FEATURE_WEIGHTS = np.array([2.0, 0.8, 1.2, 0.8, 0.8, 1.0, 0.6], dtype=float)


class MarketDataProvider(Protocol):
    """Minimal data contract required by :class:`KlineSimilarityEngine`."""

    def get_symbols(self) -> list[str]: ...

    def get_symbol_data(self, symbol: str) -> pd.DataFrame: ...


def aggregate_bars(data: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Return clean daily, weekly, or calendar-monthly OHLCV bars.

    Weekly bars end on Friday.  Input dates may be strings or timestamps and
    are normalized to midnight.  The returned frame is sorted by date and has
    the standard columns ``date, open, high, low, close, vol, amount``.
    """
    timeframe = _normalize_timeframe(timeframe)
    required = {"date", *PRICE_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"market data is missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", *PRICE_COLUMNS]).sort_values("date")
    frame = frame.drop_duplicates("date", keep="last").set_index("date")
    if "vol" not in frame:
        frame["vol"] = 0.0
    if "amount" not in frame:
        frame["amount"] = 0.0
    if timeframe == "1d":
        out = frame.reset_index()
    else:
        rule = "W-FRI" if timeframe == "1w" else "ME"
        out = frame.resample(rule, label="right", closed="right").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            vol=("vol", "sum"), amount=("amount", "sum"),
        ).dropna(subset=list(PRICE_COLUMNS)).reset_index()
    return out[["date", "open", "high", "low", "close", "vol", "amount"]].reset_index(drop=True)


def _normalize_timeframe(value: str) -> str:
    aliases = {"d": "1d", "day": "1d", "daily": "1d", "1d": "1d",
               "w": "1w", "week": "1w", "weekly": "1w", "1w": "1w",
               "m": "1m", "month": "1m", "monthly": "1m", "1m": "1m"}
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError("timeframe must be one of 1d/day, 1w/week, or 1m/month") from exc


class ParquetDataProvider:
    """Provider for the repository's ``stock_daily_qfq.parquet`` file.

    The parquet is read once and indexed by symbol.  For very large external
    files, callers should provide a database-backed provider instead.
    """

    def __init__(self, path: str | Path = "database/processed/stock_daily_qfq.parquet"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        frame = pd.read_parquet(self.path)
        frame["date"] = pd.to_datetime(frame["date"])
        self._data = {
            str(symbol): group.sort_values("date").reset_index(drop=True)
            for symbol, group in frame.groupby("ts_code", sort=False, observed=True)
        }
        self._bars_cache: dict[str, dict[str, pd.DataFrame]] = {}

    def get_symbols(self) -> list[str]:
        return list(self._data)

    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        try:
            return self._data[str(symbol)].copy()
        except KeyError as exc:
            raise KeyError(f"unknown symbol: {symbol}") from exc

    def get_symbol_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Cached aggregation used by repeated full-universe searches."""
        timeframe = _normalize_timeframe(timeframe)
        cache = self._bars_cache.setdefault(timeframe, {})
        if str(symbol) not in cache:
            cache[str(symbol)] = aggregate_bars(self.get_symbol_data(symbol), timeframe)
        return cache[str(symbol)].copy()


@dataclass(frozen=True)
class SimilarityConfig:
    local_top_n: int = 20
    recall_n: int = 1000
    top_k: int = 50
    exclusion_ratio: float = 0.25
    sakoe_chiba_ratio: float = 0.08
    clip_value: float = 5.0
    mass_weight: float = 0.20
    dtw_weight: float = 0.35
    aligned_weight: float = 0.25
    return_weight: float = 0.15
    volatility_weight: float = 0.05
    use_volume: bool = True
    feature_weights: tuple[float, ...] = tuple(DEFAULT_FEATURE_WEIGHTS.tolist())


@dataclass
class SimilarityMatch:
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    score: float
    mass_distance: float
    dtw_distance: float
    aligned_distance: float
    total_return_diff: float
    volatility_diff: float
    mass_rank: float = 0.0
    dtw_rank: float = 0.0
    aligned_rank: float = 0.0
    return_rank: float = 0.0
    volatility_rank: float = 0.0
    window_length: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("start_date", "end_date"):
            result[key] = pd.Timestamp(result[key]).strftime("%Y-%m-%d")
        return result


class KlineSimilarityEngine:
    """Search historical equal-length K-line windows across an entire universe."""

    def __init__(self, provider: MarketDataProvider, config: SimilarityConfig | None = None):
        self.provider = provider
        self.config = config or SimilarityConfig()
        if len(self.config.feature_weights) != len(FEATURE_NAMES):
            raise ValueError("feature_weights must contain seven values")

    def search(
        self,
        query_symbol: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        *,
        timeframe: str = "1d",
        top_k: int | None = None,
        history_only: bool = True,
        recall_n: int | None = None,
        exclude_query: bool = True,
    ) -> list[SimilarityMatch]:
        """Find the most similar historical windows.

        ``history_only=True`` (the default) requires every candidate to end
        before the query starts, preventing look-ahead.  Set it to ``False``
        for a symmetric historical scan; overlapping windows from the query
        stock are still removed by default.
        """
        tf = _normalize_timeframe(timeframe)
        query = _get_bars(self.provider, query_symbol, tf)
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        q = query[(query.date >= start) & (query.date <= end)].reset_index(drop=True)
        if len(q) < 3:
            raise ValueError("query interval must contain at least three complete K bars")
        q = _valid_bars(q)
        if len(q) < 3:
            raise ValueError("query interval has fewer than three valid K bars")
        m = len(q)
        q_close = np.log(q.close.to_numpy(float))
        q_features = _features(q, self.config.use_volume)
        candidates: list[dict] = []
        limit = int(recall_n or self.config.recall_n)
        for symbol in self.provider.get_symbols():
            frame = _valid_bars(_get_bars(self.provider, symbol, tf))
            if len(frame) < m:
                continue
            distances = _mass_distances(q_close, np.log(frame.close.to_numpy(float)))
            radius = max(5, int(m * self.config.exclusion_ratio))
            selected = _local_minima(distances, self.config.local_top_n, radius)
            for idx, mass_distance in selected:
                c_start, c_end = frame.date.iloc[idx], frame.date.iloc[idx + m - 1]
                if history_only and c_end >= start:
                    continue
                if exclude_query and str(symbol) == str(query_symbol):
                    overlap = _overlap_ratio(c_start, c_end, start, end)
                    if overlap > 0.2:
                        continue
                candidates.append({"symbol": str(symbol), "frame": frame.iloc[idx:idx + m].reset_index(drop=True),
                                   "mass_distance": float(mass_distance)})
        candidates.sort(key=lambda x: x["mass_distance"])
        candidates = candidates[:limit]
        if not candidates:
            return []
        # Global robust feature scale.  Median/IQR is deliberately computed
        # over the query and recalled windows, avoiding a second full-market pass.
        matrices = [q_features] + [_features(c["frame"], self.config.use_volume) for c in candidates]
        stack = np.vstack(matrices)
        med = np.nanmedian(stack, axis=0)
        iqr = np.nanpercentile(stack, 75, axis=0) - np.nanpercentile(stack, 25, axis=0)
        qn = np.clip((q_features - med) / (iqr + 1e-9), -self.config.clip_value, self.config.clip_value)
        weights = np.asarray(self.config.feature_weights, dtype=float)
        if not self.config.use_volume:
            weights[-1] = 0.0
        metrics: list[SimilarityMatch] = []
        for item in candidates:
            raw = _features(item["frame"], self.config.use_volume)
            cn = np.clip((raw - med) / (iqr + 1e-9), -self.config.clip_value, self.config.clip_value)
            weighted_q, weighted_c = qn * np.sqrt(weights), cn * np.sqrt(weights)
            dtw_distance = _constrained_dtw(weighted_q, weighted_c,
                                             max(1, round(m * self.config.sakoe_chiba_ratio)))
            aligned = float(np.sqrt(np.mean(np.sum((weighted_q - weighted_c) ** 2, axis=1))))
            cframe = item["frame"]
            qret = float(np.log(q.close.iloc[-1] / q.close.iloc[0]))
            cret = float(np.log(cframe.close.iloc[-1] / cframe.close.iloc[0]))
            qvol = float(np.std(np.diff(q_close)))
            cvol = float(np.std(np.diff(np.log(cframe.close.to_numpy(float)))))
            metrics.append(SimilarityMatch(
                symbol=item["symbol"], start_date=cframe.date.iloc[0], end_date=cframe.date.iloc[-1],
                score=0.0, mass_distance=item["mass_distance"], dtw_distance=dtw_distance,
                aligned_distance=aligned, total_return_diff=abs(qret - cret),
                volatility_diff=float(abs(np.log((cvol + 1e-9) / (qvol + 1e-9)))), window_length=m,
            ))
        _assign_rank_scores(metrics, self.config)
        metrics.sort(key=lambda x: (-x.score, x.symbol, x.start_date))
        return _global_nms(metrics, top_k or self.config.top_k, radius=max(5, int(m * self.config.exclusion_ratio)))


def _valid_bars(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    vals = frame[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    mask = vals.gt(0).all(axis=1) & np.isfinite(vals).all(axis=1)
    return frame.loc[mask].reset_index(drop=True)


def _get_bars(provider: MarketDataProvider, symbol: str, timeframe: str) -> pd.DataFrame:
    getter = getattr(provider, "get_symbol_bars", None)
    if getter is not None:
        return getter(symbol, timeframe)
    return aggregate_bars(provider.get_symbol_data(symbol), timeframe)


def _features(frame: pd.DataFrame, use_volume: bool = True) -> np.ndarray:
    o, h, l, c = [frame[x].to_numpy(float) for x in PRICE_COLUMNS]
    prev = np.r_[c[0], c[:-1]]
    eps = 1e-12
    vol = pd.to_numeric(frame.get("vol", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0).to_numpy(float)
    prev_vol = np.r_[vol[0], vol[:-1]]
    result = np.column_stack([
        np.log(np.maximum(c, eps) / np.maximum(prev, eps)),
        np.log(np.maximum(o, eps) / np.maximum(prev, eps)),
        np.log(np.maximum(c, eps) / np.maximum(o, eps)),
        (h - np.maximum(o, c)) / np.maximum(prev, eps),
        (np.minimum(o, c) - l) / np.maximum(prev, eps),
        (h - l) / np.maximum(prev, eps),
        np.log((vol + eps) / (prev_vol + eps)),
    ])
    result[0, 0] = result[0, 1] = result[0, 6] = 0.0
    result[:, 3:5] = np.maximum(result[:, 3:5], 0.0)
    if not use_volume:
        result[:, 6] = 0.0
    return result


def _mass_distances(query: np.ndarray, series: np.ndarray) -> np.ndarray:
    m, n = len(query), len(series)
    if n < m:
        return np.empty(0)
    q = (query - query.mean()) / max(query.std(), 1e-12)
    windows = np.lib.stride_tricks.sliding_window_view(series, m)
    means, stds = windows.mean(axis=1), windows.std(axis=1)
    z = (windows - means[:, None]) / np.maximum(stds[:, None], 1e-12)
    return np.sqrt(np.mean((z - q[None, :]) ** 2, axis=1))


def _local_minima(distances: np.ndarray, count: int, radius: int) -> list[tuple[int, float]]:
    work = distances.copy()
    result = []
    for _ in range(max(0, count)):
        if not len(work) or not np.isfinite(work).any():
            break
        idx = int(np.nanargmin(work))
        result.append((idx, float(work[idx])))
        work[max(0, idx - radius): min(len(work), idx + radius + 1)] = np.inf
    return result


def _constrained_dtw(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    n, m = len(a), len(b)
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - radius), min(m, i + radius)
        for j in range(lo, hi + 1):
            d = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[n, m] / max(n + m, 1)))


def _rank(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float) / (len(values) - 1)
    return ranks


def _assign_rank_scores(items: list[SimilarityMatch], config: SimilarityConfig) -> None:
    rs = [_rank([x.mass_distance for x in items]), _rank([x.dtw_distance for x in items]),
          _rank([x.aligned_distance for x in items]), _rank([x.total_return_diff for x in items]),
          _rank([x.volatility_diff for x in items])]
    for i, item in enumerate(items):
        item.mass_rank, item.dtw_rank, item.aligned_rank, item.return_rank, item.volatility_rank = [float(x[i]) for x in rs]
        distance = (config.mass_weight * item.mass_rank + config.dtw_weight * item.dtw_rank +
                    config.aligned_weight * item.aligned_rank + config.return_weight * item.return_rank +
                    config.volatility_weight * item.volatility_rank)
        item.score = float(1.0 - distance)


def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    left, right = max(pd.Timestamp(a_start), pd.Timestamp(b_start)), min(pd.Timestamp(a_end), pd.Timestamp(b_end))
    if right < left:
        return 0.0
    return (right - left).days / max((pd.Timestamp(b_end) - pd.Timestamp(b_start)).days + 1, 1)


def _global_nms(items: list[SimilarityMatch], top_k: int, radius: int) -> list[SimilarityMatch]:
    selected: list[SimilarityMatch] = []
    for item in items:
        if any(item.symbol == old.symbol and abs((item.start_date - old.start_date).days) <= radius for old in selected):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected
