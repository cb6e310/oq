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
import heapq
import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
FEATURE_NAMES = (
    "r_close", "gap", "body_return", "upper_wick", "lower_wick",
    "range_pct", "volume_log_change",
)
EVENT_NAMES = ("limit_up", "limit_down", "one_word_up", "one_word_down")
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
    top_k: int = 20
    exclusion_ratio: float = 0.25
    sakoe_chiba_ratio: float = 0.08
    clip_value: float = 5.0
    mass_weight: float = 0.15
    dtw_weight: float = 0.30
    aligned_weight: float = 0.20
    return_weight: float = 0.10
    volatility_weight: float = 0.05
    special_event_weight: float = 0.20
    event_recall_weight: float = 0.75
    hard_event_match: bool = True
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
    event_distance: float = 0.0
    mass_rank: float = 0.0
    dtw_rank: float = 0.0
    aligned_rank: float = 0.0
    return_rank: float = 0.0
    volatility_rank: float = 0.0
    event_rank: float = 0.0
    window_length: int = 0
    start_idx: int = -1
    end_idx: int = -1

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
        self.last_search_stats: dict[str, int | float | str] = {}

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
        q_source = query[(query.date >= start) & (query.date <= end)]
        if q_source.empty:
            raise ValueError("query interval has no complete K bars")
        first_query_pos = int(q_source.index[0])
        query_prev_close = (float(query.close.iloc[first_query_pos - 1])
                            if first_query_pos > 0 else None)
        q = q_source.reset_index(drop=True)
        if len(q) < 3:
            raise ValueError("query interval must contain at least three complete K bars")
        q = _valid_bars(q)
        if len(q) < 3:
            raise ValueError("query interval has fewer than three valid K bars")
        m = len(q)
        q_close = np.log(q.close.to_numpy(float))
        q_features = _features(q, self.config.use_volume)
        q_events = _event_features(q, query_symbol, previous_close=query_prev_close)
        candidates: list[dict] = []
        stats = {
            "symbols_total": 0, "symbols_eligible": 0, "windows_total": 0,
            "local_recall_total": 0, "mass_path_total": 0,
            "structure_path_total": 0, "event_path_total": 0,
            "candidates_before_global_limit": 0,
        }
        limit = int(recall_n or self.config.recall_n)
        for symbol in self.provider.get_symbols():
            stats["symbols_total"] += 1
            frame = _valid_bars(_get_bars(self.provider, symbol, tf))
            if len(frame) < m:
                continue
            stats["symbols_eligible"] += 1
            distances = _mass_distances(q_close, np.log(frame.close.to_numpy(float)))
            structure_distances = _structure_distances(q, frame, self.config.use_volume)
            frame_events = _event_features(frame, str(symbol))
            event_distances = _event_window_distances(q_events, frame_events)
            stats["windows_total"] += len(distances)
            recall_distances = distances + self.config.event_recall_weight * event_distances
            # Mask illegal windows before local minima. Otherwise future or
            # query-overlapping windows can consume the local_top_n budget.
            legal = np.ones(len(distances), dtype=bool)
            if history_only:
                legal &= frame.date.to_numpy()[np.arange(len(distances)) + m - 1] < start.to_datetime64()
            if exclude_query and str(symbol) == str(query_symbol):
                self_legal = np.ones(len(legal), dtype=bool)
                for idx in range(len(legal)):
                    self_legal[idx] = _overlap_ratio(frame.date.iloc[idx], frame.date.iloc[idx + m - 1], start, end) <= 0.2
                legal &= self_legal
            recall_distances[~legal] = np.inf
            event_distances[~legal] = np.inf
            structure_distances[~legal] = np.inf
            radius = max(5, int(m * self.config.exclusion_ratio))
            selected = _local_minima(recall_distances, self.config.local_top_n, radius)
            selected = [(idx, value) for idx, value in selected if np.isfinite(value)]
            selected_paths: dict[int, set[str]] = {idx: {"mass"} for idx, _ in selected}
            # A second cheap path recalls windows with similar candle bodies,
            # gaps and shadows even when their closing-price trajectory is
            # not close enough for the MASS path.
            structure_selected = _local_minima(structure_distances, self.config.local_top_n, radius)
            for idx, value in structure_selected:
                if np.isfinite(value):
                    selected_paths.setdefault(idx, set()).add("structure")
            # Event-bearing queries get a second, independent event shortlist.
            # This prevents a strong close-shape window from crowding an exact
            # limit-up/limit-down signature out of the per-stock recall list.
            if float(q_events.sum()) > 0:
                event_selected = _local_minima(event_distances, self.config.local_top_n, radius)
                for idx, _ in event_selected:
                    if np.isfinite(event_distances[idx]):
                        selected_paths.setdefault(idx, set()).add("event")
            stats["local_recall_total"] += len(selected_paths)
            stats["mass_path_total"] += sum("mass" in paths for paths in selected_paths.values())
            stats["structure_path_total"] += sum("structure" in paths for paths in selected_paths.values())
            stats["event_path_total"] += sum("event" in paths for paths in selected_paths.values())
            for idx, paths in selected_paths.items():
                c_start, c_end = frame.date.iloc[idx], frame.date.iloc[idx + m - 1]
                previous_close = float(frame.close.iloc[idx - 1]) if idx > 0 else None
                candidates.append({"symbol": str(symbol),
                                   "mass_distance": float(distances[idx]),
                                   "event_distance": float(event_distances[idx]), "start_idx": idx,
                                   "end_idx": idx + m - 1, "previous_close": previous_close,
                                   "recall_paths": "+".join(sorted(paths))})
        stats["candidates_before_global_limit"] = len(candidates)
        candidates = _bounded_candidates(candidates, limit, self.config.event_recall_weight)
        candidates = candidates[:limit]
        if not candidates:
            stats["candidates_after_global_limit"] = 0
            stats["final_results"] = 0
            self.last_search_stats = stats
            return []
        stats["candidates_after_global_limit"] = len(candidates)
        # Only now materialize OHLCV windows for the bounded Stage-2 set.
        # Stage 1 candidates remain metadata-only across the full universe.
        for item in candidates:
            bars = _valid_bars(_get_bars(self.provider, item["symbol"], tf))
            item["frame"] = bars.iloc[item["start_idx"]:item["end_idx"] + 1].reset_index(drop=True)
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
            # The first bar of a candidate must be compared with the close
            # immediately before the window.  Recomputing events on the
            # sliced frame alone would silently lose a first-bar limit-up or
            # limit-down event (the most important event in many queries).
            candidate_prev_close = item.get("previous_close")
            candidate_events = _event_features(
                item["frame"], item["symbol"], previous_close=candidate_prev_close
            )
            # Event channels are deliberately not robust-scaled: 0/1 event
            # identity must remain visible to DTW and cannot be washed out by
            # a large cross-market IQR.
            event_scale = np.sqrt(max(self.config.special_event_weight, 0.0))
            weighted_q = np.hstack([weighted_q, q_events * event_scale])
            weighted_c = np.hstack([weighted_c, candidate_events * event_scale])
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
                event_distance=float(np.sqrt(np.mean((q_events - candidate_events) ** 2))),
                volatility_diff=float(abs(np.log((cvol + 1e-9) / (qvol + 1e-9)))), window_length=m,
                start_idx=item["start_idx"], end_idx=item["end_idx"],
            ))
        _assign_rank_scores(metrics, self.config, event_active=float(q_events.sum()) > 0)
        # Event features still affect both recall and the composite score
        # (and therefore remain a hard consideration for board-event-heavy
        # queries), but the public ranking must be ordered by the displayed
        # similarity score.  Sorting by event distance first made a lower
        # score appear above a higher score in the UI.
        metrics.sort(key=lambda x: (-x.score, x.symbol, x.start_date))
        result = _global_nms(metrics, top_k or self.config.top_k, overlap_threshold=self.config.exclusion_ratio)
        stats["final_results"] = len(result)
        self.last_search_stats = stats
        return result


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


def _structure_distances(query: pd.DataFrame, series: pd.DataFrame,
                         use_volume: bool = True) -> np.ndarray:
    """Cheap candle-structure recall distance for every equal-length window.

    This deliberately excludes the close-trajectory channel used by MASS and
    compares gap/body/wicks/range (plus volume change when enabled).  It is a
    recall signal only; the full OHLCV + event DTW remains the final judge.
    """
    qf = _features(query, use_volume)[:, 1:]
    sf = _features(series, use_volume)[:, 1:]
    m = len(qf)
    if len(sf) < m:
        return np.empty(0)
    scale = np.nanstd(qf, axis=0)
    scale = np.maximum(scale, 1e-3)
    qn = qf / scale
    windows = np.lib.stride_tricks.sliding_window_view(sf, m, axis=0)
    if windows.ndim == 3 and windows.shape[1] != m:
        windows = np.swapaxes(windows, 1, 2)
    return np.sqrt(np.mean((windows / scale - qn[None, :, :]) ** 2, axis=(1, 2)))


def _limit_pct(symbol: str | None) -> float:
    """Best-effort board limit based on A-share code when ST metadata is absent."""
    code = str(symbol or "").split(".")[0]
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8")):
        return 0.30
    return 0.10


def _event_features(frame: pd.DataFrame, symbol: str | None = None,
                    previous_close: float | None = None) -> np.ndarray:
    """Return independent limit-up/down and one-word event channels.

    A one-word board is detected from OHLC equality and a board-sized return.
    Prices are rounded to cents before equality checks, matching Chinese daily
    limit-price quotation.  For weekly/monthly bars this is a conservative
    period-level approximation; daily searches use exact daily events.
    """
    close = frame.close.to_numpy(float)
    open_ = frame.open.to_numpy(float)
    high = frame.high.to_numpy(float)
    low = frame.low.to_numpy(float)
    prev = np.r_[close[0] if previous_close is None else previous_close, close[:-1]]
    limit = _limit_pct(symbol)
    # A-share limit prices are quoted to cents.  Use the rounded theoretical
    # limit price rather than a broad percentage band (e.g. 8.5%~11.5%),
    # otherwise ordinary large candles are incorrectly treated as boards.
    valid_prev = np.isfinite(prev) & (prev > 0)
    up_price = np.round(prev * (1.0 + limit), 2)
    down_price = np.round(prev * (1.0 - limit), 2)
    up = valid_prev & (np.abs(close - up_price) <= 0.011)
    down = valid_prev & (np.abs(close - down_price) <= 0.011)
    equal = (np.round(open_, 2) == np.round(high, 2)) & (np.round(high, 2) == np.round(low, 2)) & (np.round(low, 2) == np.round(close, 2))
    return np.column_stack([up.astype(float), down.astype(float),
                            (up & equal).astype(float), (down & equal).astype(float)])


def _event_window_distances(query_events: np.ndarray, series_events: np.ndarray) -> np.ndarray:
    m = len(query_events)
    if len(series_events) < m:
        return np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(series_events, m, axis=0)
    # sliding_window_view places the window dimension last for a 2-D input.
    if windows.shape[1] != m:
        windows = np.swapaxes(windows, 1, 2)
    return np.sqrt(np.mean((windows - query_events[None, :, :]) ** 2, axis=(1, 2)))


def _mass_distances(query: np.ndarray, series: np.ndarray) -> np.ndarray:
    m, n = len(query), len(series)
    if n < m:
        return np.empty(0)
    q = (query - query.mean()) / max(query.std(), 1e-12)
    windows = np.lib.stride_tricks.sliding_window_view(series, m)
    means, stds = windows.mean(axis=1), windows.std(axis=1)
    z = (windows - means[:, None]) / np.maximum(stds[:, None], 1e-12)
    return np.sqrt(np.mean((z - q[None, :]) ** 2, axis=1))


def _bounded_candidates(candidates: list[dict], limit: int, event_weight: float) -> list[dict]:
    """Keep only the global best recall candidates without retaining all ties."""
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda x: x["mass_distance"] + event_weight * x["event_distance"])
    # Reserve part of the finite DTW budget for structure-path recalls so a
    # close-shape-heavy market cannot crowd them all out.
    structure_candidates = [x for x in candidates if "structure" in str(x.get("recall_paths", ""))]
    reserve_n = min(max(1, limit // 3), len(structure_candidates))
    reserved = _heap_best_candidates(structure_candidates, reserve_n, event_weight)
    reserved_ids = {id(x) for x in reserved}
    remainder = [x for x in candidates if id(x) not in reserved_ids]
    return reserved + _heap_best_candidates(remainder, max(0, limit - len(reserved)), event_weight)


def _heap_best_candidates(candidates: list[dict], limit: int, event_weight: float) -> list[dict]:
    if limit <= 0 or not candidates:
        return []
    heap: list[tuple[float, int, dict]] = []
    for serial, item in enumerate(candidates):
        key = item["mass_distance"] + event_weight * item["event_distance"]
        entry = (-float(key), serial, item)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [x[2] for x in sorted(heap, key=lambda x: -x[0])]


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
    """Banded rolling-row constrained DTW; memory is O(window * features)."""
    n, m = len(a), len(b)
    inf = float("inf")
    prev = np.full(m + 1, inf)
    curr = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr.fill(inf)
        lo, hi = max(1, i - radius), min(m, i + radius)
        for j in range(lo, hi + 1):
            d = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            curr[j] = d + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[m] / max(n + m, 1)))


def _rank(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    first_rank = np.maximum.accumulate(np.where(starts, np.arange(len(values)), 0))
    ranks[order] = first_rank / (len(values) - 1)
    return ranks


def _assign_rank_scores(items: list[SimilarityMatch], config: SimilarityConfig,
                        event_active: bool = False) -> None:
    rs = [_rank([x.mass_distance for x in items]), _rank([x.dtw_distance for x in items]),
          _rank([x.aligned_distance for x in items]), _rank([x.total_return_diff for x in items]),
          _rank([x.volatility_diff for x in items]), _rank([x.event_distance for x in items])]
    for i, item in enumerate(items):
        item.mass_rank, item.dtw_rank, item.aligned_rank, item.return_rank, item.volatility_rank, item.event_rank = [float(x[i]) for x in rs]
        distance = (config.mass_weight * item.mass_rank + config.dtw_weight * item.dtw_rank +
                    config.aligned_weight * item.aligned_rank + config.return_weight * item.return_rank +
                    config.volatility_weight * item.volatility_rank + config.special_event_weight * item.event_rank)
        item.score = float(1.0 - distance)
        # ``hard_event_match`` gives exact event signatures a clear priority
        # band, while keeping a continuous penalty among non-exact windows.
        # This avoids collapsing every result below 0.5 when the market has
        # no perfect historical event sequence.
        if config.hard_event_match and event_active:
            base_score = item.score
            if item.event_distance <= 1e-12:
                item.score = 0.5 + 0.5 * base_score
            else:
                item.score = 0.5 * base_score - 0.5 * min(item.event_distance, 1.0)


def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    left, right = max(pd.Timestamp(a_start), pd.Timestamp(b_start)), min(pd.Timestamp(a_end), pd.Timestamp(b_end))
    if right < left:
        return 0.0
    return (right - left).days / max((pd.Timestamp(b_end) - pd.Timestamp(b_start)).days + 1, 1)


def _global_nms(items: list[SimilarityMatch], top_k: int, overlap_threshold: float = 0.25) -> list[SimilarityMatch]:
    selected: list[SimilarityMatch] = []
    for item in items:
        if any(item.symbol == old.symbol and _interval_overlap(item.start_idx, item.end_idx, old.start_idx, old.end_idx) > overlap_threshold for old in selected):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def _interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    if min(end_a, end_b) < max(start_a, start_b):
        return 0.0
    intersection = min(end_a, end_b) - max(start_a, start_b) + 1
    return intersection / max(min(end_a - start_a + 1, end_b - start_b + 1), 1)
