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

from .elastic_fast import coarse_distances as _compiled_coarse_distances


PRICE_COLUMNS = ("open", "high", "low", "close")
FEATURE_NAMES = (
    "r_close", "gap", "body_return", "upper_wick", "lower_wick",
    "range_pct", "volume_log_change",
)
EVENT_NAMES = ("limit_up", "limit_down", "one_word_up", "one_word_down",
               "gap_limit_up", "t_board_up")
SPECIAL_EVENT_CHANNELS = (2, 3, 4, 5)
SPECIAL_EVENT_WEIGHTS = np.array((3.0, 3.0, 1.0, 4.0))
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
    length_mode: str = "fixed"
    min_length_ratio: float = 0.75
    max_length_ratio: float = 1.35
    max_length_delta: int = 20
    max_length_candidates: int = 9
    elastic_coarse_step: int = 8
    min_window_length: int = 8
    max_window_length: int = 250
    normalized_warp_band: float = 0.10
    max_repeat_step: int = 2
    gap_penalty: float = 0.35
    salient_gap_multiplier: float = 2.5
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
    special_event_match: float = 0.0
    missing_special_events: int = 0
    special_event_required: bool = False
    mass_rank: float = 0.0
    dtw_rank: float = 0.0
    aligned_rank: float = 0.0
    return_rank: float = 0.0
    volatility_rank: float = 0.0
    event_rank: float = 0.0
    window_length: int = 0
    start_idx: int = -1
    end_idx: int = -1
    query_length: int = 0
    length_ratio: float = 1.0
    match_mode: str = "fixed"
    elastic_shape_distance: float = 0.0
    state_distance: float = 0.0
    anchor_distance: float = 0.0
    length_penalty: float = 0.0
    warp_path_length: int = 0
    skipped_query_bars: int = 0
    skipped_candidate_bars: int = 0

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
        match_mode: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        return_diagnostics: bool = False,
        latest_only: bool = False,
    ) -> list[SimilarityMatch] | dict[str, object]:
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
        latest_date = None
        if latest_only:
            latest_date = max(
                (_get_bars(self.provider, symbol, tf).date.max()
                 for symbol in self.provider.get_symbols()), default=pd.NaT,
            )
            if pd.isna(latest_date):
                raise ValueError("market data has no latest K bars")
        mode = str(match_mode or self.config.length_mode).lower()
        if mode not in {"fixed", "elastic"}:
            raise ValueError("match_mode must be 'fixed' or 'elastic'")
        if latest_only and top_k is None:
            top_k = 10
        if mode == "elastic":
            result = _elastic_search(
                self, query_symbol, q, q_close, q_features, q_events, start, end,
                tf, top_k, history_only, recall_n, exclude_query,
                min_length=min_length, max_length=max_length,
                latest_date=latest_date,
            )
            if return_diagnostics:
                return {"results": result, "diagnostics": dict(self.last_search_stats)}
            return result
        candidates: list[dict] = []
        stats = {
            "candidate_scope": "latest" if latest_only else "history",
            "symbols_total": 0, "symbols_eligible": 0, "windows_total": 0,
            "local_recall_total": 0, "mass_path_total": 0,
            "structure_path_total": 0, "event_path_total": 0,
            "candidates_before_global_limit": 0,
        }
        limit = int(recall_n or self.config.recall_n)
        for symbol in self.provider.get_symbols():
            stats["symbols_total"] += 1
            if latest_only and exclude_query and str(symbol) == str(query_symbol):
                continue
            frame = _valid_bars(_get_bars(self.provider, symbol, tf))
            if len(frame) < m:
                continue
            if latest_only and (latest_date - frame.date.iloc[-1]).days > _latest_lag_days(tf):
                continue
            stats["symbols_eligible"] += 1
            if latest_only:
                idx = len(frame) - m
                window = frame.iloc[idx:].reset_index(drop=True)
                previous_close = float(frame.close.iloc[idx - 1]) if idx else None
                frame_events = _event_features(window, str(symbol), previous_close=previous_close)
                coverage, missing = _special_event_match(q_events, frame_events)
                candidates.append({
                    "symbol": str(symbol), "mass_distance": float(_mass_distances(q_close, np.log(window.close.to_numpy(float)))[0]),
                    "event_distance": float(_event_window_distances(q_events, frame_events)[0]),
                    "start_idx": idx, "end_idx": len(frame) - 1, "previous_close": previous_close,
                    "special_event_match": coverage, "missing_special_events": missing,
                    "recall_paths": "mass+structure+event",
                })
                stats["windows_total"] += 1
                stats["local_recall_total"] += 1
                continue
            distances = _mass_distances(q_close, np.log(frame.close.to_numpy(float)))
            structure_distances = _structure_distances(q, frame, self.config.use_volume)
            frame_events = _event_features(frame, str(symbol))
            event_distances = _event_window_distances(q_events, frame_events)
            special_coverage, special_missing = _special_event_coverage_windows(
                q_events, frame_events, np.arange(len(distances)), m,
            )
            stats["windows_total"] += len(distances)
            recall_distances = (distances + self.config.event_recall_weight * event_distances
                                + 2.0 * (1.0 - special_coverage))
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
                                   "special_event_match": float(special_coverage[idx]),
                                   "missing_special_events": int(special_missing[idx]),
                                   "recall_paths": "+".join(sorted(paths))})
        stats["candidates_before_global_limit"] = len(candidates)
        candidates = _bounded_candidates(candidates, limit, self.config.event_recall_weight,
                                         special_required=bool(q_events[:, SPECIAL_EVENT_CHANNELS].any()))
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
            special_match, missing_special = _special_event_match(q_events, candidate_events)
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
                special_event_match=special_match, missing_special_events=missing_special,
                special_event_required=bool(q_events[:, SPECIAL_EVENT_CHANNELS].any()),
                volatility_diff=float(abs(np.log((cvol + 1e-9) / (qvol + 1e-9)))), window_length=m,
                start_idx=item["start_idx"], end_idx=item["end_idx"],
                query_length=m, length_ratio=1.0, match_mode="fixed",
            ))
        _assign_rank_scores(metrics, self.config, event_active=float(q_events.sum()) > 0)
        _prioritize_special_events(metrics, q_events)
        # Event features still affect both recall and the composite score
        # (and therefore remain a hard consideration for board-event-heavy
        # queries), but the public ranking must be ordered by the displayed
        # similarity score.  Sorting by event distance first made a lower
        # score appear above a higher score in the UI.
        metrics.sort(key=lambda x: (-x.score, x.symbol, x.start_date))
        result = _global_nms(metrics, top_k or self.config.top_k, overlap_threshold=self.config.exclusion_ratio)
        stats["final_results"] = len(result)
        self.last_search_stats = stats
        if return_diagnostics:
            return {"results": result, "diagnostics": dict(self.last_search_stats)}
        return result


def _valid_bars(frame: pd.DataFrame) -> pd.DataFrame:
    vals = frame.loc[:, PRICE_COLUMNS].to_numpy(dtype=float, na_value=np.nan)
    mask = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1)
    return frame.loc[mask].reset_index(drop=True)


def _latest_lag_days(timeframe: str) -> int:
    return {"1d": 7, "1w": 14, "1m": 45}[timeframe]


def _candidate_lengths(length: int, config: SimilarityConfig,
                       min_length: int | None = None,
                       max_length: int | None = None,
                       mode: str | None = None) -> list[int]:
    """Build a bounded length grid without expanding short queries too far."""
    if (mode or config.length_mode) == "fixed":
        return [length]
    if length < 8:
        values = {length - 1, length, length + 1}
    else:
        delta = min(max(2, round(0.20 * length)), config.max_length_delta)
        values = {length - delta, length, length + delta}
        values.update(round(length * ratio) for ratio in (0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35))
    lo = max(config.min_window_length, min_length or 0, 3)
    hi = min(config.max_window_length, max_length or config.max_window_length)
    if lo > hi:
        raise ValueError("min_length must not exceed max_length")
    values = sorted(x for x in values if lo <= x <= hi)
    if not values:
        values = [min(max(length, lo), hi)]
    return sorted(set(values), key=lambda x: (abs(x - length), x))[:max(1, config.max_length_candidates)]


def _coarse_bars(frame: pd.DataFrame, buckets: int) -> pd.DataFrame:
    """Aggregate a window into positional OHLCV buckets for cheap recall."""
    if len(frame) <= buckets:
        return frame.reset_index(drop=True)
    rows = []
    for group in np.array_split(np.arange(len(frame)), buckets):
        part = frame.iloc[group]
        rows.append({
            "date": part.date.iloc[-1], "open": part.open.iloc[0], "high": part.high.max(),
            "low": part.low.min(), "close": part.close.iloc[-1],
            "vol": part.get("vol", pd.Series(0.0, index=part.index)).sum(),
        })
    return pd.DataFrame(rows)


def _coarse_shape_distance(query: pd.DataFrame, series: pd.DataFrame, length: int) -> np.ndarray:
    buckets = min(len(query), length, int(np.clip(round(len(query) / 2), 8, 32)))
    q_close = np.log(_coarse_bars(query, buckets).close.to_numpy(float))
    q_norm = (q_close - q_close.mean()) / max(q_close.std(), 1e-12)
    if len(series) < length:
        return np.empty(0)
    result = np.full(len(series) - length + 1, np.inf)
    for start in range(len(result)):
        c_close = np.log(_coarse_bars(series.iloc[start:start + length], buckets).close.to_numpy(float))
        c_norm = (c_close - c_close.mean()) / max(c_close.std(), 1e-12)
        result[start] = float(np.sqrt(np.mean((q_norm - c_norm) ** 2)))
    return result


def _coarse_structure_distance(query: pd.DataFrame, series: pd.DataFrame,
                               length: int, use_volume: bool = True) -> np.ndarray:
    buckets = min(len(query), length, int(np.clip(round(len(query) / 2), 8, 32)))
    qf = _features(_coarse_bars(query, buckets), use_volume)[:, 1:]
    if len(series) < length:
        return np.empty(0)
    result = np.full(len(series) - length + 1, np.inf)
    scale = np.maximum(np.nanstd(qf, axis=0), 1e-3)
    for start in range(len(result)):
        cf = _features(_coarse_bars(series.iloc[start:start + length], buckets), use_volume)[:, 1:]
        result[start] = float(np.sqrt(np.mean(((qf - cf) / scale) ** 2)))
    return result


def _state_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return low-cardinality causal states and per-bar salience."""
    features = _features(frame, use_volume=True)
    returns, body = features[:, 0], np.abs(features[:, 2])
    upper, lower, ranges, volume = features[:, 3], features[:, 4], features[:, 5], features[:, 6]
    direction = np.select([returns > 0.004, returns < -0.004], [2, 0], default=1)
    volatility = np.select([ranges < 0.8, ranges > 1.5], [0, 2], default=1)
    body_type = np.select([body < 0.003, body > 0.025], [2, 0], default=1)
    wick_type = np.select([upper > lower * 1.8, lower > upper * 1.8], [0, 2], default=1)
    volume_state = np.select([volume < -0.35, volume > 0.50], [0, 2], default=1)
    states = np.column_stack([direction, volatility, body_type, wick_type, volume_state]).astype(np.int8)
    event = _event_features(frame)
    salience = np.clip(0.15 + 0.20 * np.minimum(ranges, 3.0) + 0.15 * np.minimum(np.abs(volume), 3.0), 0.0, 0.85)
    salience = np.maximum(salience, event.max(axis=1))
    return states, salience


def _state_bucket_modes(states: np.ndarray, buckets: int) -> np.ndarray:
    rows = []
    for group in np.array_split(np.arange(len(states)), min(buckets, len(states))):
        rows.append([np.bincount(states[group, col], minlength=3).argmax() for col in range(states.shape[1])])
    return np.asarray(rows, dtype=np.int8)


def _coarse_state_distance(query_states: np.ndarray, series_states: np.ndarray,
                           length: int) -> np.ndarray:
    if len(series_states) < length:
        return np.empty(0)
    buckets = min(len(query_states), length, int(np.clip(round(len(query_states) / 2), 8, 32)))
    q = _state_bucket_modes(query_states, buckets)
    result = np.full(len(series_states) - length + 1, np.inf)
    for start in range(len(result)):
        c = _state_bucket_modes(series_states[start:start + length], buckets)
        result[start] = float(np.mean(q != c))
    return result


def _event_bucket_modes(events: np.ndarray, buckets: int) -> np.ndarray:
    return np.asarray([events[group].max(axis=0) for group in np.array_split(np.arange(len(events)), min(buckets, len(events)))])


def _coarse_event_distance(query_events: np.ndarray, series_events: np.ndarray,
                           length: int) -> np.ndarray:
    if len(series_events) < length:
        return np.empty(0)
    buckets = min(len(query_events), length, int(np.clip(round(len(query_events) / 2), 8, 32)))
    q = _event_bucket_modes(query_events, buckets)
    result = np.full(len(series_events) - length + 1, np.inf)
    for start in range(len(result)):
        c = _event_bucket_modes(series_events[start:start + length], buckets)
        result[start] = float(np.sqrt(np.mean((q - c) ** 2)))
    return result


def _prepare_elastic_query(query: pd.DataFrame, query_states: np.ndarray,
                           query_events: np.ndarray, length: int,
                           use_volume: bool) -> tuple:
    buckets = min(len(query), length, int(np.clip(round(len(query) / 2), 8, 32)))
    groups = np.array_split(np.arange(length), buckets)
    q_coarse = _coarse_bars(query, buckets)
    q_close = np.log(q_coarse.close.to_numpy(float))
    q_norm = (q_close - q_close.mean()) / max(q_close.std(), 1e-12)
    q_features = _features(q_coarse, use_volume)[:, 1:]
    scale = np.maximum(np.nanstd(q_features, axis=0), 1e-3)
    q_states = _state_bucket_modes(query_states, buckets)
    q_events = _event_bucket_modes(query_events, buckets)
    return groups, q_norm, q_features, scale, q_states, q_events


def _prepare_elastic_frame(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(frame[col].to_numpy(float) for col in ("open", "high", "low", "close", "vol"))


def _elastic_coarse_distances(query: pd.DataFrame, frame: pd.DataFrame,
                              query_states: np.ndarray, frame_states: np.ndarray,
                              query_events: np.ndarray, frame_events: np.ndarray,
                              length: int, use_volume: bool, prepared: tuple | None = None,
                              frame_arrays: tuple[np.ndarray, ...] | None = None,
                              compiled_prices: np.ndarray | None = None,
                              start_step: int = 1
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the four coarse recall channels in window batches.

    This preserves the positional bucket rules of the scalar helpers while
    avoiding a DataFrame slice and feature calculation for every window.
    """
    count = len(frame) - length + 1
    if count <= 0:
        empty = np.empty(0)
        return empty, empty, empty, empty
    groups, q_norm, q_features, scale, q_states, q_events = (prepared or
        _prepare_elastic_query(query, query_states, query_events, length, use_volume))
    buckets = len(groups)
    if compiled_prices is not None and _compiled_coarse_distances is not None:
        starts = np.array([group[0] for group in groups], dtype=np.int64)
        ends = np.array([group[-1] + 1 for group in groups], dtype=np.int64)
        return _compiled_coarse_distances(compiled_prices, frame_states, frame_events,
                                          starts, ends, q_norm, q_features, scale,
                                          q_states, q_events, use_volume, start_step)
    opens, highs, lows, closes, volumes = (frame_arrays if frame_arrays is not None
                                           else _prepare_elastic_frame(frame))
    shape = np.empty(count)
    structure = np.empty(count)
    state = np.empty(count)
    event = np.empty(count)
    for base in range(0, count, 2048):
        stop = min(base + 2048, count)
        starts = np.arange(base, stop)
        n = len(starts)
        o = np.empty((n, buckets)); h = np.empty_like(o)
        l = np.empty_like(o); c = np.empty_like(o); v = np.empty_like(o)
        states = np.empty((n, buckets, frame_states.shape[1]), dtype=np.int8)
        events = np.empty((n, buckets, frame_events.shape[1]))
        for col, offsets in enumerate(groups):
            positions = starts[:, None] + offsets[None, :]
            o[:, col] = opens[starts + offsets[0]]
            h[:, col] = highs[positions].max(axis=1)
            l[:, col] = lows[positions].min(axis=1)
            c[:, col] = closes[starts + offsets[-1]]
            v[:, col] = volumes[positions].sum(axis=1)
            state_group = frame_states[positions]
            for channel in range(frame_states.shape[1]):
                counts = np.stack([(state_group[:, :, channel] == value).sum(axis=1)
                                   for value in range(3)], axis=1)
                states[:, col, channel] = counts.argmax(axis=1)
            events[:, col] = frame_events[positions].max(axis=1)
        log_close = np.log(c)
        normalized = (log_close - log_close.mean(axis=1, keepdims=True)) / np.maximum(log_close.std(axis=1, keepdims=True), 1e-12)
        shape[base:stop] = np.sqrt(np.mean((normalized - q_norm) ** 2, axis=1))
        prev = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
        prev_vol = np.concatenate([v[:, :1], v[:, :-1]], axis=1)
        eps = 1e-12
        features = np.stack([
            np.log(np.maximum(o, eps) / np.maximum(prev, eps)),
            np.log(np.maximum(c, eps) / np.maximum(o, eps)),
            np.maximum((h - np.maximum(o, c)) / np.maximum(prev, eps), 0.0),
            np.maximum((np.minimum(o, c) - l) / np.maximum(prev, eps), 0.0),
            (h - l) / np.maximum(prev, eps),
            np.log((v + eps) / (prev_vol + eps)),
        ], axis=2)
        features[:, 0, 0] = 0.0
        features[:, 0, -1] = 0.0
        if not use_volume:
            features[:, :, -1] = 0.0
        structure[base:stop] = np.sqrt(np.mean(((features - q_features) / scale) ** 2, axis=(1, 2)))
        state[base:stop] = np.mean(states != q_states, axis=(1, 2))
        event[base:stop] = np.sqrt(np.mean((events - q_events) ** 2, axis=(1, 2)))
    return shape, structure, state, event


def _state_edit_distance(query_states: np.ndarray, candidate_states: np.ndarray,
                         query_salience: np.ndarray, candidate_salience: np.ndarray) -> float:
    """Weighted Levenshtein distance for direction/volatility/candle states."""
    n, m = len(query_states), len(candidate_states)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + 0.5 * (1 + 2.5 * query_salience[i - 1])
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + 0.5 * (1 + 2.5 * candidate_salience[j - 1])
    weights = np.array([1.5, 0.8, 0.8, 0.6, 0.5])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = float(np.sum(weights * (query_states[i - 1] != candidate_states[j - 1])))
            dp[i, j] = min(
                dp[i - 1, j] + 0.5 * (1 + 2.5 * query_salience[i - 1]),
                dp[i, j - 1] + 0.5 * (1 + 2.5 * candidate_salience[j - 1]),
                dp[i - 1, j - 1] + substitution,
            )
    return float(dp[n, m] / max(n + m, 1))


def _anchor_signature(frame: pd.DataFrame) -> np.ndarray:
    close = np.log(frame.close.to_numpy(float))
    peak = int(np.argmax(close))
    trough = int(np.argmin(close[peak:])) + peak
    return np.array([
        peak / max(len(close) - 1, 1), trough / max(len(close) - 1, 1),
        float(close[peak] - close[0]), float(close[-1] - close[trough]),
    ])


def _anchor_distance(query: pd.DataFrame, candidate: pd.DataFrame) -> float:
    scale = np.array([1.0, 1.0, 0.10, 0.10])
    return float(np.sqrt(np.mean(((_anchor_signature(query) - _anchor_signature(candidate)) / scale) ** 2)))


def _resampled_distance(query: np.ndarray, candidate: np.ndarray) -> float:
    """Compare different-length arrays after temporary linear projection."""
    if len(query) == len(candidate):
        return float(np.sqrt(np.mean((query - candidate) ** 2)))
    target = np.linspace(0.0, 1.0, len(query))
    source = np.linspace(0.0, 1.0, len(candidate))
    projected = np.column_stack([
        np.interp(target, source, candidate[:, col])
        for col in range(candidate.shape[1])
    ])
    return float(np.sqrt(np.mean((query - projected) ** 2)))


def _elastic_dtw(a: np.ndarray, b: np.ndarray, a_salience: np.ndarray,
                 b_salience: np.ndarray, normalized_band: float,
                 max_repeat_step: int, gap_penalty: float,
                 salient_gap_multiplier: float) -> tuple[float, int, int, int]:
    """Elastic DTW with normalized band and bounded horizontal/vertical runs."""
    n, m = len(a), len(b)
    inf = float("inf")
    runs = max(1, int(max_repeat_step))
    dp = np.full((n + 1, m + 1, 1 + 2 * runs), inf)
    path_len = np.zeros((n + 1, m + 1, 1 + 2 * runs), dtype=np.int16)
    skipped_q = np.zeros((n + 1, m + 1, 1 + 2 * runs), dtype=np.int16)
    skipped_c = np.zeros((n + 1, m + 1, 1 + 2 * runs), dtype=np.int16)
    dp[0, 0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            legal = abs(i / max(n, 1) - j / max(m, 1)) <= normalized_band + 1e-12
            if not legal:
                continue
            if i and j:
                prev = int(np.argmin(dp[i - 1, j - 1]))
                if np.isfinite(dp[i - 1, j - 1, prev]):
                    dp[i, j, 0] = dp[i - 1, j - 1, prev] + float(np.sqrt(np.mean((a[i - 1] - b[j - 1]) ** 2)))
                    path_len[i, j, 0] = path_len[i - 1, j - 1, prev] + 1
                    skipped_q[i, j, 0] = skipped_q[i - 1, j - 1, prev]
                    skipped_c[i, j, 0] = skipped_c[i - 1, j - 1, prev]
            if j:
                for prev in range(1 + 2 * runs):
                    if prev > runs:
                        continue
                    if np.isfinite(dp[i, j - 1, prev]):
                        value = dp[i, j - 1, prev] + gap_penalty * (1 + salient_gap_multiplier * float(b_salience[j - 1]))
                        if prev == runs:
                            continue
                        state = 1 if prev == 0 else prev + 1
                        if value < dp[i, j, state]:
                            dp[i, j, state] = value
                            path_len[i, j, state] = path_len[i, j - 1, prev] + 1
                            skipped_q[i, j, state] = skipped_q[i, j - 1, prev]
                            skipped_c[i, j, state] = skipped_c[i, j - 1, prev] + 1
            if i:
                for prev in range(1 + 2 * runs):
                    if prev != 0 and prev < 1 + runs:
                        continue
                    if np.isfinite(dp[i - 1, j, prev]):
                        value = dp[i - 1, j, prev] + gap_penalty * (1 + salient_gap_multiplier * float(a_salience[i - 1]))
                        if prev == 2 * runs:
                            continue
                        state = 1 + runs if prev == 0 else prev + 1
                        if value < dp[i, j, state]:
                            dp[i, j, state] = value
                            path_len[i, j, state] = path_len[i - 1, j, prev] + 1
                            skipped_q[i, j, state] = skipped_q[i - 1, j, prev] + 1
                            skipped_c[i, j, state] = skipped_c[i - 1, j, prev]
    state = int(np.argmin(dp[n, m]))
    if not np.isfinite(dp[n, m, state]):
        return float("inf"), 0, 0, 0
    return (float(np.sqrt(dp[n, m, state] / max(path_len[n, m, state], 1))),
            int(path_len[n, m, state]), int(skipped_q[n, m, state]), int(skipped_c[n, m, state]))


def _elastic_search(engine: KlineSimilarityEngine, query_symbol: str, q: pd.DataFrame,
                    q_close: np.ndarray, q_features: np.ndarray, q_events: np.ndarray,
                    start: pd.Timestamp, end: pd.Timestamp, timeframe: str,
                    top_k: int | None, history_only: bool, recall_n: int | None,
                    exclude_query: bool, *, min_length: int | None,
                    max_length: int | None, latest_date: pd.Timestamp | None = None) -> list[SimilarityMatch]:
    """Run bounded multi-length recall followed by elastic Stage 2 alignment."""
    config = engine.config
    query_length = len(q)
    query_states, query_salience = _state_features(q)
    lengths = _candidate_lengths(query_length, config, min_length, max_length, mode="elastic")
    if min_length is None and max_length is None and len(lengths) > 3:
        shorter = min((value for value in lengths if value < query_length),
                      key=lambda value: abs(value - query_length * 0.8), default=None)
        longer = min((value for value in lengths if value > query_length),
                     key=lambda value: abs(value - query_length * 1.2), default=None)
        lengths = sorted({query_length, shorter or query_length, longer or query_length})
    prepared_queries = {length: _prepare_elastic_query(q, query_states, q_events,
                                                        length, config.use_volume)
                        for length in lengths}
    candidates: list[dict] = []
    stats: dict[str, int | float | str] = {
        "candidate_scope": "latest" if latest_date is not None else "history",
        "match_mode": "elastic", "query_length": query_length,
        "length_candidates": len(lengths), "length_grid": ",".join(map(str, lengths)),
        "symbols_total": 0, "symbols_eligible": 0, "windows_total": 0,
        "local_recall_total": 0, "mass_path_total": 0, "structure_path_total": 0,
        "state_path_total": 0, "event_path_total": 0, "candidates_before_global_limit": 0,
    }
    limit = int(recall_n or config.recall_n)
    event_active = bool(q_events.any())
    for symbol in engine.provider.get_symbols():
        stats["symbols_total"] += 1
        if latest_date is not None and exclude_query and str(symbol) == str(query_symbol):
            continue
        frame = _valid_bars(_get_bars(engine.provider, symbol, timeframe))
        if len(frame) < min(lengths):
            continue
        if latest_date is not None and (latest_date - frame.date.iloc[-1]).days > _latest_lag_days(timeframe):
            continue
        stats["symbols_eligible"] += 1
        if latest_date is not None:
            for length in lengths:
                start_idx = len(frame) - length
                if start_idx < 0:
                    continue
                window = frame.iloc[start_idx:].reset_index(drop=True)
                previous_close = float(frame.close.iloc[start_idx - 1]) if start_idx else None
                events = _event_features(window, str(symbol), previous_close=previous_close)
                coverage, missing = _special_event_match(q_events, events)
                states, _ = _state_features(window)
                candidates.append({
                    "symbol": str(symbol), "start_idx": start_idx, "end_idx": len(frame) - 1,
                    "length": length, "mass_distance": float(_coarse_shape_distance(q, window, length)[0]),
                    "event_distance": float(_coarse_event_distance(q_events, events, length)[0]),
                    "special_event_match": coverage, "missing_special_events": missing,
                    "previous_close": previous_close, "recall_paths": "mass+structure+event",
                })
                stats["windows_total"] += 1
                stats["local_recall_total"] += 1
            continue
        frame_states, _ = _state_features(frame)
        frame_events = _event_features(frame, str(symbol))
        frame_arrays = _prepare_elastic_frame(frame)
        compiled_prices = np.ascontiguousarray(np.column_stack(frame_arrays)) if _compiled_coarse_distances is not None else None
        frame_dates = frame.date.to_numpy()
        query_dates = frame_dates.astype("datetime64[D]") if exclude_query and str(symbol) == str(query_symbol) else None
        for length in lengths:
            scan_step = (min(max(1, config.elastic_coarse_step), max(1, length // 4))
                         if _compiled_coarse_distances is not None else 1)
            shape, structure, state, event = _elastic_coarse_distances(
                q, frame, query_states, frame_states, q_events, frame_events,
                length, config.use_volume, prepared_queries[length], frame_arrays,
                compiled_prices, scan_step,
            )
            count = len(shape)
            stats["windows_total"] += count
            legal = np.ones(count, dtype=bool)
            starts = np.arange(count) * scan_step
            ends = starts + length - 1
            special_coverage, special_missing = _special_event_coverage_windows(
                q_events, frame_events, starts, length,
            )
            if history_only:
                legal &= frame_dates[ends] < start.to_datetime64()
            if exclude_query and str(symbol) == str(query_symbol):
                left = np.maximum(query_dates[starts], np.datetime64(start, "D"))
                right = np.minimum(query_dates[ends], np.datetime64(end, "D"))
                overlap_days = np.maximum((right - left).astype("timedelta64[D]").astype(int), 0)
                query_days = max((end - start).days + 1, 1)
                legal &= (right < left) | (overlap_days / query_days <= 0.2)
            for values in (shape, structure, state, event):
                values[~legal] = np.inf
            combined = (shape + 0.35 * structure + 0.25 * state
                        + config.event_recall_weight * event + 2.0 * (1.0 - special_coverage))
            radius = max(1, round(max(3, length * config.exclusion_ratio) / scan_step))
            selected_paths: dict[int, set[str]] = {}
            paths = (("mass", combined), ("structure", structure), ("state", state))
            if event_active:
                paths += (("event", event),)
            for name, values in paths:
                for idx, value in _local_minima(values, config.local_top_n, radius):
                    if np.isfinite(value):
                        selected_paths.setdefault(idx, set()).add(name)
            stats["local_recall_total"] += len(selected_paths)
            stats["mass_path_total"] += sum("mass" in path for path in selected_paths.values())
            stats["structure_path_total"] += sum("structure" in path for path in selected_paths.values())
            stats["state_path_total"] += sum("state" in path for path in selected_paths.values())
            stats["event_path_total"] += sum("event" in path for path in selected_paths.values())
            for idx, paths in selected_paths.items():
                candidates.append({
                    "symbol": str(symbol), "start_idx": int(starts[idx]),
                    "end_idx": int(ends[idx]),
                    "length": length, "mass_distance": float(shape[idx]),
                    "event_distance": float(event[idx]),
                    "special_event_match": float(special_coverage[idx]),
                    "missing_special_events": int(special_missing[idx]),
                    "previous_close": float(frame.close.iloc[starts[idx] - 1]) if starts[idx] else None,
                    "recall_paths": "+".join(sorted(paths)),
                })
    stats["candidates_before_global_limit"] = len(candidates)
    candidates = _bounded_candidates(candidates, limit, config.event_recall_weight,
                                     special_required=bool(q_events[:, SPECIAL_EVENT_CHANNELS].any()))
    stats["candidates_after_global_limit"] = len(candidates)
    if not candidates:
        stats["final_results"] = 0
        engine.last_search_stats = stats
        return []
    return _elastic_rerank(
        engine, q, q_close, q_features, q_events, query_states, query_salience,
        candidates, timeframe, top_k, stats,
    )


def _elastic_rerank(engine: KlineSimilarityEngine, q: pd.DataFrame,
                    q_close: np.ndarray, q_features: np.ndarray,
                    q_events: np.ndarray, query_states: np.ndarray,
                    query_salience: np.ndarray, candidates: list[dict],
                    timeframe: str, top_k: int | None,
                    stats: dict[str, int | float | str]) -> list[SimilarityMatch]:
    config = engine.config
    query_length = len(q)
    feature_scale = np.maximum(np.nanstd(q_features, axis=0), 1e-3)
    weights = np.sqrt(np.asarray(config.feature_weights, dtype=float))
    if not config.use_volume:
        weights[-1] = 0.0
    weighted_q = q_features / feature_scale * weights
    metrics: list[SimilarityMatch] = []
    qret = float(np.log(q.close.iloc[-1] / q.close.iloc[0]))
    qvol = float(np.std(np.diff(q_close)))
    bars_by_symbol = {symbol: _valid_bars(_get_bars(engine.provider, symbol, timeframe))
                      for symbol in {item["symbol"] for item in candidates}}
    for item in candidates:
        bars = bars_by_symbol[item["symbol"]]
        cframe = bars.iloc[item["start_idx"]:item["end_idx"] + 1].reset_index(drop=True)
        c_features = _features(cframe, config.use_volume)
        weighted_c = c_features / feature_scale * weights
        candidate_events = _event_features(cframe, item["symbol"], previous_close=item.get("previous_close"))
        special_match, missing_special = _special_event_match(q_events, candidate_events)
        event_scale = np.sqrt(max(config.special_event_weight, 0.0))
        aligned_q = np.hstack([weighted_q, q_events * event_scale])
        aligned_c = np.hstack([weighted_c, candidate_events * event_scale])
        candidate_states, candidate_salience = _state_features(cframe)
        elastic_shape, warp_length, skipped_q, skipped_c = _elastic_dtw(
            aligned_q, aligned_c, query_salience, candidate_salience,
            config.normalized_warp_band, config.max_repeat_step,
            config.gap_penalty, config.salient_gap_multiplier,
        )
        state_distance = _state_edit_distance(query_states, candidate_states, query_salience, candidate_salience)
        cret = float(np.log(cframe.close.iloc[-1] / cframe.close.iloc[0]))
        cvol = float(np.std(np.diff(np.log(cframe.close.to_numpy(float)))))
        event_distance = float(np.sqrt(np.mean((q_events.mean(axis=0) - candidate_events.mean(axis=0)) ** 2)))
        aligned = _resampled_distance(aligned_q, aligned_c)
        metrics.append(SimilarityMatch(
            symbol=item["symbol"], start_date=cframe.date.iloc[0], end_date=cframe.date.iloc[-1],
            score=0.0, mass_distance=float(item["mass_distance"]), dtw_distance=elastic_shape,
            aligned_distance=aligned, total_return_diff=abs(qret - cret),
            volatility_diff=float(abs(np.log((cvol + 1e-9) / (qvol + 1e-9)))), event_distance=event_distance,
            special_event_match=special_match, missing_special_events=missing_special,
            special_event_required=bool(q_events[:, SPECIAL_EVENT_CHANNELS].any()),
            window_length=len(cframe), start_idx=item["start_idx"], end_idx=item["end_idx"],
            query_length=query_length, length_ratio=len(cframe) / query_length, match_mode="elastic",
            elastic_shape_distance=elastic_shape, state_distance=state_distance,
            anchor_distance=_anchor_distance(q, cframe), length_penalty=abs(np.log(len(cframe) / query_length)),
            warp_path_length=warp_length, skipped_query_bars=skipped_q, skipped_candidate_bars=skipped_c,
        ))
    for item in metrics:
        path_summary = 0.5 * min(item.total_return_diff, 2.0) + 0.5 * min(item.volatility_diff, 2.0)
        total = (0.30 * item.elastic_shape_distance + 0.25 * item.state_distance +
                 0.20 * item.anchor_distance + 0.10 * path_summary +
                 0.08 * item.event_distance + 0.05 * item.length_penalty)
        item.score = float(1.0 / (1.0 + max(total, 0.0)))
    _prioritize_special_events(metrics, q_events)
    metrics.sort(key=lambda x: (-x.score, x.symbol, x.start_date))
    result = _global_nms(metrics, top_k or config.top_k, overlap_threshold=config.exclusion_ratio)
    stats["stage2_candidates"] = len(metrics)
    stats["candidate_length_min"] = min(x.window_length for x in metrics)
    stats["candidate_length_max"] = max(x.window_length for x in metrics)
    stats["skipped_query_bars_total"] = int(sum(x.skipped_query_bars for x in metrics))
    stats["skipped_candidate_bars_total"] = int(sum(x.skipped_candidate_bars for x in metrics))
    stats["final_results"] = len(result)
    engine.last_search_stats = stats
    return result


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
    """Return limit events, including high-open, T-board and one-word boards.

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
    # QFQ prices are fractional after adjustment, so the return test also
    # recognizes boards whose exact cent price is no longer recoverable.
    up_return = close / np.maximum(prev, 1e-12)
    up = valid_prev & ((np.abs(close - up_price) <= 0.011) |
                       (np.abs(up_return - (1.0 + limit)) <= 0.003))
    down = valid_prev & ((np.abs(close - down_price) <= 0.011) |
                         (np.abs(up_return - (1.0 - limit)) <= 0.003))
    equal = (np.round(open_, 2) == np.round(high, 2)) & (np.round(high, 2) == np.round(low, 2)) & (np.round(low, 2) == np.round(close, 2))
    high_open = up & (open_ / np.maximum(prev, 1e-12) >= 1.02)
    at_limit_open = high_open & (np.abs(open_ - close) / np.maximum(close, 1e-12) <= 0.006)
    t_board = at_limit_open & ~equal & (np.abs(high - close) / np.maximum(close, 1e-12) <= 0.006) & (low < close * 0.994)
    return np.column_stack([up.astype(float), down.astype(float),
                            (up & equal).astype(float), (down & equal).astype(float),
                            high_open.astype(float), t_board.astype(float)])


def _special_event_match(query_events: np.ndarray,
                         candidate_events: np.ndarray) -> tuple[float, int]:
    """Measure whether the candidate contains the query's rare board types.

    A T-board also has the high-open channel; a one-word board also has it.
    Missing either subtype is therefore penalized more than a generic limit-up.
    The count comparison allows elastic windows to differ in length or position.
    """
    channels = SPECIAL_EVENT_CHANNELS
    weights = SPECIAL_EVENT_WEIGHTS
    required = np.rint(query_events[:, channels].sum(axis=0)).astype(int)
    if not required.any():
        return 1.0, 0
    observed = np.rint(candidate_events[:, channels].sum(axis=0)).astype(int)
    missing = np.maximum(required - observed, 0)
    need_weight = float(np.dot(required, weights))
    coverage = 1.0 - float(np.dot(missing, weights)) / need_weight
    return max(0.0, coverage), int(missing.sum())


def _special_event_coverage_windows(query_events: np.ndarray, series_events: np.ndarray,
                                    starts: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    channels = SPECIAL_EVENT_CHANNELS
    weights = SPECIAL_EVENT_WEIGHTS
    required = np.rint(query_events[:, channels].sum(axis=0)).astype(int)
    if not required.any():
        return np.ones(len(starts)), np.zeros(len(starts), dtype=int)
    flags = np.rint(series_events[:, channels]).astype(int)
    prefix = np.vstack((np.zeros((1, len(channels)), dtype=int), np.cumsum(flags, axis=0)))
    observed = prefix[starts + length] - prefix[starts]
    missing = np.maximum(required - observed, 0)
    coverage = 1.0 - (missing @ weights) / float(required @ weights)
    return np.maximum(coverage, 0.0), missing.sum(axis=1)


def _prioritize_special_events(items: list[SimilarityMatch], query_events: np.ndarray) -> None:
    if not query_events[:, SPECIAL_EVENT_CHANNELS].any():
        return
    for item in items:
        # Exact subtype coverage receives the upper score band; partial
        # coverage remains rankable if no exact historical analogue exists.
        item.score = (0.5 + 0.5 * item.score if item.missing_special_events == 0
                      else 0.5 * item.score * item.special_event_match)


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


def _bounded_candidates(candidates: list[dict], limit: int, event_weight: float,
                        special_required: bool = False) -> list[dict]:
    """Keep only the global best recall candidates without retaining all ties."""
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda x: _candidate_recall_key(x, event_weight))
    special = []
    if special_required:
        complete = [x for x in candidates if x.get("missing_special_events", 1) == 0]
        special = _heap_best_candidates(complete, min(max(1, limit // 2), len(complete)), event_weight)
        chosen_ids = {id(x) for x in special}
        candidates = [x for x in candidates if id(x) not in chosen_ids]
        limit -= len(special)
        if limit <= 0:
            return special
    # Reserve part of the finite DTW budget for structure-path recalls so a
    # close-shape-heavy market cannot crowd them all out.
    structure_candidates = [x for x in candidates if "structure" in str(x.get("recall_paths", ""))]
    reserve_n = min(max(1, limit // 3), len(structure_candidates))
    reserved = _heap_best_candidates(structure_candidates, reserve_n, event_weight)
    reserved_ids = {id(x) for x in reserved}
    remainder = [x for x in candidates if id(x) not in reserved_ids]
    return special + reserved + _heap_best_candidates(remainder, max(0, limit - len(reserved)), event_weight)


def _heap_best_candidates(candidates: list[dict], limit: int, event_weight: float) -> list[dict]:
    if limit <= 0 or not candidates:
        return []
    heap: list[tuple[float, int, dict]] = []
    for serial, item in enumerate(candidates):
        key = _candidate_recall_key(item, event_weight)
        entry = (-float(key), serial, item)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [x[2] for x in sorted(heap, key=lambda x: -x[0])]


def _candidate_recall_key(item: dict, event_weight: float) -> float:
    return (item["mass_distance"] + event_weight * item["event_distance"]
            + 2.0 * (1.0 - item.get("special_event_match", 1.0)))


def _local_minima(distances: np.ndarray, count: int, radius: int) -> list[tuple[int, float]]:
    work = distances.copy()
    result = []
    for _ in range(max(0, count)):
        if not len(work):
            break
        idx = int(np.argmin(work))
        if not np.isfinite(work[idx]):
            break
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
