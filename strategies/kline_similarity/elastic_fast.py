"""Compiled coarse recall for elastic K-line search.

Numba is optional; the engine retains its NumPy implementation as a fallback.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised by the engine fallback
    njit = None


if njit is not None:
    @njit(cache=True)
    def coarse_distances(prices, states, events, group_starts, group_ends,
                         q_norm, q_features, scale, q_states, q_events,
                         use_volume, start_step=1):
        count = (len(prices) - int(group_ends[-1]) + start_step) // start_step
        buckets = len(group_starts)
        shape = np.empty(count)
        structure = np.empty(count)
        state = np.empty(count)
        event = np.empty(count)
        eps = 1e-12
        for start in range(count):
            o = np.empty(buckets); h = np.empty(buckets); l = np.empty(buckets)
            c = np.empty(buckets); v = np.empty(buckets)
            state_mismatch = 0.0
            event_error = 0.0
            for bucket in range(buckets):
                first = start * start_step + group_starts[bucket]
                last = start * start_step + group_ends[bucket]
                o[bucket] = prices[first, 0]
                h[bucket] = prices[first, 1]
                l[bucket] = prices[first, 2]
                c[bucket] = prices[last - 1, 3]
                v[bucket] = 0.0
                for idx in range(first, last):
                    h[bucket] = max(h[bucket], prices[idx, 1])
                    l[bucket] = min(l[bucket], prices[idx, 2])
                    v[bucket] += prices[idx, 4]
                for channel in range(states.shape[1]):
                    counts = np.zeros(3, dtype=np.int64)
                    for idx in range(first, last):
                        counts[states[idx, channel]] += 1
                    mode = 0
                    if counts[1] > counts[mode]: mode = 1
                    if counts[2] > counts[mode]: mode = 2
                    if mode != q_states[bucket, channel]: state_mismatch += 1.0
                for channel in range(events.shape[1]):
                    maximum = 0.0
                    for idx in range(first, last):
                        maximum = max(maximum, events[idx, channel])
                    difference = maximum - q_events[bucket, channel]
                    event_error += difference * difference
            log_close = np.log(c)
            mean = np.mean(log_close)
            std = np.std(log_close)
            if std < eps: std = eps
            shape_error = 0.0
            structure_error = 0.0
            for bucket in range(buckets):
                difference = (log_close[bucket] - mean) / std - q_norm[bucket]
                shape_error += difference * difference
                prev = c[bucket - 1] if bucket else c[0]
                prev_vol = v[bucket - 1] if bucket else v[0]
                feat = np.empty(6)
                feat[0] = np.log(max(o[bucket], eps) / max(prev, eps)) if bucket else 0.0
                feat[1] = np.log(max(c[bucket], eps) / max(o[bucket], eps))
                feat[2] = max((h[bucket] - max(o[bucket], c[bucket])) / max(prev, eps), 0.0)
                feat[3] = max((min(o[bucket], c[bucket]) - l[bucket]) / max(prev, eps), 0.0)
                feat[4] = (h[bucket] - l[bucket]) / max(prev, eps)
                feat[5] = np.log((v[bucket] + eps) / (prev_vol + eps)) if bucket and use_volume else 0.0
                for channel in range(6):
                    difference = (feat[channel] - q_features[bucket, channel]) / scale[channel]
                    structure_error += difference * difference
            shape[start] = np.sqrt(shape_error / buckets)
            structure[start] = np.sqrt(structure_error / (buckets * 6))
            state[start] = state_mismatch / (buckets * states.shape[1])
            event[start] = np.sqrt(event_error / (buckets * events.shape[1]))
        return shape, structure, state, event
else:
    coarse_distances = None
