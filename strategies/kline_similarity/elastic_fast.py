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
        state_prefix = np.zeros((len(states) + 1, states.shape[1], 3), dtype=np.int64)
        event_prefix = np.zeros((len(events) + 1, events.shape[1]), dtype=np.int64)
        for idx in range(len(states)):
            for channel in range(states.shape[1]):
                for value in range(3):
                    state_prefix[idx + 1, channel, value] = state_prefix[idx, channel, value]
                state_prefix[idx + 1, channel, states[idx, channel]] += 1
            for channel in range(events.shape[1]):
                event_prefix[idx + 1, channel] = event_prefix[idx, channel] + (events[idx, channel] > 0.0)
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
                    count0 = state_prefix[last, channel, 0] - state_prefix[first, channel, 0]
                    count1 = state_prefix[last, channel, 1] - state_prefix[first, channel, 1]
                    count2 = state_prefix[last, channel, 2] - state_prefix[first, channel, 2]
                    mode = 0
                    largest = count0
                    if count1 > largest:
                        mode = 1
                        largest = count1
                    if count2 > largest:
                        mode = 2
                    if mode != q_states[bucket, channel]: state_mismatch += 1.0
                for channel in range(events.shape[1]):
                    maximum = 1.0 if event_prefix[last, channel] > event_prefix[first, channel] else 0.0
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


if njit is not None:
    @njit(cache=True)
    def fixed_dtw(a, b, radius):
        n, m = len(a), len(b)
        prev = np.full(m + 1, np.inf)
        curr = np.full(m + 1, np.inf)
        prev[0] = 0.0
        for i in range(1, n + 1):
            curr[:] = np.inf
            lo = max(1, i - radius)
            hi = min(m, i + radius)
            for j in range(lo, hi + 1):
                distance = 0.0
                for k in range(a.shape[1]):
                    diff = a[i - 1, k] - b[j - 1, k]
                    distance += diff * diff
                curr[j] = distance + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        return np.sqrt(prev[m] / max(n + m, 1))

    @njit(cache=True)
    def state_edit_distance(a, b, a_salience, b_salience):
        n, m = len(a), len(b)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        weights = np.array((1.5, 0.8, 0.8, 0.6, 0.5))
        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] + 0.5 * (1.0 + 2.5 * a_salience[i - 1])
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + 0.5 * (1.0 + 2.5 * b_salience[j - 1])
        for i in range(1, n + 1):
            delete = 0.5 * (1.0 + 2.5 * a_salience[i - 1])
            for j in range(1, m + 1):
                substitute = 0.0
                for k in range(a.shape[1]):
                    if a[i - 1, k] != b[j - 1, k]:
                        substitute += weights[k]
                insert = 0.5 * (1.0 + 2.5 * b_salience[j - 1])
                dp[i, j] = min(dp[i - 1, j] + delete,
                               dp[i, j - 1] + insert,
                               dp[i - 1, j - 1] + substitute)
        return dp[n, m] / max(n + m, 1)


    @njit(cache=True)
    def elastic_dtw(a, b, a_salience, b_salience, normalized_band,
                    max_repeat_step, gap_penalty, salient_gap_multiplier):
        n, m = len(a), len(b)
        runs = max(1, int(max_repeat_step))
        states = 1 + 2 * runs
        dp = np.full((n + 1, m + 1, states), np.inf)
        path_len = np.zeros((n + 1, m + 1, states), dtype=np.int16)
        skipped_q = np.zeros((n + 1, m + 1, states), dtype=np.int16)
        skipped_c = np.zeros((n + 1, m + 1, states), dtype=np.int16)
        dp[0, 0, 0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                if abs(i / max(n, 1) - j / max(m, 1)) > normalized_band + 1e-12:
                    continue
                if i and j:
                    prev = int(np.argmin(dp[i - 1, j - 1]))
                    if np.isfinite(dp[i - 1, j - 1, prev]):
                        distance = 0.0
                        for k in range(a.shape[1]):
                            diff = a[i - 1, k] - b[j - 1, k]
                            distance += diff * diff
                        dp[i, j, 0] = dp[i - 1, j - 1, prev] + np.sqrt(distance / a.shape[1])
                        path_len[i, j, 0] = path_len[i - 1, j - 1, prev] + 1
                        skipped_q[i, j, 0] = skipped_q[i - 1, j - 1, prev]
                        skipped_c[i, j, 0] = skipped_c[i - 1, j - 1, prev]
                if j:
                    for prev in range(1 + 2 * runs):
                        if prev > runs:
                            continue
                        if np.isfinite(dp[i, j - 1, prev]):
                            value = dp[i, j - 1, prev] + gap_penalty * (1 + salient_gap_multiplier * b_salience[j - 1])
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
                            value = dp[i - 1, j, prev] + gap_penalty * (1 + salient_gap_multiplier * a_salience[i - 1])
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
            return np.inf, 0, 0, 0
        return (np.sqrt(dp[n, m, state] / max(path_len[n, m, state], 1)),
                int(path_len[n, m, state]), int(skipped_q[n, m, state]),
                int(skipped_c[n, m, state]))
else:
    fixed_dtw = None
    state_edit_distance = None
    elastic_dtw = None
