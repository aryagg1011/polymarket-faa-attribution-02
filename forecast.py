"""Dependency-free quantile forecasts for make.fun program/v1 cases."""
from __future__ import annotations

import math

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
Z = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
BASE_SIGMA = 0.0025


def clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0.0:
        return 0.0
    target = q * total
    cum = 0.0
    prev_value = pairs[0][0]
    prev_cum = 0.0
    for value, weight in pairs:
        next_cum = cum + weight
        if target <= next_cum:
            if next_cum <= prev_cum:
                return value
            frac = min(1.0, max(0.0, (target - prev_cum) / (next_cum - prev_cum)))
            return prev_value + frac * (value - prev_value)
        prev_value, prev_cum, cum = value, next_cum, next_cum
    return pairs[-1][0]


def _rms_step(prices: list[float], window: int) -> float:
    n = len(prices)
    lo = max(1, n - window)
    count = n - lo
    if count <= 0:
        return BASE_SIGMA
    return math.sqrt(sum((prices[i] - prices[i - 1]) ** 2 for i in range(lo, n)) / count)


def _empirical_deltas(
    histories: list[list[float]],
    current_price: float,
    horizon: int,
    *,
    step: int,
    half_life: float,
    price_band: float,
    max_per_history: int = 0,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    weights: list[float] = []
    for hist in histories:
        n = len(hist)
        last_origin = n - 1 - horizon
        if last_origin < 0:
            continue
        start = 0
        if max_per_history:
            start = max(0, last_origin - max_per_history * step)
        for j in range(start, last_origin + 1, step):
            age = last_origin - j
            w = 1.0
            if half_life > 0.0:
                w *= 0.5 ** (age / half_life)
            if price_band > 0.0:
                distance = abs(hist[j] - current_price) / price_band
                w *= math.exp(-0.5 * distance * distance)
            if w >= 1e-8:
                values.append(hist[j + horizon] - hist[j])
                weights.append(w)
    return values, weights


def forecast_market(
    prices: list[float],
    horizon: int,
    *,
    model: str = "baseline",
    peer_histories: list[list[float]] | None = None,
    sigma_scale: float = 1.0,
    drift_window: int = 0,
    drift_scale: float = 0.0,
    step: int = 30,
    half_life: float = 0.0,
    price_band: float = 0.0,
    empirical_blend: float = 1.0,
    peer_blend: float = 0.0,
    max_origins: int = 48,
    calc_stride: int = 1,
) -> list[list[float]]:
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    sigma = BASE_SIGMA
    if model == "adaptive":
        short = _rms_step(prices, 120)
        medium = _rms_step(prices, 720)
        long = _rms_step(prices, 2880)
        # A mild recent emphasis, shrunk toward the contest's fixed prior.
        sigma = math.sqrt(0.50 * short * short + 0.30 * medium * medium + 0.20 * long * long)
        sigma = math.sqrt(0.65 * sigma * sigma + 0.35 * BASE_SIGMA * BASE_SIGMA)
    sigma *= sigma_scale
    drift = 0.0
    if drift_window > 0 and len(prices) > drift_window:
        drift = drift_scale * (prices[-1] - prices[-1 - drift_window]) / drift_window

    peers = peer_histories or [prices]
    rows: list[list[float]] = []
    for h in range(calc_stride, horizon + 1, calc_stride):
        gaussian = [z * sigma * math.sqrt(h) + drift * h for z in Z]
        if model in ("empirical", "hybrid"):
            local_values, local_weights = _empirical_deltas(
                [prices], x, h, step=step, half_life=half_life,
                price_band=price_band, max_per_history=max_origins,
            )
            local = [_weighted_quantile(local_values, local_weights, q) for q in QUANTILES]
            if peer_blend > 0.0:
                peer_values, peer_weights = _empirical_deltas(
                    peers, x, h, step=step, half_life=half_life,
                    price_band=price_band, max_per_history=max_origins,
                )
                pooled = [_weighted_quantile(peer_values, peer_weights, q) for q in QUANTILES]
                empirical = [
                    (1.0 - peer_blend) * local[i] + peer_blend * pooled[i]
                    for i in range(5)
                ]
            else:
                empirical = local
            deltas = [
                empirical_blend * empirical[i] + (1.0 - empirical_blend) * gaussian[i]
                for i in range(5)
            ]
        else:
            deltas = gaussian
        row = [round(clamp(x + d), 6) for d in deltas]
        # Clipping at 0/1 can tie adjacent quantiles; sorting also guards roundoff.
        rows.append(sorted(row))
    return rows


def forecast_case(case: dict, *, model: str = "baseline", **params) -> dict:
    histories = [m["prices"] for m in case["markets"]]
    forecasts = []
    for market in case["markets"]:
        q = forecast_market(
            market["prices"], case["horizon_minutes"], model=model,
            peer_histories=histories, **params,
        )
        forecasts.append({"id": market["id"], "q": q})
    return {"forecasts": forecasts}

