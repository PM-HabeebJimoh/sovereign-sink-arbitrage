"""Leakage-safe technical, regime and optional cross-market features."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .data import prepare_bars


_PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    change = series.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).clip(0.0, 100.0)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return name or "external"


def _higher_timeframe_features(bars: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    """Create features from completed higher-timeframe bars only.

    The source timestamps are treated as close times.  A higher-timeframe row
    is forward-filled to 30m rows only after that higher-timeframe bar closes;
    no future bar is merged backwards.
    """

    aggregate = bars[["open", "high", "low", "close"]].resample(
        rule, label="right", closed="right"
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if aggregate.empty:
        return pd.DataFrame(index=bars.index)

    close = aggregate["close"]
    log_close = np.log(close)
    result = pd.DataFrame(index=aggregate.index)
    result[f"{prefix}_ret_1"] = log_close.diff()
    result[f"{prefix}_ret_3"] = log_close.diff(3)
    result[f"{prefix}_ema_gap_8_21"] = _ema(close, 8) / _ema(close, 21) - 1.0
    tr = pd.concat(
        [
            aggregate["high"] - aggregate["low"],
            (aggregate["high"] - close.shift(1)).abs(),
            (aggregate["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result[f"{prefix}_atr_pct"] = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean() / close
    result[f"{prefix}_rsi_14"] = _rsi(close, 14) / 100.0
    result[f"{prefix}_range_pct"] = (aggregate["high"] - aggregate["low"]) / close
    # ``reindex(method='ffill')`` never uses a higher timeframe row dated in
    # the future relative to the 30-minute observation.
    return result.reindex(bars.index, method="ffill")


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build a numeric feature matrix using only current and past bars.

    Feature groups:
    - price action and candle geometry;
    - multi-horizon momentum, volatility and trend;
    - 2h/4h/daily context;
    - UTC/session and day-of-week cyclic encodings;
    - optional, explicitly named exogenous columns (DXY, rates, VIX, etc.).

    No feature is shifted from the future.  The first rows are naturally NaN
    until rolling windows warm up; the modelling layer drops/imputes them
    without forward filling prices.
    """

    bars = prepare_bars(bars, resample=False)
    close = bars["close"].astype(float)
    open_ = bars["open"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    log_close = np.log(close)
    returns = log_close.diff()

    out = pd.DataFrame(index=bars.index)

    # Price action.  These are dimensionless so a model can be trained across
    # gold regimes and broker price formatting.
    out["ret_1"] = returns
    for window in (2, 3, 6, 12, 24, 48, 96):
        out[f"ret_{window}"] = log_close.diff(window)
        out[f"ret_mean_{window}"] = returns.rolling(window, min_periods=window).mean()
        out[f"ret_std_{window}"] = returns.rolling(window, min_periods=window).std()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    out["atr_14_pct"] = atr14 / close
    out["atr_48_pct"] = true_range.rolling(48, min_periods=48).mean() / close
    out["range_pct"] = (high - low) / close
    out["gap_pct"] = (open_ / prev_close) - 1.0
    out["body_pct"] = (close - open_) / open_
    out["body_to_range"] = (close - open_).abs() / (high - low).replace(0.0, np.nan)
    out["upper_wick_to_range"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low).replace(0.0, np.nan)
    out["lower_wick_to_range"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low).replace(0.0, np.nan)
    out["close_location"] = (close - low) / (high - low).replace(0.0, np.nan)
    out["atr_return_1"] = returns / (atr14 / close).replace(0.0, np.nan)

    # Trend and mean reversion.
    for span in (8, 21, 55, 100):
        ema = _ema(close, span)
        out[f"ema_gap_{span}"] = close / ema - 1.0
    out["ema_slope_8"] = _ema(close, 8).pct_change(3)
    out["ema_slope_21"] = _ema(close, 21).pct_change(6)
    out["rsi_7"] = _rsi(close, 7) / 100.0
    out["rsi_14"] = _rsi(close, 14) / 100.0
    out["rsi_28"] = _rsi(close, 28) / 100.0
    for window in (12, 24, 48):
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        out[f"breakout_high_{window}"] = (close - rolling_high.shift(1)) / atr14
        out[f"breakout_low_{window}"] = (close - rolling_low.shift(1)) / atr14
        out[f"range_position_{window}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)

    # Volatility regime and directional persistence.
    out["vol_ratio_12_48"] = out["ret_std_12"] / out["ret_std_48"].replace(0.0, np.nan)
    out["vol_ratio_48_96"] = out["ret_std_48"] / out["ret_std_96"].replace(0.0, np.nan)
    signed = np.sign(returns)
    out["directional_persistence_6"] = signed.rolling(6, min_periods=6).mean()
    out["directional_persistence_24"] = signed.rolling(24, min_periods=24).mean()

    # Volume is optional because spot FX feeds often only offer tick volume.
    # If present, use relative volume, not the raw vendor-specific number.
    if "volume" in bars.columns and bars["volume"].notna().any():
        volume = bars["volume"].astype(float)
        out["volume_ratio_24"] = volume / volume.rolling(24, min_periods=24).median()
        volume_log = np.log1p(volume)
        out["volume_z_48"] = (volume_log - volume_log.rolling(48, min_periods=48).mean()) / volume_log.rolling(48, min_periods=48).std()

    # Session context.  UTC is deterministic; users can add broker-local
    # session variables upstream if their feed uses a different day boundary.
    minute_of_day = bars.index.hour * 60 + bars.index.minute
    day_fraction = minute_of_day / (24.0 * 60.0)
    week_fraction = (bars.index.dayofweek * 24 * 60 + minute_of_day) / (7.0 * 24.0 * 60.0)
    out["day_sin"] = np.sin(2.0 * np.pi * day_fraction)
    out["day_cos"] = np.cos(2.0 * np.pi * day_fraction)
    out["week_sin"] = np.sin(2.0 * np.pi * week_fraction)
    out["week_cos"] = np.cos(2.0 * np.pi * week_fraction)
    out["london_session"] = ((minute_of_day >= 7 * 60) & (minute_of_day < 16 * 60)).astype(float)
    out["ny_session"] = ((minute_of_day >= 13 * 60) & (minute_of_day < 22 * 60)).astype(float)
    out["london_ny_overlap"] = ((minute_of_day >= 13 * 60) & (minute_of_day < 16 * 60)).astype(float)
    out["weekday"] = (bars.index.dayofweek < 5).astype(float)

    # Higher-timeframe context is a feature, not a separate target.  Because
    # all windows run on completed aggregate bars, this avoids the common
    # mistake of leaking the final daily close into an earlier intraday row.
    for rule, prefix in (("2h", "htf_2h"), ("4h", "htf_4h"), ("1D", "htf_1d")):
        out = out.join(_higher_timeframe_features(bars, rule, prefix), how="left")

    # Optional aligned market drivers.  Only columns explicitly allowed by
    # data.py reach this point.  Use changes and rolling z-scores so the model
    # learns cross-market pressure without relying on absolute price units.
    for column in bars.columns:
        if column in _PRICE_COLUMNS:
            continue
        safe = _safe_name(column)
        series = pd.to_numeric(bars[column], errors="coerce")
        if not series.notna().any():
            continue
        level = series.replace([np.inf, -np.inf], np.nan)
        out[f"external_{safe}_ret_1"] = np.log(level.where(level > 0)).diff()
        out[f"external_{safe}_change_1"] = level.diff()
        mean = level.rolling(48, min_periods=48).mean()
        std = level.rolling(48, min_periods=48).std()
        out[f"external_{safe}_z_48"] = (level - mean) / std.replace(0.0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "timestamp"
    return out


def build_labeled_frame(
    bars: pd.DataFrame,
    *,
    horizon_bars: int = 1,
    label_threshold_bps: float = 5.0,
    label_atr_fraction: float = 0.10,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return features plus an execution-aligned direction label.

    A prediction is made after bar ``t`` closes and entered at bar ``t+1``'s
    open.  For ``horizon_bars=1`` the outcome is the next bar's open-to-close
    log return.  A neutral/dead-zone outcome is excluded from training; this
    is explicit abstention, not a silently relabelled win.
    """

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if label_threshold_bps < 0 or label_atr_fraction < 0:
        raise ValueError("label thresholds cannot be negative")

    bars = prepare_bars(bars, resample=False)
    features = build_features(bars)
    entry = bars["open"].shift(-1)
    exit_ = bars["close"].shift(-horizon_bars)
    future_return = np.log(exit_ / entry)

    atr_bps = features["atr_14_pct"].abs() * 10000.0
    dead_zone_bps = np.maximum(float(label_threshold_bps), float(label_atr_fraction) * atr_bps)
    future_bps = future_return * 10000.0

    label = pd.Series(np.nan, index=bars.index, dtype=float)
    label.loc[future_bps > dead_zone_bps] = 1.0
    label.loc[future_bps < -dead_zone_bps] = 0.0

    labelled = features.copy()
    labelled["label"] = label
    labelled["future_return"] = future_return
    labelled["future_return_bps"] = future_bps
    labelled["dead_zone_bps"] = dead_zone_bps
    feature_columns = list(features.columns)
    # Keep rows with a known direction and let the estimator impute only
    # warm-up feature gaps.  Rows at the end with unknown future returns are
    # necessarily removed here.
    labelled = labelled.loc[labelled["label"].notna() & labelled["future_return"].notna()].copy()
    return labelled, feature_columns
