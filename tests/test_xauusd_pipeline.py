from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd.data import prepare_bars
from xauusd.evaluation import evaluate_predictions, select_threshold, wilson_lower_bound
from xauusd.features import build_features, build_labeled_frame


def make_bars(rows: int = 1_000) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="30min", tz="UTC")
    # Deterministic, non-flat gold-like series with a modest trend/regime mix.
    t = np.arange(rows)
    close = 2_000 + 0.03 * t + 4 * np.sin(t / 17) + 1.5 * np.sin(t / 61)
    open_ = close - 0.8 * np.sin(t / 5)
    high = np.maximum(open_, close) + 1.2
    low = np.minimum(open_, close) - 1.2
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100 + (t % 20),
        }
    )


def test_prepare_bars_is_utc_and_accepts_vendor_case() -> None:
    frame = make_bars(100).rename(
        columns={"timestamp": "Gmt time", "open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    bars, report = prepare_bars(frame, return_report=True)
    assert str(bars.index.tz) == "UTC"
    assert len(bars) == 100
    assert report.median_interval_minutes == 30.0


def test_features_do_not_use_a_future_close() -> None:
    frame = make_bars()
    first = prepare_bars(frame)
    original = build_features(first)
    changed = first.copy()
    changed.loc[changed.index >= changed.index[700], "close"] += 500
    changed["high"] = np.maximum(changed["high"], changed["close"])
    changed_built = build_features(changed)
    # A row well before the changed region must be bit-for-bit causal.
    pd.testing.assert_frame_equal(original.iloc[:600], changed_built.iloc[:600])


def test_one_hour_normalisation_is_supported() -> None:
    bars, report = prepare_bars(make_bars(120), timeframe_minutes=60, return_report=True)
    assert report.timeframe_minutes == 60
    assert report.median_interval_minutes == 60.0
    assert len(bars) < 120


def test_label_uses_next_open_to_next_close() -> None:
    bars = make_bars(300)
    labelled, _ = build_labeled_frame(
        bars,
        horizon_bars=1,
        label_threshold_bps=0.0,
        label_atr_fraction=0.0,
    )
    # The last eligible row is not the final bar because it needs t+1 open and
    # t+1 close.  Its return agrees with the explicit execution convention.
    row = labelled.iloc[-1]
    timestamp = labelled.index[-1]
    source = prepare_bars(bars)
    expected = np.log(source.loc[timestamp + pd.Timedelta(minutes=30), "close"] / source.loc[timestamp + pd.Timedelta(minutes=30), "open"])
    assert np.isclose(row["future_return"], expected)


def test_selective_metrics_make_coverage_visible() -> None:
    p_up = [0.95, 0.94, 0.51, 0.49, 0.06, 0.05]
    y = [1, 1, 1, 0, 0, 0]
    metrics = evaluate_predictions(p_up, y, threshold=0.90)
    assert metrics["signals"] == 4
    assert metrics["coverage"] == 4 / 6
    assert metrics["hit_rate"] == 1.0
    policy = select_threshold(p_up, y, target_hit_rate=0.80, min_signals=2)
    assert policy["target_met_on_validation"] is True
    assert 0.0 < wilson_lower_bound(4, 4) < 1.0
