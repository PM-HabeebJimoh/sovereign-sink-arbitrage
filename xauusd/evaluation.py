"""Out-of-sample metrics, threshold selection and purged time splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Indices for chronological train/validation/test partitions."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    embargo_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_rows": int(len(self.train)),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
            "embargo_bars": int(self.embargo_bars),
        }


def purged_time_split(
    index: Sequence[Any],
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    embargo_bars: int = 1,
) -> TimeSplit:
    """Create a chronological split with a gap on both label boundaries.

    The gap prevents the last training labels and first validation/test labels
    from sharing the same future bar.  No random shuffle is ever used.
    """

    n = len(index)
    if n < 60:
        raise ValueError("at least 60 labelled rows are needed for a time split")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must be positive and leave a test period")
    embargo = max(0, int(embargo_bars))
    train_cut = int(n * train_fraction)
    validation_cut = int(n * (train_fraction + validation_fraction))
    train_end = max(1, train_cut - embargo)
    validation_start = min(n, train_cut + embargo)
    validation_end = max(validation_start, validation_cut - embargo)
    test_start = min(n, validation_cut + embargo)

    train = np.arange(0, train_end, dtype=int)
    validation = np.arange(validation_start, validation_end, dtype=int)
    test = np.arange(test_start, n, dtype=int)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("time split produced an empty partition; supply more history")
    return TimeSplit(train=train, validation=validation, test=test, embargo_bars=embargo)


def _as_arrays(
    p_up: Iterable[float],
    y: Iterable[int],
    future_returns: Optional[Iterable[float]] = None,
    agreement: Optional[Iterable[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    probabilities = np.asarray(list(p_up), dtype=float)
    target = np.asarray(list(y), dtype=int)
    if len(probabilities) != len(target):
        raise ValueError("probabilities and labels have different lengths")
    returns = None if future_returns is None else np.asarray(list(future_returns), dtype=float)
    if returns is not None and len(returns) != len(target):
        raise ValueError("future returns and labels have different lengths")
    agreements = None if agreement is None else np.asarray(list(agreement), dtype=float)
    if agreements is not None and len(agreements) != len(target):
        raise ValueError("agreement and labels have different lengths")
    valid = np.isfinite(probabilities) & np.isfinite(target)
    if returns is not None:
        valid &= np.isfinite(returns)
    if agreements is not None:
        valid &= np.isfinite(agreements)
    probabilities = np.clip(probabilities[valid], 1e-6, 1.0 - 1e-6)
    target = target[valid]
    returns = returns[valid] if returns is not None else None
    agreements = agreements[valid] if agreements is not None else None
    if len(target) == 0:
        raise ValueError("no finite predictions remain")
    return probabilities, target, returns, agreements


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))


def wilson_lower_bound(wins: int, observations: int, z: float = 1.96) -> Optional[float]:
    """Two-sided 95% Wilson lower bound for a Bernoulli hit rate.

    The lower bound is deliberately reported separately from the observed
    hit rate.  It prevents four lucky signals from being presented as a
    statistically established 80% process.
    """

    if observations <= 0:
        return None
    n = float(observations)
    p = float(wins) / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return float((centre - margin) / denominator)


def evaluate_predictions(
    p_up: Iterable[float],
    y: Iterable[int],
    *,
    threshold: float = 0.80,
    min_agreement: float = 0.0,
    agreement: Optional[Iterable[float]] = None,
    future_returns: Optional[Iterable[float]] = None,
    total_cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """Calculate all-bar and selective out-of-sample performance.

    ``hit_rate`` is only for bars passing the abstention policy.  ``coverage``
    and ``signals`` make the trade-off impossible to hide.  P&L uses the
    execution-aligned future return supplied by ``build_labeled_frame`` and
    deducts a configurable cost on every selected signal.
    """

    probabilities, target, returns, agreements = _as_arrays(p_up, y, future_returns, agreement)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    direction = (probabilities >= 0.5).astype(int)
    all_hits = direction == target
    agreement_mask = np.ones(len(target), dtype=bool) if agreements is None else agreements >= min_agreement
    selected = (confidence >= float(threshold)) & agreement_mask
    signal_count = int(selected.sum())

    brier = float(np.mean((probabilities - target) ** 2))
    logloss = float(-np.mean(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)))
    result: Dict[str, Any] = {
        "rows": int(len(target)),
        "all_bar_accuracy": float(np.mean(all_hits)),
        "brier_score": brier,
        "log_loss": logloss,
        "threshold": float(threshold),
        "min_agreement": float(min_agreement),
        "signals": signal_count,
        "coverage": float(signal_count / len(target)),
        "hit_rate": None,
        "wilson_lower_95": None,
        "wins": 0,
        "losses": 0,
        "avg_selected_return_bps": None,
        "net_selected_return_bps": None,
        "profit_factor": None,
        "max_drawdown": None,
        "cost_assumption_bps": float(total_cost_bps),
    }

    if signal_count:
        selected_hits = all_hits[selected]
        result["hit_rate"] = float(np.mean(selected_hits))
        result["wilson_lower_95"] = wilson_lower_bound(int(selected_hits.sum()), signal_count)
        result["wins"] = int(selected_hits.sum())
        result["losses"] = int(signal_count - selected_hits.sum())

    if returns is not None and signal_count:
        signed_return = returns[selected] * np.where(direction[selected] == 1, 1.0, -1.0)
        net_return_bps = signed_return * 10000.0 - float(total_cost_bps)
        result["avg_selected_return_bps"] = float(np.mean(net_return_bps))
        result["net_selected_return_bps"] = float(np.sum(net_return_bps))
        gross_wins = float(np.sum(net_return_bps[net_return_bps > 0]))
        gross_losses = float(-np.sum(net_return_bps[net_return_bps < 0]))
        result["profit_factor"] = float(gross_wins / gross_losses) if gross_losses > 0 else None
        equity = np.cumprod(np.exp(net_return_bps / 10000.0))
        result["max_drawdown"] = float(_max_drawdown(equity))

    return result


def select_threshold(
    p_up: Iterable[float],
    y: Iterable[int],
    *,
    target_hit_rate: float = 0.80,
    min_signals: int = 25,
    threshold_min: float = 0.50,
    threshold_max: float = 0.97,
    threshold_step: float = 0.01,
    min_agreement: float = 0.0,
    agreement: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    """Choose an abstention threshold using validation data only.

    First preference is the largest validation coverage that meets the hit
    target and minimum signal count.  If the target is not achievable, the
    policy chooses the highest observed hit rate with enough signals and marks
    the failure.  It never looks at the final test period.
    """

    probabilities, target, _, agreements = _as_arrays(p_up, y, agreement=agreement)
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        metrics = evaluate_predictions(
            probabilities,
            target,
            threshold=float(threshold),
            min_agreement=min_agreement,
            agreement=agreements,
        )
        rows.append(metrics)

    enough = [row for row in rows if row["signals"] >= int(min_signals) and row["hit_rate"] is not None]
    target_rows = [row for row in enough if row["hit_rate"] >= float(target_hit_rate)]
    if target_rows:
        # At a fixed target, lower threshold / higher coverage is more useful.
        chosen = max(target_rows, key=lambda row: (row["coverage"], -row["threshold"]))
        target_met = True
    elif enough:
        # Fail honestly rather than overstate an unattainable target.
        chosen = max(enough, key=lambda row: (row["hit_rate"], row["coverage"]))
        target_met = False
    else:
        # A validation slice with fewer than min_signals cannot certify an
        # operating point.  Use the strictest threshold and fail closed.
        chosen = max(rows, key=lambda row: row["threshold"])
        target_met = False

    return {
        "chosen_threshold": float(chosen["threshold"]),
        "target_hit_rate": float(target_hit_rate),
        "target_met_on_validation": bool(target_met),
        "minimum_validation_signals": int(min_signals),
        "chosen_validation_metrics": chosen,
        "sweep": [
            {
                "threshold": row["threshold"],
                "signals": row["signals"],
                "coverage": row["coverage"],
                "hit_rate": row["hit_rate"],
            }
            for row in rows
        ],
    }


def calibration_bins(p_up: Iterable[float], y: Iterable[int], bins: int = 10) -> List[Dict[str, Any]]:
    """Return reliability bins for a simple probability sanity check."""

    probabilities, target, _, _ = _as_arrays(p_up, y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: List[Dict[str, Any]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)
        result.append(
            {
                "lower": float(left),
                "upper": float(right),
                "count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
                "observed_up_rate": float(target[mask].mean()) if mask.any() else None,
            }
        )
    return result
