"""Rolling-origin evaluation for regime robustness.

The final holdout in :mod:`xauusd.research` is the certification gate.  This
module adds a stricter optional check: each test block is predicted by a model
trained only on earlier history, with a fresh validation threshold per fold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .evaluation import evaluate_predictions, select_threshold, wilson_lower_bound
from .features import build_labeled_frame
from .model import DirectionEnsemble


def _make_model(cfg: ResearchConfig) -> DirectionEnsemble:
    return DirectionEnsemble(
        random_state=cfg.random_state,
        rf_estimators=cfg.rf_estimators,
        hgb_max_iter=cfg.hgb_max_iter,
        hgb_learning_rate=cfg.hgb_learning_rate,
        hgb_max_leaf_nodes=cfg.hgb_max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
    )


def _aggregate_variable_threshold(
    probabilities: np.ndarray,
    target: np.ndarray,
    returns: np.ndarray,
    agreements: np.ndarray,
    thresholds: np.ndarray,
    cfg: ResearchConfig,
) -> Dict[str, Any]:
    """Aggregate folds whose thresholds were intentionally different."""

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    direction = (probabilities >= 0.5).astype(int)
    selected = (confidence >= thresholds) & (agreements >= cfg.min_ensemble_agreement)
    hits = direction == target
    signals = int(selected.sum())
    net = returns[selected] * np.where(direction[selected] == 1, 1.0, -1.0) * 10000.0 - cfg.total_cost_bps
    result = evaluate_predictions(probabilities, target, threshold=0.0, future_returns=returns, total_cost_bps=0.0)
    result.update(
        {
            "threshold_policy": "per_fold_validation_threshold",
            "signals": signals,
            "coverage": float(signals / len(target)) if len(target) else 0.0,
            "hit_rate": float(hits[selected].mean()) if signals else None,
            "wilson_lower_95": wilson_lower_bound(int(hits[selected].sum()), signals),
            "wins": int(hits[selected].sum()) if signals else 0,
            "losses": int(signals - hits[selected].sum()) if signals else 0,
            "avg_selected_return_bps": float(net.mean()) if signals else None,
            "net_selected_return_bps": float(net.sum()) if signals else None,
            "cost_assumption_bps": cfg.total_cost_bps,
        }
    )
    if signals:
        gross_wins = float(np.sum(net[net > 0]))
        gross_losses = float(-np.sum(net[net < 0]))
        result["profit_factor"] = gross_wins / gross_losses if gross_losses else None
        equity = np.cumprod(np.exp(net / 10000.0))
        peaks = np.maximum.accumulate(equity)
        result["max_drawdown"] = float(np.min(equity / peaks - 1.0))
    return result


def run_walk_forward(
    bars: pd.DataFrame,
    config: Optional[ResearchConfig] = None,
    *,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Run rolling-origin train/validation/test folds and return a report."""

    cfg = config or ResearchConfig()
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    labelled, feature_columns = build_labeled_frame(
        bars,
        horizon_bars=cfg.horizon_bars,
        label_threshold_bps=cfg.label_threshold_bps,
        label_atr_fraction=cfg.label_atr_fraction,
    )
    n = len(labelled)
    test_size = max(20, n // (n_splits + 2))
    embargo = max(cfg.embargo_bars, cfg.horizon_bars)
    first_test_start = n - (n_splits - 1) * test_size - test_size
    if first_test_start - test_size - embargo <= 30:
        raise ValueError(
            f"not enough labelled rows ({n}) for {n_splits} walk-forward folds; "
            "supply more history or use fewer folds"
        )

    fold_reports: List[Dict[str, Any]] = []
    all_probabilities: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_returns: List[np.ndarray] = []
    all_agreements: List[np.ndarray] = []
    all_thresholds: List[np.ndarray] = []

    for fold in range(n_splits):
        test_end = n - (n_splits - 1 - fold) * test_size
        test_start = test_end - test_size
        validation_end = test_start - embargo
        validation_start = validation_end - test_size
        train_end = validation_start - embargo
        if train_end <= 30 or validation_start < 0:
            raise ValueError("walk-forward fold has insufficient training history")

        train = labelled.iloc[:train_end]
        validation = labelled.iloc[validation_start:validation_end]
        test = labelled.iloc[test_start:test_end]
        model = _make_model(cfg)
        model.fit(train.loc[:, feature_columns], train["label"].astype(int))
        validation_components = model.predict_components(validation.loc[:, feature_columns])
        min_signals = min(cfg.min_validation_signals, max(5, int(len(validation) * 0.20)))
        policy = select_threshold(
            validation_components["p_up"],
            validation["label"].astype(int),
            target_hit_rate=cfg.target_hit_rate,
            min_signals=min_signals,
            threshold_min=cfg.threshold_min,
            threshold_max=cfg.threshold_max,
            threshold_step=cfg.threshold_step,
            min_agreement=cfg.min_ensemble_agreement,
            agreement=validation_components["agreement"],
        )
        threshold = float(policy["chosen_threshold"])
        components = model.predict_components(test.loc[:, feature_columns])
        metrics = evaluate_predictions(
            components["p_up"],
            test["label"].astype(int),
            threshold=threshold,
            min_agreement=cfg.min_ensemble_agreement,
            agreement=components["agreement"],
            future_returns=test["future_return"],
            total_cost_bps=cfg.total_cost_bps,
        )
        metrics.update(
            {
                "fold": fold + 1,
                "train_end": str(train.index[-1]),
                "validation_start": str(validation.index[0]),
                "validation_end": str(validation.index[-1]),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "threshold_selection": policy,
            }
        )
        fold_reports.append(metrics)
        all_probabilities.append(np.asarray(components["p_up"]))
        all_targets.append(test["label"].astype(int).to_numpy())
        all_returns.append(test["future_return"].astype(float).to_numpy())
        all_agreements.append(np.asarray(components["agreement"]))
        all_thresholds.append(np.full(len(test), threshold, dtype=float))

    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets)
    returns = np.concatenate(all_returns)
    agreements = np.concatenate(all_agreements)
    thresholds = np.concatenate(all_thresholds)
    aggregate = _aggregate_variable_threshold(probabilities, targets, returns, agreements, thresholds, cfg)
    enough = aggregate["signals"] >= max(10, cfg.min_validation_signals)
    target_met = bool(enough and aggregate["hit_rate"] is not None and aggregate["hit_rate"] >= cfg.target_hit_rate)
    statistical_support = bool(
        aggregate.get("wilson_lower_95") is not None
        and aggregate["wilson_lower_95"] >= cfg.target_hit_rate
    )
    positive_expectancy = bool(
        aggregate.get("net_selected_return_bps") is not None
        and aggregate["net_selected_return_bps"] > 0
    )
    if target_met and statistical_support and positive_expectancy:
        status = "target_met_on_walk_forward"
    elif target_met and not statistical_support:
        status = "target_met_but_low_statistical_confidence"
    elif target_met:
        status = "target_met_but_negative_after_costs"
    else:
        status = "target_not_met_or_insufficient_coverage"
    return {
        "status": status,
        "folds": fold_reports,
        "aggregate": aggregate,
        "certification": {
            "requested_hit_rate": cfg.target_hit_rate,
            "target_met": target_met,
            "statistical_support_at_95pct": statistical_support,
            "positive_net_return_after_costs": positive_expectancy,
            "signals": aggregate["signals"],
            "coverage": aggregate["coverage"],
            "n_splits": n_splits,
            "test_size_per_fold": test_size,
        },
    }
