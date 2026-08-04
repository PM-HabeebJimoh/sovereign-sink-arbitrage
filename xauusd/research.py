"""End-to-end leakage-resistant training and holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .evaluation import (
    calibration_bins,
    evaluate_predictions,
    purged_time_split,
    select_threshold,
)
from .features import build_labeled_frame
from .model import DirectionEnsemble


@dataclass
class ResearchResult:
    """Model, policy threshold and machine-readable experiment report."""

    model: DirectionEnsemble
    threshold: float
    min_agreement: float
    report: Dict[str, Any]
    feature_columns: Tuple[str, ...]

    def save(self, model_path: str | Path) -> None:
        """Save the fitted model with all policy and data metadata."""

        self.model.save(
            model_path,
            threshold=self.threshold,
            metadata={
                "min_agreement": self.min_agreement,
                "feature_columns": list(self.feature_columns),
                "report_summary": {
                    "test": self.report.get("test", {}),
                    "validation": self.report.get("validation", {}),
                    "status": self.report.get("status"),
                },
            },
        )


def _frame_arrays(frame: pd.DataFrame, feature_columns: Tuple[str, ...], rows: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    subset = frame.iloc[rows]
    return (
        subset.loc[:, list(feature_columns)],
        subset["label"].astype(int).to_numpy(),
        subset["future_return"].astype(float).to_numpy(),
        subset.index.to_numpy(),
    )


def _validate_min_history(frame: pd.DataFrame, config: ResearchConfig) -> Dict[str, Any]:
    counts = frame["label"].value_counts().to_dict()
    return {
        "rows_after_labeling": int(len(frame)),
        "up_labels": int(counts.get(1.0, counts.get(1, 0))),
        "down_labels": int(counts.get(0.0, counts.get(0, 0))),
        "minimum_recommended_rows": int(config.min_rows),
        "history_warning": bool(len(frame) < config.min_rows),
    }


def run_research(bars: pd.DataFrame, config: Optional[ResearchConfig] = None) -> ResearchResult:
    """Train, select a no-trade threshold and score one untouched test period.

    Procedure:

    1. create execution-aligned labels and causal features;
    2. chronological 60/20/20 split with an embargo;
    3. train the ensemble on the first period;
    4. select confidence threshold on validation only;
    5. refit on train + validation (threshold remains frozen);
    6. evaluate once on the untouched final test period.

    The test result is the only number that matters for the requested hit-rate
    claim.  The function will report failure when it does not meet 80% with
    sufficient coverage; it never changes the threshold to rescue the test.
    """

    cfg = config or ResearchConfig()
    labelled, feature_columns_list = build_labeled_frame(
        bars,
        horizon_bars=cfg.horizon_bars,
        label_threshold_bps=cfg.label_threshold_bps,
        label_atr_fraction=cfg.label_atr_fraction,
    )
    feature_columns = tuple(feature_columns_list)
    if len(labelled) < 60:
        raise ValueError("not enough labelled 30-minute history; supply at least 60 usable rows")

    split = purged_time_split(
        labelled.index,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
        embargo_bars=max(cfg.embargo_bars, cfg.horizon_bars),
    )
    report: Dict[str, Any] = {
        "status": "not_certified",
        "config": cfg.to_dict(),
        "dataset": _validate_min_history(labelled, cfg),
        "split": split.to_dict(),
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
    }

    train_x, train_y, _, _ = _frame_arrays(labelled, feature_columns, split.train)
    validation_x, validation_y, validation_returns, _ = _frame_arrays(labelled, feature_columns, split.validation)
    model = DirectionEnsemble(
        random_state=cfg.random_state,
        rf_estimators=cfg.rf_estimators,
        hgb_max_iter=cfg.hgb_max_iter,
        hgb_learning_rate=cfg.hgb_learning_rate,
        hgb_max_leaf_nodes=cfg.hgb_max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
    )
    model.fit(train_x, train_y)
    validation_components = model.predict_components(validation_x)

    # A small validation slice should not pretend to have 25 independent
    # trades.  Keep the configured minimum where possible, but make the
    # limitation visible in the report.
    minimum_validation_signals = min(
        cfg.min_validation_signals,
        max(5, int(len(validation_y) * 0.20)),
    )
    threshold_report = select_threshold(
        validation_components["p_up"],
        validation_y,
        target_hit_rate=cfg.target_hit_rate,
        min_signals=minimum_validation_signals,
        threshold_min=cfg.threshold_min,
        threshold_max=cfg.threshold_max,
        threshold_step=cfg.threshold_step,
        min_agreement=cfg.min_ensemble_agreement,
        agreement=validation_components["agreement"],
    )
    threshold = float(threshold_report["chosen_threshold"])
    report["threshold_selection"] = threshold_report
    report["validation"] = evaluate_predictions(
        validation_components["p_up"],
        validation_y,
        threshold=threshold,
        min_agreement=cfg.min_ensemble_agreement,
        agreement=validation_components["agreement"],
        future_returns=validation_returns,
        total_cost_bps=cfg.total_cost_bps,
    )
    report["validation"]["calibration_bins"] = calibration_bins(validation_components["p_up"], validation_y)

    # Refitting after policy selection is legitimate: validation data may be
    # used for fitting the final model, but it may not be used to retune the
    # operating threshold after looking at the test period.
    fit_rows = np.concatenate([split.train, split.validation])
    fit_x, fit_y, _, _ = _frame_arrays(labelled, feature_columns, fit_rows)
    final_model = DirectionEnsemble(
        random_state=cfg.random_state,
        rf_estimators=cfg.rf_estimators,
        hgb_max_iter=cfg.hgb_max_iter,
        hgb_learning_rate=cfg.hgb_learning_rate,
        hgb_max_leaf_nodes=cfg.hgb_max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
    )
    final_model.fit(fit_x, fit_y)

    test_x, test_y, test_returns, test_index = _frame_arrays(labelled, feature_columns, split.test)
    test_components = final_model.predict_components(test_x)
    test_metrics = evaluate_predictions(
        test_components["p_up"],
        test_y,
        threshold=threshold,
        min_agreement=cfg.min_ensemble_agreement,
        agreement=test_components["agreement"],
        future_returns=test_returns,
        total_cost_bps=cfg.total_cost_bps,
    )
    test_metrics["calibration_bins"] = calibration_bins(test_components["p_up"], test_y)
    test_metrics["test_start"] = str(test_index[0]) if len(test_index) else None
    test_metrics["test_end"] = str(test_index[-1]) if len(test_index) else None
    report["test"] = test_metrics

    # Baselines are intentionally simple.  A complex ensemble that cannot
    # beat a majority or current-bar-momentum baseline is not disruptive; it
    # is just more complicated.
    majority_class = int(np.mean(train_y) >= 0.5)
    majority_probability = 0.51 if majority_class == 1 else 0.49
    momentum_probability = np.where(test_x["ret_1"].fillna(0.0).to_numpy() >= 0.0, 0.51, 0.49)
    report["baselines"] = {
        "majority_direction": evaluate_predictions(
            np.full(len(test_y), majority_probability),
            test_y,
            threshold=0.5,
            future_returns=test_returns,
            total_cost_bps=cfg.total_cost_bps,
        ),
        "current_bar_momentum": evaluate_predictions(
            momentum_probability,
            test_y,
            threshold=0.5,
            future_returns=test_returns,
            total_cost_bps=cfg.total_cost_bps,
        ),
    }

    enough_test_signals = test_metrics["signals"] >= max(10, min(20, minimum_validation_signals))
    target_met = bool(
        enough_test_signals
        and test_metrics["hit_rate"] is not None
        and test_metrics["hit_rate"] >= cfg.target_hit_rate
    )
    positive_expectancy = bool(
        test_metrics.get("net_selected_return_bps") is not None
        and test_metrics["net_selected_return_bps"] > 0
    )
    statistical_support = bool(
        test_metrics.get("wilson_lower_95") is not None
        and test_metrics["wilson_lower_95"] >= cfg.target_hit_rate
    )
    report["certification"] = {
        "requested_hit_rate": cfg.target_hit_rate,
        "out_of_sample_hit_rate": test_metrics["hit_rate"],
        "out_of_sample_wilson_lower_95": test_metrics["wilson_lower_95"],
        "minimum_test_signals": int(max(10, min(20, minimum_validation_signals))),
        "enough_test_signals": enough_test_signals,
        "target_met_out_of_sample": target_met,
        "statistical_support_at_95pct": statistical_support,
        "positive_net_return_after_costs": positive_expectancy,
        "coverage_required_to_interpret": "Report hit rate together with signals and coverage; a zero-signal 100% is not a model.",
    }
    if target_met and not statistical_support:
        report["status"] = "target_met_but_low_statistical_confidence"
    elif target_met and not positive_expectancy:
        report["status"] = "target_met_but_negative_after_costs"
    else:
        report["status"] = "target_met_on_holdout" if target_met else "target_not_met_or_insufficient_coverage"
    return ResearchResult(
        model=final_model,
        threshold=threshold,
        min_agreement=cfg.min_ensemble_agreement,
        report=report,
        feature_columns=feature_columns,
    )


def predict_latest_from_bars(
    bars: pd.DataFrame,
    model_path: str | Path,
) -> Dict[str, Any]:
    """Generate one research signal from the latest complete bar."""

    model, threshold, metadata = DirectionEnsemble.load(model_path)
    from .features import build_features

    features = build_features(bars)
    # Preserve the artifact's training order and drop only rows with enough
    # information to be transformed; the model's imputer handles warm-up NaN.
    feature_columns = metadata.get("feature_columns") or model.feature_names_
    features = features.loc[:, [column for column in feature_columns if column in features.columns]]
    if not features.columns.tolist() == list(feature_columns):
        missing = sorted(set(feature_columns) - set(features.columns))
        raise ValueError(f"latest data is missing trained features: {missing[:5]}")
    signal = model.latest_signal(features, threshold=threshold, min_agreement=float(metadata.get("min_agreement", 0.55)))
    result = signal.to_dict()
    result["model_path"] = str(model_path)
    result["metadata"] = metadata
    return result
