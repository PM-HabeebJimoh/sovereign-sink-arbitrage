"""Configuration for the XAUUSD research pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ResearchConfig:
    """Reproducible defaults for a 30-minute or 1-hour, cost-aware experiment.

    The 80% value is a *selection target*, not a guaranteed outcome.  The
    policy is allowed to abstain.  A result is only considered useful when it
    also reports the number and fraction of bars on which it traded.
    """

    timeframe_minutes: int = 30
    horizon_bars: int = 1

    # Meaningful-move label.  A bar whose future return is inside this dead
    # zone is removed rather than turning micro-noise into a fake direction.
    label_threshold_bps: float = 5.0
    label_atr_fraction: float = 0.10

    # The operating threshold is chosen on a calibration/validation period,
    # never on the final test period.
    target_hit_rate: float = 0.80
    min_validation_signals: int = 25
    threshold_min: float = 0.50
    threshold_max: float = 0.97
    threshold_step: float = 0.01
    min_ensemble_agreement: float = 0.55

    # Approximate all-in cost for one round trip.  XAUUSD costs depend on the
    # broker, spread, session and account; this must be replaced with measured
    # costs before a live decision.
    round_trip_cost_bps: float = 4.0
    slippage_bps: float = 1.0

    # Time split.  All splits are chronological and have a purge/embargo.
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    embargo_bars: int = 1

    # Ensemble / reproducibility.
    random_state: int = 42
    rf_estimators: int = 180
    hgb_max_iter: int = 220
    hgb_learning_rate: float = 0.045
    hgb_max_leaf_nodes: int = 15
    min_samples_leaf: int = 20

    # The minimum number of rows needed to make a credible experiment.  The
    # CLI refuses to certify a result below this amount, but unit tests and
    # exploratory callers can lower it explicitly.
    min_rows: int = 500

    def __post_init__(self) -> None:
        if self.timeframe_minutes not in (30, 60):
            raise ValueError("timeframe_minutes must be 30 or 60")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if not 0 < self.target_hit_rate < 1:
            raise ValueError("target_hit_rate must be between 0 and 1")
        if not 0.5 <= self.threshold_min < self.threshold_max <= 0.999:
            raise ValueError("invalid probability threshold range")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be positive")
        if self.train_fraction <= 0 or self.validation_fraction <= 0:
            raise ValueError("train_fraction and validation_fraction must be positive")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train + validation fractions must leave a test period")
        if self.min_ensemble_agreement < 0 or self.min_ensemble_agreement > 1:
            raise ValueError("min_ensemble_agreement must be between 0 and 1")
        if self.round_trip_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs cannot be negative")

    @property
    def total_cost_bps(self) -> float:
        """Round-trip costs plus slippage, expressed in basis points."""

        return self.round_trip_cost_bps + self.slippage_bps

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable configuration metadata."""

        return asdict(self)
