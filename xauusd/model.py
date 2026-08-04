"""A calibrated, abstaining ensemble for 30-minute direction signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Keep import errors actionable for users who forgot requirements.txt.
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only without deps
    joblib = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


class ModelDependencyError(ImportError):
    """Raised when the scientific Python dependencies are not installed."""


def _require_ml_dependencies() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ModelDependencyError(
            "The XAUUSD model needs numpy, pandas, scikit-learn and joblib. "
            "Run `pip install -r requirements.txt`."
        ) from _SKLEARN_IMPORT_ERROR


@dataclass(frozen=True)
class Signal:
    """A prediction with an explicit no-trade state."""

    timestamp: str
    direction: str
    p_up: float
    confidence: float
    agreement: float
    threshold: float
    take_signal: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "direction": self.direction,
            "p_up": round(float(self.p_up), 6),
            "confidence": round(float(self.confidence), 6),
            "agreement": round(float(self.agreement), 6),
            "threshold": round(float(self.threshold), 6),
            "take_signal": bool(self.take_signal),
        }


class DirectionEnsemble:
    """Tree + linear ensemble with internal time-ordered probability calibration.

    The base models capture complementary structures:

    * histogram gradient boosting: non-linear interactions and regimes;
    * random forest: robust threshold/interaction diversity;
    * regularised logistic regression: a stable low-variance anchor.

    Their raw probabilities are averaged, then passed through a one-variable
    Platt calibrator trained on the *tail* of the training period.  The final
    operating threshold is still selected on a separate validation period.
    """

    ARTIFACT_VERSION = 1

    def __init__(
        self,
        *,
        random_state: int = 42,
        rf_estimators: int = 180,
        hgb_max_iter: int = 220,
        hgb_learning_rate: float = 0.045,
        hgb_max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
    ) -> None:
        _require_ml_dependencies()
        self.random_state = int(random_state)
        self.rf_estimators = int(rf_estimators)
        self.hgb_max_iter = int(hgb_max_iter)
        self.hgb_learning_rate = float(hgb_learning_rate)
        self.hgb_max_leaf_nodes = int(hgb_max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.feature_names_: List[str] = []
        self.imputer_: Any = None
        self.models_: List[Any] = []
        self.calibrator_: Any = None
        self.fitted_: bool = False

    def _new_models(self) -> List[Any]:
        return [
            HistGradientBoostingClassifier(
                max_iter=self.hgb_max_iter,
                learning_rate=self.hgb_learning_rate,
                max_leaf_nodes=self.hgb_max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.random_state,
            ),
            RandomForestClassifier(
                n_estimators=self.rf_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            ),
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=0.20,
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
        ]

    @staticmethod
    def _as_frame(features: Any, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        else:
            array = np.asarray(features)
            if array.ndim != 2:
                raise ValueError("features must be a two-dimensional array or DataFrame")
            names = list(feature_names or [f"feature_{i}" for i in range(array.shape[1])])
            frame = pd.DataFrame(array, columns=names)
        if frame.empty:
            raise ValueError("features are empty")
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.replace([np.inf, -np.inf], np.nan)

    def _transform(self, features: Any) -> np.ndarray:
        if not self.fitted_ and self.imputer_ is None:
            raise RuntimeError("model is not fitted")
        frame = self._as_frame(features)
        missing = [name for name in self.feature_names_ if name not in frame.columns]
        if missing:
            raise ValueError(f"prediction data is missing features: {missing[:5]}")
        # Extra columns are ignored, but feature order is always explicit.
        frame = frame.loc[:, self.feature_names_]
        return self.imputer_.transform(frame)

    @staticmethod
    def _raw_probability(models: Sequence[Any], matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probabilities: List[np.ndarray] = []
        for model in models:
            proba = np.asarray(model.predict_proba(matrix))[:, 1]
            probabilities.append(proba)
        stacked = np.vstack(probabilities)
        return stacked.mean(axis=0), stacked

    def _fit_base_models(self, matrix: np.ndarray, y: np.ndarray) -> None:
        self.models_ = self._new_models()
        for model in self.models_:
            model.fit(matrix, y)

    def fit(self, features: Any, y: Iterable[int]) -> "DirectionEnsemble":
        """Fit the ensemble with a chronological internal calibration split."""

        _require_ml_dependencies()
        frame = self._as_frame(features)
        target = np.asarray(list(y), dtype=int)
        if len(frame) != len(target):
            raise ValueError("features and target lengths differ")
        if len(target) < 30:
            raise ValueError("at least 30 labelled rows are required to fit")
        unique = np.unique(target)
        if not np.array_equal(unique, np.array([0, 1])):
            raise ValueError("target must contain both direction classes 0 and 1")

        self.feature_names_ = list(frame.columns)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        matrix = self.imputer_.fit_transform(frame)

        # Calibrate on the last 20% of the training data, preserving time
        # order.  The model is then refit on all training observations.
        cut = int(len(target) * 0.80)
        cut = min(max(cut, 20), len(target) - 10)
        internal_train_x, internal_cal_x = matrix[:cut], matrix[cut:]
        internal_train_y, internal_cal_y = target[:cut], target[cut:]
        if np.unique(internal_train_y).size == 2 and np.unique(internal_cal_y).size == 2:
            calibration_models = self._new_models()
            for model in calibration_models:
                model.fit(internal_train_x, internal_train_y)
            raw, _ = self._raw_probability(calibration_models, internal_cal_x)
            self.calibrator_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(raw.reshape(-1, 1), internal_cal_y)
        else:
            # Very small or unusually one-sided samples cannot calibrate.  The
            # operating threshold is still chosen out of sample by the caller.
            self.calibrator_ = None

        self._fit_base_models(matrix, target)
        self.fitted_ = True
        return self

    def predict_components(self, features: Any) -> Dict[str, np.ndarray]:
        """Return calibrated score, base-model scores and agreement."""

        _require_ml_dependencies()
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        matrix = self._transform(features)
        raw, components = self._raw_probability(self.models_, matrix)
        if self.calibrator_ is not None:
            p_up = self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            p_up = raw
        # Agreement is high when independent models point to the same side.
        # 1.0 means identical probabilities; 0.0 means maximal disagreement.
        agreement = 1.0 - np.clip(2.0 * components.std(axis=0), 0.0, 1.0)
        return {
            "p_up": np.clip(p_up, 0.0, 1.0),
            "raw_p_up": np.clip(raw, 0.0, 1.0),
            "agreement": agreement,
            "model_p_up": components,
        }

    def predict_proba(self, features: Any) -> np.ndarray:
        """Return ``[P(down), P(up)]`` for sklearn compatibility."""

        p_up = self.predict_components(features)["p_up"]
        return np.column_stack([1.0 - p_up, p_up])

    def signal_frame(
        self,
        features: pd.DataFrame,
        *,
        threshold: float,
        min_agreement: float = 0.55,
    ) -> pd.DataFrame:
        """Convert probabilities into UP/DOWN/NO_TRADE decisions."""

        components = self.predict_components(features)
        p_up = components["p_up"]
        confidence = np.maximum(p_up, 1.0 - p_up)
        direction_up = p_up >= 0.5
        take = (confidence >= float(threshold)) & (components["agreement"] >= float(min_agreement))
        result = pd.DataFrame(
            {
                "p_up": p_up,
                "confidence": confidence,
                "agreement": components["agreement"],
                "direction": np.where(direction_up, "UP", "DOWN"),
                "take_signal": take,
            },
            index=features.index,
        )
        result.loc[~take, "direction"] = "NO_TRADE"
        return result

    def latest_signal(
        self,
        features: pd.DataFrame,
        *,
        threshold: float,
        min_agreement: float = 0.55,
    ) -> Signal:
        if features.empty:
            raise ValueError("cannot predict an empty feature frame")
        row = self.signal_frame(features.tail(1), threshold=threshold, min_agreement=min_agreement).iloc[0]
        timestamp = features.index[-1].isoformat() if hasattr(features.index[-1], "isoformat") else str(features.index[-1])
        return Signal(
            timestamp=timestamp,
            direction=str(row["direction"]),
            p_up=float(row["p_up"]),
            confidence=float(row["confidence"]),
            agreement=float(row["agreement"]),
            threshold=float(threshold),
            take_signal=bool(row["take_signal"]),
        )

    def save(self, path: str | Path, *, threshold: float, metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Persist model and policy metadata in a versioned joblib artifact."""

        _require_ml_dependencies()
        if not self.fitted_:
            raise RuntimeError("cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": self.ARTIFACT_VERSION,
            "model": self,
            "threshold": float(threshold),
            "metadata": dict(metadata or {}),
        }
        joblib.dump(payload, destination, compress=3)

    @staticmethod
    def load(path: str | Path) -> Tuple["DirectionEnsemble", float, Dict[str, Any]]:
        """Load ``(model, threshold, metadata)`` from an artifact."""

        _require_ml_dependencies()
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("invalid XAUUSD model artifact")
        if payload.get("artifact_version") != DirectionEnsemble.ARTIFACT_VERSION:
            raise ValueError("unsupported XAUUSD model artifact version")
        model = payload["model"]
        if not isinstance(model, DirectionEnsemble) or not model.fitted_:
            raise ValueError("artifact does not contain a fitted DirectionEnsemble")
        return model, float(payload["threshold"]), dict(payload.get("metadata", {}))
