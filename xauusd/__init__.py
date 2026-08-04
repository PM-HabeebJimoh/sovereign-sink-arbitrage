"""Research-grade XAUUSD 30-minute direction modelling toolkit.

This package is deliberately research and signal generation only.  It never
places trades, manages wallets, or treats a model score as a promise of
profitability.
"""

from .config import ResearchConfig
from .data import DataQualityReport, load_bars, prepare_bars
from .features import build_features, build_labeled_frame
from .research import ResearchResult, run_forward_dry_run, run_research

__all__ = [
    "DataQualityReport",
    "ResearchConfig",
    "ResearchResult",
    "run_forward_dry_run",
    "build_features",
    "build_labeled_frame",
    "load_bars",
    "prepare_bars",
    "run_research",
]

__version__ = "1.0.0"
