"""Public data-source registry and decision rules for XAUUSD experiments.

The registry is intentionally explicit: a publicly viewable chart, a gold
futures ticker and a spot XAU/USD feed are not interchangeable datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class PublicSource:
    name: str
    url: str
    instrument: str
    exact_spot_xauusd: bool
    timeframes: Tuple[str, ...]
    access: str
    role: str
    limitations: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PUBLIC_SOURCES: Tuple[PublicSource, ...] = (
    PublicSource(
        name="Dukascopy historical feed",
        url="https://www.dukascopy-node.app/instrument/xauusd",
        instrument="XAUUSD",
        exact_spot_xauusd=True,
        timeframes=("m30", "h1"),
        access="public downloader; local Node.js/npm required",
        role="primary intraday source",
        limitations="Bid/ask feed is broker/vendor specific; retain raw file hash and UTC convention.",
    ),
    PublicSource(
        name="Twelve Data XAU/USD",
        url="https://twelvedata.com/markets/300755/commodity/xau-usd/historical-data",
        instrument="XAU/USD",
        exact_spot_xauusd=True,
        timeframes=("daily", "intraday via API plan"),
        access="public chart; API key/plan for programmatic intraday history",
        role="reference/secondary validation",
        limitations="Do not treat a daily chart as a 30m or 1h dataset; API quotas and licensing apply.",
    ),
    PublicSource(
        name="XAUS spot API",
        url="https://xaus.com/api/",
        instrument="XAUUSD",
        exact_spot_xauusd=True,
        timeframes=("daily", "2-minute recent intraday"),
        access="keyless public API",
        role="live/reference sanity check",
        limitations="Intraday retention is short; it cannot reconstruct the full July 2026 month.",
    ),
    PublicSource(
        name="Yahoo Finance GC=F",
        url="https://finance.yahoo.com/quote/GC=F/history/",
        instrument="COMEX gold futures",
        exact_spot_xauusd=False,
        timeframes=("30m", "1h"),
        access="public chart endpoint with vendor retention limits",
        role="diagnostic futures proxy only",
        limitations="Not XAUUSD spot; futures basis, roll and session differ. Never mix into a spot certification.",
    ),
    PublicSource(
        name="Kaggle/Hugging Face archives",
        url="https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024",
        instrument="XAUUSD-labelled archives",
        exact_spot_xauusd=False,
        timeframes=("m30", "h1", "daily"),
        access="public download subject to dataset terms",
        role="reproducibility/legacy research only",
        limitations="Versions can stop before the requested month and provenance/vendor definitions may differ.",
    ),
)


def source_matrix() -> List[Dict[str, Any]]:
    """Return the source registry as JSON-friendly records."""

    return [source.to_dict() for source in PUBLIC_SOURCES]


def exact_intraday_source() -> PublicSource:
    """Return the source allowed for a strict spot XAUUSD dry-run."""

    return PUBLIC_SOURCES[0]
