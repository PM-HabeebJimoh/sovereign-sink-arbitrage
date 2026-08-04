"""OHLCV ingestion and data-quality checks for XAUUSD research.

The model treats timestamps as the *close time* of a bar.  This convention is
important: a signal generated at timestamp ``t`` may only use information in
that row and is executed at the next row's open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pandas as pd


class DataError(ValueError):
    """Raised when a price file cannot be safely used for research."""


@dataclass(frozen=True)
class DataQualityReport:
    """Compact, serialisable description of an input data set."""

    rows: int
    start: str
    end: str
    timeframe_minutes: int
    median_interval_minutes: float
    largest_gap_minutes: float
    gaps_over_expected: int
    duplicate_timestamps_removed: int
    resampled_to_timeframe: bool
    has_volume: bool
    external_columns: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Brokers and vendors use dozens of variants for these columns.  Normalising
# names makes the ingestion layer useful with MT5, Dukascopy and exported CSVs.
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "timestamp": (
        "timestamp",
        "time",
        "datetime",
        "date",
        "gmt time",
        "utc time",
        "opentime",
        "closetime",
    ),
    "open": ("open", "o", "bidopen", "openprice"),
    "high": ("high", "h", "bidhigh", "highprice"),
    "low": ("low", "l", "bidlow", "lowprice"),
    "close": ("close", "c", "bidclose", "closeprice", "last"),
    "volume": ("volume", "vol", "tickvol", "tickvolume", "realvolume", "realvol"),
}


def _clean_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _column_map(columns: Iterable[Any]) -> Dict[str, str]:
    cleaned = {_clean_name(c): str(c) for c in columns}
    result: Dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            key = _clean_name(alias)
            if key in cleaned:
                result[canonical] = cleaned[key]
                break
    return result


def _parse_timestamp(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps or epoch seconds/milliseconds as UTC."""

    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        median = float(numeric.dropna().abs().median()) if numeric.notna().any() else 0.0
        unit = "ms" if median > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _normalise_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataError("price data is empty")

    # Internal callers naturally pass a prepared frame with a DatetimeIndex.
    # Treat that index as the timestamp instead of requiring a redundant
    # timestamp column on every feature-engineering call.
    frame = frame.copy()
    if "timestamp" not in _column_map(frame.columns) and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()
        if "timestamp" not in _column_map(frame.columns):
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})

    # MT5 commonly exports DATE and TIME as two columns.  Combining them
    # before alias resolution prevents the DATE-only alias from silently
    # collapsing every intraday row onto midnight.
    cleaned_columns = {_clean_name(column): column for column in frame.columns}
    if "timestamp" not in cleaned_columns and "date" in cleaned_columns and "time" in cleaned_columns:
        frame["timestamp"] = (
            frame[cleaned_columns["date"]].astype(str).str.strip()
            + " "
            + frame[cleaned_columns["time"]].astype(str).str.strip()
        )

    mapping = _column_map(frame.columns)
    missing = [name for name in ("timestamp", "open", "high", "low", "close") if name not in mapping]
    if missing:
        raise DataError(
            "missing required columns: " + ", ".join(missing) +
            "; expected timestamp/open/high/low/close (case and vendor aliases are accepted)"
        )

    renamed = frame.rename(columns={source: target for target, source in mapping.items()}).copy()
    keep = ["timestamp", "open", "high", "low", "close"]
    if "volume" in renamed.columns:
        keep.append("volume")

    # Preserve optional, numeric, time-aligned drivers such as dxy_close,
    # us10y, vix_close or silver_close.  They are only used when their names
    # are explicitly prefixed with an external/market hint.
    canonical = set(keep)
    optional = []
    for column in renamed.columns:
        if column in canonical or column in optional:
            continue
        safe = _clean_name(column)
        if safe.startswith(("external", "dxy", "us10y", "us2y", "vix", "silver", "oil", "btc", "eurusd")):
            if pd.api.types.is_numeric_dtype(renamed[column]):
                optional.append(column)
    keep.extend(optional)
    renamed = renamed[keep]

    renamed["timestamp"] = _parse_timestamp(renamed["timestamp"])
    for column in keep:
        if column != "timestamp":
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

    renamed = renamed.dropna(subset=["timestamp", "open", "high", "low", "close"])
    duplicate_count = int(renamed["timestamp"].duplicated(keep="last").sum())
    renamed = renamed.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    renamed = renamed.set_index("timestamp")
    renamed.index.name = "timestamp"

    if renamed.empty:
        raise DataError("no valid rows remain after timestamp parsing")

    # Reject malformed candles rather than silently correcting them.  A bad
    # source should be fixed at ingestion, not hidden by a model.
    invalid = (
        (renamed[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (renamed["high"] < renamed[["open", "close"]].max(axis=1))
        | (renamed["low"] > renamed[["open", "close"]].min(axis=1))
        | (renamed["high"] < renamed["low"])
    )
    if invalid.any():
        bad_count = int(invalid.sum())
        raise DataError(f"found {bad_count} malformed OHLC rows")

    if "volume" in renamed.columns:
        renamed["volume"] = renamed["volume"].clip(lower=0)

    return renamed, duplicate_count


def _resample_bars(frame: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Convert lower-frequency bars into right-labelled target bars."""

    aggregations: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in frame.columns:
        aggregations["volume"] = "sum"

    rule = f"{int(timeframe_minutes)}min"
    resampler = frame.resample(rule, label="right", closed="right")
    result = resampler.agg(aggregations)
    # Do not keep a partially populated first/last aggregate.  A 15-minute
    # source should contribute two observations to a complete 30-minute bar;
    # a 1-minute source should contribute roughly thirty.
    source_deltas = frame.index.to_series().diff().dropna().dt.total_seconds().div(60)
    expected_count = max(1, int(round(float(timeframe_minutes) / float(source_deltas.median())))) if not source_deltas.empty else 1
    source_count = frame["close"].resample(rule, label="right", closed="right").count()
    result = result.loc[source_count >= expected_count]
    for column in frame.columns:
        if column in aggregations:
            continue
        # External drivers are sampled at the last known value in the bar;
        # they must already be time-aligned by their provider.
        result[column] = frame[column].resample(rule, label="right", closed="right").last()
    result = result.dropna(subset=["open", "high", "low", "close"])
    result.index.name = "timestamp"
    return result


def prepare_bars(
    frame: pd.DataFrame,
    *,
    timeframe_minutes: int = 30,
    resample: bool = True,
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, DataQualityReport]]:
    """Validate and optionally normalise bars to a 30-minute or 1-hour index.

    Parameters
    ----------
    frame:
        A DataFrame with timestamp/open/high/low/close.  Timestamps are
        converted to UTC.  They are interpreted as bar close times.
    timeframe_minutes:
        Target bar size. The research models support 30 and 60 minutes.
    resample:
        Resample lower-frequency source data when the median interval is
        different from the target. Higher-frequency source bars can be
        aggregated; higher-timeframe data is rejected rather than fabricated.
    return_report:
        Return ``(bars, quality_report)`` when true.
    """

    if timeframe_minutes not in (30, 60):
        raise DataError("timeframe_minutes must be 30 or 60")
    normalised, duplicate_count = _normalise_frame(frame)
    deltas = normalised.index.to_series().diff().dropna().dt.total_seconds().div(60)
    median_interval = float(deltas.median()) if not deltas.empty else float(timeframe_minutes)
    if resample and not deltas.empty and median_interval > float(timeframe_minutes) + 1.0:
        raise DataError(
            f"source interval is {median_interval:.1f} minutes; a {timeframe_minutes}-minute model cannot reconstruct "
            "missing intrabar prices from higher-timeframe data"
        )
    should_resample = bool(
        resample and (not deltas.empty) and abs(median_interval - float(timeframe_minutes)) > 1.0
    )
    bars = _resample_bars(normalised, timeframe_minutes) if should_resample else normalised

    if bars.empty:
        raise DataError(f"no complete {timeframe_minutes}-minute bars remain")

    post_deltas = bars.index.to_series().diff().dropna().dt.total_seconds().div(60)
    largest_gap = float(post_deltas.max()) if not post_deltas.empty else 0.0
    gaps = int((post_deltas > float(timeframe_minutes) * 1.5).sum())
    external = tuple(
        column for column in bars.columns
        if column not in {"open", "high", "low", "close", "volume"}
    )
    report = DataQualityReport(
        rows=int(len(bars)),
        start=bars.index.min().isoformat(),
        end=bars.index.max().isoformat(),
        timeframe_minutes=int(timeframe_minutes),
        median_interval_minutes=float(post_deltas.median()) if not post_deltas.empty else float(timeframe_minutes),
        largest_gap_minutes=largest_gap,
        gaps_over_expected=gaps,
        duplicate_timestamps_removed=duplicate_count,
        resampled_to_timeframe=should_resample,
        has_volume="volume" in bars.columns and bool(bars["volume"].notna().any()),
        external_columns=external,
    )
    return (bars, report) if return_report else bars


def load_bars(
    path: Union[str, Path],
    *,
    timeframe_minutes: int = 30,
    resample: bool = True,
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, DataQualityReport]]:
    """Load a CSV price history and pass it through :func:`prepare_bars`."""

    source = Path(path)
    if not source.exists():
        raise DataError(f"price file does not exist: {source}")
    try:
        # The Python engine detects the comma/semicolon variants emitted by
        # MT5 and several broker export tools.  Users may still pass a
        # pre-loaded DataFrame to prepare_bars for unusual formats.
        frame = pd.read_csv(source, sep=None, engine="python")
    except Exception as exc:  # pragma: no cover - parser-specific errors
        raise DataError(f"could not read {source}: {exc}") from exc
    return prepare_bars(
        frame,
        timeframe_minutes=timeframe_minutes,
        resample=resample,
        return_report=return_report,
    )


def download_yahoo_30m(
    symbol: str = "GC=F",
    *,
    period1: Optional[int] = None,
    period2: Optional[int] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download Yahoo chart data without adding a heavyweight data package.

    Yahoo's 30-minute retention is limited and ``GC=F`` is gold futures, not
    broker-specific spot XAUUSD.  This helper is for quick experiments only;
    use a broker/tick vendor export for a production-quality history.
    """

    import time
    from urllib.parse import quote
    import requests

    params: Dict[str, Any] = {"interval": "30m", "events": "history", "includePrePost": "true"}
    if period1 is None or period2 is None:
        params["range"] = "60d"
    else:
        params.update({"period1": int(period1), "period2": int(period2)})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "xauusd-research/1.0"})
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise DataError(f"Yahoo chart error: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise DataError("Yahoo returned no chart data")
    timestamps = result.get("timestamp") or []
    quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote_data.get("open", []),
            "high": quote_data.get("high", []),
            "low": quote_data.get("low", []),
            "close": quote_data.get("close", []),
            "volume": quote_data.get("volume", []),
        }
    )
    if frame.empty:
        raise DataError(f"Yahoo returned no rows for {symbol}")
    # Avoid a cache/proxy serving an old response being mistaken for current
    # data in a notebook; the timestamp is retained in the returned frame.
    frame.attrs["source"] = f"Yahoo chart {symbol} fetched {time.time():.0f}"
    return prepare_bars(frame, resample=False)
