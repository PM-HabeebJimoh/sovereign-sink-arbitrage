"""Command line interface for the XAUUSD research-only model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import ResearchConfig
from .data import DataError, download_dukascopy_xauusd, download_yahoo_intraday, load_bars
from .features import build_features
from .research import predict_latest_from_bars, run_forward_dry_run, run_research
from .sources import source_matrix


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        timeframe_minutes=args.timeframe,
        horizon_bars=args.horizon,
        label_threshold_bps=args.label_threshold_bps,
        label_atr_fraction=args.label_atr_fraction,
        target_hit_rate=args.target_hit_rate,
        min_validation_signals=args.min_validation_signals,
        round_trip_cost_bps=args.round_trip_cost_bps,
        slippage_bps=args.slippage_bps,
        min_rows=args.min_rows,
        random_state=args.random_state,
        rf_estimators=args.rf_estimators,
        hgb_max_iter=args.hgb_max_iter,
    )


def _add_research_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeframe", type=int, choices=(30, 60), default=30, help="bar size in minutes")
    parser.add_argument("--horizon", type=int, default=1, help="bars ahead; 1 means the next target-timeframe bar")
    parser.add_argument("--label-threshold-bps", type=float, default=5.0, help="fixed dead-zone for labels")
    parser.add_argument("--label-atr-fraction", type=float, default=0.10, help="ATR-relative dead-zone")
    parser.add_argument("--target-hit-rate", type=float, default=0.80, help="validation/test selection target")
    parser.add_argument("--min-validation-signals", type=int, default=25)
    parser.add_argument("--round-trip-cost-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--min-rows", type=int, default=500, help="warn when less history is supplied")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=180)
    parser.add_argument("--hgb-max-iter", type=int, default=220)
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=0,
        help="also run rolling-origin folds (5 is a good research check; adds compute)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xauusd",
        description="Leakage-resistant XAUUSD 30-minute direction research; no order execution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train and score a chronological holdout")
    train.add_argument("--csv", required=True, help="broker/vendor OHLCV CSV")
    train.add_argument("--model", default="artifacts/xauusd_30m.joblib")
    train.add_argument("--report", default="reports/xauusd_30m.json")
    _add_research_options(train)

    dry_run = subparsers.add_parser(
        "dry-run",
        help="fit only before a date and replay a later period with no orders",
    )
    dry_run.add_argument("--csv", required=True, help="real broker/vendor OHLCV CSV")
    dry_run.add_argument("--start", required=True, help="inclusive UTC start, e.g. 2026-07-01")
    dry_run.add_argument("--end", default=None, help="exclusive UTC end, e.g. 2026-08-01")
    dry_run.add_argument("--model", default="artifacts/xauusd_dry_run.joblib")
    dry_run.add_argument("--report", default="reports/xauusd_dry_run.json")
    _add_research_options(dry_run)

    predict = subparsers.add_parser("predict", help="score the latest complete bar; never places an order")
    predict.add_argument("--csv", required=True)
    predict.add_argument("--model", default="artifacts/xauusd_30m.joblib")
    predict.add_argument("--json", action="store_true", help="emit JSON only")

    download = subparsers.add_parser("download-yahoo", help="download a small real gold-futures experiment data set")
    download.add_argument("--symbol", default="GC=F", help="GC=F futures; not broker spot XAUUSD")
    download.add_argument("--timeframe", type=int, choices=(30, 60), default=30)
    download.add_argument("--out", required=True)

    dukascopy = subparsers.add_parser(
        "download-dukascopy",
        help="download exact public XAU/USD candles through dukascopy-node",
    )
    dukascopy.add_argument("--from", dest="date_from", required=True, help="inclusive UTC date, YYYY-MM-DD")
    dukascopy.add_argument("--to", dest="date_to", required=True, help="exclusive UTC date, YYYY-MM-DD")
    dukascopy.add_argument("--timeframe", type=int, choices=(30, 60), default=30)
    dukascopy.add_argument("--price-type", choices=("bid", "ask"), default="bid")
    dukascopy.add_argument("--out", required=True)

    sources = subparsers.add_parser("sources", help="show researched public data-source policy")
    sources.add_argument("--json", action="store_true", help="emit JSON")

    quality = subparsers.add_parser("quality", help="validate a CSV and print its data-quality report")
    quality.add_argument("--csv", required=True)
    quality.add_argument("--timeframe", type=int, choices=(30, 60), default=30)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "sources":
            records = source_matrix()
            if args.json:
                print(json.dumps(records, indent=2, default=str))
            else:
                for record in records:
                    exact = "EXACT SPOT" if record["exact_spot_xauusd"] else "NOT SPOT"
                    print(f"{record['name']}: {exact} | {record['role']} | {record['url']}")
                    print(f"  limits: {record['limitations']}")
            return 0

        if args.command == "quality":
            _, report = load_bars(
                args.csv,
                timeframe_minutes=args.timeframe,
                return_report=True,
            )
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0

        if args.command == "download-yahoo":
            bars = download_yahoo_intraday(args.symbol, timeframe_minutes=args.timeframe)
            destination = Path(args.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            bars.reset_index().to_csv(destination, index=False)
            print(f"wrote {len(bars)} {args.timeframe}-minute bars to {destination}")
            return 0

        if args.command == "download-dukascopy":
            bars = download_dukascopy_xauusd(
                args.out,
                date_from=args.date_from,
                date_to=args.date_to,
                timeframe_minutes=args.timeframe,
                price_type=args.price_type,
            )
            print(
                f"wrote {len(bars)} exact XAU/USD {args.timeframe}-minute "
                f"{args.price_type} bars to {args.out}"
            )
            return 0

        if args.command == "train":
            config = _config_from_args(args)
            bars, data_report = load_bars(
                args.csv,
                timeframe_minutes=config.timeframe_minutes,
                return_report=True,
            )
            result = run_research(bars, config)
            result.report["data_quality"] = data_report.to_dict()
            if args.walk_forward_folds:
                if args.walk_forward_folds < 2:
                    raise ValueError("--walk-forward-folds must be 0 or at least 2")
                from .walk_forward import run_walk_forward
                result.report["walk_forward"] = run_walk_forward(
                    bars, config, n_splits=args.walk_forward_folds
                )
            result.save(args.model)
            _write_json(args.report, result.report)
            print(json.dumps({
                "status": result.report["status"],
                "model": str(args.model),
                "report": str(args.report),
                "validation": result.report.get("validation", {}),
                "test": result.report.get("test", {}),
                "certification": result.report.get("certification", {}),
            }, indent=2, default=str))
            return 0 if result.report["status"] == "target_met_on_holdout" else 2

        if args.command == "dry-run":
            config = _config_from_args(args)
            bars, data_report = load_bars(
                args.csv,
                timeframe_minutes=config.timeframe_minutes,
                return_report=True,
            )
            result = run_forward_dry_run(bars, args.start, args.end, config)
            result.report["data_quality"] = data_report.to_dict()
            result.save(args.model)
            _write_json(args.report, result.report)
            print(json.dumps({
                "status": result.report["status"],
                "timeframe_minutes": config.timeframe_minutes,
                "forward_period": result.report.get("forward_period", {}),
                "model": str(args.model),
                "report": str(args.report),
                "forward_dry_run": result.report.get("forward_dry_run", {}),
                "certification": result.report.get("certification", {}),
            }, indent=2, default=str))
            return 0 if result.report["status"] == "target_met_on_forward_dry_run" else 2

        if args.command == "predict":
            # The artifact records whether it was trained at 30m or 1h; the
            # helper performs the correct normalization after loading it.
            bars = load_bars(args.csv, resample=False)
            signal = predict_latest_from_bars(bars, args.model)
            if args.json:
                print(json.dumps(signal, indent=2, default=str))
            else:
                print(
                    f"{signal['timestamp']} | {signal['direction']} | "
                    f"p_up={signal['p_up']:.3f} confidence={signal['confidence']:.3f} "
                    f"agreement={signal['agreement']:.3f} take_signal={signal['take_signal']}"
                )
            return 0
    except (DataError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 1
