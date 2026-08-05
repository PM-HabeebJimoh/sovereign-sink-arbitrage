# XAU30 — research-grade XAUUSD direction model

This repository has been rebuilt around one problem: **predict the direction of the next XAUUSD 30-minute bar without leaking future information**.

It is a research and signal-generation system. It does **not** place orders, hold keys, promise an 80% hit rate, or turn a model score into a trading instruction. No honest engineer can guarantee `>80%` out-of-sample hits before seeing a sufficiently large, broker-specific holdout.

## The critical review

The previous project was a crypto/DeFi arbitrage bot, not an XAUUSD forecasting system. Its stablecoin prices were used as a proxy for gold, its confidence was a hand-written formula rather than a validated probability, and its README reported trading performance without a reproducible dataset or leakage-safe test. It also contained live credentials. That architecture could not answer the requested question.

The replacement is deliberately **fail-closed**:

- the target is execution-aligned: signal after bar `t` closes, enter at bar `t+1` open, score the next bar's open-to-close direction;
- the label has a transparent dead zone so micro-noise is not relabelled as a confident move;
- features are causal and include price action, volatility, trend, 2h/4h/daily context, UTC sessions and optional aligned DXY/rates/VIX/silver drivers;
- train, validation and test are chronological with an embargo; no random shuffle and no test-driven threshold tuning;
- a tree + linear ensemble is calibrated on an internal time-ordered tail;
- the policy can abstain (`NO_TRADE`); `hit_rate` is always reported with `signals` and `coverage`;
- costs and slippage are deducted in the holdout report;
- the code refuses to execute trades. Any live execution must be a separately reviewed project.

## “50x” / first-principles design

The useful part of an aggressive, first-principles approach is not copying a celebrity or using a larger neural network. It is removing the highest-leverage failure modes:

1. **First principles:** define exactly what “next-bar direction” and a hit mean before choosing a model.
2. **Delete before adding:** remove the old arbitrage/live-wallet stack and all unsupported performance claims.
3. **One source of truth:** broker-quality 30m OHLC data, UTC timestamps, data-quality report and immutable experiment JSON.
4. **Physics of the problem:** gold is regime-, session-, volatility- and macro-sensitive; multi-timeframe and optional cross-market features express those drivers without future values.
5. **Test the hardest thing:** the final chronological holdout is touched once. A target is only marked met when it clears 80% with meaningful holdout signal count.
6. **Fail fast:** malformed candles, missing columns, missing classes, insufficient history and missing features raise errors rather than being silently “fixed.”
7. **Compounding feedback loop:** save model, threshold, feature list, costs, data range, calibration bins and certification status so every experiment is reproducible.

The result is not a magical 20x guarantee. It is a system that makes a genuine 80% result difficult to fake.

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get real XAUUSD data

The repository can download exact public Dukascopy XAU/USD candles when Node.js/npm is available:

```bash
python -m xauusd download-dukascopy \
  --from 2024-01-01 --to 2026-08-01 \
  --timeframe 30 --price-type bid \
  --out data/xauusd_m30.csv

python -m xauusd download-dukascopy \
  --from 2024-01-01 --to 2026-08-01 \
  --timeframe 60 --price-type bid \
  --out data/xauusd_h1.csv
```

This uses the public `dukascopy-node` downloader and does not use synthetic prices. If the public feed is unavailable from the current network, export the same instrument/timeframe from your broker and use that CSV instead.

Use a broker or institutional export whenever possible. The CSV must contain:

```text
timestamp,open,high,low,close,volume
2024-01-02T00:00:00Z,2061.2,2062.0,2060.7,2061.8,1234
```

- timestamps are interpreted as **bar close times** and normalised to UTC;
- `volume` is optional; tick volume is accepted but is not treated as centralized volume;
- `dxy_close`, `us10y`, `vix_close`, `silver_close`, etc. may be included as aligned optional drivers. Only columns with explicit market prefixes are accepted;
- the data must be XAUUSD, not stablecoins. `GC=F` is gold futures and is only a quick Yahoo experiment, not a substitute for a broker's spot feed;
- supply at least 2 years of 30m history if possible. The default minimum warning is 500 labelled rows, not a claim of statistical sufficiency.

### 3. Inspect data quality

```bash
python -m xauusd quality --csv data/xauusd_30m.csv
```

### 4. Train and score one untouched holdout

```bash
python -m xauusd train \
  --csv data/xauusd_30m.csv \
  --model artifacts/xauusd_30m.joblib \
  --report reports/xauusd_30m.json
```

Exit code `0` means the selected policy met 80% on the final holdout, had meaningful signal count, a 95% Wilson lower bound at or above 80%, and positive net return under the configured cost assumption. Exit code `2` means the hit-rate target, statistical support, coverage or cost-aware expectancy failed; inspect the report rather than lowering the standard.

Useful experiment controls:

```bash
# Predict a 2-bar (1 hour) horizon, with a 10 bp fixed dead zone
python -m xauusd train --csv data/xauusd_30m.csv \
  --horizon 2 --label-threshold-bps 10

# Use measured broker costs, not optimistic defaults
python -m xauusd train --csv data/xauusd_30m.csv \
  --round-trip-cost-bps 7 --slippage-bps 2
```

### 5. Forward dry-run a real historical month

The `dry-run` command fits only on data before the requested start date. It then replays the forward period with no broker connection or order placement. This is the correct way to test July 2026 without using July outcomes to tune the model:

```bash
# 30-minute spot XAUUSD broker/Dukascopy CSV
python -m xauusd dry-run \
  --csv data/xauusd_m30.csv --timeframe 30 \
  --start 2026-07-01 --end 2026-08-01 \
  --model reports/xauusd_m30.joblib \
  --report reports/xauusd_m30.json

# 1-hour spot XAUUSD CSV
python -m xauusd dry-run \
  --csv data/xauusd_h1.csv --timeframe 60 \
  --start 2026-07-01 --end 2026-08-01 \
  --model reports/xauusd_h1.joblib \
  --report reports/xauusd_h1.json
```

The forward report includes exact data range, signal count, coverage, hit rate, Wilson confidence bound, cost-adjusted returns and a `status`. It is a dry-run even when the status is successful; it never places orders.

### 6. Generate a latest-bar research signal

```bash
python -m xauusd predict \
  --csv data/xauusd_30m.csv \
  --model artifacts/xauusd_30m.joblib \
  --json
```

Possible directions are `UP`, `DOWN` and `NO_TRADE`. `NO_TRADE` is a designed outcome, not a bug. The command only reads a CSV and prints a signal; it never connects to a broker.

### Optional quick Yahoo experiment

Yahoo's intraday retention is limited and `GC=F` is COMEX gold futures, not spot XAUUSD:

```bash
python -m xauusd download-yahoo --symbol 'GC=F' --timeframe 30 --out /tmp/gold_30m.csv
python -m xauusd download-yahoo --symbol 'GC=F' --timeframe 60 --out /tmp/gold_1h.csv
```

Do not use this small, vendor-mismatched sample to certify a production XAUUSD hit rate. For the requested July 2026 dry-run, use real broker/Dukascopy spot XAUUSD candles instead of substituting `GC=F`.

## What is measured

The JSON report contains:

- all-bar accuracy, Brier score, log loss and calibration bins;
- selected `hit_rate`, 95% Wilson lower bound, `signals`, `coverage`, wins and losses;
- average/net selected return after configured costs, profit factor and max drawdown;
- validation threshold sweep and whether validation met its target;
- untouched holdout result and an explicit `target_met_out_of_sample` flag;
- data range, gaps, duplicates, feature count and the exact config.

An 80% number with 3 signals is not equivalent to 80% with 3,000 signals. Treat high hit rate / tiny coverage as a selective alert policy, not as an always-on directional model. Also compare against a no-skill baseline, a simple momentum baseline and a cost-aware strategy return.

## Public data research

See [`DATA_SOURCES.md`](DATA_SOURCES.md) for the researched public-source matrix, why futures proxies are rejected, and the exact provenance rules for the July 2026 dry-run. The short version is: Dukascopy is the primary exact spot XAU/USD intraday source; Yahoo `GC=F` is real but futures and must not be used as spot evidence.

## Layout

```text
xauusd/
  data.py          CSV validation, 30m normalisation, optional Yahoo download
  features.py      causal features and execution-aligned labels
  model.py         calibrated ensemble, abstention and model artifact I/O
  evaluation.py    purged splits, threshold selection and cost-aware metrics
  research.py      train → validate → refit → untouched holdout
  cli.py           quality, train, predict and download commands
  config.py        explicit experiment defaults
xauusd_direction.py  convenience CLI entry point
tests/                causality and metric tests
```

Run tests with:

```bash
pytest -q
```

## Security and deployment

The old tracked `.env` contained credentials. They have been removed from the working tree, but anything that was ever committed must be treated as compromised: **rotate the private key, Telegram token and API keys outside this repository**. `.env` is now ignored; use `.env.example` for non-secret local settings.

This repository has no live-trading Procfile by design. Do not attach the research signal directly to an account. Before any paper or live integration, add a separately reviewed execution layer with broker-specific spread checks, order acknowledgements, position reconciliation, hard daily limits, kill switches, audit logs and independent monitoring.

## Limitations

Gold markets change. A calibrated score is not a probability guarantee in a new regime; 30-minute bars can be vendor-dependent; economic releases create gaps and slippage; and a selective 80% hit rate can coexist with negative expectancy. Walk-forward validation across multiple years, event windows and brokers is required before drawing any conclusion.
