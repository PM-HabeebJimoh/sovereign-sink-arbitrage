# Public-data review for the XAUUSD July 2026 dry-run

**Research date:** 2026-08-05 (UTC)

## Decision

For a strict XAU/USD spot experiment at 30 minutes and 1 hour, the primary
source is the public Dukascopy historical feed accessed through
[`dukascopy-node`](https://www.dukascopy-node.app/instrument/xauusd). It exposes
`m30` and `h1` aggregation, bid/ask selection and UTC timestamps. It is the
only source in this review that is both explicitly spot XAU/USD and designed
for downloadable intraday history.

The downloader is now built into the repository:

```bash
python -m xauusd download-dukascopy \
  --from 2024-01-01 --to 2026-08-05 \
  --timeframe 30 --price-type bid \
  --out data/xauusd_m30.csv

python -m xauusd download-dukascopy \
  --from 2024-01-01 --to 2026-08-05 \
  --timeframe 60 --price-type bid \
  --out data/xauusd_h1.csv
```

The public feed is not a single global “official XAUUSD price”: OTC spot
quotes are vendor/broker specific. The source, side (bid/ask), timestamp
convention and file hash must therefore be stored with every experiment.

## Source matrix

| Source | Exact spot? | 30m/1h July 2026? | Role | Decision |
|---|---:|---:|---|---|
| [Dukascopy XAUUSD](https://www.dukascopy-node.app/instrument/xauusd) | Yes | Yes | Primary intraday OHLC | Use |
| [Twelve Data XAU/USD](https://twelvedata.com/markets/300755/commodity/xau-usd/historical-data) | Yes | API-dependent | Reference and secondary validation | Use only with an API plan/key and captured metadata |
| [XAUS API](https://xaus.com/api/) | Yes | Recent intraday only | Current quote/daily sanity check | Cannot reconstruct the full July month |
| Yahoo `GC=F` | No: COMEX futures | Yes, vendor retention limited | Futures diagnostic | Never mix into spot certification |
| Kaggle/Hugging Face archives | Uncertain | Often stale before July 2026 | Reproducibility/legacy research | Do not use for July certification without checking version/date |
| Myfxbook/Investing/Barchart | Mostly daily/manual or paid intraday | Source/plan dependent | Visual/reference | Not an automated primary source |
| Philadelphia XAU index datasets | No | Sometimes | Equity index data | Not XAU/USD |

The CLI exposes this policy with:

```bash
python -m xauusd sources
python -m xauusd sources --json
```

## Why the earlier online run stopped

The repository attempted the public Dukascopy downloader in the sandbox. The
source itself is public, but the sandbox's outbound connection to the binary
Dukascopy datafeed returned `fetch failed`. Yahoo was deliberately not used as
a substitute because `GC=F` is futures, not spot XAU/USD. A result produced by
mixing those instruments would be precise-looking but invalid for the user's
request.

This is a network-transport limitation, not a claim that the data does not
exist. Running the same command on a normal local/CI network, or attaching the
resulting CSVs, enables the exact July dry-run.

## Data acquisition is not the same as model evidence

A source being public does not make its data interchangeable or its hit rate
credible. Before accepting a run, the pipeline must check:

1. `instrument == XAUUSD`, not a futures or index proxy;
2. bid/ask side and UTC timestamps are recorded;
3. data ends after 2026-07-31 so the final July labels are observable;
4. no bars are fabricated over weekend/daily breaks;
5. no July row influences model fitting or threshold selection;
6. raw CSV SHA-256, row count, start/end and quality report are archived;
7. the 30m and 1h runs are evaluated separately;
8. costs use measured spread/slippage, not an arbitrary zero-cost assumption.

## 50x first-principles research architecture

The high-leverage approach is not “add a bigger neural network.” It is a
closed-loop evidence system:

- **Delete the wrong problem:** never train XAUUSD on stablecoin or crypto
  arbitrage data.
- **Instrument truth:** use the exact spot instrument and a declared quote
  side; maintain separate 30m and 1h artifacts.
- **Causal information set:** merge macro features only at their publication
  timestamp; never use a revised series value that was unavailable at the
  prediction time.
- **Regime decomposition:** score London/NY overlap, Asia, news windows,
  volatility quartiles and trend/range regimes independently.
- **Selective prediction:** require model agreement and abstain when the
  expected edge does not clear spread plus slippage.
- **Nested validation:** choose thresholds before the July forward period and
  use rolling-origin folds across multiple prior regimes.
- **Statistical gate:** require a meaningful number of forward signals, a 95%
  Wilson lower bound, positive net return after costs and comparison against
  majority/momentum baselines.
- **Reproducibility:** each result includes source, side, timeframe, dates,
  file hash, config, feature list and report status.

No public data source can guarantee an 80% future hit rate. A genuine result
must survive the untouched July forward period, costs, multiple timeframes,
and a later month that was not selected after seeing July.
