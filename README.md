# FAA attribution forecast

A dependency-free Python 3.13 entry for MakeFun bounty `polymarket-faa-attribution-02`.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The entry reads every `*.in` in the input directory and writes `<case>.out` JSON files. In the grader, the directory contains the current case. It uses only the Python standard library and the supplied price history; it makes no network calls.

## Forecast method

For each horizon minute, the program pools empirical price changes from the last four available 30-minute-spaced historical origins. Origins are weighted by age (720-minute half-life) and distance from the current price (Gaussian kernel, 0.15 price bandwidth). It blends the empirical quantiles 80% with the contest random-walk Gaussian prior (20%), then blends local and within-case peer-market empirical distributions equally. Quantiles are clipped to `[0,1]`, sorted, and emitted at the five required levels.

## Local causal evaluation

Run `python3 backtest.py --input-dir /path/to/kit/input --stride 1` with the MakeFun kit's practice inputs. The backtest uses each next checkpoint's history as labels, so it does not use a case's future prices as that case's inputs. Use `--stride 10` for quick screening only.

The exact five-transition evaluation available here (p00–p04; p05's full label tail was unavailable) scored **0.360483 pp** for the four-origin candidate, against **0.392518 pp** for the official random-walk formula on the same five transitions. This is not the official six-case practice score; its sample differs from the published six-case house score of 0.363589 pp. The bounty states that practice is from a different event and does not predict final performance.

## Constraints and license

The runtime entry uses only the standard library. Source is released under the MIT License; see `LICENSE`.


