"""Causal rolling-origin backtest on the public dev cases in the MakeFun kit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import forecast_market


ROOT = Path(__file__).resolve().parent
KIT_INPUT = ROOT / "input"
OUT = ROOT / "backtest-results.csv"


def minute_index(iso: str, start: str) -> int:
    a = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return int((a - s).total_seconds() // 60)


def loss(y: float, f: float, q: float) -> float:
    d = y - f
    return max(q * d, (q - 1.0) * d) * 100.0


def load_cases():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(KIT_INPUT.glob("p*.in"))]


def configs():
    yield "baseline", "baseline", {}
    yield "fixed-s0.60", "baseline", {"sigma_scale": 0.60}
    yield "adaptive-s0.20", "adaptive", {"sigma_scale": 0.20}
    for cap in (1, 2, 3, 4, 6):
        for blend in (0.60, 0.80):
            name = f"emp-n{cap}-b0.15-hl720-mix{blend:g}-peer0.5"
            yield name, "hybrid", {
                "step": 30,
                "price_band": 0.15,
                "half_life": 720.0,
                "empirical_blend": blend,
                "peer_blend": 0.5,
                "max_origins": cap,
            }


def run_backtest(selected=None, stride=1):
    cases = load_cases()
    names = {name: (model, params) for name, model, params in configs()}
    if selected:
        names = {name: value for name, value in names.items() if name in selected}
    totals = {name: {"loss": 0.0, "items": 0, "abs": 0.0, "covered": 0, "rows": 0} for name in names}
    per_case = {name: {} for name in names}
    rows_out = []

    # A case's labels are read only from the next checkpoint's history.
    for ci in range(min(len(cases) - 1, 5)):
        case, future = cases[ci], cases[ci + 1]
        future_by_id = {m["id"]: m for m in future["markets"]}
        peers = [m["prices"] for m in case["markets"]]
        for name, (model, params) in names.items():
            c_loss = c_items = c_abs = c_covered = c_rows = 0
            for market in case["markets"]:
                later = future_by_id.get(market["id"])
                if later is None:
                    continue
                offset = minute_index(case["at"], later["start"])
                actual = later["prices"][offset + 1 : offset + case["horizon_minutes"] + 1]
                if len(actual) != case["horizon_minutes"]:
                    continue
                pred = forecast_market(
                    market["prices"], case["horizon_minutes"], model=model,
                    peer_histories=peers, calc_stride=stride, **params,
                )
                sampled_actual = actual[stride - 1 :: stride]
                for y, row in zip(sampled_actual, pred):
                    for q, f in zip((0.05, 0.25, 0.50, 0.75, 0.95), row):
                        c_loss += loss(y, f, q)
                        c_items += 1
                    c_abs += abs(y - row[2]) * 100.0
                    c_covered += int(row[0] <= y <= row[4])
                    c_rows += 1
            totals[name]["loss"] += c_loss
            totals[name]["items"] += c_items
            totals[name]["abs"] += c_abs
            totals[name]["covered"] += c_covered
            totals[name]["rows"] += c_rows
            per_case[name][case["case"]] = (c_loss / c_items if c_items else float("nan"), c_rows)

    for name, total in totals.items():
        item = {
            "name": name,
            "pinball_pp": total["loss"] / total["items"] if total["items"] else float("nan"),
            "median_mae_pp": total["abs"] / total["rows"] if total["rows"] else float("nan"),
            "coverage_90": total["covered"] / total["rows"] if total["rows"] else float("nan"),
            "rows": total["rows"],
            "cases": " | ".join(f"{c}:{v[0]:.6f}/{v[1]}" for c, v in per_case[name].items()),
        }
        rows_out.append(item)
    rows_out.sort(key=lambda r: r["pinball_pp"])
    return rows_out


def main():
    global KIT_INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="restrict to named configurations")
    parser.add_argument("--input-dir", type=Path, default=KIT_INPUT, help="directory containing practice inputs")
    parser.add_argument("--stride", type=int, default=1, help="score every Nth horizon minute (screening only; default 1 is exact)")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    KIT_INPUT = args.input_dir
    results = run_backtest(set(args.only) if args.only else None, stride=max(1, args.stride))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for r in results[:40]:
        print(f"{r['name']:<36} loss={r['pinball_pp']:.6f} median={r['median_mae_pp']:.6f} cov90={r['coverage_90']:.4f} rows={r['rows']}")
    print(f"saved {len(results)} configs to {args.output}")


if __name__ == "__main__":
    main()



