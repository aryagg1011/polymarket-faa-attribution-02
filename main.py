"""Run the causal quantile forecaster on MakeFun program/v1 case files."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

# Keep imports local to the submitted repository, independent of the grader's
# current working directory and Python module search path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import forecast_market

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
OUTPUT_LIMIT = 1_048_576
CASE_SCHEMA = "makefun/polymarket-case/v1"
CASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# The forecast pools each market's own causal history with peer histories.
# These settings are shared across every market and checkpoint.
PARAMS = {
    "step": 30,
    "price_band": 0.15,
    "half_life": 720.0,
    "empirical_blend": 0.80,
    "peer_blend": 0.50,
    "max_origins": 4,
}


def _object(value: Any, where: str) -> dict:
    """Require a JSON object and attach the input location to any error."""
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _number(value: Any) -> bool:
    """JSON booleans are integers in Python, but are never valid prices."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _case_name(value: Any) -> str:
    if not isinstance(value, str) or CASE_NAME.fullmatch(value) is None:
        raise ValueError("case must be a short safe filename token")
    return value


def validate_case(case: Any) -> tuple[str, int, list[dict]]:
    """Validate the input before doing the more expensive forecasts."""
    case = _object(case, "case")
    if case.get("schema") != CASE_SCHEMA:
        raise ValueError("unsupported case schema")
    name = _case_name(case.get("case"))

    horizon = case.get("horizon_minutes")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon_minutes must be a positive integer")

    quantiles = case.get("quantiles")
    if not isinstance(quantiles, list) or len(quantiles) != len(QUANTILES):
        raise ValueError("quantiles must contain the five protocol levels")
    if any(not _number(q) for q in quantiles):
        raise ValueError("quantiles must be finite numbers")
    if tuple(float(q) for q in quantiles) != QUANTILES:
        raise ValueError("quantile levels do not match the entry protocol")

    raw_markets = case.get("markets")
    if not isinstance(raw_markets, list) or not raw_markets:
        raise ValueError("case must contain at least one market")

    markets: list[dict] = []
    seen_ids: set[str] = set()
    for index, raw_market in enumerate(raw_markets):
        market = _object(raw_market, f"markets[{index}]")
        market_id = market.get("id")
        if not isinstance(market_id, str) or not market_id:
            raise ValueError(f"markets[{index}].id must be a non-empty string")
        if market_id in seen_ids:
            raise ValueError(f"duplicate market id {market_id}")
        seen_ids.add(market_id)

        prices = market.get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"market {market_id} has no price history")
        for price_index, price in enumerate(prices):
            if not _number(price) or not 0.0 <= price <= 1.0:
                raise ValueError(
                    f"market {market_id} price {price_index} is outside [0, 1]"
                )
        markets.append(market)

    return name, horizon, markets


def _make_market_row(
    market: dict,
    horizon: int,
    histories: list[list[float]],
) -> dict:
    """Forecast one market using only histories provided in this case."""
    forecast = forecast_market(
        market["prices"],
        horizon,
        model="hybrid",
        peer_histories=histories,
        **PARAMS,
    )
    return {"id": market["id"], "q": forecast}


def make_output(case: Any) -> dict:
    """Build one result while preserving complete market coverage."""
    _name, horizon, markets = validate_case(case)
    histories = [market["prices"] for market in markets]
    forecasts = [
        _make_market_row(market, horizon, histories)
        for market in markets
    ]
    result = {"forecasts": forecasts}
    validate_output(result, markets, horizon)
    return result


def validate_output(result: Any, markets: list[dict], horizon: int) -> None:
    """Reject malformed forecasts before they can be written for scoring."""
    result = _object(result, "output")
    if set(result) != {"forecasts"}:
        raise ValueError("output must contain exactly the forecasts key")
    rows = result["forecasts"]
    if not isinstance(rows, list) or len(rows) != len(markets):
        raise ValueError("output must include one row for every market")

    expected_ids = {market["id"] for market in markets}
    found_ids: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"forecasts[{index}]")
        if set(row) != {"id", "q"}:
            raise ValueError(f"forecasts[{index}] must contain only id and q")
        market_id = row["id"]
        if not isinstance(market_id, str) or market_id not in expected_ids:
            raise ValueError(f"forecast row {index} names an unknown market")
        if market_id in found_ids:
            raise ValueError(f"duplicate forecast row for {market_id}")
        found_ids.add(market_id)

        values_by_minute = row["q"]
        if not isinstance(values_by_minute, list) or len(values_by_minute) != horizon:
            raise ValueError(f"horizon mismatch for market {market_id}")
        for minute, values in enumerate(values_by_minute):
            if not isinstance(values, list) or len(values) != len(QUANTILES):
                raise ValueError(
                    f"market {market_id} minute {minute} needs five quantiles"
                )
            if any(not _number(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(
                    f"market {market_id} minute {minute} has an invalid quantile"
                )
            if any(left > right for left, right in zip(values, values[1:])):
                raise ValueError(
                    f"market {market_id} minute {minute} has unordered quantiles"
                )

    if found_ids != expected_ids:
        raise ValueError("output market coverage does not match the input")


def _read_case(source: Path) -> Any:
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read case {source.name}: {exc}") from exc


def _serialize(result: dict, case_name: str) -> bytes:
    try:
        encoded = json.dumps(
            result,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot serialize case {case_name}: {exc}") from exc
    if len(encoded) > OUTPUT_LIMIT:
        raise ValueError(f"output too large for case {case_name}")
    return encoded


def _write_result(output_dir: Path, case_name: str, encoded: bytes) -> None:
    """Replace a complete result file atomically, never leaving partial JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{case_name}.out"
    temporary = output_dir / f".{case_name}.out.tmp"
    try:
        temporary.write_bytes(encoded)
        temporary.replace(target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"cannot write output for {case_name}: {exc}") from exc


def run_cases(input_dir: Path, output_dir: Path) -> None:
    """Process every supplied case in stable filename order."""
    sources = sorted(input_dir.glob("*.in"), key=lambda path: path.name)
    if not sources:
        raise ValueError(f"no case files found in {input_dir}")
    for source in sources:
        case = _read_case(source)
        case_name, horizon, markets = validate_case(case)
        result = make_output(case)
        validate_output(result, markets, horizon)
        encoded = _serialize(result, case_name)
        _write_result(output_dir, case_name, encoded)


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    run_cases(input_dir, output_dir)


if __name__ == "__main__":
    main()

