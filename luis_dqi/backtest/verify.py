"""Read-only checks of a complete reference run, optionally against another run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .policies import POLICIES
from .runner import ENGINE_COMMIT, REFERENCE_INPUTS, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_run(path: Path, against: Path | None = None) -> dict:
    path = path.expanduser().resolve()
    manifest = json.loads((path / "run_manifest.json").read_text())
    require(manifest["status"] == "completed", "Run is not completed")
    metrics = manifest["metrics"]
    if "metrics" in metrics:
        metrics = metrics["metrics"]
    require(manifest["contract"]["qbacktest_commit"] == ENGINE_COMMIT, "Engine commit differs")
    require({key: manifest["inputs"][key]["sha256"] for key in REFERENCE_INPUTS} == REFERENCE_INPUTS, "Input hashes differ")
    for key in REFERENCE_INPUTS:
        source = manifest["inputs"][key]
        require(sha256(Path(source["path"])) == source["sha256"], f"Physical input differs: {key}")
    engine_prices = manifest["inputs"]["engine_prices"]
    require(sha256(path / "engine_prices.parquet") == engine_prices["sha256"], "Engine price subset differs")
    engine_dir = Path(manifest["contract"]["qbacktest_import_path"]).parent
    for filename, fingerprint in manifest["contract"]["engine_sources"].items():
        require(not Path(filename).is_absolute() and ".." not in Path(filename).parts, "Invalid engine source path")
        require(sha256(engine_dir / filename) == fingerprint, f"Current engine source differs: {filename}")
    daily = pd.read_csv(path / "daily.csv", parse_dates=["date"])
    nav_file = pd.read_csv(path / "daily_nav.csv", parse_dates=["date"])
    trades = pd.read_csv(path / "trades.csv")
    execution = pd.read_csv(path / "execution.csv")
    audit = pd.read_csv(path / "rebalance_audit.csv", parse_dates=["execution_date", "price_cutoff_date", "score_date"])
    require(len(daily) == len(nav_file) == 2978, "Expected all 2,978 sessions")
    require(daily.date.is_monotonic_increasing and not daily.date.duplicated().any(), "Dates are not unique and chronological")
    require((str(daily.date.iloc[0].date()), str(daily.date.iloc[-1].date())) == ("2014-03-03", "2025-12-31"), "Period differs")
    require((daily.date == nav_file.date).all() and np.array_equal(daily.nav, nav_file.nav), "NAV files differ")
    require((execution.rejected_trades == 0).all(), "Rejected native orders")
    require(json.loads((path / "strategy_errors.json").read_text()) == [], "Native strategy errors")
    require(not audit.execution_date.duplicated().any(), "Duplicate rebalance dates")
    require((audit.price_cutoff_date < audit.execution_date).all() and (audit.score_date < audit.execution_date).all(), "Noncausal decision")
    monthly = int((audit.trigger_reason == "monthly_scheduled").sum())
    require(monthly == 142, "Expected all 142 monthly events")
    params = manifest["parameters"]
    require((params["fee_bps"], params["borrow_rate"], params["recovery"], params["initial_cash"]) == (10., .02, 1., 1_000_000.), "Reference costs/cash differ")
    policy = manifest["rebalance_policy"].get("policy", manifest["rebalance_policy"].get("mode"))
    if policy in POLICIES and POLICIES[policy]["mode"] != "monthly":
        require((params["n_positions"], params["scenario"]) == (30, "p75"), "Event mandate differs")
    short_cap = manifest["rebalance_policy"].get("short_cap", manifest.get("markowitz_adapter", {}).get("short_cap"))
    if short_cap is None:
        expression = manifest["contract"]["target_short"]
        short_cap = 0. if params["scenario"] == "lo" else float(expression.split("*")[0])
    threshold = None if params["scenario"] == "lo" else float(params["scenario"][1:])
    if policy in POLICIES and POLICIES[policy]["mode"] != "monthly":
        require(short_cap == .30, "Event short cap differs")
    target = np.zeros(len(audit)) if threshold is None else short_cap * np.clip((audit.stress_value.to_numpy() - threshold) / (100. - threshold), 0., 1.)
    target[~np.isfinite(target)] = 0.
    require(np.allclose(target, audit.target_short_fraction, atol=1e-12, rtol=0), "Short formula differs")
    wealth = np.r_[1_000_000., daily.nav.to_numpy(dtype=float)]
    require(np.isfinite(wealth).all() and (wealth > 0).all(), "Invalid NAV")
    returns = wealth[1:] / wealth[:-1] - 1.
    years = (daily.date.iloc[-1] - pd.Timestamp("2014-02-28")).days / 365.25
    expected = {
        "sharpe_standard":float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
        "cagr":float((wealth[-1] / wealth[0]) ** (1 / years) - 1),
        "annual_volatility":float(returns.std(ddof=1) * np.sqrt(252)),
        "max_drawdown":float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "final_nav":float(wealth[-1]), "total_fees":float(trades.fee.sum()),
        "total_borrow":float(-(daily.interests_settled.iloc[-1] + daily.interests.iloc[-1])),
        "trades":len(trades),
    }
    require(metrics["initial_nav_date"] == "2014-02-28", "Initial NAV date differs")
    for key, value in expected.items():
        require(np.isclose(value, metrics[key], atol=1e-9, rtol=1e-12), f"Metric differs from raw files: {key}")
    require(np.allclose(returns, daily.daily_return, atol=1e-13, rtol=0), "Daily returns differ from NAV")
    require(np.isclose(daily.cost.sum(), expected["total_fees"], atol=1e-8, rtol=1e-12), "Fee accounting differs")
    require(np.isclose(daily.borrow_fee.sum(), expected["total_borrow"], atol=1e-8, rtol=1e-12), "Borrow accounting differs")
    code = manifest.get("shared_adapter_sha256", {})
    for filename, fingerprint in code.items():
        require(Path(filename).name == filename, "Invalid code manifest path")
        copied = filename.removesuffix(".py") + ".executed.py" if filename.endswith(".py") else filename
        require(sha256(path / copied) == fingerprint, f"Executed code hash differs: {filename}")
    comparison = {}
    if against is not None:
        against = against.expanduser().resolve()
        for filename in ("daily.csv", "daily_nav.csv", "trades.csv", "weights.csv", "target_weights.csv", "rebalance_audit.csv"):
            actual, reference = pd.read_csv(path / filename), pd.read_csv(against / filename)
            try:
                pd.testing.assert_frame_equal(actual, reference, check_exact=True)
            except AssertionError as exc:
                raise ValueError(f"Full-series comparison differs: {filename}") from exc
            comparison[filename] = "exact"
    return {"status":"verified", "policy":policy, "sessions":len(daily), "monthly_events":monthly,
            "events":len(audit), "extra_events":len(audit)-monthly, "metrics":expected,
            "full_series_comparison":comparison, "input_hashes":REFERENCE_INPUTS,
            "engine_commit":ENGINE_COMMIT}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--against", type=Path, help="Compare all complete series exactly, not rounded Sharpe")
    args = parser.parse_args(argv)
    try:
        result = verify_run(args.run, args.against)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
