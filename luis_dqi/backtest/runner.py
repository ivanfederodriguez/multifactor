"""Portable CLI for full Luis DQI runs on a separate, pinned qbacktest checkout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import traceback
import _thread

from .policies import POLICIES

ENGINE_COMMIT = "a9f0edbc1d0fdc3e318ca9e991027b4131fcf301"
REFERENCE_INPUTS = {
    "scores": "1837df55bed8eb50b05399a12e81a0a9c7a98308d66d5cd46b259a4722720996",
    "prices": "4bd8a70cd04ae5e6a984fee0a7f525474222071b0172b92ec256adf622308854",
    "macro": "7a8d2f2e0c79b8dfcbd4e2626630b8a0c8cfd6e92a536dedda821fb986bb5e42",
}
PACKAGE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_engine(root: Path) -> Path:
    """Resolve authorized source, reject a different commit or modified engine."""
    root = root.expanduser().resolve()
    if not (root / "src/qbacktest/__init__.py").is_file():
        raise ValueError("--qbacktest-root must be the authorized Git checkout, containing src/qbacktest")
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        changed = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src/qbacktest"], text=True)
        new_sources = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", "src/qbacktest"], text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError("Cannot verify the qbacktest checkout") from exc
    if head != ENGINE_COMMIT:
        raise ValueError(f"qbacktest must be pinned at {ENGINE_COMMIT}; found {head}")
    if changed or any(name.endswith(".py") for name in new_sources.splitlines()):
        raise ValueError("qbacktest source differs from the pinned commit")
    loaded = sys.modules.get("qbacktest")
    if loaded is not None and Path(loaded.__file__).resolve() != root / "src/qbacktest/__init__.py":
        raise ValueError("A different qbacktest has already been imported; use a fresh process")
    return root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qbacktest-root", type=Path, required=True)
    parser.add_argument("--scores-path", type=Path, required=True)
    parser.add_argument("--prices-path", type=Path, required=True)
    parser.add_argument("--macro-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=tuple(POLICIES), default="monthly")
    parser.add_argument("--scenario", choices=("lo", "p50", "p75", "p90"), default="p75")
    parser.add_argument("--n-positions", type=int, default=30)
    parser.add_argument("--short-cap", type=float, default=.30)
    parser.add_argument("--fee-bps", type=float, default=10.)
    parser.add_argument("--borrow-rate", type=float, default=.02)
    parser.add_argument("--recovery", type=float, default=1.)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.)
    parser.add_argument("--start", default="2014-03-03")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--deadline-utc", help="Optional timezone-aware execution cutoff; no stale default")
    parser.add_argument("--reference", action="store_true", help="Require frozen Base B inputs and the complete 2,978-session contract")
    return parser.parse_args(argv)


def validate_args(args):
    import math
    numeric = (args.short_cap, args.fee_bps, args.borrow_rate, args.recovery, args.initial_cash)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Portfolio and cost parameters must be finite")
    if args.n_positions < 1 or not 0 <= args.short_cap <= 1 or min(args.fee_bps, args.borrow_rate, args.recovery) < 0 or args.initial_cash <= 0:
        raise ValueError("Invalid portfolio or cost parameters")
    if (args.scenario == "lo") != (args.short_cap == 0):
        raise ValueError("Long-only requires --short-cap 0; short scenarios require a positive cap")
    if POLICIES[args.policy]["mode"] != "monthly" and (
        args.scenario != "p75" or args.n_positions != 30 or args.short_cap != .30
    ):
        raise ValueError("Event rebalance policies keep Top 30, P75 and short cap 30% fixed")
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    if start.tzinfo is not None or end.tzinfo is not None or start > end:
        raise ValueError("--start/--end must be ordered, timezone-naive dates")
    if args.reference and (args.start, args.end, args.fee_bps, args.borrow_rate, args.recovery, args.initial_cash) != (
        "2014-03-03", "2025-12-31", 10., .02, 1., 1_000_000.
    ):
        raise ValueError("--reference requires the complete frozen period and reference costs/cash")
    for key in REFERENCE_INPUTS:
        path = getattr(args, key + "_path").expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Missing {key} input: {path}")
        setattr(args, key + "_path", path)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError("Use a new --output-dir; existing runs are never overwritten")
    if args.deadline_utc:
        deadline = datetime.fromisoformat(args.deadline_utc)
        if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
            raise ValueError("--deadline-utc must be timezone-aware and in the future")


def run(args):
    """One isolated CLI run. Stateful frozen adapters are not thread-safe."""
    validate_args(args)
    args.qbacktest_root = check_engine(args.qbacktest_root)
    inputs_before = {key: sha256(getattr(args, key + "_path")) for key in REFERENCE_INPUTS}
    if args.reference and inputs_before != REFERENCE_INPUTS:
        raise ValueError("Input hashes differ from the frozen Base B reference")
    sys.path.insert(0, str(args.qbacktest_root / "src"))
    from . import _markowitz as A
    from ._strategy import StrictPolicyStrategy
    if Path(A.BASE.qbacktest.__file__).resolve() != args.qbacktest_root / "src/qbacktest/__init__.py":
        raise ValueError("Imported qbacktest does not match the authorized checkout")
    cfg = dict(POLICIES[args.policy])
    A.RUN_CONTEXT.clear()
    A.MKW_STATS.clear()
    A.RUN_CONTEXT.update(mode=cfg["mode"], scenario=args.scenario, policy=args.policy,
                         policy_config=cfg, threshold=None if args.scenario == "lo" else float(args.scenario[1:]),
                         short_cap=args.short_cap, short_enabled=args.scenario != "lo")
    A.BASE.make_decisions = A._patched_make_decisions
    A.BASE.FrozenMonthlyStrategy = StrictPolicyStrategy
    runner_args = argparse.Namespace(model="dqi", scenario=args.scenario, output_dir=str(args.output_dir),
        scores_path=str(args.scores_path), prices_path=str(args.prices_path), macro_path=str(args.macro_path),
        start=args.start, end=args.end, n_positions=args.n_positions, selection_fraction=None,
        borrow_rate=args.borrow_rate, recovery=args.recovery, fee_bps=args.fee_bps,
        initial_cash=args.initial_cash, deadline_utc=args.deadline_utc)
    timer = None
    if args.deadline_utc:
        remaining = (datetime.fromisoformat(args.deadline_utc) - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise ValueError("Execution cutoff reached during preparation")
        # KeyboardInterrupt escapes the engine's ordinary Exception handler on
        # both Windows and macOS. A deadline-aborted run is never completed.
        timer = threading.Timer(remaining, _thread.interrupt_main)
        timer.daemon = True
        timer.start()
    try:
        result = A.BASE.run_case(runner_args)
        metrics = result["metrics"]
        if result["status"] != "completed" or metrics["strategy_errors"] or metrics["rejected_trades"]:
            raise ValueError("The native run contains strategy errors or rejected trades")
        if args.reference and (metrics["start"], metrics["end"], metrics["sessions"]) != (
            "2014-03-03", "2025-12-31", 2978
        ):
            raise ValueError("Reference backtest does not cover all 2,978 sessions")
        if any(sha256(getattr(args, key + "_path")) != value for key, value in inputs_before.items()):
            raise ValueError("An input changed during the run")
        pd = A.pd
        audit = pd.read_csv(args.output_dir / "rebalance_audit.csv")
        counts = audit.trigger_reason.value_counts().to_dict()
        scheduled = int(counts.get("monthly_scheduled", 0))
        if args.reference and scheduled != 142:
            raise ValueError("Reference run must retain all 142 monthly rebalance dates")
        metrics.update(rebalance_policy=args.policy, total_rebalance_events=len(audit),
                       offcycle_rebalance_events=len(audit)-scheduled, rebalance_trigger_counts=counts)
        pd.DataFrame(A.MKW_STATS).to_csv(args.output_dir / "markowitz_weight_stats.csv", index=False)
        manifest = json.loads((args.output_dir / "run_manifest.json").read_text())
        files = ["runner.py", "policies.py", "_strategy.py", "_markowitz.py", "_engine_adapter.py", "provenance.json"]
        code = {name: sha256(PACKAGE / name) for name in files}
        for name in files:
            shutil.copy2(PACKAGE / name, args.output_dir / (name.removesuffix(".py") + ".executed.py" if name.endswith(".py") else name))
        manifest.update(status="completed", metrics=metrics, shared_adapter_sha256=code,
            reference_inputs_verified=bool(args.reference), rebalance_policy={"policy":args.policy, **cfg,
                "monthly_base":True, "cash_in_drift_gap":False, "signed_weights":True,
                "decision_timing":"prior-close weights/prices; available macro; current session MOC",
                "baskets":"fixed between monthly score updates; daily covariance and macro short split"},
            runtime_dependencies={"python":platform.python_version(), "platform":platform.platform(),
                "packages":{name:importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scikit-learn")}})
        threshold = None if args.scenario == "lo" else int(args.scenario[1:])
        manifest["contract"].update(selection=f"top{args.n_positions} long, bottom{args.n_positions} short; Markowitz per side",
            target_short="0" if threshold is None else f"{args.short_cap}*clip((stress-{threshold})/{100-threshold},0,1)",
            fee_bps=args.fee_bps)
        for filename in ("run_manifest.json", "metrics_manifest.json"):
            A.BASE.write_json(args.output_dir / filename, manifest)
        result.update(status="completed", policy=args.policy, metrics=metrics)
        A.BASE.write_json(args.output_dir / "status.json", result)
        return result
    except BaseException as exc:
        if A.BASE.CREATED_OUTPUT and args.output_dir.is_dir():
            state = "aborted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            error = {"status":state, "error":str(exc), "traceback":traceback.format_exc()}
            A.BASE.write_json(args.output_dir / "status.json", error)
            path = args.output_dir / "run_manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                manifest.update(error)
                A.BASE.write_json(path, manifest)
        raise
    finally:
        if timer:
            timer.cancel()


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
