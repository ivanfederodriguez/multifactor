"""Public qbacktest adapter; no engine implementation is distributed here."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import qbacktest
from qbacktest.engine import BacktestEngine
from qbacktest.environment import Currency
from qbacktest.presets import SpotEquityPreset
from qbacktest.presets.costs.fee_models import PercentOfNotional
from qbacktest.strategies import Strategy
from qbacktest.utils import Frequency

QBACKTEST = Path(qbacktest.__file__).resolve().parents[2]
CREATED_OUTPUT = False


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default, allow_nan=False) + "\n")


class StudyEquityPreset(SpotEquityPreset):
    """Only public instrument configuration; native fee callback and broker."""
    def parse_params(self, dataset, fee_bps=10., lot_size=1e-8, tick_size=1e-8, **kwargs):
        params = super().parse_params(dataset, **kwargs)
        params["investment_universe"].update(
            fee_model=PercentOfNotional(pct=fee_bps / 10000.),
            lot_size=lot_size, tick_size=tick_size,
        )
        return params


def fee_reserved_weights(raw, current, nav, tradable, fee_rate, epsilon=1e-8):
    """Self-financing target sizing, not cash/NAV accounting.

    Frozen untradable existing positions keep their marked weights. New missing
    entry quotes retain their intended fraction as cash (no replacements or
    renormalization). A common scalar reserves expected native notional fees.
    Tiny explicit numerical cash buffer handles native tick/lot rounding.
    """
    frozen = {s: w for s, w in current.items() if s not in tradable}
    frozen_gross = sum(abs(w) for w in frozen.values())
    raw_tradable = {s: w for s, w in raw.items() if s in tradable}
    symbols = sorted(set(raw_tradable) | (set(current) & set(tradable)))
    fees = 0.
    for _ in range(100):
        scale = max(0., 1. - frozen_gross - fees / nav - epsilon)
        new_fees = fee_rate * nav * sum(abs(raw_tradable.get(s, 0.) * scale - current.get(s, 0.)) for s in symbols)
        if abs(new_fees - fees) < 1e-9:
            fees = new_fees
            break
        fees = new_fees
    scale = max(0., 1. - frozen_gross - fees / nav - epsilon)
    result = {s: w * scale for s, w in raw_tradable.items()}
    result.update(frozen)
    return result, dict(fee_reserve=fees, target_scale=scale,
                        frozen_gross_weight=frozen_gross,
                        missing_entry_weight=sum(abs(w) for s, w in raw.items() if s not in tradable),
                        numerical_cash_buffer_fraction=epsilon)


class FrozenMonthlyStrategy(Strategy):
    """Public strategy interface; timestamps supplied without backdating."""
    def setup(self, config):
        self.decisions = config["decisions"]
        self.fee_rate = config.get("fee_bps", 10.) / 10000.
        self.progress_path = config.get("progress_path")
        self.audit = []
        self.bars = 0

    def reset(self):
        self.audit.clear()
        self.bars = 0

    def run(self):
        self.bars += 1
        date = pd.Timestamp(self._context.current_date)
        if self.progress_path and (self.bars == 1 or self.bars % 252 == 0):
            write_json(self.progress_path, {"status": "running", "bar": self.bars, "date": date})
        decision = self.decisions.get(date)
        if decision is None:
            return None
        assert pd.Timestamp(decision["score_date"]) < date
        # Read prices only through QBacktest's current-date-filtered Context.
        history = self._context.data.get_instrument_prices()
        close = history["close_price"]
        close = close[np.isfinite(close) & (close > 0)]
        marks = close.groupby(level="symbol").last().to_dict()
        try:
            quotes = close.xs(date, level="date").to_dict()
        except KeyError:
            quotes = {}
        broker = self._context.broker
        nav = broker.nav(marks, {})
        current = broker.get_portfolio_weigths(marks, {})
        targets, sizing = fee_reserved_weights(decision["raw_weights"], current, nav,
                                               quotes, self.fee_rate)
        info = {k: v for k, v in decision.items() if k != "raw_weights"}
        info.update(sizing, execution_date=date, pretrade_nav=nav,
                    raw_target_gross=sum(abs(v) for v in decision["raw_weights"].values()),
                    requested_target_gross=sum(abs(v) for v in targets.values()))
        self.audit.append(info)
        return targets, info


def make_decisions(scores, wide, calendar, macro, threshold, n_positions=20, selection_fraction=None):
    decisions, selection_rows = {}, []
    dates = wide.index
    for score_date, group in scores.groupby("date", sort=True):
        j = calendar.searchsorted(score_date, side="right")
        if j == len(calendar):
            continue
        date = calendar[j]
        # Exclude old pre-study signals, rather than combining months at bar0.
        all_j = dates.searchsorted(score_date, side="right")
        if all_j == len(dates) or dates[all_j] != date:
            continue
        cutoff = dates[all_j - 1]
        quotes = wide.loc[cutoff]
        eligible_symbols = quotes.index[np.isfinite(quotes) & (quotes > 0)]
        eligible = group[group.symbol.isin(eligible_symbols)].dropna(subset=["composite_score"])
        eligible = eligible[np.isfinite(eligible.composite_score)]
        ranked = eligible.sort_values(["composite_score", "symbol"], ascending=[False, True], kind="stable")
        n = max(1, int(np.floor(len(ranked) * selection_fraction))) if selection_fraction is not None else n_positions
        if len(ranked) < (2 * n if threshold is not None else n):
            raise ValueError(f"Insufficient eligible rows at {score_date}: {len(ranked)}")
        m = macro[(macro.FIRST_TRADABLE_DATE <= date) & (macro.BATCH_DATE < date)]
        if threshold is not None and m.empty:
            raise ValueError(f"No causal macro row at {date}")
        stress = float(m.iloc[-1].STRESS_PERCENTILE_0_100) if len(m) else None
        short = 0. if threshold is None else .2 * float(np.clip((stress - threshold) / (100. - threshold), 0., 1.))
        longs = ranked.head(n)
        # Ascending score, symbol ascending, exactly the prior study's tie rule.
        shorts = eligible.sort_values(["composite_score", "symbol"], kind="stable").head(n) if short > 0 else eligible.iloc[:0]
        raw = {str(s): (1. - short) / n for s in longs.symbol}
        raw.update({str(s): -short / n for s in shorts.symbol})
        if set(longs.symbol) & set(shorts.symbol):
            raise ValueError("Long and short baskets overlap")
        decision = dict(score_date=score_date, price_cutoff_date=cutoff, eligible_count=len(ranked),
                        n_positions=n, stress_value=stress, target_short_fraction=short,
                        stress_batch_date=m.iloc[-1].BATCH_DATE if len(m) else None,
                        stress_first_tradable_date=m.iloc[-1].FIRST_TRADABLE_DATE if len(m) else None,
                        raw_weights=raw)
        if "weight_effective_from" in group:
            decision["weight_effective_from"] = pd.to_datetime(group.weight_effective_from).max()
            assert decision["weight_effective_from"] <= score_date
        if date in decisions:
            raise ValueError(f"Multiple score dates execute on {date}")
        decisions[date] = decision
        for side, basket in [("LONG", longs), ("SHORT", shorts)]:
            for rank, row in enumerate(basket.itertuples(), 1):
                selection_rows.append(dict(execution_date=date, score_date=score_date, symbol=row.symbol,
                                           side=side, rank=rank, score=row.composite_score, raw_weight=raw[row.symbol]))
    return decisions, pd.DataFrame(selection_rows)


def build_engine(prices, calendar, decisions, *, borrow=.02, recovery=1., fee_bps=10., progress_path=None, starting_cash=1_000_000., max_leverage=1.5):
    preset = StudyEquityPreset(
        prices, starting_cash=starting_cash, cash_rates={Currency.USD: (0., 0.)},
        max_leverage=max_leverage, add_noise=False, interest_settlement_frequency=Frequency.DAILY,
        min_weight=1e-12, margin_rate=.5,
        financing_rates={"LONG": 0., "SHORT": borrow},
        collateral_rates={"LONG": 0., "SHORT": 1.}, add_slippage=False,
        fee_bps=fee_bps, lot_size=1e-8, tick_size=1e-8,
    )
    config = preset()
    config["environment_config"].update(delisting_stale_days=10, delisting_recovery=recovery)
    engine = BacktestEngine(
        strategy_config={"class": FrozenMonthlyStrategy, "params": {
            "decisions": decisions, "fee_bps": fee_bps, "progress_path": progress_path}},
        environment_config=config["environment_config"], data=config["data"],
        calendar=list(calendar.to_pydatetime()), random_seed=42, show_progress=False,
    )
    return engine


def metrics_and_daily(output, initial_cash, initial_nav_date=None):
    daily = output.portfolio.copy()
    daily.index.name = "date"
    daily["daily_return"] = daily.nav / daily.nav.shift(1).fillna(initial_cash) - 1.
    daily["gross_exposure"] = (daily.long_exposure + daily.short_exposure) / daily.nav
    daily["net_exposure"] = (daily.long_exposure - daily.short_exposure) / daily.nav
    daily["long_weight"] = daily.long_exposure / daily.nav
    daily["short_weight"] = daily.short_exposure / daily.nav
    daily["borrow_fee"] = -(daily.interests_settled + daily.interests).diff().fillna(-(daily.interests_settled + daily.interests).iloc[0])
    trade_notional = output.trades.qty * output.trades.price * output.trades.contract_size if len(output.trades) else pd.Series(dtype=float)
    fees = output.trades.fee.groupby(level=0).sum() if len(output.trades) else pd.Series(dtype=float)
    daily["cost"] = fees.reindex(daily.index, fill_value=0.)
    r = daily.daily_return
    vol = float(r.std(ddof=1) * np.sqrt(252))
    cagr252 = float((daily.nav.iloc[-1] / initial_cash) ** (252 / len(daily)) - 1.)
    initial_nav_date = pd.Timestamp(initial_nav_date if initial_nav_date is not None else daily.index[0])
    years = (daily.index[-1] - initial_nav_date).days / 365.25
    cagr = float((daily.nav.iloc[-1] / initial_cash) ** (1. / years) - 1.) if years > 0 else cagr252
    wealth_with_initial = np.r_[initial_cash, daily.nav.to_numpy()]
    m = dict(start=str(daily.index.min().date()), end=str(daily.index.max().date()), sessions=len(daily),
             initial_nav_date=str(initial_nav_date.date()), elapsed_calendar_years=years,
             total_return=float(daily.nav.iloc[-1] / initial_cash - 1.), cagr=cagr, cagr_252=cagr252, annual_volatility=vol,
             sharpe_standard=float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if vol else None,
             sharpe_cagr_over_vol=cagr / vol if vol else None,
             max_drawdown=float(np.min(wealth_with_initial / np.maximum.accumulate(wealth_with_initial) - 1.)),
             final_nav=float(daily.nav.iloc[-1]), total_fees=float(fees.sum()), total_borrow=float(daily.borrow_fee.sum()),
             trade_notional=float(trade_notional.sum()), trades=len(output.trades),
             rebalance_decisions=len(output.execution), rejected_trades=int(output.execution.rejected_trades.sum()) if len(output.execution) else 0,
             strategy_errors=len(output.strategy_errors), mean_gross=float(daily.gross_exposure.mean()), max_gross=float(daily.gross_exposure.max()),
             mean_net=float(daily.net_exposure.mean()), mean_short=float(daily.short_weight.mean()), max_short=float(daily.short_weight.max()),
             fraction_sessions_short_active=float((daily.short_weight > 1e-10).mean()),
             min_cash_available=float(daily.cash_available.min()), min_cash_total=float(daily.cash_total.min()))
    return m, daily


def save_output(output, engine, outdir, initial_cash):
    for attr in ["portfolio", "positions", "trades", "weights", "target_weights", "execution"]:
        frame = getattr(output, attr).copy()
        frame.index.name = "date"
        frame.to_csv(outdir / f"{attr}.csv")
    write_json(outdir / "events_history.json", output.events_history)
    write_json(outdir / "model_outputs.json", {str(k): v for k, v in output.model_outputs.items()})
    write_json(outdir / "strategy_errors.json", output.strategy_errors)
    pd.DataFrame(engine.strategy.audit).to_csv(outdir / "rebalance_audit.csv", index=False)
    initial_nav_date = engine.strategy.decisions[min(engine.strategy.decisions)]["price_cutoff_date"]
    metrics, daily = metrics_and_daily(output, initial_cash, initial_nav_date)
    daily.to_csv(outdir / "daily.csv")
    daily[["nav"]].to_csv(outdir / "daily_nav.csv")
    daily[["daily_return"]].to_csv(outdir / "daily_returns.csv")
    daily.daily_return.resample("ME").apply(lambda x: (1. + x).prod() - 1.).rename("monthly_return").to_csv(outdir / "monthly_returns.csv")
    daily.daily_return.resample("YE").apply(lambda x: (1. + x).prod() - 1.).rename("annual_return").to_csv(outdir / "annual_returns.csv")
    return metrics


def run_case(args):
    global CREATED_OUTPUT
    started = time.monotonic()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    CREATED_OUTPUT = True
    status_path = outdir / "status.json"
    write_json(status_path, {"status": "preparing", "model": args.model, "scenario": args.scenario})
    logging.basicConfig(filename=outdir / "engine.log", level=logging.WARNING, force=True)
    spec = {"model": "dqi_v2", "scores_path": str(Path(args.scores_path).resolve())}
    score_path = Path(args.scores_path)
    scores = pd.read_parquet(score_path)
    scores["date"] = pd.to_datetime(scores.date).dt.normalize()
    assert not scores.duplicated(["date", "symbol"]).any()
    if "weight_effective_from" in scores:
        assert (pd.to_datetime(scores.weight_effective_from) <= scores.date).all()
    scores = scores[(scores.date >= pd.Timestamp(args.start) - pd.Timedelta(days=40)) & (scores.date < pd.Timestamp(args.end))]
    prices = pd.read_parquet(args.prices_path)
    if "date" not in prices.columns:
        prices = prices.reset_index()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    assert not prices.duplicated(["date", "symbol"]).any()
    if "volume_in_units" not in prices:
        raise ValueError("Missing native volume_in_units column; not fabricating market data")
    prices = prices[(prices.date >= scores.date.min() - pd.Timedelta(days=35)) & (prices.date <= pd.Timestamp(args.end))]
    calendar = pd.DatetimeIndex(sorted(prices.date.unique()))
    calendar = calendar[(calendar >= pd.Timestamp(args.start)) & (calendar <= pd.Timestamp(args.end))]
    wide = prices.pivot(index="date", columns="symbol", values="close_price")
    macro = pd.read_csv(args.macro_path)
    for col in ["BATCH_DATE", "FIRST_TRADABLE_DATE"]:
        macro[col] = pd.to_datetime(macro[col]).dt.normalize()
    macro = macro.dropna(subset=["STRESS_PERCENTILE_0_100"]).sort_values(["FIRST_TRADABLE_DATE", "BATCH_DATE"])
    threshold = None if args.scenario == "lo" else float(args.scenario[1:])
    decisions, selections = make_decisions(scores, wide, calendar, macro, threshold, args.n_positions, args.selection_fraction)
    if not decisions or min(decisions) != calendar.min():
        raise ValueError(f"First decision mismatch: {min(decisions) if decisions else None}, {calendar.min()}")
    selections.to_csv(outdir / "selections.csv", index=False)
    # Excluding never-targeted symbols is a storage/runtime optimization only:
    # rankings above use the full original cutoff universe, not this union.
    union = sorted({s for d in decisions.values() for s in d["raw_weights"]})
    px = prices[prices.symbol.isin(union)][["date", "symbol", "close_price", "volume_in_units"]].copy()
    for col in ["close_price", "volume_in_units"]:
        px[col] = px[col].astype(float)
    px = px.set_index(["date", "symbol"]).sort_index()
    px.to_parquet(outdir / "engine_prices.parquet")
    del prices, wide, scores
    input_records = {}
    for label, path in [("scores", score_path), ("prices", Path(args.prices_path)), ("macro", Path(args.macro_path)),
                        ("engine_prices", outdir / "engine_prices.parquet"), ("runner", Path(__file__))]:
        input_records[label] = {"path": str(path.resolve()), "sha256": sha(path)}
    engine_files = ["engine/backtest_engine.py", "environment/broker.py", "environment/portfolio.py", "environment/instruments.py",
                    "data/context.py", "presets/equity_presets.py", "presets/costs/fee_models.py"]
    engine_sources = {p: sha(QBACKTEST / "src/qbacktest" / p) for p in engine_files}
    contract = dict(engine="unmodified qbacktest candidate BacktestEngine/Broker/Portfolio",
                    qbacktest_import_path=qbacktest.__file__, engine_sources=engine_sources,
                    qbacktest_commit=subprocess.check_output(["git", "-C", str(QBACKTEST), "rev-parse", "HEAD"], text=True).strip(),
                    decision_timing="first observed trading session strictly after score_date; frozen signals, no re-training",
                    eligibility="positive finite exact quote on last market session <= score_date; no per-symbol stale cutoff fill",
                    execution="same decision-day MOC through native broker, positive exact quote required",
                    selection="top20 long; bottom20 short; symbol ascending tie-break; EW within each side",
                    target_short="0 for LO; .20*clip((stress-P)/(100-P),0,1) otherwise",
                    fee_bps=args.fee_bps, fee_callback="native PercentOfNotional; charged within Broker.place_order",
                    fee_sizing="public Strategy reserves predicted fees, common scalar, numerical cash buffer1e-8NAV; no NAV alterations",
                    lot_size=1e-8, tick_size=1e-8, fractional_positions=True, max_leverage=1.5, margin_rate=.5,
                    leverage_limit_reason="technical broker headroom for drifted gross>1: native preview rejects even risk-reducing individual orders unless each intermediate portfolio satisfies its limit; raw strategy gross target remains1",
                    metrics_cagr="calendar ACT/365.25 from initial cash dated last market session <= first score; cagr_252 separately",
                    metrics_sharpe="mean(daily_return)/sample_std(daily_return)*sqrt252, rf0, includes first-day fee return",
                    cash_credit_rate=0., cash_debit_rate=0., borrow_rate=args.borrow_rate,
                    borrow_convention="native ACT/360 on post-trade current-close short liability; calendar days; daily settlement",
                    delisting_stale_sessions=10, delisting_recovery=args.recovery,
                    missing_entry="cash for missing intended fraction, no replacements or opportunistic renormalization",
                    missing_held="native last valid mark and force-close after10 missing sessions; untradable weights frozen in sizing",
                    price_universe="full universe used for rankings; native engine receives exact original rows for union of target symbols",
                    limitations=["macro vintages current_or_archived_snapshot_not_asof_vintage; estimated18:30ET availability",
                                 "No borrow availability, locate failures, dividends or tax modeled",
                                 "Not identical to previous research engine: financing timing/day count and native execution differ",
                                 "Trade ranking/macro known before current MOC; exact close selection/availability remains simulation convention"])
    manifest = dict(status="running", runner_revision="r2_technical_leverage_headroom", panel="base_b_f95e", model=spec["model"], scenario=args.scenario,
                    parameters=vars(args), model_metadata=spec, inputs=input_records, contract=contract,
                    calendar_sessions=len(calendar), selected_union_count=len(union), price_rows=len(px))
    write_json(outdir / "run_manifest.json", manifest)
    # Retain the exact adapter source used by this run, not engine modifications.
    (outdir / "run_case.executed.py").write_bytes(Path(__file__).read_bytes())
    engine = build_engine(px, calendar, decisions, borrow=args.borrow_rate, recovery=args.recovery,
                          fee_bps=args.fee_bps, progress_path=status_path, starting_cash=args.initial_cash)
    prep_seconds = time.monotonic() - started
    output = engine.run()
    metrics = save_output(output, engine, outdir, args.initial_cash)
    elapsed = time.monotonic() - started
    manifest.update(status="completed" if not metrics["strategy_errors"] else "failed_strategy", metrics=metrics,
                    runtime_seconds=elapsed, preparation_seconds=prep_seconds)
    write_json(outdir / "run_manifest.json", manifest)
    write_json(outdir / "metrics_manifest.json", manifest)
    result = dict(status=manifest["status"], model=spec["model"], scenario=args.scenario, output_dir=str(outdir),
                  runtime_seconds=elapsed, preparation_seconds=prep_seconds, metrics=metrics)
    write_json(status_path, result)
    print(json.dumps(result, default=json_default, allow_nan=False), flush=True)
    return result
