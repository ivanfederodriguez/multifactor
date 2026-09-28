"""Frozen Markowitz v2 sizing and original Point-3 execution policies."""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from . import _engine_adapter as BASE

_BASE_MAKE_DECISIONS = BASE.make_decisions

LOOKBACK_PRICES = 253
MIN_OBS = 60
VALID_RETURN_FRACTION = 0.90
RIDGE = 1e-6
SHORT_CAP = 0.30
P_THRESHOLD = 75.0
FEE_DRIFT_BUFFER = 1e-8
SINGLE_NAME_DRIFT_TRIGGER = 0.05
PORTFOLIO_ONE_WAY_DRIFT_TRIGGER = 0.10

RUN_CONTEXT: dict = {}
MKW_STATS: list[dict] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capped_proportional_weights(raw: np.ndarray, cap: float, total: float) -> np.ndarray:
    """Normalize nonnegative scores to `total`, iteratively redistributing caps."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Markowitz weights require a non-empty vector")
    if not np.isfinite(raw).all() or cap <= 0 or total < 0:
        raise ValueError("Invalid cap or non-finite raw weights")
    if total == 0:
        return np.zeros_like(raw)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        raw = np.ones_like(raw)
    if total > cap * len(raw) + 1e-12:
        raise ValueError(f"Cap infeasible: target={total}, cap={cap}, n={len(raw)}")

    out = np.zeros_like(raw)
    free = np.ones(len(raw), dtype=bool)
    remaining = float(total)
    while free.any():
        free_raw = raw[free]
        if free_raw.sum() <= 0:
            proposal = np.full(free_raw.size, remaining / free_raw.size)
        else:
            proposal = remaining * free_raw / free_raw.sum()
        over = proposal > cap + 1e-14
        free_idx = np.flatnonzero(free)
        if not over.any():
            out[free_idx] = proposal
            break
        capped_idx = free_idx[over]
        out[capped_idx] = cap
        free[capped_idx] = False
        remaining = float(total - out[~free].sum())
        if remaining < -1e-12:
            raise ArithmeticError("Capped reallocation exceeded requested total")
    # Correct harmless floating-point drift without breaking any cap.
    residual = total - out.sum()
    if abs(residual) > 1e-12:
        room = np.flatnonzero(out < cap - 1e-12)
        if room.size:
            out[room] += residual * out[room] / out[room].sum() if out[room].sum() else residual / room.size
    return out


def stress_short_fraction(stress: float | None, threshold: float | None = None) -> float:
    if stress is None or not np.isfinite(stress):
        return 0.0
    threshold = RUN_CONTEXT.get("threshold", P_THRESHOLD) if threshold is None else threshold
    cap = RUN_CONTEXT.get("short_cap", SHORT_CAP)
    if threshold is None or cap <= 0:
        return 0.0
    return cap * float(np.clip((stress - threshold) / (100.0 - threshold), 0.0, 1.0))


def markowitz_side_weights(
    tickers: list[str],
    scores: pd.Series,
    wide_prices: pd.DataFrame,
    price_cutoff: pd.Timestamp,
    cap: float,
    *,
    side: str,
    execution_date: pd.Timestamp,
    score_date: pd.Timestamp,
) -> tuple[dict[str, float], dict]:
    """Reconstruct the documented v2 LW allocator for one side of the book."""
    n = len(tickers)
    if n == 0:
        return {}, {"state": "empty", "n_positions": 0}
    equal = {str(ticker): 1.0 / n for ticker in tickers}
    stats = {
        "execution_date": str(pd.Timestamp(execution_date).date()),
        "score_date": str(pd.Timestamp(score_date).date()),
        "side": side,
        "n_positions": n,
        "cap_per_name": cap,
        "lookback_price_rows": LOOKBACK_PRICES,
        "min_observations": MIN_OBS,
        "valid_return_fraction": VALID_RETURN_FRACTION,
        "ridge": RIDGE,
        "state": "mlw",
        "n_return_rows": 0,
        "n_covariance_names": 0,
        "n_equal_fallback_names": 0,
        "fallback_reason": None,
    }

    if price_cutoff not in wide_prices.index:
        stats.update(state="eq_short_history", fallback_reason="price_cutoff_missing")
        return equal, stats
    eligible_history = wide_prices.loc[wide_prices.index <= price_cutoff]
    symbols_in_prices = [ticker for ticker in tickers if ticker in eligible_history.columns]
    if not symbols_in_prices:
        stats.update(state="eq_short_history", fallback_reason="no_price_columns")
        return equal, stats

    px = eligible_history[symbols_in_prices].tail(LOOKBACK_PRICES).ffill(limit=10)
    returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    stats["n_return_rows"] = int(len(returns))
    if len(returns) < MIN_OBS:
        stats.update(state="eq_short_history", fallback_reason=f"{len(returns)}_returns_below_{MIN_OBS}")
        return equal, stats

    min_valid = int(math.ceil(VALID_RETURN_FRACTION * min(252, len(returns))))
    valid_cols = [col for col in returns.columns if int(returns[col].notna().sum()) >= min_valid]
    stats["n_covariance_names"] = int(len(valid_cols))
    missing = [ticker for ticker in tickers if ticker not in valid_cols]
    stats["n_equal_fallback_names"] = int(len(missing))
    if len(valid_cols) < 2:
        stats.update(state="eq_short_history", fallback_reason="fewer_than_two_valid_return_series")
        return equal, stats

    try:
        clean_returns = returns[valid_cols].fillna(0.0)
        covariance = LedoitWolf().fit(clean_returns.to_numpy(dtype=float)).covariance_
        score_values = scores.reindex(valid_cols).fillna(0.0).astype(float)
        if side == "SHORT":
            score_values = -score_values
        mu = score_values.to_numpy(dtype=float)
        mu = mu - float(mu.min()) + 1e-6
        raw = np.linalg.solve(covariance + RIDGE * np.eye(len(valid_cols)), mu)
        raw = np.clip(raw, 0.0, None)
        if raw.sum() <= 0 or not np.isfinite(raw).all():
            stats.update(state="eq_error", fallback_reason="nonpositive_or_nonfinite_optimizer_output")
            return equal, stats

        fixed = {str(ticker): 1.0 / n for ticker in missing}
        active_total = 1.0 - len(missing) / n
        active = capped_proportional_weights(raw, cap, active_total)
        weights = {str(ticker): float(value) for ticker, value in zip(valid_cols, active)}
        weights.update(fixed)
        # Keep the side fully invested while honoring the per-name cap.
        total = sum(weights.values())
        if not np.isclose(total, 1.0, atol=1e-10):
            raise ArithmeticError(f"Side weights sum to {total}")
        if max(weights.values()) > cap + 1e-9:
            raise ArithmeticError("Per-name cap exceeded after missing-name fallback")
        return weights, stats
    except Exception as exc:
        stats.update(state="eq_error", fallback_reason=f"{type(exc).__name__}:{exc}")
        return equal, stats


def _rank_candidates(
    score_group: pd.DataFrame,
    wide: pd.DataFrame,
    cutoff: pd.Timestamp,
    n_positions: int,
    short_enabled: bool,
) -> tuple[list[str], list[str], pd.Series]:
    if cutoff not in wide.index:
        raise ValueError(f"No exact ranking price cutoff {cutoff}")
    quotes = wide.loc[cutoff]
    tradable = quotes.index[np.isfinite(quotes.to_numpy(dtype=float)) & (quotes.to_numpy(dtype=float) > 0)]
    eligible = score_group[score_group.symbol.isin(tradable)].dropna(subset=["composite_score"])
    eligible = eligible[np.isfinite(eligible.composite_score.to_numpy(dtype=float))]
    ranked_desc = eligible.sort_values(["composite_score", "symbol"], ascending=[False, True], kind="stable")
    ranked_asc = eligible.sort_values(["composite_score", "symbol"], ascending=[True, True], kind="stable")
    required = 2 * n_positions if short_enabled else n_positions
    if len(ranked_desc) < required:
        raise ValueError(f"Need {required} eligible names, found {len(ranked_desc)}")
    return (
        ranked_desc.head(n_positions).symbol.astype(str).tolist(),
        ranked_asc.head(n_positions).symbol.astype(str).tolist() if short_enabled else [],
        eligible.set_index("symbol").composite_score.astype(float),
    )


def _make_raw_target(
    *,
    execution_date: pd.Timestamp,
    score_date: pd.Timestamp,
    price_cutoff: pd.Timestamp,
    score_group: pd.DataFrame,
    wide: pd.DataFrame,
    stress_value: float | None,
    stress_row: pd.Series | None,
    n_positions: int,
    cap: float,
    short_enabled: bool,
    fixed_baskets: dict | None = None,
    record_stats: bool = True,
) -> tuple[dict, dict, list[dict]]:
    if fixed_baskets is None:
        longs, shorts, score_map = _rank_candidates(
            score_group, wide, price_cutoff, n_positions, short_enabled
        )
    else:
        longs = list(fixed_baskets["longs"])
        shorts = list(fixed_baskets["shorts"]) if short_enabled else []
        score_map = fixed_baskets["score_map"]
    lw, lw_stats = markowitz_side_weights(longs, score_map, wide, price_cutoff, cap,
                                          side="LONG", execution_date=execution_date,
                                          score_date=score_date)
    if shorts:
        short_weights, short_stats = markowitz_side_weights(
            shorts, score_map, wide, price_cutoff, cap, side="SHORT",
            execution_date=execution_date, score_date=score_date
        )
    else:
        short_weights, short_stats = {}, None
    short_fraction = stress_short_fraction(stress_value) if short_enabled else 0.0
    raw = {ticker: float((1.0 - short_fraction) * weight) for ticker, weight in lw.items()}
    raw.update({ticker: float(-short_fraction * weight) for ticker, weight in short_weights.items()})
    if set(longs) & set(shorts):
        raise ValueError("Long and short candidate lists overlap")
    stress_batch = None if stress_row is None else stress_row.BATCH_DATE
    stress_effective = None if stress_row is None else stress_row.FIRST_TRADABLE_DATE
    decision = {
        "score_date": score_date,
        "price_cutoff_date": price_cutoff,
        "eligible_count": len(score_map),
        "n_positions": n_positions,
        "stress_value": stress_value,
        "target_short_fraction": short_fraction,
        "target_long_fraction": 1.0 - short_fraction,
        "stress_batch_date": stress_batch,
        "stress_first_tradable_date": stress_effective,
        "weighting": "Markowitz-LedoitWolf-v2-reconstructed",
        "weight_cap_per_name": cap,
        "raw_weights": raw,
    }
    selections = []
    for side, basket, weights, side_fraction in [
        ("LONG", longs, lw, 1.0 - short_fraction),
        ("SHORT", shorts, short_weights, short_fraction),
    ]:
        for rank, symbol in enumerate(basket, 1):
            selections.append({
                "execution_date": execution_date,
                "score_date": score_date,
                "symbol": symbol,
                "side": side,
                "rank": rank,
                "score": float(score_map.loc[symbol]),
                "raw_weight": float(side_fraction * weights[symbol]) * (1.0 if side == "LONG" else -1.0),
            })
    if record_stats:
        MKW_STATS.append(lw_stats)
        if short_stats is not None:
            MKW_STATS.append(short_stats)
    return decision, {"longs": longs, "shorts": shorts, "score_map": score_map}, selections


def _causal_macro(macro: pd.DataFrame, date: pd.Timestamp) -> tuple[float | None, pd.Series | None]:
    rows = macro[(macro.FIRST_TRADABLE_DATE <= date) & (macro.BATCH_DATE < date)]
    if rows.empty:
        return None, None
    row = rows.iloc[-1]
    return float(row.STRESS_PERCENTILE_0_100), row


def _patched_make_decisions(scores, wide, calendar, macro, threshold, n_positions=20, selection_fraction=None):
    """Keep the frozen runner's date/eligibility policy; replace only sizing."""
    expected = RUN_CONTEXT.get("threshold")
    if (threshold is None) != (expected is None) or (
        threshold is not None and not np.isclose(float(threshold), float(expected))
    ):
        raise ValueError(f"Scenario threshold mismatch: runner={threshold}, adapter={expected}")
    short_enabled = threshold is not None and RUN_CONTEXT.get("short_cap", 0.0) > 0
    base_decisions, _ = _BASE_MAKE_DECISIONS(scores, wide, calendar, macro, threshold,
                                              n_positions, selection_fraction)
    score_groups = {pd.Timestamp(k): group.copy() for k, group in scores.groupby("date", sort=True)}
    candidates = {}
    rebuilt_selections = []
    cap = 3.0 / n_positions
    for execution_date, old in base_decisions.items():
        score_date = pd.Timestamp(old["score_date"])
        cutoff = pd.Timestamp(old["price_cutoff_date"])
        stress, stress_row = _causal_macro(macro, pd.Timestamp(execution_date))
        group = score_groups[score_date]
        decision, candidate, rows = _make_raw_target(
            execution_date=pd.Timestamp(execution_date), score_date=score_date,
            price_cutoff=cutoff, score_group=group, wide=wide, stress_value=stress,
            stress_row=stress_row, n_positions=n_positions, cap=cap,
            short_enabled=short_enabled,
        )
        # Include zero-weight short candidates in the engine's price union. This
        # lets an intramonth P75 crossing open the short with native exact-date prices.
        for ticker in candidate["shorts"]:
            decision["raw_weights"].setdefault(ticker, 0.0)
        base_decisions[execution_date] = decision
        candidates[score_date] = {"group": group, **candidate, "price_cutoff": cutoff}
        rebuilt_selections.extend(rows)

    RUN_CONTEXT.update(
        scores=scores,
        score_groups=score_groups,
        wide=wide,
        calendar=calendar,
        macro=macro,
        threshold=None if threshold is None else float(threshold),
        n_positions=int(n_positions),
        cap=cap,
        short_cap=float(RUN_CONTEXT.get("short_cap", 0.0)),
        short_enabled=short_enabled,
        decisions=base_decisions,
        candidates=candidates,
        decision_dates=sorted(base_decisions),
    )
    return base_decisions, pd.DataFrame(rebuilt_selections)


def _last_close_before(date: pd.Timestamp) -> pd.Timestamp:
    calendar = RUN_CONTEXT["calendar"]
    index = int(calendar.searchsorted(date, side="left")) - 1
    if index >= 0:
        return pd.Timestamp(calendar[index])
    decisions = RUN_CONTEXT["decision_dates"]
    return pd.Timestamp(RUN_CONTEXT["decisions"][decisions[0]]["price_cutoff_date"])


def _latest_signal(date: pd.Timestamp) -> tuple[pd.Timestamp, dict] | None:
    dates = RUN_CONTEXT["decision_dates"]
    index = bisect_right(dates, date) - 1
    if index < 0:
        return None
    execution_date = dates[index]
    decision = RUN_CONTEXT["decisions"][execution_date]
    return pd.Timestamp(decision["score_date"]), decision


def _daily_decision(date: pd.Timestamp) -> dict | None:
    signal = _latest_signal(date)
    if signal is None:
        return None
    score_date, source_decision = signal
    candidate = RUN_CONTEXT["candidates"].get(score_date)
    if candidate is None:
        return None
    stress, stress_row = _causal_macro(RUN_CONTEXT["macro"], date)
    cutoff = _last_close_before(date)
    decision, _, _ = _make_raw_target(
        execution_date=date,
        score_date=score_date,
        price_cutoff=cutoff,
        score_group=candidate["group"],
        wide=RUN_CONTEXT["wide"],
        stress_value=stress,
        stress_row=stress_row,
        n_positions=RUN_CONTEXT["n_positions"],
        cap=RUN_CONTEXT["cap"],
        short_enabled=RUN_CONTEXT["short_enabled"],
        fixed_baskets=candidate,
        record_stats=True,
    )
    # Retain the original signal's eligibility snapshot; no new instruments
    # enter a basket merely because a later price became available.
    for ticker in candidate["shorts"]:
        decision["raw_weights"].setdefault(ticker, 0.0)
    decision["source_monthly_execution_date"] = source_decision.get("execution_date")
    return decision


def _actual_weights_at_previous_close(strategy, date: pd.Timestamp) -> dict[str, float]:
    history = strategy._context.data.get_instrument_prices()
    close = history["close_price"]
    dates = close.index.get_level_values("date")
    close = close[(dates < date) & np.isfinite(close.to_numpy(dtype=float)) & (close.to_numpy(dtype=float) > 0)]
    if close.empty:
        return {}
    marks = close.groupby(level="symbol").last().to_dict()
    return strategy._context.broker.get_portfolio_weigths(marks, {})


def _current_stress_regime(date: pd.Timestamp) -> tuple[float | None, bool]:
    stress, _ = _causal_macro(RUN_CONTEXT["macro"], date)
    threshold = RUN_CONTEXT.get("threshold")
    return stress, bool(threshold is not None and stress is not None and stress >= threshold)


class Point3Strategy(BASE.Strategy):
    """Monthly base plus either macro-stress or target-weight drift triggers."""

    def setup(self, config):
        self.decisions = config["decisions"]
        self.fee_rate = config.get("fee_bps", 10.0) / 10000.0
        self.progress_path = config.get("progress_path")
        self.audit = []
        self.bars = 0
        self.mode = RUN_CONTEXT["mode"]
        self.last_stress_regime: bool | None = None
        self.last_rebalanced_short: float | None = None

    def reset(self):
        self.audit.clear()
        self.bars = 0
        self.last_stress_regime = None
        self.last_rebalanced_short = None

    def run(self):
        self.bars += 1
        date = pd.Timestamp(self._context.current_date)
        if self.progress_path and (self.bars == 1 or self.bars % 252 == 0):
            BASE.write_json(self.progress_path, {"status": "running", "bar": self.bars, "date": date})

        decision = self.decisions.get(date)
        stress, regime = _current_stress_regime(date)
        trigger_reason = None
        max_name_gap = None
        one_way_gap = None

        if decision is not None:
            trigger_reason = "monthly_scheduled"
            self.last_stress_regime = regime
            self.last_rebalanced_short = float(decision["target_short_fraction"])
        elif self.mode == "stress":
            short_target = stress_short_fraction(stress)
            regime_cross = self.last_stress_regime is not None and regime != self.last_stress_regime
            short_move = (self.last_rebalanced_short is not None and
                          abs(short_target - self.last_rebalanced_short) >= 0.05 - 1e-12)
            if regime_cross or short_move:
                decision = _daily_decision(date)
                trigger_reason = "stress_regime_cross" if regime_cross else "stress_short_target_delta_5pp"
                if regime_cross and short_move:
                    trigger_reason = "stress_regime_cross+short_target_delta_5pp"
                if decision is not None:
                    self.last_rebalanced_short = float(decision["target_short_fraction"])
            self.last_stress_regime = regime
        elif self.mode == "drift":
            decision = _daily_decision(date)
            if decision is not None:
                current = _actual_weights_at_previous_close(self, date)
                target = decision["raw_weights"]
                symbols = set(current) | set(target)
                gaps = {symbol: abs(float(current.get(symbol, 0.0)) - float(target.get(symbol, 0.0)))
                        for symbol in symbols}
                max_name_gap = max(gaps.values(), default=0.0)
                one_way_gap = 0.5 * sum(gaps.values())
                reasons = []
                if max_name_gap >= SINGLE_NAME_DRIFT_TRIGGER - 1e-12:
                    reasons.append("single_name_gap_ge_5pp")
                if one_way_gap >= PORTFOLIO_ONE_WAY_DRIFT_TRIGGER - 1e-12:
                    reasons.append("one_way_turnover_gap_ge_10pp")
                if reasons:
                    trigger_reason = "+".join(reasons)
                else:
                    decision = None
            self.last_stress_regime = regime
            if trigger_reason is not None and decision is not None:
                self.last_rebalanced_short = float(decision["target_short_fraction"])
        elif self.mode != "monthly":
            raise ValueError(f"Unsupported rebalance mode {self.mode}")

        if decision is None:
            return None

        # Same native price marks, fee reserve, Broker, and order path as the
        # frozen runner. Trigger detection above uses only the prior close.
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
        targets, sizing = BASE.fee_reserved_weights(decision["raw_weights"], current, nav,
                                                    quotes, self.fee_rate, epsilon=FEE_DRIFT_BUFFER)
        info = {key: value for key, value in decision.items() if key != "raw_weights"}
        info.update(sizing, execution_date=date, pretrade_nav=nav,
                    raw_target_gross=sum(abs(value) for value in decision["raw_weights"].values()),
                    requested_target_gross=sum(abs(value) for value in targets.values()),
                    rebalance_mode=self.mode, trigger_reason=trigger_reason,
                    max_single_name_gap=max_name_gap, one_way_target_turnover_gap=one_way_gap)
        self.audit.append(info)
        return targets, info
