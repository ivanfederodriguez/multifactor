from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


SCHEMA_LABELS = {
    "EM_Fijo": "EM fijo",
    "HRP_Fijo": "HRP fijo",
    "HRPRolling": "HRP rolling",
    "KalmanEM_SB": "Kalman EM · Style buckets",
    "KalmanEM_base": "Kalman EM · Base",
    "SB_Fijo": "Style buckets fijo",
}

WEIGHTS_MODE_LABELS = {
    "equal": "Igual",
    "hrp": "HRP",
    "markowitz_lw": "Markowitz LW",
}

FACTOR_LABELS = {
    "factor_w_value": "Value",
    "factor_w_quality": "Quality",
    "factor_w_growth": "Growth",
    "factor_w_momentum": "Momentum",
    "factor_w_revisions": "Revisions",
    "factor_w_prof_momentum": "Profitability momentum",
    "factor_w_lowvol": "Low volatility",
}

SUBFACTOR_LABELS = {
    "subfactor_w_ev_ebitda": "EV / EBITDA",
    "subfactor_w_fcf_yield": "FCF Yield",
    "subfactor_w_earnings_yield": "Earnings Yield",
    "subfactor_w_roic": "ROIC",
    "subfactor_w_net_debt_ebitda": "Net Debt / EBITDA",
    "subfactor_w_gross_margin": "Gross Margin",
    "subfactor_w_rev_growth_ttm": "Revenue Growth TTM",
    "subfactor_w_eps_growth_fy1": "EPS Growth FY1",
    "subfactor_w_rev_growth_ntm": "Revenue Growth NTM",
    "subfactor_w_return_12m_ex1": "Return 12M ex-1",
    "subfactor_w_return_6m": "Return 6M",
    "subfactor_w_return_12m": "Return 12M",
    "subfactor_w_eps_revision_1m": "EPS Revision 1M",
    "subfactor_w_delta_roic": "Delta ROIC",
    "subfactor_w_delta_gross_margin": "Delta Gross Margin",
    "subfactor_w_delta_oper_margin": "Delta Operating Margin",
    "subfactor_w_gp_over_assets": "Gross Profit / Assets",
    "subfactor_w_accruals": "Accruals",
    "subfactor_w_piotroski_f_score": "Piotroski F-Score",
    "subfactor_w_ev_ebit": "EV / EBIT",
    "subfactor_w_book_to_price": "Book to Price",
    "subfactor_w_rd_intensity": "R&D Intensity",
    "subfactor_w_high52w_proximity": "52W High Proximity",
    "subfactor_w_market_beta_756d": "Market Beta 756D",
    "subfactor_w_idiosyncratic_vol_63d": "Idiosyncratic Vol 63D",
}

SUBFACTOR_FACTORS = {
    "subfactor_w_ev_ebitda": "Value",
    "subfactor_w_fcf_yield": "Value",
    "subfactor_w_earnings_yield": "Value",
    "subfactor_w_ev_ebit": "Value",
    "subfactor_w_book_to_price": "Value",
    "subfactor_w_roic": "Quality",
    "subfactor_w_net_debt_ebitda": "Quality",
    "subfactor_w_gross_margin": "Quality",
    "subfactor_w_gp_over_assets": "Quality",
    "subfactor_w_accruals": "Quality",
    "subfactor_w_piotroski_f_score": "Quality",
    "subfactor_w_rd_intensity": "Quality",
    "subfactor_w_rev_growth_ttm": "Growth",
    "subfactor_w_eps_growth_fy1": "Growth",
    "subfactor_w_rev_growth_ntm": "Growth",
    "subfactor_w_return_12m_ex1": "Momentum",
    "subfactor_w_return_6m": "Momentum",
    "subfactor_w_return_12m": "Momentum",
    "subfactor_w_high52w_proximity": "Momentum",
    "subfactor_w_eps_revision_1m": "Revisions",
    "subfactor_w_delta_roic": "Profitability momentum",
    "subfactor_w_delta_gross_margin": "Profitability momentum",
    "subfactor_w_delta_oper_margin": "Profitability momentum",
    "subfactor_w_market_beta_756d": "Low volatility",
    "subfactor_w_idiosyncratic_vol_63d": "Low volatility",
}


def _csv_record_count(path: Path) -> int:
    """Count records without loading the trade ledger into memory."""
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _sortino_ratio(returns: pd.Series, periods: int = 252) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    downside = clean[clean < 0]
    if clean.empty or len(downside) < 2:
        return math.nan
    downside_deviation = downside.std(ddof=1) * math.sqrt(periods)
    if downside_deviation == 0:
        return math.nan
    return float(clean.mean() * periods / downside_deviation)


def _fallback_config_values(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    strategy = config.get("strategy", {})
    model = strategy.get("model_config", {})
    constructor = model.get("portfolio_constructor", {}).get("params", {})
    risk = model.get("risk_overlay", {}).get("params", {})
    signal = model.get("signal_generator", {}).get("params", {})
    testing = config.get("testing", {})
    weights_file = str(signal.get("weights_file", "")).lower()
    return {
        "n_positions": constructor.get("n_positions"),
        "target_vol": risk.get("target_vol"),
        "max_leverage": risk.get("max_leverage"),
        "weights_mode": constructor.get("weights_mode"),
        "is_dynamic": "fixed" not in weights_file,
        "start_date": testing.get("start_date"),
        "end_date": testing.get("end_date"),
    }


def load_experiment_catalog(experiments_root: str | Path) -> pd.DataFrame:
    root = Path(experiments_root)
    summary_path = root / "experiments_summary.csv"
    summary = pd.read_csv(summary_path).set_index("run_key") if summary_path.exists() else pd.DataFrame()

    records: list[dict] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        run_key = folder.name
        required_files = [
            folder / "analytics.json",
            folder / "config.yaml",
            folder / "portfolio_nav.csv",
            folder / "trades.csv",
        ]
        if not all(path.exists() for path in required_files):
            continue

        with (folder / "analytics.json").open(encoding="utf-8") as handle:
            analytics = json.load(handle)

        config_values = _fallback_config_values(folder / "config.yaml")
        if not summary.empty and run_key in summary.index:
            summary_row = summary.loc[run_key]
            schema = summary_row.get("schema")
            n_positions = summary_row.get("n_positions")
            target_vol = summary_row.get("target_vol")
            max_leverage = summary_row.get("max_leverage")
            weights_mode = summary_row.get("weights_mode")
            training_years = summary_row.get("training_years")
            is_dynamic = summary_row.get("is_dynamic", config_values["is_dynamic"])
            status = summary_row.get("status", "Completed")
        else:
            schema = run_key.split("_")[0]
            n_positions = config_values["n_positions"]
            target_vol = config_values["target_vol"]
            max_leverage = config_values["max_leverage"]
            weights_mode = config_values["weights_mode"]
            training_years = 2 if run_key.endswith("_2y") else 4
            is_dynamic = config_values["is_dynamic"]
            status = "Completed"

        if isinstance(is_dynamic, str):
            is_dynamic = is_dynamic.strip().lower() in {"true", "1", "yes", "si", "sí"}

        nav = pd.read_csv(folder / "portfolio_nav.csv", index_col=0, usecols=[0, 1])
        nav_values = pd.to_numeric(nav["nav"], errors="coerce").dropna()
        returns = nav_values.pct_change()
        total_return = nav_values.iloc[-1] / nav_values.iloc[0] - 1 if len(nav_values) > 1 else math.nan

        records.append(
            {
                "run_key": run_key,
                "schema": str(schema),
                "schema_label": SCHEMA_LABELS.get(str(schema), str(schema).replace("_", " ")),
                "n_positions": int(n_positions),
                "target_vol": float(target_vol),
                "max_leverage": float(max_leverage),
                "weights_mode": str(weights_mode),
                "weights_mode_label": WEIGHTS_MODE_LABELS.get(str(weights_mode), str(weights_mode)),
                "training_years": int(training_years),
                "is_dynamic": bool(is_dynamic),
                "status": str(status),
                "cagr": float(analytics["cagr"]),
                "annual_volatility": float(analytics["annual_volatility"]),
                "sharpe_ratio": float(analytics["sharpe_ratio"]),
                "max_drawdown": float(analytics["max_drawdown"]),
                "calmar_ratio": float(analytics["calmar_ratio"]),
                "sortino_ratio": _sortino_ratio(returns),
                "monthly_win_rate_pct": float(analytics["monthly_win_rate_pct"]),
                "total_trading_days": int(analytics["total_trading_days"]),
                "total_trades": _csv_record_count(folder / "trades.csv"),
                "total_return": float(total_return),
                "final_nav": float(nav_values.iloc[-1]) if not nav_values.empty else math.nan,
                "start_date": config_values["start_date"],
                "end_date": config_values["end_date"],
            }
        )

    catalog = pd.DataFrame.from_records(records)
    if catalog.empty:
        return catalog
    return catalog.sort_values(
        ["cagr", "sharpe_ratio", "run_key"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def load_nav_series(experiments_root: str | Path, run_keys: Iterable[str]) -> pd.DataFrame:
    root = Path(experiments_root)
    frames: list[pd.DataFrame] = []
    for run_key in run_keys:
        path = root / run_key / "portfolio_nav.csv"
        nav = pd.read_csv(path, index_col=0, usecols=[0, 1])
        nav.index = pd.to_datetime(nav.index, errors="coerce")
        nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
        nav = nav.dropna(subset=["nav"])
        if nav.empty:
            continue
        base = nav["nav"].iloc[0]
        nav["base_100"] = nav["nav"] / base * 100
        nav["experiment"] = run_key
        frames.append(nav.reset_index(names="date")[["date", "experiment", "nav", "base_100"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_weight_series(experiments_root: str | Path, run_keys: Iterable[str]) -> pd.DataFrame:
    root = Path(experiments_root)
    frames: list[pd.DataFrame] = []
    for run_key in run_keys:
        path = root / run_key / "dqi_active_weights_evolution.csv"
        if not path.exists():
            continue
        weights = pd.read_csv(path)
        weights["date"] = pd.to_datetime(weights["date"], errors="coerce")
        weights["experiment"] = run_key
        frames.append(weights)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def factor_weights_long(weights: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in FACTOR_LABELS if column in weights.columns]
    if weights.empty or not columns:
        return pd.DataFrame()
    long = weights.melt(
        id_vars=["date", "experiment"],
        value_vars=columns,
        var_name="factor_key",
        value_name="weight",
    )
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce")
    long["Factor"] = long["factor_key"].map(FACTOR_LABELS)
    return long.dropna(subset=["date", "weight"])


def annual_subfactor_weights(weights: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SUBFACTOR_LABELS if column in weights.columns]
    if weights.empty or not columns:
        return pd.DataFrame()
    long = weights.melt(
        id_vars=["date", "experiment"],
        value_vars=columns,
        var_name="subfactor_key",
        value_name="weight",
    )
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce")
    long["year"] = long["date"].dt.year
    annual = (
        long.dropna(subset=["year", "weight"])
        .groupby(["experiment", "year", "subfactor_key"], as_index=False)["weight"]
        .mean()
    )
    annual["Subfactor"] = annual["subfactor_key"].map(SUBFACTOR_LABELS)
    annual["year"] = annual["year"].astype(int)
    return annual


def top_subfactor_timeline(weights: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return annual mean weights for the highest-weighted subfactors."""
    columns = [column for column in SUBFACTOR_LABELS if column in weights.columns]
    if weights.empty or not columns:
        return pd.DataFrame()

    long = weights.melt(
        id_vars=["date", "experiment"],
        value_vars=columns,
        var_name="subfactor_key",
        value_name="weight",
    )
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce")
    long = long.dropna(subset=["date", "weight"])
    if long.empty:
        return pd.DataFrame()

    mean_weights = long.groupby("subfactor_key")["weight"].mean()
    top_keys = mean_weights.nlargest(min(top_n, len(mean_weights))).index.tolist()
    rank_map = {key: rank for rank, key in enumerate(top_keys, start=1)}

    selected = long[long["subfactor_key"].isin(top_keys)].copy()
    selected["year"] = selected["date"].dt.year
    timeline = selected.groupby(["year", "subfactor_key"], as_index=False)["weight"].mean()
    timeline["date"] = pd.to_datetime(timeline["year"].astype(str) + "-12-31")
    timeline["Subfactor"] = timeline["subfactor_key"].map(SUBFACTOR_LABELS)
    timeline["Factor"] = timeline["subfactor_key"].map(SUBFACTOR_FACTORS)
    timeline["rank"] = timeline["subfactor_key"].map(rank_map).astype(int)
    timeline["mean_weight"] = timeline["subfactor_key"].map(mean_weights)
    return timeline.sort_values(["date", "rank"]).reset_index(drop=True)


def annual_subfactor_ranking(weights: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank the highest mean subfactor weights independently within each year."""
    columns = [column for column in SUBFACTOR_LABELS if column in weights.columns]
    if weights.empty or not columns:
        return pd.DataFrame()

    long = weights.melt(
        id_vars=["date", "experiment"],
        value_vars=columns,
        var_name="subfactor_key",
        value_name="weight",
    )
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce")
    long["year"] = long["date"].dt.year
    annual = (
        long.dropna(subset=["year", "weight"])
        .groupby(["year", "subfactor_key"], as_index=False)["weight"]
        .mean()
    )
    annual["Subfactor"] = annual["subfactor_key"].map(SUBFACTOR_LABELS)
    annual["Factor"] = annual["subfactor_key"].map(SUBFACTOR_FACTORS)
    annual = annual.sort_values(
        ["year", "weight", "Subfactor"],
        ascending=[True, False, True],
    )
    annual["rank"] = annual.groupby("year").cumcount() + 1
    return annual[annual["rank"] <= top_n].reset_index(drop=True)


def annual_subfactor_comparison(weights: pd.DataFrame) -> pd.DataFrame:
    """Return annual subfactor means without averaging across experiments."""
    columns = [column for column in SUBFACTOR_LABELS if column in weights.columns]
    if weights.empty or not columns:
        return pd.DataFrame()

    long = weights.melt(
        id_vars=["date", "experiment"],
        value_vars=columns,
        var_name="subfactor_key",
        value_name="weight",
    )
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce")
    long["year"] = long["date"].dt.year
    annual = (
        long.dropna(subset=["year", "weight"])
        .groupby(["experiment", "year", "subfactor_key"], as_index=False)["weight"]
        .mean()
    )
    annual["year"] = annual["year"].astype(int)
    annual["Subfactor"] = annual["subfactor_key"].map(SUBFACTOR_LABELS)
    annual["Factor"] = annual["subfactor_key"].map(SUBFACTOR_FACTORS)
    return annual.sort_values(
        ["year", "experiment", "weight", "Subfactor"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def build_short_label(row: pd.Series) -> str:
    dynamics = "Dinámicos" if bool(row["is_dynamic"]) else "Fijos"
    return (
        f"{row['schema_label']} · N{int(row['n_positions'])} · "
        f"{row['target_vol']:.0%} vol · {row['max_leverage']:.1f}x · "
        f"{row['weights_mode_label']} · {int(row['training_years'])}a · {dynamics}"
    )
