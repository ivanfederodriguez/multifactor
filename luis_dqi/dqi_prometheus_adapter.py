"""Adapter from the PROMETHEUS historical score tape to the DQI v2 scorer.

The original DQI v2 implementation remains untouched.  This module only maps
PROMETHEUS column names into the cache contract accepted by
``dqi_factor_scores.score_universe`` and serialises the resulting monthly score
tape.  It deliberately does not fill missing R&D intensity with zero.

Example
-------
python dqi_prometheus_adapter.py \
    --source "/path/to/Proyecto Multifactor" \
    --output "/tmp/dqi_v2_prometheus_scores.parquet"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dqi_factor_scores import SUBFACTOR_FIELD_MAP, score_universe


SOURCE_COLUMN_MAP = {
    "ev_to_t12m_ebitda": "raw_ev_ebitda",
    "fcf_yield_with_cur_entp_valu": "raw_fcf_yield",
    "earn_yld_hist": "raw_earnings_yield",
    "return_on_inv_capital": "raw_roic",
    "net_debt_to_ebitda": "raw_net_debt_ebitda",
    "gross_margin": "raw_gross_margin",
    "rev_growth_ntm": "raw_rev_growth_ntm",
    "eps_growth_fy1": "raw_eps_growth_fy1",
    # PROMETHEUS' current historical score tape has no raw R&D intensity.
    "rd_intensity": None,
    "return_12m_ex1": "raw_return_12m_ex1",
    "return_6m": "raw_return_6m",
    "return_12m": "raw_return_12m",
    "best_eps_4wk_pct_chg": "raw_eps_revision_1m",
    "delta_roic": "raw_delta_roic",
    "delta_gross_margin": "raw_delta_gm",
    "delta_oper_margin": "raw_delta_em",
}

META_COLUMN_MAP = {
    "name": "name",
    "gics_sub_industry_name": "sub_industry",
    "px_last": "px_last",
    "cur_mkt_cap": "mkt_cap",
}

# The corrected f95e tape spells out the same two margin-delta fields.
# This is a schema alias only; no financial values or scorer logic change.
SOURCE_COLUMN_ALIASES = {
    "raw_delta_gm": "raw_delta_gross_margin",
    "raw_delta_em": "raw_delta_oper_margin",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(value: str | Path) -> Path:
    """Resolve either a parquet file or the PROMETHEUS workspace root."""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / "Output" / "dqi_historical_scores.parquet",
        path / "dqi_historical_scores.parquet",
        path / "Prometheus-git" / "Output" / "dqi_historical_scores.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find dqi_historical_scores.parquet below {path}. "
        f"Checked: {[str(item) for item in candidates]}"
    )


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_dqi_cache(month: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, dict[str, Any]]:
    """Translate one PROMETHEUS month into the unmodified DQI v2 cache schema."""
    cache: dict[str, dict[str, Any]] = {}
    for row in month.itertuples(index=False):
        source = row._asdict()
        symbol = str(source["symbol"])
        record: dict[str, Any] = {
            target: _clean_scalar(source.get(origin)) if origin else None
            for target, origin in SOURCE_COLUMN_MAP.items()
        }
        record.update(
            {
                target: _clean_scalar(source.get(origin))
                for target, origin in META_COLUMN_MAP.items()
            }
        )
        record["pull_date"] = as_of.date().isoformat()
        cache[symbol] = record
    return cache


def score_month(month: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Run the original DQI v2 scorer for one timestamp."""
    cache = build_dqi_cache(month, as_of)
    scored = score_universe(cache)
    records: list[dict[str, Any]] = []
    for symbol, item in scored.items():
        record: dict[str, Any] = {
            "date": as_of,
            "symbol": symbol,
            "composite_score": item["composite_z"],
            "composite_z": item["composite_z"],
            "percentile": item["percentile"],
            "rating": item["rating"],
            "sector": item["sector"],
            "name": item["name"],
            "price": item["price"],
            "mkt_cap": item["mkt_cap"],
        }
        record.update({f"factor_{key}": value for key, value in item["factor_z"].items()})
        record.update({f"subfactor_{key}": value for key, value in item["subfactor_z"].items()})
        records.append(record)
    return pd.DataFrame.from_records(records)


def adapt_score_tape(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert every month while recording coverage and provenance."""
    frame = pd.read_parquet(source)
    aliases_used = {}
    for canonical, alternative in SOURCE_COLUMN_ALIASES.items():
        if canonical not in frame.columns and alternative in frame.columns:
            frame[canonical] = frame[alternative]
            aliases_used[canonical] = alternative
    required = {"date", "symbol", *[c for c in SOURCE_COLUMN_MAP.values() if c], *META_COLUMN_MAP.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PROMETHEUS score tape is missing required columns: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["date", "symbol"], kind="mergesort")

    chunks: list[pd.DataFrame] = []
    for as_of, month in frame.groupby("date", sort=True):
        chunks.append(score_month(month, pd.Timestamp(as_of)))
    scores = pd.concat(chunks, ignore_index=True)
    scores = scores.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    mapped_coverage = {}
    for target, origin in SOURCE_COLUMN_MAP.items():
        mapped_coverage[target] = {
            "source_column": origin,
            "non_null_pct": 0.0 if origin is None else round(float(frame[origin].notna().mean() * 100), 4),
        }
    rated = scores["composite_score"].notna()
    manifest = {
        "schema_version": 1,
        "adapter": "dqi_prometheus_adapter",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_rows": int(len(frame)),
        "source_months": int(frame["date"].nunique()),
        "source_first_date": frame["date"].min().isoformat(),
        "source_last_date": frame["date"].max().isoformat(),
        "output_rows": int(len(scores)),
        "rated_rows": int(rated.sum()),
        "rated_pct": round(float(rated.mean() * 100), 4),
        "mapping": mapped_coverage,
        "source_column_aliases_used": aliases_used,
        "original_scorer_contract": sorted(SUBFACTOR_FIELD_MAP.values()),
        "known_gap": "rd_intensity is unavailable and remains null; DQI v2 renormalizes Growth across the two available signals.",
        "original_files_sha256": {
            "dqi_factor_scores.py": sha256_file(Path(__file__).with_name("dqi_factor_scores.py")),
            "dqi_config.py": sha256_file(Path(__file__).with_name("dqi_config.py")),
        },
    }
    return scores, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="PROMETHEUS root or historical score parquet")
    parser.add_argument("--output", required=True, help="Destination parquet")
    args = parser.parse_args()

    source = resolve_source(args.source)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scores, manifest = adapt_score_tape(source)
    scores.to_parquet(output, index=False)
    manifest["output_path"] = str(output)
    manifest["output_sha256"] = sha256_file(output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **{k: manifest[k] for k in ("output_rows", "rated_rows", "rated_pct")}}, indent=2))


if __name__ == "__main__":
    main()
