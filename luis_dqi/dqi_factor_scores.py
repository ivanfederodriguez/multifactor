"""
DQI Factor Scoring Engine
Pipeline: Winsorize → Sector-neutral z-score → Weighted composite → Quintile rating

Audit fixes applied:
  - #4:  Removed double z-scoring of factor z-scores (was re-z-scoring already-z-scored values)
  - #7:  Added factor re-normalization cap (MAX_FACTOR_WEIGHT_MULT) to prevent weight drift
  - #16: Raised min sample for winsorization (MIN_WINSORIZE_SAMPLE)
  - #17: Fixed percentile tie handling using midpoint ranking
  - Sector valid count now checks valid observations against MIN_SECTOR_SIZE (not just 2)

Usage:
    python dqi_factor_scores.py
    # (reads dqi_cache.json, writes dqi_scores.json)
"""
import json
import os
import sys

import numpy as np

from .dqi_config import (
    FACTOR_WEIGHTS, SUBFACTOR_WEIGHTS, INVERT_SUBFACTORS,
    WINSORIZE_PCT, MIN_WINSORIZE_SAMPLE, MIN_SECTOR_SIZE,
    MIN_COMPOSITE_COVERAGE, MAX_FACTOR_WEIGHT_MULT,
    RATING_BANDS, CACHE_FILES,
)


# ── Mapping: cache key → sub-factor name ─────────────────────────────────────
SUBFACTOR_FIELD_MAP = {
    # Value
    'ev_ebitda':       'ev_to_t12m_ebitda',
    'fcf_yield':       'fcf_yield_with_cur_entp_valu',
    'earnings_yield':  'earn_yld_hist',
    # Quality
    'roic':            'return_on_inv_capital',
    'net_debt_ebitda': 'net_debt_to_ebitda',
    'gross_margin':    'gross_margin',
    # Growth
    'rev_growth_ntm':  'rev_growth_ntm',
    'eps_growth_fy1':  'eps_growth_fy1',
    'rd_intensity':    'rd_intensity',
    # Momentum
    'return_12m_ex1':  'return_12m_ex1',
    'return_6m':       'return_6m',
    'return_12m':      'return_12m',
    # Revisions
    'eps_revision_1m': 'best_eps_4wk_pct_chg',
    # Profitability momentum
    'delta_roic':         'delta_roic',
    'delta_gross_margin': 'delta_gross_margin',
    'delta_oper_margin':  'delta_oper_margin',
}

UNKNOWN_SECTOR = 'Unknown'


def winsorize(arr, pct_low=WINSORIZE_PCT[0], pct_high=WINSORIZE_PCT[1]):
    """Winsorize a numpy array at pct_low/pct_high percentiles.
    FIX #16: Skip winsorization if fewer than MIN_WINSORIZE_SAMPLE valid values,
    since percentile estimates on tiny samples are unreliable.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_WINSORIZE_SAMPLE:
        return arr
    lo = np.nanpercentile(arr, pct_low * 100)
    hi = np.nanpercentile(arr, pct_high * 100)
    return np.clip(arr, lo, hi)


def sector_neutral_zscore(values, sectors, min_sector_size=MIN_SECTOR_SIZE):
    """
    Compute sector-neutral z-scores.
    values:  np.array of floats (may contain NaN)
    sectors: list of sector strings (same length as values)
    Returns: np.array of z-scores.

    FIX: Valid observation count is now checked against min_sector_size,
    not just < 2. A sector with 5 members but only 2 valid values will
    fall back to universe stats instead of producing noisy z-scores.
    """
    z = np.full(len(values), np.nan)
    unique_sectors = set(s for s in sectors if s and s != UNKNOWN_SECTOR)

    for sector in unique_sectors:
        idx = [i for i, s in enumerate(sectors) if s == sector]
        if len(idx) < min_sector_size:
            continue
        sect_vals = values[idx]
        valid_mask = ~np.isnan(sect_vals)
        # FIX: require min_sector_size valid observations, not just 2
        if valid_mask.sum() < min_sector_size:
            continue
        mean = np.nanmean(sect_vals)
        std  = np.nanstd(sect_vals)
        if std < 1e-10:
            z[idx] = 0.0
        else:
            for i in idx:
                if not np.isnan(values[i]):
                    z[i] = (values[i] - mean) / std

    # Any remaining NaN sector or sector too small: use full-universe stats
    no_z = np.isnan(z) & ~np.isnan(values)
    if no_z.sum() > 0:
        universe_mean = np.nanmean(values)
        universe_std  = np.nanstd(values)
        if universe_std > 1e-10:
            for i in np.where(no_z)[0]:
                z[i] = (values[i] - universe_mean) / universe_std
        else:
            z[no_z] = 0.0

    return z


def assign_rating(percentile: float) -> str:
    """Map a percentile rank (0-1) to a rating string."""
    for threshold, rating in RATING_BANDS:
        if percentile >= threshold:
            return rating
    return 'POOR'


def score_universe(cache: dict, factor_weights=None) -> dict:
    """
    Score all tickers in the cache.

    Args:
        cache: dqi_cache.json contents — {ticker: {...fields...}}
        factor_weights: override FACTOR_WEIGHTS (for regime switching)

    Returns:
        scores: {ticker: {composite_z, factor_z_scores, sub_factor_z_scores,
                           percentile, rating, sector, name, price}}
    """
    if factor_weights is None:
        factor_weights = FACTOR_WEIGHTS

    tickers = list(cache.keys())
    n = len(tickers)
    if n == 0:
        return {}

    sectors = [cache[t].get('gics_sub_industry_name') or UNKNOWN_SECTOR for t in tickers]

    # ── Step 1: Extract and winsorize each sub-factor ──────────────────────
    # sub_factor_raw[subfactor_name] = np.array of raw values (with NaN)
    sub_factor_raw = {}
    for subfactor, cache_key in SUBFACTOR_FIELD_MAP.items():
        raw = np.array([
            float(cache[t].get(cache_key, np.nan))
            if cache[t].get(cache_key) is not None and cache[t].get(cache_key) != ''
            else np.nan
            for t in tickers
        ], dtype=float)
        # Invert where lower = better
        if subfactor in INVERT_SUBFACTORS:
            raw = -raw
        sub_factor_raw[subfactor] = winsorize(raw)

    # ── Step 2: Sector-neutral z-score per sub-factor ──────────────────────
    sub_factor_z = {}
    for subfactor, raw in sub_factor_raw.items():
        sub_factor_z[subfactor] = sector_neutral_zscore(raw, sectors)

    # ── Step 3: Factor z-scores (weighted avg of sub-factor z-scores) ──────
    # FIX #4: Removed double z-scoring. The weighted average of sector-neutral
    # z-scores is already sector-neutral and on a comparable scale. The previous
    # re-z-scoring distorted rankings by compressing/expanding factors with
    # different coverage rates, systematically favouring large-cap stocks with
    # complete data.
    factor_z = {}
    for factor, subfactors in SUBFACTOR_WEIGHTS.items():
        weights = subfactors
        composite = np.zeros(n)
        weight_sum = np.zeros(n)
        for sf, w in weights.items():
            z = sub_factor_z.get(sf, np.full(n, np.nan))
            valid = ~np.isnan(z)
            composite[valid] += w * z[valid]
            weight_sum[valid] += w
        # Normalize by actual weight used (handles NaN sub-factors)
        valid_mask = weight_sum > 0.10  # need at least 10% weight coverage
        f_z = np.full(n, np.nan)
        f_z[valid_mask] = composite[valid_mask] / weight_sum[valid_mask]
        # FIX #4: NO re-z-scoring here — weighted avg of z-scores is already comparable
        factor_z[factor] = f_z

    # ── Step 4: Composite z-score ──────────────────────────────────────────
    # FIX #7: Cap factor re-normalization so no factor exceeds
    # MAX_FACTOR_WEIGHT_MULT × its base weight. This prevents stocks missing
    # momentum/revisions from having Value+Quality inflated to ~50% each.
    composite_z = np.zeros(n)
    composite_weight_sum = np.zeros(n)
    for factor, w in factor_weights.items():
        fz = factor_z.get(factor, np.full(n, np.nan))
        valid = ~np.isnan(fz)
        composite_z[valid] += w * fz[valid]
        composite_weight_sum[valid] += w

    insufficient = composite_weight_sum < MIN_COMPOSITE_COVERAGE
    composite_z_final = np.full(n, np.nan)
    valid_mask = ~insufficient

    for i in np.where(valid_mask)[0]:
        raw_sum = composite_weight_sum[i]
        renorm_factor = 1.0 / raw_sum
        # Check if any factor would exceed the cap after re-normalization
        max_allowed_renorm = MAX_FACTOR_WEIGHT_MULT
        if renorm_factor > max_allowed_renorm:
            # Too much weight drift — mark as insufficient
            composite_z_final[i] = np.nan
        else:
            composite_z_final[i] = composite_z[i] / raw_sum

    # ── Step 5: Percentile and rating ─────────────────────────────────────
    # FIX #17: Use midpoint ranking for ties. Stocks with the same composite_z
    # now get the average of their left and right positions in the sorted list.
    valid_tickers = [i for i, v in enumerate(composite_z_final) if not np.isnan(v)]
    valid_scores  = [composite_z_final[i] for i in valid_tickers]
    sorted_scores = np.array(sorted(valid_scores))
    n_valid       = len(sorted_scores)

    def get_percentile(z):
        if np.isnan(z):
            return np.nan
        if n_valid <= 1:
            return 0.5
        # Midpoint ranking: average of leftmost and rightmost positions
        left = int(np.searchsorted(sorted_scores, z, side='left'))
        right = int(np.searchsorted(sorted_scores, z, side='right'))
        midpoint = (left + right - 1) / 2.0
        return midpoint / max(n_valid - 1, 1)

    # ── Step 6: Build output dict ──────────────────────────────────────────
    scores = {}
    for i, ticker in enumerate(tickers):
        czf = composite_z_final[i]
        pct = get_percentile(czf)
        rating = assign_rating(pct) if not np.isnan(pct) else 'N/A'

        ticker_data = cache[ticker]
        scores[ticker] = {
            'composite_z':  float(czf) if not np.isnan(czf) else None,
            'percentile':   float(pct) if not np.isnan(pct) else None,
            'rating':       rating,
            'sector':       ticker_data.get('gics_sub_industry_name', UNKNOWN_SECTOR),
            'name':         ticker_data.get('name', ''),
            'price':        ticker_data.get('px_last'),
            'beta':         ticker_data.get('beta_adjusted'),
            'mkt_cap':      ticker_data.get('cur_mkt_cap'),
            'factor_z': {
                f: float(factor_z[f][i]) if not np.isnan(factor_z[f][i]) else None
                for f in factor_z
            },
            'subfactor_z': {
                sf: float(sub_factor_z[sf][i]) if not np.isnan(sub_factor_z[sf][i]) else None
                for sf in sub_factor_z
            },
            'raw': {
                'ev_ebitda':      ticker_data.get('ev_to_t12m_ebitda'),
                'fcf_yield':      ticker_data.get('fcf_yield_with_cur_entp_valu'),
                'roic':           ticker_data.get('return_on_inv_capital'),
                'gross_margin':   ticker_data.get('gross_margin'),
                'rev_growth_ntm': ticker_data.get('rev_growth_ntm'),
                'eps_growth_fy1': ticker_data.get('eps_growth_fy1'),
                'return_12m_ex1': ticker_data.get('return_12m_ex1'),
                'return_6m':      ticker_data.get('return_6m'),
                'eps_revision':   ticker_data.get('best_eps_4wk_pct_chg'),
            },
            'pull_date': ticker_data.get('pull_date'),
        }

    return scores


def main() -> dict:
    """Load cache, score the universe, and save results to cache/dqi_scores.json."""
    if not os.path.exists(CACHE_FILES['live']):
        print(f"ERROR: cache not found at {CACHE_FILES['live']}")
        print("Run dqi_blpapi_pull.py first.")
        sys.exit(1)

    print("Loading cache...")
    with open(CACHE_FILES['live']) as f:
        cache = json.load(f)
    print(f"  {len(cache)} tickers loaded")

    print("Scoring universe...")
    scores = score_universe(cache)

    # Stats
    rated = [s for s in scores.values() if s['rating'] != 'N/A']
    from collections import Counter
    rating_counts = Counter(s['rating'] for s in rated)
    print(f"  Scored {len(rated)} tickers")
    print(f"  Ratings: {dict(rating_counts)}")

    # Top 10 and bottom 10
    valid = [(t, s) for t, s in scores.items() if s['composite_z'] is not None]
    valid.sort(key=lambda x: x[1]['composite_z'], reverse=True)
    print("\n  Top 10 STRONG:")
    for t, s in valid[:10]:
        print(f"    {t:25s} {s['rating']:8s} z={s['composite_z']:+.2f}  {s['name'][:30]}")
    print("\n  Bottom 10 POOR:")
    for t, s in valid[-10:]:
        print(f"    {t:25s} {s['rating']:8s} z={s['composite_z']:+.2f}  {s['name'][:30]}")

    # Save
    with open(CACHE_FILES['scores'], 'w') as f:
        json.dump(scores, f)
    print(f"\nSaved scores to {CACHE_FILES['scores']}")
    return scores


if __name__ == '__main__':
    main()
