"""
DQI Multi-Factor Equity Framework - Configuration
Single source of truth for all parameters.
"""
import os

# ── Factor weights (must sum to 1.0) ────────────────────────────────────────
FACTOR_WEIGHTS = {
    'value':         0.20,
    'quality':       0.20,
    'growth':        0.15,
    'momentum':      0.20,
    'revisions':     0.10,
    'prof_momentum': 0.15,
}

# ── Sub-factor weights per factor (must sum to 1.0 within each) ─────────────
SUBFACTOR_WEIGHTS = {
    'value': {
        'ev_ebitda':      0.40,  # inverted: lower = better
        'fcf_yield':      0.30,
        'earnings_yield': 0.30,
    },
    'quality': {
        'roic':           0.40,
        'net_debt_ebitda':0.30,  # inverted: lower = better
        'gross_margin':   0.30,
    },
    'growth': {
        'rev_growth_ntm': 0.40,
        'eps_growth_fy1': 0.40,
        'rd_intensity':   0.20,
    },
    'momentum': {
        'return_12m_ex1': 0.40,  # Jegadeesh-Titman
        'return_6m':      0.30,
        'return_12m':     0.30,
    },
    'revisions': {
        'eps_revision_1m': 1.00,
    },
    'prof_momentum': {
        'delta_roic':         0.50,
        'delta_gross_margin': 0.30,
        'delta_oper_margin':  0.20,
    },
}

# Sub-factors where LOWER value = BETTER (will be negated before z-scoring)
INVERT_SUBFACTORS = {'ev_ebitda', 'net_debt_ebitda'}

# ── Scoring pipeline parameters ──────────────────────────────────────────────
WINSORIZE_PCT = (0.01, 0.99)   # 1st/99th percentile
MIN_WINSORIZE_SAMPLE = 20      # min valid values to winsorize; below this, skip winsorization
MIN_SECTOR_SIZE = 5            # min stocks in sector to use sector stats; else use full universe
MIN_COMPOSITE_COVERAGE = 0.70  # min factor weight coverage to compute composite (was 0.60)
MAX_FACTOR_WEIGHT_MULT = 1.50  # max allowed re-normalization multiplier per factor (caps drift)

# ── Quintile ratings (cumulative percentile thresholds) ─────────────────────
# top 20% = STRONG, bottom 20% = POOR
RATING_BANDS = [
    (0.80, 'STRONG'),
    (0.60, 'GOOD'),
    (0.40, 'NEUTRAL'),
    (0.20, 'WEAK'),
    (0.00, 'POOR'),
]

# ── Regime-based factor weights ──────────────────────────────────────────────
VIX_REGIME_THRESHOLDS = {'CALM': 15, 'NORMAL': 25}  # STRESS = above 25

REGIME_FACTOR_WEIGHTS = {
    'CALM': {
        'value': 0.15, 'quality': 0.15, 'growth': 0.20,
        'momentum': 0.25, 'revisions': 0.10, 'prof_momentum': 0.15,
    },
    'NORMAL': {  # base weights (same as FACTOR_WEIGHTS above)
        'value': 0.20, 'quality': 0.20, 'growth': 0.15,
        'momentum': 0.20, 'revisions': 0.10, 'prof_momentum': 0.15,
    },
    'STRESS': {
        'value': 0.25, 'quality': 0.30, 'growth': 0.10,
        'momentum': 0.10, 'revisions': 0.10, 'prof_momentum': 0.15,
    },
}

# ── Volatility signal thresholds ─────────────────────────────────────────────
VOL_THRESHOLDS = {
    'iv_hv_cheap':     0.85,   # IV/HV below this = cheap options
    'iv_hv_normal_hi': 1.20,   # IV/HV above this = expensive options
    'vol_shock_hi':    1.30,   # HV30/HV90 above this = near-term spike
    'vol_shock_lo':    0.80,   # HV30/HV90 below this = vol compressing
    'ts_inverted':     0.90,   # IV3M/IV1M below this = strong inversion (stress)
    'ts_contango':     1.05,   # IV3M/IV1M above this = normal contango
    'ts_invert_mild':  0.95,   # IV3M/IV1M below this = mild backwardation
}

# Short setup signal thresholds
SHORT_STRONG_IV_HV = 1.20

# ── Position sizing ───────────────────────────────────────────────────────────
POSITION_SIZING = {
    'max_weight':              0.20,   # 20% cap per name
    'iv_hv_cheap_mult':        1.25,   # cheap vol + strong = oversize
    'iv_hv_expensive_mult':    0.75,   # expensive vol = undersize
    'iv_hv_cheap_threshold':   0.90,
    'iv_hv_expensive_threshold': 1.20,
    'etf_exclusions':          {'RSP', 'ARGT'},  # excluded from active sizing
}

# ── Portfolio construction ────────────────────────────────────────────────────
LONG_QUINTILE_PCT  = 0.20   # top 20% = long universe
SHORT_QUINTILE_PCT = 0.20   # bottom 20% = short universe

# ── Bloomberg pull settings ──────────────────────────────────────────────────
BBG_HOST = 'localhost'
BBG_PORT = 8194
BBG_BATCH_SIZE = 50          # tickers per refdata request batch
BBG_TIMEOUT_MS = 60_000      # 60-second request timeout

# ── Backtest settings ─────────────────────────────────────────────────────────
BACKTEST_YEARS = 10          # years of history to pull
BACKTEST_N_LONG  = None      # None = top quintile; or integer
BACKTEST_N_SHORT = None      # None = bottom quintile
HV_ROLLING_MONTHS = 3        # months for rolling HV in backtest
VOL_SHOCK_HV_SHORT = 3       # HV short window for vol shock (was 2; min 3 for meaningful std)
VOL_SHOCK_HV_LONG  = 6       # HV_6M window for backtest vol shock
MIN_HV_OBSERVATIONS = 3      # min monthly returns needed to compute HV (ddof=1 needs ≥3)
STRESS_FILTER_THRESHOLD = 1.40  # BT4: exclude long stocks with shock > this
HV_WEIGHT_CAP = 0.05         # 5% per-name cap in vol-scaled strategies
TRANSACTION_COST_BPS = 10    # one-way transaction cost in basis points (applied to turnover)
DELISTING_RETURN = -0.30     # assumed return for stocks with missing exit prices
REBALANCE_LAG_DAYS = 1       # days between scoring date and return-start date

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, 'cache')
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'OneDrive', 'DQI', 'Multistrategy Fund - version 2')

CACHE_FILES = {
    'live':       os.path.join(CACHE_DIR, 'dqi_cache.json'),
    'scores':     os.path.join(CACHE_DIR, 'dqi_scores.json'),
    'vol':        os.path.join(CACHE_DIR, 'dqi_vol_signals.json'),
    'historical': os.path.join(CACHE_DIR, 'dqi_historical_cache.json'),
    'vix':        os.path.join(CACHE_DIR, 'vix_history.json'),
    'tickers':    os.path.join(CACHE_DIR, 'sp500_tickers.json'),
}
