"""Canonical shared scorer: Luis DQI v2, without portfolio/backtest execution."""
from .dqi_factor_scores import score_universe

__version__ = "0.1.0"
__all__ = ["score_universe", "__version__"]
