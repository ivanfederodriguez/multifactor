"""Canonical Luis DQI v2 scorer, with optional external-qbacktest adapters."""
from .dqi_factor_scores import score_universe

__version__ = "0.2.0"
__all__ = ["score_universe", "__version__"]
