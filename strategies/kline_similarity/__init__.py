"""Historical stock K-line pattern similarity search.

The package intentionally keeps the search engine independent from the data
store.  ``ParquetDataProvider`` is included for this repository's local TDX
derived data, while applications can implement ``MarketDataProvider`` for a
database or an API.
"""

from .engine import (
    KlineSimilarityEngine,
    MarketDataProvider,
    ParquetDataProvider,
    SimilarityConfig,
    SimilarityMatch,
    aggregate_bars,
)


def build_visual_results(*args, **kwargs):
    """Lazy import wrapper, keeping ``python -m ...visualize`` warning-free."""
    from .visualize import build_visual_results as _build_visual_results
    return _build_visual_results(*args, **kwargs)

__all__ = [
    "KlineSimilarityEngine",
    "MarketDataProvider",
    "ParquetDataProvider",
    "SimilarityConfig",
    "SimilarityMatch",
    "aggregate_bars",
    "build_visual_results",
]
