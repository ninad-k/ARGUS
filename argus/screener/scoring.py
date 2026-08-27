"""Shared percentile-rank scoring helper for screener strategies.

Factored out of ``momentum.py`` (the first strategy to need it) so
``value.py``/``mean_reversion.py`` don't each carry their own copy.
"""

from __future__ import annotations


def percentile_rank(values: list[float], value: float) -> float:
    """Percentile rank of ``value`` within ``values``, 0-100 (100 = highest).

    A single-candidate ``values`` list (``n <= 1``) ranks at 100 -- a lone
    survivor of a strategy's gate has nothing to be relatively worse than.
    """
    n = len(values)
    if n <= 1:
        return 100.0
    below_or_equal = sum(1 for v in values if v <= value)
    return 100.0 * (below_or_equal - 1) / (n - 1)
