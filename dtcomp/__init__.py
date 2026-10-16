"""CSI300 index-enhancement competition toolkit (shared library).

Layout
------
dtcomp.data      panel loading / caching / label construction
dtcomp.features  causal (<=120 trading-day look-back) feature engineering
dtcomp.cv        walk-forward splits + RankIC metrics
dtcomp.models    model zoo + training entry points

Hard invariants (competition rules)
-----------------------------------
* Point-in-time only: a feature at date ``t`` may use rows dated ``<= t``.
* Max look-back for market series: 120 trading days.
* Universe for a signal date is the ``Barra_factor`` code set of that date.
* Label: ``open(t+2)/open(t+1) - 1`` on adjusted prices.
"""

DATA_START = 20160104
DATA_END = 20221230

__all__ = ["DATA_START", "DATA_END"]
