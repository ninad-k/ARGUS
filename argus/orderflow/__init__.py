"""Orderflow & liquidity analytics (Task 13).

Everything in this package is derived from OHLCV bars (and, optionally, an
option chain snapshot) -- there is no tick-by-tick trade tape or L2
order-book feed anywhere in ARGUS. Every function here is therefore an
*approximation* of order flow, built from the coarser signal actually
available: bar-level open/high/low/close/volume. See each module's docstring
for exactly what's approximated and how.
"""
