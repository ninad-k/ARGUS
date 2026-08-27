"""``PaperRiskGate.validate`` -- each check pass/fail and short-circuit ordering."""

from __future__ import annotations

from argus.config.paper import PaperSettings
from argus.paper.engine import OrderIntent
from argus.paper.risk import PaperRiskGate


def _settings(**overrides: object) -> PaperSettings:
    return PaperSettings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def _gate(
    *,
    cash: float = 100_000.0,
    equity: float = 100_000.0,
    open_positions: frozenset[tuple[str, str]] = frozenset(),
    open_position_count: int = 0,
    settings: PaperSettings | None = None,
) -> PaperRiskGate:
    return PaperRiskGate(
        settings=settings or _settings(),
        cash=cash,
        equity=equity,
        open_positions=open_positions,
        open_position_count=open_position_count,
    )


def _buy(
    symbol: str = "AAPL", market: str = "US_NASDAQ", qty: float = 10, price: float = 100.0
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol, market_code=market, side="buy", qty=qty, ref_price=price, pick_id=None
    )


def test_validate_passes_a_well_formed_buy() -> None:
    gate = _gate(cash=100_000.0, equity=100_000.0)
    result = gate.validate(_buy(qty=10, price=100.0))
    assert result.ok is True
    assert result.failed_check is None


def test_check_cash_fails_when_required_exceeds_available() -> None:
    gate = _gate(cash=500.0, equity=100_000.0)
    result = gate.validate(_buy(qty=10, price=100.0))
    assert result.ok is False
    assert result.failed_check == "cash_sufficient"


def test_check_cash_includes_slippage_buffer() -> None:
    # cash exactly equal to the *unbuffered* notional should still fail once
    # the slippage buffer is added on top.
    settings = _settings(slippage_bps=100)  # 1%
    gate = _gate(cash=1_000.0, equity=1_000_000.0, settings=settings)
    result = gate.validate(_buy(qty=10, price=100.0))  # notional = 1000.0 exactly
    assert result.ok is False
    assert result.failed_check == "cash_sufficient"


def test_check_position_size_fails_when_notional_exceeds_cap() -> None:
    settings = _settings(position_size_pct=5.0)
    # equity 10,000 -> cap 500; order notional = 10*100 = 1000 > 500
    gate = _gate(cash=1_000_000.0, equity=10_000.0, settings=settings)
    result = gate.validate(_buy(qty=10, price=100.0))
    assert result.ok is False
    assert result.failed_check == "position_size_limit"


def test_check_max_positions_fails_for_a_new_symbol_at_cap() -> None:
    settings = _settings(max_positions=2)
    gate = _gate(
        cash=1_000_000.0,
        equity=1_000_000.0,
        settings=settings,
        open_positions=frozenset({("MSFT", "US_NASDAQ"), ("GOOG", "US_NASDAQ")}),
        open_position_count=2,
    )
    result = gate.validate(_buy(symbol="AAPL"))
    assert result.ok is False
    assert result.failed_check == "max_positions"


def test_check_max_positions_allows_adding_to_an_existing_symbol_at_cap() -> None:
    settings = _settings(max_positions=1)
    gate = _gate(
        cash=1_000_000.0,
        equity=1_000_000.0,
        settings=settings,
        open_positions=frozenset({("AAPL", "US_NASDAQ")}),
        open_position_count=1,
    )
    # AAPL already held -- adding to it must not trip max_positions, but it
    # DOES trip the duplicate-position check further down the chain.
    result = gate.validate(_buy(symbol="AAPL"))
    assert result.ok is False
    assert result.failed_check == "duplicate_position"


def test_check_duplicate_fails_for_symbol_already_open_same_market() -> None:
    gate = _gate(open_positions=frozenset({("AAPL", "US_NASDAQ")}), open_position_count=1)
    result = gate.validate(_buy(symbol="AAPL", market="US_NASDAQ"))
    assert result.ok is False
    assert result.failed_check == "duplicate_position"


def test_check_duplicate_passes_for_same_symbol_different_market() -> None:
    gate = _gate(open_positions=frozenset({("AAPL", "US_NASDAQ")}), open_position_count=1)
    result = gate.validate(_buy(symbol="AAPL", market="US_NYSE"))
    assert result.ok is True


def test_check_qty_positive_fails_for_zero_or_negative_qty() -> None:
    gate = _gate()
    result = gate.validate(_buy(qty=0))
    assert result.ok is False
    assert result.failed_check == "qty_positive"

    result_negative = gate.validate(_buy(qty=-5))
    assert result_negative.ok is False
    assert result_negative.failed_check == "qty_positive"


def test_sell_orders_skip_all_buy_specific_checks() -> None:
    # Cash is far too small, position size would blow the cap, max_positions
    # is already exceeded, and the symbol is a duplicate -- none of that
    # should matter for a sell; only qty>0 is checked.
    settings = _settings(max_positions=0, position_size_pct=0.001)
    gate = _gate(
        cash=0.0,
        equity=100.0,
        settings=settings,
        open_positions=frozenset({("AAPL", "US_NASDAQ")}),
        open_position_count=5,
    )
    sell = OrderIntent(
        symbol="AAPL", market_code="US_NASDAQ", side="sell", qty=10, ref_price=100.0, pick_id=None
    )
    result = gate.validate(sell)
    assert result.ok is True


def test_validate_short_circuits_on_first_failure_cash_before_position_size() -> None:
    # Both cash and position-size would fail; cash_sufficient runs first.
    settings = _settings(position_size_pct=0.001)
    gate = _gate(cash=1.0, equity=100.0, settings=settings)
    result = gate.validate(_buy(qty=10, price=100.0))
    assert result.ok is False
    assert result.failed_check == "cash_sufficient"
