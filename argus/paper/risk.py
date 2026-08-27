"""Pre-trade risk validation for simulated paper orders.

Scoped-down version of the DRUVA risk-engine idiom (see
``DRUVA/backend/app/core/execution/risk_engine.py``) for a single-account
simulator with no concurrency/margin/broker concerns: checks run in a fixed
order and short-circuit on the first failure. No broker/execution code
exists here -- everything is a paper-money simulation.

Only ``side == "buy"`` orders exercise the capital/exposure checks (cash,
position-size, max-positions, duplicate-position) -- a sell only ever
reduces existing exposure, so those checks would be meaningless for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from argus.config.paper import PaperSettings

if TYPE_CHECKING:
    from argus.paper.engine import OrderIntent


@dataclass(frozen=True)
class RiskCheckResult:
    ok: bool
    failed_check: str | None
    detail: str


_OK = RiskCheckResult(ok=True, failed_check=None, detail="ok")


@dataclass(frozen=True)
class PaperRiskGate:
    """A snapshot of simulated portfolio state a single ``validate()`` call is judged against.

    Callers (``argus.paper.engine.queue_orders_for_picks``) build this fresh
    from the current cash/equity/open-position state before validating each
    candidate order.
    """

    settings: PaperSettings
    cash: float
    equity: float
    open_positions: frozenset[tuple[str, str]]  # (symbol, market_code) pairs currently open
    open_position_count: int

    def validate(self, order: OrderIntent) -> RiskCheckResult:
        """Run every check in order; return the first failure, or an ok result."""
        for check in (
            self._check_cash,
            self._check_position_size,
            self._check_max_positions,
            self._check_duplicate,
            self._check_qty_positive,
        ):
            result = check(order)
            if not result.ok:
                return result
        return _OK

    def _check_cash(self, order: OrderIntent) -> RiskCheckResult:
        if order.side != "buy":
            return _OK
        buffer = 1 + self.settings.slippage_bps / 10_000
        required = order.qty * order.ref_price * buffer
        if required > self.cash:
            return RiskCheckResult(
                ok=False,
                failed_check="cash_sufficient",
                detail=f"required {required:.2f} (incl. slippage buffer) exceeds "
                f"available cash {self.cash:.2f}",
            )
        return _OK

    def _check_position_size(self, order: OrderIntent) -> RiskCheckResult:
        if order.side != "buy":
            return _OK
        notional = order.qty * order.ref_price
        cap = self.equity * self.settings.position_size_pct / 100.0
        if notional > cap:
            return RiskCheckResult(
                ok=False,
                failed_check="position_size_limit",
                detail=f"notional {notional:.2f} exceeds {self.settings.position_size_pct}% "
                f"of equity cap {cap:.2f}",
            )
        return _OK

    def _check_max_positions(self, order: OrderIntent) -> RiskCheckResult:
        if order.side != "buy":
            return _OK
        key = (order.symbol, order.market_code)
        if key in self.open_positions:
            return _OK  # adding to an already-open position doesn't consume a new slot
        if self.open_position_count >= self.settings.max_positions:
            return RiskCheckResult(
                ok=False,
                failed_check="max_positions",
                detail=f"open position count {self.open_position_count} >= "
                f"max {self.settings.max_positions}",
            )
        return _OK

    def _check_duplicate(self, order: OrderIntent) -> RiskCheckResult:
        if order.side != "buy":
            return _OK
        key = (order.symbol, order.market_code)
        if key in self.open_positions:
            return RiskCheckResult(
                ok=False,
                failed_check="duplicate_position",
                detail=f"an open position already exists for {order.symbol}/{order.market_code}",
            )
        return _OK

    def _check_qty_positive(self, order: OrderIntent) -> RiskCheckResult:
        if order.qty <= 0:
            return RiskCheckResult(ok=False, failed_check="qty_positive", detail="qty must be > 0")
        return _OK
