"""Pure FIFO economics derived from causally ordered confirmed fills."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from trading.domain._decimal import exact_decimal_product, exact_decimal_sum
from trading.domain._validation import require_clean_text, require_positive_decimal
from trading.domain.positions import (
    InvalidPositionTransition,
    Position,
    PositionEffect,
    PositionExitContext,
    PositionFill,
    PositionSide,
    new_position,
    reduce_position,
)

_BIGINT_MAX = (1 << 63) - 1
_EVENT_ID_MAX_LENGTH = 255


def _require_event_id(value: object) -> None:
    require_clean_text("event_id", value)
    if len(value) > _EVENT_ID_MAX_LENGTH:
        raise ValueError("event_id exceeds its durable storage limit")
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("event_id is not representable in durable storage")


def _decimal_payload_identity(value: Decimal) -> tuple[object, ...]:
    components = value.as_tuple()
    return (components.sign, components.digits, components.exponent)


def _datetime_payload_identity(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _require_utc_representable(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
        value.astimezone(timezone.utc)
    except TypeError as exc:
        raise TypeError(f"{name} cannot be represented in UTC") from exc
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented in UTC") from exc


def _exit_context_payload_identity(
    exit_context: PositionExitContext | None,
) -> object:
    if exit_context is None:
        return None
    return (
        exit_context.take_profit_profile,
        _decimal_payload_identity(exit_context.take_profit_fraction),
        _decimal_payload_identity(exit_context.stop_loss_fraction),
        (
            None
            if exit_context.trailing_stop_fraction is None
            else _decimal_payload_identity(exit_context.trailing_stop_fraction)
        ),
    )


def _position_fill_payload_identity(position_fill: PositionFill) -> tuple[object, ...]:
    instruction = position_fill.instruction
    intent = instruction.order_intent
    exit_context = instruction.exit_context
    fill = position_fill.fill
    return (
        instruction.position_key,
        instruction.effect,
        intent.client_order_id,
        intent.decision_id,
        intent.symbol,
        intent.side,
        _decimal_payload_identity(intent.quantity),
        intent.order_type,
        _decimal_payload_identity(intent.reference_price),
        intent.leverage,
        _datetime_payload_identity(intent.created_at),
        _exit_context_payload_identity(exit_context),
        fill.client_order_id,
        fill.venue_order_id,
        fill.trade_id,
        fill.symbol,
        fill.side,
        _decimal_payload_identity(fill.quantity),
        _decimal_payload_identity(fill.price),
        _decimal_payload_identity(fill.fee_amount),
        _datetime_payload_identity(fill.executed_at),
        fill.fee_asset,
    )


def _position_payload_identity(position: Position) -> tuple[object, ...]:
    return (
        position.position_key,
        position.symbol,
        position.side,
        position.leverage,
        _exit_context_payload_identity(position.exit_context),
        position.state,
        tuple(_position_fill_payload_identity(fill) for fill in position.fills),
    )


def _record_payload_identity(record: "PaperFillRecord") -> tuple[object, ...]:
    return (
        record.position_version,
        record.event_id,
        _position_fill_payload_identity(record.position_fill),
    )


def _lot_payload_identity(lot: "PaperCostLot") -> tuple[object, ...]:
    return (
        lot.fill_identity,
        _decimal_payload_identity(lot.remaining_quantity),
        _decimal_payload_identity(lot.entry_price),
    )


def _fee_payload_identity(fee: "PaperFeeTotal") -> tuple[object, ...]:
    return (fee.asset, _decimal_payload_identity(fee.amount))


class PaperLotMethod(str, Enum):
    """Supported cost-allocation methods for paper economics."""

    FIFO = "FIFO"


class InvalidPaperEconomicTransition(ValueError):
    """Raised when a fill contradicts the causal paper-economic history."""


@dataclass(frozen=True, slots=True)
class PaperFillRecord:
    """One confirmed position fill with its durable causal metadata."""

    position_version: int
    event_id: str
    position_fill: PositionFill

    def __post_init__(self) -> None:
        if isinstance(self.position_version, bool) or not isinstance(
            self.position_version, int
        ):
            raise TypeError("position_version must be an integer")
        if self.position_version < 1:
            raise ValueError("position_version must be positive")
        if self.position_version > _BIGINT_MAX:
            raise ValueError("position_version exceeds its durable storage limit")
        _require_event_id(self.event_id)
        if type(self.position_fill) is not PositionFill:
            raise TypeError("position_fill must be a PositionFill")
        _require_utc_representable(
            "instruction.created_at",
            self.position_fill.instruction.order_intent.created_at,
        )
        _require_utc_representable(
            "fill.executed_at",
            self.position_fill.fill.executed_at,
        )

    @property
    def fill_identity(self) -> tuple[str, str]:
        """Return the stable client-order and venue-trade identity."""
        return self.position_fill.identity

    @property
    def event_identity(self) -> tuple[str, str]:
        """Return the journal-scoped client-order and event identity."""
        return (self.position_fill.fill.client_order_id, self.event_id)


@dataclass(frozen=True, slots=True)
class PaperCostLot:
    """The unconsumed quantity of one causally ordered opening fill."""

    fill_identity: tuple[str, str]
    remaining_quantity: Decimal
    entry_price: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.fill_identity, tuple) or len(self.fill_identity) != 2:
            raise TypeError("fill_identity must be a two-item tuple")
        client_order_id, trade_id = self.fill_identity
        require_clean_text("fill_identity client_order_id", client_order_id)
        require_clean_text("fill_identity trade_id", trade_id)
        require_positive_decimal("remaining_quantity", self.remaining_quantity)
        require_positive_decimal("entry_price", self.entry_price)


@dataclass(frozen=True, slots=True)
class PaperFeeTotal:
    """An exact cumulative positive fee amount for one asset."""

    asset: str
    amount: Decimal

    def __post_init__(self) -> None:
        require_clean_text("asset", self.asset)
        require_positive_decimal("amount", self.amount)


@dataclass(frozen=True, slots=True)
class _DerivedEconomics:
    position: Position
    lots: tuple[PaperCostLot, ...]
    open_quantity: Decimal
    open_cost: Decimal
    gross_realized_pnl: Decimal
    fees: tuple[PaperFeeTotal, ...]


def _add_fee(
    totals: dict[str, Decimal],
    position_fill: PositionFill,
) -> None:
    fill = position_fill.fill
    if not fill.fee_amount:
        return
    if fill.fee_asset is None:
        raise AssertionError("validated positive fee lost its asset")
    totals[fill.fee_asset] = exact_decimal_sum(
        (totals.get(fill.fee_asset, Decimal("0")), fill.fee_amount)
    )


def _consume_fifo(
    lots: tuple[PaperCostLot, ...],
    position_fill: PositionFill,
    position_side: PositionSide,
) -> tuple[tuple[PaperCostLot, ...], Decimal]:
    remaining = position_fill.fill.quantity
    exit_price = position_fill.fill.price
    next_lots: list[PaperCostLot] = []
    realized_parts: list[Decimal] = []

    for lot in lots:
        if not remaining:
            next_lots.append(lot)
            continue

        consumed = min(remaining, lot.remaining_quantity)
        price_delta = (
            exact_decimal_sum((exit_price, lot.entry_price.copy_negate()))
            if position_side is PositionSide.LONG
            else exact_decimal_sum((lot.entry_price, exit_price.copy_negate()))
        )
        realized_parts.append(exact_decimal_product((consumed, price_delta)))
        remaining = exact_decimal_sum((remaining, consumed.copy_negate()))
        lot_remainder = exact_decimal_sum(
            (lot.remaining_quantity, consumed.copy_negate())
        )
        if lot_remainder:
            next_lots.append(
                PaperCostLot(
                    fill_identity=lot.fill_identity,
                    remaining_quantity=lot_remainder,
                    entry_price=lot.entry_price,
                )
            )

    if remaining:
        raise InvalidPaperEconomicTransition(
            "REDUCE_ONLY fill exceeds the available FIFO quantity"
        )
    return tuple(next_lots), exact_decimal_sum(tuple(realized_parts))


def _derive(records: tuple[PaperFillRecord, ...]) -> _DerivedEconomics:
    if not records:
        raise ValueError("paper economics requires at least one fill record")

    versions = tuple(record.position_version for record in records)
    if any(later <= earlier for earlier, later in zip(versions, versions[1:])):
        raise InvalidPaperEconomicTransition(
            "fill records must use strictly increasing position versions"
        )

    event_ids = tuple(record.event_identity for record in records)
    if len(event_ids) != len(set(event_ids)):
        raise InvalidPaperEconomicTransition("fill event identities must be unique")
    fill_ids = tuple(record.fill_identity for record in records)
    if len(fill_ids) != len(set(fill_ids)):
        raise InvalidPaperEconomicTransition("fill identities must be unique")

    first = records[0].position_fill
    try:
        position = new_position(first)
        first_fill = first.fill
        lots = (
            PaperCostLot(
                fill_identity=first.identity,
                remaining_quantity=first_fill.quantity,
                entry_price=first_fill.price,
            ),
        )
        gross_realized_pnl = Decimal("0")
        fee_totals: dict[str, Decimal] = {}
        _add_fee(fee_totals, first)
    except InvalidPositionTransition as exc:
        raise InvalidPaperEconomicTransition(
            "paper economics must start with an OPEN fill"
        ) from exc
    except ValueError as exc:
        raise InvalidPaperEconomicTransition(
            "first fill cannot be represented exactly"
        ) from exc

    for record in records[1:]:
        position_fill = record.position_fill
        prior_side = position.side
        try:
            next_position = reduce_position(position, position_fill)
        except InvalidPositionTransition as exc:
            raise InvalidPaperEconomicTransition(
                "confirmed fill contradicts paper economics"
            ) from exc
        except ValueError as exc:
            raise InvalidPaperEconomicTransition(
                "confirmed fill cannot be represented exactly"
            ) from exc

        if position_fill.instruction.effect is PositionEffect.OPEN:
            lots = lots + (
                PaperCostLot(
                    fill_identity=position_fill.identity,
                    remaining_quantity=position_fill.fill.quantity,
                    entry_price=position_fill.fill.price,
                ),
            )
        else:
            try:
                lots, realized = _consume_fifo(lots, position_fill, prior_side)
                gross_realized_pnl = exact_decimal_sum((gross_realized_pnl, realized))
            except ValueError as exc:
                raise InvalidPaperEconomicTransition(
                    "confirmed fill cannot be represented exactly"
                ) from exc

        try:
            _add_fee(fee_totals, position_fill)
        except ValueError as exc:
            raise InvalidPaperEconomicTransition(
                "confirmed fee cannot be represented exactly"
            ) from exc
        position = next_position

    try:
        lot_quantity = exact_decimal_sum(tuple(lot.remaining_quantity for lot in lots))
        open_cost = exact_decimal_sum(
            tuple(
                exact_decimal_product((lot.remaining_quantity, lot.entry_price))
                for lot in lots
            )
        )
    except ValueError as exc:
        raise InvalidPaperEconomicTransition(
            "paper economics cannot be represented exactly"
        ) from exc
    if lot_quantity != position.remaining_quantity:
        raise InvalidPaperEconomicTransition(
            "FIFO lots contradict the remaining position quantity"
        )

    fees = tuple(
        PaperFeeTotal(asset=asset, amount=amount)
        for asset, amount in sorted(fee_totals.items())
    )
    return _DerivedEconomics(
        position=position,
        lots=lots,
        open_quantity=lot_quantity,
        open_cost=open_cost,
        gross_realized_pnl=gross_realized_pnl,
        fees=fees,
    )


@dataclass(frozen=True, slots=True)
class PaperEconomics:
    """A validated economic projection derived from causal fill records."""

    lot_method: PaperLotMethod
    records: tuple[PaperFillRecord, ...]
    position: Position
    lots: tuple[PaperCostLot, ...]
    open_quantity: Decimal
    open_cost: Decimal
    gross_realized_pnl: Decimal
    fees: tuple[PaperFeeTotal, ...]

    def __post_init__(self) -> None:
        if type(self.lot_method) is not PaperLotMethod:
            raise TypeError("lot_method must be a PaperLotMethod")
        if self.lot_method is not PaperLotMethod.FIFO:
            raise ValueError("only FIFO paper economics is supported")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple")
        if any(type(record) is not PaperFillRecord for record in self.records):
            raise TypeError("records must contain only PaperFillRecord values")
        if type(self.position) is not Position:
            raise TypeError("position must be a Position")
        if not isinstance(self.lots, tuple):
            raise TypeError("lots must be a tuple")
        if any(type(lot) is not PaperCostLot for lot in self.lots):
            raise TypeError("lots must contain only PaperCostLot values")
        if not isinstance(self.open_quantity, Decimal):
            raise TypeError("open_quantity must be a Decimal")
        if not self.open_quantity.is_finite() or self.open_quantity < 0:
            raise ValueError("open_quantity must be finite and non-negative")
        if not isinstance(self.open_cost, Decimal):
            raise TypeError("open_cost must be a Decimal")
        if not self.open_cost.is_finite() or self.open_cost < 0:
            raise ValueError("open_cost must be finite and non-negative")
        if not isinstance(self.gross_realized_pnl, Decimal):
            raise TypeError("gross_realized_pnl must be a Decimal")
        if not self.gross_realized_pnl.is_finite():
            raise ValueError("gross_realized_pnl must be finite")
        if not isinstance(self.fees, tuple):
            raise TypeError("fees must be a tuple")
        if any(type(fee) is not PaperFeeTotal for fee in self.fees):
            raise TypeError("fees must contain only PaperFeeTotal values")

        expected = _derive(self.records)
        if _position_payload_identity(self.position) != _position_payload_identity(
            expected.position
        ):
            raise ValueError("position must be derived from the causal records")
        if tuple(_lot_payload_identity(lot) for lot in self.lots) != tuple(
            _lot_payload_identity(lot) for lot in expected.lots
        ):
            raise ValueError("lots must be derived from the causal records")
        if _decimal_payload_identity(self.open_quantity) != _decimal_payload_identity(
            expected.open_quantity
        ):
            raise ValueError("open_quantity must be derived from the causal records")
        if _decimal_payload_identity(self.open_cost) != _decimal_payload_identity(
            expected.open_cost
        ):
            raise ValueError("open_cost must be derived from the causal records")
        if _decimal_payload_identity(
            self.gross_realized_pnl
        ) != _decimal_payload_identity(expected.gross_realized_pnl):
            raise ValueError(
                "gross_realized_pnl must be derived from the causal records"
            )
        if tuple(_fee_payload_identity(fee) for fee in self.fees) != tuple(
            _fee_payload_identity(fee) for fee in expected.fees
        ):
            raise ValueError("fees must be derived from the causal records")

    @property
    def projected_through_version(self) -> int:
        """Return the latest causally applied fill position version."""
        return self.records[-1].position_version


def _from_records(
    lot_method: PaperLotMethod,
    records: tuple[PaperFillRecord, ...],
) -> PaperEconomics:
    if type(lot_method) is not PaperLotMethod:
        raise TypeError("lot_method must be a PaperLotMethod")
    derived = _derive(records)
    return PaperEconomics(
        lot_method=lot_method,
        records=records,
        position=derived.position,
        lots=derived.lots,
        open_quantity=derived.open_quantity,
        open_cost=derived.open_cost,
        gross_realized_pnl=derived.gross_realized_pnl,
        fees=derived.fees,
    )


def new_paper_economics(
    record: PaperFillRecord,
    *,
    lot_method: PaperLotMethod = PaperLotMethod.FIFO,
) -> PaperEconomics:
    """Create paper economics from the first confirmed OPEN fill."""
    if type(record) is not PaperFillRecord:
        raise TypeError("record must be a PaperFillRecord")
    return _from_records(lot_method, (record,))


def reduce_paper_economics(
    economics: PaperEconomics,
    record: PaperFillRecord,
) -> PaperEconomics:
    """Apply one later fill or return the same object for an exact replay."""
    if type(economics) is not PaperEconomics:
        raise TypeError("economics must be a PaperEconomics")
    if type(record) is not PaperFillRecord:
        raise TypeError("record must be a PaperFillRecord")

    for existing in economics.records:
        if _record_payload_identity(existing) == _record_payload_identity(record):
            return economics
        if existing.event_identity == record.event_identity:
            raise InvalidPaperEconomicTransition(
                "event identity conflicts with an existing fill record"
            )
        if existing.fill_identity == record.fill_identity:
            raise InvalidPaperEconomicTransition(
                "fill identity conflicts with existing causal metadata"
            )
        if existing.position_version == record.position_version:
            raise InvalidPaperEconomicTransition(
                "position_version conflicts with an existing fill record"
            )

    if record.position_version <= economics.projected_through_version:
        raise InvalidPaperEconomicTransition(
            "fill position_version must advance causal order"
        )
    return _from_records(economics.lot_method, economics.records + (record,))


__all__ = [
    "InvalidPaperEconomicTransition",
    "PaperCostLot",
    "PaperEconomics",
    "PaperFeeTotal",
    "PaperFillRecord",
    "PaperLotMethod",
    "new_paper_economics",
    "reduce_paper_economics",
]
