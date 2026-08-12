"""Pure, immutable position-effect state transitions."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from trading.domain._decimal import exact_decimal_sum
from trading.domain._validation import (
    protect_frozen_dataclass_state,
    require_clean_text,
    require_positive_decimal,
)
from trading.domain.order_lifecycle import ConfirmedFill, OrderLifecycle
from trading.domain.orders import OrderIntent, OrderSide


class PositionEffect(str, Enum):
    """The only effects an order may have on a position."""

    OPEN = "OPEN"
    REDUCE_ONLY = "REDUCE_ONLY"


class PositionSide(str, Enum):
    """Position orientation, deliberately distinct from order direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class TakeProfitProfile(str, Enum):
    """Validated entry-time profile understood by the current exit policy."""

    TRENDING = "TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"


class PositionState(str, Enum):
    """Whether a position key still has a positive confirmed remainder."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class InvalidPositionTransition(ValueError):
    """Raised when a confirmed fill contradicts the known position."""


def _require_fraction(name: str, value: object) -> None:
    require_positive_decimal(name, value)
    if value >= Decimal("1"):
        raise ValueError(f"{name} must be less than one")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PositionExitContext:
    """Exit policy resolved once and retained for the position lifetime."""

    take_profit_profile: TakeProfitProfile
    take_profit_fraction: Decimal
    stop_loss_fraction: Decimal
    trailing_stop_fraction: Decimal | None = None

    def __post_init__(self) -> None:
        if type(self.take_profit_profile) is not TakeProfitProfile:
            raise TypeError("take_profit_profile must be a TakeProfitProfile")
        _require_fraction("take_profit_fraction", self.take_profit_fraction)
        _require_fraction("stop_loss_fraction", self.stop_loss_fraction)
        if self.trailing_stop_fraction is not None:
            _require_fraction("trailing_stop_fraction", self.trailing_stop_fraction)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PositionInstruction:
    """An explicit position effect attached to one approved order intent."""

    position_key: str
    effect: PositionEffect
    order_intent: OrderIntent
    exit_context: PositionExitContext | None = None

    def __post_init__(self) -> None:
        require_clean_text("position_key", self.position_key)
        if type(self.effect) is not PositionEffect:
            raise TypeError("effect must be a PositionEffect")
        if type(self.order_intent) is not OrderIntent:
            raise TypeError("order_intent must be an OrderIntent")
        if (
            self.exit_context is not None
            and type(self.exit_context) is not PositionExitContext
        ):
            raise TypeError("exit_context must be a PositionExitContext or None")
        if self.effect is PositionEffect.OPEN and self.exit_context is None:
            raise ValueError("an OPEN instruction requires an exit_context")
        if self.effect is PositionEffect.REDUCE_ONLY and self.exit_context is not None:
            raise ValueError("a REDUCE_ONLY instruction cannot replace exit_context")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PositionFill:
    """One confirmed fill coupled to its pre-submission position instruction."""

    instruction: PositionInstruction
    fill: ConfirmedFill

    def __post_init__(self) -> None:
        if type(self.instruction) is not PositionInstruction:
            raise TypeError("instruction must be a PositionInstruction")
        if type(self.fill) is not ConfirmedFill:
            raise TypeError("fill must be a ConfirmedFill")

        intent = self.instruction.order_intent
        if self.fill.client_order_id != intent.client_order_id:
            raise ValueError("fill client_order_id must match the instruction intent")
        if self.fill.symbol != intent.symbol:
            raise ValueError("fill symbol must match the instruction intent")
        if self.fill.side is not intent.side:
            raise ValueError("fill side must match the instruction intent")
        if self.fill.quantity > intent.quantity:
            raise ValueError("one fill cannot exceed its instruction quantity")

    @property
    def identity(self) -> tuple[str, str]:
        """Return the position-local idempotency key for this confirmed fill."""
        return (self.fill.client_order_id, self.fill.trade_id)


def _side_for_open(order_side: OrderSide) -> PositionSide:
    return PositionSide.LONG if order_side is OrderSide.BUY else PositionSide.SHORT


def _expected_reduce_side(position_side: PositionSide) -> OrderSide:
    return OrderSide.SELL if position_side is PositionSide.LONG else OrderSide.BUY


def _quantity_for_effect(
    fills: tuple[PositionFill, ...],
    effect: PositionEffect,
) -> Decimal:
    return exact_decimal_sum(
        tuple(
            event.fill.quantity for event in fills if event.instruction.effect is effect
        )
    )


def _remaining_quantity(fills: tuple[PositionFill, ...]) -> Decimal:
    opened = _quantity_for_effect(fills, PositionEffect.OPEN)
    reduced = _quantity_for_effect(fills, PositionEffect.REDUCE_ONLY)
    return exact_decimal_sum((opened, reduced.copy_negate()))


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class Position:
    """A validated projection of confirmed fills for one stable position key."""

    position_key: str
    symbol: str
    side: PositionSide
    leverage: int
    exit_context: PositionExitContext
    state: PositionState
    fills: tuple[PositionFill, ...]

    def __post_init__(self) -> None:
        require_clean_text("position_key", self.position_key)
        require_clean_text("symbol", self.symbol)
        if type(self.side) is not PositionSide:
            raise TypeError("side must be a PositionSide")
        if isinstance(self.leverage, bool) or not isinstance(self.leverage, int):
            raise TypeError("leverage must be an integer")
        if self.leverage < 1:
            raise ValueError("leverage must be positive")
        if type(self.exit_context) is not PositionExitContext:
            raise TypeError("exit_context must be a PositionExitContext")
        if type(self.state) is not PositionState:
            raise TypeError("state must be a PositionState")
        if not isinstance(self.fills, tuple):
            raise TypeError("fills must be a tuple")
        if not self.fills:
            raise ValueError("a position requires at least one confirmed fill")
        if any(type(event) is not PositionFill for event in self.fills):
            raise TypeError("fills must contain only PositionFill values")

        identities = tuple(event.identity for event in self.fills)
        if len(identities) != len(set(identities)):
            raise ValueError("position fill identities must be unique")
        if identities != tuple(sorted(identities)):
            raise ValueError("position fills must use canonical identity order")

        instructions_by_client: dict[str, PositionInstruction] = {}
        venue_ids_by_client: dict[str, str] = {}
        fills_by_client: dict[str, list[Decimal]] = {}
        for event in self.fills:
            instruction = event.instruction
            intent = instruction.order_intent
            if instruction.position_key != self.position_key:
                raise ValueError("fill instruction must target the position key")
            if intent.symbol != self.symbol:
                raise ValueError("fill symbol must match the position")

            if instruction.effect is PositionEffect.OPEN:
                if _side_for_open(intent.side) is not self.side:
                    raise ValueError("OPEN fill side conflicts with the position")
                if intent.leverage != self.leverage:
                    raise ValueError("OPEN fill leverage must match the position")
                if instruction.exit_context != self.exit_context:
                    raise ValueError("scale-in cannot replace the entry exit_context")
            else:
                if intent.side is not _expected_reduce_side(self.side):
                    raise ValueError("REDUCE_ONLY fill must oppose the position side")

            client_order_id = intent.client_order_id
            previous_instruction = instructions_by_client.setdefault(
                client_order_id,
                instruction,
            )
            if previous_instruction != instruction:
                raise ValueError("one client order cannot change position instruction")
            previous_venue_id = venue_ids_by_client.setdefault(
                client_order_id,
                event.fill.venue_order_id,
            )
            if previous_venue_id != event.fill.venue_order_id:
                raise ValueError("one client order cannot span venue order IDs")
            fills_by_client.setdefault(client_order_id, []).append(event.fill.quantity)

        for client_order_id, quantities in fills_by_client.items():
            filled = exact_decimal_sum(tuple(quantities))
            if filled > instructions_by_client[client_order_id].order_intent.quantity:
                raise ValueError("position fills exceed their instruction quantity")

        opened = self.opened_quantity
        reduced = self.reduced_quantity
        remaining = exact_decimal_sum((opened, reduced.copy_negate()))
        if opened <= 0:
            raise ValueError("a position requires a confirmed OPEN fill")
        if remaining < 0:
            raise ValueError("REDUCE_ONLY fills cannot exceed opened quantity")
        if self.state is PositionState.OPEN and remaining <= 0:
            raise ValueError("an open position requires positive remaining quantity")
        if self.state is PositionState.CLOSED and remaining != 0:
            raise ValueError("a closed position requires zero remaining quantity")

    @property
    def opened_quantity(self) -> Decimal:
        """Return the exact confirmed quantity opened under this key."""
        return _quantity_for_effect(self.fills, PositionEffect.OPEN)

    @property
    def reduced_quantity(self) -> Decimal:
        """Return the exact confirmed reduce-only quantity."""
        return _quantity_for_effect(self.fills, PositionEffect.REDUCE_ONLY)

    @property
    def remaining_quantity(self) -> Decimal:
        """Return the exact still-open quantity."""
        return _remaining_quantity(self.fills)


def position_fill_from_lifecycle(
    instruction: PositionInstruction,
    lifecycle: OrderLifecycle,
    fill: ConfirmedFill,
) -> PositionFill:
    """Bind a position effect only to a fill present in its order lifecycle."""
    if type(instruction) is not PositionInstruction:
        raise TypeError("instruction must be a PositionInstruction")
    if type(lifecycle) is not OrderLifecycle:
        raise TypeError("lifecycle must be an OrderLifecycle")
    if type(fill) is not ConfirmedFill:
        raise TypeError("fill must be a ConfirmedFill")
    if instruction.order_intent != lifecycle.intent:
        raise ValueError("instruction intent must match the order lifecycle")
    if fill not in lifecycle.fills:
        raise ValueError("fill must be confirmed by the order lifecycle")
    return PositionFill(instruction=instruction, fill=fill)


def new_position(event: PositionFill) -> Position:
    """Create a new position only from an explicit confirmed OPEN fill."""
    if type(event) is not PositionFill:
        raise TypeError("event must be a PositionFill")
    if event.instruction.effect is not PositionEffect.OPEN:
        raise InvalidPositionTransition("REDUCE_ONLY cannot create a position")

    intent = event.instruction.order_intent
    exit_context = event.instruction.exit_context
    if exit_context is None:
        raise AssertionError("validated OPEN instruction lost its exit_context")
    try:
        return Position(
            position_key=event.instruction.position_key,
            symbol=intent.symbol,
            side=_side_for_open(intent.side),
            leverage=intent.leverage,
            exit_context=exit_context,
            state=PositionState.OPEN,
            fills=(event,),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidPositionTransition(
            "confirmed fill cannot create position"
        ) from exc


def reduce_position(position: Position, event: PositionFill) -> Position:
    """Apply one later confirmed fill without I/O or implicit netting."""
    if type(position) is not Position:
        raise TypeError("position must be a Position")
    if type(event) is not PositionFill:
        raise TypeError("event must be a PositionFill")

    existing = next(
        (known for known in position.fills if known.identity == event.identity),
        None,
    )
    if existing is not None:
        if existing == event:
            return position
        raise InvalidPositionTransition("position fill identity has conflicting data")
    if position.state is PositionState.CLOSED:
        raise InvalidPositionTransition("a closed position key cannot be reused")

    fills = tuple(
        sorted(
            position.fills + (event,),
            key=lambda known: known.identity,
        )
    )
    try:
        remaining = _remaining_quantity(fills)
        if remaining < 0:
            raise ValueError("REDUCE_ONLY fill exceeds remaining quantity")
        state = PositionState.CLOSED if remaining == 0 else PositionState.OPEN
        return Position(
            position_key=position.position_key,
            symbol=position.symbol,
            side=position.side,
            leverage=position.leverage,
            exit_context=position.exit_context,
            state=state,
            fills=fills,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidPositionTransition("confirmed fill contradicts position") from exc
