"""Contract tests for pure confirmed-fill position transitions."""

import ast
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Rounded, localcontext
from pathlib import Path

import pytest

from trading.domain import (
    ConfirmedFill,
    InvalidPositionTransition,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    PositionEffect,
    PositionExitContext,
    PositionFill,
    PositionInstruction,
    PositionSide,
    PositionState,
    SubmissionAcknowledged,
    TakeProfitProfile,
    new_order_lifecycle,
    new_position,
    position_fill_from_lifecycle,
    reduce_order_lifecycle,
    reduce_position,
)

NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(seconds=1)


def make_exit_context(**overrides: object) -> PositionExitContext:
    values = {
        "take_profit_profile": TakeProfitProfile.RANGING,
        "take_profit_fraction": Decimal("0.0025"),
        "stop_loss_fraction": Decimal("0.005"),
        "trailing_stop_fraction": Decimal("0.02"),
    }
    values.update(overrides)
    return PositionExitContext(**values)


def make_intent(**overrides: object) -> OrderIntent:
    values = {
        "client_order_id": "open-order-1",
        "decision_id": "decision-1",
        "symbol": "BTCUSDT",
        "side": OrderSide.BUY,
        "quantity": Decimal("1.0"),
        "order_type": OrderType.MARKET,
        "reference_price": Decimal("50000"),
        "leverage": 3,
        "created_at": NOW,
    }
    values.update(overrides)
    return OrderIntent(**values)


def make_instruction(
    intent: OrderIntent | None = None,
    **overrides: object,
) -> PositionInstruction:
    values = {
        "position_key": "position-1",
        "effect": PositionEffect.OPEN,
        "order_intent": intent or make_intent(),
        "exit_context": make_exit_context(),
    }
    values.update(overrides)
    return PositionInstruction(**values)


def make_fill(intent: OrderIntent | None = None, **overrides: object) -> ConfirmedFill:
    order_intent = intent or make_intent()
    values = {
        "client_order_id": order_intent.client_order_id,
        "venue_order_id": f"venue-{order_intent.client_order_id}",
        "trade_id": "trade-1",
        "symbol": order_intent.symbol,
        "side": order_intent.side,
        "quantity": Decimal("0.4"),
        "price": Decimal("50010"),
        "fee_amount": Decimal("0.2"),
        "fee_asset": "USDT",
        "executed_at": LATER,
    }
    values.update(overrides)
    return ConfirmedFill(**values)


def direct_event(
    instruction: PositionInstruction | None = None,
    fill: ConfirmedFill | None = None,
) -> PositionFill:
    position_instruction = instruction or make_instruction()
    confirmed_fill = fill or make_fill(position_instruction.order_intent)
    return PositionFill(position_instruction, confirmed_fill)


def lifecycle_events(
    instruction: PositionInstruction,
    *fills: ConfirmedFill,
) -> tuple[PositionFill, ...]:
    lifecycle = new_order_lifecycle(instruction.order_intent)
    lifecycle = reduce_order_lifecycle(
        lifecycle,
        SubmissionAcknowledged(
            client_order_id=instruction.order_intent.client_order_id,
            venue_order_id=fills[0].venue_order_id,
            observed_at=LATER,
        ),
    )
    for fill in fills:
        lifecycle = reduce_order_lifecycle(lifecycle, fill)
    return tuple(
        position_fill_from_lifecycle(instruction, lifecycle, fill) for fill in fills
    )


def make_open_position(
    *,
    quantity: Decimal = Decimal("1.0"),
    side: OrderSide = OrderSide.BUY,
    position_key: str = "position-1",
) -> Position:
    intent = make_intent(quantity=quantity, side=side)
    instruction = make_instruction(intent, position_key=position_key)
    fill = make_fill(intent, quantity=quantity)
    event = lifecycle_events(instruction, fill)[0]
    return new_position(event)


@pytest.mark.parametrize(
    ("order_side", "position_side"),
    [(OrderSide.BUY, PositionSide.LONG), (OrderSide.SELL, PositionSide.SHORT)],
)
def test_new_position_is_immutable_exact_and_oriented(
    order_side: OrderSide,
    position_side: PositionSide,
) -> None:
    position = make_open_position(side=order_side)

    assert position.side is position_side
    assert position.state is PositionState.OPEN
    assert position.opened_quantity == Decimal("1.0")
    assert position.reduced_quantity == Decimal("0")
    assert position.remaining_quantity == Decimal("1.0")
    assert isinstance(hash(position), int)
    with pytest.raises(FrozenInstanceError):
        position.state = PositionState.CLOSED  # type: ignore[misc]


@pytest.mark.parametrize("profile", list(TakeProfitProfile))
def test_exit_context_accepts_only_produced_profiles(
    profile: TakeProfitProfile,
) -> None:
    context = make_exit_context(take_profit_profile=profile)
    assert context.take_profit_profile is profile


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("take_profit_profile", "RANGING", TypeError),
        ("take_profit_profile", "REVERSAL", TypeError),
        ("take_profit_fraction", 0.1, TypeError),
        ("take_profit_fraction", Decimal("0"), ValueError),
        ("take_profit_fraction", Decimal("1"), ValueError),
        ("take_profit_fraction", Decimal("NaN"), ValueError),
        ("stop_loss_fraction", True, TypeError),
        ("stop_loss_fraction", Decimal("Infinity"), ValueError),
        ("trailing_stop_fraction", Decimal("-0.1"), ValueError),
        ("trailing_stop_fraction", Decimal("1.1"), ValueError),
    ],
)
def test_exit_context_rejects_invalid_values(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        make_exit_context(**{field: value})


def test_none_is_the_only_way_to_disable_trailing_stop() -> None:
    context = make_exit_context(trailing_stop_fraction=None)
    assert context.trailing_stop_fraction is None


@pytest.mark.parametrize("position_key", ["", " ", " padded ", 1, None])
def test_instruction_requires_a_clean_position_key(position_key: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_instruction(position_key=position_key)


def test_instruction_requires_explicit_typed_effect_and_intent() -> None:
    with pytest.raises(TypeError):
        make_instruction(effect="OPEN")
    with pytest.raises(TypeError):
        make_instruction(order_intent={})


def test_open_requires_context_and_reduce_only_forbids_it() -> None:
    with pytest.raises(ValueError):
        make_instruction(exit_context=None)
    with pytest.raises(ValueError):
        make_instruction(effect=PositionEffect.REDUCE_ONLY)

    reduce_intent = make_intent(side=OrderSide.SELL)
    instruction = make_instruction(
        reduce_intent,
        effect=PositionEffect.REDUCE_ONLY,
        exit_context=None,
    )
    assert instruction.effect is PositionEffect.REDUCE_ONLY


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_order_id", "other-order"),
        ("symbol", "BNBUSDT"),
        ("side", OrderSide.SELL),
        ("quantity", Decimal("1.1")),
    ],
)
def test_position_fill_requires_full_instruction_correlation(
    field: str,
    value: object,
) -> None:
    instruction = make_instruction()
    fill = make_fill(instruction.order_intent, **{field: value})
    with pytest.raises(ValueError):
        PositionFill(instruction, fill)


def test_position_fill_requires_exact_types() -> None:
    with pytest.raises(TypeError):
        PositionFill({}, make_fill())
    with pytest.raises(TypeError):
        PositionFill(make_instruction(), {})


def test_lifecycle_factory_accepts_only_a_member_confirmed_fill() -> None:
    intent = make_intent(quantity=Decimal("1"))
    instruction = make_instruction(intent)
    fill = make_fill(intent, quantity=Decimal("0.4"))
    event = lifecycle_events(instruction, fill)[0]

    assert event.instruction is instruction
    assert event.fill is fill

    acknowledged = reduce_order_lifecycle(
        new_order_lifecycle(intent),
        SubmissionAcknowledged(
            client_order_id=intent.client_order_id,
            venue_order_id=fill.venue_order_id,
            observed_at=LATER,
        ),
    )
    with pytest.raises(ValueError):
        position_fill_from_lifecycle(instruction, acknowledged, fill)
    with pytest.raises(ValueError):
        position_fill_from_lifecycle(
            make_instruction(make_intent(client_order_id="other-order")),
            acknowledged,
            fill,
        )


@pytest.mark.parametrize("value", [None, {}, object()])
def test_lifecycle_factory_rejects_untyped_inputs(value: object) -> None:
    instruction = make_instruction()
    fill = make_fill(instruction.order_intent)
    lifecycle = new_order_lifecycle(instruction.order_intent)

    with pytest.raises(TypeError):
        position_fill_from_lifecycle(value, lifecycle, fill)
    with pytest.raises(TypeError):
        position_fill_from_lifecycle(instruction, value, fill)
    with pytest.raises(TypeError):
        position_fill_from_lifecycle(instruction, lifecycle, value)


def test_reduce_only_cannot_create_a_position() -> None:
    intent = make_intent(side=OrderSide.SELL, client_order_id="reduce-order-1")
    instruction = make_instruction(
        intent,
        effect=PositionEffect.REDUCE_ONLY,
        exit_context=None,
    )
    event = direct_event(instruction, make_fill(intent))

    with pytest.raises(InvalidPositionTransition):
        new_position(event)


def test_new_position_and_reducer_require_exact_types() -> None:
    with pytest.raises(TypeError):
        new_position(object())
    with pytest.raises(TypeError):
        reduce_position(object(), direct_event())
    with pytest.raises(TypeError):
        reduce_position(make_open_position(), object())


def test_partial_open_fills_and_scale_in_keep_one_entry_context() -> None:
    first_intent = make_intent(quantity=Decimal("1"))
    first_instruction = make_instruction(first_intent)
    first_fill = make_fill(first_intent, trade_id="trade-a", quantity=Decimal("0.4"))
    second_fill = make_fill(
        first_intent,
        trade_id="trade-b",
        quantity=Decimal("0.6"),
    )
    first_event, second_event = lifecycle_events(
        first_instruction,
        first_fill,
        second_fill,
    )
    position = new_position(first_event)
    position = reduce_position(position, second_event)

    scale_intent = make_intent(
        client_order_id="open-order-2",
        decision_id="decision-2",
        quantity=Decimal("0.5"),
    )
    scale_instruction = make_instruction(
        scale_intent,
        exit_context=position.exit_context,
    )
    scale_fill = make_fill(
        scale_intent,
        trade_id="trade-scale",
        quantity=Decimal("0.5"),
    )
    scale_event = lifecycle_events(scale_instruction, scale_fill)[0]
    scaled = reduce_position(position, scale_event)

    assert scaled.state is PositionState.OPEN
    assert scaled.opened_quantity == Decimal("1.5")
    assert scaled.remaining_quantity == Decimal("1.5")
    assert scaled.exit_context is position.exit_context


@pytest.mark.parametrize(
    ("intent_overrides", "instruction_overrides"),
    [
        ({"symbol": "BNBUSDT"}, {}),
        ({"side": OrderSide.SELL}, {}),
        ({"leverage": 5}, {}),
        ({}, {"position_key": "other-position"}),
        (
            {},
            {
                "exit_context": make_exit_context(
                    take_profit_profile=TakeProfitProfile.TRENDING
                )
            },
        ),
    ],
)
def test_scale_in_rejects_changed_position_identity_or_policy(
    intent_overrides: dict[str, object],
    instruction_overrides: dict[str, object],
) -> None:
    position = make_open_position()
    intent = make_intent(
        client_order_id="open-order-2",
        decision_id="decision-2",
        quantity=Decimal("0.2"),
        **intent_overrides,
    )
    instruction_values = {"exit_context": position.exit_context}
    instruction_values.update(instruction_overrides)
    instruction = make_instruction(intent, **instruction_values)
    event = direct_event(
        instruction,
        make_fill(intent, trade_id="trade-scale", quantity=Decimal("0.2")),
    )

    with pytest.raises(InvalidPositionTransition):
        reduce_position(position, event)
    assert position.remaining_quantity == Decimal("1.0")


def make_reduce_event(
    position: Position,
    *,
    client_order_id: str = "reduce-order-1",
    trade_id: str = "reduce-trade-1",
    quantity: Decimal = Decimal("0.4"),
    side: OrderSide | None = None,
    leverage: int = 9,
) -> PositionFill:
    reduce_side = side or (
        OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
    )
    intent = make_intent(
        client_order_id=client_order_id,
        decision_id=f"decision-{client_order_id}",
        side=reduce_side,
        quantity=quantity,
        leverage=leverage,
    )
    instruction = make_instruction(
        intent,
        position_key=position.position_key,
        effect=PositionEffect.REDUCE_ONLY,
        exit_context=None,
    )
    fill = make_fill(intent, trade_id=trade_id, quantity=quantity)
    return lifecycle_events(instruction, fill)[0]


def test_reduce_only_partially_reduces_then_closes_exactly() -> None:
    position = make_open_position()

    partial = reduce_position(position, make_reduce_event(position))
    closed = reduce_position(
        partial,
        make_reduce_event(
            partial,
            client_order_id="reduce-order-2",
            trade_id="reduce-trade-2",
            quantity=Decimal("0.6"),
        ),
    )

    assert partial.state is PositionState.OPEN
    assert partial.reduced_quantity == Decimal("0.4")
    assert partial.remaining_quantity == Decimal("0.6")
    assert closed.state is PositionState.CLOSED
    assert closed.reduced_quantity == Decimal("1.0")
    assert closed.remaining_quantity == Decimal("0.0")


def test_reduce_only_rejects_same_side_and_over_reduction_without_flipping() -> None:
    position = make_open_position()
    same_side = make_reduce_event(position, side=OrderSide.BUY)
    over_reduce = make_reduce_event(position, quantity=Decimal("1.0001"))

    with pytest.raises(InvalidPositionTransition):
        reduce_position(position, same_side)
    with pytest.raises(InvalidPositionTransition):
        reduce_position(position, over_reduce)

    assert position.state is PositionState.OPEN
    assert position.remaining_quantity == Decimal("1.0")


def test_reduce_only_leverage_does_not_change_or_block_the_position() -> None:
    position = make_open_position()
    reduced = reduce_position(
        position,
        make_reduce_event(position, quantity=Decimal("0.1"), leverage=11),
    )

    assert reduced.leverage == 3
    assert reduced.remaining_quantity == Decimal("0.9")


def test_exact_duplicate_is_an_identity_noop_before_and_after_close() -> None:
    position = make_open_position()
    opening_event = position.fills[0]
    assert reduce_position(position, opening_event) is position

    closing_event = make_reduce_event(position, quantity=Decimal("1.0"))
    closed = reduce_position(position, closing_event)
    assert reduce_position(closed, closing_event) is closed


@pytest.mark.parametrize(
    "change",
    [
        "price",
        "fee",
        "time",
        "instruction",
    ],
)
def test_same_position_fill_identity_with_changed_payload_is_a_conflict(
    change: str,
) -> None:
    position = make_open_position()
    known = position.fills[0]
    if change == "instruction":
        changed_instruction = replace(
            known.instruction,
            exit_context=make_exit_context(
                take_profit_profile=TakeProfitProfile.TRENDING
            ),
        )
        conflicting = PositionFill(changed_instruction, known.fill)
    else:
        field_values = {
            "price": {"price": Decimal("50011")},
            "fee": {"fee_amount": Decimal("0.3")},
            "time": {"executed_at": known.fill.executed_at + timedelta(seconds=1)},
        }
        conflicting = PositionFill(
            known.instruction,
            replace(known.fill, **field_values[change]),
        )

    with pytest.raises(InvalidPositionTransition):
        reduce_position(position, conflicting)


def test_closed_position_rejects_every_new_open_or_reduce_fill() -> None:
    position = make_open_position()
    closed = reduce_position(
        position,
        make_reduce_event(position, quantity=Decimal("1.0")),
    )
    new_open_intent = make_intent(
        client_order_id="open-order-2",
        decision_id="decision-2",
        quantity=Decimal("0.1"),
    )
    new_open_instruction = make_instruction(
        new_open_intent,
        exit_context=closed.exit_context,
    )
    new_open = direct_event(
        new_open_instruction,
        make_fill(new_open_intent, trade_id="new-open", quantity=Decimal("0.1")),
    )
    new_reduce = make_reduce_event(
        closed,
        client_order_id="reduce-order-2",
        trade_id="new-reduce",
        quantity=Decimal("0.1"),
    )

    with pytest.raises(InvalidPositionTransition):
        reduce_position(closed, new_open)
    with pytest.raises(InvalidPositionTransition):
        reduce_position(closed, new_reduce)


def test_same_trade_id_from_distinct_client_orders_is_not_a_duplicate() -> None:
    position = make_open_position(quantity=Decimal("0.5"))
    intent = make_intent(
        client_order_id="open-order-2",
        decision_id="decision-2",
        quantity=Decimal("0.5"),
    )
    instruction = make_instruction(intent, exit_context=position.exit_context)
    event = direct_event(
        instruction,
        make_fill(intent, trade_id="trade-1", quantity=Decimal("0.5")),
    )

    scaled = reduce_position(position, event)
    assert len(scaled.fills) == 2
    assert scaled.opened_quantity == Decimal("1.0")


def test_partial_fills_cannot_exceed_one_client_order_intent() -> None:
    intent = make_intent(quantity=Decimal("1.0"))
    instruction = make_instruction(intent)
    first = direct_event(
        instruction,
        make_fill(intent, trade_id="trade-a", quantity=Decimal("0.6")),
    )
    second = direct_event(
        instruction,
        make_fill(intent, trade_id="trade-b", quantity=Decimal("0.5")),
    )
    position = new_position(first)

    with pytest.raises(InvalidPositionTransition):
        reduce_position(position, second)
    assert position.opened_quantity == Decimal("0.6")


@pytest.mark.parametrize("conflict", ["instruction", "venue"])
def test_one_client_order_cannot_change_instruction_or_venue(conflict: str) -> None:
    intent = make_intent(quantity=Decimal("1.0"))
    instruction = make_instruction(intent)
    first = direct_event(
        instruction,
        make_fill(intent, trade_id="trade-a", quantity=Decimal("0.4")),
    )
    position = new_position(first)
    second_instruction = instruction
    second_fill = make_fill(intent, trade_id="trade-b", quantity=Decimal("0.4"))
    if conflict == "instruction":
        second_instruction = replace(
            instruction,
            exit_context=make_exit_context(
                take_profit_profile=TakeProfitProfile.TRENDING
            ),
        )
    else:
        second_fill = replace(second_fill, venue_order_id="other-venue")

    with pytest.raises(InvalidPositionTransition):
        reduce_position(position, PositionFill(second_instruction, second_fill))


def test_scale_in_arrival_order_converges_canonically() -> None:
    seed = make_open_position(quantity=Decimal("0.5"))

    def open_event(client: str, trade: str) -> PositionFill:
        intent = make_intent(
            client_order_id=client,
            decision_id=f"decision-{client}",
            quantity=Decimal("0.25"),
        )
        instruction = make_instruction(intent, exit_context=seed.exit_context)
        return direct_event(
            instruction,
            make_fill(intent, trade_id=trade, quantity=Decimal("0.25")),
        )

    event_a = open_event("open-order-a", "trade-a")
    event_b = open_event("open-order-b", "trade-b")
    a_then_b = reduce_position(reduce_position(seed, event_a), event_b)
    b_then_a = reduce_position(reduce_position(seed, event_b), event_a)

    assert a_then_b == b_then_a
    assert tuple(event.identity for event in a_then_b.fills) == tuple(
        sorted(event.identity for event in a_then_b.fills)
    )


def test_independently_valid_reductions_converge_canonically() -> None:
    seed = make_open_position()
    reduction_a = make_reduce_event(
        seed,
        client_order_id="reduce-order-a",
        trade_id="reduce-a",
        quantity=Decimal("0.2"),
    )
    reduction_b = make_reduce_event(
        seed,
        client_order_id="reduce-order-b",
        trade_id="reduce-b",
        quantity=Decimal("0.3"),
    )

    a_then_b = reduce_position(reduce_position(seed, reduction_a), reduction_b)
    b_then_a = reduce_position(reduce_position(seed, reduction_b), reduction_a)

    assert a_then_b == b_then_a
    assert a_then_b.remaining_quantity == Decimal("0.5")


def test_exact_large_quantities_ignore_ambient_decimal_context() -> None:
    quantity = Decimal("1000000000000000000000000000001")
    with localcontext() as ambient:
        ambient.prec = 4
        ambient.rounding = ROUND_DOWN
        ambient.traps[Rounded] = True
        position = make_open_position(quantity=quantity)
        partial = reduce_position(
            position,
            make_reduce_event(
                position,
                client_order_id="reduce-order-a",
                trade_id="reduce-a",
                quantity=Decimal("1000000000000000000000000000000"),
            ),
        )
        closed = reduce_position(
            partial,
            make_reduce_event(
                partial,
                client_order_id="reduce-order-b",
                trade_id="reduce-b",
                quantity=Decimal("1"),
            ),
        )

    assert partial.remaining_quantity == Decimal("1")
    assert closed.state is PositionState.CLOSED
    assert closed.remaining_quantity == Decimal("0")


@pytest.mark.parametrize("precision", [4, 80])
def test_exact_over_reduction_is_rejected_under_ambient_contexts(
    precision: int,
) -> None:
    quantity = Decimal("1000000000000000000000000000001")
    with localcontext() as ambient:
        ambient.prec = precision
        position = make_open_position(quantity=quantity)
        with pytest.raises(InvalidPositionTransition):
            reduce_position(
                position,
                make_reduce_event(
                    position,
                    quantity=Decimal("1000000000000000000000000000002"),
                ),
            )
    assert position.remaining_quantity == quantity


def test_new_position_rejects_values_outside_exact_arithmetic_bounds() -> None:
    quantity = Decimal("1E-1000000000")
    intent = make_intent(quantity=quantity)
    instruction = make_instruction(intent)
    event = direct_event(instruction, make_fill(intent, quantity=quantity))

    with pytest.raises(InvalidPositionTransition):
        new_position(event)


def test_direct_position_construction_rejects_noncanonical_and_impossible_states() -> (
    None
):
    seed = make_open_position()
    event = seed.fills[0]

    with pytest.raises(TypeError):
        replace(seed, fills=[event])
    with pytest.raises(ValueError):
        replace(seed, fills=())
    with pytest.raises(ValueError):
        replace(seed, state=PositionState.CLOSED)
    with pytest.raises(ValueError):
        replace(seed, fills=(event, event))

    reduction = make_reduce_event(seed, quantity=Decimal("0.4"))
    with pytest.raises(ValueError):
        replace(seed, fills=(reduction,))

    over_reduction = make_reduce_event(seed, quantity=Decimal("1.1"))
    over_reduced_fills = tuple(
        sorted((event, over_reduction), key=lambda known: known.identity)
    )
    with pytest.raises(ValueError):
        replace(seed, fills=over_reduced_fills)

    second_intent = make_intent(
        client_order_id="open-order-2",
        decision_id="decision-2",
        quantity=Decimal("0.1"),
    )
    second_instruction = make_instruction(second_intent, exit_context=seed.exit_context)
    second = direct_event(
        second_instruction,
        make_fill(second_intent, trade_id="trade-2", quantity=Decimal("0.1")),
    )
    canonical = tuple(sorted((event, second), key=lambda known: known.identity))
    with pytest.raises(ValueError):
        replace(seed, fills=tuple(reversed(canonical)))


def test_direct_position_requires_typed_fields() -> None:
    position = make_open_position()
    with pytest.raises(TypeError):
        replace(position, side="LONG")
    with pytest.raises(TypeError):
        replace(position, state="OPEN")
    with pytest.raises(TypeError):
        replace(position, leverage=True)
    with pytest.raises(ValueError):
        replace(position, position_key=" padded ")


_POSITION_SYMBOLS = {
    "InvalidPositionTransition",
    "Position",
    "PositionEffect",
    "PositionExitContext",
    "PositionFill",
    "PositionInstruction",
    "PositionSide",
    "PositionState",
    "TakeProfitProfile",
    "new_position",
    "position_fill_from_lifecycle",
    "reduce_position",
}


def _literal_import_target(call: ast.Call) -> str | None:
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    target = call.args[0].value
    return target if isinstance(target, str) else None


def _uses_position_contract(source: str) -> bool:
    """Detect position-contract imports without matching generic names."""
    tree = ast.parse(source)
    builtins_aliases = {"builtins"}
    builtin_import_aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "trading":
                    # The package facade can expose ``domain.Position`` through
                    # arbitrary attribute access, so keep production imports on
                    # explicit domain submodules instead.
                    return True
                if alias.name == "trading.domain":
                    return True
                if alias.name == "trading.domain.positions" or alias.name.startswith(
                    "trading.domain.positions."
                ):
                    return True
                if alias.name.startswith("trading.domain.") and alias.asname is None:
                    # A dotted import without ``as`` also binds ``trading`` and
                    # therefore exposes the package facade to later access.
                    return True
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_names = {alias.name for alias in node.names}
            module = node.module or ""
            if module == "importlib" and "import_module" in imported_names:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if module == "builtins" and "__import__" in imported_names:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )
            if module == "trading" and "domain" in imported_names:
                return True
            if module == "trading.domain.positions" or (
                node.level and module in {"positions", "domain.positions"}
            ):
                return True
            if module == "trading.domain" or (node.level and module == "domain"):
                if imported_names & (_POSITION_SYMBOLS | {"*", "positions"}):
                    return True
            if node.level and not module and "domain" in imported_names:
                return True

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            is_builtin_import = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "__import__"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            )
            is_import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "import_module"
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
            )
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for assigned in targets:
                if not isinstance(assigned, ast.Name):
                    continue
                if is_builtin_import and assigned.id not in builtin_import_aliases:
                    builtin_import_aliases.add(assigned.id)
                    changed = True
                if is_import_module and assigned.id not in import_module_aliases:
                    import_module_aliases.add(assigned.id)
                    changed = True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _literal_import_target(node)
        is_builtin_import = (
            isinstance(node.func, ast.Name) and node.func.id in builtin_import_aliases
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "__import__"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in builtins_aliases
        )
        if is_builtin_import and target is not None and target.startswith("trading"):
            return True
        if target not in {
            "trading",
            "trading.domain",
            "trading.domain.positions",
        }:
            continue
        if isinstance(node.func, ast.Name) and node.func.id in (
            import_module_aliases | {"__import__"}
        ):
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        ):
            return True

    return False


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.positions import Position",
        "from trading.domain import Position",
        "import trading.domain.positions as positions",
        "import trading.domain as domain\nvalue = domain.Position",
        "import trading as trading_alias\nvalue = trading_alias.domain.Position",
        "import trading.domain.orders\nvalue = trading.domain.Position",
        "from ..domain import Position",
        "from ..domain.positions import Position",
        "from importlib import import_module as load\nload('trading.domain.positions')",
        "import importlib as loader\nloader.import_module('trading.domain.positions')",
        "from trading import domain as d\nvalue = getattr(d, 'Position')",
        (
            "from importlib import import_module as load\n"
            "domain = load('trading.domain')\n"
            "value = getattr(domain, 'Position')"
        ),
        "root = __import__('trading')\nvalue = root.domain.PositionInstruction",
        (
            "__import__('trading.application.order_service')"
            ".domain.PositionInstruction"
        ),
        (
            "load = __import__\n"
            "load('trading.application.order_service').domain.PositionInstruction"
        ),
        (
            "import importlib as loader\n"
            "load = loader.import_module\n"
            "load('trading').domain.PositionInstruction"
        ),
        (
            "from importlib import import_module as load\n"
            "root = load('trading')\n"
            "value = root.domain.PositionInstruction"
        ),
    ],
)
def test_position_consumer_detector_rejects_facade_and_indirect_imports(
    source: str,
) -> None:
    assert _uses_position_contract(source)


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain import OrderIntent",
        "import trading.domain.orders as orders",
        "from trading.domain.orders import OrderIntent",
        "from importlib import import_module as load\nload('trading.domain.orders')",
    ],
)
def test_position_consumer_detector_allows_explicit_unrelated_domain_imports(
    source: str,
) -> None:
    assert not _uses_position_contract(source)


def test_position_only_approved_modules_consume_contract() -> None:
    root = Path(__file__).parents[1]
    domain_root = root / "trading" / "domain"
    module_path = domain_root / "positions.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    allowed_standard_library = {"dataclasses", "decimal", "enum"}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert {
        module
        for module in imports
        if module.split(".", 1)[0] not in allowed_standard_library
        and not module.startswith("trading.domain")
    } == set()

    consumers = []
    excluded = {
        module_path,
        domain_root / "__init__.py",
        domain_root / "paper_economics.py",
    }
    for source_path in root.rglob("*.py"):
        if (
            source_path in excluded
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
        ):
            continue
        if _uses_position_contract(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert sorted(consumers) == [
        Path("trading/application/durable_submission.py"),
        Path("trading/application/journaled_order_service.py"),
        Path("trading/persistence/journal_codec.py"),
        Path("trading/persistence/order_position_journal.py"),
    ]
