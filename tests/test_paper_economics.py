"""Contract tests for pure FIFO paper-economic projections."""

import ast
import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Inexact, Rounded, localcontext
from pathlib import Path

import pytest

from trading.domain.order_lifecycle import ConfirmedFill
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_economics import (
    InvalidPaperEconomicTransition,
    PaperCostLot,
    PaperEconomics,
    PaperFeeTotal,
    PaperFillRecord,
    PaperLotMethod,
    new_paper_economics,
    reduce_paper_economics,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionFill,
    PositionInstruction,
    PositionSide,
    PositionState,
    TakeProfitProfile,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
PUBLIC_EXPORTS = {
    "InvalidPaperEconomicTransition",
    "PaperCostLot",
    "PaperEconomics",
    "PaperFeeTotal",
    "PaperFillRecord",
    "PaperLotMethod",
    "new_paper_economics",
    "reduce_paper_economics",
}


def make_record(
    *,
    version: int = 1,
    event_id: str | None = None,
    client_order_id: str = "open-order-1",
    trade_id: str = "trade-1",
    position_key: str = "position-1",
    effect: PositionEffect = PositionEffect.OPEN,
    position_side: PositionSide = PositionSide.LONG,
    symbol: str = "BTCUSDT",
    quantity: Decimal = Decimal("1"),
    price: Decimal = Decimal("100"),
    reference_price: Decimal | None = None,
    leverage: int = 3,
    fee_amount: Decimal = Decimal("0.1"),
    fee_asset: str | None = "USDT",
) -> PaperFillRecord:
    if position_side is PositionSide.LONG:
        order_side = OrderSide.BUY if effect is PositionEffect.OPEN else OrderSide.SELL
    else:
        order_side = OrderSide.SELL if effect is PositionEffect.OPEN else OrderSide.BUY
    intent = OrderIntent(
        client_order_id=client_order_id,
        decision_id=f"decision-{client_order_id}",
        symbol=symbol,
        side=order_side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        reference_price=price if reference_price is None else reference_price,
        leverage=leverage,
        created_at=NOW,
    )
    instruction = PositionInstruction(
        position_key=position_key,
        effect=effect,
        order_intent=intent,
        exit_context=(
            PositionExitContext(
                take_profit_profile=TakeProfitProfile.RANGING,
                take_profit_fraction=Decimal("0.0025"),
                stop_loss_fraction=Decimal("0.005"),
            )
            if effect is PositionEffect.OPEN
            else None
        ),
    )
    fill = ConfirmedFill(
        client_order_id=client_order_id,
        venue_order_id=f"venue-{client_order_id}",
        trade_id=trade_id,
        symbol=symbol,
        side=order_side,
        quantity=quantity,
        price=price,
        fee_amount=fee_amount,
        fee_asset=fee_asset,
        executed_at=NOW + timedelta(seconds=version),
    )
    return PaperFillRecord(
        position_version=version,
        event_id=event_id or f"fill-{version}",
        position_fill=PositionFill(instruction, fill),
    )


def apply_records(*records: PaperFillRecord) -> PaperEconomics:
    economics = new_paper_economics(records[0])
    for record in records[1:]:
        economics = reduce_paper_economics(economics, record)
    return economics


@pytest.mark.parametrize(
    ("side", "expected_side"),
    [(PositionSide.LONG, PositionSide.LONG), (PositionSide.SHORT, PositionSide.SHORT)],
)
def test_open_creates_one_exact_fifo_lot(side, expected_side) -> None:
    record = make_record(
        position_side=side, quantity=Decimal("1.25"), price=Decimal("80")
    )
    economics = new_paper_economics(record)

    assert economics.lot_method is PaperLotMethod.FIFO
    assert economics.position.side is expected_side
    assert economics.position.state is PositionState.OPEN
    assert economics.open_quantity == Decimal("1.25")
    assert economics.open_cost == Decimal("100.00")
    assert economics.gross_realized_pnl == Decimal("0")
    assert economics.lots == (
        PaperCostLot(record.fill_identity, Decimal("1.25"), Decimal("80")),
    )


def test_scale_in_preserves_causal_fifo_lots_and_exact_cost() -> None:
    first = make_record(quantity=Decimal("1"), price=Decimal("100"))
    second = make_record(
        version=4,
        client_order_id="open-order-2",
        trade_id="trade-2",
        quantity=Decimal("2"),
        price=Decimal("125"),
    )
    economics = apply_records(first, second)

    assert tuple(lot.fill_identity for lot in economics.lots) == (
        first.fill_identity,
        second.fill_identity,
    )
    assert economics.open_quantity == Decimal("3")
    assert economics.open_cost == Decimal("350")


def test_fill_prices_drive_cost_and_pnl_while_reference_and_leverage_do_not() -> None:
    def projection(leverage: int) -> PaperEconomics:
        opened = make_record(
            price=Decimal("100"),
            reference_price=Decimal("999"),
            leverage=leverage,
        )
        reduced = make_record(
            version=2,
            client_order_id="reduce-order-1",
            trade_id="reduce-1",
            effect=PositionEffect.REDUCE_ONLY,
            quantity=Decimal("0.5"),
            price=Decimal("120"),
            reference_price=Decimal("1"),
            leverage=leverage,
        )
        return apply_records(opened, reduced)

    at_one_x = projection(1)
    at_one_hundred_x = projection(100)

    assert at_one_x.open_cost == at_one_hundred_x.open_cost == Decimal("50.0")
    assert (
        at_one_x.gross_realized_pnl
        == at_one_hundred_x.gross_realized_pnl
        == Decimal("10.0")
    )


@pytest.mark.parametrize(
    ("side", "exit_price", "expected"),
    [
        (PositionSide.LONG, Decimal("120"), Decimal("10.0")),
        (PositionSide.LONG, Decimal("90"), Decimal("-5.0")),
        (PositionSide.SHORT, Decimal("90"), Decimal("5.0")),
        (PositionSide.SHORT, Decimal("120"), Decimal("-10.0")),
    ],
)
def test_partial_reduce_realizes_exact_long_and_short_gain_or_loss(
    side, exit_price, expected
) -> None:
    opened = make_record(
        position_side=side, quantity=Decimal("1"), price=Decimal("100")
    )
    reduced = make_record(
        version=2,
        client_order_id="reduce-order-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        position_side=side,
        quantity=Decimal("0.5"),
        price=exit_price,
    )
    economics = apply_records(opened, reduced)

    assert economics.position.state is PositionState.OPEN
    assert economics.open_quantity == Decimal("0.5")
    assert economics.open_cost == Decimal("50.0")
    assert economics.gross_realized_pnl == expected


def test_fifo_partial_then_full_reduce_consumes_oldest_lot_first() -> None:
    first = make_record(
        client_order_id="z-open", trade_id="z-trade", price=Decimal("100")
    )
    second = make_record(
        version=2,
        client_order_id="a-open",
        trade_id="a-trade",
        price=Decimal("200"),
    )
    partial = make_record(
        version=3,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("1.5"),
        price=Decimal("150"),
    )
    final = make_record(
        version=8,
        client_order_id="reduce-2",
        trade_id="reduce-2",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("0.5"),
        price=Decimal("180"),
    )

    halfway = apply_records(first, second, partial)
    closed = reduce_paper_economics(halfway, final)

    assert tuple(record.fill_identity for record in halfway.records[:2]) == (
        first.fill_identity,
        second.fill_identity,
    )
    assert tuple(event.identity for event in halfway.position.fills) == tuple(
        sorted(event.identity for event in halfway.position.fills)
    )
    assert halfway.lots == (
        PaperCostLot(second.fill_identity, Decimal("0.5"), Decimal("200")),
    )
    assert halfway.open_cost == Decimal("100.0")
    assert halfway.gross_realized_pnl == Decimal("25.0")
    assert closed.position.state is PositionState.CLOSED
    assert closed.lots == ()
    assert closed.open_quantity == Decimal("0")
    assert closed.open_cost == Decimal("0")
    assert closed.gross_realized_pnl == Decimal("15.0")


def test_fees_accumulate_exactly_and_canonically_by_asset() -> None:
    records = (
        make_record(fee_amount=Decimal("0.1"), fee_asset="USDT"),
        make_record(
            version=2,
            client_order_id="open-2",
            trade_id="trade-2",
            quantity=Decimal("0.5"),
            fee_amount=Decimal("0.2"),
            fee_asset="USDT",
        ),
        make_record(
            version=3,
            client_order_id="reduce-1",
            trade_id="reduce-1",
            effect=PositionEffect.REDUCE_ONLY,
            quantity=Decimal("0.25"),
            fee_amount=Decimal("0.3"),
            fee_asset="BNB",
        ),
        make_record(
            version=4,
            client_order_id="reduce-2",
            trade_id="reduce-2",
            effect=PositionEffect.REDUCE_ONLY,
            quantity=Decimal("0.25"),
            fee_amount=Decimal("0"),
            fee_asset=None,
        ),
    )
    economics = apply_records(*records)
    assert economics.fees == (
        PaperFeeTotal("BNB", Decimal("0.3")),
        PaperFeeTotal("USDT", Decimal("0.3")),
    )


def test_zero_fee_is_ignored_even_when_an_asset_is_present() -> None:
    economics = new_paper_economics(
        make_record(fee_amount=Decimal("0"), fee_asset="USDT")
    )
    assert economics.fees == ()


def test_unsupported_exact_fee_arithmetic_fails_as_a_typed_transition() -> None:
    unsupported = Decimal("1E-1000000000")
    with pytest.raises(InvalidPaperEconomicTransition):
        new_paper_economics(make_record(fee_amount=unsupported))

    economics = new_paper_economics(make_record())
    later = make_record(
        version=2,
        client_order_id="open-2",
        trade_id="trade-2",
        fee_amount=unsupported,
    )
    with pytest.raises(InvalidPaperEconomicTransition):
        reduce_paper_economics(economics, later)


@pytest.mark.parametrize(
    ("quantity", "price"),
    [
        (Decimal("10"), Decimal("1E999999999")),
        (Decimal("9" * 6000), Decimal("9" * 6000)),
    ],
)
def test_unrepresentable_open_cost_product_fails_as_a_typed_transition(
    quantity, price
) -> None:
    with pytest.raises(InvalidPaperEconomicTransition):
        new_paper_economics(make_record(quantity=quantity, price=price))


def test_decimal_economics_ignore_hostile_ambient_context() -> None:
    opened = make_record(
        quantity=Decimal("1000000000000000000000000000001"),
        price=Decimal("0.000000000000000000000000000003"),
        fee_amount=Decimal("0.1"),
    )
    reduced = make_record(
        version=9,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("1000000000000000000000000000000"),
        price=Decimal("0.000000000000000000000000000004"),
        fee_amount=Decimal("0.2"),
    )
    with localcontext() as ambient:
        ambient.prec = 3
        ambient.rounding = ROUND_DOWN
        ambient.traps[Inexact] = True
        ambient.traps[Rounded] = True
        economics = apply_records(opened, reduced)

    assert economics.open_quantity == Decimal("1")
    assert economics.open_cost == Decimal("3E-30")
    assert economics.gross_realized_pnl == Decimal("1")
    assert economics.fees == (PaperFeeTotal("USDT", Decimal("0.3")),)


def test_exact_duplicate_is_same_object_and_conflicts_fail_closed() -> None:
    opened = make_record()
    economics = new_paper_economics(opened)
    assert reduce_paper_economics(economics, opened) is economics

    conflicts = (
        replace(opened, event_id="other-event"),
        replace(opened, position_version=2),
        PaperFillRecord(
            position_version=2,
            event_id=opened.event_id,
            position_fill=make_record(
                client_order_id=opened.position_fill.fill.client_order_id,
                trade_id="trade-2",
            ).position_fill,
        ),
        PaperFillRecord(
            position_version=2,
            event_id="other-event",
            position_fill=replace(
                opened.position_fill,
                fill=replace(opened.position_fill.fill, price=Decimal("101")),
            ),
        ),
    )
    for conflict in conflicts:
        with pytest.raises(InvalidPaperEconomicTransition):
            reduce_paper_economics(economics, conflict)


def test_event_id_scope_is_the_correlated_client_order() -> None:
    opened = make_record(event_id="fill-1")
    same_event_id_for_another_order = make_record(
        version=2,
        event_id="fill-1",
        client_order_id="open-order-2",
        trade_id="trade-2",
    )

    economics = apply_records(opened, same_event_id_for_another_order)

    assert tuple(record.event_identity for record in economics.records) == (
        ("open-order-1", "fill-1"),
        ("open-order-2", "fill-1"),
    )


def test_exact_duplicate_preserves_every_decimal_payload_quantum() -> None:
    opened = make_record()
    economics = new_paper_economics(opened)
    position_fill = opened.position_fill
    instruction = position_fill.instruction
    intent = instruction.order_intent
    exit_context = instruction.exit_context
    assert exit_context is not None
    fill = position_fill.fill

    changed_position_fills = (
        replace(position_fill, fill=replace(fill, quantity=Decimal("1.00"))),
        replace(position_fill, fill=replace(fill, price=Decimal("100.0"))),
        replace(position_fill, fill=replace(fill, fee_amount=Decimal("0.10"))),
        replace(
            position_fill,
            instruction=replace(
                instruction,
                order_intent=replace(intent, quantity=Decimal("1.00")),
            ),
        ),
        replace(
            position_fill,
            instruction=replace(
                instruction,
                order_intent=replace(intent, reference_price=Decimal("100.0")),
            ),
        ),
        replace(
            position_fill,
            instruction=replace(
                instruction,
                exit_context=replace(
                    exit_context,
                    take_profit_fraction=Decimal("0.00250"),
                ),
            ),
        ),
    )

    for changed in changed_position_fills:
        with pytest.raises(InvalidPaperEconomicTransition):
            reduce_paper_economics(
                economics,
                replace(opened, position_fill=changed),
            )


def test_version_gaps_are_allowed_but_duplicates_and_regressions_are_rejected() -> None:
    opened = make_record(version=2)
    later = make_record(
        version=20,
        client_order_id="open-2",
        trade_id="trade-2",
        quantity=Decimal("0.5"),
    )
    economics = apply_records(opened, later)
    assert economics.projected_through_version == 20

    for version in (20, 19, 1):
        conflict = make_record(
            version=version,
            event_id=f"late-{version}",
            client_order_id=f"late-{version}",
            trade_id=f"late-{version}",
            quantity=Decimal("0.1"),
        )
        with pytest.raises(InvalidPaperEconomicTransition):
            reduce_paper_economics(economics, conflict)


@pytest.mark.parametrize(
    "record",
    [
        make_record(effect=PositionEffect.REDUCE_ONLY),
        make_record(version=2, position_key="other-position", client_order_id="open-2"),
        make_record(version=2, symbol="ETHUSDT", client_order_id="open-2"),
        make_record(
            version=2,
            effect=PositionEffect.REDUCE_ONLY,
            position_side=PositionSide.SHORT,
            client_order_id="wrong-side",
        ),
        make_record(
            version=2,
            effect=PositionEffect.REDUCE_ONLY,
            quantity=Decimal("1.1"),
            client_order_id="over-reduce",
        ),
    ],
)
def test_invalid_initial_or_later_transitions_fail_closed(record) -> None:
    if record.position_version == 1:
        with pytest.raises(InvalidPaperEconomicTransition):
            new_paper_economics(record)
    else:
        economics = new_paper_economics(make_record())
        with pytest.raises(InvalidPaperEconomicTransition):
            reduce_paper_economics(economics, record)


def test_closed_projection_rejects_new_open_or_reduce_fill() -> None:
    opened = make_record()
    closed = apply_records(
        opened,
        make_record(
            version=2,
            effect=PositionEffect.REDUCE_ONLY,
            client_order_id="close-1",
            trade_id="close-1",
        ),
    )
    assert closed.position.state is PositionState.CLOSED
    for effect in (PositionEffect.OPEN, PositionEffect.REDUCE_ONLY):
        with pytest.raises(InvalidPaperEconomicTransition):
            reduce_paper_economics(
                closed,
                make_record(
                    version=3,
                    effect=effect,
                    client_order_id=f"after-{effect.value}",
                    trade_id=f"after-{effect.value}",
                ),
            )


def test_fold_replay_and_checkpoint_suffix_are_identical() -> None:
    records = (
        make_record(),
        make_record(version=3, client_order_id="open-2", trade_id="trade-2"),
        make_record(
            version=7,
            effect=PositionEffect.REDUCE_ONLY,
            client_order_id="reduce-1",
            trade_id="reduce-1",
            quantity=Decimal("0.4"),
            price=Decimal("120"),
        ),
    )
    replayed = apply_records(*records)
    checkpoint = apply_records(*records[:2])
    resumed = reduce_paper_economics(checkpoint, records[2])
    assert replayed == resumed


def test_contract_values_are_frozen_slotted_and_hashable() -> None:
    record = make_record()
    economics = new_paper_economics(record)
    values = (
        record,
        economics.lots[0],
        PaperFeeTotal("USDT", Decimal("0.1")),
        economics,
    )
    assert all(not hasattr(value, "__dict__") for value in values)
    assert len({hash(value) for value in values}) == len(values)
    with pytest.raises(FrozenInstanceError):
        economics.open_cost = Decimal("0")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("records", [], TypeError),
        ("lots", [], TypeError),
        ("fees", [], TypeError),
        ("open_quantity", 1.0, TypeError),
        ("open_cost", Decimal("NaN"), ValueError),
        ("gross_realized_pnl", Decimal("Infinity"), ValueError),
    ],
)
def test_direct_economics_construction_rejects_invalid_or_derived_state(
    field, value, error
) -> None:
    economics = new_paper_economics(make_record())
    with pytest.raises(error):
        replace(economics, **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"lots": ()},
        {"open_quantity": Decimal("2")},
        {"open_cost": Decimal("99")},
        {"gross_realized_pnl": Decimal("1")},
        {"fees": ()},
    ],
)
def test_direct_construction_cannot_forge_derived_economics(change) -> None:
    with pytest.raises(ValueError):
        replace(new_paper_economics(make_record()), **change)


def test_direct_construction_preserves_derived_decimal_quantums() -> None:
    economics = new_paper_economics(make_record())
    position_fill = economics.position.fills[0]
    forged_position = replace(
        economics.position,
        fills=(
            replace(
                position_fill,
                fill=replace(position_fill.fill, price=Decimal("100.0")),
            ),
        ),
    )
    for change in (
        {"position": forged_position},
        {"lots": (replace(economics.lots[0], entry_price=Decimal("100.0")),)},
        {"open_quantity": Decimal("1.0")},
        {"open_cost": Decimal("100.0")},
        {"gross_realized_pnl": Decimal("0.0")},
        {"fees": (PaperFeeTotal("USDT", Decimal("0.10")),)},
    ):
        with pytest.raises(ValueError):
            replace(economics, **change)


def test_direct_construction_rejects_reordered_causal_records() -> None:
    economics = apply_records(
        make_record(),
        make_record(version=3, client_order_id="open-2", trade_id="trade-2"),
    )
    with pytest.raises(InvalidPaperEconomicTransition):
        replace(economics, records=tuple(reversed(economics.records)))


def test_record_storage_bounds_and_exact_types() -> None:
    record = make_record()
    assert (
        replace(record, position_version=(1 << 63) - 1).position_version
        == (1 << 63) - 1
    )
    for values, error in (
        ({"position_version": True}, TypeError),
        ({"position_version": 0}, ValueError),
        ({"position_version": 1 << 63}, ValueError),
        ({"event_id": "x" * 256}, ValueError),
        ({"event_id": "bad\x00event"}, ValueError),
        ({"event_id": "bad\ud800event"}, ValueError),
        ({"position_fill": object()}, TypeError),
    ):
        with pytest.raises(error):
            replace(record, **values)


@pytest.mark.parametrize("field", ["created_at", "executed_at"])
def test_record_rejects_timestamps_that_cannot_be_normalized_to_utc(field) -> None:
    record = make_record()
    invalid_time = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    if field == "created_at":
        instruction = record.position_fill.instruction
        changed = replace(
            record.position_fill,
            instruction=replace(
                instruction,
                order_intent=replace(
                    instruction.order_intent,
                    created_at=invalid_time,
                ),
            ),
        )
    else:
        changed = replace(
            record.position_fill,
            fill=replace(record.position_fill.fill, executed_at=invalid_time),
        )

    with pytest.raises(ValueError):
        replace(record, position_fill=changed)


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: PaperCostLot(("client",), Decimal("1"), Decimal("1")), TypeError),
        (
            lambda: PaperCostLot(("client", "trade"), Decimal("0"), Decimal("1")),
            ValueError,
        ),
        (
            lambda: PaperCostLot(("client", "trade"), Decimal("1"), Decimal("0")),
            ValueError,
        ),
        (lambda: PaperFeeTotal(" padded ", Decimal("1")), ValueError),
        (lambda: PaperFeeTotal("USDT", Decimal("0")), ValueError),
    ],
)
def test_direct_lot_and_fee_construction_rejects_invalid_state(factory, error) -> None:
    with pytest.raises(error):
        factory()


def _uses_paper_economics(source: str) -> bool:
    tree = ast.parse(source)
    module = "trading.domain.paper_economics"
    builtins_aliases = {"builtins"}
    builtin_import_aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"trading", "trading.domain"}:
                    return True
                if alias.name == module or alias.name.startswith(f"{module}."):
                    return True
                if alias.name.startswith("trading.domain.") and not alias.asname:
                    return True
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            imported_module = node.module or ""
            if imported_module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )
            if imported_module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if imported_module == module:
                return True
            if node.level and imported_module.endswith("paper_economics"):
                return True
            if node.level and imported & (PUBLIC_EXPORTS | {"paper_economics", "*"}):
                return True
            if node.level and "domain" in imported:
                return True
            if imported_module == "trading.domain" and imported & (
                PUBLIC_EXPORTS | {"paper_economics", "*"}
            ):
                return True
            if imported_module == "trading" and "domain" in imported:
                return True

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            built_in = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "__import__"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            )
            import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "import_module"
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
            )
            if not built_in and not import_module:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                aliases = builtin_import_aliases if built_in else import_module_aliases
                if target.id not in aliases:
                    aliases.add(target.id)
                    changed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = (
                node.args[0].value
                if node.args and isinstance(node.args[0], ast.Constant)
                else next(
                    (
                        keyword.value.value
                        for keyword in node.keywords
                        if keyword.arg == "name"
                        and isinstance(keyword.value, ast.Constant)
                    ),
                    None,
                )
            )
            is_builtin_import = (
                isinstance(node.func, ast.Name)
                and node.func.id in builtin_import_aliases
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "__import__"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in builtins_aliases
            )
            is_import_module = (
                isinstance(node.func, ast.Name)
                and node.func.id in import_module_aliases
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in importlib_aliases
            )
            if (
                is_builtin_import
                and isinstance(target, str)
                and (target == "trading" or target.startswith("trading."))
            ):
                return True
            if is_import_module and target in {module, "trading.domain", "trading"}:
                return True
            if isinstance(target, str) and target.startswith("."):
                package = (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else next(
                        (
                            keyword.value.value
                            for keyword in node.keywords
                            if keyword.arg == "package"
                            and isinstance(keyword.value, ast.Constant)
                        ),
                        None,
                    )
                )
                if package:
                    try:
                        resolved = importlib.util.resolve_name(target, package)
                        if resolved in {module, "trading.domain", "trading"}:
                            return True
                    except (ImportError, ValueError):
                        pass
    return False


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.paper_economics import PaperEconomics",
        "import trading.domain.paper_economics as economics",
        "from trading.domain import PaperEconomics",
        "from trading.domain import paper_economics\n"
        "value = paper_economics.PaperEconomics",
        "from trading.domain import *\nvalue = PaperEconomics",
        "import trading.domain as domain\ndomain.PaperEconomics",
        "import trading.domain.orders\ntrading.domain.PaperEconomics",
        "from trading import domain as domain\ndomain.PaperEconomics",
        "from .. import domain as domain\ndomain.PaperEconomics",
        "from .paper_economics import PaperEconomics",
        "from . import paper_economics",
        "from . import *\nvalue = PaperEconomics",
        "from importlib import import_module\n"
        "import_module('trading.domain.paper_economics')",
        "import importlib\n"
        "importlib.import_module('.paper_economics', package='trading.domain')",
        "__import__('trading.domain.paper_economics')",
        "__import__('trading.domain.orders').domain.PaperEconomics",
        "load = __import__\nload('trading.domain.paper_economics')",
        "from builtins import __import__ as load\n"
        "load('trading.domain.paper_economics')",
        "import importlib\nload = importlib.import_module\n"
        "load('trading.domain.paper_economics')",
        "from importlib import import_module as load\n"
        "load('.paper_economics', 'trading.domain')",
        "import importlib as loader\n"
        "loader.import_module(name='.paper_economics', package='trading.domain')",
        "from importlib import import_module\n"
        "import_module('.domain', package='trading').PaperEconomics",
    ],
)
def test_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_paper_economics(source)


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.positions import Position",
        "from trading.domain import Position",
        "from trading.application import OrderService",
        "name = 'trading.domain.paper_economics'",
    ],
)
def test_consumer_detector_allows_unrelated_forms(source) -> None:
    assert not _uses_paper_economics(source)


def test_paper_economics_is_pure_and_has_no_runtime_consumer() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "domain" / "paper_economics.py"
    facade_path = root / "trading" / "domain" / "__init__.py"
    settlement_path = root / "trading" / "domain" / "paper_settlement.py"
    accounting_path = root / "trading" / "domain" / "paper_accounting.py"
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path in {module_path, facade_path, settlement_path, accounting_path}
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_paper_economics(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))
    assert consumers == []

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.level == 0
            imported_roots.add(node.module.split(".")[0])
            if node.module.startswith("trading."):
                assert node.module.startswith("trading.domain.")
    assert imported_roots <= set(sys.stdlib_module_names) | {"trading"}
