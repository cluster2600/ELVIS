"""Contract tests for pure, explicitly denominated paper settlement."""

import ast
import copy
import importlib.util
import inspect
import pickle
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Inexact, Rounded, localcontext
from pathlib import Path

import pytest

from trading.domain.order_lifecycle import ConfirmedFill
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_economics import (
    PaperCostLot,
    PaperEconomics,
    PaperFeeTotal,
    PaperFillRecord,
)
from trading.domain.paper_settlement import (
    InvalidPaperSettlement,
    PaperAssetAmount,
    PaperLinearInstrument,
    PaperSettlement,
    PaperSettlementCheckpoint,
    PaperSettlementDisposition,
    settle_paper_fill,
)
from trading.domain.positions import (
    Position,
    PositionEffect,
    PositionExitContext,
    PositionFill,
    PositionInstruction,
    PositionSide,
    TakeProfitProfile,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
BTCUSDT = PaperLinearInstrument("BTCUSDT", "BTC", "USDT")
BNBBTC = PaperLinearInstrument("BNBBTC", "BNB", "BTC")
PUBLIC_EXPORTS = {
    "InvalidPaperSettlement",
    "PaperAssetAmount",
    "PaperLinearInstrument",
    "PaperSettlement",
    "PaperSettlementCheckpoint",
    "PaperSettlementDisposition",
    "settle_paper_fill",
}


def make_record(
    *,
    version: int = 1,
    event_id: str | None = None,
    client_order_id: str = "open-1",
    trade_id: str = "trade-1",
    position_key: str = "position-1",
    effect: PositionEffect = PositionEffect.OPEN,
    position_side: PositionSide = PositionSide.LONG,
    symbol: str = "BTCUSDT",
    quantity: Decimal = Decimal("1"),
    price: Decimal = Decimal("100"),
    fee_amount: Decimal = Decimal("0"),
    fee_asset: str | None = None,
) -> PaperFillRecord:
    if position_side is PositionSide.LONG:
        side = OrderSide.BUY if effect is PositionEffect.OPEN else OrderSide.SELL
    else:
        side = OrderSide.SELL if effect is PositionEffect.OPEN else OrderSide.BUY
    intent = OrderIntent(
        client_order_id=client_order_id,
        decision_id=f"decision-{client_order_id}",
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        reference_price=price,
        leverage=3,
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
        side=side,
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


def apply(instrument: PaperLinearInstrument, *records: PaperFillRecord):
    settlement = settle_paper_fill(instrument, None, records[0])
    for record in records[1:]:
        settlement = settle_paper_fill(instrument, settlement.after, record)
    return settlement


@pytest.mark.parametrize(
    ("instrument", "expected_asset"),
    [(BTCUSDT, "USDT"), (BNBBTC, "BTC")],
)
def test_instrument_uses_explicit_quote_as_settlement_asset(
    instrument, expected_asset
) -> None:
    assert instrument.settlement_asset == expected_asset
    opened = make_record(symbol=instrument.symbol)
    result = settle_paper_fill(instrument, None, opened)
    assert result.gross_realized_pnl_delta == PaperAssetAmount(
        expected_asset, Decimal("0")
    )


def test_non_usdt_symbol_realizes_pnl_in_explicit_quote_asset() -> None:
    opened = make_record(symbol="BNBBTC", price=Decimal("0.010"))
    before = settle_paper_fill(BNBBTC, None, opened).after
    reduced = make_record(
        version=2,
        client_order_id="reduce-bnb",
        trade_id="reduce-bnb",
        effect=PositionEffect.REDUCE_ONLY,
        symbol="BNBBTC",
        quantity=Decimal("0.5"),
        price=Decimal("0.020"),
    )

    result = settle_paper_fill(BNBBTC, before, reduced)

    assert result.gross_realized_pnl_delta == PaperAssetAmount("BTC", Decimal("0.0050"))
    assert result.cash_deltas == (PaperAssetAmount("BTC", Decimal("0.0050")),)


@pytest.mark.parametrize(
    ("side", "exit_price", "expected"),
    [
        (PositionSide.LONG, Decimal("120"), Decimal("-20.0")),
        (PositionSide.SHORT, Decimal("80"), Decimal("80.0")),
    ],
)
def test_fifo_long_and_short_realized_delta_uses_quote_asset(
    side, exit_price, expected
) -> None:
    first = make_record(position_side=side, price=Decimal("100"))
    second = make_record(
        version=3,
        client_order_id="open-2",
        trade_id="trade-2",
        position_side=side,
        price=Decimal("200"),
    )
    reduced = make_record(
        version=8,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        position_side=side,
        quantity=Decimal("1.5"),
        price=exit_price,
    )
    result = apply(BTCUSDT, first, second, reduced)

    assert result.disposition is PaperSettlementDisposition.APPLIED
    assert result.gross_realized_pnl_delta == PaperAssetAmount("USDT", expected)
    assert result.cash_deltas == (PaperAssetAmount("USDT", expected),)
    assert result.after.economics.open_quantity == Decimal("0.5")
    assert result.after.economics.open_cost == Decimal("100.0")


def test_open_is_fee_only_and_keeps_fee_in_its_own_asset() -> None:
    opened = make_record(fee_amount=Decimal("0.25"), fee_asset="BNB")
    result = settle_paper_fill(BTCUSDT, None, opened)

    assert result.gross_realized_pnl_delta == PaperAssetAmount("USDT", Decimal("0"))
    assert result.fee_debits == (PaperAssetAmount("BNB", Decimal("0.25")),)
    assert result.cash_deltas == (PaperAssetAmount("BNB", Decimal("-0.25")),)


def test_realized_pnl_and_same_asset_fee_are_aggregated_into_one_cash_delta() -> None:
    opened = make_record()
    before = settle_paper_fill(BTCUSDT, None, opened).after
    reduced = make_record(
        version=2,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("0.5"),
        price=Decimal("120"),
        fee_amount=Decimal("0.75"),
        fee_asset="USDT",
    )
    result = settle_paper_fill(BTCUSDT, before, reduced)

    assert result.gross_realized_pnl_delta == PaperAssetAmount("USDT", Decimal("10.0"))
    assert result.fee_debits == (PaperAssetAmount("USDT", Decimal("0.75")),)
    assert result.cash_deltas == (PaperAssetAmount("USDT", Decimal("9.25")),)


def test_equal_quote_fee_and_realized_pnl_omit_the_zero_cash_delta() -> None:
    before = settle_paper_fill(BTCUSDT, None, make_record()).after
    reduced = make_record(
        version=2,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("0.5"),
        price=Decimal("102"),
        fee_amount=Decimal("1.0"),
        fee_asset="USDT",
    )

    result = settle_paper_fill(BTCUSDT, before, reduced)

    assert result.gross_realized_pnl_delta.amount == Decimal("1.0")
    assert result.fee_debits == (PaperAssetAmount("USDT", Decimal("1.0")),)
    assert result.cash_deltas == ()


def test_realized_pnl_and_other_asset_fee_remain_separate_and_sorted() -> None:
    before = settle_paper_fill(BTCUSDT, None, make_record()).after
    reduced = make_record(
        version=2,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("0.5"),
        price=Decimal("120"),
        fee_amount=Decimal("0.2"),
        fee_asset="BNB",
    )
    result = settle_paper_fill(BTCUSDT, before, reduced)
    assert result.cash_deltas == (
        PaperAssetAmount("BNB", Decimal("-0.2")),
        PaperAssetAmount("USDT", Decimal("10.0")),
    )


def test_exact_replay_returns_zero_delta_and_reuses_economics_object() -> None:
    opened = make_record(fee_amount=Decimal("0.1"), fee_asset="USDT")
    applied = settle_paper_fill(BTCUSDT, None, opened)
    replayed = settle_paper_fill(BTCUSDT, applied.after, opened)

    assert replayed.disposition is PaperSettlementDisposition.REPLAYED
    assert replayed.after is applied.after
    assert replayed.gross_realized_pnl_delta == PaperAssetAmount("USDT", Decimal("0"))
    assert replayed.fee_debits == ()
    assert replayed.cash_deltas == ()


@pytest.mark.parametrize(
    "changed",
    [
        lambda record: replace(
            record,
            position_fill=replace(
                record.position_fill,
                fill=replace(record.position_fill.fill, price=Decimal("100.0")),
            ),
        ),
        lambda record: replace(
            record,
            position_fill=replace(
                record.position_fill,
                fill=replace(record.position_fill.fill, fee_amount=Decimal("0.10")),
            ),
        ),
    ],
)
def test_same_identity_with_different_decimal_quantum_is_a_conflict(changed) -> None:
    opened = make_record(fee_amount=Decimal("0.1"), fee_asset="USDT")
    before = settle_paper_fill(BTCUSDT, None, opened).after
    with pytest.raises(InvalidPaperSettlement):
        settle_paper_fill(BTCUSDT, before, changed(opened))


def test_mismatched_instrument_or_prior_history_fails_closed() -> None:
    with pytest.raises(InvalidPaperSettlement, match="symbol"):
        settle_paper_fill(BNBBTC, None, make_record(symbol="BTCUSDT"))

    before = settle_paper_fill(BTCUSDT, None, make_record()).after
    with pytest.raises(InvalidPaperSettlement):
        settle_paper_fill(
            BNBBTC,
            before,
            make_record(
                version=2,
                symbol="BNBBTC",
                client_order_id="other-open",
                trade_id="other-trade",
            ),
        )

    changed_denomination = PaperLinearInstrument("BTCUSDT", "DOGE", "CHF")
    with pytest.raises(InvalidPaperSettlement, match="prior settlement"):
        settle_paper_fill(
            changed_denomination,
            before,
            make_record(
                version=2,
                client_order_id="reduce-renominated",
                trade_id="reduce-renominated",
                effect=PositionEffect.REDUCE_ONLY,
                quantity=Decimal("0.5"),
                price=Decimal("120"),
            ),
        )


def test_settlement_arithmetic_ignores_hostile_ambient_decimal_context() -> None:
    opened = make_record(
        quantity=Decimal("1000000000000000000000000000001"),
        price=Decimal("0.000000000000000000000000000003"),
    )
    before = settle_paper_fill(BTCUSDT, None, opened).after
    reduced = make_record(
        version=2,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("1000000000000000000000000000000"),
        price=Decimal("0.000000000000000000000000000004"),
        fee_amount=Decimal("0.1"),
        fee_asset="USDT",
    )
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        result = settle_paper_fill(BTCUSDT, before, reduced)

    assert result.gross_realized_pnl_delta.amount == Decimal("1")
    assert result.cash_deltas == (PaperAssetAmount("USDT", Decimal("0.9")),)


@pytest.mark.parametrize(
    "record",
    [
        make_record(fee_amount=Decimal("1E-1000000000"), fee_asset="USDT"),
        make_record(quantity=Decimal("9" * 6000), price=Decimal("9" * 6000)),
    ],
)
def test_unsupported_large_exact_arithmetic_is_wrapped(record) -> None:
    with pytest.raises(InvalidPaperSettlement):
        settle_paper_fill(BTCUSDT, None, record)


def test_contract_values_are_frozen_slotted_and_hashable() -> None:
    result = settle_paper_fill(BTCUSDT, None, make_record())
    assert type(result) is PaperSettlement
    values = (BTCUSDT, PaperAssetAmount("USDT", Decimal("1")), result)
    assert all(not hasattr(value, "__dict__") for value in values)
    assert len({hash(value) for value in values}) == len(values)
    with pytest.raises(FrozenInstanceError):
        result.cash_deltas = ()


def test_checkpoint_is_only_created_by_the_settlement_factory() -> None:
    economics = settle_paper_fill(BTCUSDT, None, make_record()).after.economics

    with pytest.raises(TypeError, match="returned by settle_paper_fill"):
        PaperSettlementCheckpoint(BTCUSDT, economics)


def test_contract_values_reject_generated_setstate_mutation() -> None:
    result = settle_paper_fill(
        BTCUSDT,
        None,
        make_record(fee_amount=Decimal("0.1"), fee_asset="USDT"),
    )
    original_identity = (repr(result), hash(result))
    visited: set[int] = set()
    protected_types: set[type] = set()

    def assert_protected(value: object) -> None:
        if id(value) in visited or not is_dataclass(value) or isinstance(value, type):
            return
        visited.add(id(value))
        state = [getattr(value, field.name) for field in fields(value)]
        if hasattr(value, "__setstate__"):
            protected_types.add(type(value))
            with pytest.raises(TypeError, match="state mutation"):
                value.__setstate__(state)
        for field in fields(value):
            child = getattr(value, field.name)
            if isinstance(child, tuple):
                for item in child:
                    assert_protected(item)
            else:
                assert_protected(child)

    assert_protected(result)

    assert protected_types == {
        ConfirmedFill,
        OrderIntent,
        PaperAssetAmount,
        PaperCostLot,
        PaperEconomics,
        PaperFeeTotal,
        PaperFillRecord,
        PaperLinearInstrument,
        PaperSettlement,
        PaperSettlementCheckpoint,
        Position,
        PositionExitContext,
        PositionFill,
        PositionInstruction,
    }
    assert (repr(result), hash(result)) == original_identity


def test_validated_state_restore_preserves_copy_and_pickle_round_trips() -> None:
    result = settle_paper_fill(
        BTCUSDT,
        None,
        make_record(fee_amount=Decimal("0.1"), fee_asset="USDT"),
    )

    for restored in (
        copy.copy(result),
        copy.deepcopy(result),
        pickle.loads(pickle.dumps(result)),
    ):
        assert restored == result
        assert hash(restored) == hash(result)


def test_validated_state_restore_rejects_and_cleans_invalid_payload() -> None:
    economics = settle_paper_fill(BTCUSDT, None, make_record()).after.economics
    field_names = tuple(field.name for field in fields(economics))
    state = [getattr(economics, name) for name in field_names]
    state[field_names.index("gross_realized_pnl")] = Decimal("100")
    restored = object.__new__(PaperEconomics)

    with pytest.raises(ValueError, match="derived"):
        restored.__setstate__(state)

    assert all(not hasattr(restored, name) for name in field_names)
    with pytest.raises(TypeError, match="invalid frozen domain state"):
        restored.__setstate__(())


def test_checkpoint_economics_cannot_be_mutated_to_corrupt_cash_delta() -> None:
    applied = settle_paper_fill(BTCUSDT, None, make_record())
    economics = applied.after.economics
    field_names = tuple(field.name for field in fields(economics))
    state = [getattr(economics, name) for name in field_names]
    state[field_names.index("gross_realized_pnl")] = Decimal("100")

    with pytest.raises(TypeError, match="state mutation"):
        economics.__setstate__(state)

    reduced = make_record(
        version=2,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        quantity=Decimal("0.5"),
        price=Decimal("120"),
    )
    result = settle_paper_fill(BTCUSDT, applied.after, reduced)
    assert result.gross_realized_pnl_delta == PaperAssetAmount("USDT", Decimal("10.0"))
    assert result.cash_deltas == (PaperAssetAmount("USDT", Decimal("10.0")),)


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: PaperLinearInstrument(" padded ", "BTC", "USDT"), ValueError),
        (lambda: PaperLinearInstrument("BTCUSDT", "BTC", "BTC"), ValueError),
        (lambda: PaperAssetAmount("USDT", 1), TypeError),
        (lambda: PaperAssetAmount("USDT", Decimal("NaN")), ValueError),
    ],
)
def test_leaf_contracts_reject_invalid_direct_construction(factory, error) -> None:
    with pytest.raises(error):
        factory()


@pytest.mark.parametrize("value", ["bad\x00asset", "bad\ud800asset"])
def test_instrument_and_amount_reject_non_durable_text(value) -> None:
    with pytest.raises(ValueError):
        PaperLinearInstrument("BTCUSDT", value, "USDT")
    with pytest.raises(ValueError):
        PaperAssetAmount(value, Decimal("1"))


def test_linear_instrument_signature_excludes_multiplier_and_inverse_terms() -> None:
    assert tuple(inspect.signature(PaperLinearInstrument).parameters) == (
        "symbol",
        "base_asset",
        "quote_asset",
    )
    with pytest.raises(TypeError):
        PaperLinearInstrument("BTCUSDT", "BTC", "USDT", multiplier=Decimal("1"))


@pytest.mark.parametrize(
    "change",
    [
        {"disposition": PaperSettlementDisposition.REPLAYED},
        {"gross_realized_pnl_delta": PaperAssetAmount("USDT", Decimal("0.0"))},
        {"fee_debits": (PaperAssetAmount("USDT", Decimal("1")),)},
        {"cash_deltas": (PaperAssetAmount("USDT", Decimal("0")),)},
    ],
)
def test_direct_construction_cannot_forge_derived_fields(change) -> None:
    result = settle_paper_fill(BTCUSDT, None, make_record())
    with pytest.raises(ValueError):
        replace(result, **change)


def test_direct_construction_cannot_forge_after_decimal_quantum() -> None:
    result = settle_paper_fill(BTCUSDT, None, make_record())
    forged_economics = object.__new__(type(result.after.economics))
    for field in fields(result.after.economics):
        object.__setattr__(
            forged_economics,
            field.name,
            (
                Decimal("100.0")
                if field.name == "open_cost"
                else getattr(result.after.economics, field.name)
            ),
        )
    forged_after = object.__new__(PaperSettlementCheckpoint)
    object.__setattr__(forged_after, "instrument", result.after.instrument)
    object.__setattr__(forged_after, "economics", forged_economics)
    with pytest.raises(ValueError):
        replace(result, after=forged_after)


def test_replays_keep_one_compact_checkpoint_without_recursive_growth() -> None:
    opened = make_record()
    result = settle_paper_fill(BTCUSDT, None, opened)
    checkpoint = result.after

    for _ in range(1_200):
        result = settle_paper_fill(BTCUSDT, result.after, opened)

    assert result.disposition is PaperSettlementDisposition.REPLAYED
    assert result.after is checkpoint
    assert len(result.after.economics.records) == 1
    hash(result)
    repr(result)


def _uses_paper_settlement(source: str) -> bool:
    tree = ast.parse(source)
    module = "trading.domain.paper_settlement"
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
            if node.level and imported_module.endswith("paper_settlement"):
                return True
            if node.level and imported & (PUBLIC_EXPORTS | {"paper_settlement", "*"}):
                return True
            if node.level and "domain" in imported:
                return True
            if imported_module == "trading.domain" and imported & (
                PUBLIC_EXPORTS | {"paper_settlement", "*"}
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
        if not isinstance(node, ast.Call):
            continue
        target = (
            node.args[0].value
            if node.args and isinstance(node.args[0], ast.Constant)
            else next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
        )
        if not isinstance(target, str):
            continue
        is_builtin_import = (
            isinstance(node.func, ast.Name) and node.func.id in builtin_import_aliases
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "__import__"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in builtins_aliases
        )
        is_import_module = (
            isinstance(node.func, ast.Name) and node.func.id in import_module_aliases
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        )
        if is_builtin_import and (target == "trading" or target.startswith("trading.")):
            return True
        if is_import_module and target in {module, "trading.domain", "trading"}:
            return True
        if target.startswith("."):
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
                    if importlib.util.resolve_name(target, package) in {
                        module,
                        "trading.domain",
                        "trading",
                    }:
                        return True
                except (ImportError, ValueError):
                    pass
    return False


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.paper_settlement import PaperSettlement",
        "from trading.domain import PaperSettlement",
        "from trading.domain import *\nvalue = PaperSettlement",
        "import trading.domain.paper_settlement as settlement",
        "import trading as root\nroot.domain.PaperSettlement",
        "from trading import domain as domain\ndomain.PaperSettlement",
        "from . import *\nvalue = PaperSettlement",
        "from .paper_settlement import settle_paper_fill",
        "from importlib import import_module\n"
        "import_module('trading.domain.paper_settlement')",
        "from importlib import import_module\n"
        "import_module('.paper_settlement', 'trading.domain')",
        "__import__('trading.domain.paper_settlement')",
        "load = __import__\nload('trading.domain.paper_settlement')",
        "import importlib\nload = importlib.import_module\n"
        "load('trading.domain.paper_settlement')",
        "import importlib as loader\n"
        "loader.import_module(name='.paper_settlement', package='trading.domain')",
    ],
)
def test_consumer_detector_catches_direct_facade_and_dynamic_imports(source) -> None:
    assert _uses_paper_settlement(source)


def test_paper_settlement_is_pure_and_has_no_runtime_consumer() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "domain" / "paper_settlement.py"
    facade_path = root / "trading" / "domain" / "__init__.py"
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path in {module_path, facade_path}
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_paper_settlement(source_path.read_text(encoding="utf-8")):
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
