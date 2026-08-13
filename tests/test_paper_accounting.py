"""Contract tests for pure global paper-account admission and accounting."""

import ast
import copy
import importlib.util
import pickle
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Inexact, Rounded, localcontext
from pathlib import Path

import pytest

from trading.domain.order_lifecycle import ConfirmedFill
from trading.domain.orders import OrderIntent, OrderSide, OrderType
from trading.domain.paper_accounting import (
    InvalidPaperAccountTransition,
    PaperAccount,
    PaperAccountAdmission,
    PaperAccountAdmissionDisposition,
    PaperAccountBalance,
    PaperAccountPolicy,
    PaperAccountPosting,
    PaperAccountPostingBucket,
    PaperAccountSettlementRecord,
    PaperAccountState,
    PaperMarginReservation,
    admit_paper_settlement,
    new_paper_account,
)
from trading.domain.paper_economics import PaperFillRecord
from trading.domain.paper_settlement import (
    PaperLinearInstrument,
    PaperSettlement,
    settle_paper_fill,
)
from trading.domain.positions import (
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
    "InvalidPaperAccountTransition",
    "PaperAccount",
    "PaperAccountAdmission",
    "PaperAccountAdmissionDisposition",
    "PaperAccountBalance",
    "PaperAccountPolicy",
    "PaperAccountPosting",
    "PaperAccountPostingBucket",
    "PaperAccountSettlementRecord",
    "PaperAccountState",
    "PaperMarginReservation",
    "admit_paper_settlement",
    "new_paper_account",
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
    price: Decimal = Decimal("1"),
    fee_amount: Decimal = Decimal("0"),
    fee_asset: str | None = None,
    leverage: int = 3,
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


def make_settlement(
    record: PaperFillRecord,
    *,
    instrument: PaperLinearInstrument = BTCUSDT,
    before=None,
) -> PaperSettlement:
    return settle_paper_fill(instrument, before, record)


def make_account(
    available: Decimal = Decimal("10"),
    *,
    reserved: Decimal = Decimal("0"),
    quantum: Decimal = Decimal("0.01"),
    collateral_asset: str = "USDT",
    extra_balances: tuple[PaperAccountBalance, ...] = (),
) -> PaperAccount:
    opening_balances = tuple(
        sorted(
            (PaperAccountBalance(collateral_asset, available, reserved),)
            + extra_balances,
            key=lambda value: value.asset,
        )
    )
    return new_paper_account(
        PaperAccountPolicy("paper-main", collateral_asset, quantum),
        opening_balances,
    )


def balance(account: PaperAccount, asset: str) -> PaperAccountBalance:
    return next(value for value in account.balances if value.asset == asset)


def reservation(account: PaperAccount, position_key: str) -> Decimal:
    return next(
        (
            value.amount
            for value in account.reservations
            if value.position_key == position_key
        ),
        Decimal("0"),
    )


def approve(
    account: PaperAccount,
    account_version: int,
    settlement: PaperSettlement,
) -> PaperAccountAdmission:
    result = admit_paper_settlement(account, account_version, settlement)
    assert result.disposition is PaperAccountAdmissionDisposition.APPLIED
    assert result.reasons == ()
    return result


def posting_total(postings: tuple[PaperAccountPosting, ...], asset: str) -> Decimal:
    return sum(
        (posting.amount for posting in postings if posting.asset == asset),
        Decimal("0"),
    )


def settlement_total(settlement: PaperSettlement, asset: str) -> Decimal:
    return next(
        (amount.amount for amount in settlement.cash_deltas if amount.asset == asset),
        Decimal("0"),
    )


def test_new_account_has_exact_opening_state_without_synthetic_records() -> None:
    policy = PaperAccountPolicy("paper-main", "USDT", Decimal("0.01"))
    opening = (
        PaperAccountBalance("BNB", Decimal("2.50"), Decimal("0")),
        PaperAccountBalance("USDT", Decimal("100.00"), Decimal("0")),
    )

    account = new_paper_account(policy, opening)

    assert type(account) is PaperAccount
    assert account.policy is policy
    assert account.balances == opening
    assert account.reservations == ()
    assert account.records == ()
    assert account.state is PaperAccountState.ACTIVE


def test_global_account_order_prevents_two_positions_spending_same_collateral() -> None:
    account = make_account(Decimal("1.00"))
    first = make_settlement(
        make_record(
            position_key="position-a",
            client_order_id="open-a",
            trade_id="trade-a",
            price=Decimal("1.80"),
        )
    )
    second = make_settlement(
        make_record(
            position_key="position-b",
            client_order_id="open-b",
            trade_id="trade-b",
            price=Decimal("1.80"),
        )
    )

    accepted = approve(account, 1, first)
    rejected = admit_paper_settlement(accepted.after, 2, second)

    assert balance(accepted.after, "USDT") == PaperAccountBalance(
        "USDT", Decimal("0.40"), Decimal("0.60")
    )
    assert rejected.disposition is PaperAccountAdmissionDisposition.REJECTED
    assert rejected.after is accepted.after
    assert rejected.postings == ()
    assert rejected.reasons
    assert rejected.after.records == accepted.after.records


def test_rejected_admission_does_not_consume_the_global_account_version() -> None:
    account = make_account(Decimal("0.50"))
    too_large = make_settlement(make_record(price=Decimal("1.80")))
    small = make_settlement(
        make_record(
            position_key="position-small",
            client_order_id="open-small",
            trade_id="trade-small",
            price=Decimal("0.30"),
        )
    )

    rejected = admit_paper_settlement(account, 1, too_large)
    accepted = approve(rejected.after, 1, small)

    assert rejected.after is account
    assert tuple(record.account_version for record in accepted.after.records) == (1,)


def test_applied_records_require_contiguous_global_account_versions() -> None:
    account = make_account()
    first = make_settlement(make_record())
    after_first = approve(account, 1, first).after
    second = make_settlement(
        make_record(
            position_key="position-2",
            client_order_id="open-2",
            trade_id="trade-2",
        )
    )

    with pytest.raises(InvalidPaperAccountTransition):
        admit_paper_settlement(after_first, 3, second)

    accepted = approve(after_first, 2, second)
    assert tuple(record.account_version for record in accepted.after.records) == (1, 2)


@pytest.mark.parametrize("account_version", [True, 0, -1, 1.0, "1"])
def test_account_version_is_a_strict_positive_integer(account_version) -> None:
    account = make_account()
    settlement = make_settlement(make_record())

    with pytest.raises((TypeError, ValueError)):
        admit_paper_settlement(account, account_version, settlement)


def test_exact_replay_after_later_records_is_a_no_op_on_current_account() -> None:
    account = make_account()
    first = make_settlement(make_record())
    after_first = approve(account, 1, first).after
    second = make_settlement(
        make_record(
            position_key="position-2",
            client_order_id="open-2",
            trade_id="trade-2",
        )
    )
    current = approve(after_first, 2, second).after

    replay = admit_paper_settlement(current, 1, first)

    assert replay.disposition is PaperAccountAdmissionDisposition.REPLAYED
    assert replay.reasons == ()
    assert replay.before is current
    assert replay.after is current
    assert replay.postings == ()
    assert replay.account_version == 1
    assert replay.settlement is first


def test_same_account_version_with_another_fill_conflicts() -> None:
    account = make_account()
    current = approve(account, 1, make_settlement(make_record())).after
    conflict = make_settlement(
        make_record(
            position_key="position-2",
            client_order_id="open-2",
            trade_id="trade-2",
        )
    )

    with pytest.raises(InvalidPaperAccountTransition):
        admit_paper_settlement(current, 1, conflict)


def test_same_fill_identity_at_another_account_version_conflicts() -> None:
    account = make_account()
    settlement = make_settlement(make_record())
    current = approve(account, 1, settlement).after

    with pytest.raises(InvalidPaperAccountTransition):
        admit_paper_settlement(current, 2, settlement)


@pytest.mark.parametrize(
    "changed_record",
    [
        make_record(price=Decimal("1.0")),
        make_record(fee_amount=Decimal("0.10"), fee_asset="USDT"),
    ],
)
def test_same_fill_or_event_identity_with_changed_payload_conflicts(
    changed_record,
) -> None:
    account = make_account()
    current = approve(account, 1, make_settlement(make_record())).after
    changed = make_settlement(changed_record)

    with pytest.raises(InvalidPaperAccountTransition):
        admit_paper_settlement(current, 2, changed)


@pytest.mark.parametrize(
    ("quantum", "expected_margin"),
    [
        (Decimal("0.01"), Decimal("0.34")),
        (Decimal("0.03"), Decimal("0.36")),
        (Decimal("0.0001"), Decimal("0.3334")),
    ],
)
def test_margin_division_uses_exact_ceiling_to_explicit_quantum(
    quantum, expected_margin
) -> None:
    account = make_account(Decimal("2"), quantum=quantum)
    settlement = make_settlement(make_record(price=Decimal("1"), leverage=3))

    result = approve(account, 1, settlement)

    assert reservation(result.after, "position-1") == expected_margin
    assert balance(result.after, "USDT").reserved == expected_margin


def test_margin_ceiling_ignores_hostile_ambient_decimal_context() -> None:
    account = make_account(Decimal("2"), quantum=Decimal("0.01"))
    settlement = make_settlement(make_record(price=Decimal("1"), leverage=3))

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        result = approve(account, 1, settlement)

    assert reservation(result.after, "position-1") == Decimal("0.34")


def test_scale_in_and_reductions_recompute_target_margin_without_drift() -> None:
    account = make_account(Decimal("5"))
    opened_record = make_record(price=Decimal("1"))
    opened = make_settlement(opened_record)
    after_open = approve(account, 1, opened).after

    scaled_record = make_record(
        version=2,
        client_order_id="scale-1",
        trade_id="scale-1",
        price=Decimal("2"),
    )
    scaled = make_settlement(scaled_record, before=opened.after)
    after_scale = approve(after_open, 2, scaled).after

    partial_record = make_record(
        version=3,
        client_order_id="reduce-1",
        trade_id="reduce-1",
        effect=PositionEffect.REDUCE_ONLY,
        price=Decimal("1"),
    )
    partial = make_settlement(partial_record, before=scaled.after)
    after_partial = approve(after_scale, 3, partial).after

    final_record = make_record(
        version=4,
        client_order_id="reduce-2",
        trade_id="reduce-2",
        effect=PositionEffect.REDUCE_ONLY,
        price=Decimal("2"),
    )
    final = make_settlement(final_record, before=partial.after)
    after_final = approve(after_partial, 4, final).after

    assert reservation(after_open, "position-1") == Decimal("0.34")
    assert reservation(after_scale, "position-1") == Decimal("1.00")
    assert reservation(after_partial, "position-1") == Decimal("0.67")
    assert reservation(after_final, "position-1") == Decimal("0")
    assert balance(after_final, "USDT") == PaperAccountBalance(
        "USDT", Decimal("5.00"), Decimal("0.00")
    )
    assert after_final.reservations == ()


def test_quote_fee_and_margin_are_debited_together_when_funded() -> None:
    account = make_account(Decimal("1.00"))
    settlement = make_settlement(
        make_record(
            price=Decimal("1.80"),
            fee_amount=Decimal("0.25"),
            fee_asset="USDT",
        )
    )

    result = approve(account, 1, settlement)

    assert balance(result.after, "USDT") == PaperAccountBalance(
        "USDT", Decimal("0.15"), Decimal("0.60")
    )
    assert posting_total(result.postings, "USDT") == Decimal("-0.25")


def test_foreign_fee_requires_and_debits_its_own_asset() -> None:
    account = make_account(
        Decimal("1.00"),
        extra_balances=(PaperAccountBalance("BNB", Decimal("0.10"), Decimal("0")),),
    )
    settlement = make_settlement(
        make_record(
            price=Decimal("1.80"),
            fee_amount=Decimal("0.10"),
            fee_asset="BNB",
        )
    )

    result = approve(account, 1, settlement)

    assert balance(result.after, "USDT") == PaperAccountBalance(
        "USDT", Decimal("0.40"), Decimal("0.60")
    )
    assert balance(result.after, "BNB") == PaperAccountBalance(
        "BNB", Decimal("0.00"), Decimal("0")
    )
    assert posting_total(result.postings, "BNB") == Decimal("-0.10")


@pytest.mark.parametrize(
    "extra_balances",
    [
        (),
        (PaperAccountBalance("BNB", Decimal("0.09"), Decimal("0")),),
    ],
)
def test_missing_or_insufficient_foreign_fee_asset_rejects(extra_balances) -> None:
    account = make_account(Decimal("1.00"), extra_balances=extra_balances)
    settlement = make_settlement(
        make_record(
            price=Decimal("1.80"),
            fee_amount=Decimal("0.10"),
            fee_asset="BNB",
        )
    )

    result = admit_paper_settlement(account, 1, settlement)

    assert result.disposition is PaperAccountAdmissionDisposition.REJECTED
    assert result.after is account
    assert result.postings == ()


def test_combined_quote_fee_and_margin_can_reject_when_each_alone_fits() -> None:
    account = make_account(Decimal("1.00"))
    settlement = make_settlement(
        make_record(
            price=Decimal("3"),
            fee_amount=Decimal("0.01"),
            fee_asset="USDT",
        )
    )

    result = admit_paper_settlement(account, 1, settlement)

    assert result.disposition is PaperAccountAdmissionDisposition.REJECTED
    assert result.after is account
    assert result.postings == ()
    assert result.reasons


def test_reduce_only_records_loss_and_may_make_account_insolvent() -> None:
    account = make_account(Decimal("0.34"))
    opened = make_settlement(make_record(price=Decimal("1")))
    after_open = approve(account, 1, opened).after
    reduced = make_settlement(
        make_record(
            version=2,
            client_order_id="reduce-1",
            trade_id="reduce-1",
            effect=PositionEffect.REDUCE_ONLY,
            price=Decimal("0.5"),
        ),
        before=opened.after,
    )

    result = approve(after_open, 2, reduced)

    assert result.after.state is PaperAccountState.INSOLVENT
    assert balance(result.after, "USDT") == PaperAccountBalance(
        "USDT", Decimal("-0.16"), Decimal("0.00")
    )
    assert result.after.reservations == ()
    assert posting_total(result.postings, "USDT") == Decimal("-0.5")


def test_insolvent_account_rejects_new_open_exposure() -> None:
    account = make_account(Decimal("0.34"))
    opened = make_settlement(make_record(price=Decimal("1")))
    after_open = approve(account, 1, opened).after
    reduced = make_settlement(
        make_record(
            version=2,
            client_order_id="reduce-1",
            trade_id="reduce-1",
            effect=PositionEffect.REDUCE_ONLY,
            price=Decimal("0.5"),
        ),
        before=opened.after,
    )
    insolvent = approve(after_open, 2, reduced).after
    another_open = make_settlement(
        make_record(
            position_key="position-2",
            client_order_id="open-2",
            trade_id="trade-2",
            price=Decimal("0.01"),
        )
    )

    result = admit_paper_settlement(insolvent, 3, another_open)

    assert result.disposition is PaperAccountAdmissionDisposition.REJECTED
    assert result.after is insolvent
    assert result.postings == ()


def test_postings_conserve_each_settlement_asset_delta() -> None:
    account = make_account(
        Decimal("10"),
        extra_balances=(PaperAccountBalance("BNB", Decimal("1"), Decimal("0")),),
    )
    opened = make_settlement(make_record(fee_amount=Decimal("0.10"), fee_asset="BNB"))
    result = approve(account, 1, opened)

    assets = {posting.asset for posting in result.postings} | {
        delta.asset for delta in opened.cash_deltas
    }
    assert assets == {"BNB", "USDT"}
    for asset in assets:
        assert posting_total(result.postings, asset) == settlement_total(opened, asset)
    assert all(posting.amount for posting in result.postings)


def test_zero_margin_change_and_zero_cash_delta_emit_no_postings() -> None:
    account = make_account(Decimal("1"), quantum=Decimal("0.01"))
    opened = make_settlement(make_record(quantity=Decimal("0.01"), price=Decimal("1")))
    after_open = approve(account, 1, opened).after
    scaled = make_settlement(
        make_record(
            version=2,
            client_order_id="scale-1",
            trade_id="scale-1",
            quantity=Decimal("0.001"),
            price=Decimal("1"),
        ),
        before=opened.after,
    )

    result = approve(after_open, 2, scaled)

    assert reservation(result.after, "position-1") == Decimal("0.01")
    assert result.postings == ()


def test_instrument_settlement_asset_must_match_account_collateral() -> None:
    account = make_account(collateral_asset="USDT")
    settlement = make_settlement(
        make_record(symbol="BNBBTC"),
        instrument=BNBBTC,
    )

    with pytest.raises(InvalidPaperAccountTransition):
        admit_paper_settlement(account, 1, settlement)


def test_margin_quantum_payload_identity_is_not_silently_rewritten() -> None:
    fine = make_account(Decimal("2"), quantum=Decimal("0.01"))
    coarse = make_account(Decimal("2"), quantum=Decimal("0.010"))

    assert (
        fine.policy.margin_quantum.as_tuple() != coarse.policy.margin_quantum.as_tuple()
    )
    assert fine.policy.margin_quantum == coarse.policy.margin_quantum
    assert fine.policy.margin_quantum is not coarse.policy.margin_quantum

    applied = approve(
        fine,
        1,
        make_settlement(make_record(price=Decimal("1"), leverage=3)),
    ).after
    changed_policy = replace(
        applied.policy,
        margin_quantum=Decimal("0.010"),
    )
    with pytest.raises(ValueError, match="derived"):
        replace(applied, policy=changed_policy)


def test_rejected_admission_cannot_substitute_an_equal_policy_quantum() -> None:
    account = make_account(Decimal("0.50"), quantum=Decimal("0.01"))
    rejected = admit_paper_settlement(
        account,
        1,
        make_settlement(make_record(price=Decimal("1.80"))),
    )
    forged_after = replace(
        account,
        policy=replace(account.policy, margin_quantum=Decimal("0.010")),
    )

    assert rejected.disposition is PaperAccountAdmissionDisposition.REJECTED
    assert forged_after.policy.margin_quantum == account.policy.margin_quantum
    with pytest.raises(ValueError, match="after is not derived"):
        replace(rejected, after=forged_after)


def test_margin_ratio_does_not_depend_on_int_string_digit_limits() -> None:
    exact_cost = Decimal((0, (1,) * 5_000, 0))
    account = make_account(exact_cost, quantum=Decimal("1"))
    settlement = make_settlement(
        make_record(price=exact_cost, leverage=1),
    )

    applied = approve(account, 1, settlement)

    assert reservation(applied.after, "position-1").as_tuple() == (
        exact_cost.as_tuple()
    )
    assert balance(applied.after, "USDT").available == Decimal("0")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PaperAccountPolicy(" padded ", "USDT", Decimal("0.01")),
        lambda: PaperAccountPolicy("paper-main", "USDT", Decimal("0")),
        lambda: PaperAccountPolicy("paper-main", "USDT", Decimal("NaN")),
        lambda: PaperAccountBalance("USDT", Decimal("1"), Decimal("-1")),
        lambda: PaperMarginReservation("position-1", Decimal("0")),
        lambda: PaperAccountPosting(
            "USDT", PaperAccountPostingBucket.AVAILABLE, Decimal("0")
        ),
    ],
)
def test_leaf_values_reject_invalid_direct_construction(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_direct_account_and_admission_construction_revalidates_fields() -> None:
    account = make_account()
    direct_account = PaperAccount(
        policy=account.policy,
        opening_balances=account.opening_balances,
        balances=account.balances,
        reservations=account.reservations,
        records=account.records,
        state=account.state,
    )
    result = approve(account, 1, make_settlement(make_record()))
    direct_result = PaperAccountAdmission(
        before=result.before,
        account_version=result.account_version,
        settlement=result.settlement,
        disposition=result.disposition,
        after=result.after,
        postings=result.postings,
        reasons=result.reasons,
    )

    assert direct_account == account
    assert direct_result == result
    with pytest.raises(ValueError, match="state must be derived"):
        replace(account, state=PaperAccountState.INSOLVENT)


def test_account_values_are_frozen_slotted_and_hashable() -> None:
    account = make_account()
    settlement = make_settlement(make_record())
    result = approve(account, 1, settlement)
    values = (
        account.policy,
        account.balances[0],
        result.after,
        result,
        result.after.records[0],
        result.after.reservations[0],
        result.postings[0],
    )

    assert all(not hasattr(value, "__dict__") for value in values)
    assert all(isinstance(hash(value), int) for value in values)
    with pytest.raises(FrozenInstanceError):
        result.after.state = PaperAccountState.INSOLVENT


def test_reachable_account_values_reject_setstate_mutation() -> None:
    account = make_account()
    result = approve(account, 1, make_settlement(make_record()))
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

    assert {
        PaperAccount,
        PaperAccountAdmission,
        PaperAccountBalance,
        PaperAccountPolicy,
        PaperAccountPosting,
        PaperAccountSettlementRecord,
        PaperMarginReservation,
    } <= protected_types
    assert (repr(result), hash(result)) == original_identity


def test_copy_and_pickle_round_trips_revalidate_account_values() -> None:
    account = make_account()
    result = approve(account, 1, make_settlement(make_record()))

    for value in (result.after, result):
        for restored in (
            copy.copy(value),
            copy.deepcopy(value),
            pickle.loads(pickle.dumps(value)),
        ):
            assert restored == value
            assert hash(restored) == hash(value)


def test_direct_account_state_restore_cannot_forge_derived_balance() -> None:
    account = approve(
        make_account(),
        1,
        make_settlement(make_record()),
    ).after
    field_names = tuple(field.name for field in fields(account))
    state = [getattr(account, name) for name in field_names]
    state[field_names.index("balances")] = (
        PaperAccountBalance("USDT", Decimal("999"), Decimal("0")),
    )
    restored = object.__new__(PaperAccount)

    with pytest.raises((TypeError, ValueError)):
        restored.__setstate__(state)

    assert all(not hasattr(restored, name) for name in field_names)


def test_direct_admission_construction_cannot_forge_postings() -> None:
    result = approve(make_account(), 1, make_settlement(make_record()))

    with pytest.raises((TypeError, ValueError)):
        replace(result, postings=())


def _uses_paper_accounting(source: str) -> bool:
    tree = ast.parse(source)
    module = "trading.domain.paper_accounting"
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
            if node.level and imported_module.endswith("paper_accounting"):
                return True
            if node.level and imported & (PUBLIC_EXPORTS | {"paper_accounting", "*"}):
                return True
            if node.level and "domain" in imported:
                return True
            if imported_module == "trading.domain" and imported & (
                PUBLIC_EXPORTS | {"paper_accounting", "*"}
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
                except ImportError, ValueError:
                    pass
    return False


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.paper_accounting import PaperAccount",
        "from trading.domain import PaperAccount",
        "from trading.domain import *\nvalue = PaperAccount",
        "import trading.domain.paper_accounting as accounting",
        "import trading as root\nroot.domain.PaperAccount",
        "from trading import domain as domain\ndomain.PaperAccount",
        "from . import *\nvalue = PaperAccount",
        "from .paper_accounting import admit_paper_settlement",
        "from importlib import import_module\n"
        "import_module('trading.domain.paper_accounting')",
        "from importlib import import_module\n"
        "import_module('.paper_accounting', 'trading.domain')",
        "__import__('trading.domain.paper_accounting')",
        "load = __import__\nload('trading.domain.paper_accounting')",
        "import importlib\nload = importlib.import_module\n"
        "load('trading.domain.paper_accounting')",
        "import importlib as loader\n"
        "loader.import_module(name='.paper_accounting', package='trading.domain')",
    ],
)
def test_consumer_detector_catches_direct_facade_and_dynamic_imports(source) -> None:
    assert _uses_paper_accounting(source)


@pytest.mark.parametrize(
    "source",
    [
        "from trading.domain.paper_settlement import PaperSettlement",
        "from trading.domain import PaperSettlement",
        "from trading.application import OrderService",
        "name = 'trading.domain.paper_accounting'",
    ],
)
def test_consumer_detector_allows_unrelated_forms(source) -> None:
    assert not _uses_paper_accounting(source)


def test_paper_accounting_is_pure_and_has_no_runtime_consumer() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "domain" / "paper_accounting.py"
    facade_path = root / "trading" / "domain" / "__init__.py"
    codec_path = root / "trading" / "persistence" / "paper_account_journal_codec.py"
    repository_path = root / "trading" / "persistence" / "paper_account_journal.py"
    owner_path = root / "trading" / "persistence" / "atomic_paper_account_owner.py"
    readiness_path = root / "trading" / "persistence" / "paper_account_readiness.py"
    reconciliation_contract_path = (
        root / "trading" / "application" / "legacy_snapshot_reconciliation.py"
    )
    reconciliation_adapter_path = (
        root / "trading" / "persistence" / "postgres_legacy_snapshot_reconciliation.py"
    )
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path
            in {
                module_path,
                facade_path,
                codec_path,
                repository_path,
                owner_path,
                readiness_path,
                reconciliation_contract_path,
                reconciliation_adapter_path,
            }
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_paper_accounting(source_path.read_text(encoding="utf-8")):
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
