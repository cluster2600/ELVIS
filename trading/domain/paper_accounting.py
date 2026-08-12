"""Pure account admission and postings for exact paper settlements."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from trading.domain._decimal import exact_decimal_product, exact_decimal_sum
from trading.domain._validation import (
    protect_frozen_dataclass_state,
    require_clean_text,
    require_positive_decimal,
)
from trading.domain.paper_economics import (
    _decimal_payload_identity,
    _record_payload_identity,
)
from trading.domain.paper_settlement import (
    PaperSettlement,
    PaperSettlementDisposition,
    _amount_identity,
    _economics_identity,
)
from trading.domain.positions import PositionEffect

_BIGINT_MAX = (1 << 63) - 1
_MAX_INTEGER_RATIO_DIGITS = 10_000


def _require_durable_text(name: str, value: object) -> None:
    require_clean_text(name, value)
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{name} is not representable in durable storage")


def _decimal_identity(value: Decimal) -> tuple[object, ...]:
    return _decimal_payload_identity(value)


def _balance_identity(balance: "PaperAccountBalance") -> tuple[object, ...]:
    return (
        balance.asset,
        _decimal_identity(balance.available),
        _decimal_identity(balance.reserved),
    )


def _reservation_identity(
    reservation: "PaperMarginReservation",
) -> tuple[object, ...]:
    return (reservation.position_key, _decimal_identity(reservation.amount))


def _posting_identity(posting: "PaperAccountPosting") -> tuple[object, ...]:
    return (
        posting.asset,
        posting.bucket,
        _decimal_identity(posting.amount),
    )


def _settlement_identity(settlement: PaperSettlement) -> tuple[object, ...]:
    before = settlement.before
    return (
        (
            settlement.instrument.symbol,
            settlement.instrument.base_asset,
            settlement.instrument.quote_asset,
        ),
        (
            None
            if before is None
            else (
                before.instrument.symbol,
                before.instrument.base_asset,
                before.instrument.quote_asset,
                _economics_identity(before.economics),
            )
        ),
        _record_payload_identity(settlement.record),
        (
            settlement.after.instrument.symbol,
            settlement.after.instrument.base_asset,
            settlement.after.instrument.quote_asset,
            _economics_identity(settlement.after.economics),
        ),
        settlement.disposition,
        _amount_identity(settlement.gross_realized_pnl_delta),
        tuple(_amount_identity(amount) for amount in settlement.fee_debits),
        tuple(_amount_identity(amount) for amount in settlement.cash_deltas),
    )


def _record_identity(
    record: "PaperAccountSettlementRecord",
) -> tuple[object, ...]:
    return (record.account_version, _settlement_identity(record.settlement))


def _policy_identity(policy: "PaperAccountPolicy") -> tuple[object, ...]:
    return (
        policy.account_key,
        policy.collateral_asset,
        _decimal_identity(policy.margin_quantum),
    )


def _account_identity(account: "PaperAccount") -> tuple[object, ...]:
    return (
        _policy_identity(account.policy),
        tuple(_balance_identity(balance) for balance in account.opening_balances),
        tuple(_balance_identity(balance) for balance in account.balances),
        tuple(
            _reservation_identity(reservation) for reservation in account.reservations
        ),
        tuple(_record_identity(record) for record in account.records),
        account.state,
    )


def _require_finite_decimal(name: str, value: object) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")


def _decimal_coefficient(digits: tuple[int, ...]) -> int:
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return coefficient


def _ceil_margin_target(
    open_cost: Decimal,
    leverage: int,
    quantum: Decimal,
) -> Decimal:
    """Return ceil(open_cost / leverage / quantum) * quantum exactly."""
    _require_finite_decimal("open_cost", open_cost)
    if open_cost < 0:
        raise ValueError("open_cost must be non-negative")
    if isinstance(leverage, bool) or not isinstance(leverage, int):
        raise TypeError("leverage must be an integer")
    if leverage < 1:
        raise ValueError("leverage must be positive")
    require_positive_decimal("margin_quantum", quantum)
    if not open_cost:
        return Decimal("0")

    cost_tuple = open_cost.as_tuple()
    quantum_tuple = quantum.as_tuple()
    if cost_tuple.sign or quantum_tuple.sign:
        raise ValueError("margin arithmetic requires non-negative values")
    cost_coefficient = _decimal_coefficient(cost_tuple.digits)
    quantum_coefficient = _decimal_coefficient(quantum_tuple.digits)
    exponent_delta = int(cost_tuple.exponent) - int(quantum_tuple.exponent)
    required_digits = (
        len(cost_tuple.digits)
        + len(quantum_tuple.digits)
        + len(str(leverage))
        + abs(exponent_delta)
    )
    if required_digits > _MAX_INTEGER_RATIO_DIGITS:
        raise ValueError("margin arithmetic exceeds the supported precision")

    if exponent_delta >= 0:
        numerator = cost_coefficient * (10**exponent_delta)
        denominator = leverage * quantum_coefficient
    else:
        numerator = cost_coefficient
        denominator = leverage * quantum_coefficient * (10 ** (-exponent_delta))
    units = (numerator + denominator - 1) // denominator
    return exact_decimal_product((Decimal(units), quantum))


class PaperAccountState(str, Enum):
    """Whether the exact paper balances can admit new exposure."""

    ACTIVE = "ACTIVE"
    INSOLVENT = "INSOLVENT"


class PaperAccountPostingBucket(str, Enum):
    """The two explicit buckets affected by account postings."""

    AVAILABLE = "AVAILABLE"
    RESERVED_MARGIN = "RESERVED_MARGIN"


class PaperAccountAdmissionDisposition(str, Enum):
    """Whether a candidate was applied, replayed, or rejected."""

    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"
    REJECTED = "REJECTED"


class InvalidPaperAccountTransition(ValueError):
    """Raised when account history or a candidate settlement conflicts."""


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountPolicy:
    """Explicit collateral and rounding policy for one isolated paper account."""

    account_key: str
    collateral_asset: str
    margin_quantum: Decimal

    def __post_init__(self) -> None:
        _require_durable_text("account_key", self.account_key)
        _require_durable_text("collateral_asset", self.collateral_asset)
        require_positive_decimal("margin_quantum", self.margin_quantum)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountBalance:
    """Exact available and reserved amounts for one asset."""

    asset: str
    available: Decimal
    reserved: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        _require_durable_text("asset", self.asset)
        _require_finite_decimal("available", self.available)
        _require_finite_decimal("reserved", self.reserved)
        if self.reserved < 0:
            raise ValueError("reserved must be non-negative")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperMarginReservation:
    """The exact current collateral reservation for one position."""

    position_key: str
    amount: Decimal

    def __post_init__(self) -> None:
        _require_durable_text("position_key", self.position_key)
        require_positive_decimal("amount", self.amount)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountPosting:
    """One exact signed movement in one account bucket."""

    asset: str
    bucket: PaperAccountPostingBucket
    amount: Decimal

    def __post_init__(self) -> None:
        _require_durable_text("asset", self.asset)
        if type(self.bucket) is not PaperAccountPostingBucket:
            raise TypeError("bucket must be a PaperAccountPostingBucket")
        _require_finite_decimal("amount", self.amount)
        if not self.amount:
            raise ValueError("zero postings must be omitted")


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountSettlementRecord:
    """One account-global version coupled to an exact paper settlement."""

    account_version: int
    settlement: PaperSettlement

    def __post_init__(self) -> None:
        if isinstance(self.account_version, bool) or not isinstance(
            self.account_version, int
        ):
            raise TypeError("account_version must be an integer")
        if self.account_version < 1:
            raise ValueError("account_version must be positive")
        if self.account_version > _BIGINT_MAX:
            raise ValueError("account_version exceeds its durable storage limit")
        if type(self.settlement) is not PaperSettlement:
            raise TypeError("settlement must be a PaperSettlement")
        if self.settlement.disposition is not PaperSettlementDisposition.APPLIED:
            raise ValueError("account records require newly applied settlements")


@dataclass(frozen=True, slots=True)
class _DerivedAccount:
    balances: tuple[PaperAccountBalance, ...]
    reservations: tuple[PaperMarginReservation, ...]
    state: PaperAccountState


def _canonical_opening_balances(
    balances: object,
) -> tuple[PaperAccountBalance, ...]:
    if type(balances) is not tuple:
        raise TypeError("opening_balances must be an exact tuple")
    if any(type(balance) is not PaperAccountBalance for balance in balances):
        raise TypeError("opening_balances must contain PaperAccountBalance values")
    if not balances:
        raise ValueError("an account requires at least one opening balance")
    assets = tuple(balance.asset for balance in balances)
    if assets != tuple(sorted(assets)) or len(assets) != len(set(assets)):
        raise ValueError("opening_balances must be unique and sorted by asset")
    if any(balance.available < 0 or balance.reserved != 0 for balance in balances):
        raise ValueError("opening balances must be solvent and unreserved")
    return balances


def _canonical_records(
    records: object,
) -> tuple[PaperAccountSettlementRecord, ...]:
    if type(records) is not tuple:
        raise TypeError("records must be an exact tuple")
    if any(type(record) is not PaperAccountSettlementRecord for record in records):
        raise TypeError("records must contain PaperAccountSettlementRecord values")
    versions = tuple(record.account_version for record in records)
    if versions != tuple(range(1, len(records) + 1)):
        raise InvalidPaperAccountTransition(
            "account versions must be the exact contiguous prefix"
        )
    event_identities = tuple(
        record.settlement.record.event_identity for record in records
    )
    if len(event_identities) != len(set(event_identities)):
        raise InvalidPaperAccountTransition(
            "account settlement event identities must be unique"
        )
    fill_identities = tuple(
        record.settlement.record.fill_identity for record in records
    )
    if len(fill_identities) != len(set(fill_identities)):
        raise InvalidPaperAccountTransition(
            "account settlement fill identities must be unique"
        )
    return records


def _build_postings(
    settlement: PaperSettlement,
    collateral_asset: str,
    margin_delta: Decimal,
) -> tuple[PaperAccountPosting, ...]:
    totals: dict[tuple[str, PaperAccountPostingBucket], Decimal] = {}

    def add(asset: str, bucket: PaperAccountPostingBucket, amount: Decimal) -> None:
        if not amount:
            return
        key = (asset, bucket)
        totals[key] = exact_decimal_sum((totals.get(key, Decimal("0")), amount))

    for cash_delta in settlement.cash_deltas:
        add(cash_delta.asset, PaperAccountPostingBucket.AVAILABLE, cash_delta.amount)
    if margin_delta:
        add(
            collateral_asset,
            PaperAccountPostingBucket.AVAILABLE,
            margin_delta.copy_negate(),
        )
        add(
            collateral_asset,
            PaperAccountPostingBucket.RESERVED_MARGIN,
            margin_delta,
        )
    return tuple(
        PaperAccountPosting(asset, bucket, amount)
        for (asset, bucket), amount in sorted(
            totals.items(),
            key=lambda item: (item[0][0], item[0][1].value),
        )
        if amount
    )


def _apply_postings(
    balances: tuple[PaperAccountBalance, ...],
    postings: tuple[PaperAccountPosting, ...],
) -> tuple[PaperAccountBalance, ...]:
    available = {balance.asset: balance.available for balance in balances}
    reserved = {balance.asset: balance.reserved for balance in balances}
    for posting in postings:
        available.setdefault(posting.asset, Decimal("0"))
        reserved.setdefault(posting.asset, Decimal("0"))
        if posting.bucket is PaperAccountPostingBucket.AVAILABLE:
            available[posting.asset] = exact_decimal_sum(
                (available[posting.asset], posting.amount)
            )
        else:
            reserved[posting.asset] = exact_decimal_sum(
                (reserved[posting.asset], posting.amount)
            )
            if reserved[posting.asset] < 0:
                raise InvalidPaperAccountTransition(
                    "margin release exceeds the reserved collateral"
                )
    return tuple(
        PaperAccountBalance(asset, available[asset], reserved[asset])
        for asset in sorted(available)
    )


def _settlement_position_key(settlement: PaperSettlement) -> str:
    return settlement.record.position_fill.instruction.position_key


def _last_settlement_by_position(
    records: tuple[PaperAccountSettlementRecord, ...],
) -> dict[str, PaperSettlement]:
    result: dict[str, PaperSettlement] = {}
    for record in records:
        result[_settlement_position_key(record.settlement)] = record.settlement
    return result


def _transition_components(
    policy: PaperAccountPolicy,
    balances: tuple[PaperAccountBalance, ...],
    reservations: tuple[PaperMarginReservation, ...],
    prior_settlements: dict[str, PaperSettlement],
    settlement: PaperSettlement,
) -> tuple[
    tuple[PaperAccountPosting, ...],
    tuple[PaperAccountBalance, ...],
    tuple[PaperMarginReservation, ...],
]:
    if settlement.instrument.settlement_asset != policy.collateral_asset:
        raise InvalidPaperAccountTransition(
            "instrument settlement asset must match account collateral"
        )
    position_key = _settlement_position_key(settlement)
    previous = prior_settlements.get(position_key)
    if previous is None:
        if settlement.before is not None:
            raise InvalidPaperAccountTransition(
                "first account settlement for a position must start its chain"
            )
    elif settlement.before is None or (
        settlement.before.instrument != previous.after.instrument
        or _economics_identity(settlement.before.economics)
        != _economics_identity(previous.after.economics)
    ):
        raise InvalidPaperAccountTransition(
            "settlement does not continue the account position chain"
        )

    reservations_by_position = {
        reservation.position_key: reservation.amount for reservation in reservations
    }
    current_margin = reservations_by_position.get(position_key, Decimal("0"))
    economics = settlement.after.economics
    try:
        target_margin = _ceil_margin_target(
            economics.open_cost,
            economics.position.leverage,
            policy.margin_quantum,
        )
        margin_delta = exact_decimal_sum((target_margin, current_margin.copy_negate()))
        postings = _build_postings(
            settlement,
            policy.collateral_asset,
            margin_delta,
        )
        next_balances = _apply_postings(balances, postings)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InvalidPaperAccountTransition):
            raise
        raise InvalidPaperAccountTransition(
            "account posting arithmetic is not exact"
        ) from exc

    if target_margin:
        reservations_by_position[position_key] = target_margin
    else:
        reservations_by_position.pop(position_key, None)
    next_reservations = tuple(
        PaperMarginReservation(known_position, amount)
        for known_position, amount in sorted(reservations_by_position.items())
    )
    return postings, next_balances, next_reservations


def _derive_account(
    policy: PaperAccountPolicy,
    opening_balances: tuple[PaperAccountBalance, ...],
    records: tuple[PaperAccountSettlementRecord, ...],
) -> _DerivedAccount:
    balances = opening_balances
    reservations: tuple[PaperMarginReservation, ...] = ()
    prior_settlements: dict[str, PaperSettlement] = {}
    for record in records:
        settlement = record.settlement
        postings, next_balances, next_reservations = _transition_components(
            policy,
            balances,
            reservations,
            prior_settlements,
            settlement,
        )
        del postings
        effect = settlement.record.position_fill.instruction.effect
        if effect is PositionEffect.OPEN and any(
            balance.available < 0 for balance in next_balances
        ):
            raise InvalidPaperAccountTransition(
                "historical OPEN settlement exceeds available balance"
            )
        balances = next_balances
        reservations = next_reservations
        prior_settlements[_settlement_position_key(settlement)] = settlement
    state = (
        PaperAccountState.INSOLVENT
        if any(balance.available < 0 for balance in balances)
        else PaperAccountState.ACTIVE
    )
    return _DerivedAccount(balances, reservations, state)


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccount:
    """A validated account projection in one global causal order."""

    policy: PaperAccountPolicy
    opening_balances: tuple[PaperAccountBalance, ...]
    balances: tuple[PaperAccountBalance, ...]
    reservations: tuple[PaperMarginReservation, ...]
    records: tuple[PaperAccountSettlementRecord, ...]
    state: PaperAccountState

    def __post_init__(self) -> None:
        if type(self.policy) is not PaperAccountPolicy:
            raise TypeError("policy must be a PaperAccountPolicy")
        opening_balances = _canonical_opening_balances(self.opening_balances)
        if not any(
            balance.asset == self.policy.collateral_asset
            for balance in opening_balances
        ):
            raise ValueError("opening balances must include the collateral asset")
        if type(self.balances) is not tuple or any(
            type(balance) is not PaperAccountBalance for balance in self.balances
        ):
            raise TypeError("balances must be an exact tuple of PaperAccountBalance")
        if type(self.reservations) is not tuple or any(
            type(reservation) is not PaperMarginReservation
            for reservation in self.reservations
        ):
            raise TypeError(
                "reservations must be an exact tuple of PaperMarginReservation"
            )
        records = _canonical_records(self.records)
        if type(self.state) is not PaperAccountState:
            raise TypeError("state must be a PaperAccountState")

        expected = _derive_account(self.policy, opening_balances, records)
        if tuple(map(_balance_identity, self.balances)) != tuple(
            map(_balance_identity, expected.balances)
        ):
            raise ValueError(
                "balances must be derived from opening balances and records"
            )
        if tuple(map(_reservation_identity, self.reservations)) != tuple(
            map(_reservation_identity, expected.reservations)
        ):
            raise ValueError("reservations must be derived from account records")
        if self.state is not expected.state:
            raise ValueError("state must be derived from account balances")


@dataclass(frozen=True, slots=True)
class _DerivedAdmission:
    disposition: PaperAccountAdmissionDisposition
    after: PaperAccount
    postings: tuple[PaperAccountPosting, ...]
    reasons: tuple[str, ...]


def _derive_admission(
    before: PaperAccount,
    account_version: int,
    settlement: PaperSettlement,
) -> _DerivedAdmission:
    candidate = PaperAccountSettlementRecord(account_version, settlement)
    existing_at_version = next(
        (
            record
            for record in before.records
            if record.account_version == account_version
        ),
        None,
    )
    if existing_at_version is not None:
        if _record_identity(existing_at_version) == _record_identity(candidate):
            return _DerivedAdmission(
                PaperAccountAdmissionDisposition.REPLAYED,
                before,
                (),
                (),
            )
        raise InvalidPaperAccountTransition(
            "account version has conflicting settlement data"
        )

    candidate_event_identity = settlement.record.event_identity
    candidate_fill_identity = settlement.record.fill_identity
    if any(
        record.settlement.record.event_identity == candidate_event_identity
        or record.settlement.record.fill_identity == candidate_fill_identity
        for record in before.records
    ):
        raise InvalidPaperAccountTransition(
            "settlement identity is already recorded at another account version"
        )
    if account_version != len(before.records) + 1:
        raise InvalidPaperAccountTransition(
            "new settlement must use the next account version"
        )

    prior_settlements = _last_settlement_by_position(before.records)
    postings, next_balances, next_reservations = _transition_components(
        before.policy,
        before.balances,
        before.reservations,
        prior_settlements,
        settlement,
    )
    effect = settlement.record.position_fill.instruction.effect
    insufficient_assets = tuple(
        balance.asset for balance in next_balances if balance.available < 0
    )
    if effect is PositionEffect.OPEN and (
        before.state is PaperAccountState.INSOLVENT or insufficient_assets
    ):
        reasons = (
            ("account is insolvent",)
            if before.state is PaperAccountState.INSOLVENT
            else tuple(
                f"insufficient available balance for {asset}"
                for asset in insufficient_assets
            )
        )
        return _DerivedAdmission(
            PaperAccountAdmissionDisposition.REJECTED,
            before,
            (),
            reasons,
        )

    next_record = candidate
    state = (
        PaperAccountState.INSOLVENT
        if any(balance.available < 0 for balance in next_balances)
        else PaperAccountState.ACTIVE
    )
    after = PaperAccount(
        policy=before.policy,
        opening_balances=before.opening_balances,
        balances=next_balances,
        reservations=next_reservations,
        records=before.records + (next_record,),
        state=state,
    )
    return _DerivedAdmission(
        PaperAccountAdmissionDisposition.APPLIED,
        after,
        postings,
        (),
    )


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAccountAdmission:
    """A non-forgeable admission outcome for one candidate settlement."""

    before: PaperAccount
    account_version: int
    settlement: PaperSettlement
    disposition: PaperAccountAdmissionDisposition
    after: PaperAccount
    postings: tuple[PaperAccountPosting, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.before) is not PaperAccount:
            raise TypeError("before must be a PaperAccount")
        if isinstance(self.account_version, bool) or not isinstance(
            self.account_version, int
        ):
            raise TypeError("account_version must be an integer")
        if self.account_version < 1 or self.account_version > _BIGINT_MAX:
            raise ValueError("account_version is outside durable storage bounds")
        if type(self.settlement) is not PaperSettlement:
            raise TypeError("settlement must be a PaperSettlement")
        if type(self.disposition) is not PaperAccountAdmissionDisposition:
            raise TypeError("disposition must be a PaperAccountAdmissionDisposition")
        if type(self.after) is not PaperAccount:
            raise TypeError("after must be a PaperAccount")
        if type(self.postings) is not tuple or any(
            type(posting) is not PaperAccountPosting for posting in self.postings
        ):
            raise TypeError("postings must be an exact tuple of PaperAccountPosting")
        if type(self.reasons) is not tuple:
            raise TypeError("reasons must be an exact tuple")
        for reason in self.reasons:
            require_clean_text("reason", reason)

        expected = _derive_admission(
            self.before,
            self.account_version,
            self.settlement,
        )
        if self.disposition is not expected.disposition:
            raise ValueError("disposition is not derived from the candidate")
        if _account_identity(self.after) != _account_identity(expected.after):
            raise ValueError("after is not derived from the admission")
        if tuple(map(_posting_identity, self.postings)) != tuple(
            map(_posting_identity, expected.postings)
        ):
            raise ValueError("postings are not derived from the admission")
        if self.reasons != expected.reasons:
            raise ValueError("reasons are not derived from the admission")


def new_paper_account(
    policy: PaperAccountPolicy,
    opening_balances: tuple[PaperAccountBalance, ...],
) -> PaperAccount:
    """Create a solvent account with no reservations or settlement records."""
    if type(policy) is not PaperAccountPolicy:
        raise TypeError("policy must be a PaperAccountPolicy")
    opening_balances = _canonical_opening_balances(opening_balances)
    if not any(
        balance.asset == policy.collateral_asset for balance in opening_balances
    ):
        raise ValueError("opening balances must include the collateral asset")
    return PaperAccount(
        policy=policy,
        opening_balances=opening_balances,
        balances=opening_balances,
        reservations=(),
        records=(),
        state=PaperAccountState.ACTIVE,
    )


def admit_paper_settlement(
    account: PaperAccount,
    account_version: int,
    settlement: PaperSettlement,
) -> PaperAccountAdmission:
    """Admit, replay, or reject one exact settlement without performing I/O."""
    if type(account) is not PaperAccount:
        raise TypeError("account must be a PaperAccount")
    if isinstance(account_version, bool) or not isinstance(account_version, int):
        raise TypeError("account_version must be an integer")
    if account_version < 1 or account_version > _BIGINT_MAX:
        raise ValueError("account_version is outside durable storage bounds")
    if type(settlement) is not PaperSettlement:
        raise TypeError("settlement must be a PaperSettlement")

    derived = _derive_admission(account, account_version, settlement)
    return PaperAccountAdmission(
        before=account,
        account_version=account_version,
        settlement=settlement,
        disposition=derived.disposition,
        after=derived.after,
        postings=derived.postings,
        reasons=derived.reasons,
    )


__all__ = [
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
]
