"""Pure quote-settled cash deltas derived from confirmed paper fills."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from trading.domain._decimal import exact_decimal_sum
from trading.domain._validation import (
    protect_frozen_dataclass_state,
    require_clean_text,
)
from trading.domain.paper_economics import (
    InvalidPaperEconomicTransition,
    PaperEconomics,
    PaperFillRecord,
    _decimal_payload_identity,
    _fee_payload_identity,
    _lot_payload_identity,
    _position_payload_identity,
    _record_payload_identity,
    new_paper_economics,
    reduce_paper_economics,
)


def _require_durable_text(name: str, value: object) -> None:
    require_clean_text(name, value)
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{name} is not representable in durable storage")


def _amount_identity(amount: "PaperAssetAmount") -> tuple[object, ...]:
    return (amount.asset, _decimal_payload_identity(amount.amount))


def _economics_identity(economics: PaperEconomics) -> tuple[object, ...]:
    return (
        economics.lot_method,
        tuple(_record_payload_identity(record) for record in economics.records),
        _position_payload_identity(economics.position),
        tuple(_lot_payload_identity(lot) for lot in economics.lots),
        _decimal_payload_identity(economics.open_quantity),
        _decimal_payload_identity(economics.open_cost),
        _decimal_payload_identity(economics.gross_realized_pnl),
        tuple(_fee_payload_identity(fee) for fee in economics.fees),
    )


def _protect_factory_checkpoint(contract_type: type) -> type:
    def _reject_state_mutation(_instance: object, _state: object) -> None:
        raise TypeError(
            "factory-only PaperSettlementCheckpoint does not support state mutation"
        )

    contract_type.__setstate__ = _reject_state_mutation
    return contract_type


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperLinearInstrument:
    """A multiplier-one instrument priced and settled in its quote asset."""

    symbol: str
    base_asset: str
    quote_asset: str

    def __post_init__(self) -> None:
        _require_durable_text("symbol", self.symbol)
        _require_durable_text("base_asset", self.base_asset)
        _require_durable_text("quote_asset", self.quote_asset)
        if self.base_asset == self.quote_asset:
            raise ValueError("base_asset and quote_asset must differ")

    @property
    def settlement_asset(self) -> str:
        """Return the explicit settlement denomination for linear PnL."""
        return self.quote_asset


@_protect_factory_checkpoint
@dataclass(frozen=True, slots=True, init=False)
class PaperSettlementCheckpoint:
    """Compact instrument-bound economics used between settlement steps."""

    instrument: PaperLinearInstrument
    economics: PaperEconomics

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "PaperSettlementCheckpoint values are returned by settle_paper_fill"
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (_new_checkpoint, (self.instrument, self.economics))

    def __post_init__(self) -> None:
        if type(self.instrument) is not PaperLinearInstrument:
            raise TypeError("instrument must be a PaperLinearInstrument")
        if type(self.economics) is not PaperEconomics:
            raise TypeError("economics must be PaperEconomics")
        if self.economics.position.symbol != self.instrument.symbol:
            raise ValueError("economics symbol must match the instrument")


def _new_checkpoint(
    instrument: PaperLinearInstrument,
    economics: PaperEconomics,
) -> PaperSettlementCheckpoint:
    checkpoint = object.__new__(PaperSettlementCheckpoint)
    object.__setattr__(checkpoint, "instrument", instrument)
    object.__setattr__(checkpoint, "economics", economics)
    checkpoint.__post_init__()
    return checkpoint


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperAssetAmount:
    """One exact signed amount denominated in a single explicit asset."""

    asset: str
    amount: Decimal

    def __post_init__(self) -> None:
        _require_durable_text("asset", self.asset)
        if not isinstance(self.amount, Decimal):
            raise TypeError("amount must be a Decimal")
        if not self.amount.is_finite():
            raise ValueError("amount must be finite")


class PaperSettlementDisposition(str, Enum):
    """Whether a fill produced new deltas or replayed an exact prior fact."""

    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"


class InvalidPaperSettlement(ValueError):
    """Raised when a fill cannot produce one coherent linear settlement."""


@dataclass(frozen=True, slots=True)
class _DerivedSettlement:
    after: PaperSettlementCheckpoint
    disposition: PaperSettlementDisposition
    gross_realized_pnl_delta: PaperAssetAmount
    fee_debits: tuple[PaperAssetAmount, ...]
    cash_deltas: tuple[PaperAssetAmount, ...]


def _derive_settlement(
    instrument: PaperLinearInstrument,
    before: PaperSettlementCheckpoint | None,
    record: PaperFillRecord,
) -> _DerivedSettlement:
    fill = record.position_fill.fill
    if fill.symbol != instrument.symbol:
        raise InvalidPaperSettlement("fill symbol does not match the instrument")
    if before is not None and before.instrument != instrument:
        raise InvalidPaperSettlement(
            "instrument does not match the prior settlement chain"
        )

    prior_economics = None if before is None else before.economics

    try:
        after = (
            new_paper_economics(record)
            if prior_economics is None
            else reduce_paper_economics(prior_economics, record)
        )
    except (InvalidPaperEconomicTransition, TypeError, ValueError) as exc:
        raise InvalidPaperSettlement(
            "fill contradicts the prior paper economics"
        ) from exc

    if prior_economics is not None and after is prior_economics:
        return _DerivedSettlement(
            after=before,
            disposition=PaperSettlementDisposition.REPLAYED,
            gross_realized_pnl_delta=PaperAssetAmount(
                instrument.settlement_asset,
                Decimal("0"),
            ),
            fee_debits=(),
            cash_deltas=(),
        )

    prior_gross = (
        Decimal("0") if prior_economics is None else prior_economics.gross_realized_pnl
    )
    try:
        realized_delta = exact_decimal_sum(
            (after.gross_realized_pnl, prior_gross.copy_negate())
        )
        fee_debits = (
            (PaperAssetAmount(fill.fee_asset, fill.fee_amount),)
            if fill.fee_amount
            else ()
        )
        cash_by_asset: dict[str, Decimal] = {}
        if realized_delta:
            cash_by_asset[instrument.settlement_asset] = realized_delta
        for debit in fee_debits:
            cash_by_asset[debit.asset] = exact_decimal_sum(
                (
                    cash_by_asset.get(debit.asset, Decimal("0")),
                    debit.amount.copy_negate(),
                )
            )
        cash_deltas = tuple(
            PaperAssetAmount(asset, amount)
            for asset, amount in sorted(cash_by_asset.items())
            if amount
        )
    except (TypeError, ValueError) as exc:
        raise InvalidPaperSettlement("settlement arithmetic is not exact") from exc

    return _DerivedSettlement(
        after=_new_checkpoint(instrument, after),
        disposition=PaperSettlementDisposition.APPLIED,
        gross_realized_pnl_delta=PaperAssetAmount(
            instrument.settlement_asset,
            realized_delta,
        ),
        fee_debits=fee_debits,
        cash_deltas=cash_deltas,
    )


@protect_frozen_dataclass_state
@dataclass(frozen=True, slots=True)
class PaperSettlement:
    """One non-forgeable economic transition and its per-asset cash deltas."""

    instrument: PaperLinearInstrument
    before: PaperSettlementCheckpoint | None
    record: PaperFillRecord
    after: PaperSettlementCheckpoint
    disposition: PaperSettlementDisposition
    gross_realized_pnl_delta: PaperAssetAmount
    fee_debits: tuple[PaperAssetAmount, ...]
    cash_deltas: tuple[PaperAssetAmount, ...]

    def __post_init__(self) -> None:
        if type(self.instrument) is not PaperLinearInstrument:
            raise TypeError("instrument must be a PaperLinearInstrument")
        if (
            self.before is not None
            and type(self.before) is not PaperSettlementCheckpoint
        ):
            raise TypeError("before must be PaperSettlementCheckpoint or None")
        if type(self.record) is not PaperFillRecord:
            raise TypeError("record must be a PaperFillRecord")
        if type(self.after) is not PaperSettlementCheckpoint:
            raise TypeError("after must be a PaperSettlementCheckpoint")
        if type(self.disposition) is not PaperSettlementDisposition:
            raise TypeError("disposition must be a PaperSettlementDisposition")
        if type(self.gross_realized_pnl_delta) is not PaperAssetAmount:
            raise TypeError("gross_realized_pnl_delta must be a PaperAssetAmount")
        if not isinstance(self.fee_debits, tuple) or any(
            type(amount) is not PaperAssetAmount for amount in self.fee_debits
        ):
            raise TypeError("fee_debits must be a tuple of PaperAssetAmount values")
        if not isinstance(self.cash_deltas, tuple) or any(
            type(amount) is not PaperAssetAmount for amount in self.cash_deltas
        ):
            raise TypeError("cash_deltas must be a tuple of PaperAssetAmount values")

        expected = _derive_settlement(self.instrument, self.before, self.record)
        if self.after.instrument != expected.after.instrument or _economics_identity(
            self.after.economics
        ) != _economics_identity(expected.after.economics):
            raise ValueError("after is not derived from before and record")
        if self.disposition is not expected.disposition:
            raise ValueError("disposition is not derived from the transition")
        if _amount_identity(self.gross_realized_pnl_delta) != _amount_identity(
            expected.gross_realized_pnl_delta
        ):
            raise ValueError("gross_realized_pnl_delta is not derived")
        if tuple(map(_amount_identity, self.fee_debits)) != tuple(
            map(_amount_identity, expected.fee_debits)
        ):
            raise ValueError("fee_debits are not derived")
        if tuple(map(_amount_identity, self.cash_deltas)) != tuple(
            map(_amount_identity, expected.cash_deltas)
        ):
            raise ValueError("cash_deltas are not derived")


def settle_paper_fill(
    instrument: PaperLinearInstrument,
    before: PaperSettlementCheckpoint | None,
    record: PaperFillRecord,
) -> PaperSettlement:
    """Apply one causal fill and return exact, explicitly denominated deltas."""
    if type(instrument) is not PaperLinearInstrument:
        raise TypeError("instrument must be a PaperLinearInstrument")
    if before is not None and type(before) is not PaperSettlementCheckpoint:
        raise TypeError("before must be PaperSettlementCheckpoint or None")
    if type(record) is not PaperFillRecord:
        raise TypeError("record must be a PaperFillRecord")

    derived = _derive_settlement(instrument, before, record)
    return PaperSettlement(
        instrument=instrument,
        before=before,
        record=record,
        after=derived.after,
        disposition=derived.disposition,
        gross_realized_pnl_delta=derived.gross_realized_pnl_delta,
        fee_debits=derived.fee_debits,
        cash_deltas=derived.cash_deltas,
    )


__all__ = [
    "InvalidPaperSettlement",
    "PaperAssetAmount",
    "PaperLinearInstrument",
    "PaperSettlement",
    "PaperSettlementCheckpoint",
    "PaperSettlementDisposition",
    "settle_paper_fill",
]
