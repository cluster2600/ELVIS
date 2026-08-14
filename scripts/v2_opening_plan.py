"""Read-only CLI for preparing one signed ELVIS V2 fresh opening.

The command validates local, secret-free JSON documents only.  It has no
database, network, signing, provisioning, or runtime-authority capability.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NoReturn, TextIO

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from trading.application.fresh_opening import (
    DetachedFreshOpeningApproval,
    FreshOpeningIntent,
    FreshOpeningPolicy,
    FreshOpeningPreparation,
    FreshOpeningPreparationDisposition,
    FreshOpeningTrustAnchor,
    FreshOpeningTrustPolicy,
    prepare_fresh_opening,
)
from trading.domain.paper_accounting import (
    PaperAccountBalance,
    PaperAccountPolicy,
    new_paper_account,
)
from trading.persistence.paper_account_journal_codec import (
    EncodedPaperAccountOpening,
    encode_paper_account_opening,
)

_MAX_FILE_BYTES = 65_536

_LOWER_HEX = frozenset("0123456789abcdef")
_INTENT_KEYS = {
    "schema_version",
    "purpose",
    "trajectory",
    "continuity",
    "logical_target",
    "execution_scope",
    "account_key",
    "owner_generation",
    "opening_codec",
    "opening_version",
    "collateral_asset",
    "collateral_amount",
    "margin_quantum",
    "opening_policy",
    "operator_identity",
    "approval_id",
    "approver_identity",
    "approval_issued_at",
    "approval_expires_at",
    "trust_policy_sha256",
    "trust_domain",
    "signer_key_id",
    "signer_public_key_sha256",
    "nonce",
}
_APPROVAL_KEYS = {"schema_version", "intent_sha256", "signature"}
_TRUST_POLICY_KEYS = {
    "schema_version",
    "purpose",
    "trust_domain",
    "max_approval_lifetime_seconds",
    "anchors",
}
_TRUST_ANCHOR_KEYS = {
    "signer_key_id",
    "approver_identity",
    "ed25519_public_key",
    "revoked",
}

_EXIT_PREPARED = 0
_EXIT_INPUT = 2
_EXIT_BLOCKED = 10
_EXIT_INTERNAL = 70


class _CanonicalOpeningCodec:
    """Compose the pure planner with the existing opening-codec boundary."""

    def encode(
        self,
        *,
        execution_scope: str,
        account_key: str,
        owner_generation: int,
        collateral_asset: str,
        collateral_amount: Decimal,
        margin_quantum: Decimal,
    ) -> EncodedPaperAccountOpening:
        policy = PaperAccountPolicy(
            account_key=account_key,
            collateral_asset=collateral_asset,
            margin_quantum=margin_quantum,
        )
        account = new_paper_account(
            policy,
            (
                PaperAccountBalance(
                    asset=collateral_asset,
                    available=collateral_amount,
                    reserved=Decimal("0"),
                ),
            ),
        )
        return encode_paper_account_opening(
            execution_scope,
            owner_generation,
            account,
        )


class _CryptographyEd25519Verifier:
    """Adapt cryptography verification without exposing any signing operation."""

    def verify(
        self,
        *,
        public_key: bytes,
        signature: bytes,
        message: bytes,
    ) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        except InvalidSignature, ValueError:
            return False
        return True


_OPENING_CODEC = _CanonicalOpeningCodec()
_SIGNATURE_VERIFIER = _CryptographyEd25519Verifier()


class _CliInputError(ValueError):
    """Secret-free signal for an invalid invocation or local input file."""


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliInputError("invalid invocation")


def _reject_json_constant(value: str) -> NoReturn:
    del value
    raise _CliInputError("invalid JSON")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _CliInputError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    """Read one bounded regular file through the descriptor opened safely."""

    descriptor: int | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise _CliInputError("safe input-file access is unavailable")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | nofollow
        )
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _CliInputError("invalid input file")
        with os.fdopen(descriptor, "rb", closefd=True) as input_file:
            descriptor = None
            payload = input_file.read(_MAX_FILE_BYTES + 1)
        if len(payload) > _MAX_FILE_BYTES:
            raise _CliInputError("input file is too large")
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except _CliInputError:
        raise
    except OSError, UnicodeError, ValueError, RecursionError:
        raise _CliInputError("invalid input file") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if type(document) is not dict:
        raise _CliInputError("input file must contain an object")
    return document


def _write_json(stream: TextIO, value: dict[str, object]) -> None:
    stream.write(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )
    stream.write("\n")


def _exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise _CliInputError("invalid object shape")


def _lower_hex(value: object, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise _CliInputError("invalid hexadecimal value")
    return value


def _sha256(value: object) -> str:
    return _lower_hex(value, 64)


def _decimal(value: object) -> Decimal:
    if type(value) is not str:
        raise _CliInputError("invalid decimal")
    integer, separator, fraction = value.partition(".")
    if (
        not integer
        or (integer != "0" and (integer.startswith("0") or not integer.isdigit()))
        or (integer == "0" and not integer.isdigit())
        or (separator and (not fraction or not fraction.isdigit()))
        or (not separator and fraction)
        or not value.isascii()
    ):
        raise _CliInputError("invalid decimal")
    try:
        decoded = Decimal(value)
    except InvalidOperation:
        raise _CliInputError("invalid decimal") from None
    if not decoded.is_finite() or str(decoded) != value:
        raise _CliInputError("invalid decimal")
    return decoded


def _utc_datetime(value: object) -> datetime:
    if type(value) is not str:
        raise _CliInputError("invalid datetime")
    try:
        decoded = datetime.fromisoformat(value)
        if decoded.tzinfo is None or decoded.utcoffset() is None:
            raise ValueError
        normalized = decoded.astimezone(timezone.utc)
    except OverflowError, TypeError, ValueError:
        raise _CliInputError("invalid datetime") from None
    if value != normalized.isoformat(timespec="microseconds"):
        raise _CliInputError("invalid datetime")
    return normalized


def _parse_intent(document: dict[str, Any]) -> FreshOpeningIntent:
    _exact_keys(document, _INTENT_KEYS)
    try:
        opening_policy = FreshOpeningPolicy(document["opening_policy"])
    except TypeError, ValueError:
        raise _CliInputError("invalid opening policy") from None
    return FreshOpeningIntent(
        schema_version=document["schema_version"],
        purpose=document["purpose"],
        trajectory=document["trajectory"],
        continuity=document["continuity"],
        logical_target=document["logical_target"],
        execution_scope=document["execution_scope"],
        account_key=document["account_key"],
        owner_generation=document["owner_generation"],
        opening_codec=document["opening_codec"],
        opening_version=document["opening_version"],
        collateral_asset=document["collateral_asset"],
        collateral_amount=_decimal(document["collateral_amount"]),
        margin_quantum=_decimal(document["margin_quantum"]),
        opening_policy=opening_policy,
        operator_identity=document["operator_identity"],
        approval_id=document["approval_id"],
        approver_identity=document["approver_identity"],
        approval_issued_at=_utc_datetime(document["approval_issued_at"]),
        approval_expires_at=_utc_datetime(document["approval_expires_at"]),
        trust_policy_sha256=document["trust_policy_sha256"],
        trust_domain=document["trust_domain"],
        signer_key_id=document["signer_key_id"],
        signer_public_key_sha256=document["signer_public_key_sha256"],
        nonce=document["nonce"],
    )


def _parse_approval(document: dict[str, Any]) -> DetachedFreshOpeningApproval:
    _exact_keys(document, _APPROVAL_KEYS)
    return DetachedFreshOpeningApproval(
        schema_version=document["schema_version"],
        intent_sha256=document["intent_sha256"],
        signature=bytes.fromhex(_lower_hex(document["signature"], 128)),
    )


def _parse_trust_anchor(value: object) -> FreshOpeningTrustAnchor:
    if type(value) is not dict:
        raise _CliInputError("invalid trust anchor")
    _exact_keys(value, _TRUST_ANCHOR_KEYS)
    return FreshOpeningTrustAnchor(
        signer_key_id=value["signer_key_id"],
        approver_identity=value["approver_identity"],
        ed25519_public_key=bytes.fromhex(_lower_hex(value["ed25519_public_key"], 64)),
        revoked=value["revoked"],
    )


def _parse_trust_policy(document: dict[str, Any]) -> FreshOpeningTrustPolicy:
    _exact_keys(document, _TRUST_POLICY_KEYS)
    anchors = document["anchors"]
    if type(anchors) is not list:
        raise _CliInputError("invalid trust anchors")
    return FreshOpeningTrustPolicy(
        schema_version=document["schema_version"],
        purpose=document["purpose"],
        trust_domain=document["trust_domain"],
        max_approval_lifetime_seconds=document["max_approval_lifetime_seconds"],
        anchors=tuple(_parse_trust_anchor(value) for value in anchors),
    )


def _safe_result(
    result: str,
    reason_code: str,
    *,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "result": result,
        "primary_reason_code": reason_code,
        "side_effect_state": "NONE",
        "account_opening_authorized": False,
        "account_provisioning_authorized": False,
        "runtime_activation_authorized": False,
        "trading_authorized": False,
        "database_contact": False,
        "nonce_registry_checked": False,
        "target_local_replay_authority": "UNAVAILABLE_UNTIL_PR3",
    }
    if details is not None:
        value.update(details)
    return value


def _preparation_result(preparation: FreshOpeningPreparation) -> dict[str, object]:
    prepared = preparation.disposition is FreshOpeningPreparationDisposition.PREPARED
    details: dict[str, object] = {
        "intent_sha256": preparation.intent_document.sha256,
        "opening_payload_sha256": (
            preparation.prospective_opening.opening_payload_sha256
        ),
        "opening_version": preparation.prospective_opening.opening_version,
        "physical_target_bound": preparation.physical_target_bound,
        "pin_source_authenticated": preparation.pin_source_authenticated,
        "stale_on_return": preparation.stale_on_return,
    }
    if preparation.candidate is not None:
        details.update(
            {
                "approval_sha256": (preparation.candidate.approval_document.sha256),
                "candidate_sha256": (preparation.candidate.candidate_document.sha256),
                "trust_policy_sha256": (
                    preparation.candidate.trust_policy_document.sha256
                ),
            }
        )
    return _safe_result(
        "PREPARED" if prepared else "BLOCKED",
        preparation.disposition.value,
        details=details,
    )


def _argument_parser() -> _StrictArgumentParser:
    parser = _StrictArgumentParser(
        prog="python3.14 -m scripts.v2_opening_plan",
        allow_abbrev=False,
        description=(
            "Validate a signed fresh-opening plan without contacting a database "
            "or granting runtime authority."
        ),
    )
    parser.add_argument("--intent", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--trust-policy")
    parser.add_argument("--pinned-trust-policy-sha256")
    parser.add_argument("--pinned-signer-public-key-sha256")
    return parser


def _run(argv: Sequence[str] | None, evaluated_at: datetime) -> dict[str, object]:
    """Parse and prepare one local plan without any external side effect."""

    arguments = _argument_parser().parse_args(argv)
    intent = _parse_intent(_read_json(Path(arguments.intent)))
    approval = (
        None
        if arguments.approval is None
        else _parse_approval(_read_json(Path(arguments.approval)))
    )
    trust_policy = (
        None
        if arguments.trust_policy is None
        else _parse_trust_policy(_read_json(Path(arguments.trust_policy)))
    )
    expected_policy_sha256 = (
        None
        if arguments.pinned_trust_policy_sha256 is None
        else _sha256(arguments.pinned_trust_policy_sha256)
    )
    expected_key_sha256 = (
        None
        if arguments.pinned_signer_public_key_sha256 is None
        else _sha256(arguments.pinned_signer_public_key_sha256)
    )
    preparation = prepare_fresh_opening(
        intent,
        approval,
        opening_codec=_OPENING_CODEC,
        signature_verifier=_SIGNATURE_VERIFIER,
        trust_policy=trust_policy,
        expected_trust_policy_sha256=expected_policy_sha256,
        expected_signer_public_key_sha256=expected_key_sha256,
        evaluated_at=evaluated_at,
    )
    result = _preparation_result(preparation)
    if (
        preparation.disposition is FreshOpeningPreparationDisposition.PREPARED
        and expected_key_sha256 is not None
    ):
        result["signer_public_key_sha256"] = expected_key_sha256
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluated_at: datetime | None = None,
) -> int:
    """Emit exactly one compact, secret-free JSON result for an invocation."""

    try:
        result = _run(argv, evaluated_at or datetime.now(timezone.utc))
    except _CliInputError, TypeError, ValueError:
        _write_json(sys.stdout, _safe_result("INVALID_INPUT", "INVALID_INPUT"))
        return _EXIT_INPUT
    except Exception:
        _write_json(sys.stdout, _safe_result("INTERNAL_ERROR", "INTERNAL_ERROR"))
        return _EXIT_INTERNAL
    _write_json(sys.stdout, result)
    return {
        "PREPARED": _EXIT_PREPARED,
        "BLOCKED": _EXIT_BLOCKED,
    }[str(result["result"])]


if __name__ == "__main__":
    raise SystemExit(main())
