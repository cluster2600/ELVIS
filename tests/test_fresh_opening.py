"""Adversarial contracts for offline fresh-opening preparation."""

import ast
import hashlib
import json
import pickle
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Inexact, Rounded, localcontext
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from trading.application.fresh_opening import (
    CanonicalFreshOpeningDocument,
    DetachedFreshOpeningApproval,
    FreshOpeningIntent,
    FreshOpeningPolicy,
    FreshOpeningPreparation,
    FreshOpeningPreparationDisposition,
    FreshOpeningTrustAnchor,
    FreshOpeningTrustPolicy,
    ProspectiveFreshOpeningCandidate,
    encode_detached_fresh_opening_approval,
    encode_fresh_opening_intent,
    encode_fresh_opening_trust_policy,
    fresh_opening_signing_bytes,
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

UTC = timezone.utc
ISSUED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
EXPIRES_AT = datetime(2026, 8, 14, 13, 0, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)
SIGNING_PREFIX = b"ELVIS\x00fresh-opening-intent\x00v1\x00"

PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
PUBLIC_KEY = PRIVATE_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
PUBLIC_KEY_SHA256 = hashlib.sha256(PUBLIC_KEY).hexdigest()
SECOND_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
SECOND_PUBLIC_KEY = SECOND_PRIVATE_KEY.public_key().public_bytes(
    Encoding.Raw,
    PublicFormat.Raw,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


class _CanonicalOpeningCodec:
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
        return encode_paper_account_opening(
            execution_scope,
            owner_generation,
            new_paper_account(
                PaperAccountPolicy(account_key, collateral_asset, margin_quantum),
                (
                    PaperAccountBalance(
                        collateral_asset,
                        collateral_amount,
                        Decimal("0"),
                    ),
                ),
            ),
        )


class _CryptographyEd25519Verifier:
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


OPENING_CODEC = _CanonicalOpeningCodec()
SIGNATURE_VERIFIER = _CryptographyEd25519Verifier()


def _anchor(**overrides: object) -> FreshOpeningTrustAnchor:
    values: dict[str, object] = {
        "signer_key_id": "opening-approver-2026-01",
        "approver_identity": "operator-reviewer-02",
        "ed25519_public_key": PUBLIC_KEY,
        "revoked": False,
    }
    values.update(overrides)
    return FreshOpeningTrustAnchor(**values)


def _policy(
    *anchors: FreshOpeningTrustAnchor, **overrides: object
) -> FreshOpeningTrustPolicy:
    values: dict[str, object] = {
        "schema_version": 1,
        "purpose": "ELVIS_V2_FRESH_PAPER_OPENING",
        "trust_domain": "elvis-paper-production",
        "max_approval_lifetime_seconds": 3600,
        "anchors": anchors or (_anchor(),),
    }
    values.update(overrides)
    return FreshOpeningTrustPolicy(**values)


def _intent(
    policy: FreshOpeningTrustPolicy,
    *,
    anchor: FreshOpeningTrustAnchor | None = None,
    **overrides: object,
) -> FreshOpeningIntent:
    selected = anchor or policy.anchors[0]
    policy_document = encode_fresh_opening_trust_policy(policy)
    values: dict[str, object] = {
        "schema_version": 1,
        "purpose": "ELVIS_V2_FRESH_PAPER_OPENING",
        "trajectory": "B",
        "continuity": "NO_V1_CONTINUITY",
        "logical_target": "paper-production-primary",
        "execution_scope": "paper:production",
        "account_key": "paper-v2-main",
        "owner_generation": 1,
        "opening_codec": "paper-account-opening",
        "opening_version": 1,
        "collateral_asset": "USDT",
        "collateral_amount": Decimal("1000.00"),
        "margin_quantum": Decimal("0.01"),
        "opening_policy": FreshOpeningPolicy.EXPLICIT_FRESH_SINGLE_COLLATERAL,
        "operator_identity": "operator-requester-01",
        "approval_id": "opening-approval-2026-08-14-001",
        "approver_identity": selected.approver_identity,
        "approval_issued_at": ISSUED_AT,
        "approval_expires_at": EXPIRES_AT,
        "trust_policy_sha256": policy_document.sha256,
        "trust_domain": policy.trust_domain,
        "signer_key_id": selected.signer_key_id,
        "signer_public_key_sha256": selected.public_key_sha256,
        "nonce": "0123456789abcdef" * 4,
    }
    values.update(overrides)
    return FreshOpeningIntent(**values)


def _approval(
    intent: FreshOpeningIntent,
    *,
    private_key: Ed25519PrivateKey = PRIVATE_KEY,
    signed_bytes: bytes | None = None,
    intent_sha256: str | None = None,
) -> DetachedFreshOpeningApproval:
    document = encode_fresh_opening_intent(intent)
    message = (
        fresh_opening_signing_bytes(document) if signed_bytes is None else signed_bytes
    )
    return DetachedFreshOpeningApproval(
        schema_version=1,
        intent_sha256=document.sha256 if intent_sha256 is None else intent_sha256,
        signature=private_key.sign(message),
    )


def _prepare(
    *,
    policy: FreshOpeningTrustPolicy | None = None,
    intent: FreshOpeningIntent | None = None,
    approval: DetachedFreshOpeningApproval | None | object = Ellipsis,
    expected_policy_sha256: str | None | object = Ellipsis,
    expected_key_sha256: str | None | object = Ellipsis,
    evaluated_at: datetime = EVALUATED_AT,
) -> FreshOpeningPreparation:
    actual_policy = policy or _policy()
    actual_intent = intent or _intent(actual_policy)
    actual_approval = _approval(actual_intent) if approval is Ellipsis else approval
    policy_document = encode_fresh_opening_trust_policy(actual_policy)
    actual_expected_policy = (
        policy_document.sha256
        if expected_policy_sha256 is Ellipsis
        else expected_policy_sha256
    )
    actual_expected_key = (
        actual_policy.anchors[0].public_key_sha256
        if expected_key_sha256 is Ellipsis
        else expected_key_sha256
    )
    return prepare_fresh_opening(
        actual_intent,
        actual_approval,
        opening_codec=OPENING_CODEC,
        signature_verifier=SIGNATURE_VERIFIER,
        trust_policy=actual_policy if policy is not None or actual_policy else None,
        expected_trust_policy_sha256=actual_expected_policy,
        expected_signer_public_key_sha256=actual_expected_key,
        evaluated_at=evaluated_at,
    )


def test_prepared_candidate_uses_exact_real_opening_codec_and_grants_no_authority() -> (
    None
):
    policy = _policy()
    intent = _intent(policy)
    result = _prepare(policy=policy, intent=intent)

    assert result.disposition is FreshOpeningPreparationDisposition.PREPARED
    assert type(result.candidate) is ProspectiveFreshOpeningCandidate
    assert result.candidate.opening == result.prospective_opening
    assert result.prospective_opening == encode_paper_account_opening(
        intent.execution_scope,
        intent.owner_generation,
        new_paper_account(
            PaperAccountPolicy(
                intent.account_key,
                intent.collateral_asset,
                intent.margin_quantum,
            ),
            (
                PaperAccountBalance(
                    intent.collateral_asset,
                    intent.collateral_amount,
                    Decimal("0"),
                ),
            ),
        ),
    )
    opening_payload = json.loads(result.prospective_opening.opening_payload)
    assert opening_payload["opening_balances"] == [
        {"asset": "USDT", "available": "1000.00", "reserved": "0"}
    ]
    assert result.nonce_replay_authority_available is False
    assert result.physical_target_bound is False
    assert result.opening_authorized is False
    assert result.provisioning_authorized is False
    assert result.runtime_authorized is False
    assert result.trading_authorized is False
    assert result.pin_source_authenticated is False
    assert result.stale_on_return is True


def test_signing_domain_and_intent_digest_are_exact() -> None:
    policy = _policy()
    document = encode_fresh_opening_intent(_intent(policy))

    expected = SIGNING_PREFIX + document.payload.encode("utf-8")

    assert fresh_opening_signing_bytes(document) == expected
    assert document.sha256 == hashlib.sha256(expected).hexdigest()


def test_policy_intent_approval_and_candidate_are_canonical_json() -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent)
    result = _prepare(policy=policy, intent=intent, approval=approval)
    documents = (
        encode_fresh_opening_trust_policy(policy),
        encode_fresh_opening_intent(intent),
        encode_detached_fresh_opening_approval(approval),
        result.candidate.candidate_document,
    )

    for document in documents:
        decoded = json.loads(document.payload)
        assert document.payload == json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        assert document.payload.isascii()

    assert set(json.loads(documents[2].payload)) == {
        "schema_version",
        "intent_sha256",
        "signature",
    }
    candidate_payload = json.loads(documents[3].payload)
    assert set(candidate_payload) == {
        "approval_sha256",
        "intent_sha256",
        "opening_codec",
        "opening_payload_sha256",
        "opening_version",
        "schema_version",
        "trust_policy_sha256",
    }
    forbidden = {
        "system_identifier",
        "runtime_candidate_sha256",
        "activation_id",
        "v1",
        "legacy",
    }
    assert not any(value in documents[3].payload.lower() for value in forbidden)


def test_golden_documents_are_frozen() -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent)
    result = _prepare(policy=policy, intent=intent, approval=approval)

    assert encode_fresh_opening_trust_policy(policy).sha256 == POLICY_GOLDEN_SHA
    assert encode_fresh_opening_intent(intent).sha256 == INTENT_GOLDEN_SHA
    assert encode_detached_fresh_opening_approval(approval).sha256 == (
        APPROVAL_GOLDEN_SHA
    )
    assert result.prospective_opening.opening_payload_sha256 == OPENING_GOLDEN_SHA
    assert result.candidate.candidate_document.sha256 == CANDIDATE_GOLDEN_SHA


@pytest.mark.parametrize(
    "missing",
    ("policy", "expected_policy", "expected_key"),
)
def test_absent_out_of_band_authority_is_explicitly_unconfigured(missing: str) -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent)
    policy_document = encode_fresh_opening_trust_policy(policy)
    kwargs = {
        "trust_policy": policy,
        "expected_trust_policy_sha256": policy_document.sha256,
        "expected_signer_public_key_sha256": PUBLIC_KEY_SHA256,
    }
    if missing == "policy":
        kwargs["trust_policy"] = None
    elif missing == "expected_policy":
        kwargs["expected_trust_policy_sha256"] = None
    else:
        kwargs["expected_signer_public_key_sha256"] = None

    result = prepare_fresh_opening(
        intent,
        approval,
        opening_codec=OPENING_CODEC,
        signature_verifier=SIGNATURE_VERIFIER,
        **kwargs,
        evaluated_at=EVALUATED_AT,
    )

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_AUTHORITY_UNCONFIGURED
    )
    assert result.candidate is None
    assert result.prospective_opening.opening_payload_sha256 == OPENING_GOLDEN_SHA


def test_prepare_requires_explicit_codec_and_signature_verifier_ports() -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent)
    common = {
        "trust_policy": policy,
        "expected_trust_policy_sha256": encode_fresh_opening_trust_policy(
            policy
        ).sha256,
        "expected_signer_public_key_sha256": PUBLIC_KEY_SHA256,
        "evaluated_at": EVALUATED_AT,
    }

    with pytest.raises(TypeError, match="opening_codec"):
        prepare_fresh_opening(
            intent,
            approval,
            opening_codec=None,
            signature_verifier=SIGNATURE_VERIFIER,
            **common,
        )
    with pytest.raises(TypeError, match="signature_verifier"):
        prepare_fresh_opening(
            intent,
            approval,
            opening_codec=OPENING_CODEC,
            signature_verifier=None,
            **common,
        )


def test_signature_verifier_port_must_return_an_exact_boolean() -> None:
    class InvalidVerifier:
        def verify(self, **kwargs: object) -> int:
            del kwargs
            return 1

    policy = _policy()
    intent = _intent(policy)
    with pytest.raises(TypeError, match="return a boolean"):
        prepare_fresh_opening(
            intent,
            _approval(intent),
            opening_codec=OPENING_CODEC,
            signature_verifier=InvalidVerifier(),
            trust_policy=policy,
            expected_trust_policy_sha256=encode_fresh_opening_trust_policy(
                policy
            ).sha256,
            expected_signer_public_key_sha256=PUBLIC_KEY_SHA256,
            evaluated_at=EVALUATED_AT,
        )


def test_missing_approval_is_blocked_after_prospective_derivation() -> None:
    policy = _policy()
    result = _prepare(policy=policy, approval=None)

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_MISSING
    )
    assert result.candidate is None
    assert result.prospective_opening.opening_payload_sha256 == OPENING_GOLDEN_SHA


def test_policy_digest_mismatch_is_blocked() -> None:
    policy = _policy()
    result = _prepare(policy=policy, expected_policy_sha256=SHA_A)

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_TRUST_POLICY_MISMATCH
    )


def test_intent_policy_digest_mismatch_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy, trust_policy_sha256=SHA_A)
    result = _prepare(policy=policy, intent=intent, approval=None)

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_TRUST_POLICY_MISMATCH
    )


def test_trust_domain_mismatch_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy, trust_domain="other-paper-domain")
    result = _prepare(policy=policy, intent=intent, approval=None)

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_TRUST_DOMAIN_MISMATCH
    )


def test_unknown_signer_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy, signer_key_id="opening-approver-unknown")
    result = _prepare(policy=policy, intent=intent, approval=None)

    assert (
        result.disposition is FreshOpeningPreparationDisposition.BLOCKED_SIGNER_UNKNOWN
    )


def test_revoked_signer_is_blocked() -> None:
    policy = _policy(_anchor(revoked=True))
    intent = _intent(policy)
    result = _prepare(policy=policy, intent=intent, approval=None)

    assert (
        result.disposition is FreshOpeningPreparationDisposition.BLOCKED_SIGNER_REVOKED
    )


def test_approver_identity_mismatch_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy, approver_identity="operator-reviewer-03")
    result = _prepare(policy=policy, intent=intent, approval=None)

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_APPROVER_MISMATCH
    )


@pytest.mark.parametrize("mismatch", ("external", "intent"))
def test_public_key_fingerprint_mismatch_is_blocked(mismatch: str) -> None:
    policy = _policy()
    intent = _intent(
        policy,
        **({"signer_public_key_sha256": SHA_A} if mismatch == "intent" else {}),
    )
    result = _prepare(
        policy=policy,
        intent=intent,
        approval=None,
        expected_key_sha256=SHA_A if mismatch == "external" else PUBLIC_KEY_SHA256,
    )

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_BINDING_MISMATCH
    )


def test_approval_intent_digest_mismatch_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent, intent_sha256=SHA_A)
    result = _prepare(policy=policy, intent=intent, approval=approval)

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_BINDING_MISMATCH
    )


def test_future_approval_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy)
    result = _prepare(
        policy=policy,
        intent=intent,
        evaluated_at=ISSUED_AT - timedelta(microseconds=1),
    )

    assert result.disposition is (
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_NOT_YET_VALID
    )


@pytest.mark.parametrize("evaluated", (EXPIRES_AT, EXPIRES_AT + timedelta(seconds=1)))
def test_expired_approval_is_blocked_at_and_after_boundary(evaluated: datetime) -> None:
    policy = _policy()
    intent = _intent(policy)
    result = _prepare(policy=policy, intent=intent, evaluated_at=evaluated)

    assert (
        result.disposition
        is FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED
    )


def test_approval_lifetime_over_policy_limit_is_blocked() -> None:
    policy = _policy(max_approval_lifetime_seconds=3599)
    intent = _intent(policy)
    result = _prepare(policy=policy, intent=intent)

    assert (
        result.disposition
        is FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED
    )


def test_structurally_valid_signature_over_other_bytes_is_blocked() -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent, signed_bytes=b"different signed message")
    result = _prepare(policy=policy, intent=intent, approval=approval)

    assert (
        result.disposition
        is FreshOpeningPreparationDisposition.BLOCKED_SIGNATURE_INVALID
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("logical_target", "paper-production-secondary"),
        ("execution_scope", "paper:production:other"),
        ("account_key", "paper-v2-secondary"),
        ("owner_generation", 2),
        ("collateral_asset", "USDC"),
        ("collateral_amount", Decimal("1001.00")),
        ("margin_quantum", Decimal("0.10")),
        ("operator_identity", "operator-requester-03"),
        ("approval_id", "opening-approval-2026-08-14-002"),
        ("nonce", "f" * 64),
    ),
)
def test_every_mutated_business_field_invalidates_detached_signature(
    field: str,
    replacement: object,
) -> None:
    policy = _policy()
    original = _intent(policy)
    original_signature = _approval(original).signature
    mutated = replace(original, **{field: replacement})
    mutated_document = encode_fresh_opening_intent(mutated)
    approval = DetachedFreshOpeningApproval(
        1,
        mutated_document.sha256,
        original_signature,
    )
    result = _prepare(policy=policy, intent=mutated, approval=approval)

    assert (
        result.disposition
        is FreshOpeningPreparationDisposition.BLOCKED_SIGNATURE_INVALID
    )


def test_decimal_scale_is_part_of_signed_and_opening_identity() -> None:
    policy = _policy()
    first = _intent(policy, collateral_amount=Decimal("1000.00"))
    second = _intent(policy, collateral_amount=Decimal("1000.0"))

    first_result = _prepare(policy=policy, intent=first)
    second_result = _prepare(policy=policy, intent=second)

    assert first.collateral_amount == second.collateral_amount
    assert encode_fresh_opening_intent(first).sha256 != (
        encode_fresh_opening_intent(second).sha256
    )
    assert first_result.prospective_opening.opening_payload_sha256 != (
        second_result.prospective_opening.opening_payload_sha256
    )


@pytest.mark.parametrize(
    "amount",
    (
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ),
)
def test_non_positive_or_non_finite_collateral_is_rejected(amount: Decimal) -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="finite and positive"):
        _intent(policy, collateral_amount=amount)


@pytest.mark.parametrize(
    "quantum",
    (Decimal("0"), Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity")),
)
def test_invalid_margin_quantum_is_rejected(quantum: Decimal) -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="finite and positive"):
        _intent(policy, margin_quantum=quantum)


def test_quantization_is_an_exact_step_not_only_a_decimal_exponent() -> None:
    policy = _policy()

    assert _intent(
        policy,
        collateral_amount=Decimal("10.00"),
        margin_quantum=Decimal("0.05"),
    )
    with pytest.raises(ValueError, match="quantized"):
        _intent(
            policy,
            collateral_amount=Decimal("10.01"),
            margin_quantum=Decimal("0.05"),
        )


def test_decimal_bounds_reject_hostile_precision_and_exponent() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="precision"):
        _intent(policy, collateral_amount=Decimal("1" * 129))
    with pytest.raises(ValueError, match="exponent"):
        _intent(policy, collateral_amount=Decimal("1E+129"))
    with pytest.raises(ValueError, match="fixed-point"):
        _intent(policy, collateral_amount=Decimal("1E+2"))
    with pytest.raises(ValueError, match="fixed-point"):
        _intent(policy, collateral_amount=Decimal("1E-7"))


def test_decimal_derivation_ignores_hostile_ambient_context() -> None:
    policy = _policy()
    intent = _intent(
        policy,
        collateral_amount=Decimal("10.00"),
        margin_quantum=Decimal("0.05"),
    )
    expected = _prepare(policy=policy, intent=intent)

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        actual = _prepare(policy=policy, intent=intent)

    assert actual.intent_document == expected.intent_document
    assert actual.prospective_opening == expected.prospective_opening
    assert actual.candidate.candidate_document == expected.candidate.candidate_document


@pytest.mark.parametrize("generation", (True, False, 0, -1, 1 << 63))
def test_owner_generation_rejects_booleans_and_storage_overflow(
    generation: object,
) -> None:
    policy = _policy()
    error = TypeError if type(generation) is bool else ValueError
    with pytest.raises(error):
        _intent(policy, owner_generation=generation)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("logical_target", ""),
        ("execution_scope", " padded "),
        ("account_key", "contains space"),
        ("collateral_asset", "USDT\x00"),
        ("operator_identity", "example"),
        ("approval_id", "todo"),
        ("approver_identity", "é"),
        ("trust_domain", "line\nbreak"),
        ("signer_key_id", "\ud800"),
    ),
)
def test_ambiguous_or_placeholder_text_is_rejected(field: str, value: str) -> None:
    policy = _policy()
    with pytest.raises((TypeError, ValueError)):
        _intent(policy, **{field: value})


def test_operator_and_approver_must_be_independent() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="independent"):
        _intent(policy, operator_identity="operator-reviewer-02")


@pytest.mark.parametrize(
    "nonce",
    ("0" * 64, "A" * 64, "a" * 63, "g" * 64),
)
def test_nonce_must_be_nonzero_256_bit_lowercase_hex(nonce: str) -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="nonce"):
        _intent(policy, nonce=nonce)


SMALL_ORDER_ENCODINGS = tuple(
    bytes.fromhex(value)
    for value in (
        "0100000000000000000000000000000000000000000000000000000000000000",
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
        "0000000000000000000000000000000000000000000000000000000000000080",
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa",
    )
)


@pytest.mark.parametrize("public_key", SMALL_ORDER_ENCODINGS)
def test_all_canonical_small_order_public_keys_are_rejected(public_key: bytes) -> None:
    with pytest.raises(ValueError, match="canonical and non-weak"):
        _anchor(ed25519_public_key=public_key)


@pytest.mark.parametrize("encoded_y", ((1 << 255) - 19, (1 << 255) - 18))
def test_noncanonical_public_key_y_is_rejected(encoded_y: int) -> None:
    noncanonical_y = encoded_y.to_bytes(32, "little")
    with pytest.raises(ValueError, match="canonical and non-weak"):
        _anchor(ed25519_public_key=noncanonical_y)


@pytest.mark.parametrize("public_key", (b"", b"x" * 31, b"x" * 33))
def test_public_key_length_is_exact(public_key: bytes) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        _anchor(ed25519_public_key=public_key)


def test_policy_anchors_are_nonempty_unique_sorted_and_not_key_aliases() -> None:
    second = _anchor(
        signer_key_id="opening-approver-2026-02",
        approver_identity="operator-reviewer-03",
        ed25519_public_key=SECOND_PUBLIC_KEY,
    )
    first = _anchor()

    assert _policy(first, second)
    with pytest.raises(ValueError, match="must not be empty"):
        _policy(anchors=())
    with pytest.raises(ValueError, match="unique sorted"):
        _policy(second, first)
    with pytest.raises(ValueError, match="unique sorted"):
        _policy(first, replace(first, ed25519_public_key=SECOND_PUBLIC_KEY))
    with pytest.raises(ValueError, match="alias"):
        _policy(
            first,
            replace(
                first,
                signer_key_id="opening-approver-2026-02",
                approver_identity="operator-reviewer-03",
            ),
        )


@pytest.mark.parametrize("r", SMALL_ORDER_ENCODINGS)
def test_signature_rejects_every_small_order_r_before_verification(r: bytes) -> None:
    valid = _approval(_intent(_policy())).signature
    with pytest.raises(ValueError, match="signature R"):
        DetachedFreshOpeningApproval(1, SHA_A, r + valid[32:])


@pytest.mark.parametrize("encoded_y", ((1 << 255) - 19, (1 << 255) - 18))
def test_signature_rejects_noncanonical_r_before_verification(encoded_y: int) -> None:
    valid = _approval(_intent(_policy())).signature
    noncanonical_y = encoded_y.to_bytes(32, "little")
    with pytest.raises(ValueError, match="signature R"):
        DetachedFreshOpeningApproval(1, SHA_A, noncanonical_y + valid[32:])


@pytest.mark.parametrize(
    "scalar",
    (
        (1 << 252) + 27742317777372353535851937790883648493,
        (1 << 256) - 1,
    ),
)
def test_signature_requires_canonical_s_less_than_group_order(scalar: int) -> None:
    valid = _approval(_intent(_policy())).signature
    with pytest.raises(ValueError, match="signature S"):
        DetachedFreshOpeningApproval(
            1,
            SHA_A,
            valid[:32] + scalar.to_bytes(32, "little"),
        )


@pytest.mark.parametrize("signature", (b"", b"x" * 63, b"x" * 65))
def test_signature_length_is_exact(signature: bytes) -> None:
    with pytest.raises(ValueError, match="64 bytes"):
        DetachedFreshOpeningApproval(1, SHA_A, signature)


def test_intent_datetime_is_normalized_to_utc_and_naive_is_rejected() -> None:
    policy = _policy()
    offset = timezone(timedelta(hours=2))
    intent = _intent(
        policy,
        approval_issued_at=ISSUED_AT.astimezone(offset),
        approval_expires_at=EXPIRES_AT.astimezone(offset),
    )

    assert intent.approval_issued_at == ISSUED_AT
    assert intent.approval_issued_at.tzinfo is UTC
    assert encode_fresh_opening_intent(intent) == (
        encode_fresh_opening_intent(_intent(policy))
    )
    with pytest.raises(TypeError, match="timezone-aware"):
        _intent(policy, approval_issued_at=ISSUED_AT.replace(tzinfo=None))


def test_signing_bytes_reject_a_forged_document_digest() -> None:
    document = CanonicalFreshOpeningDocument("{}", SHA_A)
    with pytest.raises(ValueError, match="does not match"):
        fresh_opening_signing_bytes(document)


@pytest.mark.parametrize(
    "payload",
    ('{ "value":1}', '{"value":1,"value":1}', "[]", '{"value":NaN}'),
)
def test_canonical_document_rejects_noncanonical_or_nonobject_json(
    payload: str,
) -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        CanonicalFreshOpeningDocument(payload, SHA_A)


def test_public_values_reject_copy_pickle_setstate_mutation() -> None:
    policy = _policy()
    intent = _intent(policy)
    approval = _approval(intent)
    result = _prepare(policy=policy, intent=intent, approval=approval)
    values = (
        policy.anchors[0],
        policy,
        intent,
        approval,
        result.intent_document,
        result.candidate,
        result,
    )

    for value in values:
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        state = [getattr(value, field.name) for field in fields(value)]
        with pytest.raises(TypeError, match="state mutation"):
            value.__setstate__(state)


def test_blocked_result_cannot_carry_candidate_and_flags_cannot_be_elevated() -> None:
    prepared = _prepare(policy=_policy())

    with pytest.raises(ValueError, match="blocked"):
        replace(
            prepared,
            disposition=FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_MISSING,
        )
    for field in (
        "nonce_replay_authority_available",
        "physical_target_bound",
        "opening_authorized",
        "provisioning_authorized",
        "runtime_authorized",
        "trading_authorized",
        "pin_source_authenticated",
    ):
        with pytest.raises(ValueError, match="flags"):
            replace(prepared, **{field: True})
    with pytest.raises(ValueError, match="stale"):
        replace(prepared, stale_on_return=False)


def test_prepared_result_and_candidate_reject_cross_evidence_splicing() -> None:
    policy = _policy()
    prepared = _prepare(policy=policy)
    other = _prepare(
        policy=policy,
        intent=_intent(policy, collateral_amount=Decimal("1001.00")),
    )

    with pytest.raises(ValueError, match="exact intent"):
        replace(prepared, intent_document=other.intent_document)
    with pytest.raises(ValueError, match="exact opening"):
        replace(prepared, prospective_opening=other.prospective_opening)
    with pytest.raises(ValueError, match="derived from its evidence"):
        replace(
            prepared.candidate,
            candidate_document=CanonicalFreshOpeningDocument("{}", SHA_A),
        )
    for field in (
        "intent_document",
        "trust_policy_document",
        "approval_document",
    ):
        with pytest.raises(ValueError, match="digest is inconsistent"):
            replace(
                prepared.candidate,
                **{field: replace(getattr(prepared.candidate, field), sha256=SHA_A)},
            )
    with pytest.raises(ValueError, match="opening digest"):
        replace(
            prepared.candidate,
            opening=replace(prepared.candidate.opening, opening_payload_sha256=SHA_A),
        )


def test_module_has_no_private_key_db_env_cli_runtime_or_legacy_capability() -> None:
    module_path = (
        Path(__file__).parents[1] / "trading" / "application" / "fresh_opening.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = []
    names = set()
    called_attributes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)

    forbidden_import_prefixes = (
        "argparse",
        "click",
        "cryptography",
        "os",
        "psycopg2",
        "sqlalchemy",
        "subprocess",
        "trading.application.legacy",
        "trading.application.paper_runtime",
        "trading.domain.paper_accounting",
        "trading.persistence",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported
        for prefix in forbidden_import_prefixes
    )
    assert "Ed25519PrivateKey" not in names
    assert "sign" not in called_attributes


def test_module_is_not_exported_from_application_facade() -> None:
    facade = Path(__file__).parents[1] / "trading" / "application" / "__init__.py"
    source = facade.read_text(encoding="utf-8")

    assert "fresh_opening" not in source
    assert "FreshOpening" not in source


# Filled from the deterministic test key and exact canonical formats above.
POLICY_GOLDEN_SHA = "9e61d9accd359f7fae6601310a1f01dc2b630982c8a0a3be7be547b601c2a2b6"
INTENT_GOLDEN_SHA = "dec1504d3b0361fdda58b5c02bf84da69cc6d6a8d170e1295c34b33d9ca04e5b"
APPROVAL_GOLDEN_SHA = "90ef12eebf27689cab28269ec2bf5d3eeffadf6f58499f69f3d900b51dcb6bb6"
OPENING_GOLDEN_SHA = "558e6e3087114e908831b900c2ef322f9312c08ffd80bffd8278640aeb92887f"
CANDIDATE_GOLDEN_SHA = (
    "5e48334662b6de9c6ff9bec3e52577c75a14cc20e7c768c8836b6f66e15f3618"
)
