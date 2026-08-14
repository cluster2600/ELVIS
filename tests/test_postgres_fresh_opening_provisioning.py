"""Fast contracts for the durable PostgreSQL fresh-opening adapter."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.application.fresh_opening import (
    DetachedFreshOpeningApproval,
    FreshOpeningIntent,
    FreshOpeningPolicy,
    FreshOpeningPreparationDisposition,
    FreshOpeningTrustAnchor,
    FreshOpeningTrustPolicy,
    derive_prospective_fresh_opening_candidate,
    encode_fresh_opening_intent,
    encode_fresh_opening_trust_policy,
    fresh_opening_signing_bytes,
)
from trading.application.fresh_opening_provisioning import (
    FreshOpeningPhysicalTarget,
    FreshOpeningProvisioningDisposition,
    FreshOpeningProvisioningRequest,
)
from trading.domain.paper_accounting import (
    PaperAccountBalance,
    PaperAccountPolicy,
    new_paper_account,
)
from trading.persistence.paper_account_journal_codec import (
    encode_paper_account_opening,
)
from trading.persistence.postgres_fresh_opening_provisioning import (
    PostgresFreshOpeningProvisioning,
    PostgresFreshOpeningProvisioningStorageError,
    _database_incarnation_id,
    _fence_evidence,
    _receipt_documents,
)

UTC = timezone.utc
EVALUATED_AT = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)
ISSUED_AT = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
EXPIRES_AT = datetime(2026, 8, 14, 13, 0, tzinfo=UTC)
MIGRATION_CHECKSUM = "a" * 64
TERMINAL_CATALOG_SHA256 = "b" * 64
PIN_AUTHORITY_SHA256 = "c" * 64


class _OpeningCodec:
    def encode(
        self,
        *,
        execution_scope,
        account_key,
        owner_generation,
        collateral_asset,
        collateral_amount,
        margin_quantum,
    ):
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


class _Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, parameters=None):
        self.executions.append((" ".join(statement.split()), parameters))

    def fetchone(self):
        return self.rows.pop(0)


class _Connection:
    def __init__(self, rows, *, commit_error=None):
        self.autocommit = False
        self.status = STATUS_READY
        self.cursor_value = _Cursor(rows)
        self.commit_error = commit_error
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def get_transaction_status(self):
        return TRANSACTION_STATUS_IDLE

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.commits += 1
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1


class _Authority:
    def __init__(
        self, candidate, disposition=FreshOpeningPreparationDisposition.PREPARED
    ):
        self.candidate = candidate
        self.disposition = disposition
        self.calls = []

    def evaluate(self, evaluated_at):
        self.calls.append(evaluated_at)
        return SimpleNamespace(
            disposition=self.disposition,
            candidate=(
                self.candidate
                if self.disposition is FreshOpeningPreparationDisposition.PREPARED
                else None
            ),
        )


class _HostileConnectionInterface:
    @property
    def cursor(self):
        raise RuntimeError("postgresql://user:secret@example.invalid/database")

    @property
    def close(self):
        raise RuntimeError("postgresql://user:secret@example.invalid/database")


class _HostileAuthorityInterface:
    @property
    def evaluate(self):
        raise RuntimeError("postgresql://user:secret@example.invalid/database")


def _case():
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    anchor = FreshOpeningTrustAnchor(
        signer_key_id="opening-approver-2026-01",
        approver_identity="operator-reviewer-02",
        ed25519_public_key=public_key,
        revoked=False,
    )
    policy = FreshOpeningTrustPolicy(
        schema_version=1,
        purpose="ELVIS_V2_FRESH_PAPER_OPENING",
        trust_domain="elvis-paper-production",
        max_approval_lifetime_seconds=3600,
        anchors=(anchor,),
    )
    policy_document = encode_fresh_opening_trust_policy(policy)
    intent = FreshOpeningIntent(
        schema_version=1,
        purpose="ELVIS_V2_FRESH_PAPER_OPENING",
        trajectory="B",
        continuity="NO_V1_CONTINUITY",
        logical_target="paper-production-primary",
        execution_scope="paper:production",
        account_key="paper-v2-main",
        owner_generation=1,
        opening_codec="paper-account-opening",
        opening_version=1,
        collateral_asset="USDT",
        collateral_amount=Decimal("1000.00"),
        margin_quantum=Decimal("0.01"),
        opening_policy=FreshOpeningPolicy.EXPLICIT_FRESH_SINGLE_COLLATERAL,
        operator_identity="operator-requester-01",
        approval_id="opening-approval-2026-08-14-001",
        approver_identity=anchor.approver_identity,
        approval_issued_at=ISSUED_AT,
        approval_expires_at=EXPIRES_AT,
        trust_policy_sha256=policy_document.sha256,
        trust_domain=policy.trust_domain,
        signer_key_id=anchor.signer_key_id,
        signer_public_key_sha256=anchor.public_key_sha256,
        nonce="0123456789abcdef" * 4,
    )
    intent_document = encode_fresh_opening_intent(intent)
    approval = DetachedFreshOpeningApproval(
        schema_version=1,
        intent_sha256=intent_document.sha256,
        signature=private_key.sign(fresh_opening_signing_bytes(intent_document)),
    )
    target = FreshOpeningPhysicalTarget(
        expected_database="elvis_paper_v2",
        expected_system_identifier=123456789,
        control_plane_role="elvis_bootstrap_admin",
        opening_anchor_role="elvis_v2_opening",
        deployment_incarnation_id="deployment-2026-08-14-001",
        terminal_catalog_sha256=TERMINAL_CATALOG_SHA256,
        pin_authority_record_sha256=PIN_AUTHORITY_SHA256,
    )
    request = FreshOpeningProvisioningRequest(
        intent=intent,
        approval=approval,
        trust_policy=policy,
        expected_trust_policy_sha256=policy_document.sha256,
        expected_signer_public_key_sha256=anchor.public_key_sha256,
        target=target,
    )
    candidate = derive_prospective_fresh_opening_candidate(
        intent,
        approval,
        policy,
        opening_codec=_OpeningCodec(),
    )
    return request, candidate


def _absent_row(request, **overrides):
    values = [
        "ABSENT",
        EVALUATED_AT,
        request.target.expected_database,
        Decimal(request.target.expected_system_identifier),
        request.target.control_plane_role,
        request.target.opening_anchor_role,
        7,
        "fresh_opening_provenance",
        MIGRATION_CHECKSUM,
        request.target.terminal_catalog_sha256,
        request.target.pin_authority_record_sha256,
        request.target.deployment_incarnation_id,
        None,
        "LEGACY",
        0,
        0,
        0,
        True,
        *([None] * 16),
    ]
    positions = {
        "resolution": 0,
        "database_name": 2,
        "system_identifier": 3,
        "control_plane_role": 4,
        "opening_anchor_role": 5,
        "migration_version": 6,
        "terminal_catalog_sha256": 9,
        "pin_authority_record_sha256": 10,
        "deployment_incarnation_id": 11,
        "database_incarnation_id": 12,
        "runtime_mode": 13,
        "v2_empty": 17,
    }
    for name, value in overrides.items():
        values[positions[name]] = value
    assert len(values) == 34
    return tuple(values)


def _present_row(request, candidate, *, resolution="EXACT_REPLAY", conflict=False):
    absent = _fence_evidence(_absent_row(request))
    documents = _receipt_documents(request, candidate, absent)
    candidate_sha256 = "d" * 64 if conflict else candidate.candidate_document.sha256
    row = (
        resolution,
        EVALUATED_AT,
        request.target.expected_database,
        Decimal(request.target.expected_system_identifier),
        request.target.control_plane_role,
        request.target.opening_anchor_role,
        7,
        "fresh_opening_provenance",
        MIGRATION_CHECKSUM,
        request.target.terminal_catalog_sha256,
        request.target.pin_authority_record_sha256,
        request.target.deployment_incarnation_id,
        _database_incarnation_id(request.target, absent),
        "LEGACY",
        0,
        0,
        0,
        False,
        EVALUATED_AT,
        EVALUATED_AT,
        candidate.intent_document.payload,
        candidate.intent_document.sha256,
        candidate.approval_document.payload,
        candidate.approval_document.sha256,
        candidate.trust_policy_document.payload,
        candidate.trust_policy_document.sha256,
        candidate.candidate_document.payload,
        candidate_sha256,
        candidate.opening.opening_payload,
        candidate.opening.opening_payload_sha256,
        documents.opening.payload,
        documents.opening.sha256,
        documents.provisioning.payload,
        documents.provisioning.sha256,
    )
    assert len(row) == 34
    return row


def _commit_row(request, candidate):
    evidence = _fence_evidence(_absent_row(request))
    documents = _receipt_documents(request, candidate, evidence)
    return (
        "CREATED",
        EVALUATED_AT,
        documents.opening.payload,
        documents.opening.sha256,
        documents.provisioning.payload,
        documents.provisioning.sha256,
    )


def test_absent_opening_uses_database_time_and_commits_once():
    request, candidate = _case()
    authority = _Authority(candidate)
    connection = _Connection([_absent_row(request), _commit_row(request, candidate)])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.CREATED
    assert result.primary_reason_code == "FRESH_OPENING_CREATED"
    assert authority.calls == [EVALUATED_AT]
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closes == 1
    assert result.receipt is not None
    receipt = json.loads(result.receipt.document.payload)
    assert set(receipt) == {
        "approval_sha256",
        "authority_evaluated_at",
        "authority_transition_sequence",
        "candidate_sha256",
        "control_plane_role",
        "database_incarnation_id",
        "database_name",
        "deployment_incarnation_id",
        "intent_sha256",
        "migration_checksum",
        "migration_head",
        "migration_name",
        "opening_payload_sha256",
        "opening_receipt_sha256",
        "opening_anchor_role",
        "pin_authority_record_sha256",
        "runtime_activation_authorized",
        "runtime_generation",
        "runtime_mode",
        "schema_version",
        "stale_on_return",
        "system_identifier",
        "terminal_catalog_sha256",
        "trading_authorized",
        "trust_policy_sha256",
        "writer_fence",
    }
    assert receipt["system_identifier"] == "123456789"
    assert receipt["runtime_mode"] == "LEGACY"
    assert receipt["writer_fence"] == 0
    assert receipt["runtime_activation_authorized"] is False
    assert receipt["trading_authorized"] is False


def test_exact_replay_precedes_current_authority_and_rolls_back():
    request, candidate = _case()
    authority = _Authority(
        candidate,
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED,
    )
    connection = _Connection([_present_row(request, candidate)])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.REPLAYED
    assert result.primary_reason_code == "EXACT_DURABLE_REPLAY"
    assert authority.calls == []
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_admission_conflict_cannot_carry_a_durable_opening():
    request, candidate = _case()

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError):
        _fence_evidence(
            _present_row(request, candidate, resolution="ADMISSION_CONFLICT")
        )


def test_durable_opening_cannot_report_an_empty_target():
    request, candidate = _case()
    row = list(_present_row(request, candidate))
    row[17] = True

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError):
        _fence_evidence(tuple(row))


@pytest.mark.parametrize("stored", [False, True])
def test_database_incarnation_presence_matches_durable_opening(stored):
    request, candidate = _case()
    if stored:
        row = list(_present_row(request, candidate))
        row[12] = None
    else:
        row = list(_absent_row(request))
        row[12] = "d" * 64

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError):
        _fence_evidence(tuple(row))


@pytest.mark.parametrize("resolution", ["NONCE_CONFLICT", "TARGET_CONFLICT"])
def test_durable_conflict_precedes_target_admission_and_current_authority(resolution):
    request, candidate = _case()
    authority = _Authority(candidate)
    row = list(_present_row(request, candidate, resolution=resolution, conflict=True))
    row[9] = "d" * 64
    connection = _Connection([tuple(row)])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.CONFLICT
    assert result.primary_reason_code == (
        "FRESH_OPENING_NONCE_CONFLICT"
        if resolution == "NONCE_CONFLICT"
        else "FRESH_OPENING_TARGET_CONFLICT"
    )
    assert result.side_effect_state == "NONE"
    assert authority.calls == []
    assert connection.rollbacks == 1


def test_physical_target_drift_blocks_before_current_authority():
    request, candidate = _case()
    authority = _Authority(candidate)
    connection = _Connection([_absent_row(request, database_name="other_database")])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "TARGET_ADMISSION_BLOCKED"
    assert authority.calls == []
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    ("position", "drifted_value"),
    [
        (2, "other_database"),
        (3, Decimal("987654321")),
        (4, "other_control_plane_role"),
        (5, "other_opening_anchor"),
        (9, "d" * 64),
        (10, "d" * 64),
        (11, "other-deployment"),
        (12, "d" * 64),
    ],
)
def test_exact_replay_target_drift_blocks_before_current_authority(
    position,
    drifted_value,
):
    request, candidate = _case()
    authority = _Authority(candidate)
    row = list(_present_row(request, candidate))
    row[position] = drifted_value
    connection = _Connection([tuple(row)])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "TARGET_ADMISSION_BLOCKED"
    assert authority.calls == []
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("pin_authority_record_sha256", "d" * 64),
        ("deployment_incarnation_id", "other-deployment"),
    ],
)
def test_admission_target_drift_blocks_before_current_authority(override, value):
    request, candidate = _case()
    authority = _Authority(candidate)
    connection = _Connection([_absent_row(request, **{override: value})])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "TARGET_ADMISSION_BLOCKED"
    assert authority.calls == []
    assert connection.rollbacks == 1


def test_admission_candidate_conflict_blocks_before_current_authority():
    request, candidate = _case()
    authority = _Authority(candidate)
    connection = _Connection([_absent_row(request, resolution="ADMISSION_CONFLICT")])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "TARGET_ADMISSION_BLOCKED"
    assert authority.calls == []
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    "missing_evidence",
    ["pin_authority_record_sha256", "deployment_incarnation_id"],
)
def test_missing_admission_evidence_is_rejected_before_current_authority(
    missing_evidence,
):
    request, candidate = _case()
    authority = _Authority(candidate)
    connection = _Connection([_absent_row(request, **{missing_evidence: None})])

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError):
        PostgresFreshOpeningProvisioning(lambda: connection).provision(
            request,
            candidate,
            authority,
        )

    assert authority.calls == []
    assert connection.rollbacks == 1


def test_absent_opening_blocked_by_current_authority_never_calls_commit():
    request, candidate = _case()
    authority = _Authority(
        candidate,
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED,
    )
    connection = _Connection([_absent_row(request)])

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "BLOCKED_APPROVAL_EXPIRED"
    assert authority.calls == [EVALUATED_AT]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert not any(
        "commit_paper_fresh_opening" in statement
        for statement, _ in connection.cursor_value.executions
    )


def test_approval_expiring_inside_commit_is_blocked_without_side_effects():
    request, candidate = _case()
    authority = _Authority(candidate)
    connection = _Connection([_absent_row(request)])
    execute = connection.cursor_value.execute

    class ApprovalExpiredError(Exception):
        pgcode = "PT004"

    def expire_at_commit(statement, parameters=None):
        if "commit_paper_fresh_opening" in statement:
            raise ApprovalExpiredError
        execute(statement, parameters)

    connection.cursor_value.execute = expire_at_commit

    result = PostgresFreshOpeningProvisioning(lambda: connection).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "BLOCKED_APPROVAL_EXPIRED"
    assert result.side_effect_state == "NONE"
    assert result.current_authority_evaluated is True
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_lost_commit_ack_is_created_only_after_exact_independent_readback():
    request, candidate = _case()
    authority = _Authority(candidate)
    write = _Connection(
        [_absent_row(request), _commit_row(request, candidate)],
        commit_error=RuntimeError("lost acknowledgement"),
    )
    readback = _Connection([_present_row(request, candidate)])
    connections = iter((write, readback))

    result = PostgresFreshOpeningProvisioning(lambda: next(connections)).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.CREATED
    assert result.primary_reason_code == "FRESH_OPENING_CREATED"
    assert result.side_effect_state == "COMMITTED"
    assert write.commits == 1
    assert write.closes == 1
    assert readback.rollbacks == 1
    assert readback.closes == 1


def test_lost_commit_ack_without_exact_readback_remains_unknown():
    request, candidate = _case()
    authority = _Authority(candidate)
    write = _Connection(
        [_absent_row(request), _commit_row(request, candidate)],
        commit_error=RuntimeError("lost acknowledgement"),
    )
    readback = _Connection([_absent_row(request)])
    connections = iter((write, readback))

    result = PostgresFreshOpeningProvisioning(lambda: next(connections)).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN
    assert result.primary_reason_code == "FRESH_OPENING_COMMIT_UNKNOWN"
    assert result.receipt is None
    assert result.side_effect_state == "UNKNOWN"
    assert result.runtime_activation_authorized is False
    assert result.trading_authorized is False


@pytest.mark.parametrize("resolution", ["NONCE_CONFLICT", "TARGET_CONFLICT"])
def test_lost_commit_ack_with_contradictory_resolution_remains_unknown(resolution):
    request, candidate = _case()
    authority = _Authority(candidate)
    write = _Connection(
        [_absent_row(request), _commit_row(request, candidate)],
        commit_error=RuntimeError("lost acknowledgement"),
    )
    readback = _Connection([_present_row(request, candidate, resolution=resolution)])
    connections = iter((write, readback))

    result = PostgresFreshOpeningProvisioning(lambda: next(connections)).provision(
        request,
        candidate,
        authority,
    )

    assert result.disposition is FreshOpeningProvisioningDisposition.COMMIT_UNKNOWN
    assert result.primary_reason_code == "FRESH_OPENING_COMMIT_UNKNOWN"
    assert result.receipt is None
    assert result.side_effect_state == "UNKNOWN"
    assert result.current_authority_evaluated is True


def test_connection_failure_is_secret_free():
    request, candidate = _case()

    def fail():
        raise RuntimeError("postgresql://user:secret@example.invalid/database")

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError) as caught:
        PostgresFreshOpeningProvisioning(fail).provision(
            request,
            candidate,
            _Authority(candidate),
        )

    assert str(caught.value) == "could not open a fresh-opening connection"
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_hostile_connection_interface_failure_is_secret_free():
    request, candidate = _case()

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError) as caught:
        PostgresFreshOpeningProvisioning(_HostileConnectionInterface).provision(
            request,
            candidate,
            _Authority(candidate),
        )

    assert str(caught.value) == "fresh-opening connection has an invalid interface"
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_hostile_authority_interface_failure_is_secret_free():
    request, candidate = _case()

    with pytest.raises(TypeError) as caught:
        PostgresFreshOpeningProvisioning(lambda: _Connection([])).provision(
            request,
            candidate,
            _HostileAuthorityInterface(),
        )

    assert str(caught.value) == (
        "current_authority must implement FreshOpeningCurrentAuthorityPort"
    )
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
