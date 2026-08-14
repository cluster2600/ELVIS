"""Disposable PostgreSQL checks for one durable V2 fresh opening."""

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from threading import Barrier
from types import SimpleNamespace

import psycopg2
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from psycopg2 import sql
from psycopg2.extensions import parse_dsn

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
    FreshOpeningProvisioningService,
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
    _ACQUIRE_FENCE_SQL,
    _COMMIT_OPENING_SQL,
    PostgresFreshOpeningProvisioning,
    PostgresFreshOpeningProvisioningStorageError,
    _fence_evidence,
    _receipt_documents,
)

_OPENING_FUNCTIONS = (
    "np.acquire_paper_fresh_opening_fence(text,text,text,text)",
    "np.commit_paper_fresh_opening(" + ",".join(["text"] * 14) + ")",
    "np.read_paper_fresh_opening(text,text,text)",
)
_INTENT_SIGNING_PREFIX = b"ELVIS\x00fresh-opening-intent\x00v1\x00"


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


class _SignatureVerifier:
    def verify(self, *, public_key, signature, message):
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        except InvalidSignature, ValueError:
            return False
        return True


class _CommitAckLostConnection:
    """Delegate whose commit is durable but whose acknowledgement is lost."""

    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def commit(self):
        self._connection.commit()
        raise psycopg2.OperationalError("simulated lost commit acknowledgement")


class _BlockedAuthority:
    def __init__(self, disposition):
        self.disposition = disposition
        self.calls = []

    def evaluate(self, evaluated_at):
        self.calls.append(evaluated_at)
        return SimpleNamespace(disposition=self.disposition, candidate=None)


@pytest.fixture
def fresh_opening_database(
    migrated_postgres_dsn,
):
    role = f"elvis_opening_{uuid.uuid4().hex[:20]}"
    other_role = f"elvis_other_{uuid.uuid4().hex[:20]}"
    target_parameters = parse_dsn(migrated_postgres_dsn)
    database_name = target_parameters["dbname"]
    control_plane_role = target_parameters["user"]
    admin = psycopg2.connect(migrated_postgres_dsn)
    admin.autocommit = True
    created_roles = []
    role_dsn = migrated_postgres_dsn
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                    "NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD NULL"
                ).format(sql.Identifier(role))
            )
            created_roles.append(role)
            cursor.execute(
                sql.SQL(
                    "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                    "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(other_role))
            )
            created_roles.append(other_role)
            cursor.execute(
                sql.SQL("COMMENT ON ROLE {} IS %s").format(sql.Identifier(role)),
                (f"elvis-postgres-bootstrap:v2:{database_name}:opening",),
            )
            cursor.execute("""
                SELECT procedure.oid::regprocedure::TEXT
                FROM pg_proc procedure
                JOIN pg_namespace namespace
                  ON namespace.oid = procedure.pronamespace
                WHERE namespace.nspname = 'np'
                """)
            for (function,) in cursor.fetchall():
                cursor.execute(
                    sql.SQL("REVOKE ALL ON FUNCTION {} FROM PUBLIC").format(
                        sql.SQL(function)
                    )
                )
            cursor.execute(
                sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC, {}").format(
                    sql.Identifier(database_name),
                    sql.Identifier(role),
                )
            )
            cursor.execute(
                sql.SQL("REVOKE ALL ON SCHEMA np FROM PUBLIC, {}").format(
                    sql.Identifier(role)
                )
            )
            cursor.execute(
                sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA np FROM {}").format(
                    sql.Identifier(role)
                )
            )
            cursor.execute(
                sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA np FROM {}").format(
                    sql.Identifier(role)
                )
            )
            cursor.execute(
                sql.SQL("REVOKE ALL ON ALL FUNCTIONS IN SCHEMA np FROM {}").format(
                    sql.Identifier(role)
                )
            )
            cursor.execute("SELECT system_identifier::numeric FROM pg_control_system()")
            system_identifier = int(cursor.fetchone()[0])
            cursor.execute("SELECT transaction_timestamp()")
            database_time = cursor.fetchone()[0]

        database = SimpleNamespace(
            admin_dsn=migrated_postgres_dsn,
            role_dsn=role_dsn,
            database_name=database_name,
            role=role,
            control_plane_role=control_plane_role,
            other_role=other_role,
            system_identifier=system_identifier,
            database_time=database_time,
            terminal_catalog_sha256="0" * 64,
        )

        def admit(request, *, tampered=False, candidate_sha256=None):
            candidate = _candidate(request)
            admitted_candidate_sha256 = (
                candidate.candidate_document.sha256
                if candidate_sha256 is None
                else candidate_sha256
            )
            admission_payload = json.dumps(
                {
                    "candidate_sha256": admitted_candidate_sha256,
                    "deployment_incarnation_id": (
                        request.target.deployment_incarnation_id
                    ),
                    "pin_authority_record_sha256": (
                        request.target.pin_authority_record_sha256
                    ),
                    "schema_version": 1,
                },
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            admission_sha256 = hashlib.sha256(
                admission_payload.encode("utf-8")
            ).hexdigest()
            if tampered:
                admission_sha256 = (
                    "e" * 64 if admission_sha256 != "e" * 64 else "f" * 64
                )
            with admin.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO np.paper_fresh_opening_admissions (
                        control_key,
                        candidate_payload_sha256,
                        pin_authority_record_sha256,
                        deployment_incarnation_id,
                        admission_payload,
                        admission_payload_sha256
                    ) VALUES (TRUE, %s, %s, %s, %s, %s)
                    """,
                    (
                        admitted_candidate_sha256,
                        request.target.pin_authority_record_sha256,
                        request.target.deployment_incarnation_id,
                        admission_payload,
                        admission_sha256,
                    ),
                )
                cursor.execute(
                    sql.SQL("COMMENT ON ROLE {} IS %s").format(sql.Identifier(role)),
                    (
                        f"elvis-postgres-bootstrap:v2:{database_name}:"
                        f"opening:{admission_sha256}",
                    ),
                )
                cursor.execute("SELECT np.paper_terminal_catalog_fingerprint()")
                terminal_catalog_sha256 = cursor.fetchone()[0]
                cursor.execute(
                    "COMMENT ON SCHEMA np IS %s",
                    (
                        "elvis-postgres-bootstrap-schema:v2:"
                        f"{database_name}:{terminal_catalog_sha256}",
                    ),
                )
            database.terminal_catalog_sha256 = terminal_catalog_sha256

        database.admit = admit
        yield database
    finally:
        try:
            with admin.cursor() as cursor:
                for managed_role in reversed(created_roles):
                    cursor.execute(
                        sql.SQL("DROP OWNED BY {}").format(sql.Identifier(managed_role))
                    )
                    cursor.execute(
                        sql.SQL("DROP ROLE {}").format(sql.Identifier(managed_role))
                    )
        finally:
            admin.close()


def _request(
    database,
    *,
    nonce="0123456789abcdef" * 4,
    logical_target=None,
    approval_issued_at=None,
    approval_expires_at=None,
    collateral_amount=Decimal("1000.00"),
    margin_quantum=Decimal("0.01"),
):
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
        max_approval_lifetime_seconds=7200,
        anchors=(anchor,),
    )
    policy_document = encode_fresh_opening_trust_policy(policy)
    intent = FreshOpeningIntent(
        schema_version=1,
        purpose="ELVIS_V2_FRESH_PAPER_OPENING",
        trajectory="B",
        continuity="NO_V1_CONTINUITY",
        logical_target=logical_target or "paper-production-primary",
        execution_scope="paper:production",
        account_key="paper-v2-main",
        owner_generation=1,
        opening_codec="paper-account-opening",
        opening_version=1,
        collateral_asset="USDT",
        collateral_amount=collateral_amount,
        margin_quantum=margin_quantum,
        opening_policy=FreshOpeningPolicy.EXPLICIT_FRESH_SINGLE_COLLATERAL,
        operator_identity="operator-requester-01",
        approval_id="opening-approval-2026-08-14-001",
        approver_identity=anchor.approver_identity,
        approval_issued_at=(
            approval_issued_at
            if approval_issued_at is not None
            else database.database_time - timedelta(minutes=5)
        ),
        approval_expires_at=(
            approval_expires_at
            if approval_expires_at is not None
            else database.database_time + timedelta(hours=1)
        ),
        trust_policy_sha256=policy_document.sha256,
        trust_domain=policy.trust_domain,
        signer_key_id=anchor.signer_key_id,
        signer_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
        nonce=nonce,
    )
    intent_document = encode_fresh_opening_intent(intent)
    approval = DetachedFreshOpeningApproval(
        schema_version=1,
        intent_sha256=intent_document.sha256,
        signature=private_key.sign(fresh_opening_signing_bytes(intent_document)),
    )
    return FreshOpeningProvisioningRequest(
        intent=intent,
        approval=approval,
        trust_policy=policy,
        expected_trust_policy_sha256=policy_document.sha256,
        expected_signer_public_key_sha256=anchor.public_key_sha256,
        target=FreshOpeningPhysicalTarget(
            expected_database=database.database_name,
            expected_system_identifier=database.system_identifier,
            control_plane_role=database.control_plane_role,
            opening_anchor_role=database.role,
            deployment_incarnation_id="deployment-2026-08-14-001",
            terminal_catalog_sha256=database.terminal_catalog_sha256,
            pin_authority_record_sha256="c" * 64,
        ),
    )


def _admitted_request(database, **overrides):
    draft = _request(database, **overrides)
    database.admit(draft)
    return _request(database, **overrides)


def _service(factory):
    return FreshOpeningProvisioningService(
        PostgresFreshOpeningProvisioning(factory),
        _OpeningCodec(),
        _SignatureVerifier(),
    )


def _candidate(request):
    return derive_prospective_fresh_opening_candidate(
        request.intent,
        request.approval,
        request.trust_policy,
        opening_codec=_OpeningCodec(),
    )


def _durable_counts(database):
    connection = psycopg2.connect(database.admin_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM np.paper_fresh_opening_nonces),
                    (SELECT COUNT(*) FROM np.paper_fresh_opening_provisionings),
                    (SELECT COUNT(*) FROM np.paper_account_streams),
                    (SELECT COUNT(*) FROM np.paper_account_balances)
                """)
            return cursor.fetchone()
    finally:
        connection.close()


def _admission_count(database):
    connection = psycopg2.connect(database.admin_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM np.paper_fresh_opening_admissions")
            return cursor.fetchone()[0]
    finally:
        connection.close()


def _commit_parameters(candidate, documents):
    return [
        candidate.intent_document.payload,
        candidate.intent_document.sha256,
        candidate.approval_document.payload,
        candidate.approval_document.sha256,
        candidate.trust_policy_document.payload,
        candidate.trust_policy_document.sha256,
        candidate.candidate_document.payload,
        candidate.candidate_document.sha256,
        candidate.opening.opening_payload,
        candidate.opening.opening_payload_sha256,
        documents.opening.payload,
        documents.opening.sha256,
        documents.provisioning.payload,
        documents.provisioning.sha256,
    ]


def _canonical_payload(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rewritten_decimal_documents(candidate, decimal_text):
    intent = json.loads(candidate.intent_document.payload)
    intent["collateral_amount"] = decimal_text
    intent["margin_quantum"] = decimal_text
    intent_payload = _canonical_payload(intent)
    intent_sha256 = hashlib.sha256(
        _INTENT_SIGNING_PREFIX + intent_payload.encode("utf-8")
    ).hexdigest()

    approval = json.loads(candidate.approval_document.payload)
    approval["intent_sha256"] = intent_sha256
    approval_payload = _canonical_payload(approval)
    approval_sha256 = _payload_sha256(approval_payload)

    opening = json.loads(candidate.opening.opening_payload)
    opening["opening_balances"][0]["available"] = decimal_text
    opening["policy"]["margin_quantum"] = decimal_text
    opening_payload = _canonical_payload(opening)
    opening_sha256 = _payload_sha256(opening_payload)

    candidate_document = json.loads(candidate.candidate_document.payload)
    candidate_document["intent_sha256"] = intent_sha256
    candidate_document["approval_sha256"] = approval_sha256
    candidate_document["opening_payload_sha256"] = opening_sha256
    candidate_payload = _canonical_payload(candidate_document)
    candidate_sha256 = _payload_sha256(candidate_payload)
    return SimpleNamespace(
        intent_payload=intent_payload,
        intent_sha256=intent_sha256,
        approval_payload=approval_payload,
        approval_sha256=approval_sha256,
        opening_payload=opening_payload,
        opening_sha256=opening_sha256,
        candidate_payload=candidate_payload,
        candidate_sha256=candidate_sha256,
    )


def _rewritten_decimal_commit_parameters(
    request,
    candidate,
    evidence,
    rewritten,
):
    receipts = _receipt_documents(request, candidate, evidence)
    opening_receipt = json.loads(receipts.opening.payload)
    opening_receipt["opening_payload_sha256"] = rewritten.opening_sha256
    opening_receipt_payload = _canonical_payload(opening_receipt)
    opening_receipt_sha256 = _payload_sha256(opening_receipt_payload)

    provisioning_receipt = json.loads(receipts.provisioning.payload)
    provisioning_receipt.update(
        {
            "approval_sha256": rewritten.approval_sha256,
            "candidate_sha256": rewritten.candidate_sha256,
            "intent_sha256": rewritten.intent_sha256,
            "opening_payload_sha256": rewritten.opening_sha256,
            "opening_receipt_sha256": opening_receipt_sha256,
        }
    )
    provisioning_receipt_payload = _canonical_payload(provisioning_receipt)
    provisioning_receipt_sha256 = _payload_sha256(provisioning_receipt_payload)
    return [
        rewritten.intent_payload,
        rewritten.intent_sha256,
        rewritten.approval_payload,
        rewritten.approval_sha256,
        candidate.trust_policy_document.payload,
        candidate.trust_policy_document.sha256,
        rewritten.candidate_payload,
        rewritten.candidate_sha256,
        rewritten.opening_payload,
        rewritten.opening_sha256,
        opening_receipt_payload,
        opening_receipt_sha256,
        provisioning_receipt_payload,
        provisioning_receipt_sha256,
    ]


def test_opening_anchor_is_nologin_and_has_no_capabilities(
    fresh_opening_database,
):
    database = fresh_opening_database
    admin = psycopg2.connect(database.admin_dsn)
    try:
        with admin.cursor() as cursor:
            for function in _OPENING_FUNCTIONS:
                cursor.execute(
                    """
                    SELECT
                        has_function_privilege(%s, %s, 'EXECUTE'),
                        has_function_privilege(%s, %s, 'EXECUTE'),
                        EXISTS (
                            SELECT 1
                            FROM pg_proc procedure
                            CROSS JOIN LATERAL aclexplode(
                                COALESCE(
                                    procedure.proacl,
                                    acldefault('f', procedure.proowner)
                                )
                            ) function_acl
                            WHERE procedure.oid = %s::regprocedure
                              AND function_acl.grantee = 0
                              AND function_acl.privilege_type = 'EXECUTE'
                        )
                    """,
                    (
                        database.role,
                        function,
                        database.other_role,
                        function,
                        function,
                    ),
                )
                assert cursor.fetchone() == (False, False, False)
            cursor.execute(
                """
                SELECT
                    auth_role.rolcanlogin,
                    auth_role.rolsuper,
                    auth_role.rolinherit,
                    auth_role.rolcreaterole,
                    auth_role.rolcreatedb,
                    auth_role.rolreplication,
                    auth_role.rolbypassrls,
                    auth_role.rolconnlimit,
                    auth_role.rolpassword IS NULL,
                    role_view.rolconfig
                FROM pg_authid auth_role
                JOIN pg_roles role_view
                  ON role_view.oid = auth_role.oid
                WHERE auth_role.rolname = %s
                """,
                (database.role,),
            )
            assert cursor.fetchone() == (
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                -1,
                True,
                None,
            )
            cursor.execute("""
                SELECT FORMAT('%I.%I', namespace.nspname, relation.relname)
                FROM pg_class relation
                JOIN pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'np'
                  AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
                ORDER BY relation.relname
                """)
            relations = tuple(row[0] for row in cursor.fetchall())
            assert relations
            cursor.execute(
                """
                SELECT procedure.oid::regprocedure::TEXT
                FROM pg_proc procedure
                JOIN pg_namespace namespace
                  ON namespace.oid = procedure.pronamespace
                WHERE namespace.nspname = 'np'
                  AND has_function_privilege(%s, procedure.oid, 'EXECUTE')
                ORDER BY 1
                """,
                (database.role,),
            )
            assert cursor.fetchall() == []
            cursor.execute(
                """
                SELECT procedure.oid::regprocedure::TEXT
                FROM pg_proc procedure
                JOIN pg_namespace namespace
                  ON namespace.oid = procedure.pronamespace
                WHERE namespace.nspname = 'np'
                  AND has_function_privilege(%s, procedure.oid, 'EXECUTE')
                ORDER BY 1
                """,
                (database.other_role,),
            )
            assert cursor.fetchall() == []
            for relation in relations:
                for privilege in (
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "TRUNCATE",
                ):
                    cursor.execute(
                        "SELECT has_table_privilege(%s, %s, %s)",
                        (database.role, relation, privilege),
                    )
                    assert cursor.fetchone() == (False,)
                for privilege in ("SELECT", "INSERT", "UPDATE", "REFERENCES"):
                    cursor.execute(
                        "SELECT has_any_column_privilege(%s, %s, %s)",
                        (database.role, relation, privilege),
                    )
                    assert cursor.fetchone() == (False,)
    finally:
        admin.close()


def test_missing_admission_fails_closed_without_opening_rows(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _request(database)

    def factory():
        return psycopg2.connect(database.role_dsn)

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError) as caught:
        _service(factory).provision(request)

    assert str(caught.value) == (
        "PostgreSQL fresh-opening provisioning failed before commit"
    )
    assert caught.value.__cause__ is None
    assert _admission_count(database) == 0
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_tampered_admission_fails_closed_without_opening_rows(
    fresh_opening_database,
):
    database = fresh_opening_database
    draft = _request(database)
    database.admit(draft, tampered=True)
    request = _request(database)

    def factory():
        return psycopg2.connect(database.role_dsn)

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError) as caught:
        _service(factory).provision(request)

    assert str(caught.value) == (
        "PostgreSQL fresh-opening provisioning failed before commit"
    )
    assert caught.value.__cause__ is None
    assert _admission_count(database) == 1
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_direct_candidate_drift_is_admission_conflict_before_authority(
    fresh_opening_database,
):
    database = fresh_opening_database
    admitted = _admitted_request(database)
    request = _request(database, logical_target="paper-production-secondary")
    candidate = _candidate(request)
    assert candidate.candidate_document.sha256 != (
        _candidate(admitted).candidate_document.sha256
    )

    connection = psycopg2.connect(database.role_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                _ACQUIRE_FENCE_SQL,
                (
                    request.intent.trust_domain,
                    request.intent.signer_key_id,
                    request.intent.nonce,
                    candidate.candidate_document.sha256,
                ),
            )
            row = cursor.fetchone()
        connection.rollback()
    finally:
        connection.close()

    assert row[0] == "ADMISSION_CONFLICT"
    assert row[10:13] == (
        request.target.pin_authority_record_sha256,
        request.target.deployment_incarnation_id,
        None,
    )
    assert row[17] is True
    assert row[18:] == (None,) * 16

    authority = _BlockedAuthority(
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED
    )
    result = PostgresFreshOpeningProvisioning(
        lambda: psycopg2.connect(database.role_dsn)
    ).provision(request, candidate, authority)

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "TARGET_ADMISSION_BLOCKED"
    assert result.current_authority_evaluated is False
    assert authority.calls == []
    assert _admission_count(database) == 1
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_decimal_with_128_fractional_places_is_accepted_exactly(
    fresh_opening_database,
):
    database = fresh_opening_database
    boundary = Decimal("0.000001" + "0" * 122)
    assert boundary.as_tuple().exponent == -128
    request = _admitted_request(
        database,
        collateral_amount=boundary,
        margin_quantum=boundary,
    )

    result = _service(lambda: psycopg2.connect(database.role_dsn)).provision(request)

    assert result.disposition is FreshOpeningProvisioningDisposition.CREATED
    connection = psycopg2.connect(database.admin_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT available_decimal FROM np.paper_account_balances")
            assert cursor.fetchone() == (str(boundary),)
    finally:
        connection.close()


def test_decimal_with_129_significant_and_fractional_digits_is_rejected(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _request(database)
    candidate = _candidate(request)
    decimal_text = "0." + "1" * 129
    hostile = Decimal(decimal_text)
    assert len(hostile.as_tuple().digits) == 129
    assert hostile.as_tuple().exponent == -129
    rewritten = _rewritten_decimal_documents(candidate, decimal_text)
    database.admit(request, candidate_sha256=rewritten.candidate_sha256)

    request = _request(database)
    candidate = _candidate(request)
    rewritten = _rewritten_decimal_documents(candidate, decimal_text)
    connection = psycopg2.connect(database.role_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                _ACQUIRE_FENCE_SQL,
                (
                    request.intent.trust_domain,
                    request.intent.signer_key_id,
                    request.intent.nonce,
                    rewritten.candidate_sha256,
                ),
            )
            evidence = _fence_evidence(cursor.fetchone())
            assert evidence.resolution == "ABSENT"
            parameters = _rewritten_decimal_commit_parameters(
                request,
                candidate,
                evidence,
                rewritten,
            )
            with pytest.raises(psycopg2.Error) as caught:
                cursor.execute(_COMMIT_OPENING_SQL, parameters)
        assert caught.value.pgcode == "22023"
        assert "paper fresh opening balance is invalid" in str(caught.value)
    finally:
        connection.rollback()
        connection.close()

    assert _admission_count(database) == 1
    assert _durable_counts(database) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "tamper_sql",
    [
        (
            "CREATE TABLE public.external_paper_stream_child () "
            "INHERITS (np.paper_account_streams)"
        ),
        (
            "CREATE POLICY external_paper_stream_policy "
            "ON np.paper_account_streams USING (TRUE)"
        ),
    ],
    ids=("external-inherits-child", "rls-policy"),
)
def test_catalog_topology_or_policy_tamper_makes_acquire_fail_closed(
    fresh_opening_database,
    tamper_sql,
):
    database = fresh_opening_database
    request = _admitted_request(database)

    admin = psycopg2.connect(database.admin_dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(tamper_sql)
    finally:
        admin.close()

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError) as caught:
        _service(lambda: psycopg2.connect(database.role_dsn)).provision(request)

    assert str(caught.value) == (
        "PostgreSQL fresh-opening provisioning failed before commit"
    )
    assert caught.value.__cause__ is None
    assert _admission_count(database) == 1
    assert _durable_counts(database) == (0, 0, 0, 0)


@pytest.mark.parametrize("marker", ("schema", "role"))
def test_missing_terminal_marker_makes_acquire_fail_closed(
    fresh_opening_database,
    marker,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    admin = psycopg2.connect(database.admin_dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            if marker == "schema":
                cursor.execute("COMMENT ON SCHEMA np IS NULL")
            else:
                cursor.execute(
                    sql.SQL("COMMENT ON ROLE {} IS NULL").format(
                        sql.Identifier(database.role)
                    )
                )
    finally:
        admin.close()

    with pytest.raises(PostgresFreshOpeningProvisioningStorageError) as caught:
        _service(lambda: psycopg2.connect(database.role_dsn)).provision(request)

    assert str(caught.value) == (
        "PostgreSQL fresh-opening provisioning failed before commit"
    )
    assert caught.value.__cause__ is None
    assert _admission_count(database) == 1
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_created_replayed_and_conflicts_leave_authority_dormant(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)

    def factory():
        return psycopg2.connect(database.role_dsn)

    created = _service(factory).provision(request)
    replayed = _service(factory).provision(request)
    nonce_conflict_request = _request(
        database,
        logical_target="paper-production-secondary",
    )
    nonce_conflict_request = replace(
        nonce_conflict_request,
        target=replace(
            nonce_conflict_request.target,
            pin_authority_record_sha256="d" * 64,
        ),
    )
    target_conflict_request = _request(
        database,
        nonce="fedcba9876543210" * 4,
    )
    target_conflict_request = replace(
        target_conflict_request,
        target=replace(
            target_conflict_request.target,
            deployment_incarnation_id="other-deployment",
        ),
    )
    nonce_conflict = _service(factory).provision(nonce_conflict_request)
    target_conflict = _service(factory).provision(target_conflict_request)

    assert created.disposition is FreshOpeningProvisioningDisposition.CREATED
    assert replayed.disposition is FreshOpeningProvisioningDisposition.REPLAYED
    assert created.receipt == replayed.receipt
    assert replayed.current_authority_evaluated is False
    assert nonce_conflict.disposition is FreshOpeningProvisioningDisposition.CONFLICT
    assert nonce_conflict.primary_reason_code == "FRESH_OPENING_NONCE_CONFLICT"
    assert nonce_conflict.current_authority_evaluated is False
    assert target_conflict.disposition is FreshOpeningProvisioningDisposition.CONFLICT
    assert target_conflict.primary_reason_code == "FRESH_OPENING_TARGET_CONFLICT"
    assert target_conflict.current_authority_evaluated is False

    connection = psycopg2.connect(database.admin_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM np.paper_fresh_opening_nonces),
                    (SELECT COUNT(*) FROM np.paper_fresh_opening_provisionings),
                    (SELECT COUNT(*) FROM np.paper_account_streams),
                    (SELECT COUNT(*) FROM np.paper_account_balances),
                    (SELECT mode FROM np.paper_runtime_control WHERE control_key),
                    (SELECT runtime_generation FROM np.paper_runtime_control
                     WHERE control_key)
                """)
            assert cursor.fetchone() == (1, 1, 1, 1, "LEGACY", 0)
    finally:
        connection.close()


def test_provenance_and_opening_identity_are_immutable(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)

    def factory():
        return psycopg2.connect(database.role_dsn)

    assert (
        _service(factory).provision(request).disposition
        is FreshOpeningProvisioningDisposition.CREATED
    )
    assert _durable_counts(database) == (1, 1, 1, 1)

    connection = psycopg2.connect(database.admin_dsn)
    try:
        for relation, unchanged_column in (
            ("paper_fresh_opening_admissions", "admitted_at"),
            ("paper_fresh_opening_nonces", "registered_at"),
            ("paper_fresh_opening_provisionings", "committed_at"),
        ):
            statements = (
                f"UPDATE np.{relation} SET {unchanged_column} = {unchanged_column}",
                f"DELETE FROM np.{relation}",
                f"TRUNCATE np.{relation} CASCADE",
            )
            for statement in statements:
                with pytest.raises(psycopg2.Error) as caught:
                    with connection.cursor() as cursor:
                        cursor.execute(statement)
                assert caught.value.pgcode == "55000"
                assert "append-only" in str(caught.value)
                connection.rollback()

        with pytest.raises(psycopg2.Error) as caught:
            with connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE np.paper_account_streams
                    SET owner_generation = owner_generation + 1
                    """)
        assert caught.value.pgcode == "55000"
        assert "opening identity is immutable" in str(caught.value)
        connection.rollback()
    finally:
        connection.close()

    assert _durable_counts(database) == (1, 1, 1, 1)


def test_exact_replay_precedes_expiry_and_revocation_revalidation(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    candidate = _candidate(request)

    def factory():
        return psycopg2.connect(database.role_dsn)

    assert (
        _service(factory).provision(request).disposition
        is FreshOpeningProvisioningDisposition.CREATED
    )
    adapter = PostgresFreshOpeningProvisioning(factory)
    for blocked_disposition in (
        FreshOpeningPreparationDisposition.BLOCKED_APPROVAL_EXPIRED,
        FreshOpeningPreparationDisposition.BLOCKED_SIGNER_REVOKED,
    ):
        authority = _BlockedAuthority(blocked_disposition)
        replayed = adapter.provision(request, candidate, authority)
        assert replayed.disposition is FreshOpeningProvisioningDisposition.REPLAYED
        assert replayed.primary_reason_code == "EXACT_DURABLE_REPLAY"
        assert replayed.current_authority_evaluated is False
        assert authority.calls == []


def test_absent_expired_approval_is_blocked_without_writes(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(
        database,
        approval_issued_at=database.database_time - timedelta(hours=2),
        approval_expires_at=database.database_time - timedelta(hours=1),
    )

    def factory():
        return psycopg2.connect(database.role_dsn)

    result = _service(factory).provision(request)

    assert result.disposition is FreshOpeningProvisioningDisposition.BLOCKED
    assert result.primary_reason_code == "BLOCKED_APPROVAL_EXPIRED"
    assert result.current_authority_evaluated is True
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_identical_concurrent_attempts_create_once_and_replay_once(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    barrier = Barrier(2)

    def factory():
        return psycopg2.connect(database.role_dsn)

    def provision():
        barrier.wait(timeout=10)
        return _service(factory).provision(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: provision(), range(2)))

    assert sorted(result.disposition.value for result in results) == [
        "CREATED",
        "REPLAYED",
    ]
    assert sorted(result.current_authority_evaluated for result in results) == [
        False,
        True,
    ]
    assert _durable_counts(database) == (1, 1, 1, 1)


def test_tampered_payload_or_digest_rolls_back_every_opening_row(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    candidate = _candidate(request)

    for tampered_parameter, tampered_value in (
        (0, candidate.intent_document.payload + " "),
        (1, "0" * 64),
    ):
        connection = psycopg2.connect(database.role_dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    _ACQUIRE_FENCE_SQL,
                    (
                        request.intent.trust_domain,
                        request.intent.signer_key_id,
                        request.intent.nonce,
                        candidate.candidate_document.sha256,
                    ),
                )
                evidence = _fence_evidence(cursor.fetchone())
                documents = _receipt_documents(request, candidate, evidence)
                parameters = _commit_parameters(candidate, documents)
                parameters[tampered_parameter] = tampered_value
                with pytest.raises(psycopg2.Error) as caught:
                    cursor.execute(_COMMIT_OPENING_SQL, parameters)
            assert caught.value.pgcode == "22023"
        finally:
            connection.rollback()
            connection.close()
        assert _durable_counts(database) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("document_index", "field", "replacement"),
    (
        (10, "schema_version", None),
        (10, "schema_version", "1"),
        (10, "owner_generation", None),
        (12, "database_name", None),
        (12, "migration_head", "7"),
        (12, "runtime_generation", "0"),
        (12, "database_incarnation_id", None),
    ),
)
def test_direct_receipt_null_or_type_confusion_is_rejected_without_writes(
    fresh_opening_database,
    document_index,
    field,
    replacement,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    candidate = _candidate(request)
    connection = psycopg2.connect(database.role_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                _ACQUIRE_FENCE_SQL,
                (
                    request.intent.trust_domain,
                    request.intent.signer_key_id,
                    request.intent.nonce,
                    candidate.candidate_document.sha256,
                ),
            )
            evidence = _fence_evidence(cursor.fetchone())
            documents = _receipt_documents(request, candidate, evidence)
            parameters = _commit_parameters(candidate, documents)
            hostile_document = json.loads(parameters[document_index])
            hostile_document[field] = replacement
            hostile_payload = _canonical_payload(hostile_document)
            parameters[document_index] = hostile_payload
            parameters[document_index + 1] = _payload_sha256(hostile_payload)
            with pytest.raises(psycopg2.Error) as caught:
                cursor.execute(_COMMIT_OPENING_SQL, parameters)
        assert caught.value.pgcode == "22023"
    finally:
        connection.rollback()
        connection.close()
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_direct_all_null_business_receipt_is_rejected_without_writes(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    candidate = _candidate(request)
    connection = psycopg2.connect(database.role_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                _ACQUIRE_FENCE_SQL,
                (
                    request.intent.trust_domain,
                    request.intent.signer_key_id,
                    request.intent.nonce,
                    candidate.candidate_document.sha256,
                ),
            )
            evidence = _fence_evidence(cursor.fetchone())
            documents = _receipt_documents(request, candidate, evidence)
            parameters = _commit_parameters(candidate, documents)
            hostile_receipt = {
                key: None for key in json.loads(documents.opening.payload)
            }
            hostile_payload = _canonical_payload(hostile_receipt)
            parameters[10] = hostile_payload
            parameters[11] = _payload_sha256(hostile_payload)
            with pytest.raises(psycopg2.Error) as caught:
                cursor.execute(_COMMIT_OPENING_SQL, parameters)
        assert caught.value.pgcode == "22023"
    finally:
        connection.rollback()
        connection.close()
    assert _durable_counts(database) == (0, 0, 0, 0)


def test_durable_commit_with_lost_ack_resolves_by_exact_readback(
    fresh_opening_database,
):
    database = fresh_opening_database
    request = _admitted_request(database)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        connection = psycopg2.connect(database.role_dsn)
        if calls == 1:
            return _CommitAckLostConnection(connection)
        return connection

    result = _service(factory).provision(request)

    assert result.disposition is FreshOpeningProvisioningDisposition.CREATED
    assert result.primary_reason_code == "FRESH_OPENING_CREATED"
    assert result.receipt is not None
    assert result.side_effect_state == "COMMITTED"
    assert calls == 2
