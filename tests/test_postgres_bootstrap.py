"""Unit contract checks for the dormant PostgreSQL authority bootstrap."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest
from psycopg2.extensions import (
    STATUS_BEGIN,
    STATUS_READY,
    TRANSACTION_STATUS_IDLE,
    TRANSACTION_STATUS_INTRANS,
)

from trading.persistence.migration_runner import MigrationApplyError
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrap,
    PostgresBootstrapAdoption,
    PostgresBootstrapCommitUnknownError,
    PostgresBootstrapContext,
    PostgresBootstrapDriftError,
    PostgresBootstrapInputError,
    PostgresBootstrapMigrationError,
    PostgresBootstrapPhase,
    PostgresBootstrapReceipt,
    PostgresBootstrapRoles,
    PostgresBootstrapStatus,
    PostgresBootstrapStorageError,
)


def make_roles(prefix: str = "elvis_test") -> PostgresBootstrapRoles:
    return PostgresBootstrapRoles(
        schema_owner=f"{prefix}_owner",
        migrator=f"{prefix}_migrator",
        legacy_runtime=f"{prefix}_legacy",
        atomic_runtime=f"{prefix}_atomic",
        activation=f"{prefix}_activation",
        readiness=f"{prefix}_readiness",
        trainer=f"{prefix}_trainer",
    )


def make_adoption(**overrides) -> PostgresBootstrapAdoption:
    values = {
        "migration_authority_role": "elvis_history_owner",
        "allowed_historical_owner_roles": ("elvis_history_owner",),
    }
    values.update(overrides)
    return PostgresBootstrapAdoption(**values)


def make_context(**overrides) -> PostgresBootstrapContext:
    values = {
        "expected_database": "elvis_test_database",
        "admin_role": "elvis_admin",
        "roles": make_roles(),
    }
    values.update(overrides)
    return PostgresBootstrapContext(**values)


def make_connection() -> MagicMock:
    connection = MagicMock()
    connection.autocommit = False
    connection.status = STATUS_READY
    connection.get_transaction_status.return_value = TRANSACTION_STATUS_IDLE
    connection.cursor.return_value.__exit__.return_value = False
    return connection


def assert_secret_absent_from_exception_graph(
    error: BaseException, secret: str
) -> None:
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(current)
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__)
            if linked is not None
        )


def managed_role_rows(
    context: PostgresBootstrapContext,
    *,
    login_roles_enabled: bool = False,
):
    purposes = (
        "schema_owner",
        "migrator",
        "legacy_runtime",
        "atomic_runtime",
        "activation",
        "readiness",
        "trainer",
    )
    rows = []
    for purpose, role in zip(purposes, context.roles.all):
        rows.append(
            (
                role,
                login_roles_enabled and purpose != "schema_owner",
                False,
                False,
                False,
                False,
                False,
                False,
                -1,
                None,
                f"elvis-postgres-bootstrap:v1:{context.expected_database}:{purpose}",
            )
        )
    return tuple(sorted(rows))


def scripted_admin_connection(
    context: PostgresBootstrapContext,
    *,
    role_row_sets,
    memberships=(("elvis_test_owner", "elvis_test_migrator", False),),
    password_states=None,
) -> MagicMock:
    connection = make_connection()
    cursor = connection.cursor.return_value.__enter__.return_value
    rows = iter(role_row_sets)
    last_role_rows = ()

    def fetchone():
        command = str(cursor.execute.call_args.args[0])
        if "FROM pg_database database_row" in command:
            return (context.admin_role,)
        return (
            context.expected_database,
            context.admin_role,
            context.admin_role,
            context.admin_role,
            True,
            True,
        )

    def fetchall():
        nonlocal last_role_rows
        command = str(cursor.execute.call_args.args[0])
        if "FROM pg_roles role_row" in command:
            last_role_rows = next(rows)
            return last_role_rows
        if "FROM pg_authid role_row" in command:
            if password_states is not None:
                return password_states
            login_enabled = {
                row[1] for row in last_role_rows if row[0] != context.roles.schema_owner
            }
            assert len(login_enabled) == 1
            credentials_provisioned = next(iter(login_enabled))
            return tuple(
                sorted(
                    (
                        role,
                        role == context.roles.schema_owner
                        or not credentials_provisioned,
                        True,
                    )
                    for role in context.roles.all
                )
            )
        if "FROM pg_auth_members" in command:
            return memberships
        if "FROM pg_db_role_setting" in command:
            return ()
        raise AssertionError(f"unexpected scripted fetchall for {command}")

    cursor.fetchone.side_effect = fetchone
    cursor.fetchall.side_effect = fetchall
    return connection


def scripted_credential_connection(
    context: PostgresBootstrapContext,
    purpose: str,
    role: str,
) -> MagicMock:
    connection = make_connection()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (
        context.expected_database,
        role,
        role,
        role,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        -1,
        None,
        f"elvis-postgres-bootstrap:v1:{context.expected_database}:{purpose}",
    )
    return connection


def test_role_manifest_has_exact_ordered_authorities() -> None:
    roles = make_roles()

    assert roles.all == (
        "elvis_test_owner",
        "elvis_test_migrator",
        "elvis_test_legacy",
        "elvis_test_atomic",
        "elvis_test_activation",
        "elvis_test_readiness",
        "elvis_test_trainer",
    )
    assert roles.login_roles == roles.all[1:]


def test_role_markers_bind_cluster_global_names_to_database_and_purpose() -> None:
    roles = make_roles()
    first = make_context(expected_database="elvis_database_a", roles=roles)
    second = make_context(expected_database="elvis_database_b", roles=roles)

    assert PostgresBootstrap._role_marker(first, "atomic_runtime") == (
        "elvis-postgres-bootstrap:v1:elvis_database_a:atomic_runtime"
    )
    assert PostgresBootstrap._role_marker(second, "atomic_runtime") != (
        PostgresBootstrap._role_marker(first, "atomic_runtime")
    )
    assert PostgresBootstrap._role_marker(first, "readiness") != (
        PostgresBootstrap._role_marker(first, "atomic_runtime")
    )


@pytest.mark.parametrize(
    "invalid_role",
    [
        "",
        "Uppercase",
        "9starts_with_number",
        "contains-dash",
        "contains space",
        'quoted"role',
        "accentué",
        "a" * 64,
        None,
        True,
    ],
)
def test_role_manifest_rejects_unsafe_identifiers(invalid_role) -> None:
    values = {
        "schema_owner": "valid_owner",
        "migrator": "valid_migrator",
        "legacy_runtime": "valid_legacy",
        "atomic_runtime": "valid_atomic",
        "activation": "valid_activation",
        "readiness": "valid_readiness",
        "trainer": invalid_role,
    }

    with pytest.raises(PostgresBootstrapInputError, match="lowercase"):
        PostgresBootstrapRoles(**values)


def test_role_manifest_accepts_maximum_postgresql_identifier_length() -> None:
    roles = make_roles(prefix="a" * 52)

    assert all(len(role) <= 63 for role in roles.all)


def test_role_manifest_rejects_aliasing_between_authorities() -> None:
    roles = make_roles()

    with pytest.raises(PostgresBootstrapInputError, match="pairwise distinct"):
        PostgresBootstrapRoles(
            schema_owner=roles.schema_owner,
            migrator=roles.migrator,
            legacy_runtime=roles.legacy_runtime,
            atomic_runtime=roles.atomic_runtime,
            activation=roles.activation,
            readiness=roles.readiness,
            trainer=roles.readiness,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"allowed_historical_owner_roles": ()},
            "bound to the migration authority",
        ),
        (
            {
                "allowed_historical_owner_roles": (
                    "elvis_history_owner",
                    "elvis_history_owner",
                )
            },
            "bound to the migration authority",
        ),
        (
            {
                "migration_authority_role": "elvis_other_owner",
                "allowed_historical_owner_roles": ("elvis_history_owner",),
            },
            "bound to the migration authority",
        ),
        (
            {
                "allowed_historical_owner_roles": (
                    "elvis_history_owner",
                    "elvis_other_owner",
                )
            },
            "bound to the migration authority",
        ),
        ({"demote_old_shared_runtime": 1}, "must be a boolean"),
        ({"demote_old_shared_runtime": True}, "required before demotion"),
        (
            {"old_shared_runtime_role": "elvis_other_old"},
            "must be the migration authority",
        ),
    ],
)
def test_adoption_rejects_ambiguous_authority(overrides, message) -> None:
    with pytest.raises(PostgresBootstrapInputError, match=message):
        make_adoption(**overrides)


@pytest.mark.parametrize(
    "invalid_role",
    ["BadOwner", "owner;drop role", "owner space", "øwner", "a" * 64],
)
def test_adoption_rejects_unsafe_role_identifiers(invalid_role) -> None:
    with pytest.raises(PostgresBootstrapInputError, match="lowercase"):
        make_adoption(
            migration_authority_role=invalid_role,
            allowed_historical_owner_roles=(invalid_role,),
        )


@pytest.mark.parametrize("invalid_database", ["", "\x00", "elvis\x00test", None])
def test_context_rejects_invalid_database_identity(invalid_database) -> None:
    with pytest.raises(PostgresBootstrapInputError, match="expected_database"):
        make_context(expected_database=invalid_database)


def test_context_keeps_database_name_as_data_not_role_identifier() -> None:
    context = make_context(expected_database='elvis-test "database"')

    assert context.expected_database == 'elvis-test "database"'


def test_context_rejects_admin_aliasing_a_managed_role() -> None:
    roles = make_roles()

    with pytest.raises(
        PostgresBootstrapInputError, match="admin role must be distinct"
    ):
        make_context(admin_role=roles.activation, roles=roles)


def test_context_rejects_managed_role_as_historical_owner() -> None:
    roles = make_roles()
    adoption = make_adoption(
        migration_authority_role=roles.schema_owner,
        allowed_historical_owner_roles=(roles.schema_owner,),
    )

    with pytest.raises(PostgresBootstrapInputError, match="historical owners"):
        make_context(roles=roles, adoption=adoption)


def test_context_rejects_old_runtime_aliasing_managed_role() -> None:
    roles = make_roles()
    adoption = make_adoption(
        migration_authority_role=roles.legacy_runtime,
        allowed_historical_owner_roles=(roles.legacy_runtime,),
        old_shared_runtime_role=roles.legacy_runtime,
    )

    with pytest.raises(PostgresBootstrapInputError, match="historical owners"):
        make_context(roles=roles, adoption=adoption)


@pytest.mark.parametrize("demote", [False, True])
def test_context_rejects_old_runtime_aliasing_admin_identity(demote) -> None:
    adoption = make_adoption(
        migration_authority_role="elvis_admin",
        allowed_historical_owner_roles=("elvis_admin",),
        old_shared_runtime_role="elvis_admin",
        demote_old_shared_runtime=demote,
    )

    with pytest.raises(PostgresBootstrapInputError, match="differ.*admin identity"):
        make_context(adoption=adoption)


@pytest.mark.parametrize("invalid_factory", [None, False, 1, object()])
def test_bootstrap_requires_callable_admin_factory(invalid_factory) -> None:
    with pytest.raises(TypeError, match="admin_connection_factory"):
        PostgresBootstrap(invalid_factory)


def test_bootstrap_rejects_non_callable_optional_factory() -> None:
    with pytest.raises(TypeError, match="role connection factories"):
        PostgresBootstrap(lambda: object(), readiness_connection_factory=object())


def test_connection_factory_representations_cannot_leak_credentials() -> None:
    secret = "postgresql://operator:never-print-this@example.invalid/elvis"

    class SecretFactory:
        def __call__(self):
            raise AssertionError("the factory must not be called by repr")

        def __repr__(self) -> str:
            return secret

    factory = SecretFactory()
    bootstrap = PostgresBootstrap(
        factory,
        migrator_connection_factory=factory,
        legacy_runtime_connection_factory=factory,
        atomic_runtime_connection_factory=factory,
        activation_connection_factory=factory,
        readiness_connection_factory=factory,
        trainer_connection_factory=factory,
    )

    assert secret not in repr(bootstrap)
    assert secret not in repr(bootstrap._credential_factories)


def test_receipt_schema_is_exactly_secret_free() -> None:
    assert tuple(field.name for field in fields(PostgresBootstrapReceipt)) == (
        "status",
        "migration_versions",
        "verified_role_probes",
        "pending_role_credentials",
        "old_shared_runtime_demoted",
    )
    receipt = PostgresBootstrapReceipt(
        status=PostgresBootstrapStatus.CREDENTIALS_REQUIRED,
        migration_versions=(1, 2, 3, 4, 5, 6),
        verified_role_probes=(),
        pending_role_credentials=("elvis_test_migrator",),
        old_shared_runtime_demoted=False,
    )

    assert "password" not in repr(receipt).lower()
    assert "dsn" not in repr(receipt).lower()


def test_commit_unknown_error_reports_only_the_durable_phase() -> None:
    error = PostgresBootstrapCommitUnknownError(PostgresBootstrapPhase.CATALOG)

    assert error.phase is PostgresBootstrapPhase.CATALOG
    assert str(error) == "bootstrap catalog commit outcome is unknown"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[:3], "partially present"),
        (
            lambda rows: (rows[0][:2] + (True,) + rows[0][3:],) + rows[1:],
            "unsafe attributes",
        ),
        (
            lambda rows: (rows[0][:9] + (("search_path=np",),) + rows[0][10:],)
            + rows[1:],
            "role-level settings",
        ),
        (
            lambda rows: (rows[0][:-1] + ("wrong-purpose-marker",),) + rows[1:],
            "invalid marker",
        ),
    ],
)
def test_phase_a_rejects_partial_or_drifted_role_catalog(mutation, message) -> None:
    context = make_context()
    rows = managed_role_rows(context)
    connection = scripted_admin_connection(
        context,
        role_row_sets=(mutation(rows),),
    )

    with pytest.raises(PostgresBootstrapDriftError, match=message):
        PostgresBootstrap(lambda: connection)._reconcile_roles(context)

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()
    connection.close.assert_called_once_with()
    cursor = connection.cursor.return_value.__enter__.return_value
    executed = " ".join(str(call.args[0]) for call in cursor.execute.call_args_list)
    assert "CREATE ROLE" not in executed


@pytest.mark.parametrize(
    "memberships",
    [
        (),
        (("elvis_test_owner", "elvis_test_migrator", True),),
        (
            ("elvis_test_owner", "elvis_test_migrator", False),
            ("elvis_test_readiness", "elvis_admin", False),
        ),
    ],
)
def test_phase_a_rejects_membership_drift(memberships) -> None:
    context = make_context()
    connection = scripted_admin_connection(
        context,
        role_row_sets=(managed_role_rows(context),),
        memberships=memberships,
    )

    with pytest.raises(PostgresBootstrapDriftError, match="memberships"):
        PostgresBootstrap(lambda: connection)._reconcile_roles(context)

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_phase_a_rejects_partially_enabled_login_roles() -> None:
    context = make_context()
    dormant_rows = managed_role_rows(context)
    mixed_rows = tuple(
        row[:1] + (role == context.roles.migrator,) + row[2:]
        for row in dormant_rows
        for role in (row[0],)
    )
    connection = scripted_admin_connection(context, role_row_sets=(mixed_rows,))

    with pytest.raises(PostgresBootstrapDriftError, match="partially provisioned"):
        PostgresBootstrap(lambda: connection)._reconcile_roles(context)


@pytest.mark.parametrize("credential_drift", ["password_null", "expired"])
def test_phase_a_rejects_absent_or_expired_login_role_credentials(
    credential_drift,
) -> None:
    context = make_context()
    rows = managed_role_rows(context, login_roles_enabled=True)
    password_states = [
        (role, role == context.roles.schema_owner, True) for role in context.roles.all
    ]
    readiness_index = context.roles.all.index(context.roles.readiness)
    role, _password_null, _valid = password_states[readiness_index]
    password_states[readiness_index] = (
        role,
        credential_drift == "password_null",
        credential_drift != "expired",
    )
    connection = scripted_admin_connection(
        context,
        role_row_sets=(rows,),
        password_states=tuple(sorted(password_states)),
    )

    with pytest.raises(
        PostgresBootstrapDriftError,
        match="credentials.*absent|credentials.*expired|credentials.*unsafe",
    ):
        PostgresBootstrap(lambda: connection)._reconcile_roles(context)

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


@pytest.mark.parametrize("login_roles_enabled", [False, True])
def test_phase_a_exact_rerun_is_read_only_and_rolls_back(
    login_roles_enabled,
) -> None:
    context = make_context()
    connection = scripted_admin_connection(
        context,
        role_row_sets=(
            managed_role_rows(
                context,
                login_roles_enabled=login_roles_enabled,
            ),
        ),
    )

    PostgresBootstrap(lambda: connection)._reconcile_roles(context)

    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()
    cursor = connection.cursor.return_value.__enter__.return_value
    commands = tuple(str(call.args[0]) for call in cursor.execute.call_args_list)
    forbidden = ("CREATE ROLE", "COMMENT ON ROLE", "GRANT ", "ALTER ROLE")
    assert not any(token in command for token in forbidden for command in commands)


def test_phase_a_partial_creation_failure_rolls_back_the_whole_role_set() -> None:
    context = make_context()
    connection = scripted_admin_connection(context, role_row_sets=((),))
    cursor = connection.cursor.return_value.__enter__.return_value
    create_count = 0

    def fail_during_creation(command, *args):
        nonlocal create_count
        if "CREATE ROLE" in str(command):
            create_count += 1
            if create_count == 3:
                raise RuntimeError("simulated role DDL failure")

    cursor.execute.side_effect = fail_during_creation

    with pytest.raises(PostgresBootstrapStorageError, match="reconciliation failed"):
        PostgresBootstrap(lambda: connection)._reconcile_roles(context)

    assert create_count == 3
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_phase_a_commit_then_raise_is_resolved_by_exact_readback() -> None:
    context = make_context()
    rows = managed_role_rows(context)
    write = scripted_admin_connection(
        context,
        role_row_sets=((), rows),
    )
    write.commit.side_effect = RuntimeError("lost commit reply")
    readback = scripted_admin_connection(context, role_row_sets=(rows,))
    connections = iter((write, readback))

    PostgresBootstrap(lambda: next(connections))._reconcile_roles(context)

    write.commit.assert_called_once_with()
    write.rollback.assert_called_once_with()
    readback.rollback.assert_called_once_with()
    readback.close.assert_called_once_with()


def test_phase_a_commit_failure_without_durable_readback_is_unknown() -> None:
    context = make_context()
    rows = managed_role_rows(context)
    write = scripted_admin_connection(context, role_row_sets=((), rows))
    sentinel = "SENTINEL-ROLE-COMMIT-SECRET"
    write.commit.side_effect = RuntimeError(f"commit rejected with {sentinel}")
    readback = scripted_admin_connection(context, role_row_sets=((),))
    connections = iter((write, readback))

    with pytest.raises(PostgresBootstrapCommitUnknownError) as caught:
        PostgresBootstrap(lambda: next(connections))._reconcile_roles(context)

    assert caught.value.phase is PostgresBootstrapPhase.ROLES
    assert caught.value.__cause__ is None
    assert_secret_absent_from_exception_graph(caught.value, sentinel)


def test_connection_interface_inspection_exception_is_fully_redacted() -> None:
    sentinel = "SENTINEL-INTERFACE-SECRET"

    class ExplosiveInterface:
        def __getattribute__(self, name):
            if name == "cursor":
                raise RuntimeError(f"driver property leaked {sentinel}")
            return object.__getattribute__(self, name)

        def close(self):
            pass

    with pytest.raises(PostgresBootstrapStorageError) as caught:
        PostgresBootstrap(lambda: ExplosiveInterface())._reconcile_roles(make_context())

    assert_secret_absent_from_exception_graph(caught.value, sentinel)


@pytest.mark.parametrize(
    ("connection_kind", "message"),
    [
        ("invalid_interface", "invalid interface"),
        ("missing_status", "transaction status"),
        ("autocommit", "disable autocommit"),
        ("connection_status", "fresh and idle"),
        ("transaction_status", "fresh and idle"),
    ],
)
def test_phase_a_rejects_invalid_admin_connection_state(
    connection_kind, message
) -> None:
    if connection_kind == "invalid_interface":
        connection = object()
    elif connection_kind == "missing_status":
        connection = type(
            "MissingStatusConnection",
            (),
            {
                "autocommit": False,
                "cursor": lambda self: None,
                "commit": lambda self: None,
                "rollback": lambda self: None,
                "close": lambda self: None,
            },
        )()
    else:
        connection = make_connection()
        if connection_kind == "autocommit":
            connection.autocommit = True
        elif connection_kind == "connection_status":
            connection.status = STATUS_BEGIN
        else:
            connection.get_transaction_status.return_value = TRANSACTION_STATUS_INTRANS

    with pytest.raises(PostgresBootstrapStorageError, match=message):
        PostgresBootstrap(lambda: connection)._reconcile_roles(make_context())


def test_mixed_factories_preserve_manifest_probe_and_pending_order() -> None:
    context = make_context()
    rows = managed_role_rows(context, login_roles_enabled=True)
    admin = scripted_admin_connection(context, role_row_sets=(rows,))
    migrator = scripted_credential_connection(
        context, "migrator", context.roles.migrator
    )
    activation = scripted_credential_connection(
        context, "activation", context.roles.activation
    )
    trainer = scripted_credential_connection(context, "trainer", context.roles.trainer)
    bootstrap = PostgresBootstrap(
        lambda: admin,
        migrator_connection_factory=lambda: migrator,
        activation_connection_factory=lambda: activation,
        trainer_connection_factory=lambda: trainer,
    )

    receipt = bootstrap.reconcile(context)

    assert receipt.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    assert receipt.verified_role_probes == (
        context.roles.migrator,
        context.roles.activation,
        context.roles.trainer,
    )
    assert receipt.pending_role_credentials == (
        context.roles.legacy_runtime,
        context.roles.atomic_runtime,
        context.roles.readiness,
    )


def test_credential_factory_wrong_identity_fails_closed() -> None:
    context = make_context()
    rows = managed_role_rows(context, login_roles_enabled=True)
    admin = scripted_admin_connection(context, role_row_sets=(rows,))
    wrong_identity = scripted_credential_connection(
        context, "migrator", context.roles.migrator
    )
    credential_cursor = wrong_identity.cursor.return_value.__enter__.return_value
    evidence = list(credential_cursor.fetchone.return_value)
    evidence[1] = context.admin_role
    evidence[2] = context.admin_role
    credential_cursor.fetchone.return_value = tuple(evidence)

    with pytest.raises(PostgresBootstrapDriftError, match="another identity"):
        PostgresBootstrap(
            lambda: admin,
            migrator_connection_factory=lambda: wrong_identity,
        ).reconcile(context)

    wrong_identity.rollback.assert_called_once_with()
    wrong_identity.close.assert_called_once_with()


def test_credential_factory_exception_is_redacted_without_a_cause() -> None:
    context = make_context()
    rows = managed_role_rows(context, login_roles_enabled=True)
    admin = scripted_admin_connection(context, role_row_sets=(rows,))
    sentinel = "SENTINEL-ROLE-PASSWORD"

    def leaking_factory():
        raise RuntimeError(f"could not connect with {sentinel}")

    with pytest.raises(PostgresBootstrapStorageError) as caught:
        PostgresBootstrap(
            lambda: admin,
            migrator_connection_factory=leaking_factory,
        ).reconcile(context)

    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_credential_probe_query_exception_drops_secret_bearing_context() -> None:
    context = make_context()
    connection = make_connection()
    sentinel = "SENTINEL-QUERY-SECRET"
    connection.cursor.return_value.__enter__.return_value.execute.side_effect = (
        RuntimeError(f"query failed with {sentinel}")
    )

    with pytest.raises(PostgresBootstrapStorageError) as caught:
        PostgresBootstrap._probe_credential(
            context,
            "migrator",
            context.roles.migrator,
            lambda: connection,
        )

    assert sentinel not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_pending_credentials_stop_before_migrations_and_catalog() -> None:
    bootstrap = PostgresBootstrap(lambda: object())
    context = make_context()

    with (
        patch.object(bootstrap, "_preflight_database"),
        patch.object(bootstrap, "_reconcile_roles") as roles,
        patch.object(
            bootstrap,
            "_probe_credentials",
            return_value=((context.roles.migrator,), context.roles.login_roles[1:]),
        ),
        patch.object(bootstrap, "_reconcile_migrations") as migrations,
        patch.object(bootstrap, "_reconcile_catalog", create=True) as catalog,
    ):
        receipt = bootstrap.reconcile(context)

    roles.assert_called_once_with(context)
    migrations.assert_not_called()
    catalog.assert_not_called()
    assert receipt.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    assert receipt.migration_versions == ()


def test_exact_terminal_readback_skips_migrations_and_catalog_writes() -> None:
    bootstrap = PostgresBootstrap(lambda: object())
    context = make_context()

    with (
        patch.object(bootstrap, "_preflight_database"),
        patch.object(bootstrap, "_reconcile_roles"),
        patch.object(
            bootstrap,
            "_probe_credentials",
            return_value=(context.roles.login_roles, ()),
        ),
        patch.object(
            bootstrap,
            "_catalog_readback_is_exact",
            return_value=True,
        ) as readback,
        patch.object(bootstrap, "_reconcile_migrations") as migrations,
        patch.object(bootstrap, "_reconcile_catalog") as catalog,
    ):
        receipt = bootstrap.reconcile(context)

    readback.assert_called_once_with(context)
    migrations.assert_not_called()
    catalog.assert_not_called()
    assert receipt.status is PostgresBootstrapStatus.COMPLETE
    assert receipt.migration_versions == (1, 2, 3, 4, 5, 6)
    assert receipt.old_shared_runtime_demoted is False


def test_existing_adoption_without_demotion_stops_before_catalog_cutover() -> None:
    bootstrap = PostgresBootstrap(lambda: object())
    adoption = make_adoption(
        migration_authority_role="elvis_old_runtime",
        allowed_historical_owner_roles=("elvis_old_runtime",),
        old_shared_runtime_role="elvis_old_runtime",
    )
    context = make_context(adoption=adoption)

    with (
        patch.object(bootstrap, "_preflight_database"),
        patch.object(bootstrap, "_reconcile_roles"),
        patch.object(
            bootstrap,
            "_probe_credentials",
            return_value=(context.roles.login_roles, ()),
        ),
        patch.object(
            bootstrap,
            "_catalog_readback_is_exact",
            return_value=False,
        ),
        patch.object(
            bootstrap,
            "_require_managed_roles_exact",
        ) as role_recheck,
        patch.object(
            bootstrap,
            "_reconcile_migrations",
            return_value=(1, 2, 3, 4, 5, 6),
        ) as migrations,
        patch.object(
            bootstrap,
            "_preflight_old_login_demotion",
        ) as demotion_preflight,
        patch.object(bootstrap, "_reconcile_catalog", create=True) as catalog,
    ):
        receipt = bootstrap.reconcile(context)

    migrations.assert_called_once_with(context)
    role_recheck.assert_called_once_with(context)
    demotion_preflight.assert_called_once_with(context)
    catalog.assert_not_called()
    assert receipt.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert receipt.old_shared_runtime_demoted is False


def test_migration_authority_commit_then_raise_uses_exact_readback() -> None:
    context = make_context()
    connection = make_connection()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (
        context.expected_database,
        context.admin_role,
        context.admin_role,
        context.admin_role,
        True,
        True,
    )
    connection.commit.side_effect = RuntimeError("lost commit reply")
    bootstrap = PostgresBootstrap(lambda: connection)

    with patch.object(
        bootstrap,
        "_fresh_migration_authority_is_exact",
        return_value=True,
    ) as readback:
        bootstrap._prepare_fresh_migration_authority(context)

    readback.assert_called_once_with(context)
    connection.rollback.assert_called_once_with()


def test_migration_authority_commit_failure_without_readback_is_unknown() -> None:
    context = make_context()
    connection = make_connection()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (
        context.expected_database,
        context.admin_role,
        context.admin_role,
        context.admin_role,
        True,
        True,
    )
    connection.commit.side_effect = RuntimeError("commit rejected")
    bootstrap = PostgresBootstrap(lambda: connection)

    with (
        patch.object(
            bootstrap,
            "_fresh_migration_authority_is_exact",
            return_value=False,
        ),
        pytest.raises(PostgresBootstrapCommitUnknownError) as caught,
    ):
        bootstrap._prepare_fresh_migration_authority(context)

    assert caught.value.phase is PostgresBootstrapPhase.MIGRATIONS


def test_migrator_set_role_failure_is_a_typed_migration_error() -> None:
    connection = make_connection()
    sentinel = "SENTINEL-SET-ROLE-SECRET"
    connection.cursor.return_value.__enter__.return_value.execute.side_effect = (
        RuntimeError(f"SET ROLE denied with {sentinel}")
    )

    with pytest.raises(PostgresBootstrapMigrationError, match="assume") as caught:
        PostgresBootstrap._set_migration_role(connection, "elvis_test_owner")

    assert caught.value.__cause__ is None
    assert_secret_absent_from_exception_graph(caught.value, sentinel)
    connection.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("ledger_readback", "expected_error"),
    [
        (False, PostgresBootstrapMigrationError),
        (None, PostgresBootstrapCommitUnknownError),
    ],
)
def test_migration_apply_failure_redacts_recursive_exception_graph(
    ledger_readback,
    expected_error,
) -> None:
    context = make_context()
    connection = make_connection()
    sentinel = "SENTINEL-MIGRATION-APPLY-SECRET"
    bootstrap = PostgresBootstrap(
        lambda: make_connection(),
        migrator_connection_factory=lambda: connection,
    )

    with (
        patch.object(
            bootstrap,
            "_require_migrator_connection_identity",
        ),
        patch.object(bootstrap, "_set_migration_role"),
        patch.object(
            bootstrap,
            "_migration_ledger_readback",
            return_value=ledger_readback,
        ),
        patch(
            "trading.persistence.postgres_bootstrap.apply_migrations",
            side_effect=MigrationApplyError(
                f"migration driver failure contained {sentinel}"
            ),
        ),
        pytest.raises(expected_error) as caught,
    ):
        bootstrap._apply_packaged_migrations(context)

    if ledger_readback is None:
        assert caught.value.phase is PostgresBootstrapPhase.MIGRATIONS
    assert_secret_absent_from_exception_graph(caught.value, sentinel)
