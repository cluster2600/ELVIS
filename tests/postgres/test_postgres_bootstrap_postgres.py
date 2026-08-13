"""PostgreSQL 15 authority and recovery checks for ``PostgresBootstrap``."""

from dataclasses import dataclass
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

from trading.persistence.migration_runner import apply_migrations, load_migrations
from trading.persistence.postgres_bootstrap import (
    PostgresBootstrap,
    PostgresBootstrapAdoption,
    PostgresBootstrapCommitUnknownError,
    PostgresBootstrapContext,
    PostgresBootstrapDriftError,
    PostgresBootstrapMigrationError,
    PostgresBootstrapPhase,
    PostgresBootstrapRoles,
    PostgresBootstrapStatus,
    PostgresBootstrapStorageError,
)

_AUTHORITY_TABLES = (
    "account_balances",
    "liquidations",
    "margin_history",
    "model_predictions",
    "open_positions",
    "order_events",
    "orders",
    "paper_account_balances",
    "paper_account_batch_manifests",
    "paper_account_postings",
    "paper_account_settlements",
    "paper_account_streams",
    "paper_margin_reservations",
    "paper_runtime_control",
    "paper_runtime_generations",
    "position_streams",
    "schema_migrations",
    "trades",
    "trading_session_resets",
)
_LEGACY_SEQUENCES = (
    "account_balances_id_seq",
    "liquidations_id_seq",
    "margin_history_id_seq",
    "model_predictions_id_seq",
    "open_positions_id_seq",
    "trades_id_seq",
    "trading_session_resets_id_seq",
)
_LEGACY_PRIVILEGES = {
    "account_balances": ("SELECT", "INSERT", "UPDATE"),
    "liquidations": ("SELECT", "INSERT"),
    "margin_history": ("SELECT", "INSERT"),
    "model_predictions": ("SELECT", "INSERT", "UPDATE"),
    "open_positions": ("SELECT", "INSERT", "DELETE"),
    "trades": ("SELECT", "INSERT", "DELETE"),
    "trading_session_resets": ("SELECT", "INSERT"),
}
_ATOMIC_PRIVILEGES = {
    "order_events": ("SELECT", "INSERT"),
    "orders": ("SELECT", "INSERT", "UPDATE"),
    "paper_account_balances": ("SELECT", "INSERT", "UPDATE"),
    "paper_account_batch_manifests": ("SELECT", "INSERT"),
    "paper_account_postings": ("SELECT", "INSERT"),
    "paper_account_settlements": ("SELECT", "INSERT"),
    "paper_account_streams": ("SELECT", "INSERT", "UPDATE"),
    "paper_margin_reservations": ("SELECT", "INSERT", "DELETE"),
    "position_streams": ("SELECT", "INSERT", "UPDATE"),
}
_PACKAGED_INDEXES = (
    "idx_model_predictions_scored",
    "idx_trades_symbol_ts",
    "order_events_fill_identity_uq",
    "order_events_order_replay_idx",
    "order_events_paper_account_fill_ref_uq",
    "order_events_paper_account_submission_ref_uq",
    "orders_paper_account_batch_ref_uq",
    "orders_paper_account_symbol_ref_uq",
    "orders_venue_identity_uq",
)


@dataclass(frozen=True)
class BootstrapCluster:
    admin_dsn: str
    database: str
    admin_role: str
    roles: PostgresBootstrapRoles
    old_runtime: str
    outsider: str
    passwords: dict[str, str]
    role_dsns: dict[str, str]

    def admin_factory(self):
        return psycopg2.connect(self.admin_dsn)

    def role_factory(self, role: str):
        dsn = self.role_dsns[role]
        return lambda: psycopg2.connect(dsn)

    def role_factory_with_options(self, role: str, options: str):
        parameters = parse_dsn(self.role_dsns[role])
        parameters["options"] = options
        dsn = make_dsn(**parameters)
        return dsn, lambda: psycopg2.connect(dsn)

    def bootstrap(
        self,
        *,
        with_credentials: bool,
        admin_factory=None,
    ) -> PostgresBootstrap:
        factories = {}
        if with_credentials:
            factories = {
                "migrator_connection_factory": self.role_factory(self.roles.migrator),
                "legacy_runtime_connection_factory": self.role_factory(
                    self.roles.legacy_runtime
                ),
                "atomic_runtime_connection_factory": self.role_factory(
                    self.roles.atomic_runtime
                ),
                "activation_connection_factory": self.role_factory(
                    self.roles.activation
                ),
                "readiness_connection_factory": self.role_factory(self.roles.readiness),
                "trainer_connection_factory": self.role_factory(self.roles.trainer),
            }
        return PostgresBootstrap(admin_factory or self.admin_factory, **factories)

    def context(
        self,
        *,
        adoption: PostgresBootstrapAdoption | None = None,
    ) -> PostgresBootstrapContext:
        return PostgresBootstrapContext(
            expected_database=self.database,
            admin_role=self.admin_role,
            roles=self.roles,
            adoption=adoption,
        )

    def provision_managed_passwords(self) -> None:
        connection = self.admin_factory()
        try:
            with connection.cursor() as cursor:
                for role in self.roles.login_roles:
                    cursor.execute(
                        sql.SQL("ALTER ROLE {} LOGIN PASSWORD %s").format(
                            sql.Identifier(role)
                        ),
                        (self.passwords[role],),
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_auxiliary_login(self, role: str) -> None:
        connection = self.admin_factory()
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD %s").format(
                        sql.Identifier(role)
                    ),
                    (self.passwords[role],),
                )
        finally:
            connection.close()


class CommitThenRaiseConnection:
    """Delegate connection whose first commit is durable but reports failure."""

    def __init__(self, connection, shared_state):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_shared_state", shared_state)

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __setattr__(self, name, value):
        setattr(self._connection, name, value)

    def commit(self):
        self._connection.commit()
        if not self._shared_state["raised"]:
            self._shared_state["raised"] = True
            raise psycopg2.OperationalError("simulated lost migration commit reply")


class NthCommitFaultConnection:
    """Inject one durable or non-durable failure at a shared commit ordinal."""

    def __init__(self, connection, shared_state, *, target, durable):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_shared_state", shared_state)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_durable", durable)

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __setattr__(self, name, value):
        setattr(self._connection, name, value)

    def commit(self):
        self._shared_state["commits"] += 1
        if self._shared_state["commits"] != self._target:
            return self._connection.commit()
        self._shared_state["fired"] = True
        if self._durable:
            self._connection.commit()
        raise psycopg2.OperationalError("simulated bootstrap commit reply loss")


class ReadbackMutationConnection:
    """Mutate catalog state immediately before the first readback query."""

    def __init__(self, connection, mutation, shared_state):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_mutation", mutation)
        object.__setattr__(self, "_shared_state", shared_state)

    def __getattr__(self, name):
        if name != "cursor":
            return getattr(self._connection, name)

        def cursor(*args, **kwargs):
            wrapped = self._connection.cursor(*args, **kwargs)
            if not self._shared_state["mutated"]:
                self._shared_state["mutated"] = True
                self._mutation()
            return wrapped

        return cursor

    def __setattr__(self, name, value):
        setattr(self._connection, name, value)


def _dsn_identity(dsn: str):
    return frozenset(parse_dsn(dsn).items())


def _database_identity(dsn: str) -> tuple[str, str]:
    parameters = parse_dsn(dsn)
    return parameters["dbname"], parameters["user"]


@pytest.fixture
def bootstrap_cluster(postgres_database_dsn, postgres_connection_allowlist):
    database, admin_role = _database_identity(postgres_database_dsn)
    suffix = uuid4().hex[:12]
    prefix = f"eb_{suffix}"
    roles = PostgresBootstrapRoles(
        schema_owner=f"{prefix}_owner",
        migrator=f"{prefix}_migrator",
        legacy_runtime=f"{prefix}_legacy",
        atomic_runtime=f"{prefix}_atomic",
        activation=f"{prefix}_activation",
        readiness=f"{prefix}_readiness",
        trainer=f"{prefix}_trainer",
    )
    old_runtime = f"{prefix}_old"
    outsider = f"{prefix}_outsider"
    login_test_roles = roles.login_roles + (old_runtime, outsider)
    passwords = {
        role: f"test-only-{suffix}-{index}"
        for index, role in enumerate(login_test_roles, start=1)
    }
    base_parameters = parse_dsn(postgres_database_dsn)
    role_dsns = {}
    for role in login_test_roles:
        parameters = dict(base_parameters)
        parameters.update(user=role, password=passwords[role])
        role_dsn = make_dsn(**parameters)
        role_dsns[role] = role_dsn
        postgres_connection_allowlist.add(_dsn_identity(role_dsn))

    cluster = BootstrapCluster(
        admin_dsn=postgres_database_dsn,
        database=database,
        admin_role=admin_role,
        roles=roles,
        old_runtime=old_runtime,
        outsider=outsider,
        passwords=passwords,
        role_dsns=role_dsns,
    )
    try:
        yield cluster
    finally:
        connection = cluster.admin_factory()
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                        sql.Identifier(database), sql.Identifier(admin_role)
                    )
                )
                for role in reversed(roles.all + (old_runtime, outsider)):
                    cursor.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)",
                        (role,),
                    )
                    if not cursor.fetchone()[0]:
                        continue
                    cursor.execute(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE usename = %s AND pid <> pg_backend_pid()",
                        (role,),
                    )
                    cursor.execute(
                        sql.SQL("REASSIGN OWNED BY {} TO {}").format(
                            sql.Identifier(role), sql.Identifier(admin_role)
                        )
                    )
                    cursor.execute(
                        sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role))
                    )
                    cursor.execute(
                        sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role))
                    )
        finally:
            connection.close()
            for role_dsn in role_dsns.values():
                postgres_connection_allowlist.discard(_dsn_identity(role_dsn))


def _role_rows(cluster: BootstrapCluster):
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT rolname, rolcanlogin, rolcreaterole, rolcreatedb,
                       rolsuper, rolreplication, rolbypassrls, rolconnlimit,
                       shobj_description(oid, 'pg_authid')
                FROM pg_roles
                WHERE rolname = ANY(%s)
                ORDER BY rolname
                """,
                (list(cluster.roles.all),),
            )
            return tuple(cursor.fetchall())
    finally:
        connection.close()


def _logical_role(cluster: BootstrapCluster, role: str) -> str:
    return {
        cluster.roles.schema_owner: "schema_owner",
        cluster.roles.migrator: "migrator",
        cluster.roles.legacy_runtime: "legacy_runtime",
        cluster.roles.atomic_runtime: "atomic_runtime",
        cluster.roles.activation: "activation",
        cluster.roles.readiness: "readiness",
        cluster.roles.trainer: "trainer",
    }[role]


def _complete_fresh_bootstrap(cluster: BootstrapCluster):
    first = cluster.bootstrap(with_credentials=False).reconcile(cluster.context())
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()
    return cluster.bootstrap(with_credentials=True).reconcile(cluster.context())


def _stage_existing_adoption(cluster: BootstrapCluster):
    cluster.create_auxiliary_login(cluster.old_runtime)
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.old_runtime),
                )
            )
    finally:
        admin.close()
    old_connection = cluster.role_factory(cluster.old_runtime)()
    old_connection.autocommit = False
    try:
        assert apply_migrations(old_connection, load_migrations()) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
    finally:
        old_connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=False,
    )
    context = cluster.context(adoption=adoption)
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(context).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()
    assert (
        cluster.bootstrap(with_credentials=True).reconcile(context).status
        is PostgresBootstrapStatus.DEMOTION_REQUIRED
    )
    return adoption


def _expected_table_grants(cluster: BootstrapCluster):
    grants = {
        (table, cluster.roles.legacy_runtime, privilege)
        for table, privileges in _LEGACY_PRIVILEGES.items()
        for privilege in privileges
    }
    grants.update(
        (table, cluster.roles.atomic_runtime, privilege)
        for table, privileges in _ATOMIC_PRIVILEGES.items()
        for privilege in privileges
    )
    grants.update(
        (table, cluster.roles.atomic_runtime, "SELECT")
        for table in ("paper_runtime_control", "paper_runtime_generations")
    )
    grants.update(
        (table, cluster.roles.readiness, "SELECT") for table in _AUTHORITY_TABLES
    )
    grants.add(("trades", cluster.roles.trainer, "SELECT"))
    grants.update(
        (table, cluster.roles.activation, privilege)
        for table in _AUTHORITY_TABLES
        for privilege in ("SELECT", "UPDATE")
    )
    grants.add(("paper_runtime_generations", cluster.roles.activation, "INSERT"))
    return grants


def _authority_snapshot(cluster: BootstrapCluster):
    connection = cluster.admin_factory()
    try:
        snapshots = []
        statements = (
            """
            SELECT rolname, rolcanlogin, rolsuper, rolinherit, rolcreaterole,
                   rolcreatedb, rolreplication, rolbypassrls, rolconnlimit,
                   rolconfig, shobj_description(oid, 'pg_authid')
            FROM pg_roles
            WHERE rolname = ANY(%s)
            ORDER BY rolname
            """,
            """
            SELECT relname, relkind, pg_get_userbyid(relowner), relacl::text
            FROM pg_class
            WHERE relnamespace = 'np'::regnamespace
            ORDER BY relkind, relname
            """,
            """
            SELECT proname, pg_get_function_identity_arguments(oid),
                   pg_get_userbyid(proowner), proacl::text
            FROM pg_proc
            WHERE pronamespace = 'np'::regnamespace
            ORDER BY proname, oid
            """,
            """
            SELECT pg_get_userbyid(nspowner), nspacl::text,
                   obj_description(oid, 'pg_namespace')
            FROM pg_namespace
            WHERE nspname = 'np'
            """,
            """
            SELECT pg_get_userbyid(datdba), datacl::text
            FROM pg_database
            WHERE datname = %s
            """,
            """
            SELECT pg_get_userbyid(defaclrole), defaclobjtype, defaclacl::text
            FROM pg_default_acl
            WHERE defaclnamespace = 'np'::regnamespace
            ORDER BY 1, 2, 3
            """,
        )
        with connection.cursor() as cursor:
            for index, statement in enumerate(statements):
                if index == 0:
                    cursor.execute(statement, (list(cluster.roles.all),))
                elif index == 4:
                    cursor.execute(statement, (cluster.database,))
                else:
                    cursor.execute(statement)
                snapshots.append(tuple(cursor.fetchall()))
        connection.rollback()
        return tuple(snapshots)
    finally:
        connection.close()


def _index_snapshot(cluster: BootstrapCluster):
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT index_class.relname,
                       table_class.relname,
                       access_method.amname,
                       index_row.indisunique,
                       index_row.indisvalid,
                       index_row.indisready,
                       index_row.indkey::text,
                       index_row.indnkeyatts,
                       index_row.indnatts,
                       pg_get_expr(index_row.indpred, index_row.indrelid),
                       pg_get_expr(index_row.indexprs, index_row.indrelid),
                       index_class.relpersistence,
                       pg_get_userbyid(index_class.relowner)
                FROM pg_index index_row
                JOIN pg_class index_class
                  ON index_class.oid = index_row.indexrelid
                JOIN pg_class table_class
                  ON table_class.oid = index_row.indrelid
                JOIN pg_am access_method
                  ON access_method.oid = index_class.relam
                WHERE index_class.relnamespace = 'np'::regnamespace
                  AND index_class.relname = ANY(%s)
                ORDER BY index_class.relname
                """,
                (list(_PACKAGED_INDEXES),),
            )
            return tuple(cursor.fetchall())
    finally:
        connection.close()


def _trades_sequence_evidence(cluster: BootstrapCluster):
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT sequence_catalog.seqincrement,
                       sequence_catalog.seqcycle,
                       COALESCE(owner_table.relname, ''),
                       COALESCE(owner_column.attname, '')
                FROM pg_class sequence_row
                JOIN pg_sequence sequence_catalog
                  ON sequence_catalog.seqrelid = sequence_row.oid
                LEFT JOIN pg_depend ownership
                  ON ownership.classid = 'pg_class'::regclass
                 AND ownership.objid = sequence_row.oid
                 AND ownership.deptype = 'a'
                LEFT JOIN pg_class owner_table
                  ON owner_table.oid = ownership.refobjid
                LEFT JOIN pg_attribute owner_column
                  ON owner_column.attrelid = ownership.refobjid
                 AND owner_column.attnum = ownership.refobjsubid
                WHERE sequence_row.oid = 'np.trades_id_seq'::regclass
                """)
            return cursor.fetchone()
    finally:
        connection.close()


def _public_schema_evidence(cluster: BootstrapCluster):
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT pg_get_userbyid(namespace_row.nspowner),
                       COALESCE(grantee_role.rolname, 'PUBLIC'),
                       schema_acl.privilege_type,
                       schema_acl.is_grantable
                FROM pg_namespace namespace_row
                CROSS JOIN LATERAL aclexplode(
                    COALESCE(
                        namespace_row.nspacl,
                        acldefault('n', namespace_row.nspowner)
                    )
                ) schema_acl
                LEFT JOIN pg_roles grantee_role
                  ON grantee_role.oid = schema_acl.grantee
                WHERE namespace_row.nspname = 'public'
                ORDER BY 2, 3
                """)
            return tuple(cursor.fetchall())
    finally:
        connection.close()


def _outside_schema_snapshot(cluster: BootstrapCluster, schema: str):
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner), nspacl::text "
                "FROM pg_namespace WHERE nspname = %s",
                (schema,),
            )
            namespace = cursor.fetchone()
            cursor.execute(
                "SELECT proname, pg_get_function_identity_arguments(oid), "
                "pg_get_userbyid(proowner), prosecdef, proacl::text, "
                "proconfig, prosrc "
                "FROM pg_proc "
                "WHERE pronamespace = %s::regnamespace "
                "ORDER BY proname, oid",
                (schema,),
            )
            functions = tuple(cursor.fetchall())
            cursor.execute(
                "SELECT relname, relkind, pg_get_userbyid(relowner), relacl::text "
                "FROM pg_class "
                "WHERE relnamespace = %s::regnamespace "
                "ORDER BY relkind, relname",
                (schema,),
            )
            relations = tuple(cursor.fetchall())
            return namespace, functions, relations
    finally:
        connection.close()


def test_first_run_creates_only_marked_roles_and_stops_before_catalog(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster

    receipt = cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert receipt.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    assert receipt.verified_role_probes == ()
    assert receipt.pending_role_credentials == cluster.roles.login_roles
    rows = _role_rows(cluster)
    assert tuple(row[0] for row in rows) == tuple(sorted(cluster.roles.all))
    for row in rows:
        role, can_login, *attributes, marker = row
        assert can_login is False
        assert attributes == [False, False, False, False, False, -1]
        assert marker == (
            "elvis-postgres-bootstrap:v1:"
            f"{cluster.database}:{_logical_role(cluster, role)}"
        )
    for role in cluster.roles.login_roles:
        with pytest.raises(psycopg2.OperationalError):
            cluster.role_factory(role)()

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regnamespace('np')")
            assert cursor.fetchone() == (None,)
            cursor.execute(
                "SELECT pg_get_userbyid(datdba), "
                "has_database_privilege(%s, %s, 'CREATE') "
                "FROM pg_database WHERE datname = %s",
                (cluster.roles.schema_owner, cluster.database, cluster.database),
            )
            assert cursor.fetchone() == (cluster.admin_role, False)
    finally:
        connection.close()


def test_exact_existing_volume_is_admitted_before_role_staging(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = False
    try:
        assert apply_migrations(connection, load_migrations()) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
    finally:
        connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.admin_role,
        allowed_historical_owner_roles=(cluster.admin_role,),
    )
    context = cluster.context(adoption=adoption)
    before = _authority_snapshot(cluster)
    assert _role_rows(cluster) == ()

    receipt = cluster.bootstrap(with_credentials=False).reconcile(context)

    assert receipt.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    assert receipt.verified_role_probes == ()
    assert receipt.pending_role_credentials == cluster.roles.login_roles
    assert tuple(row[0] for row in _role_rows(cluster)) == tuple(
        sorted(cluster.roles.all)
    )
    assert _authority_snapshot(cluster)[1:] == before[1:]


def test_fresh_database_owner_drift_fails_before_roles_or_schema_mutation(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.outsider)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.outsider),
                )
            )
    finally:
        connection.close()

    def preflight_state():
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_get_userbyid(datdba), datacl::text "
                    "FROM pg_database WHERE datname = %s",
                    (cluster.database,),
                )
                database = cursor.fetchone()
                cursor.execute(
                    "SELECT nspname, pg_get_userbyid(nspowner), nspacl::text "
                    "FROM pg_namespace WHERE nspname = 'np'"
                )
                schema = tuple(cursor.fetchall())
                cursor.execute(
                    "SELECT count(*) FROM pg_class relation_row "
                    "JOIN pg_namespace namespace_row "
                    "ON namespace_row.oid = relation_row.relnamespace "
                    "WHERE namespace_row.nspname = 'np'"
                )
                object_count = cursor.fetchone()[0]
            return database, schema, object_count, _role_rows(cluster)
        finally:
            connection.close()

    before = preflight_state()
    assert before[0][0] == cluster.outsider
    assert before[1:] == ((), 0, ())

    with pytest.raises(
        PostgresBootstrapDriftError,
        match="database.*owner|owner.*database",
    ):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert preflight_state() == before


@pytest.mark.parametrize(
    "wrong_marker",
    [
        None,
        "elvis-postgres-bootstrap:v1:other_database:atomic_runtime",
        "elvis-postgres-bootstrap:v1:database:readiness",
    ],
)
def test_preexisting_role_collision_fails_closed_without_partial_role_set(
    bootstrap_cluster,
    wrong_marker,
):
    cluster = bootstrap_cluster
    collision = cluster.roles.atomic_runtime
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(collision))
            )
            if wrong_marker is not None:
                cursor.execute(
                    sql.SQL("COMMENT ON ROLE {} IS %s").format(
                        sql.Identifier(collision)
                    ),
                    (wrong_marker,),
                )
    finally:
        connection.close()

    with pytest.raises(PostgresBootstrapDriftError, match="role"):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert tuple(row[0] for row in _role_rows(cluster)) == (collision,)


def test_role_factory_must_authenticate_as_the_declared_authority(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    first = cluster.bootstrap(with_credentials=False).reconcile(cluster.context())
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()
    bootstrap = PostgresBootstrap(
        cluster.admin_factory,
        migrator_connection_factory=cluster.role_factory(cluster.roles.migrator),
        legacy_runtime_connection_factory=cluster.role_factory(
            cluster.roles.legacy_runtime
        ),
        atomic_runtime_connection_factory=cluster.role_factory(
            cluster.roles.atomic_runtime
        ),
        activation_connection_factory=cluster.role_factory(cluster.roles.activation),
        readiness_connection_factory=cluster.admin_factory,
        trainer_connection_factory=cluster.role_factory(cluster.roles.trainer),
    )

    with pytest.raises(PostgresBootstrapDriftError, match="readiness"):
        bootstrap.reconcile(cluster.context())

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regnamespace('np')")
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_role_factory_cannot_hide_an_admin_session_behind_set_role(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context()).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()

    def admin_as_readiness():
        connection = cluster.admin_factory()
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET ROLE {}").format(sql.Identifier(cluster.roles.readiness))
            )
        connection.autocommit = False
        return connection

    bootstrap = PostgresBootstrap(
        cluster.admin_factory,
        migrator_connection_factory=cluster.role_factory(cluster.roles.migrator),
        legacy_runtime_connection_factory=cluster.role_factory(
            cluster.roles.legacy_runtime
        ),
        atomic_runtime_connection_factory=cluster.role_factory(
            cluster.roles.atomic_runtime
        ),
        activation_connection_factory=cluster.role_factory(cluster.roles.activation),
        readiness_connection_factory=admin_as_readiness,
        trainer_connection_factory=cluster.role_factory(cluster.roles.trainer),
    )

    with pytest.raises(PostgresBootstrapDriftError, match="identity"):
        bootstrap.reconcile(cluster.context())


def test_database_scoped_role_setting_is_rejected_even_when_rolconfig_is_null(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context()).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "ALTER ROLE {} IN DATABASE {} SET statement_timeout = '2s'"
                ).format(
                    sql.Identifier(cluster.roles.readiness),
                    sql.Identifier(cluster.database),
                )
            )
    finally:
        admin.close()

    with pytest.raises(PostgresBootstrapDriftError, match="setting|config"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())


@pytest.mark.parametrize("credential_drift", ["password_null", "expired"])
def test_managed_login_requires_nonexpired_password_even_if_hba_could_trust(
    bootstrap_cluster,
    credential_drift,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if credential_drift == "password_null":
                cursor.execute(
                    sql.SQL("ALTER ROLE {} PASSWORD NULL").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
            else:
                cursor.execute(
                    sql.SQL("ALTER ROLE {} VALID UNTIL '2000-01-01'").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
    finally:
        connection.close()

    def credential_state():
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT rolcanlogin, rolpassword IS NULL, "
                    "rolvaliduntil IS NULL OR rolvaliduntil > clock_timestamp() "
                    "FROM pg_authid WHERE rolname = %s",
                    (cluster.roles.readiness,),
                )
                return cursor.fetchone()
        finally:
            connection.close()

    expected_state = (
        (True, True, True)
        if credential_drift == "password_null"
        else (True, False, False)
    )
    assert credential_state() == expected_state
    catalog_before = _authority_snapshot(cluster)

    with pytest.raises(
        PostgresBootstrapDriftError,
        match="credentials.*absent|credentials.*expired|credentials.*unsafe",
    ):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert credential_state() == expected_state
    assert _authority_snapshot(cluster) == catalog_before


def test_schema_owner_drift_is_rejected_without_repair(bootstrap_cluster):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER SCHEMA np OWNER TO {}").format(
                    sql.Identifier(cluster.admin_role)
                )
            )
    finally:
        admin.close()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|owner|schema"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner) FROM pg_namespace "
                "WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (cluster.admin_role,)
    finally:
        admin.close()


def test_factory_failure_redacts_exception_and_chained_cause(bootstrap_cluster):
    cluster = bootstrap_cluster
    cluster.bootstrap(with_credentials=False).reconcile(cluster.context())
    cluster.provision_managed_passwords()
    sentinel = "SENTINEL-SECRET-MUST-NOT-ESCAPE"

    def leaking_factory():
        raise RuntimeError(f"connection failed for {sentinel}")

    bootstrap = PostgresBootstrap(
        cluster.admin_factory,
        migrator_connection_factory=leaking_factory,
        legacy_runtime_connection_factory=cluster.role_factory(
            cluster.roles.legacy_runtime
        ),
        atomic_runtime_connection_factory=cluster.role_factory(
            cluster.roles.atomic_runtime
        ),
        activation_connection_factory=cluster.role_factory(cluster.roles.activation),
        readiness_connection_factory=cluster.role_factory(cluster.roles.readiness),
        trainer_connection_factory=cluster.role_factory(cluster.roles.trainer),
    )

    with pytest.raises(PostgresBootstrapStorageError) as caught:
        bootstrap.reconcile(cluster.context())

    rendered = [repr(bootstrap), repr(caught.value), str(caught.value)]
    nested = [caught.value.__cause__, caught.value.__context__]
    seen = set()
    while nested:
        exception = nested.pop()
        if exception is None or id(exception) in seen:
            continue
        seen.add(id(exception))
        rendered.extend((repr(exception), str(exception)))
        nested.extend((exception.__cause__, exception.__context__))
    assert all(sentinel not in value for value in rendered)


def test_fresh_bootstrap_installs_exact_ownership_and_runtime_boundaries(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    receipt = _complete_fresh_bootstrap(cluster)

    assert receipt.status is PostgresBootstrapStatus.COMPLETE
    assert receipt.migration_versions == (1, 2, 3, 4, 5, 6)
    assert receipt.verified_role_probes == cluster.roles.login_roles
    assert receipt.pending_role_credentials == ()
    assert receipt.old_shared_runtime_demoted is False
    assert _public_schema_evidence(cluster) == (
        ("pg_database_owner", "PUBLIC", "USAGE", False),
        ("pg_database_owner", "pg_database_owner", "CREATE", False),
        ("pg_database_owner", "pg_database_owner", "USAGE", False),
    )

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_get_userbyid(datdba),
                       has_database_privilege(%s, datname, 'CREATE'),
                       ARRAY(
                           SELECT managed_role
                           FROM unnest(%s::text[]) AS managed_role
                           WHERE has_database_privilege(
                               managed_role, datname, 'CREATE'
                           )
                           ORDER BY managed_role
                       ),
                       EXISTS (
                           SELECT 1
                           FROM aclexplode(datacl)
                           WHERE grantee = 0 AND privilege_type = 'CREATE'
                       )
                FROM pg_database
                WHERE datname = %s
                """,
                (
                    cluster.roles.schema_owner,
                    list(cluster.roles.login_roles),
                    cluster.database,
                ),
            )
            assert cursor.fetchone() == (cluster.admin_role, True, [], False)

            cursor.execute(
                """
                SELECT COALESCE(grantee_row.rolname, 'PUBLIC'),
                       database_acl.privilege_type
                FROM pg_database database_row
                CROSS JOIN LATERAL aclexplode(database_row.datacl) database_acl
                LEFT JOIN pg_roles grantee_row
                  ON grantee_row.oid = database_acl.grantee
                WHERE database_row.datname = %s
                  AND database_acl.grantee <> database_row.datdba
                ORDER BY 1, 2
                """,
                (cluster.database,),
            )
            assert cursor.fetchall() == sorted(
                [(cluster.roles.schema_owner, "CREATE")]
                + [(role, "CONNECT") for role in cluster.roles.login_roles]
            )

            cursor.execute(
                """
                SELECT parent.rolname, member.rolname
                FROM pg_auth_members membership
                JOIN pg_roles parent ON parent.oid = membership.roleid
                JOIN pg_roles member ON member.oid = membership.member
                WHERE parent.rolname = ANY(%s)
                   OR member.rolname = ANY(%s)
                ORDER BY parent.rolname, member.rolname
                """,
                (list(cluster.roles.all), list(cluster.roles.all)),
            )
            assert cursor.fetchall() == [
                (cluster.roles.schema_owner, cluster.roles.migrator)
            ]

            cursor.execute("""
                SELECT relation_row.relname,
                       COALESCE(grantee_row.rolname, 'PUBLIC'),
                       relation_acl.privilege_type
                FROM pg_class relation_row
                CROSS JOIN LATERAL aclexplode(relation_row.relacl) relation_acl
                LEFT JOIN pg_roles grantee_row
                  ON grantee_row.oid = relation_acl.grantee
                WHERE relation_row.relnamespace = 'np'::regnamespace
                  AND relation_row.relkind = 'r'
                  AND relation_acl.grantee <> relation_row.relowner
                ORDER BY 1, 2, 3
                """)
            assert set(cursor.fetchall()) == _expected_table_grants(cluster)

            cursor.execute("""
                SELECT relation_row.relname,
                       COALESCE(grantee_row.rolname, 'PUBLIC'),
                       relation_acl.privilege_type
                FROM pg_class relation_row
                CROSS JOIN LATERAL aclexplode(relation_row.relacl) relation_acl
                LEFT JOIN pg_roles grantee_row
                  ON grantee_row.oid = relation_acl.grantee
                WHERE relation_row.relnamespace = 'np'::regnamespace
                  AND relation_row.relkind = 'S'
                  AND relation_acl.grantee <> relation_row.relowner
                ORDER BY 1, 2, 3
                """)
            assert cursor.fetchall() == [
                (sequence, cluster.roles.legacy_runtime, "USAGE")
                for sequence in sorted(_LEGACY_SEQUENCES)
            ]

            cursor.execute("""
                SELECT COALESCE(grantee_row.rolname, 'PUBLIC'),
                       schema_acl.privilege_type
                FROM pg_namespace namespace_row
                CROSS JOIN LATERAL aclexplode(namespace_row.nspacl) schema_acl
                LEFT JOIN pg_roles grantee_row
                  ON grantee_row.oid = schema_acl.grantee
                WHERE namespace_row.nspname = 'np'
                  AND schema_acl.grantee <> namespace_row.nspowner
                ORDER BY 1, 2
                """)
            assert cursor.fetchall() == [
                (role, "USAGE") for role in sorted(cluster.roles.login_roles)
            ]

            cursor.execute("""
                SELECT pg_get_userbyid(nspowner),
                       obj_description(oid, 'pg_namespace')
                FROM pg_namespace
                WHERE nspname = 'np'
                """)
            assert cursor.fetchone() == (
                cluster.roles.schema_owner,
                f"elvis-postgres-bootstrap-schema:v1:{cluster.database}",
            )

            cursor.execute("""
                SELECT function_row.proname,
                       COALESCE(grantee_row.rolname, 'PUBLIC'),
                       function_acl.privilege_type
                FROM pg_proc function_row
                CROSS JOIN LATERAL aclexplode(function_row.proacl) function_acl
                LEFT JOIN pg_roles grantee_row
                  ON grantee_row.oid = function_acl.grantee
                WHERE function_row.pronamespace = 'np'::regnamespace
                  AND function_acl.grantee <> function_row.proowner
                ORDER BY 1, 2, 3
                """)
            assert cursor.fetchall() == []

            cursor.execute("""
                SELECT COALESCE(grantee_row.rolname, 'PUBLIC'),
                       default_acl.defaclobjtype,
                       expanded_acl.privilege_type
                FROM pg_default_acl default_acl
                CROSS JOIN LATERAL aclexplode(default_acl.defaclacl) expanded_acl
                LEFT JOIN pg_roles grantee_row
                  ON grantee_row.oid = expanded_acl.grantee
                WHERE default_acl.defaclnamespace = 'np'::regnamespace
                  AND expanded_acl.grantee <> default_acl.defaclrole
                ORDER BY 1, 2, 3
                """)
            assert cursor.fetchall() == []

            cursor.execute("""
                SELECT relname, relkind, pg_get_userbyid(relowner)
                FROM pg_class
                WHERE relnamespace = 'np'::regnamespace
                  AND relkind IN ('r', 'S')
                ORDER BY relkind, relname
                """)
            relations = tuple(cursor.fetchall())
            assert {(name, kind) for name, kind, _owner in relations} == {
                *((table, "r") for table in _AUTHORITY_TABLES),
                *((sequence, "S") for sequence in _LEGACY_SEQUENCES),
            }
            assert {owner for _name, _kind, owner in relations} == {
                cluster.roles.schema_owner
            }

            cursor.execute("""
                SELECT proname, pg_get_function_identity_arguments(function_row.oid),
                       pg_get_userbyid(proowner)
                FROM pg_proc function_row
                WHERE pronamespace = 'np'::regnamespace
                ORDER BY proname
                """)
            functions = tuple(cursor.fetchall())
            activation_owners = {
                owner
                for name, _arguments, owner in functions
                if name
                in {
                    "acquire_paper_runtime_activation_fence",
                    "activate_paper_runtime_generation",
                }
            }
            other_owners = {
                owner
                for name, _arguments, owner in functions
                if name
                not in {
                    "acquire_paper_runtime_activation_fence",
                    "activate_paper_runtime_generation",
                }
            }
            assert activation_owners == {cluster.roles.activation}
            assert other_owners == {cluster.roles.schema_owner}

            cursor.execute(
                """
                SELECT table_name,
                       has_table_privilege(%s, 'np.' || table_name, 'SELECT')
                FROM information_schema.tables
                WHERE table_schema = 'np'
                ORDER BY table_name
                """,
                (cluster.roles.activation,),
            )
            assert cursor.fetchall() == [(table, True) for table in _AUTHORITY_TABLES]

            cursor.execute(
                """
                SELECT
                    has_column_privilege(
                        %s, 'np.paper_runtime_control', 'control_key', 'UPDATE'
                    ),
                    has_column_privilege(
                        %s, 'np.paper_runtime_control', 'mode', 'UPDATE'
                    ),
                    has_column_privilege(
                        %s,
                        'np.paper_runtime_control',
                        'runtime_generation',
                        'UPDATE'
                    ),
                    has_column_privilege(
                        %s,
                        'np.paper_runtime_generations',
                        'activation_id',
                        'UPDATE'
                    ),
                    has_column_privilege(
                        %s,
                        'np.paper_runtime_generations',
                        'runtime_generation',
                        'UPDATE'
                    )
                """,
                (cluster.roles.atomic_runtime,) * 5,
            )
            assert cursor.fetchone() == (True, False, False, True, False)
    finally:
        connection.close()

    atomic = cluster.role_factory(cluster.roles.atomic_runtime)()
    atomic.autocommit = False
    try:
        with atomic.cursor() as cursor:
            cursor.execute("SELECT control_key FROM np.paper_runtime_control FOR SHARE")
            assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT activation_id FROM np.paper_runtime_generations FOR SHARE"
            )
            assert cursor.fetchall() == []
        atomic.rollback()

        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            with atomic.cursor() as cursor:
                cursor.execute(
                    "UPDATE np.paper_runtime_control SET mode = 'ACTIVE' "
                    "WHERE control_key IS TRUE"
                )
        atomic.rollback()
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            with atomic.cursor() as cursor:
                cursor.execute(
                    "UPDATE np.paper_runtime_control SET runtime_generation = 1 "
                    "WHERE control_key IS TRUE"
                )
    finally:
        atomic.close()


def test_six_login_roles_enforce_representative_allow_and_deny_matrix(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE

    def execute(role, statement, *, denied=False):
        connection = cluster.role_factory(role)()
        connection.autocommit = False
        try:
            if denied:
                with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                    with connection.cursor() as cursor:
                        cursor.execute(statement)
                connection.rollback()
                return
            with connection.cursor() as cursor:
                cursor.execute(statement)
                if cursor.description is not None:
                    cursor.fetchall()
            connection.rollback()
        finally:
            connection.close()

    execute(cluster.roles.legacy_runtime, "SELECT * FROM np.trades")
    execute(
        cluster.roles.legacy_runtime,
        "INSERT INTO np.trades (id, symbol) VALUES (920001, 'BTCUSDT')",
    )
    execute(cluster.roles.legacy_runtime, "SELECT nextval('np.trades_id_seq')")
    execute(cluster.roles.legacy_runtime, "SELECT * FROM np.orders", denied=True)
    execute(
        cluster.roles.legacy_runtime,
        "UPDATE np.trades SET symbol = symbol WHERE FALSE",
        denied=True,
    )

    execute(cluster.roles.atomic_runtime, "SELECT * FROM np.position_streams")
    execute(
        cluster.roles.atomic_runtime,
        "INSERT INTO np.position_streams (position_key, execution_scope) "
        "VALUES ('acl-matrix', 'paper:acl-matrix')",
    )
    execute(cluster.roles.atomic_runtime, "SELECT * FROM np.trades", denied=True)
    execute(
        cluster.roles.atomic_runtime,
        "UPDATE np.paper_runtime_control SET mode = mode WHERE FALSE",
        denied=True,
    )

    execute(cluster.roles.readiness, "SELECT * FROM np.trades")
    execute(cluster.roles.readiness, "SELECT * FROM np.orders")
    execute(
        cluster.roles.readiness,
        "INSERT INTO np.trades (id) VALUES (920002)",
        denied=True,
    )
    execute(cluster.roles.readiness, "SELECT nextval('np.trades_id_seq')", denied=True)

    execute(cluster.roles.trainer, "SELECT * FROM np.trades")
    execute(
        cluster.roles.trainer,
        "SELECT * FROM np.model_predictions",
        denied=True,
    )

    execute(cluster.roles.activation, "SELECT * FROM np.schema_migrations")
    execute(
        cluster.roles.activation,
        "SELECT np.acquire_paper_runtime_activation_fence()",
    )
    execute(
        cluster.roles.activation,
        "UPDATE np.trades SET symbol = symbol WHERE FALSE",
    )
    execute(
        cluster.roles.activation,
        "DELETE FROM np.trades WHERE FALSE",
        denied=True,
    )

    migrator = cluster.role_factory(cluster.roles.migrator)()
    migrator.autocommit = False
    try:
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            with migrator.cursor() as cursor:
                cursor.execute("SELECT * FROM np.trades")
        migrator.rollback()
        with migrator.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET ROLE {}").format(
                    sql.Identifier(cluster.roles.schema_owner)
                )
            )
            cursor.execute("SELECT * FROM np.trades")
            cursor.fetchall()
        migrator.rollback()
    finally:
        migrator.close()

    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            for role in (
                cluster.roles.migrator,
                cluster.roles.legacy_runtime,
                cluster.roles.atomic_runtime,
                cluster.roles.readiness,
                cluster.roles.trainer,
            ):
                cursor.execute(
                    "SELECT has_function_privilege("
                    "%s, 'np.acquire_paper_runtime_activation_fence()', 'EXECUTE')",
                    (role,),
                )
                assert cursor.fetchone() == (False,)
            cursor.execute(
                "SELECT has_function_privilege("
                "%s, 'np.acquire_paper_runtime_activation_fence()', 'EXECUTE')",
                (cluster.roles.activation,),
            )
            assert cursor.fetchone() == (True,)
    finally:
        admin.close()


def test_complete_catalog_reread_is_idempotent(bootstrap_cluster):
    cluster = bootstrap_cluster
    first = _complete_fresh_bootstrap(cluster)
    assert first.status is PostgresBootstrapStatus.COMPLETE
    before = _authority_snapshot(cluster)

    second = cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert second == first
    assert _authority_snapshot(cluster) == before


def test_schema_marker_tamper_is_rejected_without_repair(bootstrap_cluster):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    expected_marker = f"elvis-postgres-bootstrap-schema:v1:{cluster.database}"
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT obj_description(oid, 'pg_namespace') "
                "FROM pg_namespace WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (expected_marker,)
            cursor.execute("COMMENT ON SCHEMA np IS 'tampered-bootstrap-marker'")
    finally:
        connection.close()

    before = _authority_snapshot(cluster)
    with pytest.raises(PostgresBootstrapDriftError, match="catalog|schema|marker"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert _authority_snapshot(cluster) == before
    assert before[3][0][2] == "tampered-bootstrap-marker"


def test_adoption_schema_marker_is_exact_idempotent_and_tamper_evident(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    staged = _stage_existing_adoption(cluster)
    adoption = PostgresBootstrapAdoption(
        migration_authority_role=staged.migration_authority_role,
        allowed_historical_owner_roles=staged.allowed_historical_owner_roles,
        old_shared_runtime_role=staged.old_shared_runtime_role,
        demote_old_shared_runtime=True,
    )
    context = cluster.context(adoption=adoption)
    first = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert first.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    completed = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert completed.status is PostgresBootstrapStatus.COMPLETE
    assert completed.old_shared_runtime_demoted is True

    expected_marker = f"elvis-postgres-bootstrap-schema:v1:{cluster.database}"
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT pg_get_userbyid(nspowner),
                       obj_description(oid, 'pg_namespace')
                FROM pg_namespace
                WHERE nspname = 'np'
                """)
            assert cursor.fetchone() == (cluster.roles.schema_owner, expected_marker)
    finally:
        connection.close()

    rerun = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert rerun == completed
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("COMMENT ON SCHEMA np IS 'tampered-adoption-marker'")
    finally:
        connection.close()
    before = _authority_snapshot(cluster)

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|schema|marker"):
        cluster.bootstrap(with_credentials=True).reconcile(context)

    assert _authority_snapshot(cluster) == before
    assert before[3][0][2] == "tampered-adoption-marker"


def test_legacy_database_owner_with_default_acl_adopts_in_two_demotion_runs(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.old_runtime)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.old_runtime),
                )
            )
            cursor.execute(
                "SELECT pg_get_userbyid(datdba), datacl IS NULL "
                "FROM pg_database WHERE datname = %s",
                (cluster.database,),
            )
            assert cursor.fetchone() == (cluster.old_runtime, True)
    finally:
        connection.close()

    old_connection = cluster.role_factory(cluster.old_runtime)()
    old_connection.autocommit = False
    try:
        assert apply_migrations(old_connection, load_migrations()) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
    finally:
        old_connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=False,
    )
    staged_context = cluster.context(adoption=adoption)
    first = cluster.bootstrap(with_credentials=False).reconcile(staged_context)
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()
    staged = cluster.bootstrap(with_credentials=True).reconcile(staged_context)
    assert staged.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert staged.old_shared_runtime_demoted is False

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(datdba), datacl IS NULL "
                "FROM pg_database WHERE datname = %s",
                (cluster.database,),
            )
            assert cursor.fetchone() == (cluster.old_runtime, True)
            cursor.execute("""
                SELECT pg_get_userbyid(namespace_row.nspowner),
                       pg_get_userbyid(ledger_row.relowner)
                FROM pg_namespace namespace_row
                JOIN pg_class ledger_row
                  ON ledger_row.relnamespace = namespace_row.oid
                 AND ledger_row.relname = 'schema_migrations'
                WHERE namespace_row.nspname = 'np'
                """)
            assert cursor.fetchone() == (cluster.old_runtime, cluster.old_runtime)
    finally:
        connection.close()

    demotion = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=True,
    )
    context = cluster.context(adoption=demotion)
    disabled = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert disabled.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert disabled.old_shared_runtime_demoted is False
    completed = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert completed.status is PostgresBootstrapStatus.COMPLETE
    assert completed.old_shared_runtime_demoted is True

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(datdba) " "FROM pg_database WHERE datname = %s",
                (cluster.database,),
            )
            assert cursor.fetchone() == (cluster.admin_role,)
            cursor.execute("""
                SELECT pg_get_userbyid(nspowner),
                       obj_description(oid, 'pg_namespace')
                FROM pg_namespace
                WHERE nspname = 'np'
                """)
            assert cursor.fetchone() == (
                cluster.roles.schema_owner,
                f"elvis-postgres-bootstrap-schema:v1:{cluster.database}",
            )
            cursor.execute(
                "SELECT rolcanlogin, rolpassword IS NULL "
                "FROM pg_authid WHERE rolname = %s",
                (cluster.old_runtime,),
            )
            assert cursor.fetchone() == (False, True)
    finally:
        connection.close()


def test_varchar_typmod_drift_is_rejected_without_repair(bootstrap_cluster):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE np.orders " "ALTER COLUMN position_effect TYPE VARCHAR(17)"
            )
    finally:
        connection.close()

    def position_effect_type():
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT format_type(column_row.atttypid, column_row.atttypmod)
                    FROM pg_attribute column_row
                    WHERE column_row.attrelid = 'np.orders'::regclass
                      AND column_row.attname = 'position_effect'
                    """)
                return cursor.fetchone()
        finally:
            connection.close()

    assert position_effect_type() == ("character varying(17)",)
    with pytest.raises(PostgresBootstrapDriftError, match="catalog|column|shape"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())
    assert position_effect_type() == ("character varying(17)",)


@pytest.mark.parametrize("sequence_drift", ["increment", "owned_by"])
def test_sequence_definition_drift_is_rejected_without_repair(
    bootstrap_cluster,
    sequence_drift,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    assert _trades_sequence_evidence(cluster) == (1, False, "trades", "id")
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if sequence_drift == "increment":
                cursor.execute("ALTER SEQUENCE np.trades_id_seq INCREMENT BY 2")
            else:
                cursor.execute("ALTER SEQUENCE np.trades_id_seq OWNED BY NONE")
    finally:
        connection.close()

    drifted = _trades_sequence_evidence(cluster)
    assert drifted == (
        (2, False, "trades", "id")
        if sequence_drift == "increment"
        else (1, False, "", "")
    )
    with pytest.raises(PostgresBootstrapDriftError, match="catalog|sequence|shape"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())
    assert _trades_sequence_evidence(cluster) == drifted


@pytest.mark.parametrize("role_drift", ["superuser", "membership"])
def test_managed_role_drift_after_probes_is_rechecked_before_catalog_mutation(
    bootstrap_cluster,
    monkeypatch,
    role_drift,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    if role_drift == "membership":
        cluster.create_auxiliary_login(cluster.outsider)
    bootstrap = cluster.bootstrap(with_credentials=True)
    catalog_readback = bootstrap._catalog_readback_is_exact
    injected_state = {}

    def inject_role_drift(context):
        connection = cluster.admin_factory()
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                if role_drift == "superuser":
                    cursor.execute(
                        sql.SQL("ALTER ROLE {} SUPERUSER").format(
                            sql.Identifier(cluster.roles.readiness)
                        )
                    )
                else:
                    cursor.execute(
                        sql.SQL("GRANT {} TO {}").format(
                            sql.Identifier(cluster.roles.trainer),
                            sql.Identifier(cluster.roles.readiness),
                        )
                    )
        finally:
            connection.close()
        injected_state["snapshot"] = _authority_snapshot(cluster)
        return catalog_readback(context)

    monkeypatch.setattr(
        bootstrap,
        "_catalog_readback_is_exact",
        inject_role_drift,
    )

    with pytest.raises(PostgresBootstrapDriftError, match="role|membership|catalog"):
        bootstrap.reconcile(cluster.context())

    assert _authority_snapshot(cluster) == injected_state["snapshot"]
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            if role_drift == "superuser":
                cursor.execute(
                    "SELECT rolsuper FROM pg_roles WHERE rolname = %s",
                    (cluster.roles.readiness,),
                )
                assert cursor.fetchone() == (True,)
            else:
                cursor.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_auth_members membership "
                    "JOIN pg_roles parent ON parent.oid = membership.roleid "
                    "JOIN pg_roles member ON member.oid = membership.member "
                    "WHERE parent.rolname = %s AND member.rolname = %s)",
                    (cluster.roles.trainer, cluster.roles.readiness),
                )
                assert cursor.fetchone() == (True,)
    finally:
        connection.close()


def test_safe_additional_plain_index_owned_by_schema_owner_is_accepted(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    migrator = cluster.role_factory(cluster.roles.migrator)()
    migrator.autocommit = False
    try:
        with migrator.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET ROLE {}").format(
                    sql.Identifier(cluster.roles.schema_owner)
                )
            )
            cursor.execute(
                "CREATE INDEX safe_additional_trades_side_idx ON np.trades (side)"
            )
        migrator.commit()
    finally:
        migrator.close()

    receipt = cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert receipt.status is PostgresBootstrapStatus.COMPLETE
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT index_row.indisunique,
                       index_row.indisvalid,
                       index_row.indisready,
                       index_row.indpred IS NULL,
                       index_row.indexprs IS NULL,
                       pg_get_userbyid(class_row.relowner)
                FROM pg_index index_row
                JOIN pg_class class_row ON class_row.oid = index_row.indexrelid
                WHERE class_row.oid = 'np.safe_additional_trades_side_idx'::regclass
                """)
            assert cursor.fetchone() == (
                False,
                True,
                True,
                True,
                True,
                cluster.roles.schema_owner,
            )
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("index_name", "replacement_sql"),
    [
        ("idx_trades_symbol_ts", None),
        ("orders_venue_identity_uq", None),
        ("order_events_paper_account_fill_ref_uq", None),
        (
            "idx_trades_symbol_ts",
            "CREATE INDEX idx_trades_symbol_ts ON np.trades (side)",
        ),
        (
            "idx_trades_symbol_ts",
            "CREATE INDEX idx_trades_symbol_ts "
            "ON np.trades (symbol DESC, timestamp)",
        ),
        (
            "orders_venue_identity_uq",
            "CREATE INDEX orders_venue_identity_uq ON np.orders (symbol)",
        ),
        (
            "order_events_paper_account_fill_ref_uq",
            "CREATE UNIQUE INDEX order_events_paper_account_fill_ref_uq "
            "ON np.order_events (event_id)",
        ),
    ],
)
def test_packaged_index_missing_or_definition_drift_is_rejected(
    bootstrap_cluster,
    index_name,
    replacement_sql,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    expected_indexes = _index_snapshot(cluster)
    assert tuple(row[0] for row in expected_indexes) == _PACKAGED_INDEXES
    migrator = cluster.role_factory(cluster.roles.migrator)()
    migrator.autocommit = False
    try:
        with migrator.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET ROLE {}").format(
                    sql.Identifier(cluster.roles.schema_owner)
                )
            )
            drop_sql = (
                "DROP INDEX np.{} CASCADE"
                if index_name == "order_events_paper_account_fill_ref_uq"
                else "DROP INDEX np.{}"
            )
            cursor.execute(sql.SQL(drop_sql).format(sql.Identifier(index_name)))
            if replacement_sql is not None:
                cursor.execute(replacement_sql)
        migrator.commit()
    finally:
        migrator.close()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|index"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())


@pytest.mark.parametrize("durable", [True, False])
def test_catalog_commit_failure_is_resolved_only_from_exact_readback(
    bootstrap_cluster,
    durable,
):
    cluster = bootstrap_cluster
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context()).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()
    state = {"commits": 0, "fired": False}

    def fault_admin_factory():
        return NthCommitFaultConnection(
            cluster.admin_factory(),
            state,
            target=2,
            durable=durable,
        )

    bootstrap = cluster.bootstrap(
        with_credentials=True,
        admin_factory=fault_admin_factory,
    )
    if durable:
        receipt = bootstrap.reconcile(cluster.context())
        assert receipt.status is PostgresBootstrapStatus.COMPLETE
        assert _authority_snapshot(cluster)
    else:
        with pytest.raises(PostgresBootstrapCommitUnknownError) as caught:
            bootstrap.reconcile(cluster.context())
        assert caught.value.phase is PostgresBootstrapPhase.CATALOG
        assert caught.value.__cause__ is None
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO np.trades (id, symbol, side) "
                    "VALUES (940001, 'FRESH-SENTINEL', 'BUY')"
                )
            connection.commit()
        finally:
            connection.close()
        receipt = cluster.bootstrap(with_credentials=True).reconcile(cluster.context())
        assert receipt.status is PostgresBootstrapStatus.COMPLETE
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT symbol, side FROM np.trades WHERE id = 940001")
                assert cursor.fetchone() == ("FRESH-SENTINEL", "BUY")
        finally:
            connection.close()
    assert state["fired"] is True


def test_complete_is_not_returned_when_post_commit_readback_has_drifted(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context()).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()
    state = {"commits": 0, "fired": False, "mutated": False, "connections": 0}

    def inject_drift():
        connection = cluster.admin_factory()
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("GRANT SELECT ON np.trades TO PUBLIC")
        finally:
            connection.close()

    def fault_admin_factory():
        state["connections"] += 1
        base = cluster.admin_factory()
        fault = NthCommitFaultConnection(
            base,
            state,
            target=2,
            durable=True,
        )
        if state["commits"] >= 2 and not state["mutated"]:
            state["mutated"] = True
            inject_drift()
        return fault

    bootstrap = cluster.bootstrap(
        with_credentials=True,
        admin_factory=fault_admin_factory,
    )

    with pytest.raises(PostgresBootstrapCommitUnknownError) as caught:
        bootstrap.reconcile(cluster.context())

    assert caught.value.phase is PostgresBootstrapPhase.CATALOG
    assert state["fired"] is True
    assert state["mutated"] is True


@pytest.mark.parametrize("durable", [True, False])
def test_demotion_commit_failure_preserves_or_proves_the_old_role_boundary(
    bootstrap_cluster,
    durable,
):
    cluster = bootstrap_cluster
    staged_adoption = _stage_existing_adoption(cluster)
    demotion = PostgresBootstrapAdoption(
        migration_authority_role=staged_adoption.migration_authority_role,
        allowed_historical_owner_roles=(staged_adoption.migration_authority_role,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=True,
    )
    context = cluster.context(adoption=demotion)
    state = {"commits": 0, "fired": False}

    def fault_admin_factory():
        return NthCommitFaultConnection(
            cluster.admin_factory(),
            state,
            target=1,
            durable=durable,
        )

    bootstrap = cluster.bootstrap(
        with_credentials=True,
        admin_factory=fault_admin_factory,
    )
    if durable:
        receipt = bootstrap.reconcile(context)
        assert receipt.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
        assert receipt.old_shared_runtime_demoted is False
        completed = cluster.bootstrap(with_credentials=True).reconcile(context)
        assert completed.status is PostgresBootstrapStatus.COMPLETE
        assert completed.old_shared_runtime_demoted is True
    else:
        with pytest.raises(PostgresBootstrapCommitUnknownError) as caught:
            bootstrap.reconcile(context)
        assert caught.value.phase is PostgresBootstrapPhase.DEMOTION
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT rolcanlogin FROM pg_roles WHERE rolname = %s",
                    (cluster.old_runtime,),
                )
                assert cursor.fetchone() == (True,)
        finally:
            connection.close()
        demoted = cluster.bootstrap(with_credentials=True).reconcile(context)
        assert demoted.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
        assert demoted.old_shared_runtime_demoted is False
        completed = cluster.bootstrap(with_credentials=True).reconcile(context)
        assert completed.status is PostgresBootstrapStatus.COMPLETE
        assert completed.old_shared_runtime_demoted is True
    assert state["fired"] is True


@pytest.mark.parametrize("durable", [True, False])
def test_post_drain_cutover_commit_failure_requires_exact_demotion_readback(
    bootstrap_cluster,
    durable,
):
    cluster = bootstrap_cluster
    staged_adoption = _stage_existing_adoption(cluster)
    demotion = PostgresBootstrapAdoption(
        migration_authority_role=staged_adoption.migration_authority_role,
        allowed_historical_owner_roles=(staged_adoption.migration_authority_role,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=True,
    )
    context = cluster.context(adoption=demotion)
    disabled = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert disabled.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert disabled.old_shared_runtime_demoted is False
    state = {"commits": 0, "fired": False}

    def fault_admin_factory():
        return NthCommitFaultConnection(
            cluster.admin_factory(),
            state,
            target=1,
            durable=durable,
        )

    bootstrap = cluster.bootstrap(
        with_credentials=True,
        admin_factory=fault_admin_factory,
    )
    if durable:
        receipt = bootstrap.reconcile(context)
        assert receipt.status is PostgresBootstrapStatus.COMPLETE
        assert receipt.old_shared_runtime_demoted is True
    else:
        with pytest.raises(PostgresBootstrapCommitUnknownError) as caught:
            bootstrap.reconcile(context)
        assert caught.value.phase is PostgresBootstrapPhase.DEMOTION
        recovered = cluster.bootstrap(with_credentials=True).reconcile(context)
        assert recovered.status is PostgresBootstrapStatus.COMPLETE
        assert recovered.old_shared_runtime_demoted is True
    assert state["fired"] is True


def test_hostile_session_search_path_catalog_is_rejected_without_mutation(
    bootstrap_cluster,
    postgres_connection_allowlist,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute("CREATE SCHEMA hostile")
            cursor.execute("GRANT USAGE ON SCHEMA hostile TO PUBLIC")
            cursor.execute(
                "CREATE FUNCTION hostile.current_database() RETURNS name "
                "LANGUAGE sql AS $$ SELECT 'hostile_database'::name $$"
            )
            cursor.execute(
                "CREATE FUNCTION hostile.clock_timestamp() RETURNS timestamptz "
                "LANGUAGE plpgsql AS $$ BEGIN "
                "RAISE EXCEPTION 'hostile search_path executed'; END $$"
            )
    finally:
        admin.close()

    def hostile_dsn(dsn):
        parameters = parse_dsn(dsn)
        parameters["options"] = "-csearch_path=hostile,pg_catalog"
        return make_dsn(**parameters)

    admin_dsn = hostile_dsn(cluster.admin_dsn)
    role_dsns = {role: hostile_dsn(dsn) for role, dsn in cluster.role_dsns.items()}
    hostile_dsns = (admin_dsn,) + tuple(role_dsns.values())
    for dsn in hostile_dsns:
        postgres_connection_allowlist.add(_dsn_identity(dsn))
    before = _authority_snapshot(cluster)
    hostile_before = _outside_schema_snapshot(cluster, "hostile")
    try:
        bootstrap = PostgresBootstrap(
            lambda: psycopg2.connect(admin_dsn),
            migrator_connection_factory=lambda: psycopg2.connect(
                role_dsns[cluster.roles.migrator]
            ),
            legacy_runtime_connection_factory=lambda: psycopg2.connect(
                role_dsns[cluster.roles.legacy_runtime]
            ),
            atomic_runtime_connection_factory=lambda: psycopg2.connect(
                role_dsns[cluster.roles.atomic_runtime]
            ),
            activation_connection_factory=lambda: psycopg2.connect(
                role_dsns[cluster.roles.activation]
            ),
            readiness_connection_factory=lambda: psycopg2.connect(
                role_dsns[cluster.roles.readiness]
            ),
            trainer_connection_factory=lambda: psycopg2.connect(
                role_dsns[cluster.roles.trainer]
            ),
        )

        with pytest.raises(PostgresBootstrapDriftError, match="catalog|schema"):
            bootstrap.reconcile(cluster.context())
    finally:
        for dsn in hostile_dsns:
            postgres_connection_allowlist.discard(_dsn_identity(dsn))

    assert _authority_snapshot(cluster) == before
    assert _outside_schema_snapshot(cluster, "hostile") == hostile_before
    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute("SELECT to_regnamespace('hostile')")
            assert cursor.fetchone() == ("hostile",)
            cursor.execute(
                "SELECT to_regprocedure('hostile.current_database()'), "
                "to_regprocedure('hostile.clock_timestamp()')"
            )
            assert cursor.fetchone() == (
                "hostile.current_database()",
                "hostile.clock_timestamp()",
            )
    finally:
        admin.close()


@pytest.mark.parametrize(
    ("object_kind", "create_sql", "exists_sql"),
    [
        (
            "collation",
            "CREATE COLLATION public.bootstrap_public_collation "
            "(provider = libc, locale = 'C')",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_collation collation_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = collation_row.collnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND collation_row.collname = 'bootstrap_public_collation'"
            ")",
        ),
        (
            "operator",
            "CREATE OPERATOR public.=== ("
            "LEFTARG = integer, RIGHTARG = integer, "
            "FUNCTION = pg_catalog.int4eq"
            ")",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_operator operator_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = operator_row.oprnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND operator_row.oprname = '==='"
            ")",
        ),
        (
            "operator_class",
            "CREATE OPERATOR CLASS public.bootstrap_public_int_hash_ops "
            "FOR TYPE integer USING hash AS "
            "OPERATOR 1 pg_catalog.= (integer, integer), "
            "FUNCTION 1 pg_catalog.hashint4(integer)",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_opclass opclass_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = opclass_row.opcnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND opclass_row.opcname = 'bootstrap_public_int_hash_ops'"
            ")",
        ),
        (
            "operator_family",
            "CREATE OPERATOR FAMILY public.bootstrap_public_int_hash_family "
            "USING hash",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_opfamily opfamily_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = opfamily_row.opfnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND opfamily_row.opfname = 'bootstrap_public_int_hash_family'"
            ")",
        ),
        (
            "conversion",
            "CREATE CONVERSION public.bootstrap_public_utf8_to_latin1 "
            "FOR 'UTF8' TO 'LATIN1' FROM pg_catalog.utf8_to_iso8859_1",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_conversion conversion_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = conversion_row.connamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND conversion_row.conname = 'bootstrap_public_utf8_to_latin1'"
            ")",
        ),
        (
            "text_search_configuration",
            "CREATE TEXT SEARCH CONFIGURATION "
            "public.bootstrap_public_ts_configuration "
            "(COPY = pg_catalog.simple)",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_ts_config config_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = config_row.cfgnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND config_row.cfgname = 'bootstrap_public_ts_configuration'"
            ")",
        ),
        (
            "text_search_dictionary",
            "CREATE TEXT SEARCH DICTIONARY "
            "public.bootstrap_public_ts_dictionary "
            "(TEMPLATE = pg_catalog.simple)",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_ts_dict dictionary_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = dictionary_row.dictnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND dictionary_row.dictname = 'bootstrap_public_ts_dictionary'"
            ")",
        ),
        (
            "text_search_parser",
            "CREATE TEXT SEARCH PARSER public.bootstrap_public_ts_parser ("
            "START = pg_catalog.prsd_start, "
            "GETTOKEN = pg_catalog.prsd_nexttoken, "
            "END = pg_catalog.prsd_end, "
            "LEXTYPES = pg_catalog.prsd_lextype, "
            "HEADLINE = pg_catalog.prsd_headline"
            ")",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_ts_parser parser_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = parser_row.prsnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND parser_row.prsname = 'bootstrap_public_ts_parser'"
            ")",
        ),
        (
            "text_search_template",
            "CREATE TEXT SEARCH TEMPLATE public.bootstrap_public_ts_template ("
            "INIT = pg_catalog.dsimple_init, "
            "LEXIZE = pg_catalog.dsimple_lexize"
            ")",
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_ts_template template_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = template_row.tmplnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND template_row.tmplname = 'bootstrap_public_ts_template'"
            ")",
        ),
        (
            "statistics",
            (
                "CREATE TEMP TABLE bootstrap_public_statistics_source "
                "(first_value INTEGER, second_value INTEGER)",
                "CREATE STATISTICS public.bootstrap_public_statistics "
                "(dependencies) ON first_value, second_value "
                "FROM bootstrap_public_statistics_source",
            ),
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_statistic_ext statistics_row "
            "JOIN pg_namespace namespace_row "
            "ON namespace_row.oid = statistics_row.stxnamespace "
            "WHERE namespace_row.nspname = 'public' "
            "AND statistics_row.stxname = 'bootstrap_public_statistics'"
            ")",
        ),
    ],
)
def test_public_standalone_catalog_object_is_rejected_before_managed_roles(
    bootstrap_cluster,
    object_kind,
    create_sql,
    exists_sql,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            statements = (create_sql,) if isinstance(create_sql, str) else create_sql
            for statement in statements:
                cursor.execute(statement)
            cursor.execute(exists_sql)
            assert cursor.fetchone() == (True,), object_kind

        assert _role_rows(cluster) == ()

        with pytest.raises(
            PostgresBootstrapDriftError,
            match="catalog|public object",
        ):
            cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

        assert _role_rows(cluster) == ()
        with connection.cursor() as cursor:
            cursor.execute(exists_sql)
            assert cursor.fetchone() == (True,), object_kind
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("object_kind", "create_statements", "exists_sql"),
    [
        (
            "event_trigger",
            (
                "CREATE FUNCTION "
                "pg_catalog.bootstrap_untrusted_event_trigger() "
                "RETURNS event_trigger LANGUAGE plpgsql "
                "AS $$ BEGIN END $$",
                "CREATE EVENT TRIGGER bootstrap_untrusted_event_trigger "
                "ON ddl_command_start EXECUTE FUNCTION "
                "pg_catalog.bootstrap_untrusted_event_trigger()",
            ),
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_event_trigger "
            "WHERE evtname = 'bootstrap_untrusted_event_trigger'"
            ")",
        ),
        (
            "foreign_data_wrapper",
            (
                "CREATE FOREIGN DATA WRAPPER bootstrap_untrusted_fdw "
                "NO HANDLER NO VALIDATOR",
            ),
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_foreign_data_wrapper "
            "WHERE fdwname = 'bootstrap_untrusted_fdw'"
            ")",
        ),
        (
            "publication",
            ("CREATE PUBLICATION bootstrap_untrusted_publication",),
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_publication "
            "WHERE pubname = 'bootstrap_untrusted_publication'"
            ")",
        ),
    ],
)
def test_database_scoped_catalog_object_is_rejected_before_managed_roles(
    bootstrap_cluster,
    object_kind,
    create_statements,
    exists_sql,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for statement in create_statements:
                cursor.execute(statement)
            cursor.execute(exists_sql)
            assert cursor.fetchone() == (True,), object_kind
    finally:
        connection.close()

    assert _role_rows(cluster) == ()

    with pytest.raises(
        PostgresBootstrapDriftError,
        match="catalog|database|object",
    ):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert _role_rows(cluster) == ()
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(exists_sql)
            assert cursor.fetchone() == (True,), object_kind
    finally:
        connection.close()


@pytest.mark.parametrize(
    "drift_kind",
    [
        "plpgsql_validator_attribute",
        "bthandler_attribute",
        "plpgsql_validator_extension_membership",
    ],
)
def test_builtin_routine_or_extension_membership_drift_is_rejected_before_roles(
    bootstrap_cluster,
    drift_kind,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if drift_kind == "plpgsql_validator_attribute":
                cursor.execute(
                    "UPDATE pg_catalog.pg_proc SET prosecdef = TRUE "
                    "WHERE oid = "
                    "'pg_catalog.plpgsql_validator(oid)'::regprocedure "
                    "RETURNING prosecdef"
                )
                assert cursor.fetchone() == (True,)
            elif drift_kind == "bthandler_attribute":
                cursor.execute(
                    "UPDATE pg_catalog.pg_proc SET proleakproof = TRUE "
                    "WHERE oid = "
                    "'pg_catalog.bthandler(internal)'::regprocedure "
                    "RETURNING proleakproof"
                )
                assert cursor.fetchone() == (True,)
            else:
                cursor.execute(
                    "DELETE FROM pg_catalog.pg_depend dependency_row "
                    "WHERE dependency_row.classid = 'pg_proc'::regclass "
                    "AND dependency_row.objid = "
                    "'pg_catalog.plpgsql_validator(oid)'::regprocedure "
                    "AND dependency_row.refclassid = 'pg_extension'::regclass "
                    "AND dependency_row.refobjid = ("
                    "SELECT extension_row.oid FROM pg_extension extension_row "
                    "WHERE extension_row.extname = 'plpgsql'"
                    ") AND dependency_row.deptype = 'e' "
                    "RETURNING dependency_row.deptype"
                )
                assert cursor.fetchone() == ("e",)
    finally:
        connection.close()

    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog"):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert _role_rows(cluster) == ()


def test_dependency_free_user_cast_is_rejected_before_managed_roles(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'CREATE CAST (boolean AS "char") ' "WITHOUT FUNCTION AS ASSIGNMENT"
            )
            cursor.execute(
                "SELECT cast_row.oid, COUNT(dependency_row.*) "
                "FROM pg_cast cast_row "
                "LEFT JOIN pg_depend dependency_row "
                "ON dependency_row.classid = 'pg_cast'::regclass "
                "AND dependency_row.objid = cast_row.oid "
                "WHERE cast_row.castsource = 'boolean'::regtype "
                "AND cast_row.casttarget = '\"char\"'::regtype "
                "GROUP BY cast_row.oid"
            )
            cast_oid, dependency_count = cursor.fetchone()
            assert type(cast_oid) is int
            assert dependency_count == 0
    finally:
        connection.close()

    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog"):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert _role_rows(cluster) == ()


def test_user_routine_in_pg_catalog_is_rejected_before_managed_roles(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE FUNCTION "
                "pg_catalog.bootstrap_untrusted_catalog_routine(integer) "
                "RETURNS integer LANGUAGE sql IMMUTABLE AS 'SELECT $1'"
            )
            cursor.execute(
                "SELECT pg_get_userbyid(proowner) FROM pg_proc "
                "WHERE oid = "
                "'pg_catalog.bootstrap_untrusted_catalog_routine(integer)'"
                "::regprocedure"
            )
            assert cursor.fetchone() == (cluster.admin_role,)
    finally:
        connection.close()

    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog"):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert _role_rows(cluster) == ()


def test_nonprocedural_language_owner_drift_is_rejected_before_managed_roles(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.outsider)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER LANGUAGE sql OWNER TO {}").format(
                    sql.Identifier(cluster.outsider)
                )
            )
            cursor.execute(
                "SELECT lanispl, pg_get_userbyid(lanowner) "
                "FROM pg_language WHERE lanname = 'sql'"
            )
            assert cursor.fetchone() == (False, cluster.outsider)
    finally:
        connection.close()

    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog"):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert _role_rows(cluster) == ()


def test_prepared_fresh_schema_object_is_rejected_before_role_mutation(
    bootstrap_cluster,
    monkeypatch,
):
    cluster = bootstrap_cluster
    context = cluster.context()
    first = cluster.bootstrap(with_credentials=False).reconcile(context)
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()
    interrupted = cluster.bootstrap(with_credentials=True)

    class SimulatedProcessStop(Exception):
        pass

    def stop_before_migrations(_context):
        raise SimulatedProcessStop

    monkeypatch.setattr(
        interrupted,
        "_apply_packaged_migrations",
        stop_before_migrations,
    )
    with pytest.raises(SimulatedProcessStop):
        interrupted.reconcile(context)

    expected_marker = f"elvis-postgres-bootstrap-schema:v1:{cluster.database}"
    exact_roles = _role_rows(cluster)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner), "
                "obj_description(oid, 'pg_namespace') "
                "FROM pg_namespace WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (cluster.roles.schema_owner, expected_marker)
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            assert cursor.fetchone() == (None,)
            cursor.execute(
                "CREATE TYPE np.bootstrap_prepared_fresh_enum AS ENUM ('sentinel')"
            )
    finally:
        connection.close()

    with pytest.raises(
        PostgresBootstrapDriftError,
        match="catalog|object|type",
    ):
        cluster.bootstrap(with_credentials=False).reconcile(context)

    assert _role_rows(cluster) == exact_roles
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regtype('np.bootstrap_prepared_fresh_enum')::text"
            )
            assert cursor.fetchone() == ("np.bootstrap_prepared_fresh_enum",)
            cursor.execute(
                "SELECT obj_description(oid, 'pg_namespace') "
                "FROM pg_namespace WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (expected_marker,)
    finally:
        connection.close()


def test_prepared_fresh_public_create_acl_is_rejected_before_migrations(
    bootstrap_cluster,
    monkeypatch,
):
    cluster = bootstrap_cluster
    context = cluster.context()
    first = cluster.bootstrap(with_credentials=False).reconcile(context)
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()
    interrupted = cluster.bootstrap(with_credentials=True)

    class SimulatedProcessStop(Exception):
        pass

    def stop_before_migrations(_context):
        raise SimulatedProcessStop

    monkeypatch.setattr(
        interrupted,
        "_apply_packaged_migrations",
        stop_before_migrations,
    )
    with pytest.raises(SimulatedProcessStop):
        interrupted.reconcile(context)

    exact_roles = _role_rows(cluster)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            assert cursor.fetchone() == (None,)
            cursor.execute("GRANT CREATE ON SCHEMA np TO PUBLIC")
            cursor.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_namespace namespace_row "
                "CROSS JOIN LATERAL aclexplode(namespace_row.nspacl) acl_row "
                "WHERE namespace_row.nspname = 'np' "
                "AND acl_row.grantee = 0 "
                "AND acl_row.privilege_type = 'CREATE'"
                ")"
            )
            assert cursor.fetchone() == (True,)
    finally:
        connection.close()

    before = _authority_snapshot(cluster)
    with pytest.raises(PostgresBootstrapDriftError, match="catalog|schema|authority"):
        cluster.bootstrap(with_credentials=False).reconcile(context)

    assert _role_rows(cluster) == exact_roles
    assert _authority_snapshot(cluster) == before
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "outside_catalog_drift",
    ["schema", "security_definer", "public_acl"],
)
def test_unexpected_user_schema_or_public_routine_is_rejected_without_repair(
    bootstrap_cluster,
    outside_catalog_drift,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if outside_catalog_drift == "schema":
                cursor.execute("CREATE SCHEMA extra_bootstrap_authority")
                cursor.execute(
                    "GRANT CREATE ON SCHEMA extra_bootstrap_authority TO PUBLIC"
                )
            elif outside_catalog_drift == "security_definer":
                cursor.execute("""
                    CREATE FUNCTION public.bootstrap_security_definer_probe()
                    RETURNS INTEGER
                    LANGUAGE sql
                    SECURITY DEFINER
                    SET search_path = pg_catalog
                    AS 'SELECT 1'
                    """)
            else:
                cursor.execute("REVOKE USAGE ON SCHEMA public FROM PUBLIC")
    finally:
        connection.close()

    before = _authority_snapshot(cluster)
    public_evidence = _public_schema_evidence(cluster)
    outside_schema = (
        "extra_bootstrap_authority" if outside_catalog_drift == "schema" else "public"
    )
    outside_evidence = _outside_schema_snapshot(cluster, outside_schema)
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            if outside_catalog_drift == "schema":
                cursor.execute(
                    "SELECT has_schema_privilege(%s, %s, 'CREATE')",
                    (cluster.roles.readiness, "extra_bootstrap_authority"),
                )
            elif outside_catalog_drift == "security_definer":
                cursor.execute(
                    "SELECT has_function_privilege("
                    "%s, 'public.bootstrap_security_definer_probe()', 'EXECUTE')",
                    (cluster.roles.readiness,),
                )
            else:
                cursor.execute(
                    "SELECT has_schema_privilege(%s, 'public', 'USAGE')",
                    (cluster.roles.readiness,),
                )
            assert cursor.fetchone() == (
                (False,) if outside_catalog_drift == "public_acl" else (True,)
            )
    finally:
        connection.close()

    with pytest.raises(
        PostgresBootstrapDriftError,
        match="catalog|schema|public object",
    ):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert _authority_snapshot(cluster) == before
    assert _public_schema_evidence(cluster) == public_evidence
    assert _outside_schema_snapshot(cluster, outside_schema) == outside_evidence
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            if outside_catalog_drift == "schema":
                cursor.execute("SELECT to_regnamespace('extra_bootstrap_authority')")
                assert cursor.fetchone() == ("extra_bootstrap_authority",)
            elif outside_catalog_drift == "security_definer":
                cursor.execute("""
                    SELECT prosecdef
                    FROM pg_proc
                    WHERE oid = (
                        'public.bootstrap_security_definer_probe()'::regprocedure
                    )
                """)
                assert cursor.fetchone() == (True,)
            else:
                assert public_evidence == (
                    ("pg_database_owner", "pg_database_owner", "CREATE", False),
                    ("pg_database_owner", "pg_database_owner", "USAGE", False),
                )
    finally:
        connection.close()


def test_runtime_created_large_object_is_rejected_without_repair(bootstrap_cluster):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    runtime = cluster.role_factory(cluster.roles.readiness)()
    runtime.autocommit = False
    try:
        with runtime.cursor() as cursor:
            cursor.execute("SELECT lo_create(0)")
            large_object_oid = cursor.fetchone()[0]
        runtime.commit()
    finally:
        runtime.close()

    def large_object_evidence():
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT oid, pg_get_userbyid(lomowner), lomacl::text "
                    "FROM pg_largeobject_metadata WHERE oid = %s",
                    (large_object_oid,),
                )
                return cursor.fetchone()
        finally:
            connection.close()

    before = _authority_snapshot(cluster)
    evidence = large_object_evidence()
    assert evidence == (large_object_oid, cluster.roles.readiness, None)

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|large object"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert _authority_snapshot(cluster) == before
    assert large_object_evidence() == evidence


def test_standalone_enum_in_np_is_rejected_without_repair(bootstrap_cluster):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    migrator = cluster.role_factory(cluster.roles.migrator)()
    migrator.autocommit = False
    try:
        with migrator.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET ROLE {}").format(
                    sql.Identifier(cluster.roles.schema_owner)
                )
            )
            cursor.execute(
                "CREATE TYPE np.hidden_bootstrap_enum "
                "AS ENUM ('sentinel_first', 'sentinel_second')"
            )
        migrator.commit()
    finally:
        migrator.close()

    def enum_evidence():
        connection = cluster.admin_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT pg_get_userbyid(type_row.typowner),
                           type_row.typtype,
                           type_row.typacl::text,
                           ARRAY(
                               SELECT enum_row.enumlabel::text
                               FROM pg_enum enum_row
                               WHERE enum_row.enumtypid = type_row.oid
                               ORDER BY enum_row.enumsortorder
                           )
                    FROM pg_type type_row
                    WHERE type_row.oid = 'np.hidden_bootstrap_enum'::regtype
                    """)
                return cursor.fetchone()
        finally:
            connection.close()

    before = _authority_snapshot(cluster)
    evidence = enum_evidence()
    assert evidence == (
        cluster.roles.schema_owner,
        "e",
        None,
        ["sentinel_first", "sentinel_second"],
    )

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|type"):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())

    assert _authority_snapshot(cluster) == before
    assert enum_evidence() == evidence


@pytest.mark.parametrize(
    "drift_kind",
    [
        "unexpected_object",
        "third_party_table_grant",
        "third_party_function_grant",
        "third_party_schema_grant",
        "third_party_database_grant",
        "third_party_default_acl",
        "third_party_membership",
        "public_table_grant",
        "public_function_grant",
        "managed_table_surplus",
        "managed_schema_surplus",
        "managed_database_surplus",
        "managed_function_surplus",
        "column_acl",
        "third_party_column_acl",
        "table_grant_option",
        "column_grant_option",
        "schema_grant_option",
        "database_grant_option",
        "function_grant_option",
        "unexpected_unique_index",
        "unexpected_partial_index",
        "unexpected_expression_index",
    ],
)
def test_reconciliation_rejects_catalog_or_authority_expansion(
    bootstrap_cluster,
    drift_kind,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if drift_kind == "unexpected_object":
                cursor.execute("CREATE TABLE np.unexpected_authority (id INTEGER)")
            if drift_kind not in {
                "unexpected_object",
                "public_table_grant",
                "public_function_grant",
                "managed_table_surplus",
                "managed_schema_surplus",
                "managed_database_surplus",
                "managed_function_surplus",
                "column_acl",
                "table_grant_option",
                "column_grant_option",
                "schema_grant_option",
                "database_grant_option",
                "function_grant_option",
                "unexpected_unique_index",
                "unexpected_partial_index",
                "unexpected_expression_index",
            }:
                cluster.create_auxiliary_login(cluster.outsider)
            if drift_kind == "third_party_table_grant":
                cursor.execute(
                    sql.SQL("GRANT SELECT ON np.trades TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
            elif drift_kind == "third_party_function_grant":
                cursor.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION "
                        "np.enforce_legacy_paper_runtime_fence() TO {}"
                    ).format(sql.Identifier(cluster.outsider))
                )
            elif drift_kind == "third_party_schema_grant":
                cursor.execute(
                    sql.SQL("GRANT CREATE ON SCHEMA np TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
            elif drift_kind == "third_party_database_grant":
                cursor.execute(
                    sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                        sql.Identifier(cluster.database),
                        sql.Identifier(cluster.outsider),
                    )
                )
            elif drift_kind == "third_party_default_acl":
                cursor.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA np "
                        "GRANT SELECT ON TABLES TO {}"
                    ).format(
                        sql.Identifier(cluster.roles.schema_owner),
                        sql.Identifier(cluster.outsider),
                    )
                )
            elif drift_kind == "third_party_membership":
                cursor.execute(
                    sql.SQL("GRANT {} TO {}").format(
                        sql.Identifier(cluster.roles.readiness),
                        sql.Identifier(cluster.outsider),
                    )
                )
            elif drift_kind == "public_table_grant":
                cursor.execute("GRANT SELECT ON np.trades TO PUBLIC")
            elif drift_kind == "public_function_grant":
                cursor.execute(
                    "GRANT EXECUTE ON FUNCTION "
                    "np.enforce_legacy_paper_runtime_fence() TO PUBLIC"
                )
            elif drift_kind == "managed_table_surplus":
                cursor.execute(
                    sql.SQL("GRANT DELETE ON np.trades TO {}").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
            elif drift_kind == "managed_schema_surplus":
                cursor.execute(
                    sql.SQL("GRANT CREATE ON SCHEMA np TO {}").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
            elif drift_kind == "managed_database_surplus":
                cursor.execute(
                    sql.SQL("GRANT TEMP ON DATABASE {} TO {}").format(
                        sql.Identifier(cluster.database),
                        sql.Identifier(cluster.roles.readiness),
                    )
                )
            elif drift_kind == "managed_function_surplus":
                cursor.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION "
                        "np.enforce_legacy_paper_runtime_fence() TO {}"
                    ).format(sql.Identifier(cluster.roles.readiness))
                )
            elif drift_kind == "column_acl":
                cursor.execute(
                    sql.SQL("GRANT UPDATE(symbol) ON np.trades TO {}").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
            elif drift_kind == "third_party_column_acl":
                cursor.execute(
                    sql.SQL("GRANT UPDATE(symbol) ON np.trades TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
            elif drift_kind == "table_grant_option":
                cursor.execute(
                    sql.SQL("GRANT SELECT ON np.trades TO {} WITH GRANT OPTION").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
            elif drift_kind == "column_grant_option":
                cursor.execute(
                    sql.SQL(
                        "GRANT SELECT(symbol) ON np.trades TO {} WITH GRANT OPTION"
                    ).format(sql.Identifier(cluster.roles.readiness))
                )
            elif drift_kind == "schema_grant_option":
                cursor.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA np TO {} WITH GRANT OPTION").format(
                        sql.Identifier(cluster.roles.readiness)
                    )
                )
            elif drift_kind == "database_grant_option":
                cursor.execute(
                    sql.SQL(
                        "GRANT CONNECT ON DATABASE {} TO {} WITH GRANT OPTION"
                    ).format(
                        sql.Identifier(cluster.database),
                        sql.Identifier(cluster.roles.readiness),
                    )
                )
            elif drift_kind == "function_grant_option":
                cursor.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION "
                        "np.acquire_paper_runtime_activation_fence() "
                        "TO {} WITH GRANT OPTION"
                    ).format(sql.Identifier(cluster.roles.readiness))
                )
            elif drift_kind == "unexpected_unique_index":
                cursor.execute(
                    "CREATE UNIQUE INDEX unexpected_unique_acl_idx "
                    "ON np.trades (id, symbol)"
                )
            elif drift_kind == "unexpected_partial_index":
                cursor.execute(
                    "CREATE INDEX unexpected_partial_acl_idx "
                    "ON np.trades (symbol) WHERE symbol IS NOT NULL"
                )
            elif drift_kind == "unexpected_expression_index":
                cursor.execute(
                    "CREATE INDEX unexpected_expression_acl_idx "
                    "ON np.trades ((lower(symbol)))"
                )
    finally:
        connection.close()

    with pytest.raises(PostgresBootstrapDriftError):
        cluster.bootstrap(with_credentials=True).reconcile(cluster.context())


@pytest.mark.parametrize("ledger_drift", ["pending", "checksum"])
def test_existing_volume_rejects_pending_or_drifted_migration_history(
    bootstrap_cluster,
    ledger_drift,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = False
    try:
        assert apply_migrations(connection, load_migrations()) == (1, 2, 3, 4, 5, 6)
        with connection.cursor() as cursor:
            if ledger_drift == "pending":
                cursor.execute("DELETE FROM np.schema_migrations WHERE version = 6")
            else:
                cursor.execute(
                    "UPDATE np.schema_migrations "
                    "SET checksum = repeat('0', 64) WHERE version = 3"
                )
        connection.commit()
    finally:
        connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.admin_role,
        allowed_historical_owner_roles=(cluster.admin_role,),
    )
    context = cluster.context(adoption=adoption)
    before = _authority_snapshot(cluster)
    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapMigrationError, match="history"):
        cluster.bootstrap(with_credentials=False).reconcile(context)

    assert _role_rows(cluster) == ()
    assert _authority_snapshot(cluster) == before


def test_migration_authority_cannot_own_plpgsql_before_role_staging(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.old_runtime)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.old_runtime),
                )
            )
    finally:
        connection.close()

    old_connection = cluster.role_factory(cluster.old_runtime)()
    old_connection.autocommit = False
    try:
        assert apply_migrations(old_connection, load_migrations()) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
    finally:
        old_connection.close()

    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER LANGUAGE plpgsql OWNER TO {}").format(
                    sql.Identifier(cluster.old_runtime)
                )
            )
            cursor.execute(
                "SELECT pg_get_userbyid(lanowner) "
                "FROM pg_language WHERE lanname = 'plpgsql'"
            )
            assert cursor.fetchone() == (cluster.old_runtime,)
    finally:
        connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
    )
    context = cluster.context(adoption=adoption)
    before = _authority_snapshot(cluster)
    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|language|owner"):
        cluster.bootstrap(with_credentials=False).reconcile(context)

    assert _role_rows(cluster) == ()
    assert _authority_snapshot(cluster) == before
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(lanowner) "
                "FROM pg_language WHERE lanname = 'plpgsql'"
            )
            assert cursor.fetchone() == (cluster.old_runtime,)
    finally:
        connection.close()


def test_migration_authority_plpgsql_owner_never_returns_false_complete(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    assert _complete_fresh_bootstrap(cluster).status is PostgresBootstrapStatus.COMPLETE
    cluster.create_auxiliary_login(cluster.old_runtime)
    exact_roles = _role_rows(cluster)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER LANGUAGE plpgsql OWNER TO {}").format(
                    sql.Identifier(cluster.old_runtime)
                )
            )
    finally:
        connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
    )
    context = cluster.context(adoption=adoption)

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|language|owner"):
        cluster.bootstrap(with_credentials=True).reconcile(context)

    assert _role_rows(cluster) == exact_roles
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(lanowner) "
                "FROM pg_language WHERE lanname = 'plpgsql'"
            )
            assert cursor.fetchone() == (cluster.old_runtime,)
    finally:
        connection.close()


@pytest.mark.parametrize("wrong_owner", ["schema", "ledger"])
def test_adoption_requires_migration_authority_to_own_schema_and_ledger(
    bootstrap_cluster,
    wrong_owner,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.old_runtime)
    cluster.create_auxiliary_login(cluster.outsider)
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.old_runtime),
                )
            )
    finally:
        connection.close()
    old_connection = cluster.role_factory(cluster.old_runtime)()
    old_connection.autocommit = False
    try:
        assert apply_migrations(old_connection, load_migrations()) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
    finally:
        old_connection.close()

    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.old_runtime),
                )
            )
            if wrong_owner == "schema":
                cursor.execute(
                    sql.SQL("ALTER SCHEMA np OWNER TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
            else:
                cursor.execute(
                    sql.SQL("ALTER TABLE np.schema_migrations OWNER TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
    finally:
        connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
    )
    context = cluster.context(adoption=adoption)
    before = _authority_snapshot(cluster)
    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="owner|authority|catalog"):
        cluster.bootstrap(with_credentials=False).reconcile(context)

    assert _role_rows(cluster) == ()
    assert _authority_snapshot(cluster) == before
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT pg_get_userbyid(namespace_row.nspowner),
                       pg_get_userbyid(ledger_row.relowner)
                FROM pg_namespace namespace_row
                JOIN pg_class ledger_row
                  ON ledger_row.relnamespace = namespace_row.oid
                 AND ledger_row.relname = 'schema_migrations'
                WHERE namespace_row.nspname = 'np'
                """)
            assert cursor.fetchone() == (
                (cluster.outsider if wrong_owner == "schema" else cluster.old_runtime),
                (cluster.outsider if wrong_owner == "ledger" else cluster.old_runtime),
            )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "historical_surplus",
    ["table", "schema", "database", "function"],
)
def test_allowed_historical_owner_cannot_retain_surplus_authority(
    bootstrap_cluster,
    historical_surplus,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.outsider)
    connection = cluster.admin_factory()
    connection.autocommit = False
    try:
        assert apply_migrations(connection, load_migrations()) == (1, 2, 3, 4, 5, 6)
        connection.commit()
    finally:
        connection.close()
    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.admin_role,
        allowed_historical_owner_roles=(cluster.admin_role,),
    )
    context = cluster.context(adoption=adoption)
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(context).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if historical_surplus == "table":
                cursor.execute(
                    sql.SQL("GRANT SELECT ON np.trades TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
            elif historical_surplus == "schema":
                cursor.execute(
                    sql.SQL("GRANT CREATE ON SCHEMA np TO {}").format(
                        sql.Identifier(cluster.outsider)
                    )
                )
            elif historical_surplus == "database":
                cursor.execute(
                    sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                        sql.Identifier(cluster.database),
                        sql.Identifier(cluster.outsider),
                    )
                )
            else:
                cursor.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION "
                        "np.enforce_legacy_paper_runtime_fence() TO {}"
                    ).format(sql.Identifier(cluster.outsider))
                )
    finally:
        connection.close()

    with pytest.raises(PostgresBootstrapDriftError):
        cluster.bootstrap(with_credentials=True).reconcile(context)

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner) FROM pg_namespace "
                "WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (cluster.admin_role,)
    finally:
        connection.close()


def test_legacy_tables_without_migration_ledger_are_rejected_without_mutation(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA np")
            cursor.execute(
                "CREATE TABLE np.trades (id SERIAL PRIMARY KEY, symbol TEXT)"
            )
            cursor.execute(
                "INSERT INTO np.trades (id, symbol) VALUES (950001, 'LEGACY-SENTINEL')"
            )
        connection.commit()
    finally:
        connection.close()
    before = _authority_snapshot(cluster)
    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.admin_role,
        allowed_historical_owner_roles=(cluster.admin_role,),
    )
    context = cluster.context(adoption=adoption)
    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapMigrationError, match="history"):
        cluster.bootstrap(with_credentials=False).reconcile(context)

    assert _role_rows(cluster) == ()
    assert _authority_snapshot(cluster) == before
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT symbol FROM np.trades WHERE id = 950001")
            assert cursor.fetchone() == ("LEGACY-SENTINEL",)
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_fresh_context_rejects_preexisting_np_schema_without_owner_mutation(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    connection = cluster.admin_factory()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA np")
            cursor.execute("CREATE TABLE np.preexisting_sentinel (id INTEGER)")
    finally:
        connection.close()
    before = _authority_snapshot(cluster)
    assert _role_rows(cluster) == ()

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|objects|owners"):
        cluster.bootstrap(with_credentials=False).reconcile(cluster.context())

    assert _role_rows(cluster) == ()
    assert _authority_snapshot(cluster) == before
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner) FROM pg_namespace "
                "WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (cluster.admin_role,)
            cursor.execute("SELECT to_regclass('np.preexisting_sentinel')")
            assert cursor.fetchone() == ("np.preexisting_sentinel",)
    finally:
        connection.close()


def test_fresh_bootstrap_resumes_after_migration_authority_commit(
    bootstrap_cluster,
    monkeypatch,
):
    cluster = bootstrap_cluster
    context = cluster.context()
    assert (
        cluster.bootstrap(with_credentials=False).reconcile(context).status
        is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    )
    cluster.provision_managed_passwords()
    exact_roles = _role_rows(cluster)
    interrupted = cluster.bootstrap(with_credentials=True)

    class SimulatedProcessStop(Exception):
        pass

    def stop_before_migrations(_context):
        raise SimulatedProcessStop

    monkeypatch.setattr(
        interrupted,
        "_apply_packaged_migrations",
        stop_before_migrations,
    )
    with pytest.raises(SimulatedProcessStop):
        interrupted.reconcile(context)

    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner) FROM pg_namespace "
                "WHERE nspname = 'np'"
            )
            assert cursor.fetchone() == (cluster.roles.schema_owner,)
            cursor.execute("SELECT to_regclass('np.schema_migrations')")
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()

    receipt = cluster.bootstrap(with_credentials=True).reconcile(context)

    assert receipt.status is PostgresBootstrapStatus.COMPLETE
    assert receipt.migration_versions == (1, 2, 3, 4, 5, 6)
    assert _role_rows(cluster) == exact_roles


def test_existing_volume_requires_explicit_quiescence_before_old_role_demotion(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    cluster.create_auxiliary_login(cluster.old_runtime)
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(cluster.database),
                    sql.Identifier(cluster.old_runtime),
                )
            )
    finally:
        admin.close()
    old_connection = cluster.role_factory(cluster.old_runtime)()
    old_connection.autocommit = False
    try:
        assert apply_migrations(old_connection, load_migrations()) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )
        with old_connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO np.trades (id, symbol, side) "
                "VALUES (930001, 'SENTINELUSDT', 'BUY')"
            )
        old_connection.commit()
    finally:
        old_connection.close()

    adoption = PostgresBootstrapAdoption(
        migration_authority_role=cluster.old_runtime,
        allowed_historical_owner_roles=(cluster.old_runtime,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=False,
    )
    context = cluster.context(adoption=adoption)
    first = cluster.bootstrap(with_credentials=False).reconcile(context)
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()

    staged = cluster.bootstrap(with_credentials=True).reconcile(context)

    assert staged.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert staged.old_shared_runtime_demoted is False
    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT rolcanlogin, has_database_privilege(%s, %s, 'CONNECT') "
                "FROM pg_roles WHERE rolname = %s",
                (cluster.old_runtime, cluster.database, cluster.old_runtime),
            )
            assert cursor.fetchone() == (True, True)
    finally:
        admin.close()

    catalog_before_demotion = _authority_snapshot(cluster)
    active_old_backend = cluster.role_factory(cluster.old_runtime)()
    try:
        demotion = PostgresBootstrapAdoption(
            migration_authority_role=cluster.old_runtime,
            allowed_historical_owner_roles=(cluster.old_runtime,),
            old_shared_runtime_role=cluster.old_runtime,
            demote_old_shared_runtime=True,
        )
        demoted = cluster.bootstrap(with_credentials=True).reconcile(
            cluster.context(adoption=demotion)
        )

        assert demoted.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
        assert demoted.old_shared_runtime_demoted is False
        assert _authority_snapshot(cluster) == catalog_before_demotion

        admin = cluster.admin_factory()
        try:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT rolcanlogin, rolsuper, rolinherit, rolcreaterole, "
                    "rolcreatedb, rolreplication, rolbypassrls, rolconnlimit, "
                    "rolpassword IS NULL "
                    "FROM pg_authid WHERE rolname = %s",
                    (cluster.old_runtime,),
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
                )
        finally:
            admin.close()
        with pytest.raises(psycopg2.OperationalError):
            cluster.role_factory(cluster.old_runtime)()
        with active_old_backend.cursor() as cursor:
            cursor.execute("SELECT symbol FROM np.trades WHERE id = 930001")
            assert cursor.fetchone() == ("SENTINELUSDT",)

        waiting = cluster.bootstrap(with_credentials=True).reconcile(
            cluster.context(adoption=demotion)
        )
        assert waiting.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
        assert waiting.old_shared_runtime_demoted is False
        assert _authority_snapshot(cluster) == catalog_before_demotion
    finally:
        active_old_backend.close()

    completed = cluster.bootstrap(with_credentials=True).reconcile(
        cluster.context(adoption=demotion)
    )
    assert completed.status is PostgresBootstrapStatus.COMPLETE
    assert completed.old_shared_runtime_demoted is True
    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT rolcanlogin, has_database_privilege(%s, %s, 'CONNECT') "
                "FROM pg_roles WHERE rolname = %s",
                (cluster.old_runtime, cluster.database, cluster.old_runtime),
            )
            assert cursor.fetchone() == (False, False)
            cursor.execute("SELECT symbol, side FROM np.trades WHERE id = 930001")
            assert cursor.fetchone() == ("SENTINELUSDT", "BUY")
    finally:
        admin.close()


def test_demotion_not_requested_still_preflights_catalog_without_mutation(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    adoption = _stage_existing_adoption(cluster)
    cluster.create_auxiliary_login(cluster.outsider)
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("GRANT SELECT ON np.trades TO {}").format(
                    sql.Identifier(cluster.outsider)
                )
            )
    finally:
        admin.close()
    context = cluster.context(adoption=adoption)
    before = _authority_snapshot(cluster)

    with pytest.raises(PostgresBootstrapDriftError, match="catalog|demotion"):
        cluster.bootstrap(with_credentials=True).reconcile(context)

    assert _authority_snapshot(cluster) == before
    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT rolcanlogin FROM pg_roles WHERE rolname = %s",
                (cluster.old_runtime,),
            )
            assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT has_table_privilege(%s, 'np.trades', 'SELECT')",
                (cluster.outsider,),
            )
            assert cursor.fetchone() == (True,)
    finally:
        admin.close()


def test_old_membership_injected_at_cutover_is_drift_without_catalog_mutation(
    bootstrap_cluster,
    monkeypatch,
):
    cluster = bootstrap_cluster
    staged = _stage_existing_adoption(cluster)
    cluster.create_auxiliary_login(cluster.outsider)
    adoption = PostgresBootstrapAdoption(
        migration_authority_role=staged.migration_authority_role,
        allowed_historical_owner_roles=staged.allowed_historical_owner_roles,
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=True,
    )
    context = cluster.context(adoption=adoption)
    disabled = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert disabled.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert disabled.old_shared_runtime_demoted is False
    before = _authority_snapshot(cluster)
    bootstrap = cluster.bootstrap(with_credentials=True)
    demote_old_role = bootstrap._demote_old_role
    injected = {"done": False}

    def inject_membership(cursor, cutover_context):
        admin = cluster.admin_factory()
        admin.autocommit = True
        try:
            with admin.cursor() as membership_cursor:
                membership_cursor.execute(
                    sql.SQL("GRANT {} TO {}").format(
                        sql.Identifier(cluster.outsider),
                        sql.Identifier(cluster.old_runtime),
                    )
                )
        finally:
            admin.close()
        injected["done"] = True
        return demote_old_role(cursor, cutover_context)

    monkeypatch.setattr(bootstrap, "_demote_old_role", inject_membership)

    with pytest.raises(PostgresBootstrapDriftError, match="memberships.*cutover"):
        bootstrap.reconcile(context)

    assert injected["done"] is True
    assert _authority_snapshot(cluster) == before
    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_auth_members membership "
                "JOIN pg_roles parent ON parent.oid = membership.roleid "
                "JOIN pg_roles member ON member.oid = membership.member "
                "WHERE parent.rolname = %s AND member.rolname = %s)",
                (cluster.outsider, cluster.old_runtime),
            )
            assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT pg_get_userbyid(datdba) " "FROM pg_database WHERE datname = %s",
                (cluster.database,),
            )
            assert cursor.fetchone() == (cluster.admin_role,)
    finally:
        admin.close()


@pytest.mark.parametrize("membership_direction", ["incoming", "outgoing"])
def test_old_role_membership_blocks_login_disable_without_mutation(
    bootstrap_cluster,
    membership_direction,
):
    cluster = bootstrap_cluster
    staged = _stage_existing_adoption(cluster)
    cluster.create_auxiliary_login(cluster.outsider)
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            parent, member = (
                (cluster.old_runtime, cluster.outsider)
                if membership_direction == "incoming"
                else (cluster.outsider, cluster.old_runtime)
            )
            cursor.execute(
                sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(parent),
                    sql.Identifier(member),
                )
            )
    finally:
        admin.close()
    context = cluster.context(
        adoption=PostgresBootstrapAdoption(
            migration_authority_role=staged.migration_authority_role,
            allowed_historical_owner_roles=(staged.migration_authority_role,),
            old_shared_runtime_role=cluster.old_runtime,
            demote_old_shared_runtime=True,
        )
    )
    before = _authority_snapshot(cluster)

    with pytest.raises(PostgresBootstrapDriftError, match="membership"):
        cluster.bootstrap(with_credentials=True).reconcile(context)

    assert _authority_snapshot(cluster) == before
    admin = cluster.admin_factory()
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT rolcanlogin, rolsuper, rolinherit, rolcreaterole, "
                "rolcreatedb, rolreplication, rolbypassrls, rolconnlimit, "
                "rolpassword IS NULL "
                "FROM pg_authid WHERE rolname = %s",
                (cluster.old_runtime,),
            )
            assert cursor.fetchone() == (
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                -1,
                False,
            )
            cursor.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_auth_members membership "
                "JOIN pg_roles parent_role "
                "ON parent_role.oid = membership.roleid "
                "JOIN pg_roles member_role "
                "ON member_role.oid = membership.member "
                "WHERE parent_role.rolname = %s "
                "AND member_role.rolname = %s)",
                (parent, member),
            )
            assert cursor.fetchone() == (True,)
    finally:
        admin.close()


@pytest.mark.parametrize("old_role_drift", ["table_acl", "membership"])
def test_demoted_historical_role_cannot_retain_or_regain_authority(
    bootstrap_cluster,
    old_role_drift,
):
    cluster = bootstrap_cluster
    staged = _stage_existing_adoption(cluster)
    demotion = PostgresBootstrapAdoption(
        migration_authority_role=staged.migration_authority_role,
        allowed_historical_owner_roles=(staged.migration_authority_role,),
        old_shared_runtime_role=cluster.old_runtime,
        demote_old_shared_runtime=True,
    )
    context = cluster.context(adoption=demotion)
    demoted = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert demoted.status is PostgresBootstrapStatus.DEMOTION_REQUIRED
    assert demoted.old_shared_runtime_demoted is False
    completed = cluster.bootstrap(with_credentials=True).reconcile(context)
    assert completed.status is PostgresBootstrapStatus.COMPLETE
    assert completed.old_shared_runtime_demoted is True
    admin = cluster.admin_factory()
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            if old_role_drift == "table_acl":
                cursor.execute(
                    sql.SQL("GRANT SELECT ON np.trades TO {}").format(
                        sql.Identifier(cluster.old_runtime)
                    )
                )
            else:
                cluster.create_auxiliary_login(cluster.outsider)
                cursor.execute(
                    sql.SQL("GRANT {} TO {}").format(
                        sql.Identifier(cluster.outsider),
                        sql.Identifier(cluster.old_runtime),
                    )
                )
    finally:
        admin.close()

    with pytest.raises(PostgresBootstrapDriftError):
        cluster.bootstrap(with_credentials=True).reconcile(context)


def test_migration_commit_then_raise_is_resolved_from_the_durable_ledger(
    bootstrap_cluster,
):
    cluster = bootstrap_cluster
    first = cluster.bootstrap(with_credentials=False).reconcile(cluster.context())
    assert first.status is PostgresBootstrapStatus.CREDENTIALS_REQUIRED
    cluster.provision_managed_passwords()
    commit_state = {"raised": False}

    def uncertain_migrator_factory():
        return CommitThenRaiseConnection(
            cluster.role_factory(cluster.roles.migrator)(), commit_state
        )

    bootstrap = PostgresBootstrap(
        cluster.admin_factory,
        migrator_connection_factory=uncertain_migrator_factory,
        legacy_runtime_connection_factory=cluster.role_factory(
            cluster.roles.legacy_runtime
        ),
        atomic_runtime_connection_factory=cluster.role_factory(
            cluster.roles.atomic_runtime
        ),
        activation_connection_factory=cluster.role_factory(cluster.roles.activation),
        readiness_connection_factory=cluster.role_factory(cluster.roles.readiness),
        trainer_connection_factory=cluster.role_factory(cluster.roles.trainer),
    )

    receipt = bootstrap.reconcile(cluster.context())

    assert commit_state == {"raised": True}
    assert receipt.status is PostgresBootstrapStatus.COMPLETE
    assert receipt.migration_versions == (1, 2, 3, 4, 5, 6)
    connection = cluster.admin_factory()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version FROM np.schema_migrations ORDER BY version")
            assert cursor.fetchall() == [(1,), (2,), (3,), (4,), (5,), (6,)]
    finally:
        connection.close()
