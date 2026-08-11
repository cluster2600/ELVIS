"""Fixtures for PostgreSQL tests that may only use disposable databases."""

import os
import uuid
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

from trading.persistence import apply_migrations, load_migrations

_ADMIN_DSN_ENV = "ELVIS_TEST_POSTGRES_ADMIN_DSN"
_REQUIRED_ENV = "ELVIS_TEST_POSTGRES_REQUIRED"
_DATABASE_PREFIX = "elvis_pytest_"
_POSTGRES_TEST_ROOT = Path(__file__).parent.resolve()


def _dsn_identity(dsn):
    return frozenset(parse_dsn(dsn).items())


def pytest_collection_modifyitems(items):
    """Mark every test collected below this directory as PostgreSQL-backed."""
    for item in items:
        item_path = Path(str(item.path)).resolve()
        if _POSTGRES_TEST_ROOT in item_path.parents:
            item.add_marker(pytest.mark.postgres)


@pytest.fixture(scope="session")
def postgres_admin_dsn():
    """Return only the explicitly configured administrative test DSN."""
    admin_dsn = os.getenv(_ADMIN_DSN_ENV)
    if admin_dsn:
        try:
            parse_dsn(admin_dsn)
        except Exception:
            pytest.fail(
                f"{_ADMIN_DSN_ENV} is not a valid PostgreSQL DSN", pytrace=False
            )
        return admin_dsn

    message = f"{_ADMIN_DSN_ENV} is required for isolated PostgreSQL tests"
    if os.getenv(_REQUIRED_ENV) == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


@pytest.fixture
def postgres_database_dsn(postgres_admin_dsn, postgres_connection_allowlist):
    """Create one empty UUID-named database and drop only that exact database."""
    database_name = f"{_DATABASE_PREFIX}{uuid.uuid4().hex}"
    target_parameters = parse_dsn(postgres_admin_dsn)
    target_parameters["dbname"] = database_name
    target_dsn = make_dsn(**target_parameters)
    admin_identity = _dsn_identity(postgres_admin_dsn)
    target_identity = _dsn_identity(target_dsn)

    admin_connection = None
    postgres_connection_allowlist.add(admin_identity)
    try:
        admin_connection = psycopg2.connect(postgres_admin_dsn)
        admin_connection.autocommit = True
        with admin_connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = %s)",
                (database_name,),
            )
            if cursor.fetchone()[0]:
                pytest.fail("generated PostgreSQL test database already exists")
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
            )
    finally:
        if admin_connection is not None:
            admin_connection.close()
        postgres_connection_allowlist.discard(admin_identity)

    postgres_connection_allowlist.add(target_identity)
    try:
        yield target_dsn
    finally:
        postgres_connection_allowlist.discard(target_identity)
        postgres_connection_allowlist.add(admin_identity)
        cleanup_connection = None
        try:
            cleanup_connection = psycopg2.connect(postgres_admin_dsn)
            cleanup_connection.autocommit = True
            with cleanup_connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                        sql.Identifier(database_name)
                    )
                )
        finally:
            if cleanup_connection is not None:
                cleanup_connection.close()
            postgres_connection_allowlist.discard(admin_identity)


@pytest.fixture
def migrated_postgres_dsn(postgres_database_dsn):
    """Apply packaged migrations, then return the disposable database DSN."""
    connection = psycopg2.connect(postgres_database_dsn)
    connection.autocommit = False
    try:
        assert apply_migrations(connection, load_migrations()) == (1,)
    finally:
        connection.close()
    return postgres_database_dsn
