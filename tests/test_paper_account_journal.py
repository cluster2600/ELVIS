"""Fast contract tests for the dormant paper-account PostgreSQL repository."""

import ast
import copy
import importlib.util
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from psycopg2.extensions import STATUS_READY, TRANSACTION_STATUS_IDLE

from trading.domain.paper_accounting import (
    PaperAccountBalance,
    PaperAccountPolicy,
    new_paper_account,
)
from trading.persistence.paper_account_journal import (
    PaperAccountCommitUnknown,
    PaperAccountConflictError,
    PaperAccountConflictKind,
    PaperAccountInputError,
    PaperAccountNotFoundError,
    PaperAccountReplayError,
    PaperAccountStorageError,
    PostgresPaperAccountJournal,
    ProvisionDisposition,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _account(
    *,
    account_key="account-1",
    margin_quantum=Decimal("0.0100"),
    available=Decimal("100.00"),
):
    return new_paper_account(
        PaperAccountPolicy(account_key, "USDT", margin_quantum),
        (
            PaperAccountBalance("BNB", Decimal("2.500"), Decimal("0")),
            PaperAccountBalance("USDT", available, Decimal("0.00")),
        ),
    )


class MemoryDatabase:
    def __init__(self):
        self.streams = {}
        self.balances = {}
        self.connections = []
        self.fail_execute = False
        self.commit_then_raise = False

    def connect(self):
        connection = MemoryConnection(self)
        self.connections.append(connection)
        return connection


class MemoryConnection:
    autocommit = False
    status = STATUS_READY

    def __init__(self, database):
        self.database = database
        self.streams = copy.deepcopy(database.streams)
        self.balances = copy.deepcopy(database.balances)
        self.commands = []
        self.cursor_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def get_transaction_status(self):
        return TRANSACTION_STATUS_IDLE

    def cursor(self):
        self.cursor_calls += 1
        return MemoryCursor(self)

    def commit(self):
        self.commits += 1
        self.database.streams = copy.deepcopy(self.streams)
        self.database.balances = copy.deepcopy(self.balances)
        if self.database.commit_then_raise:
            raise RuntimeError("lost acknowledgement after commit")

    def rollback(self):
        self.rollbacks += 1
        self.streams = copy.deepcopy(self.database.streams)
        self.balances = copy.deepcopy(self.database.balances)

    def close(self):
        self.closed = True


class MemoryCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, parameters=None):
        if self.connection.database.fail_execute:
            raise RuntimeError("database unavailable")
        sql = " ".join(statement.split())
        params = parameters or ()
        self.connection.commands.append(sql)
        self.rows = []

        if sql.startswith("SET TRANSACTION"):
            return
        if sql.startswith("INSERT INTO np.paper_account_streams"):
            (
                account_key,
                execution_scope,
                owner_generation,
                collateral_asset,
                opening_version,
                opening_payload,
                opening_payload_sha256,
            ) = params
            if account_key not in self.connection.streams:
                self.connection.streams[account_key] = {
                    "account_key": account_key,
                    "execution_scope": execution_scope,
                    "owner_generation": owner_generation,
                    "collateral_asset": collateral_asset,
                    "account_version": 0,
                    "account_state": "ACTIVE",
                    "opening_version": opening_version,
                    "opening_payload": json.loads(opening_payload),
                    "opening_payload_sha256": opening_payload_sha256,
                    "created_at": NOW,
                    "updated_at": NOW,
                }
                self.rows = [(account_key,)]
            return
        if sql.startswith("INSERT INTO np.paper_account_balances"):
            account_key, asset, available, reserved = params
            self.connection.balances.setdefault(account_key, []).append(
                {
                    "asset": asset,
                    "available": available,
                    "reserved": reserved,
                    "updated_at": NOW,
                }
            )
            return
        if sql.startswith("SELECT account_key FROM np.paper_account_streams"):
            scope = params[0]
            self.rows = [
                (account_key,)
                for account_key, row in reversed(tuple(self.connection.streams.items()))
                if row["execution_scope"] == scope
            ]
            return
        if "FROM np.paper_account_streams" in sql:
            row = self.connection.streams.get(params[0])
            if row is not None:
                self.rows = [
                    (
                        row["account_key"],
                        row["execution_scope"],
                        row["owner_generation"],
                        row["collateral_asset"],
                        row["account_version"],
                        row["account_state"],
                        row["opening_version"],
                        row["opening_payload"],
                        row["opening_payload_sha256"],
                        row["created_at"],
                        row["updated_at"],
                    )
                ]
            return
        if "FROM np.paper_account_balances" in sql:
            self.rows = [
                (
                    row["asset"],
                    row["available"],
                    row["reserved"],
                    row["updated_at"],
                )
                for row in reversed(self.connection.balances.get(params[0], []))
            ]
            return
        if "FROM np.paper_account_batch_manifests" in sql:
            return
        if "FROM np.paper_account_settlements" in sql:
            return
        if "FROM np.paper_margin_reservations" in sql:
            return
        if "FROM np.paper_account_postings" in sql:
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


@pytest.fixture
def journal():
    database = MemoryDatabase()
    return database, PostgresPaperAccountJournal(database.connect)


def test_provision_commits_opening_then_exact_retry_is_existing(journal):
    database, repository = journal
    account = _account()

    created = repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=account,
    )
    existing = repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=account,
    )

    assert created.disposition is ProvisionDisposition.CREATED
    assert created.is_created is True
    assert existing.disposition is ProvisionDisposition.EXISTING
    assert existing.is_created is False
    assert created.account == existing.account == account
    assert created.execution_scope == "paper:test"
    assert created.owner_generation == 7
    assert created.current.batches == ()
    assert tuple(
        (row["asset"], row["available"], row["reserved"])
        for row in database.balances["account-1"]
    ) == (
        ("BNB", "2.500", "0"),
        ("USDT", "100.00", "0.00"),
    )
    assert len(database.streams) == 1
    assert len(database.balances["account-1"]) == 2
    assert all(connection.closed for connection in database.connections)


@pytest.mark.parametrize(
    ("execution_scope", "owner_generation", "account", "kind"),
    (
        (
            "paper:other",
            7,
            _account(),
            PaperAccountConflictKind.EXECUTION_SCOPE,
        ),
        (
            "paper:test",
            8,
            _account(),
            PaperAccountConflictKind.OWNER_GENERATION,
        ),
        (
            "paper:test",
            7,
            _account(margin_quantum=Decimal("0.01")),
            PaperAccountConflictKind.OPENING_IDENTITY,
        ),
        (
            "paper:test",
            7,
            _account(available=Decimal("100.0")),
            PaperAccountConflictKind.OPENING_IDENTITY,
        ),
    ),
)
def test_provision_rejects_immutable_opening_conflicts_without_mutation(
    journal,
    execution_scope,
    owner_generation,
    account,
    kind,
):
    database, repository = journal
    repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=_account(),
    )
    before = (copy.deepcopy(database.streams), copy.deepcopy(database.balances))

    with pytest.raises(PaperAccountConflictError) as caught:
        repository.provision_account(
            execution_scope=execution_scope,
            owner_generation=owner_generation,
            account=account,
        )

    assert caught.value.kind is kind
    assert (database.streams, database.balances) == before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_scope", " padded "),
        ("execution_scope", "x" * 129),
        ("execution_scope", "bad\x00scope"),
        ("owner_generation", True),
        ("owner_generation", 0),
        ("owner_generation", 1 << 63),
    ),
)
def test_invalid_provision_input_fails_before_connect(field, value):
    database = MemoryDatabase()
    repository = PostgresPaperAccountJournal(database.connect)
    kwargs = {
        "execution_scope": "paper:test",
        "owner_generation": 7,
        "account": _account(),
    }
    kwargs[field] = value

    with pytest.raises(PaperAccountInputError):
        repository.provision_account(**kwargs)

    assert database.connections == []


def test_provision_requires_an_empty_account_before_connect():
    database = MemoryDatabase()
    repository = PostgresPaperAccountJournal(database.connect)
    forged = copy.copy(_account())
    object.__setattr__(forged, "reservations", (object(),))

    with pytest.raises(PaperAccountInputError):
        repository.provision_account(
            execution_scope="paper:test",
            owner_generation=7,
            account=forged,
        )

    assert database.connections == []


def test_commit_acknowledgement_loss_is_unknown_then_exact_retry_is_existing(journal):
    database, repository = journal
    database.commit_then_raise = True

    with pytest.raises(PaperAccountCommitUnknown) as caught:
        repository.provision_account(
            execution_scope="paper:test",
            owner_generation=7,
            account=_account(),
        )

    assert caught.value.execution_scope == "paper:test"
    assert caught.value.account_key == "account-1"
    assert caught.value.owner_generation == 7
    assert caught.value.requires_reconciliation is True
    database.commit_then_raise = False
    replay = repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=_account(),
    )
    assert replay.disposition is ProvisionDisposition.EXISTING
    assert len(database.streams) == 1
    assert len(database.balances["account-1"]) == 2


def test_replay_opening_only_uses_repeatable_read_and_rolls_back(journal):
    database, repository = journal
    repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=_account(),
    )

    replay = repository.replay_account(
        execution_scope="paper:test",
        account_key="account-1",
    )

    assert replay.account == _account()
    assert replay.owner_generation == 7
    assert replay.batches == ()
    connection = database.connections[-1]
    assert connection.commands[0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )
    assert all("FOR UPDATE" not in command for command in connection.commands)
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_replay_reports_missing_and_wrong_scope(journal):
    _, repository = journal
    with pytest.raises(PaperAccountNotFoundError):
        repository.replay_account(
            execution_scope="paper:test",
            account_key="missing",
        )

    repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=_account(),
    )
    with pytest.raises(PaperAccountConflictError) as caught:
        repository.replay_account(
            execution_scope="paper:other",
            account_key="account-1",
        )
    assert caught.value.kind is PaperAccountConflictKind.EXECUTION_SCOPE


def test_list_accounts_filters_sorts_and_fully_replays_one_snapshot(journal):
    database, repository = journal
    for scope, key, generation in (
        ("paper:test", "z-account", 9),
        ("paper:test", "a-account", 7),
        ("paper:other", "other-account", 11),
    ):
        repository.provision_account(
            execution_scope=scope,
            owner_generation=generation,
            account=_account(account_key=key),
        )

    accounts = repository.list_accounts(execution_scope="paper:test")

    assert tuple(value.account.policy.account_key for value in accounts) == (
        "a-account",
        "z-account",
    )
    assert tuple(value.owner_generation for value in accounts) == (7, 9)
    connection = database.connections[-1]
    assert connection.cursor_calls == 1
    assert (
        connection.commands.count(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
        )
        == 1
    )
    assert all("FOR UPDATE" not in command for command in connection.commands)
    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "opening_hash",
        "stream_version",
        "stream_state",
        "missing_balance",
        "noncanonical_balance",
        "balance_scale",
    ),
)
def test_replay_fails_closed_on_opening_or_projection_corruption(journal, mutation):
    database, repository = journal
    repository.provision_account(
        execution_scope="paper:test",
        owner_generation=7,
        account=_account(),
    )
    stream = database.streams["account-1"]
    if mutation == "opening_hash":
        stream["opening_payload_sha256"] = "0" * 64
    elif mutation == "stream_version":
        stream["account_version"] = 1
    elif mutation == "stream_state":
        stream["account_state"] = "INSOLVENT"
    elif mutation == "missing_balance":
        database.balances["account-1"].pop()
    elif mutation == "noncanonical_balance":
        database.balances["account-1"][1]["available"] = "01.00"
    else:
        database.balances["account-1"][1]["available"] = "100.0"

    with pytest.raises(PaperAccountReplayError):
        repository.replay_account(
            execution_scope="paper:test",
            account_key="account-1",
        )

    connection = database.connections[-1]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_list_accounts_is_all_or_nothing_when_one_account_is_corrupt(journal):
    database, repository = journal
    for key in ("a-healthy", "z-corrupt"):
        repository.provision_account(
            execution_scope="paper:test",
            owner_generation=7,
            account=_account(account_key=key),
        )
    database.balances["z-corrupt"][1]["available"] = "99.00"

    with pytest.raises(PaperAccountReplayError):
        repository.list_accounts(execution_scope="paper:test")


def test_query_failure_is_typed_and_connection_is_closed(journal):
    database, repository = journal
    database.fail_execute = True

    with pytest.raises(PaperAccountStorageError, match="replay failed"):
        repository.replay_account(
            execution_scope="paper:test",
            account_key="account-1",
        )

    connection = database.connections[-1]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


def test_provision_execute_failure_rolls_back_closes_and_changes_nothing(journal):
    database, repository = journal
    database.fail_execute = True

    with pytest.raises(PaperAccountStorageError, match="before commit"):
        repository.provision_account(
            execution_scope="paper:test",
            owner_generation=7,
            account=_account(),
        )

    assert database.streams == {}
    assert database.balances == {}
    connection = database.connections[-1]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True


_REPOSITORY_MODULE = "trading.persistence.paper_account_journal"
_REPOSITORY_EXPORTS = {
    "PaperAccountCommitUnknown",
    "PaperAccountConflictError",
    "PaperAccountConflictKind",
    "PaperAccountInputError",
    "PaperAccountJournalError",
    "PaperAccountNotFoundError",
    "PaperAccountReplayError",
    "PaperAccountStorageError",
    "PostgresPaperAccountJournal",
    "ProvisionDisposition",
    "ProvisionedPaperAccount",
    "ReplayedPaperAccount",
}


def _attribute_path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return (*parent, node.attr) if parent is not None else None
    return None


def _uses_paper_account_journal(source):
    tree = ast.parse(source)
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    builtin_import_aliases = {"__import__"}
    trading_aliases = set()
    persistence_aliases = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _REPOSITORY_MODULE or alias.name.startswith(
                    f"{_REPOSITORY_MODULE}."
                ):
                    return True
                if alias.name == "trading":
                    trading_aliases.add(alias.asname or "trading")
                elif alias.name == "trading.persistence":
                    persistence_aliases.add(alias.asname or "persistence")
                    if alias.asname is None:
                        trading_aliases.add("trading")
                elif alias.name == "importlib":
                    importlib_aliases.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            module = node.module or ""
            if module == _REPOSITORY_MODULE or (
                node.level and module.endswith("paper_account_journal")
            ):
                return True
            if module == "trading.persistence" or (
                node.level and module == "persistence"
            ):
                if imported & (_REPOSITORY_EXPORTS | {"paper_account_journal", "*"}):
                    return True
            if (
                node.level
                and not module
                and imported
                & {
                    "paper_account_journal",
                    "*",
                }
            ):
                return True
            if module == "trading" and "persistence" in imported:
                persistence_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "persistence"
                )
            if module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if not names or value is None:
                continue
            path = _attribute_path(value)
            is_import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                path is not None
                and len(path) == 2
                and path[0] in importlib_aliases
                and path[1] == "import_module"
            )
            is_builtin_import = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            )
            target_set = (
                import_module_aliases if is_import_module else builtin_import_aliases
            )
            if is_import_module or is_builtin_import:
                for name in names:
                    if name not in target_set:
                        target_set.add(name)
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
        function_path = _attribute_path(node.func)
        dynamic = (
            isinstance(node.func, ast.Name)
            and node.func.id in import_module_aliases | builtin_import_aliases
        ) or (
            function_path is not None
            and len(function_path) == 2
            and function_path[0] in importlib_aliases
            and function_path[1] == "import_module"
        )
        if dynamic and (
            target == _REPOSITORY_MODULE or target.startswith(f"{_REPOSITORY_MODULE}.")
        ):
            return True
        if dynamic and target.startswith("."):
            package = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "package"
                    and isinstance(keyword.value, ast.Constant)
                ),
                (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else None
                ),
            )
            if package:
                try:
                    if (
                        importlib.util.resolve_name(target, package)
                        == _REPOSITORY_MODULE
                    ):
                        return True
                except (ImportError, ValueError):
                    pass

    for node in ast.walk(tree):
        path = _attribute_path(node)
        if path is None or path[-1] not in (
            _REPOSITORY_EXPORTS | {"paper_account_journal"}
        ):
            continue
        if path[0] in persistence_aliases:
            return True
        if len(path) >= 3 and path[0] in trading_aliases and path[1] == "persistence":
            return True
    return False


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence.paper_account_journal import "
        "PostgresPaperAccountJournal",
        "import trading.persistence.paper_account_journal as accounts",
        "from trading.persistence import PostgresPaperAccountJournal",
        "from trading.persistence import paper_account_journal",
        "from trading.persistence import *",
        "import trading as root\nroot.persistence.PostgresPaperAccountJournal",
        "from trading import persistence as store\n" "store.paper_account_journal",
        "from .persistence.paper_account_journal import ProvisionDisposition",
        "from .persistence import paper_account_journal",
        "from importlib import import_module as load\n"
        "load('trading.persistence.paper_account_journal')",
        "import importlib as loader\n"
        "loader.import_module(name='trading.persistence.paper_account_journal')",
        "__import__('trading.persistence.paper_account_journal')",
        "load = __import__\nload('trading.persistence.paper_account_journal')",
        "from importlib import import_module\nload = import_module\n"
        "load('.paper_account_journal', 'trading.persistence')",
    ),
)
def test_repository_consumer_detector_catches_supported_forms(source):
    assert _uses_paper_account_journal(source)


@pytest.mark.parametrize(
    "source",
    (
        "from trading.persistence import apply_migrations",
        "import trading.persistence",
        "from trading.domain.paper_accounting import PaperAccount",
        "name = 'trading.persistence.paper_account_journal'",
    ),
)
def test_repository_consumer_detector_allows_unrelated_forms(source):
    assert not _uses_paper_account_journal(source)


def test_paper_account_repository_is_unwired_and_not_facade_exported():
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "persistence" / "paper_account_journal.py"
    facade_path = root / "trading" / "persistence" / "__init__.py"
    consumers = []
    for source_path in root.rglob("*.py"):
        if (
            source_path == module_path
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        if _uses_paper_account_journal(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert consumers == []
    assert not _uses_paper_account_journal(facade_path.read_text(encoding="utf-8"))
