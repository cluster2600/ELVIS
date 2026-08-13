"""Test-only SQL observation helpers for the bounded legacy snapshot import."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SqlEventLog:
    """Ordered connection events without parameters or copied business values."""

    events: list[str] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)
    fetch_sizes: list[int] = field(default_factory=list)
    parameter_batch_sizes: list[int] = field(default_factory=list)
    commits: int = 0
    rollbacks: int = 0


class RecordingCursor:
    """Record statement shapes while deliberately omitting bound row values."""

    def __init__(self, cursor: object, record: SqlEventLog) -> None:
        self._cursor = cursor
        self._record = record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __enter__(self) -> "RecordingCursor":
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._cursor.__exit__(*args)

    def execute(self, query: object, variables: object = None) -> object:
        rendered = (
            query.as_string(self._cursor) if hasattr(query, "as_string") else str(query)
        )
        normalized = " ".join(rendered.split())
        self._record.statements.append(normalized)
        self._record.events.append(f"sql:{normalized}")
        if variables is not None:
            self._record.parameter_batch_sizes.append(_parameter_group_size(variables))
        if variables is None:
            return self._cursor.execute(query)
        return self._cursor.execute(query, variables)

    def executemany(self, query: object, variables: object) -> object:
        rendered = (
            query.as_string(self._cursor) if hasattr(query, "as_string") else str(query)
        )
        normalized = " ".join(rendered.split())
        self._record.statements.append(normalized)
        self._record.events.append(f"sql-many:{normalized}")
        self._record.parameter_batch_sizes.append(_parameter_group_size(variables))
        return self._cursor.executemany(query, variables)

    def fetchmany(self, size: int | None = None) -> object:
        if size is None:
            self._record.fetch_sizes.append(self._cursor.arraysize)
            return self._cursor.fetchmany()
        self._record.fetch_sizes.append(size)
        return self._cursor.fetchmany(size)


class RecordingConnection:
    """Transparent psycopg connection proxy with ordered commit evidence."""

    def __init__(self, connection: object, record: SqlEventLog) -> None:
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_record", record)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._connection, name, value)

    def cursor(self, *args: object, **kwargs: object) -> RecordingCursor:
        return RecordingCursor(
            self._connection.cursor(*args, **kwargs),
            self._record,
        )

    def commit(self) -> None:
        self._record.events.append("commit:before")
        self._record.commits += 1
        self._connection.commit()
        self._record.events.append("commit:confirmed")

    def rollback(self) -> None:
        self._record.events.append("rollback")
        self._record.rollbacks += 1
        self._connection.rollback()

    def close(self) -> None:
        self._record.events.append("close")
        self._connection.close()


class CommitUnknownConnection(RecordingConnection):
    """Commit successfully, then simulate loss of the acknowledgement."""

    def commit(self) -> None:
        self._record.events.append("commit:before")
        self._record.commits += 1
        self._connection.commit()
        self._record.events.append("commit:unknown")
        raise ConnectionError("simulated commit acknowledgement loss")


class CommitNotSentConnection(RecordingConnection):
    """Lose the connection before PostgreSQL receives the commit request."""

    def commit(self) -> None:
        self._record.events.append("commit:not-sent")
        self._record.commits += 1
        raise ConnectionError("simulated connection loss before commit")


class FailBeforeCommitConnection(RecordingConnection):
    """Fail a selected target INSERT before the transaction can commit."""

    def __init__(
        self,
        connection: object,
        record: SqlEventLog,
        *,
        fail_on_insert_number: int,
    ) -> None:
        super().__init__(connection, record)
        object.__setattr__(self, "_fail_on_insert_number", fail_on_insert_number)
        object.__setattr__(self, "_insert_count", 0)

    def cursor(self, *args: object, **kwargs: object) -> RecordingCursor:
        cursor = super().cursor(*args, **kwargs)
        owner = self
        original_execute = cursor.execute
        original_executemany = cursor.executemany

        def maybe_fail(query: object, variables: object = None) -> object:
            rendered = (
                query.as_string(cursor._cursor)
                if hasattr(query, "as_string")
                else str(query)
            )
            if rendered.lstrip().upper().startswith("INSERT"):
                object.__setattr__(owner, "_insert_count", owner._insert_count + 1)
                if owner._insert_count == owner._fail_on_insert_number:
                    owner._record.events.append("insert:failed")
                    raise ConnectionError("simulated pre-commit insert failure")
            return original_execute(query, variables)

        def maybe_fail_many(query: object, variables: object) -> object:
            rendered = (
                query.as_string(cursor._cursor)
                if hasattr(query, "as_string")
                else str(query)
            )
            if rendered.lstrip().upper().startswith("INSERT"):
                object.__setattr__(owner, "_insert_count", owner._insert_count + 1)
                if owner._insert_count == owner._fail_on_insert_number:
                    owner._record.events.append("insert:failed")
                    raise ConnectionError("simulated pre-commit insert failure")
            return original_executemany(query, variables)

        cursor.execute = maybe_fail
        cursor.executemany = maybe_fail_many
        return cursor


class FailOnSequenceConnection(RecordingConnection):
    """Interrupt one sequence-normalization statement after the row commit."""

    def __init__(
        self,
        connection: object,
        record: SqlEventLog,
        *,
        fail_on_setval_number: int,
    ) -> None:
        super().__init__(connection, record)
        object.__setattr__(self, "_fail_on_setval_number", fail_on_setval_number)
        object.__setattr__(self, "_setval_count", 0)

    def cursor(self, *args: object, **kwargs: object) -> RecordingCursor:
        cursor = super().cursor(*args, **kwargs)
        owner = self
        original_execute = cursor.execute

        def maybe_fail(query: object, variables: object = None) -> object:
            rendered = (
                query.as_string(cursor._cursor)
                if hasattr(query, "as_string")
                else str(query)
            )
            if "SETVAL" in rendered.upper():
                object.__setattr__(owner, "_setval_count", owner._setval_count + 1)
                if owner._setval_count == owner._fail_on_setval_number:
                    owner._record.events.append("setval:failed")
                    raise ConnectionError(
                        "simulated sequence-normalization interruption"
                    )
            return original_execute(query, variables)

        cursor.execute = maybe_fail
        return cursor


class AfterTargetIdentityCursor(RecordingCursor):
    """Commit a test drift after target identity acquires its SQL snapshot."""

    def __init__(
        self,
        cursor: object,
        record: SqlEventLog,
        *,
        callback: Callable[[], None],
        fired: list[bool],
    ) -> None:
        super().__init__(cursor, record)
        self._callback = callback
        self._fired = fired

    def execute(self, query: object, variables: object = None) -> object:
        result = super().execute(query, variables)
        statement = self._record.statements[-1].upper()
        if (
            not self._fired[0]
            and "SYSTEM_IDENTIFIER FROM PG_CONTROL_SYSTEM()" in statement
        ):
            self._record.events.append("target-identity:snapshot-acquired")
            self._callback()
            self._fired[0] = True
            self._record.events.append("target-identity:external-drift-committed")
        return result


class AfterTargetIdentityConnection(RecordingConnection):
    """Wrap only the transaction whose post-identity race is under test."""

    def __init__(
        self,
        connection: object,
        record: SqlEventLog,
        *,
        callback: Callable[[], None],
    ) -> None:
        super().__init__(connection, record)
        object.__setattr__(self, "_callback", callback)
        object.__setattr__(self, "_fired", [False])

    def cursor(self, *args: object, **kwargs: object) -> RecordingCursor:
        return AfterTargetIdentityCursor(
            self._connection.cursor(*args, **kwargs),
            self._record,
            callback=self._callback,
            fired=self._fired,
        )


def _parameter_group_size(variables: object) -> int:
    """Record only collection cardinality, never bound business values."""

    if isinstance(variables, (list, tuple)):
        return len(variables)
    return 1


def statement_keyword(statement: str) -> str:
    """Return the first SQL keyword from one normalized observed statement."""

    return statement.lstrip().split(None, 1)[0].upper()


def first_event_index(events: list[str], fragment: str) -> int:
    """Find an event fragment and fail loudly when test evidence is absent."""

    return next(index for index, event in enumerate(events) if fragment in event)
