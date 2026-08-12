import ast
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib.util import resolve_name
from pathlib import Path

import pytest

from trading.application.journaled_order_service import (
    JournaledOrderService,
    JournaledSubmissionDisposition,
    JournaledSubmissionResult,
    SubmissionObservationNotRecorded,
)
from trading.application.order_service import OrderService
from trading.domain.order_lifecycle import (
    SubmissionAcknowledged,
    SubmissionAmbiguous,
    SubmissionFailed,
)
from trading.domain.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    RetrySafety,
    SubmissionReport,
    SubmissionStatus,
)
from trading.domain.positions import (
    PositionEffect,
    PositionExitContext,
    PositionInstruction,
    TakeProfitProfile,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
OBSERVED_AT = NOW + timedelta(seconds=1)


def make_instruction() -> PositionInstruction:
    return PositionInstruction(
        position_key="position-1",
        effect=PositionEffect.OPEN,
        order_intent=OrderIntent(
            client_order_id="order-1",
            decision_id="decision-1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.10"),
            order_type=OrderType.MARKET,
            reference_price=Decimal("50000.125"),
            leverage=3,
            created_at=NOW,
        ),
        exit_context=PositionExitContext(
            take_profit_profile=TakeProfitProfile.RANGING,
            take_profit_fraction=Decimal("0.0025"),
            stop_loss_fraction=Decimal("0.005"),
        ),
    )


def make_report(status: SubmissionStatus) -> SubmissionReport:
    if status is SubmissionStatus.SUBMITTED:
        return SubmissionReport(
            client_order_id="order-1",
            status=status,
            retry_safety=RetrySafety.UNSAFE,
            venue_order_id="venue-1",
            venue_status="FILLED",
        )
    return SubmissionReport(
        client_order_id="order-1",
        status=status,
        retry_safety=(
            RetrySafety.UNSAFE
            if status is SubmissionStatus.AMBIGUOUS
            else RetrySafety.SAFE
        ),
        reason=f"{status.value.lower()} outcome",
        venue_status=(
            "REJECTED" if status is SubmissionStatus.VENUE_REJECTED else None
        ),
    )


@dataclass(frozen=True)
class FakeReservationReceipt:
    is_created: object


@dataclass(frozen=True)
class FakeEventReceipt:
    durable_event_id: str = "submission-attempt-1"
    position_version: int = 1


class FakeClock:
    def __init__(self, value=OBSERVED_AT, *, error=None, timeline=None) -> None:
        self.value = value
        self.error = error
        self.timeline = timeline
        self.calls = 0

    def now(self):
        self.calls += 1
        if self.timeline is not None:
            self.timeline.append("clock")
        if self.error is not None:
            raise self.error
        return self.value


class FakeExecution:
    def __init__(self, report, *, timeline=None) -> None:
        self.report = report
        self.timeline = timeline
        self.calls = []

    def submit(self, intent, /):
        self.calls.append(intent)
        if self.timeline is not None:
            self.timeline.append("submit")
        if isinstance(self.report, BaseException):
            raise self.report
        return self.report


class FakeJournal:
    def __init__(
        self,
        *,
        reservations=(True,),
        reserve_error=None,
        append_error=None,
        event_receipt=None,
        timeline=None,
    ) -> None:
        self.reservations = list(reservations)
        self.reserve_error = reserve_error
        self.append_error = append_error
        self.event_receipt = event_receipt or FakeEventReceipt()
        self.timeline = timeline
        self.reserve_calls = []
        self.append_calls = []

    def reserve_instruction(self, *, execution_scope, instruction):
        self.reserve_calls.append((execution_scope, instruction))
        if self.timeline is not None:
            self.timeline.append("reserve")
        if self.reserve_error is not None:
            raise self.reserve_error
        return FakeReservationReceipt(self.reservations.pop(0))

    def append_event(
        self,
        *,
        execution_scope,
        position_key,
        event_id,
        event,
    ):
        self.append_calls.append((execution_scope, position_key, event_id, event))
        if self.timeline is not None:
            self.timeline.append("append")
        if self.append_error is not None:
            raise self.append_error
        return self.event_receipt


class ReserveOnlyJournal:
    def reserve_instruction(self, *, execution_scope, instruction):
        raise AssertionError("constructor validation must not call the journal")


def make_service(*, report=None, journal=None, clock=None, timeline=None):
    execution = FakeExecution(
        report or make_report(SubmissionStatus.SUBMITTED),
        timeline=timeline,
    )
    selected_journal = journal or FakeJournal(timeline=timeline)
    selected_clock = clock or FakeClock(timeline=timeline)
    return (
        JournaledOrderService(
            OrderService(execution),
            selected_journal,
            selected_clock,
        ),
        execution,
        selected_journal,
        selected_clock,
    )


@pytest.mark.parametrize(
    ("status", "event_type", "requires_reconciliation"),
    [
        (SubmissionStatus.SUBMITTED, SubmissionAcknowledged, False),
        (SubmissionStatus.AMBIGUOUS, SubmissionAmbiguous, True),
        (SubmissionStatus.NOT_SENT, SubmissionFailed, False),
        (SubmissionStatus.VENUE_REJECTED, SubmissionFailed, False),
    ],
)
def test_created_reservation_submits_once_and_records_each_report(
    status,
    event_type,
    requires_reconciliation,
) -> None:
    timeline = []
    service, execution, journal, clock = make_service(
        report=make_report(status),
        timeline=timeline,
    )
    instruction = make_instruction()

    result = service.submit(instruction, execution_scope="paper:test")

    assert timeline == ["reserve", "clock", "submit", "append"]
    assert execution.calls == [instruction.order_intent]
    assert clock.calls == 1
    assert journal.reserve_calls == [("paper:test", instruction)]
    assert len(journal.append_calls) == 1
    scope, position_key, event_id, event = journal.append_calls[0]
    assert (scope, position_key, event_id) == (
        "paper:test",
        "position-1",
        "submission-attempt-1",
    )
    assert type(event) is event_type
    assert event.observed_at == OBSERVED_AT
    assert result.disposition is JournaledSubmissionDisposition.RECORDED
    assert result.report is execution.report
    assert result.event is event
    assert result.durable_event_id == "submission-attempt-1"
    assert result.position_version == 1
    assert result.execution_attempted is True
    assert result.requires_reconciliation is requires_reconciliation


def test_filled_venue_status_is_only_a_submission_acknowledgement() -> None:
    service, _, journal, _ = make_service()

    result = service.submit(make_instruction(), execution_scope="paper:test")

    assert result.report.venue_status == "FILLED"
    assert type(result.event) is SubmissionAcknowledged
    assert not hasattr(result.event, "trade_id")
    assert len(journal.append_calls) == 1


def test_existing_reservation_never_reads_clock_submits_or_appends() -> None:
    journal = FakeJournal(reservations=(False,))
    clock = FakeClock(error=AssertionError("clock must not be read"))
    service, execution, _, _ = make_service(journal=journal, clock=clock)

    result = service.submit(make_instruction(), execution_scope="paper:test")

    assert result == JournaledSubmissionResult(
        disposition=JournaledSubmissionDisposition.EXISTING_RESERVATION
    )
    assert result.execution_attempted is False
    assert result.requires_reconciliation is True
    assert execution.calls == []
    assert clock.calls == 0
    assert journal.append_calls == []


def test_reservation_failure_propagates_before_clock_or_execution() -> None:
    failure = RuntimeError("commit acknowledgement unknown")
    journal = FakeJournal(reserve_error=failure)
    service, execution, _, clock = make_service(journal=journal)

    with pytest.raises(RuntimeError) as caught:
        service.submit(make_instruction(), execution_scope="paper:test")

    assert caught.value is failure
    assert execution.calls == []
    assert clock.calls == 0
    assert journal.append_calls == []


@pytest.mark.parametrize(
    "clock",
    [
        FakeClock(value=datetime(2026, 8, 12, 12, 0)),
        FakeClock(error=RuntimeError("clock failed")),
    ],
)
def test_clock_failure_after_reservation_propagates_before_execution(clock) -> None:
    service, execution, journal, _ = make_service(clock=clock)

    with pytest.raises((ValueError, RuntimeError)):
        service.submit(make_instruction(), execution_scope="paper:test")

    assert len(journal.reserve_calls) == 1
    assert execution.calls == []
    assert journal.append_calls == []


def test_invalid_reservation_receipt_never_becomes_permission_to_submit() -> None:
    journal = FakeJournal(reservations=(1,))
    service, execution, _, clock = make_service(journal=journal)

    with pytest.raises(TypeError, match="is_created must be a bool"):
        service.submit(make_instruction(), execution_scope="paper:test")

    assert execution.calls == []
    assert clock.calls == 0
    assert journal.append_calls == []


def test_adapter_exception_becomes_one_durable_ambiguous_observation() -> None:
    execution_error = TimeoutError("private venue detail")
    service, execution, journal, _ = make_service(report=execution_error)

    result = service.submit(make_instruction(), execution_scope="paper:test")

    assert len(execution.calls) == 1
    assert result.report.status is SubmissionStatus.AMBIGUOUS
    assert result.report.retry_safety is RetrySafety.UNSAFE
    assert "TimeoutError" in result.report.reason
    assert "private venue detail" not in result.report.reason
    assert type(result.event) is SubmissionAmbiguous
    assert result.requires_reconciliation is True
    assert len(journal.append_calls) == 1


def test_append_failure_preserves_exact_report_event_and_identity() -> None:
    append_failure = RuntimeError("commit acknowledgement unknown")
    journal = FakeJournal(append_error=append_failure)
    report = make_report(SubmissionStatus.SUBMITTED)
    service, execution, _, _ = make_service(report=report, journal=journal)

    with pytest.raises(SubmissionObservationNotRecorded) as caught:
        service.submit(make_instruction(), execution_scope="paper:test")

    error = caught.value
    assert error.__cause__ is append_failure
    assert error.report is report
    assert error.event is journal.append_calls[0][3]
    assert error.event_id == "submission-attempt-1"
    assert error.requires_reconciliation is True
    assert len(execution.calls) == 1


@pytest.mark.parametrize(
    "event_receipt",
    [
        FakeEventReceipt(durable_event_id="other-event"),
        FakeEventReceipt(durable_event_id=""),
        FakeEventReceipt(position_version=0),
        FakeEventReceipt(position_version=True),
        object(),
    ],
)
def test_malformed_append_receipt_is_not_reported_as_durable(event_receipt) -> None:
    journal = FakeJournal(event_receipt=event_receipt)
    service, execution, _, _ = make_service(journal=journal)

    with pytest.raises(SubmissionObservationNotRecorded) as caught:
        service.submit(make_instruction(), execution_scope="paper:test")

    assert isinstance(caught.value.__cause__, (AttributeError, TypeError, ValueError))
    assert caught.value.event is journal.append_calls[0][3]
    assert caught.value.event_id == "submission-attempt-1"
    assert len(execution.calls) == 1


def test_reentry_after_append_failure_finds_existing_without_resubmission() -> None:
    failure = RuntimeError("append failed")
    journal = FakeJournal(
        reservations=(True, False),
        append_error=failure,
    )
    service, execution, _, clock = make_service(journal=journal)

    with pytest.raises(SubmissionObservationNotRecorded):
        service.submit(make_instruction(), execution_scope="paper:test")
    journal.append_error = None
    result = service.submit(make_instruction(), execution_scope="paper:test")

    assert result.disposition is JournaledSubmissionDisposition.EXISTING_RESERVATION
    assert len(execution.calls) == 1
    assert clock.calls == 1
    assert len(journal.append_calls) == 1


def test_invalid_instruction_is_rejected_before_any_collaborator_call() -> None:
    service, execution, journal, clock = make_service()

    with pytest.raises(TypeError):
        service.submit(object(), execution_scope="paper:test")

    assert journal.reserve_calls == []
    assert clock.calls == 0
    assert execution.calls == []


@pytest.mark.parametrize("execution_scope", [None, 1, "", "   ", " padded "])
def test_invalid_execution_scope_is_rejected_before_reservation(
    execution_scope,
) -> None:
    service, execution, journal, clock = make_service()

    with pytest.raises((TypeError, ValueError)):
        service.submit(make_instruction(), execution_scope=execution_scope)

    assert journal.reserve_calls == []
    assert clock.calls == 0
    assert execution.calls == []


@pytest.mark.parametrize(
    ("order_service", "journal", "clock", "message"),
    [
        (object(), FakeJournal(), FakeClock(), "order_service"),
        (
            OrderService(FakeExecution(make_report(SubmissionStatus.SUBMITTED))),
            object(),
            FakeClock(),
            "reserve_instruction",
        ),
        (
            OrderService(FakeExecution(make_report(SubmissionStatus.SUBMITTED))),
            ReserveOnlyJournal(),
            FakeClock(),
            "append_event",
        ),
        (
            OrderService(FakeExecution(make_report(SubmissionStatus.SUBMITTED))),
            FakeJournal(),
            object(),
            "clock",
        ),
    ],
)
def test_constructor_rejects_invalid_collaborators(
    order_service,
    journal,
    clock,
    message,
) -> None:
    with pytest.raises(TypeError, match=message):
        JournaledOrderService(order_service, journal, clock)


@pytest.mark.parametrize(
    "values",
    [
        {"report": make_report(SubmissionStatus.SUBMITTED)},
        {"event": SubmissionAcknowledged("order-1", "venue-1", OBSERVED_AT)},
        {"durable_event_id": "submission-attempt-1"},
        {"position_version": 1},
    ],
)
def test_existing_result_rejects_attempt_details(values) -> None:
    with pytest.raises(ValueError, match="existing reservation"):
        JournaledSubmissionResult(
            disposition=JournaledSubmissionDisposition.EXISTING_RESERVATION,
            **values,
        )


@pytest.mark.parametrize(
    "event",
    [
        SubmissionAcknowledged("order-1", "other-venue", OBSERVED_AT),
        SubmissionFailed(
            client_order_id="order-1",
            status=SubmissionStatus.NOT_SENT,
            retry_safety=RetrySafety.SAFE,
            reason="nothing was sent",
            observed_at=OBSERVED_AT,
        ),
    ],
)
def test_recorded_result_rejects_event_that_contradicts_report(event) -> None:
    with pytest.raises(ValueError, match="exactly represent"):
        JournaledSubmissionResult(
            disposition=JournaledSubmissionDisposition.RECORDED,
            report=make_report(SubmissionStatus.SUBMITTED),
            event=event,
            durable_event_id="submission-attempt-1",
            position_version=1,
        )


def test_recorded_result_requires_stable_submission_event_identity() -> None:
    report = make_report(SubmissionStatus.SUBMITTED)
    event = SubmissionAcknowledged("order-1", "venue-1", OBSERVED_AT)

    with pytest.raises(ValueError, match="event identity"):
        JournaledSubmissionResult(
            disposition=JournaledSubmissionDisposition.RECORDED,
            report=report,
            event=event,
            durable_event_id="another-attempt",
            position_version=1,
        )


_JOURNALED_SERVICE_EXPORTS = {
    "Clock",
    "EventReceipt",
    "JournaledOrderService",
    "JournaledSubmissionDisposition",
    "JournaledSubmissionResult",
    "OrderJournalPort",
    "ReservationReceipt",
    "SubmissionObservationNotRecorded",
}


def _literal_dynamic_import(
    node: ast.Call,
    *,
    builtins_aliases: set[str],
    builtin_import_aliases: set[str],
    importlib_aliases: set[str],
    import_module_aliases: set[str],
) -> str | None:
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
        return None
    uses_builtin_import = (
        isinstance(node.func, ast.Name) and node.func.id in builtin_import_aliases
    )
    uses_import_module = isinstance(node.func, ast.Name) and node.func.id in (
        import_module_aliases
    )
    uses_builtin_import = uses_builtin_import or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "__import__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in builtins_aliases
    )
    uses_import_module = uses_import_module or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_aliases
    )
    if not uses_builtin_import and not uses_import_module:
        return None
    if uses_builtin_import and target.startswith("trading."):
        return "trading"
    if not target.startswith("."):
        return target
    package = (
        node.args[1].value
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
        else next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "package"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            None,
        )
    )
    if package is None:
        return None
    try:
        return resolve_name(target, package)
    except (ImportError, ValueError):
        return None


def _uses_journaled_order_service(source: str) -> bool:
    """Conservatively detect direct, facade, and literal dynamic consumers."""
    tree = ast.parse(source)
    module = "trading.application.journaled_order_service"
    builtins_aliases = {"builtins"}
    builtin_import_aliases = {"__import__"}
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"trading", "trading.application"}:
                    return True
                if alias.name == module or alias.name.startswith(f"{module}."):
                    return True
                if alias.name.startswith("trading.application.") and not alias.asname:
                    return True
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            imported_module = node.module or ""
            if imported_module == "importlib" and "import_module" in imported:
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if imported_module == "builtins" and "__import__" in imported:
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )
            if imported_module == module or (
                node.level and imported_module.endswith("journaled_order_service")
            ):
                return True
            if (
                node.level
                and imported_module in {"", "application"}
                and imported
                & (_JOURNALED_SERVICE_EXPORTS | {"journaled_order_service", "*"})
            ):
                return True
            if imported_module == "trading" and "application" in imported:
                return True
            if node.level and not imported_module and "application" in imported:
                return True
            if imported_module == "trading.application" and imported & (
                _JOURNALED_SERVICE_EXPORTS | {"journaled_order_service", "*"}
            ):
                return True
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            is_builtin_import = (
                isinstance(value, ast.Name) and value.id in builtin_import_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "__import__"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            )
            is_import_module = (
                isinstance(value, ast.Name) and value.id in import_module_aliases
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "import_module"
                and isinstance(value.value, ast.Name)
                and value.value.id in importlib_aliases
            )
            if not is_builtin_import and not is_import_module:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if is_builtin_import and target.id not in builtin_import_aliases:
                    builtin_import_aliases.add(target.id)
                    changed = True
                if is_import_module and target.id not in import_module_aliases:
                    import_module_aliases.add(target.id)
                    changed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _literal_dynamic_import(
                node,
                builtins_aliases=builtins_aliases,
                builtin_import_aliases=builtin_import_aliases,
                importlib_aliases=importlib_aliases,
                import_module_aliases=import_module_aliases,
            )
            if target in {"trading", "trading.application", module}:
                return True
            if target and target.startswith(f"{module}."):
                return True
    return False


@pytest.mark.parametrize(
    "source",
    [
        (
            "from trading.application.journaled_order_service "
            "import JournaledOrderService"
        ),
        "from trading.application import JournaledOrderService",
        "import trading.application as app\napp.JournaledOrderService",
        "from trading import application as app\napp.JournaledOrderService",
        "import trading as root\nroot.application.JournaledOrderService",
        "from ..application import JournaledOrderService",
        "from . import journaled_order_service",
        "from . import application as app\napp.JournaledOrderService",
        "from .. import application as app\napp.JournaledOrderService",
        (
            "from importlib import import_module as fetch\n"
            "fetch('trading.application.journaled_order_service')"
        ),
        (
            "import importlib as loader\n"
            "loader.import_module('trading').application.JournaledOrderService"
        ),
        ("root = __import__('trading')\n" "root.application.JournaledOrderService"),
        ("load = __import__\n" "load('trading.application.journaled_order_service')"),
        (
            "from builtins import __import__ as load\n"
            "load('trading.application.journaled_order_service')"
        ),
        (
            "from builtins import __import__ as load\n"
            "load('trading.domain.orders').application.JournaledOrderService"
        ),
        (
            "import builtins as b\n"
            "b.__import__('trading.application.journaled_order_service')"
        ),
        (
            "__import__('trading.application.order_service')"
            ".application.JournaledOrderService"
        ),
        (
            "from importlib import import_module\n"
            "import_module('.journaled_order_service', "
            "package='trading.application')"
        ),
        (
            "from importlib import import_module\n"
            "import_module('.journaled_order_service', 'trading.application')"
        ),
        (
            "from importlib import import_module\n"
            "import_module('.application', package='trading')"
            ".JournaledOrderService"
        ),
        (
            "from importlib import import_module\n"
            "import_module(name='.application', package='trading')"
            ".JournaledOrderService"
        ),
        (
            "import importlib\n"
            "importlib.import_module('..application.journaled_order_service', "
            "package='trading.execution')"
        ),
    ],
)
def test_journaled_service_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_journaled_order_service(source)


@pytest.mark.parametrize(
    "source",
    [
        "from trading.application.order_service import OrderService",
        "from trading.domain.orders import OrderIntent",
        "from trading.persistence import apply_migrations",
    ],
)
def test_journaled_service_consumer_detector_allows_unrelated_imports(source) -> None:
    assert not _uses_journaled_order_service(source)


def test_journaled_order_service_has_no_runtime_consumer() -> None:
    root = Path(__file__).parents[1]
    module_path = root / "trading" / "application" / "journaled_order_service.py"
    facade_path = root / "trading" / "application" / "__init__.py"
    consumers = []
    scanned = []

    for source_path in root.rglob("*.py"):
        if (
            source_path in {module_path, facade_path}
            or "tests" in source_path.parts
            or ".venv" in source_path.parts
            or "build" in source_path.parts
            or "dist" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue
        scanned.append(source_path.relative_to(root))
        if _uses_journaled_order_service(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))

    assert consumers == []
    assert {
        Path("main.py"),
        Path("core/bootstrap.py"),
        Path("trading/execution/legacy_paper_adapter.py"),
        Path("utils/paper_trade_db.py"),
    } <= set(scanned)
