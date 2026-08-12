"""Fast contract tests for the unwired atomic PostgreSQL paper owner."""

import ast
import copy
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tests.test_order_position_journal import MemoryDatabase, make_instruction
from trading.application.durable_submission import (
    DurableSubmissionDisposition,
    PaperPlannedFill,
    PaperSubmissionPlan,
    SubmissionAttemptContext,
    SubmissionCommitUnknown,
    SubmissionReconciliationRequired,
)
from trading.domain.order_lifecycle import ConfirmedFill, SubmissionAcknowledged
from trading.domain.orders import OrderIntent, OrderSide
from trading.domain.positions import PositionEffect, PositionInstruction
from trading.persistence.atomic_paper_submission_owner import (
    PostgresAtomicPaperSubmissionOwner,
)
from trading.persistence.order_position_journal import (
    JournalConflictError,
    JournalConflictKind,
    JournalStorageError,
    PostgresOrderPositionJournal,
)

NOW = __import__("tests.test_order_position_journal", fromlist=["NOW"]).NOW


def make_attempt(
    *,
    instruction: PositionInstruction | None = None,
) -> SubmissionAttemptContext:
    return SubmissionAttemptContext.first(
        instruction or make_instruction(),
        "paper:test",
        NOW + timedelta(seconds=1),
    )


def make_plan(
    attempt: SubmissionAttemptContext,
    *,
    quantities: tuple[Decimal, ...] = (Decimal("0.40"), Decimal("0.60")),
) -> PaperSubmissionPlan:
    venue_order_id = f"venue-{attempt.client_order_id}"
    fills = tuple(
        PaperPlannedFill(
            event_id=f"fill-event-{index}",
            fill=ConfirmedFill(
                client_order_id=attempt.client_order_id,
                venue_order_id=venue_order_id,
                trade_id=f"trade-{attempt.client_order_id}-{index}",
                symbol=attempt.instruction.order_intent.symbol,
                side=attempt.instruction.order_intent.side,
                quantity=quantity,
                price=Decimal("50001.250"),
                fee_amount=Decimal("0.10"),
                fee_asset="USDT",
                executed_at=attempt.observed_at + timedelta(seconds=index),
            ),
        )
        for index, quantity in enumerate(quantities, start=1)
    )
    return PaperSubmissionPlan(
        attempt=attempt,
        submission=SubmissionAcknowledged(
            client_order_id=attempt.client_order_id,
            venue_order_id=venue_order_id,
            observed_at=attempt.observed_at,
        ),
        fills=fills,
    )


class CountingPlanner:
    def __init__(self) -> None:
        self.calls = []

    def plan(self, attempt, /):
        self.calls.append(attempt)
        return make_plan(attempt)


class ExplodingPlanner:
    def plan(self, attempt, /):
        raise AssertionError("planner must not be called")


def test_execute_commits_one_terminal_batch_and_replays_without_replanning() -> None:
    database = MemoryDatabase()
    planner = CountingPlanner()
    owner = PostgresAtomicPaperSubmissionOwner(database.connect, planner)
    attempt = make_attempt()

    committed = owner.execute(attempt)
    replayed = owner.execute(attempt)

    assert committed.disposition is DurableSubmissionDisposition.COMMITTED
    assert replayed.disposition is DurableSubmissionDisposition.REPLAYED
    assert committed.submission.position_version == 1
    assert tuple(fill.position_version for fill in committed.fills) == (2, 3)
    assert replayed.submission.event == committed.submission.event
    assert replayed.fills == committed.fills
    assert planner.calls == [attempt]
    assert database.state.streams["position-1"]["stream_version"] == 3
    assert len(database.state.orders) == 1
    assert len(database.state.events) == 3
    assert all(connection.commits == 1 for connection in database.connections)
    assert all(connection.closed for connection in database.connections)


def test_exact_terminal_history_replays_with_a_new_owner_and_no_planner() -> None:
    database = MemoryDatabase()
    attempt = make_attempt()
    PostgresAtomicPaperSubmissionOwner(database.connect, CountingPlanner()).execute(
        attempt
    )
    snapshot = copy.deepcopy(database.state)

    receipt = PostgresAtomicPaperSubmissionOwner(
        database.connect,
        ExplodingPlanner(),
    ).execute(attempt)

    assert receipt.disposition is DurableSubmissionDisposition.REPLAYED
    assert database.state == snapshot


def test_exact_terminal_shape_from_separate_commits_is_adopted_without_planning() -> (
    None
):
    database = MemoryDatabase()
    attempt = make_attempt()
    plan = make_plan(attempt)
    journal = PostgresOrderPositionJournal(database.connect)
    journal.reserve_instruction(
        execution_scope=attempt.execution_scope,
        instruction=attempt.instruction,
    )
    journal.append_event(
        execution_scope=attempt.execution_scope,
        position_key=attempt.instruction.position_key,
        event_id=attempt.event_id,
        event=plan.submission,
    )
    for candidate in plan.fills:
        journal.append_event(
            execution_scope=attempt.execution_scope,
            position_key=attempt.instruction.position_key,
            event_id=candidate.event_id,
            event=candidate.fill,
        )
    snapshot = copy.deepcopy(database.state)

    receipt = PostgresAtomicPaperSubmissionOwner(
        database.connect,
        ExplodingPlanner(),
    ).execute(attempt)

    assert receipt.disposition is DurableSubmissionDisposition.REPLAYED
    assert database.state == snapshot


def test_exact_batch_replays_after_a_later_terminal_sibling_batch() -> None:
    database = MemoryDatabase()
    first = make_attempt()
    owner = PostgresAtomicPaperSubmissionOwner(database.connect, CountingPlanner())
    first_receipt = owner.execute(first)
    second = make_attempt(
        instruction=make_instruction(
            client_order_id="order-2",
            decision_id="decision-2",
            position_key="position-1",
        )
    )
    second_receipt = owner.execute(second)

    replayed = PostgresAtomicPaperSubmissionOwner(
        database.connect,
        ExplodingPlanner(),
    ).execute(first)

    assert first_receipt.submission.position_version == 1
    assert second_receipt.submission.position_version == 4
    assert replayed.disposition is DurableSubmissionDisposition.REPLAYED
    assert replayed.submission.position_version == 1
    assert tuple(fill.position_version for fill in replayed.fills) == (2, 3)
    assert database.state.streams["position-1"]["stream_version"] == 6


def test_commit_acknowledgement_loss_reconciles_by_exact_replay() -> None:
    database = MemoryDatabase()
    planner = CountingPlanner()
    attempt = make_attempt()
    database.commit_then_raise = True

    with pytest.raises(SubmissionCommitUnknown) as raised:
        PostgresAtomicPaperSubmissionOwner(database.connect, planner).execute(attempt)

    assert raised.value.attempt is attempt
    assert len(database.state.events) == 3
    database.commit_then_raise = False
    replayed = PostgresAtomicPaperSubmissionOwner(
        database.connect,
        ExplodingPlanner(),
    ).execute(attempt)
    assert replayed.disposition is DurableSubmissionDisposition.REPLAYED
    assert planner.calls == [attempt]
    assert len(database.state.events) == 3


def test_planner_failure_rolls_back_new_stream_and_reservation() -> None:
    database = MemoryDatabase()

    with pytest.raises(JournalStorageError, match="failed before commit") as raised:
        PostgresAtomicPaperSubmissionOwner(
            database.connect,
            ExplodingPlanner(),
        ).execute(make_attempt())

    assert isinstance(raised.value.__cause__, AssertionError)
    assert database.state.streams == {}
    assert database.state.orders == {}
    assert database.state.events == []
    assert database.connections[-1].rollbacks == 1
    assert database.connections[-1].commits == 0


def test_planner_must_retain_the_exact_attempt_object() -> None:
    database = MemoryDatabase()
    attempt = make_attempt()

    class CloningPlanner:
        def plan(self, candidate, /):
            return make_plan(copy.deepcopy(candidate))

    with pytest.raises(ValueError, match="exact attempt object"):
        PostgresAtomicPaperSubmissionOwner(
            database.connect,
            CloningPlanner(),
        ).execute(attempt)

    assert database.state.streams == {}
    assert database.state.orders == {}


def test_existing_pending_order_requires_reconciliation_without_planning() -> None:
    database = MemoryDatabase()
    attempt = make_attempt()
    PostgresOrderPositionJournal(database.connect).reserve_instruction(
        execution_scope=attempt.execution_scope,
        instruction=attempt.instruction,
    )
    snapshot = copy.deepcopy(database.state)

    with pytest.raises(SubmissionReconciliationRequired) as raised:
        PostgresAtomicPaperSubmissionOwner(
            database.connect,
            ExplodingPlanner(),
        ).execute(attempt)

    assert raised.value.attempt is attempt
    assert database.state == snapshot


def test_unresolved_sibling_blocks_a_new_order_on_the_same_stream() -> None:
    database = MemoryDatabase()
    first = make_attempt()
    PostgresOrderPositionJournal(database.connect).reserve_instruction(
        execution_scope=first.execution_scope,
        instruction=first.instruction,
    )
    second_instruction = make_instruction(
        client_order_id="order-2",
        decision_id="decision-2",
        position_key="position-1",
    )
    second = make_attempt(instruction=second_instruction)
    snapshot = copy.deepcopy(database.state)

    with pytest.raises(SubmissionReconciliationRequired):
        PostgresAtomicPaperSubmissionOwner(
            database.connect,
            ExplodingPlanner(),
        ).execute(second)

    assert database.state == snapshot
    assert "order-2" not in database.state.orders


def test_instruction_decimal_quantum_conflicts_before_planning() -> None:
    database = MemoryDatabase()
    initial = make_attempt(instruction=make_instruction(quantity=Decimal("1.0")))
    PostgresAtomicPaperSubmissionOwner(database.connect, CountingPlanner()).execute(
        initial
    )
    changed = make_attempt(instruction=make_instruction(quantity=Decimal("1.00")))

    with pytest.raises(JournalConflictError) as raised:
        PostgresAtomicPaperSubmissionOwner(
            database.connect,
            ExplodingPlanner(),
        ).execute(changed)

    assert raised.value.kind is JournalConflictKind.CLIENT_ORDER_ID


def test_invalid_position_transition_rolls_back_the_whole_batch() -> None:
    database = MemoryDatabase()
    base = make_instruction()
    reduce_instruction = PositionInstruction(
        position_key=base.position_key,
        effect=PositionEffect.REDUCE_ONLY,
        order_intent=OrderIntent(
            client_order_id=base.order_intent.client_order_id,
            decision_id=base.order_intent.decision_id,
            symbol=base.order_intent.symbol,
            side=OrderSide.SELL,
            quantity=base.order_intent.quantity,
            order_type=base.order_intent.order_type,
            reference_price=base.order_intent.reference_price,
            leverage=base.order_intent.leverage,
            created_at=base.order_intent.created_at,
        ),
        exit_context=None,
    )

    with pytest.raises(JournalConflictError) as raised:
        PostgresAtomicPaperSubmissionOwner(
            database.connect,
            CountingPlanner(),
        ).execute(make_attempt(instruction=reduce_instruction))

    assert raised.value.kind is JournalConflictKind.INVALID_TRANSITION
    assert database.state.streams == {}
    assert database.state.orders == {}
    assert database.state.events == []


@pytest.mark.parametrize("value", [None, object(), lambda: None])
def test_constructor_requires_a_planner_protocol(value) -> None:
    with pytest.raises(TypeError, match="planner"):
        PostgresAtomicPaperSubmissionOwner(MemoryDatabase().connect, value)


def test_execute_is_positional_only() -> None:
    owner = PostgresAtomicPaperSubmissionOwner(
        MemoryDatabase().connect,
        CountingPlanner(),
    )
    with pytest.raises(TypeError):
        owner.execute(attempt=make_attempt())


_OWNER_MODULE = "trading.persistence.atomic_paper_submission_owner"
_OWNER_EXPORTS = {"PostgresAtomicPaperSubmissionOwner"}


def _uses_atomic_owner(source: str) -> bool:
    """Conservatively detect static and literal-dynamic owner consumers."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"trading", "trading.persistence", _OWNER_MODULE}:
                    return True
                if alias.name.startswith(f"{_OWNER_MODULE}."):
                    return True
                if (
                    alias.name.startswith("trading.persistence.")
                    and alias.asname is None
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            module = node.module or ""
            if module == _OWNER_MODULE:
                return True
            if node.level and (
                module.endswith("atomic_paper_submission_owner")
                or "atomic_paper_submission_owner" in imported
                or imported & _OWNER_EXPORTS
            ):
                return True
            if module == "trading" and "persistence" in imported:
                return True
            if module == "trading.persistence" and imported & (
                _OWNER_EXPORTS | {"atomic_paper_submission_owner", "*"}
            ):
                return True
        elif isinstance(node, ast.Constant) and node.value in {
            _OWNER_MODULE,
            "trading.persistence",
        }:
            return True
    return False


@pytest.mark.parametrize(
    "source",
    [
        "from trading.persistence.atomic_paper_submission_owner "
        "import PostgresAtomicPaperSubmissionOwner",
        "import trading.persistence.atomic_paper_submission_owner as owner",
        "from trading.persistence import atomic_paper_submission_owner",
        "from .atomic_paper_submission_owner import "
        "PostgresAtomicPaperSubmissionOwner",
        "from importlib import import_module as load\n"
        "load('trading.persistence.atomic_paper_submission_owner')",
        "load = __import__\n"
        "load('trading.persistence.atomic_paper_submission_owner')",
    ],
)
def test_owner_consumer_detector_catches_supported_forms(source) -> None:
    assert _uses_atomic_owner(source)


def test_atomic_owner_is_unwired_and_uses_only_approved_boundaries() -> None:
    root = Path(__file__).parents[1]
    module_path = (
        root / "trading" / "persistence" / ("atomic_paper_submission_owner.py")
    )
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
        if _uses_atomic_owner(source_path.read_text(encoding="utf-8")):
            consumers.append(source_path.relative_to(root))
    assert consumers == [
        Path("trading/persistence/atomic_paper_account_owner.py"),
    ]

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    allowed = {
        "psycopg2",
        "typing",
        "trading.application.durable_submission",
        "trading.domain.order_lifecycle",
        "trading.domain.positions",
        "trading.persistence.journal_codec",
        "trading.persistence.order_position_journal",
    }
    assert imports <= allowed
    source = module_path.read_text(encoding="utf-8")
    assert "reserve_instruction(" not in source
    assert "append_event(" not in source
    assert "trading.execution" not in source
    assert "paper_economics" not in source
    assert "paper_settlement" not in source
