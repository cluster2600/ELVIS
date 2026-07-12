"""Adaptive-ensemble feedback pipeline (roadmap #11): scoring + weight math."""

import datetime as dt

import pytest

from trading.signals import model_feedback
from trading.signals.adaptive_ensemble import AdaptiveEnsembleWeights, combine_weights

TS = dt.datetime(2026, 7, 12, 10, 0, 0)


@pytest.fixture
def fresh_weights(tmp_path, monkeypatch):
    """Isolated shared-weights instance persisted under tmp_path."""
    monkeypatch.setattr(model_feedback, "_shared_weights", None)
    monkeypatch.setenv("ELVIS_ADAPTIVE_STATE", str(tmp_path / "weights.json"))
    monkeypatch.setenv("ELVIS_ADAPTIVE_ALPHA", "0.5")
    return model_feedback.get_shared_weights()


# ---------------------------------------------------------------- combine


def test_combine_weights_uniform_adaptive_is_identity():
    static = {"Technical": 0.3, "Research": 0.35, "RL": 0.25}
    adaptive = {"Technical": 1 / 3, "Research": 1 / 3, "RL": 1 / 3}
    combined = combine_weights(static, adaptive)
    total = sum(static.values())
    for k in static:
        assert combined[k] == pytest.approx(static[k] / total)


def test_combine_weights_skew_shifts_and_normalizes():
    static = {"A": 0.5, "B": 0.5}
    adaptive = {"A": 0.8, "B": 0.2}  # neutral = 0.5 -> multipliers 1.6 / 0.4
    combined = combine_weights(static, adaptive)
    assert combined["A"] == pytest.approx(0.8)
    assert combined["B"] == pytest.approx(0.2)
    assert sum(combined.values()) == pytest.approx(1.0)


def test_combine_weights_clamps_multiplier():
    static = {"A": 0.5, "B": 0.5}
    # B's accuracy collapsed to ~0 -> raw multiplier ~0 but floors at 0.25
    combined = combine_weights(static, {"A": 0.999, "B": 0.001})
    assert combined["B"] > 0  # never silenced by a bad streak
    assert combined["B"] / combined["A"] >= 0.25 / 4.0


def test_combine_weights_unknown_source_neutral():
    static = {"A": 0.5, "NewModel": 0.5}
    combined = combine_weights(static, {"A": 0.5, "B": 0.5})  # neutral = 0.5
    # A's multiplier = 1.0, NewModel unknown -> 1.0: ratios preserved
    assert combined["A"] == pytest.approx(combined["NewModel"])


def test_combine_weights_empty_static():
    assert combine_weights({}, {"A": 1.0}) == {}


# ---------------------------------------------------------------- scoring


def _rows(*specs):
    """(id, model, vote) specs -> DB-shaped unscored rows for one batch."""
    return [(i, TS, "BTCUSDT", "BUY", model, vote) for i, model, vote in specs]


def test_score_right_and_wrong_votes(fresh_weights, monkeypatch):
    rows = _rows((1, "Technical", "BUY"), (2, "Research", "SELL"), (3, "RL", "HOLD"))
    marked = []
    monkeypatch.setattr(model_feedback._db, "get_unscored_predictions", lambda: rows)
    monkeypatch.setattr(
        model_feedback._db,
        "get_first_closing_trade_after",
        lambda symbol, ts: (TS, 12.5),  # profitable close -> BUY was right
    )
    monkeypatch.setattr(
        model_feedback._db, "mark_predictions_scored", lambda ids: marked.extend(ids)
    )

    scored = model_feedback.score_closed_trades()

    assert scored == 2  # HOLD not scored
    assert sorted(marked) == [1, 2, 3]  # but every row marked handled
    acc = fresh_weights.accuracies
    assert acc["Technical"] == pytest.approx(1.0)  # voted BUY, pnl > 0
    assert acc["Research"] == pytest.approx(0.0)  # voted SELL against a win
    assert "RL" not in acc


def test_score_losing_close_inverts(fresh_weights, monkeypatch):
    rows = _rows((1, "Technical", "BUY"), (2, "Research", "SELL"))
    monkeypatch.setattr(model_feedback._db, "get_unscored_predictions", lambda: rows)
    monkeypatch.setattr(
        model_feedback._db,
        "get_first_closing_trade_after",
        lambda symbol, ts: (TS, -3.0),  # losing close
    )
    monkeypatch.setattr(model_feedback._db, "mark_predictions_scored", lambda ids: True)

    model_feedback.score_closed_trades()

    assert fresh_weights.accuracies["Technical"] == pytest.approx(0.0)
    assert fresh_weights.accuracies["Research"] == pytest.approx(1.0)


def test_open_position_left_unscored(fresh_weights, monkeypatch):
    rows = _rows((1, "Technical", "BUY"))
    marked = []
    monkeypatch.setattr(model_feedback._db, "get_unscored_predictions", lambda: rows)
    monkeypatch.setattr(
        model_feedback._db, "get_first_closing_trade_after", lambda s, t: None
    )
    monkeypatch.setattr(
        model_feedback._db, "mark_predictions_scored", lambda ids: marked.extend(ids)
    )

    assert model_feedback.score_closed_trades() == 0
    assert marked == []  # stays unscored for a later cycle
    assert fresh_weights.accuracies == {}


def test_zero_pnl_marks_without_updating(fresh_weights, monkeypatch):
    rows = _rows((1, "Technical", "BUY"))
    marked = []
    monkeypatch.setattr(model_feedback._db, "get_unscored_predictions", lambda: rows)
    monkeypatch.setattr(
        model_feedback._db, "get_first_closing_trade_after", lambda s, t: (TS, 0.0)
    )
    monkeypatch.setattr(
        model_feedback._db, "mark_predictions_scored", lambda ids: marked.extend(ids)
    )

    assert model_feedback.score_closed_trades() == 0
    assert marked == [1]
    assert fresh_weights.accuracies == {}


def test_scoring_persists_state(fresh_weights, monkeypatch, tmp_path):
    rows = _rows((1, "Technical", "BUY"))
    monkeypatch.setattr(model_feedback._db, "get_unscored_predictions", lambda: rows)
    monkeypatch.setattr(
        model_feedback._db, "get_first_closing_trade_after", lambda s, t: (TS, 5.0)
    )
    monkeypatch.setattr(model_feedback._db, "mark_predictions_scored", lambda ids: True)

    model_feedback.score_closed_trades()

    reloaded = AdaptiveEnsembleWeights(
        initial_accuracies={}, state_path=str(tmp_path / "weights.json")
    )
    assert reloaded.accuracies.get("Technical") == pytest.approx(1.0)


def test_sql_zero_pnl_close_is_returned_not_skipped():
    """SQL-semantics regression (review finding on #41): a breakeven close
    (pnl == 0) must be RETURNED by get_first_closing_trade_after — filtering
    it out would misattribute the NEXT position's exit to this vote batch.
    Runs against the real local Postgres; skips where none is available (CI).
    """
    import datetime as _dt
    import uuid

    from utils import paper_trade_db as db

    conn = db.get_conn()
    if conn is None:
        pytest.skip("Postgres unavailable")
    conn.close()

    symbol = f"TEST{uuid.uuid4().hex[:8].upper()}"
    t0 = _dt.datetime.now()
    try:
        # entry fill BEFORE the votes (pnl 0) must not match
        db.record_trade(
            symbol,
            "BUY",
            100.0,
            1.0,
            pnl=0.0,
            timestamp=t0 - _dt.timedelta(seconds=5),
        )
        # breakeven close AFTER the votes
        db.record_trade(
            symbol,
            "SELL",
            100.0,
            1.0,
            pnl=0.0,
            timestamp=t0 + _dt.timedelta(seconds=5),
        )
        # a LATER, unrelated profitable close that a pnl<>0 filter would
        # have wrongly attributed to this batch
        db.record_trade(
            symbol,
            "SELL",
            105.0,
            1.0,
            pnl=5.0,
            timestamp=t0 + _dt.timedelta(seconds=60),
        )

        closing = db.get_first_closing_trade_after(symbol, t0)
        assert closing is not None, "breakeven close was skipped"
        _, pnl = closing
        assert pnl == 0.0, f"expected the breakeven close, got pnl={pnl}"
    finally:
        cleanup = db.get_conn()
        if cleanup is not None:
            cur = cleanup.cursor()
            cur.execute("DELETE FROM np.trades WHERE symbol = %s", (symbol,))
            cleanup.commit()
            cleanup.close()


def test_record_entry_filters_and_delegates(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        model_feedback._db,
        "record_model_predictions",
        lambda symbol, side, votes: captured.update(
            symbol=symbol, side=side, votes=votes
        )
        or True,
    )
    ok = model_feedback.record_entry(
        "BTCUSDT", "BUY", {"Technical": "BUY", "Junk": "MAYBE", "RL": "HOLD"}
    )
    assert ok
    assert captured["votes"] == {"Technical": "BUY", "RL": "HOLD"}
    assert not model_feedback.record_entry("BTCUSDT", "BUY", {})
