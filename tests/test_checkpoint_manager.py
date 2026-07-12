"""Lifecycle tests for trading.utils.checkpoint.CheckpointManager.

Covers the pieces that were previously unwired: best/latest tracking, bounded
cleanup that preserves the best checkpoint, and metadata self-healing when
checkpoint files disappear out-of-band.
"""

import json
import os

import pytest

# CheckpointManager (and these tests) need torch; CI runs without it.
torch = pytest.importorskip("torch", reason="checkpoint I/O requires torch")

from trading.utils.checkpoint import CheckpointManager


def _mgr(tmp_path):
    return CheckpointManager({"checkpoint_dir": str(tmp_path)})


def _state(epoch):
    return {"epoch": epoch, "model_state": {"w": torch.tensor([float(epoch)])}}


def test_best_latest_and_load(tmp_path):
    m = _mgr(tmp_path)
    m.save_checkpoint(_state(1))
    best = m.save_checkpoint(_state(2), is_best=True)
    latest = m.save_checkpoint(_state(3))

    assert m.get_best_checkpoint() == best
    assert m.get_latest_checkpoint() == latest
    loaded = m.load_checkpoint()  # None -> latest via metadata
    assert loaded["epoch"] == 3


def test_cleanup_keeps_last_n_plus_best(tmp_path):
    m = _mgr(tmp_path)
    m.save_checkpoint(_state(1), is_best=True)  # oldest, but the best
    for e in range(2, 8):  # 6 more, none best
        m.save_checkpoint(_state(e))

    m.cleanup_old_checkpoints(keep_last_n=2)

    on_disk = list(tmp_path.glob("*.pt"))
    assert len(on_disk) == 3  # 2 most-recent + the best
    assert os.path.exists(m.get_best_checkpoint())  # best survived despite age
    meta = json.load(open(tmp_path / "checkpoints.json"))
    assert len(meta["checkpoints"]) == 3
    # every surviving metadata entry points at a real file
    assert all(os.path.exists(c["path"]) for c in meta["checkpoints"])


def test_reconcile_drops_dangling_entries(tmp_path):
    m = _mgr(tmp_path)
    p1 = m.save_checkpoint(_state(1), is_best=True)
    p2 = m.save_checkpoint(_state(2))
    os.remove(p2)  # a checkpoint vanishes out from under the metadata

    # re-init -> __init__ self-heals the metadata against disk
    m2 = _mgr(tmp_path)

    meta = json.load(open(tmp_path / "checkpoints.json"))
    paths = [c["path"] for c in meta["checkpoints"]]
    assert p1 in paths and p2 not in paths
    assert m2.get_latest_checkpoint() == p1  # recomputed to a surviving file
    assert os.path.exists(m2.get_latest_checkpoint())
    assert m2.load_checkpoint() is not None  # latest is loadable, not dangling


def test_reconcile_dedupes_duplicate_paths(tmp_path):
    m = _mgr(tmp_path)
    p = m.save_checkpoint(_state(1))
    # Simulate a legacy second-resolution collision: two metadata entries that
    # point at the same file (the old timestamp format reused one filename).
    meta = json.load(open(tmp_path / "checkpoints.json"))
    meta["checkpoints"].append(dict(meta["checkpoints"][0]))
    json.dump(meta, open(tmp_path / "checkpoints.json", "w"))

    m2 = _mgr(tmp_path)  # __init__ reconcile collapses the duplicate

    meta2 = json.load(open(tmp_path / "checkpoints.json"))
    assert len(meta2["checkpoints"]) == 1
    assert meta2["checkpoints"][0]["path"] == p
    assert m2.get_latest_checkpoint() == p


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
