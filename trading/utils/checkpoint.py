"""
Checkpoint management utilities for saving and loading model states.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import torch


class CheckpointManager:
    """Manages model checkpoints and state saving/loading."""

    def __init__(self, config: Dict):
        """
        Initialize the checkpoint manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Create checkpoint metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        if not self.metadata_file.exists():
            self._init_metadata()
        else:
            # Heal any drift between the metadata and what is actually on disk
            # (checkpoints rotated/deleted out-of-band, stale latest/best).
            self._reconcile_with_disk()

    def _init_metadata(self):
        """Initialize checkpoint metadata file."""
        metadata = {
            "checkpoints": [],
            "latest_checkpoint": None,
            "best_checkpoint": None,
        }
        self._save_metadata(metadata)

    def _save_metadata(self, metadata: Dict):
        """Save checkpoint metadata atomically.

        Writes to a temp file then os.replace()s it into place so a concurrent
        reader (or another rank) never observes a half-written JSON file.
        """
        tmp = self.metadata_file.with_name(self.metadata_file.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp, self.metadata_file)

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from file."""
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def _reconcile_with_disk(self):
        """Drop metadata entries whose checkpoint file no longer exists and
        recompute latest/best so neither ever points at a missing file.

        Self-heals a stale ``checkpoints.json`` (e.g. after checkpoints were
        rotated, deleted, or partially restored out-of-band). Idempotent and a
        no-op when metadata already matches disk.
        """
        try:
            metadata = self._load_metadata()
        except (json.JSONDecodeError, OSError):
            # Corrupt/unreadable metadata: start clean rather than crash.
            self._init_metadata()
            return

        checkpoints = metadata.get("checkpoints") or []
        present = [c for c in checkpoints if os.path.exists(c["path"])]
        # Collapse duplicate-path entries left by legacy second-resolution
        # timestamps (several saves in one second reused a single filename).
        by_path = {}
        for c in present:
            if c["path"] not in by_path or c.get("is_best"):
                by_path[c["path"]] = c
        surviving = list(by_path.values())

        def _points_at_missing(entry):
            return bool(entry) and not os.path.exists(entry["path"])

        if (
            len(surviving) == len(checkpoints)
            and not _points_at_missing(metadata.get("latest_checkpoint"))
            and not _points_at_missing(metadata.get("best_checkpoint"))
        ):
            return  # already consistent

        surviving.sort(key=lambda c: c["timestamp"])
        best = [c for c in surviving if c.get("is_best")]
        metadata["checkpoints"] = surviving
        metadata["latest_checkpoint"] = surviving[-1] if surviving else None
        metadata["best_checkpoint"] = best[-1] if best else None
        self._save_metadata(metadata)

        dropped = len(checkpoints) - len(surviving)
        if dropped:
            self.logger.info(
                f"Reconciled checkpoint metadata: dropped {dropped} stale/"
                f"duplicate entr{'y' if dropped == 1 else 'ies'}, "
                f"{len(surviving)} remain"
            )

    def save_checkpoint(
        self, state_dict: Dict, is_final: bool = False, is_best: bool = False
    ) -> str:
        """
        Save model checkpoint.

        Args:
            state_dict: Dictionary containing model state and other information
            is_final: Whether this is the final checkpoint
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        try:
            # Generate checkpoint filename (microseconds keep rapid saves unique)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if is_final:
                filename = f"final_checkpoint_{timestamp}.pt"
            elif is_best:
                filename = f"best_checkpoint_{timestamp}.pt"
            else:
                filename = f"checkpoint_{timestamp}.pt"

            checkpoint_path = self.checkpoint_dir / filename

            # Save checkpoint
            torch.save(state_dict, checkpoint_path)

            # Update metadata
            metadata = self._load_metadata()
            checkpoint_info = {
                "path": str(checkpoint_path),
                "timestamp": timestamp,
                "epoch": state_dict.get("epoch", 0),
                "is_final": is_final,
                "is_best": is_best,
            }

            metadata["checkpoints"].append(checkpoint_info)
            if is_best:
                metadata["best_checkpoint"] = checkpoint_info
            metadata["latest_checkpoint"] = checkpoint_info

            self._save_metadata(metadata)

            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest checkpoint.

        Returns:
            Loaded state dictionary or None if no checkpoint found
        """
        try:
            if checkpoint_path is None:
                metadata = self._load_metadata()
                if not metadata["checkpoints"]:
                    return None
                checkpoint_path = metadata["latest_checkpoint"]["path"]

            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None

            try:
                state_dict = torch.load(checkpoint_path, weights_only=True)
            except Exception:
                # Older/richer checkpoints (metrics dicts etc.) need the full
                # loader; fall back so existing files keep loading.
                state_dict = torch.load(checkpoint_path, weights_only=False)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state_dict

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        metadata = self._load_metadata()
        if metadata["best_checkpoint"]:
            return metadata["best_checkpoint"]["path"]
        return None

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        metadata = self._load_metadata()
        if metadata["latest_checkpoint"]:
            return metadata["latest_checkpoint"]["path"]
        return None

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Remove old checkpoints, keeping only the specified number of most recent ones.

        Args:
            keep_last_n: Number of most recent checkpoints to keep
        """
        try:
            metadata = self._load_metadata()
            checkpoints = metadata["checkpoints"]

            if len(checkpoints) <= keep_last_n:
                return

            # Sort checkpoints by timestamp
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

            # Keep the most recent checkpoints and best checkpoint
            keep_paths = {c["path"] for c in checkpoints[:keep_last_n]}
            if metadata["best_checkpoint"]:
                keep_paths.add(metadata["best_checkpoint"]["path"])

            # Remove old checkpoints
            for checkpoint in checkpoints:
                if checkpoint["path"] not in keep_paths:
                    try:
                        os.remove(checkpoint["path"])
                    except Exception as e:
                        self.logger.warning(
                            f"Error removing checkpoint {checkpoint['path']}: {str(e)}"
                        )

            # Update metadata
            metadata["checkpoints"] = [
                c for c in checkpoints if c["path"] in keep_paths
            ]
            self._save_metadata(metadata)

            self.logger.info(
                f"Cleaned up old checkpoints, keeping {len(keep_paths)} most recent ones"
            )

        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {str(e)}")
            raise

    def backup_checkpoints(self, backup_dir: str):
        """
        Create a backup of all checkpoints.

        Args:
            backup_dir: Directory to store backup
        """
        try:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copy all checkpoint files
            for checkpoint in self._load_metadata()["checkpoints"]:
                src = Path(checkpoint["path"])
                dst = backup_dir / src.name
                shutil.copy2(src, dst)

            # Copy metadata
            shutil.copy2(self.metadata_file, backup_dir / self.metadata_file.name)

            self.logger.info(f"Created checkpoint backup in {backup_dir}")

        except Exception as e:
            self.logger.error(f"Error creating checkpoint backup: {str(e)}")
            raise
