"""Model registry with version control + approval workflow.

Persists a JSON manifest at models/registry.json mapping each model name to its
versions. Each version records a content hash (sha256 of the model file), the source path,
optional metrics, an approval status, and a timestamp. Promotion to
"production" requires explicit approval, giving a minimal deploy-approval gate.

Pure stdlib; writes are atomic (tmp file + os.replace).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "registry.json",
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    """Versioned, approval-gated registry of trained model artifacts."""

    def __init__(self, path: str = _DEFAULT_PATH, clock=time.time):
        self.path = path
        self._clock = clock  # injectable for deterministic tests
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    # --- persistence -----------------------------------------------------
    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)

    def _save(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        d = os.path.dirname(self.path)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # --- API -------------------------------------------------------------
    def register(
        self, path: str, name: str, metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Register a model artifact as a new pending version. Returns the entry."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = self._load()
        versions = data.setdefault(name, [])
        entry = {
            "version": len(versions) + 1,
            "sha256": _sha256(path),
            "path": path,
            "metrics": metrics or {},
            "status": "pending",
            "ts": self._clock(),
        }
        versions.append(entry)
        self._save(data)
        return entry

    def _set_status(self, name: str, version: int, status: str) -> Dict[str, Any]:
        data = self._load()
        for e in data.get(name, []):
            if e["version"] == version:
                e["status"] = status
                self._save(data)
                return e
        raise KeyError(f"{name} v{version} not found")

    def approve(self, name: str, version: int) -> Dict[str, Any]:
        return self._set_status(name, version, "production")

    def reject(self, name: str, version: int) -> Dict[str, Any]:
        return self._set_status(name, version, "rejected")

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        return list(self._load().get(name, []))

    def get_production(self, name: str) -> Optional[Dict[str, Any]]:
        """Latest approved ('production') version, or None."""
        approved = [
            e for e in self._load().get(name, []) if e["status"] == "production"
        ]
        return max(approved, key=lambda e: e["version"]) if approved else None
