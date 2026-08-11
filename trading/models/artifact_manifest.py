"""Strict feature manifests for joblib model components."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from trading.models.feature_schema import FeatureContractError, FeatureSchema

FORMAT_VERSION = 1


class ArtifactCompatibilityError(FeatureContractError):
    """An artefact cannot safely be used by the current model adapter."""


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Model implementation identity checked before deserialization."""

    model_kind: str
    library: str
    library_version: str

    def __post_init__(self) -> None:
        for field_name in ("model_kind", "library", "library_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a non-empty trimmed string")

    def manifest_payload(self) -> dict[str, str]:
        return {
            "model_kind": self.model_kind,
            "library": self.library,
            "library_version": self.library_version,
        }


def _component_paths(
    components: Mapping[str, Path], manifest_path: Path
) -> dict[str, Path]:
    if not isinstance(components, Mapping) or not components:
        raise TypeError("components must be a non-empty mapping")
    expected_parent = manifest_path.parent.resolve()
    paths: dict[str, Path] = {}
    for name, raw_path in components.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise TypeError("component names must be non-empty strings")
        path = Path(raw_path)
        if not path.is_file():
            raise ArtifactCompatibilityError(f"model component {name!r} is missing")
        if path.is_symlink():
            raise ArtifactCompatibilityError(
                f"model component {name!r} must not be a symbolic link"
            )
        if path.parent.resolve() != expected_parent:
            raise ArtifactCompatibilityError(
                f"model component {name!r} must be beside its manifest"
            )
        paths[name] = path
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise ArtifactCompatibilityError("model components must be distinct files")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_feature_manifest(
    manifest_path: Path,
    schema: FeatureSchema,
    descriptor: ArtifactDescriptor,
    components: Mapping[str, Path],
) -> None:
    """Write a canonical manifest last, after all model components exist."""
    if not isinstance(schema, FeatureSchema):
        raise TypeError("schema must be a FeatureSchema")
    if not isinstance(descriptor, ArtifactDescriptor):
        raise TypeError("descriptor must be an ArtifactDescriptor")
    path = Path(manifest_path)
    resolved_components = _component_paths(components, path)
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": descriptor.manifest_payload(),
        "feature_schema": schema.manifest_payload(),
        "components": {
            name: {"filename": component.name, "sha256": _sha256(component)}
            for name, component in sorted(resolved_components.items())
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def validate_feature_manifest(
    manifest_path: Path,
    schema: FeatureSchema,
    descriptor: ArtifactDescriptor,
    components: Mapping[str, Path],
) -> None:
    """Validate schema identity/order and every component digest."""
    if not isinstance(schema, FeatureSchema):
        raise TypeError("schema must be a FeatureSchema")
    if not isinstance(descriptor, ArtifactDescriptor):
        raise TypeError("descriptor must be an ArtifactDescriptor")
    path = Path(manifest_path)
    if path.is_symlink():
        raise ArtifactCompatibilityError("feature manifest must not be a symbolic link")
    resolved_components = _component_paths(components, path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            "feature manifest is missing or malformed"
        ) from exc

    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "created_at",
        "artifact",
        "feature_schema",
        "components",
    }:
        raise ArtifactCompatibilityError("feature manifest has an invalid structure")
    if payload["format_version"] != FORMAT_VERSION:
        raise ArtifactCompatibilityError("feature manifest format is unsupported")
    try:
        created_at = datetime.fromisoformat(payload["created_at"])
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "feature manifest creation time is invalid"
        ) from exc
    if created_at.utcoffset() is None:
        raise ArtifactCompatibilityError(
            "feature manifest creation time must be timezone-aware"
        )
    if payload["artifact"] != descriptor.manifest_payload():
        raise ArtifactCompatibilityError(
            "feature manifest model implementation is incompatible"
        )
    if payload["feature_schema"] != schema.manifest_payload():
        raise ArtifactCompatibilityError(
            f"feature manifest schema is incompatible with {schema.identity}"
        )

    component_payload = payload["components"]
    if not isinstance(component_payload, dict) or set(component_payload) != set(
        resolved_components
    ):
        raise ArtifactCompatibilityError("feature manifest components are incompatible")

    for name, component in resolved_components.items():
        metadata = component_payload[name]
        if not isinstance(metadata, dict) or set(metadata) != {"filename", "sha256"}:
            raise ArtifactCompatibilityError(
                f"feature manifest component {name!r} is malformed"
            )
        if metadata["filename"] != component.name:
            raise ArtifactCompatibilityError(
                f"feature manifest component {name!r} has another filename"
            )
        if metadata["sha256"] != _sha256(component):
            raise ArtifactCompatibilityError(
                f"feature manifest component {name!r} digest does not match"
            )


def validate_classifier_classes(
    classifier: object, expected_classes: Sequence[object], owner: str
) -> None:
    """Require a fitted classifier to expose the expected class order."""
    actual = getattr(classifier, "classes_", None)
    if actual is None or tuple(actual) != tuple(expected_classes):
        raise ArtifactCompatibilityError(
            f"{owner} class order is incompatible with the strategy"
        )
