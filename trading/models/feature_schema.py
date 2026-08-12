"""Versioned, immutable feature contracts shared by training and inference."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral, Real
from typing import Mapping, Sequence, Tuple


class FeatureContractError(ValueError):
    """A feature vector or fitted component is incompatible with its schema."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One named feature and its logical interchange dtype."""

    name: str
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("feature name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("feature name must not have surrounding whitespace")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("feature dtype must be float32 or float64")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Ordered feature schema with a stable identity and strict validation."""

    schema_id: str
    version: int
    features: Tuple[FeatureSpec, ...]
    preprocessing_id: str = "identity.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.schema_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9_.-]*", self.schema_id
        ):
            raise ValueError("schema_id must be a lowercase dotted identifier")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("schema version must be an integer")
        if self.version < 1:
            raise ValueError("schema version must be positive")
        if not isinstance(self.features, tuple) or not self.features:
            raise TypeError("features must be a non-empty tuple")
        if not all(isinstance(feature, FeatureSpec) for feature in self.features):
            raise TypeError("features must contain only FeatureSpec values")
        if len(set(self.names)) != len(self.features):
            raise ValueError("feature names must be unique")
        if not isinstance(self.preprocessing_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9_.-]*", self.preprocessing_id
        ):
            raise ValueError("preprocessing_id must be a lowercase dotted identifier")

    @property
    def identity(self) -> str:
        return f"{self.schema_id}@{self.version}"

    @property
    def size(self) -> int:
        return len(self.features)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def dtypes(self) -> Tuple[str, ...]:
        return tuple(feature.dtype for feature in self.features)

    def vectorize(self, values: Mapping[str, object]) -> Tuple[float, ...]:
        """Return finite values in schema order; unrelated context is ignored."""
        if not isinstance(values, Mapping):
            raise TypeError("feature values must be a mapping")

        result = []
        for feature in self.features:
            if feature.name not in values:
                raise FeatureContractError(
                    f"feature {feature.name!r} is missing for schema {self.identity}"
                )
            raw_value = values[feature.name]
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (Real, Decimal)
            ):
                raise FeatureContractError(
                    f"feature {feature.name!r} must be a real number"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise FeatureContractError(f"feature {feature.name!r} must be finite")
            result.append(value)
        return tuple(result)

    def validate_names(self, names: Sequence[object], owner: str) -> None:
        """Require exact feature names and order for a named model component."""
        actual = tuple(names)
        if actual != self.names:
            raise FeatureContractError(
                f"{owner} feature names do not match schema {self.identity}"
            )

    def validate_fitted_component(self, component: object, owner: str) -> None:
        """Check fitted sklearn-like dimensional and optional name metadata."""
        count = getattr(component, "n_features_in_", None)
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise FeatureContractError(
                f"{owner} does not declare a fitted feature dimension"
            )
        if int(count) != self.size:
            raise FeatureContractError(
                f"{owner} expects {int(count)} features, schema {self.identity} "
                f"declares {self.size}"
            )

        names = getattr(component, "feature_names_in_", None)
        if names is not None:
            self.validate_names(tuple(names), owner)

    def manifest_payload(self) -> dict[str, object]:
        """Return the canonical JSON representation embedded in manifests."""
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "preprocessing_id": self.preprocessing_id,
            "features": [
                {"name": feature.name, "dtype": feature.dtype}
                for feature in self.features
            ],
        }
