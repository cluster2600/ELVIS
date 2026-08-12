import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from trading.models.artifact_manifest import (
    ArtifactCompatibilityError,
    ArtifactDescriptor,
    FeatureContractError,
    validate_feature_manifest,
    write_feature_manifest,
)
from trading.models.feature_schemas import (
    RESEARCH_FINANCIAL_9_V1,
    RESEARCH_SOCIAL_11_V1,
)

SKLEARN_RF = ArtifactDescriptor(
    model_kind="random-forest-classifier",
    library="scikit-learn",
    library_version=sklearn.__version__,
)


def component_files(tmp_path: Path) -> dict[str, Path]:
    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    model_path.write_bytes(b"model-v1")
    scaler_path.write_bytes(b"scaler-v1")
    return {"model": model_path, "scaler": scaler_path}


def test_manifest_round_trip_validates_schema_order_and_hashes(tmp_path: Path) -> None:
    components = component_files(tmp_path)
    manifest_path = tmp_path / "feature_manifest.json"

    write_feature_manifest(
        manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
    )
    validate_feature_manifest(
        manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["artifact"] == SKLEARN_RF.manifest_payload()
    assert payload["feature_schema"]["schema_id"] == RESEARCH_FINANCIAL_9_V1.schema_id
    assert [item["name"] for item in payload["feature_schema"]["features"]] == list(
        RESEARCH_FINANCIAL_9_V1.names
    )


def test_manifest_rejects_a_different_expected_schema_before_loading(
    tmp_path: Path,
) -> None:
    components = component_files(tmp_path)
    manifest_path = tmp_path / "feature_manifest.json"
    write_feature_manifest(manifest_path, RESEARCH_SOCIAL_11_V1, SKLEARN_RF, components)

    with pytest.raises(FeatureContractError, match="schema"):
        validate_feature_manifest(
            manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
        )


def test_manifest_rejects_component_tampering(tmp_path: Path) -> None:
    components = component_files(tmp_path)
    manifest_path = tmp_path / "feature_manifest.json"
    write_feature_manifest(
        manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
    )
    components["model"].write_bytes(b"tampered")

    with pytest.raises(FeatureContractError, match="digest"):
        validate_feature_manifest(
            manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
        )


@pytest.mark.parametrize("payload", [b"", b"[]", b"{not-json}"])
def test_manifest_rejects_malformed_content(tmp_path: Path, payload: bytes) -> None:
    components = component_files(tmp_path)
    manifest_path = tmp_path / "feature_manifest.json"
    manifest_path.write_bytes(payload)

    with pytest.raises(FeatureContractError, match="manifest"):
        validate_feature_manifest(
            manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
        )


def test_manifest_rejects_another_library_version(tmp_path: Path) -> None:
    components = component_files(tmp_path)
    manifest_path = tmp_path / "feature_manifest.json"
    write_feature_manifest(
        manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
    )
    another_runtime = ArtifactDescriptor(
        model_kind=SKLEARN_RF.model_kind,
        library=SKLEARN_RF.library,
        library_version="0.0-incompatible",
    )

    with pytest.raises(ArtifactCompatibilityError, match="implementation"):
        validate_feature_manifest(
            manifest_path,
            RESEARCH_FINANCIAL_9_V1,
            another_runtime,
            components,
        )


def test_manifest_rejects_a_component_outside_its_directory(tmp_path: Path) -> None:
    manifest_directory = tmp_path / "manifest"
    manifest_directory.mkdir()
    outside = tmp_path / "model.joblib"
    outside.write_bytes(b"model")

    with pytest.raises(ArtifactCompatibilityError, match="beside"):
        write_feature_manifest(
            manifest_directory / "feature_manifest.json",
            RESEARCH_FINANCIAL_9_V1,
            SKLEARN_RF,
            {"model": outside},
        )


def test_manifest_rejects_a_symbolic_link_component(tmp_path: Path) -> None:
    real_model = tmp_path / "real-model.joblib"
    linked_model = tmp_path / "linked-model.joblib"
    real_model.write_bytes(b"model")
    linked_model.symlink_to(real_model)

    with pytest.raises(ArtifactCompatibilityError, match="symbolic link"):
        write_feature_manifest(
            tmp_path / "feature_manifest.json",
            RESEARCH_FINANCIAL_9_V1,
            SKLEARN_RF,
            {"model": linked_model},
        )


def test_training_persistence_inference_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    training = rng.normal(size=(30, RESEARCH_FINANCIAL_9_V1.size))
    labels = np.array([0, 1] * 15)
    scaler = StandardScaler().fit(training)
    model = RandomForestClassifier(n_estimators=3, random_state=7).fit(
        scaler.transform(training), labels
    )
    components = {
        "model": tmp_path / "model.joblib",
        "scaler": tmp_path / "scaler.joblib",
    }
    joblib.dump(model, components["model"])
    joblib.dump(scaler, components["scaler"])
    manifest_path = tmp_path / "feature_manifest.json"
    write_feature_manifest(
        manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
    )

    validate_feature_manifest(
        manifest_path, RESEARCH_FINANCIAL_9_V1, SKLEARN_RF, components
    )
    loaded_model = joblib.load(components["model"])
    loaded_scaler = joblib.load(components["scaler"])
    RESEARCH_FINANCIAL_9_V1.validate_fitted_component(loaded_model, "model")
    RESEARCH_FINANCIAL_9_V1.validate_fitted_component(loaded_scaler, "scaler")
    context = {
        name: float(training[0, index])
        for index, name in enumerate(RESEARCH_FINANCIAL_9_V1.names)
    }
    inference = np.asarray([RESEARCH_FINANCIAL_9_V1.vectorize(context)])

    assert np.array_equal(
        loaded_model.predict(loaded_scaler.transform(inference)),
        model.predict(scaler.transform(training[[0]])),
    )
