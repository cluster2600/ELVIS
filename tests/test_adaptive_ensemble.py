"""Tests for trading.signals.adaptive_ensemble (roadmap item #11)."""

import json
import math

import pytest

from trading.signals.adaptive_ensemble import AdaptiveEnsembleWeights


class TestWeightsNormalization:
    def test_weights_sum_to_one_and_keep_proportions(self):
        ens = AdaptiveEnsembleWeights({"a": 0.6, "b": 0.4})
        weights = ens.weights
        assert math.isclose(sum(weights.values()), 1.0)
        assert math.isclose(weights["a"], 0.6)
        assert math.isclose(weights["b"], 0.4)

    def test_all_zero_accuracies_give_uniform_weights(self):
        ens = AdaptiveEnsembleWeights({"a": 0.0, "b": 0.0, "c": 0.0})
        weights = ens.weights
        assert weights == {
            "a": pytest.approx(1 / 3),
            "b": pytest.approx(1 / 3),
            "c": pytest.approx(1 / 3),
        }

    def test_empty_initials_give_empty_weights(self):
        ens = AdaptiveEnsembleWeights({})
        assert ens.weights == {}

    def test_out_of_range_initial_accuracies_are_clamped(self):
        ens = AdaptiveEnsembleWeights({"hot": 1.5, "cold": -0.2})
        assert ens.accuracies == {"hot": 1.0, "cold": 0.0}

    def test_non_finite_initial_accuracy_becomes_zero(self):
        ens = AdaptiveEnsembleWeights({"a": float("nan"), "b": 0.5})
        assert ens.accuracies["a"] == 0.0
        assert math.isclose(ens.weights["b"], 1.0)


class TestEmaUpdate:
    def test_ema_math_pinned(self):
        # new = 0.7 * 0.9 + 0.3 * 0.5 = 0.78
        ens = AdaptiveEnsembleWeights({"a": 0.5}, ema_alpha=0.7)
        ens.update("a", 0.9)
        assert ens.accuracies["a"] == pytest.approx(0.78)

    def test_ema_second_step(self):
        # 0.78 -> 0.7 * 0.4 + 0.3 * 0.78 = 0.514
        ens = AdaptiveEnsembleWeights({"a": 0.5}, ema_alpha=0.7)
        ens.update("a", 0.9)
        ens.update("a", 0.4)
        assert ens.accuracies["a"] == pytest.approx(0.514)

    def test_unknown_model_is_added_with_observed_accuracy(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        ens.update("new_model", 0.6)
        assert ens.accuracies["new_model"] == pytest.approx(0.6)

    def test_nan_accuracy_update_is_ignored(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        ens.update("a", float("nan"))
        assert ens.accuracies["a"] == pytest.approx(0.5)

    def test_non_numeric_accuracy_update_is_ignored(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        ens.update("a", "oops")
        assert ens.accuracies["a"] == pytest.approx(0.5)

    def test_out_of_range_update_is_clamped_before_ema(self):
        # observed 2.0 clamps to 1.0 -> new = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        ens = AdaptiveEnsembleWeights({"a": 0.5}, ema_alpha=0.7)
        ens.update("a", 2.0)
        assert ens.accuracies["a"] == pytest.approx(0.85)

    def test_boundary_alpha_zero_and_one(self):
        frozen = AdaptiveEnsembleWeights({"a": 0.5}, ema_alpha=0.0)
        frozen.update("a", 0.9)
        assert frozen.accuracies["a"] == pytest.approx(0.5)

        instant = AdaptiveEnsembleWeights({"a": 0.5}, ema_alpha=1.0)
        instant.update("a", 0.9)
        assert instant.accuracies["a"] == pytest.approx(0.9)

    def test_out_of_range_alpha_is_clamped(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5}, ema_alpha=1.7)
        ens.update("a", 0.9)  # clamped alpha == 1.0
        assert ens.accuracies["a"] == pytest.approx(0.9)


class TestWeightedSignal:
    def test_full_intersection_weighted_average(self):
        ens = AdaptiveEnsembleWeights({"a": 0.6, "b": 0.4})
        signal = ens.weighted_signal({"a": 1.0, "b": -1.0})
        assert signal == pytest.approx(0.6 * 1.0 + 0.4 * -1.0)

    def test_subset_is_renormalized(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5, "b": 0.3, "c": 0.2})
        # Only a and b report: weights become 0.5/0.8 and 0.3/0.8.
        signal = ens.weighted_signal({"a": 1.0, "b": -1.0})
        assert signal == pytest.approx((0.5 - 0.3) / 0.8)

    def test_empty_intersection_returns_neutral_zero(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        assert ens.weighted_signal({"unknown": 1.0}) == 0.0

    def test_empty_predictions_return_neutral_zero(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        assert ens.weighted_signal({}) == 0.0

    def test_no_models_tracked_returns_neutral_zero(self):
        ens = AdaptiveEnsembleWeights({})
        assert ens.weighted_signal({"a": 1.0}) == 0.0

    def test_nan_prediction_is_skipped(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5, "b": 0.5})
        signal = ens.weighted_signal({"a": float("nan"), "b": 1.0})
        assert signal == pytest.approx(1.0)

    def test_all_nan_predictions_return_neutral_zero(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        assert ens.weighted_signal({"a": float("nan")}) == 0.0

    def test_zero_weight_subset_falls_back_to_plain_mean(self):
        # a dominates the weights, but only zero-accuracy models report.
        ens = AdaptiveEnsembleWeights({"a": 0.9, "b": 0.0, "c": 0.0})
        signal = ens.weighted_signal({"b": 1.0, "c": 0.5})
        assert signal == pytest.approx(0.75)


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        state = str(tmp_path / "weights.json")
        first = AdaptiveEnsembleWeights(
            {"a": 0.5, "b": 0.5}, ema_alpha=0.7, state_path=state
        )
        first.update("a", 0.9)
        first.save()

        second = AdaptiveEnsembleWeights(
            {"a": 0.5, "b": 0.5}, ema_alpha=0.7, state_path=state
        )
        assert second.accuracies["a"] == pytest.approx(0.78)
        assert second.accuracies["b"] == pytest.approx(0.5)

    def test_missing_state_file_falls_back_to_initials(self, tmp_path):
        state = str(tmp_path / "does_not_exist.json")
        ens = AdaptiveEnsembleWeights({"a": 0.6}, state_path=state)
        assert ens.accuracies == {"a": 0.6}

    def test_corrupt_state_file_falls_back_to_initials(self, tmp_path):
        state = tmp_path / "weights.json"
        state.write_text("{not valid json!!", encoding="utf-8")
        ens = AdaptiveEnsembleWeights({"a": 0.6}, state_path=str(state))
        assert ens.accuracies == {"a": 0.6}
        assert ens.load() is False

    def test_wrong_shape_state_file_falls_back_to_initials(self, tmp_path):
        state = tmp_path / "weights.json"
        state.write_text(json.dumps({"accuracies": [1, 2, 3]}), encoding="utf-8")
        ens = AdaptiveEnsembleWeights({"a": 0.6}, state_path=str(state))
        assert ens.accuracies == {"a": 0.6}

    def test_loaded_state_merges_over_initials(self, tmp_path):
        state = tmp_path / "weights.json"
        state.write_text(
            json.dumps({"ema_alpha": 0.7, "accuracies": {"a": 0.9}}), encoding="utf-8"
        )
        # "b" is new in code and absent from the saved state: it must survive.
        ens = AdaptiveEnsembleWeights({"a": 0.5, "b": 0.4}, state_path=str(state))
        assert ens.accuracies == {"a": pytest.approx(0.9), "b": pytest.approx(0.4)}

    def test_save_without_state_path_is_noop(self):
        ens = AdaptiveEnsembleWeights({"a": 0.5})
        ens.save()  # must not raise
        assert ens.load() is False

    def test_save_writes_valid_json_atomically(self, tmp_path):
        state = tmp_path / "weights.json"
        ens = AdaptiveEnsembleWeights({"a": 0.5}, state_path=str(state))
        ens.save()
        payload = json.loads(state.read_text(encoding="utf-8"))
        assert payload["accuracies"] == {"a": 0.5}
        assert payload["ema_alpha"] == pytest.approx(0.7)
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_save_creates_missing_parent_directory(self, tmp_path):
        state = tmp_path / "nested" / "dir" / "weights.json"
        ens = AdaptiveEnsembleWeights({"a": 0.5}, state_path=str(state))
        ens.save()
        assert state.exists()
