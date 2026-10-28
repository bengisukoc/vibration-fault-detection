import numpy as np
import pytest

from vibration_analysis import analyse_signal, classify_condition, generate_signal


@pytest.mark.parametrize("condition", ["normal", "imbalance", "bearing_fault"])
def test_generated_signals_have_expected_length(condition):
    t, signal = generate_signal(condition, fs=2000, duration=1.0)
    assert len(t) == len(signal) == 2000


def test_imbalance_has_larger_rms_than_normal():
    _, normal = generate_signal("normal")
    _, imbalance = generate_signal("imbalance")
    assert analyse_signal(imbalance, 12000)["rms"] > 3 * analyse_signal(normal, 12000)["rms"]


def test_default_cases_are_classified_correctly():
    for index, condition in enumerate(["normal", "imbalance", "bearing_fault"]):
        _, signal = generate_signal(condition, seed=42 + index)
        assert classify_condition(analyse_signal(signal, 12000)) == condition


def test_invalid_condition_is_rejected():
    with pytest.raises(ValueError):
        generate_signal("unknown")



def test_envelope_spectrum_reveals_impact_frequency():
    _, signal = generate_signal("bearing_fault", seed=44)
    assert analyse_signal(signal, 12000)["envelope_peak_hz"] == pytest.approx(120.0, abs=1.0)


def test_bearing_fault_raises_resonance_energy():
    _, normal = generate_signal("normal", seed=42)
    _, fault = generate_signal("bearing_fault", seed=44)
    assert analyse_signal(fault, 12000)["resonance_energy"] > 10 * analyse_signal(normal, 12000)["resonance_energy"]


@pytest.mark.parametrize("seed", range(100, 120))
def test_classification_is_robust_to_noise_seed(seed):
    for condition in ["normal", "imbalance", "bearing_fault"]:
        _, signal = generate_signal(condition, seed=seed)
        assert classify_condition(analyse_signal(signal, 12000)) == condition
