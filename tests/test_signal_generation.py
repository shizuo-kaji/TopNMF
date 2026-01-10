"""Tests for signal_generation module."""

import pytest
import numpy as np
from TopNMF.signal_generation import (
    generate_signals,
    generate_mixed_periodic_nonperiodic,
    generate_noisy_periodic,
    generate_complex_signals,
    generate_noisy_signals,
    generate_step_signals,
    normalize_signals,
    create_time_array,
)


class TestCreateTimeArray:
    """Tests for create_time_array function."""

    def test_default_parameters(self):
        """Test default time array creation."""
        t = create_time_array()
        assert len(t) == 100
        assert t[0] == 0
        assert np.isclose(t[-1], 2 * np.pi)

    def test_custom_parameters(self):
        """Test custom time array parameters."""
        t = create_time_array(start=1, stop=10, n_points=50)
        assert len(t) == 50
        assert t[0] == 1
        assert t[-1] == 10

    def test_linearly_spaced(self):
        """Test that array is linearly spaced."""
        t = create_time_array(start=0, stop=10, n_points=11)
        diffs = np.diff(t)
        assert np.allclose(diffs, 1.0)


class TestGenerateSignals:
    """Tests for generate_signals function."""

    def test_cosine_signals(self, time_array):
        """Test cosine signal generation."""
        signals = generate_signals(time_array, kind="cosine")
        assert "cosine 1" in signals
        assert "cosine 2" in signals
        assert len(signals["cosine 1"]) == len(time_array)
        assert len(signals["cosine 2"]) == len(time_array)

    def test_triangle_signals(self, time_array):
        """Test triangle signal generation."""
        signals = generate_signals(time_array, kind="triangle")
        assert "cosine 1" in signals
        assert "triangle 1" in signals

    def test_case_insensitive(self, time_array):
        """Test that kind parameter is case insensitive."""
        signals1 = generate_signals(time_array, kind="COSINE")
        signals2 = generate_signals(time_array, kind="cosine")
        np.testing.assert_array_equal(signals1["cosine 1"], signals2["cosine 1"])

    def test_invalid_kind_raises(self, time_array):
        """Test that invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be either"):
            generate_signals(time_array, kind="invalid")


class TestGenerateMixedPeriodicNonperiodic:
    """Tests for generate_mixed_periodic_nonperiodic function."""

    def test_signal_keys(self, time_array):
        """Test that correct signal keys are present."""
        signals = generate_mixed_periodic_nonperiodic(time_array)
        assert "linear_cosine" in signals
        assert "quadratic_cosine" in signals
        assert "gaussian_cosine" in signals

    def test_signal_shapes(self, time_array):
        """Test that all signals have correct shape."""
        signals = generate_mixed_periodic_nonperiodic(time_array)
        for name, sig in signals.items():
            assert len(sig) == len(time_array)


class TestGenerateNoisyPeriodic:
    """Tests for generate_noisy_periodic function."""

    def test_signal_count(self, time_array):
        """Test that correct number of signals generated."""
        signals = generate_noisy_periodic(time_array)
        assert len(signals) == 4

    def test_signal_values_finite(self, time_array):
        """Test that all signal values are finite."""
        signals = generate_noisy_periodic(time_array)
        for name, sig in signals.items():
            assert np.all(np.isfinite(sig))


class TestGenerateComplexSignals:
    """Tests for generate_complex_signals function."""

    def test_signal_keys(self, time_array):
        """Test that all expected signals are present."""
        signals = generate_complex_signals(time_array)
        expected_keys = [
            "ramp_cosine", "quadratic_cosine", "gaussian_bump",
            "chirp", "step_cosine", "sawtooth_cosine"
        ]
        for key in expected_keys:
            assert key in signals


class TestGenerateNoisySignals:
    """Tests for generate_noisy_signals function."""

    def test_default_sample_count(self, time_array):
        """Test default number of samples."""
        signals = generate_noisy_signals(time_array)
        assert len(signals) == 5

    def test_custom_sample_count(self, time_array):
        """Test custom number of samples."""
        signals = generate_noisy_signals(time_array, n_samples=3)
        assert len(signals) == 3

    def test_reproducibility_with_seed(self, time_array):
        """Test that seed makes results reproducible."""
        signals1 = generate_noisy_signals(time_array, seed=42)
        signals2 = generate_noisy_signals(time_array, seed=42)
        for key in signals1:
            np.testing.assert_array_equal(signals1[key], signals2[key])

    def test_noise_scale(self, time_array):
        """Test that noise scale affects variance."""
        signals_low = generate_noisy_signals(time_array, noise_scale=0.01, seed=42)
        signals_high = generate_noisy_signals(time_array, noise_scale=1.0, seed=42)
        # Higher noise should lead to higher variance
        var_low = np.var(list(signals_low.values())[0])
        var_high = np.var(list(signals_high.values())[0])
        assert var_high > var_low


class TestGenerateStepSignals:
    """Tests for generate_step_signals function."""

    def test_step_presence(self, time_array):
        """Test that step discontinuity is present."""
        signals = generate_step_signals(time_array, n_samples=1, noise_scale=0, seed=42)
        sig = list(signals.values())[0]
        # Signal should have different mean before and after 2*pi
        mid_idx = np.searchsorted(time_array, 2 * np.pi)
        mean_before = np.mean(sig[:mid_idx])
        mean_after = np.mean(sig[mid_idx:])
        assert mean_after > mean_before


class TestNormalizeSignals:
    """Tests for normalize_signals function."""

    def test_max_normalization(self, time_array):
        """Test max normalization."""
        signals = {"test": np.array([1, 2, 3, 4, 5])}
        normalized = normalize_signals(signals, method='max')
        assert np.max(np.abs(normalized["test"])) == 1.0

    def test_std_normalization(self, time_array):
        """Test standard normalization."""
        signals = {"test": np.array([1, 2, 3, 4, 5], dtype=float)}
        normalized = normalize_signals(signals, method='std')
        assert np.isclose(np.mean(normalized["test"]), 0, atol=1e-10)
        assert np.isclose(np.std(normalized["test"]), 1, atol=1e-10)

    def test_minmax_normalization(self, time_array):
        """Test min-max normalization."""
        signals = {"test": np.array([1, 2, 3, 4, 5], dtype=float)}
        normalized = normalize_signals(signals, method='minmax')
        assert np.min(normalized["test"]) == 0.0
        assert np.max(normalized["test"]) == 1.0

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        signals = {"test": np.array([1, 2, 3])}
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_signals(signals, method='invalid')
