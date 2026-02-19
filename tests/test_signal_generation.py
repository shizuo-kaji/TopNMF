"""Tests for signal generation utilities."""

from __future__ import annotations

import pytest

from TopNMF.signal_generation import (
    generate_ichimatsu_pattern,
    generate_signals,
    normalize_signals,
)


def test_generate_ichimatsu_pattern_is_reproducible(np):
    first = generate_ichimatsu_pattern(
        num_samples=3, image_shape=(10, 10), min_pat=1, max_pat=3, seed=123,
    )
    second = generate_ichimatsu_pattern(
        num_samples=3, image_shape=(10, 10), min_pat=1, max_pat=3, seed=123,
    )

    assert first.shape == (3, 10, 10)
    assert np.array_equal(first, second)


def test_generate_ichimatsu_pattern_binarize_outputs_binary_values(np):
    generated = generate_ichimatsu_pattern(
        num_samples=2, image_shape=(9, 9), min_pat=1, max_pat=2,
        binarize=True, seed=7,
    )

    unique_values = set(np.unique(generated).tolist())
    assert unique_values.issubset({0.0, 1.0})


def test_generate_ichimatsu_pattern_rejects_large_pattern(np):
    with pytest.raises(ValueError, match="smaller than or equal"):
        generate_ichimatsu_pattern(
            num_samples=1, image_shape=(4, 4), pat=np.ones((5, 5)),
            min_pat=1, max_pat=1,
        )


def test_generate_signals_returns_list_with_correct_length(time_array):
    generated = generate_signals(time_array, kind="triangle", num=3)

    assert isinstance(generated, list)
    assert len(generated) == 3
    assert all(s.shape == time_array.shape for s in generated)


def test_generate_signals_rejects_unknown_kind(time_array):
    with pytest.raises(ValueError, match="kind must be either"):
        generate_signals(time_array, kind="unknown")


def test_normalize_signals_1d(np):
    normalized = normalize_signals(np.array([-2.0, 0.0, 2.0]))
    np.testing.assert_allclose(normalized, np.array([0.0, 0.5, 1.0]))


def test_normalize_signals_2d(np):
    data = np.array([[1.0, 3.0], [0.0, 4.0]])
    normalized = normalize_signals(data)
    np.testing.assert_allclose(normalized, np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_normalize_signals_list(np):
    result = normalize_signals([np.array([2.0, 4.0]), np.array([0.0, 10.0])])
    assert isinstance(result, list)
    np.testing.assert_allclose(result[0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(result[1], np.array([0.0, 1.0]))
