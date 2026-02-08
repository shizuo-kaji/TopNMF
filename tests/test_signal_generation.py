from __future__ import annotations

import pytest


def test_create_ichimatsu_pattern_is_reproducible(signal_generation_module, np):
    first = signal_generation_module.create_ichimatsu_pattern(
        num_samples=3,
        image_shape=(10, 10),
        min_pat=1,
        max_pat=3,
        seed=123,
    )
    second = signal_generation_module.create_ichimatsu_pattern(
        num_samples=3,
        image_shape=(10, 10),
        min_pat=1,
        max_pat=3,
        seed=123,
    )

    assert first.shape == (3, 10, 10)
    assert np.array_equal(first, second)


def test_create_ichimatsu_pattern_binarize_outputs_binary_values(signal_generation_module, np):
    generated = signal_generation_module.create_ichimatsu_pattern(
        num_samples=2,
        image_shape=(9, 9),
        min_pat=1,
        max_pat=2,
        binarize=True,
        seed=7,
    )

    unique_values = set(np.unique(generated).tolist())
    assert unique_values.issubset({0.0, 1.0})


def test_create_ichimatsu_pattern_rejects_large_pattern(signal_generation_module, np):
    with pytest.raises(ValueError, match="smaller than or equal"):
        signal_generation_module.create_ichimatsu_pattern(
            num_samples=1,
            image_shape=(4, 4),
            pat=np.ones((5, 5)),
            min_pat=1,
            max_pat=1,
        )


def test_generate_signals_triangle_keys_and_shapes(signal_generation_module, time_array):
    generated = signal_generation_module.generate_signals(time_array, kind="triangle")

    assert set(generated) == {"triangle 1", "triangle 2"}
    assert all(values.shape == time_array.shape for values in generated.values())


def test_generate_signals_rejects_unknown_kind(signal_generation_module, time_array):
    with pytest.raises(ValueError, match="kind must be either"):
        signal_generation_module.generate_signals(time_array, kind="unknown")


def test_normalize_signals_minmax(signal_generation_module, np):
    normalized = signal_generation_module.normalize_signals(
        {"sample": np.array([-2.0, 0.0, 2.0])},
        method="minmax",
    )
    np.testing.assert_allclose(normalized["sample"], np.array([0.0, 0.5, 1.0]))


def test_normalize_signals_rejects_unknown_method(signal_generation_module, np):
    with pytest.raises(ValueError, match="Unknown normalization method"):
        signal_generation_module.normalize_signals(
            {"sample": np.array([1.0, 2.0, 3.0])},
            method="median",
        )


def test_create_time_array_respects_bounds_and_size(signal_generation_module):
    values = signal_generation_module.create_time_array(start=-1.0, stop=1.0, n_points=11)

    assert values.shape == (11,)
    assert values[0] == pytest.approx(-1.0)
    assert values[-1] == pytest.approx(1.0)
