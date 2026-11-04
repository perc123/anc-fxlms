import numpy as np
import pytest

from anc.signals import (
    generate_band_limited_noise,
    generate_tone_mixture,
)


def test_generate_single_tone():
    signal = generate_tone_mixture(
        sample_rate_hz=2000,
        duration_s=1.0,
        tones=[{"frequency_hz": 80.0, "amplitude": 1.0}],
        seed=42,
    )

    assert signal.shape == (2000,)
    assert np.isfinite(signal).all()


def test_generate_white_noise():
    signal = generate_tone_mixture(
        sample_rate_hz=2000,
        duration_s=2.0,
        tones=[],
        white_noise_std=0.5,
        seed=42,
    )

    assert signal.shape == (4000,)
    assert np.std(signal) == pytest.approx(0.5, abs=0.03)


def test_band_limited_noise_is_reproducible():
    kwargs = dict(
        sample_rate_hz=2000,
        duration_s=2.0,
        low_cutoff_hz=50.0,
        high_cutoff_hz=300.0,
        noise_std=0.5,
        seed=42,
    )

    first = generate_band_limited_noise(**kwargs)
    second = generate_band_limited_noise(**kwargs)

    np.testing.assert_array_equal(first, second)


def test_band_limited_noise_has_expected_shape_and_std():
    signal = generate_band_limited_noise(
        sample_rate_hz=2000,
        duration_s=2.0,
        low_cutoff_hz=50.0,
        high_cutoff_hz=300.0,
        noise_std=0.5,
        seed=42,
    )

    assert signal.shape == (4000,)
    assert np.isfinite(signal).all()
    assert np.std(signal) == pytest.approx(0.5)


def test_band_limited_noise_zero_std():
    signal = generate_band_limited_noise(
        sample_rate_hz=2000,
        duration_s=1.0,
        low_cutoff_hz=50.0,
        high_cutoff_hz=300.0,
        noise_std=0.0,
    )

    np.testing.assert_array_equal(
        signal,
        np.zeros(2000),
    )


def test_band_limited_noise_rejects_invalid_cutoffs():
    with pytest.raises(ValueError):
        generate_band_limited_noise(
            sample_rate_hz=2000,
            duration_s=1.0,
            low_cutoff_hz=300.0,
            high_cutoff_hz=50.0,
        )


def test_band_limited_noise_rejects_even_filter_length():
    with pytest.raises(ValueError):
        generate_band_limited_noise(
            sample_rate_hz=2000,
            duration_s=1.0,
            low_cutoff_hz=50.0,
            high_cutoff_hz=300.0,
            filter_length=128,
        )