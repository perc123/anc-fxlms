import numpy as np
import pytest

from experiments.step2_signal_variants import (
    generate_variant_reference,
)


SAMPLE_RATE = 2000
DURATION = 2.0


def test_generate_single_tone_variant():
    signal = generate_variant_reference(
        {
            "type": "single_tone",
            "frequency_hz": 80.0,
            "amplitude": 1.0,
        },
        SAMPLE_RATE,
        DURATION,
        42,
    )

    assert signal.shape == (4000,)
    assert np.isfinite(signal).all()
    assert np.std(signal) == pytest.approx(
        1.0 / np.sqrt(2.0),
        abs=0.01,
    )


def test_generate_multi_tone_variant():
    signal = generate_variant_reference(
        {
            "type": "multi_tone",
            "tones": [
                {"frequency_hz": 80.0, "amplitude": 0.7},
                {"frequency_hz": 120.0, "amplitude": 0.4},
            ],
        },
        SAMPLE_RATE,
        DURATION,
        42,
    )

    assert signal.shape == (4000,)
    assert np.isfinite(signal).all()


def test_generate_white_noise_variant():
    signal = generate_variant_reference(
        {
            "type": "white_noise",
            "noise_std": 0.5,
        },
        SAMPLE_RATE,
        DURATION,
        42,
    )

    assert signal.shape == (4000,)
    assert np.isfinite(signal).all()
    assert np.std(signal) == pytest.approx(0.5, abs=0.03)


def test_generate_band_limited_noise_variant():
    signal = generate_variant_reference(
        {
            "type": "band_limited_noise",
            "low_cutoff_hz": 50.0,
            "high_cutoff_hz": 300.0,
            "noise_std": 0.5,
            "filter_length": 129,
        },
        SAMPLE_RATE,
        DURATION,
        42,
    )

    assert signal.shape == (4000,)
    assert np.isfinite(signal).all()
    assert np.std(signal) == pytest.approx(0.5)


def test_variant_generation_is_reproducible():
    variant = {
        "type": "band_limited_noise",
        "low_cutoff_hz": 50.0,
        "high_cutoff_hz": 300.0,
        "noise_std": 0.5,
    }

    first = generate_variant_reference(
        variant, SAMPLE_RATE, DURATION, 42
    )
    second = generate_variant_reference(
        variant, SAMPLE_RATE, DURATION, 42
    )

    np.testing.assert_array_equal(first, second)


def test_unsupported_signal_type_raises():
    with pytest.raises(ValueError):
        generate_variant_reference(
            {"type": "unsupported"},
            SAMPLE_RATE,
            DURATION,
            42,
        )
