import numpy as np
import pytest

from experiments.step3_causality_sweep import (
    add_primary_delay,
    causality_margin_samples,
)


def test_add_primary_delay_prepends_zeros():
    taps = np.array([0.8, 0.2, -0.1])

    result = add_primary_delay(taps, 2)

    np.testing.assert_array_equal(
        result,
        [0.0, 0.0, 0.8, 0.2, -0.1],
    )


def test_add_zero_primary_delay_preserves_taps():
    taps = np.array([0.8, 0.2, -0.1])

    result = add_primary_delay(taps, 0)

    np.testing.assert_array_equal(result, taps)


def test_add_primary_delay_rejects_non_finite_taps():
    with pytest.raises(ValueError):
        add_primary_delay(np.array([1.0, np.nan]), 2)


def test_causality_margin_at_boundary():
    assert causality_margin_samples(
        primary_delay_samples=2,
        secondary_delay_samples=2,
        processing_delay_samples=0,
    ) == 0


def test_causality_margin_with_insufficient_preview():
    assert causality_margin_samples(
        primary_delay_samples=1,
        secondary_delay_samples=2,
        processing_delay_samples=0,
    ) == -1


def test_causality_margin_includes_processing_delay():
    assert causality_margin_samples(
        primary_delay_samples=3,
        secondary_delay_samples=2,
        processing_delay_samples=2,
    ) == -1


def test_causality_margin_rejects_negative_delay():
    with pytest.raises(ValueError):
        causality_margin_samples(
            primary_delay_samples=2,
            secondary_delay_samples=-1,
            processing_delay_samples=0,
        )