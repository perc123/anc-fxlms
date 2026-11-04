import pytest
from experiments.step3_phase_error_sweep import phase_error_degrees


def test_zero_phase_error_for_matched_delay():
    result = phase_error_degrees(
        frequency_hz=125.0,
        sample_rate_hz=2000,
        estimated_delay_samples=8,
        actual_delay_samples=8,
    )

    assert result == pytest.approx(0.0)
    

def test_positive_ninety_degree_phase_error():
    result = phase_error_degrees(
        frequency_hz=125.0,
        sample_rate_hz=2000,
        estimated_delay_samples=4,
        actual_delay_samples=8,
    )

    assert result == pytest.approx(90.0)


def test_negative_ninety_degree_phase_error():
    result = phase_error_degrees(
        frequency_hz=125.0,
        sample_rate_hz=2000,
        estimated_delay_samples=12,
        actual_delay_samples=8,
    )

    assert result == pytest.approx(-90.0)


def test_positive_180_degree_phase_error():
    result = phase_error_degrees(
        frequency_hz=125.0,
        sample_rate_hz=2000,
        estimated_delay_samples=0,
        actual_delay_samples=8,
    )

    assert result == pytest.approx(180.0)


def test_negative_180_degree_phase_error():
    result = phase_error_degrees(
        frequency_hz=125.0,
        sample_rate_hz=2000,
        estimated_delay_samples=16,
        actual_delay_samples=8,
    )

    assert result == pytest.approx(-180.0)


def test_phase_error_rejects_invalid_frequency():
    with pytest.raises(ValueError):
        phase_error_degrees(
            frequency_hz=1000.0,
            sample_rate_hz=2000,
            estimated_delay_samples=8,
            actual_delay_samples=8,
        )


def test_phase_error_rejects_negative_delay():
    with pytest.raises(ValueError):
        phase_error_degrees(
            frequency_hz=125.0,
            sample_rate_hz=2000,
            estimated_delay_samples=-1,
            actual_delay_samples=8,
        )