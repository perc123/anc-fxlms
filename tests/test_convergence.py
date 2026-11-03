import numpy as np
import pytest
from anc.convergence import analyze_convergence


def test_steady_state_rms_and_attenuation():
    primary = np.full(100, 2.0)
    residual = np.concatenate([
        np.full(80, 2.0),
        np.full(20, 1.0),
    ])

    result = analyze_convergence(
        primary_noise=primary,
        residual=residual,
        sample_rate_hz=10,
        block_duration_s=1.0,
        steady_state_fraction=0.2,
    )

    assert result.steady_state_start_sample == 80

    assert result.steady_state_primary_rms == pytest.approx(2.0)
    assert result.steady_state_residual_rms == pytest.approx(1.0)

    assert result.steady_state_attenuation_db == pytest.approx(
        20.0 * np.log10(2.0)
    )


def test_convergence_time_detects_three_stable_blocks():
    primary = np.full(100, 2.0)

    residual = np.concatenate([
        np.full(80, 2.0),
        np.full(20, 1.0),
    ])           

    result = analyze_convergence(
        primary_noise=primary,
        residual=residual,
        sample_rate_hz=10,
        block_duration_s=1.0,
        steady_state_fraction=0.2,
        tolerance_db=1.0,
        consecutive_blocks=2,
    )

    assert result.convergence_time_s == pytest.approx(8.0)                                                  

def test_convergence_time_is_none_without_enough_stable_blocks():
    primary = np.full(100, 2.0)

    residual = np.concatenate([
        np.full(90, 2.0),
        np.full(10, 1.0),
    ])

    result = analyze_convergence(
        primary_noise=primary,
        residual=residual,
        sample_rate_hz=10,
        block_duration_s=1.0,
        steady_state_fraction=0.1,
        tolerance_db=1.0,
        consecutive_blocks=3,
    )

    assert result.convergence_time_s is None


def test_block_rms_arrays_have_expected_length():
    primary = np.ones(100)
    residual = np.full(100, 0.5)

    result = analyze_convergence(
        primary_noise=primary,
        residual=residual,
        sample_rate_hz=100,
        block_duration_s=0.2,
    )

    assert result.block_start_times_s.shape == (5,)
    assert result.block_primary_rms.shape == (5,)
    assert result.block_residual_rms.shape == (5,)



def test_evaluation_can_start_after_transient():
    primary = np.full(100, 2.0)
    residual = np.full(100, 1.0)

    result = analyze_convergence(
        primary_noise=primary,
        residual=residual,
        sample_rate_hz=10,
        start_sample=20,
        stop_sample=100,
        block_duration_s=1.0,
        steady_state_fraction=0.25,
    )   

    assert result.steady_state_start_sample == 80
    assert result.block_start_times_s[0] == pytest.approx(2.0)


def test_mismatched_signal_lengths_raise():
    with pytest.raises(ValueError):
        analyze_convergence(
            primary_noise=np.ones(100),
            residual=np.ones(99),
            sample_rate_hz=100,
        )


def test_non_finite_residual_raises():
    residual = np.ones(100)
    residual[50] = np.nan

    with pytest.raises(ValueError):
        analyze_convergence(
            primary_noise=np.ones(100),
            residual=residual,
            sample_rate_hz=100,
        )
