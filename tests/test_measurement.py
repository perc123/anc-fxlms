import numpy as np
import pytest

from anc.measurement import (
    generate_measurement_noise,
    is_floor_limited,
    measurement_floor_db,
    noise_std_for_snr,
)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _tone(num_samples=200_000, fs=2000.0, f=125.0):
    n = np.arange(num_samples)
    return np.sin(2 * np.pi * f * n / fs)


def test_noise_hits_requested_snr():
    d = _tone()
    std = noise_std_for_snr(d, snr_db=40.0)
    v = generate_measurement_noise(d.size, std, seed=1)
    achieved = 20 * np.log10(_rms(d) / _rms(v))
    assert achieved == pytest.approx(40.0, abs=0.1)


def test_noise_is_reproducible_for_same_seed():
    a = generate_measurement_noise(1000, 0.1, seed=3)
    b = generate_measurement_noise(1000, 0.1, seed=3)
    np.testing.assert_array_equal(a, b)


def test_noise_differs_for_different_seeds():
    a = generate_measurement_noise(1000, 0.1, seed=3)
    b = generate_measurement_noise(1000, 0.1, seed=4)
    assert not np.array_equal(a, b)


def test_noise_stream_is_independent_of_signal_stream():
    signal_rng_draw = np.random.default_rng(3).normal(0.0, 0.1, 1000)
    noise = generate_measurement_noise(1000, 0.1, seed=3)
    assert not np.array_equal(signal_rng_draw, noise)


def test_zero_std_gives_zeros_and_infinite_floor():
    d = _tone(1000)
    v = generate_measurement_noise(d.size, 0.0, seed=1)
    assert np.all(v == 0.0)
    assert measurement_floor_db(d, v) == float("inf")


def test_floor_is_close_to_snr_at_high_snr():
    d = _tone()
    v = generate_measurement_noise(d.size, noise_std_for_snr(d, 60.0), seed=1)
    assert measurement_floor_db(d, v) == pytest.approx(60.0, abs=0.1)


@pytest.mark.parametrize(
    "measured, expected",
    [(50.0, False), (56.9, False), (57.0, True), (59.9, True)],
)
def test_is_floor_limited(measured, expected):
    assert is_floor_limited(measured, floor_db=60.0, margin_db=3.0) is expected


def test_infinite_floor_is_never_limited():
    assert is_floor_limited(300.0, float("inf")) is False


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        noise_std_for_snr(np.zeros(10), 40.0)
    with pytest.raises(ValueError):
        noise_std_for_snr(_tone(10), float("nan"))
    with pytest.raises(ValueError):
        generate_measurement_noise(-1, 0.1, seed=0)
    with pytest.raises(ValueError):
        generate_measurement_noise(10, -0.1, seed=0)
    with pytest.raises(ValueError):
        is_floor_limited(10.0, 60.0, margin_db=-1.0)