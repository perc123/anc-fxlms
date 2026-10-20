import numpy as np
import pytest

from anc.metrics import attenuation_db, rms


def test_rms_known_value():
    signal = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.isclose(rms(signal), 1.0)


def test_rms_dc_signal():
    signal = np.full(100, 3.0)
    assert np.isclose(rms(signal), 3.0)


def test_rms_empty_raises():
    with pytest.raises(ValueError):
        rms(np.array([]))


def test_rms_2d_raises():
    with pytest.raises(ValueError):
        rms(np.ones((4, 4)))


def test_attenuation_db_equal_signals():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    assert np.isclose(attenuation_db(x, x), 0.0, atol=1e-12)


def test_attenuation_db_known_reduction():
    before = np.ones(100)
    after = np.full(100, 0.5)
    result = attenuation_db(before, after)
    assert np.isclose(result, 20.0 * np.log10(2.0))


def test_attenuation_db_zero_after_returns_inf():
    before = np.ones(100)
    after = np.zeros(100)
    assert attenuation_db(before, after) == float("inf")


def test_attenuation_db_zero_before_raises():
    with pytest.raises(ValueError):
        attenuation_db(np.zeros(100), np.ones(100))


def test_attenuation_db_start_sample():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1000)
    result_full = attenuation_db(x, x, start_sample=0)
    result_partial = attenuation_db(x, x, start_sample=500)
    assert np.isclose(result_full, 0.0, atol=1e-12)
    assert np.isclose(result_partial, 0.0, atol=1e-12)


def test_attenuation_db_invalid_start_sample_raises():
    x = np.ones(100)
    with pytest.raises(ValueError):
        attenuation_db(x, x, start_sample=100)


def test_attenuation_db_length_mismatch_raises():
    with pytest.raises(ValueError):
        attenuation_db(np.ones(100), np.ones(50))
