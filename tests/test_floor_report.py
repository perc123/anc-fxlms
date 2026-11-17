import numpy as np
import pytest

from anc.measurement import (
    floor_report,
    generate_measurement_noise,
    noise_std_for_snr,
)
from anc.metrics import attenuation_db


def _signals(true_att_db, snr_db, n=50_000):
    t = np.arange(n)
    d = np.sin(2 * np.pi * 125 * t / 2000)
    r = d * 10 ** (-true_att_db / 20)
    if snr_db is None:
        v = np.zeros(n)
    else:
        v = generate_measurement_noise(n, noise_std_for_snr(d, snr_db), seed=1)
    return d, r, v


def test_true_attenuation_matches_existing_metric():
    """The 'true' column must agree exactly with what the CSVs already contain."""
    d, r, v = _signals(30.0, 40.0)
    report = floor_report(d, r, v, start_sample=1000)
    assert report.true_attenuation_db == attenuation_db(d, r, start_sample=1000)


def test_noise_free_report_is_unchanged_and_never_flagged():
    d, r, v = _signals(90.0, None)
    report = floor_report(d, r, v)
    assert report.measured_attenuation_db == report.true_attenuation_db
    assert report.floor_db == float("inf")
    assert report.floor_limited is False


def test_well_above_floor_is_not_flagged():
    d, r, v = _signals(20.0, 60.0)
    report = floor_report(d, r, v)
    assert report.measured_attenuation_db == pytest.approx(20.0, abs=0.05)
    assert report.floor_limited is False


def test_residual_equal_to_noise_sits_3db_below_floor():
    """The 3 dB rule from Lesson 1.4."""
    d, r, v = _signals(60.0, 60.0)
    report = floor_report(d, r, v)
    assert report.floor_db - report.measured_attenuation_db == pytest.approx(3.0, abs=0.1)


def test_deep_cancellation_is_pinned_to_floor_and_flagged():
    d, r, v = _signals(90.0, 40.0)
    report = floor_report(d, r, v)
    assert report.true_attenuation_db == pytest.approx(90.0, abs=0.01)
    assert report.measured_attenuation_db == pytest.approx(report.floor_db, abs=0.05)
    assert report.floor_limited is True


def test_window_and_shape_validation():
    d, r, v = _signals(20.0, 40.0, n=100)
    with pytest.raises(ValueError):
        floor_report(d, r[:-1], v)
    with pytest.raises(ValueError):
        floor_report(d, r, v, start_sample=100)
    with pytest.raises(ValueError):
        floor_report(d, r, v, start_sample=50, stop_sample=40)


def test_nan_inside_window_raises():
    d, r, v = _signals(20.0, 40.0, n=100)
    r = r.copy()
    r[60] = np.nan
    with pytest.raises(ValueError, match="finite"):
        floor_report(d, r, v)
    floor_report(d, r, v, stop_sample=60)   # NaN outside the window is fine