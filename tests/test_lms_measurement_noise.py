import numpy as np
import pytest

from anc.adaptive import run_lms
from anc.controllers import LMSController
from anc.measurement import make_measurement_noise, measurement_floor_db
from anc.metrics import attenuation_db
from anc.plant import apply_fir

PRIMARY_TAPS = np.array([0.80, 0.00, 0.15, 0.00, -0.08, 0.00, 0.04])


def _setup(step_size=0.01):
    reference = np.random.default_rng(42).normal(size=8000)
    desired = apply_fir(reference, PRIMARY_TAPS)
    controller = LMSController(filter_length=32, step_size=step_size)
    return reference, desired, controller


def test_no_noise_and_zero_noise_are_bit_identical():
    reference, desired, controller = _setup()
    a = run_lms(reference, desired, controller)
    b = run_lms(reference, desired, controller,
                measurement_noise=np.zeros_like(desired))

    np.testing.assert_array_equal(a.error, b.error)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.measured_error, a.error)


def test_measured_error_is_error_plus_noise():
    reference, desired, controller = _setup()
    v = make_measurement_noise(desired, 40.0, seed=1)
    trace = run_lms(reference, desired, controller, measurement_noise=v)

    np.testing.assert_allclose(trace.measured_error, trace.error + v)
    # The true error and the stored desired signal stay noise-free:
    np.testing.assert_allclose(trace.error, desired - trace.control)
    np.testing.assert_array_equal(trace.desired, desired)


def test_noise_enters_the_adaptation():
    reference, desired, controller = _setup()
    clean = run_lms(reference, desired, controller)
    v = make_measurement_noise(desired, 40.0, seed=1)
    noisy = run_lms(reference, desired, controller, measurement_noise=v)

    assert not np.array_equal(clean.weights, noisy.weights)


def test_weights_still_converge_near_true_path():
    """Observation noise is uncorrelated with x: the optimum is unchanged (4.2)."""
    reference, desired, controller = _setup()
    v = make_measurement_noise(desired, 40.0, seed=1)
    trace = run_lms(reference, desired, controller, measurement_noise=v)

    final = trace.weights[-1]
    np.testing.assert_allclose(final[:PRIMARY_TAPS.size], PRIMARY_TAPS, atol=0.01)
    np.testing.assert_allclose(final[PRIMARY_TAPS.size:], 0.0, atol=0.01)


def test_measured_reduction_cannot_exceed_floor():
    reference, desired, controller = _setup()
    v = make_measurement_noise(desired, 40.0, seed=1)
    trace = run_lms(reference, desired, controller, measurement_noise=v)

    start = 6000
    measured = attenuation_db(desired[start:] + v[start:],
                              trace.measured_error[start:])
    floor = measurement_floor_db(desired[start:], v[start:])

    assert trace.status == "completed"
    assert measured <= floor + 0.1


def test_noise_shape_mismatch_raises():
    reference, desired, controller = _setup()
    with pytest.raises(ValueError, match="shape"):
        run_lms(reference, desired, controller,
                measurement_noise=np.zeros(desired.size - 1))


def test_non_finite_noise_raises():
    reference, desired, controller = _setup()
    v = np.zeros_like(desired)
    v[10] = np.inf
    with pytest.raises(ValueError, match="finite"):
        run_lms(reference, desired, controller, measurement_noise=v)