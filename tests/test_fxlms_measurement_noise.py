import numpy as np
import pytest

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.measurement import make_measurement_noise, measurement_floor_db
from anc.metrics import attenuation_db
from anc.plant import ANCPlant


def _setup(step_size=0.005):
    # Causality-sweep plant at zero margin: cancels very well when noise-free.
    plant = ANCPlant(
        primary_taps=np.array([0, 0, 0.80, 0.00, 0.15, 0.00, -0.08, 0.00, 0.04]),
        secondary_taps=np.array([0.70, 0.00, 0.20, 0.00, -0.05]),
        secondary_delay_samples=2,
    )
    controller = FxLMSController(
        filter_length=32,
        step_size=step_size,
        estimated_secondary_taps=np.array([0.70, 0.00, 0.20, 0.00, -0.05]),
        estimated_secondary_delay_samples=2,
    )
    reference = np.random.default_rng(42).normal(size=8000)
    return reference, plant, controller


def test_no_noise_and_zero_noise_are_bit_identical():
    """The backward-compatibility guarantee from Lesson 3.2."""
    reference, plant, controller = _setup()
    a = run_fxlms(reference, plant, controller)
    b = run_fxlms(reference, plant, controller,
                  measurement_noise=np.zeros_like(reference))

    np.testing.assert_array_equal(a.residual, b.residual)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.measured_error, a.residual)


def test_measured_error_is_residual_plus_noise():
    reference, plant, controller = _setup()
    v = make_measurement_noise(plant.primary_output(reference), 40.0, seed=1)
    trace = run_fxlms(reference, plant, controller, measurement_noise=v)

    np.testing.assert_allclose(trace.measured_error, trace.residual + v)
    # The acoustic residual must stay noise-free physics:
    np.testing.assert_allclose(trace.residual,
                               trace.primary_noise + trace.anti_noise)
    np.testing.assert_array_equal(trace.measurement_noise, v)


def test_noise_enters_the_adaptation():
    """Proves the controller adapts on e_m, not on r."""
    reference, plant, controller = _setup()
    clean = run_fxlms(reference, plant, controller)
    v = make_measurement_noise(plant.primary_output(reference), 40.0, seed=1)
    noisy = run_fxlms(reference, plant, controller, measurement_noise=v)

    assert not np.array_equal(clean.weights, noisy.weights)


def test_measured_attenuation_cannot_exceed_floor():
    reference, plant, controller = _setup()
    d = plant.primary_output(reference)
    v = make_measurement_noise(d, 40.0, seed=1)
    trace = run_fxlms(reference, plant, controller, measurement_noise=v)

    start = 6000
    measured = attenuation_db(d[start:] + v[start:], trace.measured_error[start:])
    floor = measurement_floor_db(d[start:], v[start:])

    assert trace.status == "completed"
    assert measured <= floor + 0.1


def test_noise_shape_mismatch_raises():
    reference, plant, controller = _setup()
    with pytest.raises(ValueError, match="shape"):
        run_fxlms(reference, plant, controller,
                  measurement_noise=np.zeros(reference.size - 1))


def test_non_finite_noise_raises():
    reference, plant, controller = _setup()
    v = np.zeros_like(reference)
    v[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        run_fxlms(reference, plant, controller, measurement_noise=v)


def test_make_measurement_noise_none_means_noise_free():
    assert make_measurement_noise(np.ones(10), None, seed=1) is None