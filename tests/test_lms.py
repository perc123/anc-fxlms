import numpy as np
import pytest

from anc.controllers import LMSController
from anc.adaptive import run_lms

def test_lms_starts_with_zero_weights():
    controller = LMSController(filter_length=4, step_size=0.01)

    np.testing.assert_array_equal(
        controller.weights,
        np.zeros(4),
    )

def test_lms_update_uses_expected_sign():
    controller = LMSController(filter_length=2, step_size=0.1)

    controller.update(
        np.array([1.0, 2.0]),
        error=-0.5,
    )

    np.testing.assert_allclose(
        controller.weights,
        [0.05, 0.1],
    )

def test_lms_rejects_invalid_step_size():
    with pytest.raises(ValueError):
        LMSController(filter_length=4, step_size=0.0)

def test_lms_rejects_wrong_reference_vector_shape():
    controller = LMSController(filter_length=4, step_size=0.01)

    with pytest.raises(ValueError):
        controller.predict(np.zeros(3))

def test_lms_trace_has_expected_shapes():
    reference = np.ones(100)
    desired = np.zeros(100)

    controller = LMSController(filter_length=4, step_size=0.01)
    trace = run_lms(reference, desired, controller)

    assert trace.control.shape == (100,)
    assert trace.error.shape == (100,)
    assert trace.weights.shape == (100, 4)
    assert np.isfinite(trace.weights).all()
    assert trace.status == "completed"
    assert trace.processed_samples == reference.size
    assert trace.divergence_sample is None
    assert trace.divergence_reason is None

def test_lms_reduces_error_for_constant_reference():
    reference = np.ones(500)
    desired = np.full(500, 0.5)

    controller = LMSController(filter_length=1, step_size=0.05)
    trace = run_lms(reference, desired, controller)

    assert abs(trace.error[-1]) < abs(trace.error[0])


def test_lms_detects_excessive_step_size_divergence():
    reference = np.ones(20)
    desired = np.ones(20)

    controller = LMSController(
        filter_length=1,
        step_size=3.0,
    )

    trace = run_lms(
        reference=reference,
        desired=desired,
        controller=controller,
        max_weight_norm=10.0,
    )

    assert trace.status == "diverged"
    assert trace.divergence_sample == 3
    assert trace.processed_samples == 4
    assert "weight norm" in trace.divergence_reason

    assert np.isfinite(trace.error[:4]).all()
    assert np.isnan(trace.error[4:]).all()


def test_lms_detects_error_amplitude_limit():
    reference = np.zeros(20)
    desired = np.ones(20)

    controller = LMSController(
        filter_length=1,
        step_size=0.001,
    )

    trace = run_lms(
        reference=reference,
        desired=desired,
        controller=controller,
        max_abs_error=0.5,
    )

    assert trace.status == "diverged"
    assert trace.divergence_sample == 0
    assert trace.processed_samples == 1
    assert "Prediction error amplitude" in trace.divergence_reason


def test_lms_rejects_invalid_safety_limits():
    reference = np.ones(20)
    desired = np.ones(20)

    controller = LMSController(
        filter_length=1,
        step_size=0.001,
    )

    with pytest.raises(ValueError):
        run_lms(
            reference,
            desired,
            controller,
            max_weight_norm=0.0,
        )

    with pytest.raises(ValueError):
        run_lms(
            reference,
            desired,
            controller,
            max_abs_error=float("inf"),
        )
