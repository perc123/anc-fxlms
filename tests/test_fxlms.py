import numpy as np

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.plant import ANCPlant

def test_fxlms_update_uses_negative_gradient():
    controller = FxLMSController(
        filter_length=2,
        step_size=0.1,
        estimated_secondary_taps=np.array([0.7]),
    )

    controller.update(
        filtered_reference_vector=np.array([1.0, 2.0]),
        error=0.5,
    )

    np.testing.assert_allclose(
        controller.weights,
        [-0.05, -0.1],
    )


def test_fxlms_run_returns_expected_shapes():
    reference = np.random.default_rng(42).normal(size=200)

    plant = ANCPlant(
        primary_taps=np.array([0.8, 0.1]),
        secondary_taps=np.array([0.7, 0.2]),
        secondary_delay_samples=2,
    )

    controller = FxLMSController(
        filter_length=8,
        step_size=0.005,
        estimated_secondary_taps=np.array([0.65, 0.25]),
        estimated_secondary_delay_samples=2,
    )

    trace = run_fxlms(reference, plant, controller)

    assert trace.primary_noise.shape == reference.shape
    assert trace.control.shape == reference.shape
    assert trace.anti_noise.shape == reference.shape
    assert trace.residual.shape == reference.shape
    assert trace.filtered_reference.shape == reference.shape
    assert trace.weights.shape == (reference.size, 8)

    for array in (
        trace.primary_noise,
        trace.control,
        trace.anti_noise,
        trace.residual,
        trace.filtered_reference,
        trace.weights,
    ):
        assert np.isfinite(array).all()


def text_fxlms_anti_noise_respects_secondary_delay():
    reference = np.ones(20)

    plant = ANCPlant(
        primary_taps=np.array([1.0]),
        secondary_taps=np.array([0.7]),
        secondary_delay_samples=2,
    )

    controller = FxLMSController(
        filter_length=4,
        step_size=0.001,
        estimated_secondary_taps=np.array([0.7]),
        estimated_secondary_delay_samples=2,
    )

    trace = run_fxlms(reference, plant, controller)

    np.testing.assert_array_equal(
        trace.anti_noise[:2],
        np.zeros,
    )


def test_zero_secondary_path_produces_np_anti_noise():
    reference = np.random.default_rng(7).normal(size=100)

    plant = ANCPlant(
        primary_taps=np.array([0.8, 0.1]),
        secondary_taps=np.array([0.0]),
        secondary_delay_samples=1,
    )

    controller = FxLMSController(
        filter_length=4,
        step_size=0.001,
        estimated_secondary_taps=np.array([0.7]),
        estimated_secondary_delay_samples=1,
    )

    trace = run_fxlms(reference, plant, controller)

    np.testing.assert_array_equal(
        trace.anti_noise,
        np.zeros_like(reference),
    )
    np.testing.assert_allclose(
        trace.residual,
        trace.primary_noise,
    )