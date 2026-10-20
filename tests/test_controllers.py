import numpy as np

from anc.controllers import FixedFIRController
from anc.metrics import attenuation_db
from anc.plant import ANCPlant
from anc.simulate import run

def make_plant() -> ANCPlant:
    return ANCPlant(
        primary_taps=np.array(
            [0.80, 0.00, 0.15, 0.00, -0.08, 0.00, 0.04]
        ),
        secondary_taps=np.array(
            [0.70, 0.00, 0.20, 0.00, -0.05]
        ),
        secondary_delay_samples=2,
    )

def test_zero_controller_gives_zero_attenuation():
    plant = make_plant()

    rng = np.random.default_rng(42)
    reference = rng.standard_normal(2000)

    controller = FixedFIRController(
        taps=np.zeros(32, dtype=np.float64)
    )

    trace = run(controller, plant, reference)

    attenuation = attenuation_db(
        trace.primary_noise,
        trace.residual,
        start_sample=100,
    )

    assert np.isclose(attenuation, 0.0, atol=1e-12)

    np.testing.assert_allclose(
        trace.residual,
        trace.primary_noise,
        rtol=1e-12,
        atol=1e-12,
    )

def test_secondary_path_delay():
    plant = make_plant()

    impulse = np.zeros(8, dtype=np.float64)
    impulse[0] = 1.0

    response = plant.secondary_output(impulse)

    excepted = np.array(
        [0.0, 0.0, 0.70, 0.0, 0.20, 0.0, -0.05, 0.0]
    )

    np.testing.assert_allclose(
        response,
        excepted,
        rtol=1e-12,
        atol=1e-12,
    )

def test_residual_consistency():
    plant = make_plant()

    rng = np.random.default_rng(123)
    reference = rng.standard_normal(2000)

    controller = FixedFIRController(
        taps=np.linspace(-0.1, 0.1, 32)
    )

    trace = run(controller, plant, reference)

    expected_primary = plant.primary_output(reference)
    expected_anti = plant.secondary_output(controller.process(reference))

    np.testing.assert_allclose(
        trace.residual,
        expected_primary + expected_anti,
        rtol=1e-12,
        atol=1e-12,
    )