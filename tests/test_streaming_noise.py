import numpy as np
import pytest

from anc.plant import ANCPlant
from anc.streaming import StreamingFxLMS, run_streaming_closed_loop


def _setup():
    plant = ANCPlant(np.array([0.8, 0.1]), np.array([0.7, 0.2]), 2)
    controller = StreamingFxLMS(8, 0.01, np.array([0.7, 0.2]), 2)
    x = np.sin(0.3 * np.arange(500))
    return x, plant, controller


def test_streaming_zero_noise_is_bit_identical():
    x, plant, controller = _setup()
    _, e_a, y_a = run_streaming_closed_loop(x, plant, controller)
    _, e_b, y_b = run_streaming_closed_loop(
        x, plant, controller, measurement_noise=np.zeros_like(x))
    np.testing.assert_array_equal(e_a, e_b)
    np.testing.assert_array_equal(y_a, y_b)


def test_streaming_noise_is_added_to_error():
    x, plant, controller = _setup()
    v = np.full_like(x, 1e-3)
    _, e, y = run_streaming_closed_loop(x, plant, controller, measurement_noise=v)

    d = plant.primary_output(x)
    y_s = plant.secondary_output(y)
    np.testing.assert_allclose(e, d + y_s + v, atol=1e-12)


def test_streaming_noise_shape_is_checked():
    x, plant, controller = _setup()
    with pytest.raises(ValueError, match="shape"):
        run_streaming_closed_loop(x, plant, controller,
                                  measurement_noise=np.zeros(x.size - 1))