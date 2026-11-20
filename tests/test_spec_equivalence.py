"""Executable checks for docs/algorithm_specification.md."""

import numpy as np
import pytest

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.plant import ANCPlant, apply_fir
from anc.streaming import StreamingFxLMS

S_HAT = np.array([0.66, 0.0, 0.23, 0.0, -0.03])
S_HAT_DELAY = 2
LENGTH = 8
MU = 0.01


def _signals(n=400, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(scale=0.1, size=n)


def test_filtered_reference_is_the_same_in_both_orders():
    """Spec section 3: x'[n] = sum_k s_hat[k] x[n - D - k], zero initial state."""
    x, e = _signals()
    batch_xf = apply_fir(x, S_HAT, delay_samples=S_HAT_DELAY)

    streaming = StreamingFxLMS(LENGTH, MU, S_HAT, S_HAT_DELAY)
    stream_xf = []
    for xn, en in zip(x, e):
        streaming.process(xn, en)
        stream_xf.append(streaming._xf_hist[0])

    np.testing.assert_allclose(stream_xf, batch_xf, rtol=0, atol=1e-15)


def test_both_orders_apply_the_same_weight_updates():
    """Spec section 5.3: both orders pair e[n] with x'[n].

    Fed the same (x, e) sequence, the batch-style controller and the streaming
    controller follow the same weight trajectory. Only the moment at which y[n]
    is computed differs.
    """
    x, e = _signals()
    xf = apply_fir(x, S_HAT, delay_samples=S_HAT_DELAY)

    batch = FxLMSController(LENGTH, MU, S_HAT, S_HAT_DELAY)
    xf_hist = np.zeros(LENGTH)
    streaming = StreamingFxLMS(LENGTH, MU, S_HAT, S_HAT_DELAY)

    for n in range(x.size):
        xf_hist[1:] = xf_hist[:-1]
        xf_hist[0] = xf[n]
        batch.update(filtered_reference_vector=xf_hist, error=e[n])

        streaming.process(x[n], e[n])

        np.testing.assert_allclose(streaming.weights, batch.weights,
                                   rtol=0, atol=1e-12)


def test_output_differs_by_one_update():
    """Spec section 5.3: batch y[n] uses w[n]; streaming y[n] uses w[n+1]."""
    x, e = _signals()
    x_hist = np.zeros(LENGTH)
    streaming = StreamingFxLMS(LENGTH, MU, S_HAT, S_HAT_DELAY)

    previous_weights = streaming.weights.copy()
    for n in range(x.size):
        x_hist[1:] = x_hist[:-1]
        x_hist[0] = x[n]

        y_batch_style = previous_weights @ x_hist     # before the update
        y_stream = streaming.process(x[n], e[n])      # after the update

        assert y_stream == pytest.approx(streaming.weights @ x_hist, abs=1e-12)
        previous_weights = streaming.weights.copy()

    # The two outputs are generally different numbers.
    assert y_batch_style != pytest.approx(y_stream, abs=1e-12)


def test_batch_allows_zero_secondary_delay_streaming_does_not():
    """Spec section 6: causality requirements of the two harnesses."""
    from anc.streaming import run_streaming_closed_loop

    plant = ANCPlant(np.array([1.0]), np.array([1.0]), secondary_delay_samples=0)
    x = np.ones(16)

    trace = run_fxlms(x, plant, FxLMSController(4, 0.01, np.array([1.0]), 0))
    assert trace.status == "completed"

    with pytest.raises(ValueError):
        run_streaming_closed_loop(x, plant, StreamingFxLMS(4, 0.01, np.array([1.0])))