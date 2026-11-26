"""Fault behaviour of the streaming controller (spec section 8.1).

Each scenario is the Python twin of a C++ test in cpp/tests/unit_tests.cpp,
with the same inputs and expectations. Every test runs in float64 and float32.
"""

import numpy as np
import pytest

from anc.streaming import Fault, StreamingFxLMS

DTYPES = [np.float64, np.float32]


@pytest.fixture(params=DTYPES, ids=["float64", "float32"])
def dtype(request):
    return request.param


def test_healthy_run_reports_no_fault(dtype):
    c = StreamingFxLMS(2, 0.1, np.array([1.0]), dtype=dtype)
    c.process(1.0, 0.5)
    c.process(2.0, 1.0)
    assert not c.diverged
    assert c.fault is Fault.NONE
    assert c.samples_processed == 2


def test_nan_error_latches_and_leaves_weights_untouched(dtype):
    c = StreamingFxLMS(2, 0.1, np.array([1.0]), dtype=dtype)
    c.process(1.0, 0.5)                       # n = 0: w = [-0.05, 0]
    assert c.process(2.0, float("nan")) == 0.0

    assert c.fault is Fault.NON_FINITE_INPUT
    assert c.fault_sample == 1
    np.testing.assert_allclose(c.weights, [-0.05, 0.0], atol=1e-7)


def test_infinite_reference_latches(dtype):
    c = StreamingFxLMS(2, 0.1, np.array([1.0]), dtype=dtype)
    assert c.process(float("inf"), 0.5) == 0.0
    assert c.fault is Fault.NON_FINITE_INPUT
    assert c.fault_sample == 0


def test_error_limit_latches_before_update(dtype):
    c = StreamingFxLMS(2, 0.1, np.array([1.0]), dtype=dtype, max_abs_error=10.0)
    c.process(1.0, 0.5)
    assert c.process(1.0, 10.0) != 0.0       # |e| equal to the limit is allowed
    w_before = c.weights.copy()
    assert c.process(1.0, -10.5) == 0.0

    assert c.fault is Fault.ERROR_LIMIT
    assert c.fault_sample == 2
    np.testing.assert_array_equal(c.weights, w_before)


def test_weight_norm_limit_restores_previous_weights(dtype):
    c = StreamingFxLMS(1, 1.0, np.array([1.0]), dtype=dtype, max_weight_norm=10.0)
    assert c.process(1.0, -4.0) == pytest.approx(4.0)
    assert c.process(1.0, -4.0) == pytest.approx(8.0)
    assert c.process(1.0, -4.0) == 0.0

    assert c.fault is Fault.WEIGHT_NORM_LIMIT
    assert c.fault_sample == 2
    np.testing.assert_allclose(c.weights, [8.0])


def test_overflowing_weights_latch_and_are_restored():
    # float32 only: the same product is finite in float64.
    c = StreamingFxLMS(1, 1.0, np.array([1.0]), dtype=np.float32)
    assert c.process(3.0e38, -3.0e38) == 0.0
    assert c.fault is Fault.WEIGHTS_NOT_FINITE
    np.testing.assert_array_equal(c.weights, [0.0])


def test_latched_controller_outputs_silence_and_keeps_state(dtype):
    c = StreamingFxLMS(2, 0.1, np.array([1.0]), dtype=dtype)
    c.process(1.0, 0.5)
    c.process(1.0, float("nan"))
    w_latched = c.weights.copy()

    for _ in range(5):
        assert c.process(1.0, 0.5) == 0.0
    np.testing.assert_array_equal(c.weights, w_latched)
    assert c.fault_sample == 1
    assert c.samples_processed == 7


def test_reset_clears_the_fault(dtype):
    c = StreamingFxLMS(2, 0.1, np.array([1.0]), dtype=dtype)
    c.process(float("nan"), 0.5)
    assert c.diverged

    c.reset()
    assert not c.diverged
    assert c.samples_processed == 0
    assert c.process(1.0, 0.5) == pytest.approx(-0.05)


@pytest.mark.parametrize("kwargs", [
    {"max_abs_error": 0.0},
    {"max_abs_error": -1.0},
    {"max_weight_norm": float("nan")},
    {"max_weight_norm": -1.0},
])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        StreamingFxLMS(2, 0.1, np.array([1.0]), **kwargs)


def test_fault_names_match_cpp():
    assert [f.value for f in Fault] == [
        "none", "non_finite_input", "error_limit",
        "weights_not_finite", "weight_norm_limit", "output_not_finite",
    ]