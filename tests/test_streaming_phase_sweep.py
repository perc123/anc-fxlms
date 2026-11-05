"""The streaming controller must share the batch FxLMS stability window.

Configuration is that of configs/step3_phase_error_sweep.yaml: 125 Hz at
2 kHz, actual secondary delay 8, so an S_hat delay of 8 - d samples gives a
phase error of 22.5 * d degrees. An update-order bug in the streaming path
shifts this window by one sample (22.5 degrees).
"""

import numpy as np
import pytest

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.metrics import attenuation_db
from anc.plant import ANCPlant
from anc.streaming import StreamingFxLMS, run_streaming_closed_loop
from experiments.step3_phase_error_sweep import (
    generate_reference,
    phase_error_degrees,
)

FS = 2000
FREQ = 125.0
ACTUAL_DELAY = 8
S_TAPS = np.array([0.70, 0.00, 0.20, 0.00, -0.05])
PRIMARY = np.concatenate((
    np.zeros(12), [0.80, 0.00, 0.15, 0.00, -0.08, 0.00, 0.04],
))
LENGTH = 32
STEP = 0.005
EVAL_START = 1000
LIMIT = 10.0

# estimated delay -> phase error: 3 -112.5, 4 -90, 5 -67.5, 11 +67.5, 12 +90
CASES = [
    (3, False), (4, False), (5, True), (11, True), (12, False),
]


def _plant() -> ANCPlant:
    return ANCPlant(PRIMARY, S_TAPS.copy(), ACTUAL_DELAY)


def _reference() -> np.ndarray:
    return generate_reference(FREQ, 1.0, FS, 4.0, seed=42)


def _converged(residual: np.ndarray, primary_noise: np.ndarray) -> bool:
    if not np.isfinite(residual).all() or np.abs(residual).max() > LIMIT:
        return False
    return attenuation_db(primary_noise, residual, EVAL_START) > 20.0


def _batch_converges(delay: int) -> bool:
    reference = _reference()
    trace = run_fxlms(
        reference, _plant(),
        FxLMSController(LENGTH, STEP, S_TAPS.copy(), delay),
        max_weight_norm=100.0, max_abs_residual=LIMIT,
    )
    return trace.status == "completed" and _converged(
        trace.residual, trace.primary_noise
    )


def _streaming_converges(delay: int) -> bool:
    reference = _reference()
    plant = _plant()
    controller = StreamingFxLMS(LENGTH, STEP, S_TAPS.copy(), delay)
    with np.errstate(all="ignore"):
        _, e, _ = run_streaming_closed_loop(reference, plant, controller)
    return _converged(e, plant.primary_output(reference))


@pytest.mark.parametrize("delay, expected", CASES)
def test_phase_error_cases_are_on_the_expected_side(delay, expected):
    phase = phase_error_degrees(FREQ, FS, delay, ACTUAL_DELAY)
    assert (abs(phase) < 90.0) == expected


@pytest.mark.parametrize("delay, expected", CASES)
def test_streaming_stability_window_matches_phase_error(delay, expected):
    assert _streaming_converges(delay) is expected


@pytest.mark.parametrize("delay, expected", CASES)
def test_streaming_agrees_with_batch(delay, expected):
    assert _streaming_converges(delay) == _batch_converges(delay)
