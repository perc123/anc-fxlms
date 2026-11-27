"""Open-loop parity between the Python streaming reference and the C++ port.

The inputs and expected outputs are frozen golden vectors in tests/golden/
(see anc/golden.py). Each case was recorded by simulating the closed loop
once in float64; both implementations are fed the same recorded (x, e), so
float32 rounding differences cannot compound through the loop.

Tolerances are stored with each case. They come from the float32 Python
reference: its drift from float64 is what float32 arithmetic costs on this
problem, and the C++ port should not be much worse.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from anc.golden import CASES, load
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture
from anc.streaming import StreamingFxLMS, run_streaming_closed_loop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


def test_streaming_closed_loop_attenuates_noise():
    from anc.metrics import attenuation_db

    reference = generate_tone_mixture(
        2000, 4.0, [{"frequency_hz": 100.0, "amplitude": 1.0}],
    )
    plant = ANCPlant(
        primary_taps=np.array([0.8, 0.1]),
        secondary_taps=np.array([0.7, 0.2]),
        secondary_delay_samples=2,
    )
    controller = StreamingFxLMS(16, 0.005, np.array([0.7, 0.2]), 2)

    _, e, _ = run_streaming_closed_loop(reference, plant, controller)

    assert attenuation_db(plant.primary_output(reference), e, 4000) > 20.0


def test_streaming_requires_secondary_delay():
    plant = ANCPlant(np.array([1.0]), np.array([1.0]), 0)
    controller = StreamingFxLMS(4, 0.01, np.array([1.0]))

    with pytest.raises(ValueError):
        run_streaming_closed_loop(np.ones(10), plant, controller)


def test_streaming_update_follows_filtered_push():
    # The update pairs e[n] with x'[n], so it sees the sample just pushed and
    # y[n] already uses the updated weights, including on the first call.
    controller = StreamingFxLMS(2, 0.1, np.array([1.0]))
    y = controller.process(1.0, 0.5)
    np.testing.assert_allclose(controller.weights, [-0.05, 0.0])
    assert y == pytest.approx(-0.05 * 1.0)

    # Second call: x' history is [2, 1], so w -= 0.1 * 1.0 * [2, 1].
    y = controller.process(2.0, 1.0)
    np.testing.assert_allclose(controller.weights, [-0.25, -0.10])
    assert y == pytest.approx(-0.25 * 2.0 - 0.10 * 1.0)


@pytest.mark.parametrize("name", [case.name for case in CASES])
def test_cpp_matches_golden_vectors(cli, name, tmp_path):
    golden = load(name)
    params = golden.params

    y_path = tmp_path / "y.csv"
    w_path = tmp_path / "w.csv"

    # The CLI skips the "x,e" header itself, so the golden file is used as is.
    result = subprocess.run(
        [str(cli),
         "--input", str(GOLDEN_DIR / name / "input.csv"),
         "--output-y", str(y_path),
         "--output-weights", str(w_path),
         "--length", params["filter_length"],
         "--mu", params["step_size"],
         "--shat", params["s_hat_taps"],
         "--delay", params["s_hat_delay"]],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("fault=none ")
    y_cpp = np.loadtxt(y_path, ndmin=1)
    w_cpp = np.loadtxt(w_path, ndmin=1)

    assert y_cpp.shape == golden.y64.shape
    assert w_cpp.shape == golden.w64.shape

    assert np.max(np.abs(y_cpp - golden.y64)) <= golden.y_tolerance
    assert np.max(np.abs(w_cpp - golden.w64)) <= golden.w_tolerance

    # float32 vs float32 should agree more tightly than either does with
    # float64; only summation order differs. Stored tolerance = 3 x drift.
    y_drift = golden.y_tolerance / 3.0
    w_drift = golden.w_tolerance / 3.0
    assert np.max(np.abs(y_cpp - golden.y32)) <= max(y_drift, 1e-7)
    assert np.max(np.abs(w_cpp - golden.w32)) <= max(w_drift, 1e-7)