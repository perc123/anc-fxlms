"""Open-loop parity between the Python streaming reference and the C++ port.

Python simulates the closed loop once in float64 and records x[n] and e[n].
Both implementations are then fed those same recorded sequences, so e does not
depend on either implementation's own output and float32 rounding differences
cannot compound through the loop.

The tolerance comes from the float32 Python reference: its drift from float64
is what float32 arithmetic costs on this problem, and the C++ port should not
be much worse.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture
from anc.streaming import StreamingFxLMS, run_streaming_closed_loop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "tests" / "data"

FILTER_LENGTH = 32
STEP_SIZE = 0.005
S_HAT_TAPS = [0.66, 0.0, 0.23, 0.0, -0.03]
S_HAT_DELAY = 2

# The C++ error may exceed the float32-vs-float64 drift by this factor.
DRIFT_MARGIN = 3.0
# Absolute floor so a lucky near-zero drift cannot make the check brittle.
ABS_FLOOR = 1e-7


@pytest.fixture(scope="module")
def cli(tmp_path_factory) -> Path:
    if shutil.which("cmake") is None:
        pytest.skip("cmake is not installed")

    build_dir = tmp_path_factory.mktemp("cpp_build")
    try:
        subprocess.run(
            ["cmake", "-S", str(PROJECT_ROOT / "cpp"), "-B", str(build_dir),
             "-DCMAKE_BUILD_TYPE=Release"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "anc_cli"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"C++ build failed:\n{error.stdout}\n{error.stderr}")

    return build_dir / "anc_cli"


@pytest.fixture(scope="module")
def recorded_io() -> tuple[np.ndarray, np.ndarray]:
    reference = generate_tone_mixture(
        sample_rate_hz=2000,
        duration_s=4.0,
        tones=[
            {"frequency_hz": 80.0, "amplitude": 0.7},
            {"frequency_hz": 120.0, "amplitude": 0.4},
            {"frequency_hz": 160.0, "amplitude": 0.2},
        ],
        white_noise_std=0.03,
        seed=42,
    )
    plant = ANCPlant(
        primary_taps=np.array([0.80, 0.0, 0.15, 0.0, -0.08, 0.0, 0.04]),
        secondary_taps=np.array([0.70, 0.0, 0.20, 0.0, -0.05]),
        secondary_delay_samples=2,
    )
    controller = StreamingFxLMS(
        FILTER_LENGTH, STEP_SIZE, np.array(S_HAT_TAPS), S_HAT_DELAY,
    )
    x, e, _ = run_streaming_closed_loop(reference, plant, controller)
    return x, e


def _run_python(x, e, dtype):
    controller = StreamingFxLMS(
        FILTER_LENGTH, STEP_SIZE, np.array(S_HAT_TAPS), S_HAT_DELAY,
        dtype=dtype,
    )
    y = np.array([
        controller.process(dtype(xn), dtype(en)) for xn, en in zip(x, e)
    ])
    return y, controller.weights.astype(np.float64)


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


def test_streaming_update_precedes_output():
    # First call: weights are zero, so y[0] = 0 and nothing is updated yet.
    controller = StreamingFxLMS(2, 0.1, np.array([1.0]))
    assert controller.process(1.0, 0.5) == 0.0
    np.testing.assert_array_equal(controller.weights, [0.0, 0.0])

    # Second call updates with x'[0] = 1 (and zeros behind it) and e = 1.0
    # before predicting, so y[1] already uses the new weights.
    y = controller.process(2.0, 1.0)
    np.testing.assert_allclose(controller.weights, [-0.1, 0.0])
    assert y == pytest.approx(-0.1 * 2.0)


def test_cpp_matches_python_open_loop(cli, recorded_io):
    x, e = recorded_io

    DATA_DIR.mkdir(exist_ok=True)
    io_path = DATA_DIR / "parity_io.csv"
    y_path = DATA_DIR / "parity_cpp_y.csv"
    w_path = DATA_DIR / "parity_cpp_weights.csv"
    np.savetxt(io_path, np.column_stack([x, e]), delimiter=",", fmt="%.17g")

    subprocess.run(
        [str(cli),
         "--input", str(io_path),
         "--output-y", str(y_path),
         "--output-weights", str(w_path),
         "--length", str(FILTER_LENGTH),
         "--mu", str(STEP_SIZE),
         "--shat", ",".join(str(tap) for tap in S_HAT_TAPS),
         "--delay", str(S_HAT_DELAY)],
        check=True, capture_output=True, text=True,
    )
    y_cpp = np.loadtxt(y_path)
    w_cpp = np.loadtxt(w_path)

    y64, w64 = _run_python(x, e, np.float64)
    y32, w32 = _run_python(x, e, np.float32)

    assert y_cpp.shape == y64.shape
    assert w_cpp.shape == w64.shape

    y_drift = np.max(np.abs(y32 - y64))
    w_drift = np.max(np.abs(w32 - w64))
    y_tol = max(DRIFT_MARGIN * y_drift, ABS_FLOOR)
    w_tol = max(DRIFT_MARGIN * w_drift, ABS_FLOOR)

    assert np.max(np.abs(y_cpp - y64)) <= y_tol
    assert np.max(np.abs(w_cpp - w64)) <= w_tol

    # float32 vs float32 should agree more tightly than either does with
    # float64; only summation order differs.
    assert np.max(np.abs(y_cpp - y32)) <= max(y_drift, ABS_FLOOR)
    assert np.max(np.abs(w_cpp - w32)) <= max(w_drift, ABS_FLOOR)
