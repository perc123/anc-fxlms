"""Golden test vectors for the streaming FxLMS controller.

A golden case is a frozen, committed set of inputs and expected outputs.
Both the Python parity test and the C++ test runner read the same files, so
the C++ port can be checked without regenerating anything.

Each case lives in tests/golden/<name>/:

    params.txt         key=value lines: controller parameters and tolerances
    input.csv          x,e            one row per sample
    expected_y.csv     y64,y32        Python float64 and float32 outputs
    expected_w.csv     w64,w32        final weights, one row per tap

Regenerate with:

    python -m anc.golden --write

Regeneration must be a deliberate act: the files are the contract that the
C++ port is tested against.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anc.measurement import make_measurement_noise
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture
from anc.streaming import StreamingFxLMS, run_streaming_closed_loop

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden"

# Bump when the file format changes, not when the numbers change.
FORMAT_VERSION = 1

# Tolerance policy (algorithm specification, section 7).
DRIFT_MARGIN = 3.0
ABS_FLOOR = 1e-7


@dataclass(frozen=True)
class GoldenCase:
    name: str
    description: str
    filter_length: int
    step_size: float
    s_hat_taps: tuple[float, ...]
    s_hat_delay: int
    seed: int
    tones: tuple[tuple[float, float], ...]
    white_noise_std: float
    duration_s: float
    sample_rate_hz: int = 2000
    primary_taps: tuple[float, ...] = (0.80, 0.0, 0.15, 0.0, -0.08, 0.0, 0.04)
    secondary_taps: tuple[float, ...] = (0.70, 0.0, 0.20, 0.0, -0.05)
    secondary_delay: int = 2
    snr_db: float | None = None


CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        name="tone_mixture",
        description="Three tones plus light broadband noise; mismatched S_hat.",
        filter_length=32,
        step_size=0.005,
        s_hat_taps=(0.66, 0.0, 0.23, 0.0, -0.03),
        s_hat_delay=2,
        seed=42,
        tones=((80.0, 0.7), (120.0, 0.4), (160.0, 0.2)),
        white_noise_std=0.03,
        duration_s=4.0,
    ),
    GoldenCase(
        name="tone_mixture_snr40",
        description="As tone_mixture, with 40 dB error-microphone noise in e[n].",
        filter_length=32,
        step_size=0.005,
        s_hat_taps=(0.66, 0.0, 0.23, 0.0, -0.03),
        s_hat_delay=2,
        seed=42,
        tones=((80.0, 0.7), (120.0, 0.4), (160.0, 0.2)),
        white_noise_std=0.03,
        duration_s=4.0,
        snr_db=40.0,
    ),
    GoldenCase(
        name="short_debug",
        description="Tiny case for reading by hand: L=4, 16 samples, one tone.",
        filter_length=4,
        step_size=0.05,
        s_hat_taps=(0.7, 0.2),
        s_hat_delay=1,
        seed=1,
        tones=((250.0, 1.0),),
        white_noise_std=0.0,
        duration_s=0.008,
        primary_taps=(0.8, 0.1),
        secondary_taps=(0.7, 0.2),
        secondary_delay=1,
    ),
)


def record_closed_loop(case: GoldenCase) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the closed loop once in float64 and return (x, e)."""
    reference = generate_tone_mixture(
        sample_rate_hz=case.sample_rate_hz,
        duration_s=case.duration_s,
        tones=[{"frequency_hz": f, "amplitude": a} for f, a in case.tones],
        white_noise_std=case.white_noise_std,
        seed=case.seed,
    )
    plant = ANCPlant(
        primary_taps=np.array(case.primary_taps),
        secondary_taps=np.array(case.secondary_taps),
        secondary_delay_samples=case.secondary_delay,
    )
    noise = make_measurement_noise(
        plant.primary_output(reference), case.snr_db, seed=case.seed
    )
    controller = StreamingFxLMS(
        case.filter_length, case.step_size,
        np.array(case.s_hat_taps), case.s_hat_delay,
    )
    x, e, _ = run_streaming_closed_loop(
        reference, plant, controller, measurement_noise=noise
    )
    return x, e


def run_open_loop(
    case: GoldenCase, x: np.ndarray, e: np.ndarray, dtype: type[np.floating]
) -> tuple[np.ndarray, np.ndarray]:
    """Feed a recorded (x, e) sequence to a fresh controller of the given dtype."""
    controller = StreamingFxLMS(
        case.filter_length, case.step_size,
        np.array(case.s_hat_taps), case.s_hat_delay, dtype=dtype,
    )
    y = np.array(
        [controller.process(dtype(xn), dtype(en)) for xn, en in zip(x, e)],
        dtype=np.float64,
    )
    return y, controller.weights.astype(np.float64)


@dataclass(frozen=True)
class GoldenData:
    params: dict[str, str]
    x: np.ndarray
    e: np.ndarray
    y64: np.ndarray
    y32: np.ndarray
    w64: np.ndarray
    w32: np.ndarray

    @property
    def y_tolerance(self) -> float:
        return float(self.params["y_tolerance"])

    @property
    def w_tolerance(self) -> float:
        return float(self.params["w_tolerance"])


def build(case: GoldenCase) -> GoldenData:
    """Compute a golden case in memory."""
    x, e = record_closed_loop(case)
    y64, w64 = run_open_loop(case, x, e, np.float64)
    y32, w32 = run_open_loop(case, x, e, np.float32)

    y_tol = max(DRIFT_MARGIN * float(np.max(np.abs(y32 - y64))), ABS_FLOOR)
    w_tol = max(DRIFT_MARGIN * float(np.max(np.abs(w32 - w64))), ABS_FLOOR)

    params = {
        "format_version": str(FORMAT_VERSION),
        "name": case.name,
        "description": case.description,
        "n_samples": str(x.size),
        "filter_length": str(case.filter_length),
        "step_size": repr(case.step_size),
        "s_hat_taps": ",".join(repr(t) for t in case.s_hat_taps),
        "s_hat_delay": str(case.s_hat_delay),
        "snr_db": "none" if case.snr_db is None else repr(case.snr_db),
        "seed": str(case.seed),
        "y_tolerance": repr(y_tol),
        "w_tolerance": repr(w_tol),
    }
    return GoldenData(params, x, e, y64, y32, w64, w32)


def write(data: GoldenData, root: Path = GOLDEN_DIR) -> Path:
    case_dir = root / data.params["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    with (case_dir / "params.txt").open("w", encoding="utf-8") as file:
        for key, value in data.params.items():
            file.write(f"{key}={value}\n")

    fmt = "%.17g"
    np.savetxt(case_dir / "input.csv", np.column_stack([data.x, data.e]),
               delimiter=",", fmt=fmt, header="x,e", comments="")
    np.savetxt(case_dir / "expected_y.csv", np.column_stack([data.y64, data.y32]),
               delimiter=",", fmt=fmt, header="y64,y32", comments="")
    np.savetxt(case_dir / "expected_w.csv", np.column_stack([data.w64, data.w32]),
               delimiter=",", fmt=fmt, header="w64,w32", comments="")
    return case_dir


def load(name: str, root: Path = GOLDEN_DIR) -> GoldenData:
    case_dir = root / name
    params = {}
    for line in (case_dir / "params.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            params[key] = value

    io = np.loadtxt(case_dir / "input.csv", delimiter=",", skiprows=1, ndmin=2)
    y = np.loadtxt(case_dir / "expected_y.csv", delimiter=",", skiprows=1, ndmin=2)
    w = np.loadtxt(case_dir / "expected_w.csv", delimiter=",", skiprows=1, ndmin=2)
    return GoldenData(params, io[:, 0], io[:, 1], y[:, 0], y[:, 1], w[:, 0], w[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the golden FxLMS vectors.")
    parser.add_argument("--write", action="store_true",
                        help="Write the files. Without it, only print a summary.")
    args = parser.parse_args()

    for case in CASES:
        data = build(case)
        print(f"{case.name}: {data.x.size} samples, "
              f"y_tol={data.y_tolerance:.3g}, w_tol={data.w_tolerance:.3g}")
        if args.write:
            print(f"  written to {write(data)}")

    if not args.write:
        print("\nDry run. Use --write to update tests/golden/.")


if __name__ == "__main__":
    main()