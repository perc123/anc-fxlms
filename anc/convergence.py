from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConvergenceResult:
    """Performance and convergence measurements for an ANC run."""

    block_start_times_s: FloatArray
    block_primary_rms: FloatArray
    block_residual_rms: FloatArray

    steady_state_start_sample: int
    steady_state_primary_rms: float
    steady_state_residual_rms: float
    steady_state_attenuation_db: float

    convergence_time_s: float | None


def analyze_convergence(
        primary_noise: FloatArray,
        residual: FloatArray,
        sample_rate_hz: int,
        start_sample: int = 0,
        stop_sample: int | None = None,
        block_duration_s: float = 1.0,
        steady_state_fraction: float = 0.2,
        tolerance_db: float = 1.0,
        consecutive_blocks: int = 3,
) -> ConvergenceResult:
    """Calculate block RMS, steady-state attenuation and convergence time for an ANC run.
    The steady-state RMS is calculated over the final fraction of the specified evaluation interval.

    Convergence time is the start time of the first sequence of 'consecutive_blocks'
    blocks whose residual RMS values all lie within +/- tolerance_db of the steady-state residual RMS value.

    Times are measured relative to the beginning of the original signal,
    not relative to the start_sample parameter.

    The caller must supply only valid, processed samples. In particular, do not include NaN tail of a diverged FxLMS run.
    """

    primary_noise = np.asarray(
        primary_noise, dtype=np.float64
    )
    residual = np.asarray(residual, dtype=np.float64)

    # ------------------------------------------
    # Validate input parameters.
    # ------------------------------------------
    if primary_noise.ndim != 1 or residual.ndim != 1:
        raise ValueError("Signals must be one-dimensional.")

    if primary_noise.size == 0 or residual.size == 0:
        raise ValueError("Signals must not be empty.")

    if primary_noise.shape != residual.shape:
        raise ValueError("Signals must have the same length.")

    if not np.isfinite(primary_noise).all():
        raise ValueError("Primary noise signal contains non-finite values.")

    if not np.isfinite(residual).all():
        raise ValueError("Residual signal contains non-finite values.")
    
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")

    if start_sample < 0:
        raise ValueError("Start sample must be non-negative.")

    if stop_sample is None:
        stop_sample = residual.size

    if stop_sample > residual.size:
        raise ValueError("Stop sample must be less than or equal to the length of the signals.")

    if stop_sample <= start_sample:
        raise ValueError("Stop sample must be greater than start sample.")

    if block_duration_s <= 0:
        raise ValueError("Block duration must be positive.")

    if not 0 < steady_state_fraction <= 1:
        raise ValueError("Steady-state fraction must be in the range (0, 1].")

    if tolerance_db < 0 or not np.isfinite(tolerance_db):
        raise ValueError("Tolerance must be non-negative and finite.")

    if consecutive_blocks < 1:
        raise ValueError("Consecutive blocks must be positive.")

    block_size = int(round(
        block_duration_s * sample_rate_hz  
    ))

    if block_size < 1:
        raise ValueError("block_duration_s is too small for the given sample rate.")

    # ------------------------------------------
    # Compute non-overlapping block RMS values for primary and residual signals.
    # ------------------------------------------
    block_starts = np.arange(
        start_sample,
        stop_sample,
        block_size,
        dtype=np.int64,
    )

    block_times: list[float] = []
    primary_values: list[float] = []
    residual_values: list[float] = []

    for block_start in block_starts:
        block_stop = min(
            int(block_start) + block_size,
            stop_sample
        )

        primary_block = primary_noise[block_start:block_stop]
        residual_block = residual[block_start:block_stop]

        primary_block_rms = float(
            np.sqrt(np.mean(primary_block**2))
        )
        residual_block_rms = float(
            np.sqrt(np.mean(residual_block**2))
        )

        block_times.append(
            float(block_start / sample_rate_hz)
        )
        primary_values.append(primary_block_rms)
        residual_values.append(residual_block_rms)

    block_start_times_s = np.asarray(block_times, dtype=np.float64)

    block_primary_rms = np.asarray(primary_values, dtype=np.float64)
    block_residual_rms = np.asarray(residual_values, dtype=np.float64)

    # ------------------------------------------
    # Estimate steady-state RMS from the final evaluation window.
    # ------------------------------------------
    evaluation_length = stop_sample - start_sample

    steady_length = max(
        1,
        int(np.ceil(
            evaluation_length * steady_state_fraction
        )),
    )

    steady_state_start_sample = stop_sample - steady_length

    steady_primary = primary_noise[
        steady_state_start_sample:stop_sample
    ]
    steady_residual = residual[
        steady_state_start_sample:stop_sample
    ]

    steady_state_primary_rms = float(
        np.sqrt(np.mean(steady_primary**2))
    )
    steady_state_residual_rms = float(
        np.sqrt(np.mean(steady_residual**2))
    )

    # ------------------------------------------
    # Calculate steady-state attenuation in dB.
    # ------------------------------------------
    if steady_state_residual_rms == 0.0:
        if steady_state_primary_rms > 0.0:
            steady_state_attenuation_db = float("inf")
        else:
            steady_state_attenuation_db = float("nan")

    elif steady_state_primary_rms == 0.0:
        steady_state_attenuation_db = float("-inf")

    else:
        steady_state_attenuation_db = float(
            20.0 * np.log10(steady_state_primary_rms / steady_state_residual_rms)
        )

    # ---------------------------------------------------------
    # Find first qualifying convergence sequence.
    # ---------------------------------------------------------
    convergence_time_s: float | None = None

    ratio = 10.0 ** (tolerance_db / 20.0)

    lower_limit = steady_state_residual_rms / ratio
    upper_limit = steady_state_residual_rms * ratio

    within_band = (
        (block_residual_rms >= lower_limit)
        & (block_residual_rms <= upper_limit)
    )

    last_start = (
        len(within_band) - consecutive_blocks + 1
    )

    for index in range(max(0, last_start)):
        candidate = within_band[
            index:index + consecutive_blocks
        ]

        if np.all(candidate):
            convergence_time_s = float(
                block_start_times_s[index]
            )
            break

    return ConvergenceResult(
        block_start_times_s=block_start_times_s,
        block_primary_rms=block_primary_rms,
        block_residual_rms=block_residual_rms,
        steady_state_start_sample=steady_state_start_sample,
        steady_state_primary_rms=steady_state_primary_rms,
        steady_state_residual_rms=steady_state_residual_rms,
        steady_state_attenuation_db=steady_state_attenuation_db,
        convergence_time_s=convergence_time_s,
    )