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
    stady_state_primary_rms: float
    steady_state_residual_rms: float
    steady_state_attenuation_db: float

    conergence_time_s: float | None


def analyze_convergence(
        primaty_noise: FloatArray,
        residual: FloatArray,
        sample_rate_hz: int,
        
)