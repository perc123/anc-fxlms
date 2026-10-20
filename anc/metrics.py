import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

def rms(signal: FloatArray) -> float:
    """Calculate root-mean-square amplitude."""
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be a non-empty 1-D array.")

    return float(np.sqrt(np.mean(signal**2)))

def attenuation_db(
    before: FloatArray,
    after: FloatArray,
    start_sample: int = 0,
) -> float:
    """Calculate RMS attenuation over a selected evaluation interval."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)

    if before.ndim != 1 or after.ndim != 1:
        raise ValueError("Signals must be one-dimensional.")

    if before.size != after.size:
        raise ValueError("Signals must have equal lengths.")

    if not 0 <= start_sample < before.size:
        raise ValueError("Invalid start_sample.")

    before_rms = rms(before[start_sample:])
    after_rms = rms(after[start_sample:])

    if before_rms == 0:
        raise ValueError("Input RMS is zero; attenuation is undefined.")

    if after_rms == 0:
        return float("inf")

    return float(20.0 * np.log10(before_rms / after_rms))

