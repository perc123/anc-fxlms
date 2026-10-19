from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray =  NDArray[np.float64]

def apply_fir(
        signal: FloatArray,
        taps: FloatArray,
        delay_samples: int = 0,
    ) -> FloatArray:
        """Applay an FIR filter and an explicit integer-sample delay."""
        signal = np.asarray(signal, dtype=np.float64)
        taps = np.asarray(taps, dtype=np.float64)

        if signal.ndim != 1 or taps.ndim != 1:
            raise ValueError("Signal and taps must be one-dimensional.")

        if taps.size == 0:
            raise ValueError("The FIR filter must have at least one tap.")

        if delay_samples < 0:
            raise ValueError("delay_samples must be non-negative.")

        filtered = cast(FloatArray, np.convolve(signal, taps, mode="full")[: signal.size])

        if delay_samples == 0:
            return filtered

        delayed = np.zeros_like(filtered)

        if delay_samples < signal.size:
            delayed[delay_samples:] = filtered[:-delay_samples]

        return delayed

@dataclass(frozen=True)
class ANCPlant:
    """Linear time-invariant primary and secondary path model for an ANC system."""

    primary_taps: FloatArray
    secondary_taps: FloatArray
    secondary_delay_samples: int = 0

    def __post_init__(self) -> None:
        if np.asarray(self.primary_taps).size == 0:
            raise ValueError("primary_taps cannot be empty.")

        if np.asarray(self.secondary_taps).size == 0:
            raise ValueError("secondary_taps cannot be empty.")

        if self.secondary_delay_samples < 0:
            raise ValueError("secondary_delay_samples must be non-negative.")

    def primary_output(self, reference: FloatArray) -> FloatArray:
        """Return the unwanted noise at the error microphone given a reference signal."""
        return apply_fir(reference, self.primary_taps)

    def secondary_output(self, control: FloatArray) -> FloatArray:
        """Return the anti-noise at the error microphone given a control signal."""
        return apply_fir(control, self.secondary_taps, self.secondary_delay_samples)