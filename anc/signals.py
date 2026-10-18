from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64] 


def generate_tone_mixture(
        sample_rate_hz: int,
        duration_s: float,
        tones: Sequence[dict[str, float]],
        white_noise_std: float = 0.0,
        seed: int = 42,
) -> FloatArray:
    """Generate a reproducable mixture of sinusoids and white noise."""
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")

    if duration_s <= 0:
        raise ValueError("Duration must be positive.")

    if white_noise_std < 0:
        raise ValueError("White noise standard deviation must be non-negative.")

    n_samples = int(round(sample_rate_hz * duration_s))

    if n_samples < 1:
        raise ValueError("The signal must contain at least one sample.")

    rng = np.random.default_rng(seed)
    n = np.arrange(n_samples, dtype=np.float64)
    t = n / sample_rate_hz


    x = np.zeros(n_samples, dtype=np.float64)

    for tone in tones:
        frequency_hz = float(tone["frequency_hz"])
        amplitude = float(tone["amplitude"])

        if not 0 <= frequency_hz < sample_rate_hz / 2:
            raise ValueError(
                f"Frequency {frequency_hz} Hz is out of range. "
                f"Must be in [0, {sample_rate_hz / 2}) Hz (Nyquist)."
            )
        x  += amplitude * np.sin(
            2.0 * np.pi * frequency_hz * t
        )

    if white_noise_std:
        x += white_noise_std * rng.standard_normal(n_samples)

    return x
