from collections.abc import Sequence

import numpy as np

from anc.types import FloatArray


def generate_tone_mixture(
        sample_rate_hz: int,
        duration_s: float,
        tones: Sequence[dict[str, float]],
        white_noise_std: float = 0.0,
        seed: int = 42,
) -> FloatArray:
    """Generate a reproducible mixture of sinusoids and white noise."""
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
    n = np.arange(n_samples, dtype=np.float64)
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


def generate_band_limited_noise(
        sample_rate_hz: int,
        duration_s: float,
        low_cutoff_hz: float,
        high_cutoff_hz: float,
        noise_std: float = 1.0,
        seed: int = 42,
        filter_length: int = 129,
) -> FloatArray:
    """Generate reproducible band-limited Gaussian noise.

    A symmetric windowed-sinc FIR bandpass filter is applied
    to white Gaussian noise.

    Warm-up samples are discarded to reduce startup transients.

    The output is normalized to the requested standard deviation.
    """

    # Validate parameters.
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    if duration_s <= 0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be finite and positive.")

    if not (
        np.isfinite(low_cutoff_hz)
        and np.isfinite(high_cutoff_hz)
        and 0 <= low_cutoff_hz < high_cutoff_hz < sample_rate_hz / 2
    ):
        raise ValueError(
            "Cutoffs must satisfy "
            "0 <= low_cutoff_hz < high_cutoff_hz < Nyquist."
        )

    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative.")

    if filter_length < 3 or filter_length % 2 == 0:
        raise ValueError("filter_length must be odd and at least 3.")

    n_samples = int(round(sample_rate_hz * duration_s))

    if n_samples < 1:
        raise ValueError("Signaö must contain at least one sample.")

    if noise_std == 0:
        return np.zeros(n_samples, dtype=np.float64)

    rng = np.random.default_rng(seed)

    warmup_samples = filter_length - 1

    white_noise = rng.standard_normal(
        warmup_samples + n_samples
    )

    # Windowed-sinc FIR bandpass design.
    sample_indices = (
        np.arange(filter_length, dtype=np.float64)
        - (filter_length - 1) / 2
    )

    window = np.hamming(filter_length)

    def lowpass_taps(cutoff_hz: float) -> FloatArray:
        normalized_cutoff = cutoff_hz / sample_rate_hz

        return (
            2.0
            * normalized_cutoff
            * np.sinc(2.0 * normalized_cutoff * sample_indices)
            * window
        )

    filter_taps = (
        lowpass_taps(high_cutoff_hz)
        - lowpass_taps(low_cutoff_hz)
    )

    tap_norm = np.linalg.norm(filter_taps)

    if not np.isfinite(tap_norm) or tap_norm == 0:
        raise ValueError("Unable to design a valid bandpass filter.")

    # Normalize filter energy.
    filter_taps /= tap_norm

    # Apply the casual FIR filter
    filtered = np.convolve(
        white_noise,
        filter_taps,
        mode="full",
    )

    filtered = filtered[
        warmup_samples:warmup_samples + n_samples
    ]

    # Normalize the finite generated realization
    measured_std = float(np.std(filtered))

    if not np.isfinite(measured_std) or measured_std == 0:
        raise ValueError("Generated noise has invalid standard deviation.")

    filtered *= noise_std / measured_std

    return np.asarray(filtered, dtype=np.float64)