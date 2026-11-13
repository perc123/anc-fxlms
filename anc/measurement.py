"""Measurement-noise model for the error microphone.

The noise represents sensor self-noise, preamp noise and ADC noise at the
error microphone. It is simulation support, not part of the controller,
and it is not ported to the C++ FxLMS core.

Signal convention (see docs/algorithm_specification.md):
    r[n]   = d[n] + y_s[n]      acoustic residual (simulation-only quantity)
    e_m[n] = r[n] + v[n]        measured error, seen by the controller
"""
from __future__ import annotations

import numpy as np

from anc.metrics import rms

# Fixed stream ID: keeps measurement noise independent of signal noise
# generated from the same seed. Never change once results are published.
_NOISE_STREAM_ID = 7_001


def noise_std_for_snr(disturbance: np.ndarray, snr_db: float) -> float:
    """Noise standard deviation giving SNR_dB = 20*log10(RMS(d) / RMS(v))."""
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be a finite number.")
    d_rms = rms(disturbance)
    if d_rms == 0.0:
        raise ValueError("Cannot set an SNR relative to an all-zero disturbance.")
    return d_rms * 10.0 ** (-snr_db / 20.0)


def generate_measurement_noise(
    num_samples: int, std: float, seed: int | None
) -> np.ndarray:
    """White Gaussian measurement noise on an independent random stream."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if std < 0.0:
        raise ValueError("std must be non-negative.")
    if std == 0.0:
        return np.zeros(num_samples, dtype=np.float64)
    entropy = None if seed is None else [seed, _NOISE_STREAM_ID]
    rng = np.random.default_rng(entropy)
    return rng.normal(0.0, std, num_samples)


def measurement_floor_db(disturbance: np.ndarray, noise: np.ndarray) -> float:
    """Highest measurable attenuation: 20*log10(RMS(d + v) / RMS(v)).

    Returns +inf when there is no noise (the noise-free simulation has no floor).
    """
    noise_rms = rms(noise)
    if noise_rms == 0.0:
        return float("inf")
    measured_uncontrolled = np.asarray(disturbance) + np.asarray(noise)
    return 20.0 * np.log10(rms(measured_uncontrolled) / noise_rms)


def is_floor_limited(
    measured_attenuation_db: float, floor_db: float, margin_db: float = 3.0
) -> bool:
    """True when the result is within margin_db of the floor.

    With the default 3 dB margin, True means the acoustic residual power is at
    or below the noise power, so the figure describes the sensor rather than
    the controller (see the 3 dB rule).
    """
    if margin_db < 0.0:
        raise ValueError("margin_db must be non-negative.")
    if not np.isfinite(floor_db):
        return False
    return bool(measured_attenuation_db >= floor_db - margin_db)

def make_measurement_noise(
    disturbance: np.ndarray, snr_db: float | None, seed: int | None
) -> np.ndarray | None:
    """Measurement noise for one run, or None when snr_db is None (noise-free)."""
    if snr_db is None:
        return None
    disturbance = np.asarray(disturbance, dtype=np.float64)
    std = noise_std_for_snr(disturbance, snr_db)
    return generate_measurement_noise(disturbance.size, std, seed)