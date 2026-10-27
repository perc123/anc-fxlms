from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anc.controllers import FxLMSController
from anc.plant import ANCPlant, apply_fir

FloatArray = NDArray[np.float64]


@dataclass
class FxLMSTrace:
    reference: FloatArray
    primary_noise: FloatArray
    control: FloatArray
    anti_noise: FloatArray
    residual: FloatArray
    filtered_reference: FloatArray
    weights:FloatArray


def run_fxlms(
        reference: FloatArray,
        plant: ANCPlant,
        controller: FxLMSController,
) -> FxLMSTrace:
    """Run sample-by-sample FxLMS active noise cancellation."""

    reference = np.asarray(reference, dtype=np.float64)

    if reference.ndim != 1:
        raise ValueError("Reference must be one-dimentional.")

    if reference.size == 0:
        raise ValueError("Reference cannot be empty.")

    if not np.isfinite(reference).all():
        raise ValueError("Reference must contain only finite values.")

    primary_noise = np.asarray(
        plant.primary_output(reference),
        dtype=np.float64,
    )

    actual_secondary_taps = np.asarray(
        plant.secondary_taps,
        dtype=np.float64,
    )

    if actual_secondary_taps.ndim != 1:
        raise ValueError("Actual secondary-path taps must be one-dimentional.")

    if actual_secondary_taps.size == 0:
        raise ValueError("Actual secondary-path taps cannot be empty.")

    if not np.isfinite(actual_secondary_taps).all():
        raise ValueError("Actual secondary-path taps must be finite.")

    if plant.secondary_delay_samples < 0:
        raise ValueError("Actual secondary-path delay must be non-negative.")

    if primary_noise.shape != reference.shape:
        raise ValueError("Primary noise must match the reference shape")

    # Filter the reference using the controller's estimated secondary path.
    # The resulting signal is used only for adaptation.
    filtered_reference = apply_fir(
        reference,
        controller.estimated_secondary_taps,
        delay_samples=controller.estimated_secondary_delay_samples,
    )

    n_samples = reference.size
    filter_length = controller.filter_length

    control = np.zeros(n_samples, dtype=np.float64)
    anti_noise = np.zeros(n_samples, dtype=np.float64)
    residual = np.zeros(n_samples, dtype=np.float64)

    weights = np.zeros(n_samples, dtype=np.float64)
    anti_noise = np.zeros(n_samples, dtype=np.float64)
    residual = np.zeros(n_samples, dtype=np.float64)

    weights = np.zeros(
        (n_samples, filter_length),
        dtype=np.float64,
    )

    reference_history = np.zeros(filter_length, dtype=np.float64)
    filtered_history = np.zeros(filter_length, dtype=np.float64)

    controller.reset()

    for n in range(n_samples):
        # 1. Build the controller input vector:
        #       [x[n], x[n-1], ..., x[n-L+1]].
        reference_history[1:] = reference_history[:1]
        reference_history[0] = reference[n]

        # 2. Gener 