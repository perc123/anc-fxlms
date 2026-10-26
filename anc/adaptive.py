from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anc.controllers import LMSController

FloatArray = NDArray[np.float64]

@dataclass
class LMSTrace:
    reference: FloatArray
    desired: FloatArray
    control: FloatArray
    error: FloatArray
    weights: FloatArray

def run_lms(
        reference: FloatArray,
        desired: FloatArray,
        controller: LMSController,
) -> LMSTrace:
    """Run ordinary LMS on a direct-error adaptive-filter problem.
    
    This experiment intentionally does not model ANC secondary path.
    It is a reference experiment for learning and validating LMS before
    implementing the filtered-x LMS algorithm.
    """
    reference =  np.asarray(reference, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)

    if reference.ndim != 1 or desired.ndim != 1:
        raise ValueError("Signals must be one-dimentional.")

    if reference.size == 0:
        raise ValueError("Signals cannot be empty.")

    if reference.shape != desired.shape:
        raise ValueError("Signals must have the same length.")

    if not np.isfinite(desired).all():
        raise ValueError("Disired signal must be finite.")

    controller.reset()

    n_samples = reference.size
    length = controller.filter_length

    control = np.zeros(n_samples, dtype=np.float64)
    error = np.zeros(n_samples, dtype=np.float64)
    weight_history = np.zeros(
        (n_samples, length),
        dtype=np.float64,
    )

    # Newest sample first: [x[n]], [x[n-1]], ..., [x[n-L+1]].
    reference_history = np.zeros(length, dtype=np.float64)

    for n in range(n_samples):
        reference_history[1:] = reference_history[:-1]
        reference_history[0] = reference_history[n]

        control[n] = controller.predict(reference_history)

        # Direct-error model: e[n] = desired[n] - control[n].
        error[n] = desired[n] - control[n]

        controller.update(reference_history, -error[n])
        weight_history[n] = controller.weights

    return LMSTrace(
        reference=reference,
        desired=desired,
        control=control,
        error=error,
        weights=weight_history,
    )