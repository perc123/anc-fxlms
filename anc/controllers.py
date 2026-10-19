from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from anc.plant import ANCPlant, apply_fir

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class FixedFIRController:
    """Fixed FIR controller with offline least-sqares design."""

    taps: FloatArray

    def __post_init__(self) -> None:
        taps = np.asarray(self.taps, dtype=np.float64)

        if taps.ndim != 1 or taps.size ==0:
            raise ValueError("Controller taps must be a nonempty vector.")

        object.__setattr__(self, "taps", taps.copy())

    def process(self, reference: FloatArray) -> FloatArray:
        """Generate the loudspeaker control signal from the reference microphone signal."""
        return apply_fir(reference, self.taps)


def _fir_regression_matrix(
        signal: FloatArray,
        filter_length: int,
) -> FloatArray:
    """Construct the casual FIR regression matrix for a given signal."""
    if filter_length < 1:
        raise ValueError("filter_length must be positive.")

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    n_samples = signal.size
    matrix = np.zeros(
        (n_samples, filter_length),
        dtype=np.float64,
    )

    for k in range(min(filter_length, n_samples)):
        matrix[k:, k] = signal[: n_samples - k]

    return matrix


def design_fixed_fir(
        reference: FloatArray,
        desired_noise: FloatArray,
        plant: ANCPlant,
        filter_length: int,
        regularization: float = 1e-6,
    ) -> FixedFIRController:
    """Design a fixed controller by regularized least squares.
    
    The optimized residual is:
        desired_noise + secondary_path(controller(reference))
        
    Coefficients are estimated offline and remain fixed during subsequent simulations."""

    reference = np.asarray(reference, dtype=np.float64)
    desired_noise = np.asarray(desired_noise, dtype=np.float64)

    if reference.ndim != 1 or desired_noise.ndim != 1:
        raise ValueError("Signals must be one-dimensional.")

    if reference.size != desired_noise.size:
        raise ValueError("Signals must have the same length.")

    if reference.size == 0:
        raise ValueError("Input signals cannot be empty.")

    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    if filter_length < 1:
        raise ValueError("filter_length must be positive.")

    # A unit impulse through the secondary path gives its impulse response, including the explicit delay.

    impulse = np.zeros(filter_length, dtype=np.float64)
    impulse[0] = 1.0

    secondary_ir = plant.secondary_output(impulse)

    # The cascade S(z)W(z) is linear in the controller taps. 
    # Each column corresponds to the one controller tap after passing through the secondary path. 

    reference_matrix = _fir_regression_matrix(
        reference,
        filter_length,
    )
    effective_matrix = np.zeros_like(reference_matrix)

    for k in range(filter_length):
        effective_matrix[:, k] = plant.secondary_output(
            reference_matrix[:, k]
        )

    # Solve:
    # min_w ||d + A w||^2 + lambda * ||w||^2

    lhs = (
        effective_matrix.T @ effective_matrix
        + regularization * np.eye(filter_length)
    )    
    rhs = -(effective_matrix.T @ desired_noise)

    taps = cast(FloatArray, np.linalg.solve(lhs, rhs))

    return FixedFIRController(taps=taps)
