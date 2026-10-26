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

        if taps.ndim != 1 or taps.size == 0:
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
    """Design a fixed FIR controller using regularized least squares.

    Minimize:
        ||desired_noise + S(W(reference))||^2
        + regularization * ||taps||^2

    where:
        reference       is the reference signal x[n],
        desired_noise   is the primary noise d[n],
        W               is the FIR controller,
        S               is the secondary path.

    The controller coefficients are estimated offline and remain fixed
    during subsequent simulations.
    """

    # Convert signals to float64 NumPy arrays.
    reference = np.asarray(reference, dtype=np.float64)
    desired_noise = np.asarray(desired_noise, dtype=np.float64)

    # Validate inputs.
    if reference.ndim != 1 or desired_noise.ndim != 1:
        raise ValueError("Signals must be one-dimensional.")

    if reference.size != desired_noise.size:
        raise ValueError("Signals must have the same length.")

    if reference.size == 0:
        raise ValueError("Input signals cannot be empty.")

    if not np.isfinite(reference).all():
        raise ValueError("Reference contains NaN or infinite values.")

    if not np.isfinite(desired_noise).all():
        raise ValueError("Desired noise contains NaN or infinite values.")

    if filter_length < 1:
        raise ValueError("filter_length must be positive.")

    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    # Construct the FIR regression matrix.
    # Each column represents the reference signal delayed by one
    # additional sample.
    reference_matrix = _fir_regression_matrix(
        reference,
        filter_length,
    )

    # Construct the effective matrix for the cascade S(z)W(z).
    # Each column passes one controller basis signal through the
    # secondary path.
    effective_matrix = np.zeros_like(
        reference_matrix,
        dtype=np.float64,
    )

    for k in range(filter_length):
        effective_matrix[:, k] = plant.secondary_output(
            reference_matrix[:, k]
        )

    # Validate the constructed matrix before solving.
    if not np.isfinite(effective_matrix).all():
        raise ValueError(
            "Effective matrix contains NaN or infinite values."
        )

    if effective_matrix.shape != (
        reference.size,
        filter_length,
    ):
        raise ValueError(
            f"Unexpected effective matrix shape: "
            f"{effective_matrix.shape}"
        )

    # Solve:
    # min_w ||A w + d||^2 + lambda * ||w||^2
    #
    # Use an augmented least-squares system rather than normal equations.
    # This avoids explicitly forming A.T @ A, which can worsen
    # numerical conditioning.

    if regularization > 0:
        augmented_matrix = np.vstack(
            (
                effective_matrix,
                np.sqrt(regularization)
                * np.eye(filter_length, dtype=np.float64),
            )
        )

        augmented_target = np.concatenate(
            (
                -desired_noise,
                np.zeros(filter_length, dtype=np.float64),
            )
        )
    else:
        augmented_matrix = effective_matrix
        augmented_target = -desired_noise

    taps, *_ = np.linalg.lstsq(
        augmented_matrix,
        augmented_target,
        rcond=None,
    )

    taps = np.asarray(taps, dtype=np.float64)

    if not np.isfinite(taps).all():
        raise ValueError(
            "The optimized controller contains NaN or infinite values."
        )

    # Verify that the optimized controller produces a finite residual.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        residual = desired_noise + effective_matrix @ taps

    if not np.isfinite(residual).all():
        raise ValueError(
            "The optimized residual contains NaN or infinite values."
        )

    return FixedFIRController(taps=taps)

@dataclass
class LMSController:
    """Adaptive FIR controller using the ordinary LMS algorithm"""

    filter_length: int
    step_size: float
    weights: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.filter_length < 1:
            raise ValueError("filter_length must be positive")

        if not np.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive")

        if self.weights is None:
            self.weights = np.zeros(
                self.filter_length,
                dtype=np.float64,
            )

        else:
            self.weights = np.asarray(
                self.weights,
                dtype=np.float64,
            ).copy()

        if self.weights.shape != (self.filter_length,):
            raise ValueError("weights must have shape (filter_length,).")

        if not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite.")

        def reset(self) -> None:
            """Reset the controller coefficients to zero"""
            self.weights = np.zeros(
                self.filter_length,
                dtype=np.float64,
            )
        def predict(self, reference_vector = FloatArray) -> float:
            """Compute the FIR output without updating the coefficients."""
            assert self.weights is not None
            reference_vector = np.asarray(
                reference_vector,
                dtype=np.float64,
            )

            if reference_vector.shape != (self.filter_length,):
                raise ValueError("reference_vector must have shape (filter_lenght,).")

            return float(self.weights @ reference_vector)

        def update(
            self,
            reference_vector: FloatArray,
            error: float,
        ) -> None:
            """Apply one ordinary LMS coefficient update.
            
            Update convention:
                weights <- weights - step_size * error * reference_vector
            """
            assert self.weights is not None
            reference_vector = np.asarray(
                reference_vector,
                dtype=np.float64,
            )

            if reference_vector.shape != (self.filter_length,):
                raise ValueError("reference_vector must have shape (filter_length,).")

            if not np.isfinite(reference_vector).all():
                raise ValueError("reference_vector must be finite.")

            if not np.isfinite(error):
                raise ValueError("error must be finite.")

            self.weights -= (
                self.step_size * error * reference_vector
            )

            if not np.isfinite(self.weights).all():
                raise ValueError("LMS coefficients became non-finite.")

            