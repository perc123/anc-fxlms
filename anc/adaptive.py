from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from anc.controllers import LMSController

FloatArray = NDArray[np.float64]
RunStatus = Literal["completed", "diverged"]


@dataclass
class LMSTrace:
    reference: FloatArray
    desired: FloatArray
    control: FloatArray
    error: FloatArray
    measurement_noise: FloatArray      # v[n] on the desired observation
    measured_error: FloatArray         # (desired + v) - control
    weights: FloatArray
    status: RunStatus
    processed_samples: int
    divergence_sample: int | None
    divergence_reason: str | None


def _validate_limit(name: str, value: float | None) -> None:
    if value is not None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be finite and positive, or None."
            )


def run_lms(
    reference: FloatArray,
    desired: FloatArray,
    controller: LMSController,
    max_weight_norm: float | None = None,
    max_abs_error: float | None = None,
    measurement_noise: FloatArray | None = None, 
) -> LMSTrace:
    """Run ordinary LMS on a direct-error adaptive-filter problem.

    The error convention is:
        error[n] = desired[n] - control[n]

    The coefficient update is:
        weights[n+1] = weights[n]
                       + step_size * error[n] * reference_vector[n]

    A run is marked "diverged" if a non-finite result occurs or
    a configured safety limit is exceeded.

    "completed" means that the complete input was processed without
    triggering these checks. It does not, by itself, prove convergence.

    Parameters
    ----------
    max_weight_norm:
        Maximum permitted Euclidean norm of the controller coefficients.
        Set to None to disable this limit.

    max_abs_error:
        Maximum permitted absolute prediction-error sample.
        Set to None to disable this limit.

    With observation noise v[n] on the desired signal, the filter
    adapts on the measured error:
        measured_error[n] = (desired[n] + v[n]) - control[n]
                          = error[n] + v[n]
    trace.error and trace.desired always hold the noise-free values.

    measurement_noise:
        Observation noise v[n], same shape as desired.
        None means a noise-free observation (identical to v[n] = 0).
    """

    _validate_limit("max_weight_norm", max_weight_norm)
    _validate_limit("max_abs_error", max_abs_error)

    reference = np.asarray(reference, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)

    if reference.ndim != 1 or desired.ndim != 1:
        raise ValueError("Signals must be one-dimensional.")

    if reference.size == 0:
        raise ValueError("Signals cannot be empty.")

    if reference.shape != desired.shape:
        raise ValueError("Signals must have the same length.")

    if not np.isfinite(reference).all():
        raise ValueError("Reference must be finite.")

    if not np.isfinite(desired).all():
        raise ValueError("Desired signal must be finite.")

    # Observation noise v[n] on the desired signal. None means a
    # noise-free observation: v[n] = 0, identical to the original.
    if measurement_noise is None:
        measurement_noise = np.zeros_like(desired)
    else:
        measurement_noise = np.asarray(measurement_noise, dtype=np.float64)

        if measurement_noise.shape != desired.shape:
            raise ValueError(
                "Measurement noise must match the desired-signal shape."
            )

        if not np.isfinite(measurement_noise).all():
            raise ValueError(
                "Measurement noise must contain only finite values."
            )

    controller.reset()

    n_samples = reference.size
    filter_length = controller.filter_length

    # NaN indicates that a sample was not processed.
    control = np.full(n_samples, np.nan, dtype=np.float64)
    error = np.full(n_samples, np.nan, dtype=np.float64)
    measured_error = np.full(n_samples, np.nan, dtype=np.float64)

    weight_history = np.full(
        (n_samples, filter_length),
        np.nan,
        dtype=np.float64,
    )

    reference_history = np.zeros(
        filter_length,
        dtype=np.float64,
    )

    status: RunStatus = "completed"
    processed_samples = 0
    divergence_sample: int | None = None
    divergence_reason: str | None = None

    for n in range(n_samples):
        # 1. Update reference history:
        #    [x[n], x[n-1], ..., x[n-L+1]].
        reference_history[1:] = reference_history[:-1]
        reference_history[0] = reference[n]

        # 2. Calculate the current FIR output.
        with np.errstate(over="ignore", invalid="ignore"):
            control_sample = controller.predict(reference_history)

        if not np.isfinite(control_sample):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller output became non-finite."
            )
            break

        # 3. Calculate prediction error.
        with np.errstate(over="ignore", invalid="ignore"):
            error_sample = desired[n] - control_sample

        if not np.isfinite(error_sample):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Prediction error became non-finite."
            )
            break

                # 3b. Form the MEASURED error against the noisy observation:
        #       e_m[n] = (d[n] + v[n]) - y[n] = e[n] + v[n]
        measured_error_sample = error_sample + measurement_noise[n]

        # The output and errors at this sample are valid.
        control[n] = control_sample
        error[n] = error_sample
        measured_error[n] = measured_error_sample
        processed_samples = n + 1

        # 4. Check the error-amplitude safety limit on the
        #    signal the adaptive filter actually observes.
        if (
            max_abs_error is not None
            and abs(error_sample) > max_abs_error
        ):
            weight_history[n] = controller.weights
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                f"Prediction error amplitude exceeded "
                f"{max_abs_error:g}."
            )
            break

        # 5. Preserve the last valid coefficients before adaptation.
        weights_before_update = controller.weights.copy()

        # 6. Apply one LMS coefficient update.
        # LMSController.update() subtracts step_size * error * x.
        # Passing -error therefore implements:
        # weights <- weights + step_size * error * x.
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                controller.update(
                    reference_history,
                    -measured_error_sample,
                )
        except FloatingPointError:
            controller.weights = weights_before_update
            weight_history[n] = weights_before_update

            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller coefficients became non-finite "
                "during adaptation."
            )
            break

        if not np.isfinite(controller.weights).all():
            controller.weights = weights_before_update
            weight_history[n] = weights_before_update

            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller coefficients became non-finite."
            )
            break

        weight_history[n] = controller.weights

        # 7. Check coefficient norm.
        with np.errstate(over="ignore", invalid="ignore"):
            weight_norm = float(
                np.linalg.norm(controller.weights)
            )

        if not np.isfinite(weight_norm):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller coefficient norm became non-finite."
            )
            break

        if (
            max_weight_norm is not None
            and weight_norm > max_weight_norm
        ):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                f"Controller weight norm exceeded "
                f"{max_weight_norm:g}."
            )
            break

    return LMSTrace(
        reference=reference,
        desired=desired,
        control=control,
        error=error,
        measurement_noise=measurement_noise,
        measured_error=measured_error,
        weights=weight_history,
        status=status,
        processed_samples=processed_samples,
        divergence_sample=divergence_sample,
        divergence_reason=divergence_reason,
    )