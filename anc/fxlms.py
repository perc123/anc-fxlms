from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from anc.controllers import FxLMSController
from anc.plant import ANCPlant, apply_fir

FloatArray = NDArray[np.float64]
RunStatus = Literal["completed", "diverged"]


@dataclass
class FxLMSTrace:
    reference: FloatArray
    primary_noise: FloatArray
    control: FloatArray
    anti_noise: FloatArray
    residual: FloatArray
    filtered_reference: FloatArray
    weights: FloatArray

    status: RunStatus
    processed_samples: int
    divergence_sample: int | None
    divergence_reason: str | None


def _validate_limit(name: str, value: float | None) -> None:
    if value is not None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, or None.")   


def run_fxlms(
        reference: FloatArray,
        plant: ANCPlant,
        controller: FxLMSController,
        max_weight_norm: float | None = 1e6,
        max_abs_residual: float | None = 1e6,
) -> FxLMSTrace:
    """Run sample-by-sample FxLMS active noise cancellation.
        The residual convention is:
            e[n] = primary_noise[n] + anti_noise[n]

        The adaptation rule is:
            w[n+1] = w[n] - mu * e[n] * filtered_reference_vector[n]

        A run is marked "diverged" if a non-finite result occurs or either
        configured safety limit is exceeded.

        "completed" means the full input was processed without triggering
        these checks. It does not, by itself, imply convergence.

        Parameters
        ----------
        max_weight_norm:
            Maximum permitted Euclidean norm of the controller coefficients.
            Set to None to disable this limit.

        max_abs_residual:
            Maximum permitted absolute residual sample amplitude.
            Set to None to disable this limit.

    """

    _validate_limit("max_weight_norm", max_weight_norm)
    _validate_limit("max_abs_residual", max_abs_residual)

    reference = np.asarray(reference, dtype=np.float64)

    if reference.ndim != 1:
        raise ValueError("Reference must be one-dimensional.")

    if reference.size == 0:
        raise ValueError("Reference cannot be empty.")

    if not np.isfinite(reference).all():
        raise ValueError("Reference must contain only finite values.")

    if plant.secondary_delay_samples < 0:
        raise ValueError(
            "Actual secondary-path delay must be non-negative."
        )

    primary_noise = np.asarray(
        plant.primary_output(reference),
        dtype=np.float64,
    )

    actual_secondary_taps = np.asarray(
        plant.secondary_taps,
        dtype=np.float64,
    )

    if actual_secondary_taps.ndim != 1:
        raise ValueError("Actual secondary-path taps must be one-dimensional.")

    if actual_secondary_taps.size == 0:
        raise ValueError("Actual secondary-path taps cannot be empty.")

    if not np.isfinite(actual_secondary_taps).all():
        raise ValueError("Actual secondary-path taps must be finite.")

    if primary_noise.shape != reference.shape:
        raise ValueError("Primary noise must match the reference shape.")

    if not np.isfinite(primary_noise).all():
        raise ValueError("Primary noise contains NaN or infinite values.")

    # Filter the reference using the controller's estimated secondary path.
    # The resulting signal is used only for adaptation.
    filtered_reference = apply_fir(
        reference,
        controller.estimated_secondary_taps,
        delay_samples=controller.estimated_secondary_delay_samples,
    )

    if not np.isfinite(filtered_reference).all():
        raise ValueError(
            "Filtered reference contains NaN or infinite values."
        )

    n_samples = reference.size
    filter_length = controller.filter_length

    # Unprocessed samples stay zero in these arrays.
    control = np.zeros(n_samples, dtype=np.float64)
    anti_noise = np.zeros(n_samples, dtype=np.float64)
    residual = np.zeros(n_samples, dtype=np.float64)

    # NaN rows in weights indicate samples that were not processed.
    weights = np.full(
        (n_samples, filter_length),
        np.nan,
        dtype=np.float64,
    )

    reference_history = np.zeros(filter_length, dtype=np.float64)
    filtered_history = np.zeros(filter_length, dtype=np.float64)

    controller.reset()

    status: RunStatus = "completed"
    processed_samples = 0
    divergence_sample: int | None = None
    divergence_reason: str | None = None

    for n in range(n_samples):
        # 1. Update the reference history:
        #       [x[n], x[n-1], ..., x[n-L+1]].
        reference_history[1:] = reference_history[:-1]
        reference_history[0] = reference[n]

        # 2. Calculate the controller output.
        with np.errstate(over="ignore", invalid="ignore"):
            control_sample = controller.predict(reference_history)

        if not np.isfinite(control_sample):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller output became non-finite."
            )
            break

        # 3. Calculate the output of the ACTUAL secondary path.
        #      The explicit delay is modeled separately from the FIR taps.
        #      control[n] must be stored first: with zero delay, tap 0
        #      acts on the current control sample.
        control[n] = control_sample
        anti_noise_sample = 0.0
        delayed_index = n - plant.secondary_delay_samples

        if delayed_index >= 0:
            with np.errstate(over="ignore", invalid="ignore"):
                for k, tap in enumerate(actual_secondary_taps):
                    control_index = delayed_index - k

                    if control_index >= 0:
                        anti_noise_sample += tap * control[control_index]

        if not np.isfinite(anti_noise_sample):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Anti-noise became non-finite."
            )
            break

        # 4. Calculate the residual at the error microphone.
        with np.errstate(over="ignore", invalid="ignore"):
            residual_sample = primary_noise[n] + anti_noise_sample

        if not np.isfinite(residual_sample):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Residual became non-finite."
            )
            break

        #These three outputs are now valid for sample n.
        control[n] = control_sample
        anti_noise[n] = anti_noise_sample
        residual[n] = residual_sample
        processed_samples = n + 1

        # Check the residual safety limit before adaptation.
        if (
            max_abs_residual is not None
            and abs(residual_sample) > max_abs_residual
        ):
            weights[n] = controller.weights
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                f"Residual amplitude exceeded {max_abs_residual:g}."
            )
            break


        # 5. Build the filtered-reference vector using the ESTIMATED
        #       secondary path, then adapt the controller coefficients.
        filtered_history[1:] = filtered_history[:-1]
        filtered_history[0] = filtered_reference[n]

        # Keep the last known finite coefficients in case the update
        # overflows. The coefficient update is performed after the
        # current sample's output and residual have been calculated.
        weights_before_update = controller.weights.copy()

        # 6. Adapt the controller coefficients.
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                controller.update(
                    filtered_reference_vector=filtered_history,
                    error=residual_sample,
                )
        except FloatingPointError:
            weights[n] = weights_before_update
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller coefficient became non-finite during adaptation."
            )
            break

        if not np.isfinite(controller.weights).all():
            weights[n] = weights_before_update
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller coefficients became non-finite."
            )
            break

        weights[n] = controller.weights

        # 7. Check the coefficient-norm safety limit.
        with np.errstate(over="ignore", invalid="ignore"):
            weight_norm = float(np.linalg.norm(controller.weights))

        if not np.isfinite(weight_norm):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                "Controller coefficient norm became non-finite."
            )
            break

        if(
            max_weight_norm is not None
            and weight_norm > max_weight_norm
        ):
            status = "diverged"
            divergence_sample = n
            divergence_reason = (
                f"Controller weight norm exceeded {max_weight_norm:g}."
            )
            break



    return FxLMSTrace(
        reference=reference,
        primary_noise=primary_noise,
        control=control,
        anti_noise=anti_noise,
        residual=residual,
        filtered_reference=filtered_reference,
        weights=weights,
        status=status,
        processed_samples=processed_samples,
        divergence_sample=divergence_sample,
        divergence_reason=divergence_reason,
    )