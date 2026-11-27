"""Sample-streaming FxLMS reference with the call order used by the C++ port.

Audio-callback convention
-------------------------
Each call receives the reference sample x[n] and the error-microphone
sample e[n] together. Because of the secondary-path latency, e[n] only
reflects controller outputs up to y[n-1]. The error follows the project
convention

    e[n] = primary_noise[n] + anti_noise[n]

so a call performs, in this exact order:

    1. push     x[n] into the reference history and the secondary-path
                input history
    2. filter   compute x'[n] = (S_hat * x)[n] and push it into the
                filtered-reference history
    3. update   w <- w - mu * e[n] * x'_hist      (x'_hist now ends at x'[n])
    4. predict  y[n] = w . x_hist

e[n] must be paired with x'[n], as in the batch harness in anc.fxlms. Updating
before x'[n] is pushed pairs e[n] with x'[n-1], which shifts the stability
window of the phase error by one sample of S_hat delay.

The batch harness computes y[n], then e[n], then updates, so its y[n] uses
weights adapted through e[n-1]; here y[n] also uses the update from e[n]. The
stability behaviour is the same, and only this module matches the C++
implementation sample for sample.

See docs/algorithm_specification.md, section 5.2.
"""

from enum import Enum
import numpy as np

from anc.plant import ANCPlant
from anc.types import FloatArray


class Fault(str, Enum):
    """Why the controller stopped adapting. Values match the C++ to_string()."""

    NONE = "none"
    NON_FINITE_INPUT = "non_finite_input"
    ERROR_LIMIT = "error_limit"
    WEIGHTS_NOT_FINITE = "weights_not_finite"
    WEIGHT_NORM_LIMIT = "weight_norm_limit"
    OUTPUT_NOT_FINITE = "output_not_finite"


def _validate_limit(name: str, value: float | None) -> float:
    """Map None to infinity (disabled); reject NaN, zero and negative limits."""
    if value is None:
        return float("inf")
    if not value > 0:   # False for NaN, zero and negative values
        raise ValueError(f"{name} must be positive, or None to disable it.")
    return float(value)


class StreamingFxLMS:
    """Controller and secondary-path estimate S_hat, processed per sample.

    Fault behaviour (algorithm specification, section 8.1): the first failed
    check latches the controller. From then on process() returns 0 and changes
    no state, apart from the call counter samples_processed, until reset().
    If the weight check fails, the weights are restored to their values
    before that update.
    """

    def __init__(
            self,
            filter_length: int,
            step_size: float,
            estimated_secondary_taps: FloatArray,
            estimated_secondary_delay_samples: int = 0,
            dtype: type[np.floating] = np.float64,
            max_abs_error: float | None = None,
            max_weight_norm: float | None = None,
    ) -> None:
        if filter_length < 1:
            raise ValueError("filter_length must be positive.")

        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive.")

        if estimated_secondary_delay_samples < 0:
            raise ValueError("Estimated secondary-path delay must be non-negative.")

        s_hat = np.asarray(estimated_secondary_taps, dtype=dtype)

        if s_hat.ndim != 1 or s_hat.size == 0:
            raise ValueError("estimated_secondary_taps must be a nonempty vector.")

        if not np.isfinite(s_hat).all():
            raise ValueError("Estimated secondary-path taps must be finite.")

        self.dtype = dtype
        self.filter_length = filter_length
        self.step_size = dtype(step_size)
        self.s_hat = s_hat.copy()
        self.s_hat_delay = estimated_secondary_delay_samples
        self.max_abs_error = _validate_limit("max_abs_error", max_abs_error)
        self.max_weight_norm = _validate_limit("max_weight_norm", max_weight_norm)
        self._max_weight_norm_sq = self.max_weight_norm ** 2
        self.reset()

    def reset(self) -> None:
        dtype = self.dtype
        self.weights = np.zeros(self.filter_length, dtype=dtype)
        # All histories are newest-first: index k holds the sample from n-k.
        self._x_hist = np.zeros(self.filter_length, dtype=dtype)
        self._xf_hist = np.zeros(self.filter_length, dtype=dtype)
        self._s_in_hist = np.zeros(
            self.s_hat_delay + self.s_hat.size, dtype=dtype
        )
        self.fault = Fault.NONE
        self.fault_sample = 0
        self.samples_processed = 0

    @property
    def diverged(self) -> bool:
        return self.fault is not Fault.NONE

    def _latch(self, fault: Fault, n: int) -> float:
        self.fault = fault
        self.fault_sample = n
        return 0.0

    def process(self, x: float, e: float) -> float:
        """Consume x[n] and e[n], return the control sample y[n]."""
        dtype = self.dtype
        n = self.samples_processed
        self.samples_processed += 1

        if self.diverged:
            return 0.0   # latched: silence until reset()

        # 0. Input checks. A faulted call changes no state except the fault and the counter.
        if not (np.isfinite(x) and np.isfinite(e)):
            return self._latch(Fault.NON_FINITE_INPUT, n)
        if abs(e) > self.max_abs_error:
            return self._latch(Fault.ERROR_LIMIT, n)

        # 1. Push the new reference sample and its filtered version.
        self._x_hist[1:] = self._x_hist[:-1]
        self._x_hist[0] = x

        self._s_in_hist[1:] = self._s_in_hist[:-1]
        self._s_in_hist[0] = x
        delayed = self._s_in_hist[self.s_hat_delay:]
        x_filtered = np.dot(self.s_hat, delayed)

        self._xf_hist[1:] = self._xf_hist[:-1]
        self._xf_hist[0] = x_filtered

        # 2. Update: e[n] is paired with x'[n], so this follows the push.
        previous_weights = self.weights.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            self.weights -= (self.step_size * dtype(e)) * self._xf_hist
            norm_sq = float(np.dot(self.weights, self.weights))

        if not np.isfinite(norm_sq):
            self.weights = previous_weights
            return self._latch(Fault.WEIGHTS_NOT_FINITE, n)
        if norm_sq > self._max_weight_norm_sq:
            self.weights = previous_weights
            return self._latch(Fault.WEIGHT_NORM_LIMIT, n)

        # 3. Controller output.
        with np.errstate(over="ignore", invalid="ignore"):
            y = float(np.dot(self.weights, self._x_hist))
        if not np.isfinite(y):
            return self._latch(Fault.OUTPUT_NOT_FINITE, n)
        return y


def run_streaming_closed_loop(
        reference: FloatArray,
        plant: ANCPlant,
        controller: StreamingFxLMS,
        measurement_noise: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Drive the streaming controller with the plant simulated in Python.

    Returns the (x, e, y) sequences. The plant needs at least one sample of
    secondary-path delay, since e[n] may not depend on y[n].

    measurement_noise:
        Error-microphone noise v[n], added to e[n] before it reaches the
        controller. None means a noise-free sensor (identical to v[n] = 0).
    """
    if plant.secondary_delay_samples < 1:
        raise ValueError(
            "Streaming operation needs secondary_delay_samples >= 1: "
            "e[n] cannot depend on y[n]."
        )

    reference = np.asarray(reference, dtype=np.float64)
    primary_noise = plant.primary_output(reference)
    s_taps = np.asarray(plant.secondary_taps, dtype=np.float64)
    delay = plant.secondary_delay_samples

    n_samples = reference.size

    if measurement_noise is None:
        measurement_noise = np.zeros(n_samples, dtype=np.float64)
    else:
        measurement_noise = np.asarray(measurement_noise, dtype=np.float64)
        if measurement_noise.shape != reference.shape:
            raise ValueError("Measurement noise must match the reference shape.")
        if not np.isfinite(measurement_noise).all():
            raise ValueError("Measurement noise must contain only finite values.")
            
    error = np.zeros(n_samples, dtype=np.float64)
    control = np.zeros(n_samples, dtype=np.float64)

    controller.reset()

    for n in range(n_samples):
        anti_noise = 0.0
        for k, tap in enumerate(s_taps):
            index = n - delay - k
            if index >= 0:
                anti_noise += tap * control[index]

        error[n] = primary_noise[n] + anti_noise + measurement_noise[n]
        control[n] = controller.process(reference[n], error[n])

    return reference, error, control
