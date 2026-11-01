"""Sample-streaming FxLMS reference with the call order used by the C++ port.

Audio-callback convention
-------------------------
Each call receives the reference sample x[n] and the error-microphone
sample e[n] together. Because of the secondary-path latency, e[n] only
reflects controller outputs up to y[n-1]. The error follows the project
convention

    e[n] = primary_noise[n] + anti_noise[n]

so a call performs, in this exact order:

    1. update   w <- w - mu * e[n] * x'_hist      (x'_hist stored by the
                                                  previous call)
    2. push     x[n] into the reference history and the secondary-path
                input history, compute x'[n] = (S_hat * x)[n] and push it
                into the filtered-reference history
    3. predict  y[n] = w . x_hist

The batch harness in anc.fxlms computes y[n], then e[n], then updates in the
same step. The two are intentionally different algorithms, and only this
module matches the C++ implementation sample for sample.
"""

import numpy as np

from anc.plant import ANCPlant
from anc.types import FloatArray


class StreamingFxLMS:
    """Controller and secondary-path estimate S_hat, processed per sample."""

    def __init__(
            self,
            filter_length: int,
            step_size: float,
            estimated_secondary_taps: FloatArray,
            estimated_secondary_delay_samples: int = 0,
            dtype: type[np.floating] = np.float64,
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

    def process(self, x: float, e: float) -> float:
        """Consume x[n] and e[n], return the control sample y[n]."""
        dtype = self.dtype

        # 1. Update with the stored filtered-reference vector.
        self.weights -= (self.step_size * dtype(e)) * self._xf_hist

        # 2. Push the new reference sample and its filtered version.
        self._x_hist[1:] = self._x_hist[:-1]
        self._x_hist[0] = x

        self._s_in_hist[1:] = self._s_in_hist[:-1]
        self._s_in_hist[0] = x
        delayed = self._s_in_hist[self.s_hat_delay:]
        x_filtered = np.dot(self.s_hat, delayed)

        self._xf_hist[1:] = self._xf_hist[:-1]
        self._xf_hist[0] = x_filtered

        # 3. Controller output.
        return float(np.dot(self.weights, self._x_hist))


def run_streaming_closed_loop(
        reference: FloatArray,
        plant: ANCPlant,
        controller: StreamingFxLMS,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Drive the streaming controller with the plant simulated in Python.

    Returns the (x, e, y) sequences. The plant needs at least one sample of
    secondary-path delay, since e[n] may not depend on y[n].
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
    error = np.zeros(n_samples, dtype=np.float64)
    control = np.zeros(n_samples, dtype=np.float64)

    controller.reset()

    for n in range(n_samples):
        anti_noise = 0.0
        for k, tap in enumerate(s_taps):
            index = n - delay - k
            if index >= 0:
                anti_noise += tap * control[index]

        error[n] = primary_noise[n] + anti_noise
        control[n] = controller.process(reference[n], error[n])

    return reference, error, control
