from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anc.controllers import FixedFIRController
from anc.plant import ANCPlant

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationTrace:
    reference: FloatArray
    primary_noise: FloatArray
    control: FloatArray
    anti_noise: FloatArray
    residual: FloatArray


def run(
        controller: FixedFIRController,
        plant: ANCPlant,
        signal: FloatArray,
) -> SimulationTrace:
    """Run a fixed-controller ANC simulation."""
    reference = np.asarray(signal, dtype=np.float64)

    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Signal must be a noempty 1-D array.")

    primary_noise = plant.primary_output(reference)

    control = controller.process(reference)

    anti_noise = plant.secondary_output(control)

    residual = primary_noise + anti_noise

    return SimulationTrace(
        reference=reference.copy(),
        primary_noise=primary_noise,
        control=control,
        anti_noise=anti_noise,
        residual=residual,
    )