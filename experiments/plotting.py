from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.controllers import FixedFIRController
from anc.simulate import SimulationTrace

def plot_results(
        trace: SimulationTrace,
        controller: FixedFIRController,
        sample_rate_hz: int,
        output_dir: Path,
        start_sample: int,
        show: bool = False,
) -> None:
    """Save the baseline ANC figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = trace.reference.size
    t = np.arange(n) / sample_rate_hz

    # Plot a short interval to make the waveform structure visible.
    end_sample = min(n, start_sample + int(0.05 * sample_rate_hz))
    selection = slice(start_sample, end_sample)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(t[selection], trace.reference[selection])
    axes[0].set_title("Reference Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(
        t[selection],
        trace.primary_noise[selection],
        label="Primary noise",
    )
    axes[1].plot(
        t[selection],
        trace.anti_noise[selection],
        label="Anti-noise at error microphone",
    )

    axes[1].set_title("Primary Noise and Anti-Noise")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t[selection], trace.residual[selection])
    axes[2].set_title("Residual Noise")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "step1_signals.png", dpi=160)

    if show:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.stem(
        np.arange(controller.taps.size),
        controller.taps,
    )
    ax.set_title("Fixed FIR Controller Coefficients")
    ax.set_xlabel("Tap index")
    ax.set_ylabel("Coefficient")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "step1_coefficients.png", dpi=160)

    if show:
        plt.show()

    plt.close()