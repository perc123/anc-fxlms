from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.controllers import FixedFIRController
from anc.simulate import SimulationTrace
from anc.adaptive import LMSTrace
from anc.fxlms import FxLMSTrace
from anc.convergence import ConvergenceMetrics


def plot_fxlms_results(
    trace: FxLMSTrace,
    sample_rate_hz: int,
    output_dir: Path | None = None,
    show: bool = False,
    convergence_metrics: ConvergenceMetrics | None = None,
) -> list[Path]:
    """Plot FxLMS signals, convergence metrics, and coefficient evolution."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = trace.reference.size
    time_s = np.arange(n_samples) / sample_rate_hz
    saved_paths: list[Path] = []

    def finish(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()

        if output_dir is not None:
            path = output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved_paths.append(path)

        if show:
            plt.show()

        plt.close(fig)

    # ---------------------------------------------------------
    # 1. Time-domain signals.
    # ---------------------------------------------------------
    shown = min(n_samples, sample_rate_hz)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True,
    )

    axes[0].plot(
        time_s[:shown],
        trace.primary_noise[:shown],
        label="Primary noise",
    )
    axes[0].plot(
        time_s[:shown],
        trace.anti_noise[:shown],
        label="Anti-noise",
        alpha=0.8,
    )

    axes[0].set_title(
        "Primary noise and secondary-path anti-noise"
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time_s[:shown],
        trace.residual[:shown],
    )
    axes[1].set_title("Residual at the error microphone")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    finish(
        fig,
        "step3_fxlms_mismatch_timeseries.png",
    )

    # ---------------------------------------------------------
    # 2. Learning curve and convergence indicators.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    if convergence_metrics is not None:
        block_times = (
            convergence_metrics.block_start_times_s
        )
        primary_rms = (
            convergence_metrics.block_primary_rms
        )
        residual_rms = (
            convergence_metrics.block_residual_rms
        )

        ax.plot(
            block_times,
            primary_rms,
            label="Primary-noise RMS",
        )
        ax.plot(
            block_times,
            residual_rms,
            label="Residual RMS",
        )

        steady_state_time = (
            convergence_metrics.steady_state_start_sample
            / sample_rate_hz
        )

        ax.axvline(
            steady_state_time,
            linestyle=":",
            linewidth=1.5,
            label="Steady-state window begins",
        )

        if convergence_metrics.convergence_time_s is not None:
            ax.axvline(
                convergence_metrics.convergence_time_s,
                linestyle="--",
                linewidth=1.5,
                label="Convergence time",
            )

    else:
        # A diverged run has no valid full-run convergence metrics.
        # Plot only complete blocks from the processed portion.
        valid_samples = trace.processed_samples
        block_size = int(round(0.1 * sample_rate_hz))

        block_times = []
        primary_rms = []
        residual_rms = []

        for start in range(0, valid_samples, block_size):
            stop = min(start + block_size, valid_samples)

            if stop - start < block_size:
                break

            block_times.append(start / sample_rate_hz)

            primary_rms.append(
                float(
                    np.sqrt(
                        np.mean(
                            trace.primary_noise[start:stop] ** 2
                        )
                    )
                )
            )

            residual_rms.append(
                float(
                    np.sqrt(
                        np.mean(
                            trace.residual[start:stop] ** 2
                        )
                    )
                )
            )

        ax.plot(
            block_times,
            primary_rms,
            label="Primary-noise RMS",
        )
        ax.plot(
            block_times,
            residual_rms,
            label="Residual RMS",
        )

        ax.set_title(
            "FxLMS learning curve — incomplete run"
        )

    ax.set_title(
        "FxLMS learning curve and convergence"
        if convergence_metrics is not None
        else "FxLMS learning curve — incomplete run"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Block RMS amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    finish(
        fig,
        "step3_fxlms_mismatch_learning_curve.png",
    )

    # ---------------------------------------------------------
    # 3. Controller coefficient evolution.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))

    for k in range(min(8, trace.weights.shape[1])):
        ax.plot(
            time_s,
            trace.weights[:, k],
            label=f"Tap {k}",
        )

    ax.set_title("FxLMS coefficient evolution")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coefficient value")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    finish(
        fig,
        "step3_fxlms_mismatch_weights.png",
    )

    return saved_paths


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

    # Power spectral density before and after control, transient excluded.
    nfft = min(512, n - start_sample)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.psd(
        trace.primary_noise[start_sample:],
        NFFT=nfft,
        Fs=sample_rate_hz,
        label="Primary noise",
    )
    ax.psd(
        trace.residual[start_sample:],
        NFFT=nfft,
        Fs=sample_rate_hz,
        label="Residual",
    )
    ax.set_title("Power Spectral Density Before and After Control")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "step1_psd.png", dpi=160)

    if show:
        plt.show()

    plt.close(fig)

def plot_lms_results(
        trace: LMSTrace,
        sample_rate_hz: int,
        output_dir: Path | None = None,
        show: bool = False,
) -> list[Path]:
    """Plot LMS signals, block-RMS learning curve and tap evolution."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    n_samples = trace.reference.size
    time_s = np.arange(n_samples) / sample_rate_hz

    def finish_figure(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()

        if output_dir is not None:
            path = output_dir / filename
            fig.savefig(path, dpi=150,bbox_inches="tight")
            saved_paths.append(path)

        if show:
            plt.show()

        plt.close(fig)

    # 1. Time-domain signals: show at most the first second.
    shown_samples = min(n_samples, sample_rate_hz)
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True
    )

    axes[0].plot(
        time_s[:shown_samples],
        trace.reference[:shown_samples],
    )
    axes[0].set_title("Reference signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time_s[:shown_samples],
        trace.desired[:shown_samples],
        label="Desired",
    )
    axes[1].plot(
        time_s[:shown_samples],
        trace.control[:shown_samples],
        label="LMS output (anti-signal)",
        alpha=0.8,
    )
    axes[1].set_title("Desired signal and adaptive-filter output (error = desired + output)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time_s[:shown_samples],
        trace.error[:shown_samples],
    )
    axes[2].set_title("Residual error")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, alpha=0.3)

    finish_figure(fig, "step2_lms_timeseries.png")

    # 2. Learning curve using non-overlapping 100 ms RMS blocks.
    block_size = max(1, int(round(0.1 * sample_rate_hz)))
    starts = np.arange(0, n_samples, block_size)

    block_times = []
    desired_rms = []
    error_rms = []

    for start in starts:
        stop = min(start + block_size, n_samples)
        block_times.append((start + stop) / (2 * sample_rate_hz))

        desired_block = trace.desired[start:stop]
        error_block = trace.error[start:stop]

        desired_rms.append(
            float(np.sqrt(np.mean(desired_block**2)))
        )
        error_rms.append(
            float(np.sqrt(np.mean(error_block**2)))
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(block_times, desired_rms, label="Desired RMS")
    ax.plot(block_times, error_rms, label="Error RMS")
    ax.set_title("LMS learning curve (100 ms block RMS)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    finish_figure(fig, "step2_lms_learning_curve.png")

    # 3. Evolution of the first eight adaptive coefficients.
    weights_to_show = min(8, trace.weights.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(weights_to_show):
        ax.plot(time_s, trace.weights[:, k], label=f"Tap {k}")

    ax.set_title("LMS coefficients evolution")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coefficient value")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    finish_figure(fig, "step2_lms_weights.png")

    return saved_paths