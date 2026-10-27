from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from anc.adaptive import run_lms
from anc.controllers import LMSController
from anc.metrics import attenuation_db, rms
from anc.plant import apply_fir
from anc.signals import generate_tone_mixture
from experiments.plotting import plot_lms_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_string: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Step 2 ordinary LMS experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/step2_lms.yaml",
        help="Path to the experiment YAML configuration."
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_cfg = config["experiment"]
    signal_cfg = config["signal"]
    plant_cfg = config["plant"]
    controller_cfg = config["controller"]
    evaluation_cfg = config["evaluation"]
    output_cfg = config["output"]

    sample_rate_hz = int(signal_cfg["sample_rate_hz"])
    duration_s = float(signal_cfg["duration_s"])
    seed = int(experiment_cfg["seed"])

    # Generate the reference signal.
    reference = generate_tone_mixture(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        tones=signal_cfg["tones"],
        white_noise_std=float(
            signal_cfg.get("white_noise_std", 0.0)
        ),
        seed=seed,
    )

    # Generate the target using the primary path.
    # This is the direct-error LMS benchmark, not the full ANC loop.
    primary_taps = np.asarray(
        plant_cfg["primary_taps"],
        dtype=np.float64,
    )
    desired_noise = apply_fir(reference, primary_taps)

    controller = LMSController(
        filter_length=int(controller_cfg["length"]),
        step_size=float(controller_cfg["step_size"]),
    )

    trace = run_lms(
        reference=reference,
        desired=desired_noise,
        controller=controller,
    )

    # Exclude the configured startup transient from evaluation.
    transient_s = float(
        evaluation_cfg.get("transient_s", 0.0)
    )
    start_sample = int(round(transient_s * sample_rate_hz))

    if not 0 <= start_sample < reference.size:
        raise ValueError(
            "evaluation.transient_s must be non-negative and " \
            "shorter than the signal duration."
        )

    target_rms = rms(trace.desired[start_sample:])
    error_rms = rms(trace.error[start_sample:])

    error_reduction_db = attenuation_db(
        trace.desired,
        trace.error,
        start_sample=start_sample,
    )

    # Validate the arrays before saving.
    for name, array in (
        ("reference", trace.reference),
        ("desired", trace.desired),
        ("control", trace.control),
        ("error", trace.error),
        ("weights", trace.weights), 
    ):
        if not np.isfinite(array).all():
            raise FloatingPointError(
                f"{name} contains NaN or infinite values."
            )

    results_dir = resolve_path(output_cfg["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    traces_path = results_dir / "traces.npz"
    np.savez_compressed(
        traces_path,
        reference=trace.reference,
        desired=trace.desired,
        control=trace.control,
        error=trace.error,
        weights=trace.weights,
    )

    metrics_path = results_dir / "metrics.txt"
    metrics_text = (
        f"Experiment: {experiment_cfg['name']}\n"
        f"Sample rate: {sample_rate_hz} Hz\n"
        f"Duration: {duration_s} s\n"
        f"Controller length: {controller.filter_length} taps\n"
        f"LMS step size: {controller.step_size}\n"
        f"Evaluation transient: {transient_s} s\n"
        f"Target RMS: {target_rms:.8f}\n"
        f"Prediction error RMS: {error_rms:.8f}\n"
        f"Prediction-error reduction: {error_reduction_db:.4f} dB\n"
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")

    save_figures = bool(output_cfg.get("save_figures", True))
    show_figures = bool(output_cfg.get("show_figures", False))

    figure_dir = (
        resolve_path(output_cfg["figure_directory"])
        if save_figures
        else None
    )

    figure_paths = plot_lms_results(
        trace=trace,
        sample_rate_hz=sample_rate_hz,
        output_dir=figure_dir,
        show=show_figures,
    )

    print(f"Experiment: {experiment_cfg['name']}")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Duration: {duration_s} s")
    print(f"Controller length: {controller.filter_length} taps")
    print(f"LMS step size: {controller.step_size}")
    print(f"Target RMS: {target_rms:.8f}")
    print(f"Prediction error RMS: {error_rms:.8f}")
    print(
        "Prediction-error reduction: "
        f"{error_reduction_db:.4f} dB"
    )
    print(f"Metrics saved to: {metrics_path}")
    print(f"Traces saved to: {traces_path}")

    for path in figure_paths:
        print(f"Figure saved to: {path}")


if __name__ == "__main__":
    main()