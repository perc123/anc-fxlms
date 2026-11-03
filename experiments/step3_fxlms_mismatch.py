from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.metrics import attenuation_db, rms
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture
from anc.convergence import analyze_convergence
from experiments.plotting import plot_fxlms_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FxLMS with secondary-path model mismatch."
    )
    parser.add_argument(
        "--config",
        default="configs/step3_fxlms_mismatch.yaml",
        help="Path to the experiment YAML configuration.",
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

    tones = [
        {
            "frequency_hz": float(tone["frequency_hz"]),
            "amplitude": float(tone["amplitude"]),
        }
        for tone in signal_cfg["tones"]
    ]

    reference = generate_tone_mixture(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        tones=tones,
        white_noise_std=float(
            signal_cfg.get("white_noise_std", 0.0)
        ),
        seed=seed,
    )

    plant = ANCPlant(
        primary_taps=np.asarray(
            plant_cfg["primary_taps"],
            dtype=np.float64,
        ),
        secondary_taps=np.asarray(
            plant_cfg["secondary_taps"],
            dtype=np.float64,
        ),
        secondary_delay_samples=int(
            plant_cfg["secondary_delay_samples"]
        ),
    )

    controller = FxLMSController(
        filter_length=int(controller_cfg["length"]),
        step_size=float(controller_cfg["step_size"]),
        estimated_secondary_taps=np.asarray(
            controller_cfg["estimated_secondary_taps"],
            dtype=np.float64,
        ),
        estimated_secondary_delay_samples=int(
            controller_cfg["estimated_secondary_delay_samples"]
        ),
    )

    safety_cfg = config.get("safety", {})

    trace = run_fxlms(
        reference=reference,
        plant=plant,
        controller=controller,
        max_weight_norm=safety_cfg.get("max_weight_norm", 1e6),
        max_abs_residual=safety_cfg.get("max_abs_residual", 1e6),
    )

    transient_s = float(
        evaluation_cfg.get("transient_s", 0.0)
    )
    start_sample = int(round(transient_s * sample_rate_hz))

    if not 0 <= start_sample < reference.size:
        raise ValueError(
            "evaluation.transient_s must be non-negative and "
            "shorter than the signal duration."
        )

    # Only analyze samples that the FxLMS simulation actually processed.
    valid_samples = trace.processed_samples

    primary_valid = trace.primary_noise[:valid_samples]
    residual_valid = trace.residual[:valid_samples]

    # Calculate performance over the available evaluation interval.
    if valid_samples > start_sample:
        primary_rms = rms(primary_valid[start_sample:])
        residual_rms = rms(residual_valid[start_sample:])

        anc_attenuation_db = attenuation_db(
            primary_valid,
            residual_valid,
            start_sample=start_sample,
        )
    else:
        primary_rms = float("nan")
        residual_rms = float("nan")
        anc_attenuation_db = float("nan")

    # Calculate steady-state and convergence metrics only for a
    # fully completed run. A diverged run cannot establish steady state.
    convergence_metrics = None

    if trace.status == "completed":
        if valid_samples != reference.size:
            raise RuntimeError(
                "Completed run does not contain the full reference signal."
            )

        convergence_metrics = analyze_convergence(
            primary_noise=primary_valid,
            residual=residual_valid,
            sample_rate_hz=sample_rate_hz,
            start_sample=start_sample,
            block_duration_s=float(evaluation_cfg.get("block_duration_s", 0.1)),
            steady_state_fraction=float(evaluation_cfg.get("steady_state_fraction", 0.2)),
            tolerance_db=float(evaluation_cfg.get("convergence_tolerance_db", 1.0)),
            consecutive_blocks=int(evaluation_cfg.get("consecutive_blocks", 3)),
        )

    results_dir = resolve_path(output_cfg["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    traces_path = results_dir / "traces.npz"

    np.savez_compressed(
        traces_path,
        reference=trace.reference,
        primary_noise=trace.primary_noise,
        control=trace.control,
        anti_noise=trace.anti_noise,
        residual=trace.residual,
        filtered_reference=trace.filtered_reference,
        weights=trace.weights,
        status=np.asarray(trace.status),
        processed_samples=np.asarray(trace.processed_samples),
        divergence_sample=np.asarray(
            -1 if trace.divergence_sample is None
            else trace.divergence_sample
        ),
        divergence_reason=np.asarray(
            trace.divergence_reason or ""
        ),
        block_start_times_s=(
            convergence_metrics.block_start_times_s
            if convergence_metrics is not None
            else np.empty(0, dtype=np.float64)
        ),
        block_primary_rms=(
            convergence_metrics.block_primary_rms
            if convergence_metrics is not None
            else np.empty(0, dtype=np.float64)
        ),
        block_residual_rms=(
            convergence_metrics.block_residual_rms
            if convergence_metrics is not None
            else np.empty(0, dtype=np.float64)
        ),
        steady_state_start_sample=np.asarray(
            convergence_metrics.steady_state_start_sample
            if convergence_metrics is not None
            else -1
        ),
        steady_state_primary_rms=np.asarray(
            convergence_metrics.steady_state_primary_rms
            if convergence_metrics is not None
            else np.nan
        ),
        steady_state_residual_rms=np.asarray(
            convergence_metrics.steady_state_residual_rms
            if convergence_metrics is not None
            else np.nan
        ),
        steady_state_attenuation_db=np.asarray(
            convergence_metrics.steady_state_attenuation_db
            if convergence_metrics is not None
            else np.nan
        ),
        convergence_time_s=np.asarray(
            convergence_metrics.convergence_time_s
            if convergence_metrics is not None
            and convergence_metrics.convergence_time_s is not None
            else np.nan
        ),
    )

    metrics_path = results_dir / "metrics.txt"
    if convergence_metrics is None:
        steady_state_primary_text = "N/A (run diverged)"
        steady_state_residual_text = "N/A (run diverged)"
        steady_state_attenuation_text = "N/A (run diverged)"
        convergence_time_text = "N/A (run diverged)"
        steady_state_start_text = "N/A (run diverged)"
    else:
        steady_state_primary_text = (
            f"{convergence_metrics.steady_state_primary_rms:.8f}"
        )
        steady_state_residual_text = (
            f"{convergence_metrics.steady_state_residual_rms:.8f}"
        )
        steady_state_attenuation_text = (
            f"{convergence_metrics.steady_state_attenuation_db:.4f} dB"
        )

        steady_state_start_text = (
            f"{convergence_metrics.steady_state_start_sample}"
        )

        if convergence_metrics.convergence_time_s is None:
            convergence_time_text = "Not reached within experiment duration"
        else:
            convergence_time_text = (
                f"{convergence_metrics.convergence_time_s:.3f} s"
            )

    metrics_text = (
        f"Experiment: {experiment_cfg['name']}\n"
        f"Sample rate: {sample_rate_hz} Hz\n"
        f"Duration: {duration_s} s\n"
        f"Controller length: {controller.filter_length} taps\n"
        f"FxLMS step size: {controller.step_size}\n"
        f"Evaluation transient: {transient_s} s\n"
        f"Run status: {trace.status}\n"
        f"Processed samples: {trace.processed_samples} / {reference.size}\n"
        f"Divergence sample: {trace.divergence_sample}\n"
        f"Divergence reason: {trace.divergence_reason or 'None'}\n"
        f"Primary-noise RMS: {primary_rms:.8f}\n"
        f"Residual RMS: {residual_rms:.8f}\n"
        f"Evaluation attenuation: {anc_attenuation_db:.4f} dB\n"
        f"Steady-state start sample: {steady_state_start_text}\n"
        f"Steady-state primary RMS: {steady_state_primary_text}\n"
        f"Steady-state residual RMS: {steady_state_residual_text}\n"
        f"Steady-state attenuation: {steady_state_attenuation_text}\n"
        f"Convergence time: {convergence_time_text}\n"
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")

    save_figures = bool(output_cfg.get("save_figures", True))
    show_figures = bool(output_cfg.get("show_figures", False))

    figure_dir = (
        resolve_path(output_cfg["figure_directory"])
        if save_figures
        else None
    )

    figure_paths = plot_fxlms_results(
        trace=trace,
        sample_rate_hz=sample_rate_hz,
        output_dir=figure_dir,
        show=show_figures,
    )

    print(f"Experiment: {experiment_cfg['name']}")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Duration: {duration_s} s")
    print(f"Controller length: {controller.filter_length} taps")
    print(f"FxLMS step size: {controller.step_size}")
    print(f"Evaluation transient: {transient_s} s")
    print(f"Primary-noise RMS: {primary_rms:.8f}")
    print(f"Residual RMS: {residual_rms:.8f}")
    print(f"ANC attenuation: {anc_attenuation_db:.4f} dB")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Traces saved to: {traces_path}")

    for path in figure_paths:
        print(f"Figure saved to: {path}")

    print(f"Run status: {trace.status}")
    print(
        f"Processed samples: "
        f"{trace.processed_samples} / {reference.size}"
    )
    print(f"Divergence sample: {trace.divergence_sample}")

    if trace.divergence_reason:
        print(f"Divergence reason: {trace.divergence_reason}")

    if convergence_metrics is not None:
        print(
            "Steady-state primary RMS: "
            f"{convergence_metrics.steady_state_primary_rms:.8f}"
        )
        print(
            "Steady-state residual RMS: "
            f"{convergence_metrics.steady_state_residual_rms:.8f}"
        )
        print(
            "Steady-state attenuation: "
            f"{convergence_metrics.steady_state_attenuation_db:.4f} dB"
        )

        if convergence_metrics.convergence_time_s is None:
            print("Convergence time: Not reached")
        else:
            print(
                "Convergence time: "
                f"{convergence_metrics.convergence_time_s:.3f} s"
            )
    else:
        print("Steady-state metrics: Not available (run diverged)")


if __name__ == "__main__":
    main()