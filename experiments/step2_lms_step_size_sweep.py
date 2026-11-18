from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from anc.adaptive import run_lms
from anc.controllers import LMSController
from anc.measurement import floor_report, make_measurement_noise
from plotting import draw_measurement_floor, mean_finite_floor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/step2_lms_step_size_sweep.yaml"


def resolve_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    return config


def generate_reference(
    sample_count: int,
    standard_deviation: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(
        0.0,
        standard_deviation,
        sample_count,
    ).astype(np.float64)


def apply_primary_path(
    reference: np.ndarray,
    taps: np.ndarray,
) -> np.ndarray:
    return np.convolve(reference, taps, mode="full")[: reference.size]


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    finite = signal[np.isfinite(signal)]

    if finite.size == 0:
        return float("nan")

    return float(np.sqrt(np.mean(finite**2)))


def error_reduction_db(
    desired: np.ndarray,
    error: np.ndarray,
) -> float | None:
    desired_rms = rms(desired)
    error_rms = rms(error)

    if (
        not np.isfinite(desired_rms)
        or not np.isfinite(error_rms)
        or desired_rms <= 0.0
        or error_rms <= 0.0
    ):
        return None

    return float(20.0 * np.log10(desired_rms / error_rms))


def get_max_weight_norm(trace) -> float:
    weights = np.asarray(trace.weights, dtype=np.float64)

    if weights.ndim != 2:
        raise ValueError("LMS weight history must be two-dimensional.")

    valid_rows = np.all(np.isfinite(weights), axis=1)

    if not np.any(valid_rows):
        return float("nan")

    norms = np.linalg.norm(weights[valid_rows], axis=1)
    return float(np.max(norms))


def run_one(
    reference: np.ndarray,
    desired: np.ndarray,
    filter_length: int,
    step_size: float,
    max_weight_norm: float,
    max_abs_error: float,
    measurement_noise: np.ndarray | None = None,
):
    controller = LMSController(
        filter_length=filter_length,
        step_size=step_size,
    )

    return run_lms(
        reference=reference,
        desired=desired,
        controller=controller,
        max_weight_norm=max_weight_norm,
        max_abs_error=max_abs_error,
        measurement_noise=measurement_noise,
    )


def make_result_row(
    seed: int,
    step_size: float,
    trace,
    desired: np.ndarray,
    sample_rate_hz: int,
    transient_samples: int,
    snr_db: float | None,
    floor_margin_db: float,
) -> dict:
    processed = min(trace.processed_samples, desired.size)
    valid_error = trace.error[:processed]

    peak_abs_error = (
        float(np.max(np.abs(valid_error)))
        if valid_error.size
        else float("nan")
    )

    evaluation_error_rms = None
    evaluation_reduction = None
    measurement_floor = None
    evaluation_measured_reduction = None
    evaluation_floor_limited = None

    if trace.status == "completed" and processed > transient_samples:
        evaluation_desired = desired[transient_samples:processed]
        evaluation_error = trace.error[transient_samples:processed]

        evaluation_error_rms = rms(evaluation_error)
        evaluation_reduction = error_reduction_db(
            evaluation_desired,
            evaluation_error,
        )

        report = floor_report(
            desired,
            trace.error,
            trace.measurement_noise,
            start_sample=transient_samples,
            stop_sample=processed,
            margin_db=floor_margin_db,
        )
        measurement_floor = report.floor_db
        evaluation_measured_reduction = report.measured_attenuation_db
        evaluation_floor_limited = report.floor_limited

    divergence_time_s = (
        trace.divergence_sample / sample_rate_hz
        if trace.divergence_sample is not None
        else None
    )

    return {
        "seed": seed,
        "step_size": step_size,
        "status": trace.status,
        "processed_samples": processed,
        "divergence_sample": trace.divergence_sample,
        "divergence_time_s": divergence_time_s,
        "divergence_reason": trace.divergence_reason,
        "max_weight_norm": get_max_weight_norm(trace),
        "peak_abs_error": peak_abs_error,
        "evaluation_error_rms": evaluation_error_rms,
        "evaluation_error_reduction_db": evaluation_reduction,
        "measurement_snr_db": snr_db,
        "measurement_floor_db": measurement_floor,
        "evaluation_measured_reduction_db": evaluation_measured_reduction,
        "evaluation_floor_limited": evaluation_floor_limited,
    }


def aggregate_results(
    rows: list[dict],
    step_sizes: list[float],
    seed_count: int,
) -> list[dict]:
    summary_rows = []

    for step_size in step_sizes:
        matching = [
            row for row in rows
            if row["step_size"] == step_size
        ]

        completed = [
            row for row in matching
            if row["status"] == "completed"
        ]
        diverged = [
            row for row in matching
            if row["status"] == "diverged"
        ]

        reductions = [
            row["evaluation_error_reduction_db"]
            for row in completed
            if row["evaluation_error_reduction_db"] is not None
        ]

        divergence_times = [
            row["divergence_time_s"]
            for row in diverged
            if row["divergence_time_s"] is not None
        ]

        measured_reductions = [
            row["evaluation_measured_reduction_db"]
            for row in completed
            if row["evaluation_measured_reduction_db"] is not None
        ]
        floor_limited_runs = sum(
            row["evaluation_floor_limited"] is True for row in completed
        )

        finite_norms = [
            row["max_weight_norm"]
            for row in matching
            if np.isfinite(row["max_weight_norm"])
        ]

        summary_rows.append(
            {
                "step_size": step_size,
                "completed_runs": len(completed),
                "diverged_runs": len(diverged),
                "completion_fraction": len(completed) / seed_count,
                "mean_evaluation_reduction_db": (
                    float(np.mean(reductions)) if reductions else None
                ),
                "std_evaluation_reduction_db": (
                    float(np.std(reductions)) if reductions else None
                ),
                "mean_measured_reduction_db": (
                    float(np.mean(measured_reductions))
                    if measured_reductions
                    else None
                ),
                "std_measured_reduction_db": (
                    float(np.std(measured_reductions))
                    if measured_reductions
                    else None
                ),
                "floor_limited_runs": floor_limited_runs,
                "mean_divergence_time_s": (
                    float(np.mean(divergence_times))
                    if divergence_times
                    else None
                ),
                "mean_max_weight_norm": (
                    float(np.mean(finite_norms)) if finite_norms else None
                ),
            }
        )

    return summary_rows


def save_plots(
    summary_rows: list[dict],
    rows: list[dict],
    figures_dir: Path,
    snr_db: float | None,
    floor_margin_db: float,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    step_sizes = np.array(
        [row["step_size"] for row in summary_rows],
        dtype=np.float64,
    )
    completion = np.array(
        [row["completion_fraction"] for row in summary_rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step_sizes, completion, marker="o")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LMS step size (μ)")
    ax.set_ylabel("Fraction of runs completed")
    ax.set_title("LMS completion rate versus step size")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    completion_path = figures_dir / "step2_lms_completion_rate.png"
    fig.savefig(completion_path, dpi=150)
    plt.close(fig)

    valid_rows = [
        row for row in summary_rows
        if row["mean_evaluation_reduction_db"] is not None
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    if valid_rows:
        x = np.array(
            [row["step_size"] for row in valid_rows],
            dtype=np.float64,
        )
        means = np.array(
            [row["mean_evaluation_reduction_db"] for row in valid_rows],
            dtype=np.float64,
        )
        stds = np.array(
            [row["std_evaluation_reduction_db"] for row in valid_rows],
            dtype=np.float64,
        )

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label="True reduction (simulation-only)",
        )

        if snr_db is not None:
            measured = np.array(
                [row["mean_measured_reduction_db"] for row in valid_rows],
                dtype=np.float64,
            )
            measured_stds = np.array(
                [row["std_measured_reduction_db"] for row in valid_rows],
                dtype=np.float64,
            )
            limited = np.array(
                [
                    row["floor_limited_runs"] == row["completed_runs"]
                    for row in valid_rows
                ],
                dtype=bool,
            )

            ax.errorbar(
                x,
                measured,
                yerr=measured_stds,
                linestyle="--",
                marker="none",
                capsize=4,
                color="C1",
                label="Measured reduction",
            )
            ax.plot(x[~limited], measured[~limited], "o", color="C1")
            ax.plot(
                x[limited],
                measured[limited],
                "o",
                color="C1",
                markerfacecolor="white",
                label="Measured, floor-limited",
            )

    draw_measurement_floor(
        ax,
        mean_finite_floor(rows),
        margin_db=floor_margin_db,
    )

    ax.set_xscale("log")
    ax.set_xlabel("LMS step size (μ)")
    ax.set_ylabel("Prediction-error reduction (dB)")
    ax.set_title(
        "LMS prediction performance for completed runs\n"
        + ("Idealised noise-free observation" if snr_db is None
           else f"Observation SNR {snr_db:g} dB")
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    performance_path = figures_dir / "step2_lms_prediction_performance.png"
    fig.savefig(performance_path, dpi=150)
    plt.close(fig)

    print(f"Figure saved to: {completion_path}")
    print(f"Figure saved to: {performance_path}")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty result set.")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    path: Path,
    config: dict,
    summary_rows: list[dict],
    snr_db: float | None,
    floor_margin_db: float,
) -> None:
    experiment = config["experiment"]
    controller = config["controller"]
    safety = config["safety"]

    with path.open("w", encoding="utf-8") as file:
        file.write("LMS step-size sweep\n")
        file.write("===================\n\n")
        file.write(f"Sample rate: {experiment['sample_rate_hz']} Hz\n")
        file.write(f"Duration: {experiment['duration_s']} s\n")
        file.write(f"Seeds: {experiment['seeds']}\n")
        file.write(f"Filter length: {controller['filter_length']} taps\n")
        file.write(
            f"Safety limits: weight norm={safety['max_weight_norm']}, "
            f"absolute error={safety['max_abs_error']}\n\n"
        )

        file.write(
            "A completed run processed the full signal without crossing "
            "either configured safety limit. Completion does not, by itself, "
            "guarantee good prediction performance. Mean reductions are "
            "averaged over completed runs only (survivor-only).\n\n"
        )

        if snr_db is None:
            file.write(
                "Measurement noise: none (idealised observation). Reductions "
                "are not bounded by a sensor floor; very high values reflect "
                "numerical precision, not achievable performance.\n\n"
            )
        else:
            file.write(
                f"Measurement noise: SNR {snr_db:g} dB re. the desired "
                f"signal. Measured values within {floor_margin_db:g} dB of "
                f"the floor are marked floor-limited.\n\n"
            )

        file.write(
            "μ | Completed | Diverged | Completion | "
            "Mean reduction (dB) | Measured (dB) | Floor-limited | "
            "Mean divergence time (s)\n"
        )
        file.write("-" * 120 + "\n")

        for row in summary_rows:
            reduction = row["mean_evaluation_reduction_db"]
            reduction_text = (
                f"{reduction:.4f}" if reduction is not None else "N/A"
            )

            measured = row["mean_measured_reduction_db"]
            measured_text = (
                f"{measured:.4f}" if measured is not None else "N/A"
            )

            divergence_time = row["mean_divergence_time_s"]
            time_text = (
                f"{divergence_time:.4f}"
                if divergence_time is not None
                else "N/A"
            )

            file.write(
                f"{row['step_size']:.4f} | "
                f"{row['completed_runs']} | "
                f"{row['diverged_runs']} | "
                f"{row['completion_fraction']:.2f} | "
                f"{reduction_text} | "
                f"{measured_text} | "
                f"{row['floor_limited_runs']}/{row['completed_runs']} | "
                f"{time_text}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the LMS step size over several seeds."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the step-size sweep YAML configuration.",
    )
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))

    # Output locations follow the experiment name, so every config
    # writes to its own folders. The default name reproduces the
    # original paths exactly.
    experiment_name = str(
        config["experiment"].get("name", "step2_lms_step_size_sweep")
    )

    config_path = resolve_path(args.config)
    if experiment_name != config_path.stem:
        raise ValueError(
            f"experiment.name '{experiment_name}' does not match the config "
            f"file name '{config_path.stem}'. Results would be written to "
            f"the wrong folder."
        )
    results_dir = PROJECT_ROOT / "results" / experiment_name
    figures_dir = PROJECT_ROOT / "docs" / "figures" / experiment_name

    sample_rate_hz = int(config["experiment"]["sample_rate_hz"])
    duration_s = float(config["experiment"]["duration_s"])
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]

    sample_count = int(round(sample_rate_hz * duration_s))
    filter_length = int(config["controller"]["filter_length"])
    step_sizes = [
        float(value) for value in config["controller"]["step_sizes"]
    ]

    noise_std = float(config["signal"]["standard_deviation"])
    primary_taps = np.asarray(
        config["plant"]["primary_taps"],
        dtype=np.float64,
    )

    noise_cfg = config.get("measurement_noise") or {}
    snr_db = noise_cfg.get("snr_db")
    snr_db = None if snr_db is None else float(snr_db)
    floor_margin_db = float(noise_cfg.get("floor_margin_db", 3.0))

    max_weight_norm = float(config["safety"]["max_weight_norm"])
    max_abs_error = float(config["safety"]["max_abs_error"])

    transient_samples = int(
        round(
            float(config["evaluation"]["transient_duration_s"])
            * sample_rate_hz
        )
    )

    if not seeds:
        raise ValueError("At least one seed is required.")
    if not step_sizes or any(
        not np.isfinite(value) or value <= 0.0 for value in step_sizes
    ):
        raise ValueError("All step sizes must be finite and positive.")
    if len(set(step_sizes)) != len(step_sizes):
        raise ValueError("Step sizes must be unique.")
    if sample_count <= transient_samples:
        raise ValueError("Evaluation transient must be shorter than the run.")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"Experiment: {experiment_name}")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Duration: {duration_s:.2f} s")
    print(f"Filter length: {filter_length} taps")
    print(f"Seeds: {seeds}")
    print(f"Step sizes: {step_sizes}")
    print(f"Measurement SNR: {'none' if snr_db is None else f'{snr_db:g} dB'}")
    print()

    for seed in seeds:
        reference = generate_reference(
            sample_count=sample_count,
            standard_deviation=noise_std,
            seed=seed,
        )
        desired = apply_primary_path(reference, primary_taps)

        # One noise realisation per seed, shared by every step size
        # (common random numbers across conditions).
        measurement_noise = make_measurement_noise(
            desired, snr_db, seed=seed
        )

        for step_size in step_sizes:
            trace = run_one(
                reference=reference,
                desired=desired,
                filter_length=filter_length,
                step_size=step_size,
                max_weight_norm=max_weight_norm,
                max_abs_error=max_abs_error,
                measurement_noise=measurement_noise,
            )

            # Guard: a configured SNR must produce non-zero noise.
            if snr_db is not None and not np.any(trace.measurement_noise):
                raise RuntimeError(
                    "measurement_noise.snr_db is set, but the run used "
                    "zero measurement noise. Is measurement_noise passed "
                    "to run_lms?"
                )

            row = make_result_row(
                seed=seed,
                step_size=step_size,
                trace=trace,
                desired=desired,
                sample_rate_hz=sample_rate_hz,
                transient_samples=transient_samples,
                snr_db=snr_db,
                floor_margin_db=floor_margin_db,
            )
            rows.append(row)

            divergence_text = (
                f"sample {trace.divergence_sample} "
                f"({row['divergence_time_s']:.4f} s)"
                if trace.divergence_sample is not None
                else "none"
            )

            print(
                f"Seed {seed}, μ={step_size:g}: "
                f"status={trace.status}, "
                f"processed={trace.processed_samples}/{sample_count}, "
                f"divergence={divergence_text}, "
                f"reason={trace.divergence_reason}"
            )

        print()

    summary_rows = aggregate_results(
        rows=rows,
        step_sizes=step_sizes,
        seed_count=len(seeds),
    )

    per_seed_path = results_dir / "per_seed_results.csv"
    summary_csv_path = results_dir / "step_size_summary.csv"
    summary_path = results_dir / "summary.txt"

    save_csv(per_seed_path, rows)
    save_csv(summary_csv_path, summary_rows)
    save_summary(summary_path, config, summary_rows, snr_db, floor_margin_db)
    save_plots(summary_rows, rows, figures_dir, snr_db, floor_margin_db)

    print("\nSweep summary")
    print("-------------")

    for row in summary_rows:
        reduction = row["mean_evaluation_reduction_db"]
        reduction_text = (
            f"{reduction:.2f} dB" if reduction is not None else "N/A"
        )

        print(
            f"μ={row['step_size']:g}: "
            f"{row['completed_runs']}/{len(seeds)} completed, "
            f"{row['diverged_runs']}/{len(seeds)} diverged, "
            f"mean reduction={reduction_text}"
        )

    print(f"\nPer-seed results: {per_seed_path}")
    print(f"Step-size summary: {summary_csv_path}")
    print(f"Text summary: {summary_path}")


if __name__ == "__main__":
    main()