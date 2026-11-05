from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from anc.adaptive import run_lms
from anc.controllers import LMSController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "step2_lms_step_size_sweep.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "step2_lms_step_size_sweep"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / "step2_lms_step_size_sweep"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
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
    )


def make_result_row(
    seed: int,
    step_size: float,
    trace,
    desired: np.ndarray,
    sample_rate_hz: int,
    transient_samples: int,
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

    if trace.status == "completed" and processed > transient_samples:
        evaluation_desired = desired[transient_samples:processed]
        evaluation_error = trace.error[transient_samples:processed]

        evaluation_error_rms = rms(evaluation_error)
        evaluation_reduction = error_reduction_db(
            evaluation_desired,
            evaluation_error,
        )

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


def save_plots(summary_rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

    completion_path = FIGURES_DIR / "step2_lms_completion_rate.png"
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
        )

    ax.set_xscale("log")
    ax.set_xlabel("LMS step size (μ)")
    ax.set_ylabel("Prediction-error reduction (dB)")
    ax.set_title("LMS prediction performance for completed runs")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    performance_path = FIGURES_DIR / "step2_lms_prediction_performance.png"
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
            "guarantee good prediction performance.\n\n"
        )

        file.write(
            "μ | Completed | Diverged | Completion | "
            "Mean reduction (dB) | Mean divergence time (s)\n"
        )
        file.write("-" * 95 + "\n")

        for row in summary_rows:
            reduction = row["mean_evaluation_reduction_db"]
            reduction_text = (
                f"{reduction:.4f}" if reduction is not None else "N/A"
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
                f"{time_text}\n"
            )


def main() -> None:
    config = load_config()

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print("Experiment: step2_lms_step_size_sweep")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Duration: {duration_s:.2f} s")
    print(f"Filter length: {filter_length} taps")
    print(f"Seeds: {seeds}")
    print(f"Step sizes: {step_sizes}")
    print()

    for seed in seeds:
        reference = generate_reference(
            sample_count=sample_count,
            standard_deviation=noise_std,
            seed=seed,
        )
        desired = apply_primary_path(reference, primary_taps)

        for step_size in step_sizes:
            trace = run_one(
                reference=reference,
                desired=desired,
                filter_length=filter_length,
                step_size=step_size,
                max_weight_norm=max_weight_norm,
                max_abs_error=max_abs_error,
            )

            row = make_result_row(
                seed=seed,
                step_size=step_size,
                trace=trace,
                desired=desired,
                sample_rate_hz=sample_rate_hz,
                transient_samples=transient_samples,
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

    per_seed_path = RESULTS_DIR / "per_seed_results.csv"
    summary_csv_path = RESULTS_DIR / "step_size_summary.csv"
    summary_path = RESULTS_DIR / "summary.txt"

    save_csv(per_seed_path, rows)
    save_csv(summary_csv_path, summary_rows)
    save_summary(summary_path, config, summary_rows)
    save_plots(summary_rows)

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