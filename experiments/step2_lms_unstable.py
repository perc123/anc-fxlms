from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from anc.adaptive import run_lms
from anc.controllers import LMSController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "step2_lms_unstable.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "step2_lms_unstable"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / "step2_lms_unstable"


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
    return rng.normal(0.0, standard_deviation, sample_count).astype(np.float64)


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


def prediction_error_reduction_db(
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


def get_weight_norms(trace) -> np.ndarray:
    weights = np.asarray(trace.weights, dtype=np.float64)

    if weights.ndim != 2:
        raise ValueError("LMS weight history must be a two-dimensional array.")

    valid_rows = np.all(np.isfinite(weights), axis=1)
    norms = np.full(weights.shape[0], np.nan, dtype=np.float64)
    norms[valid_rows] = np.linalg.norm(weights[valid_rows], axis=1)

    return norms


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
    variant: str,
    seed: int,
    step_size: float,
    trace,
    desired: np.ndarray,
    sample_rate_hz: int,
    transient_samples: int,
) -> dict:
    processed = min(trace.processed_samples, desired.size)
    valid_desired = desired[:processed]
    valid_error = trace.error[:processed]

    pre_divergence_reduction = prediction_error_reduction_db(
        valid_desired,
        valid_error,
    )

    evaluation_reduction = None
    evaluation_error_rms = None

    if trace.status == "completed" and processed > transient_samples:
        evaluation_desired = desired[transient_samples:processed]
        evaluation_error = trace.error[transient_samples:processed]

        evaluation_reduction = prediction_error_reduction_db(
            evaluation_desired,
            evaluation_error,
        )
        evaluation_error_rms = rms(evaluation_error)

    weight_norms = get_weight_norms(trace)
    finite_norms = weight_norms[np.isfinite(weight_norms)]
    max_observed_weight_norm = (
        float(np.max(finite_norms)) if finite_norms.size else float("nan")
    )

    divergence_time_s = (
        trace.divergence_sample / sample_rate_hz
        if trace.divergence_sample is not None
        else None
    )

    return {
        "variant": variant,
        "seed": seed,
        "step_size": step_size,
        "status": trace.status,
        "processed_samples": processed,
        "divergence_sample": trace.divergence_sample,
        "divergence_time_s": divergence_time_s,
        "divergence_reason": trace.divergence_reason,
        "max_observed_weight_norm": max_observed_weight_norm,
        "evaluation_error_rms": evaluation_error_rms,
        "evaluation_error_reduction_db": evaluation_reduction,
        "pre_divergence_error_reduction_db": pre_divergence_reduction,
    }


def plot_comparison(
    traces: dict,
    sample_rate_hz: int,
    seed: int,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    stable_trace = traces["stable"]
    excessive_trace = traces["excessive"]

    sample_count = min(
        stable_trace.error.size,
        excessive_trace.error.size,
        int(0.5 * sample_rate_hz),
    )
    time_s = np.arange(sample_count) / sample_rate_hz

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(
        time_s,
        stable_trace.error[:sample_count],
        label="Stable step size",
    )
    axes[0].plot(
        time_s,
        excessive_trace.error[:sample_count],
        label="Excessive step size",
    )
    axes[0].set_title("LMS prediction error")
    axes[0].set_ylabel("Error amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    stable_norms = get_weight_norms(stable_trace)[:sample_count]
    excessive_norms = get_weight_norms(excessive_trace)[:sample_count]

    axes[1].plot(time_s, stable_norms, label="Stable step size")
    axes[1].plot(time_s, excessive_norms, label="Excessive step size")
    axes[1].set_title("LMS coefficient-vector norm")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Weight-vector norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"LMS step-size comparison — seed {seed}")
    fig.tight_layout()

    output_path = FIGURES_DIR / f"step2_lms_unstable_seed_{seed}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


def main() -> None:
    config = load_config()

    sample_rate_hz = int(config["experiment"]["sample_rate_hz"])
    duration_s = float(config["experiment"]["duration_s"])
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]

    sample_count = int(round(sample_rate_hz * duration_s))
    filter_length = int(config["controller"]["filter_length"])

    stable_step_size = float(config["controller"]["stable_step_size"])
    excessive_step_size = float(config["controller"]["excessive_step_size"])

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print("Experiment: step2_lms_unstable")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Duration: {duration_s:.2f} s")
    print(f"Controller length: {filter_length} taps")
    print(f"Stable step size: {stable_step_size}")
    print(f"Excessive step size: {excessive_step_size}")
    print(f"Safety limits: weight norm={max_weight_norm}, error={max_abs_error}")
    print()

    for seed_index, seed in enumerate(seeds):
        reference = generate_reference(
            sample_count=sample_count,
            standard_deviation=noise_std,
            seed=seed,
        )
        desired = apply_primary_path(reference, primary_taps)

        traces = {
            "stable": run_one(
                reference=reference,
                desired=desired,
                filter_length=filter_length,
                step_size=stable_step_size,
                max_weight_norm=max_weight_norm,
                max_abs_error=max_abs_error,
            ),
            "excessive": run_one(
                reference=reference,
                desired=desired,
                filter_length=filter_length,
                step_size=excessive_step_size,
                max_weight_norm=max_weight_norm,
                max_abs_error=max_abs_error,
            ),
        }

        rows.append(
            make_result_row(
                variant="stable",
                seed=seed,
                step_size=stable_step_size,
                trace=traces["stable"],
                desired=desired,
                sample_rate_hz=sample_rate_hz,
                transient_samples=transient_samples,
            )
        )
        rows.append(
            make_result_row(
                variant="excessive",
                seed=seed,
                step_size=excessive_step_size,
                trace=traces["excessive"],
                desired=desired,
                sample_rate_hz=sample_rate_hz,
                transient_samples=transient_samples,
            )
        )

        for variant in ("stable", "excessive"):
            trace = traces[variant]
            row = rows[-2] if variant == "stable" else rows[-1]

            divergence_text = (
                f"sample {trace.divergence_sample} "
                f"({row['divergence_time_s']:.4f} s)"
                if trace.divergence_sample is not None
                else "none"
            )

            print(
                f"Seed {seed}, {variant}: "
                f"status={trace.status}, "
                f"processed={trace.processed_samples}/{sample_count}, "
                f"divergence={divergence_text}, "
                f"reason={trace.divergence_reason}"
            )

        if seed_index == 0:
            plot_comparison(
                traces=traces,
                sample_rate_hz=sample_rate_hz,
                seed=seed,
            )

        print()

    csv_path = RESULTS_DIR / "per_seed_results.csv"
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS_DIR / "summary.txt"

    stable_rows = [row for row in rows if row["variant"] == "stable"]
    excessive_rows = [row for row in rows if row["variant"] == "excessive"]

    stable_reductions = [
        row["evaluation_error_reduction_db"]
        for row in stable_rows
        if row["evaluation_error_reduction_db"] is not None
    ]

    divergence_times = [
        row["divergence_time_s"]
        for row in excessive_rows
        if row["divergence_time_s"] is not None
    ]

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("LMS step-size instability experiment\n")
        file.write("====================================\n\n")
        file.write(f"Seeds: {seeds}\n")
        file.write(f"Stable step size: {stable_step_size}\n")
        file.write(f"Excessive step size: {excessive_step_size}\n")
        file.write(
            f"Safety limits: weight norm={max_weight_norm}, "
            f"absolute error={max_abs_error}\n\n"
        )

        if stable_reductions:
            file.write(
                "Stable-step evaluation prediction-error reduction: "
                f"{np.mean(stable_reductions):.4f} +/- "
                f"{np.std(stable_reductions):.4f} dB\n"
            )

        diverged_count = sum(
            row["status"] == "diverged" for row in excessive_rows
        )
        file.write(
            f"Excessive-step runs diverged: "
            f"{diverged_count}/{len(excessive_rows)}\n"
        )

        if divergence_times:
            file.write(
                "Mean detected divergence time: "
                f"{np.mean(divergence_times):.4f} s\n"
            )

        file.write("\nPer-run outcomes:\n")
        for row in excessive_rows:
            file.write(
                f"  Seed {row['seed']}: "
                f"{row['status']}, "
                f"sample={row['divergence_sample']}, "
                f"time={row['divergence_time_s']}, "
                f"reason={row['divergence_reason']}\n"
            )

    print("Summary")
    print("-------")

    if stable_reductions:
        print(
            "Stable-step evaluation error reduction: "
            f"{np.mean(stable_reductions):.4f} +/- "
            f"{np.std(stable_reductions):.4f} dB"
        )

    diverged_count = sum(
        row["status"] == "diverged" for row in excessive_rows
    )
    print(
        f"Excessive-step runs diverged: "
        f"{diverged_count}/{len(excessive_rows)}"
    )

    if divergence_times:
        print(
            "Mean detected divergence time: "
            f"{np.mean(divergence_times):.4f} s"
        )

    print(f"\nPer-seed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()