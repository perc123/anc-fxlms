from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from anc.convergence import analyze_convergence
from anc.controllers import FxLMSController, design_fixed_fir
from anc.fxlms import run_fxlms
from anc.metrics import attenuation_db, rms
from anc.plant import ANCPlant, apply_fir
from anc.signals import generate_tone_mixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    "no_control",
    "fixed_fir",
    "fxlms_matched",
    "fxlms_mismatched",
]

METHOD_LABELS = {
    "no_control": "No control",
    "fixed_fir": "Fixed FIR",
    "fxlms_matched": "FxLMS matched",
    "fxlms_mismatched": "FxLMS mismatched",
}


def resolve_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def generate_reference(
    config: dict,
    seed: int,
    duration_s: float,
) -> np.ndarray:
    signal_cfg = config["signal"]

    tones = [
        (
            float(item["frequency_hz"]),
            float(item["amplitude"]),
        )
        for item in signal_cfg["tones"]
    ]

    return generate_tone_mixture(
        sample_rate_hz=int(signal_cfg["sample_rate_hz"]),
        duration_s=duration_s,
        tones=tones,
        white_noise_std=float(
            signal_cfg.get("white_noise_std", 0.0)
        ),
        seed=seed,
    )


def make_plant(config: dict) -> ANCPlant:
    plant_cfg = config["plant"]

    return ANCPlant(
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


def evaluate_method(
    seed: int,
    method: str,
    primary: np.ndarray,
    residual: np.ndarray,
    sample_rate_hz: int,
    start_sample: int,
    processed_samples: int,
    status: str,
    adaptive: bool,
    benchmark_cfg: dict,
) -> dict:
    """Calculate comparable performance metrics for one method."""

    row = {
        "seed": seed,
        "method": method,
        "status": status,
        "processed_samples": processed_samples,
        "evaluation_primary_rms": None,
        "evaluation_residual_rms": None,
        "evaluation_attenuation_db": None,
        "steady_state_primary_rms": None,
        "steady_state_residual_rms": None,
        "steady_state_attenuation_db": None,
        "convergence_time_s": None,
        "convergence_status": "not_applicable",
    }

    # Do not treat incomplete or diverged runs as valid
    # full-duration performance measurements.
    if status != "completed" or processed_samples != primary.size:
        row["convergence_status"] = (
            "unavailable_diverged"
            if status == "diverged"
            else "unavailable_incomplete"
        )
        return row

    primary_valid = primary[:processed_samples]
    residual_valid = residual[:processed_samples]

    if processed_samples <= start_sample:
        row["convergence_status"] = "unavailable"
        return row

    # Overall evaluation-window measurements.
    row["evaluation_primary_rms"] = rms(
        primary_valid[start_sample:]
    )
    row["evaluation_residual_rms"] = rms(
        residual_valid[start_sample:]
    )

    row["evaluation_attenuation_db"] = attenuation_db(
        primary_valid,
        residual_valid,
        start_sample=start_sample,
    )

    # Steady-state and block convergence measurements.
    metrics = analyze_convergence(
        primary_noise=primary_valid,
        residual=residual_valid,
        sample_rate_hz=sample_rate_hz,
        start_sample=start_sample,
        block_duration_s=float(
            benchmark_cfg.get("block_duration_s", 0.1)
        ),
        steady_state_fraction=float(
            benchmark_cfg.get("steady_state_fraction", 0.2)
        ),
        tolerance_db=float(
            benchmark_cfg.get("convergence_tolerance_db", 1.0)
        ),
        consecutive_blocks=int(
            benchmark_cfg.get("consecutive_blocks", 3)
        ),
    )

    row["steady_state_primary_rms"] = (
        metrics.steady_state_primary_rms
    )
    row["steady_state_residual_rms"] = (
        metrics.steady_state_residual_rms
    )
    row["steady_state_attenuation_db"] = (
        metrics.steady_state_attenuation_db
    )

    # Static methods have no adaptive convergence time.
    if adaptive:
        row["convergence_time_s"] = metrics.convergence_time_s

        row["convergence_status"] = (
            "detected"
            if metrics.convergence_time_s is not None
            else "not_detected"
        )

    return row


def metric_statistics(
    rows: list[dict],
    method: str,
    metric_name: str,
) -> tuple[float, float, int]:
    """Return mean, sample standard deviation, and valid count."""

    values = [
        float(row[metric_name])
        for row in rows
        if row["method"] == method
        and row[metric_name] is not None
        and np.isfinite(float(row[metric_name]))
    ]

    if not values:
        return float("nan"), float("nan"), 0

    mean = float(np.mean(values))
    std = (
        float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0
    )

    return mean, std, len(values)


def plot_benchmark(
    rows: list[dict],
    figure_dir: Path,
) -> list[Path]:
    """Create attenuation and convergence-time comparison figures."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # ---------------------------------------------------------
    # Figure 1: overall and steady-state attenuation.
    # ---------------------------------------------------------
    x = np.arange(len(METHODS))
    width = 0.36

    evaluation_means = []
    evaluation_stds = []
    steady_means = []
    steady_stds = []

    for method in METHODS:
        mean, std, _ = metric_statistics(
            rows,
            method,
            "evaluation_attenuation_db",
        )
        evaluation_means.append(mean)
        evaluation_stds.append(std)

        mean, std, _ = metric_statistics(
            rows,
            method,
            "steady_state_attenuation_db",
        )
        steady_means.append(mean)
        steady_stds.append(std)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - width / 2,
        evaluation_means,
        width,
        yerr=evaluation_stds,
        capsize=4,
        label="Evaluation attenuation",
    )

    ax.bar(
        x + width / 2,
        steady_means,
        width,
        yerr=steady_stds,
        capsize=4,
        label="Steady-state attenuation",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in METHODS]
    )
    ax.set_ylabel("Attenuation (dB)")
    ax.set_title(
        "ANC method comparison on held-out test signals"
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    attenuation_path = figure_dir / "step4_benchmark.png"
    fig.savefig(
        attenuation_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    saved_paths.append(attenuation_path)

    # ---------------------------------------------------------
    # Figure 2: convergence time for adaptive methods only.
    # ---------------------------------------------------------
    adaptive_methods = [
        "fxlms_matched",
        "fxlms_mismatched",
    ]

    means = []
    stds = []
    counts = []

    for method in adaptive_methods:
        mean, std, count = metric_statistics(
            rows,
            method,
            "convergence_time_s",
        )
        means.append(mean)
        stds.append(std)
        counts.append(count)

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = np.arange(len(adaptive_methods))
    valid_positions = [
        i for i, count in enumerate(counts) if count > 0
    ]

    if valid_positions:
        ax.bar(
            [positions[i] for i in valid_positions],
            [means[i] for i in valid_positions],
            yerr=[stds[i] for i in valid_positions],
            capsize=5,
        )

    for i, count in enumerate(counts):
        if count == 0:
            ax.text(
                positions[i],
                0.02,
                "Not detected",
                ha="center",
                va="bottom",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in adaptive_methods]
    )
    ax.set_ylabel("Convergence time (s)")
    ax.set_title(
        "FxLMS convergence time across benchmark seeds"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    convergence_path = (
        figure_dir / "step4_convergence_time.png"
    )
    fig.savefig(
        convergence_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    saved_paths.append(convergence_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed FIR and FxLMS ANC methods."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark YAML configuration.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_cfg = config["experiment"]
    signal_cfg = config["signal"]
    benchmark_cfg = config["benchmark"]
    fixed_cfg = config["fixed_controller"]
    fxlms_cfg = config["fxlms"]
    mismatch_cfg = config["mismatch"]
    output_cfg = config["output"]
    safety_cfg = config.get("safety", {})

    sample_rate_hz = int(signal_cfg["sample_rate_hz"])
    training_duration_s = float(
        signal_cfg["training_duration_s"]
    )
    test_duration_s = float(signal_cfg["test_duration_s"])

    transient_s = float(benchmark_cfg["transient_s"])
    start_sample = int(round(
        transient_s * sample_rate_hz
    ))

    if transient_s < 0 or transient_s >= test_duration_s:
        raise ValueError(
            "transient_s must be non-negative and shorter "
            "than test_duration_s."
        )

    seeds = [int(seed) for seed in benchmark_cfg["seeds"]]

    if not seeds:
        raise ValueError("At least one benchmark seed is required.")

    plant = make_plant(config)

    mismatch_taps = np.asarray(
        mismatch_cfg["estimated_secondary_taps"],
        dtype=np.float64,
    )
    mismatch_delay = int(
        mismatch_cfg["estimated_secondary_delay_samples"]
    )

    if mismatch_taps.ndim != 1 or mismatch_taps.size == 0:
        raise ValueError("Mismatch taps must be a non-empty vector.")

    if not np.isfinite(mismatch_taps).all():
        raise ValueError("Mismatch taps must be finite.")

    if mismatch_delay < 0:
        raise ValueError("Mismatch delay must be non-negative.")

    rows: list[dict] = []

    for seed in seeds:
        # Independent training and test records.
        training_reference = generate_reference(
            config,
            seed=seed,
            duration_s=training_duration_s,
        )

        test_reference = generate_reference(
            config,
            seed=seed + 10_000,
            duration_s=test_duration_s,
        )

        training_primary = plant.primary_output(
            training_reference
        )
        test_primary = plant.primary_output(
            test_reference
        )

        n_samples = test_reference.size

        # -----------------------------------------------------
        # 1. No control.
        # -----------------------------------------------------
        no_control_residual = test_primary.copy()

        rows.append(
            evaluate_method(
                seed=seed,
                method="no_control",
                primary=test_primary,
                residual=no_control_residual,
                sample_rate_hz=sample_rate_hz,
                start_sample=start_sample,
                processed_samples=n_samples,
                status="completed",
                adaptive=False,
                benchmark_cfg=benchmark_cfg,
            )
        )

        # -----------------------------------------------------
        # 2. Fixed FIR: train offline, evaluate on test record.
        # -----------------------------------------------------
        fixed_controller = design_fixed_fir(
            reference=training_reference,
            desired_noise=training_primary,
            plant=plant,
            filter_length=int(fixed_cfg["length"]),
            regularization=float(
                fixed_cfg["regularization"]
            ),
        )

        fixed_control = apply_fir(
            test_reference,
            fixed_controller.taps,
        )

        fixed_anti_noise = plant.secondary_output(
            fixed_control
        )

        fixed_residual = test_primary + fixed_anti_noise

        rows.append(
            evaluate_method(
                seed=seed,
                method="fixed_fir",
                primary=test_primary,
                residual=fixed_residual,
                sample_rate_hz=sample_rate_hz,
                start_sample=start_sample,
                processed_samples=n_samples,
                status="completed",
                adaptive=False,
                benchmark_cfg=benchmark_cfg,
            )
        )

        # -----------------------------------------------------
        # 3. FxLMS with matched secondary-path estimate.
        # -----------------------------------------------------
        matched_controller = FxLMSController(
            filter_length=int(fxlms_cfg["length"]),
            step_size=float(fxlms_cfg["step_size"]),
            estimated_secondary_taps=plant.secondary_taps.copy(),
            estimated_secondary_delay_samples=(
                plant.secondary_delay_samples
            ),
        )

        matched_trace = run_fxlms(
            reference=test_reference,
            plant=plant,
            controller=matched_controller,
            max_weight_norm=safety_cfg.get(
                "max_weight_norm", 1e6
            ),
            max_abs_residual=safety_cfg.get(
                "max_abs_residual", 1e6
            ),
        )

        rows.append(
            evaluate_method(
                seed=seed,
                method="fxlms_matched",
                primary=matched_trace.primary_noise,
                residual=matched_trace.residual,
                sample_rate_hz=sample_rate_hz,
                start_sample=start_sample,
                processed_samples=matched_trace.processed_samples,
                status=matched_trace.status,
                adaptive=True,
                benchmark_cfg=benchmark_cfg,
            )
        )

        # -----------------------------------------------------
        # 4. FxLMS with mismatched secondary-path estimate.
        # -----------------------------------------------------
        mismatch_controller = FxLMSController(
            filter_length=int(fxlms_cfg["length"]),
            step_size=float(fxlms_cfg["step_size"]),
            estimated_secondary_taps=mismatch_taps.copy(),
            estimated_secondary_delay_samples=mismatch_delay,
        )

        mismatch_trace = run_fxlms(
            reference=test_reference,
            plant=plant,
            controller=mismatch_controller,
            max_weight_norm=safety_cfg.get(
                "max_weight_norm", 1e6
            ),
            max_abs_residual=safety_cfg.get(
                "max_abs_residual", 1e6
            ),
        )

        rows.append(
            evaluate_method(
                seed=seed,
                method="fxlms_mismatched",
                primary=mismatch_trace.primary_noise,
                residual=mismatch_trace.residual,
                sample_rate_hz=sample_rate_hz,
                start_sample=start_sample,
                processed_samples=mismatch_trace.processed_samples,
                status=mismatch_trace.status,
                adaptive=True,
                benchmark_cfg=benchmark_cfg,
            )
        )

        # Per-seed report.
        for row in rows[-4:]:
            attenuation = row["evaluation_attenuation_db"]
            steady_attenuation = row["steady_state_attenuation_db"]
            convergence_time = row["convergence_time_s"]

            attenuation_text = (
                "N/A"
                if attenuation is None
                else f"{attenuation:.2f} dB"
            )

            steady_text = (
                "N/A"
                if steady_attenuation is None
                else f"{steady_attenuation:.2f} dB"
            )

            convergence_text = (
                "N/A"
                if convergence_time is None
                else f"{convergence_time:.2f} s"
            )

            print(
                f"Seed {seed} | {row['method']}: "
                f"status={row['status']}, "
                f"evaluation={attenuation_text}, "
                f"steady-state={steady_text}, "
                f"convergence={convergence_text}"
            )

    # ---------------------------------------------------------
    # Save per-seed, per-method results.
    # ---------------------------------------------------------
    results_dir = resolve_path(output_cfg["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "per_seed_results.csv"

    fieldnames = [
        "seed",
        "method",
        "status",
        "processed_samples",
        "evaluation_primary_rms",
        "evaluation_residual_rms",
        "evaluation_attenuation_db",
        "steady_state_primary_rms",
        "steady_state_residual_rms",
        "steady_state_attenuation_db",
        "convergence_time_s",
        "convergence_status",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Aggregate statistics.
    # ---------------------------------------------------------
    summary_lines = [
        f"Benchmark: {experiment_cfg['name']}",
        f"Sample rate: {sample_rate_hz} Hz",
        f"Training duration: {training_duration_s} s",
        f"Test duration: {test_duration_s} s",
        f"Evaluation transient: {transient_s} s",
        f"Number of seeds: {len(seeds)}",
        "",
    ]

    for method in METHODS:
        method_rows = [
            row for row in rows
            if row["method"] == method
        ]

        completed_count = sum(
            row["status"] == "completed"
            for row in method_rows
        )

        summary_lines.extend([
            METHOD_LABELS[method],
            f"  Completed runs: {completed_count}/{len(seeds)}",
        ])

        for metric_name, label in (
            ("evaluation_attenuation_db", "Evaluation attenuation"),
            ("steady_state_attenuation_db", "Steady-state attenuation"),
        ):
            mean, std, count = metric_statistics(
                rows,
                method,
                metric_name,
            )

            if count:
                summary_lines.append(
                    f"  {label}: {mean:.4f} +/- {std:.4f} dB "
                    f"(n={count})"
                )
            else:
                summary_lines.append(
                    f"  {label}: unavailable"
                )

        if method in ("fxlms_matched", "fxlms_mismatched"):
            mean, std, count = metric_statistics(
                rows,
                method,
                "convergence_time_s",
            )

            detected_count = sum(
                row["convergence_status"] == "detected"
                for row in method_rows
            )

            summary_lines.append(
                f"  Convergence detected: {detected_count}/{len(seeds)}"
            )

            if count:
                summary_lines.append(
                    f"  Convergence time: {mean:.4f} +/- {std:.4f} s "
                    f"(n={count})"
                )
            else:
                summary_lines.append(
                    "  Convergence time: not detected"
                )
        else:
            summary_lines.append(
                "  Convergence time: not applicable (non-adaptive method)"
            )

        summary_lines.append("")

    summary_path = results_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Generate benchmark figures.
    # ---------------------------------------------------------
    figure_paths: list[Path] = []

    if bool(output_cfg.get("save_figures", True)):
        figure_dir = resolve_path(
            output_cfg["figure_directory"]
        )

        figure_paths = plot_benchmark(
            rows=rows,
            figure_dir=figure_dir,
        )

    print("\nBenchmark summary:")
    print("\n".join(summary_lines))

    print(f"Per-seed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")

    for path in figure_paths:
        print(f"Figure saved to: {path}")


if __name__ == "__main__":
    main()