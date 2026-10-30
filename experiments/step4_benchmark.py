from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from anc.controllers import FxLMSController, design_fixed_fir
from anc.fxlms import run_fxlms
from anc.metrics import attenuation_db
from anc.plant import ANCPlant, apply_fir
from anc.signals import generate_tone_mixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def evaluate_attenuation(
        primary: np.ndarray,
        residual: np.ndarray,
        start_sample: int,
) -> float:
    return attenuation_db(
        primary,
        residual,
        start_sample=start_sample,
    )


def main() -> None:
    parser =  argparse.ArgumentParser(
        description="Benchmark fixed FIR and FxLMS ANC mathods."
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

    sample_rate_hz = int(signal_cfg["sample_rate_hz"])
    training_duration_s = float(
        signal_cfg["training_duration_s"]
    )
    test_duration_s = float(signal_cfg["test_duration_s"])

    transient_s = float(benchmark_cfg["transient_s"])
    start_sample = int(round(transient_s * sample_rate_hz))

    plant = make_plant(config)

    if transient_s < 0 or transient_s >= test_duration_s:
        raise ValueError(
            "transient_s must be non-negative and shorter "
            "than test_duration_s."
        )

    mismatch_taps = np.asarray(
        mismatch_cfg["estimated_secondary_taps"],
        dtype=np.float64,
    )

    mismatch_delay = int(
        mismatch_cfg["estimated_secondary_delay_samples"]
    )

    if mismatch_delay < 0:
        raise ValueError("Estimated secondary-path delay must be non-negative.")

    if not np.isfinite(mismatch_taps).all():
        raise ValueError("Mismatch taps must be finite.")

    seeds = [int(seed) for seed in benchmark_cfg["seeds"]]

    if not seeds:
        raise ValueError("At least one benchmark seed is required.")

    rows: list[dict[str, float | int]] = []

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
        test_primary = plant.primary_output(test_reference)

        # ---------------------------------------------------------
        # 1. No-control reference.
        # ---------------------------------------------------------
        no_control_residual = test_primary.copy()

        no_control_db = evaluate_attenuation(
            test_primary,
            no_control_residual,
            start_sample,
        )

        # ---------------------------------------------------------
        # 2. Fixed FIR: design on training data, evaluate on test.
        # ---------------------------------------------------------
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

        fixed_db = evaluate_attenuation(
            test_primary,
            fixed_residual,
            start_sample,
        )

        # ---------------------------------------------------------
        # 3. FxLMS with a matched secondary-path estimate.
        # ---------------------------------------------------------
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
        )

        matched_db = evaluate_attenuation(
            test_primary,
            matched_trace.residual,
            start_sample,
        )

        # ---------------------------------------------------------
        # 4. FxLMS with a mismatched secondary-path estimate.
        # ---------------------------------------------------------
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
        )

        mismatch_db = evaluate_attenuation(
            test_primary,
            mismatch_trace.residual,
            start_sample,
        )

        # Ensure all results are valid before recording them.
        results = {
            "no_control_db": no_control_db,
            "fixed_fir_db": fixed_db,
            "fxlms_matched_db": matched_db,
            "fxlms_mismatched_db": mismatch_db,
        }

        for name, value in results.items():
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"{name} is non-finite for seed {seed}: {value}"
                )

        rows.append({
            "seed": seed,
            **results,
        })

        print(
            f"Seed {seed}: "
            f"no control={no_control_db:.2f} dB, "
            f"fixed={fixed_db:.2f} dB, "
            f"FxLMS matched={matched_db:.2f} dB, "
            f"FxLMS mismatched={mismatch_db:.2f} dB"
        )

    # -------------------------------------------------------------
    # Aggregate results.
    # -------------------------------------------------------------
    methods = [
        "no_control_db",
        "fixed_fir_db",
        "fxlms_matched_db",
        "fxlms_mismatched_db",
    ]

    labels = {
        "no_control_db": "No control",
        "fixed_fir_db": "Fixed FIR",
        "fxlms_matched_db": "FxLMS matched",
        "fxlms_mismatched_db": "FxLMS mismatched",
    }

    summary: dict[str, tuple[float, float]] = {}

    for method in methods:
        values = np.asarray(
            [float(row[method]) for row in rows],
            dtype=np.float64,
        )

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        summary[method] = (mean, std)

    # -------------------------------------------------------------
    # Save results.
    # -------------------------------------------------------------
    results_dir = resolve_path(output_cfg["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "per_seed_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seed", *methods],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = results_dir / "summary.txt"

    lines = [
        f"Benchmark: {experiment_cfg['name']}",
        f"Sample rate: {sample_rate_hz} Hz",
        f"Training duration: {training_duration_s} s",
        f"Test duration: {test_duration_s} s",
        f"Evaluation transient: {transient_s} s",
        f"Number of seeds: {len(seeds)}",
        "",
        "Attenuation (mean +/- standard deviation):",
    ]

    for method in methods:
        mean, std = summary[method]
        lines.append(
            f"{labels[method]}: {mean:.4f} +/- {std:.4f} dB"
        )

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Plot aggregate attenuation.
    # -------------------------------------------------------------
    if bool(output_cfg.get("save_figures", True)):
        figure_dir = resolve_path(
            output_cfg["figure_directory"]
        )
        figure_dir.mkdir(parents=True, exist_ok=True)

        means = [summary[method][0] for method in methods]
        stds = [summary[method][1] for method in methods]

        fig, ax = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(methods))

        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=5,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [labels[method] for method in methods]
        )
        ax.set_ylabel("Attenuation (dB)")
        ax.set_title(
            "ANC method comparison on held-out test signals"
        )
        ax.axhline(0.0, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        figure_path = figure_dir / "step4_benchmark.png"
        fig.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Benchmark figure saved to: {figure_path}")

    print(f"Per-seed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nBenchmark summary:")
    for method in methods:
        mean, std = summary[method]
        print(
            f"{labels[method]}: {mean:.4f} +/- {std:.4f} dB"
        )


if __name__ == "__main__":
    main()