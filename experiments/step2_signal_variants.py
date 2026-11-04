from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from anc.controllers import LMSController
from anc.metrics import attenuation_db, rms
from anc.plant import apply_fir
from anc.signals import (
    generate_band_limited_noise,
    generate_tone_mixture,
)
from anc.adaptive import run_lms
from experiments.plotting import plot_lms_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_string: str) -> Path:
    path = Path(path_string)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def generate_variant_reference(
        variant: dict,
        sample_rate_hz: int,
        duration_s: float,
        seed: int,
) -> np.ndarray:
    """Generate one configured LMS reference-signal variant."""

    signal_type = variant["type"]

    if signal_type == "single_tone":
        tones = [
            {
                "frequency_hz": float(variant["frequency_hz"]),
                "amplitude": float(variant["amplitude"]),
            }
        ]

        return generate_tone_mixture(
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            tones=tones,
            seed=seed,
        )

    if signal_type == "multi_tone":
        tones = [
            {
                "frequency_hz": float(tone["frequency_hz"]),
                "amplitude": float(tone["amplitude"]),
            }
            for tone in variant["tones"]
        ]

        return generate_tone_mixture(
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            tones=tones,
            seed=seed,
        )

    if signal_type == "white_noise":
        return generate_tone_mixture(
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            tones=[],
            white_noise_std=float(variant["noise_std"]),
            seed=seed,
        )

    if signal_type == "band_limited_noise":
        return generate_band_limited_noise(
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            low_cutoff_hz=float(variant["low_cutoff_hz"]),
            high_cutoff_hz=float(variant["high_cutoff_hz"]),
            noise_std=float(variant["noise_std"]),
            seed=seed,
            filter_length=int(
                variant.get("filter_length", 129)
            ),
        )

    raise ValueError(
        f"Unsupported signal type: {signal_type}"
    )


def metric_statistics(
        rows: list[dict],
        variant_name: str,
        metric_name: str,
) -> tuple[float, float, int]:
    """Return mean, sample standard deviation and valid run count."""

    values = [
        float(row[metric_name])
        for row in rows
        if row["variant"] == variant_name
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ordinary LMS across reference-signal types."
    )

    parser.add_argument(
        "--config",
        default="configs/step2_signal_variants.yaml",
        help="Path to signal-variants YAML configuration.",
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
    plant_cfg = config["plant"]
    controller_cfg = config["controller"]
    evaluation_cfg = config["evaluation"]
    output_cfg = config["output"]

    sample_rate_hz = int(signal_cfg["sample_rate_hz"])
    duration_s = float(signal_cfg["duration_s"])

    seeds = [
        int(seed)
        for seed in experiment_cfg["seeds"]
    ]

    variants = signal_cfg["variants"]

    if not seeds:
        raise ValueError("At least one seed is required.")

    if not variants:
        raise ValueError("At least one signal variant is required.")

    variant_names = [
        str(variant["name"])
        for variant in variants
    ]

    if len(set(variant_names)) != len(variant_names):
        raise ValueError("Signal variant names must be unique.")

    transient_s = float(
        evaluation_cfg.get("transient_s", 0.0)
    )

    start_sample = int(round(
        transient_s * sample_rate_hz
    ))

    n_samples = int(round(
        duration_s * sample_rate_hz
    ))

    if not 0 <= start_sample < n_samples:
        raise ValueError("Invalid evaluation transient.")

    primary_taps = np.asarray(
        plant_cfg["primary_taps"],
        dtype=np.float64,
    )

    if primary_taps.ndim != 1 or primary_taps.size == 0:
        raise ValueError(
            "Primary-path taps must be a non-empty vector."
        )

    if not np.isfinite(primary_taps).all():
        raise ValueError("Primary-path taps must be finite.")

    filter_length = int(controller_cfg["length"])
    step_size = float(controller_cfg["step_size"])

    rows: list[dict] = []

    results_dir = resolve_path(output_cfg["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

    figure_root = resolve_path(
        output_cfg["figure_directory"]
    )

    save_figures = bool(
        output_cfg.get("save_figures", True)
    )
    show_figures = bool(
        output_cfg.get("show_figures", False)
    )

    # ---------------------------------------------------------
    # Run all configured signal variants.
    # ---------------------------------------------------------
    for variant in variants:
        variant_name = str(variant["name"])
        signal_type = str(variant["type"])

        print(f"\nSignal variant: {variant_name}")
        print(f"Signal type: {signal_type}")

        for seed in seeds:
            reference = generate_variant_reference(
                variant=variant,
                sample_rate_hz=sample_rate_hz,
                duration_s=duration_s,
                seed=seed,
            )

            desired_noise = apply_fir(
                reference,
                primary_taps,
            )

            controller = LMSController(
                filter_length=filter_length,
                step_size=step_size,
            )

            trace = run_lms(
                reference=reference,
                desired=desired_noise,
                controller=controller,
            )

            for name, array in (
                ("reference", trace.reference),
                ("desired", trace.desired),
                ("control", trace.control),
                ("error", trace.error),
                ("weights", trace.weights),
            ):
                if not np.isfinite(array).all():
                    raise FloatingPointError(
                        f"{name} contains non-finite values "
                        f"for {variant_name}, seed {seed}."
                    )

            target_rms = rms(
                trace.desired[start_sample:]
            )

            error_rms = rms(
                trace.error[start_sample:]
            )

            prediction_error_reduction_db = attenuation_db(
                trace.desired,
                trace.error,
                start_sample=start_sample,
            )

            weight_norms = np.linalg.norm(
                trace.weights,
                axis=1,
            )

            row = {
                "seed": seed,
                "variant": variant_name,
                "signal_type": signal_type,
                "sample_rate_hz": sample_rate_hz,
                "duration_s": duration_s,
                "filter_length": filter_length,
                "step_size": step_size,
                "target_rms": target_rms,
                "prediction_error_rms": error_rms,
                "prediction_error_reduction_db": (
                    prediction_error_reduction_db
                ),
                "final_weight_norm": float(weight_norms[-1]),
                "max_weight_norm": float(np.max(weight_norms)),
                "status": "completed",
            }

            rows.append(row)

            print(
                f"  Seed {seed}: "
                f"target RMS={target_rms:.6f}, "
                f"error RMS={error_rms:.6f}, "
                f"error reduction="
                f"{prediction_error_reduction_db:.2f} dB, "
                f"max weight norm={np.max(weight_norms):.4f}"
            )

            # Save detailed traces and figures for the first seed
            # of each variant. All seeds are retained in the CSV.
            if seed == seeds[0]:
                variant_results_dir = (
                    results_dir / variant_name
                )
                variant_results_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.savez_compressed(
                    variant_results_dir / "traces.npz",
                    reference=trace.reference,
                    desired=trace.desired,
                    control=trace.control,
                    error=trace.error,
                    weights=trace.weights,
                )

                if save_figures:
                    variant_figure_dir = (
                        figure_root / variant_name
                    )

                    figure_paths = plot_lms_results(
                        trace=trace,
                        sample_rate_hz=sample_rate_hz,
                        output_dir=variant_figure_dir,
                        show=show_figures,
                    )

                    for path in figure_paths:
                        print(f"  Figure saved to: {path}")

    # ---------------------------------------------------------
    # Save individual results.
    # ---------------------------------------------------------
    csv_path = results_dir / "per_seed_results.csv"

    fieldnames = list(rows[0].keys())

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
    # Aggregate by signal type.
    # ---------------------------------------------------------
    summary_lines = [
        f"Experiment: {experiment_cfg['name']}",
        f"Sample rate: {sample_rate_hz} Hz",
        f"Duration: {duration_s} s",
        f"Controller length: {filter_length} taps",
        f"LMS step size: {step_size}",
        f"Evaluation transient: {transient_s} s",
        f"Seeds: {seeds}",
        "",
        "Prediction-error reduction by signal variant:",
    ]

    for variant in variants:
        name = str(variant["name"])

        mean, std, count = metric_statistics(
            rows,
            name,
            "prediction_error_reduction_db",
        )

        mean_error, std_error, count_error = metric_statistics(
            rows,
            name,
            "prediction_error_rms",
        )

        summary_lines.extend([
            f"{name} ({variant['type']})",
            (
                f"  Prediction-error reduction: "
                f"{mean:.4f} +/- {std:.4f} dB (n={count})"
                if count
                else "  Prediction-error reduction: unavailable"
            ),
            (
                f"  Prediction-error RMS: "
                f"{mean_error:.8f} +/- {std_error:.8f} "
                f"(n={count_error})"
                if count_error
                else "  Prediction-error RMS: unavailable"
            ),
            "",
        ])

    summary_path = results_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Plot mean prediction-error reduction by signal type.
    # ---------------------------------------------------------
    if save_figures:
        figure_root.mkdir(parents=True, exist_ok=True)

        labels = [
            str(variant["name"]).replace("_", "\n")
            for variant in variants
        ]

        means = []
        stds = []

        for variant in variants:
            mean, std, count = metric_statistics(
                rows,
                str(variant["name"]),
                "prediction_error_reduction_db",
            )

            means.append(mean if count else np.nan)
            stds.append(std if count else 0.0)

        fig, ax = plt.subplots(figsize=(10, 5))

        positions = np.arange(len(variants))

        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=5,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Prediction-error reduction (dB)")
        ax.set_title(
            "Ordinary LMS performance across reference-signal types"
        )
        ax.axhline(0.0, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        figure_path = (
            figure_root / "step2_signal_variants_comparison.png"
        )

        fig.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
        )

        if show_figures:
            plt.show()

        plt.close(fig)

        print(f"Comparison figure saved to: {figure_path}")

    print("\nSignal-variant summary:")
    print("\n".join(summary_lines))
    print(f"Per-seed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()