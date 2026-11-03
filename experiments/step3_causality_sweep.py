from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from anc.convergence import analyze_convergence
from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.metrics import attenuation_db, rms
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def add_primary_delay(
        primary_taps: np.ndarray,
        delay_samples: int,
) -> np.ndarray:
    """Add an explicit leading delay to the primary-path FIR."""
    taps = np.asarray(primary_taps, dtype=np.float64)

    if taps.ndim != 1 or taps.size == 0:
        raise ValueError("primary_taps must be a non-empty vector.")

    if not np.isfinite(taps).all():
        raise ValueError("primary_taps must contain only finite values.")

    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative.")

    return np.concatenate(
        (
            np.zeros(delay_samples, dtype=np.float64),
            taps,
        )
    )


def causality_margin_samples(
        primary_delay_samples: int,
        secondary_delay_samples: int,
        processing_delay_samples: int,
) -> int:
    """Calculate primary preview minus total secondary/processing delay."""
    if primary_delay_samples < 0:
        raise ValueError("Primary delay must be non-negative.")

    if secondary_delay_samples < 0:
        raise ValueError("Secondary delay must be non-negative.")

    if processing_delay_samples < 0:
        raise ValueError("Processing delay must be non-negative.")

    return (
        primary_delay_samples
        - secondary_delay_samples
        - processing_delay_samples
    )


def metric_statistics(
        rows: list[dict],
        margin: int,
        metric_name: str,
) -> tuple[float, float, int]:
    values = [
        float(row[metric_name])
        for row in rowsif row["causality_margin_sample"] == margin
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
        description="Measure FxLMS performance versus ANC causality margin."
    )
    parser.add_argument(
        "--config",
        default="configs/step3_causality_sweep.yaml",
        help="Path to causality-sweep YAML configuration.",
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
    sweep_cfg = config["sweep"]
    plant_cfg = config["plant"]
    controller_cfg = config["controller"]
    evaluation_cfg = config["evaluation"]
    safety_cfg = config.get("safety", {})
    output_cfg = config["output"]

    sample_rate_hz = int(signal_cfg["sample_rate_hz"])
    duration_s = float(signal_cfg["duration_s"])

    seeds = [int(seed) for seed in sweep_cfg["seeds"]]
    primary_delays = [
        int(delay)
        for delay in sweep_cfg["primary_delay_samples"]
    ]

    if not seeds:
        raise ValueError("At least one random seed is required.")

    if not primary_delays:
        raise ValueError("At least one primary delay is required.")

    if len(set(primary_delays)) != len(primary_delays):
        raise ValueError("Primary delay values must be unique.")

    processing_delay = int(
        sweep_cfg["processing_delay_samples"]
    )
    physical_secondary_delay = int(
        plant_cfg["secondary_delay_samples"]
    )

    if processing_delay < 0 or physical_secondary_delay < 0:
        raise ValueError("Delays must be non-negative.")

    effective_secondary_delay = (
        physical_secondary_delay + processing_delay
    )

    base_primary_taps = np.asarray(
        plant_cfg["primary_taps"],
        dtype=np.float64,
    )

    if not np.isfinite(secondary_taps).all():
        raise ValueError("Secondary-path taps must be finite.")

    transient_s = float(
        evaluation_cfg.get("transient_s", 0.0)
    )
    start_sample = int(round(transient_s * sample_rate_hz))

    if not 0 <= start_sample < int(
        round(sample_rate_hz * duration_s)
    ):
        raise ValueError("Invalid evaluation transient.")

    rows: list[dict] = []

    # Generate each reference once, then reuse it for every delay
    # condition belonging to that seed.
    for seed in seeds:
        reference = generate_tone_mixture(
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            tones=signal_cfg.get["tones", []],
            white_noise_std=float(
                signal_cfg.get("white_noies_std", 0.0)
            ),
            seed=seed,
        )

        for primary_delay in primary_delays:
            margin = causality_margin_samples(
                primary_delay_samples=primary_delay,
                secondary_delay_samples=physical_secondary_delay,
                processing_delay_samples=processing_delay,
            )

            delayed_primary_taps = add_primary_delay(
                base_primary_taps,
                primary_delay,
            )

            plant = ANCPlant(
                primary_taps=delayed_primary_taps,
                secondary_taps=secondary_taps.copy(),
                secondary_delay_samples=effective_secondary_delay,
            )

            # The estimated secondary path is deliberately exact.
            # This sweep isolates the causality constraint.
            controller = FxLMSController(
                filter_length=int(controller_cfg["length"]),
                step_size=float(controller_cfg["step_size"]),
                estimated_secondary_taps=secondary_taps.copy(),
                estimated_secondary_delay_samples=(
                    effective_secondary_delay
                ),
            )

            trace = run_fxlms(
                reference=reference,
                plant=plant,
                controller=controller,
                max_weight_norm=safety_cfg.get(
                    "max_weight_norm", 1e6
                ),
                max_abs_residual=safety_cfg.get(
                    "max_abs_residual", 1e6
                ),
            )

            evaluation_primary_rms = None
            evaluation_residual_rms = None
            evaluation_attenuation = None
            steady_primary_rms = None
            steady_residual_rms = None
            steady_attenuation = None

            convergence_time = None
            convergence_status = "unavailable"

            # Only calculate full-duration metrics for complete runs.
            if (
                trace.status == "completed"
                and trace.processed_samples == reference.size
            ):
                primary_valid = trace.primary_noise
                residual_valid = trace.residual

                if reference.size <= start_sample:
                    raise ValueError(
                        "Evaluation transient leaves no evaluation samples."
                    )

                evaluation_primary_rms = rms(
                    primary_valid[start_sample:]
                )
                evaluation_residual_rms = rms(
                    residual_valid[start_sample:]
                )

                evaluation_attenuation = attenuation_db(
                    primary_valid,
                    residual_valid,
                    start_sample=start_sample,
                )

                convergence = analyze_convergence(
                    primary_noise=primary_valid,
                    residual=residual_valid,
                    sample_rate_hz=sample_rate_hz,
                    start_sample=start_sample,
                    block_duration_s=float(
                        evaluation_cfg.get(
                            "block_duration_s", 0.1
                        )
                    ),
                    steady_state_fraction=float(
                        evaluation_cfg.get(
                            "steady_state_fraction", 0.2
                        )
                    ),
                    tolerance_db=float(
                        evaluation_cfg.get(
                            "convergence_tolerance_db", 1.0
                        )
                    ),
                    consecutive_blocks=int(
                        evaluation_cfg.get(
                            "consecutive_blocks", 3
                        )
                    ),
                )

                steady_primary_rms = (
                    convergence.steady_state_primary_rms
                )
                steady_residual_rms = (
                    convergence.steady_state_residual_rms
                )
                steady_attenuation = (
                    convergence.steady_state_attenuation_db
                )

                # Convergence time is intentionally not used as
                # the primary result of this causality experiment.
                # A stable but uncancellable residual can also habe
                # a stable block-RMS sequence.

            row = {
                "seed": seed,
                "primary_delay_samples": primary_delay,
                "physical_secondary_delay_samples": (
                    physical_secondary_delay
                ),
                "processing_delay_samples": processing_delay,
                "effective_secondary_delay_samples": (
                    effective_secondary_delay
                ),
                "causality_margin_samples": margin,
                "status": trace.status,
                "processed_samples": trace.processed_samples,
                "divergence_sample": trace.divergence_sample,
                "divergence_reason": (
                    trace.divergence_reason or ""
                ),
                "evaluation_primary_rms": evaluation_primary_rms,
                "evaluation_residual_rms": evaluation_residual_rms,
                "evaluation_attenuation_db": evaluation_attenuation,
                "steady_state_primary_rms": steady_primary_rms,
                "steady_state_residual_rms": steady_residual_rms,
                "steady_state_attenuation_db": steady_attenuation,
            }

            rows.append(row)

            attenuation_text = (
                "N/A"
                if evaluation_attenuation is None
                else f"{evaluation_attenuation:.2f} dB"
            )

            print(
                f"Seed {seed} | "
                f"primary delay={primary_delay} samples | "
                f"margin={margin:+d} samples | "
                f"status={trace.status} | "
                f"attenuation={attenuation_text}"
            )
    # --------------------------------------------------------------
    # Save per-seed results
    # --------------------------------------------------------------
    results_dir = resolve_path(output_cfg["directory"])
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # --------------------------------------------------------------
    # Aggregate by causality margin.
    # --------------------------------------------------------------
    margins = sorted({
        int(row["causality_margin_samples"])
        for row in rows
    })

    summary_lines = [
        f"Experiment: {experiment_cfg['name']}",
        f"Sample rate: {sample_rate_hz} Hz",
        f"Duration: {duration_s} s",
        f"Seeds: {seeds}",
        f"Physical secondary delay: {physical_secondary_delay} samples",
        f"Processing delay: {processing_delay} samples",
        f"Effective secondary delay: {effective_secondary_delay} samples",
        "",
        "Results by causality margin:",
    ]

    for margin in margins:
        group = [
            row for row in rowsif row ["causality_margin_samples"] == margin
        ]

        completed = sum(
            row["status"] == "completed"
            for row in group
        )

        mean_eval, std_eval, count_eval = metric_statistics(
            rows,
            margin,
            "evaluation_attenuation_db",
        )

        mean_steady, std_steady, count_steady = metric_statistics(
            rows,
            margin,
            "steady_state_attenuation_db",
        )

        summary_lines.extend([
            f"Margin: {margin:+d} samples",
            f"  Completed runs: {completed}/{len(group)}",
            f"  Evaluation attenuation: "
            f"{mean_eval:.4f} +/- {std_eval:.4f} dB "
            f"(n={count_eval})"
            if count_eval else
            "  Evaluation attenuation: unavailable",
            f"  Steady-state attenuation: "
            f"{mean_steady:.4f} +/- {std_steady:.4f} dB "
            f"(n={count_steady})"
            if count_steady else
            "  Steady-state attenuation: unavailable",
            "",
        ])

        summary_path = results_dir / "summary.txt"
        summary_path.write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )

        # ---------------------------------------------------------
    # Plot mean evaluation attenuation versus causality margin.
    # ---------------------------------------------------------
    figure_paths: list[Path] = []

    if bool(output_cfg.get("save_figures", True)):
        figure_dir = resolve_path(
            output_cfg["figure_directory"]
        )
        figure_dir.mkdir(parents=True, exist_ok=True)

        means = []
        stds = []
        valid_margins = []

        for margin in margins:
            mean, std, count = metric_statistics(
                rows,
                margin,
                "evaluation_attenuation_db",
            )

            if count:
                valid_margins.append(margin)
                means.append(mean)
                stds.append(std)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(
            valid_margins,
            means,
            yerr=stds,
            marker="o",
            capsize=5,
            linewidth=1.5,
            label="Mean evaluation attenuation",
        )

        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Causality boundary",
        )

        ax.axhline(
            0,
            color="gray",
            linestyle=":",
            linewidth=1.0,
        )

        ax.set_xticks(margins)
        ax.set_xlabel(
            "Causality margin: Dp - (Ds + Dc) [samples]"
        )
        ax.set_ylabel("Evaluation attenuation (dB)")
        ax.set_title(
            "FxLMS performance versus feedforward causality margin\n"
            "Broadband white-noise reference, matched secondary path"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        figure_path = (
            figure_dir / "step3_causality_sweep.png"
        )
        fig.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        figure_paths.append(figure_path)

    print("\nCausality sweep summary:")
    print("\n".join(summary_lines))
    print(f"Per-seed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")

    for path in figure_paths:
        print(f"Figure saved to: {path}")


if __name__ == "__main__":
    main()
