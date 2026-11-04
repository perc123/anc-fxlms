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
from anc.metrics import attenuation_db
from anc.plant import ANCPlant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DISPLAY_CEILING_DB = 60.0


def resolve_path(path_string: str) -> Path:
    path = Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def phase_error_degrees(
        frequency_hz: float,
        sample_rate_hz: int,
        estimated_delay_samples: int,
        actual_delay_samples: int,
) -> float:
    """Return estimated-minus-actual phase at the test frequency.
    
    For a pure delay difference:
        phase_error = -360 * f / fs * (D_est - D_actual)
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    if not 0.0 < frequency_hz < sample_rate_hz / 2:
        raise ValueError(
            "frequency_hz must lie strictly between zero and Nyquist."
        )

    if estimated_delay_samples < 0 or actual_delay_samples < 0:
        raise ValueError("Delays must be non-negative.")

    return float(
        -360.0
        * frequency_hz
        / sample_rate_hz
        * (
            estimated_delay_samples
            - actual_delay_samples
        )
    )


def generate_reference(
        frequency_hz: float,
        amplitude: float,
        sample_rate_hz: int,
        duration_s: float,
        seed: int,
        white_noise_std: float = 0.0,
) -> np.ndarray:
    """Generate a sinusoid with a reproducible randpom starting phase."""
    if amplitude <= 0 or not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite and positive.")

    if white_noise_std < 0 or not np.isfinite(white_noise_std):
        raise ValueError(
            "white_noise_std must be finite and non-negative."
        )

    n_samples = int(round(sample_rate_hz * duration_s))

    if n_samples < 1:
        raise ValueError("Signal must contain at least one sample.")

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(-np.pi, np.pi))

    n = np.arange(n_samples, dtype=np.float64)

    reference = amplitude * np.sin(
        2.0 * np.pi * frequency_hz
        * n / sample_rate_hz
        + phase
    )
    
    if white_noise_std > 0.0:
        reference += (
            white_noise_std
            * rng.standatd_normal(n_samples)
        )

    return reference


def metric_statistics(
    rows: list[dict],
    phase_error: float,
    metric_name: str,
) -> tuple[float, float, int]:
    """Return mean, sample standard deviation, and valid run count."""
    values = [
        float(row[metric_name])
        for row in rows
        if np.isclose(row["phase_error_deg"], phase_error)
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
        description=(
            "Measure FxLMS performance versus secondary-path"
            "phase-estimation error"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/step3_phase_error_sweep.yaml",
        help="Path to phase-error sweep YAML configuration.",
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
    frequency_hz = float(signal_cfg["frequency_hz"])
    amplitude = float(signal_cfg["amplitude"])
    white_noise_std = float(
        signal_cfg.get("white_noise_std", 0.0)
    )

    seeds = [int(seed) for seed in sweep_cfg["seeds"]]
    estimated_delays = [
        int(delay)
        for delay in sweep_cfg[
            "estimated_secondary_delay_samples"
        ]
    ]

    if not seeds:
        raise ValueError("At least one seed is required.")

    if not estimated_delays:
        raise ValueError("At least one estimated delay is required.")

    if any(delay < 0 for delay in estimated_delays):
        raise ValueError(
            "Estimated secondary delays must be non-negative."
        )

    if len(set(estimated_delays)) != len(estimated_delays):
        raise ValueError(
            "Estimated secondary delays must be unique."
        )

    actual_secondary_delay = int(
        plant_cfg["secondary_delay_samples"]
    )
    primary_delay = int(
        plant_cfg["primary_delay_samples"]
    )

    if actual_secondary_delay < 0 or primary_delay < 0:
        raise ValueError("Plant delays must be non-negative")

    # Keep the physical system casual with a fixed positive margin.
    causality_margin = (
        primary_delay - actual_secondary_delay
    )

    if causality_margin < 0:
        raise ValueError(
            "This phase-error experiment requires a non-negative" 
            "physical causality margin"
        )

    base_primary_taps = np.asarray(
        plant_cfg["primary_taps"],
        dtype=np.float64,
    )

    if base_primary_taps.ndim != 1 or base_primary_taps.size == 0:
        raise ValueError("Primary-path taps must be a non-empty vector.")

    secondary_taps = np.asarray(
        plant_cfg["secondary_taps"],
        dtype=np.float64,
    )

    if secondary_taps.ndim != 1 or secondary_taps.size == 0:
        raise ValueError("Secondary-path taps must be a non-empty vector.")

    if not np.isfinite(base_primary_taps).all():
        raise ValueError("Primary-path taps must be finite.")

    if not np.isfinite(secondary_taps).all():
        raise ValueError("Secondary-path taps must be finite.")

    # Add the actual primary-delay.
    primary_taps = np.concatenate(
        (
            np.zeros(primary_delay, dtype=np.float64),
            base_primary_taps,
        )
    )

    plant = ANCPlant(
        primary_taps=primary_taps,
        secondary_taps=secondary_taps.copy(),
        secondary_delay_samples=actual_secondary_delay,
    )

    transient_s = float(
        evaluation_cfg.get("transient_s", 0.0)
    )
    start_sample = int(round(
        transient_s * sample_rate_hz
    ))

    n_samples = int(round(
        sample_rate_hz * duration_s
    ))

    if not 0 <= start_sample < n_samples:
        raise ValueError("Invalid evaluation transient.")

    rows: list[dict] = []

    # Same reference per seed across all phase-error conditions.
    for seed in seeds:
        reference = generate_reference(
            frequency_hz=frequency_hz,
            amplitude=amplitude,
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            seed=seed,
            white_noise_std=white_noise_std,
        )

        for estimated_delay in estimated_delays:
            phase_error = phase_error_degrees(
                frequency_hz=frequency_hz,
                sample_rate_hz=sample_rate_hz,
                estimated_delay_samples=estimated_delay,
                actual_delay_samples=actual_secondary_delay,
            )

            controller = FxLMSController(
                filter_length=int(controller_cfg["length"]),
                step_size=float(controller_cfg["step_size"]),
                estimated_secondary_taps=secondary_taps.copy(),
                estimated_secondary_delay_samples=estimated_delay,
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

            evaluation_attenuation = None
            steady_state_attenuation = None
            steady_state_residual_rms = None

            # Evaluate attenuation only for complete runs.
            if (
                trace.status == "completed"
                and trace.processed_samples == reference.size
            ):
                evaluation_attenuation = attenuation_db(
                    trace.primary_noise,
                    trace.residual,
                    start_sample=start_sample,
                )

                convergence_metrics = analyze_convergence(
                    primary_noise=trace.primary_noise,
                    residual=trace.residual,
                    sample_rate_hz=sample_rate_hz,
                    start_sample=start_sample,
                    steady_state_fraction=float(
                        evaluation_cfg.get(
                            "steady_state_fraction", 0.2
                        )
                    ),
                    block_duration_s=0.1,
                    tolerance_db=1.0,
                    consecutive_blocks=3,
                )

                steady_state_attenuation = (
                    convergence_metrics.steady_state_attenuation_db
                )
                steady_state_residual_rms = (
                    convergence_metrics.steady_state_residual_rms
                )

            valid_weights = trace.weights[
                :trace.processed_samples
            ]

            max_weight_norm = (
                float(
                    np.max(
                        np.linalg.norm(
                            valid_weights,
                            axis=1,
                        )
                    )
                )
                if valid_weights.size
                else 0.0
            )

            row = {
                "seed": seed,
                "frequency_hz": frequency_hz,
                "actual_secondary_delay_samples": (
                    actual_secondary_delay
                ),
                "estimated_secondary_delay_samples": (
                    estimated_delay
                ),
                "phase_error_deg": phase_error,
                "causality_margin_samples": causality_margin,
                "status": trace.status,
                "processed_samples": trace.processed_samples,
                "divergence_sample": trace.divergence_sample,
                "divergence_reason": (
                    trace.divergence_reason or ""
                ),
                "max_weight_norm": max_weight_norm,
                "evaluation_attenuation_db": (
                    evaluation_attenuation
                ),
                "steady_state_residual_rms": (
                    steady_state_residual_rms
                ),
                "steady_state_attenuation_db": (
                    steady_state_attenuation
                ),
            }

            rows.append(row)

            attenuation_text = (
                "N/A"
                if evaluation_attenuation is None
                else f"{evaluation_attenuation:.2f} dB"
            )

            print(
                f"Seed {seed} | "
                f"phase error={phase_error:+7.2f} deg | "
                f"estimated delay={estimated_delay:2d} | "
                f"status={trace.status:9s} | "
                f"attenuation={attenuation_text}"
            )

    # Sort by phase error for readable CSV output.
    rows.sort(
        key=lambda row: (
            row["phase_error_deg"],
            row["seed"],
        )
    )

    # ---------------------------------------------------------
    # Save per-seed results.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Summarize every phase-error condition.
    # ---------------------------------------------------------
    phase_errors = sorted({
        float(row["phase_error_deg"])
        for row in rows
    })

    summary_lines = [
        f"Experiment: {experiment_cfg['name']}",
        f"Sample rate: {sample_rate_hz} Hz",
        f"Test frequency: {frequency_hz} Hz",
        f"Duration: {duration_s} s",
        f"Actual secondary delay: {actual_secondary_delay} samples",
        f"Primary delay: {primary_delay} samples",
        f"Causality margin: {causality_margin} samples",
        f"Seeds: {seeds}",
        "",
        "Results by secondary-path phase error:",
    ]

    for phase_error in phase_errors:
        group = [
            row for row in rows
            if np.isclose(
                row["phase_error_deg"],#
                phase_error,
            )
        ]

        completed = sum(
            row["status"] == "completed"
            for row in group
        )
        diverged = sum(
            row["status"] == "diverged"
            for row in group
        )

        mean_eval, std_eval, count_eval = metric_statistics(
            rows,
            phase_error,
            "evaluation_attenuation_db",
        )

        mean_steady, std_steady, count_steady = metric_statistics(
            rows,
            phase_error,
            "steady_state_attenuation_db",
        )

        summary_lines.extend([
            f"Phase error: {phase_error:+.2f} degrees",
            f"  Completed runs: {completed}/{len(group)}",
            f"  Diverged runs: {diverged}/{len(group)}",
            (
                f"  Evaluation attenuation: "
                f"{mean_eval:.4f} +/- {std_eval:.4f} dB "
                f"(n={count_eval})"
                if count_eval else
                "  Evaluation attenuation: unavailable"
            ),
            (
                f"  Steady-state attenuation: "
                f"{mean_steady:.4f} +/- {std_steady:.4f} dB "
                f"(n={count_steady})"
                if count_steady else
                "  Steady-state attenuation: unavailable"
            ),
            "",
        ])

    summary_path = results_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Plot attenuation and divergence rate.
    # ---------------------------------------------------------
    figure_paths: list[Path] = []

    if bool(output_cfg.get("save_figures", True)):
        figure_dir = resolve_path(
            output_cfg["figure_directory"]
        )
        figure_dir.mkdir(parents=True, exist_ok=True)

        means = []
        stds = []
        divergence_fractions = []

        for phase_error in phase_errors:
            mean, std, count = metric_statistics(
                rows,
                phase_error,
                "evaluation_attenuation_db",
            )

            means.append(mean if count else np.nan)
            stds.append(std if count else 0.0)

            group = [
                row for row in rows
                if np.isclose(
                    row["phase_error_deg"],
                    phase_error,
                )
            ]

            divergence_fractions.append(
                sum(row["status"] == "diverged" for row in group)
                / len(group)
            )

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(11, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # Display ceiling only. The simulation is noiseless with a
        # deterministic tone, so the residual can become arbitrarily small;
        # values above the ceiling are not physical ANC performance.
        # CSV and summary keep the unclipped measurements.
        means_arr = np.asarray(means, dtype=np.float64)
        stds_arr = np.asarray(stds, dtype=np.float64)
        plot_means = np.minimum(means_arr, DISPLAY_CEILING_DB)
        plot_upper = np.minimum(
            means_arr + stds_arr, DISPLAY_CEILING_DB
        )
        plot_lower = np.minimum(
            means_arr - stds_arr, DISPLAY_CEILING_DB
        )
        yerr = np.vstack(
            (plot_means - plot_lower, plot_upper - plot_means)
        )

        ax1.errorbar(
            phase_errors,
            plot_means,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label="Mean evaluation attenuation",
        )

        for boundary in (-90.0, 90.0):
            ax1.axvline(
                boundary,
                linestyle="--",
                linewidth=1.2,
                label=f"{boundary:+.0f} degree boundary",
            )

        ax1.axhline(
            0.0,
            color="gray",
            linestyle=":",
            linewidth=1.0,
        )

        ax1.axhline(
            DISPLAY_CEILING_DB,
            color="red",
            linestyle="-.",
            linewidth=1.0,
            label=(
                f"{DISPLAY_CEILING_DB:g} dB display ceiling "
                "(visualization limit)"
            ),
        )
        ax1.set_ylim(top=DISPLAY_CEILING_DB * 1.05)

        ax1.set_ylabel("Evaluation attenuation (dB, clipped for display)")
        ax1.set_title(
            "Narrowband FxLMS performance versus "
            "secondary-path phase error"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(
            phase_errors,
            divergence_fractions,
            width=8.0,
        )

        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("Diverged\nfraction")
        ax2.set_xlabel(
            "Estimated minus actual secondary-path phase "
            f"at {frequency_hz:g} Hz (degrees)"
        )
        ax2.set_yticks([0.0, 0.5, 1.0])
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        figure_path = (
            figure_dir / "step3_phase_error_sweep.png"
        )
        fig.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        figure_paths.append(figure_path)

    print("\nPhase-error sweep summary:")
    print("\n".join(summary_lines))
    print(f"Per-seed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")

    for path in figure_paths:
        print(f"Figure saved to: {path}")


if __name__ == "__main__":
    main()