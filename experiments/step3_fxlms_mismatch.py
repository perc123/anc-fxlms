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
from experiments.plotting import plot_fxlms_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path_string: str) -> Path:
    path =  Path(path_string)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def main() -> None:
    parser =  argparse.ArgumentParser(
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
        (
            float(tone["frequency_hz"]),
            float(tone["amplitude"]),
        )
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
        )
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

    trace = run_fxlms(
        reference=reference,
        plant=plant,
        controller=controller,
    )

    transient_s = float(
        evaluation_cfg.get("transient_s", 0.0)
    )
    start_sample = int(round(transient_s * sample_rate_hz))

    if not 0 <= start_sample < reference.size:
        raise ValueError(
            "evaluation.transient_s must be non-negative and" \
            "shorter than the signal duration."
        )

    primary_rms = rms(trace.primary_noise[start_sample:])
    residual_rms = rms(trace.residual[start_sample:])

    anc_attenuation_db = attenuation_db(
        
    )