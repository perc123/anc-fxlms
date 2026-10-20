from pathlib import Path
import numpy as np
import yaml

from anc.controllers import design_fixed_fir
from anc.metrics import attenuation_db, rms
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture
from anc.simulate import run
from experiments.plotting import plot_results

def main() -> None:
    config_path = Path("configs/step1_fixed.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    signal_cfg = config["signal"]
    plant_cfg = config["plant"]
    controller_cfg = config["controller"]
    evaluation_cfg = config["evaluation"]
    output_cfg = config["output"]

    sample_rate_hz = int(signal_cfg["sample_rate_hz"])

    reference = generate_tone_mixture(
        sample_rate_hz=sample_rate_hz,
        duration_s=float(signal_cfg["duration_s"]),
        tones=signal_cfg["tones"],
        white_noise_std=float(signal_cfg["white_noise_std"]),
        seed=int(config["experiment"]["seed"]),
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

    primary_noise = plant.primary_output(reference)

    controller = design_fixed_fir(
        reference=reference,
        desired_noise=primary_noise,
        plant=plant,
        filter_length=int(controller_cfg["length"]),
        regularization=float(controller_cfg["regularization"]),
    )

    trace = run(
        controller=controller,
        plant=plant,
        signal=reference,
    )

    start_sample = int(
        float(evaluation_cfg["transient_s"]) * sample_rate_hz
    )

    input_rms = rms(trace.primary_noise[start_sample:])
    residual_rms = rms(trace.residual[start_sample:])

    attenuation = attenuation_db(
        trace.primary_noise,
        trace.residual,
        start_sample=start_sample,
    )

    output_dir = Path(output_cfg["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_text = (
        f"Experiment: {config['experiment']['name']}\n"
        f"Sample rate: {sample_rate_hz} Hz\n"
        f"Duration: {signal_cfg['duration_s']} s\n"
        f"Controller length: {controller.taps.size} taps\n"
        f"Input RMS: {input_rms:.8f}\n"
        f"Residual RMS: {residual_rms:.8f}\n"
        f"Attenuation: {attenuation:.4f} dB\n"
    )

    (output_dir / "metrics.txt").write_text(
        results_text,
        encoding="utf-8",
    )

    np.savez(
        output_dir / "traces.npz",
        reference=trace.reference,
        primary_noise=trace.primary_noise,
        control=trace.control,
        anti_noise=trace.anti_noise,
        residual=trace.residual,
        controller_taps=controller.taps,
    )

    if output_cfg["save_figures"]:
        plot_results(
            trace=trace,
            controller=controller,
            sample_rate_hz=sample_rate_hz,
            output_dir=Path(output_cfg["figure_directory"]),
            start_sample=start_sample,
            show=bool(output_cfg["show_figures"]),
        )

    print(results_text)


if __name__ == "__main__":
    main()