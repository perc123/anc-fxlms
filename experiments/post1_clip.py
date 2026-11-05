"""Render the LinkedIn clip for post 1: 1.5 ms vs 2.0 ms secondary-path model error.

Run from the project root:
    python -m experiments.post1_clip
Writes docs/media/post1_model_error.mp4 (needs ffmpeg on PATH).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from anc.controllers import FxLMSController
from anc.fxlms import run_fxlms
from anc.plant import ANCPlant
from anc.signals import generate_tone_mixture

# --- Simulation: same setup as configs/step3_phase_error_sweep.yaml ---------
FS = 2000                 # Hz
DURATION_S = 1.5        # everything interesting happens in the first second
TONE_HZ = 125.0
PRIMARY_DELAY = 12        # samples
SECONDARY_DELAY = 8       # samples (4 ms)
PRIMARY_TAPS = [0.80, 0.00, 0.15, 0.00, -0.08, 0.00, 0.04]
SECONDARY_TAPS = np.array([0.70, 0.00, 0.20, 0.00, -0.05])
LENGTH, MU = 32, 0.005

CASES = [  # (model delay error in samples, label, colour)
    (3, "Model off by 1.5 ms (67.5°)", "#1b9e77"),
    (4, "Model off by 2.0 ms (90°)", "#d95f02"),
]

# --- Video ------------------------------------------------------------------
VIDEO_S = 12.0            # length of the animated part (8x slow motion)
HOLD_S = 2.0              # final frame held
FPS = 30
WINDOW_S = 0.040          # scrolling waveform window
BLOCK_S = 0.020           # level-meter block
DB_FLOOR = -60.0
OUT = Path("docs/media/post1_model_error.mp4")


def simulate(error_samples: int):
    reference = generate_tone_mixture(
        FS, DURATION_S, [{"frequency_hz": TONE_HZ, "amplitude": 1.0}], 0.0
    )
    plant = ANCPlant(
        primary_taps=np.concatenate([np.zeros(PRIMARY_DELAY), PRIMARY_TAPS]),
        secondary_taps=SECONDARY_TAPS,
        secondary_delay_samples=SECONDARY_DELAY,
    )
    controller = FxLMSController(
        LENGTH, MU, SECONDARY_TAPS, SECONDARY_DELAY + error_samples
    )
    return run_fxlms(reference, plant, controller,
                     max_weight_norm=100.0, max_abs_residual=10.0)


def block_level_db(primary, residual):
    """Residual level per block in dB re. the uncontrolled noise."""
    n = int(BLOCK_S * FS)
    times, levels = [], []
    for start in range(0, residual.size - n + 1, n):
        r = residual[start:start + n]
        if np.isnan(r).any():
            break
        p = primary[start:start + n]
        level = 20 * np.log10(np.sqrt(np.mean(r**2)) / np.sqrt(np.mean(p**2)) + 1e-12)
        times.append((start + n) / FS)
        levels.append(max(level, DB_FLOOR))
    return np.array(times), np.array(levels)


def main() -> None:
    runs = []
    for err, label, colour in CASES:
        trace = simulate(err)
        t_lvl, lvl = block_level_db(trace.primary_noise, trace.residual)
        stop_s = (trace.divergence_sample / FS
                  if trace.status == "diverged" else DURATION_S)
        runs.append(dict(trace=trace, label=label, colour=colour,
                         t_lvl=t_lvl, lvl=lvl, stop_s=stop_s))
        print(f"{label}: {trace.status}, stops at {stop_s:.2f} s")

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 9.0), dpi=150,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.08,
                        hspace=0.35, wspace=0.12)
    fig.suptitle("Same noise, same controller.\n"
                 "Only the speaker-path model differs.",
                 fontsize=16, fontweight="bold", y=0.975)
    clock = fig.text(0.5, 0.885, "", ha="center", fontsize=12, color="#555")

    win = int(WINDOW_S * FS)
    artists = []
    for col, run in enumerate(runs):
        ax_w, ax_l = axes[0, col], axes[1, col]
        ax_w.set_title(run["label"], fontsize=12, color=run["colour"],
                       fontweight="bold")
        ax_w.set_ylim(-2.5, 2.5)
        ax_w.set_xticks([])
        ax_w.set_yticks([] if col else [-2, 0, 2])
        if col == 0:
            ax_w.set_ylabel("error mic")
        noise_line, = ax_w.plot([], [], color="#bbbbbb", lw=1.5,
                                label="no control")
        res_line, = ax_w.plot([], [], color=run["colour"], lw=2.2,
                              label="with ANC")
        status = ax_w.text(0.5, 0.5, "", transform=ax_w.transAxes,
                           ha="center", va="center", fontsize=20,
                           fontweight="bold", color=run["colour"])
        if col == 0:
            ax_w.legend(loc="lower left", fontsize=9, frameon=False)

        ax_l.set_xlim(0, DURATION_S)
        ax_l.set_ylim(DB_FLOOR - 3, 25)
        ax_l.axhline(0, color="#999", lw=1, ls="--")
        ax_l.text(DURATION_S, 1.5, "no control", ha="right",
                  fontsize=9, color="#777")
        ax_l.set_xlabel("time (s)")
        if col == 0:
            ax_l.set_ylabel("residual level (dB)")
        else:
            ax_l.set_yticklabels([])
        level_line, = ax_l.plot([], [], color=run["colour"], lw=2.5)
        artists.append((ax_w, noise_line, res_line, status, level_line))

    n_anim = int(VIDEO_S * FPS)
    n_frames = n_anim + int(HOLD_S * FPS)

    def update(frame):
        t_now = min(frame, n_anim) / n_anim * DURATION_S
        clock.set_text(f"simulation time  {t_now:4.2f} s   (8x slow motion)")
        out = [clock]
        for run, (ax_w, noise_line, res_line, status, level_line) in zip(runs, artists):
            t_show = min(t_now, run["stop_s"])
            end = max(int(t_show * FS), win)
            idx = np.arange(end - win, end)
            noise_line.set_data(idx, run["trace"].primary_noise[idx])
            residual = np.clip(run["trace"].residual[idx], -2.45, 2.45)
            res_line.set_data(idx, residual)
            ax_w.set_xlim(idx[0], idx[-1])

            mask = run["t_lvl"] <= t_show
            level_line.set_data(run["t_lvl"][mask], run["lvl"][mask])

            if t_now >= run["stop_s"] and run["trace"].status == "diverged":
                status.set_text("DIVERGED")
            elif t_now >= DURATION_S and mask.any():
                status.set_text(f"{-run['lvl'][mask][-1]:.0f}+ dB quieter"
                                if run["lvl"][mask][-1] <= DB_FLOOR
                                else f"{-run['lvl'][mask][-1]:.0f} dB quieter")
            out += [noise_line, res_line, status, level_line]
        return out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=4000,
              extra_args=["-pix_fmt", "yuv420p"]))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
