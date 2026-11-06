"""Render the LinkedIn clip for post 1: 1.5 ms vs 2.0 ms secondary-path model error.

Run from the project root:
    python -m experiments.post1_clip
Writes docs/media/post1_model_error.mp4 (needs ffmpeg on PATH).
    python -m experiments.post1_clip --preview
saves a few PNG frames to a folder instead (no ffmpeg needed).
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors
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
    (3, "Model off by 1.5 ms (67.5°)", "#2ee6a6"),
    (4, "Model off by 2.0 ms (90°)", "#ff5a5f"),
]

# --- Look -------------------------------------------------------------------
BG = "#0b0f19"
PANEL = "#121826"
GRID = "#243049"
TEXT = "#e8ecf4"
MUTED = "#8793ab"
NOISE = "#5b6781"
ALERT = "#ff3b47"

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


def glow(ax, x, y, colour, lw):
    """A line with a soft halo; returns the artists so they can be updated."""
    halo, = ax.plot(x, y, color=colour, lw=lw * 3.5, alpha=0.18,
                    solid_capstyle="round")
    core, = ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round")
    return halo, core


def style_axes(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


def build(runs):
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans",
                         "text.color": TEXT, "axes.labelcolor": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 9.0), dpi=150,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.12,
                        hspace=0.30, wspace=0.18)
    fig.text(0.5, 0.945, "Same noise. Same controller.", ha="center",
             fontsize=19, fontweight="bold")
    fig.text(0.5, 0.905, "Only the speaker-path model is off by a hair.",
             ha="center", fontsize=13, color=MUTED)
    clock = fig.text(0.5, 0.855, "", ha="center", fontsize=11, color=MUTED,
                     family="monospace")
    # Progress bar along the bottom.
    bar_ax = fig.add_axes([0.12, 0.055, 0.85, 0.008])
    bar_ax.set_axis_off()
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.barh(0.5, 1, height=1, color=GRID)
    bar = bar_ax.barh(0.5, 0, height=1, color="#6ea8ff")[0]
    fig.text(0.5, 0.025, "Active noise cancellation  ·  FxLMS  ·  125 Hz tone",
             ha="center", fontsize=10, color=MUTED)

    win = int(WINDOW_S * FS)
    cols = []
    for col, run in enumerate(runs):
        ax_w, ax_l = axes[0, col], axes[1, col]
        style_axes(ax_w)
        style_axes(ax_l)
        colour = run["colour"]
        ax_w.set_title(run["label"], fontsize=12, color=colour,
                       fontweight="bold", pad=10)
        ax_w.set_ylim(-2.5, 2.5)
        ax_w.set_xticks([])
        ax_w.axhline(0, color=GRID, lw=1)
        ax_w.set_yticks([] if col else [-2, 0, 2])
        if col == 0:
            ax_w.set_ylabel("error mic")
        noise_line, = ax_w.plot([], [], color=NOISE, lw=1.6, label="no control")
        res = glow(ax_w, [], [], colour, 2.2)
        status = ax_w.text(0.5, 0.5, "", transform=ax_w.transAxes, ha="center",
                           va="center", fontsize=26, fontweight="bold",
                           color=ALERT)
        if col == 0:
            ax_w.legend([noise_line, res[1]], ["no control", "with ANC"],
                        loc="lower left", fontsize=9, frameon=False,
                        labelcolor=MUTED)

        ax_l.set_xlim(0, DURATION_S)
        ax_l.set_ylim(DB_FLOOR - 3, 25)
        ax_l.grid(color=GRID, lw=0.8)
        ax_l.set_axisbelow(True)
        ax_l.axhline(0, color=MUTED, lw=1, ls="--")
        ax_l.text(DURATION_S * 0.98, -3, "no control", ha="right",
                  va="top", fontsize=9, color=MUTED)
        ax_l.set_xlabel("time (s)")
        ax_l.set_xticks([0, 0.5, 1.0, 1.5])
        if col == 0:
            ax_l.set_ylabel("residual level (dB)")
        else:
            ax_l.set_yticklabels([])
        level = glow(ax_l, [], [], colour, 2.6)
        dot, = ax_l.plot([], [], "o", color=colour, ms=8, mec=BG, mew=1.5)
        readout = ax_l.text(0.6, 0.9, "", transform=ax_l.transAxes,
                            ha="center", va="center", fontsize=30,
                            fontweight="bold", color=colour)
        verdict = ax_l.text(0.6, 0.8, "", transform=ax_l.transAxes,
                            ha="center", va="center", fontsize=11, color=MUTED)
        cols.append(dict(ax_w=ax_w, ax_l=ax_l, noise=noise_line, res=res,
                         status=status, level=level, dot=dot, readout=readout,
                         verdict=verdict, fill=None))
    return fig, clock, bar, cols, win


def make_update(runs, fig, clock, bar, cols, win, n_anim):
    def update(frame):
        t_now = min(frame, n_anim) / n_anim * DURATION_S
        clock.set_text(f"simulation time {t_now:4.2f} s  ·  8x slow motion")
        bar.set_width(t_now / DURATION_S)
        for run, c in zip(runs, cols):
            diverged = run["trace"].status == "diverged"
            t_show = min(t_now, run["stop_s"])
            end = max(int(t_show * FS), win)
            idx = np.arange(end - win, end)
            c["noise"].set_data(idx, run["trace"].primary_noise[idx])
            residual = np.clip(run["trace"].residual[idx], -2.45, 2.45)
            for line in c["res"]:
                line.set_data(idx, residual)
            if c["fill"] is not None:
                c["fill"].remove()
            c["fill"] = c["ax_w"].fill_between(
                idx, residual, 0, color=run["colour"], alpha=0.18, lw=0)
            c["ax_w"].set_xlim(idx[0], idx[-1])

            mask = run["t_lvl"] <= t_show
            t, y = run["t_lvl"][mask], run["lvl"][mask]
            for line in c["level"]:
                line.set_data(t, y)
            c["dot"].set_data(t[-1:], y[-1:])
            if y.size:
                shown = y[-1]
                c["readout"].set_text(
                    f"{shown:+.0f} dB" if shown > DB_FLOOR else "< -60 dB")

            blown = diverged and t_now >= run["stop_s"]
            since = t_now - run["stop_s"]
            if blown:
                # Red flash that decays, then a steady alarm tint.
                flash = 0.35 * np.exp(-since * 6.0) + 0.08
                tint = (ALERT, flash)
                c["status"].set_text("DIVERGED")
                c["readout"].set_color(ALERT)
                c["verdict"].set_text("filter blew up")
                edge = ALERT
            else:
                tint, edge = None, GRID
                c["status"].set_text("")
                if t_now >= DURATION_S and y.size and not diverged:
                    c["verdict"].set_text("stable  ·  noise cancelled")
            base = np.array(matplotlib.colors.to_rgb(PANEL))
            if tint:
                mix = np.array(matplotlib.colors.to_rgb(tint[0]))
                face = base * (1 - tint[1]) + mix * tint[1]
            else:
                face = base
            c["ax_w"].set_facecolor(face)
            for ax in (c["ax_w"], c["ax_l"]):
                for spine in ax.spines.values():
                    spine.set_color(edge)
                    spine.set_linewidth(2.0 if blown else 0.8)
        return []
    return update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="save PNG frames instead of rendering the mp4")
    args = parser.parse_args()

    runs = []
    for err, label, colour in CASES:
        trace = simulate(err)
        t_lvl, lvl = block_level_db(trace.primary_noise, trace.residual)
        stop_s = (trace.divergence_sample / FS
                  if trace.status == "diverged" else DURATION_S)
        runs.append(dict(trace=trace, label=label, colour=colour,
                         t_lvl=t_lvl, lvl=lvl, stop_s=stop_s))
        print(f"{label}: {trace.status}, stops at {stop_s:.2f} s")

    fig, clock, bar, cols, win = build(runs)
    n_anim = int(VIDEO_S * FPS)
    n_frames = n_anim + int(HOLD_S * FPS)
    update = make_update(runs, fig, clock, bar, cols, win, n_anim)

    if args.preview:
        folder = Path("docs/media/preview")
        folder.mkdir(parents=True, exist_ok=True)
        for frame in (int(n_anim * f) for f in (0.15, 0.4, 0.7, 1.0)):
            update(frame)
            fig.savefig(folder / f"frame_{frame:04d}.png", facecolor=BG)
        print(f"Saved preview frames to {folder}")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=6000,
              extra_args=["-pix_fmt", "yuv420p"]), savefig_kwargs={"facecolor": BG})
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
