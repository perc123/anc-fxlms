"""LinkedIn clip for post 2: the hum always cancels, the hiss needs a head start.

Self-contained (NumPy, SciPy, Matplotlib, ffmpeg). Uses the same plant as the
project configs: 2 kHz, P = [0.80, 0, 0.15, 0, -0.08, 0, 0.04],
S = [0.70, 0, 0.20, 0, -0.05] with 2 samples of delay, 32-tap controller.

For each causality margin M = D_p - D_s it designs the best possible causal
FIR controller (regularized least squares, as in Step 1) and animates the
residual spectrum at the error mic while the margin steps from late to early.

    python post2_clip.py            -> post2_causality.mp4 + post2_cover.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import FancyBboxPatch
from scipy.signal import lfilter, welch

FS = 2000
DUR_S = 8.0
TONES = [(80.0, 0.7), (120.0, 0.4), (160.0, 0.2)]
HISS_STD = 0.08
MIC_FLOOR_STD = 3e-4        # error-mic self-noise, so the plot has a physical floor
P_TAPS = np.array([0.80, 0.00, 0.15, 0.00, -0.08, 0.00, 0.04])
S_TAPS = np.array([0.70, 0.00, 0.20, 0.00, -0.05])
S_DELAY = 2
L = 32
MARGINS = [-2, -1, 0, 1, 2]          # samples; 1 sample = 0.5 ms ~ 17 cm of air
SOUND_CM_PER_SAMPLE = 343.0 / FS * 100

FPS = 30
HOLD_S = 2.4
FADE_S = 0.7
END_HOLD_S = 1.5
DB_MIN, DB_MAX = -100, 5
OUT = Path("post2_causality.mp4")
COVER = Path("post2_cover.png")

GREY, LATE, EARLY = "#9a9a9a", "#d95f02", "#1b9e77"


def delayed(taps, d):
    return np.concatenate([np.zeros(d), taps])


def design(x, d, s_full):
    """Least-squares causal FIR w minimizing ||d + s*(w*x)||."""
    xf = lfilter(s_full, 1.0, x)
    A = np.zeros((x.size, L))
    for k in range(L):
        A[k:, k] = xf[: x.size - k]
    A = np.vstack([A, 1e-3 * np.eye(L)])
    b = np.concatenate([-d, np.zeros(L)])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def residual(x, p_full, s_full, w):
    return lfilter(p_full, 1.0, x) + lfilter(s_full, 1.0, lfilter(w, 1.0, x))


def mic(sig, seed=7):
    return sig + MIC_FLOOR_STD * np.random.default_rng(seed).standard_normal(sig.size)


def att_db(before, after, skip=400):
    b, a = before[skip:], after[skip:]
    return 20 * np.log10(np.sqrt(np.mean(b**2)) / max(np.sqrt(np.mean(a**2)), 1e-12))


def simulate():
    n = int(FS * DUR_S)
    t = np.arange(n) / FS
    hum = sum(a * np.sin(2 * np.pi * f * t) for f, a in TONES)
    hiss = HISS_STD * np.random.default_rng(42).standard_normal(n)
    x = hum + hiss
    s_full = delayed(S_TAPS, S_DELAY)
    states = []
    for m in MARGINS:
        p_full = delayed(P_TAPS, S_DELAY + m)
        w = design(x, lfilter(p_full, 1.0, x), s_full)
        d = mic(lfilter(p_full, 1.0, x))
        e = mic(residual(x, p_full, s_full, w))
        f, pd = welch(d[400:], FS, nperseg=512)
        _, pe = welch(e[400:], FS, nperseg=512)
        ref = pd.max()
        states.append(dict(
            margin=m,
            f=f,
            noise_db=np.clip(10 * np.log10(pd / ref + 1e-30), DB_MIN, DB_MAX),
            anc_db=np.clip(10 * np.log10(pe / ref + 1e-30), DB_MIN, DB_MAX),
            hum_att=att_db(lfilter(p_full, 1, hum), mic(residual(hum, p_full, s_full, w))),
            hiss_att=att_db(lfilter(p_full, 1, hiss), mic(residual(hiss, p_full, s_full, w))),
        ))
        print(f"margin {m:+d}: hum {states[-1]['hum_att']:6.1f} dB, "
              f"hiss {states[-1]['hiss_att']:6.1f} dB")
    return states


def margin_text(m):
    ms = abs(m) * 1000 / FS
    if m < 0:
        return f"anti-noise arrives {ms:.1f} ms LATE", LATE
    if m == 0:
        return "anti-noise arrives exactly on time", EARLY
    return f"reference mic hears the noise {ms:.1f} ms EARLY", EARLY


def fmt_att(v):
    return "60+ dB" if v >= 60 else f"{v:.1f} dB"


def main():
    states = simulate()
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 9.0), dpi=150)
    fig.text(0.5, 0.955, "The hum always cancels.\nThe hiss needs a head start.",
             ha="center", va="center", fontsize=19, fontweight="bold")

    # --- duct schematic ------------------------------------------------------
    ax_d = fig.add_axes([0.06, 0.70, 0.88, 0.17])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 3)
    ax_d.axis("off")
    ax_d.add_patch(FancyBboxPatch((0.2, 1.0), 9.6, 1.0,
                                  boxstyle="round,pad=0.02", fc="#f2f2f2", ec="#777"))
    ax_d.text(0.6, 1.5, "noise", ha="center", va="center", fontsize=11, color="#555")
    ax_d.add_patch(plt.Rectangle((6.6, 2.0), 0.6, 0.45, color="#444"))
    ax_d.text(6.9, 2.75, "anti-noise\nspeaker", ha="center", va="center", fontsize=9)
    ax_d.plot([9.2], [1.5], "o", ms=13, color="#444")
    ax_d.text(9.2, 0.55, "error\nmic", ha="center", va="center", fontsize=9)
    ref_dot, = ax_d.plot([], [], "o", ms=13)
    ref_lbl = ax_d.text(0, 0.55, "reference\nmic", ha="center", va="center", fontsize=9)
    status = ax_d.text(5, -0.35, "", ha="center", va="center", fontsize=13,
                       fontweight="bold")

    # --- spectrum ------------------------------------------------------------
    ax_s = fig.add_axes([0.14, 0.27, 0.80, 0.36])
    f = states[0]["f"]
    ax_s.set_xlim(0, 1000)
    ax_s.set_ylim(DB_MIN, DB_MAX + 3)
    ax_s.set_xlabel("frequency (Hz)")
    ax_s.set_ylabel("level at error mic (dB)")
    ax_s.grid(alpha=0.25)
    noise_ln, = ax_s.plot(f, states[0]["noise_db"], color=GREY, lw=2, label="no control")
    anc_ln, = ax_s.plot(f, states[0]["anc_db"], lw=2.4, label="with ANC")
    ax_s.text(140, DB_MAX - 2, "hum", ha="center", fontsize=10, color="#555")
    ax_s.text(650, -12, "hiss (broadband)", ha="center", fontsize=10, color="#555")
    ax_s.set_title("grey: no control     coloured: with ANC", fontsize=10.5, color="#555")
    ax_s.text(990, -93, "error-mic self-noise", ha="right", fontsize=9, color="#888")

    # --- counters ------------------------------------------------------------
    fig.text(0.30, 0.165, "hum reduced by", ha="center", fontsize=12, color="#555")
    fig.text(0.72, 0.165, "hiss reduced by", ha="center", fontsize=12, color="#555")
    hum_v = fig.text(0.30, 0.105, "", ha="center", fontsize=30, fontweight="bold")
    hiss_v = fig.text(0.72, 0.105, "", ha="center", fontsize=30, fontweight="bold")
    fig.text(0.5, 0.035, "Best possible causal controller, 32 taps, 2 kHz simulation."
             "  1 sample = 0.5 ms ≈ 17 cm of air.",
             ha="center", fontsize=8.5, color="#777")

    hold, fade = int(HOLD_S * FPS), int(FADE_S * FPS)
    seg = hold + fade
    n_frames = seg * (len(states) - 1) + hold + int(END_HOLD_S * FPS)

    def draw(frame):
        i = min(frame // seg, len(states) - 1)
        k = frame - i * seg
        a = states[i]
        if i < len(states) - 1 and k >= hold:
            b = states[i + 1]
            u = (k - hold) / fade
            u = u * u * (3 - 2 * u)
        else:
            b, u = a, 0.0
        lerp = lambda key: (1 - u) * a[key] + u * b[key]
        cur = a if u < 0.5 else b

        noise_ln.set_ydata(lerp("noise_db"))
        anc_ln.set_ydata(lerp("anc_db"))
        txt, col = margin_text(cur["margin"])
        anc_ln.set_color(col)
        status.set_text(txt)
        status.set_color(col)

        m = (1 - u) * a["margin"] + u * b["margin"]
        x_ref = 4.2 - 0.9 * m           # moves upstream as the head start grows
        ref_dot.set_data([x_ref], [1.5])
        ref_dot.set_color(col)
        ref_lbl.set_x(x_ref)

        hum_v.set_text(fmt_att(lerp("hum_att")))
        hiss_v.set_text(fmt_att(lerp("hiss_att")))
        hum_v.set_color(EARLY)
        hiss_v.set_color(col)
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(OUT, writer=FFMpegWriter(fps=FPS, bitrate=4000,
                                       extra_args=["-pix_fmt", "yuv420p"]))
    draw(n_frames - 1)
    fig.savefig(COVER, dpi=150)
    print(f"Saved {OUT} and {COVER}")


if __name__ == "__main__":
    main()