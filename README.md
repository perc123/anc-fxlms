# anc-fxlms

A reproducible Active Noise Cancellation (ANC) DSP portfolio project, progressing from a fixed offline FIR controller through adaptive LMS/FxLMS algorithms to a real-time C++ implementation.

## Overview

The project models a single-channel feedforward ANC system: a reference microphone picks up the incoming noise, a controller generates an anti-noise signal, a loudspeaker drives the secondary path, and an error microphone measures the residual. All components are implemented as linear time-invariant FIR models, making results fully reproducible.

```text
reference ──► Controller ──► secondary path ──► anti-noise ──►┐
                                                               + ──► residual (error mic)
                              primary path  ──► primary noise ─┘
```

## Project structure

```text
anc/                  Core DSP library
  plant.py            ANCPlant: primary and secondary path FIR models
  controllers.py      FixedFIRController + offline least-squares design
  signals.py          Tone mixture and white-noise signal generation
  adaptive.py         run_lms(): direct-error LMS loop
  fxlms.py            run_fxlms(): batch FxLMS loop with divergence checks
  streaming.py        StreamingFxLMS: per-sample reference for the C++ port
  simulate.py         run(): fixed-controller simulation loop
  metrics.py          rms(), attenuation_db()
  convergence.py      analyze_convergence(): block RMS, steady-state
                      attenuation, convergence time
  types.py            Shared FloatArray type alias

experiments/          Numbered experiment scripts
  step1_fixed_baseline.py   Offline FIR controller (least-squares)
  step2_lms.py              Online LMS adaptive controller
  step3_fxlms_mismatch.py   FxLMS with secondary-path mismatch
  step3_causality_sweep.py  FxLMS versus causality margin (primary delay
                            minus secondary delay)
  step3_phase_error_sweep.py  FxLMS versus secondary-path phase error
                            (estimated-delay sweep at one tone)
  step4_benchmark.py        Comparative benchmark across controllers
  plotting.py               Shared plotting helpers

configs/              YAML configuration files (one per experiment)
tests/                pytest suite
cpp/                  C++ port of the FxLMS controller + S_hat filter (CMake)
realtime/             Placeholders for secondary-path measurement and live
                      ANC (empty, not implemented yet)
.github/workflows/    CI: build the C++ port and run pytest
results/              Generated outputs (gitignored)
docs/figures/         Generated figures
```

## Installation

Requires Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running experiments

Each experiment reads a YAML config and writes results to `results/`.

```bash
python -m experiments.step1_fixed_baseline
python -m experiments.step2_lms
python -m experiments.step3_fxlms_mismatch
python -m experiments.step3_causality_sweep
python -m experiments.step3_phase_error_sweep
python -m experiments.step4_benchmark
```

## Conventions

Every controller in the project uses the same sign convention. The error
(residual) is the sum of the unwanted noise and the anti-noise,

```text
e[n] = d[n] + y'[n]        w <- w - mu * e[n] * x[n]      (x' for FxLMS)
```

so a converged filter produces the *negative* of the unwanted signal.
`run_lms` follows this too: its filter converges to the negative of the
path that generates the desired signal.

## Results

All results use a 2 kHz sample rate and a 4-second mixture of 80, 120 and
160 Hz tones plus white noise (std 0.03), a 32-tap controller, and a 0.5 s
transient excluded from the evaluation.

The causality and phase-error sweeps use their own signals and are described
in their sections below.

| Step | Method | Attenuation |
| ---- | ------ | ----------- |
| 1 | Fixed FIR, offline least squares | 24.5 dB |
| 2 | LMS, direct error (no secondary path) | 15.4 dB |
| 3 | FxLMS, mismatched S_hat | 18.8 dB |
| 4 | Benchmark, 5 seeds, held-out test signal | see below |

Step 4 (mean +/- std over seeds 42-46):

| Method | Attenuation |
| ------ | ----------- |
| Fixed FIR | 24.43 +/- 0.11 dB |
| FxLMS, matched S_hat | 18.98 +/- 0.04 dB |
| FxLMS, mismatched S_hat | 18.79 +/- 0.04 dB |

The mismatch costs only ~0.2 dB, so FxLMS tolerates the modeled error in the
secondary-path estimate. The fixed FIR is better because it is a one-shot
least-squares fit on the whole signal, whereas FxLMS pays a misadjustment
cost for tracking with a small step size. The Step 2 figure is not
comparable to the others: it has no secondary path.

### Why Step 1 stops at ~24.5 dB

The tones are cancelled almost completely (see `docs/figures/step1_psd.png`,
where the three peaks drop by roughly 50 dB). The limit comes from the white
noise. The primary path has no delay (its first tap is 0.80) while the
secondary path adds 2 samples, so the controller would need to see two samples
into the future to cancel the broadband part of the noise. A causal controller
cannot, and the sinusoids are the only part predictable enough to cancel. The
unpredictable noise through the primary path has an RMS of about 0.025 against
a 0.52 total, which caps the attenuation near 26.5 dB. The fit also redistributes
noise: the residual PSD is a few dB above the primary PSD between roughly 200
and 600 Hz.

### Step 3 causality sweep

`step3_causality_sweep.py` runs FxLMS on white noise (no tones) with a
2-sample secondary delay and varies the primary-path delay, so the causality
margin (primary delay minus secondary delay) ranges from -2 to +4 samples,
over 5 seeds. With a negative margin the anti-noise cannot arrive in time and
attenuation is about 0 dB (0.01 +/- 0.03 dB). From a margin of 0 upward it is
roughly 30 to 33 dB (33.3 dB at margin 0, 30.5 dB at +4).

### Step 3 phase-error sweep

`step3_phase_error_sweep.py` drives a single 125 Hz tone through a plant with
an 8-sample secondary delay and a 12-sample primary delay (margin +4). The
secondary-path estimate S_hat is given a delay from 0 to 16 samples, which
produces phase errors from -180 to +180 degrees at 125 Hz, 5 seeds each. In
this run every seed converged for phase errors from -67.5 to +67.5 degrees and
every seed diverged at +/-90 degrees and beyond, consistent with the usual
+/-90 degree stability limit of FxLMS.

#### Note on large attenuation values

The sweep uses a deterministic sinusoidal reference and no sensor noise, so
the residual can become extremely small and the raw attenuation reaches well
above 100 dB. Those numbers are not achievable physical ANC performance. The
figure `docs/figures/step3_phase_error_sweep.png` clips the plotted
attenuation at 60 dB and labels this as a visualization limit. The unclipped
measurements remain in `results/step3_phase_error_sweep/per_seed_results.csv`
for reproducibility.

## C++ port

`cpp/` ports only the controller and the S_hat filter. The plant model stays
in Python as part of the test setup.

The interface is `float FxLMS::process(float x, float e)`, in the
audio-callback order: x[n] and e[n] arrive together and e[n] only reflects
outputs up to y[n-1]. Each call (1) updates the weights with the stored
filtered-reference history, (2) pushes x[n] and x'[n] into the histories, and
(3) returns y[n]. `anc/streaming.py` is the Python reference with exactly that
order. The batch `run_fxlms` computes y[n], then e[n], then updates in the same
step, so it is a different algorithm and is not what the port is compared to.

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
pytest tests/test_cpp_parity.py
```

The parity test is open-loop. Python simulates the loop once, records x and e
under `tests/data/`, and both implementations consume those same sequences,
so float32 differences cannot compound through the feedback path. It compares
y[n] and the final weights. The tolerance is derived from running the Python
reference in float32 once: the C++ error against float64 must stay within 3x
of that float32-vs-float64 drift (about 2.4e-6 on y for this configuration).

## Tests

```bash
pytest
```

## Experiment steps

| Step | Controller | Config |
| ---- | ---------- | ------ |
| 1 | Fixed FIR (offline least-squares) | `configs/step1_fixed.yaml` |
| 2 | Online LMS | `configs/step2_lms.yaml` |
| 3 | FxLMS with secondary-path mismatch | `configs/step3_fxlms_mismatch.yaml` |
| 3 | FxLMS causality sweep | `configs/step3_causality_sweep.yaml` |
| 3 | FxLMS phase-error sweep | `configs/step3_phase_error_sweep.yaml` |
| 4 | Benchmark | `configs/benchmark.yaml` |

## Dependencies

| Package | Version |
| ------- | ------- |
| numpy | 2.2.6 |
| matplotlib | 3.10.3 |
| PyYAML | 6.0.2 |
| pytest (dev) | 8.4.1 |
