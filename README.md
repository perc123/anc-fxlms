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
  controllers.py      FixedFIRController (offline least-squares design),
                      LMSController, FxLMSController
  signals.py          Tone mixture, white noise and band-limited noise
  adaptive.py         run_lms(): direct-error LMS loop
  fxlms.py            run_fxlms(): batch FxLMS loop with divergence checks
  streaming.py        StreamingFxLMS: per-sample reference for the C++ port,
                      with fault latching and safety limits
  measurement.py      Error-microphone noise model and true-vs-measured
                      attenuation (floor_report)
  golden.py           Frozen golden test vectors (python -m anc.golden --write)
  simulate.py         run(): fixed-controller simulation loop
  metrics.py          rms(), attenuation_db()
  convergence.py      analyze_convergence(): block RMS, steady-state
                      attenuation, convergence time
  types.py            Shared FloatArray type alias

experiments/          Experiment scripts (each reads a YAML config)
  step1_fixed_baseline.py     Offline FIR controller (least-squares)
  step2_lms.py                Online LMS adaptive controller
  step2_lms_step_size_sweep.py  LMS step-size sweep (several SNR configs)
  step2_lms_unstable.py       LMS beyond the stability limit
  step2_signal_variants.py    LMS on single tone, multi-tone, white and
                              band-limited noise
  step3_fxlms_mismatch.py     FxLMS with secondary-path mismatch
  step3_causality_sweep.py    FxLMS versus causality margin (primary delay
                              minus secondary delay)
  step3_phase_error_sweep.py  FxLMS versus secondary-path phase error
                              (estimated-delay sweep at one tone)
  step4_benchmark.py          Comparative benchmark across controllers
  plotting.py                 Shared plotting helpers
  post1_clip.py, post2_clip.py  Animated clips for write-ups (need ffmpeg;
                              post2_clip.py also needs SciPy:
                              `pip install -e ".[clips]"`)

configs/              YAML configuration files (one per experiment run)
tests/                pytest suite (Python, C++ CLI and C++ parity)
  golden/             Frozen golden vectors shared by Python and C++ tests
cpp/                  C++ port of the FxLMS controller + S_hat filter (CMake)
  include/, src/      anc::FxLMS library
  tools/anc_cli.cpp   Command-line driver used by the parity tests
  tests/              C++ unit tests and golden-vector runner
  bench/              Per-sample timing benchmark
docs/                 algorithm_specification.md, results.md, figures/
realtime/             Placeholders for secondary-path measurement and live
                      ANC (empty files, not implemented yet)
.github/workflows/    CI: build the C++ port and run pytest
results/              Generated outputs (gitignored)
```

## Installation

Requires Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running experiments

Each experiment reads a YAML config and writes results to `results/` and
figures to `docs/figures/`. Experiments with a `--config` option (all except
step 1, the unstable-LMS run and the clips) accept an alternative file, for
example `configs/step2_lms_step_size_sweep_snr60.yaml`.

```bash
python -m experiments.step1_fixed_baseline
python -m experiments.step2_lms
python -m experiments.step2_lms_step_size_sweep
python -m experiments.step2_lms_unstable
python -m experiments.step2_signal_variants
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

**A note on the attenuation figures.** Early simulations used an idealised,
noise-free error sensor and reported values up to 255 dB. With a modelled
60 dB sensor SNR, measured steady-state attenuation is about 60 dB (floor-limited), while the
true acoustic residual lies about 73 dB below the disturbance. See
[docs/results.md](docs/results.md#measurement-floor-true-vs-measured-attenuation)
for the model, the revised conclusions and the limitations.

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
outputs up to y[n-1]. Each call (1) pushes x[n] and x'[n] into the histories,
(2) updates the weights with e[n] against the history ending at x'[n], and
(3) returns y[n]. `anc/streaming.py` is the Python reference with exactly that
order. The order matters: updating before pushing x'[n] pairs e[n] with
x'[n-1] and shifts the stability window of the phase-error sweep by one
sample of S_hat delay. The batch `run_fxlms` computes y[n], then e[n], then
updates in the same step, so its y[n] uses weights one update older; it has the
same stability window, and `tests/test_streaming_phase_sweep.py` checks that
the streaming path reproduces it.

Safety: the controller latches on the first fault (non-finite input, error
above `max_abs_error`, non-finite or oversized weights, non-finite output) and
returns 0 until `reset()`. Both limits are optional and disabled by default.
Python and C++ use the same fault names. See
[docs/algorithm_specification.md](docs/algorithm_specification.md), section 8.1.

### Build and test

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure   # C++ unit + golden tests
pytest tests/test_cpp_parity.py tests/test_cli.py
```

### Command-line driver

`anc_cli` runs the C++ controller open loop over an `x,e` CSV file (an
optional `x,e` header line is skipped):

```bash
cpp/build/anc_cli --input tests/golden/short_debug/input.csv \
    --output-y y.csv --output-weights w.csv \
    --length 4 --mu 0.05 --shat 0.7,0.2 --delay 1 \
    [--max-abs-error E] [--max-weight-norm W]
```

It prints `fault=<name> fault_sample=<n> samples=<n>`. Exit codes: 0 ok,
1 runtime error, 2 usage error, 3 controller fault (outputs are still
written). `cpp/build/bench_fxlms` reports nanoseconds per sample.

### Golden vectors and parity

The parity test is open loop. Python simulates the loop once and records x
and e as golden vectors under `tests/golden/` (three cases: `short_debug`,
`tone_mixture`, `tone_mixture_snr40`). Both implementations consume those
same sequences, so float32 differences cannot compound through the feedback
path. The tests compare y[n] and the final weights. The tolerance is stored
with each case and derived from running the Python reference in float32: the
C++ error against float64 must stay within 3x of that float32-vs-float64
drift, with an absolute floor of 1e-7. The C++ golden runner
(`anc_golden_tests`) checks the same files without Python. The files are the
contract: regenerate them only deliberately, with
`python -m anc.golden --write`.

## Tests

```bash
pytest
```

The suite includes the C++ parity and CLI tests, which build `anc_cli` with
CMake in a temporary directory and are skipped when `cmake` is not installed.
CI runs the same `pytest` after building the C++ port.

## Experiment steps

| Step | Controller | Config |
| ---- | ---------- | ------ |
| 1 | Fixed FIR (offline least-squares) | `configs/step1_fixed.yaml` |
| 2 | Online LMS | `configs/step2_lms.yaml` |
| 3 | FxLMS with secondary-path mismatch | `configs/step3_fxlms_mismatch.yaml` |
| 3 | FxLMS causality sweep | `configs/step3_causality_sweep.yaml` |
| 3 | FxLMS phase-error sweep | `configs/step3_phase_error_sweep.yaml` |
| 4 | Benchmark | `configs/benchmark.yaml` |

Additional configs: `step2_lms_step_size_sweep*.yaml` (step-size sweeps at
several SNRs), `step2_lms_unstable.yaml` and `step2_signal_variants.yaml`.

## Further reading

- [docs/algorithm_specification.md](docs/algorithm_specification.md): the FxLMS definition shared by Python and C++
- [docs/results.md](docs/results.md): measurement-floor model and sweep findings

## Dependencies

| Package | Version |
| ------- | ------- |
| numpy | 2.2.6 |
| matplotlib | 3.10.3 |
| PyYAML | 6.0.2 |
| SciPy (`clips` extra) | >=1.15, only for `experiments/post2_clip.py` |
| pytest (dev) | 8.4.1 |
