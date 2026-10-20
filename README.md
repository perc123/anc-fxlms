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
  simulate.py         run(): fixed-controller simulation loop
  metrics.py          rms(), attenuation_db()

experiments/          Numbered experiment scripts
  step1_fixed_baseline.py   Offline FIR controller (least-squares)
  step2_lms.py              Online LMS adaptive controller
  step3_fxlms_mismatch.py   FxLMS with secondary-path mismatch
  step4_benchmark.py        Comparative benchmark across controllers

configs/              YAML configuration files (one per experiment)
tests/                pytest suite
cpp/                  C++ re-implementation (CMake)
realtime/             Real-time scripts (secondary-path measurement, live ANC)
results/              Generated outputs (gitignored)
docs/figures/         Generated figures (gitignored)
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
python -m experiments.step4_benchmark
```

The step 1 baseline achieves ~24.5 dB attenuation at 2 kHz sample rate on a 4-second tone mixture (80, 120, 160 Hz) with a 32-tap offline FIR controller.

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
| 4 | Benchmark | `configs/benchmark.yaml` |

## Dependencies

| Package | Version |
| ------- | ------- |
| numpy | 2.2.6 |
| matplotlib | 3.10.3 |
| PyYAML | 6.0.2 |
| pytest (dev) | 8.4.1 |
