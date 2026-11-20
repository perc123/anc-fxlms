# FxLMS algorithm specification

This document defines the FxLMS algorithm as implemented in this project, so
that the Python reference and the C++ port can be checked against one
definition. Executable checks of the main claims live in
`tests/test_spec_equivalence.py`.

## 1. Implementations

| Implementation | File | Role |
|---|---|---|
| Batch harness | `anc/fxlms.py` (`run_fxlms`) | Closed-loop simulation with the plant, safety checks and measurement noise. Used by all experiments. |
| Streaming reference | `anc/streaming.py` (`StreamingFxLMS`) | Per-sample controller in the audio-callback order. The reference for the C++ port. |
| C++ port | `cpp/include/anc/fxlms.hpp`, `cpp/src/fxlms.cpp` (`anc::FxLMS`) | float32 streaming controller. Mirrors `StreamingFxLMS` sample for sample. |

The plant (primary and secondary path) is simulation support. It is part of
the batch harness and of `run_streaming_closed_loop`, never of the C++ class.

## 2. Notation and conventions

| Symbol | Meaning |
|---|---|
| x[n] | Reference signal |
| d[n] | Disturbance at the error microphone, d = P(z) x |
| y[n] | Controller output (control signal) |
| y_s[n] | Anti-noise at the error microphone, y_s = S(z) y |
| r[n] | Acoustic residual, r = d + y_s (simulation-only) |
| v[n] | Measurement noise of the error microphone |
| e[n] | Error signal used for adaptation |
| x'[n] | Filtered reference, x' = Ŝ(z) x |
| w | Controller weights, length L |
| μ | Step size |

**Sign convention.** The anti-noise is *added* at the microphone:
r[n] = d[n] + y_s[n]. The update therefore *subtracts*:

    w ← w − μ · e[n] · x'_hist

**FIR taps.** Tap k multiplies the sample from k samples ago:
(h * u)[n] = Σ_k h[k] · u[n − k].

**Delays.** An explicit integer delay D is applied in addition to the taps:
x'[n] = Σ_k ŝ[k] · x[n − D − k]. The same applies to the actual secondary
path S with its delay D_s.

**Histories.** All history vectors are newest-first: index k holds the sample
from n − k. Weight w[k] multiplies x[n − k].

**Initial state.** All signals are zero for n < 0. Weights start at zero.
All histories start at zero.

## 3. Signal model

    d[n]   = Σ_k p[k] · x[n − k]
    y[n]   = Σ_k w[k] · x[n − k]
    y_s[n] = Σ_k s[k] · y[n − D_s − k]
    r[n]   = d[n] + y_s[n]
    e[n]   = r[n] + v[n]
    x'[n]  = Σ_k ŝ[k] · x[n − D̂ − k]

S (taps s, delay D_s) is the actual secondary path; Ŝ (taps ŝ, delay D̂) is
the controller's estimate. They may differ: that is how secondary-path
mismatch and phase error are modelled.

v[n] is supplied as an array to the batch harness (`measurement_noise`;
zero if omitted). The streaming controller and the C++ port receive e[n]
directly, so any measurement noise is already part of the input.

## 4. Update rule

    w ← w − μ · e[n] · [x'[n], x'[n−1], ..., x'[n−L+1]]

e[n] is always paired with the filtered-reference history that **ends at
x'[n]**. Pairing it with the history ending at x'[n−1] shifts the phase-error
stability window by one sample of Ŝ delay.

## 5. Per-sample order

### 5.1 Batch harness (`run_fxlms`)

For n = 0, 1, ...:

1. Push x[n] into the reference history.
2. y[n] = w · x_hist (w has been updated through e[n−1]).
3. y_s[n] from the actual secondary path (with D_s = 0, tap 0 acts on y[n]).
4. r[n] = d[n] + y_s[n];  e[n] = r[n] + v[n].
5. Safety: if |e[n]| > max_abs_residual, stop (diverged) before updating.
6. Push x'[n] into the filtered-reference history.
7. Update w with e[n] (section 4).
8. Safety: if ‖w‖ > max_weight_norm, stop (diverged).

x' is computed for the whole run up front (`apply_fir`); this yields the
same values as computing it per sample.

### 5.2 Streaming reference and C++ port (`process(x[n], e[n])`)

The caller supplies e[n], which reflects outputs up to y[n−1].

1. Push x[n] into the reference history and the Ŝ input history.
2. x'[n] = Σ_k ŝ[k] · x[n − D̂ − k]; push it into the filtered-reference history.
3. Update w with e[n] (section 4).
4. Return y[n] = w · x_hist.

### 5.3 Relationship between the two orders

- Both apply the same update with the same pairing (e[n] with x'[n]).
  Fed the same (x, e) sequence, they follow the same weight trajectory.
- Batch y[n] uses the weights *before* the update with e[n]; streaming
  y[n] uses them *after*. Closed-loop outputs therefore differ slightly
  between the two.
- Experiments use the batch harness. The C++ port is verified against the
  streaming reference, not against the batch harness.

## 6. Causality requirements

- Batch harness: D_s ≥ 0. With D_s = 0, y[n] reaches the microphone in the
  same sample, which the harness can model because it computes e[n] itself.
- Streaming: D_s ≥ 1. e[n] arrives together with x[n] and cannot depend on
  y[n]. `run_streaming_closed_loop` rejects D_s = 0.

## 7. Numerical precision and parity policy

| Implementation | Type | Accumulation order |
|---|---|---|
| Python (both) | float64 | NumPy dot product |
| C++ | float32 | k = 0, 1, ..., K−1 |

Rationale for float32 in C++: TODO (your decision, in one or two sentences).

Parity test (`tests/test_cpp_parity.py`), open loop: Python simulates the
closed loop once in float64 and records (x, e). Both implementations are
then fed the same recorded sequence, so differences cannot compound through
the loop.

- Tolerance vs Python float64: 3 × (float32 − float64 drift of the Python
  reference), with an absolute floor of 1e-7.
- C++ vs Python float32: within the float32 drift (only summation order
  differs).

## 8. Validation and safety

| Rule | Batch | Streaming | C++ |
|---|---|---|---|
| L ≥ 1 | ✓ | ✓ | ✓ |
| μ finite and > 0 | ✓ | ✓ | ✓ |
| D̂ ≥ 0 | ✓ | ✓ | by type (`size_t`) |
| Ŝ non-empty | ✓ | ✓ | ✓ |
| Ŝ finite | ✓ | ✓ | ✗ |
| x[n], e[n] finite | ✓ (reference, residual) | ✗ | ✗ |
| Residual amplitude limit | ✓ | ✗ | ✗ |
| Weight-norm limit | ✓ | ✗ | ✗ |
| Non-finite weights detected | ✓ | ✗ | ✗ |

On divergence, the batch harness stops, marks the run "diverged", records
the reason and sample, and leaves NaN in the unprocessed tail.

**Known gap:** the streaming reference and the C++ port have no safety
behaviour. Defining it (detection, and what `process()` returns afterwards)
is the next planned change.

## Appendix: direct-error LMS (not ported)

`anc/adaptive.py` (`run_lms`) uses e[n] = d[n] − y[n] and
w ← w + μ · e[n] · x_hist. Because `LMSController.update()` subtracts its
argument, `run_lms` passes −e[n]. This experiment is system identification,
not ANC, and has no streaming or C++ counterpart.