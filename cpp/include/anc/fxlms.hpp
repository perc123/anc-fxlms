#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "anc/history.hpp"

namespace anc {

// Why the controller stopped adapting. Names match the Python twin
// (anc.streaming.Fault) via to_string().
enum class Fault {
    None,
    NonFiniteInput,    // x[n] or e[n] is NaN ot finite
    ErrorLimit,         // |e[n]| exceeded Limits::max_abs_error
    WeightsNotFinite,   // a weight, or the squared weight norm, is not finite
    WeightNormLimit,    // ||w|| exceeded Limits::max_weight_norm
    OutputNotFinite,    // y[n] is NaN or infinite 
};

const char* to_string(Fault fault);

// Optional safety limits. The defaults (infinity) disable both checks, which
// reproduces the unprotected constoller exactly.
struct Limits {
    float max_abs_error = std::numeric_limits<float>::infinity();
    float max_weight_norm = std::numeric_limits<float>::infinity();
};

// Streaming FxLMS controller with its secondary-path estimate S_hat.
//
// Error convention (shared with the Python code): e[n] = d[n] + anti_noise[n]
// and the update is w <- w - mu * e[n] * x'_hist.
//
// process() mirrors anc/streaming.py exactly. Each call receives x[n] and
// e[n] together; e[n] reflects outputs up to y[n-1]. In this order it:
//   1. pushes x[n] and x'[n] = (S_hat * x)[n] into the histories,
//   2. updates the weights with e[n] against the history ending at x'[n],
//   3. returns y[n] = w . x_hist.
//
// Fault behaviour: the first failed check latches the controller in the 
// diverged state. From then on process() returns = 0 (silence for the
// secondary speaker) and changes no state until reset(). If the weight check
// fails, the weights are restored to their values before that update.
// process() never allocates and never throws.
//
// The plant model is not part of this class.
// Specification: docs/algorithm_specification.md, section 5.2.
class FxLMS {
public:
    FxLMS(std::size_t filter_length,
          float step_size,
          std::vector<float> secondary_taps_estimate,
          std::size_t secondary_delay_estimate = 0,
          Limits limits = {});

    float process(float x, float e);

    void reset();

    const std::vector<float>& weights() const { return weights_; }
    std::size_t filter_length() const { return filter_length_; }

    bool diverged() const { return fault_ != Fault::None;}
    Fault fault() const { return fault_; }
    // Index of the call (0-based) on which the fault was detected.
    std::uint64_t fault_sample() const { return fault_sample_; }
    std::uint64_t samples_processed() const { return samples_processed_; }

private:
    float latch(Fault fault, std::uint64_t n);

    std::size_t filter_length_;
    float step_size_;
    std::vector<float> s_hat_;
    std::size_t s_hat_delay_;
    Limits limits_;
    float max_weight_norm_sq_;

    std::vector<float> weights_;
    std::vector<float> previous_weights_;
    History x_hist_;
    History xf_hist_;
    History s_in_hist_;

    Fault fault_ = Fault::None;
    std::uint64_t fault_sample_ = 0;
    std::uint64_t samples_processed_ = 0;
};

}  // namespace anc
