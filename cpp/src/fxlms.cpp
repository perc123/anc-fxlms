#include "anc/fxlms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace anc {

const char* to_string(Fault fault) {
    switch (fault) {
        case Fault::None:             return "none";
        case Fault::NonFiniteInput:   return "non_finite_input";
        case Fault::ErrorLimit:       return "error_limit";
        case Fault::WeightsNotFinite: return "weights_not_finite";
        case Fault::WeightNormLimit:  return "weight_norm_limit";
        case Fault::OutputNotFinite:  return "output_not_finite";
    }
    return "unknown";
}

namespace {

// A limit is valid if it is positive (infinity included) and not NaN.
bool valid_limit(float limit) {
    return limit > 0.0f; // false for NaN, zero and negative values
}

} // namespace

FxLMS::FxLMS(std::size_t filter_length,
             float step_size,
             std::vector<float> secondary_taps_estimate,
             std::size_t secondary_delay_estimate,
             Limits limits)
    : filter_length_(filter_length),
      step_size_(step_size),
      s_hat_(std::move(secondary_taps_estimate)),
      s_hat_delay_(secondary_delay_estimate),
      limits_(limits),
      max_weight_norm_sq_(limits.max_weight_norm * limits.max_weight_norm) {
    if (filter_length_ < 1) {
        throw std::invalid_argument("filter_length must be positive");
    }
    if (!std::isfinite(step_size_) || step_size_ <= 0.0f) {
        throw std::invalid_argument("step_size must be finite and positive");
    }
    if (s_hat_.empty()) {
        throw std::invalid_argument("secondary-path estimate cannot be empty");
    }
    if (!std::all_of(s_hat_.begin(), s_hat_.end(), [](float tap) { return std::isfinite(tap); })) {
        throw std::invalid_argument("secondary-path estimate must be finite");
                    }
    if (!valid_limit(limits_.max_abs_error) || !valid_limit(limits_.max_weight_norm))
        throw std::invalid_argument("limits must be positive (or infinity)");
    reset();
}

void FxLMS::reset() {
    weights_.assign(filter_length_, 0.0f);
    previous_weights_.assign(filter_length_, 0.0f);
    x_hist_.resize(filter_length_);
    xf_hist_.resize(filter_length_);
    s_in_hist_.resize(s_hat_delay_ + s_hat_.size());
    fault_ = Fault::None;
    fault_sample_ = 0;
    samples_processed_ = 0;
}

float FxLMS::latch(Fault fault, std::uint64_t n) {
    fault_ = fault;
    fault_sample_ = n;
    return 0.0f;
}

float FxLMS::process(float x, float e) {
    const std::uint64_t n = samples_processed_++;

        if (fault_ != Fault::None) {
        return 0.0f;   // latched: silence until reset()
    }

    // 0. Input checks. A faulted call changes no state except the fault.
    if (!std::isfinite(x) || !std::isfinite(e)) {
        return latch(Fault::NonFiniteInput, n);
    }
    if (std::fabs(e) > limits_.max_abs_error) {
        return latch(Fault::ErrorLimit, n);
    }

    // 1. Push x[n], then x'[n] = sum_k S_hat[k] * x[n - delay - k].
    x_hist_.push(x);
    s_in_hist_.push(x);

    float x_filtered = 0.0f;
    for (std::size_t k = 0; k < s_hat_.size(); ++k) {
        x_filtered += s_hat_[k] * s_in_hist_.at(s_hat_delay_ + k);
    }
    xf_hist_.push(x_filtered);

    // 2. Update with e[n], paired with x'[n] (the newest xf_hist_ entry).
    //    Keep the previous weights so a failed check can restore them.
    std::copy(weights_.begin(), weights_.end(), previous_weights_.begin());

    const float step = step_size_ * e;
    float norm_sq = 0.0f;
    for (std::size_t k = 0; k < filter_length_; ++k) {
        weights_[k] -= step * xf_hist_.at(k);
        norm_sq += weights_[k] * weights_[k];
    }

    // NaN or inf in any weight makes norm_sq non-finite; so does overflow.
    if (!std::isfinite(norm_sq)) {
        weights_.swap(previous_weights_);
        return latch(Fault::WeightsNotFinite, n);
    }
    // Compare squared values: no sqrt per sample.
    if (norm_sq > max_weight_norm_sq_) {
        weights_.swap(previous_weights_);
        return latch(Fault::WeightNormLimit, n);
    }

    // 3. Controller output.
    float y = 0.0f;
    for (std::size_t k = 0; k < filter_length_; ++k) {
        y += weights_[k] * x_hist_.at(k);
    }
    if (!std::isfinite(y)) {
        return latch(Fault::OutputNotFinite, n);
    }
    return y;
}

}  // namespace anc
