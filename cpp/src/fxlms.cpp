#include "anc/fxlms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace anc {

void FxLMS::History::resize(std::size_t length) {
    data_.assign(length, 0.0f);
    newest_ = 0;
}

void FxLMS::History::clear() {
    std::fill(data_.begin(), data_.end(), 0.0f);
    newest_ = 0;
}

void FxLMS::History::push(float value) {
    newest_ = (newest_ == 0) ? data_.size() - 1 : newest_ - 1;
    data_[newest_] = value;
}

float FxLMS::History::at(std::size_t k) const {
    std::size_t index = newest_ + k;
    if (index >= data_.size()) {
        index -= data_.size();
    }
    return data_[index];
}

FxLMS::FxLMS(std::size_t filter_length,
             float step_size,
             std::vector<float> secondary_taps_estimate,
             std::size_t secondary_delay_estimate)
    : filter_length_(filter_length),
      step_size_(step_size),
      s_hat_(std::move(secondary_taps_estimate)),
      s_hat_delay_(secondary_delay_estimate) {
    if (filter_length_ < 1) {
        throw std::invalid_argument("filter_length must be positive");
    }
    if (!std::isfinite(step_size_) || step_size_ <= 0.0f) {
        throw std::invalid_argument("step_size must be finite and positive");
    }
    if (s_hat_.empty()) {
        throw std::invalid_argument("secondary-path estimate cannot be empty");
    }
    reset();
}

void FxLMS::reset() {
    weights_.assign(filter_length_, 0.0f);
    x_hist_.resize(filter_length_);
    xf_hist_.resize(filter_length_);
    s_in_hist_.resize(s_hat_delay_ + s_hat_.size());
}

float FxLMS::process(float x, float e) {
    // 1. Update with the filtered-reference vector stored by the last call.
    const float step = step_size_ * e;
    for (std::size_t k = 0; k < filter_length_; ++k) {
        weights_[k] -= step * xf_hist_.at(k);
    }

    // 2. Push x[n], then x'[n] = sum_k S_hat[k] * x[n - delay - k].
    x_hist_.push(x);
    s_in_hist_.push(x);

    float x_filtered = 0.0f;
    for (std::size_t k = 0; k < s_hat_.size(); ++k) {
        x_filtered += s_hat_[k] * s_in_hist_.at(s_hat_delay_ + k);
    }
    xf_hist_.push(x_filtered);

    // 3. Controller output.
    float y = 0.0f;
    for (std::size_t k = 0; k < filter_length_; ++k) {
        y += weights_[k] * x_hist_.at(k);
    }
    return y;
}

}  // namespace anc
