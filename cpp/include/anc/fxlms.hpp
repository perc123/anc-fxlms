#pragma once

#include <cstddef>
#include <vector>

namespace anc {

// Streaming FxLMS controller with its secondary-path estimate S_hat.
//
// Error convention (shared with the Python code): e[n] = d[n] + anti_noise[n]
// and the update is w <- w - mu * e[n] * x'_hist.
//
// process() mirrors anc/streaming.py exactly. Each call receives x[n] and
// e[n] together; e[n] reflects outputs up to y[n-1]. In this order it:
//   1. updates the weights using the stored filtered-reference history,
//   2. pushes x[n] and x'[n] = (S_hat * x)[n] into the histories,
//   3. returns y[n] = w . x_hist.
// The plant model is not part of this class.
class FxLMS {
public:
    FxLMS(std::size_t filter_length,
          float step_size,
          std::vector<float> secondary_taps_estimate,
          std::size_t secondary_delay_estimate = 0);

    float process(float x, float e);

    void reset();

    const std::vector<float>& weights() const { return weights_; }
    std::size_t filter_length() const { return filter_length_; }

private:
    // Fixed-length history; at(k) is the sample pushed k calls ago.
    class History {
    public:
        explicit History(std::size_t length = 0) { resize(length); }
        void resize(std::size_t length);
        void clear();
        void push(float value);
        float at(std::size_t k) const;

    private:
        std::vector<float> data_;
        std::size_t newest_ = 0;
    };

    std::size_t filter_length_;
    float step_size_;
    std::vector<float> s_hat_;
    std::size_t s_hat_delay_;

    std::vector<float> weights_;
    History x_hist_;
    History xf_hist_;
    History s_in_hist_;
};

}  // namespace anc
