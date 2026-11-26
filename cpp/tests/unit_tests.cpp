// Unit tests for the C++ FxLMS building blocks.
// Each test names the specification section it checks
// (docs/algorithm_specification.md).

#include <cmath>
#include <limits>
#include <vector>

#include "anc/fxlms.hpp"
#include "anc/history.hpp"
#include "test_support.hpp"

namespace {
    constexpr double kExact = 1e-7; // float 32 results of exact decimal inputs
}

// ---------------------------------------------------------------------------
// History (spec section 2: newest-first, zero initial state)
//

TEST_CASE("History starts at zero") {
    anc::History h(3);
    CHECK(h.size() == 3);
    for (std::size_t k = 0; k < 3; ++k) {
        CHECK(h.at(k) == 0.0f);
    }
}

TEST_CASE("History is newest-first") {
    anc::History h(3);
    h.push(1.0f);
    h.push(2.0f);
    CHECK(h.at(0) == 2.0f);
    CHECK(h.at(1) == 1.0f);
    CHECK(h.at(2) == 0.0f); // not yet written: still the zero initial state
}

TEST_CASE("History wraps around and drops the older sample") {
    anc::History h(3);
    for (int value = 1; value <= 5; ++value) {
        h.push(static_cast<float>(value));
    }
    // Pushed 1..5 into length 3: only 5, 4, 3 remain
    CHECK(h.at(0) == 5.0f);
    CHECK(h.at(1) == 4.0f);
    CHECK(h.at(2) == 3.0f);
}

TEST_CASE("History of length 1 holds only the newest sample") {
    anc::History h(1);
    h.push(7.0f);
    h.push(8.0f);
    CHECK(h.at(0) == 8.0f);
}

TEST_CASE("History of length 0 ignores pushes and reads as zero") {
    anc::History h;   // default length is 0
    CHECK(h.size() == 0);
    h.push(1.0f);
    CHECK(h.size() == 0);
    CHECK(h.at(0) == 0.0f);
}

TEST_CASE("History clear restores the zero state") {
    anc::History h(2);
    h.push(1.0f);
    h.push(2.0f);
    h.clear();
    CHECK(h.at(0) == 0.0f);
    CHECK(h.at(1) == 0.0f);
    h.push(3.0f);
    CHECK(h.at(0) == 3.0f);
    CHECK(h.at(1) == 0.0f);
}

// ---------------------------------------------------------------------------
// FxLMS validation (spec section 8)
// ---------------------------------------------------------------------------

TEST_CASE("FxLMS rejects invalid parameters") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(anc::FxLMS(0, 0.01f, {1.0f}));
    CHECK_THROWS(anc::FxLMS(4, 0.0f, {1.0f}));
    CHECK_THROWS(anc::FxLMS(4, -0.01f, {1.0f}));
    CHECK_THROWS(anc::FxLMS(4, nan, {1.0f}));
    CHECK_THROWS(anc::FxLMS(4, 0.01f, {}));
}

TEST_CASE("FxLMS starts with zero weights") {
    anc::FxLMS controller(4, 0.01f, {1.0f});
    CHECK(controller.filter_length() == 4);
    CHECK(controller.weights().size() == 4);
    for (float w : controller.weights()) {
        CHECK(w == 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Per-sample order (spec sections 4 and 5.2), computed by hand
// ---------------------------------------------------------------------------

TEST_CASE("Update follows the filtered push (C++ twin of the Python test)") {
    // L = 2, mu = 0.1, S_hat = [1], no delay, so x'[n] = x[n].
    anc::FxLMS controller(2, 0.1f, {1.0f});

    // n = 0: x' hist = [1, 0]; w = 0 - 0.1 * 0.5 * [1, 0] = [-0.05, 0]
    //        y = w . [1, 0] = -0.05   (uses the UPDATED weights)
    float y = controller.process(1.0f, 0.5f);
    CHECK_NEAR(controller.weights()[0], -0.05f, kExact);
    CHECK_NEAR(controller.weights()[1], 0.0f, kExact);
    CHECK_NEAR(y, -0.05, kExact);

    // n = 1: x' hist = [2, 1]; w = [-0.05, 0] - 0.1 * 1.0 * [2, 1] = [-0.25, -0.10]
    //        y = -0.25 * 2 + -0.10 * 1 = -0.6
    y = controller.process(2.0f, 1.0f);
    CHECK_NEAR(controller.weights()[0], -0.25, kExact);
    CHECK_NEAR(controller.weights()[1], -0.10, kExact);
    CHECK_NEAR(y, -0.6, kExact);
}

TEST_CASE("S_hat delay shifts the filtered reference") {
    // L = 1, S_hat = [1], delay 2, so x'[n] = x[n - 2].
    // Feed one impulse x = [1, 0, 0, 0] with e = 1 throughout. The weight can
    // only change once x' = 1, which happens at n = 2.
    anc::FxLMS controller(1, 0.5f, {1.0f}, 2);
    const float x[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const double expected_w[] = {0.0, 0.0, -0.5, -0.5};

    for (int n = 0; n < 4; ++n) {
        controller.process(x[n], 1.0f);
        CHECK_NEAR(controller.weights()[0], expected_w[n], kExact);
    }
}

TEST_CASE("S_hat taps are applied in order") {
    // L = 1, S_hat = [0.5, 0.25], no delay: x'[n] = 0.5 x[n] + 0.25 x[n-1].
    // Impulse x = [1, 0, 0], e = 1: x' = [0.5, 0.25, 0], mu = 1.
    anc::FxLMS controller(1, 1.0f, {0.5f, 0.25f});
    controller.process(1.0f, 1.0f);
    CHECK_NEAR(controller.weights()[0], -0.5, kExact);
    controller.process(0.0f, 1.0f);
    CHECK_NEAR(controller.weights()[0], -0.75, kExact);
    controller.process(0.0f, 1.0f);
    CHECK_NEAR(controller.weights()[0], -0.75, kExact);
}

TEST_CASE("reset restores the initial state") {
    anc::FxLMS controller(2, 0.1f, {1.0f});
    controller.process(1.0f, 0.5f);
    controller.process(2.0f, 1.0f);
    controller.reset();

    for (float w : controller.weights()) {
        CHECK(w == 0.0f);
    }
    // After reset, the first call must behave exactly like the first call
    // of a fresh controller: histories are zero again too.
    const float y = controller.process(1.0f, 0.5f);
    CHECK_NEAR(y, -0.05, kExact);
    CHECK_NEAR(controller.weights()[1], 0.0, kExact);
}

// ---------------------------------------------------------------------------
// Fault behaviour (spec section 8.1). Each scenario has a Python twin in
// tests/test_streaming_faults.py with the same inputs and expectations.
// ---------------------------------------------------------------------------

TEST_CASE("Default limits: a healthy run reports no fault") {
    anc::FxLMS controller(2, 0.1f, {1.0f});
    controller.process(1.0f, 0.5f);
    controller.process(2.0f, 1.0f);
    CHECK(!controller.diverged());
    CHECK(controller.fault() == anc::Fault::None);
    CHECK(controller.samples_processed() == 2);
}

TEST_CASE("NaN error latches non_finite_input and leaves the weights untouched") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    anc::FxLMS controller(2, 0.1f, {1.0f});
    controller.process(1.0f, 0.5f);           // n = 0: w = [-0.05, 0]
    const float y = controller.process(2.0f, nan);   // n = 1: fault

    CHECK(y == 0.0f);
    CHECK(controller.diverged());
    CHECK(controller.fault() == anc::Fault::NonFiniteInput);
    CHECK(controller.fault_sample() == 1);
    CHECK_NEAR(controller.weights()[0], -0.05, kExact);
    CHECK_NEAR(controller.weights()[1], 0.0, kExact);
}

TEST_CASE("Infinite reference latches non_finite_input") {
    const float inf = std::numeric_limits<float>::infinity();
    anc::FxLMS controller(2, 0.1f, {1.0f});
    CHECK(controller.process(inf, 0.5f) == 0.0f);
    CHECK(controller.fault() == anc::Fault::NonFiniteInput);
    CHECK(controller.fault_sample() == 0);
}

TEST_CASE("Error above the limit latches error_limit before any update") {
    anc::Limits limits;
    limits.max_abs_error = 10.0f;
    anc::FxLMS controller(2, 0.1f, {1.0f}, 0, limits);

    controller.process(1.0f, 0.5f);                    // n = 0: fine
    CHECK(controller.process(1.0f, 10.0f) != 0.0f);    // n = 1: |e| = limit is allowed
    const auto w_before = controller.weights();
    CHECK(controller.process(1.0f, -10.5f) == 0.0f);   // n = 2: |e| > limit

    CHECK(controller.fault() == anc::Fault::ErrorLimit);
    CHECK(controller.fault_sample() == 2);
    CHECK(controller.weights() == w_before);
}

TEST_CASE("Weight norm above the limit restores the previous weights") {
    // L = 1, mu = 1, S_hat = [1], x = 1, e = -4: w grows 4, 8, 12, ...
    // With max_weight_norm = 10 the update at n = 2 (w = 12) is rejected.
    anc::Limits limits;
    limits.max_weight_norm = 10.0f;
    anc::FxLMS controller(1, 1.0f, {1.0f}, 0, limits);

    CHECK_NEAR(controller.process(1.0f, -4.0f), 4.0, kExact);   // n = 0
    CHECK_NEAR(controller.process(1.0f, -4.0f), 8.0, kExact);   // n = 1
    CHECK(controller.process(1.0f, -4.0f) == 0.0f);             // n = 2

    CHECK(controller.fault() == anc::Fault::WeightNormLimit);
    CHECK(controller.fault_sample() == 2);
    CHECK_NEAR(controller.weights()[0], 8.0, kExact);   // restored
}

TEST_CASE("Overflowing weights latch weights_not_finite and are restored") {
    // mu * e * x' overflows float32 on the first update.
    anc::FxLMS controller(1, 1.0f, {1.0f});
    CHECK(controller.process(3.0e38f, -3.0e38f) == 0.0f);
    CHECK(controller.fault() == anc::Fault::WeightsNotFinite);
    CHECK(controller.weights()[0] == 0.0f);
}

TEST_CASE("A latched controller outputs silence and changes no state") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    anc::FxLMS controller(2, 0.1f, {1.0f});
    controller.process(1.0f, 0.5f);
    controller.process(1.0f, nan);   // n = 1: fault
    const auto w_latched = controller.weights();

    for (int i = 0; i < 5; ++i) {
        CHECK(controller.process(1.0f, 0.5f) == 0.0f);   // healthy inputs
    }
    CHECK(controller.weights() == w_latched);
    CHECK(controller.fault_sample() == 1);   // the first fault is kept
    CHECK(controller.samples_processed() == 7);
}

TEST_CASE("reset clears the fault") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    anc::FxLMS controller(2, 0.1f, {1.0f});
    controller.process(nan, 0.5f);
    CHECK(controller.diverged());

    controller.reset();
    CHECK(!controller.diverged());
    CHECK(controller.samples_processed() == 0);
    CHECK_NEAR(controller.process(1.0f, 0.5f), -0.05, kExact);   // as fresh
}

TEST_CASE("Constructor rejects non-finite S_hat and invalid limits") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(anc::FxLMS(2, 0.1f, {1.0f, nan}));

    anc::Limits zero_error;
    zero_error.max_abs_error = 0.0f;
    CHECK_THROWS(anc::FxLMS(2, 0.1f, {1.0f}, 0, zero_error));

    anc::Limits nan_norm;
    nan_norm.max_weight_norm = nan;
    CHECK_THROWS(anc::FxLMS(2, 0.1f, {1.0f}, 0, nan_norm));

    anc::Limits negative_norm;
    negative_norm.max_weight_norm = -1.0f;
    CHECK_THROWS(anc::FxLMS(2, 0.1f, {1.0f}, 0, negative_norm));
}

TEST_CASE("Fault names match the Python twin") {
    CHECK(std::string(anc::to_string(anc::Fault::None)) == "none");
    CHECK(std::string(anc::to_string(anc::Fault::NonFiniteInput)) == "non_finite_input");
    CHECK(std::string(anc::to_string(anc::Fault::ErrorLimit)) == "error_limit");
    CHECK(std::string(anc::to_string(anc::Fault::WeightsNotFinite)) == "weights_not_finite");
    CHECK(std::string(anc::to_string(anc::Fault::WeightNormLimit)) == "weight_norm_limit");
    CHECK(std::string(anc::to_string(anc::Fault::OutputNotFinite)) == "output_not_finite");
}

int main() { return anc_test::run_all(); }