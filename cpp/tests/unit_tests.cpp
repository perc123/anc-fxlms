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
    CHECK(h.at(0) == 1.0f);
    CHECK(h.at(0) == 0.0f); // not yet written: still the zero initial state
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

int main() { return anc_test::run_all(); }