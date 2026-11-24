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

TEST_CASE("History wraps around and drops the older sample") {
    anc::History h(1);
}