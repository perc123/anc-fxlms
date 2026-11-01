#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "anc/fxlms.hpp"

int main() {
    constexpr std::size_t kSamples = 2'000'000;
    anc::FxLMS controller(32, 0.005f, {0.66f, 0.0f, 0.23f, 0.0f, -0.03f}, 2);

    float sink = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t n = 0; n < kSamples; ++n) {
        const float x = std::sin(0.05f * static_cast<float>(n));
        sink += controller.process(x, 0.1f * x);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const double ns_per_sample =
        std::chrono::duration<double, std::nano>(elapsed).count() / kSamples;
    std::cout << ns_per_sample << " ns/sample (checksum " << sink << ")\n";
    return 0;
}
