// Golden-vector parity: run the C++ controller on every case in
// tests/golden/ and compare against the frozen Python references.
//
//   anc_golden_tests <path-to-tests/golden>
//
// File format and tolerance policy: anc/golden.py and
// docs/algorithm_specification.md, section 7. No Python is involved.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "anc/fxlms.hpp"

namespace fs = std::filesystem;

namespace {
    constexpr int kSupportedFormat = 1;
    constexpr double kDriftMargin = 3.0; // stored tolerance = 3 x drift
    constexpr double kAbsFloor = 1e-7;

    // Removes a trailing carriage return left by CRLF line endings.
    void strip_cr(std::string& line) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
    }

    std::map<std::string, std::string> read_params(const fs::path& path) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("cannot open " + path.string());
        }
        std::map<std::string, std::string> params;
        std::string line;
        while (std::getline(file, line)) {
            strip_cr(line);
            const auto eq = line.find('=');
            if (eq != std::string::npos) {
                params[line.substr(0, eq)] = line.substr(eq + 1);
            }
        }
        return params;
    }

    // Reads a CSV with one header line and `columns` numeric columns.
    // Values are parsed as double (they are exact float64 text) so that the
    // float32 conversion below rounds exactly like numpy.float32(value).
    std::vector<std::vector<double>> read_csv(const fs::path& path, std::size_t columns) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("cannot open " + path.string());
        }
        std::vector<std::vector<double>> data(columns);
        std::string line;
        std::getline(file, line);   // header
        while (std::getline(file, line)) {
            strip_cr(line);
            if (line.empty()) {
                continue;
            }
            std::stringstream stream(line);
            std::string cell;
            for (std::size_t c = 0; c < columns; ++c) {
                if (!std::getline(stream, cell, ',')) {
                    throw std::runtime_error("too few columns in " + path.string());
                }
                data[c].push_back(std::stod(cell));
            }
            if (std::getline(stream, cell, ',')) {
                throw std::runtime_error("too many columns in " + path.string());
            }
        }
        return data;
    }

    std::vector<float> parse_float_list(const std::string& text) {
        std::vector<float> values;
        std::stringstream stream(text);
        std::string item;
        while (std::getline(stream, item, ',')) {
            values.push_back(static_cast<float>(std::stod(item)));
        }
        return values;
    }

    double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
        double worst = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            worst = std::max(worst, std::fabs(a[i] - b[i]));
        }
        return worst;
    }

    // Returns true if the case passes; prints one result line either way.
    bool run_case(const fs::path& dir) {
        const auto params = read_params(dir / "params.txt");
        const std::string name = params.at("name");

        if (std::stoi(params.at("format_version")) != kSupportedFormat) {
            std::cout << "[ FAIL ] " << name << ": unsupported format_version\n";
            return false;
        }

        const auto io = read_csv(dir / "input.csv", 2);
        const auto expected_y = read_csv(dir / "expected_y.csv", 2);
        const auto expected_w = read_csv(dir / "expected_w.csv", 2);

        const std::size_t n = std::stoul(params.at("n_samples"));
        const std::size_t length = std::stoul(params.at("filter_length"));
        if (io[0].size() != n || expected_y[0].size() != n || expected_w[0].size() != length) {
            std::cout << "[ FAIL ] " << name << ": file sizes do not match params.txt\n";
            return false;
        }

        anc::FxLMS controller(
            length,
            static_cast<float>(std::stod(params.at("step_size"))),
            parse_float_list(params.at("s_hat_taps")),
            std::stoul(params.at("s_hat_delay")));

        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = controller.process(static_cast<float>(io[0][i]),
                                    static_cast<float>(io[1][i]));
        }
        const std::vector<double> w(controller.weights().begin(),
                                    controller.weights().end());

        const double y_tol = std::stod(params.at("y_tolerance"));
        const double w_tol = std::stod(params.at("w_tolerance"));
        const double y_drift = std::max(y_tol / kDriftMargin, kAbsFloor);
        const double w_drift = std::max(w_tol / kDriftMargin, kAbsFloor);

        const double y_err64 = max_abs_diff(y, expected_y[0]);
        const double w_err64 = max_abs_diff(w, expected_w[0]);
        const double y_err32 = max_abs_diff(y, expected_y[1]);
        const double w_err32 = max_abs_diff(w, expected_w[1]);

        const bool ok = y_err64 <= y_tol && w_err64 <= w_tol &&
                        y_err32 <= y_drift && w_err32 <= w_drift;

        std::cout << (ok ? "[ PASS ] " : "[ FAIL ] ") << name
                << "  y: " << y_err64 << " / " << y_tol << " (f64), "
                << y_err32 << " / " << y_drift << " (f32)"
                << "  w: " << w_err64 << " / " << w_tol << " (f64), "
                << w_err32 << " / " << w_drift << " (f32)\n";
        return ok;
    }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: anc_golden_tests <path-to-tests/golden>\n";
        return 2;
    }
    const fs::path root = argv[1];
    if (!fs::is_directory(root)) {
        std::cerr << "not a directory: " << root << "\n";
        return 2;
    }

    std::vector<fs::path> cases;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory() && fs::exists(entry.path() / "params.txt")) {
            cases.push_back(entry.path());
        }
    }
    std::sort(cases.begin(), cases.end());

    if (cases.empty()) {
        std::cerr << "no golden cases found in " << root << "\n";
        return 1;   // an empty run must not pass silently
    }

    std::size_t failed = 0;
    for (const auto& dir : cases) {
        try {
            failed += run_case(dir) ? 0 : 1;
        } catch (const std::exception& error) {
            std::cout << "[ FAIL ] " << dir.filename().string()
                      << ": " << error.what() << "\n";
            ++failed;
        }
    }
    std::cout << cases.size() - failed << "/" << cases.size()
              << " golden cases passed\n";
    return failed == 0 ? 0 : 1;
}