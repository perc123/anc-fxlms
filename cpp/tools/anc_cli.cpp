// Open-loop driver for the streaming FxLMS controller.
//
//   anc_cli --input io.csv --output-y y.csv --output-weights w.csv
//           --length L --mu MU --shat a,b,c [--delay D]
//           [--max-abs-error E] [--max-weight-norm W]
//   anc_cli --help
//
// io.csv holds one "x,e" pair per line. An optional first line that is not
// numeric (for example "x,e") is treated as a header and skipped, so golden
// input files can be passed directly. Values are parsed as double and then
// rounded to float, exactly like numpy.float32(value).
//
// Output: y[n], one per line (zeros after a fault), and the final weights,
// one per line, each with enough digits to round-trip a float. A final status
// line goes to stdout: "fault=<name> fault_sample=<n> samples=<n>".
//
// Exit codes:
//   0  completed, no fault
//   1  runtime error (file cannot be opened or written, malformed input line)
//   2  usage error (unknown, missing or invalid option)
//   3  completed, but the controller latched a fault (outputs are written)

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "anc/fxlms.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitRuntime = 1;
constexpr int kExitUsage = 2;
constexpr int kExitFault = 3;

const char* const kUsage =
    "usage: anc_cli --input FILE --output-y FILE --output-weights FILE\n"
    "               --length L --mu MU --shat a,b,c [--delay D]\n"
    "               [--max-abs-error E] [--max-weight-norm W]\n"
    "       anc_cli --help\n";

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1);
}

// Parses the whole string as a doule, or returns false.
// std::stod alone would accept "12abc" as 12.
bool try_parse_double(const std::string& raw, double& value) {
    const std::string text = trim(raw);
    if (text.empty()) {
        return false;
    }
    try {
        std::size_t used = 0;
        value = std::stod(text, &used);
        return used == text.size();
    } catch (const std::exception&) {
        return false;
    }
}

double option_double(const std::string& name,const std::string& text) {
    double value = 0.0;
    if (!try_parse_double(text, value)) {
        throw UsageError(name + ": not a number: '" + text + "'");
    }
    return value;
}

std::size_t option_size(const std::string& name, const std::string& text) {
    // std::stoul("-1") wraps around to a huge value instead of failing.
    const std::string t = trim(text);
    if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) {
        throw UsageError(name + ": not a non-negative integer: '" + text + "'");
    }
    try {
        return static_cast<std::size_t>(std::stoull(t));
    } catch (const std::exception&) {
        throw UsageError(name + ": out of range: '" + text + "'");
    }
}

std::vector<float> option_float_list(const std::string& name, const std::string& text) {
    std::vector<float> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        values.push_back(static_cast<float>(option_double(name, item)));
    }
    if (values.empty()) {
        throw UsageError(name + ": empty list");
    }
    return values;
}

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    static const std::set<std::string> known = {
        "--input", "--output-y", "--output-weights", "--length", "--mu",
        "--shat", "--delay", "--max-abs-error", "--max-weight-norm"};

    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (!known.count(key)) {
            throw UsageError("unknown option: " + key);
        }
        if (i + 1 >= argc) {
            throw UsageError("missing value for " + key);
        }
        if (args.count(key)) {
            throw UsageError("option given twice: " + key);
        }
        args[key] = argv[++i];
    }

    for (const char* key : {"--input", "--output-y", "--output-weights",
                            "--length", "--mu", "--shat"}) {
        if (!args.count(key)) {
            throw UsageError(std::string("missing option ") + key);
        }
    }
    return args;
}

// Splits "x,e" into two numbers. Returns false if the line is not numeric
// (used to recognise a header on line 1); throws on any other malformed line.
bool parse_io_line(const std::string& line, std::size_t line_number,
                   double& x, double& e) {
    const auto comma = line.find(',');
    if (comma == std::string::npos) {
        throw InputError("line " + std::to_string(line_number) +
                         ": expected two comma-separated values");
    }
    const std::string first = line.substr(0, comma);
    const std::string second = line.substr(comma + 1);
    if (second.find(',') != std::string::npos) {
        throw InputError("line " + std::to_string(line_number) +
                         ": expected two values, found more");
    }
    const bool ok_x = try_parse_double(first, x);
    const bool ok_e = try_parse_double(second, e);
    if (ok_x && ok_e) {
        return true;
    }
    if (!ok_x && !ok_e && line_number == 1) {
        return false;   // header
    }
    throw InputError("line " + std::to_string(line_number) +
                     ": not a number: '" + trim(line) + "'");
}

int run(const std::map<std::string, std::string>& args) {
    anc::Limits limits;
    if (args.count("--max-abs-error")) {
        limits.max_abs_error =
            static_cast<float>(option_double("--max-abs-error", args.at("--max-abs-error")));
    }
    if (args.count("--max-weight-norm")) {
        limits.max_weight_norm = 
            static_cast<float>(option_double("--max-weight-norm", args.at("--max-weight-norm")));
    }

    const std::size_t length = option_size("--length", args.at("--length"));
    const float mu = static_cast<float>(option_double("--mu", args.at("--mu")));
    std::vector<float> s_hat = option_float_list("--shat", args.at("--shat"));
    const std::size_t delay = 
        args.count("--delay") ? option_size("--delay", args.at("--delay")) : 0;
    
    // The constructor validates the values; report its message as a usage error.
    std::unique_ptr<anc::FxLMS> controller;
    try {
        controller = std::make_unique<anc::FxLMS>(length, mu, std::move(s_hat), delay, limits);
    } catch (const std::invalid_argument& error) {
        throw UsageError(error.what());
    }

    std::ifstream input(args.at("--input"));
    if (!input) {
        std::cerr << "error: cannot open input file " << args.at("--input") << "\n";
        return kExitRuntime;
    }
    std::ofstream out_y(args.at("--output-y"));
    std::ofstream out_w(args.at("--output-weights"));
    if (!out_y || !out_w) {
        std::cerr << "error: cannot open output files\n";
        return kExitRuntime;
    }
    out_y.precision(std::numeric_limits<float>::max_digits10);
    out_w.precision(std::numeric_limits<float>::max_digits10);
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }
        double x = 0.0;
        double e = 0.0;
        if (!parse_io_line(line, line_number, x, e)) {
            continue;   // header on line 1
        }
        out_y << controller->process(static_cast<float>(x), static_cast<float>(e)) << "\n";
    }

    for (float w : controller->weights()) {
        out_w << w << "\n";
    }

    out_y.flush();
    out_w.flush();
    if (!out_y || !out_w) {
        std::cerr << "error: writing the output files failed\n";
        return kExitRuntime;
    }

    std::cout << "fault=" << anc::to_string(controller->fault())
              << " fault_sample=" << controller->fault_sample()
              << " samples=" << controller->samples_processed() << "\n";
    return controller->diverged() ? kExitFault : kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "help"){
        std::cout << kUsage;
        return kExitOk;
    }
    try {
        return run(parse_args(argc, argv));
    } catch (const UsageError& error) {
        std::cerr << "error: " <<error.what() << "\n" << kUsage;
        return kExitUsage;
    } catch (const InputError& error) {
        std::cerr << "error: " <<error.what() << "\n";
        return kExitRuntime;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return kExitRuntime;
    }
}
