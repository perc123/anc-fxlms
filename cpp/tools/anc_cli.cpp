// Open-loop driver for the parity test.
//
//   anc_cli --input io.csv --output-y y.csv --output-weights w.csv
//           --length L --mu MU --shat a,b,c [--delay D]
//
// io.csv holds one "x,e" pair per line. The tool writes y[n] (one per line)
// and the final weights (one per line) with enough digits to round-trip a
// float.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "anc/fxlms.hpp"

namespace {

std::vector<float> parse_list(const std::string& text) {
    std::vector<float> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        values.push_back(std::stof(item));
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) {
        args[argv[i]] = argv[i + 1];
    }

    for (const char* key : {"--input", "--output-y", "--output-weights",
                            "--length", "--mu", "--shat"}) {
        if (!args.count(key)) {
            std::cerr << "missing argument " << key << "\n";
            return 2;
        }
    }

    try {
        anc::FxLMS controller(
            std::stoul(args["--length"]),
            std::stof(args["--mu"]),
            parse_list(args["--shat"]),
            args.count("--delay") ? std::stoul(args["--delay"]) : 0);

        std::ifstream input(args["--input"]);
        std::ofstream out_y(args["--output-y"]);
        std::ofstream out_w(args["--output-weights"]);
        if (!input || !out_y || !out_w) {
            std::cerr << "cannot open input/output files\n";
            return 2;
        }
        out_y.precision(9);
        out_w.precision(9);

        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            const auto comma = line.find(',');
            const float x = std::stof(line.substr(0, comma));
            const float e = std::stof(line.substr(comma + 1));
            out_y << controller.process(x, e) << "\n";
        }

        for (float w : controller.weights()) {
            out_w << w << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
