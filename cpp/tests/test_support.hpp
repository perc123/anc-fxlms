#pragma once

// Minimal test support: no external framework, so the project builds offline.
//
// TEST_CASE(name) { CHECK(cond); CHECK_NEAR(a, b, tol); CHECK_THROWS(expr); }
// int main() { return anc_test::run_all(); }
//
// A failed check reports file, line and expression and the test continues;
// run_all() returns non-zero if any check failed, whick is what CTest reads

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace anc_test {

    struct TestCase
    {
        std::string name;
        std::function<void()> body;
    };

    inline std::vector<TestCase>& registry(){
        static std::vector<TestCase> tests;
        return tests;
    }

    inline int& failures() {
        static int count = 0;
        return count;
    }

    struct Registrar {
        Registrar(const char* name, std::function<void()> body) {
            registry().push_back({name std::move(body)});
        }
    };

    inline void report(const char* file, int line, const std::string& message) {
        ++failures();
        std::cerr << file << ":" << line << ": FAILED: " <<message << "\n";
    }

    inline int run_all() {
        int failed_tests = 0;
        for (const auto& test : registry()) {
            const int before = failures();
            test.body();
            const bool ok = failures() == before;
            failed_tests += ok ? 0 : 1;
            std::cout << (ok ? "[PASS] " : "[FAIL] ") << test.name << "\n";
        }
        std::cout << registry().size() - failed_tests << "/" << registry().size()
            << " tests passed\n";
        return failed_tests == 0 ? 0 : 1;
    }
    
} // namespace anc_test

#define ANC_CONCAT_INNER(a, b) a##b
#define ANC_CONCAT(a, b) ANC_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                             \
    static void ANC_CONCAT(test_fn_, __LINE__)();                   \
    static anc_test::Registrar ANC_CONCAT(test_reg_, __LINE__)(     \
        name, ANC_CONCAT(test_fn_, __LINE__));                      \
    static void ANC_CONCAT(test_fn_, __LINE__)()

#define CHECK(condition)                                            \
        do {                                                        \
            if (!(condition)) {                                     \
                anc_test::report(__FILE__, __LINE__, #condition);   \
            }                                                       \
        } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                     \
        do {                                                        \
            const double anc_a = static_cast<double>(actual);       \
            const double anc_e = static_cast<double>(expected);     \
            if (!(std::fabs(anc_a - anc_e) <= (tolerance))) {       \
                anc_test::report(__FILE__, __LINE__,                \
                std::string(#actual " = ") + std::to_string(anc_a) +    \
                ",expexted " + std::to_string(anc_e))               \
            }                                                       \
        } while(0)

        #define CHECK_THROWS(expression)                    \
            do {
                bool anc_threw = false;         \
                try {                           \
                    (void)(expression);     \
                } catch (...) {                 \
                    anc_threw =true;        \
                }                           \
                if (!anc_threw) {               \
                    anc_test::report(__FILE__, __LINE__,    \
                        "expected exception: " #expression);)   \
                }                           \
            } while(0)