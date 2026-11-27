cmake_minimum_required(VERSION 3.16)
project(anc_fxlms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(anc_fxlms src/fxlms.cpp src/history.cpp)
target_include_directories(anc_fxlms PUBLIC include)
target_compile_options(anc_fxlms PRIVATE -Wall -Wextra -Wpedantic)

add_executable(anc_cli tools/anc_cli.cpp)
target_link_libraries(anc_cli PRIVATE anc_fxlms)
target_compile_options(anc_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bench_fxlms bench/bench_fxlms.cpp)
target_link_libraries(bench_fxlms PRIVATE anc_fxlms)

# ---------------------------------------------------------------------------
# Tests (run with: ctest --test-dir <build-dir> --output-on-failure)
# ---------------------------------------------------------------------------
enable_testing()

add_executable(anc_unit_tests tests/unit_tests.cpp)
target_link_libraries(anc_unit_tests PRIVATE anc_fxlms)
target_compile_options(anc_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME unit COMMAND anc_unit_tests)

add_executable(anc_golden_tests tests/golden_tests.cpp)
target_link_libraries(anc_golden_tests PRIVATE anc_fxlms)
target_compile_options(anc_golden_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME golden
         COMMAND anc_golden_tests ${CMAKE_CURRENT_SOURCE_DIR}/../tests/golden)