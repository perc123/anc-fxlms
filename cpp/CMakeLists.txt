cmake_minimum_required(VERSION 3.16)
project(anc_fxlms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(anc_fxlms src/fxlms.cpp)
target_include_directories(anc_fxlms PUBLIC include)
target_compile_options(anc_fxlms PRIVATE -Wall -Wextra -Wpedantic)

add_executable(anc_cli tools/anc_cli.cpp)
target_link_libraries(anc_cli PRIVATE anc_fxlms)

add_executable(bench_fxlms bench/bench_fxlms.cpp)
target_link_libraries(bench_fxlms PRIVATE anc_fxlms)
