cmake_minimum_required(VERSION 3.10)

project(gbmc_core_flow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gbmc_flow_demo src/main.cpp)
target_include_directories(gbmc_flow_demo PRIVATE include)
