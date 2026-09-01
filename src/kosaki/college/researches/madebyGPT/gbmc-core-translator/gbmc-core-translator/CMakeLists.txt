cmake_minimum_required(VERSION 3.10)

project(gbmc_core_translator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gbmc_translator_demo src/main.cpp)
target_include_directories(gbmc_translator_demo PRIVATE include)
