cmake_minimum_required(VERSION 3.10)

project(gbmc_concept_translator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gbmc_translate src/main.cpp)
target_include_directories(gbmc_translate PRIVATE include)
