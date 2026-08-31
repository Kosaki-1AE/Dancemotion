cmake_minimum_required(VERSION 3.16)
project(kos3d_viewer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)

add_executable(kos3d src/main.cpp)
target_link_libraries(kos3d PRIVATE OpenGL::GL glfw assimp::assimp)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kos3d PRIVATE -Wall -Wextra -O2)
endif()
