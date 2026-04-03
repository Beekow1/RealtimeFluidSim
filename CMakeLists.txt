cmake_minimum_required(VERSION 3.16)
project(RealtimeFluidSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build GLFW from source 
add_subdirectory(vendor/glfw)

# Build GLM from source 
add_subdirectory(vendor/glm)

# Find system OpenGL
find_package(OpenGL REQUIRED)

# Find OpenMP on the system
find_package(OpenMP REQUIRED)

add_executable(RealtimeFluidSim
    "src/RealtimeFluidSim.cpp" "src/RealtimeFluidSim.h"
)

target_link_libraries(RealtimeFluidSim
    PRIVATE
        glfw
        glm
        OpenGL::GL
        OpenMP::OpenMP_CXX
)