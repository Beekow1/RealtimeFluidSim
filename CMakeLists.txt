cmake_minimum_required(VERSION 3.16)
project(RealtimeFluidSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build GLFW from source
add_subdirectory(vendor/glfw)

# Build GLM from source
add_subdirectory(vendor/glm)

# Build GLAD from source
add_library(glad vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

# Find system OpenGL
find_package(OpenGL REQUIRED)

# Find OpenMP on the system
find_package(OpenMP REQUIRED)

add_executable(RealtimeFluidSim
    src/RealtimeFluidSim.cpp
    src/RealtimeFluidSim.h
 "src/Camera.h" "src/Camera.cpp" "src/Shader.h" "src/Shader.cpp" "src/ParticleSystem.h" "src/ParticleSystem.cpp" "src/App.cpp")

target_link_libraries(RealtimeFluidSim
    PRIVATE
        glad
        glfw
        glm
        OpenGL::GL
        OpenMP::OpenMP_CXX
)