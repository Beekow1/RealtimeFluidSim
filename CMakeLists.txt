cmake_minimum_required(VERSION 3.18)
project(RealtimeFluidSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(USE_SOLVER_GPU "CPU" CACHE STRING "Fluid backend to build: CPU or GPU")
set_property(CACHE USE_SOLVER_GPU PROPERTY STRINGS CPU GPU)
string(TOUPPER "${USE_SOLVER_GPU}" USE_SOLVER_GPU)

if(NOT USE_SOLVER_GPU STREQUAL "CPU" AND NOT USE_SOLVER_GPU STREQUAL "GPU")
    message(FATAL_ERROR "USE_SOLVER_GPU must be CPU or GPU")
endif()

set(RFS_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "USE_SOLVER_GPU = ${USE_SOLVER_GPU}")

add_subdirectory(vendor/glfw)
add_subdirectory(vendor/glm)

add_library(glad vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

find_package(OpenGL REQUIRED)

if(USE_SOLVER_GPU STREQUAL "CPU")
    find_package(OpenMP REQUIRED)
endif()

if(USE_SOLVER_GPU STREQUAL "GPU")
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "USE_SOLVER_GPU=GPU but no CUDA compiler was found. "
            "Install the CUDA Toolkit or switch to USE_SOLVER_GPU=CPU"
        )
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(MSVC)
        string(REPLACE "/RTC1" "" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
        string(REPLACE "/RTCs" "" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
        string(REPLACE "/RTCu" "" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
    endif()

    message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
endif()

set(APP_SOURCES
    src/Backend/RealtimeFluidSim.cpp
    src/Backend/RealtimeFluidSim.h
    src/Backend/Camera.h
    src/Backend/Camera.cpp
    src/Backend/Shader.h
    src/Backend/Shader.cpp
    src/Backend/ParticleSystem.h
    src/Backend/ParticleSystem.cpp
    src/App.cpp
    src/Fluid/FLIPSolver.hpp
)

set(FLUID_SOURCES)
if(USE_SOLVER_GPU STREQUAL "CPU")
    list(APPEND FLUID_SOURCES
        src/Fluid/CPU/FLIPSolver.cpp
        src/Fluid/CPU/FLIPSolver.hpp
        src/Fluid/CPU/FLIPParticle.hpp
        src/Fluid/CPU/MACGrid.hpp
    )
else()
    list(APPEND FLUID_SOURCES
        src/Fluid/GPU/FLIPSolver.cu
        src/Fluid/GPU/FLIPSolver.hpp
        src/Fluid/GPU/FLIPParticle.hpp
    )
endif()

add_executable(RealtimeFluidSim
    ${APP_SOURCES}
    ${FLUID_SOURCES}
)

target_include_directories(RealtimeFluidSim
    PRIVATE
        src
        vendor/glad/include
)

target_link_libraries(RealtimeFluidSim
    PRIVATE
        glad
        glfw
        glm
        OpenGL::GL
)

if(USE_SOLVER_GPU STREQUAL "CPU")
    target_link_libraries(RealtimeFluidSim PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(RealtimeFluidSim PRIVATE USE_SOLVER_CPU=1)
else()
    target_link_libraries(RealtimeFluidSim PRIVATE CUDA::cudart)
    target_compile_definitions(RealtimeFluidSim PRIVATE USE_SOLVER_GPU=1)

    set_target_properties(RealtimeFluidSim PROPERTIES
        CUDA_ARCHITECTURES "${RFS_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Keep CUDA compile flags simple on MSVC
    if(MSVC)
        target_compile_options(RealtimeFluidSim PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Zi,-Od,-MDd>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/O2,-MD>
        )
    else()
        target_compile_options(RealtimeFluidSim PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        )
    endif()
endif()

if(USE_SOLVER_GPU STREQUAL "CPU")
    if(MSVC)
        target_compile_options(RealtimeFluidSim PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
        )
    else()
        target_compile_options(RealtimeFluidSim PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        )
    endif()
endif()