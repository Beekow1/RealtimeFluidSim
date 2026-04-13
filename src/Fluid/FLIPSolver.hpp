#pragma once

#if defined(USE_SOLVER_GPU)
#include "GPU/FLIPSolver.hpp"
#else
#include "CPU/FLIPSolver.hpp"
#endif