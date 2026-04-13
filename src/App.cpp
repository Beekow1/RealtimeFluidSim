#include "Backend/RealtimeFluidSim.h"
#include <iostream>
// Entry point //
int main() {
    RealtimeFluidSim app;
#if defined(USE_SOLVER_GPU)
    std::cout << "GPU ACTIVE" << std::endl;
#else
    std::cout << "CPU ACTIVE" << std::endl;
#endif
    return app.run();
}  