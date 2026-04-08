#pragma once

struct GLFWwindow;
class ParticleSystem;
class FLIPSolver;

class RealtimeFluidSim {
public:
    RealtimeFluidSim();
    ~RealtimeFluidSim();

    int run();

private:
    GLFWwindow* window;
    ParticleSystem* particleSystem;
    FLIPSolver* fluidSolver;

    bool initWindow();
    bool initGL();
    void setupCallbacks();
    void mainLoop();
    void cleanup();
};