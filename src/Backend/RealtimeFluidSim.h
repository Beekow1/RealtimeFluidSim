#pragma once

struct GLFWwindow;

class RealtimeFluidSim {
public:
    RealtimeFluidSim();
    ~RealtimeFluidSim();

    int run();

private:
    GLFWwindow* window;

    bool initWindow();
    bool initGL();
    void setupCallbacks();
    void mainLoop();
    void cleanup();
};