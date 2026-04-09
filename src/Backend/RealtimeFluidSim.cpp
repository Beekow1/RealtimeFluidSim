#include "RealtimeFluidSim.h"
#include "Camera.h"
#include "ParticleSystem.h"
#include "../Fluid/FLIPSolver.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <filesystem>

namespace {
    constexpr int WINDOW_WIDTH = 800;
    constexpr int WINDOW_HEIGHT = 600;

    Camera gCamera(glm::vec3(0.0f, 0.0f, 0.0f), 8.0f, -90.0f, 20.0f);

    float gLastX = 0.0f;
    float gLastY = 0.0f;
    bool gFirstMouse = true;

    ParticleSystem* gParticleSystem = nullptr;
    FLIPSolver* gFluidSolver = nullptr;
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    (void)window;
    glViewport(0, 0, width, height);

    if (gParticleSystem) {
        gParticleSystem->setViewportSize(width, height);
    }
}

static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    float mouseX = static_cast<float>(xpos);
    float mouseY = static_cast<float>(ypos);

    if (gFirstMouse) {
        gLastX = mouseX;
        gLastY = mouseY;
        gFirstMouse = false;
        return;
    }

    float deltaX = mouseX - gLastX;
    float deltaY = mouseY - gLastY;

    gLastX = mouseX;
    gLastY = mouseY;

    const bool altPressed =
        glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

    if (!altPressed) {
        return;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        gCamera.orbit(deltaX, -deltaY);
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        gCamera.pan(deltaX, deltaY);
    }
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window;
    (void)xoffset;
    gCamera.zoom(static_cast<float>(yoffset));
}

static void processInput(GLFWwindow* window, float deltaTime) {
    (void)deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

RealtimeFluidSim::RealtimeFluidSim() : window(nullptr), particleSystem(nullptr), fluidSolver(nullptr) {}

RealtimeFluidSim::~RealtimeFluidSim() {
    cleanup();
}

int RealtimeFluidSim::run() {
    if (!initWindow()) return -1;
    if (!initGL()) return -1;

    setupCallbacks();
    mainLoop();
    return 0;
}

bool RealtimeFluidSim::initWindow() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Fluid Simulator", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);

    return true;
}

bool RealtimeFluidSim::initGL() {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    std::filesystem::path shaderDir = "../../../src/Shader";
    std::string vertPath = (shaderDir / "Particle.vert").string();
    std::string fragPath = (shaderDir / "Particle.frag").string();

    gParticleSystem = new ParticleSystem(vertPath.c_str(), fragPath.c_str());
    particleSystem = gParticleSystem;
    gParticleSystem->setViewportSize(WINDOW_WIDTH, WINDOW_HEIGHT);

    const int gridResX = 20;
    const int gridResY = 20;
    const int gridResZ = 20;
    const float cellSize = 0.3f;
    
    gFluidSolver = new FLIPSolver(gridResX, gridResY, gridResZ, cellSize);
    fluidSolver = gFluidSolver;

    gParticleSystem->generateGrid(gridResX, gridResY, gridResZ, cellSize);

    std::vector<Particle> initialParticles;
    initialParticles.reserve(gridResX * gridResY * gridResZ);
    
    float domainCenterX = (gridResX - 1) * cellSize * 0.5f;
    float domainCenterY = (gridResY - 1) * cellSize * 0.5f;
    float domainCenterZ = (gridResZ - 1) * cellSize * 0.5f;

    const int ppc = 6;
    const float invPpc = 1.0f / static_cast<float>(ppc);

    const int startX = 0;
    const int startY = 6;
    const int startZ = 0;

    for (int z = 0; z < gridResZ / 2; ++z) {
        for (int y = 0; y < gridResY / 2; ++y) {
            for (int x = 0; x < gridResX / 2; ++x) {
                for (int oz = 0; oz < ppc; ++oz) {
                    for (int oy = 0; oy < ppc; ++oy) {
                        for (int ox = 0; ox < ppc; ++ox) {
                            Particle p;
                            p.pos = Vec3(
                                (startX + x + (ox + 0.5f) * invPpc) * cellSize,
                                (startY + y + (oy + 0.5f) * invPpc) * cellSize,
                                (startZ + z + (oz + 0.5f) * invPpc) * cellSize
                            );

                            // optional: give it a little shove so it sloshes
                            p.vel = Vec3(0.6f, 0.0f, 0.2f);

                            initialParticles.push_back(p);
                        }
                    }
                }
            }
        }
    }
    
    gFluidSolver->addParticles(initialParticles);

    // Sync initial positions to rendering system
    std::vector<glm::vec3> displayPositions;
    displayPositions.reserve(initialParticles.size());
    for (const auto& p : initialParticles) {
        displayPositions.push_back(glm::vec3(p.pos.x, p.pos.y, p.pos.z));
    }
    gParticleSystem->setGridPos(displayPositions);

    return true;
}

void RealtimeFluidSim::setupCallbacks() {
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
}

void RealtimeFluidSim::mainLoop() {
    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Cap deltaTime to prevent large jumps
        if (deltaTime > 0.033f) deltaTime = 0.033f;

        processInput(window, deltaTime);

        // Update fluid simulation
        if (gFluidSolver) {
            gFluidSolver->step(deltaTime);

            const auto& solverParticles = gFluidSolver->getParticles();

            std::vector<glm::vec3> displayPositions;
            std::vector<glm::vec3> displayVelocities;
            displayPositions.reserve(solverParticles.size());
            displayVelocities.reserve(solverParticles.size());

            for (const auto& p : solverParticles) {
                displayPositions.push_back(glm::vec3(p.pos.x, p.pos.y, p.pos.z));
                displayVelocities.push_back(glm::vec3(p.vel.x, p.vel.y, p.vel.z));
            }

            if (gParticleSystem) {
                gParticleSystem->setGridPos(displayPositions);
                gParticleSystem->setVelocity(displayVelocities);
            }
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (gParticleSystem) {
            gParticleSystem->draw(gCamera);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void RealtimeFluidSim::cleanup() {
    delete gFluidSolver;
    gFluidSolver = nullptr;
    fluidSolver = nullptr;

    delete gParticleSystem;
    gParticleSystem = nullptr;
    particleSystem = nullptr;

    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }

    glfwTerminate();
}