// RealtimeFluidSim.cpp : Defines the entry point for the application.

#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Fluid Simulator", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();


    glm::vec3 t1(1, 2, 3);
    glm::vec3 t2(3, 2, 1);
    return glm::dot(t1, t2);
}