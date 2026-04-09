#pragma once

#include "Shader.h"
#include <glm/glm.hpp>
#include <vector>

class Camera;

class ParticleSystem {
public:
    struct Particle {
        glm::vec3 pos;
        glm::vec3 col;
        glm::vec3 vel;
    };

    ParticleSystem(const char* vertexShaderPath, const char* fragmentShaderPath);
    ~ParticleSystem();

    void generateGrid(int countX, int countY, int countZ, float spacing);
    void draw(const Camera& camera);
    void setViewportSize(int width, int height);
    void setGridPos(const std::vector<glm::vec3>& posP);
    void moveGridPos(const std::vector<glm::vec3>& posP);
    void setVelocity(const std::vector<glm::vec3>& velP);
    void update(float dt);
private:
    unsigned int vao;
    unsigned int vbo;

    Shader shader;
    std::vector<Particle> particles;

    int viewportWidth;
    int viewportHeight;
};