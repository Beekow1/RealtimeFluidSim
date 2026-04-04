#include "ParticleSystem.h"
#include "Camera.h"

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>

ParticleSystem::ParticleSystem(const char* vertexShaderPath, const char* fragmentShaderPath)
    : vao(0),
    vbo(0),
    shader(vertexShaderPath, fragmentShaderPath),
    viewportWidth(800),
    viewportHeight(600) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
}

ParticleSystem::~ParticleSystem() {
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
}

void ParticleSystem::setViewportSize(int width, int height) {
    viewportWidth = width;
    viewportHeight = height;
}

// Should only be run on startup
void ParticleSystem::generateGrid(int countX, int countY, int countZ, float spacing) {
    particles.clear();
    particles.reserve(countX * countY * countZ);

    glm::vec3 center(
        (countX - 1) * spacing * 0.5f,
        (countY - 1) * spacing * 0.5f,
        (countZ - 1) * spacing * 0.5f
    );

    for (int z = 0; z < countZ; ++z) {
        for (int y = 0; y < countY; ++y) {
            for (int x = 0; x < countX; ++x) {
                glm::vec3 pos(
                    x * spacing,
                    y * spacing,
                    z * spacing
                );
                particles.push_back({ pos - center, glm::vec3 (x/(float) countX, y/(float) countY, z/(float) countZ)});
            }
        }
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), particles.data(), GL_STREAM_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, pos));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, col));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void ParticleSystem::setGridPos(std::vector<glm::vec3> posP) {
    for (int i = 0; i < particles.size(); i++) { // Could maybe optimize with OMP paralellism?
        particles[i].pos = posP[i];
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSystem::setVelocity (std::vector<glm::vec3> velP) {
    for (int i = 0; i < particles.size(); i++) { // Could maybe optimize with OMP paralellism?
        particles[i].vel = velP[i];
    }
}

void ParticleSystem::moveGridPos(std::vector<glm::vec3> posP) {
    for (int i = 0; i < particles.size(); i++) { // Could maybe optimize with OMP paralellism?
        particles[i].pos += posP[i];
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSystem::update(float dt) {
    for (int i = 0; i < particles.size(); i++) { // Could maybe optimize with OMP paralellism?
        particles[i].pos += particles[i].vel * dt;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// draw particles.
void ParticleSystem::draw(const Camera& camera) {
    shader.use();

    glm::mat4 model(1.0f);
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix(
        static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight)
    );

    shader.setMat4("uModel", model);
    shader.setMat4("uView", view);
    shader.setMat4("uProjection", projection);
    shader.setFloat("uPointSize", 6.0f);

    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particles.size()));
    glBindVertexArray(0);
}