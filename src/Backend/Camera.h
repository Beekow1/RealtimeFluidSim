#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Basic blender camera implementation
class Camera {
public:
    Camera(
        glm::vec3 target = glm::vec3(0.0f),
        float distance = 8.0f,
        float yaw = -90.0f,
        float pitch = 20.0f
    );

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;

    void orbit(float deltaX, float deltaY);
    void pan(float deltaX, float deltaY);
    void zoom(float delta);

    glm::vec3 getPosition() const;
    glm::vec3 getTarget() const;
    float getZoom() const;

private:
    void updatePosition();

private:
    glm::vec3 target;
    glm::vec3 position;
    glm::vec3 worldUp;

    float distance;
    float yaw;
    float pitch;
    float fov;

    float orbitSensitivity;
    float panSensitivity;
    float zoomSensitivity;
};