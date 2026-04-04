#include "Camera.h"

#include <algorithm>
#include <cmath>


// Basic Blender style camera implementation. 

Camera::Camera(glm::vec3 target, float distance, float yaw, float pitch)
    : target(target),
    position(0.0f, 0.0f, distance),
    worldUp(0.0f, 1.0f, 0.0f),
    distance(distance),
    yaw(yaw),
    pitch(pitch),
    fov(45.0f),
    orbitSensitivity(0.3f),
    panSensitivity(0.0025f),
    zoomSensitivity(0.8f) {
    updatePosition();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, target, worldUp);
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const {
    return glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 100.0f);
}

void Camera::orbit(float deltaX, float deltaY) {
    yaw += deltaX * orbitSensitivity;
    pitch += -deltaY * orbitSensitivity;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    updatePosition();
}

void Camera::pan(float deltaX, float deltaY) {
    glm::vec3 forward = glm::normalize(target - position);
    glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    glm::vec3 up = glm::normalize(glm::cross(right, forward));

    float scale = distance * panSensitivity;
    glm::vec3 offset = (-right * deltaX + up * deltaY) * scale;

    target += offset;
    position += offset;
}

void Camera::zoom(float delta) {
    distance -= delta * zoomSensitivity;

    if (distance < 0.5f) distance = 0.5f;
    if (distance > 100.0f) distance = 100.0f;

    updatePosition();
}

glm::vec3 Camera::getPosition() const {
    return position;
}

glm::vec3 Camera::getTarget() const {
    return target;
}

float Camera::getZoom() const {
    return fov;
}

void Camera::updatePosition() {
    float pitchRad = glm::radians(pitch);
    float yawRad = glm::radians(yaw);

    position.x = target.x + distance * std::cos(pitchRad) * std::cos(yawRad);
    position.y = target.y + distance * std::sin(pitchRad);
    position.z = target.z + distance * std::cos(pitchRad) * std::sin(yawRad);
}