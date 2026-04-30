#pragma once

#include <glm/glm.hpp>
#include <vector>

using Vec3 = glm::vec3;

// Structure of Arrays (SoA) for better SIMD utilization
struct ParticleBuffer {
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz;

    void resize(size_t n) {
        px.resize(n); py.resize(n); pz.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
    }

    void reserve(size_t n) {
        px.reserve(n); py.reserve(n); pz.reserve(n);
        vx.reserve(n); vy.reserve(n); vz.reserve(n);
    }

    size_t size() const { return px.size(); }
    bool empty() const { return px.empty(); }
    void clear() {
        px.clear(); py.clear(); pz.clear();
        vx.clear(); vy.clear(); vz.clear();
    }

    void swap(ParticleBuffer& other) {
        px.swap(other.px); py.swap(other.py); pz.swap(other.pz);
        vx.swap(other.vx); vy.swap(other.vy); vz.swap(other.vz);
    }

    void addParticle(const Vec3& pos, const Vec3& vel) {
        px.push_back(pos.x); py.push_back(pos.y); pz.push_back(pos.z);
        vx.push_back(vel.x); vy.push_back(vel.y); vz.push_back(vel.z);
    }
};

struct Particle {
    Vec3 pos;
    Vec3 vel;
};
