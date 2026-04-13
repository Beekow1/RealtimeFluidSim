#pragma once

#include "FLIPParticle.hpp"

#include <memory>
#include <vector>

class FLIPSolver {
public:
    FLIPSolver(int nx, int ny, int nz, float h);
    ~FLIPSolver();

    FLIPSolver(const FLIPSolver&) = delete;
    FLIPSolver& operator=(const FLIPSolver&) = delete;
    FLIPSolver(FLIPSolver&&) = delete;
    FLIPSolver& operator=(FLIPSolver&&) = delete;

    void step(float dt);

    std::vector<Particle>& getParticles() { return particles; }
    const std::vector<Particle>& getParticles() const { return particles; }

    void addParticles(const std::vector<Particle>& newParticles);
    void clearParticles();

    void setMaterialDensity(float density);
    void setFlipRatio(float ratio);
    void setPressureIterations(int iterations);

    int getNx() const noexcept { return nx; }
    int getNy() const noexcept { return ny; }
    int getNz() const noexcept { return nz; }
    float getCellSize() const noexcept { return h; }

private:
    struct Impl;

    int nx;
    int ny;
    int nz;
    float h;

    float materialDensity;
    float flipRatio;
    int pressureIterations;

    std::vector<Particle> particles;
    std::unique_ptr<Impl> impl;
};
