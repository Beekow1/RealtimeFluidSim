#pragma once

#include "FLIPParticle.hpp"
#include "MACGrid.hpp"

#include <glm/glm.hpp>
#include <vector>

using Vec3 = glm::vec3;

enum CellType {
    AIR,
    WATER,
    SOLID
};

class FLIPSolver {
public:
    FLIPSolver(int nx, int ny, int nz, float h);

    void step(float dt);

    std::vector<Particle>& getParticles() { return particles; }
    const std::vector<Particle>& getParticles() const { return particles; }

    void addParticles(const std::vector<Particle>& newParticles) {
        particles.insert(particles.end(), newParticles.begin(), newParticles.end());
    }

    void clearParticles() { particles.clear(); }

private:
    MACGrid grid;
    std::vector<Particle> particles;
    std::vector<CellType> cellType;

    float materialDensity;
    float flipRatio;

    int pressureIterations;

    void advectParticles(float dt);
    void markFluidCells();
    void particlesToGrid();
    void addGravity(float dt);
    void solvePressure(float dt);
    void applyBoundaryConditions();
    void applyGridBoundaryConditions();
    void gridToParticles(const MACGrid& oldGrid, float dt);

    float sampleMACComponent(const MACGrid& g, const Vec3& pos, const Vec3& offset, int maxI, int maxJ, int maxK, float (MACGrid::* accessor)(int, int, int) const) const;

    Vec3 sampleMAC(const MACGrid& g, const Vec3& x) const;

    int cellIndex(int i, int j, int k) const;
    bool isValidCell(int i, int j, int k) const;
};