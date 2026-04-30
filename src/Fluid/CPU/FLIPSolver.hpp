#pragma once

#include "FLIPParticle.hpp"
#include "MACGrid.hpp"

#include <glm/glm.hpp>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Vec3 = glm::vec3;

enum CellType {
    AIR,
    WATER,
    SOLID
};

enum class SimulationMode {
    SERIAL,
    PARALLEL,      // Jacobi
    PARALLEL_RBGS  // Red-Black Gauss-Seidel
};

struct SimulationStats {
    double t_p2g = 0;
    double t_grid = 0;
    double t_g2p = 0;
    double t_advect = 0;

    void reset() {
        t_p2g = t_grid = t_g2p = t_advect = 0;
    }
};

struct AggregatedStats {
    double sum_p2g = 0, sum_grid = 0, sum_g2p = 0, sum_advect = 0;
    int frameCount = 0;

    void add(const SimulationStats& s) {
        sum_p2g += s.t_p2g;
        sum_grid += s.t_grid;
        sum_g2p += s.t_g2p;
        sum_advect += s.t_advect;
        frameCount++;
    }

    void reset() {
        sum_p2g = sum_grid = sum_g2p = sum_advect = 0;
        frameCount = 0;
    }
};

class FLIPSolver {
public:
    FLIPSolver(int nx, int ny, int nz, float h);
    int nthreads = 16;

    void step(float dt);

    ParticleBuffer& getParticles() { return particles; }
    const ParticleBuffer& getParticles() const { return particles; }

    void addParticles(const std::vector<Particle>& newParticles) {
        particles.reserve(particles.size() + newParticles.size());
        for (const auto& p : newParticles) {
            particles.addParticle(p.pos, p.vel);
        }
    }

    void clearParticles() { particles.clear(); }

    SimulationStats stats;
    AggregatedStats aggrStats;
    SimulationMode mode = SimulationMode::PARALLEL_RBGS;

private:
    MACGrid grid;
    ParticleBuffer particles;
    std::vector<CellType> cellType;

    float materialDensity;
    float flipRatio;

    int pressureIterations;

    void advectParticles(float dt);
    void markFluidCells();
    void particlesToGrid();
    void addGravity(float dt);
    void solvePressure(float dt);
    void solvePressureRBGS(float dt);
    void applyBoundaryConditions();
    void applyGridBoundaryConditions();
    void gridToParticles(const MACGrid& oldGrid, float dt);

    float sampleMACComponent(const MACGrid& g, const Vec3& pos, const Vec3& offset, int maxI, int maxJ, int maxK, float (MACGrid::* accessor)(int, int, int) const) const;
    Vec3 sampleMAC(const MACGrid& g, const Vec3& x) const;

    int cellIndex(int i, int j, int k) const;
    bool isValidCell(int i, int j, int k) const;
};
