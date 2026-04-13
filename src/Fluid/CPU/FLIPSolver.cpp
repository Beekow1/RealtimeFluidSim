#include "FLIPSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/norm.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float trilinearWeight(float fx, float fy, float fz, int wi, int wj, int wk) {
    const float wx = (wi == 0) ? (1.0f - fx) : fx;
    const float wy = (wj == 0) ? (1.0f - fy) : fy;
    const float wz = (wk == 0) ? (1.0f - fz) : fz;
    return wx * wy * wz;
}

FLIPSolver::FLIPSolver(int nx, int ny, int nz, float h)
    : grid(nx, ny, nz, h),
    cellType(nx * ny * nz, AIR),
    materialDensity(1.0f),
    flipRatio(0.95f),
    pressureIterations(100) {
}

int FLIPSolver::cellIndex(int i, int j, int k) const {
    return i + grid.getNx() * (j + grid.getNy() * k);
}

bool FLIPSolver::isValidCell(int i, int j, int k) const {
    return i >= 0 && i < grid.getNx() &&
        j >= 0 && j < grid.getNy() &&
        k >= 0 && k < grid.getNz();
}

void FLIPSolver::step(float dt) {
    if (particles.empty()) return;

    const float maxSubstep = 0.04f;
    const int substeps = std::max(1, static_cast<int>(std::ceil(dt / maxSubstep)));
    const float subDt = dt / static_cast<float>(substeps);

    for (int s = 0; s < substeps; ++s) {
        markFluidCells();
        particlesToGrid();
        applyGridBoundaryConditions();

        MACGrid oldGrid = grid;

        addGravity(subDt);
        applyGridBoundaryConditions();
        solvePressure(subDt);
        applyGridBoundaryConditions();


        gridToParticles(oldGrid, subDt);
        advectParticles(subDt);
        applyBoundaryConditions();
    }
}

void FLIPSolver::markFluidCells() {
    // Mark all cells as AIR initially
    std::fill(cellType.begin(), cellType.end(), AIR);

    const float h = grid.getDims();
    const float maxX = grid.getNx() * h;
    const float maxY = grid.getNy() * h;
    const float maxZ = grid.getNz() * h;
    const int totalCells = grid.getNx() * grid.getNy() * grid.getNz();

    const int threadCount = omp_get_max_threads();
    std::vector<std::vector<std::uint8_t>> localMasks(
        threadCount,
        std::vector<std::uint8_t>(totalCells, 0)
    );

    // Mark cells containing particles as WATER and OCCUPIED
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& mask = localMasks[tid];

#pragma omp for
        for (int pIdx = 0; pIdx < static_cast<int>(particles.size()); ++pIdx) {
            const auto& particle = particles[pIdx];
            const Vec3 p(
                std::clamp(particle.pos.x, 0.0f, maxX - 1e-5f),
                std::clamp(particle.pos.y, 0.0f, maxY - 1e-5f),
                std::clamp(particle.pos.z, 0.0f, maxZ - 1e-5f)
            );

            const int i = std::clamp(static_cast<int>(p.x / h), 0, grid.getNx() - 1);
            const int j = std::clamp(static_cast<int>(p.y / h), 0, grid.getNy() - 1);
            const int k = std::clamp(static_cast<int>(p.z / h), 0, grid.getNz() - 1);

            mask[cellIndex(i, j, k)] = 1;
        }
    }

#pragma omp parallel for
    for (int idx = 0; idx < totalCells; ++idx) {
        bool occupied = false;
        for (int t = 0; t < threadCount; ++t) {
            if (localMasks[t][idx]) {
                occupied = true;
                break;
            }
        }
        cellType[idx] = occupied ? WATER : AIR;
    }
}

void FLIPSolver::particlesToGrid() {
    grid.clearVelocities();
    grid.clearWeights();

    float h = grid.getDims();

    // For each particle, distribute its velocity to surrounding grid nodes
    for (const auto& particle : particles) {
        // Convert world position to grid coordinates
        Vec3 gridPos = particle.pos / h;

        int i = static_cast<int>(std::floor(gridPos.x));
        int j = static_cast<int>(std::floor(gridPos.y));
        int k = static_cast<int>(std::floor(gridPos.z));

        float fx = gridPos.x - i;
        float fy = gridPos.y - j;
        float fz = gridPos.z - k;

        // Clamp to valid range
        i = std::clamp(i, 0, grid.getNx() - 1);
        j = std::clamp(j, 0, grid.getNy() - 1);
        k = std::clamp(k, 0, grid.getNz() - 1);

        fx = std::max(0.0f, std::min(1.0f, fx));
        fy = std::max(0.0f, std::min(1.0f, fy));
        fz = std::max(0.0f, std::min(1.0f, fz));

        // Distribute to U grid
        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int ui = i + di;
                    int uj = j + dj;
                    int uk = k + dk;

                    if (ui >= 0 && ui <= grid.getNx() &&
                        uj >= 0 && uj < grid.getNy() &&
                        uk >= 0 && uk < grid.getNz()) {

                        float weight = trilinearWeight(fx, fy, fz, di, dj, dk);
                        grid.U(ui, uj, uk) += weight * particle.vel.x;
                        grid.getWeightU(ui, uj, uk) += weight;
                    }
                }
            }
        }

        // Distribute to V grid
        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int vi = i + di;
                    int vj = j + dj;
                    int vk = k + dk;

                    if (vi >= 0 && vi < grid.getNx() &&
                        vj >= 0 && vj <= grid.getNy() &&
                        vk >= 0 && vk < grid.getNz()) {

                        float weight = trilinearWeight(fx, fy, fz, di, dj, dk);
                        grid.V(vi, vj, vk) += weight * particle.vel.y;
                        grid.getWeightV(vi, vj, vk) += weight;
                    }
                }
            }
        }

        // Distribute to W grid
        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int wi = i + di;
                    int wj = j + dj;
                    int wk = k + dk;

                    if (wi >= 0 && wi < grid.getNx() &&
                        wj >= 0 && wj < grid.getNy() &&
                        wk >= 0 && wk <= grid.getNz()) {

                        float weight = trilinearWeight(fx, fy, fz, di, dj, dk);
                        grid.W(wi, wj, wk) += weight * particle.vel.z;
                        grid.getWeightW(wi, wj, wk) += weight;
                    }
                }
            }
        }
    }

    // Normalize by weights
#pragma omp parallel for collapse(3) if(grid.getNx() * grid.getNy() * grid.getNz() > 1024)
    for (int i = 0; i <= grid.getNx(); ++i) {
        for (int j = 0; j < grid.getNy(); ++j) {
            for (int k = 0; k < grid.getNz(); ++k) {
                float w = grid.getWeightU(i, j, k);
                if (w > 0.0f) {
                    grid.U(i, j, k) /= w;
                }
            }
        }
    }
#pragma omp parallel for collapse(3) if(grid.getNx() * grid.getNy() * grid.getNz() > 1024)
    for (int i = 0; i < grid.getNx(); ++i) {
        for (int j = 0; j <= grid.getNy(); ++j) {
            for (int k = 0; k < grid.getNz(); ++k) {
                float w = grid.getWeightV(i, j, k);
                if (w > 0.0f) {
                    grid.V(i, j, k) /= w;
                }
            }
        }
    }
#pragma omp parallel for collapse(3) if(grid.getNx() * grid.getNy() * grid.getNz() > 1024)
    for (int i = 0; i < grid.getNx(); ++i) {
        for (int j = 0; j < grid.getNy(); ++j) {
            for (int k = 0; k <= grid.getNz(); ++k) {
                float w = grid.getWeightW(i, j, k);
                if (w > 0.0f) {
                    grid.W(i, j, k) /= w;
                }
            }
        }
    }
}

void FLIPSolver::addGravity(float dt) {
	const Vec3 gravity(0.0f, -9.81f, 0.0f);
#pragma omp parallel for collapse(3) if(grid.getNx() * grid.getNy() * grid.getNz() > 1024)
	for (int i = 0; i < grid.getNx(); ++i) {
		for (int j = 0; j <= grid.getNy(); ++j) {
			for (int k = 0; k < grid.getNz(); ++k) {
				grid.V(i, j, k) += gravity.y * dt;
			}
		}
	}
}
// jacobi solver.
void FLIPSolver::solvePressure(float dt) {
    const int nx = grid.getNx();
    const int ny = grid.getNy();
    const int nz = grid.getNz();
    const float h = grid.getDims();
    const float safeDt = std::max(dt, 1e-6f);
    const int nCells = nx * ny * nz;

    grid.clearPressure();

    std::vector<float> pOld(nCells, 0.0f);
    std::vector<float> pNew(nCells, 0.0f);

    for (int iter = 0; iter < pressureIterations; ++iter) {
#pragma omp parallel for collapse(3) if(nCells > 1024)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int idx = cellIndex(i, j, k);
                    if (cellType[idx] != WATER) {
                        pNew[idx] = 0.0f;
                        continue;
                    }

                    const float rhs = (materialDensity / safeDt) * grid.divergence(i, j, k);

                    float sum = 0.0f;
                    int diag = 0;

                    auto consider = [&](int ni, int nj, int nk) {
                        if (!isValidCell(ni, nj, nk)) return;
                        const CellType t = cellType[cellIndex(ni, nj, nk)];
                        if (t == SOLID) return;

                        ++diag;
                        if (t == WATER) {
                            sum += pOld[cellIndex(ni, nj, nk)];
                        }
                    };

                    consider(i - 1, j, k);
                    consider(i + 1, j, k);
                    consider(i, j - 1, k);
                    consider(i, j + 1, k);
                    consider(i, j, k - 1);
                    consider(i, j, k + 1);

                    pNew[idx] = (diag > 0)
                        ? (sum - rhs * h * h) / static_cast<float>(diag)
                        : 0.0f;
                }
            }
        }

        pOld.swap(pNew);
    }

#pragma omp parallel for collapse(3) if(nCells > 1024)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                grid.P(i, j, k) = pOld[cellIndex(i, j, k)];
            }
        }
    }

#pragma omp parallel for collapse(3) if(nCells > 1024)
    for (int i = 1; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const CellType leftType = cellType[cellIndex(i - 1, j, k)];
                const CellType rightType = cellType[cellIndex(i, j, k)];

                if (leftType == SOLID || rightType == SOLID) {
                    grid.U(i, j, k) = 0.0f;
                    continue;
                }

                if (leftType == AIR && rightType == AIR) {
                    continue;
                }

                const float pL = (leftType == WATER) ? grid.P(i - 1, j, k) : 0.0f;
                const float pR = (rightType == WATER) ? grid.P(i, j, k) : 0.0f;
                grid.U(i, j, k) -= (safeDt / materialDensity) * (pR - pL) / h;
            }
        }
    }

#pragma omp parallel for collapse(3) if(nCells > 1024)
    for (int i = 0; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const CellType downType = cellType[cellIndex(i, j - 1, k)];
                const CellType upType = cellType[cellIndex(i, j, k)];

                if (downType == SOLID || upType == SOLID) {
                    grid.V(i, j, k) = 0.0f;
                    continue;
                }

                if (downType == AIR && upType == AIR) {
                    continue;
                }

                const float pD = (downType == WATER) ? grid.P(i, j - 1, k) : 0.0f;
                const float pU = (upType == WATER) ? grid.P(i, j, k) : 0.0f;
                grid.V(i, j, k) -= (safeDt / materialDensity) * (pU - pD) / h;
            }
        }
    }

#pragma omp parallel for collapse(3) if(nCells > 1024)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 1; k < nz; ++k) {
                const CellType backType = cellType[cellIndex(i, j, k - 1)];
                const CellType frontType = cellType[cellIndex(i, j, k)];

                if (backType == SOLID || frontType == SOLID) {
                    grid.W(i, j, k) = 0.0f;
                    continue;
                }

                if (backType == AIR && frontType == AIR) {
                    continue;
                }

                const float pB = (backType == WATER) ? grid.P(i, j, k - 1) : 0.0f;
                const float pF = (frontType == WATER) ? grid.P(i, j, k) : 0.0f;
                grid.W(i, j, k) -= (safeDt / materialDensity) * (pF - pB) / h;
            }
        }
    }
}

void FLIPSolver::applyGridBoundaryConditions() {

#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int j = 0; j < grid.getNy(); ++j) {
                for (int k = 0; k < grid.getNz(); ++k) {
                    grid.U(0, j, k) = 0.0f;
                    grid.U(grid.getNx(), j, k) = 0.0f;
                }
            }
        }

#pragma omp section
        {
            for (int i = 0; i < grid.getNx(); ++i) {
                for (int k = 0; k < grid.getNz(); ++k) {
                    grid.V(i, 0, k) = 0.0f;
                    grid.V(i, grid.getNy(), k) = 0.0f;
                }
            }
        }

#pragma omp section
        {
            for (int i = 0; i < grid.getNx(); ++i) {
                for (int j = 0; j < grid.getNy(); ++j) {
                    grid.W(i, j, 0) = 0.0f;
                    grid.W(i, j, grid.getNz()) = 0.0f;
                }
            }
        }
    }
}

void FLIPSolver::applyBoundaryConditions() {
    const float h = grid.getDims();
    const float eps = 0.05f * h;
    const float maxX = grid.getNx() * h;
    const float maxY = grid.getNy() * h;
    const float maxZ = grid.getNz() * h;

#pragma omp parallel for if(particles.size() > 1024)
    for (int pIdx = 0; pIdx < static_cast<int>(particles.size()); ++pIdx) {
        auto& p = particles[pIdx];

        if (p.pos.x < eps) {
            p.pos.x = eps;
            if (p.vel.x < 0.0f) p.vel.x = 0.0f;
            p.vel.y *= 0.0f;
            p.vel.z *= 0.0f;
        }
        else if (p.pos.x > maxX - eps) {
            p.pos.x = maxX - eps;
            if (p.vel.x > 0.0f) p.vel.x = 0.0f;
            p.vel.y = 0.0f;
            p.vel.z = 0.0f;
        }

        if (p.pos.y < eps) {
            p.pos.y = eps;
            if (p.vel.y < 0.0f) p.vel.y = 0.0f;
            p.vel.x = 0.0f;
            p.vel.z = 0.0f;
        }
        else if (p.pos.y > maxY - eps) {
            p.pos.y = maxY - eps;
            if (p.vel.y > 0.0f) p.vel.y = 0.0f;
            p.vel.x = 0.0f;
            p.vel.z = 0.0f;
        }

        if (p.pos.z < eps) {
            p.pos.z = eps;
            if (p.vel.z < 0.0f) p.vel.z = 0.0f;
            p.vel.x = 0.0f;
            p.vel.y = 0.0f;
        }
        else if (p.pos.z > maxZ - eps) {
            p.pos.z = maxZ - eps;
            if (p.vel.z > 0.0f) p.vel.z = 0.0f;
            p.vel.x = 0.0f;
            p.vel.y = 0.0f;
        }
    }
}

void FLIPSolver::gridToParticles(const MACGrid& oldGrid, float dt) {
#pragma omp parallel for if(particles.size() > 1024)
    for (int p = 0; p < static_cast<int>(particles.size()); ++p) {
        auto& particle = particles[p];
        const Vec3 picVel = sampleMAC(grid, particle.pos);
        const Vec3 oldVel = sampleMAC(oldGrid, particle.pos);
        const Vec3 flipDelta = picVel - oldVel;
        const Vec3 flipVel = particle.vel + flipDelta;
        particle.vel = flipRatio * flipVel + (1.0f - flipRatio) * picVel;
    }
}

void FLIPSolver::advectParticles(float dt) {
    const float h = grid.getDims();
    const float eps = 0.05f * h;
    const float maxX = grid.getNx() * h;
    const float maxY = grid.getNy() * h;
    const float maxZ = grid.getNz() * h;
    // RK2 integration
#pragma omp parallel for if(particles.size() > 1024)
    for (int p = 0; p < static_cast<int>(particles.size()); ++p) {
        float t = 0.0f;

        while (t < dt) {
            const Vec3 v0 = sampleMAC(grid, particles[p].pos);
            const float speed = glm::length(v0);
            const float subDt = (speed > 1e-6f)
                ? std::min(dt - t, 0.9f * h / speed)
                : (dt - t);

            const Vec3 midPos = particles[p].pos + 0.5f * subDt * v0;
            const Vec3 vMid = sampleMAC(grid, midPos);

            particles[p].pos += subDt * vMid;
            t += subDt;

            particles[p].pos.x = std::clamp(particles[p].pos.x, eps, maxX - eps);
            particles[p].pos.y = std::clamp(particles[p].pos.y, eps, maxY - eps);
            particles[p].pos.z = std::clamp(particles[p].pos.z, eps, maxZ - eps);
        }
    }
}

float FLIPSolver::sampleMACComponent(const MACGrid& g, const Vec3& pos, const Vec3& offset, int maxI, int maxJ, int maxK, float (MACGrid::* accessor)(int, int, int) const ) const {
    const float h = g.getDims();
    const Vec3 p = pos / h - offset;

    const int i = static_cast<int>(std::floor(p.x));
    const int j = static_cast<int>(std::floor(p.y));
    const int k = static_cast<int>(std::floor(p.z));

    const float fx = p.x - static_cast<float>(i);
    const float fy = p.y - static_cast<float>(j);
    const float fz = p.z - static_cast<float>(k);

    float value = 0.0f;

    if (i >= 0 && i <= maxI && j >= 0 && j <= maxJ && k >= 0 && k <= maxK)
        value += trilinearWeight(fx, fy, fz, 0, 0, 0) * (g.*accessor)(i, j, k);
    if (i >= 0 && i <= maxI && j >= 0 && j <= maxJ && k + 1 >= 0 && k + 1 <= maxK)
        value += trilinearWeight(fx, fy, fz, 0, 0, 1) * (g.*accessor)(i, j, k + 1);
    if (i >= 0 && i <= maxI && j + 1 >= 0 && j + 1 <= maxJ && k >= 0 && k <= maxK)
        value += trilinearWeight(fx, fy, fz, 0, 1, 0) * (g.*accessor)(i, j + 1, k);
    if (i >= 0 && i <= maxI && j + 1 >= 0 && j + 1 <= maxJ && k + 1 >= 0 && k + 1 <= maxK)
        value += trilinearWeight(fx, fy, fz, 0, 1, 1) * (g.*accessor)(i, j + 1, k + 1);
    if (i + 1 >= 0 && i + 1 <= maxI && j >= 0 && j <= maxJ && k >= 0 && k <= maxK)
        value += trilinearWeight(fx, fy, fz, 1, 0, 0) * (g.*accessor)(i + 1, j, k);
    if (i + 1 >= 0 && i + 1 <= maxI && j >= 0 && j <= maxJ && k + 1 >= 0 && k + 1 <= maxK)
        value += trilinearWeight(fx, fy, fz, 1, 0, 1) * (g.*accessor)(i + 1, j, k + 1);
    if (i + 1 >= 0 && i + 1 <= maxI && j + 1 >= 0 && j + 1 <= maxJ && k >= 0 && k <= maxK)
        value += trilinearWeight(fx, fy, fz, 1, 1, 0) * (g.*accessor)(i + 1, j + 1, k);
    if (i + 1 >= 0 && i + 1 <= maxI && j + 1 >= 0 && j + 1 <= maxJ && k + 1 >= 0 && k + 1 <= maxK)
        value += trilinearWeight(fx, fy, fz, 1, 1, 1) * (g.*accessor)(i + 1, j + 1, k + 1);

    return value;
}

Vec3 FLIPSolver::sampleMAC(const MACGrid& g, const Vec3& x) const {
    const float h = g.getDims();
    const float maxX = g.getNx() * h;
    const float maxY = g.getNy() * h;
    const float maxZ = g.getNz() * h;

    const Vec3 xc(
        std::clamp(x.x, 0.0f, maxX - 1e-5f),
        std::clamp(x.y, 0.0f, maxY - 1e-5f),
        std::clamp(x.z, 0.0f, maxZ - 1e-5f)
    );

    const float u = sampleMACComponent(
        g,
        xc,
        Vec3(0.0f, 0.5f, 0.5f),
        g.getNx(),
        g.getNy() - 1,
        g.getNz() - 1,
        &MACGrid::U
    );

    const float v = sampleMACComponent(
        g,
        xc,
        Vec3(0.5f, 0.0f, 0.5f),
        g.getNx() - 1,
        g.getNy(),
        g.getNz() - 1,
        &MACGrid::V
    );

    const float w = sampleMACComponent(
        g,
        xc,
        Vec3(0.5f, 0.5f, 0.0f),
        g.getNx() - 1,
        g.getNy() - 1,
        g.getNz(),
        &MACGrid::W
    );

    return Vec3(u, v, w);
}