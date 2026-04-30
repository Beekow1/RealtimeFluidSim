#include "FLIPSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <vector>
#include <numeric>
#include <immintrin.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/norm.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

// AVX2 helper for trilinear interpolation of 8 particles
static inline __m256 trilinearWeight_AVX2(__m256 fx, __m256 fy, __m256 fz, int wi, int wj, int wk) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 wx = (wi == 0) ? _mm256_sub_ps(one, fx) : fx;
    __m256 wy = (wj == 0) ? _mm256_sub_ps(one, fy) : fy;
    __m256 wz = (wk == 0) ? _mm256_sub_ps(one, fz) : fz;
    return _mm256_mul_ps(_mm256_mul_ps(wx, wy), wz);
}

static inline __m256 sampleComponent_AVX2(
    const GridNode* data,
    __m256 px, __m256 py, __m256 pz,
    __m256 offX, __m256 offY, __m256 offZ,
    int nx, int ny, int nz,
    int maxI, int maxJ, int maxK,
    int dimX, int dimY, float h
) {
    __m256 invH = _mm256_set1_ps(1.0f / h);
    __m256 gx = _mm256_sub_ps(_mm256_mul_ps(px, invH), offX);
    __m256 gy = _mm256_sub_ps(_mm256_mul_ps(py, invH), offY);
    __m256 gz = _mm256_sub_ps(_mm256_mul_ps(pz, invH), offZ);

    __m256 ix_f = _mm256_floor_ps(gx);
    __m256 iy_f = _mm256_floor_ps(gy);
    __m256 iz_f = _mm256_floor_ps(gz);

    __m256i ix = _mm256_cvtps_epi32(ix_f);
    __m256i iy = _mm256_cvtps_epi32(iy_f);
    __m256i iz = _mm256_cvtps_epi32(iz_f);

    __m256 fx = _mm256_sub_ps(gx, ix_f);
    __m256 fy = _mm256_sub_ps(gy, iy_f);
    __m256 fz = _mm256_sub_ps(gz, iz_f);

    __m256 result = _mm256_setzero_ps();

    __m256i v_dimX = _mm256_set1_epi32(dimX);
    __m256i v_dimY = _mm256_set1_epi32(dimY);
    __m256i v_maxI = _mm256_set1_epi32(maxI);
    __m256i v_maxJ = _mm256_set1_epi32(maxJ);
    __m256i v_maxK = _mm256_set1_epi32(maxK);

    for (int dk = 0; dk <= 1; ++dk) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int di = 0; di <= 1; ++di) {
                __m256i ni = _mm256_add_epi32(ix, _mm256_set1_epi32(di));
                __m256i nj = _mm256_add_epi32(iy, _mm256_set1_epi32(dj));
                __m256i nk = _mm256_add_epi32(iz, _mm256_set1_epi32(dk));

                // Mask for valid cells
                __m256i m_low = _mm256_and_si256(_mm256_cmpgt_epi32(ni, _mm256_set1_epi32(-1)),
                                    _mm256_and_si256(_mm256_cmpgt_epi32(nj, _mm256_set1_epi32(-1)),
                                                     _mm256_cmpgt_epi32(nk, _mm256_set1_epi32(-1))));
                __m256i m_high = _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_add_epi32(v_maxI, _mm256_set1_epi32(1)), ni),
                                     _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_add_epi32(v_maxJ, _mm256_set1_epi32(1)), nj),
                                                      _mm256_cmpgt_epi32(_mm256_add_epi32(v_maxK, _mm256_set1_epi32(1)), nk)));
                __m256 mask = _mm256_castsi256_ps(_mm256_and_si256(m_low, m_high));

                // Index calculation: ni + dimX * (nj + dimY * nk)
                __m256i indices = _mm256_add_epi32(ni, _mm256_mullo_epi32(v_dimX, _mm256_add_epi32(nj, _mm256_mullo_epi32(v_dimY, nk))));
                
                // Gather GridNode::val (GridNode is 2 floats, so scale 8)
                __m256 vals = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), (const float*)data, indices, mask, 8);
                
                __m256 weights = trilinearWeight_AVX2(fx, fy, fz, di, dj, dk);
                result = _mm256_fmadd_ps(weights, vals, result);
            }
        }
    }
    return result;
}

static inline float trilinearWeight(float fx, float fy, float fz, int wi, int wj, int wk) {
    const float wx = (wi == 0) ? (1.0f - fx) : fx;
    const float wy = (wj == 0) ? (1.0f - fy) : fy;
    const float wz = (wk == 0) ? (1.0f - fz) : fz;
    return wx * wy * wz;
}

FLIPSolver::FLIPSolver(int nx, int ny, int nz, float h)
    : grid(nx, ny, nz, h),
    materialDensity(1.0f),
    flipRatio(0.95f),
    pressureIterations(100) {
    const int BS = 8;
    const int nxb = (nx + BS - 1) / BS;
    const int nyb = (ny + BS - 1) / BS;
    const int nzb = (nz + BS - 1) / BS;
    cellType.resize(nxb * nyb * nzb * BS * BS * BS, AIR);
}

int FLIPSolver::cellIndex(int i, int j, int k) const {
    const int BS = 8; // Block size for tiling
    const int nxb = (grid.getNx() + BS - 1) / BS;
    const int nyb = (grid.getNy() + BS - 1) / BS;

    const int bi = i / BS;
    const int bj = j / BS;
    const int bk = k / BS;

    const int si = i % BS;
    const int sj = j % BS;
    const int sk = k % BS;

    const int blockIdx = bi + nxb * (bj + nyb * bk);
    return blockIdx * (BS * BS * BS) + (si + BS * (sj + BS * sk));
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

    stats.reset();

    for (int s = 0; s < substeps; ++s) {
        auto start = std::chrono::high_resolution_clock::now();
        markFluidCells();
        particlesToGrid();
        auto end = std::chrono::high_resolution_clock::now();
        stats.t_p2g += std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        applyGridBoundaryConditions();
        MACGrid oldGrid = grid;
        addGravity(subDt);
        applyGridBoundaryConditions();
        
        if (mode == SimulationMode::PARALLEL_RBGS) {
            solvePressureRBGS(subDt);
        } else {
            solvePressure(subDt);
        }

        applyGridBoundaryConditions();
        end = std::chrono::high_resolution_clock::now();
        stats.t_grid += std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        gridToParticles(oldGrid, subDt);
        end = std::chrono::high_resolution_clock::now();
        stats.t_g2p += std::chrono::duration<double>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        advectParticles(subDt);
        applyBoundaryConditions();
        end = std::chrono::high_resolution_clock::now();
        stats.t_advect += std::chrono::duration<double>(end - start).count();
    }

    aggrStats.add(stats);
    
    // Print aggregated stats every 60 frames
    if (aggrStats.frameCount >= 60) {
        double total = aggrStats.sum_p2g + aggrStats.sum_grid + aggrStats.sum_g2p + aggrStats.sum_advect;
        double f = 1.0 / aggrStats.frameCount;
        
        printf("\n--- AVG OVER 60 FRAMES (Mode: %d, Threads: %d) ---\n", static_cast<int>(mode), nthreads);
        printf("P2G:    %.4f ms\n", aggrStats.sum_p2g * f * 1000.0);
        printf("Grid:   %.4f ms\n", aggrStats.sum_grid * f * 1000.0);
        printf("G2P:    %.4f ms\n", aggrStats.sum_g2p * f * 1000.0);
        printf("Advect: %.4f ms\n", aggrStats.sum_advect * f * 1000.0);
        printf("TOTAL:  %.4f ms (%.1f FPS)\n", total * f * 1000.0, 1.0 / (total * f));
        printf("------------------------------------\n");
        
        aggrStats.reset();
    }
}

void FLIPSolver::markFluidCells() {
    // Mark all cells as AIR initially
    std::fill(cellType.begin(), cellType.end(), AIR);

    const float h = grid.getDims();
    const float maxX = grid.getNx() * h;
    const float maxY = grid.getNy() * h;
    const float maxZ = grid.getNz() * h;
    const int totalCells = static_cast<int>(cellType.size());

    const int threadCount = (mode == SimulationMode::SERIAL) ? 1 : nthreads;
    std::vector<std::vector<std::uint8_t>> localMasks(
        threadCount,
        std::vector<std::uint8_t>(totalCells, 0)
    );

    // Mark cells containing particles as WATER
#pragma omp parallel num_threads(threadCount) if(mode != SimulationMode::SERIAL)
    {
        const int tid = (mode == SimulationMode::SERIAL) ? 0 : omp_get_thread_num();
        auto& mask = localMasks[tid];

#pragma omp for
        for (int pIdx = 0; pIdx < static_cast<int>(particles.size()); ++pIdx) {
            const Vec3 p(
                std::clamp(particles.px[pIdx], 0.0f, maxX - 1e-5f),
                std::clamp(particles.py[pIdx], 0.0f, maxY - 1e-5f),
                std::clamp(particles.pz[pIdx], 0.0f, maxZ - 1e-5f)
            );

            const int i = std::clamp(static_cast<int>(p.x / h), 0, grid.getNx() - 1);
            const int j = std::clamp(static_cast<int>(p.y / h), 0, grid.getNy() - 1);
            const int k = std::clamp(static_cast<int>(p.z / h), 0, grid.getNz() - 1);

            mask[cellIndex(i, j, k)] = 1;
        }
    }

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL)
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

#pragma omp parallel num_threads(nthreads) if(mode != SimulationMode::SERIAL)
    {
        // For each particle, distribute its velocity to surrounding grid nodes
#pragma omp for
        for (int pIdx = 0; pIdx < static_cast<int>(particles.size()); ++pIdx) {
            // Convert world position to grid coordinates
            Vec3 gridPos(particles.px[pIdx] / h, particles.py[pIdx] / h, particles.pz[pIdx] / h);

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

            const float pvx = particles.vx[pIdx];
            const float pvy = particles.vy[pIdx];
            const float pvz = particles.vz[pIdx];

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
                            float val = weight * pvx;
                            if (mode != SimulationMode::SERIAL) {
                                
                                grid.U(ui, uj, uk) += val;
                                
                                grid.getWeightU(ui, uj, uk) += weight;
                            } else {
                                grid.U(ui, uj, uk) += val;
                                grid.getWeightU(ui, uj, uk) += weight;
                            }
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
                            float val = weight * pvy;
                            if (mode != SimulationMode::SERIAL) {
                                
                                grid.V(vi, vj, vk) += val;
                                
                                grid.getWeightV(vi, vj, vk) += weight;
                            } else {
                                grid.V(vi, vj, vk) += val;
                                grid.getWeightV(vi, vj, vk) += weight;
                            }
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
                            float val = weight * pvz;
                            if (mode != SimulationMode::SERIAL) {
                                
                                grid.W(wi, wj, wk) += val;
                                
                                grid.getWeightW(wi, wj, wk) += weight;
                            } else {
                                grid.W(wi, wj, wk) += val;
                                grid.getWeightW(wi, wj, wk) += weight;
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalize by weights
#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && grid.getNx() * grid.getNy() * grid.getNz() > 1024)
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
#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && grid.getNx() * grid.getNy() * grid.getNz() > 1024)
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
#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && grid.getNx() * grid.getNy() * grid.getNz() > 1024)
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
#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && grid.getNx() * grid.getNy() * grid.getNz() > 1024)
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
    const int nCells = static_cast<int>(cellType.size());

    grid.clearPressure();

    std::vector<float> pOld(nCells, 0.0f);
    std::vector<float> pNew(nCells, 0.0f);

    for (int iter = 0; iter < pressureIterations; ++iter) {
#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL)
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
                        if (t == WATER) sum += pOld[cellIndex(ni, nj, nk)];
                    };

                    consider(i - 1, j, k);
                    consider(i + 1, j, k);
                    consider(i, j - 1, k);
                    consider(i, j + 1, k);
                    consider(i, j, k - 1);
                    consider(i, j, k + 1);

                    pNew[idx] = (diag > 0) ? (sum - rhs * h * h) / static_cast<float>(diag) : 0.0f;
                }
            }
        }
        pOld.swap(pNew);
    }

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                grid.P(i, j, k) = pOld[cellIndex(i, j, k)];
            }
        }
    }

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
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

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
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

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
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

#pragma omp parallel sections num_threads(nthreads) if(mode != SimulationMode::SERIAL)
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

#pragma omp parallel for num_threads(nthreads) schedule(static) if(mode != SimulationMode::SERIAL)
    for (int pIdx = 0; pIdx < static_cast<int>(particles.size()); ++pIdx) {
        float px = particles.px[pIdx];
        float py = particles.py[pIdx];
        float pz = particles.pz[pIdx];
        float vx = particles.vx[pIdx];
        float vy = particles.vy[pIdx];
        float vz = particles.vz[pIdx];

        if (px < eps) {
            px = eps;
            if (vx < 0.0f) vx = 0.0f;
            vy = 0.0f;
            vz = 0.0f;
        }
        else if (px > maxX - eps) {
            px = maxX - eps;
            if (vx > 0.0f) vx = 0.0f;
            vy = 0.0f;
            vz = 0.0f;
        }

        if (py < eps) {
            py = eps;
            if (vy < 0.0f) vy = 0.0f;
            vx = 0.0f;
            vz = 0.0f;
        }
        else if (py > maxY - eps) {
            py = maxY - eps;
            if (vy > 0.0f) vy = 0.0f;
            vx = 0.0f;
            vz = 0.0f;
        }

        if (pz < eps) {
            pz = eps;
            if (vz < 0.0f) vz = 0.0f;
            vx = 0.0f;
            vy = 0.0f;
        }
        else if (pz > maxZ - eps) {
            pz = maxZ - eps;
            if (vz > 0.0f) vz = 0.0f;
            vx = 0.0f;
            vy = 0.0f;
        }

        particles.px[pIdx] = px;
        particles.py[pIdx] = py;
        particles.pz[pIdx] = pz;
        particles.vx[pIdx] = vx;
        particles.vy[pIdx] = vy;
        particles.vz[pIdx] = vz;
    }
}

struct Vec3_AVX2 {
    __m256 x, y, z;
};

static inline Vec3_AVX2 sampleMAC_AVX2(const MACGrid& g, __m256 px, __m256 py, __m256 pz) {
    float h = g.getDims();
    int nx = g.getNx();
    int ny = g.getNy();
    int nz = g.getNz();

    __m256 u = sampleComponent_AVX2(g.getUData(), px, py, pz, _mm256_setzero_ps(), _mm256_set1_ps(0.5f), _mm256_set1_ps(0.5f), nx, ny, nz, nx, ny - 1, nz - 1, nx + 1, ny, h);
    __m256 v = sampleComponent_AVX2(g.getVData(), px, py, pz, _mm256_set1_ps(0.5f), _mm256_setzero_ps(), _mm256_set1_ps(0.5f), nx, ny, nz, nx - 1, ny, nz - 1, nx, ny + 1, h);
    __m256 w = sampleComponent_AVX2(g.getWData(), px, py, pz, _mm256_set1_ps(0.5f), _mm256_set1_ps(0.5f), _mm256_setzero_ps(), nx, ny, nz, nx - 1, ny - 1, nz, nx, ny, h);

    return { u, v, w };
}

void FLIPSolver::gridToParticles(const MACGrid& oldGrid, float dt) {
    const int nParticles = static_cast<int>(particles.size());
    const int n8 = nParticles - (nParticles % 8);
    const __m256 v_flipRatio = _mm256_set1_ps(flipRatio);
    const __m256 v_oneMinusFlip = _mm256_set1_ps(1.0f - flipRatio);

#pragma omp parallel for num_threads(nthreads) schedule(static) if(mode != SimulationMode::SERIAL)
    for (int p = 0; p < n8; p += 8) {
        __m256 px = _mm256_loadu_ps(&particles.px[p]);
        __m256 py = _mm256_loadu_ps(&particles.py[p]);
        __m256 pz = _mm256_loadu_ps(&particles.pz[p]);
        __m256 vx = _mm256_loadu_ps(&particles.vx[p]);
        __m256 vy = _mm256_loadu_ps(&particles.vy[p]);
        __m256 vz = _mm256_loadu_ps(&particles.vz[p]);

        Vec3_AVX2 picVel = sampleMAC_AVX2(grid, px, py, pz);
        Vec3_AVX2 oldVel = sampleMAC_AVX2(oldGrid, px, py, pz);

        __m256 flipDeltaX = _mm256_sub_ps(picVel.x, oldVel.x);
        __m256 flipDeltaY = _mm256_sub_ps(picVel.y, oldVel.y);
        __m256 flipDeltaZ = _mm256_sub_ps(picVel.z, oldVel.z);

        __m256 flipVelX = _mm256_add_ps(vx, flipDeltaX);
        __m256 flipVelY = _mm256_add_ps(vy, flipDeltaY);
        __m256 flipVelZ = _mm256_add_ps(vz, flipDeltaZ);

        __m256 newVelX = _mm256_add_ps(_mm256_mul_ps(v_flipRatio, flipVelX), _mm256_mul_ps(v_oneMinusFlip, picVel.x));
        __m256 newVelY = _mm256_add_ps(_mm256_mul_ps(v_flipRatio, flipVelY), _mm256_mul_ps(v_oneMinusFlip, picVel.y));
        __m256 newVelZ = _mm256_add_ps(_mm256_mul_ps(v_flipRatio, flipVelZ), _mm256_mul_ps(v_oneMinusFlip, picVel.z));

        _mm256_storeu_ps(&particles.vx[p], newVelX);
        _mm256_storeu_ps(&particles.vy[p], newVelY);
        _mm256_storeu_ps(&particles.vz[p], newVelZ);
    }

    // Handle remaining particles
    for (int p = n8; p < nParticles; ++p) {
        Vec3 pos(particles.px[p], particles.py[p], particles.pz[p]);
        Vec3 vel(particles.vx[p], particles.vy[p], particles.vz[p]);

        const Vec3 picVel = sampleMAC(grid, pos);
        const Vec3 oldVel = sampleMAC(oldGrid, pos);
        const Vec3 flipDelta = picVel - oldVel;
        const Vec3 flipVel = vel + flipDelta;
        Vec3 newVel = flipRatio * flipVel + (1.0f - flipRatio) * picVel;

        particles.vx[p] = newVel.x;
        particles.vy[p] = newVel.y;
        particles.vz[p] = newVel.z;
    }
}

void FLIPSolver::advectParticles(float dt) {
    const float h = grid.getDims();
    const float eps = 0.05f * h;
    const float maxX = grid.getNx() * h;
    const float maxY = grid.getNy() * h;
    const float maxZ = grid.getNz() * h;
    // RK2 integration
#pragma omp parallel for num_threads(nthreads) schedule(static) if(mode != SimulationMode::SERIAL)
    for (int p = 0; p < static_cast<int>(particles.size()); ++p) {
        float t = 0.0f;
        Vec3 pos(particles.px[p], particles.py[p], particles.pz[p]);

        while (t < dt) {
            const Vec3 v0 = sampleMAC(grid, pos);
            const float speed = glm::length(v0);
            const float subDt = (speed > 1e-6f)
                ? std::min(dt - t, 0.9f * h / speed)
                : (dt - t);

            const Vec3 midPos = pos + 0.5f * subDt * v0;
            const Vec3 vMid = sampleMAC(grid, midPos);

            pos += subDt * vMid;
            t += subDt;

            pos.x = std::clamp(pos.x, eps, maxX - eps);
            pos.y = std::clamp(pos.y, eps, maxY - eps);
            pos.z = std::clamp(pos.z, eps, maxZ - eps);
        }

        particles.px[p] = pos.x;
        particles.py[p] = pos.y;
        particles.pz[p] = pos.z;
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

void FLIPSolver::solvePressureRBGS(float dt) {
    const int nx = grid.getNx();
    const int ny = grid.getNy();
    const int nz = grid.getNz();
    const float h = grid.getDims();
    const float safeDt = std::max(dt, 1e-6f);
    const int nCells = nx * ny * nz;

    grid.clearPressure();

    for (int iter = 0; iter < pressureIterations; ++iter) {
        for (int pass = 0; pass < 2; ++pass) {
#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    int i_start = (pass + j + k) % 2;
                    for (int i = i_start; i < nx; i += 2) {
                        const int idx = cellIndex(i, j, k);
                        if (cellType[idx] != WATER) {
                            grid.P(i, j, k) = 0.0f;
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
                                sum += grid.P(ni, nj, nk);
                            }
                        };

                        consider(i - 1, j, k);
                        consider(i + 1, j, k);
                        consider(i, j - 1, k);
                        consider(i, j + 1, k);
                        consider(i, j, k - 1);
                        consider(i, j, k + 1);

                        if (diag > 0) {
                            grid.P(i, j, k) = (sum - rhs * h * h) / static_cast<float>(diag);
                        } else {
                            grid.P(i, j, k) = 0.0f;
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
    for (int i = 1; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const CellType leftType = cellType[cellIndex(i - 1, j, k)];
                const CellType rightType = cellType[cellIndex(i, j, k)];
                if (leftType == SOLID || rightType == SOLID) { grid.U(i, j, k) = 0.0f; continue; }
                if (leftType == AIR && rightType == AIR) continue;
                const float pL = (leftType == WATER) ? grid.P(i - 1, j, k) : 0.0f;
                const float pR = (rightType == WATER) ? grid.P(i, j, k) : 0.0f;
                grid.U(i, j, k) -= (safeDt / materialDensity) * (pR - pL) / h;
            }
        }
    }

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
    for (int i = 0; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const CellType downType = cellType[cellIndex(i, j - 1, k)];
                const CellType upType = cellType[cellIndex(i, j, k)];
                if (downType == SOLID || upType == SOLID) { grid.V(i, j, k) = 0.0f; continue; }
                if (downType == AIR && upType == AIR) continue;
                const float pD = (downType == WATER) ? grid.P(i, j - 1, k) : 0.0f;
                const float pU = (upType == WATER) ? grid.P(i, j, k) : 0.0f;
                grid.V(i, j, k) -= (safeDt / materialDensity) * (pU - pD) / h;
            }
        }
    }

#pragma omp parallel for num_threads(nthreads) if(mode != SimulationMode::SERIAL && nCells > 1024)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 1; k < nz; ++k) {
                const CellType backType = cellType[cellIndex(i, j, k - 1)];
                const CellType frontType = cellType[cellIndex(i, j, k)];
                if (backType == SOLID || frontType == SOLID) { grid.W(i, j, k) = 0.0f; continue; }
                if (backType == AIR && frontType == AIR) continue;
                const float pB = (backType == WATER) ? grid.P(i, j, k - 1) : 0.0f;
                const float pF = (frontType == WATER) ? grid.P(i, j, k) : 0.0f;
                grid.W(i, j, k) -= (safeDt / materialDensity) * (pF - pB) / h;
            }
        }
    }
}




