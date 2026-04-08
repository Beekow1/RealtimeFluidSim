#include "FLIPSolver.hpp"
#include <algorithm>
#include <cmath>

// ============================================================================
// Helper Functions (static to this file)
// ============================================================================

static inline int clamp(int val, int minVal, int maxVal) {
	return std::max(minVal, std::min(val, maxVal));
}

static inline float lerp(float a, float b, float t) {
	return a + t * (b - a);
}

static inline float trilinearWeight(float fx, float fy, float fz, int wi, int wj, int wk) {
	float wx = (wi == 0) ? (1.0f - fx) : fx;
	float wy = (wj == 0) ? (1.0f - fy) : fy;
	float wz = (wk == 0) ? (1.0f - fz) : fz;
	return wx * wy * wz;
}

// ============================================================================
// Main Solver Implementation
// ============================================================================

void FLIPSolver::step(float dt) {
	// Mark which cells contain fluid
	markFluidCells();

	// Transfer particle velocities to grid
	particlesToGrid();

	// Add gravity
	addGravity(dt);

	// Solve for pressure
	solvePressure(dt);

	// Apply boundary conditions
	applyBoundaryConditions();

	// Transfer grid velocities back to particles
	// Note: This requires oldGrid to be saved before modifications
	MACGrid oldGrid = grid;
	gridToParticles(oldGrid, dt);

	// Advect particles
	advectParticles(dt);

	// Apply boundary conditions again
	applyBoundaryConditions();
}

void FLIPSolver::markFluidCells() {
	// Mark all cells as AIR initially
	std::fill(cellType.begin(), cellType.end(), AIR);

	// Mark cells containing particles as WATER
	float h = grid.getDims();
	for (const auto& particle : particles) {
		int i = static_cast<int>(particle.pos.x / h);
		int j = static_cast<int>(particle.pos.y / h);
		int k = static_cast<int>(particle.pos.z / h);

		i = clamp(i, 0, grid.getNx() - 1);
		j = clamp(j, 0, grid.getNy() - 1);
		k = clamp(k, 0, grid.getNz() - 1);

		cellType[i + grid.getNx() * (j + grid.getNy() * k)] = WATER;
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
		i = clamp(i, 0, grid.getNx() - 1);
		j = clamp(j, 0, grid.getNy() - 1);
		k = clamp(k, 0, grid.getNz() - 1);
		
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

	for (int i = 0; i < grid.getNx(); ++i) {
		for (int j = 0; j <= grid.getNy(); ++j) {
			for (int k = 0; k < grid.getNz(); ++k) {
				grid.V(i, j, k) += gravity.y * dt;
			}
		}
	}
}

void FLIPSolver::solvePressure(float dt) {
	// TODO: Implement pressure projection solver
	(void)dt;
}

void FLIPSolver::applyBoundaryConditions() {
	float h = grid.getDims();

	// Simple boundary: clamp particles to domain
	for (auto& particle : particles) {
		particle.pos.x = std::max(0.0f, std::min(particle.pos.x, grid.getNx() * h));
		particle.pos.y = std::max(0.0f, std::min(particle.pos.y, grid.getNy() * h));
		particle.pos.z = std::max(0.0f, std::min(particle.pos.z, grid.getNz() * h));
	}
}

void FLIPSolver::advectParticles(float dt) {
	for (auto& particle : particles) {
		Vec3 vel = sampleMAC(grid, particle.pos);
		particle.pos += vel * dt;
	}
}

void FLIPSolver::gridToParticles(const MACGrid& oldGrid, float dt) {
	static constexpr float FLIP_RATIO = 0.95f;

	// FLIP: blend between grid velocity and particle velocity
	for (auto& particle : particles) {
		Vec3 gridVel = sampleMAC(grid, particle.pos);
		Vec3 oldGridVel = sampleMAC(oldGrid, particle.pos);
		Vec3 velDiff = gridVel - oldGridVel;

		// FLIP combines particle and grid velocity updates
		particle.vel = particle.vel + velDiff;
		particle.vel = FLIP_RATIO * particle.vel + (1.0f - FLIP_RATIO) * gridVel;
	}
}
