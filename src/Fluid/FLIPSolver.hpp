#pragma once

#include "MACGrid.hpp"
#include "FLIPParticle.hpp"
#include <vector>
#include <glm/glm.hpp>

using Vec3 = glm::vec3;

enum CellType {
	AIR,
	WATER,
	SOLID
};

class FLIPSolver {
public:
	FLIPSolver(int nx, int ny, int nz, float h)
		: grid(nx, ny, nz, h),
		cellType(nx * ny * nz, AIR) {}

	void step(float dt);
	
	std::vector<Particle>& getParticles() { return particles; }
	const std::vector<Particle>& getParticles() const { return particles; }
	
	void addParticles(const std::vector<Particle>& newParticles) {
		particles.insert(particles.end(), newParticles.begin(), newParticles.end());
	}

private:
	MACGrid grid;
	std::vector<Particle> particles;
	std::vector<CellType> cellType;

	void advectParticles(float dt);
	void markFluidCells();
	void particlesToGrid();
	void addGravity(float dt);
	void solvePressure(float dt);
	void applyBoundaryConditions();
	void gridToParticles(const MACGrid& oldGrid, float dt);

	Vec3 sampleMAC(const MACGrid& g, const Vec3& x) const;
};