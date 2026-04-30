#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <glm/glm.hpp>

using Vec3 = glm::vec3;

struct GridNode {
	float val = 0.0f;
	float weight = 0.0f;
};

class MACGrid {
public:
	MACGrid(int nx, int ny, int nz, float unit) 
		: nx(nx), ny(ny), nz(nz), unit(unit),
		u((nx + 1) * ny * nz),
		v(nx * (ny + 1) * nz),
		w(nx * ny * (nz + 1)),
		p(nx * ny * nz, 0.0f)
	{
	}

	int getNx() const { return nx; }
	int getNy() const { return ny; }
	int getNz() const { return nz; }
	float getDims() const { return unit; }

	// U velocity accessors
	float& U(int i, int j, int k) {
		assert(i >= 0 && i <= nx && j >= 0 && j < ny && k >= 0 && k < nz);
		return u[i + (nx + 1) * (j + ny * k)].val;
	}
	float U(int i, int j, int k) const {
		assert(i >= 0 && i <= nx && j >= 0 && j < ny && k >= 0 && k < nz);
		return u[i + (nx + 1) * (j + ny * k)].val;
	}

	// V velocity accessors
	float& V(int i, int j, int k) {
		assert(i >= 0 && i < nx && j >= 0 && j <= ny && k >= 0 && k < nz);
		return v[i + nx * (j + (ny + 1) * k)].val;
	}
	float V(int i, int j, int k) const {
		assert(i >= 0 && i < nx && j >= 0 && j <= ny && k >= 0 && k < nz);
		return v[i + nx * (j + (ny + 1) * k)].val;
	}

	// W velocity accessors
	float& W(int i, int j, int k) {
		assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k <= nz);
		return w[i + nx * (j + ny * k)].val;
	}
	float W(int i, int j, int k) const {
		assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k <= nz);
		return w[i + nx * (j + ny * k)].val;
	}

	// Pressure accessors
	float& P(int i, int j, int k) {
		assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz);
		return p[i + nx * (j + ny * k)];
	}
	float P(int i, int j, int k) const {
		assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz);
		return p[i + nx * (j + ny * k)];
	}

	// Weight accessors
	float& getWeightU(int i, int j, int k) {
		assert(i >= 0 && i <= nx && j >= 0 && j < ny && k >= 0 && k < nz);
		return u[i + (nx + 1) * (j + ny * k)].weight;
	}
	float getWeightU(int i, int j, int k) const {
		assert(i >= 0 && i <= nx && j >= 0 && j < ny && k >= 0 && k < nz);
		return u[i + (nx + 1) * (j + ny * k)].weight;
	}

	float& getWeightV(int i, int j, int k) {
		assert(i >= 0 && i < nx && j >= 0 && j <= ny && k >= 0 && k < nz);
		return v[i + nx * (j + (ny + 1) * k)].weight;
	}
	float getWeightV(int i, int j, int k) const {
		assert(i >= 0 && i < nx && j >= 0 && j <= ny && k >= 0 && k < nz);
		return v[i + nx * (j + (ny + 1) * k)].weight;
	}

	float& getWeightW(int i, int j, int k) {
		assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k <= nz);
		return w[i + nx * (j + ny * k)].weight;
	}
	float getWeightW(int i, int j, int k) const {
		assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k <= nz);
		return w[i + nx * (j + ny * k)].weight;
	}

	void clearVelocities() {
		for (auto& n : u) n.val = 0.0f;
		for (auto& n : v) n.val = 0.0f;
		for (auto& n : w) n.val = 0.0f;
	}

	void clearWeights() {
		for (auto& n : u) n.weight = 0.0f;
		for (auto& n : v) n.weight = 0.0f;
		for (auto& n : w) n.weight = 0.0f;
	}

	void clearPressure() {
		std::fill(p.begin(), p.end(), 0.0f);
	}

	const GridNode* getUData() const { return u.data(); }
	const GridNode* getVData() const { return v.data(); }
	const GridNode* getWData() const { return w.data(); }

	float divergence(int i, int j, int k) const {
		float du = U(i + 1, j, k) - U(i, j, k);
		float dv = V(i, j + 1, k) - V(i, j, k);
		float dw = W(i, j, k + 1) - W(i, j, k);
		return (du + dv + dw) / unit;
	}

	float cellCenterU(int i, int j, int k) const {
		return 0.5f * (U(i, j, k) + U(i + 1, j, k));
	}

	float cellCenterV(int i, int j, int k) const {
		return 0.5f * (V(i, j, k) + V(i, j + 1, k));
	}

	float cellCenterW(int i, int j, int k) const {
		return 0.5f * (W(i, j, k) + W(i, j, k + 1));
	}

private:
	int nx, ny, nz;
	float unit;

	std::vector<GridNode> u;
	std::vector<GridNode> v;
	std::vector<GridNode> w;
	std::vector<float> p;
};
