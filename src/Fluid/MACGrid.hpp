#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <glm/glm.hpp>

using Vec3 = glm::vec3;

class MACGrid {
public:
	MACGrid(int nx, int ny, int nz, float unit) 
		: nx(nx), ny(ny), nz(nz), unit(unit),
		u((nx + 1) * ny * nz, 0.0f),
		v(nx * (ny + 1) * nz, 0.0f),
		w(nx * ny * (nz + 1), 0.0f),
		p(nx * ny * nz, 0.0f),
		weightU((nx + 1) * ny * nz, 0.0f),
		weightV(nx * (ny + 1) * nz, 0.0f),
		weightW(nx * ny * (nz + 1), 0.0f)
	{
	}

	int getNx() const { return nx; }
	int getNy() const { return ny; }
	int getNz() const { return nz; }
	float getDims() const { return unit; }

	// U velocity accessors
	float& U(int i, int j, int k) {
		assert(i >= 0 && i <= nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);
		return u[i + (nx + 1) * (j + ny * k)];
	}
	float U(int i, int j, int k) const {
		assert(i >= 0 && i <= nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);
		return u[i + (nx + 1) * (j + ny * k)];
	}

	// V velocity accessors
	float& V(int i, int j, int k) {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j <= ny);
		assert(k >= 0 && k < nz);
		return v[i + nx * (j + (ny + 1) * k)];
	}
	float V(int i, int j, int k) const {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j <= ny);
		assert(k >= 0 && k < nz);
		return v[i + nx * (j + (ny + 1) * k)];
	}

	// W velocity accessors
	float& W(int i, int j, int k) {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k <= nz);
		return w[i + nx * (j + ny * k)];
	}
	float W(int i, int j, int k) const {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k <= nz);
		return w[i + nx * (j + ny * k)];
	}

	// Pressure accessors
	float& P(int i, int j, int k) {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);
		return p[i + nx * (j + ny * k)];
	}
	float P(int i, int j, int k) const {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);
		return p[i + nx * (j + ny * k)];
	}

	// Weight accessors for U
	float& getWeightU(int i, int j, int k) {
		assert(i >= 0 && i <= nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);
		return weightU[i + (nx + 1) * (j + ny * k)];
	}
	float getWeightU(int i, int j, int k) const {
		assert(i >= 0 && i <= nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);
		return weightU[i + (nx + 1) * (j + ny * k)];
	}

	// Weight accessors for V
	float& getWeightV(int i, int j, int k) {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j <= ny);
		assert(k >= 0 && k < nz);
		return weightV[i + nx * (j + (ny + 1) * k)];
	}
	float getWeightV(int i, int j, int k) const {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j <= ny);
		assert(k >= 0 && k < nz);
		return weightV[i + nx * (j + (ny + 1) * k)];
	}

	// Weight accessors for W
	float& getWeightW(int i, int j, int k) {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k <= nz);
		return weightW[i + nx * (j + ny * k)];
	}
	float getWeightW(int i, int j, int k) const {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k <= nz);
		return weightW[i + nx * (j + ny * k)];
	}

	void clearVelocities() {
		std::fill(u.begin(), u.end(), 0.0f);
		std::fill(v.begin(), v.end(), 0.0f);
		std::fill(w.begin(), w.end(), 0.0f);
	}

	void clearWeights() {
		std::fill(weightU.begin(), weightU.end(), 0.0f);
		std::fill(weightV.begin(), weightV.end(), 0.0f);
		std::fill(weightW.begin(), weightW.end(), 0.0f);
	}

	void clearPressure() {
		std::fill(p.begin(), p.end(), 0.0f);
	}

	float divergence(int i, int j, int k) const {
		assert(i >= 0 && i < nx);
		assert(j >= 0 && j < ny);
		assert(k >= 0 && k < nz);

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

	std::vector<float> u;
	std::vector<float> v;
	std::vector<float> w;
	std::vector<float> p;

	std::vector<float> weightU;
	std::vector<float> weightV;
	std::vector<float> weightW;
};