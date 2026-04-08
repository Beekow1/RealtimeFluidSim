#include "FLIPSolver.hpp"
#include <algorithm>
#include <cmath>

// ============================================================================
// Helper Functions
// ============================================================================

static inline int clamp(int val, int minVal, int maxVal) {
	return std::max(minVal, std::min(val, maxVal));
}

static inline float lerp(float a, float b, float t) {
	return a + t * (b - a);
}

// ============================================================================
// Interpolation Implementation
// ============================================================================

static float sampleU(const MACGrid& g, const Vec3& x) {
	float h = g.getDims();

	// U is located at (i+0.5, j, k)
	Vec3 uPos = x / h - Vec3(0.5f, 0.0f, 0.0f);

	int i = static_cast<int>(std::floor(uPos.x));
	int j = static_cast<int>(std::floor(uPos.y));
	int k = static_cast<int>(std::floor(uPos.z));

	float fx = uPos.x - i;
	float fy = uPos.y - j;
	float fz = uPos.z - k;

	i = clamp(i, 0, g.getNx() - 1);
	j = clamp(j, 0, g.getNy() - 1);
	k = clamp(k, 0, g.getNz() - 1);

	fx = std::max(0.0f, std::min(1.0f, fx));
	fy = std::max(0.0f, std::min(1.0f, fy));
	fz = std::max(0.0f, std::min(1.0f, fz));

	// Trilinear interpolation
	float u000 = g.U(i, j, k);
	float u100 = g.U(i + 1, j, k);
	float u010 = g.U(i, j + 1, k);
	float u110 = g.U(i + 1, j + 1, k);
	float u001 = g.U(i, j, k + 1);
	float u101 = g.U(i + 1, j, k + 1);
	float u011 = g.U(i, j + 1, k + 1);
	float u111 = g.U(i + 1, j + 1, k + 1);

	float u00 = lerp(u000, u100, fx);
	float u10 = lerp(u010, u110, fx);
	float u01 = lerp(u001, u101, fx);
	float u11 = lerp(u011, u111, fx);

	float u0 = lerp(u00, u10, fy);
	float u1 = lerp(u01, u11, fy);

	return lerp(u0, u1, fz);
}

static float sampleV(const MACGrid& g, const Vec3& x) {
	float h = g.getDims();

	// V is located at (i, j+0.5, k)
	Vec3 vPos = x / h - Vec3(0.0f, 0.5f, 0.0f);

	int i = static_cast<int>(std::floor(vPos.x));
	int j = static_cast<int>(std::floor(vPos.y));
	int k = static_cast<int>(std::floor(vPos.z));

	float fx = vPos.x - i;
	float fy = vPos.y - j;
	float fz = vPos.z - k;

	i = clamp(i, 0, g.getNx() - 2);
	j = clamp(j, 0, g.getNy() - 1);
	k = clamp(k, 0, g.getNz() - 2);

	fx = std::max(0.0f, std::min(1.0f, fx));
	fy = std::max(0.0f, std::min(1.0f, fy));
	fz = std::max(0.0f, std::min(1.0f, fz));

	float v000 = g.V(i, j, k);
	float v100 = g.V(i + 1, j, k);
	float v010 = g.V(i, j + 1, k);
	float v110 = g.V(i + 1, j + 1, k);
	float v001 = g.V(i, j, k + 1);
	float v101 = g.V(i + 1, j, k + 1);
	float v011 = g.V(i, j + 1, k + 1);
	float v111 = g.V(i + 1, j + 1, k + 1);

	float v00 = lerp(v000, v100, fx);
	float v10 = lerp(v010, v110, fx);
	float v01 = lerp(v001, v101, fx);
	float v11 = lerp(v011, v111, fx);

	float v0 = lerp(v00, v10, fy);
	float v1 = lerp(v01, v11, fy);

	return lerp(v0, v1, fz);
}

static float sampleW(const MACGrid& g, const Vec3& x) {
	float h = g.getDims();

	// W is located at (i, j, k+0.5)
	Vec3 wPos = x / h - Vec3(0.0f, 0.0f, 0.5f);

	int i = static_cast<int>(std::floor(wPos.x));
	int j = static_cast<int>(std::floor(wPos.y));
	int k = static_cast<int>(std::floor(wPos.z));

	float fx = wPos.x - i;
	float fy = wPos.y - j;
	float fz = wPos.z - k;

	i = clamp(i, 0, g.getNx() - 1);
	j = clamp(j, 0, g.getNy() - 1);
	k = clamp(k, 0, g.getNz() - 1);

	fx = std::max(0.0f, std::min(1.0f, fx));
	fy = std::max(0.0f, std::min(1.0f, fy));
	fz = std::max(0.0f, std::min(1.0f, fz));

	// Trilinear interpolation
	float w000 = g.W(i, j, k);
	float w100 = g.W(i + 1, j, k);
	float w010 = g.W(i, j + 1, k);
	float w110 = g.W(i + 1, j + 1, k);
	float w001 = g.W(i, j, k + 1);
	float w101 = g.W(i + 1, j, k + 1);
	float w011 = g.W(i, j + 1, k + 1);
	float w111 = g.W(i + 1, j + 1, k + 1);

	float w00 = lerp(w000, w100, fx);
	float w10 = lerp(w010, w110, fx);
	float w01 = lerp(w001, w101, fx);
	float w11 = lerp(w011, w111, fx);

	float w0 = lerp(w00, w10, fy);
	float w1 = lerp(w01, w11, fy);

	return lerp(w0, w1, fz);
}

// ============================================================================
// Public Member Function Implementation
// ============================================================================

Vec3 FLIPSolver::sampleMAC(const MACGrid& g, const Vec3& x) const {
	return Vec3(
		sampleU(g, x),
		sampleV(g, x),
		sampleW(g, x)
	);
}
