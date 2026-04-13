#include "FLIPSolver.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
enum CellTypeGPU : std::uint8_t {
    AIR_GPU = 0,
    WATER_GPU = 1,
    SOLID_GPU = 2,
};

struct DeviceGrid {
    int nx;
    int ny;
    int nz;
    float h;

    float* u;
    float* v;
    float* w;

    float* weightU;
    float* weightV;
    float* weightW;
};

__host__ __device__ inline int cellIndex(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

__host__ __device__ inline int uIndex(int i, int j, int k, int nx, int ny) {
    return i + (nx + 1) * (j + ny * k);
}

__host__ __device__ inline int vIndex(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + (ny + 1) * k);
}

__host__ __device__ inline int wIndex(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ float trilinearWeight(float fx, float fy, float fz, int wi, int wj, int wk) {
    const float wx = (wi == 0) ? (1.0f - fx) : fx;
    const float wy = (wj == 0) ? (1.0f - fy) : fy;
    const float wz = (wk == 0) ? (1.0f - fz) : fz;
    return wx * wy * wz;
}

__device__ __forceinline__ float3 add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 scale3(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float length3(const float3& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ __forceinline__ float componentAt(const DeviceGrid& g, int component, int i, int j, int k) {
    if (component == 0) {
        return g.u[uIndex(i, j, k, g.nx, g.ny)];
    }
    if (component == 1) {
        return g.v[vIndex(i, j, k, g.nx, g.ny)];
    }
    return g.w[wIndex(i, j, k, g.nx, g.ny)];
}

__device__ float sampleComponent(
    const DeviceGrid& g,
    const float3& pos,
    const float3& offset,
    int maxI,
    int maxJ,
    int maxK,
    int component)
{
    const float invH = 1.0f / g.h;
    const float3 p = make_float3(
        pos.x * invH - offset.x,
        pos.y * invH - offset.y,
        pos.z * invH - offset.z);

    const int i = static_cast<int>(floorf(p.x));
    const int j = static_cast<int>(floorf(p.y));
    const int k = static_cast<int>(floorf(p.z));

    const float fx = p.x - static_cast<float>(i);
    const float fy = p.y - static_cast<float>(j);
    const float fz = p.z - static_cast<float>(k);

    float value = 0.0f;

#define ACCUM_SAMPLE(di, dj, dk)                                                                         \
    do {                                                                                                  \
        const int ii = i + (di);                                                                          \
        const int jj = j + (dj);                                                                          \
        const int kk = k + (dk);                                                                          \
        if (ii >= 0 && ii <= maxI && jj >= 0 && jj <= maxJ && kk >= 0 && kk <= maxK) {                  \
            value += trilinearWeight(fx, fy, fz, di, dj, dk) * componentAt(g, component, ii, jj, kk);    \
        }                                                                                                 \
    } while (0)

    ACCUM_SAMPLE(0, 0, 0);
    ACCUM_SAMPLE(0, 0, 1);
    ACCUM_SAMPLE(0, 1, 0);
    ACCUM_SAMPLE(0, 1, 1);
    ACCUM_SAMPLE(1, 0, 0);
    ACCUM_SAMPLE(1, 0, 1);
    ACCUM_SAMPLE(1, 1, 0);
    ACCUM_SAMPLE(1, 1, 1);

#undef ACCUM_SAMPLE

    return value;
}

__device__ float3 sampleMAC(const DeviceGrid& g, const float3& x) {
    const float maxX = g.nx * g.h;
    const float maxY = g.ny * g.h;
    const float maxZ = g.nz * g.h;

    const float3 xc = make_float3(
        clampf(x.x, 0.0f, maxX - 1e-5f),
        clampf(x.y, 0.0f, maxY - 1e-5f),
        clampf(x.z, 0.0f, maxZ - 1e-5f));

    const float u = sampleComponent(g, xc, make_float3(0.0f, 0.5f, 0.5f), g.nx,     g.ny - 1, g.nz - 1, 0);
    const float v = sampleComponent(g, xc, make_float3(0.5f, 0.0f, 0.5f), g.nx - 1, g.ny,     g.nz - 1, 1);
    const float w = sampleComponent(g, xc, make_float3(0.5f, 0.5f, 0.0f), g.nx - 1, g.ny - 1, g.nz,     2);

    return make_float3(u, v, w);
}

__device__ float divergenceAt(const DeviceGrid& g, int i, int j, int k) {
    const float du = g.u[uIndex(i + 1, j, k, g.nx, g.ny)] - g.u[uIndex(i, j, k, g.nx, g.ny)];
    const float dv = g.v[vIndex(i, j + 1, k, g.nx, g.ny)] - g.v[vIndex(i, j, k, g.nx, g.ny)];
    const float dw = g.w[wIndex(i, j, k + 1, g.nx, g.ny)] - g.w[wIndex(i, j, k, g.nx, g.ny)];
    return (du + dv + dw) / g.h;
}

__global__ void markFluidCellsKernel(
    const float4* pos,
    int particleCount,
    std::uint8_t* cellType,
    int nx,
    int ny,
    int nz,
    float h)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particleCount) {
        return;
    }

    const float maxX = nx * h;
    const float maxY = ny * h;
    const float maxZ = nz * h;

    const float4 pp = pos[p];
    const float x = clampf(pp.x, 0.0f, maxX - 1e-5f);
    const float y = clampf(pp.y, 0.0f, maxY - 1e-5f);
    const float z = clampf(pp.z, 0.0f, maxZ - 1e-5f);

    const int i = min(max(static_cast<int>(x / h), 0), nx - 1);
    const int j = min(max(static_cast<int>(y / h), 0), ny - 1);
    const int k = min(max(static_cast<int>(z / h), 0), nz - 1);

    cellType[cellIndex(i, j, k, nx, ny)] = WATER_GPU;
}

__global__ void particlesToGridKernel(DeviceGrid g, const float4* pos, const float4* vel, int particleCount) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particleCount) {
        return;
    }

    const float invH = 1.0f / g.h;
    const float4 xp = pos[p];
    const float4 vp = vel[p];

    const float gx = xp.x * invH;
    const float gy = xp.y * invH;
    const float gz = xp.z * invH;

    int i = static_cast<int>(floorf(gx));
    int j = static_cast<int>(floorf(gy));
    int k = static_cast<int>(floorf(gz));

    float fx = gx - static_cast<float>(i);
    float fy = gy - static_cast<float>(j);
    float fz = gz - static_cast<float>(k);

    i = min(max(i, 0), g.nx - 1);
    j = min(max(j, 0), g.ny - 1);
    k = min(max(k, 0), g.nz - 1);

    fx = clampf(fx, 0.0f, 1.0f);
    fy = clampf(fy, 0.0f, 1.0f);
    fz = clampf(fz, 0.0f, 1.0f);

    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                const float weight = trilinearWeight(fx, fy, fz, di, dj, dk);

                const int ui = i + di;
                const int uj = j + dj;
                const int uk = k + dk;
                if (ui >= 0 && ui <= g.nx && uj >= 0 && uj < g.ny && uk >= 0 && uk < g.nz) {
                    const int idx = uIndex(ui, uj, uk, g.nx, g.ny);
                    atomicAdd(&g.u[idx], weight * vp.x);
                    atomicAdd(&g.weightU[idx], weight);
                }

                const int vi = i + di;
                const int vj = j + dj;
                const int vk = k + dk;
                if (vi >= 0 && vi < g.nx && vj >= 0 && vj <= g.ny && vk >= 0 && vk < g.nz) {
                    const int idx = vIndex(vi, vj, vk, g.nx, g.ny);
                    atomicAdd(&g.v[idx], weight * vp.y);
                    atomicAdd(&g.weightV[idx], weight);
                }

                const int wi = i + di;
                const int wj = j + dj;
                const int wk = k + dk;
                if (wi >= 0 && wi < g.nx && wj >= 0 && wj < g.ny && wk >= 0 && wk <= g.nz) {
                    const int idx = wIndex(wi, wj, wk, g.nx, g.ny);
                    atomicAdd(&g.w[idx], weight * vp.z);
                    atomicAdd(&g.weightW[idx], weight);
                }
            }
        }
    }
}

__global__ void normalizeUKernel(DeviceGrid g) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (g.nx + 1) * g.ny * g.nz;
    if (idx >= count) {
        return;
    }
    const float w = g.weightU[idx];
    if (w > 0.0f) {
        g.u[idx] /= w;
    }
}

__global__ void normalizeVKernel(DeviceGrid g) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = g.nx * (g.ny + 1) * g.nz;
    if (idx >= count) {
        return;
    }
    const float w = g.weightV[idx];
    if (w > 0.0f) {
        g.v[idx] /= w;
    }
}

__global__ void normalizeWKernel(DeviceGrid g) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = g.nx * g.ny * (g.nz + 1);
    if (idx >= count) {
        return;
    }
    const float w = g.weightW[idx];
    if (w > 0.0f) {
        g.w[idx] /= w;
    }
}

__global__ void addGravityKernel(float* v, int nx, int ny, int nz, float dt) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) {
        return;
    }
    v[idx] += -9.81f * dt;
}

__global__ void enforceUBoundaryKernel(float* u, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx + 1) * ny * nz;
    if (idx >= count) {
        return;
    }
    const int i = idx % (nx + 1);
    if (i == 0 || i == nx) {
        u[idx] = 0.0f;
    }
}

__global__ void enforceVBoundaryKernel(float* v, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny + 1) * nz;
    if (idx >= count) {
        return;
    }
    const int j = (idx / nx) % (ny + 1);
    if (j == 0 || j == ny) {
        v[idx] = 0.0f;
    }
}

__global__ void enforceWBoundaryKernel(float* w, int nx, int ny, int nz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz + 1);
    if (idx >= count) {
        return;
    }
    const int k = idx / (nx * ny);
    if (k == 0 || k == nz) {
        w[idx] = 0.0f;
    }
}

__global__ void jacobiPressureKernel(
    DeviceGrid g,
    const std::uint8_t* cellType,
    const float* pOld,
    float* pNew,
    float density,
    float dt)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nCells = g.nx * g.ny * g.nz;
    if (idx >= nCells) {
        return;
    }

    if (cellType[idx] != WATER_GPU) {
        pNew[idx] = 0.0f;
        return;
    }

    const int i = idx % g.nx;
    const int j = (idx / g.nx) % g.ny;
    const int k = idx / (g.nx * g.ny);

    const float safeDt = fmaxf(dt, 1e-6f);
    const float rhs = (density / safeDt) * divergenceAt(g, i, j, k);

    float sum = 0.0f;
    int diag = 0;

    if (i - 1 >= 0) {
        const int nidx = cellIndex(i - 1, j, k, g.nx, g.ny);
        const std::uint8_t t = cellType[nidx];
        if (t != SOLID_GPU) {
            ++diag;
            if (t == WATER_GPU) {
                sum += pOld[nidx];
            }
        }
    }
    if (i + 1 < g.nx) {
        const int nidx = cellIndex(i + 1, j, k, g.nx, g.ny);
        const std::uint8_t t = cellType[nidx];
        if (t != SOLID_GPU) {
            ++diag;
            if (t == WATER_GPU) {
                sum += pOld[nidx];
            }
        }
    }
    if (j - 1 >= 0) {
        const int nidx = cellIndex(i, j - 1, k, g.nx, g.ny);
        const std::uint8_t t = cellType[nidx];
        if (t != SOLID_GPU) {
            ++diag;
            if (t == WATER_GPU) {
                sum += pOld[nidx];
            }
        }
    }
    if (j + 1 < g.ny) {
        const int nidx = cellIndex(i, j + 1, k, g.nx, g.ny);
        const std::uint8_t t = cellType[nidx];
        if (t != SOLID_GPU) {
            ++diag;
            if (t == WATER_GPU) {
                sum += pOld[nidx];
            }
        }
    }
    if (k - 1 >= 0) {
        const int nidx = cellIndex(i, j, k - 1, g.nx, g.ny);
        const std::uint8_t t = cellType[nidx];
        if (t != SOLID_GPU) {
            ++diag;
            if (t == WATER_GPU) {
                sum += pOld[nidx];
            }
        }
    }
    if (k + 1 < g.nz) {
        const int nidx = cellIndex(i, j, k + 1, g.nx, g.ny);
        const std::uint8_t t = cellType[nidx];
        if (t != SOLID_GPU) {
            ++diag;
            if (t == WATER_GPU) {
                sum += pOld[nidx];
            }
        }
    }

    pNew[idx] = (diag > 0) ? (sum - rhs * g.h * g.h) / static_cast<float>(diag) : 0.0f;
}

__global__ void applyPressureUKernel(
    float* u,
    int nx,
    int ny,
    int nz,
    float h,
    float density,
    float dt,
    const std::uint8_t* cellType,
    const float* pressure)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (nx - 1) * ny * nz;
    if (idx >= count) {
        return;
    }

    const int i = 1 + (idx % (nx - 1));
    const int j = (idx / (nx - 1)) % ny;
    const int k = idx / ((nx - 1) * ny);

    const std::uint8_t leftType = cellType[cellIndex(i - 1, j, k, nx, ny)];
    const std::uint8_t rightType = cellType[cellIndex(i, j, k, nx, ny)];

    const int faceIdx = uIndex(i, j, k, nx, ny);

    if (leftType == SOLID_GPU || rightType == SOLID_GPU) {
        u[faceIdx] = 0.0f;
        return;
    }

    if (leftType == AIR_GPU && rightType == AIR_GPU) {
        return;
    }

    const float pL = (leftType == WATER_GPU) ? pressure[cellIndex(i - 1, j, k, nx, ny)] : 0.0f;
    const float pR = (rightType == WATER_GPU) ? pressure[cellIndex(i, j, k, nx, ny)] : 0.0f;
    const float safeDt = fmaxf(dt, 1e-6f);

    u[faceIdx] -= (safeDt / density) * (pR - pL) / h;
}

__global__ void applyPressureVKernel(
    float* v,
    int nx,
    int ny,
    int nz,
    float h,
    float density,
    float dt,
    const std::uint8_t* cellType,
    const float* pressure)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * (ny - 1) * nz;
    if (idx >= count) {
        return;
    }

    const int i = idx % nx;
    const int j = 1 + ((idx / nx) % (ny - 1));
    const int k = idx / (nx * (ny - 1));

    const std::uint8_t downType = cellType[cellIndex(i, j - 1, k, nx, ny)];
    const std::uint8_t upType = cellType[cellIndex(i, j, k, nx, ny)];

    const int faceIdx = vIndex(i, j, k, nx, ny);

    if (downType == SOLID_GPU || upType == SOLID_GPU) {
        v[faceIdx] = 0.0f;
        return;
    }

    if (downType == AIR_GPU && upType == AIR_GPU) {
        return;
    }

    const float pD = (downType == WATER_GPU) ? pressure[cellIndex(i, j - 1, k, nx, ny)] : 0.0f;
    const float pU = (upType == WATER_GPU) ? pressure[cellIndex(i, j, k, nx, ny)] : 0.0f;
    const float safeDt = fmaxf(dt, 1e-6f);

    v[faceIdx] -= (safeDt / density) * (pU - pD) / h;
}

__global__ void applyPressureWKernel(
    float* w,
    int nx,
    int ny,
    int nz,
    float h,
    float density,
    float dt,
    const std::uint8_t* cellType,
    const float* pressure)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * (nz - 1);
    if (idx >= count) {
        return;
    }

    const int i = idx % nx;
    const int j = (idx / nx) % ny;
    const int k = 1 + idx / (nx * ny);

    const std::uint8_t backType = cellType[cellIndex(i, j, k - 1, nx, ny)];
    const std::uint8_t frontType = cellType[cellIndex(i, j, k, nx, ny)];

    const int faceIdx = wIndex(i, j, k, nx, ny);

    if (backType == SOLID_GPU || frontType == SOLID_GPU) {
        w[faceIdx] = 0.0f;
        return;
    }

    if (backType == AIR_GPU && frontType == AIR_GPU) {
        return;
    }

    const float pB = (backType == WATER_GPU) ? pressure[cellIndex(i, j, k - 1, nx, ny)] : 0.0f;
    const float pF = (frontType == WATER_GPU) ? pressure[cellIndex(i, j, k, nx, ny)] : 0.0f;
    const float safeDt = fmaxf(dt, 1e-6f);

    w[faceIdx] -= (safeDt / density) * (pF - pB) / h;
}

__global__ void gridToParticlesKernel(
    DeviceGrid currentGrid,
    DeviceGrid oldGrid,
    const float4* pos,
    float4* vel,
    int particleCount,
    float flipRatio)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particleCount) {
        return;
    }

    const float3 x = make_float3(pos[p].x, pos[p].y, pos[p].z);
    const float3 currentVel = sampleMAC(currentGrid, x);
    const float3 oldVel = sampleMAC(oldGrid, x);
    const float3 particleVel = make_float3(vel[p].x, vel[p].y, vel[p].z);

    const float3 flipDelta = sub3(currentVel, oldVel);
    const float3 flipVel = add3(particleVel, flipDelta);
    const float3 blended = add3(scale3(flipVel, flipRatio), scale3(currentVel, 1.0f - flipRatio));

    vel[p] = make_float4(blended.x, blended.y, blended.z, 0.0f);
}

__global__ void advectParticlesKernel(DeviceGrid g, float4* pos, int particleCount, float dt) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particleCount) {
        return;
    }

    const float h = g.h;
    const float eps = 0.05f * h;
    const float maxX = g.nx * h;
    const float maxY = g.ny * h;
    const float maxZ = g.nz * h;

    float3 x = make_float3(pos[p].x, pos[p].y, pos[p].z);
    float t = 0.0f;

    while (t < dt) {
        const float3 v0 = sampleMAC(g, x);
        const float speed = length3(v0);
        const float subDt = (speed > 1e-6f)
            ? fminf(dt - t, 0.9f * h / speed)
            : (dt - t);

        const float3 midPos = add3(x, scale3(v0, 0.5f * subDt));
        const float3 vMid = sampleMAC(g, midPos);
        x = add3(x, scale3(vMid, subDt));
        t += subDt;

        x.x = clampf(x.x, eps, maxX - eps);
        x.y = clampf(x.y, eps, maxY - eps);
        x.z = clampf(x.z, eps, maxZ - eps);
    }

    pos[p] = make_float4(x.x, x.y, x.z, 0.0f);
}

__global__ void applyParticleBoundaryKernel(
    float4* pos,
    float4* vel,
    int particleCount,
    int nx,
    int ny,
    int nz,
    float h)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particleCount) {
        return;
    }

    const float eps = 0.05f * h;
    const float maxX = nx * h;
    const float maxY = ny * h;
    const float maxZ = nz * h;

    float4 x = pos[p];
    float4 v = vel[p];

    if (x.x < eps) {
        x.x = eps;
        if (v.x < 0.0f) v.x = 0.0f;
        v.y = 0.0f;
        v.z = 0.0f;
    } else if (x.x > maxX - eps) {
        x.x = maxX - eps;
        if (v.x > 0.0f) v.x = 0.0f;
        v.y = 0.0f;
        v.z = 0.0f;
    }

    if (x.y < eps) {
        x.y = eps;
        if (v.y < 0.0f) v.y = 0.0f;
        v.x = 0.0f;
        v.z = 0.0f;
    } else if (x.y > maxY - eps) {
        x.y = maxY - eps;
        if (v.y > 0.0f) v.y = 0.0f;
        v.x = 0.0f;
        v.z = 0.0f;
    }

    if (x.z < eps) {
        x.z = eps;
        if (v.z < 0.0f) v.z = 0.0f;
        v.x = 0.0f;
        v.y = 0.0f;
    } else if (x.z > maxZ - eps) {
        x.z = maxZ - eps;
        if (v.z > 0.0f) v.z = 0.0f;
        v.x = 0.0f;
        v.y = 0.0f;
    }

    pos[p] = x;
    vel[p] = v;
}

inline int divUp(int n, int d) {
    return (n + d - 1) / d;
}

}

struct FLIPSolver::Impl {
    int nx;
    int ny;
    int nz;
    float h;

    int cellCount;
    int uCount;
    int vCount;
    int wCount;

    float* d_u = nullptr;
    float* d_v = nullptr;
    float* d_w = nullptr;

    float* d_uPrev = nullptr;
    float* d_vPrev = nullptr;
    float* d_wPrev = nullptr;

    float* d_weightU = nullptr;
    float* d_weightV = nullptr;
    float* d_weightW = nullptr;

    float* d_pOld = nullptr;
    float* d_pNew = nullptr;

    std::uint8_t* d_cellType = nullptr;

    float4* d_pos = nullptr;
    float4* d_vel = nullptr;
    int particleCapacity = 0;

    explicit Impl(int nx_, int ny_, int nz_, float h_)
        : nx(nx_),
          ny(ny_),
          nz(nz_),
          h(h_),
          cellCount(nx * ny * nz),
          uCount((nx + 1) * ny * nz),
          vCount(nx * (ny + 1) * nz),
          wCount(nx * ny * (nz + 1))
    {
        cudaMalloc(&d_u,       sizeof(float) * uCount);
        cudaMalloc(&d_v,       sizeof(float) * vCount);
        cudaMalloc(&d_w,       sizeof(float) * wCount);

        cudaMalloc(&d_uPrev,   sizeof(float) * uCount);
        cudaMalloc(&d_vPrev,   sizeof(float) * vCount);
        cudaMalloc(&d_wPrev,   sizeof(float) * wCount);

        cudaMalloc(&d_weightU, sizeof(float) * uCount);
        cudaMalloc(&d_weightV, sizeof(float) * vCount);
        cudaMalloc(&d_weightW, sizeof(float) * wCount);

        cudaMalloc(&d_pOld,    sizeof(float) * cellCount);
        cudaMalloc(&d_pNew,    sizeof(float) * cellCount);
        cudaMalloc(&d_cellType, sizeof(std::uint8_t) * cellCount);
    }

    ~Impl() {
        cudaFree(d_u);
        cudaFree(d_v);
        cudaFree(d_w);
        cudaFree(d_uPrev);
        cudaFree(d_vPrev);
        cudaFree(d_wPrev);
        cudaFree(d_weightU);
        cudaFree(d_weightV);
        cudaFree(d_weightW);
        cudaFree(d_pOld);
        cudaFree(d_pNew);
        cudaFree(d_cellType);
        cudaFree(d_pos);
        cudaFree(d_vel);
    }

    void ensureParticleCapacity(int count) {
        if (count <= particleCapacity) {
            return;
        }

        cudaFree(d_pos);
        cudaFree(d_vel);
        d_pos = nullptr;
        d_vel = nullptr;

        particleCapacity = std::max(count, std::max(1024, particleCapacity * 2));
        cudaMalloc(&d_pos, sizeof(float4) * particleCapacity);
        cudaMalloc(&d_vel, sizeof(float4) * particleCapacity);
    }

    DeviceGrid currentGrid() const {
        DeviceGrid g{};
        g.nx = nx;
        g.ny = ny;
        g.nz = nz;
        g.h = h;
        g.u = d_u;
        g.v = d_v;
        g.w = d_w;
        g.weightU = d_weightU;
        g.weightV = d_weightV;
        g.weightW = d_weightW;
        return g;
    }

    DeviceGrid previousGrid() const {
        DeviceGrid g{};
        g.nx = nx;
        g.ny = ny;
        g.nz = nz;
        g.h = h;
        g.u = d_uPrev;
        g.v = d_vPrev;
        g.w = d_wPrev;
        g.weightU = nullptr;
        g.weightV = nullptr;
        g.weightW = nullptr;
        return g;
    }
};

FLIPSolver::FLIPSolver(int nx_, int ny_, int nz_, float h_)
    : nx(nx_),
      ny(ny_),
      nz(nz_),
      h(h_),
      materialDensity(1.0f),
      flipRatio(0.95f),
      pressureIterations(100),
      impl(std::make_unique<Impl>(nx_, ny_, nz_, h_))
{
}

FLIPSolver::~FLIPSolver() = default;

void FLIPSolver::addParticles(const std::vector<Particle>& newParticles) {
    particles.insert(particles.end(), newParticles.begin(), newParticles.end());
}

void FLIPSolver::clearParticles() {
    particles.clear();
}

void FLIPSolver::setMaterialDensity(float density) {
    materialDensity = std::max(density, 1e-6f);
}

void FLIPSolver::setFlipRatio(float ratio) {
    flipRatio = std::max(0.0f, std::min(1.0f, ratio));
}

void FLIPSolver::setPressureIterations(int iterations) {
    pressureIterations = std::max(iterations, 1);
}

void FLIPSolver::step(float dt) {
    if (particles.empty() || dt <= 0.0f) {
        return;
    }

    impl->ensureParticleCapacity(static_cast<int>(particles.size()));

    std::vector<float4> hostPos(particles.size());
    std::vector<float4> hostVel(particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i) {
        hostPos[i] = make_float4(particles[i].pos.x, particles[i].pos.y, particles[i].pos.z, 0.0f);
        hostVel[i] = make_float4(particles[i].vel.x, particles[i].vel.y, particles[i].vel.z, 0.0f);
    }

    cudaMemcpy(
        impl->d_pos,
        hostPos.data(),
        sizeof(float4) * particles.size(),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        impl->d_vel,
        hostVel.data(),
        sizeof(float4) * particles.size(),
        cudaMemcpyHostToDevice);

    const float maxSubstep = 0.04f;
    const int substeps = std::max(1, static_cast<int>(std::ceil(dt / maxSubstep)));
    const float subDt = dt / static_cast<float>(substeps);

    constexpr int threads = 256;
    const int particleBlocks = divUp(static_cast<int>(particles.size()), threads);

    for (int s = 0; s < substeps; ++s) {
        cudaMemset(impl->d_cellType, 0, sizeof(std::uint8_t) * impl->cellCount);
        cudaMemset(impl->d_u, 0, sizeof(float) * impl->uCount);
        cudaMemset(impl->d_v, 0, sizeof(float) * impl->vCount);
        cudaMemset(impl->d_w, 0, sizeof(float) * impl->wCount);
        cudaMemset(impl->d_weightU, 0, sizeof(float) * impl->uCount);
        cudaMemset(impl->d_weightV, 0, sizeof(float) * impl->vCount);
        cudaMemset(impl->d_weightW, 0, sizeof(float) * impl->wCount);

        markFluidCellsKernel<<<particleBlocks, threads>>>(
            impl->d_pos,
            static_cast<int>(particles.size()),
            impl->d_cellType,
            nx,
            ny,
            nz,
            h);

        DeviceGrid grid = impl->currentGrid();
        particlesToGridKernel<<<particleBlocks, threads>>>(
            grid,
            impl->d_pos,
            impl->d_vel,
            static_cast<int>(particles.size()));

        normalizeUKernel<<<divUp(impl->uCount, threads), threads>>>(grid);
        normalizeVKernel<<<divUp(impl->vCount, threads), threads>>>(grid);
        normalizeWKernel<<<divUp(impl->wCount, threads), threads>>>(grid);


        enforceUBoundaryKernel<<<divUp(impl->uCount, threads), threads>>>(impl->d_u, nx, ny, nz);
        enforceVBoundaryKernel<<<divUp(impl->vCount, threads), threads>>>(impl->d_v, nx, ny, nz);
        enforceWBoundaryKernel<<<divUp(impl->wCount, threads), threads>>>(impl->d_w, nx, ny, nz);


        cudaMemcpy(impl->d_uPrev, impl->d_u, sizeof(float) * impl->uCount, cudaMemcpyDeviceToDevice);
        cudaMemcpy(impl->d_vPrev, impl->d_v, sizeof(float) * impl->vCount, cudaMemcpyDeviceToDevice);
        cudaMemcpy(impl->d_wPrev, impl->d_w, sizeof(float) * impl->wCount, cudaMemcpyDeviceToDevice);

        addGravityKernel<<<divUp(impl->vCount, threads), threads>>>(impl->d_v, nx, ny, nz, subDt);


        enforceUBoundaryKernel<<<divUp(impl->uCount, threads), threads>>>(impl->d_u, nx, ny, nz);
        enforceVBoundaryKernel<<<divUp(impl->vCount, threads), threads>>>(impl->d_v, nx, ny, nz);
        enforceWBoundaryKernel<<<divUp(impl->wCount, threads), threads>>>(impl->d_w, nx, ny, nz);


        cudaMemset(impl->d_pOld, 0, sizeof(float) * impl->cellCount);
        cudaMemset(impl->d_pNew, 0, sizeof(float) * impl->cellCount);

        for (int iter = 0; iter < pressureIterations; ++iter) {
            jacobiPressureKernel<<<divUp(impl->cellCount, threads), threads>>>(
                grid,
                impl->d_cellType,
                impl->d_pOld,
                impl->d_pNew,
                materialDensity,
                subDt);
    
            std::swap(impl->d_pOld, impl->d_pNew);
        }

        if (nx > 1) {
            applyPressureUKernel<<<divUp((nx - 1) * ny * nz, threads), threads>>>(
                impl->d_u,
                nx,
                ny,
                nz,
                h,
                materialDensity,
                subDt,
                impl->d_cellType,
                impl->d_pOld);
    
        }
        if (ny > 1) {
            applyPressureVKernel<<<divUp(nx * (ny - 1) * nz, threads), threads>>>(
                impl->d_v,
                nx,
                ny,
                nz,
                h,
                materialDensity,
                subDt,
                impl->d_cellType,
                impl->d_pOld);
    
        }
        if (nz > 1) {
            applyPressureWKernel<<<divUp(nx * ny * (nz - 1), threads), threads>>>(
                impl->d_w,
                nx,
                ny,
                nz,
                h,
                materialDensity,
                subDt,
                impl->d_cellType,
                impl->d_pOld);
    
        }

        enforceUBoundaryKernel<<<divUp(impl->uCount, threads), threads>>>(impl->d_u, nx, ny, nz);
        enforceVBoundaryKernel<<<divUp(impl->vCount, threads), threads>>>(impl->d_v, nx, ny, nz);
        enforceWBoundaryKernel<<<divUp(impl->wCount, threads), threads>>>(impl->d_w, nx, ny, nz);


        gridToParticlesKernel<<<particleBlocks, threads>>>(
            impl->currentGrid(),
            impl->previousGrid(),
            impl->d_pos,
            impl->d_vel,
            static_cast<int>(particles.size()),
            flipRatio);


        advectParticlesKernel<<<particleBlocks, threads>>>(
            impl->currentGrid(),
            impl->d_pos,
            static_cast<int>(particles.size()),
            subDt);


        applyParticleBoundaryKernel<<<particleBlocks, threads>>>(
            impl->d_pos,
            impl->d_vel,
            static_cast<int>(particles.size()),
            nx,
            ny,
            nz,
            h);


        cudaDeviceSynchronize();
    }

    cudaMemcpy(
        hostPos.data(),
        impl->d_pos,
        sizeof(float4) * particles.size(),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        hostVel.data(),
        impl->d_vel,
        sizeof(float4) * particles.size(),
        cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].pos = Vec3(hostPos[i].x, hostPos[i].y, hostPos[i].z);
        particles[i].vel = Vec3(hostVel[i].x, hostVel[i].y, hostVel[i].z);
    }
}
