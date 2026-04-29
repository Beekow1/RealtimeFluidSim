#include "FLIPSolver.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

    enum CellTypeGPU : std::uint8_t {
        AIR_GPU = 0,
        WATER_GPU = 1,
        SOLID_GPU = 2,
    };

#define CUDA_CHECK(expr) \
    do {\
        cudaError_t _err = (expr);\
        if (_err != cudaSuccess) {\
            throw std::runtime_error(cudaGetErrorString(_err));\
        }\
    } while (0)

    struct DeviceGrid {
        int nx;
        int ny;
        int nz;
        float h;

        float* __restrict__ u;
        float* __restrict__ v;
        float* __restrict__ w;

        float* __restrict__ weightU;
        float* __restrict__ weightV;
        float* __restrict__ weightW;
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
        if (component == 0) return g.u[uIndex(i, j, k, g.nx, g.ny)];
        if (component == 1) return g.v[vIndex(i, j, k, g.nx, g.ny)];
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

#define ACCUM_SAMPLE(di, dj, dk)\
    do {\
        const int ii = i + (di);\
        const int jj = j + (dj); \
        const int kk = k + (dk); \
        if (ii >= 0 && ii <= maxI && jj >= 0 && jj <= maxJ && kk >= 0 && kk <= maxK) { \
            value += trilinearWeight(fx, fy, fz, di, dj, dk) * componentAt(g, component, ii, jj, kk);\
        } \
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

        const float u = sampleComponent(g, xc, make_float3(0.0f, 0.5f, 0.5f), g.nx, g.ny - 1, g.nz - 1, 0);
        const float v = sampleComponent(g, xc, make_float3(0.5f, 0.0f, 0.5f), g.nx - 1, g.ny, g.nz - 1, 1);
        const float w = sampleComponent(g, xc, make_float3(0.5f, 0.5f, 0.0f), g.nx - 1, g.ny - 1, g.nz, 2);

        return make_float3(u, v, w);
    }

    __device__ float divergenceAt(const DeviceGrid& g, int i, int j, int k) {
        const float du = g.u[uIndex(i + 1, j, k, g.nx, g.ny)] - g.u[uIndex(i, j, k, g.nx, g.ny)];
        const float dv = g.v[vIndex(i, j + 1, k, g.nx, g.ny)] - g.v[vIndex(i, j, k, g.nx, g.ny)];
        const float dw = g.w[wIndex(i, j, k + 1, g.nx, g.ny)] - g.w[wIndex(i, j, k, g.nx, g.ny)];
        return (du + dv + dw) / g.h;
    }

    __global__ void markFluidCellsKernel(
        const float4* __restrict__ pos,
        int particleCount,
        std::uint8_t* __restrict__ cellType,
        int nx,
        int ny,
        int nz,
        float h)
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particleCount) return;

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

    __global__ void particlesToGridKernel(
        DeviceGrid g,
        const float4* __restrict__ pos,
        const float4* __restrict__ vel,
        int particleCount)
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particleCount) return;

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

#pragma unroll
        for (int di = 0; di <= 1; ++di) {
#pragma unroll
            for (int dj = 0; dj <= 1; ++dj) {
#pragma unroll
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


    __global__ void finalizeCopyUKernel(
        float* __restrict__ u,
        float* __restrict__ uPrev,
        const float* __restrict__ weightU,
        int nx,
        int ny,
        int nz)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int count = (nx + 1) * ny * nz;
        if (idx >= count) return;

        float value = u[idx];
        const float weight = weightU[idx];
        if (weight > 0.0f) value /= weight;

        const int i = idx % (nx + 1);
        if (i == 0 || i == nx) value = 0.0f;

        uPrev[idx] = value;
        u[idx] = value;
    }

    __global__ void finalizeCopyVApplyGravityKernel(
        float* __restrict__ v,
        float* __restrict__ vPrev,
        const float* __restrict__ weightV,
        int nx,
        int ny,
        int nz,
        float dt)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int count = nx * (ny + 1) * nz;
        if (idx >= count) return;

        float value = v[idx];
        const float weight = weightV[idx];
        if (weight > 0.0f) value /= weight;

        const int j = (idx / nx) % (ny + 1);
        const bool boundary = (j == 0 || j == ny);
        if (boundary) value = 0.0f;
        vPrev[idx] = value;

        if (!boundary) {
            value += -9.81f * dt;
        }

        v[idx] = value;
    }

    __global__ void finalizeCopyWKernel(
        float* __restrict__ w,
        float* __restrict__ wPrev,
        const float* __restrict__ weightW,
        int nx,
        int ny,
        int nz)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int count = nx * ny * (nz + 1);
        if (idx >= count) return;

        float value = w[idx];
        const float weight = weightW[idx];
        if (weight > 0.0f) value /= weight;

        const int k = idx / (nx * ny);
        if (k == 0 || k == nz) value = 0.0f;

        wPrev[idx] = value;
        w[idx] = value;
    }


    __global__ void buildPressureSystemKernel(
        DeviceGrid g,
        const std::uint8_t* __restrict__ cellType,
        float* __restrict__ rhs,
        float* __restrict__ diagInv,
        float density,
        float dt)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int nCells = g.nx * g.ny * g.nz;
        if (idx >= nCells) return;

        if (cellType[idx] != WATER_GPU) {
            rhs[idx] = 0.0f;
            diagInv[idx] = 0.0f;
            return;
        }

        const int i = idx % g.nx;
        const int j = (idx / g.nx) % g.ny;
        const int k = idx / (g.nx * g.ny);

        const float safeDt = fmaxf(dt, 1e-6f);
        rhs[idx] = -(density / safeDt) * divergenceAt(g, i, j, k) * g.h * g.h;

        int diag = 0;

        if (i - 1 >= 0) {
            const std::uint8_t t = cellType[cellIndex(i - 1, j, k, g.nx, g.ny)];
            if (t != SOLID_GPU) ++diag;
        }
        if (i + 1 < g.nx) {
            const std::uint8_t t = cellType[cellIndex(i + 1, j, k, g.nx, g.ny)];
            if (t != SOLID_GPU) ++diag;
        }
        if (j - 1 >= 0) {
            const std::uint8_t t = cellType[cellIndex(i, j - 1, k, g.nx, g.ny)];
            if (t != SOLID_GPU) ++diag;
        }
        if (j + 1 < g.ny) {
            const std::uint8_t t = cellType[cellIndex(i, j + 1, k, g.nx, g.ny)];
            if (t != SOLID_GPU) ++diag;
        }
        if (k - 1 >= 0) {
            const std::uint8_t t = cellType[cellIndex(i, j, k - 1, g.nx, g.ny)];
            if (t != SOLID_GPU) ++diag;
        }
        if (k + 1 < g.nz) {
            const std::uint8_t t = cellType[cellIndex(i, j, k + 1, g.nx, g.ny)];
            if (t != SOLID_GPU) ++diag;
        }

        diagInv[idx] = (diag > 0) ? (1.0f / static_cast<float>(diag)) : 0.0f;
    }

    __global__ void updateSearchDirectionKernel(
        const std::uint8_t* __restrict__ cellType,
        float* __restrict__ search,
        const float* __restrict__ z,
        float beta,
        int nCells)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nCells || cellType[idx] != WATER_GPU) return;

        search[idx] = z[idx] + beta * search[idx];
    }

    __global__ void reducePartialSumsKernel(
        float* __restrict__ data,
        int count)
    {
        __shared__ float sdata[256];

        const int tid = threadIdx.x;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        float val = (idx < count) ? data[idx] : 0.0f;
        sdata[tid] = val;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            data[blockIdx.x] = sdata[0];
        }
    }


    __global__ void initializePCGAndDotKernel(
        const std::uint8_t* __restrict__ cellType,
        const float* __restrict__ rhs,
        const float* __restrict__ diagInv,
        float* __restrict__ pressure,
        float* __restrict__ residual,
        float* __restrict__ z,
        float* __restrict__ search,
        float* __restrict__ partial,
        int nCells)
    {
        __shared__ float sdata[256];

        const int tid = threadIdx.x;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        float dotVal = 0.0f;
        if (idx < nCells) {
            if (cellType[idx] == WATER_GPU) {
                const float r = rhs[idx];
                const float zi = diagInv[idx] * r;
                pressure[idx] = 0.0f;
                residual[idx] = r;
                z[idx] = zi;
                search[idx] = zi;
                dotVal = r * zi;
            }
            else {
                pressure[idx] = 0.0f;
                residual[idx] = 0.0f;
                z[idx] = 0.0f;
                search[idx] = 0.0f;
            }
        }

        sdata[tid] = dotVal;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial[blockIdx.x] = sdata[0];
        }
    }

    __global__ void applyPressureMatrixAndDotKernel(
        int nx,
        int ny,
        int nz,
        const std::uint8_t* __restrict__ cellType,
        const float* __restrict__ x,
        float* __restrict__ y,
        float* __restrict__ partial)
    {
        __shared__ float sdata[256];

        const int tid = threadIdx.x;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int nCells = nx * ny * nz;

        float dotVal = 0.0f;

        if (idx < nCells) {
            if (cellType[idx] != WATER_GPU) {
                y[idx] = 0.0f;
            }
            else {
                const int i = idx % nx;
                const int j = (idx / nx) % ny;
                const int k = idx / (nx * ny);

                float result = 0.0f;
                int diag = 0;

                if (i - 1 >= 0) {
                    const int nidx = cellIndex(i - 1, j, k, nx, ny);
                    const std::uint8_t t = cellType[nidx];
                    if (t != SOLID_GPU) {
                        ++diag;
                        if (t == WATER_GPU) result -= x[nidx];
                    }
                }
                if (i + 1 < nx) {
                    const int nidx = cellIndex(i + 1, j, k, nx, ny);
                    const std::uint8_t t = cellType[nidx];
                    if (t != SOLID_GPU) {
                        ++diag;
                        if (t == WATER_GPU) result -= x[nidx];
                    }
                }
                if (j - 1 >= 0) {
                    const int nidx = cellIndex(i, j - 1, k, nx, ny);
                    const std::uint8_t t = cellType[nidx];
                    if (t != SOLID_GPU) {
                        ++diag;
                        if (t == WATER_GPU) result -= x[nidx];
                    }
                }
                if (j + 1 < ny) {
                    const int nidx = cellIndex(i, j + 1, k, nx, ny);
                    const std::uint8_t t = cellType[nidx];
                    if (t != SOLID_GPU) {
                        ++diag;
                        if (t == WATER_GPU) result -= x[nidx];
                    }
                }
                if (k - 1 >= 0) {
                    const int nidx = cellIndex(i, j, k - 1, nx, ny);
                    const std::uint8_t t = cellType[nidx];
                    if (t != SOLID_GPU) {
                        ++diag;
                        if (t == WATER_GPU) result -= x[nidx];
                    }
                }
                if (k + 1 < nz) {
                    const int nidx = cellIndex(i, j, k + 1, nx, ny);
                    const std::uint8_t t = cellType[nidx];
                    if (t != SOLID_GPU) {
                        ++diag;
                        if (t == WATER_GPU) result -= x[nidx];
                    }
                }

                const float Ax = static_cast<float>(diag) * x[idx] + result;
                y[idx] = Ax;
                dotVal = x[idx] * Ax;
            }
        }

        sdata[tid] = dotVal;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial[blockIdx.x] = sdata[0];
        }
    }

    __global__ void updatePressureResidualZAndDotKernel(
        const std::uint8_t* __restrict__ cellType,
        float* __restrict__ pressure,
        float* __restrict__ residual,
        float* __restrict__ z,
        const float* __restrict__ search,
        const float* __restrict__ Ap,
        const float* __restrict__ diagInv,
        float alpha,
        float* __restrict__ partial,
        int nCells)
    {
        __shared__ float sdata[256];

        const int tid = threadIdx.x;
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        float dotVal = 0.0f;
        if (idx < nCells&& cellType[idx] == WATER_GPU) {
            pressure[idx] += alpha * search[idx];

            const float r = residual[idx] - alpha * Ap[idx];
            residual[idx] = r;

            const float zi = diagInv[idx] * r;
            z[idx] = zi;

            dotVal = r * zi;
        }

        sdata[tid] = dotVal;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            partial[blockIdx.x] = sdata[0];
        }
    }


    __global__ void gridToParticlesAdvectBoundaryKernel(
        DeviceGrid currentGrid,
        DeviceGrid oldGrid,
        float4* __restrict__ pos,
        float4* __restrict__ vel,
        int particleCount,
        float flipRatio,
        float dt)
    {
        const int p = blockIdx.x * blockDim.x + threadIdx.x;
        if (p >= particleCount) return;

        const float h = currentGrid.h;
        const float eps = 0.05f * h;
        const float maxX = currentGrid.nx * h;
        const float maxY = currentGrid.ny * h;
        const float maxZ = currentGrid.nz * h;

        float3 x = make_float3(pos[p].x, pos[p].y, pos[p].z);
        const float3 particleVel = make_float3(vel[p].x, vel[p].y, vel[p].z);

        const float3 currentVel = sampleMAC(currentGrid, x);
        const float3 oldVel = sampleMAC(oldGrid, x);
        const float3 flipDelta = sub3(currentVel, oldVel);
        const float3 flipVel = add3(particleVel, flipDelta);
        float3 newVel = add3(
            scale3(flipVel, flipRatio),
            scale3(currentVel, 1.0f - flipRatio));

        float t = 0.0f;
        while (t < dt) {
            const float3 v0 = sampleMAC(currentGrid, x);
            const float speed = length3(v0);
            const float subDt =
                (speed > 1e-6f)
                ? fminf(dt - t, 0.9f * h / speed)
                : (dt - t);

            const float3 midPos = add3(x, scale3(v0, 0.5f * subDt));
            const float3 vMid = sampleMAC(currentGrid, midPos);
            x = add3(x, scale3(vMid, subDt));
            t += subDt;

            x.x = clampf(x.x, eps, maxX - eps);
            x.y = clampf(x.y, eps, maxY - eps);
            x.z = clampf(x.z, eps, maxZ - eps);
        }

        if (x.x < eps) {
            x.x = eps;
            if (newVel.x < 0.0f) newVel.x = 0.0f;
            newVel.y = 0.0f;
            newVel.z = 0.0f;
        }
        else if (x.x > maxX - eps) {
            x.x = maxX - eps;
            if (newVel.x > 0.0f) newVel.x = 0.0f;
            newVel.y = 0.0f;
            newVel.z = 0.0f;
        }

        if (x.y < eps) {
            x.y = eps;
            if (newVel.y < 0.0f) newVel.y = 0.0f;
            newVel.x = 0.0f;
            newVel.z = 0.0f;
        }
        else if (x.y > maxY - eps) {
            x.y = maxY - eps;
            if (newVel.y > 0.0f) newVel.y = 0.0f;
            newVel.x = 0.0f;
            newVel.z = 0.0f;
        }

        if (x.z < eps) {
            x.z = eps;
            if (newVel.z < 0.0f) newVel.z = 0.0f;
            newVel.x = 0.0f;
            newVel.y = 0.0f;
        }
        else if (x.z > maxZ - eps) {
            x.z = maxZ - eps;
            if (newVel.z > 0.0f) newVel.z = 0.0f;
            newVel.x = 0.0f;
            newVel.y = 0.0f;
        }

        pos[p] = make_float4(x.x, x.y, x.z, 0.0f);
        vel[p] = make_float4(newVel.x, newVel.y, newVel.z, 0.0f);
    }

    inline int divUp(int n, int d) {
        return (n + d - 1) / d;
    }
    __global__ void applyPressureUKernel(
        float* __restrict__ u,
        int nx,
        int ny,
        int nz,
        float h,
        float density,
        float dt,
        const std::uint8_t* __restrict__ cellType,
        const float* __restrict__ pressure)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int count = (nx - 1) * ny * nz;
        if (idx >= count) return;

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
        float* __restrict__ v,
        int nx,
        int ny,
        int nz,
        float h,
        float density,
        float dt,
        const std::uint8_t* __restrict__ cellType,
        const float* __restrict__ pressure)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int count = nx * (ny - 1) * nz;
        if (idx >= count) return;

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
        float* __restrict__ w,
        int nx,
        int ny,
        int nz,
        float h,
        float density,
        float dt,
        const std::uint8_t* __restrict__ cellType,
        const float* __restrict__ pressure)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int count = nx * ny * (nz - 1);
        if (idx >= count) return;

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

    std::uint8_t* d_cellType = nullptr;

    float4* d_pos = nullptr;
    float4* d_vel = nullptr;
    int particleCapacity = 0;
    int deviceParticleCount = 0;

    float4* h_posPinned = nullptr;
    float4* h_velPinned = nullptr;
    int hostParticleCapacity = 0;

    float* d_rhs = nullptr;
    float* d_residual = nullptr;
    float* d_z = nullptr;
    float* d_search = nullptr;
    float* d_Ap = nullptr;
    float* d_diagInv = nullptr;

    float* d_reduceBuffer = nullptr;
    int reduceBufferCount = 0;

    cudaStream_t stream = nullptr;

    explicit Impl(int nx_, int ny_, int nz_, float h_)
        : nx(nx_),
        ny(ny_),
        nz(nz_),
        h(h_),
        cellCount(nx* ny* nz),
        uCount((nx + 1)* ny* nz),
        vCount(nx* (ny + 1)* nz),
        wCount(nx* ny* (nz + 1))
    {
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMalloc(&d_u, sizeof(float) * uCount));
        CUDA_CHECK(cudaMalloc(&d_v, sizeof(float) * vCount));
        CUDA_CHECK(cudaMalloc(&d_w, sizeof(float) * wCount));

        CUDA_CHECK(cudaMalloc(&d_uPrev, sizeof(float) * uCount));
        CUDA_CHECK(cudaMalloc(&d_vPrev, sizeof(float) * vCount));
        CUDA_CHECK(cudaMalloc(&d_wPrev, sizeof(float) * wCount));

        CUDA_CHECK(cudaMalloc(&d_weightU, sizeof(float) * uCount));
        CUDA_CHECK(cudaMalloc(&d_weightV, sizeof(float) * vCount));
        CUDA_CHECK(cudaMalloc(&d_weightW, sizeof(float) * wCount));

        CUDA_CHECK(cudaMalloc(&d_pOld, sizeof(float) * cellCount));
        CUDA_CHECK(cudaMalloc(&d_cellType, sizeof(std::uint8_t) * cellCount));

        CUDA_CHECK(cudaMalloc(&d_rhs, sizeof(float) * cellCount));
        CUDA_CHECK(cudaMalloc(&d_residual, sizeof(float) * cellCount));
        CUDA_CHECK(cudaMalloc(&d_z, sizeof(float) * cellCount));
        CUDA_CHECK(cudaMalloc(&d_search, sizeof(float) * cellCount));
        CUDA_CHECK(cudaMalloc(&d_Ap, sizeof(float) * cellCount));
        CUDA_CHECK(cudaMalloc(&d_diagInv, sizeof(float) * cellCount));

        reduceBufferCount = divUp(cellCount, 256);
        CUDA_CHECK(cudaMalloc(&d_reduceBuffer, sizeof(float) * reduceBufferCount));
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
        cudaFree(d_cellType);
        cudaFree(d_pos);
        cudaFree(d_vel);

        cudaFree(d_rhs);
        cudaFree(d_residual);
        cudaFree(d_z);
        cudaFree(d_search);
        cudaFree(d_Ap);
        cudaFree(d_diagInv);
        cudaFree(d_reduceBuffer);

        if (h_posPinned) cudaFreeHost(h_posPinned);
        if (h_velPinned) cudaFreeHost(h_velPinned);

        if (stream) cudaStreamDestroy(stream);
    }

    void ensureParticleCapacity(int count) {
        if (count <= particleCapacity) return;

        cudaFree(d_pos);
        cudaFree(d_vel);
        d_pos = nullptr;
        d_vel = nullptr;

        particleCapacity = std::max(count, std::max(1024, particleCapacity * 2));
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float4) * particleCapacity));
        CUDA_CHECK(cudaMalloc(&d_vel, sizeof(float4) * particleCapacity));
    }


    void ensurePinnedHostCapacity(int count) {
        if (count <= hostParticleCapacity) return;

        if (h_posPinned) cudaFreeHost(h_posPinned);
        if (h_velPinned) cudaFreeHost(h_velPinned);
        h_posPinned = nullptr;
        h_velPinned = nullptr;

        hostParticleCapacity = std::max(count, std::max(1024, hostParticleCapacity * 2));
        CUDA_CHECK(cudaMallocHost(&h_posPinned, sizeof(float4) * hostParticleCapacity));
        CUDA_CHECK(cudaMallocHost(&h_velPinned, sizeof(float4) * hostParticleCapacity));
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

    void uploadParticlesIfNeeded(const std::vector<Particle>& particles, bool force) {
        if (!force && static_cast<int>(particles.size()) == deviceParticleCount) {
            return;
        }

        const int count = static_cast<int>(particles.size());
        ensureParticleCapacity(count);
        ensurePinnedHostCapacity(count);

        for (std::size_t i = 0; i < particles.size(); ++i) {
            h_posPinned[i] = make_float4(particles[i].pos.x, particles[i].pos.y, particles[i].pos.z, 0.0f);
            h_velPinned[i] = make_float4(particles[i].vel.x, particles[i].vel.y, particles[i].vel.z, 0.0f);
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_pos,
            h_posPinned,
            sizeof(float4) * particles.size(),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_vel,
            h_velPinned,
            sizeof(float4) * particles.size(),
            cudaMemcpyHostToDevice,
            stream));

        deviceParticleCount = count;
    }

    void downloadParticles(std::vector<Particle>& particles) {
        const int count = static_cast<int>(particles.size());
        ensurePinnedHostCapacity(count);

        CUDA_CHECK(cudaMemcpyAsync(
            h_posPinned,
            d_pos,
            sizeof(float4) * particles.size(),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            h_velPinned,
            d_vel,
            sizeof(float4) * particles.size(),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (std::size_t i = 0; i < particles.size(); ++i) {
            particles[i].pos = Vec3(h_posPinned[i].x, h_posPinned[i].y, h_posPinned[i].z);
            particles[i].vel = Vec3(h_velPinned[i].x, h_velPinned[i].y, h_velPinned[i].z);
        }
    }


    float finishReductionFromPartials(int count) const {
        constexpr int threads = 256;

        while (count > 1) {
            const int reduceBlocks = divUp(count, threads);
            reducePartialSumsKernel << <reduceBlocks, threads, 0, stream >> > (
                d_reduceBuffer, count);
            count = reduceBlocks;
        }

        float result = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(
            &result,
            d_reduceBuffer,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }

    void solvePressurePCG(float density, float dt, int maxIterations, float tolerance) {
        constexpr int threads = 256;
        const int blocks = divUp(cellCount, threads);

        DeviceGrid g = currentGrid();

        buildPressureSystemKernel << <blocks, threads, 0, stream >> > (
            g,
            d_cellType,
            d_rhs,
            d_diagInv,
            density,
            dt);

        initializePCGAndDotKernel << <blocks, threads, 0, stream >> > (
            d_cellType,
            d_rhs,
            d_diagInv,
            d_pOld,
            d_residual,
            d_z,
            d_search,
            d_reduceBuffer,
            cellCount);

        float rzOld = finishReductionFromPartials(blocks);
        if (rzOld <= 1e-20f) {
            CUDA_CHECK(cudaMemsetAsync(d_pOld, 0, sizeof(float) * cellCount, stream));
            return;
        }

        const float tol2 = tolerance * tolerance;

        for (int iter = 0; iter < maxIterations; ++iter) {
            applyPressureMatrixAndDotKernel << <blocks, threads, 0, stream >> > (
                nx,
                ny,
                nz,
                d_cellType,
                d_search,
                d_Ap,
                d_reduceBuffer);

            const float pAp = finishReductionFromPartials(blocks);
            if (std::fabs(pAp) <= 1e-20f) {
                break;
            }

            const float alpha = rzOld / pAp;

            updatePressureResidualZAndDotKernel << <blocks, threads, 0, stream >> > (
                d_cellType,
                d_pOld,
                d_residual,
                d_z,
                d_search,
                d_Ap,
                d_diagInv,
                alpha,
                d_reduceBuffer,
                cellCount);

            const float rzNew = finishReductionFromPartials(blocks);
            if (rzNew < tol2) {
                break;
            }

            const float beta = rzNew / rzOld;

            updateSearchDirectionKernel << <blocks, threads, 0, stream >> > (
                d_cellType,
                d_search,
                d_z,
                beta,
                cellCount);

            rzOld = rzNew;
        }
    }
};

FLIPSolver::FLIPSolver(int nx_, int ny_, int nz_, float h_)
    : nx(nx_),
    ny(ny_),
    nz(nz_),
    h(h_),
    materialDensity(1.0f),
    flipRatio(0.95f),
    pressureIterations(15),
    pressureTolerance(1e-4f),
    impl(std::make_unique<Impl>(nx_, ny_, nz_, h_))
{
}

FLIPSolver::~FLIPSolver() = default;

void FLIPSolver::addParticles(const std::vector<Particle>& newParticles) {
    particles.insert(particles.end(), newParticles.begin(), newParticles.end());
}

void FLIPSolver::clearParticles() {
    particles.clear();
    impl->deviceParticleCount = 0;
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

void FLIPSolver::setPressureTolerance(float tolerance) {
    pressureTolerance = std::max(tolerance, 1e-8f);
}

void FLIPSolver::step(float dt) {
    if (particles.empty() || dt <= 0.0f) {
        return;
    }

    impl->uploadParticlesIfNeeded(
        particles,
        static_cast<int>(particles.size()) != impl->deviceParticleCount);

    const float maxSubstep = 0.04f;
    const int substeps = std::max(1, static_cast<int>(std::ceil(dt / maxSubstep)));
    const float subDt = dt / static_cast<float>(substeps);

    constexpr int threads = 256;
    const int particleCount = static_cast<int>(particles.size());
    const int particleBlocks = divUp(particleCount, threads);

    DeviceGrid grid = impl->currentGrid();
    DeviceGrid prev = impl->previousGrid();

    for (int s = 0; s < substeps; ++s) {
        CUDA_CHECK(cudaMemsetAsync(impl->d_cellType, 0, sizeof(std::uint8_t) * impl->cellCount, impl->stream));

        CUDA_CHECK(cudaMemsetAsync(impl->d_u, 0, sizeof(float) * impl->uCount, impl->stream));
        CUDA_CHECK(cudaMemsetAsync(impl->d_v, 0, sizeof(float) * impl->vCount, impl->stream));
        CUDA_CHECK(cudaMemsetAsync(impl->d_w, 0, sizeof(float) * impl->wCount, impl->stream));

        CUDA_CHECK(cudaMemsetAsync(impl->d_weightU, 0, sizeof(float) * impl->uCount, impl->stream));
        CUDA_CHECK(cudaMemsetAsync(impl->d_weightV, 0, sizeof(float) * impl->vCount, impl->stream));
        CUDA_CHECK(cudaMemsetAsync(impl->d_weightW, 0, sizeof(float) * impl->wCount, impl->stream));

        markFluidCellsKernel << <particleBlocks, threads, 0, impl->stream >> > (
            impl->d_pos,
            particleCount,
            impl->d_cellType,
            nx,
            ny,
            nz,
            h);

        particlesToGridKernel << <particleBlocks, threads, 0, impl->stream >> > (
            grid,
            impl->d_pos,
            impl->d_vel,
            particleCount);

        finalizeCopyUKernel << <divUp(impl->uCount, threads), threads, 0, impl->stream >> > (
            impl->d_u,
            impl->d_uPrev,
            impl->d_weightU,
            nx,
            ny,
            nz);
        finalizeCopyVApplyGravityKernel << <divUp(impl->vCount, threads), threads, 0, impl->stream >> > (
            impl->d_v,
            impl->d_vPrev,
            impl->d_weightV,
            nx,
            ny,
            nz,
            subDt);
        finalizeCopyWKernel << <divUp(impl->wCount, threads), threads, 0, impl->stream >> > (
            impl->d_w,
            impl->d_wPrev,
            impl->d_weightW,
            nx,
            ny,
            nz);

        impl->solvePressurePCG(materialDensity, subDt, pressureIterations, pressureTolerance);

        if (nx > 1) {
            applyPressureUKernel << <divUp((nx - 1) * ny * nz, threads), threads, 0, impl->stream >> > (
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
            applyPressureVKernel << <divUp(nx * (ny - 1) * nz, threads), threads, 0, impl->stream >> > (
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
            applyPressureWKernel << <divUp(nx * ny * (nz - 1), threads), threads, 0, impl->stream >> > (
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

        gridToParticlesAdvectBoundaryKernel << <particleBlocks, threads, 0, impl->stream >> > (
            impl->currentGrid(),
            prev,
            impl->d_pos,
            impl->d_vel,
            particleCount,
            flipRatio,
            subDt);
    }

    CUDA_CHECK(cudaGetLastError());

    impl->downloadParticles(particles);
}