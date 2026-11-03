#include "mace_polar_cuda.hpp"
#include "mace_foundation_tables.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numbers>
#include <numeric>
#include <utility>

#include <cub/cub.cuh>
#include <cublas_v2.h>

#include "nlohmann/json.hpp"

using json = nlohmann::json;
using Vec = std::vector<double>;

namespace {

// MACE-POLAR-1 tensor products; the loader rejects models with others
using Polar = Tables<1>;
constexpr Term Phi1_terms[Polar::num_terms] = {
    {0, 0, 0, 0, 1.0}, {3, 3, 0, 1, 1.0}, {6, 3, 0, 2, 1.0}, {9, 3, 0, 3, 1.0}, {2, 2, 1, 0, 1.0},
    {1, 1, 1, 1, 0.5773502691896258}, {18, 6, 1, 1, -0.40824828356277404}, {24, 6, 1, 1, -0.7071067764335216},
    {15, 6, 1, 2, 0.7071067764335216}, {12, 6, 1, 3, 0.7071067764335216}, {5, 2, 2, 0, 1.0},
    {15, 6, 2, 1, 0.7071067764335216}, {1, 1, 2, 2, 0.5773502691896258}, {18, 6, 2, 2, 0.8164965671255481},
    {21, 6, 2, 3, 0.7071067764335216}, {8, 2, 3, 0, 1.0}, {12, 6, 3, 1, 0.7071067764335216},
    {21, 6, 3, 2, 0.7071067764335216}, {1, 1, 3, 3, 0.5773502691896258}, {18, 6, 3, 3, -0.40824828356277404},
    {24, 6, 3, 3, 0.7071067764335216}, {11, 5, 4, 0, 1.0}, {10, 4, 4, 1, 0.547722553823488},
    {35, 9, 4, 1, -0.1825741827875604}, {39, 9, 4, 1, -0.7071067679409303}, {29, 9, 4, 2, 0.5773502802989676},
    {4, 4, 4, 3, 0.547722553823488}, {27, 9, 4, 3, 0.7071067679409303}, {31, 9, 4, 3, -0.1825741827875604},
    {14, 5, 5, 0, 1.0}, {7, 4, 5, 1, 0.547722553823488}, {33, 9, 5, 1, -0.4472135802063231},
    {37, 9, 5, 1, -0.5773502802989676}, {4, 4, 5, 2, 0.547722553823488}, {31, 9, 5, 2, 0.7302967311502416},
    {29, 9, 5, 3, 0.5773502802989676}, {17, 5, 6, 0, 1.0}, {4, 4, 6, 1, -0.3162277606712774},
    {31, 9, 6, 1, 0.6324555383989546}, {7, 4, 6, 2, 0.6324555213425548}, {33, 9, 6, 2, 0.7745966857238762},
    {10, 4, 6, 3, -0.3162277606712774}, {35, 9, 6, 3, 0.6324555383989546}, {20, 5, 7, 0, 1.0},
    {29, 9, 7, 1, 0.5773502802989676}, {10, 4, 7, 2, 0.547722553823488}, {35, 9, 7, 2, 0.7302967311502416},
    {7, 4, 7, 3, 0.547722553823488}, {33, 9, 7, 3, -0.4472135802063231}, {37, 9, 7, 3, 0.5773502802989676},
    {23, 5, 8, 0, 1.0}, {4, 4, 8, 1, -0.547722553823488}, {27, 9, 8, 1, 0.7071067679409303},
    {31, 9, 8, 1, 0.1825741827875604}, {37, 9, 8, 2, 0.5773502802989676}, {10, 4, 8, 3, 0.547722553823488},
    {35, 9, 8, 3, -0.1825741827875604}, {39, 9, 8, 3, 0.7071067679409303}, {26, 8, 9, 0, 1.0},
    {25, 7, 9, 1, 0.5976142934726071}, {13, 7, 9, 3, 0.5976142934726071}, {28, 8, 10, 0, 1.0},
    {22, 7, 10, 1, 0.4879500458633741}, {13, 7, 10, 2, 0.4879500458633741}, {16, 7, 10, 3, 0.4879500458633741},
    {30, 8, 11, 0, 1.0}, {19, 7, 11, 1, 0.5345224892044912}, {25, 7, 11, 1, 0.15430334738648857},
    {16, 7, 11, 2, 0.6172133895459543}, {13, 7, 11, 3, -0.15430334738648857}, {32, 8, 12, 0, 1.0},
    {16, 7, 12, 1, -0.3779644600837467}, {19, 7, 12, 2, 0.6546536846381416}, {22, 7, 12, 3, -0.3779644600837467},
    {34, 8, 13, 0, 1.0}, {13, 7, 13, 1, -0.15430334738648857}, {22, 7, 13, 2, 0.6172133895459543},
    {19, 7, 13, 3, 0.5345224892044912}, {25, 7, 13, 3, -0.15430334738648857}, {36, 8, 14, 0, 1.0},
    {16, 7, 14, 1, -0.4879500458633741}, {25, 7, 14, 2, 0.4879500458633741}, {22, 7, 14, 3, 0.4879500458633741},
    {38, 8, 15, 0, 1.0}, {13, 7, 15, 1, -0.5976142934726071}, {25, 7, 15, 3, 0.5976142934726071}};
constexpr int num_lme = [] {
    int n = 0;
    for (int l : Polar::Phi1_l) n += 2*l + 1;
    return n;
}();

constexpr int Ce = 128;
constexpr int BATCH = 64;

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("MACEPolarCuda: ") + what + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what)
{
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("MACEPolarCuda: ") + what + ": cuBLAS error " + std::to_string(s));
}
#define CHECK(x) check((x), #x)

int blocks(long n, int per_block)
{
    return std::max(1, int((n + per_block - 1)/per_block));
}

std::vector<int> offsets(const std::vector<int>& mul)
{
    std::vector<int> off(mul.size() + 1, 0);
    for (int l = 0; l < int(mul.size()); ++l)
        off[l+1] = off[l] + (2*l+1)*mul[l];
    return off;
}

struct Lay { int L, mul[4], off[5]; };
Lay lay(const std::vector<int>& mul)
{
    Lay y{int(mul.size()) - 1, {}, {}};
    const auto off = offsets(mul);
    for (int l = 0; l < int(mul.size()); ++l)
        y.mul[l] = mul[l];
    for (int l = 0; l <= int(mul.size()); ++l)
        y.off[l] = off[l];
    return y;
}

Lay Phi1_lay()
{
    std::vector<int> mul(4, 0);
    for (int l : Polar::Phi1_l)
        mul[l] += 1;
    return lay(mul);
}

__host__ __device__ constexpr long row(int lm, int node, int N) {
    const int l = lm_l(lm);
    return (long)N*l*l + (long)node*(2*l+1) + lm - l*l;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}
__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf(float x) { return 1/(1 + expf(-x)); }
__device__ __forceinline__ float silu_deriv(float x) {
    const float s = sigmoidf(x);
    return s*(1 + x*(1 - s));
}

__host__ __device__ inline void spherical_harmonics(float x, float y, float z, float* Y, float* Y_grad) {
    const float r_inv = 1/sqrtf(x*x + y*y + z*z);
    const float xx = x*x;
    const float yy = y*y;
    const float zz = z*z;
    const float S[16] = {1, y, z, x, x*y, y*z, 2*zz - xx - yy, x*z, xx - yy, y*(3*xx - yy), x*y*z, y*(4*zz - xx - yy),
                         z*(2*zz - 3*xx - 3*yy), x*(4*zz - xx - yy), z*(xx - yy), x*(xx - 3*yy)};
    const float dS[3][16] = {
        {0, 0, 0, 1, y, 0, -2*x, z, 2*x, 6*x*y, y*z, -2*x*y, -6*x*z, 4*zz - 3*xx - yy, 2*x*z, 3*xx - 3*yy},
        {0, 1, 0, 0, x, z, -2*y, 0, -2*y, 3*xx - 3*yy, x*z, 4*zz - xx - 3*yy, -6*y*z, -2*x*y, -2*y*z, -6*x*y},
        {0, 0, 1, 0, 0, y, 4*z, x, 0, 0, x*y, 8*y*z, 6*zz - 3*xx - 3*yy, 8*x*z, xx - yy, 0}};
    const float norm[16] = {0.28209479177387814f, 0.4886025119029199f, 0.4886025119029199f, 0.4886025119029199f,
                            1.0925484305920792f, 1.0925484305920792f, 0.31539156525252005f, 1.0925484305920792f,
                            0.5462742152960396f, 0.5900435899266435f, 2.890611442640554f, 0.4570457994644658f,
                            0.3731763325901154f, 0.4570457994644658f, 1.445305721320277f, 0.5900435899266435f};
    const float u[3] = {x*r_inv, y*r_inv, z*r_inv};
    const float r_inv_l[4] = {1, r_inv, r_inv*r_inv, r_inv*r_inv*r_inv};
    for (int lm = 0; lm < 16; ++lm) {
        const int l = lm_l(lm);
        const float c = norm[lm]*r_inv_l[l];
        Y[lm] = c*S[lm];
        if (Y_grad)
            for (int a = 0; a < 3; ++a) Y_grad[a*16 + lm] = c*(dS[a][lm] - l*S[lm]*u[a]*r_inv);
    }
}

__device__ __forceinline__ float cubic(const float* b, int num, int f, float x) {
    return b[f] + x*(b[num + f] + x*(b[2*num + f] + x*b[3*num + f]));
}
__device__ __forceinline__ void cubic(const float* b, int num, int f, float x, float& v, float& d) {
    const float c1 = b[num + f];
    const float c2 = b[2*num + f];
    const float c3 = b[3*num + f];
    v = b[f] + x*(c1 + x*(c2 + x*c3));
    d = c1 + x*(2*c2 + 3*x*c3);
}

__device__ __forceinline__ float cubic(const float* b, const __half2* h, int num, int f, float x) {
    const float2 c23 = __half22float2(h[f]);
    return b[f] + x*(b[num + f] + x*(c23.x + x*c23.y));
}
__device__ __forceinline__ void cubic(const float* b, const __half2* h, int num, int f, float x, float& v, float& d) {
    const float2 c23 = __half22float2(h[f]);
    const float c1 = b[num + f];
    v = b[f] + x*(c1 + x*(c23.x + x*c23.y));
    d = c1 + x*(2*c23.x + 3*x*c23.y);
}

struct Spl { int K; float r_min, inv_h; };
struct Edge { float x, y, z, r, t; int knot, pair; };
__device__ __forceinline__ Edge edge(const Spl& s, const float4* geometry, const int* pairs, int e) {
    const float4 g = geometry[e];
    const float t = (g.w - s.r_min)*s.inv_h;
    const int knot = min(max(int(t), 0), s.K - 1);
    return {g.x, g.y, g.z, g.w, t - knot - 0.5f, knot, pairs[e]};
}

constexpr unsigned ERR_R = 1;

// ---------------------------------------------------------------------------------------------------- graph

__global__ void k_edges(int N, int T, const int* types, const int* first, const int* neigh_types, const double* xyz,
                        const double* r, float r_min, float r_cut, float4* geometry, int* pairs, unsigned* err) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    for (int e = first[n] + lane; e < first[n+1]; e += 32) {
        const double d = r[e];
        geometry[e] = make_float4(float(xyz[3*e]/d), float(xyz[3*e+1]/d), float(xyz[3*e+2]/d), float(d));
        pairs[e] = types[n]*T + neigh_types[e];
        if (!(d >= r_min && d <= r_cut)) atomicOr(err, ERR_R);
    }
}

__global__ void k_density(int N, Spl sp, const int* first, const float4* geometry, const int* pairs, const float* table,
                          int num_pairs, float* s) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    float d0 = 0;
    float d1 = 0;
    for (int e = first[n] + lane; e < first[n+1]; e += 32) {
        const Edge q = edge(sp, geometry, pairs, e);
        d0 += cubic(table + ((long)q.pair*sp.K + q.knot)*4, 1, 0, q.t);
        d1 += cubic(table + ((long)(num_pairs + q.pair)*sp.K + q.knot)*4, 1, 0, q.t);
    }
    d0 = warp_sum(d0);
    d1 = warp_sum(d1);
    if (lane == 0) {
        s[n] = 1/(1 + d0);
        s[N + n] = 1/(1 + d1);
    }
}

__global__ void k_density_forces(int N, Spl sp, const int* first, const float4* geometry, const int* pairs,
                                 const float* table, int num_pairs, const float* d_adj, float* fe) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    for (int e = first[n] + lane; e < first[n+1]; e += 32) {
        const Edge q = edge(sp, geometry, pairs, e);
        float v, g0, g1;
        cubic(table + ((long)q.pair*sp.K + q.knot)*4, 1, 0, q.t, v, g0);
        cubic(table + ((long)(num_pairs + q.pair)*sp.K + q.knot)*4, 1, 0, q.t, v, g1);
        const float g = (d_adj[n]*g0 + d_adj[N + n]*g1)*sp.inv_h;
        fe[3*e] -= g*q.x;
        fe[3*e+1] -= g*q.y;
        fe[3*e+2] -= g*q.z;
    }
}

// ---------------------------------------------------------------------------------------------------- first layer

__global__ void k_Phi0(int N, int T, int T0, int NB, int ld, Spl sp, const int* first, const float4* geometry,
                       const int* pairs, const float* basis, float* Phi0) {
    extern __shared__ float acc[];
    __shared__ float Y[32][16];
    __shared__ float t[32];
    __shared__ int knot[32];
    __shared__ int pair[32];
    const int n = blockIdx.x;
    const int f = threadIdx.x;
    const int K0 = T*NB;
    for (int b0 = 0; b0 < T; b0 += T0) {
        const int nb = min(T0, T - b0);
        __syncthreads();
        for (int x = f; x < nb*16*NB; x += blockDim.x) acc[x] = 0;
        for (int e0 = first[n]; e0 < first[n+1]; e0 += 32) {
            __syncthreads();
            if (f < 32 && e0 + f < first[n+1]) {
                const Edge q = edge(sp, geometry, pairs, e0 + f);
                spherical_harmonics(q.x, q.y, q.z, Y[f], nullptr);
                t[f] = q.t;
                knot[f] = q.knot;
                pair[f] = q.pair;
            }
            __syncthreads();
            for (int g = f; g < NB; g += blockDim.x)
                for (int j = 0; j < min(32, first[n+1] - e0); ++j) {
                    const int b = pair[j]%T - b0;
                    if (b < 0 || b >= nb) continue;
                    const float phi = cubic(basis + ((long)pair[j]*sp.K + knot[j])*4*NB, NB, g, t[j]);
                    float* a = acc + b*16*NB + g;
                    for (int lm = 0; lm < 16; ++lm) a[lm*NB] += phi*Y[j][lm];
                }
        }
        __syncthreads();
        for (int x = f; x < nb*16*NB; x += blockDim.x) {
            const int b = x/(16*NB);
            const int lm = x/NB%16;
            const int g = x%NB;
            Phi0[row(lm, n, N)*ld + (b0 + b)*NB + g] = acc[x];
        }
    }
    for (int x = f; x < 16*(ld - K0); x += blockDim.x) Phi0[row(x/(ld - K0), n, N)*ld + K0 + x%(ld - K0)] = 0;
}

__global__ void k_Phi0_reverse(int N, int T, int NB, int ld, Spl sp, const int* first, const float4* geometry,
                               const int* pairs, const float* basis, const float* Phi0_adj, float* fe) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    for (int e = first[n] + lane; e < first[n+1]; e += 32) {
        const Edge q = edge(sp, geometry, pairs, e);
        float Y[16];
        float Y_grad[48];
        float u[16] = {};
        float w[16] = {};
        spherical_harmonics(q.x, q.y, q.z, Y, Y_grad);
        const float* b = basis + ((long)q.pair*sp.K + q.knot)*4*NB;
        const int el = q.pair%T;
        for (int f = 0; f < NB; ++f) {
            float phi, dphi;
            cubic(b, NB, f, q.t, phi, dphi);
            for (int lm = 0; lm < 16; ++lm) {
                const float P = Phi0_adj[row(lm, n, N)*ld + el*NB + f];
                u[lm] += P*phi;
                w[lm] += P*dphi;
            }
        }
        float dr = 0;
        float g[3] = {};
        for (int lm = 0; lm < 16; ++lm) {
            dr += w[lm]*Y[lm]*sp.inv_h;
            for (int a = 0; a < 3; ++a) g[a] += u[lm]*Y_grad[a*16 + lm];
        }
        fe[3*e] -= dr*q.x + g[0]/q.r;
        fe[3*e+1] -= dr*q.y + g[1]/q.r;
        fe[3*e+2] -= dr*q.z + g[2]/q.r;
    }
}

__global__ void k_scale_residual(int N, Lay y, float* X, const float* s, const int* types, const float* res_elem,
                                 const float* res) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*y.off[y.L+1]) return;
    int l = 0;
    while (i >= (long)N*y.off[l+1]) ++l;
    const int n = int((i - (long)N*y.off[l])/((2*l+1)*y.mul[l]));
    float x = X[i]*s[n];
    if (res) x += res[i];
    else if (l == 0) x += res_elem[types[n]*y.mul[0] + int(i%y.mul[0])];
    X[i] = x;
}

__global__ void k_scale_reverse(int N, Lay y, const float* X, float* X_adj, const float* s, const int* types,
                                const float* res_elem, const float* res, float* d_adj) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    float d = 0;
    for (int l = 0; l <= y.L; ++l)
        for (int x = lane; x < (2*l+1)*y.mul[l]; x += 32) {
            const long i = (long)N*y.off[l] + (long)n*(2*l+1)*y.mul[l] + x;
            const float r = res ? res[i] : l == 0 ? res_elem[types[n]*y.mul[0] + x] : 0.f;
            d += X_adj[i]*(X[i] - r);
            X_adj[i] *= s[n];
        }
    d = warp_sum(d);
    if (lane == 0) d_adj[n] = -s[n]*d;
}

__global__ void k_add_elem(int N, int C, float* y, const int* types, const float* v) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (long)N*C) y[i] += v[types[i/C]*C + i%C];
}

// ---------------------------------------------------------------------------------------------------- blocks

__global__ void k_bias(int rows, int n, float* y, const float* b) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (long)rows*n) y[i] += b[i%n];
}

__global__ void k_layer_norm(int rows, int n, float eps, const float* z, const float* w, const float* b, float* x_hat,
                             float* rstd, float* a) {
    const int r = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* zr = z + (long)r*n;
    float m = 0;
    for (int k = lane; k < n; k += 32) m += zr[k];
    m = warp_sum(m)/n;
    float v = 0;
    for (int k = lane; k < n; k += 32) v += (zr[k] - m)*(zr[k] - m);
    const float rs = rsqrtf(warp_sum(v)/n + eps);
    if (lane == 0) rstd[r] = rs;
    for (int k = lane; k < n; k += 32) {
        const float xh = (zr[k] - m)*rs;
        const float y = w[k]*xh + b[k];
        x_hat[(long)r*n + k] = xh;
        a[(long)r*n + k] = y*sigmoidf(y);
    }
}

__global__ void k_layer_norm_reverse(int rows, int n, const float* x_hat, const float* rstd, const float* w,
                                     const float* b, float* g) {
    const int r = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    float* gr = g + (long)r*n;
    const float* xh = x_hat + (long)r*n;
    float m1 = 0;
    float m2 = 0;
    for (int k = lane; k < n; k += 32) {
        gr[k] *= silu_deriv(w[k]*xh[k] + b[k])*w[k];
        m1 += gr[k];
        m2 += gr[k]*xh[k];
    }
    m1 = warp_sum(m1)/n;
    m2 = warp_sum(m2)/n;
    for (int k = lane; k < n; k += 32) gr[k] = rstd[r]*(gr[k] - m1 - xh[k]*m2);
}

__global__ void k_gate(int N, Lay in, Lay out, int scalars, float cs, float cg, const float* x, float* y) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int width = scalars;
    for (int l = 1; l <= in.L; ++l) width = max(width, in.mul[l]);
    if (i >= (long)N*width) return;
    const int n = int(i/width);
    const int k = int(i%width);
    const float* x0 = x + (long)n*in.mul[0];
    if (k < scalars) y[(long)n*scalars + k] = cs*x0[k]*sigmoidf(x0[k]);
    for (int l = 1, g = scalars; l <= in.L; g += in.mul[l], ++l)
        if (k < in.mul[l]) {
            const float s = cg*sigmoidf(x0[g + k]);
            for (int m = 0; m < 2*l+1; ++m)
                y[(long)N*out.off[l] + ((long)n*(2*l+1) + m)*out.mul[l] + k] =
                    s*x[(long)N*in.off[l] + ((long)n*(2*l+1) + m)*in.mul[l] + k];
        }
}

__global__ void k_gate_reverse(int N, Lay in, Lay out, int scalars, float cs, float cg, const float* x,
                               const float* y_adj, float* x_adj) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    int width = scalars;
    for (int l = 1; l <= in.L; ++l) width = max(width, in.mul[l]);
    if (i >= (long)N*width) return;
    const int n = int(i/width);
    const int k = int(i%width);
    const float* x0 = x + (long)n*in.mul[0];
    float* x0_adj = x_adj + (long)n*in.mul[0];
    if (k < scalars) x0_adj[k] = cs*silu_deriv(x0[k])*y_adj[(long)n*scalars + k];
    for (int l = 1, g = scalars; l <= in.L; g += in.mul[l], ++l)
        if (k < in.mul[l]) {
            const float s = sigmoidf(x0[g + k]);
            float d = 0;
            for (int m = 0; m < 2*l+1; ++m) {
                const long xi = (long)N*in.off[l] + ((long)n*(2*l+1) + m)*in.mul[l] + k;
                const float ya = y_adj[(long)N*out.off[l] + ((long)n*(2*l+1) + m)*out.mul[l] + k];
                d += x[xi]*ya;
                x_adj[xi] = cg*s*ya;
            }
            x0_adj[g + k] = cg*s*(1 - s)*d;
        }
}

__global__ void k_dot(int N, int C, const float* x, const float* z, float* out, int ld) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int u = int(i%C);
    float s = 0;
    for (int lm = 0; lm < 4; ++lm) s += x[row(lm, n, N)*C + u]*z[row(lm, n, N)*C + u];
    out[(long)n*ld + u] = s;
}

__global__ void k_dot_reverse(int N, int C, const float* x, const float* z, const float* out_adj, int ld,
                              float* x_adj, float* z_adj) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int u = int(i%C);
    const float g = out_adj[(long)n*ld + u];
    for (int lm = 0; lm < 4; ++lm) {
        const long r = row(lm, n, N)*C + u;
        x_adj[r] += g*z[r];
        z_adj[r] = g*x[r];
    }
}

__global__ void k_modulate(int N, int C, const float* x, const float* w, float* out) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int u = int(i%C);
    for (int lm = 0; lm < 4; ++lm) out[row(lm, n, N)*C + u] = x[row(lm, n, N)*C + u]*w[((long)(lm > 0)*N + n)*C + u];
}

__global__ void k_modulate_reverse(int N, int C, const float* x, const float* w, const float* out_adj,
                                   float* x_adj, float* w_adj) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int u = int(i%C);
    float s[2] = {};
    for (int lm = 0; lm < 4; ++lm) {
        const long r = row(lm, n, N)*C + u;
        x_adj[r] += out_adj[r]*w[((long)(lm > 0)*N + n)*C + u];
        s[lm > 0] += out_adj[r]*x[r];
    }
    w_adj[(long)n*C + u] = s[0];
    w_adj[((long)N + n)*C + u] = s[1];
}

__global__ void k_act(long n, float act, const float* y, float* a) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) a[i] = act*y[i]*sigmoidf(y[i]);
}
__global__ void k_act_reverse(long n, float act, const float* y, float* g) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) g[i] *= act*silu_deriv(y[i]);
}

__global__ void k_contraction(int N, int C, const float* M_weights, const float* A, float* M) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int k = int(i%C);
    float v[Polar::num_values];
    float w[Polar::num_m];
    float out[4] = {};
    for (int lm = 0; lm < 16; ++lm) v[lm] = A[row(lm, n, N)*C + k];
    for (int q = 16; q < Polar::num_values; ++q) v[q] = 0;
    for (int e = 0; e < Polar::num_m; ++e) w[e] = M_weights[(long)e*C + k];
    static_for<Polar::num_b>([&](auto e) {
        constexpr BTerm t = Polar::b[e];
        v[t.q] += float(t.c)*v[t.a]*v[t.b];
    });
    static_for<Polar::num_m>([&](auto e) {
        constexpr MTerm t = Polar::m[e];
        for (int m = 0; m < t.n; ++m) out[t.o + m] += w[e]*v[t.q + m];
    });
    for (int LM = 0; LM < 4; ++LM) M[row(LM, n, N)*C + k] = out[LM];
}

__global__ void k_contraction_reverse(int N, int C, const float* M_weights, const float* A,
                                      const float* M_adj, float* A_adj) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int k = int(i%C);
    float v[Polar::num_values];
    float w[Polar::num_m];
    float g_out[4];
    float g[Polar::num_values] = {};
    for (int lm = 0; lm < 16; ++lm) v[lm] = A[row(lm, n, N)*C + k];
    for (int q = 16; q < Polar::num_values; ++q) v[q] = 0;
    for (int e = 0; e < Polar::num_m; ++e) w[e] = M_weights[(long)e*C + k];
    for (int LM = 0; LM < 4; ++LM) g_out[LM] = M_adj[row(LM, n, N)*C + k];
    static_for<Polar::num_b>([&](auto e) {
        constexpr BTerm t = Polar::b[e];
        v[t.q] += float(t.c)*v[t.a]*v[t.b];
    });
    static_for<Polar::num_m>([&](auto e) {
        constexpr MTerm t = Polar::m[e];
        for (int m = 0; m < t.n; ++m) g[t.q + m] += w[e]*g_out[t.o + m];
    });
    static_for<Polar::num_b>([&](auto e) {
        constexpr BTerm t = Polar::b[Polar::num_b - 1 - e];
        const float s = float(t.c)*g[t.q];
        g[t.a] += s*v[t.b];
        g[t.b] += s*v[t.a];
    });
    for (int lm = 0; lm < 16; ++lm) A_adj[row(lm, n, N)*C + k] += g[lm];
}

// ---------------------------------------------------------------------------------------------------- readouts and charges

__global__ void k_energies(int N, const int* types, const double* E0, const float* e1, double* out) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n < N) out[n] = E0[types[n]] + e1[n];
}

__global__ void k_add_energies(int N, const float* e, double* out) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n < N) out[n] += e[n];
}

__global__ void k_fukui_sums(int N, const float* f, int ldf, const float* q, double* sums) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    double v[4] = {};
    if (n < N)
        for (int s = 0; s < 2; ++s) {
            v[s] = f[(long)n*ldf + s];
            v[2+s] = q[2*n + s];
        }
    for (int c = 0; c < 4; ++c) {
        const double t = warp_sum(v[c]);
        if ((threadIdx.x & 31) == 0) atomicAdd(sums + c, t);
    }
}

__global__ void k_normalize(int N, const float* f, int ldf, const double* sums, double Q0, double Q1,
                            float* q, float* fn) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int s = 0; s < 2; ++s) {
        const double F = sums[s] == 0 ? 1 : sums[s];
        const double c = (s ? Q1 : Q0) - sums[2+s];
        fn[2*n + s] = float(f[(long)n*ldf + s]/F);
        q[2*n + s] += float(fn[2*n + s]*c);
    }
}

__global__ void k_normalize_sums(int N, const float* a, const float* fn, double* sums) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    double v[2] = {};
    if (n < N) for (int s = 0; s < 2; ++s) v[s] = double(a[2*n + s])*fn[2*n + s];
    for (int c = 0; c < 2; ++c) {
        const double t = warp_sum(v[c]);
        if ((threadIdx.x & 31) == 0) atomicAdd(sums + c, t);
    }
}

__global__ void k_normalize_reverse(int N, const double* A, const double* sums, double Q0, double Q1, float* a,
                                    float* f_adj, int ldf) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int s = 0; s < 2; ++s) {
        const double F = sums[s];
        const double c = (s ? Q1 : Q0) - sums[2+s];
        const double x = a[2*n + s];
        f_adj[(long)n*ldf + s] = float(c*(F == 0 ? x : (x - A[s])/F));
        a[2*n + s] = float(x - A[s]);
    }
}

__global__ void k_add_dq(int N, const float* q, const float* o, float* q1) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 8*N) return;
    q1[i] = q[i] + (i < 2*N ? o[i/2*4 + i%2] : o[4*N + i - 2*N]);
}

__global__ void k_dq_reverse(int N, const float* p_adj, float* o_adj) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 8*N) return;
    if (i < 2*N) o_adj[i/2*4 + i%2] = p_adj[i];
    else o_adj[4*N + i - 2*N] = p_adj[i];
}
__global__ void k_scatter_f(int N, const float* f_adj, float* o_adj) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < 2*N) o_adj[i/2*4 + 2 + i%2] = f_adj[i];
}

__global__ void k_source(int N, int C, const int* types, const float* source, float* in) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (long)N*C) in[(i/C)*2*C + C + i%C] = source[(long)types[i/C]*C + i%C];
}

__global__ void k_add(long n, const float* x, const float* y, float* z) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

// ---------------------------------------------------------------------------------------------------- k-space

struct KS {
    int K, N, S;
    const float *frac, *kv, *kn, *k2inv, *Bd, *Bf, *b1d, *b1f, *sigmas;   // sigmas: density, then features
};

__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 conjf2(float2 a) { return make_float2(a.x, -a.y); }

__device__ __forceinline__ float2 basis(const KS& s, int k, int lm, int g, bool features) {
    const float* B = features ? s.Bf + 2*((k*4 + lm)*s.S + g) : s.Bd + 2*(k*4 + lm);
    return make_float2(B[0], B[1]);
}

__global__ void k_phases(KS s, float* Ec, float* Es) {
    const long x = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= (long)s.N*s.K) return;
    const int i = int(x/s.K);
    const int k = int(x%s.K);
    float t = s.kn[3*k]*s.frac[3*i] + s.kn[3*k+1]*s.frac[3*i+1] + s.kn[3*k+2]*s.frac[3*i+2];
    t -= rintf(t);
    sincospif(2*t, Es + x, Ec + x);
}

__global__ void k_complex(int n, const float* re, const float* im, float2* R) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x < n) R[x] = make_float2(re[x], -im[x]);
}

__global__ void k_project_coef(KS s, const float2* Z, int nc, bool features, float* Ure, float* Uim) {
    const int ng = features ? s.S : 1;
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= s.K*4*nc*ng) return;
    const int col = x%(nc*ng);
    const int lm = x/(nc*ng)%4;
    const int k = x/(4*nc*ng);
    const float2 u = cmul(conjf2(basis(s, k, lm, col%ng, features)), Z[(long)(col/ng)*s.K + k]);
    Ure[x] = u.x;
    Uim[x] = u.y;
}

__global__ void k_force_coef(KS s, const float2* A, int nc, bool features, float* P, float* Q) {
    const int ng = features ? s.S : 1;
    const int cols = nc*ng;
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= s.K*4*cols) return;
    const int col = x%cols;
    const int lm = x/cols%4;
    const int k = x/(4*cols);
    const float2 X = cmul(conjf2(A[(long)(col/ng)*s.K + k]), basis(s, k, lm, col%ng, features));
    for (int a = 0; a < 3; ++a) {
        const long y = (((long)k*3 + a)*4 + lm)*cols + col;
        P[y] = s.k2inv[k]*s.kv[3*k + a]*X.y;
        Q[y] = s.k2inv[k]*s.kv[3*k + a]*X.x;
    }
}

__global__ void k_force_finish(int N, int cols, const float* w, const float* Mc, const float* Ms, float cst,
                               double* f) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int a = 0; a < 3; ++a) {
        float v = 0;
        for (int lm = 0; lm < 4; ++lm)
            for (int col = 0; col < cols; ++col) {
                const long y = (((long)i*3 + a)*4 + lm)*cols + col;
                v += w[row(lm, i, N)*cols + col]*(Mc[y] - Ms[y]);
            }
        f[3*i + a] += cst*v;
    }
}

__global__ void k_on_basis(KS s, const float2* R, int nc, bool features, float2* X) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc*s.K) return;
    const int c = i/s.K;
    const int k = i%s.K;
    float2 x = make_float2(0, 0);
    for (int lm = 0; lm < 4; ++lm)
        for (int g = 0; g < (features ? s.S : 1); ++g) {
            const float* B = features ? s.Bf + 2*((k*4 + lm)*s.S + g) : s.Bd + 2*(k*4 + lm);
            const float2 r = R[(long)(features ? (lm*nc + c)*s.S + g : lm*nc + c)*s.K + k];
            x = make_float2(x.x + B[0]*r.x - B[1]*r.y, x.y + B[0]*r.y + B[1]*r.x);
        }
    X[i] = x;
}

__global__ void k_kspace_virial(KS s, const float2* Rx, bool fx, const float2* Ry, bool fy, int nc,
                                float cst, double* out) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    constexpr int axis[4] = {0, 1, 2, 0};
    double v[10] = {};
    if (k < s.K)
        for (int c = 0; c < nc; ++c) {
            float2 XY[2], dXY[2][3];
            for (int side = 0; side < 2; ++side) {
                const float2* R = side ? Ry : Rx;
                const bool f = side ? fy : fx;
                float2 X = make_float2(0, 0);
                float2 dX[3] = {};
                for (int lm = 0; lm < 4; ++lm)
                    for (int g = 0; g < (f ? s.S : 1); ++g) {
                        const float2 r = R[(long)(f ? (lm*nc + c)*s.S + g : lm*nc + c)*s.K + k];
                        const float* Bp = f ? s.Bf + 2*((k*4 + lm)*s.S + g) : s.Bd + 2*(k*4 + lm);
                        const float2 B = cmul(make_float2(Bp[0], Bp[1]), r);
                        const float sigma = f ? s.sigmas[1 + g] : s.sigmas[0];
                        X.x += B.x;
                        X.y += B.y;
                        for (int a = 0; a < 3; ++a) {
                            dX[a].x -= sigma*sigma*s.kv[3*k + a]*B.x;
                            dX[a].y -= sigma*sigma*s.kv[3*k + a]*B.y;
                        }
                        if (lm > 0) {
                            const float* b1 = f ? s.b1f + 2*(k*s.S + g) : s.b1d + 2*k;
                            const float2 t = cmul(make_float2(b1[0], b1[1]), r);
                            dX[axis[lm]].x += t.x;
                            dX[axis[lm]].y += t.y;
                        }
                    }
                XY[side] = X;
                for (int a = 0; a < 3; ++a) dXY[side][a] = dX[a];
            }
            const float k2i = s.k2inv[k];
            const float re = cmul(conjf2(XY[0]), XY[1]).x;
            v[9] += cst*k2i*re;
            for (int a = 0; a < 3; ++a) {
                const float d = cst*(-2*s.kv[3*k + a]*k2i*k2i*re
                                     + k2i*(cmul(conjf2(dXY[0][a]), XY[1]).x + cmul(conjf2(XY[0]), dXY[1][a]).x));
                for (int b = 0; b < 3; ++b) v[3*a + b] += d*s.kv[3*k + b];
            }
        }
    for (int x = 0; x < 10; ++x) {
        const double t = warp_sum(v[x]);
        if ((threadIdx.x & 31) == 0) atomicAdd(out + x, t);
    }
}

__global__ void k_coulomb(int n, int K, float c, const float* k2inv, const float2* X, float2* Z) {
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x < n) Z[x] = make_float2(c*k2inv[x%K]*X[x].x, c*k2inv[x%K]*X[x].y);
}

__global__ void k_features_finish(int N, int S, const float* q, const float* self, const float* norms, float* V) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 8*S*N) return;
    const int g = i%S;
    const int s = i/S%2;
    const int r = i/(2*S);
    const int lm = r < N ? 0 : 1 + (r - N)%3;
    V[i] = (V[i] - self[lm*S + g]*q[r*2 + s])/norms[lm*S + g];
}

__global__ void k_features_reverse(int N, int S, const float* V_adj, const float* self, const float* norms,
                                   float* v, float* q_adj) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 2*N*4) return;
    const int r = i/2;
    const int lm = r < N ? 0 : 1 + (r - N)%3;
    float d = 0;
    for (int g = 0; g < S; ++g) {
        const float x = V_adj[(long)i*S + g]/norms[lm*S + g];
        v[(long)i*S + g] = x;
        d += x*self[lm*S + g];
    }
    q_adj[i] -= d;
}

__global__ void k_rho(int N, const float* q, float* rho) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < 4*N) rho[i] = q[2*i] + q[2*i + 1];
}
__global__ void k_coulomb_finish(int N, const float* rho, const float* psi, float* q_adj,
                                 double* energies, double* total) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    double e = 0;
    if (n < N) {
        for (int lm = 0; lm < 4; ++lm) {
            const long r = row(lm, n, N);
            e += double(rho[r])*psi[r];
            q_adj[2*r] += 2*psi[r];
            q_adj[2*r + 1] += 2*psi[r];
        }
        energies[n] += e;
    }
    e = warp_sum(e);
    if ((threadIdx.x & 31) == 0) atomicAdd(total, e);
}

struct Rec { double b[9]; };   // reciprocal vectors over 2 pi, as rows

__global__ void k_frac(int N, const double* x, Rec r, float* frac) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int d = 0; d < 3; ++d) {
        const double s = -(x[3*i]*r.b[3*d] + x[3*i+1]*r.b[3*d+1] + x[3*i+2]*r.b[3*d+2]);
        frac[3*i + d] = float(s - floor(s));
    }
}

__global__ void k_edge_virial(int E, const double* xyz, const float* fe, double* out) {
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    double v[6] = {};
    if (e < E) {
        const double* x = xyz + 3*e;
        const float* f = fe + 3*e;
        v[0] = x[0]*f[0];
        v[1] = x[1]*f[1];
        v[2] = x[2]*f[2];
        v[3] = 0.5*(x[0]*f[1] + x[1]*f[0]);
        v[4] = 0.5*(x[0]*f[2] + x[2]*f[0]);
        v[5] = 0.5*(x[1]*f[2] + x[2]*f[1]);
    }
    for (int c = 0; c < 6; ++c) {
        const double t = warp_sum(v[c]);
        if ((threadIdx.x & 31) == 0) atomicAdd(out + c, t);
    }
}

__global__ void k_to_double(long n, const float* x, double* y) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

__global__ void k_fill(long n, float v, float* x) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void k_invert(long n0, long n, const float* x, float* y) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = i < n0 ? x[i] : -x[i];
}

// C = alpha A Bt^T + beta C on fp16 tensor cores, fp32 accurate: A and B split into fp16 hi + lo, hi hi + hi lo + lo hi
struct GemmDesc {
    const float* A;
    const __half *Bh, *Bl;
    float* C;
    int M, N, K, lda, ldc, tile0, shift;
    float alpha, beta;
};
struct GemmBatch { GemmDesc g[32]; int n; };

__host__ __device__ inline int scale_shift(float a) {   // a 2^shift in [2^12, 2^13)
    if (!(a > 0)) return 0;
    int e;
    frexpf(a, &e);
    return 13 - e;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define TENSOR_CORES
#endif

#ifdef TENSOR_CORES
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
                 "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void split(float2 v, float scale, uint32_t& hi, uint32_t& lo) {
    const __half hx = __float2half_rn(v.x*scale), hy = __float2half_rn(v.y*scale);
    const __half2 h = __halves2half2(hx, hy);
    const __half2 l = __halves2half2(__float2half_rn(v.x*scale - __half2float(hx)),
                                     __float2half_rn(v.y*scale - __half2float(hy)));
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
__device__ __forceinline__ void row_scale(float m, float& scale, float& unscale) {
    m = fmaxf(m, __shfl_xor_sync(~0u, m, 1));
    m = fmaxf(m, __shfl_xor_sync(~0u, m, 2));
    const int s = min(max(12 - int((__float_as_uint(m) >> 23) & 255) + 127, -100), 100);
    scale = __uint_as_float(uint32_t(127 + s) << 23);
    unscale = __uint_as_float(uint32_t(127 - s) << 23);
}
__device__ __forceinline__ float amax4(float2 a, float2 b) { return fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)),
        fmaxf(fabsf(b.x), fabsf(b.y))); }
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                 "r"(b[0]), "r"(b[1]));
}
#endif

constexpr int SA = 24;
constexpr int SB = 24;
template <int BM, int BN> constexpr int gemm_smem = 3*(sizeof(float)*BM*SA + 2*sizeof(__half)*BN*SB);

__device__ __forceinline__ void store(const GemmDesc& d, int r, int c, float x, float y, float u) {
    float2* p = reinterpret_cast<float2*>(d.C + (size_t)r*d.ldc + c);
    float2 v = make_float2(x*u, y*u);
    if (d.beta != 0) {
        const float2 o = *p;
        v.x += d.beta*o.x;
        v.y += d.beta*o.y;
    }
    *p = v;
}

template <int BM, int BN, int WM, int WN> __global__ void __launch_bounds__(32*WM*WN, 2) k_gemm(GemmBatch batch) {
#ifdef TENSOR_CORES
    constexpr int NT = 32*WM*WN, TM = BM/WM/16, TN = BN/WN/8;
    extern __shared__ __align__(16) unsigned char smem[];
    auto As = reinterpret_cast<float (*)[BM][SA]>(smem);
    auto Bh = reinterpret_cast<__half (*)[BN][SB]>(smem + sizeof(float)*3*BM*SA);
    auto Bl = reinterpret_cast<__half (*)[BN][SB]>(smem + sizeof(float)*3*BM*SA + sizeof(__half)*3*BN*SB);
    int gi = 0;
    while (gi + 1 < batch.n && batch.g[gi + 1].tile0 <= int(blockIdx.x)) ++gi;
    const GemmDesc d = batch.g[gi];
    const int tile = blockIdx.x - d.tile0;
    const int m0 = tile/(d.N/BN)*BM;
    const int n0 = tile%(d.N/BN)*BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp/WN;
    const int wn = warp%WN;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const float unscale = d.alpha*ldexpf(1.f, -d.shift);
    float acc[TM][TN][4] = {};
    const int ktiles = d.K/16;
    auto load = [&](int st, int kt) {
        for (int c = threadIdx.x; c < BM*4; c += NT) {
            const int r = c >> 2;
            const int gr = m0 + r;
            cp_async16(&As[st][r][(c & 3)*4], d.A + (size_t)(gr < d.M ? gr : 0)*d.lda + kt*16 + (c & 3)*4, gr < d.M);
        }
        for (int c = threadIdx.x; c < BN*2; c += NT) {
            const size_t g = (size_t)(n0 + (c >> 1))*d.K + kt*16 + (c & 1)*8;
            cp_async16(&Bh[st][c >> 1][(c & 1)*8], d.Bh + g, true);
            cp_async16(&Bl[st][c >> 1][(c & 1)*8], d.Bl + g, true);
        }
        asm volatile("cp.async.commit_group;");
    };
    load(0, 0);
    if (ktiles > 1)
        load(1, 1);
    else
        asm volatile("cp.async.commit_group;");
    for (int kt = 0; kt < ktiles; ++kt) {
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        if (kt + 2 < ktiles)
            load((kt + 2)%3, kt + 2);
        else
            asm volatile("cp.async.commit_group;");
        const int st = kt%3;
        uint32_t ah[TM][4], al[TM][4], bh[TN][2], bl[TN][2];
        float us[TM][2];
        for (int i = 0; i < TM; ++i) {
            const int r = wm*(BM/WM) + i*16 + gid;
            const float2 a0 = *reinterpret_cast<const float2*>(&As[st][r][2*tig]);
            const float2 a1 = *reinterpret_cast<const float2*>(&As[st][r + 8][2*tig]);
            const float2 a2 = *reinterpret_cast<const float2*>(&As[st][r][2*tig + 8]);
            const float2 a3 = *reinterpret_cast<const float2*>(&As[st][r + 8][2*tig + 8]);
            float s0, s1;
            row_scale(amax4(a0, a2), s0, us[i][0]);
            row_scale(amax4(a1, a3), s1, us[i][1]);
            split(a0, s0, ah[i][0], al[i][0]);
            split(a1, s1, ah[i][1], al[i][1]);
            split(a2, s0, ah[i][2], al[i][2]);
            split(a3, s1, ah[i][3], al[i][3]);
        }
        for (int j = 0; j < TN; ++j) {
            const int n = wn*(BN/WN) + j*8 + gid;
            bh[j][0] = *reinterpret_cast<const uint32_t*>(&Bh[st][n][2*tig]);
            bh[j][1] = *reinterpret_cast<const uint32_t*>(&Bh[st][n][2*tig + 8]);
            bl[j][0] = *reinterpret_cast<const uint32_t*>(&Bl[st][n][2*tig]);
            bl[j][1] = *reinterpret_cast<const uint32_t*>(&Bl[st][n][2*tig + 8]);
        }
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j) {
                float part[4] = {};
                mma(part, al[i], bh[j]);
                mma(part, ah[i], bl[j]);
                mma(part, ah[i], bh[j]);
                for (int q = 0; q < 4; ++q) acc[i][j][q] += part[q]*us[i][q/2];
            }
    }
    asm volatile("cp.async.wait_group 0;");
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            const int r = m0 + wm*(BM/WM) + i*16 + gid, c = n0 + wn*(BN/WN) + j*8 + 2*tig;
            if (r < d.M) store(d, r, c, acc[i][j][0], acc[i][j][1], unscale);
            if (r + 8 < d.M) store(d, r + 8, c, acc[i][j][2], acc[i][j][3], unscale);
        }
#endif
}

template <int BM, int BN, int WM, int WN> void gemm_batch(GemmBatch b, cudaStream_t s)
{
    int tiles = 0;
    for (int i = 0; i < b.n; ++i) {
        b.g[i].tile0 = tiles;
        tiles += (b.g[i].M + BM - 1)/BM*(b.g[i].N/BN);
    }
    if (not tiles)
        return;
    k_gemm<BM, BN, WM, WN><<<tiles, 32*WM*WN, gemm_smem<BM, BN>, s>>>(b);
}
void gemm_batches(std::vector<GemmDesc>& pending, int num_sms, cudaStream_t s)
{
    for (size_t i0 = 0; i0 < pending.size(); i0 += 32) {
        GemmBatch b{};
        bool n128 = true;
        bool n64 = true;
        long big = 0;
        for (size_t i = i0; i < std::min(pending.size(), i0 + 32); ++i) {
            b.g[b.n++] = pending[i];
            n128 = n128 and pending[i].N%128 == 0;
            n64 = n64 and pending[i].N%64 == 0;
            big += (pending[i].M + 127)/128*(pending[i].N/32);
        }
        if (big < 4*num_sms)
            gemm_batch<32, 32, 1, 1>(b, s);
        else if (n128)
            gemm_batch<128, 128, 4, 2>(b, s);
        else if (n64)
            gemm_batch<128, 64, 4, 2>(b, s);
        else
            gemm_batch<128, 32, 4, 1>(b, s);
    }
    pending.clear();
}

void hermite(const Vec& v, const Vec& d, double h, int num, int f, float* out)
{
    const int K = int(v.size()) - 1;
    for (int i = 0; i < K; ++i) {
        const double a0 = v[i];
        const double a1 = h*d[i];
        const double a2 = 3*(v[i+1] - v[i]) - h*(2*d[i] + d[i+1]);
        const double a3 = 2*(v[i] - v[i+1]) + h*(d[i] + d[i+1]);
        float* c = out + (long)i*4*num + f;
        c[0] = float(a0 + a1/2 + a2/4 + a3/8);
        c[num] = float(a1 + a2 + 0.75*a3);
        c[2*num] = float(a2 + 1.5*a3);
        c[3*num] = float(a3);
    }
}

// ---------------------------------------------------------------------------------------------------- second layer

struct Staged { float Y[16]; float G[3][16]; float u[3]; float t; int knot, pair, nbr; };
__device__ __forceinline__ void stage(const Spl& sp, const float4* geometry, const int* pairs, const int* neigh_indices,
                                      int e, bool grad, Staged& s) {
    const Edge q = edge(sp, geometry, pairs, e);
    float Y_grad[48];
    spherical_harmonics(q.x, q.y, q.z, s.Y, grad ? Y_grad : nullptr);
    if (grad)
        for (int a = 0; a < 3; ++a)
            for (int lm = 0; lm < 16; ++lm) s.G[a][lm] = Y_grad[a*16 + lm]/q.r;
    s.u[0] = q.x;
    s.u[1] = q.y;
    s.u[2] = q.z;
    s.t = q.t;
    s.knot = q.knot;
    s.pair = q.pair;
    s.nbr = neigh_indices[e];
}

__device__ __forceinline__ long phi1_row(const Lay& y, int lme, int n, int N) {
    int l = 0;
    while (lme >= y.off[l+1]) ++l;
    return (long)N*y.off[l] + (long)n*(2*l+1)*y.mul[l] + lme - y.off[l];
}

__global__ void __launch_bounds__(Ce) k_Phi1(int N, Lay y, Spl sp, const int* first, const float4* geometry,
                                             const int* pairs, const int* neigh_indices, const float* R1,
                                             const __half2* R1h, const float* H1e, float* Phi1) {
    __shared__ Staged st[BATCH];
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    constexpr int F = Polar::num_paths*Ce;
    float a[num_lme] = {};
    for (int e0 = first[n]; e0 < first[n+1]; e0 += BATCH) {
        const int num_j = min(BATCH, first[n+1] - e0);
        __syncthreads();
        if (threadIdx.x < num_j) stage(sp, geometry, pairs, neigh_indices, e0 + threadIdx.x, false, st[threadIdx.x]);
        __syncthreads();
        for (int j = 0; j < num_j; ++j) {
            const Staged& q = st[j];
            const long blk = (long)q.pair*sp.K + q.knot;
            float Y[16];
            float h[4];
            float R[Polar::num_paths];
            for (int lm = 0; lm < 16; ++lm) Y[lm] = q.Y[lm];
            for (int lm2 = 0; lm2 < 4; ++lm2) h[lm2] = H1e[((long)q.nbr*4 + lm2)*Ce + k];
            for (int p = 0; p < Polar::num_paths; ++p)
                R[p] = R1h ? cubic(R1 + blk*2*F, R1h + blk*F, F, p*Ce + k, q.t) : cubic(R1 + blk*4*F, F, p*Ce + k, q.t);
            static_for<Polar::num_terms>([&](auto q) {
                constexpr Term t = Phi1_terms[q];
                a[t.lme] += (float(t.cg)*Y[t.lm1])*(R[t.path]*h[t.lm2]);
            });
        }
    }
    for (int lme = 0; lme < num_lme; ++lme) Phi1[phi1_row(y, lme, n, N)*Ce + k] = a[lme];
}

__global__ void __launch_bounds__(Ce, 3) k_Phi1_reverse(int N, Lay y, Spl sp, const int* first, const float4* geometry,
                                                        const int* pairs, const int* neigh_indices, const float* R1,
                                                        const __half2* R1h, const float* H1e, const float* dPhi1,
                                                        float* H1e_adj, float* fe) {
    __shared__ Staged st[BATCH];
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    constexpr int F = Polar::num_paths*Ce;
    float adj[num_lme];
    for (int lme = 0; lme < num_lme; ++lme) adj[lme] = dPhi1[phi1_row(y, lme, n, N)*Ce + k];
    for (int e0 = first[n]; e0 < first[n+1]; e0 += BATCH) {
        const int num_j = min(BATCH, first[n+1] - e0);
        __syncthreads();
        if (threadIdx.x < num_j) stage(sp, geometry, pairs, neigh_indices, e0 + threadIdx.x, true, st[threadIdx.x]);
        __syncthreads();
        for (int j = 0; j < num_j; ++j) {
            const Staged& q = st[j];
            const long blk = (long)q.pair*sp.K + q.knot;
            float Y[16];
            float h[4];
            float R[Polar::num_paths];
            float dR[Polar::num_paths];
            float g_R[Polar::num_paths] = {};
            float g_Y[16] = {};
            float g_h[4] = {};
            for (int lm = 0; lm < 16; ++lm) Y[lm] = q.Y[lm];
            for (int lm2 = 0; lm2 < 4; ++lm2) h[lm2] = H1e[((long)q.nbr*4 + lm2)*Ce + k];
            for (int p = 0; p < Polar::num_paths; ++p) {
                if (R1h) cubic(R1 + blk*2*F, R1h + blk*F, F, p*Ce + k, q.t, R[p], dR[p]);
                else cubic(R1 + blk*4*F, F, p*Ce + k, q.t, R[p], dR[p]);
            }
            static_for<Polar::num_terms>([&](auto q) {
                constexpr Term t = Phi1_terms[q];
                const float a = float(t.cg)*adj[t.lme];
                const float aY = a*Y[t.lm1];
                g_R[t.path] += aY*h[t.lm2];
                g_h[t.lm2] += aY*R[t.path];
                g_Y[t.lm1] += a*(R[t.path]*h[t.lm2]);
            });
            float f_r = 0;
            for (int p = 0; p < Polar::num_paths; ++p) f_r += dR[p]*g_R[p];
            f_r *= sp.inv_h;
            float f[3];
            for (int a = 0; a < 3; ++a) {
                f[a] = f_r*q.u[a];
                for (int lm = 0; lm < 16; ++lm) f[a] += q.G[a][lm]*g_Y[lm];
                f[a] = warp_sum(f[a]);
            }
            for (int lm2 = 0; lm2 < 4; ++lm2) atomicAdd(H1e_adj + ((long)q.nbr*4 + lm2)*Ce + k, g_h[lm2]);
            if ((threadIdx.x & 31) == 0)
                for (int a = 0; a < 3; ++a) atomicAdd(fe + 3L*(e0 + j) + a, -f[a]);
        }
    }
}

__global__ void k_ex(int N, int Ce, const float* h, float* ex) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < 4L*N*Ce) ex[i] = h[row(i/Ce%4, i/(4*Ce), N)*Ce + i%Ce];
}
__global__ void k_unex(int N, int Ce, const float* ex, float* h, bool add) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 4L*N*Ce) return;
    float& y = h[row(i/Ce%4, i/(4*Ce), N)*Ce + i%Ce];
    y = (add ? y : 0.f) + ex[i];
}

std::vector<float> floats(const json& j)
{
    std::vector<float> v;
    if (j.is_number())
        v.push_back(j.get<float>());
    else
        for (const auto& x : j)
            for (float y : floats(x))
                v.push_back(y);
    return v;
}

}

// ---------------------------------------------------------------------------------------------------- building blocks

void MACEPolarCuda::gemm(bool trans, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float beta,
                         float* c, int ldc, float alpha) const
{
    if (m == 0 or n == 0)
        return;
    if (not (float32_mixed and tensor_cores and k%16 == 0 and n%32 == 0 and lda%4 == 0 and ldc%2 == 0
          and reinterpret_cast<uintptr_t>(a)%16 == 0 and reinterpret_cast<uintptr_t>(c)%8 == 0
          and (m + 127)/128*(n/32) >= 4*num_sms))
        return sgemm(trans, m, n, k, a, lda, b, ldb, beta, c, ldc, alpha);
    auto [it, fresh] = halves.try_emplace({b, trans});
    Halves& H = it->second;
    if (fresh) {
        const int cols = trans ? k : n;
        std::vector<float> w((long)n*k);
        CHECK(cudaMemcpy2D(w.data(), cols*sizeof(float), b, ldb*sizeof(float), cols*sizeof(float), trans ? n : k,
                           cudaMemcpyDeviceToHost));
        float top = 0;
        for (float x : w)
            top = std::max(top, std::abs(x));
        H.shift = scale_shift(top);
        std::vector<__half> hi(w.size()), lo(w.size());
        for (long j = 0; j < n; ++j)
            for (long i = 0; i < k; ++i) {
                const float x = std::ldexp(trans ? w[j*k + i] : w[i*n + j], H.shift);
                hi[j*k + i] = __float2half_rn(x);
                lo[j*k + i] = __float2half_rn(x - __half2float(hi[j*k + i]));
            }
        H.hi.upload(hi);
        H.lo.upload(lo);
        CHECK(cudaDeviceSynchronize());
    }
    std::vector<GemmDesc> g{{a, H.hi, H.lo, c, m, n, k, lda, ldc, 0, H.shift, alpha, beta}};
    gemm_batches(g, num_sms, stream);
}

void MACEPolarCuda::sgemm(bool trans, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float beta,
                          float* c, int ldc, float alpha) const
{
    if (m == 0 or n == 0)
        return;
    CHECK(cublasSgemm(blas, trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, ldb, a, lda, &beta,
                      c, ldc));
}

void MACEPolarCuda::Linear::forward(const MACEPolarCuda& p, int n, const float* x, float* y, bool add) const
{
    for (int l = 0; l < int(out.size()); ++l) {
        float* y_l = y + (long)n*off_out[l];
        if (in[l] > 0)
            p.gemm(false, n*(2*l+1), out[l], in[l], x + (long)n*off_in[l], in[l], weights + weights_offset[l], out[l],
                   add, y_l, out[l]);
        else if (not add)
            CHECK(cudaMemsetAsync(y_l, 0, (size_t)n*(2*l+1)*out[l]*sizeof(float), p.stream));
    }
    k_bias<<<blocks((long)n*out[0], 256), 256, 0, p.stream>>>(n, out[0], y, bias);
}

void MACEPolarCuda::Linear::reverse(const MACEPolarCuda& p, int n, const float* y_adj, float* x_adj) const
{
    for (int l = 0; l < int(out.size()); ++l)
        if (in[l] > 0)
            p.gemm(true, n*(2*l+1), in[l], out[l], y_adj + (long)n*off_out[l], out[l],
                   weights + weights_offset[l], out[l], 1,
                   x_adj + (long)n*off_in[l], in[l]);
}

void MACEPolarCuda::MLP::forward(const MACEPolarCuda& p, int n, const float* x, float* y)
{
    const float* in = x;
    for (size_t t = 0; t < layers.size(); ++t) {
        auto& L = layers[t];
        const bool last = t + 1 == layers.size();
        L.z.reserve((long)n*L.out);
        float* out = last ? y : L.z.p;
        p.gemm(false, n, L.out, L.in, in, L.in, L.weights, L.out, 0, out, L.out);
        k_bias<<<blocks((long)n*L.out, 256), 256, 0, p.stream>>>(n, L.out, out, L.bias);
        if (last)
            break;
        L.x_hat.reserve((long)n*L.out);
        L.rstd.reserve(n);
        L.a.reserve((long)n*L.out);
        k_layer_norm<<<blocks(n, 4), 128, 0, p.stream>>>(n, L.out, L.ln_eps, L.z, L.ln_weights, L.ln_bias,
                                                         L.x_hat, L.rstd, L.a);
        in = L.a;
    }
}

void MACEPolarCuda::MLP::reverse(const MACEPolarCuda& p, int n, const float* y_adj, float* x_adj)
{
    g.reserve((long)n*layers.back().out);
    CHECK(cudaMemcpyAsync(g, y_adj, (size_t)n*layers.back().out*sizeof(float), cudaMemcpyDeviceToDevice, p.stream));
    for (int t = int(layers.size()) - 1; t >= 0; --t) {
        const auto& L = layers[t];
        if (t + 1 < int(layers.size()))
            k_layer_norm_reverse<<<blocks(n, 4), 128, 0, p.stream>>>(n, L.out, L.x_hat, L.rstd, L.ln_weights,
                                                                     L.ln_bias, g);
        float* dst = x_adj;
        if (t > 0) {
            h.reserve((long)n*L.in);
            dst = h.p;
        }
        p.gemm(true, n, L.in, L.out, g, L.out, L.weights, L.out, 0, dst, L.in);
        if (t > 0)
            std::swap(g, h);
    }
}

void MACEPolarCuda::Gate::forward(const MACEPolarCuda& p, int n, const float* x_in, float* y)
{
    std::vector<int> out = mul;
    out[0] = scalars;
    const long dim = (long)n*offsets(mul).back();
    x.reserve(dim);
    CHECK(cudaMemcpyAsync(x, x_in, dim*sizeof(float), cudaMemcpyDeviceToDevice, p.stream));
    int width = scalars;
    for (size_t l = 1; l < mul.size(); ++l)
        width = std::max(width, mul[l]);
    k_gate<<<blocks((long)n*width, 256), 256, 0, p.stream>>>(n, lay(mul), lay(out), scalars, scalar_constant,
                                                             gate_constant, x, y);
}

void MACEPolarCuda::Gate::reverse(const MACEPolarCuda& p, int n, const float* y_adj, float* x_adj) const
{
    std::vector<int> out = mul;
    out[0] = scalars;
    int width = scalars;
    for (size_t l = 1; l < mul.size(); ++l)
        width = std::max(width, mul[l]);
    k_gate_reverse<<<blocks((long)n*width, 256), 256, 0, p.stream>>>(n, lay(mul), lay(out), scalars, scalar_constant,
                                                                     gate_constant, x, y_adj, x_adj);
}

void MACEPolarCuda::Dot::forward(const MACEPolarCuda& p, int n, const float* x, const float* y, float* out, int ld)
{
    z.reserve((long)n*4*num_channels);
    for (int l = 0; l < 2; ++l)
        p.gemm(false, n*(2*l+1), num_channels, num_channels, y + (long)n*l*num_channels, num_channels,
               weights + (long)l*num_channels*num_channels, num_channels, 0, z + (long)n*l*num_channels, num_channels);
    k_dot<<<blocks((long)n*num_channels, 256), 256, 0, p.stream>>>(n, num_channels, x, z, out, ld);
}

void MACEPolarCuda::Dot::reverse(const MACEPolarCuda& p, int n, const float* x, const float* out_adj, int ld,
                                 float* x_adj, float* y_adj)
{
    z_adj.reserve((long)n*4*num_channels);
    k_dot_reverse<<<blocks((long)n*num_channels, 256), 256, 0, p.stream>>>(n, num_channels, x, z, out_adj, ld,
            x_adj, z_adj);
    for (int l = 0; l < 2; ++l)
        p.gemm(true, n*(2*l+1), num_channels, num_channels, z_adj + (long)n*l*num_channels, num_channels,
               weights + (long)l*num_channels*num_channels, num_channels, 1,
               y_adj + (long)n*l*num_channels, num_channels);
}

void MACEPolarCuda::Modulation::forward(const MACEPolarCuda& p, int n, const float* x, const float* s, float* out)
{
    w.reserve((long)n*2*num_channels);
    for (int l = 0; l < 2; ++l)
        p.gemm(false, n, num_channels, num_channels, s, num_channels, weights + (long)l*num_channels*num_channels,
               num_channels, 0, w + (long)l*n*num_channels, num_channels);
    k_modulate<<<blocks((long)n*num_channels, 256), 256, 0, p.stream>>>(n, num_channels, x, w, out);
}

void MACEPolarCuda::Modulation::reverse(const MACEPolarCuda& p, int n, const float* x, const float* out_adj,
                                        float* x_adj, float* s_adj)
{
    w_adj.reserve((long)n*2*num_channels);
    k_modulate_reverse<<<blocks((long)n*num_channels, 256), 256, 0, p.stream>>>(n, num_channels, x, w, out_adj,
            x_adj, w_adj);
    for (int l = 0; l < 2; ++l)
        p.gemm(true, n, num_channels, num_channels, w_adj + (long)l*n*num_channels, num_channels,
               weights + (long)l*num_channels*num_channels, num_channels, 1, s_adj, num_channels);
}

void MACEPolarCuda::BiasReadout::forward(const MACEPolarCuda& p, int n, const float* x, float* out)
{
    const int h = linear_1.out[0];
    y1.reserve((long)n*h);
    y2.reserve((long)n*h);
    a.reserve((long)n*h);
    linear_1.forward(p, n, x, y1, false);
    k_act<<<blocks((long)n*h, 256), 256, 0, p.stream>>>((long)n*h, act, y1, a);
    linear_mid.forward(p, n, a, y2, false);
    k_act<<<blocks((long)n*h, 256), 256, 0, p.stream>>>((long)n*h, act, y2, a);
    linear_2.forward(p, n, a, out, false);
}

void MACEPolarCuda::BiasReadout::reverse(const MACEPolarCuda& p, int n, const float* out_adj, float* x_adj)
{
    const int h = linear_1.out[0];
    g.reserve((long)n*h);
    f.reserve((long)n*h);
    CHECK(cudaMemsetAsync(g, 0, (size_t)n*h*sizeof(float), p.stream));
    CHECK(cudaMemsetAsync(f, 0, (size_t)n*h*sizeof(float), p.stream));
    linear_2.reverse(p, n, out_adj, g);
    k_act_reverse<<<blocks((long)n*h, 256), 256, 0, p.stream>>>((long)n*h, act, y2, g);
    linear_mid.reverse(p, n, g, f);
    k_act_reverse<<<blocks((long)n*h, 256), 256, 0, p.stream>>>((long)n*h, act, y1, f);
    linear_1.reverse(p, n, f, x_adj);
}

void MACEPolarCuda::GatedReadout::forward(const MACEPolarCuda& p, int n, const float* x, float* out)
{
    const long di = (long)n*linear_1.off_out.back();
    const long dg = (long)n*linear_mid.off_in.back();
    y.reserve(di);
    g.reserve(dg);
    linear_1.forward(p, n, x, y, false);
    gate_1.forward(p, n, y, g);
    linear_mid.forward(p, n, g, y, false);
    gate_2.forward(p, n, y, g);
    linear_2.forward(p, n, g, out, false);
}

void MACEPolarCuda::GatedReadout::reverse(const MACEPolarCuda& p, int n, const float* out_adj, float* x_adj)
{
    const long di = (long)n*linear_1.off_out.back();
    const long dg = (long)n*linear_mid.off_in.back();
    g_adj.reserve(dg);
    y_adj.reserve(di);
    CHECK(cudaMemsetAsync(g_adj, 0, dg*sizeof(float), p.stream));
    linear_2.reverse(p, n, out_adj, g_adj);
    gate_2.reverse(p, n, g_adj, y_adj);
    CHECK(cudaMemsetAsync(g_adj, 0, dg*sizeof(float), p.stream));
    linear_mid.reverse(p, n, y_adj, g_adj);
    gate_1.reverse(p, n, g_adj, y_adj);
    linear_1.reverse(p, n, y_adj, x_adj);
}

// ---------------------------------------------------------------------------------------------------- k-space

static KS ks(const MACEPolarCuda& p)
{
    return {p.num_k, p.num_nodes, int(p.feature_sigmas.size()), p.frac, p.kv, p.kn, p.k2inv, p.density_basis,
            p.feature_basis, p.density_basis_deriv, p.feature_basis_deriv, p.consts};
}

void MACEPolarCuda::setup_kspace(const double* positions, std::span<const double> cell)
{
    const double* a = cell.data();
    const bool same = kspace_cell == std::vector<double>(cell.begin(), cell.end());
    const double det = a[0]*(a[4]*a[8]-a[5]*a[7]) - a[1]*(a[3]*a[8]-a[5]*a[6]) + a[2]*(a[3]*a[7]-a[4]*a[6]);
    volume = std::abs(det);
    double b[3][3];
    for (int i = 0; i < 3; ++i) {
        const double* u = a + 3*((i+1)%3);
        const double* w = a + 3*((i+2)%3);
        b[i][0] = 2*std::numbers::pi*(u[1]*w[2]-u[2]*w[1])/det;
        b[i][1] = 2*std::numbers::pi*(u[2]*w[0]-u[0]*w[2])/det;
        b[i][2] = 2*std::numbers::pi*(u[0]*w[1]-u[1]*w[0])/det;
    }
    Rec rec;
    for (int x = 0; x < 9; ++x)
        rec.b[x] = b[x/3][x%3]/(2*std::numbers::pi);
    frac.reserve(3*num_nodes);
    k_frac<<<blocks(num_nodes, 128), 128, 0, stream>>>(num_nodes, positions, rec, frac);
    if (not same) {
        kspace_cell.assign(cell.begin(), cell.end());
        kspace_vectors(a, b);
    }
    phase_cos.reserve((long)num_nodes*num_k);
    phase_sin.reserve((long)num_nodes*num_k);
    kre.reserve(16L*num_k);
    kim.reserve(16L*num_k);
    kP.reserve(48L*num_k);
    kQ.reserve(48L*num_k);
    kMc.reserve(48L*num_nodes);
    kMs.reserve(48L*num_nodes);
    k_phases<<<blocks((long)num_nodes*num_k, 256), 256, 0, stream>>>(ks(*this), phase_cos, phase_sin);
}

void MACEPolarCuda::kspace_vectors(const double* a, const double (*b)[3])
{
    int nmax[3];
    for (int i = 0; i < 3; ++i) {
        const double norm = std::sqrt(a[3*i]*a[3*i] + a[3*i+1]*a[3*i+1] + a[3*i+2]*a[3*i+2]);
        nmax[i] = int(std::ceil(kspace_cutoff/((b[i][0]*a[3*i] + b[i][1]*a[3*i+1] + b[i][2]*a[3*i+2])/norm)));
    }
    std::vector<float> hk, hn, hk2, hBd, hBf, hb1d, hb1f;
    const int num_sigmas = int(feature_sigmas.size());
    const double y0 = 0.5/std::sqrt(std::numbers::pi);
    const double y1 = std::sqrt(0.75/std::numbers::pi);
    std::vector<std::array<int, 3>> ns;
    for (int n1 = 1; n1 < nmax[0]; ++n1)
        for (int n2 = -nmax[1]; n2 < nmax[1]; ++n2)
            for (int n3 = -nmax[2]; n3 < nmax[2]; ++n3)
                ns.push_back({n1, n2, n3});
    for (int n2 = 1; n2 < nmax[1]; ++n2)
        for (int n3 = -nmax[2]; n3 < nmax[2]; ++n3)
            ns.push_back({0, n2, n3});
    for (int n3 = 1; n3 < nmax[2]; ++n3)
        ns.push_back({0, 0, n3});
    for (const auto& [n1, n2, n3] : ns) {
        double k[3];
        for (int d = 0; d < 3; ++d)
            k[d] = n1*b[0][d] + n2*b[1][d] + n3*b[2][d];
        const double k2 = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
        if (k2 > kspace_cutoff*kspace_cutoff)
            continue;
        const double kn2 = std::sqrt(k2);
        const double Y[4] = {y0, y1*k[1]/kn2, y1*k[2]/kn2, y1*k[0]/kn2};
        for (int d = 0; d < 3; ++d)
            hk.push_back(float(k[d]));
        hn.insert(hn.end(), {float(n1), float(n2), float(n3)});
        hk2.push_back(float(1/k2));
        const double fd = std::exp(-0.5*density_sigma*density_sigma*k2);
        hBd.insert(hBd.end(), {float(density_a[0]*fd*Y[0]), 0.f});
        for (int lm = 1; lm < 4; ++lm)
            hBd.insert(hBd.end(), {0.f, float(-density_a[1]*kn2*fd*Y[lm])});
        hb1d.insert(hb1d.end(), {0.f, float(-density_a[1]*y1*fd)});
        for (int lm = 0; lm < 4; ++lm)
            for (int g = 0; g < num_sigmas; ++g) {
                const double f = std::exp(-0.5*feature_sigmas[g]*feature_sigmas[g]*k2);
                if (lm == 0)
                    hBf.insert(hBf.end(), {float(feature_a[g][0]*f*Y[0]), 0.f});
                else
                    hBf.insert(hBf.end(), {0.f, float(-feature_a[g][1]*kn2*f*Y[lm])});
            }
        for (int g = 0; g < num_sigmas; ++g) {
            const double f = std::exp(-0.5*feature_sigmas[g]*feature_sigmas[g]*k2);
            hb1f.insert(hb1f.end(), {0.f, float(-feature_a[g][1]*y1*f)});
        }
    }
    num_k = int(hk2.size());
    kv.upload(hk);
    kn.upload(hn);
    k2inv.upload(hk2);
    density_basis.upload(hBd);
    feature_basis.upload(hBf);
    density_basis_deriv.upload(hb1d);
    feature_basis_deriv.upload(hb1f);
}

void MACEPolarCuda::structure(const float* w, int num_weights, float2* out) const
{
    if (num_k == 0)
        return;
    const float one = 1;
    const float zero = 0;
    for (int lm = 0; lm < 4 and num_nodes > 0; ++lm) {
        const float* a = w + (lm == 0 ? 0 : (long)num_nodes*num_weights + (lm - 1)*num_weights);
        const int lda = lm == 0 ? num_weights : 3*num_weights;
        CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, num_k, num_weights, num_nodes, &one, phase_cos, num_k, a, lda,
                          &zero, kre + (long)lm*num_weights*num_k, num_k));
        CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, num_k, num_weights, num_nodes, &one, phase_sin, num_k, a, lda,
                          &zero, kim + (long)lm*num_weights*num_k, num_k));
    }
    k_complex<<<blocks(4*num_weights*num_k, 128), 128, 0, stream>>>(4*num_weights*num_k, kre, kim, out);
}

void MACEPolarCuda::on_basis(const float2* structure_factors, int num_columns, bool f, float2* X) const
{
    k_on_basis<<<blocks(num_columns*num_k, 128), 128, 0, stream>>>(ks(*this), structure_factors, num_columns, f, X);
}

void MACEPolarCuda::project(const float2* Zp, int num_columns, bool f, float* out, float scale) const
{
    if (num_k == 0)
        return;
    const int cols = num_columns*(f ? int(feature_sigmas.size()) : 1);
    k_project_coef<<<blocks(num_k*4*cols, 128), 128, 0, stream>>>(ks(*this), Zp, num_columns, f, kre, kim);
    for (int lm = 0; lm < 4; ++lm) {
        float* o = out + (lm == 0 ? 0 : (long)num_nodes*cols + (lm - 1)*cols);
        const int ldo = lm == 0 ? cols : 3*cols;
        sgemm(false, num_nodes, cols, num_k, phase_cos, num_k, kre + lm*cols, 4*cols, 1, o, ldo, scale);
        sgemm(false, num_nodes, cols, num_k, phase_sin, num_k, kim + lm*cols, 4*cols, 1, o, ldo, -scale);
    }
}

void MACEPolarCuda::phase_forces(const float2* A, const float* w, int num_columns, bool f, float prefactor) const
{
    if (num_k == 0)
        return;
    const int cols = num_columns*(f ? int(feature_sigmas.size()) : 1);
    k_force_coef<<<blocks(num_k*4*cols, 128), 128, 0, stream>>>(ks(*this), A, num_columns, f, kP, kQ);
    sgemm(false, num_nodes, 12*cols, num_k, phase_cos, num_k, kP, 12*cols, 0, kMc, 12*cols);
    sgemm(false, num_nodes, 12*cols, num_k, phase_sin, num_k, kQ, 12*cols, 0, kMs, 12*cols);
    k_force_finish<<<blocks(num_nodes, 128), 128, 0, stream>>>(num_nodes, cols, w, kMc, kMs, prefactor,
                                                               position_forces);
}

void MACEPolarCuda::kspace_virial(const float2* Rx, bool fx, const float2* Ry, bool fy, int num_columns,
                                  float prefactor) const
{
    k_kspace_virial<<<blocks(num_k, 128), 128, 0, stream>>>(ks(*this), Rx, fx, Ry, fy, num_columns,
                                                            prefactor, sums + 32);
}

void MACEPolarCuda::features(const float* qq, Buffer<float2>& structure_factors,
                             Buffer<float2>& basis_factors, float* VV)
{
    const int num_sigmas = int(feature_sigmas.size());
    structure_factors.reserve(8*num_k);
    basis_factors.reserve(2*num_k);
    Z.reserve(2*num_k);
    CHECK(cudaMemsetAsync(VV, 0, (size_t)8*num_sigmas*num_nodes*sizeof(float), stream));
    if (features_hook) {
        hook_q.reserve(8L*num_nodes);
        k_invert<<<blocks(8L*num_nodes, 128), 128, 0, stream>>>(2L*num_nodes, 8L*num_nodes, qq, hook_q);
        features_hook(hook_q, VV);
        k_invert<<<blocks(8L*num_sigmas*num_nodes, 128), 128, 0, stream>>>(2L*num_sigmas*num_nodes,
                8L*num_sigmas*num_nodes, VV, VV);
    } else {
        structure(qq, 2, structure_factors);
        on_basis(structure_factors, 2, false, basis_factors);
        k_coulomb<<<blocks(2*num_k, 128), 128, 0, stream>>>(2*num_k, num_k, float(2*field_constant/volume), k2inv,
                                                            basis_factors, Z);
        project(Z, 2, true, VV, 1);
    }
    k_features_finish<<<blocks(8*num_sigmas*num_nodes, 128), 128, 0, stream>>>(num_nodes, num_sigmas, qq,
            consts + 1 + num_sigmas, consts + 1 + num_sigmas + 4*num_sigmas, VV);
}

void MACEPolarCuda::features_reverse(const float* qq, const Buffer<float2>& structure_factors,
                                     const Buffer<float2>& basis_factors, const float* V_adj_, float* q_adj_)
{
    const int num_sigmas = int(feature_sigmas.size());
    const float prefactor = float(2*field_constant/volume);
    v.reserve((long)8*num_sigmas*num_nodes);
    structure_v.reserve(8*num_sigmas*num_k);
    basis_v.reserve(2*num_k);
    Z.reserve(2*num_k);
    k_features_reverse<<<blocks(8*num_nodes, 128), 128, 0, stream>>>(num_nodes, num_sigmas, V_adj_,
                                                                     consts + 1 + num_sigmas,
                                                                     consts + 1 + num_sigmas + 4*num_sigmas, v, q_adj_);
    if (features_reverse_hook) {
        hook_q.reserve(8L*num_nodes);
        hook_v.reserve(8L*num_sigmas*num_nodes);
        hook_a.reserve(8L*num_nodes);
        k_invert<<<blocks(8L*num_nodes, 128), 128, 0, stream>>>(2L*num_nodes, 8L*num_nodes, qq, hook_q);
        k_invert<<<blocks(8L*num_sigmas*num_nodes, 128), 128, 0, stream>>>(2L*num_sigmas*num_nodes,
                8L*num_sigmas*num_nodes, v, hook_v);
        CHECK(cudaMemsetAsync(hook_a, 0, 8L*num_nodes*sizeof(float), stream));
        features_reverse_hook(hook_q, hook_v, hook_a, position_forces, hook_virial.data());
        k_invert<<<blocks(8L*num_nodes, 128), 128, 0, stream>>>(2L*num_nodes, 8L*num_nodes, hook_a, hook_a);
        k_add<<<blocks(8L*num_nodes, 128), 128, 0, stream>>>(8L*num_nodes, q_adj_, hook_a, q_adj_);
        return;
    }
    structure(v, 2*num_sigmas, structure_v);
    on_basis(structure_v, 2, true, basis_v);
    k_coulomb<<<blocks(2*num_k, 128), 128, 0, stream>>>(2*num_k, num_k, prefactor, k2inv, basis_v, Z);
    project(Z, 2, false, q_adj_, 1);
    phase_forces(basis_v, qq, 2, false, prefactor);
    phase_forces(basis_factors, v, 2, true, prefactor);
    kspace_virial(structure_v, true, structure_factors, false, 2, prefactor);
}

// ---------------------------------------------------------------------------------------------------- compute

void MACEPolarCuda::compute_node_energies_forces(int num_nodes, std::span<const int> node_types,
                                                 std::span<const int> num_neigh, std::span<const int> neigh_indices,
                                                 std::span<const int> neigh_types, std::span<const double> xyz,
                                                 std::span<const double> r, std::span<const double> positions,
                                                 std::span<const double> cell, double total_charge, double total_spin,
                                                 cudaStream_t stream)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    // \mathbf{F}_i = -\partial E / \partial \mathbf{x}_i through the phases e^{\pm i\mathbf{k}\cdot\mathbf{p}_i},  \mathbf{p}_i = -\mathbf{x}_i
    compute_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r, stream);
    compute_second_layer(neigh_indices);
    compute_readouts();
    compute_charges(positions, cell, total_charge, total_spin);
    reverse_charges();
    reverse_second_layer(neigh_indices);
    reverse_first_layer(xyz);
}

void MACEPolarCuda::compute_first_layer(int num_nodes, std::span<const int> node_types, std::span<const int> num_neigh,
                                        std::span<const int> neigh_types, std::span<const double> xyz,
                                        std::span<const double> r, cudaStream_t stream, int num_ghost_nodes)
{
    this->num_nodes = num_nodes;
    this->num_edges = int(r.size());
    this->num_ghost_nodes = num_ghost_nodes;
    this->stream = stream;
    CHECK(cublasSetStream(blas, stream));
    const long N = num_nodes;
    const int C = num_channels;
    const int num_X = off_X.back();
    const int num_Q = std::accumulate(mul_X.begin(), mul_X.end(), 0);
    const Spl sp{num_knots, float(radial_spline_r_min), float(1/knot_spacing)};

    // neighbor list
    {
        first_neigh.reserve(N + 1);
        CHECK(cudaMemsetAsync(first_neigh, 0, sizeof(int), stream));
        size_t bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, bytes, num_neigh.data(), first_neigh.p + 1, num_nodes, stream);
        scan.reserve(bytes);
        if (num_nodes > 0)
            CHECK(cub::DeviceScan::InclusiveSum(scan.p, bytes, num_neigh.data(), first_neigh.p + 1, num_nodes, stream));
        this->node_types.reserve(N);
        CHECK(cudaMemcpyAsync(this->node_types, node_types.data(), N*sizeof(int), cudaMemcpyDeviceToDevice, stream));
        geometry.reserve(std::max(num_edges, 1));
        pairs.reserve(std::max(num_edges, 1));
        *error_flags = 0;
        k_edges<<<blocks(N, 4), 128, 0, stream>>>(num_nodes, num_elements, this->node_types, first_neigh,
                                                   neigh_types.data(), xyz.data(), r.data(), float(radial_spline_r_min),
                                                   float(r_cut), geometry, pairs, error_flags);
    }

    // Phi0
    // \Phi^{(0)}_{i,lm,bn} = \sum_{j:a_j=b} R^{(0)}_{a_ib,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij}),  s^{(t)}_i = 1 + \sum_j \rho^{(t)}_{a_ia_j}(r_{ij}) (t = 0, 1)
    {
        A_scale.reserve(2*N);
        k_density<<<blocks(N, 4), 128, 0, stream>>>(num_nodes, sp, first_neigh, geometry, pairs, densities,
                                                     num_elements*num_elements, A_scale);
        Phi0.reserve(N*16*num_X0_rows);
        if (num_nodes > 0)
            k_Phi0<<<num_nodes, 96, Phi0_elements*16*num_basis_0*sizeof(float), stream>>>(
                num_nodes, num_elements, Phi0_elements, num_basis_0, num_X0_rows, sp, first_neigh, geometry, pairs,
                radial_0, Phi0);
    }

    // X0
    // X^{(0)}_{i,klm} = \sum_{bn} w_{kl,bn} \Phi^{(0)}_{i,lm,bn} / s^{(0)}_i + \delta_{l0} r_{a_i,k}
    {
        X0.reserve(N*num_X);
        for (int l = 0; l < 4; ++l)
            gemm(false, N*(2*l+1), mul_X[l], num_X0_rows, Phi0 + N*l*l*num_X0_rows, num_X0_rows,
                 X0_weights + std::accumulate(mul_X.begin(), mul_X.begin() + l, 0), num_Q, 0,
                 X0 + N*off_X[l], mul_X[l]);
        k_scale_residual<<<blocks(N*num_X, 256), 256, 0, stream>>>(num_nodes, lay(mul_X), X0, A_scale, this->node_types,
                                                                   residual_0, nullptr);
    }

    // A0
    // T_{i,k00} = c_s X^{(0)}_{i,k00} \sigma(X^{(0)}_{i,k00}),  T_{i,klm} = c_g X^{(0)}_{i,klm} \sigma(X^{(0)}_{i,(lC+k)00}) (l > 0)
    // A^{(0)}_{i,klm} = \sum_{k'} w_{kl,k'} T_{i,k'lm}
    {
        G0.reserve(N*16*C);
        gate_0.forward(*this, num_nodes, X0, G0);
        A0.reserve(N*16*C);
        linear_2_0.forward(*this, num_nodes, G0, A0, false);
    }

    // M0
    // M^{(0)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{klm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(0)}_{i,kl'_\xi m'_\xi}
    {
        M0.reserve(N*4*C);
        k_contraction<<<blocks(N*C, 128), 128, 0, stream>>>(num_nodes, C, M0_weights, A0, M0);
    }

    // H1
    // H^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(0)}_{i,k'lm} + \delta_{l0} b_{a_i,k},  \tilde{H}^{(1)}_{i,klm} = \sum_{k'} \tilde{w}_{kl,k'} H^{(1)}_{i,k'lm}
    {
        H1.reserve(N*4*C);
        product_linear_0.forward(*this, num_nodes, M0, H1, false);
        k_add_elem<<<blocks(N*C, 256), 256, 0, stream>>>(num_nodes, C, H1, this->node_types, skip_0);
        H1e.reserve(N*4*num_edge_channels);
        linear_up_1.forward(*this, num_nodes, H1, H1e, false);
        H1e_ex.reserve(4L*(N + num_ghost_nodes)*num_edge_channels);
        k_ex<<<blocks(4*N*num_edge_channels, 256), 256, 0, stream>>>(num_nodes, num_edge_channels, H1e, H1e_ex);
    }
    CHECK(cudaGetLastError());
}

void MACEPolarCuda::compute_second_layer(std::span<const int> neigh_indices)
{
    const long N = num_nodes;
    const int C = num_channels;
    const int num_X = off_X.back();
    const Spl sp{num_knots, float(radial_spline_r_min), float(1/knot_spacing)};

    // Phi1
    // \Phi^{(1)}_{i,klm\eta} = \sum_j \sum_{l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \tilde{H}^{(1)}_{j,kl_2m_2}
    {
        Phi1.reserve(N*num_lme*Ce);
        if (num_nodes > 0)
            k_Phi1<<<num_nodes, Ce, 0, stream>>>(num_nodes, Phi1_lay(), sp, first_neigh, geometry, pairs,
                                                 neigh_indices.data(), radial_1,
                                                 float32_mixed ? radial_1_half.p : nullptr, H1e_ex, Phi1);
    }

    // X1
    // X^{(1)}_{i,klm} = \sum_{k'\eta'} w_{kl,k'\eta'} \Phi^{(1)}_{i,k'lm\eta'} / s^{(1)}_i + \sum_{k'} w'_{kl,k'} \tilde{H}^{(1)}_{i,k'lm}
    {
        X1.reserve(N*num_X);
        linear_1_1.forward(*this, num_nodes, Phi1, X1, false);
        X1_residual.reserve(N*num_X);
        residual_1.forward(*this, num_nodes, H1e, X1_residual, false);
        k_scale_residual<<<blocks(N*num_X, 256), 256, 0, stream>>>(num_nodes, lay(mul_X), X1, A_scale + N, node_types,
                                                                   nullptr, X1_residual);
    }

    // A1
    // A^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} T_{i,k'lm},  T from X^{(1)} as in compute_first_layer
    {
        G1.reserve(N*16*C);
        gate_1.forward(*this, num_nodes, X1, G1);
        A1.reserve(N*16*C);
        linear_2_1.forward(*this, num_nodes, G1, A1, false);
    }

    // M1
    // M^{(1)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{klm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(1)}_{i,kl'_\xi m'_\xi}
    {
        M1.reserve(N*4*C);
        k_contraction<<<blocks(N*C, 128), 128, 0, stream>>>(num_nodes, C, M1_weights, A1, M1);
    }

    // H2
    // H^{(2)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(1)}_{i,k'lm} + \sum_{k'} w'_{kl,k'} H^{(1)}_{i,k'lm}
    {
        H2.reserve(N*4*C);
        product_linear_1.forward(*this, num_nodes, M1, H2, false);
        skip_1.forward(*this, num_nodes, H1, H2, true);
    }
    CHECK(cudaGetLastError());
}

void MACEPolarCuda::compute_readouts()
{
    // readouts
    // E_i = E^{(0)}_{a_i} + \sum_k w_k H^{(1)}_{i,k00} + \mathrm{MLP}(\{H^{(2)}_{i,k00}\})
    // \bar{H}^{(1)}_{i,k00} = w_k,  \bar{H}^{(2)}_{i,k00} = \partial \mathrm{MLP} / \partial H^{(2)}_{i,k00}
    const long N = num_nodes;
    node_energies.reserve(N);
    energies.reserve(N);
    readout_1.forward(*this, num_nodes, H2, energies);
    readout_0.forward(*this, num_nodes, H1, energies, true);
    k_energies<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, node_types, atomic_energies, energies, node_energies);
    ones.reserve(N);
    k_fill<<<blocks(N, 128), 128, 0, stream>>>(N, 1.f, ones);
    H1_adj.reserve(N*4*num_channels);
    H2_adj.reserve(N*4*num_channels);
    CHECK(cudaMemsetAsync(H1_adj, 0, N*4*num_channels*sizeof(float), stream));
    CHECK(cudaMemsetAsync(H2_adj, 0, N*4*num_channels*sizeof(float), stream));
    readout_1.reverse(*this, num_nodes, ones, H2_adj);
    readout_0.reverse(*this, num_nodes, ones, H1_adj);
    CHECK(cudaGetLastError());
}

void MACEPolarCuda::reduce(double* x, int n)
{
    if (not allreduce)
        return;
    double h[4];
    CHECK(cudaMemcpyAsync(h, x, n*sizeof(double), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    allreduce(h, n);
    CHECK(cudaMemcpyAsync(x, h, n*sizeof(double), cudaMemcpyHostToDevice, stream));
}

void MACEPolarCuda::normalize(int t, const float* f, int ldf)
{
    const long N = num_nodes;
    fukui_n[t].reserve(2*N);
    k_fukui_sums<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, f, ldf, q[t], sums + 4*t);
    reduce(sums + 4*t, 4);
    k_normalize<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, f, ldf, sums + 4*t, spin_charges[0],
                                                    spin_charges[1], q[t],
                                                     fukui_n[t]);
}

void MACEPolarCuda::normalize_reverse(int t, float* f_adj, int ldf)
{
    const long N = num_nodes;
    k_normalize_sums<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, q_adj[t], fukui_n[t], sums + 16 + 2*t);
    reduce(sums + 16 + 2*t, 2);
    k_normalize_reverse<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, sums + 16 + 2*t, sums + 4*t, spin_charges[0],
                                                             spin_charges[1], q_adj[t], f_adj, ldf);
}

void MACEPolarCuda::compute_charges(std::span<const double> positions, std::span<const double> cell,
                                    double total_charge, double total_spin)
{
    const long N = num_nodes;
    const int C = num_channels;
    const int num_V = 8*int(feature_sigmas.size());
    hook_virial = {};
    sums.reserve(64);
    CHECK(cudaMemsetAsync(sums, 0, 64*sizeof(double), stream));
    position_forces.reserve(3*N);
    CHECK(cudaMemsetAsync(position_forces, 0, 3*N*sizeof(double), stream));
    if (not coulomb_hook)
        setup_kspace(positions.data(), cell);
    spin_charges = {(total_charge + total_spin - 1)/2, (total_charge - total_spin + 1)/2};

    // q0
    // q^{(0)}_{i,lms} = \sum_k (w_{ls,k} H^{(1)}_{i,klm} + w'_{ls,k} H^{(2)}_{i,klm}),  q^{(0)}_{i,00s} \leftarrow q^{(0)}_{i,00s} + \frac{f_{i,s}}{\sum_j f_{j,s}} (Q_s - \sum_j q^{(0)}_{j,00s})
    // f_{i,s} = \mathrm{MLP}_s(\{H^{(2)}_{i,k00}\}),  Q_{\uparrow\downarrow} = (Q \pm (S - 1))/2,  \mu_{i,klm} = \sum_{k'} (w_{kl,k'} H^{(1)}_{i,k'lm} + w'_{kl,k'} H^{(2)}_{i,k'lm})
    {
        for (int t = 0; t < 3; ++t)
            q[t].reserve(8*N);
        sources[0].forward(*this, num_nodes, H1, q[0], false);
        sources[1].forward(*this, num_nodes, H2, q[0], true);
        f0.reserve(2*N);
        fukui.forward(*this, num_nodes, H2, f0);
        normalize(0, f0, 2);
        mixed.reserve(N*4*C);
        mixer[0].forward(*this, num_nodes, H1, mixed, false);
        mixer[1].forward(*this, num_nodes, H2, mixed, true);
    }

    // q1, q2
    // V^{(t)}_{i,lms\sigma} = (\phi^{\sigma}_{i,lm}[q^{(t)}_s] - c_{lm\sigma} q^{(t)}_{i,lms}) / n_{lm\sigma},  \phi^{\sigma}_{i,lm}[q] = \sum_{\mathbf{k}} \frac{2\kappa}{\Omega k^2} \mathrm{Re}[\overline{B^{\sigma}_{lm}(\mathbf{k})} e^{i\mathbf{k}\cdot\mathbf{p}_i} \sum_{jl'm'} B_{l'm'}(\mathbf{k}) e^{-i\mathbf{k}\cdot\mathbf{p}_j} q_{j,l'm'}]
    // B_{lm}(\mathbf{k}) = a_l (-ik)^l e^{-\sigma_\rho^2k^2/2} Y_{lm}(\hat{\mathbf{k}}),  B^{\sigma}_{lm} with a^{\sigma}_l and \sigma,  \mathbf{k} over half the sphere k \le k_c,  \Omega the cell volume
    // e_{i,klm} = \sum_{s\sigma} w_{kl,s\sigma} V^{(t)}_{i,lms\sigma} + \sum_{k'} w_{kl,k'} \mu_{i,k'lm} + \sum_s w_{kl,s} q^{(t)}_{i,lms},  z_i = \mathrm{MLP}(\{\sum_{lm} \mu_{i,klm} \sum_{k'} D_{l,kk'} e_{i,k'lm}\}_k, u_{a_i})
    // q^{(t+1)}_{i,lms} = q^{(t)}_{i,lms} + \mathcal{G}_{lms}(\{\mu_{i,klm} \sum_{k'} M_{l,kk'} z_{i,k'}\}),  normalized as q^{(0)} with f_{i,s} = \mathcal{G}_{f,s}(\ldots)
    for (int t = 0; t < 2; ++t) {
        auto& u = updates[t];
        V[t].reserve(N*num_V);
        features(q[t], structure_factors[t], basis_factors[t], V[t]);
        u.embedding.reserve(N*4*C);
        u.potential.forward(*this, num_nodes, V[t], u.embedding, false);
        u.node.forward(*this, num_nodes, mixed, u.embedding, true);
        u.charge.forward(*this, num_nodes, q[t], u.embedding, true);
        u.mlp_in.reserve(N*2*C);
        u.dot.forward(*this, num_nodes, mixed, u.embedding, u.mlp_in, 2*C);
        k_source<<<blocks(N*C, 256), 256, 0, stream>>>(num_nodes, C, node_types, u.source, u.mlp_in);
        u.mlp_out.reserve(N*C);
        u.mlp.forward(*this, num_nodes, u.mlp_in, u.mlp_out);
        u.modulated.reserve(N*4*C);
        u.out.forward(*this, num_nodes, mixed, u.mlp_out, u.modulated);
        u.readout_out.reserve(N*10);
        u.readout.forward(*this, num_nodes, u.modulated, u.readout_out);
        k_add_dq<<<blocks(8*N, 128), 128, 0, stream>>>(num_nodes, q[t], u.readout_out, q[t+1]);
        normalize(t + 1, u.readout_out + 2, 4);
    }

    // electron energy
    // E_i += E^{\mathrm{e}}_i = \mathrm{MLP}(\{\sum_{lm} H^{(2)}_{i,klm} \sum_{k'} D_{l,kk'} \hat{q}_{i,k'lm}\}_k, \{\sum_{lm} H^{(2)}_{i,klm} \sum_{k'} D'_{l,kk'} \hat{v}_{i,k'lm}\}_k)
    // \hat{q}_{i,klm} = \sum_s w_{kl,s} (q^{(0)}_{i,lms} + q^{(2)}_{i,lms}) + \delta_{l0} b_k,  \hat{v}_{i,klm} = \sum_{s\sigma} w_{kl,s\sigma} V^{(1)}_{i,lms\sigma} + \delta_{l0} b'_k
    {
        auto& e = electron;
        e.q_sum.reserve(8*N);
        k_add<<<blocks(8*N, 128), 128, 0, stream>>>(8*N, q[2], q[0], e.q_sum);
        e.q_up.reserve(N*4*C);
        e.q.forward(*this, num_nodes, e.q_sum, e.q_up, false);
        e.v_up.reserve(N*4*C);
        e.v.forward(*this, num_nodes, V[1], e.v_up, false);
        e.mlp_in.reserve(N*2*C);
        e.dot_q.forward(*this, num_nodes, H2, e.q_up, e.mlp_in, 2*C);
        e.dot_v.forward(*this, num_nodes, H2, e.v_up, e.mlp_in + C, 2*C);
        e.mlp.forward(*this, num_nodes, e.mlp_in, energies);
        k_add_energies<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, energies, node_energies);
    }

    // electrostatic energy
    // \rho_{i,lm} = \sum_s q^{(2)}_{i,lms},  \psi_{i,lm} = \frac{1}{2} \phi_{i,lm}[\rho] (\phi^{\sigma} with B_{lm} for B^{\sigma}_{lm}),  E_i += \sum_{lm} \rho_{i,lm} \psi_{i,lm}
    // \bar{q}^{(2)}_{i,lms} = 2 \psi_{i,lm}
    {
        for (int t = 0; t < 3; ++t) {
            q_adj[t].reserve(8*N);
            CHECK(cudaMemsetAsync(q_adj[t], 0, 8*N*sizeof(float), stream));
        }
        rho.reserve(4*N);
        k_rho<<<blocks(4*N, 128), 128, 0, stream>>>(num_nodes, q[2], rho);
        psi.reserve(4*N);
        CHECK(cudaMemsetAsync(psi, 0, 4*N*sizeof(float), stream));
        if (coulomb_hook) {
            hook_q.reserve(4*N);
            k_invert<<<blocks(4*N, 128), 128, 0, stream>>>(N, 4*N, rho, hook_q);
            coulomb_hook(hook_q, psi, position_forces, hook_virial.data());
            k_invert<<<blocks(4*N, 128), 128, 0, stream>>>(N, 4*N, psi, psi);
        } else {
            rho_structure.reserve(4*num_k);
            rho_basis.reserve(num_k);
            Z.reserve(2*num_k);
            structure(rho, 1, rho_structure);
            on_basis(rho_structure, 1, false, rho_basis);
            k_coulomb<<<blocks(num_k, 128), 128, 0, stream>>>(num_k, num_k, float(field_constant/volume),
                                                              k2inv, rho_basis,
                                                               Z);
            project(Z, 1, false, psi, 1);
            phase_forces(rho_basis, rho, 1, false, float(2*field_constant/volume));
            kspace_virial(rho_structure, false, rho_structure, false, 1, float(field_constant/volume));
        }
        k_coulomb_finish<<<blocks(N, 128), 128, 0, stream>>>(num_nodes, rho, psi, q_adj[2], node_energies, sums + 12);
    }
    CHECK(cudaGetLastError());
}

void MACEPolarCuda::reverse_charges()
{
    const long N = num_nodes;
    const int C = num_channels;
    const int num_V = 8*int(feature_sigmas.size());
    mixed_adj.reserve(N*4*C);
    CHECK(cudaMemsetAsync(mixed_adj, 0, N*4*C*sizeof(float), stream));
    for (int t = 0; t < 2; ++t) {
        V_adj[t].reserve(N*num_V);
        CHECK(cudaMemsetAsync(V_adj[t], 0, N*num_V*sizeof(float), stream));
    }
    mlp_in_adj.reserve(N*2*C);
    q_up_adj.reserve(N*4*C);
    v_up_adj.reserve(N*4*C);
    modulated_adj.reserve(N*4*C);
    mlp_out_adj.reserve(N*C);
    embedding_adj.reserve(N*4*C);
    readout_out_adj.reserve(10*N);
    f_adj.reserve(2*N);

    // electron energy
    // \bar{H}^{(2)}_{i,klm} += \partial E^{\mathrm{e}}_i / \partial H^{(2)}_{i,klm},  \bar{q}^{(0)}_{i,lms} = \bar{q}^{(2)}_{i,lms} = \partial E^{\mathrm{e}}_i / \partial q^{(2)}_{i,lms},  \bar{V}^{(1)}_{i,lms\sigma} = \partial E^{\mathrm{e}}_i / \partial V^{(1)}_{i,lms\sigma}
    {
        auto& e = electron;
        CHECK(cudaMemsetAsync(q_up_adj, 0, N*4*C*sizeof(float), stream));
        CHECK(cudaMemsetAsync(v_up_adj, 0, N*4*C*sizeof(float), stream));
        e.mlp.reverse(*this, num_nodes, ones, mlp_in_adj);
        e.dot_q.reverse(*this, num_nodes, H2, mlp_in_adj, 2*C, H2_adj, q_up_adj);
        e.dot_v.reverse(*this, num_nodes, H2, mlp_in_adj + C, 2*C, H2_adj, v_up_adj);
        e.q.reverse(*this, num_nodes, q_up_adj, q_adj[2]);
        e.q.reverse(*this, num_nodes, q_up_adj, q_adj[0]);
        e.v.reverse(*this, num_nodes, v_up_adj, V_adj[1]);
    }

    // q2, q1
    // \bar{V}^{(t)} += \bar{q}^{(t+1)} \cdot \partial q^{(t+1)} / \partial V^{(t)},  \bar{\mu} += \bar{q}^{(t+1)} \cdot \partial q^{(t+1)} / \partial \mu,  \bar{q}^{(t)} += \bar{q}^{(t+1)} \cdot \partial q^{(t+1)} / \partial q^{(t)} + \bar{V}^{(t)} \cdot \partial V^{(t)} / \partial q^{(t)}
    // \mathbf{F}_i -= \bar{V}^{(t)} \cdot \partial V^{(t)} / \partial \mathbf{x}_i
    for (int t = 1; t >= 0; --t) {
        auto& u = updates[t];
        normalize_reverse(t + 1, f_adj, 2);
        k_dq_reverse<<<blocks(8*N, 128), 128, 0, stream>>>(num_nodes, q_adj[t+1], readout_out_adj);
        k_scatter_f<<<blocks(2*N, 128), 128, 0, stream>>>(num_nodes, f_adj, readout_out_adj);
        k_add<<<blocks(8*N, 128), 128, 0, stream>>>(8*N, q_adj[t], q_adj[t+1], q_adj[t]);
        CHECK(cudaMemsetAsync(modulated_adj, 0, N*4*C*sizeof(float), stream));
        CHECK(cudaMemsetAsync(mlp_out_adj, 0, N*C*sizeof(float), stream));
        CHECK(cudaMemsetAsync(embedding_adj, 0, N*4*C*sizeof(float), stream));
        u.readout.reverse(*this, num_nodes, readout_out_adj, modulated_adj);
        u.out.reverse(*this, num_nodes, mixed, modulated_adj, mixed_adj, mlp_out_adj);
        u.mlp.reverse(*this, num_nodes, mlp_out_adj, mlp_in_adj);
        u.dot.reverse(*this, num_nodes, mixed, mlp_in_adj, 2*C, mixed_adj, embedding_adj);
        u.potential.reverse(*this, num_nodes, embedding_adj, V_adj[t]);
        u.node.reverse(*this, num_nodes, embedding_adj, mixed_adj);
        u.charge.reverse(*this, num_nodes, embedding_adj, q_adj[t]);
        features_reverse(q[t], structure_factors[t], basis_factors[t], V_adj[t], q_adj[t]);
    }

    // q0
    // \bar{H}^{(1)} += \bar{q}^{(0)} \cdot \partial q^{(0)} / \partial H^{(1)} + \bar{\mu} \cdot \partial \mu / \partial H^{(1)},  \bar{H}^{(2)} += \bar{q}^{(0)} \cdot \partial q^{(0)} / \partial H^{(2)} + \bar{\mu} \cdot \partial \mu / \partial H^{(2)}
    {
        normalize_reverse(0, f_adj, 2);
        fukui.reverse(*this, num_nodes, f_adj, H2_adj);
        sources[0].reverse(*this, num_nodes, q_adj[0], H1_adj);
        sources[1].reverse(*this, num_nodes, q_adj[0], H2_adj);
        mixer[0].reverse(*this, num_nodes, mixed_adj, H1_adj);
        mixer[1].reverse(*this, num_nodes, mixed_adj, H2_adj);
    }
    CHECK(cudaGetLastError());
}

void MACEPolarCuda::reverse_second_layer(std::span<const int> neigh_indices)
{
    const long N = num_nodes;
    const int C = num_channels;
    const int num_X = off_X.back();
    const Spl sp{num_knots, float(radial_spline_r_min), float(1/knot_spacing)};
    edge_forces.reserve(3L*std::max(num_edges, 1));
    CHECK(cudaMemsetAsync(edge_forces, 0, 3L*num_edges*sizeof(float), stream));

    // H2
    // \bar{M}^{(1)}_{i,k'lm} = \sum_k w_{kl,k'} \bar{H}^{(2)}_{i,klm},  \bar{H}^{(1)}_{i,k'lm} += \sum_k w'_{kl,k'} \bar{H}^{(2)}_{i,klm}
    {
        M_adj.reserve(N*4*C);
        CHECK(cudaMemsetAsync(M_adj, 0, N*4*C*sizeof(float), stream));
        product_linear_1.reverse(*this, num_nodes, H2_adj, M_adj);
        skip_1.reverse(*this, num_nodes, H2_adj, H1_adj);
    }

    // M1
    // \bar{A}^{(1)}_{i,klm} = \sum_{l'm'} \bar{M}^{(1)}_{i,kl'm'} \partial M^{(1)}_{i,kl'm'} / \partial A^{(1)}_{i,klm}
    {
        A_adj.reserve(N*16*C);
        CHECK(cudaMemsetAsync(A_adj, 0, N*16*C*sizeof(float), stream));
        k_contraction_reverse<<<blocks(N*C, 128), 128, 0, stream>>>(num_nodes, C, M1_weights, A1, M_adj, A_adj);
    }

    // A1
    // \bar{T}_{i,k'lm} = \sum_k w_{kl,k'} \bar{A}^{(1)}_{i,klm},  \bar{X}^{(1)}_{i,k00} = c_s \bar{T}_{i,k00} \sigma(x) (1 + x (1 - \sigma(x))),  x = X^{(1)}_{i,k00}
    // \bar{X}^{(1)}_{i,klm} = c_g \bar{T}_{i,klm} \sigma(\gamma),  \bar{X}^{(1)}_{i,(lC+k)00} = c_g \sigma(\gamma) (1 - \sigma(\gamma)) \sum_m X^{(1)}_{i,klm} \bar{T}_{i,klm},  \gamma = X^{(1)}_{i,(lC+k)00} (l > 0)
    {
        G_adj.reserve(N*16*C);
        CHECK(cudaMemsetAsync(G_adj, 0, N*16*C*sizeof(float), stream));
        linear_2_1.reverse(*this, num_nodes, A_adj, G_adj);
        X_adj.reserve(N*num_X);
        gate_1.reverse(*this, num_nodes, G_adj, X_adj);
    }

    // X1
    // \bar{\tilde{H}}^{(1)}_{i,k'lm} = \sum_k w'_{kl,k'} \bar{X}^{(1)}_{i,klm},  \bar{\Phi}^{(1)}_{i,k'lm\eta'} = \sum_k w_{kl,k'\eta'} \bar{X}^{(1)}_{i,klm} / s^{(1)}_i
    // d^{(1)}_i = \sum_{klm} \bar{X}^{(1)}_{i,klm} (X^{(1)}_{i,klm} - \sum_{k'} w'_{kl,k'} \tilde{H}^{(1)}_{i,k'lm}) / s^{(1)}_i
    {
        H1e_adj.reserve(N*4*Ce);
        CHECK(cudaMemsetAsync(H1e_adj, 0, N*4*Ce*sizeof(float), stream));
        residual_1.reverse(*this, num_nodes, X_adj, H1e_adj);
        scale_adj.reserve(2*N);
        k_scale_reverse<<<blocks(N, 4), 128, 0, stream>>>(num_nodes, lay(mul_X), X1, X_adj, A_scale + N, node_types,
                                                          nullptr, X1_residual, scale_adj + N);
        dPhi1.reserve(N*num_lme*Ce);
        CHECK(cudaMemsetAsync(dPhi1, 0, N*num_lme*Ce*sizeof(float), stream));
        linear_1_1.reverse(*this, num_nodes, X_adj, dPhi1);
    }

    // Phi1
    // \bar{\tilde{H}}^{(1)}_{j,kl_2m_2} += \sum_i \sum_{lm\eta l_1m_1} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(1)}_{i,klm\eta}
    // \mathbf{F}_{ij} -= \sum_{klm\eta l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij})] \tilde{H}^{(1)}_{j,kl_2m_2} \bar{\Phi}^{(1)}_{i,klm\eta}
    {
        H1e_ex_adj.reserve(4L*(N + num_ghost_nodes)*Ce);
        CHECK(cudaMemsetAsync(H1e_ex_adj, 0, 4L*(N + num_ghost_nodes)*Ce*sizeof(float), stream));
        if (num_nodes > 0)
            k_Phi1_reverse<<<num_nodes, Ce, 0, stream>>>(num_nodes, Phi1_lay(), sp, first_neigh, geometry, pairs,
                                                         neigh_indices.data(), radial_1,
                                                         float32_mixed ? radial_1_half.p : nullptr, H1e_ex, dPhi1,
                                                         H1e_ex_adj, edge_forces);
    }
    CHECK(cudaGetLastError());
}

void MACEPolarCuda::reverse_first_layer(std::span<const double> xyz)
{
    const long N = num_nodes;
    const int C = num_channels;
    const int num_Q = std::accumulate(mul_X.begin(), mul_X.end(), 0);
    const Spl sp{num_knots, float(radial_spline_r_min), float(1/knot_spacing)};

    // H1
    // \bar{H}^{(1)}_{i,k'lm} += \sum_k \tilde{w}_{kl,k'} \bar{\tilde{H}}^{(1)}_{i,klm}
    {
        k_unex<<<blocks(4*N*num_edge_channels, 256), 256, 0, stream>>>(num_nodes, num_edge_channels,
                                                                       H1e_ex_adj, H1e_adj,
                                                                        true);
        linear_up_1.reverse(*this, num_nodes, H1e_adj, H1_adj);
    }

    // M0
    // \bar{M}^{(0)}_{i,k'lm} = \sum_k w_{kl,k'} \bar{H}^{(1)}_{i,klm},  \bar{A}^{(0)}_{i,klm} = \sum_{l'm'} \bar{M}^{(0)}_{i,kl'm'} \partial M^{(0)}_{i,kl'm'} / \partial A^{(0)}_{i,klm}
    {
        CHECK(cudaMemsetAsync(M_adj, 0, N*4*C*sizeof(float), stream));
        product_linear_0.reverse(*this, num_nodes, H1_adj, M_adj);
        CHECK(cudaMemsetAsync(A_adj, 0, N*16*C*sizeof(float), stream));
        k_contraction_reverse<<<blocks(N*C, 128), 128, 0, stream>>>(num_nodes, C, M0_weights, A0, M_adj, A_adj);
    }

    // A0
    // \bar{T}_{i,k'lm} = \sum_k w_{kl,k'} \bar{A}^{(0)}_{i,klm},  \bar{X}^{(0)} from \bar{T} as in reverse_second_layer
    {
        CHECK(cudaMemsetAsync(G_adj, 0, N*16*C*sizeof(float), stream));
        linear_2_0.reverse(*this, num_nodes, A_adj, G_adj);
        gate_0.reverse(*this, num_nodes, G_adj, X_adj);
    }

    // X0
    // \bar{\Phi}^{(0)}_{i,lm,bn} = \sum_k w_{kl,bn} \bar{X}^{(0)}_{i,klm} / s^{(0)}_i,  d^{(0)}_i = \sum_{klm} \bar{X}^{(0)}_{i,klm} (X^{(0)}_{i,klm} - \delta_{l0} r_{a_i,k}) / s^{(0)}_i
    {
        k_scale_reverse<<<blocks(N, 4), 128, 0, stream>>>(num_nodes, lay(mul_X), X0, X_adj, A_scale, node_types,
                                                          residual_0, nullptr, scale_adj);
        Phi0_adj.reserve(N*16*num_X0_rows);
        for (int l = 0; l < 4; ++l)
            gemm(true, N*(2*l+1), num_X0_rows, mul_X[l], X_adj + N*off_X[l], mul_X[l],
                 X0_weights + std::accumulate(mul_X.begin(), mul_X.begin() + l, 0), num_Q, 0,
                 Phi0_adj + N*l*l*num_X0_rows, num_X0_rows);
    }

    // Phi0
    // \mathbf{F}_{ij} += \sum_t \partial_r \rho^{(t)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(t)}_i - \sum_{lmn} \nabla_{\mathbf{r}_{ij}}[R^{(0)}_{a_ia_j,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})] \bar{\Phi}^{(0)}_{i,lm,a_jn}
    {
        k_Phi0_reverse<<<blocks(N, 4), 128, 0, stream>>>(num_nodes, num_elements, num_basis_0, num_X0_rows,
                                                         sp, first_neigh,
                                                         geometry, pairs, radial_0, Phi0_adj, edge_forces);
        k_density_forces<<<blocks(N, 4), 128, 0, stream>>>(num_nodes, sp, first_neigh, geometry, pairs, densities,
                                                           num_elements*num_elements, scale_adj, edge_forces);
    }

    // virial
    // W += \sum_{ij} \mathrm{sym}(\mathbf{r}_{ij} \otimes \mathbf{F}_{ij})
    {
        node_forces.reserve(3L*std::max(num_edges, 1));
        k_to_double<<<blocks(3L*num_edges, 256), 256, 0, stream>>>(3L*num_edges, edge_forces, node_forces);
        k_edge_virial<<<blocks(num_edges, 256), 256, 0, stream>>>(num_edges, xyz.data(), edge_forces, sums + 42);
        double h[64];
        CHECK(cudaMemcpyAsync(h, sums, sizeof(h), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        if (*error_flags & ERR_R)
            throw std::invalid_argument("MACEPolarCuda: a neighbor is out of the range of the splines");
        electrostatic_energy = h[12];
        const double* G = h + 32;
        const double* W = h + 42;
        virial = {G[0] + G[9] + W[0], G[4] + G[9] + W[1], G[8] + G[9] + W[2],
                  (G[1] + G[3])/2 + W[3], (G[2] + G[6])/2 + W[4], (G[5] + G[7])/2 + W[5]};
        for (int k = 0; k < 6; ++k)
            virial[k] += hook_virial[k];
    }
    CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------- loading

static MACEPolarCuda::Linear linear_from_json(const json& j)
{
    MACEPolarCuda::Linear L;
    L.in = j["in"].get<std::vector<int>>();
    L.out = j["out"].get<std::vector<int>>();
    L.off_in = offsets(L.in);
    L.off_out = offsets(L.out);
    std::vector<float> w;
    for (const auto& wl : j["weights"]) {
        L.weights_offset.push_back(long(w.size()));
        for (double x : wl.get<Vec>())
            w.push_back(float(x));
    }
    L.weights.upload(w.empty() ? std::vector<float>{0} : w);
    L.bias.upload(floats(j["bias"]));
    return L;
}

static MACEPolarCuda::MLP mlp_from_json(const json& j)
{
    MACEPolarCuda::MLP m;
    for (const auto& l : j) {
        MACEPolarCuda::MLP::Layer L;
        L.in = l["in"];
        L.out = l["out"];
        L.weights.upload(floats(l["weights"]));
        L.bias.upload(floats(l["bias"]));
        L.ln_eps = 0;
        if (l.contains("ln_weight")) {
            L.ln_weights.upload(floats(l["ln_weight"]));
            L.ln_bias.upload(floats(l["ln_bias"]));
            L.ln_eps = l["ln_eps"];
        }
        m.layers.push_back(std::move(L));
    }
    return m;
}

static MACEPolarCuda::Gate gate_from_json(const json& j, const std::vector<int>& mul)
{
    MACEPolarCuda::Gate g;
    g.scalars = j["scalars"];
    g.scalar_constant = j["csts"][0];
    g.gate_constant = j["csts"][1];
    g.mul = mul;
    return g;
}

static MACEPolarCuda::BiasReadout bias_readout_from_json(const json& j)
{
    MACEPolarCuda::BiasReadout r;
    r.linear_1 = linear_from_json(j["linear_1"]);
    r.linear_mid = linear_from_json(j["linear_mid"]);
    r.linear_2 = linear_from_json(j["linear_2"]);
    r.act = j["act"];
    return r;
}

static MACEPolarCuda::GatedReadout gated_readout_from_json(const json& j)
{
    MACEPolarCuda::GatedReadout r;
    r.linear_1 = linear_from_json(j["linear_1"]);
    r.linear_mid = linear_from_json(j["linear_mid"]);
    r.linear_2 = linear_from_json(j["linear_2"]);
    r.gate_1 = gate_from_json(j["gate"], r.linear_1.out);
    r.gate_2 = gate_from_json(j["gate"], r.linear_1.out);
    return r;
}

MACEPolarCuda::MACEPolarCuda(std::string filename, bool float32_mixed) : float32_mixed(float32_mixed)
{
    CHECK(cublasCreate(&blas));
    CHECK(cudaHostAlloc(&error_flags, sizeof(unsigned), cudaHostAllocMapped));
    stream = 0;
    if (float32_mixed) {
        int device;
        CHECK(cudaGetDevice(&device));
        CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
        cudaFuncAttributes attributes;
        CHECK(cudaFuncGetAttributes(&attributes, k_gemm<32, 32, 1, 1>));
        tensor_cores = attributes.ptxVersion >= 80;
        CHECK(cudaFuncSetAttribute(k_gemm<128, 128, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   gemm_smem<128, 128>));
        CHECK(cudaFuncSetAttribute(k_gemm<128, 64, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   gemm_smem<128, 64>));
    }
    load_from_json(filename);
}

MACEPolarCuda::~MACEPolarCuda()
{
    cublasDestroy(blas);
    cudaFreeHost(error_flags);
}

void MACEPolarCuda::load_from_json(std::string filename)
{
    std::ifstream f(filename);
    if (not f)
        throw std::runtime_error("MACEPolarCuda: cannot open " + filename);
    const json js = json::parse(f);
    if (js.value("format_version", 0) != 2)
        throw std::runtime_error(filename + " was extracted by an incompatible symmetrix version; extract it again");
    if (js.value("model_type", "") != "PolarMACE")
        throw std::invalid_argument("MACEPolarCuda: " + filename + " is not a PolarMACE model file");

    // Basic model information
    num_elements = js["num_elements"];
    num_channels = js["num_channels"];
    num_edge_channels = js["num_edge_channels"];
    r_cut = js["r_cut"];
    atomic_numbers = js["atomic_numbers"].get<std::vector<int>>();
    const int T = num_elements;
    const int C = num_channels;
    if (js["l_max"] != 3 or js["L_max"] != 1 or js["updates"].size() != 2)
        throw std::invalid_argument("MACEPolarCuda requires l_max = 3, L_max = 1 and two charge updates");

    // Phi1, M0, M1
    {
        const auto Phi1_spec = js["Phi1_spec"].get<std::vector<std::array<int,3>>>();
        const auto Phi1_cg_spec = js["Phi1_cg_spec"].get<std::vector<std::array<int,2>>>();
        const auto cg = js["Phi1_clebsch_gordan"].get<Vec>();
        bool same = num_edge_channels == Ce
                    and js["Phi1_l"].get<std::vector<int>>() == std::vector<int>(Polar::Phi1_l,
                            Polar::Phi1_l + Polar::num_paths)
                    and int(Phi1_cg_spec.size()) == Polar::num_terms and int(cg.size()) == Polar::num_terms;
        for (int t = 0; same and t < Polar::num_terms; ++t) {
            const auto [q, lm1] = Phi1_cg_spec[t];
            const auto [lme, p, lm2] = Phi1_spec.at(q);
            const Term& x = Phi1_terms[t];
            same = x.lme == lme and x.path == p and x.lm1 == lm1 and x.lm2 == lm2 and std::abs(x.cg - cg[t]) < 1e-6;
        }
        for (std::string l : {"0", "1"}) {
            const auto b = js["B" + l + "_spec"].get<std::vector<std::array<int,3>>>();
            const auto c = js["B" + l + "_coefficients"].get<Vec>();
            const auto m = js["M" + l + "_spec"].get<std::vector<std::array<int,3>>>();
            same = same and int(b.size()) == Polar::num_b and int(c.size()) == Polar::num_b
                   and int(m.size()) == Polar::num_m;
            for (int t = 0; same and t < Polar::num_b; ++t)
                same = b[t] == std::array<int,3>{Polar::b[t].q, Polar::b[t].a, Polar::b[t].b}
                       and std::abs(c[t] - Polar::b[t].c) < 1e-6;
            for (int e = 0; same and e < Polar::num_m; ++e)
                same = m[e] == std::array<int,3>{Polar::m[e].o, Polar::m[e].q, Polar::m[e].n};
        }
        if (not same)
            throw std::invalid_argument("MACEPolarCuda supports the tensor products and edge channels of MACE-POLAR-1 "
                                        "only; use symmetrix/mace/polar for this model");
    }

    // Radial functions
    {
        radial_spline_r_min = js["radial_spline_r_min"];
        const auto v0 = js["radial_spline_values_0"].get<std::vector<std::vector<Vec>>>();
        const auto d0 = js["radial_spline_derivs_0"].get<std::vector<std::vector<Vec>>>();
        num_knots = int(v0[0][0].size()) - 1;
        knot_spacing = (r_cut - radial_spline_r_min)/num_knots;
        num_basis_0 = int(v0[0].size());
        const int NB = num_basis_0;
        std::vector<float> table((long)T*T*num_knots*4*NB);
        for (int p = 0; p < T*T; ++p)
            for (int b = 0; b < NB; ++b)
                hermite(v0[p][b], d0[p][b], knot_spacing, NB, b, table.data() + (long)p*num_knots*4*NB);
        radial_0.upload(table);
        int device;
        CHECK(cudaGetDevice(&device));
        int max_shared;
        CHECK(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        cudaFuncAttributes attributes;
        CHECK(cudaFuncGetAttributes(&attributes, k_Phi0));
        const int bytes = 16*NB*sizeof(float);
        Phi0_elements = std::min(T, int((max_shared - attributes.sharedSizeBytes)/bytes));
        if (Phi0_elements < 1)
            throw std::invalid_argument("MACEPolarCuda: too many first-layer radial functions for shared memory");
        CHECK(cudaFuncSetAttribute(k_Phi0, cudaFuncAttributeMaxDynamicSharedMemorySize, Phi0_elements*bytes));
    }
    {
        const auto v1 = js["radial_spline_values_1"].get<std::vector<std::vector<Vec>>>();
        const auto d1 = js["radial_spline_derivs_1"].get<std::vector<std::vector<Vec>>>();
        const auto W = js["radial_weights_1"].get<std::vector<Vec>>();
        const int F = Polar::num_paths*Ce;
        if (int(W[0].size()) != F)
            throw std::invalid_argument("MACEPolarCuda: radial_weights_1 does not match Phi1");
        std::vector<float> table((long)T*T*num_knots*4*F);
        for (int p = 0; p < T*T; ++p)
            for (int x = 0; x < F; ++x) {
                Vec v(num_knots + 1, 0.0);
                Vec d(num_knots + 1, 0.0);
                for (size_t b = 0; b < W.size(); ++b)
                    for (int s = 0; s <= num_knots; ++s) {
                        v[s] += W[b][x]*v1[p][b][s];
                        d[s] += W[b][x]*d1[p][b][s];
                    }
                hermite(v, d, knot_spacing, F, x, table.data() + (long)p*num_knots*4*F);
            }
        if (not float32_mixed)
            radial_1.upload(table);
        else {
            const long B = (long)T*T*num_knots;
            std::vector<float> c01(B*2*F);
            std::vector<__half2> c23(B*F);
            for (long b = 0; b < B; ++b)
                for (long x = 0; x < F; ++x) {
                    const float* c = table.data() + b*4*F + x;
                    if (std::max(std::abs(c[2*F]), std::abs(c[3*F])) > 6e4f)
                        throw std::runtime_error("MACEPolarCuda: second-layer spline coefficients "
                                                 "exceed the fp16 range");
                    c01[b*2*F + x] = c[0];
                    c01[(b*2 + 1)*F + x] = c[F];
                    c23[b*F + x] = __floats2half2_rn(c[2*F], c[3*F]);
                }
            radial_1.upload(c01);
            radial_1_half.upload(c23);
        }
    }
    {
        std::vector<float> table((long)2*T*T*num_knots*4);
        for (int t = 0; t < 2; ++t) {
            const auto v = js["A" + std::to_string(t) + "_spline_values"].get<std::vector<Vec>>();
            const auto d = js["A" + std::to_string(t) + "_spline_derivs"].get<std::vector<Vec>>();
            for (int p = 0; p < T*T; ++p)
                hermite(v[p], d[p], knot_spacing, 1, 0, table.data() + ((long)t*T*T + p)*num_knots*4);
        }
        densities.upload(table);
    }

    // X0
    linear_1_1 = linear_from_json(js["linear_1_1"]);
    mul_X = linear_1_1.out;
    off_X = offsets(mul_X);
    num_X0_rows = float32_mixed ? (T*num_basis_0 + 31)/32*32 : T*num_basis_0;
    auto x0 = floats(js["X0_weights"]);
    const size_t width_X0 = std::accumulate(mul_X.begin(), mul_X.end(), 0);
    bool shapes = mul_X.size() == 4 and x0.size() == width_X0*T*num_basis_0;
    x0.resize(x0.size()/(T*num_basis_0)*num_X0_rows, 0.f);
    X0_weights.upload(x0);
    residual_0.upload(floats(js["residual_0"]));

    // A0, A1
    gate_0 = gate_from_json(js["gate_0"], mul_X);
    gate_1 = gate_from_json(js["gate_1"], mul_X);
    linear_2_0 = linear_from_json(js["linear_2_0"]);
    linear_2_1 = linear_from_json(js["linear_2_1"]);

    // M0, M1
    M0_weights.upload(floats(js["M0_weights"][0]));
    M1_weights.upload(floats(js["M1_weights"][0]));

    // H1, H2
    product_linear_0 = linear_from_json(js["product_linear_0"]);
    product_linear_1 = linear_from_json(js["product_linear_1"]);
    skip_0.upload(floats(js["skip_0"]));
    linear_up_1 = linear_from_json(js["linear_up_1"]);
    residual_1 = linear_from_json(js["residual_1"]);
    skip_1 = linear_from_json(js["skip_1"]);

    // Readouts
    readout_0.in = {C};
    readout_0.out = {1};
    readout_0.off_in = offsets(readout_0.in);
    readout_0.off_out = offsets(readout_0.out);
    readout_0.weights_offset = {0};
    readout_0.weights.upload(floats(js["readout_0"]));
    readout_0.bias.upload({0.f});
    readout_1 = bias_readout_from_json(js["readout_1"]);
    atomic_energies.upload(js["atomic_energies"].get<Vec>());
    const std::vector<int> hidden = {C, C};
    const std::vector<int> edge = {Ce, Ce};
    const std::vector<int> target(4, C);
    int width_Phi1 = 0;
    for (int l = 0; l < int(linear_1_1.in.size()); ++l)
        width_Phi1 += (2*l+1)*linear_1_1.in[l];
    if (not shapes or product_linear_0.in != hidden or product_linear_0.out != hidden or product_linear_1.in != hidden
        or product_linear_1.out != hidden or skip_1.in != hidden or skip_1.out != hidden or linear_up_1.in != hidden
        or linear_up_1.out != edge or residual_1.in != std::vector<int>{Ce, Ce, 0, 0} or linear_2_0.out != target
        or linear_2_1.out != target or floats(js["skip_0"]).size() != size_t(T*C)
        or js["readout_0"].size() != size_t(C) or width_Phi1 != num_lme*Ce)
        throw std::invalid_argument("MACEPolarCuda: " + filename + " has feature shapes other than MACE-POLAR-1's");

    // Charges
    for (int t = 0; t < 2; ++t) {
        sources[t] = linear_from_json(js["sources"][t]);
        mixer[t] = linear_from_json(js["mixer"][t]);
        const auto& j = js["updates"][t];
        auto& u = updates[t];
        u.potential = linear_from_json(j["potential"]);
        u.node = linear_from_json(j["node"]);
        u.charge = linear_from_json(j["charge"]);
        u.source.upload(floats(j["source"]));
        u.dot.num_channels = C;
        u.dot.weights.upload(floats(j["dot"]));
        u.mlp = mlp_from_json(j["mlp"]);
        u.out.num_channels = C;
        u.out.weights.upload(floats(j["out"]));
        u.readout = gated_readout_from_json(j["readout"]);
    }
    fukui = bias_readout_from_json(js["fukui"]);

    // Electron energy
    const auto& j = js["electron"];
    electron.q = linear_from_json(j["q"]);
    electron.v = linear_from_json(j["v"]);
    electron.dot_q.num_channels = C;
    electron.dot_q.weights.upload(floats(j["dot_q"]));
    electron.dot_v.num_channels = C;
    electron.dot_v.weights.upload(floats(j["dot_v"]));
    electron.mlp = mlp_from_json(j["mlp"]);

    // Electrostatics
    kspace_cutoff = js["kspace_cutoff"];
    field_constant = js["field_constant"];
    density_sigma = js["density_basis"]["sigmas"][0];
    density_a = js["density_basis"]["a"][0].get<std::array<double,2>>();
    feature_sigmas = js["feature_basis"]["sigmas"].get<Vec>();
    feature_a = js["feature_basis"]["a"].get<std::vector<std::array<double,2>>>();
    field_self = floats(js["field_self"]);
    field_norms = floats(js["field_norms"]);
    std::vector<float> c = {float(density_sigma)};
    for (double x : feature_sigmas)
        c.push_back(float(x));
    c.insert(c.end(), field_self.begin(), field_self.end());
    c.insert(c.end(), field_norms.begin(), field_norms.end());
    consts.upload(c);
}
