#pragma once

#include "mace_foundation_tables.hpp"

#include "../external/json/single_include/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace mace_foundation {

template <class real> struct Model {
    int T, K, K0;
    double r_cut2;
    real r_min, inv_h;
    const real* basis;
    const real* R1;
    const __half2* R1h;
    unsigned* amax;
    const real *A0_splines, *A1_splines;
    const real *M0_weights, *M1_weights;
    const real *z_weights_for_M0, *z_weights_for_M1;
    const real *readout_1_weights_for_M0, *readout_2_weights_2;
    real readout_2_scale_factor;
    const double* atomic_energies;
    const real* zbl;
    real zbl_c[4];
    int zbl_p;
    bool ordered;
    const real *residual_0, *H1_bias, *z_bias;
};

template <class real> struct Graph {
    int N, NR;
    const int *atom, *elem, *row_of_atom, *elem_all;
    int *count, *start, *seg;
    real* geometry;
    int* neigh_indices;
    int2* row_pair;
    real* edge_forces;
    real *Phi0, *A0, *M0, *H1, *Phi1, *A1, *H1_adj, *A1_adj, *dPhi1, *M0_adj, *A0_adj, *Phi0_adj;
    real *s0, *s1, *dot0, *dot1;
    double* node_energies;
    unsigned* err;
    real *X0, *X1, *Tg, *T_adj, *X_adj, *B1, *B1_adj, *M0s_adj, *M1, *M1_adj;
};

}  // namespace mace_foundation

namespace {

using mace_foundation::Graph;
using mace_foundation::Model;

template <class T> struct Buffer {
    T* p = nullptr;
    size_t n = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    ~Buffer() { cudaFree(p); }
    void reserve(size_t m) {
        if (m <= n and p) return;
        cudaFree(p);
        n = m + m/8 + 1;
        if (cudaMalloc(&p, n*sizeof(T)) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("MACEFoundation: out of device memory");
        }
    }
    void upload(const std::vector<T>& v, cudaStream_t s) {
        reserve(v.size());
        cudaMemcpyAsync(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
    }
    operator T*() const { return p; }
};

template <class real>
struct GemmDesc { const real *A, *B, *W; const __half *Bh, *Bl; real* C; int M, N, K, lda, ldc, tile0, shift,
                 slot; bool trans; };

}  // namespace

template <class Precision, bool mixed>
class MACEFoundationCuda {

public:

struct Weight {
    Buffer<Precision> w, tw;
    Buffer<__half> hi, lo, thi, tlo;
    int shift = 0;
};

MACEFoundationCuda(const std::string& filename, const std::string& prefix);
~MACEFoundationCuda();

// Basic model information
int num_elements;
int num_channels;
int num_channels_node;
double r_cut;
int L_max, num_LM;
int num_lme;
bool gated, ordered;
std::vector<int> atomic_numbers;
Buffer<double> atomic_energies;

// Device
cudaStream_t stream = nullptr;
cublasContext* blas = nullptr;
Buffer<unsigned char> blas_workspace;
bool tensor_cores;
int num_sms;
Buffer<unsigned> amax;
unsigned* err = nullptr;

// Graph
int N = 0;
int NR = 0;
std::vector<std::array<int, 3>> segments;
Buffer<int> atom_of_node, elem, elem_all, row_of_atom, count, start, seg, neigh_indices;
Buffer<int2> row_pair;
Buffer<Precision> geometry, edge_forces;
Buffer<unsigned char> scan_storage;
size_t scan_bytes = 0;
void reserve(long);
Model<Precision> model() const;
Graph<Precision> graph() const;

// Node energies and forces
Buffer<double> node_energies, outputs;
template <class Sh> void compute_first_layer(const Model<Precision>&, const Graph<Precision>&);
template <class Sh> void compute_second_layer(const Model<Precision>&, const Graph<Precision>&);
template <class Sh> void compute_readouts(const Model<Precision>&, const Graph<Precision>&);
template <class Sh> void reverse_second_layer(const Model<Precision>&, const Graph<Precision>&);
template <class Sh> void reverse_first_layer(const Model<Precision>&, const Graph<Precision>&);

// ZBL
bool has_zbl;
Buffer<Precision> zbl;
Precision zbl_c[4];
int zbl_p;

// Radial functions
int K;
double r_min, h;
int NB, K0;
Buffer<Precision> basis;

// A0
Buffer<Precision> Phi0, Phi0_adj, A0, A0_adj;
Weight Q;

// A0 rescaling
Buffer<Precision> A0_splines, s0, dot0;

// M0
Buffer<Precision> M0_weights, M0, M0_adj;

// H1
Weight H1_weights;
Buffer<Precision> H1_bias, H1, H1_adj;

// Phi1
Buffer<Precision> R1;
Buffer<__half2> R1h;
Buffer<Precision> Phi1, dPhi1;

// A1
Weight A1_weights;
std::vector<long> A1_weights_offset;
Buffer<Precision> A1, A1_adj;

// A1 rescaling
Buffer<Precision> A1_splines, s1, dot1;

// M1
Buffer<Precision> M1_weights, M1, M1_adj;

// Gates
Weight linear_2_0, linear_2_1, residual_1_weights;
Buffer<Precision> residual_0, Tg, T_adj, X0, X1, X_adj, B1, B1_adj;

// Readouts
Buffer<Precision> z_weights_for_M0, z_weights_for_M1, z_bias;
Buffer<Precision> readout_1_weights_for_M0, readout_2_weights_2;
Precision readout_2_scale_factor;
Buffer<Precision> M0s_adj;

// Matrix products
std::vector<GemmDesc<Precision>> queue;
void gemm(bool, int, int, int, const Precision*, int, const Weight&, long, Precision*, int, int);
void products();
void set_weight(Weight&, const std::vector<Precision>&, const std::vector<std::array<long, 3>>&);

// Initializer
void load(const std::string&, const std::string&);

};

namespace {

using json = nlohmann::json;

template <class Sh> constexpr int fwd_blocks = Sh::C < 640 ? 640/Sh::C : 1;
template <class Sh> constexpr int rev_blocks = Sh::C < 512 ? 512/Sh::C : 1;
template <class Sh> constexpr int groups = Sh::L_max == 2 ? 3 : 1;
template <class Sh> constexpr int group(int p) { return p/2%groups<Sh>; }

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("MACEFoundation: ") + what + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what)
{
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("MACEFoundation: ") + what + ": cuBLAS error " + std::to_string(s));
}
#define CHECK(x) check((x), #x)

__host__ __device__ constexpr long row(int lm, int node, int N) {
    const int l = lm_l(lm);
    return (long)N*l*l + (long)node*(2*l+1) + lm - l*l;
}
template <class Sh> __device__ constexpr long phi1_row(int lme, int node, int N) {
    constexpr int o[4] = {0, Sh::lme_offset(1), Sh::lme_offset(2), Sh::lme_offset(3)};
    constexpr int ne[4] = {Sh::num_e(0), Sh::num_e(1), Sh::num_e(2), Sh::num_e(3)};
    const int l = lme < o[1] ? 0 : lme < o[2] ? 1 : lme < o[3] ? 2 : 3;
    const int r = lme - o[l];
    return (long)N*o[l] + ((long)node*(2*l+1) + r/ne[l])*ne[l] + r%ne[l];
}
__device__ constexpr int pair_index(int a, int b, int T) {
    return a <= b ? a*(2*T - a - 1)/2 + b : b*(2*T - b - 1)/2 + a;
}

template <class real> struct alignas(4*sizeof(real)) Vec4 { real x, y, z, w; };
template <class real> struct Edge { real x, y, z, r, t; int knot, row, pair; };
template <class real> __device__ __forceinline__ Edge<real> edge(const Model<real>& M, const Graph<real>& G, int e) {
    const Vec4<real> g = *reinterpret_cast<const Vec4<real>*>(G.geometry + 4*e);
    const int2 mt = G.row_pair[e];
    const real t = (g.w - M.r_min)*M.inv_h;
    const int knot = min(max(int(t), 0), M.K - 1);
    return {g.x, g.y, g.z, g.w, t - knot - 0.5f, knot, mt.x, mt.y};
}

template <class real> struct __align__(16) Staged { real S[16]; real x, y, z, rinv, t; int knot, row, pair; };
template <class real>
__device__ __forceinline__ void stage(const Model<real>& M, const Graph<real>& G, int e, Staged<real>& s) {
    const Edge<real> q = edge(M, G, e);
    solid_harmonics(q.x, q.y, q.z, s.S);
    s.x = q.x;
    s.y = q.y;
    s.z = q.z;
    s.rinv = 1/q.r;
    s.t = q.t;
    s.knot = q.knot;
    s.row = q.row;
    s.pair = q.pair;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}
__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}
__device__ __forceinline__ unsigned match_any(int v) {
#if __CUDA_ARCH__ >= 700
    return __match_any_sync(~0u, v);
#else
    unsigned m = 0;
    for (int i = 0; i < 32; ++i) m |= unsigned(__shfl_sync(~0u, v, i) == v) << i;
    return m;
#endif
}
template <class real> __device__ __forceinline__ void warp_add3(real* dst, real f0, real f1, real f2) {
    const int lane = threadIdx.x & 31;
    const bool hi = lane & 16;
    const bool b8 = lane & 8;
    const real k0 = hi ? f2 : f0;
    const real k1 = hi ? 0.f : f1;
    const real s0 = hi ? f0 : f2;
    const real s1 = hi ? f1 : 0.f;
    const real a = k0 + __shfl_xor_sync(~0u, s0, 16);
    const real b = k1 + __shfl_xor_sync(~0u, s1, 16);
    real c = (b8 ? b : a) + __shfl_xor_sync(~0u, b8 ? a : b, 8);
    c += __shfl_xor_sync(~0u, c, 4);
    c += __shfl_xor_sync(~0u, c, 2);
    c += __shfl_xor_sync(~0u, c, 1);
    if ((lane & 7) == 0 and lane < 24) atomicAdd(dst + (lane >> 3), c);
}

template <class real> __device__ __forceinline__ real cubic(const real* b, int num, int f, real x) {
    return b[f] + x*(b[num + f] + x*(b[2*num + f] + x*b[3*num + f]));
}
template <class real> __device__ __forceinline__ void cubic(const real* b, int num, int f, real x, real& v, real& d) {
    const real c1 = b[num + f];
    const real c2 = b[2*num + f];
    const real c3 = b[3*num + f];
    v = b[f] + x*(c1 + x*(c2 + x*c3));
    d = c1 + x*(2*c2 + 3*x*c3);
}

template <class Sh, bool H, class real> struct R1Block {
    static constexpr int F = Sh::num_paths*Sh::C;
    const real* c;
    const __half2* h;
    __device__ R1Block(const Model<real>& M, int pair, int knot) {
        const long b = (long)pair*M.K + knot;
        c = M.R1 + b*(H ? 2 : 4)*F;
        h = H ? M.R1h + b*F : nullptr;
    }
    __device__ real value(int f, real x) const {
        if constexpr (not H) return cubic(c, F, f, x);
        const float2 c23 = __half22float2(h[f]);
        return c[f] + x*(c[F + f] + x*(c23.x + x*c23.y));
    }
    __device__ void value_deriv(int f, real x, real& v, real& d) const {
        if constexpr (not H) { cubic(c, F, f, x, v, d); return; }
        const float2 c23 = __half22float2(h[f]);
        const real c1 = c[F + f];
        v = c[f] + x*(c1 + x*(c23.x + x*c23.y));
        d = c1 + x*(2*c23.x + 3*x*c23.y);
    }
};

template <class real> __device__ __forceinline__ void amax(const Model<real>& M, int slot, float v) {
    if (not M.amax) return;
    v = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if ((threadIdx.x & 31) == 0) atomicMax(M.amax + slot, __float_as_uint(v));
}

constexpr int NUM_T = 27;
__host__ __device__ constexpr int t_index(int l, int d, int mp) {
    return (l == 1 ? 0 : l == 2 ? 3 : 12) + d*(2*l - 1) + mp - (l-1)*(l-1);
}
template <class real> __device__ __forceinline__ void build_T(const real* a, real* T) {
    for (int t = 0; t < NUM_T; ++t) T[t] = 0;
    static_for<num_grad_terms>([&](auto g) {
        constexpr GradTerm gt = grad_terms[g];
        T[t_index(lm_l(gt.lm), gt.d, gt.mp)] += real(gt.c)*a[gt.lm];
    });
}
template <class real> struct __align__(16) StagedD : Staged<real> { real D[num_grad_terms + 1]; };
template <class real> __device__ __forceinline__ void stage_grad(StagedD<real>& s) {
    static_for<num_grad_terms>([&](auto g) {
        constexpr GradTerm t = grad_terms[g];
        s.D[g] = real(t.c)*s.S[t.mp]*s.rinv;
    });
}
template <class real> __device__ __forceinline__ void grad_staged(const real* g, const real* D, real* Gd) {
    static_for<num_grad_terms>([&](auto i) {
        constexpr GradTerm t = grad_terms[i];
        Gd[t.d] += D[i]*g[t.lm];
    });
}
template <class real>
__device__ __forceinline__ void warp_force(real* dst, real a, const real* Gd, real ux, real uy, real uz) {
    const int lane = threadIdx.x & 31;
    const bool hi = lane & 16;
    const bool b8 = lane & 8;
    const real p = (hi ? Gd[2] : a) + __shfl_xor_sync(~0u, hi ? a : Gd[2], 16);
    const real q = (hi ? Gd[0] : Gd[1]) + __shfl_xor_sync(~0u, hi ? Gd[1] : Gd[0], 16);
    real c = (b8 ? q : p) + __shfl_xor_sync(~0u, b8 ? p : q, 8);
    c += __shfl_xor_sync(~0u, c, 4);
    c += __shfl_xor_sync(~0u, c, 2);
    c += __shfl_xor_sync(~0u, c, 1);
    const real A = __shfl_sync(~0u, c, 0);
    const real gx = __shfl_sync(~0u, c, 8);
    const real gy = __shfl_sync(~0u, c, 16);
    const real gz = __shfl_sync(~0u, c, 24);
    const real ug = ux*gx + uy*gy + uz*gz;
    if (lane < 3) {
        const real u = lane == 0 ? ux : lane == 1 ? uy : uz;
        const real g = lane == 0 ? gx : lane == 1 ? gy : gz;
        atomicAdd(dst + lane, -(A*u + g - u*ug));
    }
}
template <class real>
__device__ __forceinline__ void edge_force(real dr, real ux, real uy, real uz, const real* Gd, real* f) {
    f[0] = -(dr*ux + Gd[1]);
    f[1] = -(dr*uy + Gd[2]);
    f[2] = -(dr*uz + Gd[0]);
}

template <class Sh, int PG, class real>
__device__ __forceinline__ void tp_forward(real* acc, const real* R, const real* S, const real* h) {
    static_for<Sh::num_terms>([&](auto q) {
        constexpr Term t = Sh::terms[q];
        constexpr real cg = real(t.cg*sh_norm(t.lm1));
        if constexpr (group<Sh>(t.path) == PG) acc[t.lme] += (cg*S[t.lm1])*(R[t.path]*h[t.lm2]);
    });
}
template <class Sh> constexpr std::array<int, Sh::num_lme> lme_groups() {
    std::array<int, Sh::num_lme> g{};
    for (const Term& t : Sh::terms) g[t.lme] = group<Sh>(t.path);
    return g;
}
template <class Sh> constexpr bool group_nonempty(int p, int lm2) {
    for (const Term& t : Sh::terms) if (t.path == p and t.lm2 == lm2) return true;
    return false;
}
template <class Sh, int Q, class real> __device__ __forceinline__ real key_w(const real* adj) {
    constexpr Term t = Sh::terms[Q];
    constexpr real cg = real(t.cg*sh_norm(t.lm1));
    constexpr int n = next_term<Sh>(Q);
    if constexpr (n < 0) return cg*adj[t.lme];
    else return cg*adj[t.lme] + key_w<Sh, n>(adj);
}
template <class Sh, int PG, class real>
__device__ __forceinline__ void tp_reverse(const real* adj, const real* R, const real* S, const real* h, real* g_S,
                                           real* g_h, real* hs) {
    static_for<Sh::num_paths*Sh::num_LM>([&](auto g) {
        constexpr int P = Sh::num_paths - 1 - g/Sh::num_LM;
        constexpr int LM2 = g%Sh::num_LM;
        if constexpr (group_nonempty<Sh>(P, LM2) and group<Sh>(P) == PG) {
            const real rh = R[P]*h[LM2];
            real s = 0;
            static_for<Sh::num_terms>([&](auto q) {
                constexpr Term t = Sh::terms[q];
                if constexpr (t.path == P and t.lm2 == LM2 and first_term<Sh>(q)) {
                    const real w = key_w<Sh, q>(adj);
                    s += w*S[t.lm1];
                    g_S[t.lm1] += w*rh;
                }
            });
            g_h[LM2] += R[P]*s;
            hs[P] += h[LM2]*s;
        }
    });
}

template <class T, class real> __device__ __forceinline__ void tree_eval(const real* w, real* v, real* out) {
    for (int i = num_lm; i < T::num_values; ++i) v[i] = 0;
    static_for<T::num_b>([&](auto e) {
        constexpr BTerm t = T::b[e];
        v[t.q] += real(t.c)*v[t.a]*v[t.b];
    });
    static_for<T::num_m>([&](auto e) {
        constexpr MTerm t = T::m[e];
        for (int j = 0; j < t.n; ++j) out[t.o + j] += w[e]*v[t.q + j];
    });
}
template <class T, class real>
__device__ __forceinline__ void tree_adjoint(const real* w, const real* v, const real* g, real* adjx) {
    real gv[T::num_values] = {};
    static_for<T::num_m>([&](auto e) {
        constexpr MTerm t = T::m[e];
        for (int j = 0; j < t.n; ++j) gv[t.q + j] += w[e]*g[t.o + j];
    });
    static_for<T::num_b>([&](auto e) {
        constexpr BTerm t = T::b[T::num_b - 1 - e];
        const real s = real(t.c)*gv[t.q];
        gv[t.a] += s*v[t.b];
        gv[t.b] += s*v[t.a];
    });
    for (int i = 0; i < num_lm; ++i) adjx[i] = gv[i];
}

constexpr int WARPS = 4;

template <class real> __global__ void k_scale1(Model<real> M, Graph<real> G) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    real s1 = 0;
    if (M.A1_splines)
        for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
            const double t = (G.geometry[4*e+3] - M.r_min)*M.inv_h;
            const int knot = min(max((int)t, 0), M.K - 1);
            s1 += cubic(M.A1_splines + ((long)G.row_pair[e].y*M.K + knot)*4, 1, 0, real(t - knot - 0.5));
        }
    s1 = warp_sum(s1);
    if (lane == 0) G.s1[n] = 1 + s1;
}

template <class real> __global__ void k_zbl(Model<real> M, Graph<real> G) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    const int T = M.T;
    const int a = G.elem[n];
    const int* seg = G.seg + n*(T + 1);
    double ez = 0;
    for (int b = 0; b < T; ++b) {
        const real* z = M.zbl ? M.zbl + 3*pair_index(a, b, T) : nullptr;
        for (int e = seg[b] + lane; e < seg[b + 1]; e += 32) {
            const Vec4<real> g = *reinterpret_cast<const Vec4<real>*>(G.geometry + 4*e);
            const real rf = g.w;
            real fz = 0;
            if (z and rf < z[1]) {
                const real ce[4] = {-3.2, -0.9423, -0.4028, -0.2016}, y = rf/z[0];
                real phi = 0;
                real dphi = 0;
                for (int q = 0; q < 4; ++q) {
                    const real v = M.zbl_c[q]*exp(ce[q]*y);
                    phi += v;
                    dphi += ce[q]*v/z[0];
                }
                const real v = z[2]*phi/rf;
                const real dv = z[2]*(rf*dphi - phi)/(rf*rf);
                const real c = real(M.zbl_p);
                const real u = rf/z[1];
                const real up = pow(u, c - 1);
                const real env = 1 - 0.5f*(c+1)*(c+2)*up*u + c*(c+2)*up*u*u - 0.5f*c*(c+1)*up*u*u*u;
                const real denv = (-0.5f*(c+1)*(c+2)*c*up + c*(c+2)*(c+1)*up*u - 0.5f*c*(c+1)*(c+2)*up*u*u)/z[1];
                ez += 0.5*v*env;
                fz = -0.5f*(v*denv + dv*env);
            }
            G.edge_forces[3*e] = fz*g.y;
            G.edge_forces[3*e+1] = fz*g.z;
            G.edge_forces[3*e+2] = fz*g.x;
        }
    }
    ez = warp_sum(ez);
    if (lane == 0) G.node_energies[n] = ez;
}

template <int NB, class real>
__global__ void __launch_bounds__(NB, (NB == 96 ? 6 : 16)*4/sizeof(real)) k_Phi0(Model<real> M, Graph<real> G) {
    constexpr int CB = NB/32;
    __shared__ Staged<real> st[CB][32];
    const int w = threadIdx.x >> 5;
    const int n = CB*blockIdx.x + w;
    const int lane = threadIdx.x & 31;
    const int T = M.T;
    if (n >= G.N) return;
    const int a = G.elem[n];
    float m = 0;
    for (int b = blockIdx.y; b < T; b += gridDim.y) {
        real acc[CB][16] = {};
        const real* table = M.basis + (long)(a*T + b)*M.K*4*NB;
        const int e1 = G.seg[n*(T + 1) + b + 1];
        for (int e0 = G.seg[n*(T + 1) + b]; e0 < e1; e0 += 32) {
            __syncwarp();
            if (e0 + lane < e1) stage(M, G, e0 + lane, st[w][lane]);
            __syncwarp();
            for (int e = e0; e < min(e0 + 32, e1); ++e) {
                const Staged<real>& q = st[w][e - e0];
                const real* blk = table + (long)q.knot*4*NB;
                for (int c = 0; c < CB; ++c) {
                    const real beta = cubic(blk, NB, lane + 32*c, q.t);
                    for (int lm = 0; lm < 16; ++lm) acc[c][lm] += beta*q.S[lm];
                }
            }
        }
        for (int c = 0; c < CB; ++c)
            for (int lm = 0; lm < 16; ++lm) {
                const real v = real(sh_norm(lm))*acc[c][lm];
                G.Phi0[row(lm, n, G.N)*M.K0 + b*NB + lane + 32*c] = v;
                m = fmaxf(m, fabsf(v));
            }
    }
    amax(M, 0, m);
}

template <int NB, class real>
__global__ void __launch_bounds__(2*NB, (NB == 32 ? 12 : NB == 96 ? 5 : 8)*4/sizeof(real)) k_Phi0_reverse(Model<real> M,
        Graph<real> G) {
    __shared__ Staged<real> st[32];
    const int n = blockIdx.x;
    const int g = threadIdx.x/NB;
    const int f = threadIdx.x%NB;
    const int T = M.T;
    const int a = G.elem[n];
    for (int b = blockIdx.y; b < T; b += gridDim.y) {
        const int e1 = G.seg[n*(T + 1) + b + 1];
        real adj[16], Tg[NUM_T];
        for (int lm = 0; lm < 16; ++lm) adj[lm] = G.Phi0_adj[row(lm, n, G.N)*M.K0 + b*NB + f]*real(sh_norm(lm));
        build_T(adj, Tg);
        const real* table = M.basis + (long)(a*T + b)*M.K*4*NB;
        for (int e0 = G.seg[n*(T + 1) + b]; e0 < e1; e0 += 32) {
            __syncthreads();
            if (threadIdx.x < 32 and e0 + threadIdx.x < e1) stage(M, G, e0 + threadIdx.x, st[threadIdx.x]);
            __syncthreads();
            for (int e = e0 + g; e < min(e0 + 32, e1); e += 2) {
                const Staged<real>& q = st[e - e0];
                real beta, beta_d, S[16];
                for (int lm = 0; lm < 16; ++lm) S[lm] = q.S[lm];
                cubic(table + (long)q.knot*4*NB, NB, f, q.t, beta, beta_d);
                beta_d *= M.inv_h;
                real vl[4] = {};
                for (int lm = 0; lm < 16; ++lm) vl[lm_l(lm)] += adj[lm]*S[lm];
                real Gd[3] = {};
                for (int d = 0; d < 3; ++d) {
                    real s = 0;
                    for (int l = 1; l <= 3; ++l)
                        for (int mp = (l-1)*(l-1); mp < l*l; ++mp) s += Tg[t_index(l, d, mp)]*S[mp];
                    Gd[d] = beta*q.rinv*s;
                }
                const real dr = beta_d*(vl[0] + vl[1] + vl[2] + vl[3]) - beta*(vl[1] + 2*vl[2] + 3*vl[3])*q.rinv;
                real fv[3];
                edge_force(dr, q.y, q.z, q.x, Gd, fv);
                warp_add3(G.edge_forces + 3*e, fv[0], fv[1], fv[2]);
            }
        }
    }
}

template <class Sh, int LAYER> using MTables = std::conditional_t<LAYER == 0, Sh, Tables<0>>;
template <class Sh, int LAYER, class real> __global__ void __launch_bounds__(Sh::CN) k_M(Model<real> M, Graph<real> G) {
    using T = MTables<Sh, LAYER>;
    constexpr int C = Sh::CN;
    constexpr int NO = LAYER == 0 ? Sh::num_LM : 1;
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    const real inv = Sh::gated ? 1.f : 1/(LAYER == 0 ? G.s0 : G.s1)[n];
    const real* x = Sh::gated ? (LAYER == 0 ? G.X0 : G.X1) : (LAYER == 0 ? G.A0 : G.A1);
    const real* tree = LAYER == 0 ? M.M0_weights : M.M1_weights;
    real v[T::num_values], w[T::num_m];
    real out[NO] = {};
    for (int lm = 0; lm < num_lm; ++lm) v[lm] = x[row(lm, n, G.N)*C + k]*inv;
    for (int e = 0; e < T::num_m; ++e) w[e] = tree[((long)G.elem[n]*T::num_m + e)*C + k];
    tree_eval<T>(w, v, out);
    if constexpr (LAYER == 1) {
        G.M1[(long)n*C + k] = out[0];
        return;
    }
    float m = 0;
    for (int LM = 0; LM < NO; ++LM) {
        G.M0[row(LM, n, G.N)*C + k] = out[LM];
        m = fmaxf(m, fabsf(out[LM]));
    }
    amax(M, 1, m);
}

template <class Sh, int LAYER, class real>
__global__ void __launch_bounds__(Sh::CN) k_M_reverse(Model<real> M, Graph<real> G) {
    using T = MTables<Sh, LAYER>;
    constexpr int C = Sh::CN;
    constexpr int NO = LAYER == 0 ? Sh::num_LM : 1;
    __shared__ real part[C/32];
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    const real inv = Sh::gated ? 1.f : 1/(LAYER == 0 ? G.s0 : G.s1)[n];
    const real* x = Sh::gated ? (LAYER == 0 ? G.X0 : G.X1) : (LAYER == 0 ? G.A0 : G.A1);
    const real* tree = LAYER == 0 ? M.M0_weights : M.M1_weights;
    real v[T::num_values], w[T::num_m], g[NO], adjx[num_lm];
    real out[NO] = {};
    for (int lm = 0; lm < num_lm; ++lm) v[lm] = x[row(lm, n, G.N)*C + k]*inv;
    for (int e = 0; e < T::num_m; ++e) w[e] = tree[((long)G.elem[n]*T::num_m + e)*C + k];
    if constexpr (LAYER == 0) {
        for (int LM = 0; LM < NO; ++LM) g[LM] = G.M0_adj[row(LM, n, G.N)*C + k];
        g[0] += G.M0s_adj[(long)n*C + k];
    } else {
        g[0] = G.M1_adj[(long)n*C + k];
    }
    tree_eval<T>(w, v, out);
    tree_adjoint<T>(w, v, g, adjx);
    if constexpr (Sh::gated) {
        float m = 0;
        for (int lm = 0; lm < num_lm; ++lm) {
            G.X_adj[row(lm, n, G.N)*C + k] = adjx[lm];
            m = fmaxf(m, fabsf(adjx[lm]));
        }
        amax(M, LAYER == 0 ? 10 : 9, m);
        return;
    }
    real* A_adj = LAYER == 0 ? G.A0_adj : G.A1_adj;
    real dot = 0;
    float m = 0;
    for (int lm = 0; lm < num_lm; ++lm) {
        A_adj[row(lm, n, G.N)*C + k] = adjx[lm]*inv;
        dot += v[lm]*adjx[lm];
        m = fmaxf(m, fabsf(adjx[lm]*inv));
    }
    amax(M, LAYER == 0 ? 5 : 3, m);
    dot = warp_sum(dot);
    if ((k & 31) == 0) part[k >> 5] = dot;
    __syncthreads();
    if (k == 0) {
        real d = 0;
        for (int q = 0; q < C/32; ++q) d += part[q];
        (LAYER == 0 ? G.dot0 : G.dot1)[n] = d*inv;
    }
}

template <class Sh, bool H, class real>
__device__ __forceinline__ void radials(const R1Block<Sh, H, real>& b, int k, real x, real* R) {
    static_for<Sh::num_paths>([&](auto p) { R[p] = b.value(p*Sh::C + k, x); });
}
template <class Sh, bool H, class real>
__device__ __forceinline__ void radials(const R1Block<Sh, H, real>& b, int k, real x, real* R, real* Rd) {
    static_for<Sh::num_paths>([&](auto p) { b.value_deriv(p*Sh::C + k, x, R[p], Rd[p]); });
}
template <class Sh, int PG, class real> __device__ __forceinline__ real radial_sum(const real* Rd, const real* hs) {
    real s = 0.f;
    static_for<Sh::num_paths>([&](auto p) { s = s + (group<Sh>(p) == PG ? Rd[p]*hs[p] : 0.f); });
    return s;
}

template <class Sh, bool H, int PG, class real>
__device__ __forceinline__ void phi1(const Model<real>& M, const Graph<real>& G, Staged<real>* st) {
    constexpr int C = Sh::C;
    constexpr auto lg = lme_groups<Sh>();
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    const int first = G.start[n];
    const int len = G.start[n+1] - first;
    const int end = first + len*int(blockIdx.z + 1)/int(gridDim.z);
    real acc[Sh::num_lme] = {};
    for (int e0 = first + len*int(blockIdx.z)/int(gridDim.z); e0 < end; e0 += C) {
        __syncthreads();
        if (e0 + k < end) stage(M, G, e0 + k, st[k]);
        __syncthreads();
        for (int e = e0; e < min(e0 + C, end); ++e) {
            const Staged<real>& q = st[e - e0];
            real h[Sh::num_LM], R[Sh::num_paths], S[16];
            for (int lm = 0; lm < 16; ++lm) S[lm] = q.S[lm];
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) h[lm2] = G.H1[row(lm2, q.row, G.NR)*C + k];
            const R1Block<Sh, H, real> blk(M, q.pair, q.knot);
            radials(blk, k, q.t, R);
            tp_forward<Sh, PG>(acc, R, S, h);
        }
    }
    float m = 0;
#pragma unroll
    for (int lme = 0; lme < Sh::num_lme; ++lme)
        if (lg[lme] == PG) {
            if (gridDim.z == 1) G.Phi1[phi1_row<Sh>(lme, n, G.N)*C + k] = acc[lme];
            else atomicAdd(G.Phi1 + phi1_row<Sh>(lme, n, G.N)*C + k, acc[lme]);
            m = fmaxf(m, fabsf(acc[lme]));
        }
    amax(M, 2, m*gridDim.z);
}
template <class Sh, bool H, class real>
__global__ void __launch_bounds__(Sh::C, fwd_blocks<Sh>*4/sizeof(real)) k_Phi1(Model<real> M, Graph<real> G) {
    __shared__ Staged<real> st[Sh::C];
    static_for<groups<Sh>>([&](auto pg) { if (blockIdx.y == pg) phi1<Sh, H, pg>(M, G, st); });
}

template <class real> constexpr int phi1_chunk(int C) { return std::min(C, sizeof(real) == 4 ? 128 : 64); }
template <class Sh, bool H, int PG, class real>
__device__ __forceinline__ void phi1_reverse(const Model<real>& M, const Graph<real>& G, StagedD<real>* st) {
    constexpr int C = Sh::C;
    constexpr int CH = phi1_chunk<real>(C);
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    const int first = G.start[n];
    const int len = G.start[n+1] - first;
    const int end = first + len*int(blockIdx.z + 1)/int(gridDim.z);
    constexpr auto lg = lme_groups<Sh>();
    real adj[Sh::num_lme];
#pragma unroll
    for (int lme = 0; lme < Sh::num_lme; ++lme)
        adj[lme] = lg[lme] == PG ? G.dPhi1[phi1_row<Sh>(lme, n, G.N)*C + k] : 0.f;
    for (int e0 = first + len*int(blockIdx.z)/int(gridDim.z); e0 < end; e0 += CH) {
        __syncthreads();
        if (k < CH and e0 + k < end) {
            stage(M, G, e0 + k, st[k]);
            stage_grad(st[k]);
        }
        __syncthreads();
        for (int e = e0; e < min(e0 + CH, end); ++e) {
            const StagedD<real>& q = st[e - e0];
            real h[Sh::num_LM], R[Sh::num_paths], Rd[Sh::num_paths], S[16];
            real g_S[16] = {};
            real g_h[Sh::num_LM] = {};
            real hs[Sh::num_paths] = {};
            real Gd[3] = {};
            S[0] = 1;
            for (int lm = 1; lm < 16; ++lm) S[lm] = q.S[lm];
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) h[lm2] = G.H1[row(lm2, q.row, G.NR)*C + k];
            const R1Block<Sh, H, real> blk(M, q.pair, q.knot);
            radials(blk, k, q.t, R, Rd);
            tp_reverse<Sh, PG>(adj, R, S, h, g_S, g_h, hs);
            const real a = radial_sum<Sh, PG>(Rd, hs)*M.inv_h;
            grad_staged(g_S, q.D, Gd);
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) atomicAdd(G.H1_adj + row(lm2, q.row, G.NR)*C + k, g_h[lm2]);
            warp_force(G.edge_forces + 3*e, a, Gd, q.y, q.z, q.x);
        }
    }
}
template <class Sh, bool H, class real>
__global__ void __launch_bounds__(Sh::C, rev_blocks<Sh>*4/sizeof(real)) k_Phi1_reverse(Model<real> M, Graph<real> G) {
    __shared__ StagedD<real> st[phi1_chunk<real>(Sh::C)];
    static_for<groups<Sh>>([&](auto pg) { if (blockIdx.y == pg) phi1_reverse<Sh, H, pg>(M, G, st); });
}

template <class Sh, class real> __global__ void __launch_bounds__(Sh::CN) k_readouts(Model<real> M, Graph<real> G) {
    constexpr int C = Sh::CN;
    __shared__ real part[17][C/32], gz[16];
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    const int t = G.elem[n];
    const int lane = k & 31;
    const int warp = k >> 5;
    const real h1 = G.M0[(long)n*C + k];
    const real m1 = G.M1[(long)n*C + k];
    real f1[16], f2[16];
    for (int h = 0; h < 16; ++h) {
        f1[h] = M.z_weights_for_M0[((long)t*16 + h)*C + k];
        f2[h] = M.z_weights_for_M1[h*C + k];
    }
    for (int h = 0; h <= 16; ++h) {
        const real v = warp_sum(h < 16 ? h1*f1[h] + m1*f2[h] : M.readout_1_weights_for_M0[k]*h1);
        if (lane == 0) part[h][warp] = v;
    }
    __syncthreads();
    if (k < 16) {
        real z = M.z_bias[t*16 + k];
        for (int q = 0; q < C/32; ++q) z += part[k][q];
        const real sg = 1/(1 + exp(-z));
        part[k][0] = M.readout_2_scale_factor*z*sg;
        gz[k] = M.readout_2_weights_2[k]*M.readout_2_scale_factor*sg*(1 + z*(1 - sg));
    }
    __syncthreads();
    if (k == 0) {
        double E = M.atomic_energies[t];
        for (int q = 0; q < C/32; ++q) E += part[16][q];
        for (int h = 0; h < 16; ++h) E += (double)M.readout_2_weights_2[h]*part[h][0];
        G.node_energies[n] += E;
    }
    real aH = M.readout_1_weights_for_M0[k];
    real aM = 0;
    for (int h = 0; h < 16; ++h) {
        aH += gz[h]*f1[h];
        aM += gz[h]*f2[h];
    }
    G.M0s_adj[(long)n*C + k] = aH;
    G.M1_adj[(long)n*C + k] = aM;
}

template <class Sh> __device__ __forceinline__ long a_index(int r, int n, int N) {
    constexpr long C = Sh::CN;
    return r == 0 ? n*4*C : r < num_lm ? N*4*C + (row(r, n, N) - N)*C : n*4*C + (r - 15)*C;
}
template <class real> __device__ __forceinline__ real sigmoid(real x) { return 1/(1 + exp(-x)); }
template <class Sh, int LAYER, class real>
__device__ __forceinline__ void gate_inputs(const Model<real>& M, const Graph<real>& G, int n, int k, real inv,
                                            real* a, real* b) {
    constexpr int C = Sh::CN;
    const real* A = LAYER ? G.A1 : G.A0;
    for (int r = 0; r < num_lm + 3; ++r) b[r] = a[r] = A[a_index<Sh>(r, n, G.N) + k]*inv;
    for (int q = 0; q < 4; ++q)
        b[q == 0 ? 0 : 15 + q] += LAYER ? G.B1[((long)n*4 + q)*C + k] : M.residual_0[((long)G.elem[n]*4 + q)*C + k];
    if (LAYER) for (int lm = 1; lm < Sh::num_LM; ++lm) b[lm] += G.B1[(long)G.N*4*C + (row(lm, n, G.N) - G.N)*C + k];
}
template <class Sh, int LAYER, class real>
__global__ void __launch_bounds__(Sh::CN) k_gate(Model<real> M, Graph<real> G) {
    constexpr int C = Sh::CN;
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    real a[num_lm + 3], b[num_lm + 3];
    gate_inputs<Sh, LAYER>(M, G, n, k, 1/(LAYER ? G.s1[n] : G.s0[n]), a, b);
    float m = fabsf(b[0]*sigmoid(b[0]));
    G.Tg[(long)n*C + k] = b[0]*sigmoid(b[0]);
    for (int l = 1; l < 4; ++l) {
        const real s = sigmoid(b[15 + l]);
        for (int lm = l*l; lm < (l + 1)*(l + 1); ++lm) {
            G.Tg[row(lm, n, G.N)*C + k] = b[lm]*s;
            m = fmaxf(m, fabsf(b[lm]*s));
        }
    }
    amax(M, 6 + LAYER, m);
}
template <class Sh, int LAYER, class real>
__global__ void __launch_bounds__(Sh::CN) k_gate_reverse(Model<real> M, Graph<real> G) {
    constexpr int C = Sh::CN;
    __shared__ real part[C/32];
    const int n = blockIdx.x;
    const int k = threadIdx.x;
    const real inv = 1/(LAYER ? G.s1[n] : G.s0[n]);
    real a[num_lm + 3], b[num_lm + 3], ba[num_lm + 3];
    gate_inputs<Sh, LAYER>(M, G, n, k, inv, a, b);
    const real s0 = sigmoid(b[0]);
    ba[0] = G.T_adj[(long)n*C + k]*s0*(1 + b[0]*(1 - s0));
    for (int l = 1; l < 4; ++l) {
        const real s = sigmoid(b[15 + l]);
        real ga = 0;
        for (int lm = l*l; lm < (l + 1)*(l + 1); ++lm) {
            const real ta = G.T_adj[row(lm, n, G.N)*C + k];
            ba[lm] = ta*s;
            ga += ta*b[lm];
        }
        ba[15 + l] = ga*s*(1 - s);
    }
    real* A_adj = LAYER ? G.A1_adj : G.A0_adj;
    real dot = 0;
    float m = 0;
    for (int r = 0; r < num_lm + 3; ++r) {
        A_adj[a_index<Sh>(r, n, G.N) + k] = ba[r]*inv;
        dot += a[r]*ba[r];
        m = fmaxf(m, fabsf(ba[r]));
    }
    amax(M, LAYER ? 3 : 5, m*inv);
    if (LAYER) amax(M, 11, m);
    if (LAYER) {
        for (int q = 0; q < 4; ++q) G.B1_adj[((long)n*4 + q)*C + k] = ba[q == 0 ? 0 : 15 + q];
        for (int lm = 1; lm < Sh::num_LM; ++lm) G.B1_adj[(long)G.N*4*C + (row(lm, n, G.N) - G.N)*C + k] = ba[lm];
    }
    dot = warp_sum(dot);
    if ((k & 31) == 0) part[k >> 5] = dot;
    __syncthreads();
    if (k == 0) {
        real d = 0;
        for (int q = 0; q < C/32; ++q) d += part[q];
        (LAYER ? G.dot1 : G.dot0)[n] = d*inv;
    }
}
template <class Sh, class real> __global__ void k_h1_bias(Model<real> M, Graph<real> G) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (long)G.N*Sh::C) G.H1[i] += M.H1_bias[(long)G.elem[i/Sh::C]*Sh::C + i%Sh::C];
}

template <class real> __global__ void k_outputs(Graph<real> G, double* eatom, double* out) {
    __shared__ double part[7][WARPS];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    double v[7] = {};
    if (n < G.N) {
        if (lane == 0) {
            v[0] = G.node_energies[n];
            if (eatom) eatom[G.atom[n]] = v[0];
        }
        for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
            const real* g = G.geometry + 4*e;
            const double x = g[1]*g[3];
            const double y = g[2]*g[3];
            const double z = g[0]*g[3];
            const double fx = G.edge_forces[3*e];
            const double fy = G.edge_forces[3*e+1];
            const double fz = G.edge_forces[3*e+2];
            v[1] += x*fx;
            v[2] += y*fy;
            v[3] += z*fz;
            v[4] += 0.5*(x*fy + y*fx);
            v[5] += 0.5*(x*fz + z*fx);
            v[6] += 0.5*(y*fz + z*fy);
        }
    }
    for (int c = 0; c < 7; ++c) {
        const double s = warp_sum(v[c]);
        if (lane == 0) part[c][warp] = s;
    }
    __syncthreads();
    if (threadIdx.x < 7) {
        double s = 0;
        for (int q = 0; q < WARPS; ++q) s += part[threadIdx.x][q];
        atomicAdd(out + threadIdx.x, s);
    }
}

template <class real> __global__ void k_amax(Model<real> M, const real* x, long n, int slot) {
    __shared__ float part[32];
    float v = 0;
    for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x*blockDim.x)
        v = fmaxf(v, fabsf(x[i]));
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < int(blockDim.x >> 5); ++w) v = fmaxf(v, part[w]);
        atomicMax(M.amax + slot, __float_as_uint(v));
    }
}

struct GemmBatch { GemmDesc<float> g[32]; int n; };

__host__ __device__ inline int scale_shift(float a) {
    if (not (a > 0)) return 0;
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
    const __half hx = __float2half_rn(v.x*scale);
    const __half hy = __float2half_rn(v.y*scale);
    const __half2 h = __halves2half2(hx, hy);
    const __half2 l = __halves2half2(__float2half_rn(v.x*scale - __half2float(hx)),
                                     __float2half_rn(v.y*scale - __half2float(hy)));
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
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

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(32*WM*WN, 2) k_gemm(GemmBatch batch, const unsigned* amax) {
#ifdef TENSOR_CORES
    constexpr int NT = 32*WM*WN;
    constexpr int TM = BM/WM/16;
    constexpr int TN = BN/WN/8;
    extern __shared__ __align__(16) unsigned char smem[];
    auto As = reinterpret_cast<float (*)[BM][SA]>(smem);
    auto Bh = reinterpret_cast<__half (*)[BN][SB]>(smem + sizeof(float)*3*BM*SA);
    auto Bl = reinterpret_cast<__half (*)[BN][SB]>(smem + sizeof(float)*3*BM*SA + sizeof(__half)*3*BN*SB);
    int gi = 0;
    while (gi + 1 < batch.n and batch.g[gi + 1].tile0 <= int(blockIdx.x)) ++gi;
    const GemmDesc<float> d = batch.g[gi];
    const int tile = blockIdx.x - d.tile0;
    const int m0 = tile/(d.N/BN)*BM;
    const int n0 = tile%(d.N/BN)*BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp/WN;
    const int wn = warp%WN;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int shift = scale_shift(__uint_as_float(amax[d.slot]));
    const float scale = ldexpf(1.f, shift);
    const float unscale = ldexpf(1.f, -(shift + d.shift));
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
        for (int i = 0; i < TM; ++i) {
            const int r = wm*(BM/WM) + i*16 + gid;
            split(*reinterpret_cast<const float2*>(&As[st][r][2*tig]), scale, ah[i][0], al[i][0]);
            split(*reinterpret_cast<const float2*>(&As[st][r + 8][2*tig]), scale, ah[i][1], al[i][1]);
            split(*reinterpret_cast<const float2*>(&As[st][r][2*tig + 8]), scale, ah[i][2], al[i][2]);
            split(*reinterpret_cast<const float2*>(&As[st][r + 8][2*tig + 8]), scale, ah[i][3], al[i][3]);
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
                for (int q = 0; q < 4; ++q) acc[i][j][q] += part[q];
            }
    }
    asm volatile("cp.async.wait_group 0;");
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            const int r = m0 + wm*(BM/WM) + i*16 + gid;
            const int c = n0 + wn*(BN/WN) + j*8 + 2*tig;
            if (r < d.M)
                *reinterpret_cast<float2*>(d.C + (size_t)r*d.ldc + c) = make_float2(acc[i][j][0]*unscale,
                        acc[i][j][1]*unscale);
            if (r + 8 < d.M)
                *reinterpret_cast<float2*>(d.C + (size_t)(r + 8)*d.ldc + c) = make_float2(acc[i][j][2]*unscale,
                        acc[i][j][3]*unscale);
        }
#endif
}

template <int BM, int BN, int WM, int WN> void gemm_batch(GemmBatch b, const unsigned* amax, cudaStream_t s) {
    int tiles = 0;
    for (int i = 0; i < b.n; ++i) {
        b.g[i].tile0 = tiles;
        tiles += (b.g[i].M + BM - 1)/BM*(b.g[i].N/BN);
    }
    if (not tiles) return;
    k_gemm<BM, BN, WM, WN><<<tiles, 32*WM*WN, gemm_smem<BM, BN>, s>>>(b, amax);
}

template <int BM, int BN> __global__ void __launch_bounds__(BM*BN/16) k_sgemm(GemmBatch batch) {
    constexpr int NT = BM*BN/16;
    constexpr int TX = BN/4;
    constexpr int LA = BM*4/NT;
    constexpr int LB = BN*4/NT;
    __shared__ __align__(16) float As[2][16][BM + 4];
    __shared__ __align__(16) float Bs[2][16][BN];
    int gi = 0;
    while (gi + 1 < batch.n and batch.g[gi + 1].tile0 <= int(blockIdx.x)) ++gi;
    const GemmDesc<float> d = batch.g[gi];
    const int tile = blockIdx.x - d.tile0;
    const int m0 = tile/(d.N/BN)*BM;
    const int n0 = tile%(d.N/BN)*BN;
    const int tx = threadIdx.x%TX;
    const int ty = threadIdx.x/TX;
    float4 ra[LA], rb[LB];
    auto fetch = [&](int k0) {
        for (int i = 0; i < LA; ++i) {
            const int c = threadIdx.x + i*NT;
            const int r = m0 + c/4;
            ra[i] = r < d.M ? *reinterpret_cast<const float4*>(d.A + (size_t)r*d.lda + k0 + c%4*4) : make_float4(0,
                    0, 0, 0);
        }
        for (int i = 0; i < LB; ++i) {
            const int c = threadIdx.x + i*NT;
            rb[i] = *reinterpret_cast<const float4*>(d.B + (size_t)(k0 + c/TX)*d.N + n0 + c%TX*4);
        }
    };
    float acc[4][4] = {};
    fetch(0);
    for (int k0 = 0; k0 < d.K; k0 += 16) {
        const int s = k0/16%2;
        for (int i = 0; i < LA; ++i) {
            const int c = threadIdx.x + i*NT;
            const int r = c/4;
            const int k = c%4*4;
            As[s][k][r] = ra[i].x;
            As[s][k + 1][r] = ra[i].y;
            As[s][k + 2][r] = ra[i].z;
            As[s][k + 3][r] = ra[i].w;
        }
        for (int i = 0; i < LB; ++i) {
            const int c = threadIdx.x + i*NT;
            *reinterpret_cast<float4*>(&Bs[s][c/TX][c%TX*4]) = rb[i];
        }
        __syncthreads();
        if (k0 + 16 < d.K) fetch(k0 + 16);
        for (int k = 0; k < 16; ++k) {
            const float4 a = *reinterpret_cast<const float4*>(&As[s][k][4*ty]);
            const float4 b = *reinterpret_cast<const float4*>(&Bs[s][k][4*tx]);
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i]*bv[j];
        }
    }
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + 4*ty + i;
        if (r < d.M)
            *reinterpret_cast<float4*>(d.C + (size_t)r*d.ldc + n0 + 4*tx) = make_float4(acc[i][0], acc[i][1],
                    acc[i][2], acc[i][3]);
    }
}

int blocks(long n, int per_block) { return std::max(1, int((n + per_block - 1)/per_block)); }
template <class Sh>
long a_off(int l, long r, long N) { return l == 0 ? r*Sh::width(0) : N*Sh::width(0) + (r - N)*Sh::CN; }
template <class Sh>
long q_off(int t, int l, long K0)
{
    return K0*((long)t*(Sh::width(0) + 3*Sh::CN) + (l == 0 ? 0 : Sh::width(0) + (l - 1)*Sh::CN));
}
template <class Sh> long r_off(int L) { return (long)Sh::C*(L == 0 ? 0 : Sh::width(0) + (L - 1)*Sh::CN); }

using Matrix = std::vector<double>;

template <class real>
void spline(const std::vector<double>& v, const std::vector<double>& d, double h, int num, int f, real* out) {
    const int K = int(v.size()) - 1;
    for (int i = 0; i < K; ++i) {
        const double a0 = v[i];
        const double a1 = h*d[i];
        const double a2 = 3*(v[i+1] - v[i]) - h*(2*d[i] + d[i+1]);
        const double a3 = 2*(v[i] - v[i+1]) + h*(d[i] + d[i+1]);
        real* c = out + (long)i*4*num + f;
        c[0] = real(a0 + a1/2 + a2/4 + a3/8);
        c[num] = real(a1 + a2 + 0.75*a3);
        c[2*num] = real(a2 + 1.5*a3);
        c[3*num] = real(a3);
    }
}

}  // namespace

template <class Precision, bool mixed>
MACEFoundationCuda<Precision, mixed>::MACEFoundationCuda(const std::string& filename, const std::string& prefix)
{
    load(filename, prefix);
}

template <class Precision, bool mixed>
MACEFoundationCuda<Precision, mixed>::~MACEFoundationCuda()
{
    if (blas) cublasDestroy(blas);
    cudaFreeHost(err);
    if (stream) cudaStreamDestroy(stream);
}

template <class Precision, bool mixed>
Model<Precision> MACEFoundationCuda<Precision, mixed>::model() const
{
    Model<Precision> M{};
    M.T = num_elements;
    M.K = K;
    M.K0 = K0;
    M.r_cut2 = r_cut*r_cut;
    M.r_min = Precision(r_min);
    M.inv_h = Precision(1/h);
    M.basis = basis;
    M.R1 = R1;
    M.R1h = R1h;
    M.amax = amax;
    M.A0_splines = A0_splines;
    M.A1_splines = A1_splines;
    M.M0_weights = M0_weights;
    M.M1_weights = M1_weights;
    M.z_weights_for_M0 = z_weights_for_M0;
    M.z_weights_for_M1 = z_weights_for_M1;
    M.readout_1_weights_for_M0 = readout_1_weights_for_M0;
    M.readout_2_weights_2 = readout_2_weights_2;
    M.readout_2_scale_factor = readout_2_scale_factor;
    M.atomic_energies = atomic_energies;
    M.zbl = zbl;
    std::copy(zbl_c, zbl_c + 4, M.zbl_c);
    M.zbl_p = zbl_p;
    M.ordered = ordered;
    M.residual_0 = residual_0;
    M.H1_bias = H1_bias;
    M.z_bias = z_bias;
    return M;
}

template <class Precision, bool mixed>
Graph<Precision> MACEFoundationCuda<Precision, mixed>::graph() const
{
    return {N, NR, atom_of_node, elem, row_of_atom, elem_all, count, start, seg, geometry, neigh_indices,
            row_pair, edge_forces,
            Phi0, A0, M0, H1, Phi1, A1, H1_adj, A1_adj, dPhi1, M0_adj, A0_adj, Phi0_adj,
            s0, s1, dot0, dot1, node_energies, err, X0, X1, Tg, T_adj, X_adj, B1, B1_adj, M0s_adj, M1, M1_adj};
}

template <class Precision, bool mixed>
void MACEFoundationCuda<Precision, mixed>::reserve(long E)
{
    const int T = num_elements;
    const long nn = N;
    const long nr = NR;
    const long C = num_channels;
    const long CN = num_channels_node;
    const long ar = gated ? num_lm + 3 : num_lm;
    count.reserve(nn + 1);
    CHECK(cudaMemsetAsync(count, 0, (nn + 1)*sizeof(int), stream));
    start.reserve(nn + 1);
    seg.reserve(nn*(T + 1) + 1);
    geometry.reserve(4*E);
    neigh_indices.reserve(E);
    row_pair.reserve(E);
    edge_forces.reserve(3*E);
    Phi0.reserve(nn*16*K0);
    Phi0_adj.reserve(nn*16*K0);
    A0.reserve(nn*ar*CN);
    A0_adj.reserve(nn*ar*CN);
    M0.reserve(nn*num_LM*CN);
    M0_adj.reserve(nn*num_LM*CN);
    H1.reserve(nr*num_LM*C);
    H1_adj.reserve(nr*num_LM*C);
    Phi1.reserve(nn*num_lme*C);
    dPhi1.reserve(nn*num_lme*C);
    A1.reserve(nn*ar*CN);
    A1_adj.reserve(nn*ar*CN);
    if (gated) {
        X0.reserve(nn*16*CN);
        X1.reserve(nn*16*CN);
        Tg.reserve(nn*16*CN);
        T_adj.reserve(nn*16*CN);
        X_adj.reserve(nn*16*CN);
        B1.reserve(nn*(4 + num_LM - 1)*CN);
        B1_adj.reserve(nn*(4 + num_LM - 1)*CN);
    }
    M0s_adj.reserve(nn*CN);
    M1.reserve(nn*CN);
    M1_adj.reserve(nn*CN);
    s0.reserve(nn);
    dot0.reserve(nn);
    s1.reserve(nn);
    dot1.reserve(nn);
    node_energies.reserve(nn);
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, count.p, start.p, int(nn + 1), stream);
    scan_storage.reserve(bytes);
    scan_bytes = bytes;
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCuda<Precision, mixed>::compute_first_layer(
    const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    const int T = num_elements;
    if (M.amax) CHECK(cudaMemsetAsync(amax, 0, 12*sizeof(unsigned), stream));

    // A0
    // \Phi^{(0)}_{i,lm,bn} = \sum_{j:a_j=b} \beta_{a_ib,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})
    // A^{(0)}_{i,klm} = \sum_{bn} Q_{a_il,k,bn} \Phi^{(0)}_{i,lm,bn},  \sum_n Q_{al,k,bn} \beta_{ab,n}(r) = \sum_{k'} w_{akl,k'} R^{(0)}_{ab,k'l}(r) H^{(0)}_{b,k'}
    if (NB == 64) k_Phi0<64><<<dim3(blocks(N, 2), T), 64, 0, stream>>>(M, G);
    else if (NB == 96) k_Phi0<96><<<dim3(blocks(N, 3), T), 96, 0, stream>>>(M, G);
    else k_Phi0<32><<<dim3(N, T), 32, 0, stream>>>(M, G);
    for (int l = 0; l < 4; ++l)
        for (const auto& [b, e, t] : segments) {
            const long r0 = (long)N*l*l + (long)b*(2*l+1);
            gemm(false, (e - b)*(2*l+1), Sh::width(l), K0, G.Phi0 + r0*K0, K0, Q, q_off<Sh>(t, l, K0),
                 G.A0 + a_off<Sh>(l, r0, N), Sh::width(l), 0);
        }
    products();

    // gate
    // G_{i,klm} = \tilde{A}^{(0)}_{i,klm} + \delta_{l0} r_{a_i,k},  T_{i,k00} = G_{i,k00} \sigma(G_{i,k00}),  T_{i,klm} = G_{i,klm} \sigma(G_{i,(lC_N+k)00}) (l > 0)
    // X^{(0)}_{i,klm} = \sum_{k'} w_{l,kk'} T_{i,k'lm}
    if constexpr (Sh::gated) {
        k_gate<Sh, 0><<<N, CN, 0, stream>>>(M, G);
        for (int l = 0; l < 4; ++l)
            gemm(false, N*(2*l+1), CN, CN, G.Tg + (long)N*l*l*CN, CN, linear_2_0, (long)l*CN*CN,
                 G.X0 + (long)N*l*l*CN, CN, 6);
        products();
    }

    // M0
    // M^{(0)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_iklm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} \tilde{A}^{(0)}_{i,kl'_\xi m'_\xi},  X^{(0)} for \tilde{A}^{(0)} if gated
    k_M<Sh, 0><<<N, CN, 0, stream>>>(M, G);

    // H1
    // H^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(0)}_{i,k'lm} + \delta_{l0} b^{(1)}_{a_i,k}
    for (int L = 0; L <= Sh::L_max; ++L)
        gemm(false, N*(2*L+1), C, CN, G.M0 + (long)N*L*L*CN, CN, H1_weights, (long)L*CN*C, G.H1 + (long)NR*L*L*C, C, 1);
    products();
    if (M.H1_bias) k_h1_bias<Sh><<<blocks((long)N*C, 256), 256, 0, stream>>>(M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCuda<Precision, mixed>::compute_second_layer(
    const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;

    // A1 scaling
    // s^{(1)}_i = 1 + \sum_j \rho^{(1)}_{a_ia_j}(r_{ij}),  \tilde{A}^{(1)}_{i,klm} = A^{(1)}_{i,klm} / s^{(1)}_i
    k_scale1<<<blocks(N, WARPS), 32*WARPS, 0, stream>>>(M, G);

    // Phi1
    // \Phi^{(1)}_{i,klm\eta} = \sum_j \sum_{l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) H^{(1)}_{j,kl_2m_2}
    const int split = std::clamp(16*num_sms/std::max(N*groups<Sh>, 1), 1, 4);
    if (split > 1) CHECK(cudaMemsetAsync(G.Phi1, 0, (size_t)N*Sh::num_lme*C*sizeof(Precision), stream));
    if (M.R1h) k_Phi1<Sh, true><<<dim3(N, groups<Sh>, split), C, 0, stream>>>(M, G);
    else k_Phi1<Sh, false><<<dim3(N, groups<Sh>, split), C, 0, stream>>>(M, G);

    // A1
    // A^{(1)}_{i,klm} = \sum_{k'\eta'} w_{kl,k'\eta'} \Phi^{(1)}_{i,k'lm\eta'}
    for (int l = 0; l < 4; ++l)
        gemm(false, N*(2*l+1), Sh::width(l), Sh::num_e(l)*C, G.Phi1 + (long)N*Sh::lme_offset(l)*C, Sh::num_e(l)*C,
             A1_weights, A1_weights_offset[l],
             G.A1 + a_off<Sh>(l, (long)N*l*l, N), Sh::width(l), 2);
    if constexpr (not Sh::gated) products();

    // gate
    // G_{i,klm} = \tilde{A}^{(1)}_{i,klm} + \sum_{k'} w_{kl,k'} H^{(1)}_{i,k'lm},  T_{i,k00} = G_{i,k00} \sigma(G_{i,k00}),  T_{i,klm} = G_{i,klm} \sigma(G_{i,(lC_N+k)00}) (l > 0)
    // X^{(1)}_{i,klm} = \sum_{k'} w_{l,kk'} T_{i,k'lm}
    if constexpr (Sh::gated) {
        if (M.amax)
            k_amax<<<std::min(432, blocks((long)N*Sh::num_LM*C, 512)), 512, 0, stream>>>(M, G.H1,
                    (long)NR*Sh::num_LM*C, 8);
        for (int L = 0; L <= Sh::L_max; ++L)
            gemm(false, N*(2*L+1), Sh::width(L), C, G.H1 + (long)NR*L*L*C, C, residual_1_weights, r_off<Sh>(L),
                 G.B1 + a_off<Sh>(L, (long)N*L*L, N), Sh::width(L), 8);
        products();
        k_gate<Sh, 1><<<N, CN, 0, stream>>>(M, G);
        for (int l = 0; l < 4; ++l)
            gemm(false, N*(2*l+1), CN, CN, G.Tg + (long)N*l*l*CN, CN, linear_2_1, (long)l*CN*CN,
                 G.X1 + (long)N*l*l*CN, CN, 7);
        products();
    }

    // M1
    // M^{(1)}_{i,k} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_ik,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} \tilde{A}^{(1)}_{i,kl'_\xi m'_\xi},  X^{(1)} for \tilde{A}^{(1)} if gated
    k_M<Sh, 1><<<N, CN, 0, stream>>>(M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCuda<Precision, mixed>::compute_readouts(
    const Model<Precision>& M, const Graph<Precision>& G)
{
    // ZBL
    // E_i += \frac{1}{2} \sum_j V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij})
    // \mathbf{F}_{ij} -= \frac{1}{2} \partial_r V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij}) \hat{\mathbf{r}}_{ij}
    k_zbl<<<blocks(N, WARPS), 32*WARPS, 0, stream>>>(M, G);

    // H2
    // H^{(2)}_{i,k} = \sum_{k'} w_{k,k'} M^{(1)}_{i,k'} + \sum_{k'} w_{a_ik,k'} M^{(0)}_{i,k'00} + b^{(2)}_{a_i,k}
    // z_{i,h} = \sum_k w_{h,k} H^{(2)}_{i,k} = \sum_k (u_{a_ih,k} M^{(0)}_{i,k00} + u_{h,k} M^{(1)}_{i,k}) + v_{a_ih}
    // readouts
    // E_i += E^{(0)}_{a_i} + \sum_k w_k M^{(0)}_{i,k00} + \mathrm{MLP}(\{H^{(2)}_{i,k}\}),  \mathrm{MLP} = \sum_h w_h c\, z_{i,h} \sigma(z_{i,h})
    // \bar{M}^{(0)}_{i,k00} = w_k,  \bar{z}_{i,h} = w_h c\, \sigma(z_{i,h}) (1 + z_{i,h} (1 - \sigma(z_{i,h})))
    // H2
    // \bar{M}^{(1)}_{i,k} = \sum_h u_{h,k} \bar{z}_{i,h},  \bar{M}^{(0)}_{i,k00} += \sum_h u_{a_ih,k} \bar{z}_{i,h}
    k_readouts<Sh><<<N, Sh::CN, 0, stream>>>(M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCuda<Precision, mixed>::reverse_second_layer(
    const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    // M1
    // \bar{\tilde{A}}^{(1)}_{i,klm} = \bar{M}^{(1)}_{i,k} \partial M^{(1)}_{i,k} / \partial \tilde{A}^{(1)}_{i,klm},  \bar{X}^{(1)} for \bar{\tilde{A}}^{(1)} if gated
    // \bar{A}^{(1)}_{i,klm} = \bar{\tilde{A}}^{(1)}_{i,klm} / s^{(1)}_i,  d^{(1)}_i = \sum_{klm} \bar{\tilde{A}}^{(1)}_{i,klm} \tilde{A}^{(1)}_{i,klm} / s^{(1)}_i
    k_M_reverse<Sh, 1><<<N, CN, 0, stream>>>(M, G);

    CHECK(cudaMemsetAsync(G.H1_adj, 0, (size_t)NR*Sh::num_LM*C*sizeof(Precision), stream));

    // gate
    // \bar{T}_{i,k'lm} = \sum_k w_{l,kk'} \bar{X}^{(1)}_{i,klm},  \bar{G}_{i,k00} = \bar{T}_{i,k00} \sigma(G_{i,k00}) (1 + G_{i,k00} (1 - \sigma(G_{i,k00})))
    // \bar{G}_{i,klm} = \bar{T}_{i,klm} \sigma(\gamma),  \bar{G}_{i,(lC_N+k)00} = \sigma(\gamma) (1 - \sigma(\gamma)) \sum_m G_{i,klm} \bar{T}_{i,klm},  \gamma = G_{i,(lC_N+k)00} (l > 0)
    // \bar{A}^{(1)}_{i,klm} = \bar{G}_{i,klm} / s^{(1)}_i,  d^{(1)}_i = \sum_{klm} \bar{G}_{i,klm} \tilde{A}^{(1)}_{i,klm} / s^{(1)}_i,  \bar{H}^{(1)}_{i,k'lm} = \sum_k w_{kl,k'} \bar{G}_{i,klm}
    if constexpr (Sh::gated) {
        for (int l = 0; l < 4; ++l)
            gemm(true, N*(2*l+1), CN, CN, G.X_adj + (long)N*l*l*CN, CN, linear_2_1, (long)l*CN*CN,
                 G.T_adj + (long)N*l*l*CN, CN, 9);
        products();
        k_gate_reverse<Sh, 1><<<N, CN, 0, stream>>>(M, G);
        for (int L = 0; L <= Sh::L_max; ++L)
            gemm(true, N*(2*L+1), C, Sh::width(L), G.B1_adj + a_off<Sh>(L, (long)N*L*L, N), Sh::width(L),
                 residual_1_weights, r_off<Sh>(L), G.H1_adj + (long)NR*L*L*C, C, 11);
    }

    // A1
    // \bar{\Phi}^{(1)}_{i,k'lm\eta'} = \sum_k w_{kl,k'\eta'} \bar{A}^{(1)}_{i,klm}
    for (int l = 0; l < 4; ++l)
        gemm(true, N*(2*l+1), Sh::num_e(l)*C, Sh::width(l), G.A1_adj + a_off<Sh>(l, (long)N*l*l, N), Sh::width(l),
             A1_weights, A1_weights_offset[l],
             G.dPhi1 + (long)N*Sh::lme_offset(l)*C, Sh::num_e(l)*C, 3);
    products();

    // Phi1
    // \bar{H}^{(1)}_{j,kl_2m_2} = \sum_i \sum_{lm\eta l_1m_1} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(1)}_{i,klm\eta}
    // \mathbf{F}_{ij} -= \sum_{klm\eta l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij})] H^{(1)}_{j,kl_2m_2} \bar{\Phi}^{(1)}_{i,klm\eta}
    const int split = std::clamp(16*num_sms/std::max(N*groups<Sh>, 1), 1, 4);
    if (M.R1h) k_Phi1_reverse<Sh, true><<<dim3(N, groups<Sh>, split), C, 0, stream>>>(M, G);
    else k_Phi1_reverse<Sh, false><<<dim3(N, groups<Sh>, split), C, 0, stream>>>(M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCuda<Precision, mixed>::reverse_first_layer(
    const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    const int T = num_elements;

    // H1
    // \bar{M}^{(0)}_{i,k'lm} += \sum_k w_{kl,k'} \bar{H}^{(1)}_{i,klm}
    if (M.amax)
        k_amax<<<std::min(432, blocks((long)NR*Sh::num_LM*C, 512)), 512, 0, stream>>>(M, G.H1_adj,
                (long)NR*Sh::num_LM*C, 4);
    for (int L = 0; L <= Sh::L_max; ++L)
        gemm(true, N*(2*L+1), CN, C, G.H1_adj + (long)NR*L*L*C, C, H1_weights, (long)L*CN*C,
             G.M0_adj + (long)N*L*L*CN, CN, 4);
    products();

    // M0
    // \bar{\tilde{A}}^{(0)}_{i,klm} = \sum_{l'm'} \bar{M}^{(0)}_{i,kl'm'} \partial M^{(0)}_{i,kl'm'} / \partial \tilde{A}^{(0)}_{i,klm},  \bar{X}^{(0)} for \bar{\tilde{A}}^{(0)} if gated
    // \bar{A}^{(0)}_{i,klm} = \bar{\tilde{A}}^{(0)}_{i,klm} / s^{(0)}_i,  d^{(0)}_i = \sum_{klm} \bar{\tilde{A}}^{(0)}_{i,klm} \tilde{A}^{(0)}_{i,klm} / s^{(0)}_i
    k_M_reverse<Sh, 0><<<N, CN, 0, stream>>>(M, G);

    // gate
    // \bar{T}_{i,k'lm} = \sum_k w_{l,kk'} \bar{X}^{(0)}_{i,klm},  \bar{G} as in reverse_second_layer
    // \bar{A}^{(0)}_{i,klm} = \bar{G}_{i,klm} / s^{(0)}_i,  d^{(0)}_i = \sum_{klm} \bar{G}_{i,klm} \tilde{A}^{(0)}_{i,klm} / s^{(0)}_i
    if constexpr (Sh::gated) {
        for (int l = 0; l < 4; ++l)
            gemm(true, N*(2*l+1), CN, CN, G.X_adj + (long)N*l*l*CN, CN, linear_2_0, (long)l*CN*CN,
                 G.T_adj + (long)N*l*l*CN, CN, 10);
        products();
        k_gate_reverse<Sh, 0><<<N, CN, 0, stream>>>(M, G);
    }

    // A0
    // \bar{\Phi}^{(0)}_{i,lm,bn} = \sum_k Q_{a_il,k,bn} \bar{A}^{(0)}_{i,klm}
    // \mathbf{F}_{ij} -= \sum_{lmn} \nabla_{\mathbf{r}_{ij}}[\beta_{a_ia_j,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})] \bar{\Phi}^{(0)}_{i,lm,a_jn}
    for (int l = 0; l < 4; ++l)
        for (const auto& [b, e, t] : segments) {
            const long r0 = (long)N*l*l + (long)b*(2*l+1);
            gemm(true, (e - b)*(2*l+1), K0, Sh::width(l), G.A0_adj + a_off<Sh>(l, r0, N), Sh::width(l), Q,
                 q_off<Sh>(t, l, K0), G.Phi0_adj + r0*K0, K0, 5);
        }
    products();
    if (NB == 64) k_Phi0_reverse<64><<<dim3(N, T), 128, 0, stream>>>(M, G);
    else if (NB == 96) k_Phi0_reverse<96><<<dim3(N, T), 192, 0, stream>>>(M, G);
    else k_Phi0_reverse<32><<<dim3(N, T), 64, 0, stream>>>(M, G);
}

template <class Precision, bool mixed>
void MACEFoundationCuda<Precision, mixed>::gemm(
    bool trans, int m, int n, int k, const Precision* a, int lda, const Weight& W, long offset, Precision* c,
    int ldc, int slot)
{
    if (m > 0)
        queue.push_back({a, (trans ? W.tw : W.w) + offset, W.w + offset, (trans ? W.hi : W.thi) + offset,
                         (trans ? W.lo : W.tlo) + offset, c, m, n, k, lda, ldc, 0, W.shift, slot, trans});
}

template <class Precision, bool mixed>
void MACEFoundationCuda<Precision, mixed>::products()
{
    if constexpr (std::is_same_v<Precision, double>) {
        const double one = 1;
        const double zero = 0;
        for (const auto& g : queue)
            CHECK(cublasDgemm(blas, g.trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, g.N, g.M, g.K, &one, g.W,
                              g.trans ? g.K : g.N, g.A, g.lda, &zero, g.C, g.ldc));
    } else if (mixed and tensor_cores) {
        for (size_t i0 = 0; i0 < queue.size(); i0 += 32) {
            GemmBatch b{};
            bool n128 = true;
            bool n64 = true;
            bool n112 = true;
            bool n96 = true;
            long big = 0;
            for (size_t i = i0; i < std::min(queue.size(), i0 + 32); ++i) {
                b.g[b.n++] = queue[i];
                n128 = n128 and queue[i].N%128 == 0;
                n64 = n64 and queue[i].N%64 == 0;
                n112 = n112 and queue[i].N%112 == 0;
                n96 = n96 and queue[i].N%96 == 0;
                big += (queue[i].M + 127)/128*(queue[i].N/32);
            }
            if (big < num_sms) gemm_batch<32, 32, 1, 1>(b, amax, stream);
            else if (n128) gemm_batch<128, 128, 4, 2>(b, amax, stream);
            else if (n64) gemm_batch<128, 64, 4, 2>(b, amax, stream);
            else if (n112 and big >= 8*num_sms) gemm_batch<128, 112, 4, 2>(b, amax, stream);
            else if (n96 and big >= 8*num_sms) gemm_batch<128, 96, 4, 2>(b, amax, stream);
            else gemm_batch<128, 32, 4, 1>(b, amax, stream);
        }
    } else {
        constexpr int BM = 64;
        constexpr int BN = 32;
        long tiles = 0;
        bool n64 = true;
        for (const auto& g : queue) {
            tiles += (g.M + BM - 1)/BM*(g.N/BN);
            n64 = n64 and g.N%64 == 0;
        }
        if (tiles > (n64 ? 40L : 80L)*num_sms) {
            const float one = 1;
            const float zero = 0;
            for (const auto& g : queue)
                CHECK(cublasSgemm(blas, g.trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, g.N, g.M, g.K, &one, g.W,
                                  g.trans ? g.K : g.N, g.A, g.lda, &zero, g.C, g.ldc));
        } else
            for (size_t i0 = 0; i0 < queue.size(); i0 += 32) {
                GemmBatch b{};
                int t = 0;
                for (size_t i = i0; i < std::min(queue.size(), i0 + 32); ++i) {
                    b.g[b.n] = queue[i];
                    b.g[b.n++].tile0 = t;
                    t += (queue[i].M + BM - 1)/BM*(queue[i].N/BN);
                }
                k_sgemm<BM, BN><<<t, BM*BN/16, 0, stream>>>(b);
            }
    }
    queue.clear();
}


template <class Precision, bool mixed>
void MACEFoundationCuda<Precision, mixed>::load(const std::string& filename, const std::string& prefix)
{
    int device;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    cudaFuncAttributes attributes;
    CHECK(cudaFuncGetAttributes(&attributes, k_gemm<32, 32, 1, 1>));
    tensor_cores = attributes.ptxVersion >= 80;
    std::ifstream file(filename);
    if (not file) throw std::runtime_error(prefix + "cannot open " + filename);
    const json js = json::parse(file);
    if (js.contains("model_type")) throw std::runtime_error(prefix + filename + " is not a MACE model");
    if (not js.contains("first_layer_weights"))
        throw std::runtime_error(prefix + "the model file has no first-layer basis; "
                                          "extract it again with this Symmetrix");
    const bool fp16_gemms = mixed and tensor_cores;
    num_elements = js.at("num_elements");
    num_channels = js.at("num_channels");
    L_max = js.at("L_max");
    num_LM = (L_max + 1)*(L_max + 1);
    r_cut = js.at("r_cut");
    atomic_numbers = js.at("atomic_numbers").get<std::vector<int>>();
    num_channels_node = js.contains("num_channels_node") ? js.at("num_channels_node").get<int>() : num_channels;
    ordered = js.contains("ordered_pairs") and js.at("ordered_pairs").get<bool>();
    const int C = num_channels;
    const int CN = num_channels_node;
    int num_paths = 0;
    int num_e[4] = {};
    int width[4] = {};
    if (js.at("l_max") != 3 or not with_preset(C, L_max, CN, [&](auto s) {
                using Sh = decltype(s);
                {
                    const auto l = js.at("Phi1_l").get<std::vector<int>>();
                    const auto spec = js.at("Phi1_spec").get<std::vector<std::array<int, 3>>>();
                    const auto cg_spec = js.at("Phi1_cg_spec").get<std::vector<std::array<int, 2>>>();
                    const auto cg = js.at("Phi1_clebsch_gordan").get<std::vector<double>>();
                    bool same = int(l.size()) == Sh::num_paths and int(cg.size()) == Sh::num_terms
                                and int(cg_spec.size()) == Sh::num_terms;
                    for (int q = 0; same and q < Sh::num_terms; ++q) {
                        const auto [lme, le, lm2] = spec.at(cg_spec[q][0]);
                        same = Sh::terms[q].lme == lme and Sh::terms[q].path == le and Sh::terms[q].lm1 == cg_spec[q][1]
                            and Sh::terms[q].lm2 == lm2 and std::abs(Sh::terms[q].cg - cg[q]) < 1e-6;
                    }
                    for (int p = 0; same and p < Sh::num_paths; ++p) same = l[p] == Sh::Phi1_l[p];
                    auto contraction = [&](const std::string& i, auto t) {
                        using T = decltype(t);
                        const auto b = js.at("B" + i + "_spec").get<std::vector<std::array<int, 3>>>();
                        const auto m = js.at("M" + i + "_spec").get<std::vector<std::array<int, 3>>>();
                        const auto c = js.at("B" + i + "_coefficients").get<std::vector<double>>();
                        bool same = int(b.size()) == T::num_b and int(m.size()) == T::num_m;
                        for (int e = 0; same and e < T::num_b; ++e)
                            same = b[e] == std::array<int, 3>{T::b[e].q, T::b[e].a, T::b[e].b}
                                   and std::abs(c[e] - T::b[e].c) < 1e-6;
                        for (int e = 0; same and e < T::num_m; ++e)
                            same = m[e] == std::array<int, 3>{T::m[e].o, T::m[e].q, T::m[e].n};
                        return same;
                    };
                    if (not same or not contraction("0", Sh{}) or not contraction("1", Tables<0>{}))
                        throw std::runtime_error(prefix + "the model's tensor products differ "
                                                          "from those of its preset");
                }
                num_paths = Sh::num_paths;
                num_lme = Sh::num_lme;
                gated = Sh::gated;
                for (int l = 0; l < 4; ++l) {
                    num_e[l] = Sh::num_e(l);
                    width[l] = Sh::width(l);
                }
            }))
        throw std::runtime_error(prefix + "the model (" + std::to_string(C) + " channels, L_max "
                                 + std::to_string(L_max) + ", " + std::to_string(CN) + " node channels) "
                                   "is not of a preset shape: " + preset_names);
    const int T = num_elements;
    const int P = ordered ? T*T : T*(T + 1)/2;
    NB = (js.at("first_layer_basis_size").get<int>() + 31)/32*32;
    if (NB > 96)
        throw std::runtime_error(prefix + "first-layer bases of more than 96 functions are not supported, "
                                          "and the extractor could not compress this one within the spline error");
    K0 = T*NB;
    K = int(js.at("radial_spline_values_0")[0][0].size()) - 1;
    r_min = js.at("radial_spline_r_min");
    h = (r_cut - r_min)/K;

    {
        std::vector<Precision> table((long)P*K*4*num_paths*C);
        const auto& values = js.at("radial_spline_values_1");
        const auto& derivs = js.at("radial_spline_derivs_1");
        for (int p = 0; p < P; ++p)
            for (int f = 0; f < num_paths*C; ++f)
                spline(values[p][f].get<std::vector<double>>(), derivs[p][f].get<std::vector<double>>(), h,
                       num_paths*C, f,
                                table.data() + (long)p*K*4*num_paths*C);
        if (not mixed) R1.upload(table, 0);
        else {
            const long F = num_paths*C;
            const long B = (long)P*K;
            std::vector<Precision> c01(B*2*F);
            std::vector<__half2> c23(B*F);
            for (long b = 0; b < B; ++b)
                for (long f = 0; f < F; ++f) {
                    const Precision* c = table.data() + b*4*F + f;
                    if (std::max(std::abs(c[2*F]), std::abs(c[3*F])) > 6e4f)
                        throw std::runtime_error(prefix + "second-layer spline coefficients exceed the fp16 range");
                    c01[b*2*F + f] = c[0];
                    c01[(b*2 + 1)*F + f] = c[F];
                    c23[b*F + f] = __floats2half2_rn(c[2*F], c[3*F]);
                }
            R1.upload(c01, 0);
            R1h.upload(c23, 0);
        }

        std::vector<Precision> w1;
        std::vector<std::array<long, 3>> blocks;
        A1_weights_offset.clear();
        for (int L = 0; L < 4; ++L) {
            A1_weights_offset.push_back(long(w1.size()));
            blocks.push_back({long(w1.size()), long(num_e[L])*C, width[L]});
            const auto w = js.at("A1_weights")[L].get<std::vector<double>>();
            if (long(w.size()) != long(num_e[L])*C*width[L]) throw std::runtime_error(prefix + "unexpected A1 weights");
            for (double v : w) w1.push_back(Precision(v));
        }
        set_weight(A1_weights, w1, blocks);
    }

    {
        const int S = K + 1;
        const int nb = js.at("first_layer_basis_size");
        const int widths = width[0] + width[1] + width[2] + width[3];
        const auto H0 = js.at("H0_weights").get<Matrix>();
        const auto U = js.at("first_layer_weights").get<Matrix>();
        const auto A0w = js.at("A0_weights").get<std::vector<std::vector<Matrix>>>();
        const int N = int(U.size())/(4*C);
        auto table = [&](const char* key, int p) {
            Matrix m;
            for (const auto& f : js.at(key)[p]) for (const auto& v : f) m.push_back(v.get<double>());
            return m;
        };
        std::vector<Precision> host_basis((long)T*T*K*4*NB, 0.f);
        std::vector<Precision> q((long)T*widths*K0, 0.f);
        for (int a = 0; a < T; ++a)
            for (int b = 0; b < T; ++b) {
                const int p = ordered ? a*T + b : std::min(a, b)*(2*T - std::min(a, b) - 1)/2 + std::max(a, b);
                Matrix V = table("first_layer_values", p);
                Matrix D = table("first_layer_derivs", p);
                Matrix map((long)widths*nb, 0.0);
                if (nb < N) {
                    const Matrix B = js.at("first_layer_basis")[a*T + b].get<Matrix>();
                    const Matrix V0 = V;
                    const Matrix D0 = D;
                    map = js.at("first_layer_map")[a*T + b].get<Matrix>();
                    for (int r = 0; r < nb; ++r)
                        for (int c = 0; c < S; ++c) {
                            double v = 0;
                            double d = 0;
                            for (int i = 0; i < N; ++i) {
                                v += B[(long)r*N + i]*V0[(long)i*S + c];
                                d += B[(long)r*N + i]*D0[(long)i*S + c];
                            }
                            V[(long)r*S + c] = v;
                            D[(long)r*S + c] = d;
                        }
                } else
                    for (int l = 0, o = 0; l < 4; o += width[l++])
                        for (int kp = 0; kp < C; ++kp) {
                            const double* u = &U[((long)l*C + kp)*N];
                            for (int k = 0; k < width[l]; ++k) {
                                const double c = A0w[a][l][(long)kp*width[l] + k]*H0[b*C + kp];
                                for (int n = 0; n < N; ++n) map[((long)o + k)*nb + n] += c*u[n];
                            }
                        }
                Precision* tab = host_basis.data() + (long)(a*T + b)*K*4*NB;
                for (int n = 0; n < nb; ++n)
                    spline(Matrix(V.begin() + n*S, V.begin() + (n + 1)*S),
                           Matrix(D.begin() + n*S, D.begin() + (n + 1)*S), h, NB, n, tab);
                for (int l = 0, o = 0; l < 4; o += width[l++])
                    for (int n = 0; n < nb; ++n)
                        for (int k = 0; k < width[l]; ++k)
                            q[((long)a*widths + o)*K0 + ((long)b*NB + n)*width[l] + k] =
                                Precision(map[((long)o + k)*nb + n]);
            }
        basis.upload(host_basis, 0);
        std::vector<std::array<long, 3>> blocks;
        for (long a = 0, o = 0; a < T; ++a)
            for (int l = 0; l < 4; o += (long)K0*width[l++]) blocks.push_back({o, K0, width[l]});
        set_weight(Q, q, blocks);
    }

    auto scale = [&](const std::string& key, Buffer<Precision>& buf) {
        if (not js.at(key + "_scaled").get<bool>()) return;
        const auto v = js.at(key + "_spline_values").get<std::vector<std::vector<double>>>();
        const auto d = js.at(key + "_spline_derivs").get<std::vector<std::vector<double>>>();
        if (js.at(key + "_spline_r_min").get<double>() != r_min or int(v[0].size()) != K + 1)
            throw std::runtime_error(prefix + "the scale-factor splines must use the radial grid");
        std::vector<Precision> table((long)P*K*4);
        for (int p = 0; p < P; ++p) spline(v[p], d[p], h, 1, 0, table.data() + (long)p*K*4);
        buf.upload(table, 0);
    };
    scale("A0", A0_splines);
    scale("A1", A1_splines);

    auto contraction = [&](const std::string& l, Buffer<Precision>& buf) {
        std::vector<Precision> w;
        for (const auto& a : js.at("M" + l + "_weights"))
            for (const auto& e : a) for (const auto& x : e) w.push_back(x.get<Precision>());
        buf.upload(w, 0);
    };
    contraction("0", M0_weights);
    contraction("1", M1_weights);

    {
        std::vector<Precision> wh1;
        for (double v : js.at("H1_weights").get<std::vector<double>>()) wh1.push_back(Precision(v));
        std::vector<std::array<long, 3>> blocks;
        for (long L = 0; L <= L_max; ++L) blocks.push_back({L*CN*C, CN, C});
        set_weight(H1_weights, wh1, blocks);
        const auto wh = js.at("H2_weights_for_M0").get<std::vector<std::vector<double>>>();
        const auto wm = js.at("H2_weights_for_M1").get<std::vector<double>>();
        const auto r1 = js.at("readout_2_weights_1").get<std::vector<double>>();
        if (r1.size() != 16*CN) throw std::runtime_error(prefix + "readout 2 must have 16 hidden units");
        std::vector<Precision> fh((long)T*16*CN), fm(16*CN), rb(16*T);
        for (int hh = 0; hh < 16; ++hh)
            for (int k = 0; k < CN; ++k) {
                for (int t = 0; t < T; ++t) {
                    double s = 0;
                    for (int kp = 0; kp < CN; ++kp) s += wh[t][k*CN + kp]*r1[hh*CN + kp];
                    fh[((long)t*16 + hh)*CN + k] = Precision(s);
                }
                double s = 0;
                for (int kp = 0; kp < CN; ++kp) s += wm[k*CN + kp]*r1[hh*CN + kp];
                fm[hh*CN + k] = Precision(s);
            }
        if (js.contains("H2_bias")) {
            const auto hb = js.at("H2_bias").get<std::vector<std::vector<double>>>();
            for (int t = 0; t < T; ++t)
                for (int hh = 0; hh < 16; ++hh) {
                    double s = 0;
                    for (int kp = 0; kp < CN; ++kp) s += hb[t][kp]*r1[hh*CN + kp];
                    rb[t*16 + hh] = Precision(s);
                }
        }
        z_weights_for_M0.upload(fh, 0);
        z_weights_for_M1.upload(fm, 0);
        z_bias.upload(rb, 0);
        std::vector<Precision> r1w, r2w;
        for (double v : js.at("readout_1_weights_for_M0").get<std::vector<double>>()) r1w.push_back(Precision(v));
        for (double v : js.at("readout_2_weights_2").get<std::vector<double>>()) r2w.push_back(Precision(v));
        readout_1_weights_for_M0.upload(r1w, 0);
        readout_2_weights_2.upload(r2w, 0);
        readout_2_scale_factor = js.at("readout_2_scale_factor").get<Precision>();
        atomic_energies.upload(js.at("atomic_energies").get<std::vector<double>>(), 0);
    }

    if (gated) {
        auto flat = [](const json& j) {
            std::vector<Precision> v;
            for (const auto& r : j) for (const auto& x : r) v.push_back(x.get<Precision>());
            return v;
        };
        residual_0.upload(flat(js.at("residual_0")), 0);
        std::vector<std::array<long, 3>> blocks;
        for (long l = 0; l < 4; ++l) blocks.push_back({l*CN*CN, CN, CN});
        set_weight(linear_2_0, flat(js.at("linear_2_0")), blocks);
        set_weight(linear_2_1, flat(js.at("linear_2_1")), blocks);
        blocks.clear();
        for (long L = 0, o = 0; L <= L_max; o += (long)C*width[L++]) blocks.push_back({o, C, width[L]});
        set_weight(residual_1_weights, js.at("residual_1_weights").get<std::vector<Precision>>(), blocks);
    }

    if (js.contains("H1_bias")) {
        std::vector<Precision> b;
        for (const auto& r : js.at("H1_bias")) for (const auto& x : r) b.push_back(x.get<Precision>());
        H1_bias.upload(b, 0);
    }

    has_zbl = js.at("has_zbl").get<bool>();
    if (has_zbl) {
        const auto& Z = atomic_numbers;
        const auto rc = js.at("zbl_covalent_radii").get<std::vector<double>>();
        const auto c = js.at("zbl_c").get<std::vector<double>>();
        const double a_exp = js.at("zbl_a_exp");
        const double a_pre = js.at("zbl_a_prefactor");
        std::vector<Precision> z(3*P);
        for (int a = 0; a < T; ++a)
            for (int b = a; b < T; ++b) {
                const int p = a*(2*T - a - 1)/2 + b;
                z[3*p] = Precision(a_pre*0.529/(std::pow(Z[a], a_exp) + std::pow(Z[b], a_exp)));
                z[3*p+1] = Precision(rc[Z[a]] + rc[Z[b]]);
                z[3*p+2] = Precision(14.3996*Z[a]*Z[b]);
            }
        zbl.upload(z, 0);
        for (int q = 0; q < 4; ++q) zbl_c[q] = Precision(c[q]);
        zbl_p = int(js.at("zbl_p").get<double>());
    }

    if (fp16_gemms) {
        amax.reserve(12);
        CHECK(cudaFuncSetAttribute(k_gemm<128, 128, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   gemm_smem<128, 128>));
        CHECK(cudaFuncSetAttribute(k_gemm<128, 64, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   gemm_smem<128, 64>));
        CHECK(cudaFuncSetAttribute(k_gemm<128, 112, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   gemm_smem<128, 112>));
        CHECK(cudaFuncSetAttribute(k_gemm<128, 96, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   gemm_smem<128, 96>));
    }
    if (not blas) {
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cublasSetStream(blas, stream));
        blas_workspace.reserve(32 << 20);
        CHECK(cublasSetWorkspace(blas, blas_workspace, blas_workspace.n));
        CHECK(cudaHostAlloc(&err, sizeof(unsigned), cudaHostAllocMapped));
    }
    *err = 0;
    outputs.reserve(7);
}

template <class Precision, bool mixed>
void MACEFoundationCuda<Precision, mixed>::set_weight(Weight& W, const std::vector<Precision>& v,
                                                      const std::vector<std::array<long, 3>>& blocks)
{
    W.w.upload(v, 0);
    std::vector<Precision> t(v.size());
    for (const auto& [off, rows, cols] : blocks)
        for (long r = 0; r < rows; ++r)
            for (long c = 0; c < cols; ++c) t[off + c*rows + r] = v[off + r*cols + c];
    W.tw.upload(t, 0);
    if (not (mixed and tensor_cores)) return;
    Precision m = 0;
    for (Precision x : v) m = std::max(m, std::abs(x));
    W.shift = scale_shift(m);
    auto halves = [&](const std::vector<Precision>& x, Buffer<__half>& hi, Buffer<__half>& lo) {
        std::vector<__half> a(x.size()), b(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            const Precision s = std::ldexp(x[i], W.shift);
            a[i] = __float2half_rn(s);
            b[i] = __float2half_rn(s - __half2float(a[i]));
        }
        hi.upload(a, 0);
        lo.upload(b, 0);
    };
    halves(v, W.hi, W.lo);
    halves(t, W.thi, W.tlo);
}
