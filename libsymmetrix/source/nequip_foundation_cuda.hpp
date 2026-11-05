#pragma once

#include "nequip_foundation_tables.hpp"

#include "../external/json/single_include/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

namespace nequip_foundation {

struct Tuning {
    static constexpr int n_big = 2000;
    static constexpr int sort_key = 2;
    static constexpr int tp_unroll = 1;
    static constexpr bool smem_edge = true;
    static constexpr int smem_ch = 8;
    static constexpr int smem_ch_big = 16;
    static constexpr int fwd_unroll = 0;
    static constexpr int rev_unroll = 0;
    static constexpr int qrev = 0;
    static constexpr int qrev_unroll = 0;
    static constexpr int qrev_minb = 0;
    static constexpr bool grp = false;
    static constexpr int cpt0 = 1;
};
struct TuningS : Tuning {
    static constexpr int sort_key = 0;
    static constexpr int tp_unroll = 2;
    static constexpr int cpt[2][5] = {{1, 1}, {4, 2}};
};
struct TuningM : Tuning {
    static constexpr bool smem_edge = false;
    static constexpr int smem_ch_big = 8;
    static constexpr int cpt[2][5] = {{4, 2, 1}, {4, 2, 1}};
};
struct TuningL : Tuning {
    static constexpr bool grp = true;
    static constexpr int cpt0 = 2;
    static constexpr int n_big = 1500;
    static constexpr int fwd_unroll = 15;
    static constexpr int qrev = 15;
    static constexpr int qrev_unroll = 8;
    static constexpr int qrev_minb = 0x3304;
    static constexpr int cpt[2][5] = {{4, 2, 1, 1}, {4, 2, 1, 1}};
};
struct TuningXL : Tuning {
    static constexpr int fwd_unroll = 30;
    static constexpr int cpt[2][5] = {{2, 1, 1, 1, 1}, {2, 1, 2, 1, 1}};
};

template <class Tune, int num_layers_, int l_max_, int emb_, int... mul_> struct Shape : Tune {
    static constexpr int num_layers = num_layers_;
    static constexpr int l_max = l_max_;
    static constexpr int emb = emb_;
    static constexpr int num_lm = (l_max + 1)*(l_max + 1);
    static constexpr int mul[5] = {mul_...};
    static constexpr int mulx(int l) { int r = 0; int q = 0; ((r = q++ == l ? mul_ : r), ...); return r; }
    static constexpr int nsc = mul[0];
    static constexpr int ngate = (0 + ... + mul_) - mul[0];
    static constexpr int ngt = nsc > ngate ? nsc : ngate;
    static constexpr int muly(int l) { return l == 0 ? nsc + ngate : mulx(l); }
    static constexpr long xblk(int l) { long o = 0; for (int q = 0; q < l; ++q) o += (2*q + 1)*mulx(q); return o; }
    static constexpr long yblk(int l) { long o = 0; for (int q = 0; q < l; ++q) o += (2*q + 1)*muly(q); return o; }
    static constexpr int xw = xblk(l_max + 1);
    static constexpr int yw = yblk(l_max + 1);
};
using Presets = std::tuple<Shape<TuningS, 2, 1, 32, 128, 64>, Shape<TuningM, 4, 2, 32, 128, 64, 32>,
                           Shape<TuningL, 6, 3, 48, 128, 64, 32, 32>, Shape<TuningXL, 6, 4, 32, 320, 96, 64, 32, 32>>;
constexpr const char* preset_names = "NequIP-OAM-S, M, L and XL";

template <class F> bool with_preset(int num_layers, int l_max, int emb, int mul0, F&& f) {
    return std::apply([&](auto... s) {
        return ((decltype(s)::num_layers == num_layers and decltype(s)::l_max == l_max
                 and decltype(s)::emb == emb and decltype(s)::mul[0] == mul0 and (f(s), true)) or ...);
    }, Presets{});
}

struct Ins { int l1, l2, l3, path, mul, roff, choff; };
template <class Sh, int LT> struct Layer {
    static constexpr int lo1 = 0;
    static constexpr int hi1 = LT == 0 ? 0 : Sh::l_max;
    static constexpr std::array<Ins, 64> make() {
        std::array<Ins, 64> a{};
        int n = 0;
        int roff = 0;
        for (int l1 = lo1; l1 <= hi1; ++l1)
            for (int l2 = 0; l2 <= Sh::l_max; ++l2)
                for (int l3 = 0; l3 <= Sh::l_max; ++l3) {
                    if (l3 < (l1 > l2 ? l1 - l2 : l2 - l1) or l3 > l1 + l2 or (l1 + l2 + l3) % 2
                        or (LT == 2 and l3 != 0))
                        continue;
                    const int mul = LT == 0 ? Sh::emb : Sh::mul[l1];
                    a[n++] = {l1, l2, l3, path_index(l1, l2, l3), mul, roff, 0};
                    roff += mul;
                }
        for (int l3 = 0; l3 <= Sh::l_max; ++l3) {
            int c = 0;
            for (int k = 0; k < n; ++k)
                if (a[k].l3 == l3) {
                    a[k].choff = c;
                    c += a[k].mul;
                }
        }
        return a;
    }
    static constexpr std::array<Ins, 64> ins = make();
    static constexpr int count() { int c = 0; for (const Ins& i : ins) c += i.mul > 0; return c; }
    static constexpr int n = count();
    static constexpr int mulm(int l3) {
        int c = 0;
        for (int k = 0; k < n; ++k) c += ins[k].l3 == l3 ? ins[k].mul : 0;
        return c;
    }
    static constexpr long mblk(int l3) { long o = 0; for (int q = 0; q < l3; ++q) o += (2*q + 1)*mulm(q); return o; }
    static constexpr int functions() { int c = 0; for (int k = 0; k < n; ++k) c += ins[k].mul; return c; }
    static constexpr int nf = functions();
    static constexpr int msg = mblk(Sh::l_max + 1);
};
template <class Sh> constexpr int mulm(int t, int l)
{
    return t == 0 ? Layer<Sh, 0>::mulm(l) : t < Sh::num_layers - 1 ? Layer<Sh, 1>::mulm(l) : Layer<Sh, 2>::mulm(l);
}
template <class Sh> constexpr long mblk(int t, int l)
{
    return t == 0 ? Layer<Sh, 0>::mblk(l) : t < Sh::num_layers - 1 ? Layer<Sh, 1>::mblk(l) : Layer<Sh, 2>::mblk(l);
}

template <class Sh, int LT, int G> struct TP {
    using Ly = Layer<Sh, LT>;
    static constexpr int MUL = LT == 0 ? Sh::emb : Sh::mul[G];
    static constexpr int NIN = 2*G + 1;
    static constexpr int count_k() { int c = 0; for (int k = 0; k < Ly::n; ++k) c += Ly::ins[k].l1 == G; return c; }
    static constexpr int NK = count_k();
    struct K { int l2, l3, path, roff, choff, out; };
    static constexpr std::array<K, NK> make_ks() {
        std::array<K, NK> a{};
        int c = 0;
        int out = 0;
        for (int k = 0; k < Ly::n; ++k)
            if (Ly::ins[k].l1 == G) {
                const Ins& i = Ly::ins[k];
                a[c++] = {i.l2, i.l3, i.path, i.roff, i.choff, out};
                out += 2*i.l3 + 1;
            }
        return a;
    }
    static constexpr std::array<K, NK> ks = make_ks();
    static constexpr int NOUT = ks[NK - 1].out + 2*ks[NK - 1].l3 + 1;
    struct GT { int k, q, m1, lm2, a; };
    static constexpr int count_t() { int c = 0; for (const K& k : ks) c += paths[k.path].num; return c; }
    static constexpr int NT = count_t();
    static constexpr std::array<GT, NT> make_gt() {
        std::array<GT, NT> a{};
        int c = 0;
        for (int k = 0; k < NK; ++k)
            for (int q = paths[ks[k].path].first; q < paths[ks[k].path].first + paths[ks[k].path].num; ++q)
                a[c++] = {k, q, terms[q].m1, ks[k].l2*ks[k].l2 + terms[q].m2, ks[k].out + terms[q].m3};
        return a;
    }
    static constexpr std::array<GT, NT> gt = make_gt();
    struct E { int k, m1, i; };
    struct Entries { int n; std::array<E, NT> e; std::array<int, NT> of; };
    static constexpr Entries make_entries(bool qform) {
        Entries r{};
        for (int t = 0; t < NT; ++t) {
            const E key{gt[t].k, gt[t].m1, qform ? gt[t].lm2 : gt[t].a};
            bool found = false;
            for (int d = 0; d < r.n; ++d)
                found = found or (r.e[d].k == key.k and r.e[d].m1 == key.m1 and r.e[d].i == key.i);
            if (not found) r.e[r.n++] = key;
        }
        for (int a = 1; a < r.n; ++a)
            for (int b = a; b > 0; --b) {
                const E x = r.e[b - 1];
                const E y = r.e[b];
                if (x.k < y.k or (x.k == y.k and (x.m1 < y.m1 or (x.m1 == y.m1 and x.i < y.i)))) break;
                r.e[b - 1] = y;
                r.e[b] = x;
            }
        for (int t = 0; t < NT; ++t)
            for (int d = 0; d < r.n; ++d)
                if (r.e[d].k == gt[t].k and r.e[d].m1 == gt[t].m1 and r.e[d].i == (qform ? gt[t].lm2 : gt[t].a))
                    r.of[t] = d;
        return r;
    }
    static constexpr Entries me = make_entries(false);
    static constexpr Entries qe = make_entries(true);
    static constexpr int DM = me.n;
    static constexpr int DMP = (DM + 3)/4*4;
    static constexpr int NQ = qe.n;
    static constexpr int NQ4 = (NQ + 3)/4*4;
    static constexpr std::array<int, NK + 1> make_first(int which) {
        std::array<int, NK + 1> f{};
        const int n = which == 0 ? DM : which == 1 ? NQ : NT;
        int d = 0;
        for (int k = 0; k <= NK; ++k) {
            while (d < n and (which == 0 ? me.e[d].k : which == 1 ? qe.e[d].k : gt[d].k) < k) ++d;
            f[k] = d;
        }
        return f;
    }
    static constexpr std::array<int, NK + 1> me_first = make_first(0);
    static constexpr std::array<int, NK + 1> qe_first = make_first(1);
    static constexpr std::array<int, NK + 1> gt_first = make_first(2);
};

template <class T> struct Model {
    int num_elements, intervals;
    T h, half_h, inv_h;
    const T* R[6];
    const __half2* Rh[6];
    float Rs[6];
    const float4* Rg01[6][5];
    const float4* Rg23[6][5];
    const T* X0;
    const T *readout_weights, *per_type_scale, *per_type_shift;
    T gate_cst;
    int has_zbl;
    double r_cut, zbl_p;
    const double *zbl_a, *zbl_c;
};

struct Graph {
    int N, NR;
    const int* elem;
    const int* start;
    const int* neigh_rows;
};

template <class T> struct Work {
    const T* geometry;
    const int* interval;
    const T* xs;
    T* H[2];
    T* X[6];
    T* A[6];
    T *Phi, *A_linear, *A_sc, *A_adj, *X_adj, *H_adj_linear, *H_adj_sc, *edge_forces;
    double* node_energies;
};

}  // namespace nequip_foundation

namespace {

using json = nlohmann::json;
using namespace nequip_foundation;

template <int N, class F> __host__ __device__ __forceinline__ void unroll(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f.template operator()<I>(), ...);
    }(std::make_integer_sequence<int, N>{});
}

__constant__ double cg_double[num_terms];
const void* cg_double_owner = nullptr;
template <class T, int q, int lm> __device__ __forceinline__ T cg() {
    constexpr float c = float(terms[q].c*sh_norm(lm));
    if constexpr (std::is_same_v<T, float>) return c;
    else return cg_double[q];
}

template <class T> struct V4 { T x, y, z, w; };
template <class T> __device__ __forceinline__ V4<T> load4(const T* p) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 v = *reinterpret_cast<const float4*>(p);
        return {v.x, v.y, v.z, v.w};
    } else {
        const double2 a = reinterpret_cast<const double2*>(p)[0];
        const double2 b = reinterpret_cast<const double2*>(p)[1];
        return {a.x, a.y, b.x, b.y};
    }
}
template <class T> __device__ __forceinline__ void store4(T* p, V4<T> v) {
    if constexpr (std::is_same_v<T, float>) *reinterpret_cast<float4*>(p) = make_float4(v.x, v.y, v.z, v.w);
    else {
        reinterpret_cast<double2*>(p)[0] = make_double2(v.x, v.y);
        reinterpret_cast<double2*>(p)[1] = make_double2(v.z, v.w);
    }
}
template <class T, int N> __device__ __forceinline__ void load_n(const T* p, T* v) {
#pragma unroll
    for (int k = 0; k < N/4; ++k) {
        const V4<T> q = load4(p + 4*k);
        v[4*k] = q.x;
        v[4*k+1] = q.y;
        v[4*k+2] = q.z;
        v[4*k+3] = q.w;
    }
}
template <class T, int N> __device__ __forceinline__ void store_n(T* p, const T* v) {
#pragma unroll
    for (int k = 0; k < N/4; ++k) store4(p + 4*k, V4<T>{v[4*k], v[4*k+1], v[4*k+2], v[4*k+3]});
}

template <int l_max, class T> __device__ __forceinline__ void solid_harmonics(T x, T y, T z, T* S) {
    const T xx = x*x;
    const T yy = y*y;
    const T zz = z*z;
    const T r2 = xx + yy + zz;
    const T xy = x*y;
    const T dxy = xx - yy;
    const T f5 = fma(T(5), zz, -r2);
    S[0] = 1;
    S[1] = y;
    S[2] = z;
    S[3] = x;
    if constexpr (l_max >= 2) { S[4] = xy; S[5] = y*z; S[6] = fma(T(3), zz, -r2); S[7] = x*z; S[8] = dxy; }
    if constexpr (l_max >= 3) {
        S[9] = y*fma(T(3), xx, -yy);
        S[10] = xy*z;
        S[11] = y*f5;
        S[12] = z*fma(T(5), zz, -3*r2);
        S[13] = x*f5;
        S[14] = z*dxy;
        S[15] = x*fma(T(-3), yy, xx);
    }
    if constexpr (l_max >= 4) {
        const T s2 = xx + yy;
        const T g7 = fma(T(7), zz, -r2);
        const T g73 = fma(T(7), zz, -3*r2);
        S[16] = xy*dxy;
        S[17] = y*z*fma(T(3), xx, -yy);
        S[18] = xy*g7;
        S[19] = y*z*g73;
        S[20] = fma(8*zz, zz - 3*s2, 3*s2*s2);
        S[21] = x*z*g73;
        S[22] = dxy*g7;
        S[23] = x*z*fma(T(-3), yy, xx);
        S[24] = fma(xx, xx - 6*yy, yy*yy);
    }
}

template <class T>
__host__ __device__ constexpr T grad_c(int e) {
    return T(grad_terms[e].c*sh_norm(grad_terms[e].mp)/sh_norm(grad_terms[e].lm));
}

template <int l_max, class T> __device__ __forceinline__ void grad_direct(const T* g, const T* S, T* Gd) {
    constexpr int NQ = l_max*l_max;
    T q[3][NQ] = {};
    unroll<num_grad_terms(l_max)>([&]<int e>() {
        constexpr GradTerm t = grad_terms[e];
        constexpr T c = grad_c<T>(e);
        q[t.d][t.mp] = fma(c, g[t.lm], q[t.d][t.mp]);
    });
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        T acc = 0;
#pragma unroll
        for (int mp = 0; mp < NQ; ++mp) acc = fma(q[d][mp], S[mp], acc);
        Gd[d] = acc;
    }
}
template <int l_max, class T> __device__ __forceinline__ void grad_factors(const T* S, T* D) {
    unroll<num_grad_terms(l_max)>([&]<int e>() {
        constexpr GradTerm t = grad_terms[e];
        constexpr T c = grad_c<T>(e);
        D[e] = c*S[t.mp];
    });
}
template <int l_max, class T> __device__ __forceinline__ void grad_staged(const T* g, const T* D, T* Gd) {
    Gd[0] = 0;
    Gd[1] = 0;
    Gd[2] = 0;
    unroll<num_grad_terms(l_max)>([&]<int e>() {
        constexpr GradTerm t = grad_terms[e];
        Gd[t.d] = fma(g[t.lm], D[e], Gd[t.d]);
    });
}

template <class T> __device__ __forceinline__ void warp_add3(T* dst, T f0, T f1, T f2) {
    const int lane = threadIdx.x & 31;
    const bool hi = lane & 16;
    const bool b8 = lane & 8;
    const T k0 = hi ? f2 : f0;
    const T k1 = hi ? T(0) : f1;
    const T s0 = hi ? f0 : f2;
    const T s1 = hi ? f1 : T(0);
    const T a = k0 + __shfl_xor_sync(~0u, s0, 16);
    const T b = k1 + __shfl_xor_sync(~0u, s1, 16);
    T c = (b8 ? b : a) + __shfl_xor_sync(~0u, b8 ? a : b, 8);
    c += __shfl_xor_sync(~0u, c, 4);
    c += __shfl_xor_sync(~0u, c, 2);
    c += __shfl_xor_sync(~0u, c, 1);
    if ((lane & 7) == 0 and lane < 24) atomicAdd(dst + (lane >> 3), c);
}
template <class T> __device__ __forceinline__ void force_solid(T a, V4<T> q, const T* Gd, T* f) {
    f[0] = -fma(a, q.y*q.w, Gd[1]);
    f[1] = -fma(a, q.z*q.w, Gd[2]);
    f[2] = -fma(a, q.x*q.w, Gd[0]);
}
template <class T> __device__ __forceinline__ void warp_force4(T* dst, T a, const T* Gd, V4<T> q, T as) {
    const int lane = threadIdx.x & 31;
    const bool hi = lane & 16;
    const bool b8 = lane & 8;
    const T k0 = hi ? Gd[1] : a;
    const T k1 = hi ? Gd[2] : Gd[0];
    const T s0 = hi ? a : Gd[1];
    const T s1 = hi ? Gd[0] : Gd[2];
    const T x = k0 + __shfl_xor_sync(~0u, s0, 16);
    const T y = k1 + __shfl_xor_sync(~0u, s1, 16);
    T c = (b8 ? y : x) + __shfl_xor_sync(~0u, b8 ? x : y, 8);
    c += __shfl_xor_sync(~0u, c, 4);
    c += __shfl_xor_sync(~0u, c, 2);
    c += __shfl_xor_sync(~0u, c, 1);
    const T A = __shfl_sync(~0u, c, 0)*as;
    const int v = lane >> 3;
    if ((lane & 7) == 0 and v > 0)
        atomicAdd(dst + (v == 1 ? 2 : v - 2), -fma(A, (v == 1 ? q.x : v == 2 ? q.y : q.z)*q.w, c));
}

template <class T, bool H> struct Spl {
    const T* c;
    __device__ __forceinline__ T value(int f, T x) const {
        const V4<T> k = load4(c + 4*f);
        return k.x + x*(k.y + x*(k.z + x*k.w));
    }
    __device__ __forceinline__ void deriv(int f, T x, T& v, T& d) const {
        const V4<T> k = load4(c + 4*f);
        v = k.x + x*(k.y + x*(k.z + x*k.w));
        d = k.y + x*(2*k.z + 3*x*k.w);
    }
};
template <> struct Spl<float, true> {
    const float* c01;
    const __half2* c23;
    int nf;
    float s;
    __device__ __forceinline__ float value(int f, float x) const {
        const float xs = x*s;
        const float2 c = __half22float2(c23[f]);
        return c01[f] + x*(c01[nf + f] + xs*(c.x + x*c.y));
    }
    __device__ __forceinline__ void deriv(int f, float x, float& v, float& d) const {
        const float xs = x*s;
        const float2 c = __half22float2(c23[f]);
        const float c1 = c01[nf + f];
        v = c01[f] + x*(c1 + xs*(c.x + x*c.y));
        d = c1 + xs*(2*c.x + 3*x*c.y);
    }
};
template <class T, bool H>
__device__ __forceinline__ Spl<T, H> spline_block(const Model<T>& M, int layer, int iv, int nf) {
    if constexpr (H) return {M.R[layer] + (long)iv*2*nf, M.Rh[layer] + (long)iv*nf, nf, M.Rs[layer]};
    else return {M.R[layer] + (long)iv*4*nf};
}

template <class TPg, class T, class B> __device__ __forceinline__ void radial(const B& b, int u, T x, T* R) {
    unroll<TPg::NK>([&]<int k>() { constexpr int r = TPg::ks[k].roff; R[k] = b.value(r + u, x); });
}
template <class TPg, class T, class B> __device__ __forceinline__ void radial(const B& b, int u, T x, T* R, T* Rd) {
    unroll<TPg::NK>([&]<int k>() { constexpr int r = TPg::ks[k].roff; b.deriv(r + u, x, R[k], Rd[k]); });
}
template <class TPg>
__device__ __forceinline__ void radial_grp(const float4* g01, const float4* g23, int iv, int u, float x, float s,
                                           float* R, float* Rd) {
    constexpr int NK = TPg::NK;
    constexpr int MUL = TPg::MUL;
    constexpr int NP = (NK + 1)/2;
    constexpr int NQ = (NK + 3)/4;
    const float4* p01 = g01 + (long)iv*NP*MUL + u;
    const float4* p23 = g23 + (long)iv*NQ*MUL + u;
    float c0[2*NP], c1[2*NP];
    float2 c23[4*NQ];
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        const float4 v = p01[j*MUL];
        c0[2*j] = v.x;
        c1[2*j] = v.y;
        c0[2*j+1] = v.z;
        c1[2*j+1] = v.w;
    }
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        const float4 w = p23[j*MUL];
        const __half2* h = reinterpret_cast<const __half2*>(&w);
#pragma unroll
        for (int t = 0; t < 4; ++t) c23[4*j + t] = __half22float2(h[t]);
    }
    const float xs = x*s;
#pragma unroll
    for (int k = 0; k < NK; ++k) {
        R[k] = c0[k] + x*(c1[k] + xs*(c23[k].x + x*c23[k].y));
        if (Rd) Rd[k] = c1[k] + xs*(2*c23[k].x + 3*x*c23[k].y);
    }
}

template <class TPg, class T> __device__ __forceinline__ void tp_forward(T* acc, const T* R, const T* S, const T* x) {
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        acc[t.a] = fma(R[t.k], cg<T, t.q, t.lm2>()*S[t.lm2]*x[t.m1], acc[t.a]);
    });
}
template <class TPg, class T>
__device__ __forceinline__ void tp_reverse(const T* adj, const T* R, const T* Rd, const T* S, const T* x, T* gx,
                                           T* gS, T& a) {
    T sk[TPg::NK] = {};
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        const T c = cg<T, t.q, t.lm2>();
        const T u = c*S[t.lm2];
        const T w = adj[t.a]*R[t.k];
        sk[t.k] = fma(adj[t.a], x[t.m1]*u, sk[t.k]);
        gx[t.m1] = fma(u, w, gx[t.m1]);
        gS[t.lm2] = fma(c*x[t.m1], w, gS[t.lm2]);
    });
#pragma unroll
    for (int k = 0; k < TPg::NK; ++k) a = fma(Rd[k], sk[k], a);
}
template <class TPg, class T> __device__ __forceinline__ void tp_coupling(const T* S, T* Msm) {
    T m[TPg::DMP] = {};
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        constexpr int d = TPg::me.of[i];
        m[d] = fma(cg<T, t.q, t.lm2>(), S[t.lm2], m[d]);
    });
    store_n<T, TPg::DMP>(Msm, m);
}
template <int d, class T> __device__ __forceinline__ T coupling(const T* Msm, V4<T>& v) {
    if constexpr (d%4 == 0) v = load4(Msm + d);
    if constexpr (d%4 == 0) return v.x;
    else if constexpr (d%4 == 1) return v.y;
    else if constexpr (d%4 == 2) return v.z;
    else return v.w;
}
template <class TPg, class T>
__device__ __forceinline__ void tp_forward2(T* acc, const T* R, const T* Msm, const T* x) {
    V4<T> v;
    unroll<TPg::NK>([&]<int k>() {
        constexpr int o = TPg::ks[k].out;
        constexpr int n = 2*TPg::ks[k].l3 + 1;
        constexpr int d0 = TPg::me_first[k];
        T t[n] = {};
        unroll<TPg::me_first[k + 1] - d0>([&]<int j>() {
            constexpr auto e = TPg::me.e[d0 + j];
            t[e.i - o] = fma(coupling<d0 + j>(Msm, v), x[e.m1], t[e.i - o]);
        });
#pragma unroll
        for (int m = 0; m < n; ++m) acc[o + m] = fma(R[k], t[m], acc[o + m]);
    });
}
template <class TPg, class T>
__device__ __forceinline__ void tp_reverse2(const T* adj, const T* R, const T* Rd, const T* Msm, const T* x,
                                            T* gx, T* gS, T& a) {
    V4<T> v;
    unroll<TPg::NK>([&]<int k>() {
        constexpr int d0 = TPg::me_first[k];
        constexpr int t0 = TPg::gt_first[k];
        T q[TPg::NIN] = {};
        unroll<TPg::me_first[k + 1] - d0>([&]<int j>() {
            constexpr auto e = TPg::me.e[d0 + j];
            q[e.m1] = fma(coupling<d0 + j>(Msm, v), adj[e.i], q[e.m1]);
        });
        T sk = 0;
#pragma unroll
        for (int m1 = 0; m1 < TPg::NIN; ++m1) {
            sk = fma(x[m1], q[m1], sk);
            gx[m1] = fma(R[k], q[m1], gx[m1]);
        }
        a = fma(Rd[k], sk, a);
        unroll<TPg::gt_first[k + 1] - t0>([&]<int j>() {
            constexpr auto t = TPg::gt[t0 + j];
            gS[t.lm2] = fma(cg<T, t.q, t.lm2>()*x[t.m1], adj[t.a]*R[k], gS[t.lm2]);
        });
    });
}
template <class TPg, class T> __device__ __forceinline__ void tp_q_build(const T* adj, T* Q) {
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        constexpr int d = TPg::qe.of[i];
        Q[d] = fma(cg<T, t.q, t.lm2>(), adj[t.a], Q[d]);
    });
}
template <class TPg, class T>
__device__ __forceinline__ void tp_q_reverse(const T* Q, const T* S, const T* R, const T* Rd, const T* x, T* gx,
                                             T* gS, T& a) {
    unroll<TPg::NK>([&]<int k>() {
        constexpr int d0 = TPg::qe_first[k];
        T q[TPg::NIN] = {};
        unroll<TPg::qe_first[k + 1] - d0>([&]<int j>() {
            constexpr auto e = TPg::qe.e[d0 + j];
            q[e.m1] = fma(S[e.i], Q[d0 + j], q[e.m1]);
            gS[e.i] = fma(Q[d0 + j], R[k]*x[e.m1], gS[e.i]);
        });
        T sk = 0;
#pragma unroll
        for (int m1 = 0; m1 < TPg::NIN; ++m1) {
            sk = fma(x[m1], q[m1], sk);
            gx[m1] = fma(R[k], q[m1], gx[m1]);
        }
        a = fma(Rd[k], sk, a);
    });
}

template <class Sh, int LT, int G> __device__ __forceinline__ long msg_index(long N, int i, int u, int o) {
    using TPg = TP<Sh, LT, G>;
    using Ly = typename TPg::Ly;
    long r = 0;
    unroll<TPg::NK>([&]<int k>() {
        constexpr auto K = TPg::ks[k];
        constexpr long mb = Ly::mblk(K.l3);
        constexpr int mm = Ly::mulm(K.l3);
        if (o >= K.out and o <= K.out + 2*K.l3) r = N*mb + ((long)i*(2*K.l3 + 1) + o - K.out)*mm + K.choff + u;
    });
    return r;
}
template <class Sh, int LT, int G, class T>
__device__ __forceinline__ void msg_store(T* M, long N, int i, int u, const T* acc, bool atomic) {
    using TPg = TP<Sh, LT, G>;
    using Ly = typename TPg::Ly;
    unroll<TPg::NK>([&]<int k>() {
        constexpr auto K = TPg::ks[k];
        constexpr long mb = Ly::mblk(K.l3);
        constexpr int mm = Ly::mulm(K.l3);
#pragma unroll
        for (int m = 0; m <= 2*K.l3; ++m) {
            T* p = M + N*mb + ((long)i*(2*K.l3 + 1) + m)*mm + K.choff + u;
            if (atomic) atomicAdd(p, acc[K.out + m]);
            else *p = acc[K.out + m];
        }
    });
}
template <class Sh, int LT, int G, class T>
__device__ __forceinline__ void msg_load(const T* M, long N, int i, int u, T* adj) {
    using TPg = TP<Sh, LT, G>;
    using Ly = typename TPg::Ly;
    unroll<TPg::NK>([&]<int k>() {
        constexpr auto K = TPg::ks[k];
        constexpr long mb = Ly::mblk(K.l3);
        constexpr int mm = Ly::mulm(K.l3);
#pragma unroll
        for (int m = 0; m <= 2*K.l3; ++m) adj[K.out + m] = M[N*mb + ((long)i*(2*K.l3 + 1) + m)*mm + K.choff + u];
    });
}

template <class Sh, int LT, int G, bool BIG> struct LPN {
    static constexpr int cp = LT == 0 ? Sh::cpt0 : not BIG ? 1 : Sh::cpt[LT - 1][G];
    static constexpr int m = TP<Sh, LT, G>::MUL/cp;
    static constexpr int v = (m + 31)/32*32;
    static constexpr int npb = v >= 128 ? 1 : 128/v;
    static constexpr int tpb = v*npb;
};

template <class T> struct EdgeB { int j, iv; T xs; V4<T> q; };
template <class T> __device__ __forceinline__ EdgeB<T> edge_load(const Graph& G, const Work<T>& W, int e) {
    return {G.neigh_rows[e], W.interval[e], W.xs[e], load4(W.geometry + 4*e)};
}
template <class T> __device__ __forceinline__ EdgeB<T> edge_batch(const Graph& G, const Work<T>& W, int e0, int e1) {
    const int e = e0 + (threadIdx.x & 31);
    if (e < e1) return edge_load(G, W, e);
    return {0, 0, T(0), {T(0), T(0), T(0), T(1)}};
}
template <class T> __device__ __forceinline__ EdgeB<T> edge_bcast(const EdgeB<T>& b, int c) {
    return {__shfl_sync(~0u, b.j, c), __shfl_sync(~0u, b.iv, c), __shfl_sync(~0u, b.xs, c),
            {__shfl_sync(~0u, b.q.x, c), __shfl_sync(~0u, b.q.y, c), __shfl_sync(~0u, b.q.z, c),
             __shfl_sync(~0u, b.q.w, c)}};
}

template <class Sh, class T, bool H, bool BIG>
__global__ void __launch_bounds__(LPN<Sh, 0, 0, BIG>::tpb) k_tp0(Model<T> M, Graph G, Work<T> W) {
    using TPg = TP<Sh, 0, 0>;
    using L = LPN<Sh, 0, 0, BIG>;
    const int i = blockIdx.x*L::npb + threadIdx.x/L::v;
    const int u = threadIdx.x%L::v;
    if (i >= G.N) return;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    const bool on = u < LL;
    T acc[CP][TPg::NOUT] = {};
    const int es0 = G.start[i];
    const int ecnt = G.start[i+1] - es0;
    const int ea = es0 + (int)((long)ecnt*blockIdx.y/gridDim.y);
    const int eb = es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y);
    for (int e0 = ea; e0 < eb; e0 += 32) {
        const EdgeB<T> b = edge_batch(G, W, e0, eb);
        const int nb = min(32, eb - e0);
#pragma unroll Sh::tp_unroll
        for (int c = 0; c < nb; ++c) {
            const EdgeB<T> ed = edge_bcast(b, c);
            T S[Sh::num_lm];
            solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
            const int tj = G.elem[ed.j];
            const auto blk = spline_block<T, false>(M, 0, ed.iv, NF);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[1] = {on ? M.X0[tj*Sh::emb + uc] : T(0)};
                T R[TPg::NK] = {};
                if (on) radial<TPg>(blk, uc, xs, R);
                tp_forward<TPg>(acc[cc], R, S, x);
            }
        }
    }
    if (on)
#pragma unroll
        for (int cc = 0; cc < CP; ++cc) msg_store<Sh, 0, 0>(W.Phi, G.N, i, u + cc*LL, acc[cc], gridDim.y > 1);
}

template <class Sh, class T, bool H, bool BIG>
__global__ void __launch_bounds__(LPN<Sh, 0, 0, BIG>::tpb) k_tp0_reverse(Model<T> M, Graph G, Work<T> W) {
    using TPg = TP<Sh, 0, 0>;
    using L = LPN<Sh, 0, 0, BIG>;
    const int i = blockIdx.x*L::npb + threadIdx.x/L::v;
    const int u = threadIdx.x%L::v;
    if (i >= G.N) return;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    const bool on = u < LL;
    T adj[CP][TPg::NOUT] = {};
    if (on)
#pragma unroll
        for (int cc = 0; cc < CP; ++cc) msg_load<Sh, 0, 0>(W.Phi, G.N, i, u + cc*LL, adj[cc]);
    const int es0 = G.start[i];
    const int ecnt = G.start[i+1] - es0;
    const int ea = es0 + (int)((long)ecnt*blockIdx.y/gridDim.y);
    const int eb = es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y);
    for (int e0 = ea; e0 < eb; e0 += 32) {
        const EdgeB<T> b = edge_batch(G, W, e0, eb);
        const int nb = min(32, eb - e0);
#pragma unroll Sh::tp_unroll
        for (int c = 0; c < nb; ++c) {
            const int e = e0 + c;
            const EdgeB<T> ed = edge_bcast(b, c);
            T S[Sh::num_lm];
            solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
            const int tj = G.elem[ed.j];
            const auto blk = spline_block<T, false>(M, 0, ed.iv, NF);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
            T gS[Sh::num_lm] = {};
            T a = 0;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[1] = {on ? M.X0[tj*Sh::emb + uc] : T(0)};
                T R[TPg::NK] = {};
                T Rd[TPg::NK] = {};
                T gx[1] = {};
                if (on) radial<TPg>(blk, uc, xs, R, Rd);
                tp_reverse<TPg>(adj[cc], R, Rd, S, x, gx, gS, a);
            }
            T Gd[3], f[3];
            grad_direct<Sh::l_max>(gS, S, Gd);
            force_solid(a*M.inv_h, ed.q, Gd, f);
            warp_add3(W.edge_forces + 3*e, f[0], f[1], f[2]);
        }
    }
}

template <class Sh, int G, bool BIG, class T> struct SMC {
    using TPg = TP<Sh, 1, G>;
    static constexpr int WPB = LPN<Sh, 1, G, BIG>::tpb/32;
    static constexpr int SP = (Sh::num_lm + 3)/4*4;
    static constexpr int fit(int ch) {
        return ch > 1 and WPB*ch*(TPg::DMP + SP + (Sh::smem_edge ? 8 : 0))*(int)sizeof(T) > 49152 ? fit(ch/2) : ch;
    }
    static constexpr int CH = fit(BIG ? Sh::smem_ch_big : Sh::smem_ch);
    static constexpr int PER_WARP = CH*(TPg::DMP + SP + (Sh::smem_edge ? 8 : 0));
};
template <class Sh, int G, bool BIG, class T>
__device__ __forceinline__ void smem_chunk(const Graph& G_, const Work<T>& W, int e0, int ne, T* Msm, T* Ssm) {
    using TPg = TP<Sh, 1, G>;
    using C = SMC<Sh, G, BIG, T>;
    const int lane = threadIdx.x & 31;
    __syncwarp();
    if (lane < ne) {
        const int e = e0 + lane;
        const V4<T> q = load4(W.geometry + 4*e);
        T S[C::SP] = {};
        solid_harmonics<Sh::l_max>(q.x, q.y, q.z, S);
        tp_coupling<TPg>(S, Msm + lane*TPg::DMP);
        store_n<T, C::SP>(Ssm + lane*C::SP, S);
        if constexpr (Sh::smem_edge) {
            T* Ed = Ssm + C::CH*C::SP + lane*8;
            int* Ei = reinterpret_cast<int*>(Ed);
            Ei[0] = G_.neigh_rows[e];
            Ei[1] = W.interval[e];
            Ed[2] = W.xs[e];
            store4(Ed + 4, q);
        }
    }
    __syncwarp();
}
template <class Sh, int G, bool BIG, class T>
__device__ __forceinline__ EdgeB<T> smem_edge(const Graph& G_, const Work<T>& W, const T* Ssm, int e, int c) {
    using C = SMC<Sh, G, BIG, T>;
    if constexpr (Sh::smem_edge) {
        const T* Ed = Ssm + C::CH*C::SP + c*8;
        const int* Ei = reinterpret_cast<const int*>(Ed);
        return {Ei[0], Ei[1], Ed[2], load4(Ed + 4)};
    }
    return edge_load(G_, W, e);
}

template <class Sh, class T, bool H, int G, bool BIG>
__global__ void __launch_bounds__(LPN<Sh, 1, G, BIG>::tpb) k_tp1(Model<T> M, Graph G_, Work<T> W, int layer) {
    using TPg = TP<Sh, 1, G>;
    using L = LPN<Sh, 1, G, BIG>;
    using C = SMC<Sh, G, BIG, T>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    __shared__ __align__(16) T smem[C::WPB*C::PER_WARP];
    const int i = blockIdx.x*L::npb + threadIdx.x/L::v;
    const int u = threadIdx.x%L::v;
    if (i >= G_.N) return;
    T* Msm = smem + (threadIdx.x >> 5)*C::PER_WARP;
    T* Ssm = Msm + C::CH*TPg::DMP;
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T acc[CP][TPg::NOUT] = {};
    constexpr bool split = not BIG;
    const int es0 = G_.start[i];
    const int e1 = G_.start[i+1];
    const int ecnt = e1 - es0;
    const int ea = split ? es0 + (int)((long)ecnt*blockIdx.y/gridDim.y) : es0;
    const int eb = split ? es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y) : e1;
    for (int e0 = ea; e0 < eb; e0 += C::CH) {
        const int ne = min(C::CH, eb - e0);
        smem_chunk<Sh, G, BIG>(G_, W, e0, ne, Msm, Ssm);
#pragma unroll (((Sh::fwd_unroll >> G) & 1) ? 2 : 1)
        for (int c = 0; c < ne; ++c) {
            const EdgeB<T> ed = smem_edge<Sh, G, BIG>(G_, W, Ssm, e0 + c, c);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m) x[m] = on ? X[((long)ed.j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) {
                    if constexpr (H and Sh::grp)
                        radial_grp<TPg>(M.Rg01[layer][G], M.Rg23[layer][G], ed.iv, uc, xs, M.Rs[layer], R, nullptr);
                    else radial<TPg>(spline_block<T, H>(M, layer, ed.iv, NF), uc, xs, R);
                }
                tp_forward2<TPg>(acc[cc], R, Msm + c*TPg::DMP, x);
            }
        }
    }
    if (on)
#pragma unroll
        for (int cc = 0; cc < CP; ++cc)
            msg_store<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, acc[cc], split and gridDim.y > 1);
}

template <class Sh, class T, bool H, int G, bool BIG>
__global__ void __launch_bounds__(LPN<Sh, 1, G, BIG>::tpb) k_tp1_reverse(Model<T> M, Graph G_, Work<T> W, int layer) {
    using TPg = TP<Sh, 1, G>;
    using L = LPN<Sh, 1, G, BIG>;
    using C = SMC<Sh, G, BIG, T>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    __shared__ __align__(16) T smem[C::WPB*C::PER_WARP];
    const int i = blockIdx.x*L::npb + threadIdx.x/L::v;
    const int u = threadIdx.x%L::v;
    if (i >= G_.N) return;
    T* Msm = smem + (threadIdx.x >> 5)*C::PER_WARP;
    T* Ssm = Msm + C::CH*TPg::DMP;
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T* X_adj = W.X_adj + NX*Sh::xblk(G);
    T adj[CP][TPg::NOUT] = {};
    if (on)
#pragma unroll
        for (int cc = 0; cc < CP; ++cc) msg_load<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, adj[cc]);
    constexpr bool split = not BIG;
    const int es0 = G_.start[i];
    const int e1 = G_.start[i+1];
    const int ecnt = e1 - es0;
    const int ea = split ? es0 + (int)((long)ecnt*blockIdx.y/gridDim.y) : es0;
    const int eb = split ? es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y) : e1;
    for (int e0 = ea; e0 < eb; e0 += C::CH) {
        const int ne = min(C::CH, eb - e0);
        smem_chunk<Sh, G, BIG>(G_, W, e0, ne, Msm, Ssm);
#pragma unroll (((Sh::rev_unroll >> G) & 1) ? 2 : 1)
        for (int c = 0; c < ne; ++c) {
            const int e = e0 + c;
            const EdgeB<T> ed = smem_edge<Sh, G, BIG>(G_, W, Ssm, e, c);
            const T* S = Ssm + c*C::SP;
            const T xs = (ed.xs - M.half_h)*M.inv_h;
            T gS[Sh::num_lm] = {};
            T a = 0;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
                T Rd[TPg::NK] = {};
                T gx[TPg::NIN] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m) x[m] = on ? X[((long)ed.j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) {
                    if constexpr (H and Sh::grp)
                        radial_grp<TPg>(M.Rg01[layer][G], M.Rg23[layer][G], ed.iv, uc, xs, M.Rs[layer], R, Rd);
                    else radial<TPg>(spline_block<T, H>(M, layer, ed.iv, NF), uc, xs, R, Rd);
                }
                tp_reverse2<TPg>(adj[cc], R, Rd, Msm + c*TPg::DMP, x, gx, gS, a);
                if (on)
#pragma unroll
                    for (int m = 0; m < TPg::NIN; ++m)
                        atomicAdd(X_adj + ((long)ed.j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
            }
            T Gd[3], f[3];
            grad_direct<Sh::l_max>(gS, S, Gd);
            force_solid(a*M.inv_h, ed.q, Gd, f);
            warp_add3(W.edge_forces + 3*e, f[0], f[1], f[2]);
        }
    }
}

template <class Sh, int G, bool BIG, class T> struct SMQ {
    using L = LPN<Sh, 1, G, BIG>;
    static constexpr int SP = (Sh::num_lm + 3)/4*4;
    static constexpr int ND = (num_grad_terms(Sh::l_max) + 3)/4*4;
    static constexpr int TPB = 128 >= L::v ? 128/L::v*L::v : L::v;
    static constexpr int NPB = TPB/L::v;
    static constexpr int WPB = TPB/32;
    static constexpr int fit(int ch) { return ch > 1 and WPB*ch*(SP + 8 + ND)*(int)sizeof(T) > 49152 ? fit(ch/2) : ch; }
    static constexpr int CH = fit(32);
    static constexpr int PER_WARP = CH*(SP + 8 + ND);
};
template <class Sh, class T, bool H, int G, bool BIG>
__global__ void __launch_bounds__(SMQ<Sh, G, BIG, T>::TPB, (Sh::qrev_minb >> (4*G)) & 15)
k_tp1_reverse_q(Model<T> M, Graph G_, Work<T> W, int layer) {
    using TPg = TP<Sh, 1, G>;
    using L = LPN<Sh, 1, G, BIG>;
    using C = SMQ<Sh, G, BIG, T>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    __shared__ __align__(16) T smem[C::WPB*C::PER_WARP];
    const int i = blockIdx.x*C::NPB + threadIdx.x/L::v;
    const int u = threadIdx.x%L::v;
    if (i >= G_.N) return;
    T* Ssm = smem + (threadIdx.x >> 5)*C::PER_WARP;
    T* Esm = Ssm + C::CH*C::SP;
    T* Dsm = Esm + C::CH*8;
    const int lane = threadIdx.x & 31;
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T* X_adj = W.X_adj + NX*Sh::xblk(G);
    T Q[CP][TPg::NQ4] = {};
#pragma unroll
    for (int cc = 0; cc < CP; ++cc) {
        T adj[TPg::NOUT] = {};
        if (on) msg_load<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, adj);
        tp_q_build<TPg>(adj, Q[cc]);
    }
    constexpr bool split = not BIG;
    const int es0 = G_.start[i];
    const int e1 = G_.start[i+1];
    const int ecnt = e1 - es0;
    const int ea = split ? es0 + (int)((long)ecnt*blockIdx.y/gridDim.y) : es0;
    const int eb = split ? es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y) : e1;
    for (int e0 = ea; e0 < eb; e0 += C::CH) {
        const int ne = min(C::CH, eb - e0);
        __syncwarp();
        if (lane < ne) {
            const int e = e0 + lane;
            const V4<T> q = load4(W.geometry + 4*e);
            T S[C::SP] = {};
            solid_harmonics<Sh::l_max>(q.x, q.y, q.z, S);
            store_n<T, C::SP>(Ssm + lane*C::SP, S);
            T* Ed = Esm + lane*8;
            int* Ei = reinterpret_cast<int*>(Ed);
            Ei[0] = G_.neigh_rows[e];
            Ei[1] = W.interval[e];
            Ed[2] = W.xs[e];
            store4(Ed + 4, q);
            T D[C::ND] = {};
            grad_factors<Sh::l_max>(S, D);
            store_n<T, C::ND>(Dsm + lane*C::ND, D);
        }
        __syncwarp();
#pragma unroll (((Sh::qrev_unroll >> G) & 1) ? 2 : 1)
        for (int c = 0; c < ne; ++c) {
            const int e = e0 + c;
            const T* Ed = Esm + c*8;
            const int* Ei = reinterpret_cast<const int*>(Ed);
            const int j = Ei[0];
            const V4<T> q = load4(Ed + 4);
            T S[C::SP];
            load_n<T, C::SP>(Ssm + c*C::SP, S);
            S[0] = 1;
            const T xs = (Ed[2] - M.half_h)*M.inv_h;
            T gS[Sh::num_lm] = {};
            T a = 0;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
                T Rd[TPg::NK] = {};
                T gx[TPg::NIN] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m) x[m] = on ? X[((long)j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) {
                    if constexpr (H and Sh::grp)
                        radial_grp<TPg>(M.Rg01[layer][G], M.Rg23[layer][G], Ei[1], uc, xs, M.Rs[layer], R, Rd);
                    else radial<TPg>(spline_block<T, H>(M, layer, Ei[1], NF), uc, xs, R, Rd);
                }
                tp_q_reverse<TPg>(Q[cc], S, R, Rd, x, gx, gS, a);
                if (on)
#pragma unroll
                    for (int m = 0; m < TPg::NIN; ++m) atomicAdd(X_adj + ((long)j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
            }
            T D[C::ND];
            load_n<T, C::ND>(Dsm + c*C::ND, D);
            T Gd[3];
            grad_staged<Sh::l_max>(gS, D, Gd);
            warp_force4(W.edge_forces + 3*e, a, Gd, q, M.inv_h);
        }
    }
}

__device__ __forceinline__ bool pair_owner(int i, int j, float x, float y, float z) {
    if (i != j) return (i < j) != (((i ^ j) & 1) != 0);
    return x > 0 or (x == 0 and (y > 0 or (y == 0 and z > 0)));
}

template <class Sh, class T, bool H, int G, bool BIG>
__device__ __forceinline__ void pair_forward(const Model<T>& M, const Graph& G_, const Work<T>& W, int layer,
                                             int i, int u) {
    using TPg = TP<Sh, 2, G>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = LPN<Sh, 2, G, BIG>::cp;
    constexpr int LL = TPg::MUL/CP;
    const long N = G_.N;
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T acc[CP][TPg::NOUT] = {};
    T xi[CP][TPg::NIN];
#pragma unroll
    for (int c = 0; c < CP; ++c)
#pragma unroll
        for (int m = 0; m < TPg::NIN; ++m) xi[c][m] = on ? X[((long)i*TPg::NIN + m)*TPg::MUL + u + c*LL] : T(0);
    const int es0 = G_.start[i];
    const int ecnt = G_.start[i+1] - es0;
    const int ea = es0 + (int)((long)ecnt*blockIdx.y/gridDim.y);
    const int eb = es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y);
    for (int e0 = ea; e0 < eb; e0 += 32) {
        const EdgeB<T> b = edge_batch(G_, W, e0, eb);
        const int nb = min(32, eb - e0);
        for (int c = 0; c < nb; ++c) {
            const EdgeB<T> ed = edge_bcast(b, c);
            const int j = ed.j;
            const bool gh = j >= G_.N;
            if (not gh and not pair_owner(i, j, ed.q.x, ed.q.y, ed.q.z)) continue;
            T S[Sh::num_lm], Sr[Sh::num_lm];
            solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
#pragma unroll
            for (int lm = 0; lm < Sh::num_lm; ++lm)
                Sr[lm] = (lm == 0 or (lm >= 4 and lm < 9) or (lm >= 16 and lm < 25)) ? S[lm] : -S[lm];
            const auto blk = spline_block<T, false>(M, layer, ed.iv, NF);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m) x[m] = on ? X[((long)j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) radial<TPg>(blk, uc, xs, R);
                tp_forward<TPg>(acc[cc], R, S, x);
                if (not gh) {
                    T t[TPg::NOUT] = {};
                    tp_forward<TPg>(t, R, Sr, xi[cc]);
                    if (on) atomicAdd(W.Phi + msg_index<Sh, 2, G>(N, j, uc, 0), t[0]);
                }
            }
        }
    }
    if (on)
#pragma unroll
        for (int c = 0; c < CP; ++c) atomicAdd(W.Phi + msg_index<Sh, 2, G>(N, i, u + c*LL, 0), acc[c][0]);
}

template <class Sh, class T, bool H, int G, bool BIG>
__device__ __forceinline__ void pair_reverse(const Model<T>& M, const Graph& G_, const Work<T>& W, int layer,
                                             int i, int u) {
    using TPg = TP<Sh, 2, G>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = LPN<Sh, 2, G, BIG>::cp;
    constexpr int LL = TPg::MUL/CP;
    const long N = G_.N;
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T* X_adj = W.X_adj + NX*Sh::xblk(G);
    T adji[CP][TPg::NOUT] = {};
    T xi[CP][TPg::NIN];
    T gxi[CP][TPg::NIN] = {};
#pragma unroll
    for (int c = 0; c < CP; ++c) {
        if (on) adji[c][0] = W.Phi[msg_index<Sh, 2, G>(N, i, u + c*LL, 0)];
#pragma unroll
        for (int m = 0; m < TPg::NIN; ++m) xi[c][m] = on ? X[((long)i*TPg::NIN + m)*TPg::MUL + u + c*LL] : T(0);
    }
    const int es0 = G_.start[i];
    const int ecnt = G_.start[i+1] - es0;
    const int ea = es0 + (int)((long)ecnt*blockIdx.y/gridDim.y);
    const int eb = es0 + (int)((long)ecnt*(blockIdx.y + 1)/gridDim.y);
    for (int e0 = ea; e0 < eb; e0 += 32) {
        const EdgeB<T> b = edge_batch(G_, W, e0, eb);
        const int nb = min(32, eb - e0);
        for (int c = 0; c < nb; ++c) {
            const int e = e0 + c;
            const EdgeB<T> ed = edge_bcast(b, c);
            const int j = ed.j;
            const bool gh = j >= G_.N;
            if (not gh and not pair_owner(i, j, ed.q.x, ed.q.y, ed.q.z)) continue;
            T S[Sh::num_lm], Sr[Sh::num_lm];
            solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
#pragma unroll
            for (int lm = 0; lm < Sh::num_lm; ++lm)
                Sr[lm] = (lm == 0 or (lm >= 4 and lm < 9) or (lm >= 16 and lm < 25)) ? S[lm] : -S[lm];
            const auto blk = spline_block<T, false>(M, layer, ed.iv, NF);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
            T gS[Sh::num_lm] = {};
            T gSr[Sh::num_lm] = {};
            T a = 0;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
                T Rd[TPg::NK] = {};
                T gx[TPg::NIN] = {};
                T adjj[TPg::NOUT] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m) x[m] = on ? X[((long)j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) {
                    radial<TPg>(blk, uc, xs, R, Rd);
                    if (not gh) adjj[0] = W.Phi[msg_index<Sh, 2, G>(N, j, uc, 0)];
                }
                tp_reverse<TPg>(adji[cc], R, Rd, S, x, gx, gS, a);
                if (not gh) tp_reverse<TPg>(adjj, R, Rd, Sr, xi[cc], gxi[cc], gSr, a);
                if (on)
#pragma unroll
                    for (int m = 0; m < TPg::NIN; ++m) atomicAdd(X_adj + ((long)j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
            }
#pragma unroll
            for (int lm = 0; lm < Sh::num_lm; ++lm)
                gS[lm] += (lm == 0 or (lm >= 4 and lm < 9) or (lm >= 16 and lm < 25)) ? gSr[lm] : -gSr[lm];
            T Gd[3], f[3];
            grad_direct<Sh::l_max>(gS, S, Gd);
            force_solid(a*M.inv_h, ed.q, Gd, f);
            warp_add3(W.edge_forces + 3*e, f[0], f[1], f[2]);
        }
    }
    if (on)
#pragma unroll
        for (int c = 0; c < CP; ++c)
#pragma unroll
            for (int m = 0; m < TPg::NIN; ++m) atomicAdd(X_adj + ((long)i*TPg::NIN + m)*TPg::MUL + u + c*LL, gxi[c][m]);
}

template <class Sh, class T, bool H, int G, bool BIG, bool REV>
__global__ void __launch_bounds__(LPN<Sh, 2, G, BIG>::tpb) k_tp2(Model<T> M, Graph G_, Work<T> W, int layer) {
    using L = LPN<Sh, 2, G, BIG>;
    const int i = blockIdx.x*L::npb + threadIdx.x/L::v;
    const int u = threadIdx.x%L::v;
    if (i >= G_.N) return;
    if constexpr (REV) pair_reverse<Sh, T, H, G, BIG>(M, G_, W, layer, i, u);
    else pair_forward<Sh, T, H, G, BIG>(M, G_, W, layer, i, u);
}

template <class Sh, bool BIG> struct PairFused {
    static constexpr int NG = Sh::l_max + 1;
    static constexpr bool fits() {
        bool r = true;
        for (int g = 0; g < NG; ++g) r = r and Sh::mul[g] <= 32*(BIG ? Sh::cpt[1][g] : 1);
        return r;
    }
    static constexpr bool ok = fits();
    static constexpr int NPB = NG >= 4 ? 1 : 4/NG;
    static constexpr int TPB = 32*NG*NPB;
};
template <class Sh, class T, bool H, bool BIG, bool REV>
__global__ void __launch_bounds__(PairFused<Sh, BIG>::TPB) k_tp2_fused(Model<T> M, Graph G_, Work<T> W, int layer) {
    using F = PairFused<Sh, BIG>;
    const int w = threadIdx.x >> 5;
    const int u = threadIdx.x & 31;
    const int i = blockIdx.x*F::NPB + w/F::NG;
    const int g = w%F::NG;
    if (i >= G_.N) return;
    unroll<F::NG>([&]<int G>() {
        if (g == G) {
            if constexpr (REV) pair_reverse<Sh, T, H, G, BIG>(M, G_, W, layer, i, u);
            else pair_forward<Sh, T, H, G, BIG>(M, G_, W, layer, i, u);
        }
    });
}

template <class T> __device__ __forceinline__ T sigm(T x) { return 1/(1 + exp(-x)); }
template <class Sh> __device__ __forceinline__ void gated_of(int k, int& l, int& c) {
    l = 1;
    c = k;
#pragma unroll
    for (int q = 1; q < Sh::l_max; ++q)
        if (l == q and c >= Sh::mulx(q)) {
            c -= Sh::mulx(q);
            l = q + 1;
        }
}
template <class Sh, class T, bool SC> __global__ void k_gate(Model<T> M, Graph G_, Work<T> W, int layer, int xout) {
    const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)G_.N*Sh::ngt) return;
    const int k = idx%Sh::ngt;
    const int i = idx/Sh::ngt;
    const long N = G_.N;
    T* A = W.A[layer];
    T* H = W.H[xout];
    const T cst = M.gate_cst;
    if (k < Sh::nsc) {
        const long os = (long)i*Sh::muly(0) + k;
        const T ys = W.A_linear[os] + (SC ? W.A_sc[os] : T(0));
        A[os] = ys;
        H[(long)i*Sh::nsc + k] = ys*sigm(ys)*cst;
    }
    if (k < Sh::ngate) {
        const long og = (long)i*Sh::muly(0) + Sh::nsc + k;
        const T yg = W.A_linear[og] + (SC ? W.A_sc[og] : T(0));
        A[og] = yg;
        const T g = yg*sigm(yg)*cst;
        int l, c;
        gated_of<Sh>(k, l, c);
        for (int m = 0; m < 2*l + 1; ++m) {
            const long oy = N*Sh::yblk(l) + ((long)i*(2*l + 1) + m)*Sh::muly(l) + c;
            const T v = W.A_linear[oy] + (SC ? W.A_sc[oy] : T(0));
            A[oy] = v;
            H[N*Sh::xblk(l) + ((long)i*(2*l + 1) + m)*Sh::mulx(l) + c] = v*g;
        }
    }
}
template <class Sh, class T> __global__ void k_gate_reverse(Model<T> M, Graph G_, Work<T> W, int layer) {
    const long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (long)G_.N*Sh::ngt) return;
    const int k = idx%Sh::ngt;
    const int i = idx/Sh::ngt;
    const long N = G_.N;
    const T* A = W.A[layer];
    const T cst = M.gate_cst;
    if (k < Sh::nsc) {
        const long os = (long)i*Sh::muly(0) + k;
        const T ys = A[os];
        const T s = sigm(ys);
        const long ox = (long)i*Sh::nsc + k;
        W.A_adj[os] = (W.H_adj_linear[ox] + W.H_adj_sc[ox])*cst*s*(1 + ys*(1 - s));
    }
    if (k < Sh::ngate) {
        const long og = (long)i*Sh::muly(0) + Sh::nsc + k;
        const T yg = A[og];
        const T sg = sigm(yg);
        const T g = yg*sg*cst;
        int l, c;
        gated_of<Sh>(k, l, c);
        T ga = 0;
        for (int m = 0; m < 2*l + 1; ++m) {
            const long oy = N*Sh::yblk(l) + ((long)i*(2*l + 1) + m)*Sh::muly(l) + c;
            const long ox = N*Sh::xblk(l) + ((long)i*(2*l + 1) + m)*Sh::mulx(l) + c;
            const T xa = W.H_adj_linear[ox] + W.H_adj_sc[ox];
            W.A_adj[oy] = xa*g;
            ga = fma(xa, A[oy], ga);
        }
        W.A_adj[og] = ga*cst*sg*(1 + yg*(1 - sg));
    }
}
template <class Sh, class T> __global__ void __launch_bounds__(Sh::nsc) k_tail(Model<T> M, Graph G_, Work<T> W) {
    __shared__ T part[Sh::nsc/32];
    const int i = blockIdx.x;
    const int k = threadIdx.x;
    const int t = G_.elem[i];
    const long o = (long)i*Sh::nsc + k;
    const T y = W.A_linear[o] + W.A_sc[o];
    const T s = sigm(y);
    const T cst = M.gate_cst;
    T v = M.readout_weights[k]*y*s*cst;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
    if ((k & 31) == 0) part[k >> 5] = v;
    __syncthreads();
    if (k == 0) {
        T sum = 0;
        for (int w = 0; w < Sh::nsc/32; ++w) sum += part[w];
        W.node_energies[i] += (double)M.per_type_scale[t]*(double)sum + (double)M.per_type_shift[t];
    }
    W.A_adj[o] = M.per_type_scale[t]*M.readout_weights[k]*cst*s*(1 + y*(1 - s));
}

constexpr int SORT_MAX = 256;

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("NequIPFoundation: ") + what + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what)
{
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("NequIPFoundation: ") + what + ": cuBLAS error " + std::to_string(s));
}
#define CHECK(x) check((x), #x)

int blocks(long n, int t) { return int((n + t - 1)/t); }

template <class Sh, class T>
__device__ __forceinline__ double edge_record(const Model<T>& M, int e, int ti, int tj, int row, int nb, double dx,
                                              double dy, double dz, int* neigh_rows, int* neigh_indices, double* vec,
                                              T* geometry, int* interval, T* xs, T* edge_forces) {
    const double r = sqrt(dx*dx + dy*dy + dz*dz);
    neigh_rows[e] = row;
    neigh_indices[e] = nb;
    vec[3*e] = dx;
    vec[3*e+1] = dy;
    vec[3*e+2] = dz;
    store4(geometry + 4*e, V4<T>{T(dz), T(dx), T(dy), T(1/r)});
    const int iv = min(int(r/M.h), M.intervals - 1);
    interval[e] = iv;
    xs[e] = T(r - M.h*iv);
    double E = 0;
    double dE = 0;
    if (M.has_zbl) {
        const int tp = ti*M.num_elements + tj;
        const double a = M.zbl_a[tp];
        const double q = a*r;
        const double e1 = 0.02817*exp(-0.20162*q);
        const double e2 = 0.28022*exp(-0.40290*q);
        const double e3 = 0.50986*exp(-0.94229*q);
        const double e4 = 0.18175*exp(-3.19980*q);
        const double psi = e1 + e2 + e3 + e4;
        const double dpsi = a*(-0.20162*e1 - 0.40290*e2 - 0.94229*e3 - 3.19980*e4);
        const double p = M.zbl_p;
        const double u = r/M.r_cut;
        double fc = 0;
        double dfc = 0;
        if (u < 1) {
            const double up = pow(u, p);
            fc = 1 - (p + 1)*(p + 2)/2*up + p*(p + 2)*up*u - p*(p + 1)/2*up*u*u;
            dfc = (-(p + 1)*(p + 2)/2*p*up/u + p*(p + 2)*(p + 1)*up - p*(p + 1)/2*(p + 2)*up*u)/M.r_cut;
        }
        const double c = M.zbl_c[tp];
        E = c/r*psi*fc;
        dE = c*(-psi*fc/(r*r) + dpsi*fc/r + psi*dfc/r);
    }
    edge_forces[3*e] = T(-dE*dx/r);
    edge_forces[3*e+1] = T(-dE*dy/r);
    edge_forces[3*e+2] = T(-dE*dz/r);
    return E;
}

__global__ void k_last_start(int N, const int* count, int* start) {
    if (threadIdx.x == 0) start[N] = N > 0 ? start[N - 1] + count[N - 1] : 0;
}

template <class T>
__global__ void k_outputs(Graph G, const double* node_energies, const double* vec, const T* edge_forces, double* out) {
    __shared__ double part[7][4];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    double v[7] = {};
    if (n < G.N) {
        if (lane == 0) v[0] = node_energies[n];
        for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
            const double x = vec[3*e];
            const double y = vec[3*e+1];
            const double z = vec[3*e+2];
            const double fx = edge_forces[3*e];
            const double fy = edge_forces[3*e+1];
            const double fz = edge_forces[3*e+2];
            v[1] += x*fx;
            v[2] += y*fy;
            v[3] += z*fz;
            v[4] += 0.5*(x*fy + y*fx);
            v[5] += 0.5*(x*fz + z*fx);
            v[6] += 0.5*(y*fz + z*fy);
        }
    }
    for (int c = 0; c < 7; ++c) {
        for (int o = 16; o > 0; o >>= 1) v[c] += __shfl_xor_sync(~0u, v[c], o);
        if (lane == 0) part[c][warp] = v[c];
    }
    __syncthreads();
    if (threadIdx.x < 7)
        atomicAdd(out + threadIdx.x,
                  part[threadIdx.x][0] + part[threadIdx.x][1] + part[threadIdx.x][2] + part[threadIdx.x][3]);
}

__global__ void k_atom_energies(int N, const int* atom, const double* node_energies, double* eatom) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n < N) eatom[atom[n]] = node_energies[n];
}

void gemm(cublasHandle_t h, bool trans, int m, int n, int k, const float* A, int lda, const float* B, int ldb,
          float* C, int ldc)
{
    const float one = 1;
    const float zero = 0;
    CHECK(cublasSgemm(h, trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, B, ldb, A, lda, &zero, C, ldc));
}
void gemm(cublasHandle_t h, bool trans, int m, int n, int k, const double* A, int lda, const double* B, int ldb,
          double* C, int ldc)
{
    const double one = 1;
    const double zero = 0;
    CHECK(cublasDgemm(h, trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, B, ldb, A, lda, &zero, C, ldc));
}

__host__ __device__ __forceinline__ void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
#ifdef __CUDA_ARCH__
    hi = (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
    lo = (__float_as_uint(x - __uint_as_float(hi)) + 0x1000u) & 0xffffe000u;
#else
    uint32_t b;
    std::memcpy(&b, &x, 4);
    hi = (b + 0x1000u) & 0xffffe000u;
    float hf;
    std::memcpy(&hf, &hi, 4);
    const float r = x - hf;
    std::memcpy(&lo, &r, 4);
    lo = (lo + 0x1000u) & 0xffffe000u;
#endif
}

struct TcDesc { const float* A; const uint32_t* Bh; const uint32_t* Bl; float* C; int M, N, K, lda, ldbt, ldc, tile0; };
struct TcBatch { TcDesc g[32]; int n; };
constexpr int TC_BK = 16;
constexpr int TC_SKEW = 20;
template <int BM, int BN> constexpr int tc_smem = 3*(BM + 2*BN)*TC_SKEW*4;

template <int BM, int BN, int WARPS_M, int WARPS_N>
__global__ void __launch_bounds__(32*WARPS_M*WARPS_N) k_gemm_tf32x3(const TcBatch batch) {
#if __CUDA_ARCH__ >= 800
    constexpr int NT = 32*WARPS_M*WARPS_N;
    constexpr int WM = BM/WARPS_M;
    constexpr int WN = BN/WARPS_N;
    constexpr int TM = WM/16;
    constexpr int TN = WN/8;
    constexpr int ST = 3;
    extern __shared__ __align__(16) unsigned char tc_shared[];
    float (*As)[BM][TC_SKEW] = reinterpret_cast<float (*)[BM][TC_SKEW]>(tc_shared);
    uint32_t (*Bh)[BN][TC_SKEW] = reinterpret_cast<uint32_t (*)[BN][TC_SKEW]>(tc_shared + 4*ST*BM*TC_SKEW);
    uint32_t (*Bl)[BN][TC_SKEW] = reinterpret_cast<uint32_t (*)[BN][TC_SKEW]>(tc_shared + 4*ST*(BM + BN)*TC_SKEW);
    int gi = 0;
    while (gi + 1 < batch.n and batch.g[gi + 1].tile0 <= (int)blockIdx.x) ++gi;
    const TcDesc d = batch.g[gi];
    const int tile = blockIdx.x - d.tile0;
    const int tiles_n = d.N/BN;
    const int m0 = (tile/tiles_n)*BM;
    const int n0 = (tile%tiles_n)*BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp/WARPS_N;
    const int wn = warp%WARPS_N;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    float acc[TM][TN][4] = {};
    float small[TM][TN][4] = {};
    const int ktiles = d.K/TC_BK;
    auto copy16 = [](void* dst, const void* src, bool pred) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(a), "l"(src), "r"(pred ? 16 : 0));
    };
    auto load_stage = [&](int s, int kt) {
        const int k0 = kt*TC_BK;
        for (int c = threadIdx.x; c < BM*4; c += NT) {
            const int r = c >> 2;
            const int q = c & 3;
            const int gr = m0 + r;
            copy16(&As[s][r][4*q], d.A + (size_t)(gr < d.M ? gr : 0)*d.lda + k0 + 4*q, gr < d.M);
        }
        for (int c = threadIdx.x; c < BN*4; c += NT) {
            const int r = c >> 2;
            const int q = c & 3;
            copy16(&Bh[s][r][4*q], d.Bh + (size_t)(n0 + r)*d.ldbt + k0 + 4*q, true);
            copy16(&Bl[s][r][4*q], d.Bl + (size_t)(n0 + r)*d.ldbt + k0 + 4*q, true);
        }
    };
    for (int s = 0; s < ST - 1; ++s) {
        if (s < ktiles) load_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    for (int kt = 0; kt < ktiles; ++kt) {
        asm volatile("cp.async.wait_group %0;" :: "n"(ST - 2));
        __syncthreads();
        if (kt + ST - 1 < ktiles) load_stage((kt + ST - 1)%ST, kt + ST - 1);
        asm volatile("cp.async.commit_group;");
        const int s = kt%ST;
#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 8) {
            float part[TM][TN][4] = {};
            uint32_t ah[TM][4], al[TM][4], bh[TN][2], bl[TN][2];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int r = wm*WM + 16*i + gid;
                split_tf32(As[s][r][kk + tig], ah[i][0], al[i][0]);
                split_tf32(As[s][r + 8][kk + tig], ah[i][1], al[i][1]);
                split_tf32(As[s][r][kk + tig + 4], ah[i][2], al[i][2]);
                split_tf32(As[s][r + 8][kk + tig + 4], ah[i][3], al[i][3]);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int c = wn*WN + 8*j + gid;
                bh[j][0] = Bh[s][c][kk + tig];
                bl[j][0] = Bl[s][c][kk + tig];
                bh[j][1] = Bh[s][c][kk + tig + 4];
                bl[j][1] = Bl[s][c][kk + tig + 4];
            }
            auto mma = [](float* c, const uint32_t* a, const uint32_t* b) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]),
                             "r"(a[3]), "r"(b[0]), "r"(b[1]));
            };
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    mma(small[i][j], al[i], bh[j]);
                    mma(small[i][j], ah[i], bl[j]);
                    mma(part[i][j], ah[i], bh[j]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][j][q] += part[i][j][q];
                }
        }
    }
    asm volatile("cp.async.wait_group 0;");
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = m0 + wm*WM + 16*i + gid;
            const int c = n0 + wn*WN + 8*j + 2*tig;
            if (r < d.M)
                *reinterpret_cast<float2*>(d.C + (size_t)r*d.ldc + c)
                    = make_float2(acc[i][j][0] + small[i][j][0], acc[i][j][1] + small[i][j][1]);
            if (r + 8 < d.M)
                *reinterpret_cast<float2*>(d.C + (size_t)(r + 8)*d.ldc + c)
                    = make_float2(acc[i][j][2] + small[i][j][2], acc[i][j][3] + small[i][j][3]);
        }
#endif
}

template <int BM, int BN> int tc_tiles(TcBatch& b) {
    int tiles = 0;
    for (int i = 0; i < b.n; ++i) {
        b.g[i].tile0 = tiles;
        tiles += ((b.g[i].M + BM - 1)/BM)*(b.g[i].N/BN);
    }
    return tiles;
}
template <int BM, int BN, int WARPS_M, int WARPS_N> void tc_launch(TcBatch& b, cudaStream_t s) {
    const int tiles = tc_tiles<BM, BN>(b);
    if (tiles) k_gemm_tf32x3<BM, BN, WARPS_M, WARPS_N><<<tiles, 32*WARPS_M*WARPS_N, tc_smem<BM, BN>, s>>>(b);
}
void tc_gemm(TcBatch& b, cudaStream_t s)
{
    bool n32 = true;
    bool n64 = true;
    bool n96 = true;
    for (int i = 0; i < b.n; ++i) {
        n32 = n32 and b.g[i].N%32 == 0;
        n64 = n64 and b.g[i].N%64 == 0;
        n96 = n96 and b.g[i].N%96 == 0;
    }
    const int big = n64 ? tc_tiles<128, 64>(b) : n96 ? tc_tiles<128, 96>(b) : tc_tiles<128, 32>(b);
    if (not n32) tc_launch<32, 16, 1, 1>(b, s);
    else if (big < 108) tc_launch<32, 32, 1, 1>(b, s);
    else if (n64) tc_launch<128, 64, 4, 2>(b, s);
    else if (n96) tc_launch<128, 96, 4, 3>(b, s);
    else tc_launch<128, 32, 4, 1>(b, s);
}

template <class Sh, int LT, bool BIG> int edge_split(int N) {
    if (BIG or N <= 0) return 1;
    int s = 1;
    unroll<LT == 0 ? 1 : Sh::l_max + 1>([&]<int G>() {
        const long warps = (long)N*(LPN<Sh, LT, G, BIG>::v/32);
        s = std::max(s, int(std::max(1L, std::min(8L, 6912/std::max(1L, warps)))));
    });
    return s;
}

template <class Sh, class T, bool H, int LT, bool REV, bool BIG>
void launch_tp(const Model<T>& M, const Graph& G, const Work<T>& W, int layer, cudaStream_t s,
               const std::array<cudaStream_t, 3>& side, cudaEvent_t fork, const std::array<cudaEvent_t, 3>& join)
{
    const int N = G.N;
    if (not REV and (LT == 2 or edge_split<Sh, LT, BIG>(N) > 1))
        CHECK(cudaMemsetAsync(W.Phi, 0, (size_t)N*Layer<Sh, LT>::msg*sizeof(T), s));
    if constexpr (LT == 2 and PairFused<Sh, BIG>::ok) {
        using F = PairFused<Sh, BIG>;
        const int es = BIG ? 1 : (int)std::max(1L, std::min(8L, 6912/std::max(1L, (long)N*F::NG)));
        k_tp2_fused<Sh, T, H, BIG, REV><<<dim3(blocks(N, F::NPB), es), F::TPB, 0, s>>>(M, G, W, layer);
        return;
    }
    constexpr int NG = LT == 0 ? 1 : Sh::l_max + 1;
    if (NG > 1) {
        CHECK(cudaEventRecord(fork, s));
        for (auto q : side) CHECK(cudaStreamWaitEvent(q, fork, 0));
    }
    unroll<NG>([&]<int G_>() {
        using L = LPN<Sh, LT, G_, BIG>;
        cudaStream_t st = G_ == 0 ? s : side[(G_ - 1)%3];
        int es = 1;
        if (not BIG) {
            const long warps = (long)N*(L::v/32);
            es = int(std::max(1L, std::min(8L, 6912/std::max(1L, warps))));
        }
        const dim3 grid(blocks(N, L::npb), es);
        if constexpr (LT == 0) {
            if constexpr (REV) k_tp0_reverse<Sh, T, H, BIG><<<grid, L::tpb, 0, st>>>(M, G, W);
            else k_tp0<Sh, T, H, BIG><<<grid, L::tpb, 0, st>>>(M, G, W);
        } else if constexpr (LT == 2) {
            k_tp2<Sh, T, H, G_, BIG, REV><<<grid, L::tpb, 0, st>>>(M, G, W, layer);
        } else if constexpr (REV and ((Sh::qrev >> G_) & 1)) {
            using C = SMQ<Sh, G_, BIG, T>;
            k_tp1_reverse_q<Sh, T, H, G_, BIG><<<dim3(blocks(N, C::NPB), grid.y), C::TPB, 0, st>>>(M, G, W, layer);
        } else if constexpr (REV) {
            k_tp1_reverse<Sh, T, H, G_, BIG><<<grid, L::tpb, 0, st>>>(M, G, W, layer);
        } else {
            k_tp1<Sh, T, H, G_, BIG><<<grid, L::tpb, 0, st>>>(M, G, W, layer);
        }
    });
    if (NG > 1)
        for (int q = 0; q < 3; ++q) {
            CHECK(cudaEventRecord(join[q], side[q]));
            CHECK(cudaStreamWaitEvent(s, join[q], 0));
        }
}

std::vector<double> spline(const std::vector<double>& v, const std::vector<double>& d, int l, double h, double r_low)
{
    const int K = int(v.size()) - 1;
    auto vt = [&](int i) { const double r = std::max(i*h, 0.5*h); return v[i]/std::pow(r, l); };
    auto dt = [&](int i) {
        const double r = std::max(i*h, 0.5*h);
        return d[i]/std::pow(r, l) - l*v[i]/std::pow(r, l + 1);
    };
    std::vector<double> c(4*K);
    const int low = int(r_low/h);
    for (int q = 0; q < K; ++q) {
        if (q < low) continue;
        const double v0 = vt(q);
        const double v1 = vt(q + 1);
        const double d0 = dt(q);
        const double d1 = dt(q + 1);
        const double k2 = (-3*v0 - 2*h*d0 + 3*v1 - h*d1)/(h*h);
        const double k3 = (2*v0 + h*d0 - 2*v1 + h*d1)/(h*h*h);
        const double m = 0.5*h;
        c[4*q] = v0 + m*(d0 + m*(k2 + m*k3));
        c[4*q+1] = (d0 + m*(2*k2 + 3*m*k3))*h;
        c[4*q+2] = (k2 + 3*m*k3)*h*h;
        c[4*q+3] = k3*h*h*h;
    }
    return c;
}

template <class Sh> void check_model(const json& js, std::vector<double>& coefficients, const std::string& prefix) {
    constexpr int NL = Sh::num_layers;
    auto fail = [&](const std::string& what) {
        throw std::runtime_error(prefix + "the model is not of its preset shape (" + what + ")");
    };
    coefficients.assign(num_terms, 0.0);
    std::vector<bool> seen(num_terms, false);
    const auto H_mul = js.at("H_mul").get<std::vector<std::vector<int>>>();
    const auto H_offset = js.at("H_offset").get<std::vector<std::vector<int>>>();
    const auto A_mul = js.at("A_mul").get<std::vector<std::vector<int>>>();
    const auto Phi_mul = js.at("Phi_mul").get<std::vector<std::vector<int>>>();
    const auto Phi_offset = js.at("Phi_offset").get<std::vector<std::vector<int>>>();
    for (int q = 0; q < NL; ++q) {
        const int LT = q == 0 ? 0 : q == NL - 1 ? 2 : 1;
        std::vector<int> in(2*Sh::l_max + 2, 0);
        std::vector<int> out(2*Sh::l_max + 2, 0);
        for (int l = 0; l <= Sh::l_max; ++l) {
            in[2*l] = LT == 0 ? (l == 0 ? Sh::emb : 0) : Sh::mulx(l);
            out[2*l] = LT == 2 ? (l == 0 ? Sh::nsc : 0) : Sh::muly(l);
        }
        if (H_mul[q] != in) fail("irreps_in of layer " + std::to_string(q));
        if (A_mul[q] != out) fail("irreps_out of layer " + std::to_string(q));
        const auto spec = js.at("Phi_spec")[q].get<std::vector<std::array<int, 5>>>();
        const auto cg = js.at("Phi_clebsch_gordan")[q].get<std::vector<double>>();
        unroll<3>([&]<int lt>() {
            if (lt != LT) return;
            using Ly = Layer<Sh, lt>;
            if (int(js.at("radial_spline_values")[q].size()) != Ly::nf)
                fail("tensor product of layer " + std::to_string(q));
            std::vector<int> matched(Ly::n, 0);
            for (int p = 0; p < int(spec.size()); ++p) {
                const auto [a, lm, w, b, n] = spec[p];
                int i1 = 0;
                while (H_offset[q][i1 + 1] <= a) ++i1;
                int i3 = 0;
                while (Phi_offset[q][i3 + 1] <= b) ++i3;
                int k = 0;
                while (k < Ly::n and Ly::ins[k].roff != w) ++k;
                if (k == Ly::n) fail("tensor-product instruction of layer " + std::to_string(q));
                const Ins& e = Ly::ins[k];
                const int l2 = int(std::sqrt(lm));
                if (i1 != 2*e.l1 or l2 != e.l2 or i3 != 2*e.l3 or n != e.mul
                    or (b - Phi_offset[q][i3]) % Phi_mul[q][i3] != e.choff)
                    fail("tensor-product instruction " + std::to_string(k) + " of layer " + std::to_string(q));
                const int m1 = (a - H_offset[q][i1])/H_mul[q][i1];
                const int m2 = lm - l2*l2;
                const int m3 = (b - Phi_offset[q][i3])/Phi_mul[q][i3];
                int t = paths[e.path].first;
                while (t < paths[e.path].first + paths[e.path].num
                       and (terms[t].m1 != m1 or terms[t].m2 != m2 or terms[t].m3 != m3))
                    ++t;
                const double c = cg[p];
                if (t == paths[e.path].first + paths[e.path].num) {
                    if (std::abs(c) > 1e-10) fail("tensor-product coefficients");
                    continue;
                }
                const double v = c*sh_norm(lm);
                if (seen[t] and std::abs(coefficients[t] - v) > 1e-12)
                    fail("tensor-product coefficients differ between layers");
                if (std::abs(c - terms[t].c) > 1e-6*std::abs(terms[t].c)) fail("tensor-product coefficients");
                coefficients[t] = v;
                seen[t] = true;
                matched[k] += 1;
            }
            for (int k = 0; k < Ly::n; ++k)
                if (matched[k] != paths[Ly::ins[k].path].num) fail("tensor-product coefficients");
        });
    }
}

template <class T> struct Buffer {
    T* p = nullptr;
    size_t n = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    ~Buffer() { cudaFree(p); }
    void reserve(size_t m) {
        if (m <= n and p) return;
        cudaFree(p);
        n = m + m/8 + 1;
        if (cudaMalloc(&p, n*sizeof(T)) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("NequIPFoundation: out of device memory");
        }
    }
    void upload(const std::vector<T>& v, cudaStream_t s) {
        reserve(v.size());
        cudaMemcpyAsync(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
    }
    operator T*() const { return p; }
};

}  // namespace

template <class Precision, bool mixed>
class NequIPFoundationCuda {

public:

struct Weight {
    Buffer<Precision> w;
    Buffer<uint32_t> hi, lo, thi, tlo;
};

NequIPFoundationCuda(const std::string& filename, const std::string& prefix);
~NequIPFoundationCuda();

// Basic model information
int num_elements;
int num_layers;
double r_cut;
int l_max;
int emb, mul0;
std::vector<int> atomic_numbers;

// Device
cudaStream_t stream = nullptr;
std::array<cudaStream_t, 3> side{};
cudaEvent_t ev_fork = nullptr;
std::array<cudaEvent_t, 3> ev_join{};
cublasContext* blas = nullptr;
Buffer<unsigned char> blas_workspace;
bool tensor_cores;

// Graph
int N = 0;
int NR = 0;
long E = 0;
std::vector<int> type_start;
Buffer<int> atom_of_node, row_of_atom, elem, count, start, neigh_indices, neigh_rows;
Buffer<unsigned short> cand;
Buffer<double> vec;
Buffer<Precision> geometry, edge_forces;
Buffer<unsigned char> scan_storage;
size_t scan_bytes = 0;
template <class Sh> void reserve(long);

// Node energies and forces
Buffer<double> node_energies, outputs;
Model<Precision> M{};
Graph G{};
Work<Precision> W{};
template <class Sh> void arguments();
template <class Sh> void compute_layer(int t);
template <class Sh> void compute_readout();
template <class Sh> void reverse_layer(int t);

// ZBL
bool has_zbl;
double zbl_p;
Buffer<double> zbl_a, zbl_c;

// Radial functions
int K;
double h;
Buffer<int> interval;
Buffer<Precision> xs;
std::vector<Buffer<Precision>> R;
std::vector<Buffer<__half2>> Rh;
std::vector<float> Rs;
std::vector<Buffer<float4>> Rg01, Rg23;

// Tensor products
std::vector<double> coefficients;
void upload_coefficients();
template <class Sh, bool REV> void tensor_product(int t);

// Layers
Buffer<Precision> X0;
Weight X_weights, A_weights, A_sc_weights;
std::vector<long> X_weights_offset, A_weights_offset, A_sc_weights_offset;
std::vector<Buffer<Precision>> X, A;
Buffer<Precision> H_even, H_odd, Phi, A_linear, A_sc, A_adj, X_adj, H_adj_linear, H_adj_sc;

// Gates
Precision gate_cst;

// Readout
Buffer<Precision> readout_weights, per_type_scale, per_type_shift;

// Matrix products
struct Product {
    const Precision* a;
    int lda;
    const Weight* w;
    long off;
    bool trans;
    Precision* c;
    int ldc;
    int m, n, k;
};
std::vector<Product> queue;
void product(const Precision*, int, const Weight&, long, bool, Precision*, int, int, int, int);
void products();

// Initializer
void load(const std::string&, const std::string&);

};

template <class Precision, bool mixed>
NequIPFoundationCuda<Precision, mixed>::NequIPFoundationCuda(const std::string& filename, const std::string& prefix)
{
    load(filename, prefix);
}

template <class Precision, bool mixed>
NequIPFoundationCuda<Precision, mixed>::~NequIPFoundationCuda()
{
    if (not blas) return;
    cublasDestroy(blas);
    cudaEventDestroy(ev_fork);
    for (int q = 0; q < 3; ++q) {
        cudaEventDestroy(ev_join[q]);
        cudaStreamDestroy(side[q]);
    }
    cudaStreamDestroy(stream);
}

template <class Precision, bool mixed>
void NequIPFoundationCuda<Precision, mixed>::load(const std::string& filename, const std::string& prefix)
{
    cudaFuncAttributes attributes;
    CHECK(cudaFuncGetAttributes(&attributes, k_gemm_tf32x3<128, 64, 4, 2>));
    tensor_cores = attributes.ptxVersion >= 80;
    if (mixed and tensor_cores) {
        CHECK(cudaFuncSetAttribute(k_gemm_tf32x3<128, 64, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   tc_smem<128, 64>));
        CHECK(cudaFuncSetAttribute(k_gemm_tf32x3<128, 96, 4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   tc_smem<128, 96>));
        CHECK(cudaFuncSetAttribute(k_gemm_tf32x3<128, 32, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   tc_smem<128, 32>));
    }
    std::ifstream file(filename);
    if (not file) throw std::runtime_error(prefix + "cannot open " + filename);
    const json js = json::parse(file);
    if (js.value("format_version", 0) != 3)
        throw std::runtime_error(prefix + filename + " was extracted by an incompatible symmetrix version; "
                                                     "extract it again");
    if (js.value("model_type", "") != "NequIP") throw std::runtime_error(prefix + filename + " is not a NequIP model");
    num_layers = js.at("num_layers");
    num_elements = js.at("num_elements");
    atomic_numbers = js.at("atomic_numbers").get<std::vector<int>>();
    r_cut = js.at("r_cut");
    gate_cst = Precision(js.at("gate_cst").get<double>());
    has_zbl = js.at("has_zbl").get<bool>();
    if (has_zbl) zbl_p = js.at("zbl_p");
    const auto embedding = js.at("embedding").get<std::vector<std::vector<double>>>();
    emb = int(embedding[0].size());
    l_max = js.at("l_max");
    mul0 = num_layers > 1 ? js.at("H_mul")[1][0].get<int>() : 0;
    const auto spl_values = js.at("radial_spline_values").get<std::vector<std::vector<std::vector<double>>>>();
    const auto spl_derivs = js.at("radial_spline_derivs").get<std::vector<std::vector<std::vector<double>>>>();
    const auto x_weights = js.at("X_weights").get<std::vector<std::vector<std::vector<double>>>>();
    const auto a_weights = js.at("A_weights").get<std::vector<std::vector<std::vector<double>>>>();
    const auto a_sc_weights = js.at("A_sc_weights").get<std::vector<std::vector<std::vector<double>>>>();
    const auto Phi_mul = js.at("Phi_mul").get<std::vector<std::vector<int>>>();
    const int T = num_elements;
    if (not with_preset(num_layers, l_max, emb, mul0, [&](auto s) {
                using Sh = decltype(s);
                constexpr int NL = Sh::num_layers;
                constexpr int LM = Sh::l_max;
                check_model<Sh>(js, coefficients, prefix);
                cg_double_owner = nullptr;
                K = int(spl_values[0][0].size()) - 1;
                h = r_cut/K;
                R.clear();
                R.resize(NL);
                Rh.clear();
                Rh.resize(NL);
                Rs.assign(NL, 1.0f);
                Rg01.clear();
                Rg01.resize(NL*(LM + 1));
                Rg23.clear();
                Rg23.resize(NL*(LM + 1));
                unroll<3>([&]<int LT>() {
                    if (LT == 1 and NL < 3) return;
                    using Ly = Layer<Sh, LT>;
                    for (int q = LT == 0 ? 0 : LT == 2 ? NL - 1 : 1; q < (LT == 0 ? 1 : LT == 2 ? NL : NL - 1); ++q) {
                        const int nf = Ly::nf;
                        std::vector<double> c((long)K*4*nf);
                        for (int k = 0; k < Ly::n; ++k)
                            for (int u = 0; u < Ly::ins[k].mul; ++u) {
                                const int fn = Ly::ins[k].roff + u;
                                const auto& v = spl_values[q][fn];
                                const auto& d = spl_derivs[q][fn];
                                if (int(v.size()) != K + 1)
                                    throw std::runtime_error(prefix + "radial splines with different knots");
                                const auto cc = spline(v, d, Ly::ins[k].l2, h, mixed and LT == 1 ? 0.5 : h);
                                for (int iv = 0; iv < K; ++iv)
                                    for (int a = 0; a < 4; ++a) c[((long)iv*nf + fn)*4 + a] = cc[4*iv + a];
                            }
                        if (not mixed or LT != 1) {
                            R[q].upload(std::vector<Precision>(c.begin(), c.end()), 0);
                            continue;
                        }
                        double worst = 0;
                        for (long i = 0; i < (long)K*nf; ++i)
                            worst = std::max({worst, std::abs(c[4*i + 2]), std::abs(c[4*i + 3])});
                        const double S = std::exp2(std::floor(std::log2(16384.0/std::max(worst, 1e-30))));
                        Rs[q] = float(1/S);
                        std::vector<Precision> c01((long)K*2*nf);
                        std::vector<__half2> c23((long)K*nf);
                        for (int iv = 0; iv < K; ++iv)
                            for (int fn = 0; fn < nf; ++fn) {
                                const double* p = &c[((long)iv*nf + fn)*4];
                                c01[((long)iv*2 + 0)*nf + fn] = Precision(p[0]);
                                c01[((long)iv*2 + 1)*nf + fn] = Precision(p[1]);
                                c23[(long)iv*nf + fn] = __floats2half2_rn(float(p[2]*S), float(p[3]*S));
                            }
                        R[q].upload(c01, 0);
                        Rh[q].upload(c23, 0);
                        if constexpr (LT == 1 and Sh::grp)
                            unroll<LM + 1>([&]<int G>() {
                                using TPg = TP<Sh, 1, G>;
                                constexpr int NK = TPg::NK;
                                constexpr int MUL = TPg::MUL;
                                constexpr int NP = (NK + 1)/2;
                                constexpr int NQ = (NK + 3)/4;
                                std::vector<float4> g01((long)K*NP*MUL);
                                std::vector<float4> g23((long)K*NQ*MUL);
                                for (int iv = 0; iv < K; ++iv)
                                    for (int u = 0; u < MUL; ++u) {
                                        for (int jj = 0; jj < NP; ++jj) {
                                            float v[4] = {0, 0, 0, 0};
                                            for (int t = 0; t < 2; ++t)
                                                if (2*jj + t < NK) {
                                                    const int fn = TPg::ks[2*jj + t].roff + u;
                                                    v[2*t] = float(c01[((long)iv*2 + 0)*nf + fn]);
                                                    v[2*t + 1] = float(c01[((long)iv*2 + 1)*nf + fn]);
                                                }
                                            g01[((long)iv*NP + jj)*MUL + u] = make_float4(v[0], v[1], v[2], v[3]);
                                        }
                                        for (int jj = 0; jj < NQ; ++jj) {
                                            __half2 hv[4];
                                            for (int t = 0; t < 4; ++t)
                                                hv[t] = 4*jj + t < NK ? c23[(long)iv*nf + TPg::ks[4*jj + t].roff + u]
                                                        : __floats2half2_rn(0.f, 0.f);
                                            std::memcpy(&g23[((long)iv*NQ + jj)*MUL + u], hv, sizeof(float4));
                                        }
                                    }
                                Rg01[q*(LM + 1) + G].upload(g01, 0);
                                Rg23[q*(LM + 1) + G].upload(g23, 0);
                            });
                    }
                });
                std::vector<std::array<long, 3>> b1, b2, bs;
                auto block = [&](std::vector<Precision>& w, std::vector<long>& off,
                                 std::vector<std::array<long, 3>>& blocks, const double* m, long rows, long cols) {
                    off.push_back(long(w.size()));
                    blocks.push_back({long(w.size()), rows, cols});
                    for (long x = 0; x < rows*cols; ++x) w.push_back(Precision(m[x]));
                };
                auto set_weight = [&](Weight& W, const std::vector<Precision>& w,
                                      const std::vector<std::array<long, 3>>& blocks) {
                    W.w.upload(w, 0);
                    if (not (mixed and tensor_cores)) return;
                    std::vector<uint32_t> hi(w.size());
                    std::vector<uint32_t> lo(w.size());
                    std::vector<uint32_t> thi(w.size());
                    std::vector<uint32_t> tlo(w.size());
                    for (const auto& [off, rows, cols] : blocks)
                        for (long r = 0; r < rows; ++r)
                            for (long c = 0; c < cols; ++c) {
                                split_tf32(float(w[off + r*cols + c]), hi[off + r*cols + c], lo[off + r*cols + c]);
                                split_tf32(float(w[off + r*cols + c]), thi[off + c*rows + r], tlo[off + c*rows + r]);
                            }
                    W.hi.upload(hi, 0);
                    W.lo.upload(lo, 0);
                    W.thi.upload(thi, 0);
                    W.tlo.upload(tlo, 0);
                };
                std::vector<Precision> w1, w2, ws;
                X_weights_offset.clear();
                A_weights_offset.clear();
                A_sc_weights_offset.clear();
                for (int q = 0; q < NL; ++q) {
                    const int LT = q == 0 ? 0 : q == NL - 1 ? 2 : 1;
                    for (int l = 0; l <= LM; ++l) {
                        if (q > 0) block(w1, X_weights_offset, b1, x_weights[q][2*l].data(), Sh::mulx(l), Sh::mulx(l));
                        else X_weights_offset.push_back(0);
                        if (LT == 2 and l > 0) {
                            A_weights_offset.push_back(0);
                            for (int t = 0; t < T; ++t) A_sc_weights_offset.push_back(0);
                            continue;
                        }
                        const long mid = Phi_mul[q][2*l];
                        const long cols = LT == 2 ? Sh::nsc : Sh::muly(l);
                        if (long(a_weights[q][2*l].size()) != mid*cols
                            or (q > 0 and long(a_sc_weights[q][2*l].size()) != T*Sh::mulx(l)*cols))
                            throw std::runtime_error(prefix + "linear_2 or self-connection of layer "
                                                     + std::to_string(q) + " does not match");
                        block(w2, A_weights_offset, b2, a_weights[q][2*l].data(), mid, cols);
                        for (int t = 0; t < T; ++t) {
                            if (q > 0)
                                block(ws, A_sc_weights_offset, bs, a_sc_weights[q][2*l].data() + t*Sh::mulx(l)*cols,
                                      Sh::mulx(l), cols);
                            else A_sc_weights_offset.push_back(0);
                        }
                    }
                }
                set_weight(X_weights, w1, b1);
                set_weight(A_weights, w2, b2);
                set_weight(A_sc_weights, ws, bs);
                std::vector<Precision> x((long)T*Sh::emb);
                for (int t = 0; t < T; ++t)
                    for (int v = 0; v < Sh::emb; ++v) {
                        double acc = 0;
                        for (int u = 0; u < Sh::emb; ++u) acc += embedding[t][u]*x_weights[0][0][u*Sh::emb + v];
                        x[(long)t*Sh::emb + v] = Precision(acc);
                    }
                X0.upload(x, 0);
            }))
        throw std::runtime_error(prefix + "the model (" + std::to_string(num_layers) + " layers, l_max "
                                 + std::to_string(l_max) + ", embedding " + std::to_string(emb) + ", "
                                 + std::to_string(mul0) + " scalars) is not of a preset shape: " + preset_names);
    auto reals = [&](const char* key) {
        std::vector<Precision> v;
        for (double a : js[key].get<std::vector<double>>()) v.push_back(Precision(a));
        return v;
    };
    readout_weights.upload(reals("readout_weights"), 0);
    per_type_scale.upload(reals("per_type_scale"), 0);
    per_type_shift.upload(reals("per_type_shift"), 0);
    {
        const double pref = has_zbl ? js.at("zbl_prefactor").get<double>() : 0.0;
        std::vector<double> za((long)T*T);
        std::vector<double> zc((long)T*T);
        for (int a = 0; a < T; ++a)
            for (int b = 0; b < T; ++b) {
                za[(long)a*T + b] = (std::pow(atomic_numbers[a], 0.23) + std::pow(atomic_numbers[b], 0.23))/0.46850;
                zc[(long)a*T + b] = pref*atomic_numbers[a]*atomic_numbers[b];
            }
        zbl_a.upload(za, 0);
        zbl_c.upload(zc, 0);
    }
    if (not blas) {
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int q = 0; q < 3; ++q) {
            CHECK(cudaStreamCreateWithFlags(&side[q], cudaStreamNonBlocking));
            CHECK(cudaEventCreateWithFlags(&ev_join[q], cudaEventDisableTiming));
        }
        CHECK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
        CHECK(cublasCreate(&blas));
        CHECK(cublasSetStream(blas, stream));
        blas_workspace.reserve(32 << 20);
        CHECK(cublasSetWorkspace(blas, blas_workspace, blas_workspace.n));
    }
    outputs.reserve(7);
}

template <class Precision, bool mixed>
void NequIPFoundationCuda<Precision, mixed>::upload_coefficients()
{
    if (std::is_same_v<Precision, double> and cg_double_owner != this) {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpyToSymbolAsync(cg_double, coefficients.data(), num_terms*sizeof(double), 0,
                                      cudaMemcpyHostToDevice, stream));
        cg_double_owner = this;
    }
}

template <class Precision, bool mixed>
template <class Sh>
void NequIPFoundationCuda<Precision, mixed>::reserve(long page)
{
    constexpr int NL = Sh::num_layers;
    using L0 = Layer<Sh, 0>;
    using L1 = Layer<Sh, 1>;
    using L2 = Layer<Sh, 2>;
    const long n = N;
    const long nr = NR;
    const long e = std::max<long>(E, 1);
    count.reserve(n + 1);
    start.reserve(n + 1);
    cand.reserve(n*SORT_MAX);
    neigh_indices.reserve(e);
    neigh_rows.reserve(e);
    vec.reserve(3*e);
    geometry.reserve(4*e);
    interval.reserve(e);
    xs.reserve(e);
    edge_forces.reserve(3*e);
    H_even.reserve(n*Sh::xw);
    H_odd.reserve(n*Sh::xw);
    X.resize(NL);
    A.resize(NL);
    for (int q = 1; q < NL; ++q) X[q].reserve(std::max(nr*Sh::xw, page));
    for (int q = 0; q < NL - 1; ++q) A[q].reserve(n*Sh::yw);
    Phi.reserve(n*std::max({L0::msg, NL > 2 ? L1::msg : 0, L2::msg}));
    A_linear.reserve(n*Sh::yw);
    A_sc.reserve(n*Sh::yw);
    A_adj.reserve(n*Sh::yw);
    X_adj.reserve(std::max(nr*Sh::xw, page));
    H_adj_linear.reserve(n*Sh::xw);
    H_adj_sc.reserve(n*Sh::xw);
    node_energies.reserve(n);
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, count.p, start.p, int(n), stream);
    scan_storage.reserve(bytes);
    scan_bytes = bytes;
}

template <class Precision, bool mixed>
template <class Sh>
void NequIPFoundationCuda<Precision, mixed>::arguments()
{
    using T = Precision;
    constexpr bool H = mixed;
    constexpr int NL = Sh::num_layers;
    constexpr int LM = Sh::l_max;
    M = {};
    M.num_elements = num_elements;
    M.intervals = K;
    M.h = T(h);
    M.half_h = T(0.5*h);
    M.inv_h = T(1/h);
    for (int q = 0; q < NL; ++q) {
        M.R[q] = R[q];
        if constexpr (H) {
            M.Rh[q] = Rh[q];
            M.Rs[q] = Rs[q];
            for (int l = 0; l <= LM and q > 0 and q < NL - 1 and Sh::grp; ++l) {
                M.Rg01[q][l] = Rg01[q*(LM + 1) + l];
                M.Rg23[q][l] = Rg23[q*(LM + 1) + l];
            }
        }
    }
    M.X0 = X0;
    M.readout_weights = readout_weights;
    M.per_type_scale = per_type_scale;
    M.per_type_shift = per_type_shift;
    M.gate_cst = gate_cst;
    M.has_zbl = has_zbl;
    M.r_cut = r_cut;
    M.zbl_p = zbl_p;
    M.zbl_a = zbl_a;
    M.zbl_c = zbl_c;
    G = {N, NR, elem, start, neigh_rows};
    W = {};
    W.geometry = geometry;
    W.interval = interval;
    W.xs = xs;
    W.H[0] = H_even;
    W.H[1] = H_odd;
    for (int q = 1; q < NL; ++q) W.X[q] = X[q];
    for (int q = 0; q < NL - 1; ++q) W.A[q] = A[q];
    W.Phi = Phi;
    W.A_linear = A_linear;
    W.A_sc = A_sc;
    W.A_adj = A_adj;
    W.X_adj = X_adj;
    W.H_adj_linear = H_adj_linear;
    W.H_adj_sc = H_adj_sc;
    W.edge_forces = edge_forces;
    W.node_energies = node_energies;
}

template <class Precision, bool mixed>
template <class Sh>
void NequIPFoundationCuda<Precision, mixed>::compute_layer(int t)
{
    constexpr int LM = Sh::l_max;
    constexpr int LL = Sh::num_layers - 1;
    const long n = N;
    const long nr = NR;
    const int cur = t & 1;

    // Phi
    // \Phi^{(t)}_{i,k\lambda m\eta} = \sum_j \sum_{m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) X^{(t)}_{j,k\lambda_1m_1},  \eta = (\lambda_1, l_2)
    tensor_product<Sh, false>(t);

    // A
    // A^{(t)}_{i,k\lambda m} = \sum_{k'\eta} w^{(t)}_{k\lambda,k'\eta} \Phi^{(t)}_{i,k'\lambda m\eta} + \sum_{k'} w^{(t)}_{a_ik\lambda,k'} H^{(t)}_{i,k'\lambda m}
    for (int l = 0; l <= (t == LL ? 0 : LM); ++l) {
        const int wy = t == LL ? Sh::nsc : Sh::muly(l);
        product(W.Phi + n*mblk<Sh>(t, l), mulm<Sh>(t, l), A_weights, A_weights_offset[t*(LM + 1) + l], false,
                W.A_linear + n*Sh::yblk(l), wy, N*(2*l + 1), wy, mulm<Sh>(t, l));
    }
    for (int l = 0; t > 0 and l <= (t == LL ? 0 : LM); ++l)
        for (int a = 0; a < num_elements; ++a) {
            const long r0 = type_start[a];
            const long rows = type_start[a + 1] - r0;
            const int wx = Sh::mulx(l);
            const int wy = t == LL ? Sh::nsc : Sh::muly(l);
            product(W.H[cur] + n*Sh::xblk(l) + r0*(2*l + 1)*wx, wx, A_sc_weights,
                    A_sc_weights_offset[(t*(LM + 1) + l)*num_elements + a], false,
                    W.A_sc + n*Sh::yblk(l) + r0*(2*l + 1)*wy, wy, int(rows*(2*l + 1)), wy, wx);
        }
    products();

    // gate
    // H^{(t+1)}_{i,k(0e)} = c_e \mathrm{SiLU}(A^{(t)}_{i,k(0e)})
    // H^{(t+1)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) A^{(t)}_{i,k\lambda m} (l > 0),  g_{k\lambda} = N^{(t+1)}_{0e} + \sum_{0o < \lambda' < \lambda} N^{(t+1)}_{\lambda'} + k
    if (t < LL and N > 0) {
        if (t == 0) k_gate<Sh, Precision, false><<<blocks(n*Sh::ngt, 128), 128, 0, stream>>>(M, G, W, t, 1 - cur);
        else k_gate<Sh, Precision, true><<<blocks(n*Sh::ngt, 128), 128, 0, stream>>>(M, G, W, t, 1 - cur);
    }

    // X
    // X^{(t+1)}_{i,k\lambda m} = \sum_{k'} w^{(t+1)}_{k\lambda,k'} H^{(t+1)}_{i,k'\lambda m}
    if (t < LL) {
        for (int l = 0; l <= LM; ++l)
            product(W.H[1 - cur] + n*Sh::xblk(l), Sh::mulx(l), X_weights, X_weights_offset[(t + 1)*(LM + 1) + l], false,
                    W.X[t + 1] + nr*Sh::xblk(l), Sh::mulx(l), N*(2*l + 1), Sh::mulx(l), Sh::mulx(l));
        products();
    }
}

template <class Precision, bool mixed>
template <class Sh>
void NequIPFoundationCuda<Precision, mixed>::compute_readout()
{
    // readout
    // E_i += \mu_{a_i} + \sigma_{a_i} \sum_k w_k H^{(T)}_{i,k},  H^{(T)}_{i,k} = c_e \mathrm{SiLU}(A^{(T-1)}_{i,k(0e)})
    // \bar{A}^{(T-1)}_{i,k(0e)} = c_e \mathrm{SiLU}'(A^{(T-1)}_{i,k(0e)}) \sigma_{a_i} w_k
    if (N > 0) k_tail<Sh, Precision><<<N, Sh::nsc, 0, stream>>>(M, G, W);
}

template <class Precision, bool mixed>
template <class Sh>
void NequIPFoundationCuda<Precision, mixed>::reverse_layer(int t)
{
    constexpr int LM = Sh::l_max;
    constexpr int LL = Sh::num_layers - 1;
    const long n = N;
    const long nr = NR;

    // X
    // \bar{H}^{(t+1)}_{i,k'\lambda m} += \sum_k w^{(t+1)}_{k\lambda,k'} \bar{X}^{(t+1)}_{i,k\lambda m}
    if (t < LL) {
        for (int l = 0; l <= LM; ++l)
            product(W.X_adj + nr*Sh::xblk(l), Sh::mulx(l), X_weights, X_weights_offset[(t + 1)*(LM + 1) + l], true,
                    W.H_adj_linear + n*Sh::xblk(l), Sh::mulx(l), N*(2*l + 1), Sh::mulx(l), Sh::mulx(l));
        products();
    }

    // gate
    // \bar{A}^{(t)}_{i,k(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,k(0e)}) \bar{H}^{(t+1)}_{i,k(0e)}
    // \bar{A}^{(t)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) \bar{H}^{(t+1)}_{i,k\lambda m},  \bar{A}^{(t)}_{i,g_{k\lambda}(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,g_{k\lambda}(0e)}) \sum_m A^{(t)}_{i,k\lambda m} \bar{H}^{(t+1)}_{i,k\lambda m}
    if (t < LL and N > 0) k_gate_reverse<Sh, Precision><<<blocks(n*Sh::ngt, 128), 128, 0, stream>>>(M, G, W, t);

    // A
    // \bar{\Phi}^{(t)}_{i,k'\lambda m\eta} = \sum_k w^{(t)}_{k\lambda,k'\eta} \bar{A}^{(t)}_{i,k\lambda m},  \bar{H}^{(t)}_{i,k'\lambda m} = \sum_k w^{(t)}_{a_ik\lambda,k'} \bar{A}^{(t)}_{i,k\lambda m} (t > 0)
    if (t == LL) {
        CHECK(cudaMemsetAsync(W.H_adj_sc, 0, n*Sh::xw*sizeof(Precision), stream));
        CHECK(cudaMemsetAsync(W.X_adj, 0, nr*Sh::xw*sizeof(Precision), stream));
    }
    for (int l = 0; l <= (t == LL ? 0 : LM); ++l) {
        const int wy = t == LL ? Sh::nsc : Sh::muly(l);
        product(W.A_adj + n*Sh::yblk(l), wy, A_weights, A_weights_offset[t*(LM + 1) + l], true,
                W.Phi + n*mblk<Sh>(t, l), mulm<Sh>(t, l), N*(2*l + 1), mulm<Sh>(t, l), wy);
    }
    for (int l = 0; t > 0 and l <= (t == LL ? 0 : LM); ++l)
        for (int a = 0; a < num_elements; ++a) {
            const long r0 = type_start[a];
            const long rows = type_start[a + 1] - r0;
            const int wx = Sh::mulx(l);
            const int wy = t == LL ? Sh::nsc : Sh::muly(l);
            product(W.A_adj + n*Sh::yblk(l) + r0*(2*l + 1)*wy, wy, A_sc_weights,
                    A_sc_weights_offset[(t*(LM + 1) + l)*num_elements + a], true,
                    W.H_adj_sc + n*Sh::xblk(l) + r0*(2*l + 1)*wx, wx, int(rows*(2*l + 1)), wx, wy);
        }
    products();

    // Phi
    // \bar{X}^{(t)}_{j,k\lambda_1m_1} = \sum_i \sum_{\lambda m\eta m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(t)}_{i,k\lambda m\eta} (t > 0)
    // \mathbf{F}_{ij} -= \sum_{k\lambda m\eta m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij})] X^{(t)}_{j,k\lambda_1m_1} \bar{\Phi}^{(t)}_{i,k\lambda m\eta}
    if (t > 0 and t < LL) CHECK(cudaMemsetAsync(W.X_adj, 0, nr*Sh::xw*sizeof(Precision), stream));
    tensor_product<Sh, true>(t);
}

template <class Precision, bool mixed>
template <class Sh, bool REV>
void NequIPFoundationCuda<Precision, mixed>::tensor_product(int t)
{
    auto launch = [&](auto lt) {
        constexpr int LT = decltype(lt)::value;
        if (N >= Sh::n_big) launch_tp<Sh, Precision, mixed, LT, REV, true>(M, G, W, t, stream, side, ev_fork, ev_join);
        else launch_tp<Sh, Precision, mixed, LT, REV, false>(M, G, W, t, stream, side, ev_fork, ev_join);
    };
    if (N == 0) return;
    if (t == 0) launch(std::integral_constant<int, 0>{});
    else if (t == Sh::num_layers - 1) launch(std::integral_constant<int, 2>{});
    else if constexpr (Sh::num_layers > 2) launch(std::integral_constant<int, 1>{});
}

template <class Precision, bool mixed>
void NequIPFoundationCuda<Precision, mixed>::product(const Precision* a, int lda, const Weight& w, long off, bool trans,
                                                     Precision* c, int ldc, int m, int n, int k)
{
    if (m > 0) queue.push_back({a, lda, &w, off, trans, c, ldc, m, n, k});
}

template <class Precision, bool mixed>
void NequIPFoundationCuda<Precision, mixed>::products()
{
    if constexpr (mixed)
        if (tensor_cores) {
            TcBatch b{};
            std::vector<Product> rest;
            for (const Product& p : queue) {
                if (p.n%16 or p.k%16) {
                    rest.push_back(p);
                    continue;
                }
                const Weight& w = *p.w;
                b.g[b.n++] = {p.a, (p.trans ? w.hi : w.thi) + p.off, (p.trans ? w.lo : w.tlo) + p.off, p.c, p.m, p.n,
                              p.k, p.lda, p.k, p.ldc, 0};
                if (b.n == 32) {
                    tc_gemm(b, stream);
                    b.n = 0;
                }
            }
            if (b.n) tc_gemm(b, stream);
            queue = rest;
        }
    if (queue.empty()) return;
    CHECK(cudaEventRecord(ev_fork, stream));
    for (auto q : side) CHECK(cudaStreamWaitEvent(q, ev_fork, 0));
    for (size_t q = 0; q < queue.size(); ++q) {
        const Product& p = queue[q];
        CHECK(cublasSetStream(blas, q%4 == 0 ? stream : side[q%4 - 1]));
        gemm(blas, p.trans, p.m, p.n, p.k, p.a, p.lda, p.w->w + p.off, p.trans ? p.k : p.n, p.c, p.ldc);
    }
    for (int q = 0; q < 3; ++q) {
        CHECK(cudaEventRecord(ev_join[q], side[q]));
        CHECK(cudaStreamWaitEvent(stream, ev_join[q], 0));
    }
    CHECK(cublasSetStream(blas, stream));
    queue.clear();
}
