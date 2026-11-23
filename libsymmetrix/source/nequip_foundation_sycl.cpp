// SYCL port of nequip_foundation_cuda.hpp and of the CUDA kernels of pair_symmetrix_nequip_foundation_kokkos.cpp,
// for Intel GPUs. Kernels and layouts follow the CUDA version; the tunings are PVC's (Tuning: sub-group widths,
// BIG thresholds and channels per lane by precision, Q-form reverse masks and fp64 path-range splits, fused launches
// for fp32); the work runs on one in-order queue; cuBLAS is oneMKL. float32-mixed computes like float32 (fp32 spline
// tables, where the CUDA version stores fp16 ones, and plain fp32 GEMMs) unless SYMMETRIX_SYCL_NQ_MIXED_GEMM=1, which
// runs its GEMMs with three bf16 terms per factor on the XMX units (oneMKL compute mode float_to_bf16x3, where the
// CUDA version uses 3xTF32 tensor-core kernels).
// Runtime knobs (environment, for tuning): SYMMETRIX_SYCL_NQ_QREV, _NBIG, _NBIG_PAIR, _SPLIT_TARGET, _SPLIT_MAX,
// _PAIR_FUSED, _FUSE, _FUSE_PAIR, _MIXED_GEMM.
// Not ported: direct_exchange (CUDA IPC) and its fp16 wire.

#include "foundation_sycl.hpp"
#include "foundation_sycl_common.hpp"
#include "nequip_foundation_tables.hpp"
#include "nlohmann_json.hpp"

#include <oneapi/mkl/blas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace symmetrix_sycl {
namespace nequip {

using namespace nequip_foundation;
using json = nlohmann::json;

// Per-preset tunings for PVC. n_big: nodes from which the tensor products use the "BIG" kernels (more channels per
// lane, no edge split); sort_key: edges of a node sorted by distance (0: neighbor-list order); smem_*: edges staged
// per chunk; *_unroll: edge-loop unroll masks (bit l1: by 2); qrev: degrees l1 whose middle-layer reverse uses the
// Q form; qsplit64: fp64 Q-form reverse split into two path ranges at this path (0: whole); cpt: channels per lane in
// BIG mode, per layer type and degree; sg64/sg32: sub-group widths.
struct Tuning {
    static constexpr int n_big = 2000;
    static constexpr int sort_key = 2;
    static constexpr int tp_unroll = 1;
    static constexpr bool smem_edge = true;
    static constexpr int smem_ch = 8;
    static constexpr int smem_ch_big = 16;
    static constexpr int fwd_unroll = 0;
    static constexpr int fwd_unroll32_small = -1;  // fp32 below n_big; -1: fwd_unroll
    static constexpr int rev_unroll = 0;
    static constexpr int qrev = 0;
    static constexpr int qsplit64[5] = {};
    static constexpr int qrev_unroll = 0;
    static constexpr int cpt0 = 1;
    static constexpr int sg64 = 16;
    static constexpr int sg32 = 16;
};
struct TuningS : Tuning {
    static constexpr const char* name = "NequIP-OAM-S";
    static constexpr int sg32 = 32;
    static constexpr int n_big = 1000;
    static constexpr int sort_key = 0;
    static constexpr int tp_unroll = 2;
    static constexpr int cpt[2][5] = {{1, 1}, {4, 2}};
};
struct TuningM : Tuning {
    static constexpr const char* name = "NequIP-OAM-M";
    static constexpr bool smem_edge = false;
    static constexpr int smem_ch_big = 8;
    static constexpr int cpt[2][5] = {{4, 2, 1}, {4, 2, 1}};
};
struct TuningL : Tuning {
    static constexpr const char* name = "NequIP-OAM-L";
    static constexpr int cpt0 = 2;
    static constexpr int n_big = 1000;
    static constexpr int fwd_unroll = 15;
    static constexpr int fwd_unroll32_small = 0;
    static constexpr int qrev = 15;
    // fp64: the Q-form reverse of l1 = 2, 3 as two path ranges ([0, 5) [5, 7) and [0, 4) [4, 6)): whole, it spills
    static constexpr int qsplit64[5] = {0, 0, 5, 4, 0};
    static constexpr int qrev_unroll = 8;
    static constexpr int cpt[2][5] = {{4, 2, 1, 1}, {4, 2, 1, 1}};
    static constexpr int cpt64[5] = {4, 1, 1, 1};  // fp64 middle layers: l1 = 1 one channel per lane (no spills)
};
struct TuningXL : Tuning {
    static constexpr const char* name = "NequIP-OAM-XL";
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
using AllPresets = std::tuple<Shape<TuningS, 2, 1, 32, 128, 64>, Shape<TuningM, 4, 2, 32, 128, 64, 32>,
                              Shape<TuningL, 6, 3, 48, 128, 64, 32, 32>,
                              Shape<TuningXL, 6, 4, 32, 320, 96, 64, 32, 32>>;
// a build may restrict the compiled presets, e.g. -DSYMMETRIX_SYCL_NEQUIP_PRESETS="2" for L only
#ifdef SYMMETRIX_SYCL_NEQUIP_PRESETS
template <int... I> auto select_presets(std::integer_sequence<int, I...>)
    -> std::tuple<std::tuple_element_t<I, AllPresets>...>;
template <int... I> using Pick = decltype(select_presets(std::integer_sequence<int, I...>{}));
using Presets = Pick<SYMMETRIX_SYCL_NEQUIP_PRESETS>;
#else
using Presets = AllPresets;
#endif
inline std::string preset_names() {
    std::string r;
    std::apply([&](auto... s) { ((r += (r.empty() ? "" : ", ") + std::string(decltype(s)::name)), ...); }, Presets{});
    return r;
}

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
    static constexpr int W = 2*Sh::l_max + 1;
    static constexpr Entries make_entries(bool qform) {
        Entries r{};
        int t = 0;
        for (int k = 0; k < NK; ++k) {
            const int t0 = t;
            while (t < NT and gt[t].k == k) ++t;
            const int base = qform ? ks[k].l2*ks[k].l2 : ks[k].out;
            std::array<int, NIN*W> slot{};
            for (int u = t0; u < t; ++u) slot[gt[u].m1*W + (qform ? gt[u].lm2 : gt[u].a) - base] = 1;
            for (int s = 0; s < NIN*W; ++s)
                if (slot[s]) {
                    r.e[r.n] = {k, s/W, base + s%W};
                    slot[s] = ++r.n;
                }
            for (int u = t0; u < t; ++u) r.of[u] = slot[gt[u].m1*W + (qform ? gt[u].lm2 : gt[u].a) - base] - 1;
        }
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

struct alignas(16) F4 { float x, y, z, w; };

template <class T> struct Model {
    int num_elements, intervals;
    T h, half_h, inv_h;
    const T* R[6];
    const T* X0;
    const T *readout_weights, *per_type_scale, *per_type_shift;
    T gate_cst;
    int has_zbl;
    double r_cut, zbl_p;
    const double *zbl_a, *zbl_c;
    const double* cgd;  // the model's tensor-product coefficients (float uses the compiled table)
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

template <int N, class F> inline void unroll(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f.template operator()<I>(), ...);
    }(std::make_integer_sequence<int, N>{});
}

template <class T, int q, int lm> inline T cg(const double* cgd) {
    constexpr float c = float(terms[q].c*sh_norm(lm));
    if constexpr (std::is_same_v<T, float>) return c;
    else return cgd[q];
}

template <class T> struct V4 { T x, y, z, w; };
template <class T> struct alignas(2*sizeof(T)) V2 { T x, y; };
template <class T> inline V4<T> load4(const T* p) {
    if constexpr (std::is_same_v<T, float>) {
        const F4 v = *reinterpret_cast<const F4*>(p);
        return {v.x, v.y, v.z, v.w};
    } else {
        const V2<T> a = reinterpret_cast<const V2<T>*>(p)[0];
        const V2<T> b = reinterpret_cast<const V2<T>*>(p)[1];
        return {a.x, a.y, b.x, b.y};
    }
}
template <class T> inline void store4(T* p, V4<T> v) {
    if constexpr (std::is_same_v<T, float>) *reinterpret_cast<F4*>(p) = F4{v.x, v.y, v.z, v.w};
    else {
        reinterpret_cast<V2<T>*>(p)[0] = V2<T>{v.x, v.y};
        reinterpret_cast<V2<T>*>(p)[1] = V2<T>{v.z, v.w};
    }
}
template <class T, int N> inline void load_n(const T* p, T* v) {
#pragma unroll
    for (int k = 0; k < N/4; ++k) {
        const V4<T> q = load4(p + 4*k);
        v[4*k] = q.x;
        v[4*k+1] = q.y;
        v[4*k+2] = q.z;
        v[4*k+3] = q.w;
    }
}
template <class T, int N> inline void store_n(T* p, const T* v) {
#pragma unroll
    for (int k = 0; k < N/4; ++k) store4(p + 4*k, V4<T>{v[4*k], v[4*k+1], v[4*k+2], v[4*k+3]});
}

template <int l_max, class T> inline void solid_harmonics(T x, T y, T z, T* S) {
    const T xx = x*x;
    const T yy = y*y;
    const T zz = z*z;
    const T r2 = xx + yy + zz;
    const T xy = x*y;
    const T dxy = xx - yy;
    const T f5 = sycl::fma(T(5), zz, -r2);
    S[0] = 1;
    S[1] = y;
    S[2] = z;
    S[3] = x;
    if constexpr (l_max >= 2) { S[4] = xy; S[5] = y*z; S[6] = sycl::fma(T(3), zz, -r2); S[7] = x*z; S[8] = dxy; }
    if constexpr (l_max >= 3) {
        S[9] = y*sycl::fma(T(3), xx, -yy);
        S[10] = xy*z;
        S[11] = y*f5;
        S[12] = z*sycl::fma(T(5), zz, -3*r2);
        S[13] = x*f5;
        S[14] = z*dxy;
        S[15] = x*sycl::fma(T(-3), yy, xx);
    }
    if constexpr (l_max >= 4) {
        const T s2 = xx + yy;
        const T g7 = sycl::fma(T(7), zz, -r2);
        const T g73 = sycl::fma(T(7), zz, -3*r2);
        S[16] = xy*dxy;
        S[17] = y*z*sycl::fma(T(3), xx, -yy);
        S[18] = xy*g7;
        S[19] = y*z*g73;
        S[20] = sycl::fma(8*zz, zz - 3*s2, 3*s2*s2);
        S[21] = x*z*g73;
        S[22] = dxy*g7;
        S[23] = x*z*sycl::fma(T(-3), yy, xx);
        S[24] = sycl::fma(xx, xx - 6*yy, yy*yy);
    }
}

template <class T> constexpr T grad_c(int e) {
    return T(grad_terms[e].c*sh_norm(grad_terms[e].mp)/sh_norm(grad_terms[e].lm));
}
template <int l_max, class T> inline void grad_direct(const T* g, const T* S, T* Gd) {
    constexpr int NQ = l_max*l_max;
    T q[3][NQ] = {};
    unroll<num_grad_terms(l_max)>([&]<int e>() {
        constexpr GradTerm t = grad_terms[e];
        constexpr T c = grad_c<T>(e);
        q[t.d][t.mp] = sycl::fma(c, g[t.lm], q[t.d][t.mp]);
    });
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        T acc = 0;
#pragma unroll
        for (int mp = 0; mp < NQ; ++mp) acc = sycl::fma(q[d][mp], S[mp], acc);
        Gd[d] = acc;
    }
}
template <int l_max, class T> inline void grad_factors(const T* S, T* D) {
    unroll<num_grad_terms(l_max)>([&]<int e>() {
        constexpr GradTerm t = grad_terms[e];
        constexpr T c = grad_c<T>(e);
        D[e] = c*S[t.mp];
    });
}
template <int l_max, class T> inline void grad_staged(const T* g, const T* D, T* Gd) {
    Gd[0] = 0;
    Gd[1] = 0;
    Gd[2] = 0;
    unroll<num_grad_terms(l_max)>([&]<int e>() {
        constexpr GradTerm t = grad_terms[e];
        Gd[t.d] = sycl::fma(g[t.lm], D[e], Gd[t.d]);
    });
}

// Sub-group width of the tensor-product kernels (Tuning::sg64, sg32): a PVC thread holds 64 (SIMD32) or 128 (SIMD16)
// doubles per lane in the large register file, so kernels sized for CUDA's 255 registers per thread spill at 32 lanes
// in fp64; in fp32 only the light S preset is faster at 32.
template <class Sh, class T> constexpr int SGW = sizeof(T) == 8 ? Sh::sg64 : Sh::sg32;
// unrolling of the per-edge loop of the pair-layer kernels (left to the compiler, it unrolls them into code that
// thrashes the instruction cache)
constexpr int pair_unroll = 1;
// edge-loop unroll masks of the tensor-product kernels (bit l1: unroll by 2); fp32 below n_big may use its own
template <class Sh, class T, bool BIG> constexpr int fwd_unroll_of() {
    return sizeof(T) == 4 and not BIG and Sh::fwd_unroll32_small >= 0 ? Sh::fwd_unroll32_small : Sh::fwd_unroll;
}
// Local memory budget (bytes) of a work-group's staged edges in the middle-layer kernels. A PVC Xe-core has 128 KiB:
// 32 KiB lets four work-groups of the 256-register kernels share one (and keeps the fp64 l1 = 2, 3 forward kernels
// free of spills); the fp32 kernels below n_big use 128 registers, and 16 KiB lets up to eight run.
template <class T, bool BIG> constexpr int nq_slm() { return sizeof(T) == 4 and not BIG ? 16384 : 32768; }

// Sums of f0, f1, f2 over the SG lanes of the sub-group, added to dst[0..2] (transposed butterfly).
template <int SG, class T> inline void warp_add3(const SubGroup& sg, T* dst, T f0, T f1, T f2) {
    constexpr int H = SG/2;
    constexpr int Q = SG/4;
    const int lane = int(sg.get_local_linear_id());
    const bool hi = lane & H;
    const bool bq = lane & Q;
    const T k0 = hi ? f2 : f0;
    const T k1 = hi ? T(0) : f1;
    const T s0 = hi ? f0 : f2;
    const T s1 = hi ? f1 : T(0);
    const T a = k0 + shfl_xor(sg, s0, H);
    const T b = k1 + shfl_xor(sg, s1, H);
    T c = (bq ? b : a) + shfl_xor(sg, bq ? a : b, Q);
#pragma unroll
    for (int o = Q/2; o > 0; o >>= 1) c += shfl_xor(sg, c, o);
    if ((lane & (Q - 1)) == 0 and lane < 3*Q) atomic_add(dst + lane/Q, c);
}
template <class T> inline void force_solid(T a, V4<T> q, const T* Gd, T* f) {
    f[0] = -sycl::fma(a, q.y*q.w, Gd[1]);
    f[1] = -sycl::fma(a, q.z*q.w, Gd[2]);
    f[2] = -sycl::fma(a, q.x*q.w, Gd[0]);
}
template <int SG, class T> inline void warp_force4(const SubGroup& sg, T* dst, T a, const T* Gd, V4<T> q, T as) {
    constexpr int H = SG/2;
    constexpr int Q = SG/4;
    const int lane = int(sg.get_local_linear_id());
    const bool hi = lane & H;
    const bool bq = lane & Q;
    const T k0 = hi ? Gd[1] : a;
    const T k1 = hi ? Gd[2] : Gd[0];
    const T s0 = hi ? a : Gd[1];
    const T s1 = hi ? Gd[0] : Gd[2];
    const T x = k0 + shfl_xor(sg, s0, H);
    const T y = k1 + shfl_xor(sg, s1, H);
    T c = (bq ? y : x) + shfl_xor(sg, bq ? x : y, Q);
#pragma unroll
    for (int o = Q/2; o > 0; o >>= 1) c += shfl_xor(sg, c, o);
    const T A = bcast(sg, c, 0)*as;
    const int v = lane/Q;
    if ((lane & (Q - 1)) == 0 and v > 0)
        atomic_add(dst + (v == 1 ? 2 : v - 2), -sycl::fma(A, (v == 1 ? q.x : v == 2 ? q.y : q.z)*q.w, c));
}

template <class T> struct Spl {
    const T* c;
    T value(int f, T x) const {
        const V4<T> k = load4(c + 4*f);
        return k.x + x*(k.y + x*(k.z + x*k.w));
    }
    void deriv(int f, T x, T& v, T& d) const {
        const V4<T> k = load4(c + 4*f);
        v = k.x + x*(k.y + x*(k.z + x*k.w));
        d = k.y + x*(2*k.z + 3*x*k.w);
    }
};
template <class T> inline Spl<T> spline_block(const Model<T>& M, int layer, int iv, int nf) {
    return {M.R[layer] + (long)iv*4*nf};
}

template <class TPg, class T, class B> inline void radial(const B& b, int u, T x, T* R) {
    unroll<TPg::NK>([&]<int k>() { constexpr int r = TPg::ks[k].roff; R[k] = b.value(r + u, x); });
}
template <class TPg, class T, class B> inline void radial(const B& b, int u, T x, T* R, T* Rd) {
    unroll<TPg::NK>([&]<int k>() { constexpr int r = TPg::ks[k].roff; b.deriv(r + u, x, R[k], Rd[k]); });
}

template <class TPg, class T>
inline void tp_forward(const double* C, T* acc, const T* R, const T* S, const T* x) {
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        acc[t.a] = sycl::fma(R[t.k], cg<T, t.q, t.lm2>(C)*S[t.lm2]*x[t.m1], acc[t.a]);
    });
}
template <class TPg, class T>
inline void tp_reverse(const double* C, const T* adj, const T* R, const T* Rd, const T* S, const T* x, T* gx, T* gS,
                       T& a) {
    T sk[TPg::NK] = {};
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        const T c = cg<T, t.q, t.lm2>(C);
        const T u = c*S[t.lm2];
        const T w = adj[t.a]*R[t.k];
        sk[t.k] = sycl::fma(adj[t.a], x[t.m1]*u, sk[t.k]);
        gx[t.m1] = sycl::fma(u, w, gx[t.m1]);
        gS[t.lm2] = sycl::fma(c*x[t.m1], w, gS[t.lm2]);
    });
#pragma unroll
    for (int k = 0; k < TPg::NK; ++k) a = sycl::fma(Rd[k], sk[k], a);
}
template <class TPg, class T> inline void tp_coupling(const double* C, const T* S, T* Msm) {
    T m[TPg::DMP] = {};
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        constexpr int d = TPg::me.of[i];
        m[d] = sycl::fma(cg<T, t.q, t.lm2>(C), S[t.lm2], m[d]);
    });
    store_n<T, TPg::DMP>(Msm, m);
}
template <int d, class T> inline T coupling(const T* Msm, V4<T>& v) {
    if constexpr (d%4 == 0) v = load4(Msm + d);
    if constexpr (d%4 == 0) return v.x;
    else if constexpr (d%4 == 1) return v.y;
    else if constexpr (d%4 == 2) return v.z;
    else return v.w;
}
template <class TPg, class T> inline void tp_forward2(T* acc, const T* R, const T* Msm, const T* x) {
    V4<T> v;
    unroll<TPg::NK>([&]<int k>() {
        constexpr int o = TPg::ks[k].out;
        constexpr int n = 2*TPg::ks[k].l3 + 1;
        constexpr int d0 = TPg::me_first[k];
        T t[n] = {};
        unroll<TPg::me_first[k + 1] - d0>([&]<int j>() {
            constexpr auto e = TPg::me.e[d0 + j];
            t[e.i - o] = sycl::fma(coupling<d0 + j>(Msm, v), x[e.m1], t[e.i - o]);
        });
#pragma unroll
        for (int m = 0; m < n; ++m) acc[o + m] = sycl::fma(R[k], t[m], acc[o + m]);
    });
}
template <class TPg, class T>
inline void tp_reverse2(const double* C, const T* adj, const T* R, const T* Rd, const T* Msm, const T* x, T* gx, T* gS,
                        T& a) {
    V4<T> v;
    unroll<TPg::NK>([&]<int k>() {
        constexpr int d0 = TPg::me_first[k];
        constexpr int t0 = TPg::gt_first[k];
        T q[TPg::NIN] = {};
        unroll<TPg::me_first[k + 1] - d0>([&]<int j>() {
            constexpr auto e = TPg::me.e[d0 + j];
            q[e.m1] = sycl::fma(coupling<d0 + j>(Msm, v), adj[e.i], q[e.m1]);
        });
        T sk = 0;
#pragma unroll
        for (int m1 = 0; m1 < TPg::NIN; ++m1) {
            sk = sycl::fma(x[m1], q[m1], sk);
            gx[m1] = sycl::fma(R[k], q[m1], gx[m1]);
        }
        a = sycl::fma(Rd[k], sk, a);
        unroll<TPg::gt_first[k + 1] - t0>([&]<int j>() {
            constexpr auto t = TPg::gt[t0 + j];
            gS[t.lm2] = sycl::fma(cg<T, t.q, t.lm2>(C)*x[t.m1], adj[t.a]*R[k], gS[t.lm2]);
        });
    });
}
template <class TPg, class T> inline void tp_q_build(const double* C, const T* adj, T* Q) {
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        constexpr int d = TPg::qe.of[i];
        Q[d] = sycl::fma(cg<T, t.q, t.lm2>(C), adj[t.a], Q[d]);
    });
}
template <class TPg, class T>
inline void tp_q_reverse(const T* Q, const T* S, const T* R, const T* Rd, const T* x, T* gx, T* gS, T& a) {
    unroll<TPg::NK>([&]<int k>() {
        constexpr int d0 = TPg::qe_first[k];
        T q[TPg::NIN] = {};
        unroll<TPg::qe_first[k + 1] - d0>([&]<int j>() {
            constexpr auto e = TPg::qe.e[d0 + j];
            q[e.m1] = sycl::fma(S[e.i], Q[d0 + j], q[e.m1]);
            gS[e.i] = sycl::fma(Q[d0 + j], R[k]*x[e.m1], gS[e.i]);
        });
        T sk = 0;
#pragma unroll
        for (int m1 = 0; m1 < TPg::NIN; ++m1) {
            sk = sycl::fma(x[m1], q[m1], sk);
            gx[m1] = sycl::fma(R[k], q[m1], gx[m1]);
        }
        a = sycl::fma(Rd[k], sk, a);
    });
}

// The Q form over a range [K0, K1) of the degree's paths (fp64, see k_tp1_reverse_qs).
template <class TPg, int K0, int K1, class B, class T> inline void radial_range(const B& b, int u, T x, T* R, T* Rd) {
    unroll<TPg::NK>([&]<int k>() {
        if constexpr (k >= K0 and k < K1) {
            constexpr int r = TPg::ks[k].roff;
            b.deriv(r + u, x, R[k], Rd[k]);
        }
    });
}
template <class TPg, int K0, int K1, class T> inline void tp_q_build_range(const double* C, const T* adj, T* Q) {
    unroll<TPg::NT>([&]<int i>() {
        constexpr auto t = TPg::gt[i];
        if constexpr (t.k >= K0 and t.k < K1) {
            constexpr int d = TPg::qe.of[i] - TPg::qe_first[K0];
            Q[d] = sycl::fma(cg<T, t.q, t.lm2>(C), adj[t.a], Q[d]);
        }
    });
}
template <class TPg, int K0, int K1, class T>
inline void tp_q_reverse_range(const T* Q, const T* S, const T* R, const T* Rd, const T* x, T* gx, T* gS, T& a) {
    unroll<TPg::NK>([&]<int k>() {
        if constexpr (k >= K0 and k < K1) {
            constexpr int d0 = TPg::qe_first[k];
            constexpr int o = TPg::qe_first[K0];
            T q[TPg::NIN] = {};
            unroll<TPg::qe_first[k + 1] - d0>([&]<int j>() {
                constexpr auto e = TPg::qe.e[d0 + j];
                q[e.m1] = sycl::fma(S[e.i], Q[d0 + j - o], q[e.m1]);
                gS[e.i] = sycl::fma(Q[d0 + j - o], R[k]*x[e.m1], gS[e.i]);
            });
            T sk = 0;
#pragma unroll
            for (int m1 = 0; m1 < TPg::NIN; ++m1) {
                sk = sycl::fma(x[m1], q[m1], sk);
                gx[m1] = sycl::fma(R[k], q[m1], gx[m1]);
            }
            a = sycl::fma(Rd[k], sk, a);
        }
    });
}

template <class Sh, int LT, int G> inline long msg_index(long N, int i, int u, int o) {
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
inline void msg_store(T* M, long N, int i, int u, const T* acc, bool atomic) {
    using TPg = TP<Sh, LT, G>;
    using Ly = typename TPg::Ly;
    unroll<TPg::NK>([&]<int k>() {
        constexpr auto K = TPg::ks[k];
        constexpr long mb = Ly::mblk(K.l3);
        constexpr int mm = Ly::mulm(K.l3);
#pragma unroll
        for (int m = 0; m <= 2*K.l3; ++m) {
            T* p = M + N*mb + ((long)i*(2*K.l3 + 1) + m)*mm + K.choff + u;
            if (atomic) atomic_add(p, acc[K.out + m]);
            else *p = acc[K.out + m];
        }
    });
}
template <class Sh, int LT, int G, class T> inline void msg_load(const T* M, long N, int i, int u, T* adj) {
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

// channels per lane in BIG mode; the fp64 middle layers use the preset's cpt64 where given
template <class Sh, class T> constexpr int cpt_of(int lt, int g) {
    if constexpr (std::is_same_v<T, double> and requires { Sh::cpt64[0]; })
        if (lt == 1) return Sh::cpt64[g];
    return Sh::cpt[lt - 1][g];
}
// lanes per node, nodes per work-group; T = double selects the presets' fp64 channels per lane (cpt64) where given
template <class Sh, int LT, int G, bool BIG, class T = void> struct LPN {
    static constexpr int cp = LT == 0 ? Sh::cpt0 : not BIG ? 1 : cpt_of<Sh, T>(LT, G);
    static constexpr int m = TP<Sh, LT, G>::MUL/cp;
    static constexpr int v = (m + 31)/32*32;
    static constexpr int npb = v >= 128 ? 1 : 128/v;
    static constexpr int tpb = v*npb;
};

template <class T> struct EdgeB { int j, iv; T xs; V4<T> q; };
template <class T> inline EdgeB<T> edge_load(const Graph& G, const Work<T>& W, int e) {
    return {G.neigh_rows[e], W.interval[e], W.xs[e], load4(W.geometry + 4*e)};
}
template <class T> inline EdgeB<T> edge_batch(const SubGroup& sg, const Graph& G, const Work<T>& W, int e0, int e1) {
    const int e = e0 + int(sg.get_local_linear_id());
    if (e < e1) return edge_load(G, W, e);
    return {0, 0, T(0), {T(0), T(0), T(0), T(1)}};
}
template <class T> inline EdgeB<T> edge_bcast(const SubGroup& sg, const EdgeB<T>& b, int c) {
    return {bcast(sg, b.j, c), bcast(sg, b.iv, c), bcast(sg, b.xs, c),
            {bcast(sg, b.q.x, c), bcast(sg, b.q.y, c), bcast(sg, b.q.z, c), bcast(sg, b.q.w, c)}};
}

#define KERNEL [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]]
#define KERNEL_SG(...) [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(__VA_ARGS__)]]

// fp64 forward middle-layer kernels in the large register file: IGC's automatic choice varies between builds of
// unchanged kernels and can pick 128 registers with spills
template <class T> inline auto fwd_props() {
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (sizeof(T) == 8) return syclex::properties{sycl::ext::intel::experimental::grf_size<256>};
    else return syclex::properties{};
}

// local memory of n values of T, 4*sizeof(T)-aligned
template <class T> using LocalT = sycl::local_accessor<sycl::vec<T, 4>, 1>;
template <class T> inline T* local_ptr(const LocalT<T>& a) {
    return reinterpret_cast<T*>(a.template get_multi_ptr<sycl::access::decorated::no>().get());
}
template <class T> inline LocalT<T> local_mem(sycl::handler& h, long n) {
    return LocalT<T>(sycl::range<1>((n + 3)/4), h);
}

template <class Sh, class T, bool BIG>
void k_tp0(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G, const Work<T>& W) {
    using L = LPN<Sh, 0, 0, BIG>;
    q.parallel_for(sycl::nd_range<3>(grid*dims(L::tpb), dims(L::tpb)), KERNEL_SG(SGW<Sh, T>) {
        using TPg = TP<Sh, 0, 0>;
        const Thread th{it};
        const SubGroup sg = th.sg();
        const int i = th.bx()*L::npb + th.tx()/L::v;
        const int u = th.tx()%L::v;
        if (i >= G.N) return;
        constexpr int NF = TPg::Ly::nf;
        constexpr int CP = L::cp;
        constexpr int LL = TPg::MUL/CP;
        const bool on = u < LL;
        T acc[CP][TPg::NOUT] = {};
        const int es0 = G.start[i];
        const int ecnt = G.start[i+1] - es0;
        const int ea = es0 + (int)((long)ecnt*th.by()/th.gy());
        const int eb = es0 + (int)((long)ecnt*(th.by() + 1)/th.gy());
        for (int e0 = ea; e0 < eb; e0 += SGW<Sh, T>) {
            const EdgeB<T> b = edge_batch(sg, G, W, e0, eb);
            const int nb = sycl::min(SGW<Sh, T>, eb - e0);
#pragma unroll Sh::tp_unroll
            for (int c = 0; c < nb; ++c) {
                const EdgeB<T> ed = edge_bcast(sg, b, c);
                T S[Sh::num_lm];
                solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
                const int tj = G.elem[ed.j];
                const auto blk = spline_block<T>(M, 0, ed.iv, NF);
                const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
                for (int cc = 0; cc < CP; ++cc) {
                    const int uc = u + cc*LL;
                    T x[1] = {on ? M.X0[tj*Sh::emb + uc] : T(0)};
                    T R[TPg::NK] = {};
                    if (on) radial<TPg>(blk, uc, xs, R);
                    tp_forward<TPg>(M.cgd, acc[cc], R, S, x);
                }
            }
        }
        if (on)
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) msg_store<Sh, 0, 0>(W.Phi, G.N, i, u + cc*LL, acc[cc], th.gy() > 1);
    });
}

template <class Sh, class T, bool BIG>
void k_tp0_reverse(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G, const Work<T>& W) {
    using L = LPN<Sh, 0, 0, BIG>;
    q.parallel_for(sycl::nd_range<3>(grid*dims(L::tpb), dims(L::tpb)), KERNEL_SG(SGW<Sh, T>) {
        using TPg = TP<Sh, 0, 0>;
        const Thread th{it};
        const SubGroup sg = th.sg();
        const int i = th.bx()*L::npb + th.tx()/L::v;
        const int u = th.tx()%L::v;
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
        const int ea = es0 + (int)((long)ecnt*th.by()/th.gy());
        const int eb = es0 + (int)((long)ecnt*(th.by() + 1)/th.gy());
        for (int e0 = ea; e0 < eb; e0 += SGW<Sh, T>) {
            const EdgeB<T> b = edge_batch(sg, G, W, e0, eb);
            const int nb = sycl::min(SGW<Sh, T>, eb - e0);
#pragma unroll Sh::tp_unroll
            for (int c = 0; c < nb; ++c) {
                const int e = e0 + c;
                const EdgeB<T> ed = edge_bcast(sg, b, c);
                T S[Sh::num_lm];
                solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
                const int tj = G.elem[ed.j];
                const auto blk = spline_block<T>(M, 0, ed.iv, NF);
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
                    tp_reverse<TPg>(M.cgd, adj[cc], R, Rd, S, x, gx, gS, a);
                }
                T Gd[3], f[3];
                grad_direct<Sh::l_max>(gS, S, Gd);
                force_solid(a*M.inv_h, ed.q, Gd, f);
                warp_add3<SGW<Sh, T>>(sg, W.edge_forces + 3*e, f[0], f[1], f[2]);
            }
        }
    });
}

template <class Sh, int G, bool BIG, class T> struct SMC {
    using TPg = TP<Sh, 1, G>;
    static constexpr int WPB = LPN<Sh, 1, G, BIG, T>::tpb/SGW<Sh, T>;
    static constexpr int SP = (Sh::num_lm + 3)/4*4;
    static constexpr int fit(int ch) {
        return ch > 1 and WPB*ch*(TPg::DMP + SP + (Sh::smem_edge ? 8 : 0))*(int)sizeof(T) > nq_slm<T, BIG>()
                   ? fit(ch/2) : ch;
    }
    static constexpr int CH0 = BIG ? Sh::smem_ch_big : Sh::smem_ch;
    static constexpr int SG = SGW<Sh, T>;
    static constexpr int CH = fit(CH0 < SG ? CH0 : SG);  // edges staged by one lane each
    static constexpr int PER_WARP = CH*(TPg::DMP + SP + (Sh::smem_edge ? 8 : 0));
};
template <class Sh, int G, bool BIG, class T>
inline void smem_chunk(const SubGroup& sg, const double* C, const Graph& G_, const Work<T>& W, int e0, int ne, T* Msm,
                       T* Ssm) {
    using TPg = TP<Sh, 1, G>;
    using Cc = SMC<Sh, G, BIG, T>;
    const int lane = int(sg.get_local_linear_id());
    sycl::group_barrier(sg);
    if (lane < ne) {
        const int e = e0 + lane;
        const V4<T> q = load4(W.geometry + 4*e);
        T S[Cc::SP] = {};
        solid_harmonics<Sh::l_max>(q.x, q.y, q.z, S);
        tp_coupling<TPg>(C, S, Msm + lane*TPg::DMP);
        store_n<T, Cc::SP>(Ssm + lane*Cc::SP, S);
        if constexpr (Sh::smem_edge) {
            T* Ed = Ssm + Cc::CH*Cc::SP + lane*8;
            int* Ei = reinterpret_cast<int*>(Ed);
            Ei[0] = G_.neigh_rows[e];
            Ei[1] = W.interval[e];
            Ed[2] = W.xs[e];
            store4(Ed + 4, q);
        }
    }
    sycl::group_barrier(sg);
}
template <class Sh, int G, bool BIG, class T>
inline EdgeB<T> smem_edge(const Graph& G_, const Work<T>& W, const T* Ssm, int e, int c) {
    using Cc = SMC<Sh, G, BIG, T>;
    if constexpr (Sh::smem_edge) {
        const T* Ed = Ssm + Cc::CH*Cc::SP + c*8;
        const int* Ei = reinterpret_cast<const int*>(Ed);
        return {Ei[0], Ei[1], Ed[2], load4(Ed + 4)};
    }
    return edge_load(G_, W, e);
}

// tp1_body, tp1_reverse_body and tp1_reverse_q_body: one work-group of k_tp1, k_tp1_reverse and k_tp1_reverse_q with
// the work-group coordinates as parameters, for the fused launch (k_tp1_fused). The per-degree kernels keep their own
// copy of the same code: built from these shared bodies, IGC allocates their registers worse (slower, and spills in
// fp64 BIG mode).
template <class Sh, class T, int G, bool BIG>
inline void tp1_body(const sycl::nd_item<3>& it, int bx, int by, int gy, T* smem, const Model<T>& M,
                     const Graph& G_, const Work<T>& W, int layer) {
    using L = LPN<Sh, 1, G, BIG, T>;
    using Cc = SMC<Sh, G, BIG, T>;
    using TPg = TP<Sh, 1, G>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    const Thread th{it};
    const SubGroup sg = th.sg();
    const int i = bx*L::npb + th.tx()/L::v;
    const int u = th.tx()%L::v;
    if (i >= G_.N) return;
    T* Msm = smem + th.warp()*Cc::PER_WARP;
    T* Ssm = Msm + Cc::CH*TPg::DMP;
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T acc[CP][TPg::NOUT] = {};
    constexpr bool split = not BIG;
    const int es0 = G_.start[i];
    const int e1 = G_.start[i+1];
    const int ecnt = e1 - es0;
    const int ea = split ? es0 + (int)((long)ecnt*by/gy) : es0;
    const int eb = split ? es0 + (int)((long)ecnt*(by + 1)/gy) : e1;
    for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
        const int ne = sycl::min(Cc::CH, eb - e0);
        smem_chunk<Sh, G, BIG>(sg, M.cgd, G_, W, e0, ne, Msm, Ssm);
#pragma unroll(((fwd_unroll_of<Sh, T, BIG>() >> G) & 1) ? 2 : 1)
        for (int c = 0; c < ne; ++c) {
            const EdgeB<T> ed = smem_edge<Sh, G, BIG>(G_, W, Ssm, e0 + c, c);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m)
                    x[m] = on ? X[((long)ed.j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) radial<TPg>(spline_block<T>(M, layer, ed.iv, NF), uc, xs, R);
                tp_forward2<TPg>(acc[cc], R, Msm + c*TPg::DMP, x);
            }
        }
    }
    if (on)
#pragma unroll
        for (int cc = 0; cc < CP; ++cc)
            msg_store<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, acc[cc], split and gy > 1);
}

template <class Sh, class T, int G, bool BIG>
void k_tp1(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G_, const Work<T>& W, int layer) {
    using L = LPN<Sh, 1, G, BIG, T>;
    using Cc = SMC<Sh, G, BIG, T>;
    q.submit([&](sycl::handler& h) {
        auto sm = local_mem<T>(h, Cc::WPB*Cc::PER_WARP);
        h.parallel_for(sycl::nd_range<3>(grid*dims(L::tpb), dims(L::tpb)), fwd_props<T>(), KERNEL_SG(SGW<Sh, T>) {
            using TPg = TP<Sh, 1, G>;
            constexpr int NF = TPg::Ly::nf;
            constexpr int CP = L::cp;
            constexpr int LL = TPg::MUL/CP;
            const Thread th{it};
            const SubGroup sg = th.sg();
            const int i = th.bx()*L::npb + th.tx()/L::v;
            const int u = th.tx()%L::v;
            if (i >= G_.N) return;
            T* Msm = local_ptr(sm) + th.warp()*Cc::PER_WARP;
            T* Ssm = Msm + Cc::CH*TPg::DMP;
            const long NX = G_.NR;
            const bool on = u < LL;
            const T* X = W.X[layer] + NX*Sh::xblk(G);
            T acc[CP][TPg::NOUT] = {};
            constexpr bool split = not BIG;
            const int es0 = G_.start[i];
            const int e1 = G_.start[i+1];
            const int ecnt = e1 - es0;
            const int ea = split ? es0 + (int)((long)ecnt*th.by()/th.gy()) : es0;
            const int eb = split ? es0 + (int)((long)ecnt*(th.by() + 1)/th.gy()) : e1;
            for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
                const int ne = sycl::min(Cc::CH, eb - e0);
                smem_chunk<Sh, G, BIG>(sg, M.cgd, G_, W, e0, ne, Msm, Ssm);
#pragma unroll(((fwd_unroll_of<Sh, T, BIG>() >> G) & 1) ? 2 : 1)
                for (int c = 0; c < ne; ++c) {
                    const EdgeB<T> ed = smem_edge<Sh, G, BIG>(G_, W, Ssm, e0 + c, c);
                    const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
                    for (int cc = 0; cc < CP; ++cc) {
                        const int uc = u + cc*LL;
                        T x[TPg::NIN];
                        T R[TPg::NK] = {};
#pragma unroll
                        for (int m = 0; m < TPg::NIN; ++m)
                            x[m] = on ? X[((long)ed.j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                        if (on) radial<TPg>(spline_block<T>(M, layer, ed.iv, NF), uc, xs, R);
                        tp_forward2<TPg>(acc[cc], R, Msm + c*TPg::DMP, x);
                    }
                }
            }
            if (on)
#pragma unroll
                for (int cc = 0; cc < CP; ++cc)
                    msg_store<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, acc[cc], split and th.gy() > 1);
        });
    });
}

template <class Sh, class T, int G, bool BIG>
inline void tp1_reverse_body(const sycl::nd_item<3>& it, int bx, int by, int gy, T* smem, const Model<T>& M,
                             const Graph& G_, const Work<T>& W, int layer) {
    using L = LPN<Sh, 1, G, BIG, T>;
    using Cc = SMC<Sh, G, BIG, T>;
    using TPg = TP<Sh, 1, G>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    const Thread th{it};
    const SubGroup sg = th.sg();
    const int i = bx*L::npb + th.tx()/L::v;
    const int u = th.tx()%L::v;
    if (i >= G_.N) return;
    T* Msm = smem + th.warp()*Cc::PER_WARP;
    T* Ssm = Msm + Cc::CH*TPg::DMP;
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
    const int ea = split ? es0 + (int)((long)ecnt*by/gy) : es0;
    const int eb = split ? es0 + (int)((long)ecnt*(by + 1)/gy) : e1;
    for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
        const int ne = sycl::min(Cc::CH, eb - e0);
        smem_chunk<Sh, G, BIG>(sg, M.cgd, G_, W, e0, ne, Msm, Ssm);
#pragma unroll(((Sh::rev_unroll >> G) & 1) ? 2 : 1)
        for (int c = 0; c < ne; ++c) {
            const int e = e0 + c;
            const EdgeB<T> ed = smem_edge<Sh, G, BIG>(G_, W, Ssm, e, c);
            const T* S = Ssm + c*Cc::SP;
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
                for (int m = 0; m < TPg::NIN; ++m)
                    x[m] = on ? X[((long)ed.j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) radial<TPg>(spline_block<T>(M, layer, ed.iv, NF), uc, xs, R, Rd);
                tp_reverse2<TPg>(M.cgd, adj[cc], R, Rd, Msm + c*TPg::DMP, x, gx, gS, a);
                if (on)
#pragma unroll
                    for (int m = 0; m < TPg::NIN; ++m)
                        atomic_add(X_adj + ((long)ed.j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
            }
            T Gd[3], f[3];
            grad_direct<Sh::l_max>(gS, S, Gd);
            force_solid(a*M.inv_h, ed.q, Gd, f);
            warp_add3<SGW<Sh, T>>(sg, W.edge_forces + 3*e, f[0], f[1], f[2]);
        }
    }
}

template <class Sh, class T, int G, bool BIG>
void k_tp1_reverse(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G_, const Work<T>& W,
                   int layer) {
    using L = LPN<Sh, 1, G, BIG, T>;
    using Cc = SMC<Sh, G, BIG, T>;
    q.submit([&](sycl::handler& h) {
        auto sm = local_mem<T>(h, Cc::WPB*Cc::PER_WARP);
        h.parallel_for(sycl::nd_range<3>(grid*dims(L::tpb), dims(L::tpb)), KERNEL_SG(SGW<Sh, T>) {
            using TPg = TP<Sh, 1, G>;
            constexpr int NF = TPg::Ly::nf;
            constexpr int CP = L::cp;
            constexpr int LL = TPg::MUL/CP;
            const Thread th{it};
            const SubGroup sg = th.sg();
            const int i = th.bx()*L::npb + th.tx()/L::v;
            const int u = th.tx()%L::v;
            if (i >= G_.N) return;
            T* Msm = local_ptr(sm) + th.warp()*Cc::PER_WARP;
            T* Ssm = Msm + Cc::CH*TPg::DMP;
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
            const int ea = split ? es0 + (int)((long)ecnt*th.by()/th.gy()) : es0;
            const int eb = split ? es0 + (int)((long)ecnt*(th.by() + 1)/th.gy()) : e1;
            for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
                const int ne = sycl::min(Cc::CH, eb - e0);
                smem_chunk<Sh, G, BIG>(sg, M.cgd, G_, W, e0, ne, Msm, Ssm);
#pragma unroll(((Sh::rev_unroll >> G) & 1) ? 2 : 1)
                for (int c = 0; c < ne; ++c) {
                    const int e = e0 + c;
                    const EdgeB<T> ed = smem_edge<Sh, G, BIG>(G_, W, Ssm, e, c);
                    const T* S = Ssm + c*Cc::SP;
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
                        for (int m = 0; m < TPg::NIN; ++m)
                            x[m] = on ? X[((long)ed.j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                        if (on) radial<TPg>(spline_block<T>(M, layer, ed.iv, NF), uc, xs, R, Rd);
                        tp_reverse2<TPg>(M.cgd, adj[cc], R, Rd, Msm + c*TPg::DMP, x, gx, gS, a);
                        if (on)
#pragma unroll
                            for (int m = 0; m < TPg::NIN; ++m)
                                atomic_add(X_adj + ((long)ed.j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
                    }
                    T Gd[3], f[3];
                    grad_direct<Sh::l_max>(gS, S, Gd);
                    force_solid(a*M.inv_h, ed.q, Gd, f);
                    warp_add3<SGW<Sh, T>>(sg, W.edge_forces + 3*e, f[0], f[1], f[2]);
                }
            }
        });
    });
}

template <class Sh, int G, bool BIG, class T> struct SMQ {
    using L = LPN<Sh, 1, G, BIG, T>;
    static constexpr int SP = (Sh::num_lm + 3)/4*4;
    static constexpr int ND = (num_grad_terms(Sh::l_max) + 3)/4*4;
    static constexpr int TPB = 128 >= L::v ? 128/L::v*L::v : L::v;
    static constexpr int NPB = TPB/L::v;
    static constexpr int WPB = TPB/SGW<Sh, T>;
    static constexpr int fit(int ch) {
        return ch > 1 and WPB*ch*(SP + 8 + ND)*(int)sizeof(T) > nq_slm<T, BIG>() ? fit(ch/2) : ch;
    }
    static constexpr int CH = fit(SGW<Sh, T>);
    static constexpr int PER_WARP = CH*(SP + 8 + ND);
};
template <class Sh, class T, int G, bool BIG>
inline void tp1_reverse_q_body(const sycl::nd_item<3>& it, int bx, int by, int gy, T* smem, const Model<T>& M,
                               const Graph& G_, const Work<T>& W, int layer) {
    using Cc = SMQ<Sh, G, BIG, T>;
    using TPg = TP<Sh, 1, G>;
    using L = LPN<Sh, 1, G, BIG, T>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = L::cp;
    constexpr int LL = TPg::MUL/CP;
    const Thread th{it};
    const SubGroup sg = th.sg();
    const int i = bx*Cc::NPB + th.tx()/L::v;
    const int u = th.tx()%L::v;
    if (i >= G_.N) return;
    T* Ssm = smem + th.warp()*Cc::PER_WARP;
    T* Esm = Ssm + Cc::CH*Cc::SP;
    T* Dsm = Esm + Cc::CH*8;
    const int lane = th.lane();
    const long NX = G_.NR;
    const bool on = u < LL;
    const T* X = W.X[layer] + NX*Sh::xblk(G);
    T* X_adj = W.X_adj + NX*Sh::xblk(G);
    T Q[CP][TPg::NQ4] = {};
#pragma unroll
    for (int cc = 0; cc < CP; ++cc) {
        T adj[TPg::NOUT] = {};
        if (on) msg_load<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, adj);
        tp_q_build<TPg>(M.cgd, adj, Q[cc]);
    }
    constexpr bool split = not BIG;
    const int es0 = G_.start[i];
    const int e1 = G_.start[i+1];
    const int ecnt = e1 - es0;
    const int ea = split ? es0 + (int)((long)ecnt*by/gy) : es0;
    const int eb = split ? es0 + (int)((long)ecnt*(by + 1)/gy) : e1;
    for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
        const int ne = sycl::min(Cc::CH, eb - e0);
        sycl::group_barrier(sg);
        if (lane < ne) {
            const int e = e0 + lane;
            const V4<T> qv = load4(W.geometry + 4*e);
            T S[Cc::SP] = {};
            solid_harmonics<Sh::l_max>(qv.x, qv.y, qv.z, S);
            store_n<T, Cc::SP>(Ssm + lane*Cc::SP, S);
            T* Ed = Esm + lane*8;
            int* Ei = reinterpret_cast<int*>(Ed);
            Ei[0] = G_.neigh_rows[e];
            Ei[1] = W.interval[e];
            Ed[2] = W.xs[e];
            store4(Ed + 4, qv);
            T D[Cc::ND] = {};
            grad_factors<Sh::l_max>(S, D);
            store_n<T, Cc::ND>(Dsm + lane*Cc::ND, D);
        }
        sycl::group_barrier(sg);
#pragma unroll(((Sh::qrev_unroll >> G) & 1) ? 2 : 1)
        for (int c = 0; c < ne; ++c) {
            const int e = e0 + c;
            const T* Ed = Esm + c*8;
            const int* Ei = reinterpret_cast<const int*>(Ed);
            const int j = Ei[0];
            const V4<T> qv = load4(Ed + 4);
            T S[Cc::SP];
            load_n<T, Cc::SP>(Ssm + c*Cc::SP, S);
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
                if (on) radial<TPg>(spline_block<T>(M, layer, Ei[1], NF), uc, xs, R, Rd);
                tp_q_reverse<TPg>(Q[cc], S, R, Rd, x, gx, gS, a);
                // skipped by sub-groups without an active lane; the uniform branch also keeps IGC from interleaving the
                // atomics with the force reduction below (slower in fp64)
                if (sycl::any_of_group(sg, on) and on)
#pragma unroll
                    for (int m = 0; m < TPg::NIN; ++m)
                        atomic_add(X_adj + ((long)j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
            }
            T D[Cc::ND];
            load_n<T, Cc::ND>(Dsm + c*Cc::ND, D);
            T Gd[3];
            grad_staged<Sh::l_max>(gS, D, Gd);
            warp_force4<SGW<Sh, T>>(sg, W.edge_forces + 3*e, a, Gd, qv, M.inv_h);
        }
    }
}

template <class Sh, class T, int G, bool BIG>
void k_tp1_reverse_q(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G_, const Work<T>& W,
                     int layer) {
    using Cc = SMQ<Sh, G, BIG, T>;
    q.submit([&](sycl::handler& h) {
        auto sm = local_mem<T>(h, Cc::WPB*Cc::PER_WARP);
        h.parallel_for(sycl::nd_range<3>(grid*dims(Cc::TPB), dims(Cc::TPB)), KERNEL_SG(SGW<Sh, T>) {
            using TPg = TP<Sh, 1, G>;
            using L = LPN<Sh, 1, G, BIG, T>;
            constexpr int NF = TPg::Ly::nf;
            constexpr int CP = L::cp;
            constexpr int LL = TPg::MUL/CP;
            const Thread th{it};
            const SubGroup sg = th.sg();
            const int i = th.bx()*Cc::NPB + th.tx()/L::v;
            const int u = th.tx()%L::v;
            if (i >= G_.N) return;
            T* Ssm = local_ptr(sm) + th.warp()*Cc::PER_WARP;
            T* Esm = Ssm + Cc::CH*Cc::SP;
            T* Dsm = Esm + Cc::CH*8;
            const int lane = th.lane();
            const long NX = G_.NR;
            const bool on = u < LL;
            const T* X = W.X[layer] + NX*Sh::xblk(G);
            T* X_adj = W.X_adj + NX*Sh::xblk(G);
            T Q[CP][TPg::NQ4] = {};
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                T adj[TPg::NOUT] = {};
                if (on) msg_load<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, adj);
                tp_q_build<TPg>(M.cgd, adj, Q[cc]);
            }
            constexpr bool split = not BIG;
            const int es0 = G_.start[i];
            const int e1 = G_.start[i+1];
            const int ecnt = e1 - es0;
            const int ea = split ? es0 + (int)((long)ecnt*th.by()/th.gy()) : es0;
            const int eb = split ? es0 + (int)((long)ecnt*(th.by() + 1)/th.gy()) : e1;
            for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
                const int ne = sycl::min(Cc::CH, eb - e0);
                sycl::group_barrier(sg);
                if (lane < ne) {
                    const int e = e0 + lane;
                    const V4<T> qv = load4(W.geometry + 4*e);
                    T S[Cc::SP] = {};
                    solid_harmonics<Sh::l_max>(qv.x, qv.y, qv.z, S);
                    store_n<T, Cc::SP>(Ssm + lane*Cc::SP, S);
                    T* Ed = Esm + lane*8;
                    int* Ei = reinterpret_cast<int*>(Ed);
                    Ei[0] = G_.neigh_rows[e];
                    Ei[1] = W.interval[e];
                    Ed[2] = W.xs[e];
                    store4(Ed + 4, qv);
                    T D[Cc::ND] = {};
                    grad_factors<Sh::l_max>(S, D);
                    store_n<T, Cc::ND>(Dsm + lane*Cc::ND, D);
                }
                sycl::group_barrier(sg);
#pragma unroll(((Sh::qrev_unroll >> G) & 1) ? 2 : 1)
                for (int c = 0; c < ne; ++c) {
                    const int e = e0 + c;
                    const T* Ed = Esm + c*8;
                    const int* Ei = reinterpret_cast<const int*>(Ed);
                    const int j = Ei[0];
                    const V4<T> qv = load4(Ed + 4);
                    T S[Cc::SP];
                    load_n<T, Cc::SP>(Ssm + c*Cc::SP, S);
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
                        if (on) radial<TPg>(spline_block<T>(M, layer, Ei[1], NF), uc, xs, R, Rd);
                        tp_q_reverse<TPg>(Q[cc], S, R, Rd, x, gx, gS, a);
                        if (sycl::any_of_group(sg, on) and on)
#pragma unroll
                            for (int m = 0; m < TPg::NIN; ++m)
                                atomic_add(X_adj + ((long)j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
                    }
                    T D[Cc::ND];
                    load_n<T, Cc::ND>(Dsm + c*Cc::ND, D);
                    T Gd[3];
                    grad_staged<Sh::l_max>(gS, D, Gd);
                    warp_force4<SGW<Sh, T>>(sg, W.edge_forces + 3*e, a, Gd, qv, M.inv_h);
                }
            }
        });
    });
}

// fp64 Q-form reverse over the paths [K0, K1) of degree G: the whole Q array of l1 = 2, 3 (111, 124 doubles per lane
// for L) spills in fp64, so it runs as two launches split at Tuning::qsplit64, each of whose parts fits. Edge staging,
// neighbor features and the atomic updates repeat per part (they are linear, so the parts add).
template <class Sh, class T, int G, bool BIG, int K0, int K1>
void k_tp1_reverse_qs(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G_, const Work<T>& W,
                      int layer) {
    static_assert(std::is_same_v<T, double>);
    using Cc = SMQ<Sh, G, BIG, T>;
    q.submit([&](sycl::handler& h) {
        auto sm = local_mem<T>(h, Cc::WPB*Cc::PER_WARP);
        h.parallel_for(sycl::nd_range<3>(grid*dims(Cc::TPB), dims(Cc::TPB)), KERNEL_SG(SGW<Sh, T>) {
            using TPg = TP<Sh, 1, G>;
            using L = LPN<Sh, 1, G, BIG, T>;
            constexpr int NF = TPg::Ly::nf;
            constexpr int CP = L::cp;
            constexpr int LL = TPg::MUL/CP;
            const Thread th{it};
            const SubGroup sg = th.sg();
            const int i = th.bx()*Cc::NPB + th.tx()/L::v;
            const int u = th.tx()%L::v;
            if (i >= G_.N) return;
            T* Ssm = local_ptr(sm) + th.warp()*Cc::PER_WARP;
            T* Esm = Ssm + Cc::CH*Cc::SP;
            T* Dsm = Esm + Cc::CH*8;
            const int lane = th.lane();
            const long NX = G_.NR;
            const bool on = u < LL;
            const T* X = W.X[layer] + NX*Sh::xblk(G);
            T* X_adj = W.X_adj + NX*Sh::xblk(G);
            constexpr int NQR = (TPg::qe_first[K1] - TPg::qe_first[K0] + 3)/4*4;
            T Q[CP][NQR] = {};
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                T adj[TPg::NOUT] = {};
                if (on) msg_load<Sh, 1, G>(W.Phi, G_.N, i, u + cc*LL, adj);
                tp_q_build_range<TPg, K0, K1>(M.cgd, adj, Q[cc]);
            }
            constexpr bool split = not BIG;
            const int es0 = G_.start[i];
            const int e1 = G_.start[i+1];
            const int ecnt = e1 - es0;
            const int ea = split ? es0 + (int)((long)ecnt*th.by()/th.gy()) : es0;
            const int eb = split ? es0 + (int)((long)ecnt*(th.by() + 1)/th.gy()) : e1;
            for (int e0 = ea; e0 < eb; e0 += Cc::CH) {
                const int ne = sycl::min(Cc::CH, eb - e0);
                sycl::group_barrier(sg);
                if (lane < ne) {
                    const int e = e0 + lane;
                    const V4<T> qv = load4(W.geometry + 4*e);
                    T S[Cc::SP] = {};
                    solid_harmonics<Sh::l_max>(qv.x, qv.y, qv.z, S);
                    store_n<T, Cc::SP>(Ssm + lane*Cc::SP, S);
                    T* Ed = Esm + lane*8;
                    int* Ei = reinterpret_cast<int*>(Ed);
                    Ei[0] = G_.neigh_rows[e];
                    Ei[1] = W.interval[e];
                    Ed[2] = W.xs[e];
                    store4(Ed + 4, qv);
                    T D[Cc::ND] = {};
                    grad_factors<Sh::l_max>(S, D);
                    store_n<T, Cc::ND>(Dsm + lane*Cc::ND, D);
                }
                sycl::group_barrier(sg);
#pragma unroll(((Sh::qrev_unroll >> G) & 1) ? 2 : 1)
                for (int c = 0; c < ne; ++c) {
                    const int e = e0 + c;
                    const T* Ed = Esm + c*8;
                    const int* Ei = reinterpret_cast<const int*>(Ed);
                    const int j = Ei[0];
                    const V4<T> qv = load4(Ed + 4);
                    T S[Cc::SP];
                    load_n<T, Cc::SP>(Ssm + c*Cc::SP, S);
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
                        if (on) radial_range<TPg, K0, K1>(spline_block<T>(M, layer, Ei[1], NF), uc, xs, R, Rd);
                        tp_q_reverse_range<TPg, K0, K1>(Q[cc], S, R, Rd, x, gx, gS, a);
                        if (sycl::any_of_group(sg, on) and on)
#pragma unroll
                            for (int m = 0; m < TPg::NIN; ++m)
                                atomic_add(X_adj + ((long)j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
                    }
                    T D[Cc::ND];
                    load_n<T, Cc::ND>(Dsm + c*Cc::ND, D);
                    T Gd[3];
                    grad_staged<Sh::l_max>(gS, D, Gd);
                    warp_force4<SGW<Sh, T>>(sg, W.edge_forces + 3*e, a, Gd, qv, M.inv_h);
                }
            }
        });
    });
}

// All degrees l1 of a middle layer in one launch (forward, or reverse with the Q form for the degrees in QMASK): the
// work-groups of degree G are the virtual blocks [off[G], off[G + 1]) of a 1-d grid, nbx[G] along x times es[G] edge
// splits. Small systems run l_max + 1 times fewer launches. Needs one work-group size for all degrees.
struct FuseGrid { int off[6]; int nbx[6]; int es[6]; };
template <class Sh, bool BIG> struct Fuse1 {
    static constexpr int NG = Sh::l_max + 1;
    static constexpr int TPB = LPN<Sh, 1, 0, BIG>::tpb;
    static constexpr bool same() {
        bool r = true;
        [&]<int... G>(std::integer_sequence<int, G...>) {
            ((r = r and LPN<Sh, 1, G, BIG>::tpb == TPB and LPN<Sh, 1, G, BIG>::npb*LPN<Sh, 1, G, BIG>::v == TPB), ...);
        }(std::make_integer_sequence<int, NG>{});
        return r;
    }
    static constexpr bool ok = same();
};
template <class Sh, class T, bool BIG, bool REV, int QMASK>
void k_tp1_fused(sycl::queue& q, const FuseGrid& F, long smem, const Model<T>& M, const Graph& G_, const Work<T>& W,
                 int layer) {
    using FU = Fuse1<Sh, BIG>;
    q.submit([&](sycl::handler& h) {
        auto sm = local_mem<T>(h, smem);
        const sycl::nd_range<3> range(dims(size_t(F.off[FU::NG])*FU::TPB), dims(FU::TPB));
        auto props = [] {
            if constexpr (REV) return sycl::ext::oneapi::experimental::properties{};
            else return fwd_props<T>();
        }();
        h.parallel_for(range, props, KERNEL_SG(SGW<Sh, T>) {
            const int b = int(it.get_group(2));
            unroll<FU::NG>([&]<int G>() {
                if (b >= F.off[G] and b < F.off[G + 1]) {
                    const int lb = b - F.off[G];
                    const int bx = lb%F.nbx[G];
                    const int by = lb/F.nbx[G];
                    if constexpr (not REV) tp1_body<Sh, T, G, BIG>(it, bx, by, F.es[G], local_ptr(sm), M, G_, W, layer);
                    else if constexpr ((QMASK >> G) & 1)
                        tp1_reverse_q_body<Sh, T, G, BIG>(it, bx, by, F.es[G], local_ptr(sm), M, G_, W, layer);
                    else tp1_reverse_body<Sh, T, G, BIG>(it, bx, by, F.es[G], local_ptr(sm), M, G_, W, layer);
                }
            });
        });
    });
}

// solid harmonics of the reversed edge: S_lm(-r) = (-1)^l S_lm(r). Written with compile-time indices: IGC does not
// unroll a loop over lm with this condition and puts S, Sr, gS and gSr in private memory (slow pair kernels).
constexpr bool even_degree(int lm) { return lm == 0 or (lm >= 4 and lm < 9) or (lm >= 16 and lm < 25); }
template <int n, class T> inline void reflect(const T* S, T* Sr) {
    unroll<n>([&]<int lm>() { Sr[lm] = even_degree(lm) ? S[lm] : -S[lm]; });
}
template <int n, class T> inline void add_reflected(T* g, const T* gr) {
    unroll<n>([&]<int lm>() { g[lm] = even_degree(lm) ? g[lm] + gr[lm] : g[lm] - gr[lm]; });
}

inline bool pair_owner(int i, int j, float x, float y, float z) {
    if (i != j) return (i < j) != (((i ^ j) & 1) != 0);
    return x > 0 or (x == 0 and (y > 0 or (y == 0 and z > 0)));
}

// a work-group's coordinates in a fused launch (virtual x block, edge split and split count)
struct VThread {
    const sycl::nd_item<3>& it;
    int bx_, by_, gy_;
    int tx() const { return int(it.get_local_id(2)); }
    int bx() const { return bx_; }
    int by() const { return by_; }
    int gy() const { return gy_; }
    SubGroup sg() const { return it.get_sub_group(); }
};

template <class Sh, class T, int G, bool BIG, class TH = Thread>
inline void pair_forward(const TH& th, const Model<T>& M, const Graph& G_, const Work<T>& W, int layer, int i,
                         int u) {
    using TPg = TP<Sh, 2, G>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = LPN<Sh, 2, G, BIG>::cp;
    constexpr int LL = TPg::MUL/CP;
    const SubGroup sg = th.sg();
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
    const int ea = es0 + (int)((long)ecnt*th.by()/th.gy());
    const int eb = es0 + (int)((long)ecnt*(th.by() + 1)/th.gy());
    for (int e0 = ea; e0 < eb; e0 += SGW<Sh, T>) {
        const EdgeB<T> b = edge_batch(sg, G_, W, e0, eb);
        const int nb = sycl::min(SGW<Sh, T>, eb - e0);
#pragma unroll pair_unroll
        for (int c = 0; c < nb; ++c) {
            const EdgeB<T> ed = edge_bcast(sg, b, c);
            const int j = ed.j;
            const bool gh = j >= G_.N;
            if (not gh and not pair_owner(i, j, ed.q.x, ed.q.y, ed.q.z)) continue;
            T S[Sh::num_lm], Sr[Sh::num_lm];
            solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
            reflect<Sh::num_lm>(S, Sr);
            const auto blk = spline_block<T>(M, layer, ed.iv, NF);
            const T xs = (ed.xs - M.half_h)*M.inv_h;
#pragma unroll
            for (int cc = 0; cc < CP; ++cc) {
                const int uc = u + cc*LL;
                T x[TPg::NIN];
                T R[TPg::NK] = {};
#pragma unroll
                for (int m = 0; m < TPg::NIN; ++m) x[m] = on ? X[((long)j*TPg::NIN + m)*TPg::MUL + uc] : T(0);
                if (on) radial<TPg>(blk, uc, xs, R);
                tp_forward<TPg>(M.cgd, acc[cc], R, S, x);
                if (not gh) {
                    T t[TPg::NOUT] = {};
                    tp_forward<TPg>(M.cgd, t, R, Sr, xi[cc]);
                    if (on) atomic_add(W.Phi + msg_index<Sh, 2, G>(N, j, uc, 0), t[0]);
                }
            }
        }
    }
    if (on)
#pragma unroll
        for (int c = 0; c < CP; ++c) atomic_add(W.Phi + msg_index<Sh, 2, G>(N, i, u + c*LL, 0), acc[c][0]);
}

template <class Sh, class T, int G, bool BIG, class TH = Thread>
inline void pair_reverse(const TH& th, const Model<T>& M, const Graph& G_, const Work<T>& W, int layer, int i,
                         int u) {
    using TPg = TP<Sh, 2, G>;
    constexpr int NF = TPg::Ly::nf;
    constexpr int CP = LPN<Sh, 2, G, BIG>::cp;
    constexpr int LL = TPg::MUL/CP;
    const SubGroup sg = th.sg();
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
    const int ea = es0 + (int)((long)ecnt*th.by()/th.gy());
    const int eb = es0 + (int)((long)ecnt*(th.by() + 1)/th.gy());
    for (int e0 = ea; e0 < eb; e0 += SGW<Sh, T>) {
        const EdgeB<T> b = edge_batch(sg, G_, W, e0, eb);
        const int nb = sycl::min(SGW<Sh, T>, eb - e0);
#pragma unroll pair_unroll
        for (int c = 0; c < nb; ++c) {
            const int e = e0 + c;
            const EdgeB<T> ed = edge_bcast(sg, b, c);
            const int j = ed.j;
            const bool gh = j >= G_.N;
            if (not gh and not pair_owner(i, j, ed.q.x, ed.q.y, ed.q.z)) continue;
            T S[Sh::num_lm], Sr[Sh::num_lm];
            solid_harmonics<Sh::l_max>(ed.q.x, ed.q.y, ed.q.z, S);
            reflect<Sh::num_lm>(S, Sr);
            const auto blk = spline_block<T>(M, layer, ed.iv, NF);
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
                tp_reverse<TPg>(M.cgd, adji[cc], R, Rd, S, x, gx, gS, a);
                if (not gh) tp_reverse<TPg>(M.cgd, adjj, R, Rd, Sr, xi[cc], gxi[cc], gSr, a);
                if (on)
#pragma unroll
                    for (int m = 0; m < TPg::NIN; ++m) atomic_add(X_adj + ((long)j*TPg::NIN + m)*TPg::MUL + uc, gx[m]);
            }
            add_reflected<Sh::num_lm>(gS, gSr);
            T Gd[3], f[3];
            grad_direct<Sh::l_max>(gS, S, Gd);
            force_solid(a*M.inv_h, ed.q, Gd, f);
            warp_add3<SGW<Sh, T>>(sg, W.edge_forces + 3*e, f[0], f[1], f[2]);
        }
    }
    if (on)
#pragma unroll
        for (int c = 0; c < CP; ++c)
#pragma unroll
            for (int m = 0; m < TPg::NIN; ++m)
                atomic_add(X_adj + ((long)i*TPg::NIN + m)*TPg::MUL + u + c*LL, gxi[c][m]);
}

template <class Sh, class T, int G, bool BIG, bool REV>
void k_tp2(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G_, const Work<T>& W, int layer) {
    using L = LPN<Sh, 2, G, BIG>;
    q.parallel_for(sycl::nd_range<3>(grid*dims(L::tpb), dims(L::tpb)), KERNEL_SG(SGW<Sh, T>) {
        const Thread th{it};
        const int i = th.bx()*L::npb + th.tx()/L::v;
        const int u = th.tx()%L::v;
        if (i >= G_.N) return;
        if constexpr (REV) pair_reverse<Sh, T, G, BIG>(th, M, G_, W, layer, i, u);
        else pair_forward<Sh, T, G, BIG>(th, M, G_, W, layer, i, u);
    });
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
template <class Sh, class T, bool BIG, bool REV>
void k_tp2_fused(sycl::queue& q, sycl::range<3> grid, const Model<T>& M, const Graph& G_, const Work<T>& W,
                 int layer) {
    using F = PairFused<Sh, BIG>;
    q.parallel_for(sycl::nd_range<3>(grid*dims(F::TPB), dims(F::TPB)), KERNEL_SG(SGW<Sh, T>) {
        const Thread th{it};
        const int w = th.tx() >> 5;
        const int u = th.tx() & 31;
        const int i = th.bx()*F::NPB + w/F::NG;
        const int g = w%F::NG;
        if (i >= G_.N) return;
        unroll<F::NG>([&]<int G>() {
            if (g == G) {
                if constexpr (REV) pair_reverse<Sh, T, G, BIG>(th, M, G_, W, layer, i, u);
                else pair_forward<Sh, T, G, BIG>(th, M, G_, W, layer, i, u);
            }
        });
    });
}

// All degrees of the pair layer in one launch where k_tp2_fused does not apply (below n_big): virtual blocks per
// degree as in k_tp1_fused; for tiny, launch-bound systems.
template <class Sh, bool BIG> struct Fuse2 {
    static constexpr int NG = Sh::l_max + 1;
    static constexpr int TPB = LPN<Sh, 2, 0, BIG>::tpb;
    static constexpr bool same() {
        bool r = true;
        [&]<int... G>(std::integer_sequence<int, G...>) {
            ((r = r and LPN<Sh, 2, G, BIG>::tpb == TPB and LPN<Sh, 2, G, BIG>::npb*LPN<Sh, 2, G, BIG>::v == TPB), ...);
        }(std::make_integer_sequence<int, NG>{});
        return r;
    }
    static constexpr bool ok = same();
};
template <class Sh, class T, bool BIG, bool REV>
void k_tp2_vfused(sycl::queue& q, const FuseGrid& F, const Model<T>& M, const Graph& G_, const Work<T>& W,
                  int layer) {
    using FU = Fuse2<Sh, BIG>;
    q.parallel_for(sycl::nd_range<3>(dims(size_t(F.off[FU::NG])*FU::TPB), dims(FU::TPB)), KERNEL_SG(SGW<Sh, T>) {
        const int b = int(it.get_group(2));
        unroll<FU::NG>([&]<int G>() {
            if (b >= F.off[G] and b < F.off[G + 1]) {
                using L = LPN<Sh, 2, G, BIG>;
                const int lb = b - F.off[G];
                const VThread th{it, lb%F.nbx[G], lb/F.nbx[G], F.es[G]};
                const int i = th.bx()*L::npb + th.tx()/L::v;
                const int u = th.tx()%L::v;
                if (i < G_.N) {
                    if constexpr (REV) pair_reverse<Sh, T, G, BIG, VThread>(th, M, G_, W, layer, i, u);
                    else pair_forward<Sh, T, G, BIG, VThread>(th, M, G_, W, layer, i, u);
                }
            }
        });
    });
}

template <class T> inline T sigm(T x) { return 1/(1 + sycl::exp(-x)); }
template <class Sh> inline void gated_of(int k, int& l, int& c) {
    l = 1;
    c = k;
#pragma unroll
    for (int q = 1; q < Sh::l_max; ++q)
        if (l == q and c >= Sh::mulx(q)) {
            c -= Sh::mulx(q);
            l = q + 1;
        }
}
template <class Sh, class T, bool SC>
void k_gate(sycl::queue& q, const Model<T>& M, const Graph& G_, const Work<T>& W, int layer, int xout) {
    const long total = (long)G_.N*Sh::ngt;
    q.parallel_for(sycl::nd_range<3>(dims(blocks(total, 128)*128), dims(128)), KERNEL {
        const long idx = (long)it.get_group(2)*128 + it.get_local_id(2);
        if (idx >= total) return;
        const int k = idx%Sh::ngt;
        const int i = idx/Sh::ngt;
        const long N = G_.N;
        T* A = W.A[layer];
        T* Hh = W.H[xout];
        const T cst = M.gate_cst;
        if (k < Sh::nsc) {
            const long os = (long)i*Sh::muly(0) + k;
            const T ys = W.A_linear[os] + (SC ? W.A_sc[os] : T(0));
            A[os] = ys;
            Hh[(long)i*Sh::nsc + k] = ys*sigm(ys)*cst;
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
                Hh[N*Sh::xblk(l) + ((long)i*(2*l + 1) + m)*Sh::mulx(l) + c] = v*g;
            }
        }
    });
}
template <class Sh, class T>
void k_gate_reverse(sycl::queue& q, const Model<T>& M, const Graph& G_, const Work<T>& W, int layer) {
    const long total = (long)G_.N*Sh::ngt;
    q.parallel_for(sycl::nd_range<3>(dims(blocks(total, 128)*128), dims(128)), KERNEL {
        const long idx = (long)it.get_group(2)*128 + it.get_local_id(2);
        if (idx >= total) return;
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
                ga = sycl::fma(xa, A[oy], ga);
            }
            W.A_adj[og] = ga*cst*sg*(1 + yg*(1 - sg));
        }
    });
}
template <class Sh, class T> void k_tail(sycl::queue& q, const Model<T>& M, const Graph& G_, const Work<T>& W) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> part(sycl::range<1>(Sh::nsc/32), h);
        h.parallel_for(sycl::nd_range<3>(dims(G_.N*Sh::nsc), dims(Sh::nsc)), KERNEL {
            const Thread th{it};
            const int i = th.bx();
            const int k = th.tx();
            const int t = G_.elem[i];
            const long o = (long)i*Sh::nsc + k;
            const T y = W.A_linear[o] + W.A_sc[o];
            const T s = sigm(y);
            const T cst = M.gate_cst;
            T v = M.readout_weights[k]*y*s*cst;
            v = warp_sum(th.sg(), v);
            if (th.lane() == 0) part[th.warp()] = v;
            th.sync();
            if (k == 0) {
                T sum = 0;
                for (int w = 0; w < Sh::nsc/32; ++w) sum += part[w];
                W.node_energies[i] += (double)M.per_type_scale[t]*(double)sum + (double)M.per_type_shift[t];
            }
            W.A_adj[o] = M.per_type_scale[t]*M.readout_weights[k]*cst*s*(1 + y*(1 - s));
        });
    });
}

constexpr int SORT_MAX = 256;

template <class T>
inline double edge_record(const Model<T>& M, int e, int ti, int tj, int row, int nb, double dx, double dy, double dz,
                          int* neigh_rows, int* neigh_indices, double* vec, T* geometry, int* interval, T* xs,
                          T* edge_forces) {
    const double r = sycl::sqrt(dx*dx + dy*dy + dz*dz);
    neigh_rows[e] = row;
    neigh_indices[e] = nb;
    vec[3*e] = dx;
    vec[3*e+1] = dy;
    vec[3*e+2] = dz;
    store4(geometry + 4*e, V4<T>{T(dz), T(dx), T(dy), T(1/r)});
    const int iv = sycl::min(int(r/M.h), M.intervals - 1);
    interval[e] = iv;
    xs[e] = T(r - M.h*iv);
    double E = 0;
    double dE = 0;
    if (M.has_zbl) {
        const int tp = ti*M.num_elements + tj;
        const double a = M.zbl_a[tp];
        const double q = a*r;
        const double e1 = 0.02817*sycl::exp(-0.20162*q);
        const double e2 = 0.28022*sycl::exp(-0.40290*q);
        const double e3 = 0.50986*sycl::exp(-0.94229*q);
        const double e4 = 0.18175*sycl::exp(-3.19980*q);
        const double psi = e1 + e2 + e3 + e4;
        const double dpsi = a*(-0.20162*e1 - 0.40290*e2 - 0.94229*e3 - 3.19980*e4);
        const double p = M.zbl_p;
        const double u = r/M.r_cut;
        double fc = 0;
        double dfc = 0;
        if (u < 1) {
            const double up = sycl::pow(u, p);
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

template <class T>
void k_outputs(sycl::queue& q, const Graph& G, const double* node_energies, const double* vec, const T* edge_forces,
               double* out) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> part(sycl::range<1>(7*4), h);
        h.parallel_for(sycl::nd_range<3>(dims(blocks(32L*G.N, 128)*128), dims(128)), KERNEL {
            const Thread th{it};
            const int n = th.bx()*4 + th.warp();
            const int lane = th.lane();
            const int warp = th.warp();
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
                const double s = warp_sum(th.sg(), v[c]);
                if (lane == 0) part[c*4 + warp] = s;
            }
            th.sync();
            if (th.tx() < 7)
                atomic_add(out + th.tx(), part[th.tx()*4] + part[th.tx()*4 + 1] + part[th.tx()*4 + 2]
                                              + part[th.tx()*4 + 3]);
        });
    });
}

inline void k_atom_energies(sycl::queue& q, int N, const int* atom, const double* node_energies, double* eatom) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> n) { eatom[atom[n]] = node_energies[n]; });
}

inline void exclusive_scan(sycl::queue& q, const int* in, int* out, int n) {
    q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
        const auto g = it.get_group();
        int carry = 0;
        for (int base = 0; base < n; base += 1024) {
            const int i = base + int(it.get_local_id(0));
            const int v = i < n ? in[i] : 0;
            const int s = sycl::exclusive_scan_over_group(g, v, sycl::plus<int>());
            if (i < n) out[i] = carry + s;
            carry += sycl::reduce_over_group(g, v, sycl::plus<int>());
        }
    });
}

// ----- LAMMPS kernels (pair_symmetrix_nequip_foundation_kokkos.cpp) -----

inline double ax(const LammpsAtoms& A, int i, int d) { return A.x[i*A.xs0 + d*A.xs1]; }
inline int aneigh(const LammpsAtoms& A, int i, int jj) { return A.neighbors[i*A.ns0 + jj*A.ns1] & A.neighmask; }

inline void k_elements(sycl::queue& q, int n, const int* type, const int* type_map, int* elem) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { elem[i] = type_map[type[i] - 1]; });
}

inline void k_count(sycl::queue& q, int N, double rc2, const int* atom, const int* node_type, const LammpsAtoms& A,
                    int* count, unsigned short* cand, unsigned* err) {
    q.parallel_for(sycl::nd_range<3>(dims(blocks(32L*N, 128)*128), dims(128)), KERNEL {
        const Thread th{it};
        const SubGroup sg = th.sg();
        const int n = th.bx()*4 + th.warp();
        const int lane = th.lane();
        if (n >= N) return;
        const int i = atom[n];
        const int nn = A.numneigh[i];
        const double xi = ax(A, i, 0);
        const double yi = ax(A, i, 1);
        const double zi = ax(A, i, 2);
        unsigned short* cn = cand + (long)n*SORT_MAX;
        int cnt = 0;
        for (int j0 = 0; j0 < nn; j0 += 32) {
            const int jj = j0 + lane;
            bool inside = false;
            if (jj < nn) {
                const int j = aneigh(A, i, jj);
                const double dx = ax(A, j, 0) - xi;
                const double dy = ax(A, j, 1) - yi;
                const double dz = ax(A, j, 2) - zi;
                inside = dx*dx + dy*dy + dz*dz < rc2;
            }
            const uint32_t m = ballot(sg, inside);
            const int c = cnt + popc(m & ((1u << lane) - 1));
            if (inside and c < SORT_MAX) cn[c] = (unsigned short)jj;
            cnt += popc(m);
        }
        if (lane == 0) {
            count[n] = cnt;
            if (A.type_map[A.type[i] - 1] != node_type[n]) atomic_or(err, ERR_TYPES);
        }
    });
}

template <class Sh, class T>
void k_fill(sycl::queue& q, const Model<T>& M, const Graph& G, double rc2, const int* atom, const int* row_of_atom,
            const LammpsAtoms& A, const int* count, const unsigned short* cand, int* neigh_rows, int* neigh_indices,
            double* vec, T* geometry, int* interval, T* xs, T* edge_forces, double* node_energies, unsigned* err) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::float4, 1> keys_acc(sycl::range<1>(4*(SORT_MAX + 4)/4), h);
        h.parallel_for(sycl::nd_range<3>(dims(blocks(32L*G.N, 128)*128), dims(128)), KERNEL {
            const Thread th{it};
            const SubGroup sg = th.sg();
            const int n = th.bx()*4 + th.warp();
            const int lane = th.lane();
            const int w = th.warp();
            if (n >= G.N) return;
            float* keys = reinterpret_cast<float*>(keys_acc.get_multi_ptr<sycl::access::decorated::no>().get())
                          + w*(SORT_MAX + 4);
            const int i = atom[n];
            const int cnt = count[n];
            const int s0 = G.start[n];
            const int ti = G.elem[n];
            const double xi = ax(A, i, 0);
            const double yi = ax(A, i, 1);
            const double zi = ax(A, i, 2);
            double ez = 0;
            auto add = [&](int e, int j, double dx, double dy, double dz) {
                const int row = row_of_atom[j];
                if (row < 0) {
                    atomic_or(err, ERR_ROWS);
                    return;
                }
                ez += edge_record(M, e, ti, G.elem[row], row, j, dx, dy, dz, neigh_rows, neigh_indices, vec, geometry,
                                  interval, xs, edge_forces);
            };
            if (cnt <= SORT_MAX) {
                const unsigned short* cn = cand + (long)n*SORT_MAX;
                if constexpr (Sh::sort_key != 0) {
                    for (int c = lane; c < cnt; c += 32) {
                        const int j = aneigh(A, i, cn[c]);
                        const double dx = ax(A, j, 0) - xi;
                        const double dy = ax(A, j, 1) - yi;
                        const double dz = ax(A, j, 2) - zi;
                        keys[c] = float(sycl::sqrt(dx*dx + dy*dy + dz*dz));
                    }
                    const int cnt4 = (cnt + 3) & ~3;
                    if (lane < cnt4 - cnt) keys[cnt + lane] = sycl::bit_cast<float>(0x7f800000u);
                    sycl::group_barrier(sg);
                }
                for (int c = lane; c < cnt; c += 32) {
                    const int j = aneigh(A, i, cn[c]);
                    const double dx = ax(A, j, 0) - xi;
                    const double dy = ax(A, j, 1) - yi;
                    const double dz = ax(A, j, 2) - zi;
                    int rank = c;
                    if constexpr (Sh::sort_key != 0) {
                        const float key = keys[c];
                        rank = 0;
                        for (int qq = 0; qq < cnt; qq += 4) {
                            const F4 k4 = *reinterpret_cast<const F4*>(keys + qq);
                            rank += (k4.x < key) + (k4.y < key) + (k4.z < key) + (k4.w < key);
                            rank += (k4.x == key and qq < c) + (k4.y == key and qq + 1 < c)
                                    + (k4.z == key and qq + 2 < c) + (k4.w == key and qq + 3 < c);
                        }
                    }
                    add(s0 + rank, j, dx, dy, dz);
                }
            } else {
                const int nn = A.numneigh[i];
                int pos = 0;
                for (int j0 = 0; j0 < nn; j0 += 32) {
                    const int jj = j0 + lane;
                    bool inside = false;
                    int j = 0;
                    double dx = 0;
                    double dy = 0;
                    double dz = 0;
                    if (jj < nn) {
                        j = aneigh(A, i, jj);
                        dx = ax(A, j, 0) - xi;
                        dy = ax(A, j, 1) - yi;
                        dz = ax(A, j, 2) - zi;
                        inside = dx*dx + dy*dy + dz*dz < rc2;
                    }
                    const uint32_t m = ballot(sg, inside);
                    if (inside) add(s0 + pos + popc(m & ((1u << lane) - 1)), j, dx, dy, dz);
                    pos += popc(m);
                }
            }
            ez = warp_sum(sg, ez);
            if (lane == 0) node_energies[n] = ez;
        });
    });
}

template <class T>
void k_forces(sycl::queue& q, const Graph& G, const int* atom, const int* neigh_indices, const T* edge_forces,
              const LammpsAtoms& A) {
    q.parallel_for(sycl::nd_range<3>(dims(blocks(32L*G.N, 128)*128), dims(128)), KERNEL {
        const Thread th{it};
        const int n = th.bx()*4 + th.warp();
        const int lane = th.lane();
        if (n >= G.N) return;
        double fx = 0;
        double fy = 0;
        double fz = 0;
        for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
            const int j = neigh_indices[e];
            const double f0 = edge_forces[3*e];
            const double f1 = edge_forces[3*e+1];
            const double f2 = edge_forces[3*e+2];
            atomic_add(A.f + j*A.fs0, f0);
            atomic_add(A.f + j*A.fs0 + A.fs1, f1);
            atomic_add(A.f + j*A.fs0 + 2*A.fs1, f2);
            fx += f0;
            fy += f1;
            fz += f2;
        }
        fx = warp_sum(th.sg(), fx);
        fy = warp_sum(th.sg(), fy);
        fz = warp_sum(th.sg(), fz);
        if (lane == 0) {
            const int i = atom[n];
            atomic_add(A.f + i*A.fs0, -fx);
            atomic_add(A.f + i*A.fs0 + A.fs1, -fy);
            atomic_add(A.f + i*A.fs0 + 2*A.fs1, -fz);
        }
    });
}

// Features of rows (atom -> row) as doubles in buf[atom][xw]: 0 export locals, 1 import ghosts, 2 export ghost
// adjoints (zero for locals), 3 add adjoints into locals.
template <class Sh, class T>
void k_exchange(sycl::queue& q, T* X, int N, int NR, int nlocal, int nall, const int* row_of_atom, double* buf,
                int direction) {
    const long total = (long)nall*Sh::xw;
    q.parallel_for(sycl::nd_range<3>(dims(blocks(total, 256)*256), dims(256)), KERNEL {
        const long idx = (long)it.get_group(2)*256 + it.get_local_id(2);
        if (idx >= total) return;
        const int a = int(idx/Sh::xw);
        const int k = int(idx%Sh::xw);
        int l = 0;
        for (int qq = 1; qq <= Sh::l_max; ++qq)
            if (k >= Sh::xblk(qq)) l = qq;
        const int row = a < nlocal ? row_of_atom[a] : N + a - nlocal;
        const long col = (long)NR*Sh::xblk(l) + (long)row*(2*l + 1)*Sh::mulx(l) + k - Sh::xblk(l);
        if (a < nlocal) {
            if (direction == 0) buf[idx] = double(X[col]);
            if (direction == 2) buf[idx] = 0;
            if (direction == 3) X[col] += T(buf[idx]);
        } else {
            if (direction == 1) X[col] = T(buf[idx]);
            if (direction == 2) buf[idx] = double(X[col]);
        }
    });
}

#undef KERNEL
#undef KERNEL_SG

// ----- host -----

// Runtime settings (environment; unset means the default):
// The edges of a node are split over up to split_max() work-groups so that about split_target() groups of 32 lanes
// run (the CUDA value), in systems of fewer than n_big nodes; from n_big nodes the tensor products use the "BIG"
// kernels (more channels per lane, no split).
inline long split_target() { static const long v = env_long("SYMMETRIX_SYCL_NQ_SPLIT_TARGET", 6912, 1); return v; }
inline long split_max() { static const long v = env_long("SYMMETRIX_SYCL_NQ_SPLIT_MAX", 8, 1); return v; }
// BIG pair (last) layer: all degrees in one kernel where they fit (default in fp32), or one kernel per degree (default
// in fp64); SYMMETRIX_SYCL_NQ_PAIR_FUSED=0/1
template <class T> bool pair_fused() {
    static const bool v = env_flag("SYMMETRIX_SYCL_NQ_PAIR_FUSED", sizeof(T) == 4);
    return v;
}
// pair layer: all degrees in one launch for tiny systems (fp32; SYMMETRIX_SYCL_NQ_FUSE_PAIR=0: never, 1: always)
inline bool fuse_pair(int N) {
    static const long v = env_long("SYMMETRIX_SYCL_NQ_FUSE_PAIR", -1);
    if (v >= 0) return v != 0;
    return N < 100;
}
// middle layers: all degrees in one kernel (where the work-group sizes agree) for tiny systems (launch-bound) and in
// BIG mode (better load balance); elsewhere one kernel per degree is faster (fp32; SYMMETRIX_SYCL_NQ_FUSE=0: never,
// 1: always)
template <bool BIG> bool fuse_tp(int N) {
    static const long v = env_long("SYMMETRIX_SYCL_NQ_FUSE", -1);
    if (v >= 0) return v != 0;
    return N < 100 or BIG;
}
template <class Sh> long n_big() {
    static const long v = env_long("SYMMETRIX_SYCL_NQ_NBIG", Sh::n_big);
    return v;
}
// threshold of the last (pair) layer alone (default: n_big)
template <class Sh> long n_big_pair() {
    static const long v = env_long("SYMMETRIX_SYCL_NQ_NBIG_PAIR", n_big<Sh>());
    return v;
}
// split point of the fp64 Q-form reverse of degree g (0: whole)
template <class Sh, class T> constexpr int qsplit_of(int g) {
    if constexpr (not std::is_same_v<T, double>) return 0;
    return Sh::qsplit64[g];
}
// degrees l1 whose middle-layer reverse uses the Q form (k_tp1_reverse_q), bit l1 set: the preset's qrev, or
// SYMMETRIX_SYCL_NQ_QREV
template <class Sh, class T> long qrev_mask(int N) {
    static const long v = env_long("SYMMETRIX_SYCL_NQ_QREV", -1);
    if (v >= 0) return v;
    // tiny fp64 systems are launch-bound: the split Q-form parts would be extra launches
    if constexpr (sizeof(T) == 8)
        if (N < 50) return Sh::qrev & 3;
    return Sh::qrev;
}
inline int split_for(long slots) {
    return int(std::max(1L, std::min(split_max(), split_target()/std::max(1L, slots))));
}

template <class Sh, int LT, bool BIG> int edge_split(int N) {
    if (BIG or N <= 0) return 1;
    int s = 1;
    unroll<LT == 0 ? 1 : Sh::l_max + 1>([&]<int G>() {
        s = std::max(s, split_for((long)N*(LPN<Sh, LT, G, BIG>::v/32)));
    });
    return s;
}

template <class Sh, class T, int LT, bool REV, bool BIG>
void launch_tp(sycl::queue& q, const Model<T>& M, const Graph& G, const Work<T>& W, int layer)
{
    const int N = G.N;
    if (not REV and (LT == 2 or edge_split<Sh, LT, BIG>(N) > 1))
        q.memset(W.Phi, 0, (size_t)N*Layer<Sh, LT>::msg*sizeof(T));
    if constexpr (LT == 2 and PairFused<Sh, BIG>::ok) {
        if (pair_fused<T>()) {
            using F = PairFused<Sh, BIG>;
            const int es = BIG ? 1 : split_for((long)N*F::NG);
            k_tp2_fused<Sh, T, BIG, REV>(q, dims(blocks(N, F::NPB), es), M, G, W, layer);
            return;
        }
    }
    constexpr int NG = LT == 0 ? 1 : Sh::l_max + 1;
    // fused launches only in fp32: instantiating them for fp64 changes the inlining of shared helpers and with it the
    // register allocation of the fp64 per-degree kernels
    if constexpr (std::is_same_v<T, float> and LT == 2 and Fuse2<Sh, BIG>::ok) {
        if (fuse_pair(N)) {
            FuseGrid F{};
            unroll<NG>([&]<int G_>() {
                using L = LPN<Sh, LT, G_, BIG>;
                F.nbx[G_] = blocks(N, L::npb);
                F.es[G_] = BIG ? 1 : split_for((long)N*(L::v/32));
                F.off[G_ + 1] = F.off[G_] + F.nbx[G_]*F.es[G_];
            });
            k_tp2_vfused<Sh, T, BIG, REV>(q, F, M, G, W, layer);
            return;
        }
    }
    if constexpr (std::is_same_v<T, float> and LT == 1 and Fuse1<Sh, BIG>::ok) {
        // all degrees in one launch, built for the preset's Q-form mask (per-degree launches if SYMMETRIX_SYCL_NQ_QREV
        // selects another)
        constexpr int QMASK = Sh::qrev;
        if (fuse_tp<BIG>(N) and (not REV or qrev_mask<Sh, T>(N) == QMASK)) {
            FuseGrid F{};
            long smem = 0;
            unroll<NG>([&]<int G_>() {
                using L = LPN<Sh, LT, G_, BIG>;
                F.off[G_ + 1] = F.off[G_] + blocks(N, L::npb)*(BIG ? 1 : split_for((long)N*(L::v/32)));
                F.nbx[G_] = blocks(N, L::npb);
                F.es[G_] = BIG ? 1 : split_for((long)N*(L::v/32));
                if constexpr (REV and ((QMASK >> G_) & 1)) {
                    using C = SMQ<Sh, G_, BIG, T>;
                    static_assert(C::TPB == Fuse1<Sh, BIG>::TPB and C::NPB == L::npb);
                    smem = std::max(smem, (long)C::WPB*C::PER_WARP);
                } else {
                    using C = SMC<Sh, G_, BIG, T>;
                    smem = std::max(smem, (long)C::WPB*C::PER_WARP);
                }
            });
            k_tp1_fused<Sh, T, BIG, REV, QMASK>(q, F, smem, M, G, W, layer);
            return;
        }
    }
    unroll<NG>([&]<int G_>() {
        // (the fp64 middle-layer kernels use the preset's fp64 channels per lane)
        using L = std::conditional_t<LT == 1, LPN<Sh, LT, G_, BIG, T>, LPN<Sh, LT, G_, BIG>>;
        const int es = BIG ? 1 : split_for((long)N*(L::v/32));
        const sycl::range<3> grid = dims(blocks(N, L::npb), es);
        if constexpr (LT == 0) {
            if constexpr (REV) k_tp0_reverse<Sh, T, BIG>(q, grid, M, G, W);
            else k_tp0<Sh, T, BIG>(q, grid, M, G, W);
        } else if constexpr (LT == 2) {
            k_tp2<Sh, T, G_, BIG, REV>(q, grid, M, G, W, layer);
        } else if constexpr (REV) {
            if ((qrev_mask<Sh, T>(N) >> G_) & 1) {
                using C = SMQ<Sh, G_, BIG, T>;
                constexpr int S1 = qsplit_of<Sh, T>(G_);
                constexpr int NK = TP<Sh, 1, G_>::NK;
                const sycl::range<3> g = dims(blocks(N, C::NPB), es);
                if constexpr (S1 > 0 and S1 < NK) {
                    k_tp1_reverse_qs<Sh, T, G_, BIG, 0, S1>(q, g, M, G, W, layer);
                    k_tp1_reverse_qs<Sh, T, G_, BIG, S1, NK>(q, g, M, G, W, layer);
                } else {
                    k_tp1_reverse_q<Sh, T, G_, BIG>(q, g, M, G, W, layer);
                }
            } else {
                k_tp1_reverse<Sh, T, G_, BIG>(q, grid, M, G, W, layer);
            }
        } else {
            k_tp1<Sh, T, G_, BIG>(q, grid, M, G, W, layer);
        }
    });
}

inline std::vector<double> spline(const std::vector<double>& v, const std::vector<double>& d, int l, double h,
                                  double r_low)
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

// ----- engine (port of NequIPFoundationCuda) -----

template <class Precision, bool mixed> class Engine {
public:
    struct Weight { Buffer<Precision> w; };

    Engine(const std::string& filename, const std::string& prefix, sycl::queue q) : q(q) { load(filename, prefix); }

    sycl::queue q;
    int num_elements, num_layers;
    double r_cut;
    int l_max, emb, mul0;
    std::vector<int> atomic_numbers;

    int N = 0;
    int NR = 0;
    long E = 0;
    std::vector<int> type_start;
    Buffer<int> atom_of_node, row_of_atom, elem, count, start, neigh_indices, neigh_rows, elem_all;
    Buffer<unsigned short> cand;
    Buffer<double> vec;
    Buffer<Precision> geometry, edge_forces;
    template <class Sh> void reserve();

    Buffer<double> node_energies, outputs;
    Buffer<unsigned> err;
    Model<Precision> M{};
    Graph G{};
    Work<Precision> W{};
    template <class Sh> void arguments();
    template <class Sh> void compute_layer(int t);
    template <class Sh> void compute_readout();
    template <class Sh> void reverse_layer(int t);

    bool has_zbl;
    double zbl_p = 0;
    Buffer<double> zbl_a, zbl_c;

    int K;
    double h;
    Buffer<int> interval;
    Buffer<Precision> xs;
    std::vector<std::unique_ptr<Buffer<Precision>>> R;

    std::vector<double> coefficients;
    Buffer<double> cg_device;
    template <class Sh, bool REV> void tensor_product(int t);

    Buffer<Precision> X0;
    Weight X_weights, A_weights, A_sc_weights;
    std::vector<long> X_weights_offset, A_weights_offset, A_sc_weights_offset;
    std::vector<std::unique_ptr<Buffer<Precision>>> X, A;
    Buffer<Precision> H_even, H_odd, Phi, A_linear, A_sc, A_adj, X_adj, H_adj_linear, H_adj_sc;

    Precision gate_cst;
    Buffer<Precision> readout_weights, per_type_scale, per_type_shift;

    struct Product {
        const Precision* a;
        int lda;
        const Weight* w;
        long off;
        bool trans;
        Precision* c;
        int ldc, m, n, k;
    };
    std::vector<Product> queue;
    void product(const Precision*, int, const Weight&, long, bool, Precision*, int, int, int, int);
    void products();

    void load(const std::string&, const std::string&);
};

template <class B> static void grow(std::vector<std::unique_ptr<B>>& v, size_t n) {
    v.resize(n);
    for (auto& b : v)
        if (not b) b = std::make_unique<B>();
}

template <class Precision, bool mixed>
void Engine<Precision, mixed>::load(const std::string& filename, const std::string& prefix)
{
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
                K = int(spl_values[0][0].size()) - 1;
                h = r_cut/K;
                R.clear();
                grow(R, NL);
                unroll<3>([&]<int LT>() {
                    if (LT == 1 and NL < 3) return;
                    using Ly = Layer<Sh, LT>;
                    const int q0 = LT == 0 ? 0 : LT == 2 ? NL - 1 : 1;
                    const int q1 = LT == 0 ? 1 : LT == 2 ? NL : NL - 1;
                    for (int qq = q0; qq < q1; ++qq) {
                        const int nf = Ly::nf;
                        std::vector<Precision> c((long)K*4*nf);
                        for (int k = 0; k < Ly::n; ++k)
                            for (int u = 0; u < Ly::ins[k].mul; ++u) {
                                const int fn = Ly::ins[k].roff + u;
                                const auto& v = spl_values[qq][fn];
                                const auto& d = spl_derivs[qq][fn];
                                if (int(v.size()) != K + 1)
                                    throw std::runtime_error(prefix + "radial splines with different knots");
                                const auto cc = spline(v, d, Ly::ins[k].l2, h, h);
                                for (int iv = 0; iv < K; ++iv)
                                    for (int a = 0; a < 4; ++a) c[((long)iv*nf + fn)*4 + a] = Precision(cc[4*iv + a]);
                            }
                        R[qq]->upload(c);
                    }
                });
                std::vector<std::array<long, 3>> b1, b2, bs;
                auto block = [&](std::vector<Precision>& w, std::vector<long>& off,
                                 std::vector<std::array<long, 3>>& blks, const double* m, long rows, long cols) {
                    off.push_back(long(w.size()));
                    blks.push_back({long(w.size()), rows, cols});
                    for (long x = 0; x < rows*cols; ++x) w.push_back(Precision(m[x]));
                };
                std::vector<Precision> w1, w2, ws;
                X_weights_offset.clear();
                A_weights_offset.clear();
                A_sc_weights_offset.clear();
                for (int qq = 0; qq < NL; ++qq) {
                    const int LT = qq == 0 ? 0 : qq == NL - 1 ? 2 : 1;
                    for (int l = 0; l <= LM; ++l) {
                        if (qq > 0)
                            block(w1, X_weights_offset, b1, x_weights[qq][2*l].data(), Sh::mulx(l), Sh::mulx(l));
                        else X_weights_offset.push_back(0);
                        if (LT == 2 and l > 0) {
                            A_weights_offset.push_back(0);
                            for (int t = 0; t < T; ++t) A_sc_weights_offset.push_back(0);
                            continue;
                        }
                        const long mid = Phi_mul[qq][2*l];
                        const long cols = LT == 2 ? Sh::nsc : Sh::muly(l);
                        if (long(a_weights[qq][2*l].size()) != mid*cols
                            or (qq > 0 and long(a_sc_weights[qq][2*l].size()) != T*Sh::mulx(l)*cols))
                            throw std::runtime_error(prefix + "linear_2 or self-connection of layer "
                                                     + std::to_string(qq) + " does not match");
                        block(w2, A_weights_offset, b2, a_weights[qq][2*l].data(), mid, cols);
                        for (int t = 0; t < T; ++t) {
                            if (qq > 0)
                                block(ws, A_sc_weights_offset, bs, a_sc_weights[qq][2*l].data() + t*Sh::mulx(l)*cols,
                                      Sh::mulx(l), cols);
                            else A_sc_weights_offset.push_back(0);
                        }
                    }
                }
                X_weights.w.upload(w1);
                A_weights.w.upload(w2);
                A_sc_weights.w.upload(ws);
                std::vector<Precision> x((long)T*Sh::emb);
                for (int t = 0; t < T; ++t)
                    for (int v = 0; v < Sh::emb; ++v) {
                        double acc = 0;
                        for (int u = 0; u < Sh::emb; ++u) acc += embedding[t][u]*x_weights[0][0][u*Sh::emb + v];
                        x[(long)t*Sh::emb + v] = Precision(acc);
                    }
                X0.upload(x);
            }))
        throw std::runtime_error(prefix + "the model (" + std::to_string(num_layers) + " layers, l_max "
                                 + std::to_string(l_max) + ", embedding " + std::to_string(emb) + ", "
                                 + std::to_string(mul0)
                                 + " scalars) is not of a preset shape compiled into this build: " + preset_names());
    auto reals = [&](const char* key) {
        std::vector<Precision> v;
        for (double a : js[key].get<std::vector<double>>()) v.push_back(Precision(a));
        return v;
    };
    readout_weights.upload(reals("readout_weights"));
    per_type_scale.upload(reals("per_type_scale"));
    per_type_shift.upload(reals("per_type_shift"));
    {
        const double pref = has_zbl ? js.at("zbl_prefactor").get<double>() : 0.0;
        std::vector<double> za((long)T*T);
        std::vector<double> zc((long)T*T);
        for (int a = 0; a < T; ++a)
            for (int b = 0; b < T; ++b) {
                za[(long)a*T + b] = (std::pow(atomic_numbers[a], 0.23) + std::pow(atomic_numbers[b], 0.23))/0.46850;
                zc[(long)a*T + b] = pref*atomic_numbers[a]*atomic_numbers[b];
            }
        zbl_a.upload(za);
        zbl_c.upload(zc);
    }
    cg_device.upload(coefficients);
    outputs.reserve(7);
    err.reserve(1);
    q.memset(err, 0, sizeof(unsigned)).wait();
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::reserve()
{
    constexpr int NL = Sh::num_layers;
    using L0 = Layer<Sh, 0>;
    using L1 = Layer<Sh, 1>;
    using L2 = Layer<Sh, 2>;
    const long n = N;
    const long nr = NR;
    const long e = std::max<long>(E, 1);
    count.reserve(n + 1);
    q.memset(count, 0, (n + 1)*sizeof(int)).wait();
    start.reserve(n + 1);
    cand.reserve(std::max(n, 1L)*SORT_MAX);
    neigh_indices.reserve(e);
    neigh_rows.reserve(e);
    vec.reserve(3*e);
    geometry.reserve(4*e);
    interval.reserve(e);
    xs.reserve(e);
    edge_forces.reserve(3*e);
    H_even.reserve(n*Sh::xw);
    H_odd.reserve(n*Sh::xw);
    grow(X, NL);
    grow(A, NL);
    for (int qq = 1; qq < NL; ++qq) X[qq]->reserve(nr*Sh::xw);
    for (int qq = 0; qq < NL - 1; ++qq) A[qq]->reserve(n*Sh::yw);
    Phi.reserve(n*std::max({L0::msg, NL > 2 ? L1::msg : 0, L2::msg}));
    A_linear.reserve(n*Sh::yw);
    A_sc.reserve(n*Sh::yw);
    A_adj.reserve(n*Sh::yw);
    X_adj.reserve(nr*Sh::xw);
    H_adj_linear.reserve(n*Sh::xw);
    H_adj_sc.reserve(n*Sh::xw);
    node_energies.reserve(n);
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::arguments()
{
    using T = Precision;
    constexpr int NL = Sh::num_layers;
    M = {};
    M.num_elements = num_elements;
    M.intervals = K;
    M.h = T(h);
    M.half_h = T(0.5*h);
    M.inv_h = T(1/h);
    for (int qq = 0; qq < NL; ++qq) M.R[qq] = *R[qq];
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
    M.cgd = cg_device;
    G = {N, NR, elem, start, neigh_rows};
    W = {};
    W.geometry = geometry;
    W.interval = interval;
    W.xs = xs;
    W.H[0] = H_even;
    W.H[1] = H_odd;
    for (int qq = 1; qq < NL; ++qq) W.X[qq] = *X[qq];
    for (int qq = 0; qq < NL - 1; ++qq) W.A[qq] = *A[qq];
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
void Engine<Precision, mixed>::compute_layer(int t)
{
    constexpr int LM = Sh::l_max;
    constexpr int LL = Sh::num_layers - 1;
    const long n = N;
    const long nr = NR;
    const int cur = t & 1;

    tensor_product<Sh, false>(t);

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

    if (t < LL and N > 0) {
        if (t == 0) k_gate<Sh, Precision, false>(q, M, G, W, t, 1 - cur);
        else k_gate<Sh, Precision, true>(q, M, G, W, t, 1 - cur);
    }

    if (t < LL) {
        for (int l = 0; l <= LM; ++l)
            product(W.H[1 - cur] + n*Sh::xblk(l), Sh::mulx(l), X_weights, X_weights_offset[(t + 1)*(LM + 1) + l], false,
                    W.X[t + 1] + nr*Sh::xblk(l), Sh::mulx(l), N*(2*l + 1), Sh::mulx(l), Sh::mulx(l));
        products();
    }
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::compute_readout()
{
    if (N > 0) k_tail<Sh, Precision>(q, M, G, W);
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::reverse_layer(int t)
{
    constexpr int LM = Sh::l_max;
    constexpr int LL = Sh::num_layers - 1;
    const long n = N;
    const long nr = NR;

    if (t < LL) {
        for (int l = 0; l <= LM; ++l)
            product(W.X_adj + nr*Sh::xblk(l), Sh::mulx(l), X_weights, X_weights_offset[(t + 1)*(LM + 1) + l], true,
                    W.H_adj_linear + n*Sh::xblk(l), Sh::mulx(l), N*(2*l + 1), Sh::mulx(l), Sh::mulx(l));
        products();
    }

    if (t < LL and N > 0) k_gate_reverse<Sh, Precision>(q, M, G, W, t);

    if (t == LL) {
        q.memset(W.H_adj_sc, 0, n*Sh::xw*sizeof(Precision));
        q.memset(W.X_adj, 0, nr*Sh::xw*sizeof(Precision));
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

    if (t > 0 and t < LL) q.memset(W.X_adj, 0, nr*Sh::xw*sizeof(Precision));
    tensor_product<Sh, true>(t);
}

template <class Precision, bool mixed>
template <class Sh, bool REV>
void Engine<Precision, mixed>::tensor_product(int t)
{
    auto launch = [&](auto lt) {
        constexpr int LT = decltype(lt)::value;
        const bool big = N >= (LT == 2 ? n_big_pair<Sh>() : n_big<Sh>());
        if (big) launch_tp<Sh, Precision, LT, REV, true>(q, M, G, W, t);
        else launch_tp<Sh, Precision, LT, REV, false>(q, M, G, W, t);
    };
    if (N == 0) return;
    if (t == 0) launch(std::integral_constant<int, 0>{});
    else if (t == Sh::num_layers - 1) launch(std::integral_constant<int, 2>{});
    else if constexpr (Sh::num_layers > 2) launch(std::integral_constant<int, 1>{});
}

template <class Precision, bool mixed>
void Engine<Precision, mixed>::product(const Precision* a, int lda, const Weight& w, long off, bool trans, Precision* c,
                                       int ldc, int m, int n, int k)
{
    if (m > 0) queue.push_back({a, lda, &w, off, trans, c, ldc, m, n, k});
}

template <class Precision, bool mixed> void Engine<Precision, mixed>::products()
{
    using oneapi::mkl::transpose;
    namespace blas = oneapi::mkl::blas::column_major;
    // float32-mixed: plain fp32 GEMMs by default (on PVC at least as fast as oneMKL's three-bf16-term XMX mode for
    // these shapes); SYMMETRIX_SYCL_NQ_MIXED_GEMM=1 selects the XMX mode
    static const bool xmx = env_flag("SYMMETRIX_SYCL_NQ_MIXED_GEMM", false);
    const auto mode = mixed and xmx ? oneapi::mkl::blas::compute_mode::float_to_bf16x3
                                    : oneapi::mkl::blas::compute_mode::unset;
    for (const Product& p : queue)
        blas::gemm(q, p.trans ? transpose::trans : transpose::nontrans, transpose::nontrans, p.n, p.m, p.k,
                   Precision(1), p.w->w + p.off, p.trans ? p.k : p.n, p.a, p.lda, Precision(0), p.c, p.ldc, mode);
    queue.clear();
}

// ----- LAMMPS interface -----

template <class Precision, bool mixed> class Lammps final : public Foundation {
public:
    Lammps(const std::string& file, const std::string& prefix, sycl::queue queue) : q(queue), prefix(prefix) {
        if (not process_queue()) process_queue() = queue;
        nq = std::make_unique<Engine<Precision, mixed>>(file, prefix, queue);
    }
    double r_cut() const override { return nq->r_cut; }
    const std::vector<int>& atomic_numbers() const override { return nq->atomic_numbers; }
    int num_parts() const override { return 2*nq->num_layers - 1; }
    int comm_kind(int part) const override {
        const int NL = nq->num_layers;
        return part < NL - 1 ? 1 : part < 2*NL - 2 ? 2 : 0;
    }
    int comm_width(int) const override {
        int w = 0;
        with_preset(nq->num_layers, nq->l_max, nq->emb, nq->mul0, [&](auto s) { w = decltype(s)::xw; });
        return w;
    }
    void setup(const LammpsAtoms& A, const int* owner, bool exchange) override;
    void run(int part, const LammpsAtoms& A, double* ext, double* ext_adj) override;
    void outputs(double* eatom, double* out) override;
    unsigned errors() override {
        unsigned e = 0;
        q.memcpy(&e, nq->err.p, sizeof(unsigned)).wait();
        return e;
    }

private:
    template <class Sh> void run_shape(int part, const LammpsAtoms& A, double* ext, double* ext_adj);
    sycl::queue q;
    std::string prefix;
    std::unique_ptr<Engine<Precision, mixed>> nq;
    bool exchange = false;
    int nlocal = 0;
    int nall = 0;
};

template <class Precision, bool mixed>
void Lammps<Precision, mixed>::setup(const LammpsAtoms& A, const int* owner, bool exch)
{
    q.wait();
    exchange = exch;
    const int T = nq->num_elements;
    const int inum = A.inum;
    nlocal = A.nlocal;
    nall = A.nall;
    if (exchange and inum != nlocal)
        throw std::runtime_error(prefix + "mpi_message_passing needs every local atom in the neighbor list");
    nq->elem_all.reserve(std::max(nall, 1));
    std::vector<int> el(nall), ilist(inum), owner_h(exchange ? 0 : nall), numneigh(nlocal);
    if (nall > 0) {
        k_elements(q, nall, A.type, A.type_map, nq->elem_all);
        q.memcpy(el.data(), nq->elem_all.p, nall*sizeof(int));
        if (not exchange) q.memcpy(owner_h.data(), owner, nall*sizeof(int));
    }
    if (inum > 0) q.memcpy(ilist.data(), A.ilist, inum*sizeof(int));
    if (nlocal > 0) q.memcpy(numneigh.data(), A.numneigh, nlocal*sizeof(int));
    q.wait();

    nq->N = inum;
    nq->NR = exchange ? inum + nall - nlocal : inum;
    std::vector<int> nodes(ilist);
    std::stable_sort(nodes.begin(), nodes.end(), [&](int a, int b) { return el[a] < el[b]; });
    std::vector<int> node_elem(std::max(nq->NR, 1), 0), rows(std::max(nall, 1), -1);
    nq->type_start.assign(T + 1, 0);
    long edges = 0;
    for (int k = 0; k < nq->N; ++k) {
        node_elem[k] = el[nodes[k]];
        rows[nodes[k]] = k;
        nq->type_start[node_elem[k] + 1] += 1;
        edges += numneigh[nodes[k]];
    }
    for (int t = 0; t < T; ++t) nq->type_start[t + 1] += nq->type_start[t];
    for (int i = nlocal; i < nall; ++i) {
        if (exchange) {
            rows[i] = nq->N + i - nlocal;
            node_elem[rows[i]] = el[i];
        } else rows[i] = owner_h[i] >= 0 ? rows[owner_h[i]] : -1;
    }
    nq->E = edges;
    if (nq->N > 0) nq->atom_of_node.upload(nodes);
    nq->elem.upload(node_elem);
    nq->row_of_atom.upload(rows);
    with_preset(nq->num_layers, nq->l_max, nq->emb, nq->mul0, [&](auto s) {
        nq->template reserve<decltype(s)>();
        nq->template arguments<decltype(s)>();
    });
}

template <class Precision, bool mixed>
void Lammps<Precision, mixed>::run(int part, const LammpsAtoms& A, double* ext, double* ext_adj)
{
    with_preset(nq->num_layers, nq->l_max, nq->emb, nq->mul0, [&](auto s) {
        this->template run_shape<decltype(s)>(part, A, ext, ext_adj);
    });
}

template <class Precision, bool mixed>
template <class Sh>
void Lammps<Precision, mixed>::run_shape(int part, const LammpsAtoms& A, double* ext, double* ext_adj)
{
    constexpr int NL = Sh::num_layers;
    Engine<Precision, mixed>& e = *nq;
    const int N = e.N;
    auto edges = [&] {
        if (N == 0) return;
        const double rc2 = e.r_cut*e.r_cut;
        k_count(q, N, rc2, e.atom_of_node, e.elem, A, e.count, e.cand, e.err);
        exclusive_scan(q, e.count, e.start, N + 1);
        k_fill<Sh, Precision>(q, e.M, e.G, rc2, e.atom_of_node, e.row_of_atom, A, e.count, e.cand, e.neigh_rows,
                   e.neigh_indices, e.vec, e.geometry, e.interval, e.xs, e.edge_forces, e.node_energies, e.err);
    };
    auto forces = [&] {
        if (N > 0) k_forces(q, e.G, e.atom_of_node, e.neigh_indices, e.edge_forces.p, A);
    };
    auto xchg = [&](Precision* X, double* buf, int dir) {
        if (nall > 0) k_exchange<Sh>(q, X, N, e.NR, nlocal, nall, e.row_of_atom, buf, dir);
    };
    if (part < 0) {
        edges();
        for (int t = 0; t < NL; ++t) e.template compute_layer<Sh>(t);
        e.template compute_readout<Sh>();
        for (int t = NL - 1; t >= 0; --t) e.template reverse_layer<Sh>(t);
        forces();
        return;
    }
    if (part < NL - 1) {  // compute layer `part`, then forward comm of X[part + 1]
        if (part == 0) edges();
        else xchg(e.W.X[part], ext, 1);
        e.template compute_layer<Sh>(part);
        xchg(e.W.X[part + 1], ext, 0);
    } else if (part == NL - 1) {  // last layer, readout and its reverse, then reverse comm of X_adj
        if (part == 0) edges();
        else xchg(e.W.X[part], ext, 1);
        e.template compute_layer<Sh>(part);
        e.template compute_readout<Sh>();
        e.template reverse_layer<Sh>(part);
        if (part > 0) xchg(e.W.X_adj, ext_adj, 2);
    } else {
        const int t = 2*NL - 2 - part;
        xchg(e.W.X_adj, ext_adj, 3);
        e.template reverse_layer<Sh>(t);
        if (t > 0) xchg(e.W.X_adj, ext_adj, 2);
        else forces();
    }
    if (NL == 1 and part == 0) forces();
}

template <class Precision, bool mixed> void Lammps<Precision, mixed>::outputs(double* eatom, double* out)
{
    std::fill(out, out + 7, 0.0);
    Engine<Precision, mixed>& e = *nq;
    if (e.N == 0) return;
    q.memset(e.outputs, 0, 7*sizeof(double));
    k_outputs(q, e.G, e.node_energies, e.vec, e.edge_forces.p, e.outputs);
    if (eatom) k_atom_energies(q, e.N, e.atom_of_node, e.node_energies, eatom);
    q.memcpy(out, e.outputs.p, 7*sizeof(double)).wait();
}

}  // namespace nequip

std::unique_ptr<Foundation> make_nequip_foundation(const std::string& file, const std::string& prefix,
                                                   const std::string& dtype, sycl::queue q)
{
#ifndef SYMMETRIX_SYCL_NO_MIXED
    if (dtype == "float32-mixed") return std::make_unique<nequip::Lammps<float, true>>(file, prefix, q);
#endif
#ifndef SYMMETRIX_SYCL_NO_FLOAT
    if (dtype == "float32") return std::make_unique<nequip::Lammps<float, false>>(file, prefix, q);
#endif
#ifndef SYMMETRIX_SYCL_NO_DOUBLE
    if (dtype == "float64") return std::make_unique<nequip::Lammps<double, false>>(file, prefix, q);
#endif
    throw std::runtime_error(prefix + "precision " + dtype + " is not compiled into this SYCL build");
}

}  // namespace symmetrix_sycl
