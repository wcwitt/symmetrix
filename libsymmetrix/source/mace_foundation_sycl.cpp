// SYCL port of mace_foundation_cuda.hpp and of the CUDA kernels of pair_symmetrix_mace_foundation_kokkos.cpp, for
// Intel GPUs. The structure and kernels follow the CUDA version, tuned for PVC: the per-channel kernels run 16-lane
// sub-groups (SGW; the warp-level helpers take the width), the second-layer radial splines are stored per sub-group
// and read with block loads, the edge chunks are sized for occupancy, and k_fill also computes the second layer's
// scale factors and the ZBL terms (separate kernels in CUDA). oneMKL replaces cuBLAS; float32-mixed runs its GEMMs on
// the XMX units with three bf16 terms per factor (oneMKL compute mode float_to_bf16x3) instead of fp16 hi/lo splits.
// Runtime settings (environment): SYMMETRIX_SYCL_MIXED_GEMM=0 (float32-mixed GEMMs in plain fp32),
// SYMMETRIX_SYCL_PAIR_SYM (pair-symmetric second-layer reverse, 0/1; default: models with L_max 0),
// SYMMETRIX_SYCL_SPLIT, _SPLIT_FACTOR, _SPLIT_MAX (edge chunks per node), SYMMETRIX_SYCL_SORT_EDGES_FROM (edges
// sorted by distance from this many atoms) and SYMMETRIX_SYCL_VERBOSE (device report).

#include "foundation_sycl.hpp"
#include "foundation_sycl_common.hpp"
#include "mace_foundation_tables.hpp"
#include "nlohmann_json.hpp"

#include <oneapi/mkl/blas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace symmetrix_sycl {
namespace mace {

namespace syclex = sycl::ext::oneapi::experimental;

// Presets compiled for SYCL; a build may restrict them by index into Presets, e.g.
// -DSYMMETRIX_SYCL_MACE_PRESETS=0,2 for 96/0 (MACE-OFF small) and 128/1 (medium)
#ifdef SYMMETRIX_SYCL_MACE_PRESETS
template <int... I> auto select_presets(std::integer_sequence<int, I...>)
    -> std::tuple<std::tuple_element_t<I, Presets>...>;
template <int... I> using Pick = decltype(select_presets(std::integer_sequence<int, I...>{}));
using SyclPresets = Pick<SYMMETRIX_SYCL_MACE_PRESETS>;
#else
using SyclPresets = Presets;
#endif
template <class F> bool with_sycl_preset(int C, int L_max, int CN, F&& f) {
    return std::apply([&](auto... s) {
        return ((s.C == C and s.L_max == L_max and s.CN == CN and (f(s), true)) or ...);
    }, SyclPresets{});
}

// the compiled shapes as channels/L_max[/node channels]
inline std::string compiled_presets() {
    std::string r;
    std::apply([&](auto... s) {
        ((r += (r.empty() ? "" : ", ") + std::to_string(s.C) + "/" + std::to_string(s.L_max)
               + (s.CN != s.C ? "/" + std::to_string(s.CN) : "")), ...);
    }, SyclPresets{});
    return r;
}

struct Half2 { sycl::half x, y; };
struct alignas(8) Int2 { int x, y; };

template <class real> struct Model {
    int T, K, K0;
    double r_cut2;
    real r_min, inv_h;
    const real* basis;
    const real* R1;
    const Half2* R1h;
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
    Int2* row_pair;
    real* edge_forces;
    real *Phi0, *A0, *M0, *H1, *Phi1, *A1, *H1_adj, *A1_adj, *dPhi1, *M0_adj, *A0_adj, *Phi0_adj;
    real *s0, *s1, *dot0, *dot1;
    double* node_energies;
    unsigned* err;
    real *X0, *X1, *Tg, *T_adj, *X_adj, *B1, *B1_adj, *M0s_adj, *M1, *M1_adj;
};

template <class real> struct GemmDesc { const real *A, *W; real* C; int M, N, K, lda, ldc; bool trans; };

constexpr int WARPS = 4;

// path groups of k_Phi1/k_Phi1_reverse (separate work-groups: fewer registers per work-item; L_max 2 only)
template <class Sh> constexpr int groups = Sh::L_max == 2 ? 3 : 1;
template <class Sh> constexpr int group(int p) { return p/2%groups<Sh>; }

constexpr long row(int lm, int node, int N) {
    const int l = lm_l(lm);
    return (long)N*l*l + (long)node*(2*l+1) + lm - l*l;
}
template <class Sh> constexpr long phi1_row(int lme, int node, int N) {
    constexpr int o[4] = {0, Sh::lme_offset(1), Sh::lme_offset(2), Sh::lme_offset(3)};
    constexpr int ne[4] = {Sh::num_e(0), Sh::num_e(1), Sh::num_e(2), Sh::num_e(3)};
    const int l = lme < o[1] ? 0 : lme < o[2] ? 1 : lme < o[3] ? 2 : 3;
    const int r = lme - o[l];
    return (long)N*o[l] + ((long)node*(2*l+1) + r/ne[l])*ne[l] + r%ne[l];
}
constexpr int pair_index(int a, int b, int T) {
    return a <= b ? a*(2*T - a - 1)/2 + b : b*(2*T - b - 1)/2 + a;
}

template <class real> struct alignas(4*sizeof(real)) Vec4 { real x, y, z, w; };
template <class real> struct Edge { real x, y, z, r, t; int knot, row, pair; };
template <class real> inline Edge<real> edge(const Model<real>& M, const Graph<real>& G, int e) {
    const Vec4<real> g = *reinterpret_cast<const Vec4<real>*>(G.geometry + 4*e);
    const Int2 mt = G.row_pair[e];
    const real t = (g.w - M.r_min)*M.inv_h;
    const int knot = sycl::min(sycl::max(int(t), 0), M.K - 1);
    return {g.x, g.y, g.z, g.w, t - knot - real(0.5), knot, mt.x, mt.y};
}

template <class real> struct alignas(16) Staged { real S[16]; real x, y, z, rinv, t; int knot, row, pair; };
template <class real> inline void stage(const Model<real>& M, const Graph<real>& G, int e, Staged<real>& s) {
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

// Sub-group width of the heavy per-channel kernels: a PVC thread holds 64 (SIMD32) or 128 (SIMD16) doubles per lane
// in the large register file, so fp64 kernels sized for CUDA's 255 registers per thread run 16 lanes wide; fp32 is
// also faster at 16 lanes (fits the default register file: more threads per vector engine).
template <class real> constexpr int SGW = 16;

// Sums of f0, f1, f2 over the SG lanes of the sub-group, added to dst[0..2] (transposed butterfly).
template <int SG, class real> inline void warp_add3(const SubGroup& sg, real* dst, real f0, real f1, real f2) {
    constexpr int H = SG/2;
    constexpr int Q = SG/4;
    const int lane = int(sg.get_local_linear_id());
    const bool hi = lane & H;
    const bool bq = lane & Q;
    const real k0 = hi ? f2 : f0;
    const real k1 = hi ? real(0) : f1;
    const real s0 = hi ? f0 : f2;
    const real s1 = hi ? f1 : real(0);
    const real a = k0 + shfl_xor(sg, s0, H);
    const real b = k1 + shfl_xor(sg, s1, H);
    real c = (bq ? b : a) + shfl_xor(sg, bq ? a : b, Q);
    for (int o = Q/2; o > 0; o >>= 1) c += shfl_xor(sg, c, o);
    if ((lane & (Q - 1)) == 0 and lane < 3*Q) atomic_add(dst + lane/Q, c);
}

template <class real> inline real cubic(const real* b, int num, int f, real x) {
    return b[f] + x*(b[num + f] + x*(b[2*num + f] + x*b[3*num + f]));
}
template <class real> inline void cubic(const real* b, int num, int f, real x, real& v, real& d) {
    const real c1 = b[num + f];
    const real c2 = b[2*num + f];
    const real c3 = b[3*num + f];
    v = b[f] + x*(c1 + x*(c2 + x*c3));
    d = c1 + x*(2*c2 + 3*x*c3);
}

// Second-layer radial splines of one (pair, knot), stored per sub-group: for path p and each block of SGW channels the
// coefficients of the SGW channels are contiguous ([p][block][coefficient][lane]), so a sub-group reads a path's
// coefficients with one block load (a scalar address) instead of one gather per coefficient. float32-mixed (H)
// keeps c0, c1 in fp32 in this layout and c2, c3 as fp16 pairs ([p][block][lane]).
template <class Sh, bool H, class real> struct R1Block {
    static constexpr int F = Sh::num_paths*Sh::C;
    const real* c;
    const Half2* h;
    R1Block(const Model<real>& M, int pair, int knot) {
        const long b = (long)pair*M.K + knot;
        c = M.R1 + b*(H ? 2 : 4)*F;
        h = H ? M.R1h + b*F : nullptr;
    }
    // the coefficients of path p for the calling sub-group (all its work-items, uniform p)
    void coeffs(int p, real& c0, real& c1, real& c2, real& c3) const {
        constexpr int SG = SGW<real>;
        static_assert(Sh::C % SG == 0);
        namespace acc = sycl::access;
        const sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        const long blk = (long)p*(Sh::C/SG) + sg.get_group_linear_id();
        if constexpr (not H) {
            const auto v = sg.load<4>(sycl::address_space_cast<acc::address_space::global_space, acc::decorated::yes>(
                const_cast<real*>(c) + blk*4*SG));
            c0 = v[0];
            c1 = v[1];
            c2 = v[2];
            c3 = v[3];
        } else {
            const auto v = sg.load<2>(sycl::address_space_cast<acc::address_space::global_space, acc::decorated::yes>(
                const_cast<real*>(c) + blk*2*SG));
            const unsigned u = sg.load(sycl::address_space_cast<acc::address_space::global_space, acc::decorated::yes>(
                reinterpret_cast<unsigned*>(const_cast<Half2*>(h) + blk*SG)));
            const Half2 h23 = sycl::bit_cast<Half2>(u);
            c0 = v[0];
            c1 = v[1];
            c2 = real(h23.x);
            c3 = real(h23.y);
        }
    }
    real value(int p, real x) const {
        real c0, c1, c2, c3;
        coeffs(p, c0, c1, c2, c3);
        return c0 + x*(c1 + x*(c2 + x*c3));
    }
    void value_deriv(int p, real x, real& v, real& d) const {
        real c0, c1, c2, c3;
        coeffs(p, c0, c1, c2, c3);
        v = c0 + x*(c1 + x*(c2 + x*c3));
        d = c1 + x*(2*c2 + 3*x*c3);
    }
};

constexpr int NUM_T = 27;
constexpr int t_index(int l, int d, int mp) {
    return (l == 1 ? 0 : l == 2 ? 3 : 12) + d*(2*l - 1) + mp - (l-1)*(l-1);
}
template <class real> inline void build_T(const real* a, real* T) {
    for (int t = 0; t < NUM_T; ++t) T[t] = 0;
    static_for<num_grad_terms>([&](auto g) {
        constexpr GradTerm gt = grad_terms[g];
        T[t_index(lm_l(gt.lm), gt.d, gt.mp)] += real(gt.c)*a[gt.lm];
    });
}
template <class real> struct alignas(16) StagedD : Staged<real> { real D[num_grad_terms + 1]; };
template <class real> inline void stage_grad(StagedD<real>& s) {
    static_for<num_grad_terms>([&](auto g) {
        constexpr GradTerm t = grad_terms[g];
        s.D[g] = real(t.c)*s.S[t.mp]*s.rinv;
    });
}
template <class real> inline void grad_staged(const real* g, const real* D, real* Gd) {
    static_for<num_grad_terms>([&](auto i) {
        constexpr GradTerm t = grad_terms[i];
        Gd[t.d] += D[i]*g[t.lm];
    });
}
template <int SG, class real>
inline void warp_force(const SubGroup& sg, real* dst, real a, const real* Gd, real ux, real uy, real uz) {
    constexpr int H = SG/2;
    constexpr int Q = SG/4;
    const int lane = int(sg.get_local_linear_id());
    const bool hi = lane & H;
    const bool bq = lane & Q;
    const real p = (hi ? Gd[2] : a) + shfl_xor(sg, hi ? a : Gd[2], H);
    const real q = (hi ? Gd[0] : Gd[1]) + shfl_xor(sg, hi ? Gd[1] : Gd[0], H);
    real c = (bq ? q : p) + shfl_xor(sg, bq ? p : q, Q);
    for (int o = Q/2; o > 0; o >>= 1) c += shfl_xor(sg, c, o);
    const real A = bcast(sg, c, 0);
    const real gx = bcast(sg, c, Q);
    const real gy = bcast(sg, c, 2*Q);
    const real gz = bcast(sg, c, 3*Q);
    const real ug = ux*gx + uy*gy + uz*gz;
    if (lane < 3) {
        const real u = lane == 0 ? ux : lane == 1 ? uy : uz;
        const real g = lane == 0 ? gx : lane == 1 ? gy : gz;
        atomic_add(dst + lane, -(A*u + g - u*ug));
    }
}
template <class real> inline void edge_force(real dr, real ux, real uy, real uz, const real* Gd, real* f) {
    f[0] = -(dr*ux + Gd[1]);
    f[1] = -(dr*uy + Gd[2]);
    f[2] = -(dr*uz + Gd[0]);
}

template <class Sh, int PG, class real> inline void tp_forward(real* acc, const real* R, const real* S, const real* h) {
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
template <class Sh, int Q, class real> inline real key_w(const real* adj) {
    constexpr Term t = Sh::terms[Q];
    constexpr real cg = real(t.cg*sh_norm(t.lm1));
    constexpr int n = next_term<Sh>(Q);
    if constexpr (n < 0) return cg*adj[t.lme];
    else return cg*adj[t.lme] + key_w<Sh, n>(adj);
}
// Flip: the reversed edge, S_lm(-u) = (-1)^l S_lm(u), folded into the compile-time coefficients (k_Phi1_reverse_sym)
template <class Sh, int PG, class real, bool Flip = false>
inline void tp_reverse(const real* adj, const real* R, const real* S, const real* h, real* g_S, real* g_h, real* hs) {
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
                    if constexpr (Flip and lm_l(t.lm1) % 2) {
                        s -= w*S[t.lm1];
                        g_S[t.lm1] -= w*rh;
                    } else {
                        s += w*S[t.lm1];
                        g_S[t.lm1] += w*rh;
                    }
                }
            });
            g_h[LM2] += R[P]*s;
            hs[P] += h[LM2]*s;
        }
    });
}

template <class T, class real> inline void tree_eval(const real* w, real* v, real* out) {
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
template <class T, class real> inline void tree_adjoint(const real* w, const real* v, const real* g, real* adjx) {
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

template <class real> inline real sigmoid(real x) { return 1/(1 + sycl::exp(-x)); }

// ----- kernels -----

#define KERNEL [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]]
#define KERNEL16 [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]]
#define KERNEL_SG(SG) [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]]

inline sycl::nd_range<3> warp_range(int N) { return {dims(blocks(N, WARPS)*32*WARPS), dims(32*WARPS)}; }

// ZBL pair energy (half of it, for the edge) and the radial force factor of an edge
template <class real> inline void zbl_edge(const Model<real>& M, const real* z, real rf, double& ez, real& fz) {
    const real ce[4] = {real(-3.2), real(-0.9423), real(-0.4028), real(-0.2016)};
    const real y = rf/z[0];
    real phi = 0;
    real dphi = 0;
    for (int qq = 0; qq < 4; ++qq) {
        const real v = M.zbl_c[qq]*sycl::exp(ce[qq]*y);
        phi += v;
        dphi += ce[qq]*v/z[0];
    }
    const real v = z[2]*phi/rf;
    const real dv = z[2]*(rf*dphi - phi)/(rf*rf);
    const real c = real(M.zbl_p);
    const real u = rf/z[1];
    const real up = sycl::pow(u, c - 1);
    const real env = 1 - real(0.5)*(c+1)*(c+2)*up*u + c*(c+2)*up*u*u - real(0.5)*c*(c+1)*up*u*u*u;
    const real denv = (real(-0.5)*(c+1)*(c+2)*c*up + c*(c+2)*(c+1)*up*u - real(0.5)*c*(c+1)*(c+2)*up*u*u)/z[1];
    ez += 0.5*v*env;
    fz = real(-0.5)*(v*denv + dv*env);
}

// One work-item per basis function f of node n and neighbor element b (the CUDA kernel's warp per node keeps
// 16*NB accumulators in one warp, which does not fit a PVC thread in fp64).
template <int NB, class real> void k_Phi0(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<Staged<real>, 1> sm(sycl::range<1>(NB), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*NB, M.T), dims(NB)), KERNEL_SG(SGW<real>) {
            const Thread t{it};
            Staged<real>* st = sm.template get_multi_ptr<sycl::access::decorated::no>().get();
            const int n = t.bx();
            const int b = t.by();
            const int f = t.tx();
            const int T = M.T;
            const int a = G.elem[n];
            real acc[16] = {};
            const real* table = M.basis + (long)(a*T + b)*M.K*4*NB;
            const int e1 = G.seg[n*(T + 1) + b + 1];
            for (int e0 = G.seg[n*(T + 1) + b]; e0 < e1; e0 += NB) {
                t.sync();
                if (e0 + f < e1) stage(M, G, e0 + f, st[f]);
                t.sync();
                for (int e = e0; e < sycl::min(e0 + NB, e1); ++e) {
                    const Staged<real>& qe = st[e - e0];
                    const real beta = cubic(table + (long)qe.knot*4*NB, NB, f, qe.t);
                    // compile-time indices: IGC left this loop rolled, with acc in private memory
                    static_for<16>([&](auto lm) { acc[lm] += beta*qe.S[lm]; });
                }
            }
            static_for<16>([&](auto lm) { G.Phi0[row(lm, n, G.N)*M.K0 + b*NB + f] = real(sh_norm(lm))*acc[lm]; });
        });
    });
}

template <int NB, class real> void k_Phi0_reverse(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<Staged<real>, 1> sm(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*2*NB, M.T), dims(2*NB)), KERNEL_SG(SGW<real>) {
            const Thread t{it};
            Staged<real>* st = sm.template get_multi_ptr<sycl::access::decorated::no>().get();
            const int n = t.bx();
            const int g = t.tx()/NB;
            const int f = t.tx()%NB;
            const int T = M.T;
            const int a = G.elem[n];
            for (int b = t.by(); b < T; b += t.gy()) {
                const int e1 = G.seg[n*(T + 1) + b + 1];
                real adj[16], Tg[NUM_T];
                for (int lm = 0; lm < 16; ++lm)
                    adj[lm] = G.Phi0_adj[row(lm, n, G.N)*M.K0 + b*NB + f]*real(sh_norm(lm));
                build_T(adj, Tg);
                const real* table = M.basis + (long)(a*T + b)*M.K*4*NB;
                for (int e0 = G.seg[n*(T + 1) + b]; e0 < e1; e0 += 32) {
                    t.sync();
                    if (t.tx() < 32 and e0 + t.tx() < e1) stage(M, G, e0 + t.tx(), st[t.tx()]);
                    t.sync();
                    for (int e = e0 + g; e < sycl::min(e0 + 32, e1); e += 2) {
                        const Staged<real>& qe = st[e - e0];
                        real beta, beta_d, S[16];
                        for (int lm = 0; lm < 16; ++lm) S[lm] = qe.S[lm];
                        cubic(table + (long)qe.knot*4*NB, NB, f, qe.t, beta, beta_d);
                        beta_d *= M.inv_h;
                        real vl[4] = {};
                        for (int lm = 0; lm < 16; ++lm) vl[lm_l(lm)] += adj[lm]*S[lm];
                        real Gd[3] = {};
                        for (int d = 0; d < 3; ++d) {
                            real s = 0;
                            for (int l = 1; l <= 3; ++l)
                                for (int mp = (l-1)*(l-1); mp < l*l; ++mp) s += Tg[t_index(l, d, mp)]*S[mp];
                            Gd[d] = beta*qe.rinv*s;
                        }
                        const real dr = beta_d*(vl[0] + vl[1] + vl[2] + vl[3])
                                        - beta*(vl[1] + 2*vl[2] + 3*vl[3])*qe.rinv;
                        real fv[3];
                        edge_force(dr, qe.y, qe.z, qe.x, Gd, fv);
                        warp_add3<SGW<real>>(t.sg(), G.edge_forces + 3*e, fv[0], fv[1], fv[2]);
                    }
                }
            }
        });
    });
}

template <class Sh, int LAYER> using MTables = std::conditional_t<LAYER == 0, Sh, Tables<0>>;

template <class Sh, int LAYER, class real> void k_M(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::CN), dims(Sh::CN)), KERNEL_SG(SGW<real>) {
        using T = MTables<Sh, LAYER>;
        constexpr int C = Sh::CN;
        constexpr int NO = LAYER == 0 ? Sh::num_LM : 1;
        const int n = int(it.get_group(2));
        const int k = int(it.get_local_id(2));
        const real inv = Sh::gated ? real(1) : 1/(LAYER == 0 ? G.s0 : G.s1)[n];
        const real* x = Sh::gated ? (LAYER == 0 ? G.X0 : G.X1) : (LAYER == 0 ? G.A0 : G.A1);
        const real* tree = LAYER == 0 ? M.M0_weights : M.M1_weights;
        real v[T::num_values], w[T::num_m];
        real out[NO] = {};
        for (int lm = 0; lm < num_lm; ++lm) v[lm] = x[row(lm, n, G.N)*C + k]*inv;
        for (int e = 0; e < T::num_m; ++e) w[e] = tree[((long)G.elem[n]*T::num_m + e)*C + k];
        tree_eval<T>(w, v, out);
        if constexpr (LAYER == 1) G.M1[(long)n*C + k] = out[0];
        else
            for (int LM = 0; LM < NO; ++LM) G.M0[row(LM, n, G.N)*C + k] = out[LM];
    });
}

template <class Sh, int LAYER, class real>
void k_M_reverse(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<real, 1> part(sycl::range<1>(Sh::CN/SGW<real>), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::CN), dims(Sh::CN)), KERNEL_SG(SGW<real>) {
            using T = MTables<Sh, LAYER>;
            constexpr int C = Sh::CN;
            constexpr int NO = LAYER == 0 ? Sh::num_LM : 1;
            const Thread t{it};
            const int n = t.bx();
            const int k = t.tx();
            const real inv = Sh::gated ? real(1) : 1/(LAYER == 0 ? G.s0 : G.s1)[n];
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
                for (int lm = 0; lm < num_lm; ++lm) G.X_adj[row(lm, n, G.N)*C + k] = adjx[lm];
            } else {
                real* A_adj = LAYER == 0 ? G.A0_adj : G.A1_adj;
                real dot = 0;
                for (int lm = 0; lm < num_lm; ++lm) {
                    A_adj[row(lm, n, G.N)*C + k] = adjx[lm]*inv;
                    dot += v[lm]*adjx[lm];
                }
                dot = warp_sum(t.sg(), dot);
                if (t.lane() == 0) part[t.warp()] = dot;
                t.sync();
                if (k == 0) {
                    real d = 0;
                    for (int qq = 0; qq < C/SGW<real>; ++qq) d += part[qq];
                    (LAYER == 0 ? G.dot0 : G.dot1)[n] = d*inv;
                }
            }
        });
    });
}

template <class Sh, bool H, class real>
inline void radials(const R1Block<Sh, H, real>& b, real x, real* R) {
    static_for<Sh::num_paths>([&](auto p) { R[p] = b.value(p, x); });
}
template <class Sh, bool H, class real>
inline void radials(const R1Block<Sh, H, real>& b, real x, real* R, real* Rd) {
    static_for<Sh::num_paths>([&](auto p) { b.value_deriv(p, x, R[p], Rd[p]); });
}
template <class Sh, int PG, class real> inline real radial_sum(const real* Rd, const real* hs) {
    real s = 0;
    static_for<Sh::num_paths>([&](auto p) { s = s + (group<Sh>(p) == PG ? Rd[p]*hs[p] : real(0)); });
    return s;
}

// edges staged per chunk in the forward kernels: all C in fp32, 64 in fp64 (192 bytes of local memory per edge:
// smaller chunks let more work-groups share an Xe-core)
template <class real> constexpr int fwd_chunk(int C) { return sizeof(real) == 8 ? std::min(C, 64) : C; }
template <class Sh, bool H, int PG, class real>
inline void phi1(const Thread& t, const Model<real>& M, const Graph<real>& G, Staged<real>* st) {
    constexpr int C = Sh::C;
    constexpr int FCH = fwd_chunk<real>(C);
    constexpr auto lg = lme_groups<Sh>();
    const int n = t.bx();
    const int k = t.tx();
    const int first = G.start[n];
    const int len = G.start[n+1] - first;
    const int end = first + len*(t.bz() + 1)/t.gz();
    real acc[Sh::num_lme] = {};
    for (int e0 = first + len*t.bz()/t.gz(); e0 < end; e0 += FCH) {
        t.sync();
        if (k < FCH and e0 + k < end) stage(M, G, e0 + k, st[k]);
        t.sync();
        for (int e = e0; e < sycl::min(e0 + FCH, end); ++e) {
            const Staged<real>& qe = st[e - e0];
            real h[Sh::num_LM], R[Sh::num_paths], S[16];
            for (int lm = 0; lm < 16; ++lm) S[lm] = qe.S[lm];
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) h[lm2] = G.H1[row(lm2, qe.row, G.NR)*C + k];
            const R1Block<Sh, H, real> blk(M, qe.pair, qe.knot);
            radials(blk, qe.t, R);
            tp_forward<Sh, PG>(acc, R, S, h);
        }
    }
#pragma unroll
    for (int lme = 0; lme < Sh::num_lme; ++lme)
        if (lg[lme] == PG) {
            if (t.gz() == 1) G.Phi1[phi1_row<Sh>(lme, n, G.N)*C + k] = acc[lme];
            else atomic_add(G.Phi1 + phi1_row<Sh>(lme, n, G.N)*C + k, acc[lme]);
        }
}
template <class Sh, bool H, class real>
void k_Phi1(sycl::queue& q, const Model<real>& M, const Graph<real>& G, int split) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<Staged<real>, 1> sm(sycl::range<1>(fwd_chunk<real>(Sh::C)), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::C, groups<Sh>, split), dims(Sh::C)), KERNEL_SG(SGW<real>) {
            const Thread t{it};
            Staged<real>* st = sm.template get_multi_ptr<sycl::access::decorated::no>().get();
            static_for<groups<Sh>>([&](auto pg) { if (t.by() == pg) phi1<Sh, H, pg>(t, M, G, st); });
        });
    });
}

// k_Phi1_reverse: in fp64 the gradient factors of the solid harmonics are computed from S in registers, in fp32
// they are staged in local memory with the edge (StagedD), whichever is faster for the precision
template <class real> constexpr bool direct_grad = sizeof(real) == 8;
template <class real> using StagedRev = std::conditional_t<direct_grad<real>, Staged<real>, StagedD<real>>;
// edges staged per chunk in the reverse kernels: 64 (fp32: 240 bytes per edge with the gradient factors, fp64: 192;
// a chunk of 128 would let only half as many work-groups share an Xe-core)
template <class real> constexpr int phi1_chunk(int C) { return std::min(C, 64); }
// grad_staged without the staged factors: Gd[d] += c g[lm] S[mp] / r
template <class real> inline void grad_direct(const real* g, const real* S, real rinv, real* Gd) {
    real q[3] = {};
    static_for<num_grad_terms>([&](auto i) {
        constexpr GradTerm t = grad_terms[i];
        q[t.d] += real(t.c)*g[t.lm]*S[t.mp];
    });
    for (int d = 0; d < 3; ++d) Gd[d] += q[d]*rinv;
}
template <class Sh, bool H, int PG, class real>
inline void phi1_reverse(const Thread& t, const Model<real>& M, const Graph<real>& G, StagedRev<real>* st) {
    constexpr int C = Sh::C;
    constexpr int CH = phi1_chunk<real>(C);
    const int n = t.bx();
    const int k = t.tx();
    const int first = G.start[n];
    const int len = G.start[n+1] - first;
    const int end = first + len*(t.bz() + 1)/t.gz();
    constexpr auto lg = lme_groups<Sh>();
    real adj[Sh::num_lme];
#pragma unroll
    for (int lme = 0; lme < Sh::num_lme; ++lme)
        adj[lme] = lg[lme] == PG ? G.dPhi1[phi1_row<Sh>(lme, n, G.N)*C + k] : real(0);
    for (int e0 = first + len*t.bz()/t.gz(); e0 < end; e0 += CH) {
        t.sync();
        if (k < CH and e0 + k < end) {
            stage(M, G, e0 + k, static_cast<Staged<real>&>(st[k]));
            if constexpr (not direct_grad<real>) stage_grad(st[k]);
        }
        t.sync();
        for (int e = e0; e < sycl::min(e0 + CH, end); ++e) {
            const StagedRev<real>& qe = st[e - e0];
            real h[Sh::num_LM], R[Sh::num_paths], Rd[Sh::num_paths], S[16];
            real g_S[16] = {};
            real g_h[Sh::num_LM] = {};
            real hs[Sh::num_paths] = {};
            real Gd[3] = {};
            S[0] = 1;
            for (int lm = 1; lm < 16; ++lm) S[lm] = qe.S[lm];
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) h[lm2] = G.H1[row(lm2, qe.row, G.NR)*C + k];
            const R1Block<Sh, H, real> blk(M, qe.pair, qe.knot);
            radials(blk, qe.t, R, Rd);
            tp_reverse<Sh, PG>(adj, R, S, h, g_S, g_h, hs);
            const real a = radial_sum<Sh, PG>(Rd, hs)*M.inv_h;
            if constexpr (direct_grad<real>) grad_direct(g_S, S, qe.rinv, Gd);
            else grad_staged(g_S, qe.D, Gd);
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) atomic_add(G.H1_adj + row(lm2, qe.row, G.NR)*C + k, g_h[lm2]);
            warp_force<SGW<real>>(t.sg(), G.edge_forces + 3*e, a, Gd, qe.y, qe.z, qe.x);
        }
    }
}
template <class Sh, bool H, class real>
void k_Phi1_reverse(sycl::queue& q, const Model<real>& M, const Graph<real>& G, int split) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<StagedRev<real>, 1> sm(sycl::range<1>(phi1_chunk<real>(Sh::C)), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::C, groups<Sh>, split), dims(Sh::C)), KERNEL_SG(SGW<real>) {
            const Thread t{it};
            StagedRev<real>* st = sm.template get_multi_ptr<sycl::access::decorated::no>().get();
            static_for<groups<Sh>>([&](auto pg) { if (t.by() == pg) phi1_reverse<Sh, H, pg>(t, M, G, st); });
        });
    });
}

// ----- pair-symmetric second-layer reverse (SYMMETRIX_SYCL_PAIR_SYM) -----
// Each unordered pair once, by the node that owns the edge (sym_owner): one spline evaluation serves both directions
// (valid for unordered-pair radial functions only), and the gradients of the reversed edge come from the parity of the
// solid harmonics. Edges to ghost rows (exchange mode: no partner on this rank) run one-sided.
inline bool sym_owner(int i, int j, float x, float y, float z) {
    if (i != j) return (i < j) != (((i ^ j) & 1) != 0);
    return x > 0 or (x == 0 and (y > 0 or (y == 0 and z > 0)));
}
template <class Sh, bool H, int PG, class real>
inline void phi1_reverse_sym(const Thread& t, const Model<real>& M, const Graph<real>& G, StagedRev<real>* st) {
    constexpr int C = Sh::C;
    constexpr int CH = phi1_chunk<real>(C);
    const int n = t.bx();
    const int k = t.tx();
    const int first = G.start[n];
    const int len = G.start[n+1] - first;
    const int end = first + len*(t.bz() + 1)/t.gz();
    real adj[Sh::num_lme];
    static_for<Sh::num_lme>([&](auto lme) {
        adj[lme] = lme_groups<Sh>()[lme] == PG ? G.dPhi1[phi1_row<Sh>(lme, n, G.N)*C + k] : real(0);
    });
    real hi[Sh::num_LM];
    real g_hi[Sh::num_LM] = {};
    for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) hi[lm2] = G.H1[row(lm2, n, G.NR)*C + k];
    for (int e0 = first + len*t.bz()/t.gz(); e0 < end; e0 += CH) {
        t.sync();
        if (k < CH and e0 + k < end) {
            stage(M, G, e0 + k, static_cast<Staged<real>&>(st[k]));
            if constexpr (not direct_grad<real>) stage_grad(st[k]);
        }
        t.sync();
        for (int e = e0; e < sycl::min(e0 + CH, end); ++e) {
            const StagedRev<real>& qe = st[e - e0];
            const int j = qe.row;
            const bool gh = j >= G.N;
            if (not gh and not sym_owner(n, j, float(qe.x), float(qe.y), float(qe.z))) continue;
            real h[Sh::num_LM], R[Sh::num_paths], Rd[Sh::num_paths], S[16];
            real g_S[16] = {};
            real g_h[Sh::num_LM] = {};
            real hs[Sh::num_paths] = {};
            real Gd[3] = {};
            S[0] = 1;
            for (int lm = 1; lm < 16; ++lm) S[lm] = qe.S[lm];
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) h[lm2] = G.H1[row(lm2, j, G.NR)*C + k];
            const R1Block<Sh, H, real> blk(M, qe.pair, qe.knot);
            radials(blk, qe.t, R, Rd);
            tp_reverse<Sh, PG>(adj, R, S, h, g_S, g_h, hs);
            for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) atomic_add(G.H1_adj + row(lm2, j, G.NR)*C + k, g_h[lm2]);
            if (not gh) {  // the reversed edge j -> n: same radials, parity-flipped S, accumulated into g_S and hs
                real adjj[Sh::num_lme];
                static_for<Sh::num_lme>([&](auto lme) {
                    adjj[lme] = lme_groups<Sh>()[lme] == PG ? G.dPhi1[phi1_row<Sh>(lme, j, G.N)*C + k] : real(0);
                });
                tp_reverse<Sh, PG, real, true>(adjj, R, S, hi, g_S, g_hi, hs);
            }
            const real a = radial_sum<Sh, PG>(Rd, hs)*M.inv_h;
            if constexpr (direct_grad<real>) grad_direct(g_S, S, qe.rinv, Gd);
            else grad_staged(g_S, qe.D, Gd);
            warp_force<SGW<real>>(t.sg(), G.edge_forces + 3*e, a, Gd, qe.y, qe.z, qe.x);
        }
    }
    for (int lm2 = 0; lm2 < Sh::num_LM; ++lm2) atomic_add(G.H1_adj + row(lm2, n, G.NR)*C + k, g_hi[lm2]);
}
// register file of k_Phi1_reverse_sym: fp64 always in the large one (IGC's automatic choice is not stable across
// builds for this kernel, and the small one spills); fp32 automatic
template <class real> inline auto sym_rev_props() {
    if constexpr (sizeof(real) == 8) return syclex::properties{sycl::ext::intel::experimental::grf_size<256>};
    else return syclex::properties{};
}
template <class Sh, bool H, class real>
void k_Phi1_reverse_sym(sycl::queue& q, const Model<real>& M, const Graph<real>& G, int split) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<StagedRev<real>, 1> sm(sycl::range<1>(phi1_chunk<real>(Sh::C)), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::C, groups<Sh>, split), dims(Sh::C)), sym_rev_props<real>(),
                       KERNEL_SG(SGW<real>) {
            const Thread t{it};
            StagedRev<real>* st = sm.template get_multi_ptr<sycl::access::decorated::no>().get();
            static_for<groups<Sh>>([&](auto pg) { if (t.by() == pg) phi1_reverse_sym<Sh, H, pg>(t, M, G, st); });
        });
    });
}

template <class Sh, class real> void k_readouts(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    constexpr int C = Sh::CN;
    constexpr int NW = C/SGW<real>;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<real, 1> part(sycl::range<1>(17*NW), h);
        sycl::local_accessor<real, 1> gz(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*C), dims(C)), KERNEL_SG(SGW<real>) {
            const Thread t{it};
            const int n = t.bx();
            const int k = t.tx();
            const int ty = G.elem[n];
            const int lane = t.lane();
            const int warp = t.warp();
            const real h1 = G.M0[(long)n*C + k];
            const real m1 = G.M1[(long)n*C + k];
            real f1[16], f2[16];
            for (int hh = 0; hh < 16; ++hh) {
                f1[hh] = M.z_weights_for_M0[((long)ty*16 + hh)*C + k];
                f2[hh] = M.z_weights_for_M1[hh*C + k];
            }
            for (int hh = 0; hh <= 16; ++hh) {
                const real v = warp_sum(t.sg(), hh < 16 ? h1*f1[hh] + m1*f2[hh] : M.readout_1_weights_for_M0[k]*h1);
                if (lane == 0) part[hh*NW + warp] = v;
            }
            t.sync();
            if (k < 16) {
                real z = M.z_bias[ty*16 + k];
                for (int qq = 0; qq < NW; ++qq) z += part[k*NW + qq];
                const real s = sigmoid(z);
                part[k*NW] = M.readout_2_scale_factor*z*s;
                gz[k] = M.readout_2_weights_2[k]*M.readout_2_scale_factor*s*(1 + z*(1 - s));
            }
            t.sync();
            if (k == 0) {
                double E = M.atomic_energies[ty];
                for (int qq = 0; qq < NW; ++qq) E += part[16*NW + qq];
                for (int hh = 0; hh < 16; ++hh) E += (double)M.readout_2_weights_2[hh]*part[hh*NW];
                G.node_energies[n] += E;
            }
            real aH = M.readout_1_weights_for_M0[k];
            real aM = 0;
            for (int hh = 0; hh < 16; ++hh) {
                aH += gz[hh]*f1[hh];
                aM += gz[hh]*f2[hh];
            }
            G.M0s_adj[(long)n*C + k] = aH;
            G.M1_adj[(long)n*C + k] = aM;
        });
    });
}

template <class Sh> inline long a_index(int r, int n, int N) {
    constexpr long C = Sh::CN;
    return r == 0 ? n*4*C : r < num_lm ? N*4*C + (row(r, n, N) - N)*C : n*4*C + (r - 15)*C;
}
template <class Sh, int LAYER, class real>
inline void gate_inputs(const Model<real>& M, const Graph<real>& G, int n, int k, real inv, real* a, real* b) {
    constexpr int C = Sh::CN;
    const real* A = LAYER ? G.A1 : G.A0;
    for (int r = 0; r < num_lm + 3; ++r) b[r] = a[r] = A[a_index<Sh>(r, n, G.N) + k]*inv;
    for (int qq = 0; qq < 4; ++qq)
        b[qq == 0 ? 0 : 15 + qq] += LAYER ? G.B1[((long)n*4 + qq)*C + k] : M.residual_0[((long)G.elem[n]*4 + qq)*C + k];
    if (LAYER) for (int lm = 1; lm < Sh::num_LM; ++lm) b[lm] += G.B1[(long)G.N*4*C + (row(lm, n, G.N) - G.N)*C + k];
}
template <class Sh, int LAYER, class real> void k_gate(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::CN), dims(Sh::CN)), KERNEL_SG(SGW<real>) {
        constexpr int C = Sh::CN;
        const int n = int(it.get_group(2));
        const int k = int(it.get_local_id(2));
        real a[num_lm + 3], b[num_lm + 3];
        gate_inputs<Sh, LAYER>(M, G, n, k, 1/(LAYER ? G.s1[n] : G.s0[n]), a, b);
        G.Tg[(long)n*C + k] = b[0]*sigmoid(b[0]);
        for (int l = 1; l < 4; ++l) {
            const real s = sigmoid(b[15 + l]);
            for (int lm = l*l; lm < (l + 1)*(l + 1); ++lm) G.Tg[row(lm, n, G.N)*C + k] = b[lm]*s;
        }
    });
}
template <class Sh, int LAYER, class real>
void k_gate_reverse(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<real, 1> part(sycl::range<1>(Sh::CN/SGW<real>), h);
        h.parallel_for(sycl::nd_range<3>(dims(G.N*Sh::CN), dims(Sh::CN)), KERNEL_SG(SGW<real>) {
            constexpr int C = Sh::CN;
            const Thread t{it};
            const int n = t.bx();
            const int k = t.tx();
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
            for (int r = 0; r < num_lm + 3; ++r) {
                A_adj[a_index<Sh>(r, n, G.N) + k] = ba[r]*inv;
                dot += a[r]*ba[r];
            }
            if (LAYER) {
                for (int qq = 0; qq < 4; ++qq) G.B1_adj[((long)n*4 + qq)*C + k] = ba[qq == 0 ? 0 : 15 + qq];
                for (int lm = 1; lm < Sh::num_LM; ++lm)
                    G.B1_adj[(long)G.N*4*C + (row(lm, n, G.N) - G.N)*C + k] = ba[lm];
            }
            dot = warp_sum(t.sg(), dot);
            if (t.lane() == 0) part[t.warp()] = dot;
            t.sync();
            if (k == 0) {
                real d = 0;
                for (int qq = 0; qq < C/SGW<real>; ++qq) d += part[qq];
                (LAYER ? G.dot1 : G.dot0)[n] = d*inv;
            }
        });
    });
}
template <class Sh, class real> void k_h1_bias(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    const long total = (long)G.N*Sh::C;
    q.parallel_for(sycl::nd_range<3>(dims(blocks(total, 256)*256), dims(256)), KERNEL16 {
        const long i = (long)it.get_group(2)*256 + it.get_local_id(2);
        if (i < total) G.H1[i] += M.H1_bias[(long)G.elem[i/Sh::C]*Sh::C + i%Sh::C];
    });
}

template <class real> void k_outputs(sycl::queue& q, const Graph<real>& G, double* eatom, double* out) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> part(sycl::range<1>(7*WARPS), h);
        h.parallel_for(warp_range(G.N), KERNEL {
            const Thread t{it};
            const int n = t.bx()*WARPS + t.warp();
            const int lane = t.lane();
            const int warp = t.warp();
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
                const double s = warp_sum(t.sg(), v[c]);
                if (lane == 0) part[c*WARPS + warp] = s;
            }
            t.sync();
            if (t.tx() < 7) {
                double s = 0;
                for (int qq = 0; qq < WARPS; ++qq) s += part[t.tx()*WARPS + qq];
                atomic_add(out + t.tx(), s);
            }
        });
    });
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

// ----- LAMMPS kernels (pair_symmetrix_mace_foundation_kokkos.cpp) -----

inline void k_elements(sycl::queue& q, int n, const int* type, const int* type_map, int* elem) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { elem[i] = type_map[type[i] - 1]; });
}

inline double ax(const LammpsAtoms& A, int i, int d) { return A.x[i*A.xs0 + d*A.xs1]; }
inline int aneigh(const LammpsAtoms& A, int i, int jj) { return A.neighbors[i*A.ns0 + jj*A.ns1] & A.neighmask; }

template <class real> void k_count(sycl::queue& q, const Model<real>& M, const Graph<real>& G, const LammpsAtoms& A) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sm(sycl::range<1>(WARPS*M.T), h);
        h.parallel_for(warp_range(G.N), KERNEL {
            const Thread t{it};
            const int n = t.bx()*WARPS + t.warp();
            const int lane = t.lane();
            const int T = M.T;
            int* cnt = sm.template get_multi_ptr<sycl::access::decorated::no>().get() + t.warp()*T;
            if (n >= G.N) return;
            for (int b = lane; b < T; b += 32) cnt[b] = 0;
            t.syncwarp();
            const int i = G.atom[n];
            const double xi = ax(A, i, 0);
            const double yi = ax(A, i, 1);
            const double zi = ax(A, i, 2);
            for (int jj = lane; jj < A.numneigh[i]; jj += 32) {
                const int j = aneigh(A, i, jj);
                const double dx = ax(A, j, 0) - xi;
                const double dy = ax(A, j, 1) - yi;
                const double dz = ax(A, j, 2) - zi;
                if (dx*dx + dy*dy + dz*dz < M.r_cut2) atomic_add_local(&cnt[A.type_map[A.type[j] - 1]], 1);
            }
            t.syncwarp();
            int total = 0;
            for (int b = lane; b < T; b += 32) {
                G.seg[n*(T + 1) + 1 + b] = cnt[b];
                total += cnt[b];
            }
            total = warp_sum(t.sg(), total);
            if (lane == 0) {
                G.count[n] = total;
                if (A.type_map[A.type[i] - 1] != G.elem[n]) atomic_or(G.err, ERR_TYPES);
            }
        });
    });
}

template <class real> void k_fill(sycl::queue& q, const Model<real>& M, const Graph<real>& G, const LammpsAtoms& A) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sm(sycl::range<1>(WARPS*M.T), h);
        h.parallel_for(warp_range(G.N), KERNEL {
            const Thread t{it};
            const SubGroup sg = t.sg();
            const int n = t.bx()*WARPS + t.warp();
            const int lane = t.lane();
            int* off = sm.template get_multi_ptr<sycl::access::decorated::no>().get() + t.warp()*M.T;
            if (n >= G.N) return;
            const int T = M.T;
            const int a = G.elem[n];
            const int i = G.atom[n];
            int* seg = G.seg + n*(T + 1);
            if (lane == 0) {
                int o = G.start[n];
                for (int b = 0; b < T; ++b) {
                    const int c = seg[b + 1];
                    seg[b] = off[b] = o;
                    o += c;
                }
                seg[T] = o;
            }
            t.syncwarp();
            const double xi = ax(A, i, 0);
            const double yi = ax(A, i, 1);
            const double zi = ax(A, i, 2);
            real s0 = 0;
            real s1 = 0;
            double ez = 0;
            for (int jj0 = 0; jj0 < A.numneigh[i]; jj0 += 32) {
                const int jj = jj0 + lane;
                int j = 0;
                int b = -1;
                double dx = 0;
                double dy = 0;
                double dz = 0;
                double r2 = 0;
                if (jj < A.numneigh[i]) {
                    j = aneigh(A, i, jj);
                    dx = ax(A, j, 0) - xi;
                    dy = ax(A, j, 1) - yi;
                    dz = ax(A, j, 2) - zi;
                    r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 < M.r_cut2) b = A.type_map[A.type[j] - 1];
                }
                const uint32_t peers = match_any(sg, b);
                if (b >= 0) {
                    const int e = off[b] + popc(peers & ((1u << lane) - 1));
                    const double r = sycl::sqrt(r2);
                    const double tt = (r - M.r_min)*M.inv_h;
                    const int knot = sycl::min(sycl::max((int)tt, 0), M.K - 1);
                    const int p = M.ordered ? a*T + b : pair_index(a, b, T);
                    const real x = real(tt - knot - 0.5);
                    *reinterpret_cast<Vec4<real>*>(G.geometry + 4*e) = {real(dz/r), real(dx/r), real(dy/r), real(r)};
                    G.neigh_indices[e] = j;
                    G.row_pair[e] = Int2{G.row_of_atom[j], p};
                    if (G.row_of_atom[j] < 0) atomic_or(G.err, ERR_ROWS);
                    if (r < M.r_min) atomic_or(G.err, ERR_SHORT);
                    const long blk = ((long)p*M.K + knot)*4;
                    if (M.A0_splines) s0 += cubic(M.A0_splines + blk, 1, 0, x);
                    if (M.A1_splines) s1 += cubic(M.A1_splines + blk, 1, 0, x);
                    // ZBL: node energy and the initial edge forces (k_zbl in CUDA)
                    const real rf = real(r);
                    const real* z = M.zbl ? M.zbl + 3*pair_index(a, b, T) : nullptr;
                    real fz = 0;
                    if (z and rf < z[1]) zbl_edge(M, z, rf, ez, fz);
                    G.edge_forces[3*e] = fz*real(dx/r);
                    G.edge_forces[3*e+1] = fz*real(dy/r);
                    G.edge_forces[3*e+2] = fz*real(dz/r);
                }
                t.syncwarp();
                if (b >= 0 and lane == int(sycl::ctz(peers))) off[b] += popc(peers);
                t.syncwarp();
            }
            s0 = warp_sum(sg, s0);
            s1 = warp_sum(sg, s1);
            ez = warp_sum(sg, ez);
            if (lane == 0) {
                G.s0[n] = 1 + s0;
                G.s1[n] = 1 + s1;
                G.node_energies[n] = ez;
            }
        });
    });
}

// Edges of each (node, neighbor type) segment in increasing distance, after k_fill: the second-layer kernels then walk
// the spline table (one block per pair and knot) in a similar order on all nodes, for some L1 reuse. Segments longer
// than 32*SORT_PER edges stay in neighbor-list order (correct either way).
constexpr int SORT_PER = 8;
template <class real> void k_sort_edges(sycl::queue& q, const Model<real>& M, const Graph<real>& G) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sm(sycl::range<1>(WARPS*32*SORT_PER), h);
        h.parallel_for(warp_range(G.N), KERNEL {
            const Thread t{it};
            const SubGroup sg = t.sg();
            const int n = t.bx()*WARPS + t.warp();
            const int lane = t.lane();
            if (n >= G.N) return;
            float* key = sm.template get_multi_ptr<sycl::access::decorated::no>().get() + t.warp()*32*SORT_PER;
            const int T = M.T;
            for (int b = 0; b < T; ++b) {
                const int s0 = G.seg[n*(T + 1) + b];
                const int cnt = G.seg[n*(T + 1) + b + 1] - s0;
                if (cnt < 2 or cnt > 32*SORT_PER) continue;
                t.syncwarp();
                for (int c = lane; c < cnt; c += 32) key[c] = float(G.geometry[4*(s0 + c) + 3]);
                t.syncwarp();
                Vec4<real> g[SORT_PER];
                int ni[SORT_PER], rank[SORT_PER];
                Int2 rp[SORT_PER];
                real f[SORT_PER][3];
#pragma unroll
                for (int k = 0; k < SORT_PER; ++k) {
                    const int c = lane + 32*k;
                    rank[k] = -1;
                    if (c < cnt) {
                        const int e = s0 + c;
                        const float kc = key[c];
                        int r = 0;
                        for (int o = 0; o < cnt; ++o) r += key[o] < kc or (key[o] == kc and o < c);
                        rank[k] = r;
                        g[k] = *reinterpret_cast<const Vec4<real>*>(G.geometry + 4*e);
                        ni[k] = G.neigh_indices[e];
                        rp[k] = G.row_pair[e];
                        f[k][0] = G.edge_forces[3*e];
                        f[k][1] = G.edge_forces[3*e+1];
                        f[k][2] = G.edge_forces[3*e+2];
                    }
                }
                sycl::group_barrier(sg, sycl::memory_scope::device);
#pragma unroll
                for (int k = 0; k < SORT_PER; ++k)
                    if (rank[k] >= 0) {
                        const int e = s0 + rank[k];
                        *reinterpret_cast<Vec4<real>*>(G.geometry + 4*e) = g[k];
                        G.neigh_indices[e] = ni[k];
                        G.row_pair[e] = rp[k];
                        G.edge_forces[3*e] = f[k][0];
                        G.edge_forces[3*e+1] = f[k][1];
                        G.edge_forces[3*e+2] = f[k][2];
                    }
            }
        });
    });
}

template <class Sh, class real>
void k_exchange(sycl::queue& q, const Graph<real>& G, int nlocal, int nall, double* buf, int direction) {
    const long total = (long)nall*Sh::num_LM*Sh::C;
    q.parallel_for(sycl::nd_range<3>(dims(blocks(total, 256)*256), dims(256)), KERNEL16 {
        constexpr int C = Sh::C;
        constexpr int num_LM = Sh::num_LM;
        const long i = (long)it.get_group(2)*256 + it.get_local_id(2);
        const int k = int(i%C);
        const int LM = int((i/C)%num_LM);
        const int a = int(i/C/num_LM);
        if (a >= nall) return;
        const long ext = ((long)a*num_LM + LM)*C + k;
        if (a < nlocal) {
            const int r = G.row_of_atom[a];
            if (direction == 0) buf[ext] = G.H1[row(LM, r, G.NR)*C + k];
            if (direction == 2) buf[ext] = 0;
            if (direction == 3) G.H1_adj[row(LM, r, G.NR)*C + k] += real(buf[ext]);
        } else {
            const long internal = row(LM, G.N + a - nlocal, G.NR)*C + k;
            if (direction == 1) G.H1[internal] = real(buf[ext]);
            if (direction == 2) buf[ext] = G.H1_adj[internal];
        }
    });
}

template <class real> void k_forces(sycl::queue& q, const Model<real>& M, const Graph<real>& G, const LammpsAtoms& A) {
    q.parallel_for(warp_range(G.N), KERNEL {
        const Thread t{it};
        const int n = t.bx()*WARPS + t.warp();
        const int lane = t.lane();
        if (n >= G.N) return;
        const real q0 = M.A0_splines ? G.dot0[n] : real(0);
        const real q1 = M.A1_splines ? G.dot1[n] : real(0);
        double fx = 0;
        double fy = 0;
        double fz = 0;
        for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
            real* fe = G.edge_forces + 3*e;
            if (q0 != 0 or q1 != 0) {
                const Edge<real> qe = edge(M, G, e);
                real v, d0 = 0, d1 = 0;
                const long b = ((long)qe.pair*M.K + qe.knot)*4;
                if (M.A0_splines) cubic(M.A0_splines + b, 1, 0, qe.t, v, d0);
                if (M.A1_splines) cubic(M.A1_splines + b, 1, 0, qe.t, v, d1);
                const real s = (q0*d0 + q1*d1)*M.inv_h;
                fe[0] += s*qe.y;
                fe[1] += s*qe.z;
                fe[2] += s*qe.x;
            }
            const int j = G.neigh_indices[e];
            atomic_add(A.f + j*A.fs0, (double)fe[0]);
            atomic_add(A.f + j*A.fs0 + A.fs1, (double)fe[1]);
            atomic_add(A.f + j*A.fs0 + 2*A.fs1, (double)fe[2]);
            fx += fe[0];
            fy += fe[1];
            fz += fe[2];
        }
        fx = warp_sum(t.sg(), fx);
        fy = warp_sum(t.sg(), fy);
        fz = warp_sum(t.sg(), fz);
        if (lane == 0) {
            const int i = G.atom[n];
            atomic_add(A.f + i*A.fs0, -fx);
            atomic_add(A.f + i*A.fs0 + A.fs1, -fy);
            atomic_add(A.f + i*A.fs0 + 2*A.fs1, -fz);
        }
    });
}

#undef KERNEL
#undef KERNEL16
#undef KERNEL_SG

// ----- host helpers -----

template <class Sh>
long a_off(int l, long r, long N) { return l == 0 ? r*Sh::width(0) : N*Sh::width(0) + (r - N)*Sh::CN; }
template <class Sh>
long q_off(int t, int l, long K0)
{
    return K0*((long)t*(Sh::width(0) + 3*Sh::CN) + (l == 0 ? 0 : Sh::width(0) + (l - 1)*Sh::CN));
}
template <class Sh> long r_off(int L) { return (long)Sh::C*(L == 0 ? 0 : Sh::width(0) + (L - 1)*Sh::CN); }

using Matrix = std::vector<double>;
using json = nlohmann::json;

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

// ----- engine (port of MACEFoundationCuda) -----

template <class Precision, bool mixed> class Engine {
public:
    struct Weight { Buffer<Precision> w; };

    Engine(const std::string& filename, const std::string& prefix, sycl::queue q) : q(q) { load(filename, prefix); }

    sycl::queue q;
    int num_elements, num_channels, num_channels_node;
    double r_cut;
    int L_max, num_LM, num_lme;
    bool gated, ordered;
    std::vector<int> atomic_numbers;
    Buffer<double> atomic_energies;
    int num_xe_cores;  // compute units/8 (PVC: 8 vector engines per Xe-core)
    // edge chunks per node in k_Phi1/k_Phi1_reverse: clamp(split_factor*num_xe_cores/(N*groups), 1, split_max), i.e.
    // a few work-groups per Xe-core in total (SYMMETRIX_SYCL_SPLIT fixes the count); small systems need the split
    // to fill the GPU, large ones get one work-group per node
    int split_factor = int(env_long("SYMMETRIX_SYCL_SPLIT_FACTOR", sizeof(Precision) == 8 ? 32 : 25, 1));
    int split_max = int(env_long("SYMMETRIX_SYCL_SPLIT_MAX", sizeof(Precision) == 8 ? 8 : 16, 1));
    int split_fixed = int(env_long("SYMMETRIX_SYCL_SPLIT", 0, 0));
    // pair-symmetric second-layer reverse (k_Phi1_reverse_sym; SYMMETRIX_SYCL_PAIR_SYM=0/1). Default for L_max 0
    // models, where it halves the per-edge kernel's spline loads; for L_max >= 1 the per-edge reverse is as fast or
    // faster (and the pair-symmetric one needs more registers than fp64 has). Never for ordered-pair models: their
    // radial functions depend on the order of the pair.
    long pair_sym_env = env_long("SYMMETRIX_SYCL_PAIR_SYM", -1);
    template <class Sh> bool pair_sym() const {
        if (ordered) return false;
        return pair_sym_env >= 0 ? pair_sym_env != 0 : Sh::L_max == 0;
    }
    Buffer<unsigned> err;

    int N = 0;
    int NR = 0;
    std::vector<std::array<int, 3>> segments;
    Buffer<int> atom_of_node, elem, elem_all, row_of_atom, count, start, seg, neigh_indices;
    Buffer<Int2> row_pair;
    Buffer<Precision> geometry, edge_forces;

    Buffer<double> node_energies, outputs;
    bool has_zbl;
    Buffer<Precision> zbl;
    Precision zbl_c[4] = {};
    int zbl_p = 0;
    int K;
    double r_min, h;
    int NB, K0;
    Buffer<Precision> basis;
    Buffer<Precision> Phi0, Phi0_adj, A0, A0_adj;
    Weight Q;
    Buffer<Precision> A0_splines, s0, dot0;
    Buffer<Precision> M0_weights, M0, M0_adj;
    Weight H1_weights;
    Buffer<Precision> H1_bias, H1, H1_adj;
    Buffer<Precision> R1;
    Buffer<Half2> R1h;
    Buffer<Precision> Phi1, dPhi1;
    Weight A1_weights;
    std::vector<long> A1_weights_offset;
    Buffer<Precision> A1, A1_adj;
    Buffer<Precision> A1_splines, s1, dot1;
    Buffer<Precision> M1_weights, M1, M1_adj;
    Weight linear_2_0, linear_2_1, residual_1_weights;
    Buffer<Precision> residual_0, Tg, T_adj, X0, X1, X_adj, B1, B1_adj;
    Buffer<Precision> z_weights_for_M0, z_weights_for_M1, z_bias;
    Buffer<Precision> readout_1_weights_for_M0, readout_2_weights_2;
    Precision readout_2_scale_factor;
    Buffer<Precision> M0s_adj;
    std::vector<GemmDesc<Precision>> queue;

    Model<Precision> model() const;
    Graph<Precision> graph() const;
    void reserve(long);
    template <class Sh> void compute_first_layer(const Model<Precision>&, const Graph<Precision>&);
    template <class Sh> void compute_second_layer(const Model<Precision>&, const Graph<Precision>&);
    template <class Sh> void compute_readouts(const Model<Precision>&, const Graph<Precision>&);
    template <class Sh> void reverse_second_layer(const Model<Precision>&, const Graph<Precision>&);
    template <class Sh> void reverse_first_layer(const Model<Precision>&, const Graph<Precision>&);
    void gemm(bool, int, int, int, const Precision*, int, const Weight&, long, Precision*, int);
    void products();
    void load(const std::string&, const std::string&);
};

template <class Precision, bool mixed> Model<Precision> Engine<Precision, mixed>::model() const
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

template <class Precision, bool mixed> Graph<Precision> Engine<Precision, mixed>::graph() const
{
    return {N, NR, atom_of_node, elem, row_of_atom, elem_all, count, start, seg, geometry, neigh_indices,
            row_pair, edge_forces,
            Phi0, A0, M0, H1, Phi1, A1, H1_adj, A1_adj, dPhi1, M0_adj, A0_adj, Phi0_adj,
            s0, s1, dot0, dot1, node_energies, err, X0, X1, Tg, T_adj, X_adj, B1, B1_adj, M0s_adj, M1, M1_adj};
}

template <class Precision, bool mixed> void Engine<Precision, mixed>::reserve(long E)
{
    const int T = num_elements;
    const long nn = N;
    const long nr = NR;
    const long C = num_channels;
    const long CN = num_channels_node;
    const long ar = gated ? num_lm + 3 : num_lm;
    count.reserve(nn + 1);
    q.memset(count, 0, (nn + 1)*sizeof(int)).wait();
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
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::compute_first_layer(const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;

    // A0
    if (NB == 64) k_Phi0<64>(q, M, G);
    else if (NB == 96) k_Phi0<96>(q, M, G);
    else k_Phi0<32>(q, M, G);
    for (int l = 0; l < 4; ++l)
        for (const auto& [b, e, t] : segments) {
            const long r0 = (long)N*l*l + (long)b*(2*l+1);
            gemm(false, (e - b)*(2*l+1), Sh::width(l), K0, G.Phi0 + r0*K0, K0, Q, q_off<Sh>(t, l, K0),
                 G.A0 + a_off<Sh>(l, r0, N), Sh::width(l));
        }
    products();

    // gate
    if constexpr (Sh::gated) {
        k_gate<Sh, 0>(q, M, G);
        for (int l = 0; l < 4; ++l)
            gemm(false, N*(2*l+1), CN, CN, G.Tg + (long)N*l*l*CN, CN, linear_2_0, (long)l*CN*CN,
                 G.X0 + (long)N*l*l*CN, CN);
        products();
    }

    // M0
    k_M<Sh, 0>(q, M, G);

    // H1
    for (int L = 0; L <= Sh::L_max; ++L)
        gemm(false, N*(2*L+1), C, CN, G.M0 + (long)N*L*L*CN, CN, H1_weights, (long)L*CN*C, G.H1 + (long)NR*L*L*C, C);
    products();
    if (M.H1_bias) k_h1_bias<Sh>(q, M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::compute_second_layer(const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;

    // A1 scaling: s1 is computed with the edges (k_fill)

    // Phi1
    const int split = split_fixed > 0 ? split_fixed
                                      : std::clamp(split_factor*num_xe_cores/std::max(N*groups<Sh>, 1), 1, split_max);
    if (split > 1) q.memset(G.Phi1, 0, (size_t)N*Sh::num_lme*C*sizeof(Precision));
    if (M.R1h) k_Phi1<Sh, true>(q, M, G, split);
    else k_Phi1<Sh, false>(q, M, G, split);

    // A1
    for (int l = 0; l < 4; ++l)
        gemm(false, N*(2*l+1), Sh::width(l), Sh::num_e(l)*C, G.Phi1 + (long)N*Sh::lme_offset(l)*C, Sh::num_e(l)*C,
             A1_weights, A1_weights_offset[l], G.A1 + a_off<Sh>(l, (long)N*l*l, N), Sh::width(l));
    if constexpr (not Sh::gated) products();

    // gate
    if constexpr (Sh::gated) {
        for (int L = 0; L <= Sh::L_max; ++L)
            gemm(false, N*(2*L+1), Sh::width(L), C, G.H1 + (long)NR*L*L*C, C, residual_1_weights, r_off<Sh>(L),
                 G.B1 + a_off<Sh>(L, (long)N*L*L, N), Sh::width(L));
        products();
        k_gate<Sh, 1>(q, M, G);
        for (int l = 0; l < 4; ++l)
            gemm(false, N*(2*l+1), CN, CN, G.Tg + (long)N*l*l*CN, CN, linear_2_1, (long)l*CN*CN,
                 G.X1 + (long)N*l*l*CN, CN);
        products();
    }

    // M1
    k_M<Sh, 1>(q, M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::compute_readouts(const Model<Precision>& M, const Graph<Precision>& G)
{
    // ZBL: node energies and initial edge forces are set with the edges (k_fill)
    k_readouts<Sh>(q, M, G);
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::reverse_second_layer(const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    k_M_reverse<Sh, 1>(q, M, G);

    q.memset(G.H1_adj, 0, (size_t)NR*Sh::num_LM*C*sizeof(Precision));

    if constexpr (Sh::gated) {
        for (int l = 0; l < 4; ++l)
            gemm(true, N*(2*l+1), CN, CN, G.X_adj + (long)N*l*l*CN, CN, linear_2_1, (long)l*CN*CN,
                 G.T_adj + (long)N*l*l*CN, CN);
        products();
        k_gate_reverse<Sh, 1>(q, M, G);
        for (int L = 0; L <= Sh::L_max; ++L)
            gemm(true, N*(2*L+1), C, Sh::width(L), G.B1_adj + a_off<Sh>(L, (long)N*L*L, N), Sh::width(L),
                 residual_1_weights, r_off<Sh>(L), G.H1_adj + (long)NR*L*L*C, C);
    }

    for (int l = 0; l < 4; ++l)
        gemm(true, N*(2*l+1), Sh::num_e(l)*C, Sh::width(l), G.A1_adj + a_off<Sh>(l, (long)N*l*l, N), Sh::width(l),
             A1_weights, A1_weights_offset[l], G.dPhi1 + (long)N*Sh::lme_offset(l)*C, Sh::num_e(l)*C);
    products();

    const int split = split_fixed > 0 ? split_fixed
                                      : std::clamp(split_factor*num_xe_cores/std::max(N*groups<Sh>, 1), 1, split_max);
    if (pair_sym<Sh>()) {
        if (M.R1h) k_Phi1_reverse_sym<Sh, true>(q, M, G, split);
        else k_Phi1_reverse_sym<Sh, false>(q, M, G, split);
    } else if (M.R1h) k_Phi1_reverse<Sh, true>(q, M, G, split);
    else k_Phi1_reverse<Sh, false>(q, M, G, split);
}

template <class Precision, bool mixed>
template <class Sh>
void Engine<Precision, mixed>::reverse_first_layer(const Model<Precision>& M, const Graph<Precision>& G)
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;

    for (int L = 0; L <= Sh::L_max; ++L)
        gemm(true, N*(2*L+1), CN, C, G.H1_adj + (long)NR*L*L*C, C, H1_weights, (long)L*CN*C,
             G.M0_adj + (long)N*L*L*CN, CN);
    products();

    k_M_reverse<Sh, 0>(q, M, G);

    if constexpr (Sh::gated) {
        for (int l = 0; l < 4; ++l)
            gemm(true, N*(2*l+1), CN, CN, G.X_adj + (long)N*l*l*CN, CN, linear_2_0, (long)l*CN*CN,
                 G.T_adj + (long)N*l*l*CN, CN);
        products();
        k_gate_reverse<Sh, 0>(q, M, G);
    }

    for (int l = 0; l < 4; ++l)
        for (const auto& [b, e, t] : segments) {
            const long r0 = (long)N*l*l + (long)b*(2*l+1);
            gemm(true, (e - b)*(2*l+1), K0, Sh::width(l), G.A0_adj + a_off<Sh>(l, r0, N), Sh::width(l), Q,
                 q_off<Sh>(t, l, K0), G.Phi0_adj + r0*K0, K0);
        }
    products();
    if (NB == 64) k_Phi0_reverse<64>(q, M, G);
    else if (NB == 96) k_Phi0_reverse<96>(q, M, G);
    else k_Phi0_reverse<32>(q, M, G);
}

template <class Precision, bool mixed>
void Engine<Precision, mixed>::gemm(bool trans, int m, int n, int k, const Precision* a, int lda, const Weight& W,
                                    long offset, Precision* c, int ldc)
{
    if (m > 0) queue.push_back({a, W.w + offset, c, m, n, k, lda, ldc, trans});
}

template <class Precision, bool mixed> void Engine<Precision, mixed>::products()
{
    using oneapi::mkl::transpose;
    namespace blas = oneapi::mkl::blas::column_major;
    // float32-mixed: GEMMs with three bf16 terms on the XMX units (SYMMETRIX_SYCL_MIXED_GEMM=0: plain fp32 GEMMs)
    static const bool xmx = env_flag("SYMMETRIX_SYCL_MIXED_GEMM", true);
    const auto mode = mixed and xmx ? oneapi::mkl::blas::compute_mode::float_to_bf16x3
                                    : oneapi::mkl::blas::compute_mode::unset;
    for (const auto& g : queue)
        blas::gemm(q, g.trans ? transpose::trans : transpose::nontrans, transpose::nontrans, g.N, g.M, g.K,
                   Precision(1), g.W, g.trans ? g.K : g.N, g.A, g.lda, Precision(0), g.C, g.ldc, mode);
    queue.clear();
}

template <class Precision, bool mixed>
void Engine<Precision, mixed>::load(const std::string& filename, const std::string& prefix)
{
    const sycl::device dev = q.get_device();
    num_xe_cores = std::max(1, int(dev.get_info<sycl::info::device::max_compute_units>())/8);
    if (env_flag("SYMMETRIX_SYCL_VERBOSE", false))
        std::fprintf(stderr, "symmetrix SYCL: %s, %d compute units (num_xe_cores %d), sub-groups fp64 %d, fp32 %d\n",
                     dev.get_info<sycl::info::device::name>().c_str(),
                     int(dev.get_info<sycl::info::device::max_compute_units>()), num_xe_cores, SGW<double>, SGW<float>);
    std::ifstream file(filename);
    if (not file) throw std::runtime_error(prefix + "cannot open " + filename);
    const json js = json::parse(file);
    if (js.value("format_version", 0) != 3)
        throw std::runtime_error(prefix + filename + " was extracted by an incompatible symmetrix version; "
                                                     "extract it again");
    if (js.contains("model_type")) throw std::runtime_error(prefix + filename + " is not a MACE model");
    if (not js.contains("first_layer_weights"))
        throw std::runtime_error(prefix + "the model file has no first-layer basis; "
                                          "extract it again with this Symmetrix");
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
    if (js.at("l_max") != 3 or not with_sycl_preset(C, L_max, CN, [&](auto s) {
                using Sh = decltype(s);
                {
                    const auto l = js.at("Phi1_l").get<std::vector<int>>();
                    const auto spec = js.at("Phi1_spec").get<std::vector<std::array<int, 3>>>();
                    const auto cg_spec = js.at("Phi1_cg_spec").get<std::vector<std::array<int, 2>>>();
                    const auto cg = js.at("Phi1_clebsch_gordan").get<std::vector<double>>();
                    bool same = int(l.size()) == Sh::num_paths and int(cg.size()) == Sh::num_terms
                                and int(cg_spec.size()) == Sh::num_terms;
                    for (int qq = 0; same and qq < Sh::num_terms; ++qq) {
                        const auto [lme, le, lm2] = spec.at(cg_spec[qq][0]);
                        same = Sh::terms[qq].lme == lme and Sh::terms[qq].path == le
                               and Sh::terms[qq].lm1 == cg_spec[qq][1] and Sh::terms[qq].lm2 == lm2
                               and std::abs(Sh::terms[qq].cg - cg[qq]) < 1e-6;
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
                                   "is not of a shape compiled into this build: " + compiled_presets());
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
                       num_paths*C, f, table.data() + (long)p*K*4*num_paths*C);
        // layout: see R1Block
        const long F = num_paths*C;
        const long B = (long)P*K;
        constexpr long SG = SGW<Precision>;
        const long NS = C/SG;  // channel blocks per path
        const int NA = mixed ? 2 : 4;  // fp32/fp64 coefficients per function (mixed: c0, c1; c2, c3 as fp16 pairs)
        std::vector<Precision> blk((long)B*NA*F);
        std::vector<Half2> c23(mixed ? B*F : 0);
        for (long b = 0; b < B; ++b)
            for (long p = 0; p < num_paths; ++p)
                for (long s = 0; s < NS; ++s)
                    for (long l = 0; l < SG; ++l) {
                        const long f = p*C + s*SG + l;
                        const Precision* c = table.data() + b*4*F + f;
                        for (int a = 0; a < NA; ++a) blk[b*NA*F + ((p*NS + s)*NA + a)*SG + l] = c[a*F];
                        if (mixed) {
                            if (std::max(std::abs(c[2*F]), std::abs(c[3*F])) > 6e4f)
                                throw std::runtime_error(prefix + "second-layer spline coefficients exceed the fp16 "
                                                                  "range");
                            c23[b*F + f] = {sycl::half(float(c[2*F])), sycl::half(float(c[3*F]))};
                        }
                    }
        R1.upload(blk);
        if (mixed) R1h.upload(c23);

        std::vector<Precision> w1;
        A1_weights_offset.clear();
        for (int L = 0; L < 4; ++L) {
            A1_weights_offset.push_back(long(w1.size()));
            const auto w = js.at("A1_weights")[L].get<std::vector<double>>();
            if (long(w.size()) != long(num_e[L])*C*width[L]) throw std::runtime_error(prefix + "unexpected A1 weights");
            for (double v : w) w1.push_back(Precision(v));
        }
        A1_weights.w.upload(w1);
    }

    {
        const int S = K + 1;
        const int nb = js.at("first_layer_basis_size");
        const int widths = width[0] + width[1] + width[2] + width[3];
        const auto H0 = js.at("H0_weights").get<Matrix>();
        const auto U = js.at("first_layer_weights").get<Matrix>();
        const auto A0w = js.at("A0_weights").get<std::vector<std::vector<Matrix>>>();
        const int NU = int(U.size())/(4*C);
        auto table = [&](const char* key, int p) {
            Matrix m;
            for (const auto& f : js.at(key)[p]) for (const auto& v : f) m.push_back(v.get<double>());
            return m;
        };
        std::vector<Precision> host_basis((long)T*T*K*4*NB, Precision(0));
        std::vector<Precision> qw((long)T*widths*K0, Precision(0));
        for (int a = 0; a < T; ++a)
            for (int b = 0; b < T; ++b) {
                const int p = ordered ? a*T + b : std::min(a, b)*(2*T - std::min(a, b) - 1)/2 + std::max(a, b);
                Matrix V = table("first_layer_values", p);
                Matrix D = table("first_layer_derivs", p);
                Matrix map((long)widths*nb, 0.0);
                if (nb < NU) {
                    const Matrix B = js.at("first_layer_basis")[a*T + b].get<Matrix>();
                    const Matrix V0 = V;
                    const Matrix D0 = D;
                    map = js.at("first_layer_map")[a*T + b].get<Matrix>();
                    for (int r = 0; r < nb; ++r)
                        for (int c = 0; c < S; ++c) {
                            double v = 0;
                            double d = 0;
                            for (int i = 0; i < NU; ++i) {
                                v += B[(long)r*NU + i]*V0[(long)i*S + c];
                                d += B[(long)r*NU + i]*D0[(long)i*S + c];
                            }
                            V[(long)r*S + c] = v;
                            D[(long)r*S + c] = d;
                        }
                } else
                    for (int l = 0, o = 0; l < 4; o += width[l++])
                        for (int kp = 0; kp < C; ++kp) {
                            const double* u = &U[((long)l*C + kp)*NU];
                            for (int k = 0; k < width[l]; ++k) {
                                const double c = A0w[a][l][(long)kp*width[l] + k]*H0[b*C + kp];
                                for (int n = 0; n < NU; ++n) map[((long)o + k)*nb + n] += c*u[n];
                            }
                        }
                Precision* tab = host_basis.data() + (long)(a*T + b)*K*4*NB;
                for (int n = 0; n < nb; ++n)
                    spline(Matrix(V.begin() + n*S, V.begin() + (n + 1)*S),
                           Matrix(D.begin() + n*S, D.begin() + (n + 1)*S), h, NB, n, tab);
                for (int l = 0, o = 0; l < 4; o += width[l++])
                    for (int n = 0; n < nb; ++n)
                        for (int k = 0; k < width[l]; ++k)
                            qw[((long)a*widths + o)*K0 + ((long)b*NB + n)*width[l] + k] =
                                Precision(map[((long)o + k)*nb + n]);
            }
        basis.upload(host_basis);
        Q.w.upload(qw);
    }

    auto scale = [&](const std::string& key, Buffer<Precision>& buf) {
        if (not js.at(key + "_scaled").get<bool>()) return;
        const auto v = js.at(key + "_spline_values").get<std::vector<std::vector<double>>>();
        const auto d = js.at(key + "_spline_derivs").get<std::vector<std::vector<double>>>();
        if (js.at(key + "_spline_r_min").get<double>() != r_min or int(v[0].size()) != K + 1)
            throw std::runtime_error(prefix + "the scale-factor splines must use the radial grid");
        std::vector<Precision> table((long)P*K*4);
        for (int p = 0; p < P; ++p) spline(v[p], d[p], h, 1, 0, table.data() + (long)p*K*4);
        buf.upload(table);
    };
    scale("A0", A0_splines);
    scale("A1", A1_splines);

    auto contraction = [&](const std::string& l, Buffer<Precision>& buf) {
        std::vector<Precision> w;
        for (const auto& a : js.at("M" + l + "_weights"))
            for (const auto& e : a) for (const auto& x : e) w.push_back(x.get<Precision>());
        buf.upload(w);
    };
    contraction("0", M0_weights);
    contraction("1", M1_weights);

    {
        std::vector<Precision> wh1;
        for (double v : js.at("H1_weights").get<std::vector<double>>()) wh1.push_back(Precision(v));
        H1_weights.w.upload(wh1);
        const auto wh = js.at("H2_weights_for_M0").get<std::vector<std::vector<double>>>();
        const auto wm = js.at("H2_weights_for_M1").get<std::vector<double>>();
        const auto r1 = js.at("readout_2_weights_1").get<std::vector<double>>();
        if (r1.size() != 16*size_t(CN)) throw std::runtime_error(prefix + "readout 2 must have 16 hidden units");
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
        z_weights_for_M0.upload(fh);
        z_weights_for_M1.upload(fm);
        z_bias.upload(rb);
        std::vector<Precision> r1w, r2w;
        for (double v : js.at("readout_1_weights_for_M0").get<std::vector<double>>()) r1w.push_back(Precision(v));
        for (double v : js.at("readout_2_weights_2").get<std::vector<double>>()) r2w.push_back(Precision(v));
        readout_1_weights_for_M0.upload(r1w);
        readout_2_weights_2.upload(r2w);
        readout_2_scale_factor = js.at("readout_2_scale_factor").get<Precision>();
        atomic_energies.upload(js.at("atomic_energies").get<std::vector<double>>());
    }

    if (gated) {
        auto flat = [](const json& j) {
            std::vector<Precision> v;
            for (const auto& r : j) for (const auto& x : r) v.push_back(x.get<Precision>());
            return v;
        };
        residual_0.upload(flat(js.at("residual_0")));
        linear_2_0.w.upload(flat(js.at("linear_2_0")));
        linear_2_1.w.upload(flat(js.at("linear_2_1")));
        residual_1_weights.w.upload(js.at("residual_1_weights").get<std::vector<Precision>>());
    }

    if (js.contains("H1_bias")) {
        std::vector<Precision> b;
        for (const auto& r : js.at("H1_bias")) for (const auto& x : r) b.push_back(x.get<Precision>());
        H1_bias.upload(b);
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
        zbl.upload(z);
        for (int qq = 0; qq < 4; ++qq) zbl_c[qq] = Precision(c[qq]);
        zbl_p = int(js.at("zbl_p").get<double>());
    }

    err.reserve(1);
    q.memset(err, 0, sizeof(unsigned)).wait();
    outputs.reserve(7);
}


// ----- LAMMPS interface -----

template <class Precision, bool mixed> class Lammps final : public Foundation {
public:
    Lammps(const std::string& file, const std::string& prefix, sycl::queue queue) : q(queue), prefix(prefix) {
        if (not process_queue()) process_queue() = queue;
        mace = std::make_unique<Engine<Precision, mixed>>(file, prefix, queue);
    }
    double r_cut() const override { return mace->r_cut; }
    const std::vector<int>& atomic_numbers() const override { return mace->atomic_numbers; }
    int num_parts() const override { return 3; }
    int comm_kind(int part) const override { return part == 0 ? 1 : part == 1 ? 2 : 0; }
    int comm_width(int) const override { return mace->num_LM*mace->num_channels; }
    void setup(const LammpsAtoms& A, const int* owner, bool exchange) override;
    void run(int part, const LammpsAtoms& A, double* ext, double* ext_adj) override;
    void outputs(double* eatom, double* out) override;
    unsigned errors() override {
        unsigned e = 0;
        q.memcpy(&e, mace->err.p, sizeof(unsigned)).wait();
        return e;
    }

private:
    template <class Sh> void run_shape(int part, const LammpsAtoms& A, double* ext, double* ext_adj);
    void edges(const Model<Precision>& M, const Graph<Precision>& G, const LammpsAtoms& A) {
        k_count(q, M, G, A);
        exclusive_scan(q, G.count, G.start, mace->N + 1);
        k_fill(q, M, G, A);
        if (mace->N >= sort_edges_from) k_sort_edges(q, M, G);
    }
    void forces(const Model<Precision>& M, const Graph<Precision>& G, const LammpsAtoms& A) {
        k_forces(q, M, G, A);
    }
    sycl::queue q;
    std::string prefix;
    std::unique_ptr<Engine<Precision, mixed>> mace;
    bool exchange = false;
    int nlocal = 0;
    int nall = 0;
    // edges sorted by distance (k_sort_edges) from this many nodes on: in fp32 the sort pays off for large systems
    // (a few percent), in fp64 it does not; SYMMETRIX_SYCL_SORT_EDGES_FROM overrides (both precisions)
    long sort_edges_from = env_long("SYMMETRIX_SYCL_SORT_EDGES_FROM",
                                    sizeof(Precision) == 4 ? 1000L : std::numeric_limits<long>::max());
};

template <class Precision, bool mixed>
void Lammps<Precision, mixed>::setup(const LammpsAtoms& A, const int* owner, bool exch)
{
    q.wait();
    exchange = exch;
    const int T = mace->num_elements;
    const int inum = A.inum;
    nlocal = A.nlocal;
    nall = A.nall;
    mace->elem_all.reserve(std::max(nall, 1));
    std::vector<int> el(nall), ilist(inum), owner_h(exchange ? 0 : nall), numneigh(nlocal);
    if (nall > 0) {
        k_elements(q, nall, A.type, A.type_map, mace->elem_all);
        q.memcpy(el.data(), mace->elem_all.p, nall*sizeof(int));
        if (not exchange) q.memcpy(owner_h.data(), owner, nall*sizeof(int));
    }
    if (inum > 0) q.memcpy(ilist.data(), A.ilist, inum*sizeof(int));
    if (nlocal > 0) q.memcpy(numneigh.data(), A.numneigh, nlocal*sizeof(int));
    q.wait();
    if (inum != nlocal) throw std::runtime_error(prefix + "needs every local atom in the neighbor list");

    std::vector<int> nodes;
    mace->segments.clear();
    for (int t = 0; t < T; ++t) {
        const int first = int(nodes.size());
        for (int i : ilist) if (el[i] == t) nodes.push_back(i);
        if (int(nodes.size()) > first) mace->segments.push_back({first, int(nodes.size()), t});
    }
    mace->N = int(nodes.size());
    mace->NR = exchange ? mace->N + nall - nlocal : mace->N;
    std::vector<int> node_elem(mace->N), rows(std::max(nall, 1), -1);
    long edges = 0;
    for (int n = 0; n < mace->N; ++n) {
        node_elem[n] = el[nodes[n]];
        rows[nodes[n]] = n;
        edges += numneigh[nodes[n]];
    }
    // ghosts without an owner keep row -1, which k_fill reports (ERR_ROWS) if a local atom has them as neighbors
    for (int i = nlocal; i < nall; ++i)
        rows[i] = exchange ? mace->N + i - nlocal : owner_h[i] >= 0 and owner_h[i] < nall ? rows[owner_h[i]] : -1;
    if (mace->N > 0) {
        mace->atom_of_node.upload(nodes);
        mace->elem.upload(node_elem);
    }
    mace->row_of_atom.upload(rows);
    mace->reserve(std::max<long>(edges, 1));
}

template <class Precision, bool mixed>
void Lammps<Precision, mixed>::run(int part, const LammpsAtoms& A, double* ext, double* ext_adj)
{
    with_sycl_preset(mace->num_channels, mace->L_max, mace->num_channels_node, [&](auto s) {
        this->template run_shape<decltype(s)>(part, A, ext, ext_adj);
    });
}

template <class Precision, bool mixed>
template <class Sh>
void Lammps<Precision, mixed>::run_shape(int part, const LammpsAtoms& A, double* ext, double* ext_adj)
{
    if (mace->N == 0) {
        if (part == 1) q.memset(ext_adj, 0, (size_t)nall*Sh::num_LM*Sh::C*sizeof(double));
        return;
    }
    const Model<Precision> M = mace->model();
    const Graph<Precision> G = mace->graph();
    if (part < 0 or part == 0) {
        edges(M, G, A);
        mace->template compute_first_layer<Sh>(M, G);
    }
    if (part == 0) k_exchange<Sh>(q, G, nlocal, nall, ext, 0);
    if (part == 1) k_exchange<Sh>(q, G, nlocal, nall, ext, 1);
    if (part < 0 or part == 1) {
        mace->template compute_second_layer<Sh>(M, G);
        mace->template compute_readouts<Sh>(M, G);
        mace->template reverse_second_layer<Sh>(M, G);
    }
    if (part == 1) k_exchange<Sh>(q, G, nlocal, nall, ext_adj, 2);
    if (part == 2) k_exchange<Sh>(q, G, nlocal, nall, ext_adj, 3);
    if (part < 0 or part == 2) {
        mace->template reverse_first_layer<Sh>(M, G);
        forces(M, G, A);
    }
}

template <class Precision, bool mixed> void Lammps<Precision, mixed>::outputs(double* eatom, double* out)
{
    std::fill(out, out + 7, 0.0);
    if (mace->N == 0) return;
    const Graph<Precision> G = mace->graph();
    q.memset(mace->outputs, 0, 7*sizeof(double));
    k_outputs(q, G, eatom, mace->outputs);
    q.memcpy(out, mace->outputs.p, 7*sizeof(double)).wait();
}

}  // namespace mace

std::unique_ptr<Foundation> make_mace_foundation(const std::string& file, const std::string& prefix,
                                                 const std::string& dtype, sycl::queue q)
{
#ifndef SYMMETRIX_SYCL_NO_MIXED
    if (dtype == "float32-mixed") return std::make_unique<mace::Lammps<float, true>>(file, prefix, q);
#endif
#ifndef SYMMETRIX_SYCL_NO_FLOAT
    if (dtype == "float32") return std::make_unique<mace::Lammps<float, false>>(file, prefix, q);
#endif
#ifndef SYMMETRIX_SYCL_NO_DOUBLE
    if (dtype == "float64") return std::make_unique<mace::Lammps<double, false>>(file, prefix, q);
#endif
    throw std::runtime_error(prefix + "precision " + dtype + " is not compiled into this SYCL build");
}

}  // namespace symmetrix_sycl
