#pragma once

#include "mace_foundation_tables.hpp"

#include "../external/json/single_include/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

template <class Precision>
class MACEFoundation {

public:

template <class T> struct Uninit : std::allocator<T> {
    template <class U> struct rebind { using other = Uninit<U>; };
    template <class U, class... A> void construct(U* p, A&&... a) { ::new ((void*)p) U(std::forward<A>(a)...); }
    template <class U> void construct(U* p) { ::new ((void*)p) U; }
};
using Floats = std::vector<Precision, Uninit<Precision>>;
struct Weight { std::vector<Precision> w, tw; };

MACEFoundation(const std::string& filename, const std::string& prefix);

// Basic model information
int num_elements;
int num_channels;
int num_channels_node;
double r_cut;
int L_max, num_LM;
int num_lme;
bool gated, ordered;
std::vector<int> atomic_numbers;
std::vector<double> atomic_energies;

// Graph
int N = 0;
int NR = 0;
std::vector<std::array<int, 3>> segments;
std::vector<int> atom_of_node, elem, elem_all, row_of_atom, start, seg, neigh_indices, row_pair;
Floats geometry, edge_forces;
std::unique_ptr<std::atomic_flag[]> locks;

// Node energies and forces
std::vector<double> node_energies;
template <class Sh> void compute_first_layer();
template <class Sh> void compute_second_layer();
template <class Sh> void compute_readouts();
template <class Sh> void reverse_second_layer();
template <class Sh> void reverse_first_layer();

// ZBL
bool has_zbl;
std::vector<Precision> zbl;
Precision zbl_c[4];
int zbl_p;

// Radial functions
int K;
double r_min, h;
int NB, K0;
std::vector<Precision> basis;

// A0
Floats Phi0, Phi0_adj, A0, A0_adj;
Weight Q;

// A0 rescaling
std::vector<Precision> A0_splines;
Floats s0, dot0;

// M0
std::vector<Precision> M0_weights;
Floats M0, M0_adj;

// H1
Weight H1_weights;
std::vector<Precision> H1_bias;
Floats H1, H1_adj;

// Phi1
Floats R1, Phi1, dPhi1;

// A1
Weight A1_weights;
std::vector<long> A1_weights_offset;
Floats A1, A1_adj;

// A1 rescaling
std::vector<Precision> A1_splines;
Floats s1, dot1;

// M1
std::vector<Precision> M1_weights;
Floats M1, M1_adj;

// Gates
Weight linear_2_0, linear_2_1, residual_1_weights;
std::vector<Precision> residual_0;
Floats Tg, T_adj, X0, X1, X_adj, B1, B1_adj;
template <class Sh, class Rows> void gate_inputs(int, int, int, Precision, Rows, Rows);
template <class Sh> void gate(int, const Weight&, Floats&);
template <class Sh> void gate_reverse(int, const Weight&);

// Readouts
std::vector<Precision> z_weights_for_M0, z_weights_for_M1, z_bias;
std::vector<Precision> readout_1_weights_for_M0, readout_2_weights_2;
Precision readout_2_scale_factor;
Floats M0s_adj;

// Matrix products
struct Product { int m, n, k; const Precision* a; int lda; const Precision* b; Precision* c; int ldc; };
std::vector<Product> queue;
void gemm(bool trans, int m, int n, int k, const Precision* a, int lda, const Weight& W, long offset,
          Precision* c, int ldc)
{
    if (m > 0) queue.push_back({m, n, k, a, lda, (trans ? W.tw : W.w).data() + offset, c, ldc});
}
void products();
void set_weight(Weight&, const std::vector<Precision>&, const std::vector<std::array<long, 3>>&);

// Initializer
void load(const std::string&, const std::string&);

};

namespace {

using json = nlohmann::json;
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



constexpr long row(int lm, int node, int N)
{
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
constexpr int pair_index(int a, int b, int T)
{
    return a <= b ? a*(2*T - a - 1)/2 + b : b*(2*T - b - 1)/2 + a;
}
template <class Sh> constexpr long a_index(int r, int n, int N) {
    constexpr long C = Sh::CN;
    return r == 0 ? n*4*C : r < num_lm ? N*4*C + (row(r, n, N) - N)*C : n*4*C + (r - 15)*C;
}
template <class Sh>
constexpr long a_off(int l, long r, int N) { return l == 0 ? r*Sh::width(0) : (long)N*Sh::width(0) + (r - N)*Sh::CN; }
template <class Sh> constexpr long q_off(int K0, int t, int l) {
    return (long)K0*((long)t*(Sh::width(0) + 3*Sh::CN) + (l == 0 ? 0 : Sh::width(0) + (l - 1)*Sh::CN));
}
template <class Sh> constexpr long r_off(int L) { return (long)Sh::C*(L == 0 ? 0 : Sh::width(0) + (L - 1)*Sh::CN); }
template <class real> inline real sigmoid(real x) { return 1/(1 + std::exp(-x)); }

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

constexpr int NUM_T = 27;
constexpr int t_index(int l, int d, int mp)
{
    return (l == 1 ? 0 : l == 2 ? 3 : 12) + d*(2*l - 1) + mp - (l-1)*(l-1);
}

template <class real> struct Edge {
    real x, y, z, r, rinv, t;
    int knot, row, pair;
    real S[16];
};
template <class real> inline void grad(const real* S, const real* gs, real* Gd) {
    static_for<num_grad_terms>([&] [[gnu::always_inline]] (auto g) {
        constexpr GradTerm t = grad_terms[g];
        Gd[t.d] += real(t.c)*S[t.mp]*gs[t.lm];
    });
}
template <class real>
inline Edge<real> edge(int K, real r_min, real inv_h, const real* geometry, const int* row_pair, int e) {
    Edge<real> q;
    q.x = geometry[4*e];
    q.y = geometry[4*e+1];
    q.z = geometry[4*e+2];
    q.r = geometry[4*e+3];
    q.rinv = 1/q.r;
    const real t = (q.r - r_min)*inv_h;
    q.knot = std::min(std::max(int(t), 0), K - 1);
    q.t = t - q.knot - 0.5f;
    q.row = row_pair[2*e];
    q.pair = row_pair[2*e+1];
    solid_harmonics(q.x, q.y, q.z, q.S);
    return q;
}

constexpr int V = 8;
constexpr int TW = 32;
using T1 = Tables<0>;
template <class Sh> constexpr int path_l1(int p) {
    for (const Term& t : Sh::terms) if (t.path == p) return lm_l(t.lm1);
    return 0;
}
template <class Sh> constexpr int path_l2(int p) {
    for (const Term& t : Sh::terms) if (t.path == p) return lm_l(t.lm2);
    return 0;
}
template <class Sh> constexpr int path_e(int p) {
    for (const Term& t : Sh::terms)
        if (t.path == p) return (t.lme - Sh::lme_offset(Sh::Phi1_l[p])) % Sh::num_e(Sh::Phi1_l[p]);
    return 0;
}
template <class Sh> constexpr int fwd_chunks(int p) {
    const int n = 2*Sh::Phi1_l[p] + 2*path_l2<Sh>(p) + 2;
    return 4*n <= 14 ? 4 : 2*n <= 14 ? 2 : 1;
}

template <class Sh> constexpr bool same_out(int a, int b) {
    return Sh::terms[a].path == Sh::terms[b].path and Sh::terms[a].lme == Sh::terms[b].lme
           and Sh::terms[a].lm2 == Sh::terms[b].lm2;
}
template <class Sh> constexpr int out_term(int q) {
    for (int r = 0; r < q; ++r) if (same_out<Sh>(r, q)) return r;
    return q;
}
template <class Sh> constexpr int out_rank(int q) {
    int n = 0;
    for (int r = 0; r < out_term<Sh>(q); ++r) n += Sh::terms[r].path == Sh::terms[q].path and out_term<Sh>(r) == r;
    return n;
}
template <class Sh> constexpr int path_outs(int p) {
    int n = 0;
    for (int q = 0; q < Sh::num_terms; ++q) n += Sh::terms[q].path == p and out_term<Sh>(q) == q;
    return n;
}
template <class Sh> constexpr int path_out(int p, int k) {
    for (int q = 0; q < Sh::num_terms; ++q)
        if (Sh::terms[q].path == p and out_term<Sh>(q) == q and k-- == 0) return q;
    return -1;
}
template <class Sh>
constexpr int path_first(int p) { return p == 0 ? 0 : path_first<Sh>(p - 1) + path_outs<Sh>(p - 1); }

template <class Sh> constexpr bool last_key(int q) {
    for (int r = q + 1; r < Sh::num_terms; ++r)
        if (Sh::terms[r].path == Sh::terms[q].path and Sh::terms[r].lm1 == Sh::terms[q].lm1 and first_term<Sh>(r))
            return false;
    return true;
}
template <class Sh> constexpr int path_keys(int p) {
    int n = 0;
    for (int q = 0; q < Sh::num_terms; ++q) n += Sh::terms[q].path == p and first_term<Sh>(q);
    return n;
}
template <class Sh> constexpr int path_key(int p, int k) {
    for (int q = 0; q < Sh::num_terms; ++q)
        if (Sh::terms[q].path == p and first_term<Sh>(q) and k-- == 0) return q;
    return -1;
}

template <class Sh, class real> [[gnu::always_inline]] inline void stage_cs(const real* S, int i, int ne, real* cs) {
    static_for<Sh::num_terms>([&] [[gnu::always_inline]] (auto q) {
        constexpr Term t = Sh::terms[q];
        real& c = cs[ne*path_first<Sh>(t.path) + i*path_outs<Sh>(t.path) + out_rank<Sh>(q)];
        c = (out_term<Sh>(q) == q ? 0 : c) + real(t.cg*sh_norm(t.lm1))*S[t.lm1];
    });
}
constexpr int EB = 8;
template <class Sh, int P, class real>
[[gnu::always_inline]] inline void fwd_path(int c, int i0, int i1, int ne, const real* cs, const real* Rb,
                                            const int* jr, const real* H1, int NR, real (*acc4)[4*V]) {
    constexpr int C = Sh::C;
    constexpr int l = Sh::Phi1_l[P];
    constexpr int l2 = path_l2<Sh>(P);
    constexpr int CH = fwd_chunks<Sh>(P);
    constexpr int o = Sh::lme_offset(l) + path_e<Sh>(P);
    for (int u0 = 0; u0 < 4; u0 += CH) {
        real acc[2*l + 1][CH*V];
        for (int m = 0; m < 2*l + 1; ++m) std::copy_n(acc4[o + m*Sh::num_e(l)] + u0*V, CH*V, acc[m]);
        for (int i = i0; i < i1; ++i) {
            real rh[2*l2 + 1][CH*V];
            for (int u = 0; u < CH; ++u) {
                const real* r = Rb + ((P*EB + i - i0)*4 + u0 + u)*V;
                const real* h = H1 + row(l2*l2, jr[i], NR)*C + (c + u0 + u)*V;
                for (int m2 = 0; m2 < 2*l2 + 1; ++m2)
#pragma omp simd
                    for (int w = 0; w < V; ++w) rh[m2][u*V + w] = r[w]*h[m2*C + w];
            }
            const real* cs_i = cs + ne*path_first<Sh>(P) + i*path_outs<Sh>(P);
            static_for<path_outs<Sh>(P)>([&] [[gnu::always_inline]] (auto k) {
                constexpr Term t = Sh::terms[path_out<Sh>(P, k)];
                constexpr int m = (t.lme - Sh::lme_offset(l))/Sh::num_e(l);
                constexpr int m2 = t.lm2 - l2*l2;
                const real c = cs_i[k];
#pragma omp simd
                for (int w = 0; w < CH*V; ++w) acc[m][w] += c*rh[m2][w];
            });
        }
        for (int m = 0; m < 2*l + 1; ++m) std::copy_n(acc[m], CH*V, acc4[o + m*Sh::num_e(l)] + u0*V);
    }
}
template <class Sh, class real>
inline void fwd_paths(int c, int i0, int i1, int ne, const real* cs, const real* Rb, const int* jr, const real* H1,
                      int NR, real (*acc4)[4*V]) {
    static_for<Sh::num_paths>([&] [[gnu::always_inline]] (auto p) {
        fwd_path<Sh, p>(c, i0, i1, ne, cs, Rb, jr, H1, NR, acc4);
    });
}

template <class Sh, int L, int Q, class real> inline void key_w(const real* adj, real* out) {
    constexpr Term t = Sh::terms[Q];
    constexpr real cg = real(t.cg*sh_norm(t.lm1));
    constexpr int n = next_term<Sh>(Q);
    const real* a = adj + t.lme*L;
    if constexpr (n < 0) {
        for (int w = 0; w < L; ++w) out[w] = cg*a[w];
    } else {
        key_w<Sh, L, n>(adj, out);
        for (int w = 0; w < L; ++w) out[w] = cg*a[w] + out[w];
    }
}
template <class Sh, int L, class real> inline void key_ws(const real* adj, real* kw) {
    static_for<Sh::num_terms>([&] [[gnu::always_inline]] (auto q) {
        if constexpr (first_term<Sh>(q)) key_w<Sh, L, q>(adj, kw + q*L);
    });
}
template <class Sh, int P, class real>
[[gnu::always_inline]] inline void rev_path(const real* b, real x, const real* S, const real* kw, const real* h, int C,
                                            real (*g_h)[V], real* gi) {
    constexpr int l1 = path_l1<Sh>(P);
    constexpr int l2 = path_l2<Sh>(P);
    constexpr int N2 = 2*l2 + 1;
    real R[V];
    real Rd[V];
#pragma omp simd
    for (int w = 0; w < V; ++w) cubic(b + P*4*V, V, w, x, R[w], Rd[w]);
    real rh[N2][V];
    for (int m2 = 0; m2 < N2; ++m2)
#pragma omp simd
        for (int w = 0; w < V; ++w) rh[m2][w] = R[w]*h[m2*C + w];
    real s[N2][V] = {};
    real gy[2*l1 + 1][V] = {};
    static_for<path_keys<Sh>(P)>([&] [[gnu::always_inline]] (auto k) {
        constexpr int q = path_key<Sh>(P, k);
        constexpr Term t = Sh::terms[q];
        constexpr int m1 = t.lm1 - l1*l1;
        constexpr int m2 = t.lm2 - l2*l2;
        const real* kq = kw + q*V;
        const real sv = S[t.lm1];
#pragma omp simd
        for (int w = 0; w < V; ++w) {
            s[m2][w] += kq[w]*sv;
            gy[m1][w] += kq[w]*rh[m2][w];
        }
        if constexpr (last_key<Sh>(q))
#pragma omp simd
            for (int w = 0; w < V; ++w) gi[t.lm1*V + w] += gy[m1][w];
    });
    real hs[V] = {};
    for (int m2 = 0; m2 < N2; ++m2)
#pragma omp simd
        for (int w = 0; w < V; ++w) {
            g_h[l2*l2 + m2][w] += R[w]*s[m2][w];
            hs[w] += h[m2*C + w]*s[m2][w];
        }
#pragma omp simd
    for (int w = 0; w < V; ++w) gi[16*V + w] += Rd[w]*hs[w];
}
template <class Sh, int P, class real>
[[gnu::noinline]] void rev_path_call(const real* b, real x, const real* S, const real* kw, const real* h, int C,
                                     real (*g_h)[V], real* gi) {
    rev_path<Sh, P>(b, x, S, kw, h, C, g_h, gi);
}
template <class Sh, class real>
[[gnu::always_inline]] inline void rev_paths(const real* b, real x, const real* S, const real* kw, const real* H1,
                                             int C, long r0, const long* r, real (*g_h)[V], real* gi) {
    static_for<Sh::num_paths>([&] [[gnu::always_inline]] (auto p) {
        if constexpr (Sh::L_max == 2) rev_path_call<Sh, p>(b, x, S, kw, H1 + r[path_l2<Sh>(p)] + r0, C, g_h, gi);
        else rev_path<Sh, p>(b, x, S, kw, H1 + r[path_l2<Sh>(p)] + r0, C, g_h, gi);
    });
}

template <class T, int L, class real> [[gnu::always_inline]] inline void tree_eval(const real* wt, real* v, real* out) {
    std::fill(v + num_lm*L, v + T::num_values*L, 0.f);
    static_for<T::num_b>([&] [[gnu::always_inline]] (auto e) {
        constexpr BTerm t = T::b[e];
        real* q = v + t.q*L;
        const real* a = v + t.a*L;
        const real* b = v + t.b*L;
#pragma omp simd
        for (int w = 0; w < L; ++w) q[w] += real(t.c)*a[w]*b[w];
    });
    static_for<T::num_m>([&] [[gnu::always_inline]] (auto e) {
        constexpr MTerm t = T::m[e];
        for (int j = 0; j < t.n; ++j)
#pragma omp simd
            for (int w = 0; w < L; ++w) out[(t.o + j)*L + w] += wt[e*L + w]*v[(t.q + j)*L + w];
    });
}
template <class T, int L, class real>
[[gnu::always_inline]] inline void tree_adjoint(const real* wt, const real* v, const real* g, real* adjx) {
    real gv[T::num_values*L] = {};
    static_for<T::num_m>([&] [[gnu::always_inline]] (auto e) {
        constexpr MTerm t = T::m[e];
        for (int j = 0; j < t.n; ++j)
#pragma omp simd
            for (int w = 0; w < L; ++w) gv[(t.q + j)*L + w] += wt[e*L + w]*g[(t.o + j)*L + w];
    });
    static_for<T::num_b>([&] [[gnu::always_inline]] (auto e) {
        constexpr BTerm t = T::b[T::num_b - 1 - e];
        const real* q = gv + t.q*L;
        real* ga = gv + t.a*L;
        real* gb = gv + t.b*L;
        const real* va = v + t.a*L;
        const real* vb = v + t.b*L;
        for (int w = 0; w < L; ++w) {
            const real s = real(t.c)*q[w];
            ga[w] += s*vb[w];
            gb[w] += s*va[w];
        }
    });
    std::copy(gv, gv + num_lm*L, adjx);
}

template <int L, class real> inline void build_T(const real* a, real* T) {
    std::fill(T, T + NUM_T*L, 0.f);
    static_for<num_grad_terms>([&] [[gnu::always_inline]] (auto g) {
        constexpr GradTerm gt = grad_terms[g];
        constexpr int t = t_index(lm_l(gt.lm), gt.d, gt.mp);
        for (int w = 0; w < L; ++w) T[t*L + w] += real(gt.c)*a[gt.lm*L + w];
    });
}

}  // namespace

template <class Precision>
MACEFoundation<Precision>::MACEFoundation(const std::string& filename, const std::string& prefix)
{
    load(filename, prefix);
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::compute_first_layer()
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    const int T = num_elements;
    const Precision rmin = Precision(r_min);
    const Precision inv_h = Precision(1/h);
    auto edge_of = [&](int e) { return edge(K, rmin, inv_h, geometry.data(), row_pair.data(), e); };

    // A0
    // \Phi^{(0)}_{i,lm,bn} = \sum_{j:a_j=b} \beta_{a_ib,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})
    // A^{(0)}_{i,klm} = \sum_{bn} Q_{a_il,k,bn} \Phi^{(0)}_{i,lm,bn},  \sum_n Q_{al,k,bn} \beta_{ab,n}(r) = \sum_{k'} w_{akl,k'} R^{(0)}_{ab,k'l}(r) H^{(0)}_{b,k'}
    // A0 scaling
    // s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij}),  \tilde{A}^{(0)}_{i,klm} = A^{(0)}_{i,klm} / s^{(0)}_i
    const long ar = gated ? num_lm + 3 : num_lm;
    Phi0.resize((long)N*16*K0);
    A0.resize((long)N*ar*CN);
    s0.resize(N);
#pragma omp parallel
    {
    std::vector<Precision> Sb;
    std::vector<Precision> beta;
#pragma omp for schedule(dynamic, 1)
    for (int nn = 0; nn < N; ++nn) {
        const int a = elem[nn];
        Precision sum = 0;
        for (int b = 0; b < T; ++b) {
            const int e0 = seg[(long)nn*(T + 1) + b];
            const int ne = seg[(long)nn*(T + 1) + b + 1] - e0;
            const Precision* table = basis.data() + (long)(a*T + b)*K*4*NB;
            Sb.resize(16L*ne);
            beta.resize((long)ne*NB);
            for (int i = 0; i < ne; ++i) {
                const Edge<Precision> q = edge_of(e0 + i);
                const Precision* blk = table + (long)q.knot*4*NB;
                for (int lm = 0; lm < 16; ++lm) Sb[16L*i + lm] = Precision(sh_norm(lm))*q.S[lm];
#pragma omp simd
                for (int f = 0; f < NB; ++f) beta[(long)i*NB + f] = cubic(blk, NB, f, q.t);
                if (not A0_splines.empty()) sum += cubic(A0_splines.data() + ((long)q.pair*K + q.knot)*4, 1, 0, q.t);
            }
            for (int lm0 = 0; lm0 < 16; lm0 += 4)
                for (int f0 = 0; f0 < NB; f0 += 16) {
                    Precision c[4][16] = {};
                    for (int i = 0; i < ne; ++i)
                        for (int r = 0; r < 4; ++r)
#pragma omp simd
                            for (int j = 0; j < 16; ++j) c[r][j] += Sb[16L*i + lm0 + r]*beta[(long)i*NB + f0 + j];
                    for (int r = 0; r < 4; ++r) std::copy_n(c[r], 16, Phi0.data() + row(lm0 + r, nn, N)*K0 + b*NB + f0);
                }
        }
        s0[nn] = 1 + sum;
    }
    }
    for (int l = 0; l < 4; ++l)
        for (const auto& [b, e, t] : segments) {
            const long r0 = (long)N*l*l + (long)b*(2*l+1);
            gemm(false, (e - b)*(2*l+1), Sh::width(l), K0, Phi0.data() + r0*K0, K0, Q, q_off<Sh>(K0, t, l),
                 A0.data() + a_off<Sh>(l, r0, N), Sh::width(l));
        }
    products();

    // gate
    // G_{i,klm} = \tilde{A}^{(0)}_{i,klm} + \delta_{l0} r_{a_i,k},  T_{i,k00} = G_{i,k00} \sigma(G_{i,k00}),  T_{i,klm} = G_{i,klm} \sigma(G_{i,(lC_N+k)00}) (l > 0)
    // X^{(0)}_{i,klm} = \sum_{k'} w_{l,kk'} T_{i,k'lm}
    if constexpr (Sh::gated) gate<Sh>(0, linear_2_0, X0);

    // M0
    // M^{(0)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_iklm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} \tilde{A}^{(0)}_{i,kl'_\xi m'_\xi},  X^{(0)} for \tilde{A}^{(0)} if gated
    M0.resize((long)N*Sh::num_LM*CN);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const Precision inv = Sh::gated ? 1.f : 1/s0[nn];
        const Precision* xin = Sh::gated ? X0.data() : A0.data();
        for (int k0 = 0; k0 < CN; k0 += V) {
            Precision v[Sh::num_values*V];
            Precision wt[Sh::num_m*V];
            Precision out[Sh::num_LM*V] = {};
            for (int lm = 0; lm < num_lm; ++lm)
                for (int w = 0; w < V; ++w) v[lm*V + w] = xin[row(lm, nn, N)*CN + k0 + w]*inv;
            for (int e = 0; e < Sh::num_m; ++e)
                for (int w = 0; w < V; ++w) wt[e*V + w] = M0_weights[((long)elem[nn]*Sh::num_m + e)*CN + k0 + w];
            tree_eval<Sh, V>(wt, v, out);
            for (int LM = 0; LM < Sh::num_LM; ++LM)
                for (int w = 0; w < V; ++w) M0[row(LM, nn, N)*CN + k0 + w] = out[LM*V + w];
        }
    }

    // H1
    // H^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(0)}_{i,k'lm} + \delta_{l0} b^{(1)}_{a_i,k}
    H1.resize((long)NR*Sh::num_LM*C);
    for (int L = 0; L <= Sh::L_max; ++L)
        gemm(false, N*(2*L+1), C, CN, M0.data() + (long)N*L*L*CN, CN, H1_weights, (long)L*CN*C,
             H1.data() + (long)NR*L*L*C, C);
    products();
    if (not H1_bias.empty())
        for (int nn = 0; nn < N; ++nn)
            for (int k = 0; k < C; ++k) H1[(long)nn*C + k] += H1_bias[(long)elem[nn]*C + k];
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::compute_second_layer()
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    const Precision rmin = Precision(r_min);
    const Precision inv_h = Precision(1/h);
    constexpr int F = Sh::num_paths*C;
    auto edge_of = [&](int e) { return edge(K, rmin, inv_h, geometry.data(), row_pair.data(), e); };

    // Phi1
    // \Phi^{(1)}_{i,klm\eta} = \sum_j \sum_{l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) H^{(1)}_{j,kl_2m_2}
    // A1 scaling
    // s^{(1)}_i = 1 + \sum_j \rho^{(1)}_{a_ia_j}(r_{ij}),  \tilde{A}^{(1)}_{i,klm} = A^{(1)}_{i,klm} / s^{(1)}_i
    Phi1.resize((long)N*Sh::num_lme*C);
    s1.resize(N);
#pragma omp parallel
    {
        std::vector<Precision> cs;
        std::vector<const Precision*> tb;
        std::vector<Precision> xt;
        std::vector<int> jr;
        std::vector<Precision*> out(Sh::num_lme);
        std::vector<Precision> Rb(Sh::num_paths*EB*4*V);
#pragma omp for schedule(dynamic, 1)
        for (int nn = 0; nn < N; ++nn) {
            const int ne = start[nn + 1] - start[nn];
            cs.resize((long)ne*path_first<Sh>(Sh::num_paths));
            tb.resize(ne);
            xt.resize(ne);
            jr.resize(ne);
            Precision sum = 0;
            for (int i = 0; i < ne; ++i) {
                const Edge<Precision> q = edge_of(start[nn] + i);
                stage_cs<Sh>(q.S, i, ne, cs.data());
                tb[i] = R1.data() + ((long)q.pair*K + q.knot)*4*F;
                xt[i] = q.t;
                jr[i] = q.row;
                if (not A1_splines.empty()) sum += cubic(A1_splines.data() + ((long)q.pair*K + q.knot)*4, 1, 0, q.t);
            }
            s1[nn] = 1 + sum;
            for (int lme = 0; lme < Sh::num_lme; ++lme) out[lme] = Phi1.data() + phi1_row<Sh>(lme, nn, N)*C;
            for (int c = 0; c < C/V; c += 4) {
                Precision acc4[Sh::num_lme][4*V] = {};
                for (int i0 = 0; i0 < ne; i0 += EB) {
                    const int i1 = std::min(ne, i0 + EB);
                    for (int i = i0; i < i1; ++i)
                        for (int u = 0; u < 4; ++u)
                            for (int p = 0; p < Sh::num_paths; ++p) {
                                const Precision* b = tb[i] + ((c + u)*Sh::num_paths + p)*4*V;
                                Precision* r = Rb.data() + ((p*EB + i - i0)*4 + u)*V;
#pragma omp simd
                                for (int w = 0; w < V; ++w) r[w] = cubic(b, V, w, xt[i]);
                            }
                    fwd_paths<Sh>(c, i0, i1, ne, cs.data(), Rb.data(), jr.data(), H1.data(), NR, acc4);
                }
                for (int lme = 0; lme < Sh::num_lme; ++lme) std::copy_n(acc4[lme], 4*V, out[lme] + c*V);
            }
        }
    }

    // A1
    // A^{(1)}_{i,klm} = \sum_{k'\eta'} w_{kl,k'\eta'} \Phi^{(1)}_{i,k'lm\eta'}
    const long ar = gated ? num_lm + 3 : num_lm;
    A1.resize((long)N*ar*CN);
    for (int l = 0; l < 4; ++l)
        gemm(false, N*(2*l+1), Sh::width(l), Sh::num_e(l)*C, Phi1.data() + (long)N*Sh::lme_offset(l)*C, Sh::num_e(l)*C,
             A1_weights, A1_weights_offset[l],
             A1.data() + a_off<Sh>(l, (long)N*l*l, N), Sh::width(l));
    products();

    // gate
    // G_{i,klm} = \tilde{A}^{(1)}_{i,klm} + \sum_{k'} w_{kl,k'} H^{(1)}_{i,k'lm},  T_{i,k00} = G_{i,k00} \sigma(G_{i,k00}),  T_{i,klm} = G_{i,klm} \sigma(G_{i,(lC_N+k)00}) (l > 0)
    // X^{(1)}_{i,klm} = \sum_{k'} w_{l,kk'} T_{i,k'lm}
    if constexpr (Sh::gated) {
        B1.resize((long)N*(4 + num_LM - 1)*CN);
        for (int L = 0; L <= Sh::L_max; ++L)
            gemm(false, N*(2*L+1), Sh::width(L), C, H1.data() + (long)NR*L*L*C, C, residual_1_weights, r_off<Sh>(L),
                 B1.data() + a_off<Sh>(L, (long)N*L*L, N), Sh::width(L));
        products();
        gate<Sh>(1, linear_2_1, X1);
    }

    // M1
    // M^{(1)}_{i,k} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_ik,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} \tilde{A}^{(1)}_{i,kl'_\xi m'_\xi},  X^{(1)} for \tilde{A}^{(1)} if gated
    M1.resize((long)N*CN);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const int t = elem[nn];
        const Precision inv = Sh::gated ? 1.f : 1/s1[nn];
        const Precision* xin = Sh::gated ? X1.data() : A1.data();
        for (int k0 = 0; k0 < CN; k0 += TW) {
            Precision v[T1::num_values*TW];
            Precision wt[T1::num_m*TW];
            Precision m1[TW] = {};
            for (int lm = 0; lm < num_lm; ++lm)
#pragma omp simd
                for (int w = 0; w < TW; ++w) v[lm*TW + w] = xin[row(lm, nn, N)*CN + k0 + w]*inv;
            for (int e = 0; e < T1::num_m; ++e)
                std::copy_n(M1_weights.data() + ((long)t*T1::num_m + e)*CN + k0, TW, wt + e*TW);
            tree_eval<T1, TW>(wt, v, m1);
            std::copy_n(m1, TW, M1.data() + (long)nn*CN + k0);
        }
    }
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::compute_readouts()
{
    constexpr int CN = Sh::CN;
    const int T = num_elements;

    // ZBL
    // E_i += \frac{1}{2} \sum_j V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij})
    // \mathbf{F}_{ij} -= \frac{1}{2} \partial_r V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij}) \hat{\mathbf{r}}_{ij}
    node_energies.resize(N);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const int a = elem[nn];
        double ez = 0;
        std::fill(edge_forces.data() + 3L*start[nn], edge_forces.data() + 3L*start[nn + 1], 0.f);
        for (int b = 0; has_zbl and b < T; ++b) {
            const Precision* z = zbl.data() + 3*pair_index(a, b, T);
            for (int e = seg[(long)nn*(T + 1) + b]; e < seg[(long)nn*(T + 1) + b + 1]; ++e) {
                const Precision rf = geometry[4*e+3];
                if (rf >= z[1]) continue;
                const Precision ce[4] = {-3.2, -0.9423, -0.4028, -0.2016};
                const Precision y = rf/z[0];
                Precision phi = 0;
                Precision dphi = 0;
                for (int q = 0; q < 4; ++q) {
                    const Precision v = zbl_c[q]*std::exp(ce[q]*y);
                    phi += v;
                    dphi += ce[q]*v/z[0];
                }
                const Precision v = z[2]*phi/rf;
                const Precision dv = z[2]*(rf*dphi - phi)/(rf*rf);
                const Precision cp = Precision(zbl_p);
                const Precision u = rf/z[1];
                const Precision up = std::pow(u, cp - 1);
                const Precision env = 1 - 0.5f*(cp+1)*(cp+2)*up*u + cp*(cp+2)*up*u*u - 0.5f*cp*(cp+1)*up*u*u*u;
                const Precision denv = (-0.5f*(cp+1)*(cp+2)*cp*up + cp*(cp+2)*(cp+1)*up*u
                                        - 0.5f*cp*(cp+1)*(cp+2)*up*u*u)/z[1];
                ez += 0.5*v*env;
                const Precision fz = -0.5f*(v*denv + dv*env);
                edge_forces[3*e] = fz*geometry[4*e+1];
                edge_forces[3*e+1] = fz*geometry[4*e+2];
                edge_forces[3*e+2] = fz*geometry[4*e];
            }
        }
        node_energies[nn] = ez;
    }

    // H2
    // H^{(2)}_{i,k} = \sum_{k'} w_{k,k'} M^{(1)}_{i,k'} + \sum_{k'} w_{a_ik,k'} M^{(0)}_{i,k'00} + b^{(2)}_{a_i,k}
    // z_{i,h} = \sum_k w_{h,k} H^{(2)}_{i,k} = \sum_k (u_{a_ih,k} M^{(0)}_{i,k00} + u_{h,k} M^{(1)}_{i,k}) + v_{a_ih}
    // \bar{M}^{(1)}_{i,k} = \sum_h u_{h,k} \bar{z}_{i,h},  \bar{M}^{(0)}_{i,k00} += \sum_h u_{a_ih,k} \bar{z}_{i,h}
    // readouts
    // E_i += E^{(0)}_{a_i} + \sum_k w_k M^{(0)}_{i,k00} + \mathrm{MLP}(\{H^{(2)}_{i,k}\}),  \mathrm{MLP} = \sum_h w_h c\, z_{i,h} \sigma(z_{i,h})
    // \bar{M}^{(0)}_{i,k00} = w_k,  \bar{z}_{i,h} = w_h c\, \sigma(z_{i,h}) (1 + z_{i,h} (1 - \sigma(z_{i,h})))
    M1_adj.resize((long)N*CN);
    M0s_adj.resize((long)N*CN);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const int t = elem[nn];
        const Precision* hin = M0.data() + (long)nn*CN;
        const Precision* m1 = M1.data() + (long)nn*CN;
        const Precision* fh = z_weights_for_M0.data() + (long)t*16*CN;
        Precision part[17][TW] = {};
        for (int k0 = 0; k0 < CN; k0 += TW) {
            for (int hh = 0; hh < 16; ++hh)
#pragma omp simd
                for (int w = 0; w < TW; ++w)
                    part[hh][w] += hin[k0 + w]*fh[hh*CN + k0 + w] + m1[k0 + w]*z_weights_for_M1[hh*CN + k0 + w];
#pragma omp simd
            for (int w = 0; w < TW; ++w) part[16][w] += readout_1_weights_for_M0[k0 + w]*hin[k0 + w];
        }
        Precision gz[16];
        Precision p16 = 0;
        for (int w = 0; w < TW; ++w) p16 += part[16][w];
        double En = atomic_energies[t] + p16;
        for (int hh = 0; hh < 16; ++hh) {
            Precision z = z_bias[t*16 + hh];
            for (int w = 0; w < TW; ++w) z += part[hh][w];
            const Precision sg = 1/(1 + std::exp(-z));
            En += (double)readout_2_weights_2[hh]*(readout_2_scale_factor*z*sg);
            gz[hh] = readout_2_weights_2[hh]*readout_2_scale_factor*sg*(1 + z*(1 - sg));
        }
        node_energies[nn] += En;
        for (int k0 = 0; k0 < CN; k0 += TW) {
            Precision aH[TW];
            Precision aM[TW] = {};
            std::copy_n(readout_1_weights_for_M0.data() + k0, TW, aH);
            for (int hh = 0; hh < 16; ++hh)
#pragma omp simd
                for (int w = 0; w < TW; ++w) {
                    aH[w] += gz[hh]*fh[hh*CN + k0 + w];
                    aM[w] += gz[hh]*z_weights_for_M1[hh*CN + k0 + w];
                }
            std::copy_n(aH, TW, M0s_adj.data() + (long)nn*CN + k0);
            std::copy_n(aM, TW, M1_adj.data() + (long)nn*CN + k0);
        }
    }
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::reverse_second_layer()
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    const Precision rmin = Precision(r_min);
    const Precision inv_h = Precision(1/h);
    constexpr int F = Sh::num_paths*C;
    auto edge_of = [&](int e) { return edge(K, rmin, inv_h, geometry.data(), row_pair.data(), e); };

    // M1
    // \bar{\tilde{A}}^{(1)}_{i,klm} = \bar{M}^{(1)}_{i,k} \partial M^{(1)}_{i,k} / \partial \tilde{A}^{(1)}_{i,klm},  \bar{X}^{(1)} for \bar{\tilde{A}}^{(1)} if gated
    // \bar{A}^{(1)}_{i,klm} = \bar{\tilde{A}}^{(1)}_{i,klm} / s^{(1)}_i,  d^{(1)}_i = \sum_{klm} \bar{\tilde{A}}^{(1)}_{i,klm} \tilde{A}^{(1)}_{i,klm} / s^{(1)}_i
    const long ar = gated ? num_lm + 3 : num_lm;
    A1_adj.resize((long)N*ar*CN);
    dot1.resize(N);
    if constexpr (Sh::gated) X_adj.resize((long)N*16*CN);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const int t = elem[nn];
        const Precision inv = Sh::gated ? 1.f : 1/s1[nn];
        const Precision* xin = Sh::gated ? X1.data() : A1.data();
        Precision dot[TW] = {};
        for (int k0 = 0; k0 < CN; k0 += TW) {
            Precision v[T1::num_values*TW];
            Precision wt[T1::num_m*TW];
            Precision m1[TW] = {};
            Precision adjx[num_lm*TW];
            for (int lm = 0; lm < num_lm; ++lm)
#pragma omp simd
                for (int w = 0; w < TW; ++w) v[lm*TW + w] = xin[row(lm, nn, N)*CN + k0 + w]*inv;
            for (int e = 0; e < T1::num_m; ++e)
                std::copy_n(M1_weights.data() + ((long)t*T1::num_m + e)*CN + k0, TW, wt + e*TW);
            tree_eval<T1, TW>(wt, v, m1);
            tree_adjoint<T1, TW>(wt, v, M1_adj.data() + (long)nn*CN + k0, adjx);
            for (int lm = 0; lm < num_lm; ++lm)
#pragma omp simd
                for (int w = 0; w < TW; ++w) {
                    if constexpr (Sh::gated) X_adj[row(lm, nn, N)*CN + k0 + w] = adjx[lm*TW + w];
                    else {
                        A1_adj[row(lm, nn, N)*CN + k0 + w] = adjx[lm*TW + w]*inv;
                        dot[w] += v[lm*TW + w]*adjx[lm*TW + w];
                    }
                }
        }
        if constexpr (not Sh::gated) {
            Precision d = 0;
            for (int w = 0; w < TW; ++w) d += dot[w];
            dot1[nn] = d*inv;
        }
    }

    H1_adj.resize((long)NR*Sh::num_LM*C);
#pragma omp parallel for schedule(static)
    for (int r = Sh::gated ? N : 0; r < NR; ++r)
        for (int LM = 0; LM < Sh::num_LM; ++LM) std::fill_n(H1_adj.data() + row(LM, r, NR)*C, C, 0.f);

    // gate
    // \bar{T}_{i,k'lm} = \sum_k w_{l,kk'} \bar{X}^{(1)}_{i,klm},  \bar{G}_{i,k00} = \bar{T}_{i,k00} \sigma(G_{i,k00}) (1 + G_{i,k00} (1 - \sigma(G_{i,k00})))
    // \bar{G}_{i,klm} = \bar{T}_{i,klm} \sigma(\gamma),  \bar{G}_{i,(lC_N+k)00} = \sigma(\gamma) (1 - \sigma(\gamma)) \sum_m G_{i,klm} \bar{T}_{i,klm},  \gamma = G_{i,(lC_N+k)00} (l > 0)
    // \bar{A}^{(1)}_{i,klm} = \bar{G}_{i,klm} / s^{(1)}_i,  d^{(1)}_i = \sum_{klm} \bar{G}_{i,klm} \tilde{A}^{(1)}_{i,klm} / s^{(1)}_i,  \bar{H}^{(1)}_{i,k'lm} = \sum_k w_{kl,k'} \bar{G}_{i,klm}
    if constexpr (Sh::gated) {
        gate_reverse<Sh>(1, linear_2_1);
        for (int L = 0; L <= Sh::L_max; ++L)
            gemm(true, N*(2*L+1), C, Sh::width(L), B1_adj.data() + a_off<Sh>(L, (long)N*L*L, N), Sh::width(L),
                 residual_1_weights, r_off<Sh>(L), H1_adj.data() + (long)NR*L*L*C, C);
        products();
    }

    // A1
    // \bar{\Phi}^{(1)}_{i,k'lm\eta'} = \sum_k w_{kl,k'\eta'} \bar{A}^{(1)}_{i,klm}
    dPhi1.resize((long)N*Sh::num_lme*C);
    for (int l = 0; l < 4; ++l)
        gemm(true, N*(2*l+1), Sh::num_e(l)*C, Sh::width(l), A1_adj.data() + a_off<Sh>(l, (long)N*l*l, N), Sh::width(l),
             A1_weights, A1_weights_offset[l],
             dPhi1.data() + (long)N*Sh::lme_offset(l)*C, Sh::num_e(l)*C);
    products();

    // A1 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(1)}_i
    // Phi1
    // \bar{H}^{(1)}_{j,kl_2m_2} = \sum_i \sum_{lm\eta l_1m_1} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(1)}_{i,klm\eta}
    // \mathbf{F}_{ij} -= \sum_{klm\eta l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij})] H^{(1)}_{j,kl_2m_2} \bar{\Phi}^{(1)}_{i,klm\eta}
#pragma omp parallel
    {
        std::vector<Precision> Sv;
        std::vector<const Precision*> tb;
        std::vector<Precision> xt;
        std::vector<int> jr;
        std::vector<Precision> ga;
        std::vector<Precision> gh;
        std::vector<Precision> kw((long)Sh::num_terms*V);
#pragma omp for schedule(dynamic, 1)
        for (int nn = 0; nn < N; ++nn) {
            const int ne = start[nn + 1] - start[nn];
            Sv.resize(16L*ne);
            tb.resize(ne);
            xt.resize(ne);
            jr.resize(ne);
            ga.assign(17L*V*ne, 0.f);
            gh.resize((long)ne*Sh::num_LM*C);
            for (int i = 0; i < ne; ++i) {
                const Edge<Precision> q = edge_of(start[nn] + i);
                std::copy_n(q.S, 16, Sv.data() + 16L*i);
                tb[i] = R1.data() + ((long)q.pair*K + q.knot)*4*F;
                xt[i] = q.t;
                jr[i] = q.row;
            }
            for (int c = 0; c < C/V; ++c) {
                Precision adj[Sh::num_lme*V];
                for (int lme = 0; lme < Sh::num_lme; ++lme)
                    std::copy_n(dPhi1.data() + phi1_row<Sh>(lme, nn, N)*C + c*V, V, adj + lme*V);
                key_ws<Sh, V>(adj, kw.data());
                for (int i = 0; i < ne; ++i) {
                    const long r[4] = {row(0, jr[i], NR)*C, row(1, jr[i], NR)*C, row(4, jr[i], NR)*C,
                                       row(9, jr[i], NR)*C};
                    Precision g_h[Sh::num_LM][V] = {};
                    rev_paths<Sh>(tb[i] + c*Sh::num_paths*4*V, xt[i], Sv.data() + 16L*i, kw.data(), H1.data(), C, c*V,
                                  r, g_h, ga.data() + 17L*V*i);
                    for (int LM = 0; LM < Sh::num_LM; ++LM)
                        std::copy_n(g_h[LM], V, gh.data() + ((long)i*Sh::num_LM + LM)*C + c*V);
                }
            }
            for (int i = 0; i < ne; ++i) {
                const int e = start[nn] + i;
                const Edge<Precision> q = edge_of(e);
                const Precision* gi = ga.data() + 17L*V*i;
                Precision gs[17];
                for (int lm = 0; lm < 17; ++lm) {
                    gs[lm] = 0;
                    for (int w = 0; w < V; ++w) gs[lm] += gi[lm*V + w];
                }
                const Precision a = gs[16]*inv_h;
                Precision Gd[3] = {};
                grad(q.S, gs, Gd);
                const Precision u[3] = {q.y, q.z, q.x};
                const Precision g3[3] = {Gd[1]*q.rinv, Gd[2]*q.rinv, Gd[0]*q.rinv};
                const Precision ug = u[0]*g3[0] + u[1]*g3[1] + u[2]*g3[2];
                for (int d = 0; d < 3; ++d) edge_forces[3*e + d] += -(a*u[d] + g3[d] - u[d]*ug);
                if (not A1_splines.empty()) {
                    Precision v;
                    Precision ds;
                    cubic(A1_splines.data() + ((long)q.pair*K + q.knot)*4, 1, 0, q.t, v, ds);
                    for (int d = 0; d < 3; ++d) edge_forces[3*e + d] += dot1[nn]*ds*inv_h*u[d];
                }
                while (locks[q.row].test_and_set(std::memory_order_acquire)) {}
                for (int LM = 0; LM < Sh::num_LM; ++LM) {
                    Precision* dst = H1_adj.data() + row(LM, q.row, NR)*C;
                    const Precision* src = gh.data() + ((long)i*Sh::num_LM + LM)*C;
#pragma omp simd
                    for (int w = 0; w < C; ++w) dst[w] += src[w];
                }
                locks[q.row].clear(std::memory_order_release);
            }
        }
    }
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::reverse_first_layer()
{
    constexpr int C = Sh::C;
    constexpr int CN = Sh::CN;
    const int T = num_elements;
    const Precision rmin = Precision(r_min);
    const Precision inv_h = Precision(1/h);
    auto edge_of = [&](int e) { return edge(K, rmin, inv_h, geometry.data(), row_pair.data(), e); };

    // H1
    // \bar{M}^{(0)}_{i,k'lm} += \sum_k w_{kl,k'} \bar{H}^{(1)}_{i,klm}
    M0_adj.resize((long)N*Sh::num_LM*CN);
    for (int L = 0; L <= Sh::L_max; ++L)
        gemm(true, N*(2*L+1), CN, C, H1_adj.data() + (long)NR*L*L*C, C, H1_weights, (long)L*CN*C,
             M0_adj.data() + (long)N*L*L*CN, CN);
    products();

    // M0
    // \bar{\tilde{A}}^{(0)}_{i,klm} = \sum_{l'm'} \bar{M}^{(0)}_{i,kl'm'} \partial M^{(0)}_{i,kl'm'} / \partial \tilde{A}^{(0)}_{i,klm},  \bar{X}^{(0)} for \bar{\tilde{A}}^{(0)} if gated
    // \bar{A}^{(0)}_{i,klm} = \bar{\tilde{A}}^{(0)}_{i,klm} / s^{(0)}_i,  d^{(0)}_i = \sum_{klm} \bar{\tilde{A}}^{(0)}_{i,klm} \tilde{A}^{(0)}_{i,klm} / s^{(0)}_i
    const long ar = gated ? num_lm + 3 : num_lm;
    A0_adj.resize((long)N*ar*CN);
    dot0.resize(N);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const Precision inv = Sh::gated ? 1.f : 1/s0[nn];
        const Precision* xin = Sh::gated ? X0.data() : A0.data();
        Precision dot = 0;
        for (int k0 = 0; k0 < CN; k0 += TW) {
            Precision v[Sh::num_values*TW];
            Precision wt[Sh::num_m*TW];
            Precision g[Sh::num_LM*TW];
            Precision adjx[num_lm*TW];
            Precision out[Sh::num_LM*TW] = {};
            for (int lm = 0; lm < num_lm; ++lm)
                for (int w = 0; w < TW; ++w) v[lm*TW + w] = xin[row(lm, nn, N)*CN + k0 + w]*inv;
            for (int e = 0; e < Sh::num_m; ++e)
                for (int w = 0; w < TW; ++w) wt[e*TW + w] = M0_weights[((long)elem[nn]*Sh::num_m + e)*CN + k0 + w];
            for (int LM = 0; LM < Sh::num_LM; ++LM)
                for (int w = 0; w < TW; ++w) g[LM*TW + w] = M0_adj[row(LM, nn, N)*CN + k0 + w];
            for (int w = 0; w < TW; ++w) g[w] += M0s_adj[(long)nn*CN + k0 + w];
            tree_eval<Sh, TW>(wt, v, out);
            tree_adjoint<Sh, TW>(wt, v, g, adjx);
            for (int lm = 0; lm < num_lm; ++lm)
                for (int w = 0; w < TW; ++w) {
                    if constexpr (Sh::gated) X_adj[row(lm, nn, N)*CN + k0 + w] = adjx[lm*TW + w];
                    else {
                        A0_adj[row(lm, nn, N)*CN + k0 + w] = adjx[lm*TW + w]*inv;
                        dot += v[lm*TW + w]*adjx[lm*TW + w];
                    }
                }
        }
        if constexpr (not Sh::gated) dot0[nn] = dot*inv;
    }

    // gate
    // \bar{T}_{i,k'lm} = \sum_k w_{l,kk'} \bar{X}^{(0)}_{i,klm},  \bar{G} as in reverse_second_layer
    // \bar{A}^{(0)}_{i,klm} = \bar{G}_{i,klm} / s^{(0)}_i,  d^{(0)}_i = \sum_{klm} \bar{G}_{i,klm} \tilde{A}^{(0)}_{i,klm} / s^{(0)}_i
    if constexpr (Sh::gated) gate_reverse<Sh>(0, linear_2_0);

    // A0 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(0)}_i
    // A0
    // \bar{\Phi}^{(0)}_{i,lm,bn} = \sum_k Q_{a_il,k,bn} \bar{A}^{(0)}_{i,klm}
    // \mathbf{F}_{ij} -= \sum_{lmn} \nabla_{\mathbf{r}_{ij}}[\beta_{a_ia_j,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})] \bar{\Phi}^{(0)}_{i,lm,a_jn}
    Phi0_adj.resize((long)N*16*K0);
    for (int l = 0; l < 4; ++l)
        for (const auto& [b, e, t] : segments) {
            const long r0 = (long)N*l*l + (long)b*(2*l+1);
            gemm(true, (e - b)*(2*l+1), K0, Sh::width(l), A0_adj.data() + a_off<Sh>(l, r0, N), Sh::width(l), Q,
                 q_off<Sh>(K0, t, l), Phi0_adj.data() + r0*K0, K0);
        }
    products();
#pragma omp parallel
    {
        std::vector<Precision> adj(16*NB);
        std::vector<Precision> Tt(NUM_T*NB);
        std::vector<Precision> W(64L*NB, 0.f);
        std::vector<Precision> B;
        std::vector<Precision> Y;
        auto tiles = [&](int ne, const Precision* A, int s0, int s1, Precision* out) {
            for (int i0 = 0; i0 < ne; i0 += 4)
                for (int s = s0; s < s1; ++s) {
                    const Precision* x[4];
                    for (int r = 0; r < 4; ++r) x[r] = A + (long)std::min(i0 + r, ne - 1)*NB;
                    Precision c[4][16] = {};
                    for (int f = 0; f < NB; ++f)
                        for (int r = 0; r < 4; ++r)
#pragma omp simd
                            for (int j = 0; j < 16; ++j) c[r][j] += x[r][f]*W[f*64L + s*16 + j];
                    for (int r = 0; r < std::min(4, ne - i0); ++r)
                        std::copy_n(c[r], 16, out + (long)(i0 + r)*64 + s*16);
                }
        };
#pragma omp for schedule(dynamic, 1)
        for (int nn = 0; nn < N; ++nn) {
            const int a = elem[nn];
            for (int b = 0; b < T; ++b) {
                const int e0 = seg[(long)nn*(T + 1) + b];
                const int ne = seg[(long)nn*(T + 1) + b + 1] - e0;
                for (int lm = 0; lm < 16; ++lm)
                    for (int f = 0; f < NB; ++f)
                        adj[lm*NB + f] = Phi0_adj[row(lm, nn, N)*K0 + b*NB + f]*Precision(sh_norm(lm));
                switch (NB) {
                    case 32: build_T<32>(adj.data(), Tt.data()); break;
                    case 64: build_T<64>(adj.data(), Tt.data()); break;
                    default: build_T<96>(adj.data(), Tt.data());
                    break;
                }
                for (int f = 0; f < NB; ++f) {
                    for (int lm = 0; lm < 16; ++lm) W[f*64L + lm] = W[f*64L + 48 + lm] = adj[lm*NB + f];
                    for (int t = 0; t < NUM_T; ++t) W[f*64L + 16 + t] = Tt[t*NB + f];
                }
                const Precision* table = basis.data() + (long)(a*T + b)*K*4*NB;
                B.resize(2L*ne*NB);
                Y.resize(64L*ne);
                for (int i = 0; i < ne; ++i) {
                    const Edge<Precision> q = edge_of(e0 + i);
                    const Precision* blk = table + (long)q.knot*4*NB;
#pragma omp simd
                    for (int f = 0; f < NB; ++f) cubic(blk, NB, f, q.t, B[(long)i*NB + f], B[((long)ne + i)*NB + f]);
                }
                tiles(ne, B.data(), 0, 3, Y.data());
                tiles(ne, B.data() + (long)ne*NB, 3, 4, Y.data());
                for (int i = 0; i < ne; ++i) {
                    const int e = e0 + i;
                    const Edge<Precision> q = edge_of(e);
                    const Precision* y = Y.data() + 64L*i;
                    Precision sy = 0;
                    Precision sl = 0;
                    for (int lm = 0; lm < 16; ++lm) {
                        sy += q.S[lm]*y[48 + lm];
                        sl += lm_l(lm)*q.S[lm]*y[lm];
                    }
                    const Precision dr = inv_h*sy - q.rinv*sl;
                    Precision Gd[3] = {};
                    for (int d = 0; d < 3; ++d)
                        for (int mp = 0; mp < 9; ++mp) Gd[d] += q.S[mp]*y[16 + t_index(lm_l(mp) + 1, d, mp)];
                    edge_forces[3*e] -= dr*q.y + q.rinv*Gd[1];
                    edge_forces[3*e+1] -= dr*q.z + q.rinv*Gd[2];
                    edge_forces[3*e+2] -= dr*q.x + q.rinv*Gd[0];
                    if (not A0_splines.empty()) {
                        Precision v;
                        Precision ds;
                        cubic(A0_splines.data() + ((long)q.pair*K + q.knot)*4, 1, 0, q.t, v, ds);
                        const Precision sf = dot0[nn]*ds*inv_h;
                        edge_forces[3*e] += sf*q.y;
                        edge_forces[3*e+1] += sf*q.z;
                        edge_forces[3*e+2] += sf*q.x;
                    }
                }
            }
        }
    }
}

template <class Precision>
template <class Sh, class Rows>
void MACEFoundation<Precision>::gate_inputs(int layer, int nn, int k0, Precision inv, Rows a, Rows b)
{
    constexpr int CN = Sh::CN;
    const Precision* A = layer ? A1.data() : A0.data();
    for (int r = 0; r < num_lm + 3; ++r)
#pragma omp simd
        for (int w = 0; w < V; ++w) b[r][w] = a[r][w] = A[a_index<Sh>(r, nn, N) + k0 + w]*inv;
    for (int q = 0; q < 4; ++q) {
        const Precision* B = layer ? B1.data() + ((long)nn*4 + q)*CN : residual_0.data() + ((long)elem[nn]*4 + q)*CN;
#pragma omp simd
        for (int w = 0; w < V; ++w) b[q == 0 ? 0 : 15 + q][w] += B[k0 + w];
    }
    if (layer)
        for (int lm = 1; lm < Sh::num_LM; ++lm)
#pragma omp simd
            for (int w = 0; w < V; ++w) b[lm][w] += B1[(long)N*4*CN + (row(lm, nn, N) - N)*CN + k0 + w];
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::gate(int layer, const Weight& L2, Floats& X)
{
    constexpr int CN = Sh::CN;
    Tg.resize((long)N*16*CN);
    X.resize((long)N*16*CN);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const Precision inv = 1/(layer ? s1[nn] : s0[nn]);
        for (int k0 = 0; k0 < CN; k0 += V) {
            Precision a[num_lm + 3][V];
            Precision b[num_lm + 3][V];
            gate_inputs<Sh>(layer, nn, k0, inv, a, b);
#pragma omp simd
            for (int w = 0; w < V; ++w) Tg[(long)nn*CN + k0 + w] = b[0][w]*sigmoid(b[0][w]);
            for (int l = 1; l < 4; ++l) {
                Precision s[V];
#pragma omp simd
                for (int w = 0; w < V; ++w) s[w] = sigmoid(b[15 + l][w]);
                for (int lm = l*l; lm < (l + 1)*(l + 1); ++lm)
#pragma omp simd
                    for (int w = 0; w < V; ++w) Tg[row(lm, nn, N)*CN + k0 + w] = b[lm][w]*s[w];
            }
        }
    }
    for (int l = 0; l < 4; ++l)
        gemm(false, N*(2*l+1), CN, CN, Tg.data() + (long)N*l*l*CN, CN, L2, (long)l*CN*CN,
             X.data() + (long)N*l*l*CN, CN);
    products();
}

template <class Precision>
template <class Sh>
void MACEFoundation<Precision>::gate_reverse(int layer, const Weight& L2)
{
    constexpr int CN = Sh::CN;
    T_adj.resize((long)N*16*CN);
    for (int l = 0; l < 4; ++l)
        gemm(true, N*(2*l+1), CN, CN, X_adj.data() + (long)N*l*l*CN, CN, L2, (long)l*CN*CN,
             T_adj.data() + (long)N*l*l*CN, CN);
    products();
    Floats& A_adj = layer ? A1_adj : A0_adj;
    A_adj.resize((long)N*(num_lm + 3)*CN);
    if (layer) B1_adj.resize((long)N*(4 + num_LM - 1)*CN);
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < N; ++nn) {
        const Precision inv = 1/(layer ? s1[nn] : s0[nn]);
        Precision dot[V] = {};
        for (int k0 = 0; k0 < CN; k0 += V) {
            Precision a[num_lm + 3][V];
            Precision b[num_lm + 3][V];
            Precision ba[num_lm + 3][V];
            gate_inputs<Sh>(layer, nn, k0, inv, a, b);
#pragma omp simd
            for (int w = 0; w < V; ++w) {
                const Precision sg = sigmoid(b[0][w]);
                ba[0][w] = T_adj[(long)nn*CN + k0 + w]*sg*(1 + b[0][w]*(1 - sg));
            }
            for (int l = 1; l < 4; ++l) {
                Precision s[V];
                Precision ga[V] = {};
#pragma omp simd
                for (int w = 0; w < V; ++w) s[w] = sigmoid(b[15 + l][w]);
                for (int lm = l*l; lm < (l + 1)*(l + 1); ++lm)
#pragma omp simd
                    for (int w = 0; w < V; ++w) {
                        const Precision ta = T_adj[row(lm, nn, N)*CN + k0 + w];
                        ba[lm][w] = ta*s[w];
                        ga[w] += ta*b[lm][w];
                    }
#pragma omp simd
                for (int w = 0; w < V; ++w) ba[15 + l][w] = ga[w]*s[w]*(1 - s[w]);
            }
            for (int r = 0; r < num_lm + 3; ++r)
#pragma omp simd
                for (int w = 0; w < V; ++w) {
                    A_adj[a_index<Sh>(r, nn, N) + k0 + w] = ba[r][w]*inv;
                    dot[w] += a[r][w]*ba[r][w];
                }
            if (layer) {
                for (int q = 0; q < 4; ++q)
                    std::copy_n(ba[q == 0 ? 0 : 15 + q], V, B1_adj.data() + ((long)nn*4 + q)*CN + k0);
                for (int lm = 1; lm < Sh::num_LM; ++lm)
                    std::copy_n(ba[lm], V, B1_adj.data() + (long)N*4*CN + (row(lm, nn, N) - N)*CN + k0);
            }
        }
        Precision d = 0;
        for (int w = 0; w < V; ++w) d += dot[w];
        (layer ? dot1 : dot0)[nn] = d*inv;
    }
}

template <class Precision>
void MACEFoundation<Precision>::load(const std::string& filename, const std::string& prefix)
{
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
        const long B = (long)P*K;
        const long F = (long)num_paths*C;
        R1.resize(table.size());
#pragma omp parallel for schedule(static)
        for (long b = 0; b < B; ++b)
            for (int k = 0; k < 4; ++k)
                for (long f = 0; f < F; ++f) {
                    const long w = f%C;
                    R1[b*4*F + ((w/V*num_paths + f/C)*4 + k)*V + w%V] = table[b*4*F + k*F + f];
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
            for (const auto& f : js.at(key)[p]) for (const auto& v : f) m.push_back(v.template get<double>());
            return m;
        };
        basis.assign((long)T*T*K*4*NB, 0.f);
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
                            for (int q = 0; q < N; ++q) {
                                v += B[(long)r*N + q]*V0[(long)q*S + c];
                                d += B[(long)r*N + q]*D0[(long)q*S + c];
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
                Precision* tab = basis.data() + (long)(a*T + b)*K*4*NB;
                for (int n = 0; n < nb; ++n)
                    spline(Matrix(V.begin() + n*S, V.begin() + (n + 1)*S),
                           Matrix(D.begin() + n*S, D.begin() + (n + 1)*S), h, NB, n, tab);
                for (int l = 0, o = 0; l < 4; o += width[l++])
                    for (int n = 0; n < nb; ++n)
                        for (int k = 0; k < width[l]; ++k)
                            q[((long)a*widths + o)*K0 + ((long)b*NB + n)*width[l] + k] =
                                Precision(map[((long)o + k)*nb + n]);
            }
        std::vector<std::array<long, 3>> blocks;
        for (long a = 0, o = 0; a < T; ++a)
            for (int l = 0; l < 4; o += (long)K0*width[l++]) blocks.push_back({o, K0, width[l]});
        set_weight(Q, q, blocks);
    }

    auto scale = [&](const std::string& key, std::vector<Precision>& buf) {
        if (not js.at(key + "_scaled").get<bool>()) return;
        const auto v = js.at(key + "_spline_values").get<std::vector<std::vector<double>>>();
        const auto d = js.at(key + "_spline_derivs").get<std::vector<std::vector<double>>>();
        if (js.at(key + "_spline_r_min").get<double>() != r_min or int(v[0].size()) != K + 1)
            throw std::runtime_error(prefix + "the scale-factor splines must use the radial grid");
        std::vector<Precision> table((long)P*K*4);
        for (int p = 0; p < P; ++p) spline(v[p], d[p], h, 1, 0, table.data() + (long)p*K*4);
        buf = table;
    };
    scale("A0", A0_splines);
    scale("A1", A1_splines);

    auto contraction = [&](const std::string& l, std::vector<Precision>& buf) {
        std::vector<Precision> w;
        for (const auto& a : js.at("M" + l + "_weights"))
            for (const auto& e : a) for (const auto& x : e) w.push_back(x.template get<Precision>());
        buf = w;
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
        z_weights_for_M0 = fh;
        z_weights_for_M1 = fm;
        z_bias = rb;
        std::vector<Precision> r1w, r2w;
        for (double v : js.at("readout_1_weights_for_M0").get<std::vector<double>>()) r1w.push_back(Precision(v));
        for (double v : js.at("readout_2_weights_2").get<std::vector<double>>()) r2w.push_back(Precision(v));
        readout_1_weights_for_M0 = r1w;
        readout_2_weights_2 = r2w;
        readout_2_scale_factor = js.at("readout_2_scale_factor").get<Precision>();
        atomic_energies = js.at("atomic_energies").get<std::vector<double>>();
    }

    if (gated) {
        auto flat = [](const json& j) {
            std::vector<Precision> v;
            for (const auto& r : j) for (const auto& x : r) v.push_back(x.get<Precision>());
            return v;
        };
        residual_0 = flat(js.at("residual_0"));
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
        H1_bias = b;
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
        zbl = z;
        for (int q = 0; q < 4; ++q) zbl_c[q] = Precision(c[q]);
        zbl_p = int(js.at("zbl_p").get<double>());
    }
}

template <class Precision>
void MACEFoundation<Precision>::products()
{
    constexpr int rows = sizeof(Precision) == 4 ? 6 : 3;
#pragma omp parallel
    for (const auto& [m, n, k, A, lda, B, C, ldc] : queue)
#pragma omp for collapse(2) schedule(static) nowait
        for (int i0 = 0; i0 < m; i0 += rows)
            for (int j0 = 0; j0 < n; j0 += 16) {
                const Precision* a[rows];
                for (int r = 0; r < rows; ++r) a[r] = A + (long)std::min(i0 + r, m - 1)*lda;
                const Precision* b = B + (long)j0*k;
                Precision c[rows][16] = {};
                for (int q = 0; q < k; ++q)
                    for (int r = 0; r < rows; ++r)
#pragma omp simd
                        for (int j = 0; j < 16; ++j) c[r][j] += a[r][q]*b[q*16 + j];
                for (int r = 0; r < std::min(rows, m - i0); ++r) std::copy_n(c[r], 16, C + (long)(i0 + r)*ldc + j0);
            }
    queue.clear();
}

template <class Precision>
void MACEFoundation<Precision>::set_weight(Weight& W, const std::vector<Precision>& v,
                                           const std::vector<std::array<long, 3>>& blocks)
{
    W.w.resize(v.size());
    W.tw.resize(v.size());
    for (const auto& [off, rows, cols] : blocks)
        for (long r = 0; r < rows; ++r)
            for (long c = 0; c < cols; ++c) {
                W.w[off + (c/16*rows + r)*16 + c%16] = v[off + r*cols + c];
                W.tw[off + (r/16*cols + c)*16 + r%16] = v[off + r*cols + c];
            }
}
