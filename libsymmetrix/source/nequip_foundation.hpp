#pragma once

#include "nequip_foundation_tables.hpp"

#include "../external/json/single_include/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
using namespace nequip_foundation;

constexpr int V = 8;
constexpr int EB = 8;

void solid_harmonics(double x, double y, double z, double* S)
{
    const double xx = x*x;
    const double yy = y*y;
    const double zz = z*z;
    const double r2 = xx + yy + zz;
    const double xy = x*y;
    const double dxy = xx - yy;
    const double f5 = 5*zz - r2;
    const double s2 = xx + yy;
    const double g7 = 7*zz - r2;
    const double g73 = 7*zz - 3*r2;
    S[0] = 1;
    S[1] = y;
    S[2] = z;
    S[3] = x;
    S[4] = xy;
    S[5] = y*z;
    S[6] = 3*zz - r2;
    S[7] = x*z;
    S[8] = dxy;
    S[9] = y*(3*xx - yy);
    S[10] = xy*z;
    S[11] = y*f5;
    S[12] = z*(5*zz - 3*r2);
    S[13] = x*f5;
    S[14] = z*dxy;
    S[15] = x*(xx - 3*yy);
    S[16] = xy*dxy;
    S[17] = y*z*(3*xx - yy);
    S[18] = xy*g7;
    S[19] = y*z*g73;
    S[20] = 8*zz*(zz - 3*s2) + 3*s2*s2;
    S[21] = x*z*g73;
    S[22] = dxy*g7;
    S[23] = x*z*(xx - 3*yy);
    S[24] = xx*(xx - 6*yy) + yy*yy;
}

template <class Precision> inline Precision sigmoid(Precision x) { return 1/(1 + std::exp(-x)); }

constexpr int iabs(int a) { return a < 0 ? -a : a; }

constexpr bool coupled(int l1, int l2, int l3, int m1, int m2, int m3)
{
    const int a = m1 - l1;
    const int b = m2 - l2;
    const int c = m3 - l3;
    return ((a < 0) != (b < 0)) == (c < 0) and (iabs(c) == iabs(iabs(a) - iabs(b)) or iabs(c) == iabs(a) + iabs(b));
}

constexpr bool coupled(int l1, int l2, int l3, int m1, int m2)
{
    for (int m3 = 0; m3 <= 2*l3; ++m3)
        if (coupled(l1, l2, l3, m1, m2, m3)) return true;
    return false;
}

template <class Precision, class Ins, int L1, int L2, int L3, int M0>
void forward_rows(const Ins& k, int nb, const Precision* Rb, int nf, const Precision* Y, int num_lm, const int* jr,
                  const Precision* X, int xs, Precision* Z)
{
    constexpr int N1 = 2*L1+1;
    constexpr int N2 = 2*L2+1;
    constexpr int MB = N1*N2 <= 9 ? N1 : N1 % 3 == 0 and 3*N2 <= 9 ? 3 : 1;
    const int mul = k.mul;
    for (int c = 0; c < mul; c += V) {
        Precision acc[MB][N2][V];
        for (int b = 0; b < MB; ++b)
            for (int m2 = 0; m2 < N2; ++m2)
#pragma omp simd
                for (int v = 0; v < V; ++v) acc[b][m2][v] = Z[((M0 + b)*N2 + m2)*mul + c + v];
        for (int i = 0; i < nb; ++i) {
            const Precision* r = Rb + (long)i*nf + k.woff + c;
            const Precision* xj = X + (long)jr[i]*xs + M0*mul + c;
            const Precision* ye = Y + (long)i*num_lm + L2*L2;
#pragma GCC unroll 9
            for (int b = 0; b < MB; ++b) {
                Precision rx[V];
#pragma omp simd
                for (int v = 0; v < V; ++v) rx[v] = r[v]*xj[b*mul + v];
#pragma GCC unroll 9
                for (int m2 = 0; m2 < N2; ++m2)
                    if (coupled(L1, L2, L3, M0 + b, m2)) {
#pragma omp simd
                        for (int v = 0; v < V; ++v) acc[b][m2][v] += (N2 == 1 ? 1 : ye[m2])*rx[v];
                    }
            }
        }
        for (int b = 0; b < MB; ++b)
            for (int m2 = 0; m2 < N2; ++m2)
#pragma omp simd
                for (int v = 0; v < V; ++v) Z[((M0 + b)*N2 + m2)*mul + c + v] = acc[b][m2][v];
    }
    if constexpr (M0 + MB < N1) {
        forward_rows<Precision, Ins, L1, L2, L3, M0 + MB>(k, nb, Rb, nf, Y, num_lm, jr, X, xs, Z);
    }
}

template <class Precision, class Ins, int L1, int L2, int L3>
void forward_ins(const Ins& k, int nb, const Precision* Rb, int nf, const Precision* Y, int num_lm, const int* jr,
                 const Precision* X, int xs, Precision* Z)
{
    forward_rows<Precision, Ins, L1, L2, L3, 0>(k, nb, Rb, nf, Y, num_lm, jr, X, xs, Z);
}

template <class Precision, class Ins, int L1, int L2, int L3>
void reverse_ins(const Ins& k, int nb, const Precision* Rb, const Precision* dRb, int nf, const Precision* Y,
                 int num_lm, const int* jr, const Precision* X, int xs, const Precision* B, Precision* gx, int gw,
                 Precision* gS, Precision* dr)
{
    constexpr int N1 = 2*L1+1;
    constexpr int N2 = 2*L2+1;
    const int mul = k.mul;
    for (int i = 0; i < nb; ++i) {
        const Precision* ye = Y + (long)i*num_lm + L2*L2;
        Precision gs[N2][V] = {};
        Precision sr[V] = {};
        for (int c = 0; c < mul; c += V) {
            const Precision* r = Rb + (long)i*nf + k.woff + c;
            const Precision* dr_ = dRb + (long)i*nf + k.woff + c;
            const Precision* xj = X + (long)jr[i]*xs + c;
            Precision* o = gx + (long)i*gw + c;
#pragma GCC unroll 9
            for (int m1 = 0; m1 < N1; ++m1) {
                const Precision* b = B + (long)m1*N2*mul + c;
                Precision g[V] = {};
                Precision x[V];
                Precision rx[V];
#pragma omp simd
                for (int v = 0; v < V; ++v) {
                    x[v] = xj[m1*mul + v];
                    rx[v] = r[v]*x[v];
                }
#pragma GCC unroll 9
                for (int m2 = 0; m2 < N2; ++m2)
                    if (coupled(L1, L2, L3, m1, m2)) {
#pragma omp simd
                        for (int v = 0; v < V; ++v) {
                            g[v] += (N2 == 1 ? 1 : ye[m2])*b[m2*mul + v];
                            gs[m2][v] += rx[v]*b[m2*mul + v];
                        }
                    }
#pragma omp simd
                for (int v = 0; v < V; ++v) {
                    o[m1*mul + v] += r[v]*g[v];
                    sr[v] += dr_[v]*x[v]*g[v];
                }
            }
        }
        if constexpr (L2 > 0) {
            for (int m2 = 0; m2 < N2; ++m2) {
                Precision t = 0;
                for (int v = 0; v < V; ++v) t += gs[m2][v];
                gS[(long)i*num_lm + L2*L2 + m2] += t;
            }
        }
        Precision t = 0;
        for (int v = 0; v < V; ++v) t += sr[v];
        dr[i] += t;
    }
}

template <class Precision, class Ins, int L1, int L2, int L3, class F, class R>
void select_ins(int l1, int l2, int l3, F& forward, R& reverse)
{
    if constexpr ((L1 + L2 + L3) % 2 == 0 and L3 >= iabs(L1 - L2) and L3 <= L1 + L2) {
        if (l1 == L1 and l2 == L2 and l3 == L3) {
            forward = forward_ins<Precision, Ins, L1, L2, L3>;
            reverse = reverse_ins<Precision, Ins, L1, L2, L3>;
        }
    }
    if constexpr (L3 < 4) {
        select_ins<Precision, Ins, L1, L2, L3+1>(l1, l2, l3, forward, reverse);
    } else if constexpr (L2 < 4) {
        select_ins<Precision, Ins, L1, L2+1, 0>(l1, l2, l3, forward, reverse);
    } else if constexpr (L1 < 4) {
        select_ins<Precision, Ins, L1+1, 0, 0>(l1, l2, l3, forward, reverse);
    }
}

}  // namespace

template <class Precision>
class NequIPFoundation {

public:

template <class T> struct Uninit : std::allocator<T> {
    template <class U> struct rebind { using other = Uninit<U>; };
    template <class U, class... A> void construct(U* p, A&&... a) { ::new ((void*)p) U(std::forward<A>(a)...); }
    template <class U> void construct(U* p) { ::new ((void*)p) U; }
};
using Floats = std::vector<Precision, Uninit<Precision>>;
struct Weight { std::vector<Precision> w, tw; };

NequIPFoundation(const std::string& filename, const std::string& prefix);

// Basic model information
int num_elements;
int num_layers;
double r_cut;
int l_max, num_lm;
std::vector<int> atomic_numbers;

// Graph
int N = 0;
int NR = 0;
std::vector<std::array<int, 3>> segments;
std::vector<int> atom_of_node, elem, elem_all, row_of_atom, order, start, neigh_indices, neigh_rows, neigh_elem;
std::vector<double> xyz, r;
Floats edge_forces;
std::unique_ptr<std::atomic_flag[]> locks;

// Node energies and forces
std::vector<double> node_energies;
void compute_layer(int);
void compute_readout();
void reverse_layer(int);

// ZBL
bool has_zbl;
double zbl_p;
std::vector<double> zbl;

// Radial functions
int K;
double h;
std::vector<int> interval;
Floats xs, r_adj;

// Spherical harmonics
Floats Y, Y_adj;

// Layers
struct Instruction;
using Forward = void (*)(const Instruction&, int, const Precision*, int, const Precision*, int, const int*,
                         const Precision*, int, Precision*);
using Reverse = void (*)(const Instruction&, int, const Precision*, const Precision*, int, const Precision*, int,
                         const int*, const Precision*, int, const Precision*, Precision*, int, Precision*, Precision*);
struct Term { int z, m3; Precision c; };
struct Instruction {
    int l1, l2, l3, mul, woff, col, zoff;
    std::vector<Term> terms;
    Forward forward;
    Reverse reverse;
};
struct Layer {
    std::vector<int> H_mul, A_mul, Phi_mul;
    std::vector<long> H_offset, A_offset, Phi_offset;
    int H_width, A_width, Phi_width, z_width;
    int num_scalars, num_gates, num_functions;
    std::vector<Instruction> instructions;
    std::vector<Precision> R;
    std::vector<Weight> X_weights, A_weights;
    std::vector<std::vector<Weight>> A_sc_weights;
};
std::vector<Layer> layers;
std::vector<Precision> ones;
std::vector<Floats> X, A;
Floats H, Phi, H_adj, X_adj, A_adj, Phi_adj;

// Gates
Precision gate_cst;

// Readout
std::vector<Precision> readout_weights;
std::vector<double> per_type_scale, per_type_shift;

// Matrix products
struct Product { int m, n, k; const Precision* a; int lda; const Precision* b; Precision* c; int ldc; bool add; };
std::vector<Product> queue;
void gemm(bool, int, int, int, const Precision*, int, const Weight&, Precision*, int, bool);
void products();
void set_weight(Weight&, const double*, long, long);

// Initializer
void load(const std::string&, const std::string&);

};

template <class Precision>
NequIPFoundation<Precision>::NequIPFoundation(const std::string& filename, const std::string& prefix)
{
    load(filename, prefix);
}

template <class Precision>
void NequIPFoundation<Precision>::compute_layer(const int t)
{
    const Layer& L = layers[t];
    const bool first = t == 0;
    const long E = start[N];

    // Y
    // Y_{lm}(\hat{\mathbf{r}}_{ij})
    if (first) {
        interval.resize(E);
        xs.resize(E);
        Y.resize(E*num_lm);
#pragma omp parallel for schedule(static)
        for (long e = 0; e < E; ++e) {
            const double dx = xyz[3*e];
            const double dy = xyz[3*e+1];
            const double dz = xyz[3*e+2];
            const double r_ij = r[e];
            const int k = std::min(int(r_ij/h), K - 1);
            interval[e] = k;
            xs[e] = Precision(r_ij/h - k);
            double S[25];
            solid_harmonics(dz/r_ij, dx/r_ij, dy/r_ij, S);
            for (int lm = 0; lm < num_lm; ++lm) Y[e*num_lm + lm] = Precision(sh_c[lm]*S[lm]);
        }
    }

    // Phi
    // \Phi^{(t)}_{i,k\lambda m\eta} = \sum_j \sum_{m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) X^{(t)}_{j,k\lambda_1m_1},  \eta = (\lambda_1, l_2)
    // \Phi^{(0)}_{i,k\lambda m\eta} = \sum_j R^{(0)}_{a_j,k\lambda\eta}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij}),  R^{(0)}_{a_j,k\lambda\eta} = c_\eta R^{(0)}_{k\lambda\eta} X^{(0)}_{a_j,k},  C^{\lambda m}_{(0e)0lm_2} = c_\eta \delta_{mm_2}
    {
        Phi.resize((long)N*L.Phi_width);
        const int nf = L.num_functions;
        const long species_stride = first ? (long)K*nf*4 : 0;
#pragma omp parallel
        {
            std::vector<Precision> Rb((long)EB*nf);
            std::vector<Precision> Z(L.z_width);
#pragma omp for schedule(dynamic, 1)
            for (int n = 0; n < N; ++n) {
                const int e0 = start[n];
                const int ne = start[n + 1] - e0;
                std::fill(Z.begin(), Z.end(), Precision(0));
                for (int b0 = 0; b0 < ne; b0 += EB) {
                    const int nb = std::min(EB, ne - b0);
                    for (int i = 0; i < nb; ++i) {
                        const long e = e0 + b0 + i;
                        const Precision* row = L.R.data() + neigh_elem[e]*species_stride + (long)interval[e]*nf*4;
                        const Precision t = xs[e];
                        for (int f = 0; f < nf; f += V) {
                            const Precision* q = row + 4*f;
#pragma omp simd
                            for (int v = 0; v < V; ++v)
                                Rb[(long)i*nf + f + v] = q[v] + t*(q[V+v] + t*(q[2*V+v] + t*q[3*V+v]));
                        }
                    }
                    for (const auto& k : L.instructions)
                        k.forward(k, nb, Rb.data(), nf, Y.data() + (e0 + b0)*(long)num_lm, num_lm,
                                  neigh_rows.data() + e0 + b0, first ? ones.data() : X[t].data() + NR*L.H_offset[k.l1],
                                  first ? 0 : (2*k.l1+1)*k.mul, Z.data() + k.zoff);
                }
                for (const auto& k : L.instructions) {
                    Precision* out = Phi.data() + N*L.Phi_offset[k.l3] + (long)n*(2*k.l3+1)*L.Phi_mul[k.l3] + k.col;
                    for (int m3 = 0; m3 < 2*k.l3+1; ++m3)
                        std::fill_n(out + (long)m3*L.Phi_mul[k.l3], k.mul, Precision(0));
                    for (const auto& t : k.terms) {
                        const Precision* z = Z.data() + k.zoff + (long)t.z*k.mul;
                        Precision* o = out + (long)t.m3*L.Phi_mul[k.l3];
#pragma omp simd
                        for (int u = 0; u < k.mul; ++u) o[u] += t.c*z[u];
                    }
                }
            }
        }
    }

    // A
    // A^{(t)}_{i,k\lambda m} = \sum_{k'\eta} w^{(t)}_{k\lambda,k'\eta} \Phi^{(t)}_{i,k'\lambda m\eta} + \sum_{k'} w^{(t)}_{a_ik\lambda,k'} H^{(t)}_{i,k'\lambda m}
    {
        A[t].resize((long)N*L.A_width);
        for (int l = 0; l <= l_max; ++l)
            if (L.A_mul[l] > 0)
                gemm(false, N*(2*l+1), L.A_mul[l], L.Phi_mul[l], Phi.data() + N*L.Phi_offset[l], L.Phi_mul[l],
                     L.A_weights[l], A[t].data() + N*L.A_offset[l], L.A_mul[l], false);
        products();
        if (not L.A_sc_weights.empty()) {
            for (const auto& [n0, n1, a] : segments)
                for (int l = 0; l <= l_max; ++l)
                    if (not L.A_sc_weights[l].empty())
                        gemm(false, (n1 - n0)*(2*l+1), L.A_mul[l], L.H_mul[l],
                             H.data() + N*L.H_offset[l] + (long)n0*(2*l+1)*L.H_mul[l], L.H_mul[l], L.A_sc_weights[l][a],
                             A[t].data() + N*L.A_offset[l] + (long)n0*(2*l+1)*L.A_mul[l], L.A_mul[l], true);
            products();
        }
    }

    // gate
    // H^{(t+1)}_{i,k(0e)} = c_e \mathrm{SiLU}(A^{(t)}_{i,k(0e)})
    // H^{(t+1)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) A^{(t)}_{i,k\lambda m} (l > 0),  g_{k\lambda} = N^{(t+1)}_{0e} + \sum_{0o < \lambda' < \lambda} N^{(t+1)}_{\lambda'} + k
    if (t + 1 < num_layers) {
        const Layer& Ln = layers[t + 1];
        H.resize((long)N*Ln.H_width);
        const Precision* yq = A[t].data();
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const Precision* y0 = yq + (long)n*L.A_mul[0];
            Precision* x0 = H.data() + (long)n*Ln.H_mul[0];
            for (int k = 0; k < L.num_scalars; ++k) x0[k] = gate_cst*y0[k]*sigmoid(y0[k]);
            int g = L.num_scalars;
            for (int l = 1; l <= l_max; ++l)
                for (int c = 0; c < L.A_mul[l]; ++c) {
                    const Precision s = gate_cst*y0[g]*sigmoid(y0[g]);
                    ++g;
                    for (int m = 0; m < 2*l+1; ++m)
                        H[N*Ln.H_offset[l] + ((long)n*(2*l+1) + m)*Ln.H_mul[l] + c]
                            = yq[N*L.A_offset[l] + ((long)n*(2*l+1) + m)*L.A_mul[l] + c]*s;
                }
        }
    }

    // X
    // X^{(t+1)}_{i,k\lambda m} = \sum_{k'} w^{(t+1)}_{k\lambda,k'} H^{(t+1)}_{i,k'\lambda m}
    if (t + 1 < num_layers) {
        const Layer& Ln = layers[t + 1];
        X[t + 1].resize((long)NR*Ln.H_width);
        for (int l = 0; l <= l_max; ++l)
            gemm(false, N*(2*l+1), Ln.H_mul[l], Ln.H_mul[l], H.data() + N*Ln.H_offset[l], Ln.H_mul[l], Ln.X_weights[l],
                 X[t + 1].data() + NR*Ln.H_offset[l], Ln.H_mul[l], false);
        products();
    }
}

template <class Precision>
void NequIPFoundation<Precision>::compute_readout()
{
    const long E = start[N];

    // ZBL
    // E_i = \frac{1}{2} \sum_j V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij}),  \mathbf{F}_{ij} = -\frac{1}{2} \nabla_{\mathbf{r}_{ij}} V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij})
    {
        node_energies.resize(N);
        edge_forces.resize(3*E);
        r_adj.resize(E);
        Y_adj.resize(E*num_lm);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const int a = elem[n];
            double ez = 0;
            for (long e = start[n]; e < start[n + 1]; ++e) {
                const double dx = xyz[3*e];
                const double dy = xyz[3*e+1];
                const double dz = xyz[3*e+2];
                const double r_ij = r[e];
                double f = 0;
                if (has_zbl) {
                    const double* z = zbl.data() + 2*(a*num_elements + neigh_elem[e]);
                    const double r2 = dx*dx + dy*dy + dz*dz;
                    const double xz = z[0]*r_ij;
                    const double e1 = 0.02817*std::exp(-0.20162*xz);
                    const double e2 = 0.28022*std::exp(-0.40290*xz);
                    const double e3 = 0.50986*std::exp(-0.94229*xz);
                    const double e4 = 0.18175*std::exp(-3.19980*xz);
                    const double psi = e1 + e2 + e3 + e4;
                    const double dpsi = z[0]*(-0.20162*e1 - 0.40290*e2 - 0.94229*e3 - 3.19980*e4);
                    const double p = zbl_p;
                    const double u = r_ij/r_cut;
                    const double up = std::pow(u, p);
                    const double fc = 1 - (p+1)*(p+2)/2*up + p*(p+2)*up*u - p*(p+1)/2*up*u*u;
                    const double dfc = (-(p+1)*(p+2)/2*p*up/u + p*(p+2)*(p+1)*up - p*(p+1)/2*(p+2)*up*u)/r_cut;
                    ez += z[1]/r_ij*psi*fc;
                    f = -z[1]*(-psi*fc/r2 + dpsi*fc/r_ij + psi*dfc/r_ij)/r_ij;
                }
                edge_forces[3*e] = Precision(f*dx);
                edge_forces[3*e+1] = Precision(f*dy);
                edge_forces[3*e+2] = Precision(f*dz);
                r_adj[e] = 0;
                std::fill_n(Y_adj.data() + e*num_lm, num_lm, Precision(0));
            }
            node_energies[n] = ez;
        }
    }

    // readout
    // E_i += \mu_{a_i} + \sigma_{a_i} \sum_k w_k H^{(T)}_{i,k},  H^{(T)}_{i,k} = c_e \mathrm{SiLU}(A^{(T-1)}_{i,k(0e)})
    // \bar{A}^{(T-1)}_{i,k(0e)} = c_e \mathrm{SiLU}'(A^{(T-1)}_{i,k(0e)}) \sigma_{a_i} w_k
    {
        const Layer& L = layers[num_layers - 1];
        const int nro = L.A_mul[0];
        const Precision* yl = A[num_layers - 1].data();
        A_adj.resize((long)N*L.A_width);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const int a = elem[n];
            double e = 0;
            for (int k = 0; k < nro; ++k) {
                const Precision yk = yl[(long)n*nro + k];
                const Precision s = sigmoid(yk);
                e += readout_weights[k]*yk*s;
                A_adj[(long)n*nro + k] = Precision(per_type_scale[a])*readout_weights[k]*gate_cst*s*(1 + yk*(1 - s));
            }
            node_energies[n] += per_type_scale[a]*gate_cst*e + per_type_shift[a];
        }
    }
}

template <class Precision>
void NequIPFoundation<Precision>::reverse_layer(const int t)
{
    const Layer& L = layers[t];
    const bool first = t == 0;

    // X
    // \bar{H}^{(t+1)}_{i,k'\lambda m} += \sum_k w^{(t+1)}_{k\lambda,k'} \bar{X}^{(t+1)}_{i,k\lambda m}
    if (t + 1 < num_layers) {
        const Layer& Ln = layers[t + 1];
        for (int l = 0; l <= l_max; ++l)
            gemm(true, N*(2*l+1), Ln.H_mul[l], Ln.H_mul[l], X_adj.data() + NR*Ln.H_offset[l], Ln.H_mul[l],
                 Ln.X_weights[l], H_adj.data() + N*Ln.H_offset[l], Ln.H_mul[l],
                 not Ln.A_sc_weights.empty() and not Ln.A_sc_weights[l].empty());
        products();
    }

    // gate
    // \bar{A}^{(t)}_{i,k(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,k(0e)}) \bar{H}^{(t+1)}_{i,k(0e)}
    // \bar{A}^{(t)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) \bar{H}^{(t+1)}_{i,k\lambda m},  \bar{A}^{(t)}_{i,g_{k\lambda}(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,g_{k\lambda}(0e)}) \sum_m A^{(t)}_{i,k\lambda m} \bar{H}^{(t+1)}_{i,k\lambda m}
    if (t + 1 < num_layers) {
        const Layer& Ln = layers[t + 1];
        A_adj.resize((long)N*L.A_width);
        const Precision* yp = A[t].data();
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const Precision* y0 = yp + (long)n*L.A_mul[0];
            Precision* ya0 = A_adj.data() + (long)n*L.A_mul[0];
            const Precision* xa0 = H_adj.data() + (long)n*Ln.H_mul[0];
            for (int k = 0; k < L.num_scalars; ++k) {
                const Precision s = sigmoid(y0[k]);
                ya0[k] = xa0[k]*gate_cst*s*(1 + y0[k]*(1 - s));
            }
            int g = L.num_scalars;
            for (int l = 1; l <= l_max; ++l)
                for (int c = 0; c < L.A_mul[l]; ++c) {
                    const Precision s = sigmoid(y0[g]);
                    const Precision gv = gate_cst*y0[g]*s;
                    Precision ga = 0;
                    for (int m = 0; m < 2*l+1; ++m) {
                        const long iy = N*L.A_offset[l] + ((long)n*(2*l+1) + m)*L.A_mul[l] + c;
                        const Precision xa = H_adj[N*Ln.H_offset[l] + ((long)n*(2*l+1) + m)*Ln.H_mul[l] + c];
                        A_adj[iy] = xa*gv;
                        ga += xa*yp[iy];
                    }
                    ya0[g] = ga*gate_cst*s*(1 + y0[g]*(1 - s));
                    ++g;
                }
        }
    }

    // A
    // \bar{\Phi}^{(t)}_{i,k'\lambda m\eta} = \sum_k w^{(t)}_{k\lambda,k'\eta} \bar{A}^{(t)}_{i,k\lambda m},  \bar{H}^{(t)}_{i,k'\lambda m} = \sum_k w^{(t)}_{a_ik\lambda,k'} \bar{A}^{(t)}_{i,k\lambda m} (t > 0)
    {
        Phi_adj.resize((long)N*L.Phi_width);
        for (int l = 0; l <= l_max; ++l)
            if (L.A_mul[l] > 0)
                gemm(true, N*(2*l+1), L.Phi_mul[l], L.A_mul[l], A_adj.data() + N*L.A_offset[l], L.A_mul[l],
                     L.A_weights[l], Phi_adj.data() + N*L.Phi_offset[l], L.Phi_mul[l], false);
        if (not first) {
            H_adj.resize((long)N*L.H_width);
            for (int l = 0; l <= l_max; ++l) {
                if (L.A_sc_weights.empty() or L.A_sc_weights[l].empty()) continue;
                for (const auto& [n0, n1, a] : segments)
                    gemm(true, (n1 - n0)*(2*l+1), L.H_mul[l], L.A_mul[l],
                         A_adj.data() + N*L.A_offset[l] + (long)n0*(2*l+1)*L.A_mul[l], L.A_mul[l], L.A_sc_weights[l][a],
                         H_adj.data() + N*L.H_offset[l] + (long)n0*(2*l+1)*L.H_mul[l], L.H_mul[l], false);
            }
        }
        products();
    }

    // Phi
    // \bar{X}^{(t)}_{j,k\lambda_1m_1} = \sum_i \sum_{\lambda m\eta m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(t)}_{i,k\lambda m\eta} (t > 0)
    // \bar{r}_{ij} += \sum_{k\lambda m\eta m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)\prime}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) X^{(t)}_{j,k\lambda_1m_1} \bar{\Phi}^{(t)}_{i,k\lambda m\eta}
    // \bar{Y}_{ij,l_2m_2} += \sum_{k\lambda m\eta m_1} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) X^{(t)}_{j,k\lambda_1m_1} \bar{\Phi}^{(t)}_{i,k\lambda m\eta} (l_2 > 0)
    {
        if (not first) {
            X_adj.resize((long)NR*L.H_width);
#pragma omp parallel for schedule(static)
            for (int r = 0; r < NR; ++r)
                for (int l = 0; l <= l_max; ++l)
                    std::fill_n(X_adj.data() + NR*L.H_offset[l] + (long)r*(2*l+1)*L.H_mul[l],
                                (2*l+1)*L.H_mul[l], Precision(0));
        }
        const Precision inv_h = Precision(1/h);
        const int nf = L.num_functions;
        const long species_stride = first ? (long)K*nf*4 : 0;
        const int gw = std::max(L.H_width, nf);
#pragma omp parallel
        {
            std::vector<Precision> Rb((long)EB*nf);
            std::vector<Precision> dRb((long)EB*nf);
            std::vector<Precision> B(L.z_width);
            std::vector<Precision> gx((long)EB*gw);
            std::vector<Precision> dr(EB);
#pragma omp for schedule(static)
            for (int ii = 0; ii < N; ++ii) {
                const int n = order[ii];
                const int e0 = start[n];
                const int ne = start[n + 1] - e0;
                for (const auto& k : L.instructions) {
                    const Precision* a = Phi_adj.data() + N*L.Phi_offset[k.l3]
                                         + (long)n*(2*k.l3+1)*L.Phi_mul[k.l3] + k.col;
                    Precision* b = B.data() + k.zoff;
                    std::fill_n(b, (long)(2*k.l1+1)*(2*k.l2+1)*k.mul, Precision(0));
                    for (const auto& t : k.terms) {
                        const Precision* at = a + (long)t.m3*L.Phi_mul[k.l3];
                        Precision* bt = b + (long)t.z*k.mul;
#pragma omp simd
                        for (int u = 0; u < k.mul; ++u) bt[u] += t.c*at[u];
                    }
                }
                for (int b0 = 0; b0 < ne; b0 += EB) {
                    const int nb = std::min(EB, ne - b0);
                    for (int i = 0; i < nb; ++i) {
                        const long e = e0 + b0 + i;
                        const Precision* row = L.R.data() + neigh_elem[e]*species_stride + (long)interval[e]*nf*4;
                        const Precision t = xs[e];
                        for (int f = 0; f < nf; f += V) {
                            const Precision* q = row + 4*f;
#pragma omp simd
                            for (int v = 0; v < V; ++v) {
                                Rb[(long)i*nf + f + v] = q[v] + t*(q[V+v] + t*(q[2*V+v] + t*q[3*V+v]));
                                dRb[(long)i*nf + f + v] = q[V+v] + t*(2*q[2*V+v] + 3*t*q[3*V+v]);
                            }
                        }
                    }
                    std::fill_n(gx.data(), (long)nb*gw, Precision(0));
                    std::fill_n(dr.data(), nb, Precision(0));
                    for (const auto& k : L.instructions)
                        k.reverse(k, nb, Rb.data(), dRb.data(), nf, Y.data() + (e0 + b0)*(long)num_lm, num_lm,
                                  neigh_rows.data() + e0 + b0, first ? ones.data() : X[t].data() + NR*L.H_offset[k.l1],
                                  first ? 0 : (2*k.l1+1)*k.mul, B.data() + k.zoff,
                                  gx.data() + (first ? 0 : L.H_offset[k.l1]), gw,
                                  Y_adj.data() + (e0 + b0)*(long)num_lm, dr.data());
                    for (int i = 0; i < nb; ++i) r_adj[e0 + b0 + i] += dr[i]*inv_h;
                    if (first) continue;
                    for (int i = 0; i < nb; ++i) {
                        const int jr = neigh_rows[e0 + b0 + i];
                        while (locks[jr].test_and_set(std::memory_order_acquire)) {}
                        for (int l = 0; l <= l_max; ++l) {
                            const int w = (2*l+1)*L.H_mul[l];
                            Precision* dst = X_adj.data() + NR*L.H_offset[l] + (long)jr*w;
                            const Precision* src = gx.data() + (long)i*gw + L.H_offset[l];
                            for (int z = 0; z < w; ++z) dst[z] += src[z];
                        }
                        locks[jr].clear(std::memory_order_release);
                    }
                }
            }
        }
    }

    // Y
    // \mathbf{F}_{ij} -= \bar{r}_{ij} \hat{\mathbf{r}}_{ij} + \sum_{lm} \bar{Y}_{ij,lm} \nabla_{\mathbf{r}_{ij}} Y_{lm}(\hat{\mathbf{r}}_{ij})
    if (first) {
        const long E = start[N];
#pragma omp parallel for schedule(static)
        for (long e = 0; e < E; ++e) {
            const double r_ij = r[e];
            const double u[3] = {xyz[3*e]/r_ij, xyz[3*e+1]/r_ij, xyz[3*e+2]/r_ij};
            double S[25];
            double G[25][3] = {};
            solid_harmonics(u[2], u[0], u[1], S);
            for (const auto& t : grad_terms)
                if (t.lm < num_lm) G[t.lm][t.d] += t.c*sh_c[t.mp]*S[t.mp];
            double g[3] = {r_adj[e]*u[0], r_adj[e]*u[1], r_adj[e]*u[2]};
            for (int lm = 1; lm < num_lm; ++lm) {
                const int l = lm < 4 ? 1 : lm < 9 ? 2 : lm < 16 ? 3 : 4;
                const double yl = l*sh_c[lm]*S[lm];
                const double w = Y_adj[e*num_lm + lm]/r_ij;
                g[0] += w*(G[lm][1] - yl*u[0]);
                g[1] += w*(G[lm][2] - yl*u[1]);
                g[2] += w*(G[lm][0] - yl*u[2]);
            }
            edge_forces[3*e] -= Precision(g[0]);
            edge_forces[3*e+1] -= Precision(g[1]);
            edge_forces[3*e+2] -= Precision(g[2]);
        }
    }
}

template <class Precision>
void NequIPFoundation<Precision>::load(const std::string& filename, const std::string& prefix)
{
    std::ifstream file(filename);
    if (not file) throw std::runtime_error(prefix + "cannot open " + filename);
    const json js = json::parse(file);
    if (js.value("format_version", 0) != 3)
        throw std::runtime_error(prefix + filename + " was extracted by an incompatible symmetrix version; "
                                                     "extract it again");
    if (js.value("model_type", "") != "NequIP") throw std::runtime_error(prefix + filename + " is not a NequIP model");
    num_elements = js.at("num_elements");
    num_layers = js.at("num_layers");
    atomic_numbers = js.at("atomic_numbers").get<std::vector<int>>();
    r_cut = js.at("r_cut");
    gate_cst = Precision(js.at("gate_cst").get<double>());
    per_type_scale = js.at("per_type_scale").get<std::vector<double>>();
    per_type_shift = js.at("per_type_shift").get<std::vector<double>>();
    for (double w : js.at("readout_weights").get<std::vector<double>>()) readout_weights.push_back(Precision(w));
    const auto embedding = js.at("embedding").get<std::vector<std::vector<double>>>();
    const int T = num_elements;

    has_zbl = js.at("has_zbl");
    zbl.assign(2*T*T, 0.0);
    if (has_zbl) {
        zbl_p = js.at("zbl_p");
        const double prefactor = js.at("zbl_prefactor");
        for (int a = 0; a < T; ++a)
            for (int b = 0; b < T; ++b) {
                zbl[2*(a*T + b)] = (std::pow(atomic_numbers[a], 0.23) + std::pow(atomic_numbers[b], 0.23))/0.46850;
                zbl[2*(a*T + b) + 1] = prefactor*atomic_numbers[a]*atomic_numbers[b];
            }
    }

    l_max = js.at("l_max");
    if (l_max > 4) throw std::runtime_error(prefix + "supports degrees up to 4");
    num_lm = (l_max + 1)*(l_max + 1);
    const auto H_mul = js.at("H_mul").get<std::vector<std::vector<int>>>();
    const auto H_offset = js.at("H_offset").get<std::vector<std::vector<int>>>();
    const auto A_mul = js.at("A_mul").get<std::vector<std::vector<int>>>();
    const auto Phi_mul = js.at("Phi_mul").get<std::vector<std::vector<int>>>();
    const auto Phi_offset = js.at("Phi_offset").get<std::vector<std::vector<int>>>();
    const auto spl_values = js.at("radial_spline_values").get<std::vector<std::vector<std::vector<double>>>>();
    const auto spl_derivs = js.at("radial_spline_derivs").get<std::vector<std::vector<std::vector<double>>>>();
    K = int(spl_values[0][0].size()) - 1;
    h = r_cut/K;

    layers.assign(num_layers, Layer());
    X.resize(num_layers);
    A.resize(num_layers);
    for (int q = 0; q < num_layers; ++q) {
        Layer& L = layers[q];
        const bool first = q == 0;
        L.H_mul.assign(l_max + 1, 0);
        L.A_mul.assign(l_max + 1, 0);
        L.Phi_mul.assign(l_max + 1, 0);
        L.num_gates = 0;
        for (int l = 0; l <= l_max; ++l) {
            if (H_mul[q][2*l + 1] > 0 or A_mul[q][2*l + 1] > 0 or Phi_mul[q][2*l + 1] > 0)
                throw std::runtime_error(prefix + "odd irreps not supported");
            L.H_mul[l] = H_mul[q][2*l];
            L.A_mul[l] = A_mul[q][2*l];
            if (l > 0) L.num_gates += L.A_mul[l];
        }
        L.num_scalars = L.A_mul[0] - L.num_gates;

        const auto spec = js.at("Phi_spec")[q].get<std::vector<std::array<int, 5>>>();
        const auto cg = js.at("Phi_clebsch_gordan")[q].get<std::vector<double>>();
        std::map<int, Instruction> by_w;
        std::map<int, std::vector<std::tuple<int, int, int, double>>> C;
        for (int p = 0; p < (int)spec.size(); ++p) {
            const auto [a, lm, w, b, n] = spec[p];
            int i1 = 0;
            while (H_offset[q][i1 + 1] <= a) ++i1;
            int i3 = 0;
            while (Phi_offset[q][i3 + 1] <= b) ++i3;
            Instruction& ins = by_w[w];
            ins.l1 = i1/2;
            ins.l2 = int(std::sqrt(lm));
            ins.l3 = i3/2;
            ins.mul = n;
            ins.woff = w;
            ins.col = (b - Phi_offset[q][i3]) % Phi_mul[q][i3];
            C[w].push_back({(a - H_offset[q][i1])/H_mul[q][i1], lm - ins.l2*ins.l2,
                            (b - Phi_offset[q][i3])/Phi_mul[q][i3], cg[p]});
        }
        std::vector<double> path;
        L.num_functions = 0;
        for (auto& [w, ins] : by_w) {
            if (ins.mul % V != 0 or ins.woff % V != 0 or ins.mul != (first ? ins.mul : L.H_mul[ins.l1]))
                throw std::runtime_error(prefix + "unsupported tensor-product instruction in layer "
                                         + std::to_string(q));
            L.Phi_mul[ins.l3] += ins.mul;
            L.num_functions += ins.mul;
            const int n2 = 2*ins.l2 + 1;
            std::sort(C[w].begin(), C[w].end());
            double cmax = 0;
            for (const auto& [m1, m2, m3, c] : C[w]) cmax = std::max(cmax, std::abs(c));
            path.push_back(std::get<3>(C[w][0]));
            if (first and (ins.l1 != 0 or ins.l2 != ins.l3 or (int)C[w].size() != n2))
                throw std::runtime_error(prefix + "unexpected first-layer tensor product");
            for (const auto& [m1, m2, m3, c] : C[w]) {
                if (first) {
                    if (m2 != m3 or std::abs(c - path.back()) > 1e-6*cmax)
                        throw std::runtime_error(prefix + "unexpected first-layer tensor product");
                    ins.terms.push_back({m2, m3, Precision(1)});
                } else if (std::abs(c) > 1e-9*cmax) {
                    if (not coupled(ins.l1, ins.l2, ins.l3, m1, m2, m3))
                        throw std::runtime_error(prefix + "unexpected tensor-product coefficients");
                    ins.terms.push_back({m1*n2 + m2, m3, Precision(c)});
                }
            }
            ins.forward = nullptr;
            select_ins<Precision, Instruction, 0, 0, 0>(ins.l1, ins.l2, ins.l3, ins.forward, ins.reverse);
            if (not ins.forward)
                throw std::runtime_error(prefix + "unsupported tensor-product instruction (" + std::to_string(ins.l1)
                                         + ", " + std::to_string(ins.l2) + ", " + std::to_string(ins.l3) + ")");
            L.instructions.push_back(ins);
        }
        L.z_width = 0;
        for (auto& ins : L.instructions) {
            ins.zoff = L.z_width;
            L.z_width += (2*ins.l1+1)*(2*ins.l2+1)*ins.mul;
        }

        L.H_offset.assign(l_max + 1, 0);
        L.A_offset.assign(l_max + 1, 0);
        L.Phi_offset.assign(l_max + 1, 0);
        L.H_width = 0;
        L.A_width = 0;
        L.Phi_width = 0;
        for (int l = 0; l <= l_max; ++l) {
            L.H_offset[l] = L.H_width;
            L.A_offset[l] = L.A_width;
            L.Phi_offset[l] = L.Phi_width;
            L.H_width += (2*l+1)*L.H_mul[l];
            L.A_width += (2*l+1)*L.A_mul[l];
            L.Phi_width += (2*l+1)*L.Phi_mul[l];
        }

        {
            const auto& values = spl_values[q];
            const auto& derivs = spl_derivs[q];
            const int nf = L.num_functions;
            const int nb = first ? T : 1;
            if ((int)values.size() != nf or (int)values[0].size() != K + 1)
                throw std::runtime_error(prefix + "radial functions of layer " + std::to_string(q)
                                         + " do not match its instructions");
            std::vector<std::vector<double>> x10(T, std::vector<double>(embedding[0].size(), 0.0));
            if (first) {
                const auto w = js.at("X_weights")[0][0].get<std::vector<double>>();
                const int cols = H_mul[0][0];
                for (int b = 0; b < T; ++b)
                    for (int u = 0; u < cols; ++u)
                        for (int v = 0; v < cols; ++v) x10[b][u] += embedding[b][v]*w[v*cols + u];
            }
            L.R.resize((long)nb*K*nf*4);
            for (int b = 0; b < nb; ++b)
                for (int k = 0; k < (int)L.instructions.size(); ++k) {
                    const auto& ins = L.instructions[k];
                    for (int u = 0; u < ins.mul; ++u) {
                        const int f = ins.woff + u;
                        const double fac = (first ? path[k]*x10[b][u] : 1.0)*(ins.l2 == 0 ? sh_c[0] : 1.0);
                        for (int n = 0; n < K; ++n) {
                            const double v0 = values[f][n];
                            const double v1 = values[f][n+1];
                            const double d0 = h*derivs[f][n];
                            const double d1 = h*derivs[f][n+1];
                            Precision* o = L.R.data() + (((long)b*K + n)*nf + f/V*V)*4 + f%V;
                            o[0] = Precision(fac*v0);
                            o[V] = Precision(fac*d0);
                            o[2*V] = Precision(fac*(3*(v1 - v0) - 2*d0 - d1));
                            o[3*V] = Precision(fac*(2*(v0 - v1) + d0 + d1));
                        }
                    }
                }
            if (first) ones.assign(nf, Precision(1));
        }

        {
            L.X_weights.resize(l_max + 1);
            L.A_weights.resize(l_max + 1);
            const auto x_weights = js.at("X_weights")[q].get<std::vector<std::vector<double>>>();
            const auto a_weights = js.at("A_weights")[q].get<std::vector<std::vector<double>>>();
            const auto a_sc_weights = js.at("A_sc_weights")[q].get<std::vector<std::vector<double>>>();
            for (int l = 0; l <= l_max; ++l) {
                if (not first and L.H_mul[l] > 0) {
                    if ((int)x_weights[2*l].size() != L.H_mul[l]*L.H_mul[l])
                        throw std::runtime_error(prefix + "linear_1 " + std::to_string(l) + " of layer "
                                                 + std::to_string(q) + " does not match");
                    set_weight(L.X_weights[l], x_weights[2*l].data(), L.H_mul[l], L.H_mul[l]);
                }
                if (L.A_mul[l] > 0) {
                    if ((int)a_weights[2*l].size() != L.Phi_mul[l]*L.A_mul[l])
                        throw std::runtime_error(prefix + "linear_2 " + std::to_string(l) + " of layer "
                                                 + std::to_string(q) + " does not match");
                    set_weight(L.A_weights[l], a_weights[2*l].data(), L.Phi_mul[l], L.A_mul[l]);
                }
                if (a_sc_weights[2*l].size() > 0) {
                    if (first) throw std::runtime_error(prefix + "self-connection in the first layer not supported");
                    L.A_sc_weights.resize(l_max + 1);
                    const long n = (long)L.H_mul[l]*L.A_mul[l];
                    if ((long)a_sc_weights[2*l].size() != T*n)
                        throw std::runtime_error(prefix + "self-connection " + std::to_string(l) + " of layer "
                                                 + std::to_string(q) + " does not match");
                    for (int t = 0; t < T; ++t)
                        set_weight(L.A_sc_weights[l].emplace_back(), a_sc_weights[2*l].data() + t*n,
                                   L.H_mul[l], L.A_mul[l]);
                }
            }
        }
    }

    for (int q = 0; q < num_layers; ++q) {
        const Layer& L = layers[q];
        bool ok = true;
        if (q < num_layers - 1) {
            const Layer& Ln = layers[q + 1];
            int gates = 0;
            for (int l = 1; l <= l_max; ++l) {
                gates += L.A_mul[l];
                ok = ok and Ln.H_mul[l] == L.A_mul[l];
            }
            ok = ok and gates == L.num_gates and Ln.H_mul[0] == L.num_scalars
                 and L.A_mul[0] == L.num_scalars + L.num_gates;
        } else {
            for (int l = 1; l <= l_max; ++l) ok = ok and L.A_mul[l] == 0;
            ok = ok and (int)readout_weights.size() == L.A_mul[0];
        }
        if (not ok) throw std::runtime_error(prefix + "the gates of layer " + std::to_string(q) + " do not match");
    }
}

template <class Precision>
void NequIPFoundation<Precision>::products()
{
#pragma omp parallel
    for (const auto& [m, n, k, A, lda, B, C, ldc, add] : queue)
#pragma omp for collapse(2) schedule(static) nowait
        for (int i0 = 0; i0 < m; i0 += 6)
            for (int j0 = 0; j0 < n; j0 += 16) {
                const Precision* a[6];
                for (int r = 0; r < 6; ++r) a[r] = A + (long)std::min(i0 + r, m - 1)*lda;
                const Precision* b = B + (long)j0*k;
                Precision c[6][16] = {};
                for (int q = 0; q < k; ++q)
                    for (int r = 0; r < 6; ++r)
#pragma omp simd
                        for (int j = 0; j < 16; ++j) c[r][j] += a[r][q]*b[q*16 + j];
                for (int r = 0; r < std::min(6, m - i0); ++r) {
                    Precision* o = C + (long)(i0 + r)*ldc + j0;
                    if (add)
                        for (int j = 0; j < 16; ++j) o[j] += c[r][j];
                    else
                        std::copy_n(c[r], 16, o);
                }
            }
    queue.clear();
}

template <class Precision>
void NequIPFoundation<Precision>::set_weight(Weight& W, const double* v, long rows, long cols)
{
    if (rows % 16 != 0 or cols % 16 != 0)
        throw std::runtime_error("NequIPFoundation: weight shapes must be multiples of 16");
    W.w.resize(rows*cols);
    W.tw.resize(rows*cols);
    for (long r = 0; r < rows; ++r)
        for (long c = 0; c < cols; ++c) {
            W.w[(c/16*rows + r)*16 + c%16] = Precision(v[r*cols + c]);
            W.tw[(r/16*cols + c)*16 + r%16] = Precision(v[r*cols + c]);
        }
}

template <class Precision>
void NequIPFoundation<Precision>::gemm(bool trans, int m, int n, int k, const Precision* a, int lda, const Weight& W,
                                       Precision* c, int ldc, bool add)
{
    if (m > 0) queue.push_back({m, n, k, a, lda, (trans ? W.tw : W.w).data(), c, ldc, add});
}
