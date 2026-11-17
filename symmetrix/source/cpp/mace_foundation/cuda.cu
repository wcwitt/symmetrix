
#include "evaluator.hpp"
#include "mace_foundation_cuda.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mace_foundation;

namespace {

constexpr unsigned ERR_SHORT = 4;

struct Atoms { const double *x, *cell; const int *first, *j, *S; double* f; };

__device__ __forceinline__ void displacement(const Atoms& A, int i, int c, double& dx, double& dy, double& dz) {
    const int j = A.j[c];
    const int* s = A.S + 3*c;
    dx = A.x[3*j] - A.x[3*i] + s[0]*A.cell[0] + s[1]*A.cell[3] + s[2]*A.cell[6];
    dy = A.x[3*j+1] - A.x[3*i+1] + s[0]*A.cell[1] + s[1]*A.cell[4] + s[2]*A.cell[7];
    dz = A.x[3*j+2] - A.x[3*i+2] + s[0]*A.cell[2] + s[1]*A.cell[5] + s[2]*A.cell[8];
}

template <class real> __global__ void k_count(Model<real> M, Graph<real> G, Atoms A) {
    extern __shared__ int shared[];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int* cnt = shared + (threadIdx.x >> 5)*M.T;
    if (n >= G.N) return;
    for (int b = lane; b < M.T; b += 32) cnt[b] = 0;
    __syncwarp();
    const int i = G.atom[n];
    for (int c = A.first[i] + lane; c < A.first[i+1]; c += 32) {
        double dx, dy, dz;
        displacement(A, i, c, dx, dy, dz);
        if (dx*dx + dy*dy + dz*dz < M.r_cut2) atomicAdd(&cnt[G.elem_all[A.j[c]]], 1);
    }
    __syncwarp();
    int total = 0;
    for (int b = lane; b < M.T; b += 32) {
        G.seg[n*(M.T + 1) + 1 + b] = cnt[b];
        total += cnt[b];
    }
    for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(~0u, total, o);
    if (lane == 0) G.count[n] = total;
}

template <class real> __global__ void k_fill(Model<real> M, Graph<real> G, Atoms A) {
    extern __shared__ int shared[];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int* off = shared + (threadIdx.x >> 5)*M.T;
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
    __syncwarp();
    real s0 = 0;
    for (int jj0 = A.first[i]; jj0 < A.first[i+1]; jj0 += 32) {
        const int jj = jj0 + lane;
        int j = 0;
        int b = -1;
        double dx = 0;
        double dy = 0;
        double dz = 0;
        double r2 = 0;
        if (jj < A.first[i+1]) {
            j = A.j[jj];
            displacement(A, i, jj, dx, dy, dz);
            r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < M.r_cut2) b = G.elem_all[j];
        }
        const unsigned peers = match_any(b);
        if (b >= 0) {
            const int e = off[b] + __popc(peers & ((1u << lane) - 1));
            const double r = sqrt(r2);
            const double t = (r - M.r_min)*M.inv_h;
            const int knot = min(max((int)t, 0), M.K - 1);
            const int p = M.ordered ? a*T + b : pair_index(a, b, T);
            const real x = real(t - knot - 0.5);
            *reinterpret_cast<Vec4<real>*>(G.geometry + 4*e) = {real(dz/r), real(dx/r), real(dy/r), real(r)};
            G.neigh_indices[e] = j;
            G.row_pair[e] = make_int2(G.row_of_atom[j], p);
            if (r < M.r_min) atomicOr(G.err, ERR_SHORT);
            const long blk = ((long)p*M.K + knot)*4;
            if (M.A0_splines) s0 += cubic(M.A0_splines + blk, 1, 0, x);
        }
        __syncwarp();
        if (b >= 0 and lane == __ffs(peers) - 1) off[b] += __popc(peers);
        __syncwarp();
    }
    s0 = warp_sum(s0);
    if (lane == 0) G.s0[n] = 1 + s0;
}

template <class real> __global__ void k_forces(Model<real> M, Graph<real> G, Atoms A) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    const real q0 = M.A0_splines ? G.dot0[n] : 0;
    const real q1 = M.A1_splines ? G.dot1[n] : 0;
    double fx = 0;
    double fy = 0;
    double fz = 0;
    for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
        real* fe = G.edge_forces + 3*e;
        if (q0 != 0 or q1 != 0) {
            const Edge<real> q = edge(M, G, e);
            const real* blk = nullptr;
            real v, d0 = 0, d1 = 0;
            const long b = ((long)q.pair*M.K + q.knot)*4;
            if (M.A0_splines) {
                blk = M.A0_splines + b;
                cubic(blk, 1, 0, q.t, v, d0);
            }
            if (M.A1_splines) {
                blk = M.A1_splines + b;
                cubic(blk, 1, 0, q.t, v, d1);
            }
            const real s = (q0*d0 + q1*d1)*M.inv_h;
            fe[0] += s*q.y;
            fe[1] += s*q.z;
            fe[2] += s*q.x;
        }
        const int j = G.neigh_indices[e];
        atomicAdd(A.f + 3*j, (double)fe[0]); atomicAdd(A.f + 3*j + 1, (double)fe[1]); atomicAdd(A.f + 3*j + 2,
                (double)fe[2]);
        fx += fe[0];
        fy += fe[1];
        fz += fe[2];
    }
    fx = warp_sum(fx);
    fy = warp_sum(fy);
    fz = warp_sum(fz);
    if (lane == 0) {
        const int i = G.atom[n];
        atomicAdd(A.f + 3*i, -fx);
        atomicAdd(A.f + 3*i + 1, -fy);
        atomicAdd(A.f + 3*i + 2, -fz);
    }
}

template <class Precision, bool mixed>
struct MACEFoundationCUDA : Evaluator {
    MACEFoundationCUDA(const std::string& file, double skin);
    ~MACEFoundationCUDA() override;
    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                   double* forces, double* virial, double* energies) override;
    double compute_device(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                          bool moved, bool* rebuilt) override;
    const double* forces_device() const override { return fd.p; }
    bool changed(int n, const int* numbers, const double* cell, const bool* pbc) const;
    void build(int n, const double* x, const int* numbers, const double* cell, const bool* pbc);
    double evaluate(int n, const double* x, const double* cell, double* forces, double* virial, double* energies);
    template <class Sh> double compute_shape(int, const double*, const double*, double*, double*, double*);
    template <class Sh>
    void compute_node_energies_forces(const Model<Precision>&, const Graph<Precision>&, const Atoms&, bool);
    void compute_edges(const Model<Precision>&, const Graph<Precision>&, const Atoms&);
    void compute_forces(const Model<Precision>&, const Graph<Precision>&, const Atoms&);

    MACEFoundationCuda<Precision, mixed> mace;
    double skin;
    int natoms = -1;
    std::vector<double> x0, cell0;
    std::vector<int> numbers0;
    std::array<bool, 3> pbc0{};
    NeighborList nl;
    Buffer<double> xd, celld, fd, eatom;
    Buffer<int> nl_first, nl_j, nl_S;
    cudaGraphExec_t graph = nullptr;
    std::vector<const void*> captured;
};

template <class Precision, bool mixed>
MACEFoundationCUDA<Precision, mixed>::MACEFoundationCUDA(const std::string& file, double skin) : mace(file,
        "MACEFoundation: "), skin(skin)
{
}

template <class Precision, bool mixed>
MACEFoundationCUDA<Precision, mixed>::~MACEFoundationCUDA()
{
    if (graph) cudaGraphExecDestroy(graph);
}

template <class Precision, bool mixed>
double MACEFoundationCUDA<Precision, mixed>::compute(int n, const double* x, const int* numbers,
                                                     const double* cell, const bool* pbc, double* forces,
                                                     double* virial, double* energies)
{
    std::fill(virial, virial + 6, 0.0);
    if (n == 0) return 0;
    bool rebuild = changed(n, numbers, cell, pbc);
    for (int i = 0; not rebuild and i < n; ++i) {
        const double dx = x[3*i] - x0[3*i];
        const double dy = x[3*i+1] - x0[3*i+1];
        const double dz = x[3*i+2] - x0[3*i+2];
        rebuild = 4*(dx*dx + dy*dy + dz*dz) > skin*skin;
    }
    if (rebuild) build(n, x, numbers, cell, pbc);
    return evaluate(n, x, cell, forces, virial, energies);
}

template <class Precision, bool mixed>
double MACEFoundationCUDA<Precision, mixed>::compute_device(int n, const double* x, const int* numbers,
                                                            const double* cell, const bool* pbc, bool moved,
                                                            bool* rebuilt)
{
    *rebuilt = false;
    if (n == 0) return 0;
    if (moved or changed(n, numbers, cell, pbc)) {
        std::vector<double> host(3*(size_t)n);
        CHECK(cudaMemcpyAsync(host.data(), x, host.size()*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
        CHECK(cudaStreamSynchronize(mace.stream));
        build(n, host.data(), numbers, cell, pbc);
        *rebuilt = true;
    }
    double virial[6];
    return evaluate(n, x, cell, nullptr, virial, nullptr);
}

template <class Precision, bool mixed>
bool MACEFoundationCUDA<Precision, mixed>::changed(int n, const int* numbers, const double* cell,
                                                   const bool* pbc) const
{
    return n != natoms or not std::equal(numbers, numbers + n, numbers0.begin())
           or not std::equal(cell, cell + 9, cell0.begin()) or not std::equal(pbc, pbc + 3, pbc0.begin());
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::build(int n, const double* x, const int* numbers, const double* cell,
                                                 const bool* pbc)
{
    natoms = -1;
    const int T = mace.num_elements;
    std::vector<int> el(n);
    for (int i = 0; i < n; ++i) {
        const auto it = std::find(mace.atomic_numbers.begin(), mace.atomic_numbers.end(), numbers[i]);
        if (it == mace.atomic_numbers.end())
            throw std::runtime_error("MACEFoundation: atomic number " + std::to_string(numbers[i])
                                     + " is not an element of the model");
        el[i] = int(it - mace.atomic_numbers.begin());
    }
    neighbor_list(n, x, cell, pbc, mace.r_cut + skin, nl);

    std::vector<int> nodes;
    mace.segments.clear();
    for (int t = 0; t < T; ++t) {
        const int first = int(nodes.size());
        for (int i = 0; i < n; ++i) if (el[i] == t) nodes.push_back(i);
        if (int(nodes.size()) > first) mace.segments.push_back({first, int(nodes.size()), t});
    }
    mace.N = n;
    mace.NR = n;
    std::vector<int> node_elem(mace.N);
    std::vector<int> rows(n);
    for (int k = 0; k < mace.N; ++k) {
        node_elem[k] = el[nodes[k]];
        rows[nodes[k]] = k;
    }
    mace.atom_of_node.upload(nodes, mace.stream);
    mace.elem.upload(node_elem, mace.stream);
    mace.row_of_atom.upload(rows, mace.stream);
    mace.elem_all.upload(el, mace.stream);
    nl_first.upload(nl.start, mace.stream);
    nl_j.upload(nl.j, mace.stream);
    nl_S.upload(nl.S, mace.stream);
    xd.reserve(3*(size_t)n);
    fd.reserve(3*(size_t)n);
    eatom.reserve(n);
    celld.reserve(9);

    mace.reserve(std::max<long>(nl.j.size(), 1));
    CHECK(cudaStreamSynchronize(mace.stream));

    natoms = n;
    x0.assign(x, x + 3*(size_t)n);
    cell0.assign(cell, cell + 9);
    numbers0.assign(numbers, numbers + n);
    pbc0 = {pbc[0], pbc[1], pbc[2]};
}

// x and forces may be on the host or on the GPU (forces may be null)
template <class Precision, bool mixed>
double MACEFoundationCUDA<Precision, mixed>::evaluate(int n, const double* x, const double* cell, double* forces,
                                                      double* virial, double* energies)
{
    double energy = 0;
    with_preset(mace.num_channels, mace.L_max, mace.num_channels_node, [&](auto s) {
        energy = this->template compute_shape<decltype(s)>(n, x, cell, forces, virial, energies);
    });
    return energy;
}

template <class Precision, bool mixed>
template <class Sh>
double MACEFoundationCUDA<Precision, mixed>::compute_shape(int n, const double* x, const double* cell, double* forces,
                                                           double* virial, double* energies)
{
    CHECK(cudaMemcpyAsync(xd, x, 3*(size_t)n*sizeof(double), cudaMemcpyDefault, mace.stream));
    CHECK(cudaMemcpyAsync(celld, cell, 9*sizeof(double), cudaMemcpyHostToDevice, mace.stream));

    const Model<Precision> M = mace.model();
    const Graph<Precision> G = mace.graph();
    const Atoms A{xd, celld, nl_first, nl_j, nl_S, fd};

    std::vector<const void*> key = {(const void*)(long)mace.N, A.x, A.cell, A.first, A.j, A.S, A.f,
        G.atom, G.elem, G.row_of_atom, G.count, G.start, G.seg, G.elem_all, G.geometry, G.neigh_indices, G.row_pair,
        G.edge_forces, G.Phi0, G.A0, G.M0, G.H1,
        G.Phi1, G.A1, G.H1_adj, G.A1_adj, G.dPhi1, G.M0_adj, G.A0_adj, G.Phi0_adj, G.s0, G.s1, G.dot0, G.dot1,
        G.node_energies, mace.scan_storage.p, G.X0, G.X1, G.Tg, G.T_adj, G.X_adj, G.B1, G.B1_adj, G.M0s_adj,
        G.M1, G.M1_adj};
    for (const auto& [b, e, t] : mace.segments)
        key.insert(key.end(), {(const void*)(long)b, (const void*)(long)e, (const void*)(long)t});
    const bool capture = key != captured;
    compute_node_energies_forces<Sh>(M, G, A, capture);
    if (capture) captured = key;

    CHECK(cudaMemsetAsync(mace.outputs, 0, 7*sizeof(double), mace.stream));
    k_outputs<<<blocks(mace.N, WARPS), 32*WARPS, 0, mace.stream>>>(G, energies ? eatom.p : nullptr, mace.outputs);
    double out[7];
    CHECK(cudaMemcpyAsync(out, mace.outputs, sizeof(out), cudaMemcpyDeviceToHost, mace.stream));
    if (forces)
        CHECK(cudaMemcpyAsync(forces, A.f, 3*(size_t)n*sizeof(double), cudaMemcpyDefault, mace.stream));
    if (energies)
        CHECK(cudaMemcpyAsync(energies, eatom, (size_t)n*sizeof(double), cudaMemcpyDeviceToHost, mace.stream));
    CHECK(cudaStreamSynchronize(mace.stream));
    if (*mace.err & ERR_SHORT) {
        *mace.err = 0;
        throw std::runtime_error("MACEFoundation: a pair is closer than the model's r_min");
    }
    std::copy(out + 1, out + 7, virial);
    return out[0];
}

template <class Precision, bool mixed>
template <class Sh>
void MACEFoundationCUDA<Precision, mixed>::compute_node_energies_forces(
    const Model<Precision>& M, const Graph<Precision>& G, const Atoms& A, bool capture)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    if (capture) {
        if (graph) CHECK(cudaGraphExecDestroy(graph));
        cudaGraph_t g;
        CHECK(cudaStreamBeginCapture(mace.stream, cudaStreamCaptureModeThreadLocal));
        compute_edges(M, G, A);
        mace.template compute_first_layer<Sh>(M, G);
        mace.template compute_second_layer<Sh>(M, G);
        mace.template compute_readouts<Sh>(M, G);
        mace.template reverse_second_layer<Sh>(M, G);
        mace.template reverse_first_layer<Sh>(M, G);
        compute_forces(M, G, A);
        CHECK(cudaStreamEndCapture(mace.stream, &g));
        CHECK(cudaGraphInstantiate(&graph, g, 0));
        CHECK(cudaGraphDestroy(g));
    }
    CHECK(cudaGraphLaunch(graph, mace.stream));
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::compute_edges(const Model<Precision>& M,
                                                         const Graph<Precision>& G, const Atoms& A)
{
    // neighbor list
    // A0 scaling
    // s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij}),  \tilde{A}^{(0)}_{i,klm} = A^{(0)}_{i,klm} / s^{(0)}_i
    const size_t shared = WARPS*mace.num_elements*sizeof(int);
    k_count<<<blocks(mace.N, WARPS), 32*WARPS, shared, mace.stream>>>(M, G, A);
    size_t bytes = mace.scan_bytes;
    CHECK(cub::DeviceScan::ExclusiveSum(mace.scan_storage.p, bytes, G.count, G.start, mace.N + 1, mace.stream));
    k_fill<<<blocks(mace.N, WARPS), 32*WARPS, shared, mace.stream>>>(M, G, A);
}

template <class Precision, bool mixed>
void MACEFoundationCUDA<Precision, mixed>::compute_forces(const Model<Precision>& M,
                                                          const Graph<Precision>& G, const Atoms& A)
{
    // A1 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(1)}_i
    // A0 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(0)}_i
    // forces
    CHECK(cudaMemsetAsync(A.f, 0, 3*(size_t)mace.N*sizeof(double), mace.stream));
    k_forces<<<blocks(mace.N, WARPS), 32*WARPS, 0, mace.stream>>>(M, G, A);
}

}  // namespace

std::unique_ptr<Evaluator> mace_foundation_cuda(const std::string& file, const std::string& dtype, double skin)
{
    if (dtype == "float32-mixed") return std::make_unique<MACEFoundationCUDA<float, true>>(file, skin);
    if (dtype == "float32") return std::make_unique<MACEFoundationCUDA<float, false>>(file, skin);
    return std::make_unique<MACEFoundationCUDA<double, false>>(file, skin);
}
