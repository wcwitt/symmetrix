#include "evaluator.hpp"
#include "nequip_foundation_cuda.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Atoms { const double *x, *cell; const int *first, *j, *S; };

__device__ __forceinline__ void displacement(const Atoms& A, int i, int c, double& dx, double& dy, double& dz) {
    const int j = A.j[c];
    const int* s = A.S + 3*c;
    dx = A.x[3*j] - A.x[3*i] + s[0]*A.cell[0] + s[1]*A.cell[3] + s[2]*A.cell[6];
    dy = A.x[3*j+1] - A.x[3*i+1] + s[0]*A.cell[1] + s[1]*A.cell[4] + s[2]*A.cell[7];
    dz = A.x[3*j+2] - A.x[3*i+2] + s[0]*A.cell[2] + s[1]*A.cell[5] + s[2]*A.cell[8];
}

__global__ void k_count(int N, double rc2, const int* atom, Atoms A, int* count, unsigned short* cand) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int i = atom[n];
    const int c0 = A.first[i];
    const int nn = A.first[i+1] - c0;
    unsigned short* cn = cand + (long)n*SORT_MAX;
    int cnt = 0;
    for (int j0 = 0; j0 < nn; j0 += 32) {
        const int jj = j0 + lane;
        bool inside = false;
        if (jj < nn) {
            double dx;
            double dy;
            double dz;
            displacement(A, i, c0 + jj, dx, dy, dz);
            inside = dx*dx + dy*dy + dz*dz < rc2;
        }
        const unsigned m = __ballot_sync(~0u, inside);
        const int c = cnt + __popc(m & ((1u << lane) - 1));
        if (inside and c < SORT_MAX) cn[c] = (unsigned short)jj;
        cnt += __popc(m);
    }
    if (lane == 0) count[n] = cnt;
}

template <class Sh, class T>
__global__ void __launch_bounds__(128) k_fill(Model<T> M, Graph G, double rc2, const int* atom, const int* row_of_atom,
                                              Atoms A, const int* count, const unsigned short* cand, int* neigh_rows,
                                              int* neigh_indices, double* vec, T* geometry, int* interval, T* xs,
                                              T* edge_forces, double* node_energies) {
    __shared__ __align__(16) float keys[4][SORT_MAX + 4];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    if (n >= G.N) return;
    const int i = atom[n];
    const int cnt = count[n];
    const int s0 = G.start[n];
    const int ti = G.elem[n];
    const int c0 = A.first[i];
    double ez = 0;
    auto add = [&](int e, int j, double dx, double dy, double dz) {
        const int row = row_of_atom[j];
        ez += edge_record<Sh>(M, e, ti, G.elem[row], row, j,
                              dx, dy, dz, neigh_rows, neigh_indices, vec, geometry, interval, xs, edge_forces);
    };
    if (cnt <= SORT_MAX) {
        const unsigned short* cn = cand + (long)n*SORT_MAX;
        if constexpr (Sh::sort_key != 0) {
            for (int c = lane; c < cnt; c += 32) {
                double dx;
                double dy;
                double dz;
                displacement(A, i, c0 + cn[c], dx, dy, dz);
                keys[w][c] = float(sqrt(dx*dx + dy*dy + dz*dz));
            }
            const int cnt4 = (cnt + 3) & ~3;
            if (lane < cnt4 - cnt) keys[w][cnt + lane] = __int_as_float(0x7f800000);
            __syncwarp();
        }
        for (int c = lane; c < cnt; c += 32) {
            double dx;
            double dy;
            double dz;
            displacement(A, i, c0 + cn[c], dx, dy, dz);
            int rank = c;
            if constexpr (Sh::sort_key != 0) {
                const float key = keys[w][c];
                rank = 0;
                for (int q = 0; q < cnt; q += 4) {
                    const float4 k4 = *reinterpret_cast<const float4*>(&keys[w][q]);
                    rank += (k4.x < key) + (k4.y < key) + (k4.z < key) + (k4.w < key);
                    rank += (k4.x == key and q < c) + (k4.y == key and q + 1 < c) + (k4.z == key and q + 2 < c)
                            + (k4.w == key and q + 3 < c);
                }
            }
            add(s0 + rank, A.j[c0 + cn[c]], dx, dy, dz);
        }
    } else {
        const int nn = A.first[i+1] - c0;
        int pos = 0;
        for (int j0 = 0; j0 < nn; j0 += 32) {
            const int jj = j0 + lane;
            bool inside = false;
            double dx = 0;
            double dy = 0;
            double dz = 0;
            if (jj < nn) {
                displacement(A, i, c0 + jj, dx, dy, dz);
                inside = dx*dx + dy*dy + dz*dz < rc2;
            }
            const unsigned m = __ballot_sync(~0u, inside);
            if (inside) add(s0 + pos + __popc(m & ((1u << lane) - 1)), A.j[c0 + jj], dx, dy, dz);
            pos += __popc(m);
        }
    }
    for (int o = 16; o > 0; o >>= 1) ez += __shfl_xor_sync(~0u, ez, o);
    if (lane == 0) node_energies[n] = ez;
}

template <class T>
__global__ void k_forces(Graph G, const int* atom, const int* neigh_indices, const T* edge_forces, double* f) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    double fx = 0;
    double fy = 0;
    double fz = 0;
    for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
        const int j = neigh_indices[e];
        const double f0 = edge_forces[3*e];
        const double f1 = edge_forces[3*e+1];
        const double f2 = edge_forces[3*e+2];
        atomicAdd(f + 3*j, f0);
        atomicAdd(f + 3*j + 1, f1);
        atomicAdd(f + 3*j + 2, f2);
        fx += f0;
        fy += f1;
        fz += f2;
    }
    for (int o = 16; o > 0; o >>= 1) {
        fx += __shfl_xor_sync(~0u, fx, o);
        fy += __shfl_xor_sync(~0u, fy, o);
        fz += __shfl_xor_sync(~0u, fz, o);
    }
    if (lane == 0) {
        const int i = atom[n];
        atomicAdd(f + 3*i, -fx);
        atomicAdd(f + 3*i + 1, -fy);
        atomicAdd(f + 3*i + 2, -fz);
    }
}

}  // namespace

template <class Precision, bool mixed>
struct NequIPFoundationCUDA : Evaluator {
    NequIPFoundationCUDA(const std::string& file, double skin);
    ~NequIPFoundationCUDA() override;
    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                   double* forces, double* virial, double* energies) override;
    double compute_device(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                          bool moved, double* virial, bool* rebuilt) override;
    const double* forces_device() const override { return fd.p; }
    bool changed(int n, const int* numbers, const double* cell, const bool* pbc) const;
    void build(int n, const double* x, const int* numbers, const double* cell, const bool* pbc);
    double evaluate(int n, const double* x, const double* cell, double* forces, double* virial, double* energies);
    template <class Sh>
    double compute_shape(int n, const double* x, const double* cell, double* forces, double* virial, double* energies);
    template <class Sh> void compute_node_energies_forces();

    NequIPFoundationCuda<Precision, mixed> nequip;
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
NequIPFoundationCUDA<Precision, mixed>::NequIPFoundationCUDA(const std::string& file, double skin)
        : nequip(file, "NequIPFoundation: "), skin(skin)
{
}

template <class Precision, bool mixed>
NequIPFoundationCUDA<Precision, mixed>::~NequIPFoundationCUDA()
{
    if (graph) cudaGraphExecDestroy(graph);
}

template <class Precision, bool mixed>
double NequIPFoundationCUDA<Precision, mixed>::compute(int n, const double* x, const int* numbers,
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
double NequIPFoundationCUDA<Precision, mixed>::compute_device(int n, const double* x, const int* numbers,
                                                              const double* cell, const bool* pbc, bool moved,
                                                              double* virial, bool* rebuilt)
{
    std::fill(virial, virial + 6, 0.0);
    *rebuilt = false;
    if (n == 0) return 0;
    if (moved or changed(n, numbers, cell, pbc)) {
        std::vector<double> host(3*(size_t)n);
        CHECK(cudaMemcpyAsync(host.data(), x, host.size()*sizeof(double), cudaMemcpyDeviceToHost, nequip.stream));
        CHECK(cudaStreamSynchronize(nequip.stream));
        build(n, host.data(), numbers, cell, pbc);
        *rebuilt = true;
    }
    return evaluate(n, x, cell, nullptr, virial, nullptr);
}

template <class Precision, bool mixed>
bool NequIPFoundationCUDA<Precision, mixed>::changed(int n, const int* numbers, const double* cell,
                                                     const bool* pbc) const
{
    return n != natoms or not std::equal(numbers, numbers + n, numbers0.begin())
           or not std::equal(cell, cell + 9, cell0.begin()) or not std::equal(pbc, pbc + 3, pbc0.begin());
}

template <class Precision, bool mixed>
void NequIPFoundationCUDA<Precision, mixed>::build(int n, const double* x, const int* numbers, const double* cell,
                                                   const bool* pbc)
{
    natoms = -1;
    const int T_ = nequip.num_elements;
    std::vector<int> el(n);
    for (int i = 0; i < n; ++i) {
        const auto it = std::find(nequip.atomic_numbers.begin(), nequip.atomic_numbers.end(), numbers[i]);
        if (it == nequip.atomic_numbers.end())
            throw std::runtime_error("NequIPFoundation: atomic number " + std::to_string(numbers[i])
                                     + " is not an element of the model");
        el[i] = int(it - nequip.atomic_numbers.begin());
    }
    neighbor_list(n, x, cell, pbc, nequip.r_cut + skin, nl);
    std::vector<int> nodes;
    nequip.type_start.assign(T_ + 1, 0);
    for (int t = 0; t < T_; ++t) {
        nequip.type_start[t] = int(nodes.size());
        for (int i = 0; i < n; ++i) if (el[i] == t) nodes.push_back(i);
    }
    nequip.type_start[T_] = int(nodes.size());
    nequip.N = n;
    nequip.NR = n;
    std::vector<int> node_type(nequip.N);
    std::vector<int> rows(n);
    for (int k = 0; k < nequip.N; ++k) {
        node_type[k] = el[nodes[k]];
        rows[nodes[k]] = k;
    }
    nequip.E = long(nl.j.size());
    nequip.atom_of_node.upload(nodes, nequip.stream);
    nequip.elem.upload(node_type, nequip.stream);
    nequip.row_of_atom.upload(rows, nequip.stream);
    nl_first.upload(nl.start, nequip.stream);
    nl_j.upload(nl.j, nequip.stream);
    nl_S.upload(nl.S, nequip.stream);
    xd.reserve(3*(size_t)n);
    fd.reserve(3*(size_t)n);
    eatom.reserve(n);
    celld.reserve(9);
    with_preset(nequip.num_layers, nequip.l_max, nequip.emb, nequip.mul0, [&](auto s) {
        nequip.template reserve<decltype(s)>(0);
    });
    CHECK(cudaStreamSynchronize(nequip.stream));
    natoms = n;
    x0.assign(x, x + 3*(size_t)n);
    cell0.assign(cell, cell + 9);
    numbers0.assign(numbers, numbers + n);
    pbc0 = {pbc[0], pbc[1], pbc[2]};
}

// x and forces may be on the host or on the GPU (forces may be null)
template <class Precision, bool mixed>
double NequIPFoundationCUDA<Precision, mixed>::evaluate(int n, const double* x, const double* cell, double* forces,
                                                        double* virial, double* energies)
{
    nequip.upload_coefficients();
    double energy = 0;
    with_preset(nequip.num_layers, nequip.l_max, nequip.emb, nequip.mul0, [&](auto s) {
        energy = this->template compute_shape<decltype(s)>(n, x, cell, forces, virial, energies);
    });
    return energy;
}

template <class Precision, bool mixed>
template <class Sh>
double NequIPFoundationCUDA<Precision, mixed>::compute_shape(int n_atoms, const double* x, const double* cell,
                                                             double* forces, double* virial, double* energies)
{
    using T = Precision;
    constexpr int NL = Sh::num_layers;
    CHECK(cudaMemcpyAsync(xd, x, 3*(size_t)n_atoms*sizeof(double), cudaMemcpyDefault, nequip.stream));
    CHECK(cudaMemcpyAsync(celld, cell, 9*sizeof(double), cudaMemcpyHostToDevice, nequip.stream));
    const Atoms A{xd, celld, nl_first, nl_j, nl_S};

    nequip.template arguments<Sh>();

    const double rc2 = nequip.r_cut*nequip.r_cut;
    const long n = nequip.N;
    std::vector<const void*> key = {(const void*)long(nequip.N), xd.p, celld.p, nl_first.p, nl_j.p, nl_S.p, fd.p,
                                    eatom.p, nequip.atom_of_node, nequip.row_of_atom, nequip.elem, nequip.count,
                                    nequip.start, nequip.cand, nequip.neigh_indices, nequip.neigh_rows, nequip.vec,
                                    nequip.geometry, nequip.interval, nequip.xs, nequip.edge_forces, nequip.H_even,
                                    nequip.H_odd, nequip.Phi, nequip.A_linear, nequip.A_sc, nequip.A_adj, nequip.X_adj,
                                    nequip.H_adj_linear, nequip.H_adj_sc, nequip.node_energies, nequip.outputs,
                                    nequip.scan_storage};
    for (int q = 1; q < NL; ++q) key.push_back(nequip.X[q]);
    for (int q = 0; q < NL - 1; ++q) key.push_back(nequip.A[q]);
    for (int t : nequip.type_start) key.push_back((const void*)long(t));
    if (key != captured) {
        CHECK(cudaStreamSynchronize(nequip.stream));
        if (graph) CHECK(cudaGraphExecDestroy(graph));
        cudaGraph_t g;
        CHECK(cudaStreamBeginCapture(nequip.stream, cudaStreamCaptureModeThreadLocal));
        if (nequip.N > 0) {
            k_count<<<blocks(32*n, 128), 128, 0, nequip.stream>>>(nequip.N, rc2, nequip.atom_of_node, A,
                                                                  nequip.count, nequip.cand);
            size_t bytes = nequip.scan_bytes;
            CHECK(cub::DeviceScan::ExclusiveSum(nequip.scan_storage.p, bytes, nequip.count.p, nequip.start.p,
                                                nequip.N, nequip.stream));
            k_last_start<<<1, 32, 0, nequip.stream>>>(nequip.N, nequip.count, nequip.start);
            k_fill<Sh, T><<<blocks(32*n, 128), 128, 0, nequip.stream>>>(nequip.M, nequip.G, rc2, nequip.atom_of_node,
                                                                        nequip.row_of_atom, A, nequip.count,
                                                                        nequip.cand, nequip.neigh_rows,
                                                                        nequip.neigh_indices, nequip.vec,
                                                                        nequip.geometry, nequip.interval, nequip.xs,
                                                                        nequip.edge_forces, nequip.node_energies);
        }
        compute_node_energies_forces<Sh>();
        CHECK(cudaMemsetAsync(nequip.outputs, 0, 7*sizeof(double), nequip.stream));
        CHECK(cudaMemsetAsync(fd, 0, 3*(size_t)n*sizeof(double), nequip.stream));
        if (nequip.N > 0) {
            k_forces<<<blocks(32*n, 128), 128, 0, nequip.stream>>>(nequip.G, nequip.atom_of_node, nequip.neigh_indices,
                                                                   nequip.edge_forces.p, fd);
            k_outputs<<<blocks(32*n, 128), 128, 0, nequip.stream>>>(nequip.G, nequip.node_energies, nequip.vec,
                                                                    nequip.edge_forces.p, nequip.outputs);
            k_atom_energies<<<blocks(nequip.N, 128), 128, 0, nequip.stream>>>(nequip.N, nequip.atom_of_node,
                    nequip.node_energies, eatom);
        }
        CHECK(cudaStreamEndCapture(nequip.stream, &g));
        CHECK(cudaGraphInstantiate(&graph, g, 0));
        CHECK(cudaGraphDestroy(g));
        captured = key;
    }
    CHECK(cudaGraphLaunch(graph, nequip.stream));
    double out[7];
    CHECK(cudaMemcpyAsync(out, nequip.outputs, 7*sizeof(double), cudaMemcpyDeviceToHost, nequip.stream));
    if (forces)
        CHECK(cudaMemcpyAsync(forces, fd, 3*(size_t)n_atoms*sizeof(double), cudaMemcpyDefault, nequip.stream));
    if (energies)
        CHECK(cudaMemcpyAsync(energies, eatom, (size_t)n_atoms*sizeof(double), cudaMemcpyDeviceToHost, nequip.stream));
    CHECK(cudaStreamSynchronize(nequip.stream));
    for (int c = 0; c < 6; ++c) virial[c] = out[c + 1];
    return out[0];
}

template <class Precision, bool mixed>
template <class Sh>
void NequIPFoundationCUDA<Precision, mixed>::compute_node_energies_forces()
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    // \lambda = (l, p),  H^{(0)}_i = H^{(0)}_{a_i},  X^{(0)}_j = X^{(0)}_{a_j},  t = 0, \dots, T-1
    for (int t = 0; t < Sh::num_layers; ++t)
        nequip.template compute_layer<Sh>(t);
    nequip.template compute_readout<Sh>();
    for (int t = Sh::num_layers - 1; t >= 0; --t)
        nequip.template reverse_layer<Sh>(t);
}

std::unique_ptr<Evaluator> nequip_foundation_cuda(const std::string& file, const std::string& dtype, double skin)
{
    if (dtype == "float32-mixed") return std::make_unique<NequIPFoundationCUDA<float, true>>(file, skin);
    if (dtype == "float32") return std::make_unique<NequIPFoundationCUDA<float, false>>(file, skin);
    if (dtype == "float64") return std::make_unique<NequIPFoundationCUDA<double, false>>(file, skin);
    throw std::invalid_argument("dtype must be 'float64', 'float32' or 'float32-mixed', not '" + dtype + "'");
}
