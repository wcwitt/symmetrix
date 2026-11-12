// The cell-list neighbor list of symmetrix/source/cpp/mace_foundation/bindings.cpp (mace_foundation::neighbor_list),
// for the candidates within the cutoff plus the skin, on the CPU or with its search on the GPU.

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "neighbor_list.h"

namespace symmetrix::openmm_neighbors {

CellGrid cell_grid(int n, const double* x, const double* cell, const bool* pbc, double cutoff)
{
    CellGrid g;
    using V = std::array<double, 3>;
    auto cross = [](const V& a, const V& b) { return V{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                                                       a[0]*b[1] - a[1]*b[0]}; };
    auto unit = [](V a) {
        const double s = 1/std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        return V{a[0]*s, a[1]*s, a[2]*s};
    };
    std::array<V, 3> a;
    std::vector<int> missing;
    for (int d = 0; d < 3; ++d) {
        a[d] = {cell[3*d], cell[3*d+1], cell[3*d+2]};
        if (a[d][0] == 0 and a[d][1] == 0 and a[d][2] == 0) {
            if (pbc[d]) throw std::runtime_error("MACEFoundation: a periodic direction without a cell vector");
            missing.push_back(d);
        }
    }
    if (missing.size() == 3) a = {V{1, 0, 0}, V{0, 1, 0}, V{0, 0, 1}};
    else if (missing.size() == 2) {
        const V v = a[3 - missing[0] - missing[1]];
        const int k = std::abs(v[0]) <= std::abs(v[1]) and std::abs(v[0]) <= std::abs(v[2]) ? 0
                    : std::abs(v[1]) <= std::abs(v[2]) ? 1 : 2;
        V e{};
        e[k] = 1;
        a[missing[0]] = unit(cross(v, e));
        a[missing[1]] = unit(cross(v, a[missing[0]]));
    } else if (missing.size() == 1) {
        const int m = missing[0];
        a[m] = unit(cross(a[(m + 1)%3], a[(m + 2)%3]));
    }
    const V c0 = cross(a[1], a[2]);
    const V c1 = cross(a[2], a[0]);
    const V c2 = cross(a[0], a[1]);
    const double det = a[0][0]*c0[0] + a[0][1]*c0[1] + a[0][2]*c0[2];
    if (not (std::abs(det) > 0)) throw std::runtime_error("MACEFoundation: the cell is singular");
    const std::array<V, 3> b = {V{c0[0]/det, c0[1]/det, c0[2]/det}, V{c1[0]/det, c1[1]/det, c1[2]/det},
                                V{c2[0]/det, c2[1]/det, c2[2]/det}};

    std::vector<double> u(3*(size_t)n);
    std::vector<int>& k = g.image;
    k.assign(3*(size_t)n, 0);
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) {
            const double f = x[3*i]*b[d][0] + x[3*i+1]*b[d][1] + x[3*i+2]*b[d][2];
            if (pbc[d]) k[3*i + d] = int(std::floor(f));
            u[3*i + d] = f - k[3*i + d];
        }

    int* nb = g.nb;
    int* reach = g.reach;
    double lo[3];
    double w[3];
    double spacing[3];
    for (int d = 0; d < 3; ++d) {
        spacing[d] = 1/std::sqrt(b[d][0]*b[d][0] + b[d][1]*b[d][1] + b[d][2]*b[d][2]);
        lo[d] = 0;
        double extent = 1;
        if (not pbc[d] and n > 0) {
            lo[d] = u[d];
            double hi = u[d];
            for (int i = 1; i < n; ++i) {
                lo[d] = std::min(lo[d], u[3*i + d]);
                hi = std::max(hi, u[3*i + d]);
            }
            extent = std::max(hi - lo[d], 1e-12);
        }
        w[d] = extent;
        nb[d] = int(std::max(1.0, std::min(1e6, std::floor(extent*spacing[d]/cutoff))));
    }
    while ((long)nb[0]*nb[1]*nb[2] > 4L*n + 64) {
        const int d = nb[0] >= nb[1] and nb[0] >= nb[2] ? 0 : nb[1] >= nb[2] ? 1 : 2;
        nb[d] = (nb[d] + 1)/2;
    }
    for (int d = 0; d < 3; ++d) {
        w[d] /= nb[d];
        const double r = std::ceil(cutoff/(w[d]*spacing[d]));
        reach[d] = pbc[d] ? int(r) : int(std::min<double>(r, nb[d] - 1));
    }

    std::vector<int> bin(n);
    std::vector<int>& first = g.first;
    first.assign(nb[0]*nb[1]*nb[2] + 1, 0);
    std::vector<int>& atoms = g.atoms;
    atoms.resize(n);
    std::vector<int>& c = g.bin;
    c.resize(3*(size_t)n);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[3*i + d] = std::min(nb[d] - 1, std::max(0, int((u[3*i + d] - lo[d])/w[d])));
        bin[i] = (c[3*i]*nb[1] + c[3*i+1])*nb[2] + c[3*i+2];
        ++first[bin[i] + 1];
    }
    for (size_t q = 1; q < first.size(); ++q) first[q] += first[q - 1];
    std::vector<int> fill(first.begin(), first.end() - 1);
    for (int i = 0; i < n; ++i) atoms[fill[bin[i]]++] = i;

    for (int d = 0; d < 3; ++d)
        g.pbc[d] = pbc[d];
    std::copy(cell, cell + 9, g.cell);
    g.cutoff2 = cutoff*cutoff;
    return g;
}

void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl)
{
    const CellGrid g = cell_grid(n, x, cell, pbc, cutoff);
    const int* nb = g.nb;
    const int* reach = g.reach;
    const std::vector<int>& k = g.image;
    const std::vector<int>& first = g.first;
    const std::vector<int>& atoms = g.atoms;
    const std::vector<int>& c = g.bin;
    const double c2max = cutoff*cutoff;
    nl.start.assign(n + 1, 0);
    std::vector<std::vector<int>> found(omp_get_max_threads());
#pragma omp parallel
    {
    std::vector<int>& p = found[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
        int from[3];
        int to[3];
        for (int d = 0; d < 3; ++d) {
            from[d] = pbc[d] ? -reach[d] : std::max(-reach[d], -c[3*i + d]);
            to[d] = pbc[d] ? reach[d] : std::min(reach[d], nb[d] - 1 - c[3*i + d]);
        }
        const size_t before = p.size();
        for (int d0 = from[0]; d0 <= to[0]; ++d0)
            for (int d1 = from[1]; d1 <= to[1]; ++d1)
                for (int d2 = from[2]; d2 <= to[2]; ++d2) {
                    const int t[3] = {c[3*i] + d0, c[3*i+1] + d1, c[3*i+2] + d2};
                    int s[3];
                    int cb[3];
                    for (int d = 0; d < 3; ++d) {
                        s[d] = t[d] >= 0 ? t[d]/nb[d] : -((nb[d] - 1 - t[d])/nb[d]);
                        cb[d] = t[d] - s[d]*nb[d];
                    }
                    const int q = (cb[0]*nb[1] + cb[1])*nb[2] + cb[2];
                    for (int a = first[q]; a < first[q + 1]; ++a) {
                        const int j = atoms[a];
                        const int S[3] = {s[0] + k[3*i] - k[3*j], s[1] + k[3*i+1] - k[3*j+1],
                                          s[2] + k[3*i+2] - k[3*j+2]};
                        if (i == j and S[0] == 0 and S[1] == 0 and S[2] == 0) continue;
                        double r2 = 0;
                        for (int e = 0; e < 3; ++e) {
                            const double D = x[3*j+e] - x[3*i+e] + S[0]*cell[e] + S[1]*cell[3+e] + S[2]*cell[6+e];
                            r2 += D*D;
                        }
                        if (r2 >= c2max) continue;
                        p.insert(p.end(), {j, S[0], S[1], S[2]});
                    }
                }
        nl.start[i + 1] = int((p.size() - before)/4);
    }
#pragma omp single
    {
    for (int i = 0; i < n; ++i) nl.start[i + 1] += nl.start[i];
    nl.j.resize(nl.start[n]);
    nl.S.resize(3*(size_t)nl.start[n]);
    }
    size_t o = 0;
    for (int t = 0; t < omp_get_thread_num(); ++t) o += found[t].size()/4;
    for (size_t q = 0; q < p.size()/4; ++q) {
        nl.j[o + q] = p[4*q];
        std::copy_n(p.data() + 4*q + 1, 3, nl.S.data() + 3*(o + q));
    }
    }
}

namespace {

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("neighbor list: ") + cudaGetErrorString(e));
}

}  // namespace

template <class T> T* GpuNeighborList::reserve(T*& p, size_t& size, size_t m)
{
    m = std::max<size_t>(m, 1);
    if (m > size) {
        if (p) cudaFree(p);
        p = nullptr;
        size = 0;
        m += m/8;
        check(cudaMalloc(reinterpret_cast<void**>(&p), m*sizeof(T)));
        size = m;
    }
    return p;
}

void GpuNeighborList::build(int n, const double* xh, const double* cell, const bool* pbc, double cutoff,
                            NeighborList& nl)
{
    const CellGrid g = cell_grid(n, xh, cell, pbc, cutoff);
    const cudaStream_t s = nullptr;  // the legacy stream, which synchronizes with the blocking ones
    auto up = [&](auto*& p, size_t& size, const auto* h, size_t m) {
        reserve(p, size, m);
        check(cudaMemcpyAsync(p, h, m*sizeof(*h), cudaMemcpyHostToDevice, s));
    };
    up(x, x_size, xh, 3*size_t(n));
    up(image, image_size, g.image.data(), g.image.size());
    up(bin, bin_size, g.bin.data(), g.bin.size());
    up(first, first_size, g.first.data(), g.first.size());
    up(atoms, atoms_size, g.atoms.data(), g.atoms.size());
    reserve(count, count_size, n);
    reserve(start, start_size, n + 1);

    openmm_kernels::CellSearch c;
    for (int d = 0; d < 3; ++d) {
        c.nb[d] = g.nb[d];
        c.reach[d] = g.reach[d];
        c.pbc[d] = g.pbc[d];
    }
    std::copy(g.cell, g.cell + 9, c.cell);
    c.cutoff2 = g.cutoff2;
    openmm_kernels::neighbor_search(n, c, x, image, bin, first, atoms, nullptr, count, nullptr, nullptr, s);
    openmm_kernels::scan(n, count, start, scratch, s);
    nl.start.resize(n + 1);
    check(cudaMemcpyAsync(nl.start.data(), start, (n + 1)*sizeof(int), cudaMemcpyDeviceToHost, s));
    check(cudaStreamSynchronize(s));
    const size_t E = nl.start[n];
    reserve(j, j_size, E);
    reserve(S, S_size, 3*E);
    openmm_kernels::neighbor_search(n, c, x, image, bin, first, atoms, start, count, j, S, s);
    nl.j.resize(E);
    nl.S.resize(3*E);
    if (E > 0) {
        check(cudaMemcpyAsync(nl.j.data(), j, E*sizeof(int), cudaMemcpyDeviceToHost, s));
        check(cudaMemcpyAsync(nl.S.data(), S, 3*E*sizeof(int), cudaMemcpyDeviceToHost, s));
    }
    check(cudaStreamSynchronize(s));
}

void GpuNeighborList::clear()
{
    for (void* p : {(void*)x, (void*)image, (void*)bin, (void*)first, (void*)atoms, (void*)count, (void*)start,
                    (void*)j, (void*)S})
        if (p) cudaFree(p);
    x = nullptr;
    image = bin = first = atoms = count = start = j = S = nullptr;
    x_size = image_size = bin_size = first_size = atoms_size = count_size = start_size = j_size = S_size = 0;
    scratch.clear();
}

}  // namespace symmetrix::openmm_neighbors
