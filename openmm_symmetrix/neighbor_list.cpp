// The cell-list neighbor list of symmetrix/source/cpp/mace_foundation/bindings.cpp (mace_foundation::neighbor_list),
// for the candidates within the cutoff plus the skin: the cell grid is set up on the CPU and searched on the GPU.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "neighbor_list.h"

namespace symmetrix::gpu {

namespace {

// The cell grid of the search: image[3i..] are the cell periods of atom i, bin[3i..] its bin, and the atoms of bin
// q are atoms[first[q]..first[q+1]]
struct CellGrid {
    CellSearch search;
    std::vector<int> image, bin, first, atoms;
};

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
            if (pbc[d]) throw std::runtime_error("Symmetrix: a periodic direction without a cell vector");
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
    if (not (std::abs(det) > 0)) throw std::runtime_error("Symmetrix: the cell is singular");
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

    int* nb = g.search.nb;
    int* reach = g.search.reach;
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
        g.search.pbc[d] = pbc[d];
    std::copy(cell, cell + 9, g.search.cell);
    g.search.cutoff2 = cutoff*cutoff;
    return g;
}

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Symmetrix neighbor list: ") + cudaGetErrorString(e));
}

}  // namespace

void GpuNeighborList::build(int n, const double* xh, const double* cell, const bool* pbc, double cutoff,
                            NeighborList& nl)
{
    const CellGrid g = cell_grid(n, xh, cell, pbc, cutoff);
    const cudaStream_t s = nullptr;  // the legacy stream, which synchronizes with the blocking ones
    upload(x, xh, 3*size_t(n), s);
    upload(image, g.image.data(), g.image.size(), s);
    upload(bin, g.bin.data(), g.bin.size(), s);
    upload(first, g.first.data(), g.first.size(), s);
    upload(atoms, g.atoms.data(), g.atoms.size(), s);
    count.reserve(n);
    start.reserve(n + 1);

    const CellSearch& c = g.search;
    neighbor_search(n, c, x, image, bin, first, atoms, nullptr, count, nullptr, nullptr, s);
    scan(n, count, start, scratch, s);
    nl.start.resize(n + 1);
    check(cudaMemcpyAsync(nl.start.data(), start, (n + 1)*sizeof(int), cudaMemcpyDeviceToHost, s));
    check(cudaStreamSynchronize(s));
    const size_t E = nl.start[n];
    j.reserve(E);
    S.reserve(3*E);
    neighbor_search(n, c, x, image, bin, first, atoms, start, count, j, S, s);
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
    x.clear(); image.clear(); bin.clear(); first.clear(); atoms.clear(); count.clear(); start.clear(); j.clear();
    S.clear();
    scratch.clear();
}

namespace {
thread_local GpuNeighborList* active_list = nullptr;
}

GpuNeighborList::Scope::Scope(GpuNeighborList& list) : previous(active_list)
{
    active_list = &list;
}

GpuNeighborList::Scope::~Scope()
{
    active_list = previous;
}

GpuNeighborList* GpuNeighborList::active()
{
    return active_list;
}

}  // namespace symmetrix::gpu
