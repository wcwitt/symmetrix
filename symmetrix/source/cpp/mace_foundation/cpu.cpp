
#include "evaluator.hpp"
#include "mace_foundation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using mace_foundation::NeighborList;
using mace_foundation::neighbor_list;

namespace {

template <class Precision>
struct MACEFoundationCPU : Evaluator {
    MACEFoundationCPU(const std::string& file, double skin);
    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                   double* forces, double* virial, double* energies) override;
    template <class Sh> void compute_node_energies_forces();

    MACEFoundation<Precision> mace;
    double skin;
    int natoms = -1;
    std::vector<double> x0, cell0;
    std::vector<int> numbers0;
    std::array<bool, 3> pbc0{};
    NeighborList nl;
};

template <class Precision>
MACEFoundationCPU<Precision>::MACEFoundationCPU(const std::string& file, double skin) : mace(file, "MACEFoundation: "),
skin(skin)
{
}

template <class Precision>
double MACEFoundationCPU<Precision>::compute(int n, const double* x, const int* numbers,
                                             const double* cell, const bool* pbc,
                                         double* forces, double* virial, double* energies) {
    std::fill(virial, virial + 6, 0.0);
    if (n == 0) return 0;

    // nodes
    bool rebuild = n != natoms or not std::equal(numbers, numbers + n, numbers0.begin()) or not std::equal(cell,
            cell + 9, cell0.begin())
                   or not std::equal(pbc, pbc + 3, pbc0.begin());
    for (int i = 0; not rebuild and i < n; ++i) {
        const double dx = x[3*i] - x0[3*i];
        const double dy = x[3*i+1] - x0[3*i+1];
        const double dz = x[3*i+2] - x0[3*i+2];
        rebuild = 4*(dx*dx + dy*dy + dz*dz) > skin*skin;
    }
    if (rebuild) {
        const int T = mace.num_elements;
        natoms = -1;
        mace.elem_all.resize(n);
        for (int i = 0; i < n; ++i) {
            const auto it = std::find(mace.atomic_numbers.begin(), mace.atomic_numbers.end(), numbers[i]);
            if (it == mace.atomic_numbers.end())
                throw std::runtime_error("MACEFoundation: atomic number " + std::to_string(numbers[i])
                                         + " is not an element of the model");
            mace.elem_all[i] = int(it - mace.atomic_numbers.begin());
        }
        neighbor_list(n, x, cell, pbc, mace.r_cut + skin, nl);
        mace.atom_of_node.clear();
        mace.segments.clear();
        for (int t = 0; t < T; ++t) {
            const int first = int(mace.atom_of_node.size());
            for (int i = 0; i < n; ++i) if (mace.elem_all[i] == t) mace.atom_of_node.push_back(i);
            if (int(mace.atom_of_node.size()) > first)
                mace.segments.push_back({first, int(mace.atom_of_node.size()), t});
        }
        mace.N = n;
        mace.NR = n;
        mace.elem.resize(mace.N);
        mace.row_of_atom.resize(n);
        for (int k = 0; k < mace.N; ++k) {
            mace.elem[k] = mace.elem_all[mace.atom_of_node[k]];
            mace.row_of_atom[mace.atom_of_node[k]] = k;
        }
        mace.locks.reset(new std::atomic_flag[mace.N]);
        natoms = n;
        x0.assign(x, x + 3*(size_t)n);
        cell0.assign(cell, cell + 9);
        numbers0.assign(numbers, numbers + n);
        pbc0 = {pbc[0], pbc[1], pbc[2]};
    }

    // edges
    const int T = mace.num_elements;
    std::vector<int> cnt((long)mace.N*T, 0);
    bool short_pair = false;
    auto candidate = [&](int i, int c, double& dx, double& dy, double& dz) {
        const int j = nl.j[c];
        const int* s = nl.S.data() + 3*c;
        dx = x[3*j] - x[3*i] + s[0]*cell[0] + s[1]*cell[3] + s[2]*cell[6];
        dy = x[3*j+1] - x[3*i+1] + s[0]*cell[1] + s[1]*cell[4] + s[2]*cell[7];
        dz = x[3*j+2] - x[3*i+2] + s[0]*cell[2] + s[1]*cell[5] + s[2]*cell[8];
        return j;
    };
    const double r_cut2 = mace.r_cut*mace.r_cut;
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < mace.N; ++nn) {
        const int i = mace.atom_of_node[nn];
        for (int c = nl.start[i]; c < nl.start[i + 1]; ++c) {
            double dx;
            double dy;
            double dz;
            const int j = candidate(i, c, dx, dy, dz);
            if (dx*dx + dy*dy + dz*dz < r_cut2) ++cnt[(long)nn*T + mace.elem_all[j]];
        }
    }
    mace.start.assign(mace.N + 1, 0);
    mace.seg.assign((long)mace.N*(T + 1), 0);
    for (int nn = 0; nn < mace.N; ++nn) {
        int o = mace.start[nn];
        for (int b = 0; b < T; ++b) {
            mace.seg[(long)nn*(T + 1) + b] = o;
            o += cnt[(long)nn*T + b];
        }
        mace.seg[(long)nn*(T + 1) + T] = o;
        mace.start[nn + 1] = o;
    }
    const long E = mace.start[mace.N];
    mace.geometry.resize(4*E);
    mace.neigh_indices.resize(E);
    mace.row_pair.resize(2*E);
    mace.edge_forces.resize(3*E);
#pragma omp parallel reduction(||:short_pair)
    {
    std::vector<int> off(T);
#pragma omp for schedule(static)
    for (int nn = 0; nn < mace.N; ++nn) {
        const int i = mace.atom_of_node[nn];
        const int a = mace.elem[nn];
        for (int b = 0; b < T; ++b) off[b] = mace.seg[(long)nn*(T + 1) + b];
        for (int c = nl.start[i]; c < nl.start[i + 1]; ++c) {
            double dx;
            double dy;
            double dz;
            const int j = candidate(i, c, dx, dy, dz);
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= r_cut2) continue;
            const int b = mace.elem_all[j];
            const int e = off[b]++;
            const double r = std::sqrt(r2);
            const int p = mace.ordered ? a*T + b : pair_index(a, b, T);
            mace.geometry[4*e] = Precision(dz/r);
            mace.geometry[4*e+1] = Precision(dx/r);
            mace.geometry[4*e+2] = Precision(dy/r);
            mace.geometry[4*e+3] = Precision(r);
            mace.neigh_indices[e] = j;
            mace.row_pair[2*e] = mace.row_of_atom[j];
            mace.row_pair[2*e+1] = p;
            if (r < mace.r_min) short_pair = true;
        }
    }
    }
    if (short_pair) throw std::runtime_error("MACEFoundation: a pair is closer than the model's r_min");

    with_preset(mace.num_channels, mace.L_max, mace.num_channels_node, [&](auto s) {
        this->template compute_node_energies_forces<decltype(s)>();
    });

    // forces
    std::fill(forces, forces + 3*(long)n, 0.0);
    double energy = 0;
    double vir[6] = {};
    for (int nn = 0; nn < mace.N; ++nn) {
        const int i = mace.atom_of_node[nn];
        energy += mace.node_energies[nn];
        if (energies) energies[i] = mace.node_energies[nn];
        for (int e = mace.start[nn]; e < mace.start[nn + 1]; ++e) {
            const int j = mace.neigh_indices[e];
            const double f0 = mace.edge_forces[3*e];
            const double f1 = mace.edge_forces[3*e+1];
            const double f2 = mace.edge_forces[3*e+2];
            forces[3*j] += f0;
            forces[3*j+1] += f1;
            forces[3*j+2] += f2;
            forces[3*i] -= f0;
            forces[3*i+1] -= f1;
            forces[3*i+2] -= f2;
            const double dx = double(mace.geometry[4*e+1])*mace.geometry[4*e+3];
            const double dy = double(mace.geometry[4*e+2])*mace.geometry[4*e+3];
            const double dz = double(mace.geometry[4*e])*mace.geometry[4*e+3];
            vir[0] += dx*f0;
            vir[1] += dy*f1;
            vir[2] += dz*f2;
            vir[3] += 0.5*(dx*f1 + dy*f0);
            vir[4] += 0.5*(dx*f2 + dz*f0);
            vir[5] += 0.5*(dy*f2 + dz*f1);
        }
    }
    std::copy(vir, vir + 6, virial);
    return energy;
}

template <class Precision>
template <class Sh>
void MACEFoundationCPU<Precision>::compute_node_energies_forces()
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    mace.template compute_first_layer<Sh>();
    mace.template compute_second_layer<Sh>();
    mace.template compute_readouts<Sh>();
    mace.template reverse_second_layer<Sh>();
    mace.template reverse_first_layer<Sh>();
}

}  // namespace

std::unique_ptr<Evaluator> mace_foundation_cpu(const std::string& file, const std::string& dtype, double skin)
{
    if (dtype == "float32") return std::make_unique<MACEFoundationCPU<float>>(file, skin);
    return std::make_unique<MACEFoundationCPU<double>>(file, skin);
}
