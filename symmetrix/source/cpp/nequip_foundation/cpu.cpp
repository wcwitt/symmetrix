#include "evaluator.hpp"
#include "nequip_foundation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Precision>
struct NequIPFoundationCPU : Evaluator {
    NequIPFoundationCPU(const std::string& file, double skin) : nequip(file, "NequIPFoundation: "), skin(skin) {}
    double compute(int n, const double* pos, const int* numbers, const double* cell, const bool* pbc,
                   double* forces, double* virial, double* energies) override;
    void compute_node_energies_forces();

    NequIPFoundation<Precision> nequip;
    double skin;
    int natoms = -1;
    std::vector<double> x0, cell0;
    std::vector<int> numbers0;
    std::array<bool, 3> pbc0{};
    NeighborList nl;
};

template <typename Precision>
double NequIPFoundationCPU<Precision>::compute(int n, const double* pos, const int* numbers, const double* cell,
                                               const bool* pbc, double* forces, double* virial, double* energies)
{
    std::fill(virial, virial + 6, 0.0);
    if (n == 0) return 0;
    const int T = nequip.num_elements;

    bool rebuild = n != natoms or not std::equal(numbers, numbers + n, numbers0.begin())
                   or not std::equal(cell, cell + 9, cell0.begin())
                   or not std::equal(pbc, pbc + 3, pbc0.begin());
    for (int i = 0; not rebuild and i < n; ++i) {
        const double dx = pos[3*i] - x0[3*i];
        const double dy = pos[3*i+1] - x0[3*i+1];
        const double dz = pos[3*i+2] - x0[3*i+2];
        rebuild = 4*(dx*dx + dy*dy + dz*dz) > skin*skin;
    }
    if (rebuild) {
        natoms = -1;
        nequip.elem_all.resize(n);
        for (int i = 0; i < n; ++i) {
            const auto it = std::find(nequip.atomic_numbers.begin(), nequip.atomic_numbers.end(), numbers[i]);
            if (it == nequip.atomic_numbers.end())
                throw std::runtime_error("NequIPFoundation: atomic number " + std::to_string(numbers[i])
                                         + " is not an element of the model");
            nequip.elem_all[i] = int(it - nequip.atomic_numbers.begin());
        }
        neighbor_list(n, pos, cell, pbc, nequip.r_cut + skin, nl);
        nequip.atom_of_node.clear();
        nequip.segments.clear();
        for (int t = 0; t < T; ++t) {
            const int first = int(nequip.atom_of_node.size());
            for (int i = 0; i < n; ++i)
                if (nequip.elem_all[i] == t) nequip.atom_of_node.push_back(i);
            if (int(nequip.atom_of_node.size()) > first)
                nequip.segments.push_back({first, int(nequip.atom_of_node.size()), t});
        }
        nequip.N = n;
        nequip.NR = n;
        nequip.elem.resize(nequip.N);
        nequip.row_of_atom.resize(n);
        for (int k = 0; k < nequip.N; ++k) {
            nequip.elem[k] = nequip.elem_all[nequip.atom_of_node[k]];
            nequip.row_of_atom[nequip.atom_of_node[k]] = k;
        }
        nequip.order.resize(nequip.N);
        for (int i = 0; i < n; ++i) nequip.order[i] = nequip.row_of_atom[i];
        nequip.locks.reset(new std::atomic_flag[nequip.NR]());
        natoms = n;
        x0.assign(pos, pos + 3*(size_t)n);
        cell0.assign(cell, cell + 9);
        numbers0.assign(numbers, numbers + n);
        pbc0 = {pbc[0], pbc[1], pbc[2]};
    }

    {
        const double r_cut2 = nequip.r_cut*nequip.r_cut;
        nequip.start.assign(nequip.N + 1, 0);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < nequip.N; ++n) {
            const int i = nequip.atom_of_node[n];
            int count = 0;
            for (int c = nl.start[i]; c < nl.start[i + 1]; ++c) {
                const int j = nl.j[c];
                const int* S = nl.S.data() + 3*c;
                const double dx = pos[3*j] - pos[3*i] + S[0]*cell[0] + S[1]*cell[3] + S[2]*cell[6];
                const double dy = pos[3*j+1] - pos[3*i+1] + S[0]*cell[1] + S[1]*cell[4] + S[2]*cell[7];
                const double dz = pos[3*j+2] - pos[3*i+2] + S[0]*cell[2] + S[1]*cell[5] + S[2]*cell[8];
                if (dx*dx + dy*dy + dz*dz < r_cut2) ++count;
            }
            nequip.start[n + 1] = count;
        }
        for (int n = 0; n < nequip.N; ++n) nequip.start[n + 1] += nequip.start[n];
        const long E = nequip.start[nequip.N];
        nequip.neigh_indices.resize(E);
        nequip.neigh_rows.resize(E);
        nequip.neigh_elem.resize(E);
        nequip.xyz.resize(3*E);
        nequip.r.resize(E);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < nequip.N; ++n) {
            const int i = nequip.atom_of_node[n];
            long e = nequip.start[n];
            for (int c = nl.start[i]; c < nl.start[i + 1]; ++c) {
                const int j = nl.j[c];
                const int* S = nl.S.data() + 3*c;
                const double dx = pos[3*j] - pos[3*i] + S[0]*cell[0] + S[1]*cell[3] + S[2]*cell[6];
                const double dy = pos[3*j+1] - pos[3*i+1] + S[0]*cell[1] + S[1]*cell[4] + S[2]*cell[7];
                const double dz = pos[3*j+2] - pos[3*i+2] + S[0]*cell[2] + S[1]*cell[5] + S[2]*cell[8];
                const double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= r_cut2) continue;
                nequip.neigh_indices[e] = j;
                nequip.neigh_rows[e] = nequip.row_of_atom[j];
                nequip.neigh_elem[e] = nequip.elem_all[j];
                nequip.xyz[3*e] = dx;
                nequip.xyz[3*e+1] = dy;
                nequip.xyz[3*e+2] = dz;
                nequip.r[e] = std::sqrt(r2);
                ++e;
            }
        }
    }

    compute_node_energies_forces();

    {
        std::fill(forces, forces + 3*(long)n, 0.0);
        double energy = 0;
        double vir[6] = {};
        for (int nn = 0; nn < nequip.N; ++nn) {
            const int i = nequip.atom_of_node[nn];
            energy += nequip.node_energies[nn];
            if (energies) energies[i] = nequip.node_energies[nn];
            for (int e = nequip.start[nn]; e < nequip.start[nn + 1]; ++e) {
                const int j = nequip.neigh_indices[e];
                const double f0 = nequip.edge_forces[3*e];
                const double f1 = nequip.edge_forces[3*e+1];
                const double f2 = nequip.edge_forces[3*e+2];
                forces[3*j] += f0;
                forces[3*j+1] += f1;
                forces[3*j+2] += f2;
                forces[3*i] -= f0;
                forces[3*i+1] -= f1;
                forces[3*i+2] -= f2;
                const double dx = nequip.xyz[3*e];
                const double dy = nequip.xyz[3*e+1];
                const double dz = nequip.xyz[3*e+2];
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
}

template <typename Precision>
void NequIPFoundationCPU<Precision>::compute_node_energies_forces()
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    // \lambda = (l, p),  H^{(0)}_i = H^{(0)}_{a_i},  X^{(0)}_j = X^{(0)}_{a_j},  t = 0, \dots, T-1
    for (int t = 0; t < nequip.num_layers; ++t)
        nequip.compute_layer(t);
    nequip.compute_readout();
    for (int t = nequip.num_layers - 1; t >= 0; --t)
        nequip.reverse_layer(t);
}

}    // namespace

std::unique_ptr<Evaluator> nequip_foundation_cpu(const std::string& file, const std::string& dtype, double skin)
{
    if (dtype == "float32") return std::make_unique<NequIPFoundationCPU<float>>(file, skin);
    return std::make_unique<NequIPFoundationCPU<double>>(file, skin);
}
