#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "evaluator.hpp"

namespace mace_foundation {

void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl)
{
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
    std::vector<int> k(3*(size_t)n, 0);
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) {
            const double f = x[3*i]*b[d][0] + x[3*i+1]*b[d][1] + x[3*i+2]*b[d][2];
            if (pbc[d]) k[3*i + d] = int(std::floor(f));
            u[3*i + d] = f - k[3*i + d];
        }

    int nb[3];
    int reach[3];
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
    std::vector<int> first(nb[0]*nb[1]*nb[2] + 1, 0);
    std::vector<int> atoms(n);
    std::vector<int> c(3*(size_t)n);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[3*i + d] = std::min(nb[d] - 1, std::max(0, int((u[3*i + d] - lo[d])/w[d])));
        bin[i] = (c[3*i]*nb[1] + c[3*i+1])*nb[2] + c[3*i+2];
        ++first[bin[i] + 1];
    }
    for (size_t q = 1; q < first.size(); ++q) first[q] += first[q - 1];
    std::vector<int> fill(first.begin(), first.end() - 1);
    for (int i = 0; i < n; ++i) atoms[fill[bin[i]]++] = i;

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

}  // namespace mace_foundation

namespace py = pybind11;

template <class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(_mace_foundation, m)
{
    py::class_<Evaluator>(m, "MACEFoundation")
        .def(py::init([](const std::string& file, const std::string& device, const std::string& dtype,
                         double skin) -> std::unique_ptr<Evaluator> {
                 if (dtype != "float64" and dtype != "float32" and dtype != "float32-mixed")
                     throw std::invalid_argument("dtype must be 'float64', 'float32' or 'float32-mixed', "
                                                 "not '" + dtype + "'");
                 if (device == "cpu") {
                     if (dtype == "float32-mixed")
                         throw std::invalid_argument("dtype must be 'float64' or 'float32' on the CPU");
                     return mace_foundation_cpu(file, dtype, skin);
                 }
                 if (device != "cuda")
                     throw std::invalid_argument("device must be 'cpu' or 'cuda', not '" + device + "'");
#ifdef MACE_FOUNDATION_CUDA
                 return mace_foundation_cuda(file, dtype, skin);
#else
                 throw std::runtime_error("symmetrix was built without CUDA");
#endif
             }),
             py::arg("model_file"), py::arg("device") = "cuda", py::arg("dtype") = "float64", py::arg("skin") = 1.0)
        .def("compute",
             [](Evaluator& self, Array<double> x, Array<int> numbers, Array<double> cell,
                Array<bool> pbc, bool energies) {
            const int n = int(numbers.size());
            if (x.size() != 3*n or cell.size() != 9 or pbc.size() != 3)
                throw std::invalid_argument("expected x[n][3], numbers[n], cell[3][3], pbc[3]");
            Array<double> forces({n, 3});
            Array<double> virial(6);
            Array<double> e(energies ? n : 0);
            double energy;
            {
                py::gil_scoped_release release;
                energy = self.compute(n, x.data(), numbers.data(), cell.data(), pbc.data(), forces.mutable_data(),
                                      virial.mutable_data(),
                                      energies ? e.mutable_data() : nullptr);
            }
            return py::make_tuple(energy, forces, virial, energies ? py::object(e) : py::none());
        }, py::arg("x"), py::arg("numbers"), py::arg("cell"), py::arg("pbc"), py::arg("energies") = false);

    m.def("neighbor_list", [](Array<double> x, Array<double> cell, Array<bool> pbc, double cutoff) {
        const int n = int(x.size()/3);
        mace_foundation::NeighborList nl;
        mace_foundation::neighbor_list(n, x.data(), cell.data(), pbc.data(), cutoff, nl);
        const long p = long(nl.j.size());
        Array<int> i(p);
        Array<int> j(p);
        Array<int> S({p, 3L});
        for (int a = 0; a < n; ++a)
            for (int q = nl.start[a]; q < nl.start[a + 1]; ++q) i.mutable_data()[q] = a;
        std::copy(nl.j.begin(), nl.j.end(), j.mutable_data());
        std::copy(nl.S.begin(), nl.S.end(), S.mutable_data());
        return py::make_tuple(i, j, S);
    });
}
