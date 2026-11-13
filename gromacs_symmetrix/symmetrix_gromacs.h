#pragma once

// Symmetrix models for the NNPot interface of GROMACS, with no GROMACS or CUDA types: a Symmetrix JSON, PolarMACE,
// MACE or NequIP by its model_type, evaluated on the GPU with a neighbor list with skin.

#include <memory>
#include <string>
#include <vector>

namespace symmetrix::gromacs {

struct Options {
    // "float32" or "float32-mixed"; foundation models also "float64"
    std::string dtype = "float32";
    // A; the neighbor list is built again when an atom has moved by half of it
    double skin = 2.0;
    // PolarMACE: of the whole system
    double total_charge = 0;
    double spin_multiplicity = 1;
};

class Model {
public:
    static std::unique_ptr<Model> load(const std::string& file, const Options& options);
    virtual ~Model() = default;

    // The energy (eV), forces (eV/A) and virial (eV; xx yy zz xy xz yz) of the n atoms at x (A) in the cell (rows are
    // the lattice vectors, A), periodic along pbc. Between calls, atoms may be put back in the cell by lattice vectors.
    double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc, double* forces,
                   double* virial);

protected:
    // As compute, for positions that move continuously between calls
    virtual double evaluate(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                            double* forces, double* virial) = 0;

private:
    std::vector<double> unwrapped;
};

}  // namespace symmetrix::gromacs
