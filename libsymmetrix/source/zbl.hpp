#pragma once

#include <span>
#include <vector>

class ZBL {

public:

ZBL();

ZBL(double a_exp,
    double a_prefactor,
    std::vector<double> c,
    std::vector<double> covalent_radii,
    int p);

double compute(const int Z_u, const int Z_v, const double r);

double compute_gradient(const int Z_u, const int Z_v, const double r);

double compute_envelope(const double r, const double r_max, const int p);

double compute_envelope_gradient(const double r, const double r_max, const int p);

void compute_ZBL(const int num_nodes,
                 std::span<const int> node_types,
                 std::span<const int> num_neigh,
                 std::span<const int> neigh_types,
                 std::span<const int> atomic_numbers,
                 std::span<const double> r,
                 std::span<const double> xyz,
                 std::span<double> node_energies,
                 std::span<double> node_forces);

private:

// values set in constructor
double a_exp;
double a_prefactor;
std::vector<double> c;
std::vector<double> covalent_radii;
int p;
    
// values taken from mace/modules/radial.py
static constexpr double c_exps_0 = -3.2;
static constexpr double c_exps_1 = -0.9423;
static constexpr double c_exps_2 = -0.4028;
static constexpr double c_exps_3 = -0.2016;
static constexpr double v_prefactor = 14.3996;

};
