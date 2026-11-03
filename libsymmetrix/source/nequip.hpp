#include <array>
#include <memory>
#include <string>
#include <vector>
#include <span>

#include "cubic_spline_set.hpp"

template <typename Precision>
class NequIP {

public:

NequIP(std::string filename);

// Basic model information
int num_elements;
int num_layers;
double r_cut;
int l_max, num_lm;
std::vector<int> atomic_numbers;

// Neighbor list
std::vector<int> first_neigh;

// Node energies and forces
std::vector<double> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  std::span<const int> node_types,
                                  std::span<const int> num_neigh,
                                  std::span<const int> neigh_indices,
                                  std::span<const int> neigh_types,
                                  std::span<const double> xyz,
                                  std::span<const double> r);
void compute_layer(const int t,
                   const int num_nodes,
                   std::span<const int> node_types,
                   std::span<const int> num_neigh,
                   std::span<const int> neigh_indices,
                   std::span<const int> neigh_types,
                   std::span<const double> xyz,
                   std::span<const double> r);
void compute_readout(const int num_nodes,
                     std::span<const int> node_types,
                     std::span<const int> num_neigh,
                     std::span<const int> neigh_types,
                     std::span<const double> xyz,
                     std::span<const double> r);
void reverse_layer(const int t,
                   const int num_nodes,
                   std::span<const int> node_types,
                   std::span<const int> num_neigh,
                   std::span<const int> neigh_indices,
                   std::span<const int> neigh_types,
                   std::span<const double> xyz,
                   std::span<const double> r);

// ZBL
bool has_zbl;
double zbl_prefactor;
double zbl_p;

// Spherical harmonics
std::vector<Precision> Y, Y_grad;

// H
std::vector<std::vector<int>> H_mul, H_offset;
std::vector<std::vector<Precision>> H;
std::vector<Precision> H_adj;

// X
std::vector<std::vector<std::vector<Precision>>> X_weights;
std::vector<std::vector<Precision>> X;
std::vector<Precision> X_adj;

// Phi
std::vector<std::unique_ptr<CubicSplineSet<Precision>>> spl_set;
std::vector<std::vector<int>> Phi_mul, Phi_offset;
std::vector<std::vector<std::array<int,5>>> Phi_spec;
std::vector<std::vector<Precision>> Phi_clebsch_gordan;
std::vector<Precision> Phi, Phi_adj;

// A
std::vector<std::vector<int>> A_mul, A_offset;
std::vector<std::vector<std::vector<Precision>>> A_weights, A_sc_weights;
std::vector<std::vector<Precision>> A;
std::vector<Precision> A_adj;

// Gates
double gate_cst;
double gate_cst_odd;

// Readout
std::vector<Precision> readout_weights;
std::vector<double> per_type_scale, per_type_shift;

// Initializer
void load_from_json(std::string filename);

};
