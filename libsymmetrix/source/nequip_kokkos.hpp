#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "radial_function_set_kokkos.hpp"

template <typename Precision>
class NequIPKokkos {

public:

NequIPKokkos(std::string filename);
~NequIPKokkos();

// Basic model information
int num_elements;
int num_layers;
double r_cut;
int l_max, num_lm;
Kokkos::View<int*> atomic_numbers;

// Neighbor list
Kokkos::View<int*> first_neigh;

// Node energies and forces
Kokkos::View<double*> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  Kokkos::View<const int*> node_types,
                                  Kokkos::View<const int*> num_neigh,
                                  Kokkos::View<const int*> neigh_indices,
                                  Kokkos::View<const int*> neigh_types,
                                  Kokkos::View<const double*> xyz,
                                  Kokkos::View<const double*> r);
void compute_layer(const int t,
                   const int num_nodes,
                   Kokkos::View<const int*> node_types,
                   Kokkos::View<const int*> num_neigh,
                   Kokkos::View<const int*> neigh_indices,
                   Kokkos::View<const int*> neigh_types,
                   Kokkos::View<const double*> xyz,
                   Kokkos::View<const double*> r);
void compute_readout(const int num_nodes,
                     Kokkos::View<const int*> node_types,
                     Kokkos::View<const int*> num_neigh,
                     Kokkos::View<const int*> neigh_types,
                     Kokkos::View<const double*> xyz,
                     Kokkos::View<const double*> r);
void reverse_layer(const int t,
                   const int num_nodes,
                   Kokkos::View<const int*> node_types,
                   Kokkos::View<const int*> num_neigh,
                   Kokkos::View<const int*> neigh_indices,
                   Kokkos::View<const int*> neigh_types,
                   Kokkos::View<const double*> xyz,
                   Kokkos::View<const double*> r);

// ZBL
bool has_zbl;
double zbl_prefactor;
double zbl_p;

// Spherical harmonics
std::shared_ptr<void> sphericart_calculator;
Kokkos::View<Precision*> xyz_sphericart;
Kokkos::View<Precision*> Y, Y_grad;

// H
std::vector<std::vector<int>> H_mul, H_offset;
std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>> H;
Kokkos::View<Precision**,Kokkos::LayoutRight> H_adj;

// X
std::vector<std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>> X_weights, X_weights_trans;
std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>> X;
Kokkos::View<Precision**,Kokkos::LayoutRight> X_adj;

// Phi
std::vector<RadialFunctionSetKokkos<Precision>> radial;
std::vector<std::vector<int>> Phi_mul, Phi_offset;
std::vector<Kokkos::View<int**,Kokkos::LayoutRight>> Phi_spec, Phi_spec_trans;
std::vector<Kokkos::View<Precision*>> Phi_clebsch_gordan, Phi_clebsch_gordan_trans;
std::vector<Kokkos::View<int*>> Phi_rows, Phi_rows_trans, Phi_columns, Phi_columns_trans;
Kokkos::View<Precision**,Kokkos::LayoutRight> Phi, Phi_adj;

// A
std::vector<std::vector<int>> A_mul, A_offset;
std::vector<std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>>> A_weights, A_weights_trans;
std::vector<std::vector<Kokkos::View<Precision***,Kokkos::LayoutRight>>> A_sc_weights;
std::vector<Kokkos::View<Precision**,Kokkos::LayoutRight>> A;
Kokkos::View<Precision**,Kokkos::LayoutRight> A_adj;

// Gates
double gate_cst;
double gate_cst_odd;

// Readout
Kokkos::View<double*> readout_weights;
Kokkos::View<double*> per_type_scale, per_type_shift;

// Initializer
void load_from_json(std::string filename);

};
