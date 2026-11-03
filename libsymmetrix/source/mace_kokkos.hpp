#include <memory>
#include <string>
#include <vector>
#include <span>

#include "Kokkos_UnorderedMap.hpp"

#include "cubic_spline_kokkos.hpp"
#include "cubic_spline_set_kokkos.hpp"
#include "multilayer_perceptron_kokkos.hpp"
#include "radial_function_set_kokkos.hpp"
#include "zbl_kokkos.hpp"

template <typename Precision>
class MACEKokkos {

public:

MACEKokkos(std::string filename);
~MACEKokkos();

// Basic model information
int num_elements;
int num_channels;
double r_cut;
int l_max, num_lm;
static constexpr int mace_l_max = 3;
int L_max, num_LM;
Kokkos::View<int*> atomic_numbers;
Kokkos::View<double*> atomic_energies;

// Neighbor list
Kokkos::View<int*> first_neigh, i_list;

// Node energies and forces
Kokkos::View<double*> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  Kokkos::View<const int*> node_types,
                                  Kokkos::View<const int*> num_neigh,
                                  Kokkos::View<const int*> neigh_indices,
                                  Kokkos::View<const int*> neigh_types,
                                  Kokkos::View<const double*> xyz,
                                  Kokkos::View<const double*> r);
void compute_first_layer(const int num_nodes,
                         Kokkos::View<const int*> node_types,
                         Kokkos::View<const int*> num_neigh,
                         Kokkos::View<const int*> neigh_types,
                         Kokkos::View<const double*> xyz,
                         Kokkos::View<const double*> r);
void compute_second_layer(const int num_nodes,
                          Kokkos::View<const int*> node_types,
                          Kokkos::View<const int*> num_neigh,
                          Kokkos::View<const int*> neigh_indices,
                          Kokkos::View<const int*> neigh_types,
                          Kokkos::View<const double*> r);
void compute_readouts(const int num_nodes,
                      Kokkos::View<const int*> node_types,
                      Kokkos::View<const int*> num_neigh,
                      Kokkos::View<const int*> neigh_types,
                      Kokkos::View<const double*> xyz,
                      Kokkos::View<const double*> r);
void reverse_second_layer(const int num_nodes,
                          Kokkos::View<const int*> node_types,
                          Kokkos::View<const int*> num_neigh,
                          Kokkos::View<const int*> neigh_indices,
                          Kokkos::View<const int*> neigh_types,
                          Kokkos::View<const double*> xyz,
                          Kokkos::View<const double*> r);
void reverse_first_layer(const int num_nodes,
                         Kokkos::View<const int*> node_types,
                         Kokkos::View<const int*> num_neigh,
                         Kokkos::View<const int*> neigh_types,
                         Kokkos::View<const double*> xyz,
                         Kokkos::View<const double*> r);

// ZBL
bool has_zbl;
ZBLKokkos zbl;

// R0
double R0_spline_r_min;
double R0_spline_h;
Kokkos::View<const Precision****,Kokkos::LayoutRight> R0_spline_coefficients;

// R1
RadialFunctionSetKokkos<Precision> radial_1;

// Spherical harmonics
std::shared_ptr<void> sphericart_calculator;
Kokkos::View<Precision*> xyz_sphericart;
Kokkos::View<Precision*> Y, Y_grad;// TODO: make multidimensional

// A0
Kokkos::View<Precision***,Kokkos::LayoutRight> A0, A0_adj;

// A0 rescaling
bool A0_scaled;
RadialFunctionSetKokkos<double> A0_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_derivs;

// M0
Kokkos::View<Precision***,Kokkos::LayoutRight> M0, M0_adj;
Kokkos::View<int**,Kokkos::LayoutRight> B0_spec, M0_spec;
Kokkos::View<Precision*> B0_coefficients;
Kokkos::View<Precision***,Kokkos::LayoutRight> M0_weights, B0, B0_adj;

// H1
Kokkos::View<Precision***,Kokkos::LayoutRight> H1, H1_adj;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_weights, H1_weights_trans;

// Phi1
int num_lme;
Kokkos::View<int**,Kokkos::LayoutRight> Phi1_lel1l2_spec;
Kokkos::View<Precision****,Kokkos::LayoutRight> Phi1_cg;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1, dPhi1;

// A1
Kokkos::View<Precision***,Kokkos::LayoutRight> A1, A1_adj;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights_trans;

// A1 rescaling
bool A1_scaled;
RadialFunctionSetKokkos<double> A1_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_derivs;

// M1
Kokkos::View<Precision**,Kokkos::LayoutRight> M1, M1_adj;
Kokkos::View<int**,Kokkos::LayoutRight> B1_spec, M1_spec;
Kokkos::View<Precision*> B1_coefficients;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_weights, B1, B1_adj;

// H2
Kokkos::View<double**,Kokkos::LayoutRight> H2, H2_adj;
Kokkos::View<double**,Kokkos::LayoutRight> H2_weights_for_M0, H2_weights_for_M0_trans;
Kokkos::View<double*> H2_weights_for_M1, H2_weights_for_M1_trans;

// Readouts
Kokkos::View<double*> readout_1_weights_for_M0;
MultilayerPerceptronKokkos readout_2;
Kokkos::View<double*> readout_2_output;

// Initializer
void load_from_json(std::string filename);

};
