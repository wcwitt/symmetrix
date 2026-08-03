#include <memory>
#include <string>
#include <vector>
#include <span>
#include <array>

#include "Kokkos_UnorderedMap.hpp"

#include "compact_radial.hpp"
#include "cubic_spline_kokkos.hpp"
#include "cubic_spline_set_kokkos.hpp"
#include "multilayer_perceptron_kokkos.hpp"
#include "multivariate_polynomial.hpp"//TODO
#include "multivariate_polynomial_kokkos.hpp"
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
int L_max, num_LM;
Kokkos::View<int*> atomic_numbers;
Kokkos::View<double*> atomic_energies;
std::vector<int> atomic_numbers_host;
std::vector<int> active_atomic_numbers;
void prepare_active_types(std::vector<int> node_types);

// Node energies and forces
Kokkos::View<double*> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  Kokkos::View<const int*> node_types,
                                  Kokkos::View<const int*> num_neigh,
                                  Kokkos::View<const int*> neigh_indices,
                                  Kokkos::View<const int*> neigh_types,
                                  Kokkos::View<const double*> xyz,
                                  Kokkos::View<const double*> r);
void compute_node_energies_forces_field(const int num_nodes,
                                        Kokkos::View<const int*> node_types,
                                        Kokkos::View<const int*> num_neigh,
                                        Kokkos::View<const int*> neigh_indices,
                                        Kokkos::View<const int*> neigh_types,
                                        Kokkos::View<const double*> xyz,
                                        Kokkos::View<const double*> r,
                                        Kokkos::View<const double*> electric_field);

// ZBL
bool has_zbl;
ZBLKokkos zbl;

// R0
bool uses_compact_radial = false;
std::unique_ptr<CompactRadialModel> compact_radial_model;
std::vector<int> active_types;
Kokkos::View<int*> type_to_active;
int num_active_types = 0;
std::vector<double> H0_weights_host;
std::vector<std::vector<std::vector<double>>> A0_weights_host;
double R0_spline_h;
double R0_spline_min = 0.0;
Kokkos::View<const Precision****,Kokkos::LayoutRight> R0_spline_coefficients;
Kokkos::View<Precision**,Kokkos::LayoutRight> R0, R0_deriv;
void compute_R0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> r);

// R1
RadialFunctionSetKokkos<Precision> radial_1;
Kokkos::View<Precision**,Kokkos::LayoutRight> R1, R1_deriv;
void compute_R1(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> r);

// Spherical harmonics
Kokkos::View<Precision*> xyz_shuffled;
Kokkos::View<Precision*> Y, Y_grad;// TODO: make multidimensional
Kokkos::View<Precision*> Y_grad_shuffled;
void compute_Y(Kokkos::View<const double*> xyz);

// A0
Kokkos::View<Precision***,Kokkos::LayoutRight> A0, A0_adj;
void compute_A0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types);
void reverse_A0(const int num_nodes,
                Kokkos::View<const int*> node_types,
                Kokkos::View<const int*> num_neigh,
                Kokkos::View<const int*> neigh_types,
                Kokkos::View<const double*> xyz,
                Kokkos::View<const double*> r);

// A0 rescaling
bool A0_scaled;
RadialFunctionSetKokkos<double> A0_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A0_spline_derivs;
void compute_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// M0
Kokkos::View<Precision***,Kokkos::LayoutRight> M0, M0_adj;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_monomials;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_weights;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_spec;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_coeff;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_values;
Kokkos::View<Kokkos::View<Precision***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_adjoints;
void compute_M0(const int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M0(const int num_nodes, Kokkos::View<const int*> node_types);

// H1
Kokkos::View<Precision***,Kokkos::LayoutRight> H1, H1_adj, H1_pre_linear_up;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_weights, H1_product_weights, H1_linear_up_weights;
void compute_H1(const int num_nodes);
void reverse_H1(const int num_nodes);
void compute_H1_product(const int num_nodes);
void compute_H1_linear_up(const int num_nodes);
void reverse_H1_linear_up(const int num_nodes);
void reverse_H1_product(const int num_nodes);

// MACEField coupling after H1 product
bool has_field_coupling;
Kokkos::View<Precision***,Kokkos::LayoutRight> H1_pre_field;
Kokkos::View<Precision**,Kokkos::LayoutRight> field_delta_scalar;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_delta_vector;
Kokkos::View<Precision**,Kokkos::LayoutRight> field_linear_scalar;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_linear_vector;
Kokkos::View<Precision*> field_feats_weight;
Kokkos::View<Precision*> field_feats_output_mask;
Kokkos::View<Precision*> field_linear_weight;
Kokkos::View<Precision*> field_linear_bias;
Kokkos::View<Precision*> field_linear_output_mask;
Kokkos::View<Precision**,Kokkos::LayoutRight> field_scalar_to_vector_up_matrix;
Kokkos::View<Precision**,Kokkos::LayoutRight> field_vector_to_scalar_up_matrix;
Kokkos::View<double*> electric_field_adj;
Kokkos::View<double*> electric_field_hessian;
Kokkos::View<double*> electric_field_force_derivative;
Kokkos::View<Precision**,Kokkos::LayoutRight> field_delta_scalar_adj;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_delta_vector_adj;
Kokkos::View<Precision***,Kokkos::LayoutRight> field_H1_pre_adj;
double field_feats_scalar_to_vector_path_weight;
double field_feats_vector_to_scalar_path_weight;
double field_linear_scalar_path_weight;
double field_linear_vector_path_weight;
void compute_field_H1(const int num_nodes, Kokkos::View<const double*> electric_field);
void reverse_field_H1(const int num_nodes, Kokkos::View<const double*> electric_field);
void compute_electric_field_hessian(const int num_nodes,
                                    Kokkos::View<const int*> node_types,
                                    Kokkos::View<const int*> num_neigh,
                                    Kokkos::View<const int*> neigh_indices,
                                    Kokkos::View<const int*> neigh_types,
                                    Kokkos::View<const double*> xyz,
                                    Kokkos::View<const double*> r,
                                    Kokkos::View<const double*> electric_field);
void compute_electric_field_force_derivative(const int num_nodes,
                                             Kokkos::View<const int*> node_types,
                                             Kokkos::View<const int*> num_neigh,
                                             Kokkos::View<const int*> neigh_indices,
                                             Kokkos::View<const int*> neigh_types,
                                             Kokkos::View<const double*> xyz,
                                             Kokkos::View<const double*> r,
                                             Kokkos::View<const double*> electric_field);

// Phi1
int num_lelm1lm2, num_lme;
Kokkos::View<int*> Phi1_l, Phi1_l1, Phi1_l2;
Kokkos::View<int*> Phi1_lme, Phi1_lelm1lm2;
Kokkos::View<Precision*> Phi1_clebsch_gordan;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1r, dPhi1r;
Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1, dPhi1;
void compute_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices);
void reverse_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices, Kokkos::View<const double*> xyz, Kokkos::View<const double*> r, bool zero_dxyz = true, bool zero_H1_adj = true);

// TODO for testing of Phi1 strategies
Kokkos::View<int*> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;

// A1
Kokkos::View<Precision***,Kokkos::LayoutRight> A1, A1_adj;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights;
Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights_trans;
void compute_A1(int num_nodes);
void reverse_A1(int num_nodes);

// A1 rescaling
bool A1_scaled;
RadialFunctionSetKokkos<double> A1_splines;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_values;
Kokkos::View<double**,Kokkos::LayoutRight> A1_spline_derivs;
void compute_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r);
void reverse_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r);

// M1
Kokkos::View<Precision**,Kokkos::LayoutRight> M1, M1_adj;
Kokkos::View<int**,Kokkos::LayoutRight> M1_monomials;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_weights;
Kokkos::View<int**,Kokkos::LayoutRight> M1_poly_spec;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_coeff;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_values;
Kokkos::View<Precision***,Kokkos::LayoutRight> M1_poly_adjoints;
void compute_M1(int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M1(int num_nodes, Kokkos::View<const int*> node_types);

// H2
Kokkos::View<double**,Kokkos::LayoutRight> H2, H2_adj;
Kokkos::View<double**,Kokkos::LayoutRight> H2_weights_for_H1;
Kokkos::View<double*> H2_weights_for_M1;
void compute_H2(int num_nodes, Kokkos::View<const int*> node_types);
void reverse_H2(int num_nodes, Kokkos::View<const int*> node_types, bool zero_H1_adj = true);

// Readouts
Kokkos::View<double*> readout_1_weights;
MultilayerPerceptronKokkos readout_2;
Kokkos::View<double*> readout_2_output;
double compute_readouts(int num_nodes, const Kokkos::View<const int*> node_types);

// Initializer
void load_from_json(std::string filename);

};
