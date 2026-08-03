#include <memory>
#include <string>
#include <vector>
#include <span>

#include "compact_radial.hpp"
#include "cubic_spline.hpp"
#include "cubic_spline_set.hpp"
#include "multilayer_perceptron.hpp"
#include "multivariate_polynomial.hpp"
#include "zbl.hpp"

class MACE {

public:

MACE(std::string filename);

// Basic model information
int num_elements;
int num_channels;
double r_cut;
int l_max, num_lm;
int L_max, num_LM;
std::vector<int> atomic_numbers;
std::vector<double> atomic_energies;
std::vector<int> active_atomic_numbers;
void prepare_active_types(std::span<const int> node_types);

// Node energies and forces
std::vector<double> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  std::span<const int> node_types,
                                  std::span<const int> num_neigh,
                                  std::span<const int> neigh_indices,
                                  std::span<const int> neigh_types,
                                  std::span<const double> xyz,
                                  std::span<const double> r);
void compute_node_energies_forces_field(const int num_nodes,
                                        std::span<const int> node_types,
                                        std::span<const int> num_neigh,
                                        std::span<const int> neigh_indices,
                                        std::span<const int> neigh_types,
                                        std::span<const double> xyz,
                                        std::span<const double> r,
                                        std::span<const double> electric_field);

// ZBL
bool has_zbl;
ZBL zbl;

// Radial functions
bool uses_compact_radial = false;
std::unique_ptr<CompactRadialModel> compact_radial_model;
std::vector<int> active_types;
std::vector<int> type_to_active;
std::vector<std::unique_ptr<CubicSplineSet>> spl_set_0;
std::vector<double> R0, R0_deriv;
void compute_R0(const int num_nodes,
                std::span<const int> node_types,
                std::span<const int> num_neigh,
                std::span<const int> neigh_types,
                std::span<const double> r);

// R1
std::vector<std::unique_ptr<CubicSplineSet>> spl_set_1;
std::vector<double> R1, R1_deriv;
void compute_R1(const int num_nodes,
                std::span<const int> node_types,
                std::span<const int> num_neigh,
                std::span<const int> neigh_types,
                std::span<const double> r);
int radial_pair_index(int type_i, int type_j) const;

// Spherical harmonics
std::vector<double> Y, Y_grad;
std::vector<double> xyz_shuffled;
void compute_Y(std::span<const double> xyz);

// H0
std::vector<double> H0_weights;

// A0
std::vector<double> A0, A0_adj;
std::vector<std::vector<std::vector<double>>> A0_weights;
void compute_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types);
void reverse_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r);

// A0 rescaling
bool A0_scaled;
std::vector<CubicSpline> A0_splines;
void compute_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r);
void reverse_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r);

// M0
std::vector<double> M0, M0_grad, M0_adj;
std::vector<MultivariatePolynomial> P0;
void compute_M0(const int num_nodes, std::span<const int> node_types);
// TODO: node_types here only for consistency with Kokkos
void reverse_M0(const int num_nodes, std::span<const int> node_types);

// H1
std::vector<double> H1, H1_adj;
std::vector<double> H1_weights;
void compute_H1(const int num_nodes);
void reverse_H1(const int num_nodes);
std::vector<double> H1_product_weights;
std::vector<double> H1_linear_up_weights;
void compute_H1_product(const int num_nodes);
void compute_H1_linear_up(const int num_nodes);
void reverse_H1_linear_up(const int num_nodes);
void reverse_H1_product(const int num_nodes);

// MACEField coupling after H1
bool has_field_coupling;
std::vector<double> H1_pre_field;
std::vector<double> field_feats_weight;
std::vector<double> field_feats_output_mask;
std::vector<double> field_linear_weight;
std::vector<double> field_linear_bias;
std::vector<double> field_linear_output_mask;
std::vector<double> electric_field_adj;
std::vector<double> electric_field_hessian;
std::vector<double> electric_field_force_derivative;
double field_feats_scalar_to_vector_path_weight;
double field_feats_vector_to_scalar_path_weight;
double field_linear_scalar_path_weight;
double field_linear_vector_path_weight;
void compute_field_H1(const int num_nodes, std::span<const double> electric_field);
void reverse_field_H1(const int num_nodes, std::span<const double> electric_field);
void compute_electric_field_hessian(const int num_nodes,
                                    std::span<const int> node_types,
                                    std::span<const int> num_neigh,
                                    std::span<const int> neigh_indices,
                                    std::span<const int> neigh_types,
                                    std::span<const double> xyz,
                                    std::span<const double> r,
                                    std::span<const double> electric_field);
void compute_electric_field_force_derivative(const int num_nodes,
                                             std::span<const int> node_types,
                                             std::span<const int> num_neigh,
                                             std::span<const int> neigh_indices,
                                             std::span<const int> neigh_types,
                                             std::span<const double> xyz,
                                             std::span<const double> r,
                                             std::span<const double> electric_field);

// Phi1
int num_lelm1lm2, num_lme;
std::vector<double> Phi1r, dPhi1r;
std::vector<double> Phi1, dPhi1;
std::vector<int> Phi1_l, Phi1_l1, Phi1_l2;
std::vector<int> Phi1_lme, Phi1_lelm1lm2;
std::vector<double> Phi1_clebsch_gordan;
void compute_Phi1(const int num_nodes, std::span<const int> num_neigh, std::span<const int> neigh_indices);
void reverse_Phi1(const int num_nodes, std::span<const int> num_neigh, std::span<const int> neigh_indices, std::span<const double> xyz, std::span<const double> r, bool zero_dxyz = true, bool zero_H1_adj = true);

// A1
std::vector<double> A1, A1_adj;
std::vector<std::vector<double>> A1_weights;
void compute_A1(const int num_nodes);
void reverse_A1(const int num_nodes);

// A1 rescaling
bool A1_scaled;
std::vector<CubicSpline> A1_splines;
void compute_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r);
void reverse_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz = true);

// M1
std::vector<double> M1, M1_grad, M1_adj;
std::vector<MultivariatePolynomial> P1;
void compute_M1(const int num_nodes, std::span<const int> node_types);
// TODO: node_types here only for consistency with Kokkos
void reverse_M1(const int num_nodes, std::span<const int> node_types);

// H2
std::vector<double> H2, H2_adj;
std::vector<std::vector<double>> H2_weights_for_H1;
std::vector<double> H2_weights_for_M1;
void compute_H2(const int num_nodes, std::span<const int> node_types);
void reverse_H2(const int num_nodes, std::span<const int> node_types, bool zero_H1_adj = true);

// Readouts
std::vector<double> readout_1_weights;
std::unique_ptr<MultilayerPerceptron> readout_2;
void compute_readouts(const int num_nodes, std::span<const int> node_types);

// Initializer
void load_from_json(std::string filename);

};
