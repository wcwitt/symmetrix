#include <array>
#include <memory>
#include <string>
#include <vector>
#include <span>

#include "cubic_spline.hpp"
#include "cubic_spline_set.hpp"
#include "multilayer_perceptron.hpp"
#include "zbl.hpp"

template <typename Precision>
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
void compute_first_layer(const int num_nodes,
                         std::span<const int> node_types,
                         std::span<const int> num_neigh,
                         std::span<const int> neigh_types,
                         std::span<const double> xyz,
                         std::span<const double> r);
void compute_second_layer(const int num_nodes,
                          std::span<const int> node_types,
                          std::span<const int> num_neigh,
                          std::span<const int> neigh_indices,
                          std::span<const int> neigh_types,
                          std::span<const double> r);
void compute_readouts(const int num_nodes,
                      std::span<const int> node_types,
                      std::span<const int> num_neigh,
                      std::span<const int> neigh_types,
                      std::span<const double> xyz,
                      std::span<const double> r);
void reverse_second_layer(const int num_nodes,
                          std::span<const int> node_types,
                          std::span<const int> num_neigh,
                          std::span<const int> neigh_indices,
                          std::span<const int> neigh_types,
                          std::span<const double> xyz,
                          std::span<const double> r);
void reverse_first_layer(const int num_nodes,
                         std::span<const int> node_types,
                         std::span<const int> num_neigh,
                         std::span<const int> neigh_types,
                         std::span<const double> xyz,
                         std::span<const double> r);

// ZBL
bool has_zbl;
ZBL zbl;

// Radial functions
std::vector<std::unique_ptr<CubicSplineSet<Precision>>> spl_set_0;
double R0_spline_r_min;

// R1
std::vector<std::unique_ptr<CubicSplineSet<Precision>>> spl_set_1;

// Spherical harmonics
std::vector<Precision> Y, Y_grad;

// H0
std::vector<Precision> H0_weights;

// A0
std::vector<Precision> A0, A0_adj;
std::vector<std::vector<std::vector<Precision>>> A0_weights;

// A0 rescaling
bool A0_scaled;
std::vector<CubicSpline> A0_splines;

// M0
std::vector<Precision> M0, M0_adj;
std::vector<Precision> B0;
std::vector<std::array<int,3>> B0_spec, M0_spec;
std::vector<Precision> B0_coefficients, M0_weights;
int num_B0;

// H1
std::vector<Precision> H1, H1_adj;
std::vector<Precision> H1_weights;

// Phi1
int num_lme;
std::vector<Precision> Phi1, dPhi1;
std::vector<int> Phi1_l;
std::vector<std::array<int,3>> Phi1_spec;
std::vector<std::array<int,2>> Phi1_cg_spec;
std::vector<Precision> Phi1_clebsch_gordan;

// A1
std::vector<Precision> A1, A1_adj;
std::vector<std::vector<Precision>> A1_weights, A1_weights_trans;

// A1 rescaling
bool A1_scaled;
std::vector<CubicSpline> A1_splines;

// M1
std::vector<Precision> M1, M1_adj;
std::vector<Precision> B1;
std::vector<std::array<int,3>> B1_spec, M1_spec;
std::vector<Precision> B1_coefficients, M1_weights;
int num_B1;

// H2
std::vector<Precision> H2, H2_adj;
std::vector<std::vector<Precision>> H2_weights_for_M0;
std::vector<Precision> H2_weights_for_M1;

// Readouts
std::vector<Precision> readout_1_weights_for_M0;
std::unique_ptr<MultilayerPerceptron> readout_2;

// Initializer
void load_from_json(std::string filename);

};
