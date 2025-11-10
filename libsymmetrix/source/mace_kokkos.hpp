#include <memory>
#include <string>
#include <vector>
#include <span>

#include "Kokkos_UnorderedMap.hpp"

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

// Node energies and forces
Kokkos::View<double*> node_energies, node_forces;
void compute_node_energies_forces(const int num_nodes,
                                  Kokkos::View<const int*> node_types,
                                  Kokkos::View<const int*> num_neigh,
                                  Kokkos::View<const int*> neigh_indices,
                                  Kokkos::View<const int*> neigh_types,
                                  Kokkos::View<const double*> xyz,
                                  Kokkos::View<const double*> r);

// ZBL
bool has_zbl;
ZBLKokkos zbl;

// R0
double R0_spline_h;
Kokkos::View<const double****,Kokkos::LayoutRight> R0_spline_coefficients;
Kokkos::View<double**,Kokkos::LayoutRight> R0, R0_deriv;
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
Kokkos::View<double*> xyz_shuffled;
Kokkos::View<double*> Y, Y_grad;// TODO: make multidimensional
Kokkos::View<double*> Y_grad_shuffled;
void compute_Y(Kokkos::View<const double*> xyz);

// A0
Kokkos::View<double***,Kokkos::LayoutRight> A0, A0_adj;
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
Kokkos::View<double***,Kokkos::LayoutRight> M0, M0_adj;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_monomials;
Kokkos::View<Kokkos::View<double***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_weights;
Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_spec;
Kokkos::View<Kokkos::View<double***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_coeff;
Kokkos::View<Kokkos::View<double***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_values;
Kokkos::View<Kokkos::View<double***,Kokkos::LayoutRight>*,Kokkos::SharedSpace> M0_poly_adjoints;
void compute_M0(const int num_nodes, Kokkos::View<const int*> node_types);
void reverse_M0(const int num_nodes, Kokkos::View<const int*> node_types);

// H1
Kokkos::View<double***,Kokkos::LayoutRight> H1, H1_adj;
Kokkos::View<double***,Kokkos::LayoutRight> H1_weights;
void compute_H1(const int num_nodes);
void reverse_H1(const int num_nodes);

// Phi1
int num_lelm1lm2, num_lme;
Kokkos::View<int*> Phi1_l, Phi1_l1, Phi1_l2;
Kokkos::View<int*> Phi1_lme, Phi1_lelm1lm2;
Kokkos::View<double*> Phi1_clebsch_gordan;
Kokkos::View<double***,Kokkos::LayoutRight> Phi1r, dPhi1r;
Kokkos::View<double***,Kokkos::LayoutRight> Phi1, dPhi1;
void compute_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices);
void reverse_Phi1(const int num_nodes, Kokkos::View<const int*> num_neigh, Kokkos::View<const int*> neigh_indices, Kokkos::View<const double*> xyz, Kokkos::View<const double*> r, bool zero_dxyz = true, bool zero_H1_adj = true);

// TODO for testing of Phi1 strategies
Kokkos::View<int*> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;

// A1
Kokkos::View<double***,Kokkos::LayoutRight> A1, A1_adj;
Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights;
Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> A1_weights_trans;
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
Kokkos::View<double**,Kokkos::LayoutRight> M1, M1_adj;
Kokkos::View<int**,Kokkos::LayoutRight> M1_monomials;
Kokkos::View<double***,Kokkos::LayoutRight> M1_weights;
Kokkos::View<int**,Kokkos::LayoutRight> M1_poly_spec;
Kokkos::View<double***,Kokkos::LayoutRight> M1_poly_coeff;
Kokkos::View<double***,Kokkos::LayoutRight> M1_poly_values;
Kokkos::View<double***,Kokkos::LayoutRight> M1_poly_adjoints;
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
