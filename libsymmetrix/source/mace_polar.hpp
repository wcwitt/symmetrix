#include <array>
#include <complex>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "cubic_spline.hpp"
#include "cubic_spline_set.hpp"

class MACEPolar {

public:

MACEPolar(std::string filename);

// Basic model information
int num_elements;
int num_channels;
int num_edge_channels;
double r_cut;
std::vector<int> atomic_numbers;
std::vector<double> atomic_energies;

// Neighbor list
std::vector<int> first_neigh;

// Node energies and forces
std::vector<double> node_energies, node_forces, position_forces;
std::array<double,6> virial;
void compute_node_energies_forces(const int num_nodes,
                                  std::span<const int> node_types,
                                  std::span<const int> num_neigh,
                                  std::span<const int> neigh_indices,
                                  std::span<const int> neigh_types,
                                  std::span<const double> xyz,
                                  std::span<const double> r,
                                  std::span<const double> positions,
                                  std::span<const double> cell,
                                  double total_charge = 0.0,
                                  double total_spin = 1.0);
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
                      std::span<const int> node_types);
void compute_charges(const int num_nodes,
                     std::span<const int> node_types,
                     std::span<const double> positions,
                     std::span<const double> cell,
                     double total_charge,
                     double total_spin);
void reverse_charges(const int num_nodes);
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

// Electrostatics replacing the direct k-space sums when set
std::function<void(const double* q, double* V)> features_hook;
std::function<void(const double* q, const double* V_adj, double* q_adj, double* f, double* W)> features_reverse_hook;
std::function<void(const double* rho, double* psi, double* f, double* W)> coulomb_hook;
std::function<void(double* x, int n)> allreduce = [] (double*, int) {};

// Building blocks
struct Linear {
    std::vector<int> in, out, off_in, off_out;
    std::vector<std::vector<double>> weights, weights_trans;
    std::vector<double> bias;
    void forward(int n, const double* x, int ldx, double* y, int ldy, bool add) const;
    void reverse(int n, const double* y_adj, int ldy, double* x_adj, int ldx) const;
};
struct MLP {
    struct Layer {
        int in;
        int out;
        std::vector<double> weights, weights_trans, bias, ln_weights, ln_bias;
        double ln_eps;
    };
    std::vector<Layer> layers;
    std::vector<std::vector<double>> x_hat, rstd;
    void forward(int n, const double* x_in, double* y);
    void reverse(int n, const double* y_adj, double* x_adj) const;
};
struct Gate {
    int scalars;
    double scalar_constant, gate_constant;
    std::vector<int> mul;
    int dim_in, dim_out;
    std::vector<double> x;
    void forward(int n, const double* x_in, double* y);
    void reverse(int n, const double* y_adj, double* x_adj) const;
};
struct Dot {
    int num_channels;
    std::vector<std::vector<double>> weights, weights_trans;
    std::vector<double> z;
    void forward(int n, const double* x, const double* y, double* out, int ld);
    void reverse(int n, const double* x, const double* out_adj, int ld, double* x_adj, double* y_adj) const;
};
struct Modulation {
    int num_channels;
    std::vector<std::vector<double>> weights, weights_trans;
    std::vector<double> w;
    void forward(int n, const double* x, const double* s, double* out);
    void reverse(int n, const double* x, const double* out_adj, double* x_adj, double* s_adj) const;
};
struct BiasReadout {
    Linear linear_1, linear_mid, linear_2;
    double act;
    std::vector<double> y1, y2;
    void forward(int n, const double* x, int ldx, double* out);
    void reverse(int n, const double* out_adj, double* x_adj, int ldx) const;
};
struct GatedReadout {
    Linear linear_1, linear_mid, linear_2;
    Gate gate_1, gate_2;
    void forward(int n, const double* x, int ldx, double* out);
    void reverse(int n, const double* out_adj, double* x_adj, int ldx) const;
};

// Radial functions
double radial_spline_r_min;
std::vector<std::unique_ptr<CubicSplineSet<double>>> spl_set_0, spl_set_1;
std::vector<CubicSpline> A0_splines, A1_splines;

// Spherical harmonics
std::vector<double> Y, Y_grad;

// X0
int num_basis_0;
std::vector<int> mul_X, off_X;
Linear linear_0;
std::vector<double> Phi0, Phi0_adj, X0, X0_adj, A0_scale, residual_0;

// A0
Gate gate_0;
Linear linear_2_0;
std::vector<double> G0, G0_adj, A0, A0_adj;

// M0
std::vector<double> M0, M0_adj;
std::vector<double> B0;
std::vector<std::array<int,3>> B0_spec, M0_spec;
std::vector<double> B0_coefficients, M0_weights;
int num_B0;

// H1
Linear product_linear_0, linear_up_1;
std::vector<double> H1, H1_adj, H1e, H1e_adj, skip_0;

// Phi1
int num_lme;
std::vector<double> Phi1, dPhi1;
std::vector<int> Phi1_l;
std::vector<std::array<int,3>> Phi1_spec;
std::vector<std::array<int,2>> Phi1_cg_spec;
std::vector<double> Phi1_clebsch_gordan;

// X1
Linear linear_1_1, residual_1;
std::vector<double> X1, X1_adj, X1_residual, A1_scale;

// A1
Gate gate_1;
Linear linear_2_1;
std::vector<double> G1, G1_adj, A1, A1_adj;

// M1
std::vector<double> M1, M1_adj;
std::vector<double> B1;
std::vector<std::array<int,3>> B1_spec, M1_spec;
std::vector<double> B1_coefficients, M1_weights;
int num_B1;

// H2
Linear product_linear_1, skip_1;
std::vector<double> H2, H2_adj;

// Readouts
std::vector<double> readout_0;
BiasReadout readout_1;

// Charges q[t][i][lm][spin]
struct Update {
    Linear potential, node, charge;
    std::vector<double> source;
    Dot dot;
    MLP mlp;
    Modulation out;
    GatedReadout readout;
    std::vector<double> embedding, mlp_in, mlp_out, modulated, readout_out;
};
std::array<Linear,2> sources, mixer;
BiasReadout fukui;
std::array<Update,2> updates;
std::array<std::vector<double>,3> q, q_adj, fukui_n;
std::array<std::array<double,2>,3> fukui_sum, fukui_c;
std::array<std::vector<double>,2> V, V_adj;
std::vector<double> f0, mixed, mixed_adj;
std::array<double,2> spin_charges;
void normalize(int num_nodes, int t, const double* f, int ldf);
void normalize_reverse(int num_nodes, int t, double* f_adj, int ldf);

// Electron energy
struct Electron {
    Linear q, v;
    Dot dot_q, dot_v;
    MLP mlp;
    std::vector<double> q_sum, q_up, v_up, mlp_in;
};
Electron electron;

// Electrostatics: direct k-space sums over the half sphere, on p = -positions
double kspace_cutoff, field_constant, volume;
double density_sigma;
std::array<double,2> density_a;
std::vector<double> feature_sigmas;
std::vector<std::array<double,2>> feature_a;
std::vector<double> field_self, field_norms;
std::vector<std::array<double,3>> kvec;
std::vector<double> k2inv, inverted_positions;
std::vector<std::complex<double>> density_basis, feature_basis, density_basis_deriv, feature_basis_deriv;
std::vector<std::complex<double>> rho_structure, rho_basis;
std::array<std::vector<std::complex<double>>,2> structure_factors, basis_factors;
std::vector<double> rho, psi;
double electrostatic_energy;
void setup_kspace(int num_nodes, std::span<const double> positions, std::span<const double> cell);
void structure(int n, const double* w, int num_weights, std::complex<double>* structure_factors) const;
void on_basis(const std::complex<double>* structure_factors, int num_columns, bool features,
              std::complex<double>* X) const;
void project(int n, const std::complex<double>* Z, int num_columns, bool features, double* out) const;
void phase_forces(int n, const std::complex<double>* A, const double* w, int num_columns,
                  bool features, double prefactor,
                  double* f) const;
void kspace_virial(const std::complex<double>* Rx, bool fx, const std::complex<double>* Ry, bool fy, int num_columns,
                   double prefactor, double* W) const;
void on_basis_deriv(int k, int c, const std::complex<double>* structure_factors, bool features, int num_columns,
                    std::complex<double>& value, std::complex<double>* deriv) const;
void features(int n, const double* q, std::vector<std::complex<double>>& structure_factors,
              std::vector<std::complex<double>>& basis_factors, double* V) const;
void features_reverse(int n, const double* q, const std::vector<std::complex<double>>& structure_factors,
                      const std::vector<std::complex<double>>& basis_factors, const double* V_adj,
                      double* q_adj, double* f,
                      double* W) const;

// Initializer
void load_from_json(std::string filename);

};
