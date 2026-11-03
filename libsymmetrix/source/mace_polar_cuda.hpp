#include <array>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

struct cublasContext;

class MACEPolarCuda {

public:

template <class T> struct Buffer {
    T* p = nullptr;
    size_t n = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(Buffer&& b)
    {
        std::swap(p, b.p);
        std::swap(n, b.n);
        return *this;
    }
    Buffer(Buffer&& b)
    {
        *this = std::move(b);
    }
    ~Buffer()
    {
        cudaFree(p);
    }
    void reserve(size_t m)
    {
        if (m <= n and p)
            return;
        cudaFree(p);
        n = m + m/8 + 1;
        if (cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(T)) != cudaSuccess)
            throw std::runtime_error("MACEPolarCuda: out of device memory");
    }
    void upload(const std::vector<T>& v)
    {
        reserve(v.size());
        cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice);
    }
    operator T*() const
    {
        return p;
    }
};

MACEPolarCuda(std::string filename, bool float32_mixed = false);
~MACEPolarCuda();
const bool float32_mixed;

// Basic model information
int num_elements;
int num_channels;
int num_edge_channels;
double r_cut;
std::vector<int> atomic_numbers;

// Node energies and forces
Buffer<double> node_energies, node_forces, position_forces;
std::array<double,6> virial;
double electrostatic_energy;
void compute_node_energies_forces(int num_nodes, std::span<const int> node_types, std::span<const int> num_neigh,
                                  std::span<const int> neigh_indices, std::span<const int> neigh_types,
                                  std::span<const double> xyz, std::span<const double> r,
                                  std::span<const double> positions, std::span<const double> cell, double total_charge,
                                  double total_spin, cudaStream_t stream);
void compute_first_layer(int num_nodes, std::span<const int> node_types, std::span<const int> num_neigh,
                         std::span<const int> neigh_types, std::span<const double> xyz, std::span<const double> r,
                         cudaStream_t stream, int num_ghost_nodes = 0);
void compute_second_layer(std::span<const int> neigh_indices);
void compute_readouts();
void compute_charges(std::span<const double> positions, std::span<const double> cell, double total_charge,
                     double total_spin);
void reverse_charges();
void reverse_second_layer(std::span<const int> neigh_indices);
void reverse_first_layer(std::span<const double> xyz);

// H1e of the nodes and ghost nodes [node][lm][k]
int num_ghost_nodes = 0;
Buffer<float> H1e_ex, H1e_ex_adj;

// Electrostatics replacing the direct k-space sums when set
std::function<void(const float* q, float* V)> features_hook;
std::function<void(const float* q, const float* V_adj, float* q_adj, double* f, double* W)> features_reverse_hook;
std::function<void(const float* rho, float* psi, double* f, double* W)> coulomb_hook;
std::function<void(double* x, int n)> allreduce;

// Building blocks
struct Linear {
    std::vector<int> in, out, off_in, off_out;
    std::vector<long> weights_offset;
    Buffer<float> weights, bias;
    void forward(const MACEPolarCuda& p, int n, const float* x, float* y, bool add) const;
    void reverse(const MACEPolarCuda& p, int n, const float* y_adj, float* x_adj) const;
};
struct MLP {
    struct Layer {
        int in;
        int out;
        float ln_eps;
        Buffer<float> weights, bias, ln_weights, ln_bias, z, x_hat, rstd, a;
    };
    std::vector<Layer> layers;
    Buffer<float> g, h;
    void forward(const MACEPolarCuda& p, int n, const float* x, float* y);
    void reverse(const MACEPolarCuda& p, int n, const float* y_adj, float* x_adj);
};
struct Gate {
    int scalars;
    float scalar_constant, gate_constant;
    std::vector<int> mul;
    Buffer<float> x;
    void forward(const MACEPolarCuda& p, int n, const float* x, float* y);
    void reverse(const MACEPolarCuda& p, int n, const float* y_adj, float* x_adj) const;
};
struct Dot {
    int num_channels;
    Buffer<float> weights, z, z_adj;
    void forward(const MACEPolarCuda& p, int n, const float* x, const float* y, float* out, int ld);
    void reverse(const MACEPolarCuda& p, int n, const float* x, const float* out_adj, int ld,
                 float* x_adj, float* y_adj);
};
struct Modulation {
    int num_channels;
    Buffer<float> weights, w, w_adj;
    void forward(const MACEPolarCuda& p, int n, const float* x, const float* s, float* out);
    void reverse(const MACEPolarCuda& p, int n, const float* x, const float* out_adj, float* x_adj, float* s_adj);
};
struct BiasReadout {
    Linear linear_1, linear_mid, linear_2;
    float act;
    Buffer<float> y1, y2, a, g, f;
    void forward(const MACEPolarCuda& p, int n, const float* x, float* out);
    void reverse(const MACEPolarCuda& p, int n, const float* out_adj, float* x_adj);
};
struct GatedReadout {
    Linear linear_1, linear_mid, linear_2;
    Gate gate_1, gate_2;
    Buffer<float> y, g, y_adj, g_adj;
    void forward(const MACEPolarCuda& p, int n, const float* x, float* out);
    void reverse(const MACEPolarCuda& p, int n, const float* out_adj, float* x_adj);
};

// Neighbor list
int num_nodes = 0;
int num_edges = 0;
int num_knots;
double radial_spline_r_min, knot_spacing;
Buffer<int> node_types, first_neigh, pairs, scan;
Buffer<float4> geometry;
Buffer<float> edge_forces;

// Radial functions
int num_basis_0, Phi0_elements;
Buffer<float> radial_0, radial_1, densities;
Buffer<__half2> radial_1_half;

// X0
int num_X0_rows;
std::vector<int> mul_X, off_X;
Buffer<float> X0_weights, residual_0, Phi0, Phi0_adj, X0, A_scale, X_adj, scale_adj;

// A0
Gate gate_0;
Linear linear_2_0;
Buffer<float> G0, A0, G_adj, A_adj;

// M0, M1
Buffer<float> M0_weights, M1_weights, M0, M1, M_adj;

// H1
Linear product_linear_0, linear_up_1;
Buffer<float> skip_0, H1, H1_adj, H1e, H1e_adj;

// Phi1
Buffer<float> Phi1, dPhi1;

// X1
Linear linear_1_1, residual_1;
Buffer<float> X1, X1_residual;

// A1
Gate gate_1;
Linear linear_2_1;
Buffer<float> G1, A1;

// H2
Linear product_linear_1, skip_1;
Buffer<float> H2, H2_adj;

// Readouts
Linear readout_0;
BiasReadout readout_1;
Buffer<double> atomic_energies;
Buffer<float> energies, ones;

// Charges
struct Update {
    Linear potential, node, charge;
    Buffer<float> source, embedding, mlp_in, mlp_out, modulated, readout_out;
    Dot dot;
    MLP mlp;
    Modulation out;
    GatedReadout readout;
};
struct Electron {
    Linear q, v;
    Dot dot_q, dot_v;
    MLP mlp;
    Buffer<float> q_sum, q_up, v_up, mlp_in;
};
std::array<Linear,2> sources, mixer;
BiasReadout fukui;
std::array<Update,2> updates;
Electron electron;
std::array<Buffer<float>,3> q, q_adj, fukui_n;
std::array<Buffer<float>,2> V, V_adj;
Buffer<float> f0, mixed, mixed_adj, mlp_in_adj, q_up_adj, v_up_adj, readout_out_adj, f_adj, modulated_adj, mlp_out_adj,
              embedding_adj;
Buffer<double> sums;
std::array<double,2> spin_charges;
void normalize(int t, const float* f, int ldf);
void normalize_reverse(int t, float* f_adj, int ldf);
void reduce(double* x, int n);

// Electrostatics
double kspace_cutoff, field_constant, volume;
double density_sigma;
std::array<double,2> density_a;
std::vector<double> feature_sigmas;
std::vector<std::array<double,2>> feature_a;
std::vector<float> field_self, field_norms;
int num_k = 0;
std::vector<double> kspace_cell;
Buffer<float> frac, kv, kn, k2inv, density_basis, feature_basis, density_basis_deriv, feature_basis_deriv, consts;
Buffer<float> phase_cos, phase_sin, kre, kim, kP, kQ, kMc, kMs;
Buffer<float2> rho_structure, rho_basis, structure_v, basis_v, Z;
std::array<Buffer<float2>,2> structure_factors, basis_factors;
Buffer<float> psi, rho, v, hook_q, hook_v, hook_a;
std::array<double,6> hook_virial;
void setup_kspace(const double* positions, std::span<const double> cell);
void kspace_vectors(const double* a, const double (*b)[3]);
void structure(const float* w, int num_weights, float2* out) const;
void on_basis(const float2* structure_factors, int num_columns, bool features, float2* X) const;
void project(const float2* Z, int num_columns, bool features, float* out, float scale) const;
void phase_forces(const float2* A, const float* w, int num_columns, bool features, float prefactor) const;
void kspace_virial(const float2* Rx, bool fx, const float2* Ry, bool fy, int num_columns, float prefactor) const;
void features(const float* q, Buffer<float2>& structure_factors, Buffer<float2>& basis_factors, float* V);
void features_reverse(const float* q, const Buffer<float2>& structure_factors, const Buffer<float2>& basis_factors,
                      const float* V_adj, float* q_adj);

// CUDA
cudaStream_t stream;
cublasContext* blas;
unsigned* error_flags;
void gemm(bool trans, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float beta, float* c,
          int ldc, float alpha = 1) const;
void sgemm(bool trans, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float beta, float* c,
           int ldc, float alpha = 1) const;
struct Halves { Buffer<__half> hi, lo; int shift; };
mutable std::map<std::pair<const float*, bool>, Halves> halves;
bool tensor_cores = false;
int num_sms = 0;

// Initializer
void load_from_json(std::string filename);

};
