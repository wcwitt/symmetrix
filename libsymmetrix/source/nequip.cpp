#include <cmath>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "nequip.hpp"

template <typename Precision>
NequIP<Precision>::NequIP(std::string filename)
{
    load_from_json(filename);
}

template <typename Precision>
void NequIP<Precision>::compute_node_energies_forces(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    // \lambda = (l, p),  H^{(0)}_i = H^{(0)}_{a_i},  X^{(0)}_j = X^{(0)}_{a_j}
    for (int t=0; t<num_layers; ++t)
        compute_layer(t, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
    compute_readout(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    for (int t=num_layers-1; t>=0; --t)
        reverse_layer(t, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
}

template <typename Precision>
void NequIP<Precision>::compute_layer(
    const int t,
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    const int H_width = H_offset[t].back();
    const int Phi_width = Phi_offset[t].back();
    const int A_width = A_offset[t].back();

    // neighbor list
    if (t == 0) {
        for (const double r_ij : r)
            if (std::isnan(r_ij) or r_ij <= 0.0 or r_ij > r_cut)
                throw std::invalid_argument("Out of bounds in NequIP. r=" + std::to_string(r_ij));
        first_neigh.resize(num_nodes);
        std::exclusive_scan(num_neigh.begin(), num_neigh.begin()+num_nodes, first_neigh.begin(), 0);
    }

    // Y
    // Y_{lm}(\hat{\mathbf{r}}_{ij})
    if (t == 0 and xyz.size() > 0) {
        const int num = xyz.size()/3;
        Y.resize(num*num_lm);
        Y_grad.resize(3*num*num_lm);
        const auto xyz_p = std::vector<Precision>(xyz.begin(), xyz.end());
        sphericart::SphericalHarmonics<Precision> sphericart(l_max);
        sphericart.compute_array_with_gradients(
            xyz_p.data(), xyz_p.size(), Y.data(), Y.size(), Y_grad.data(), Y_grad.size());
    }

    // Phi
    // \Phi^{(t)}_{i,k\lambda m\eta} = \sum_j \sum_{m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) X^{(t)}_{j,k\lambda_1m_1},  \eta = (\lambda_1, l_2)
    {
        Phi.assign(num_nodes*Phi_width, 0.0);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            auto Phi_i = Phi.data()+i*Phi_width;
            auto R_ij = std::vector<Precision>(spl_set[t]->num_splines);
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                spl_set[t]->evaluate(r[ij], R_ij);
                auto Y_ij = Y.data()+ij*num_lm;
                auto X_ij = X[t].data()+((t > 0) ? neigh_indices[ij] : neigh_types[ij])*H_width;
                for (int p=0; p<Phi_spec[t].size(); ++p) {
                    const auto [a, lm, w, b, n] = Phi_spec[t][p];
                    const Precision C_Y = Phi_clebsch_gordan[t][p] * Y_ij[lm];
                    for (int k=0; k<n; ++k)
                        Phi_i[b+k] += C_Y * R_ij[w+k] * X_ij[a+k];
                }
            }
        }
    }

    // A
    // A^{(t)}_{i,k\lambda m} = \sum_{k'\eta} w^{(t)}_{k\lambda,k'\eta} \Phi^{(t)}_{i,k'\lambda m\eta} + \sum_{k'} w^{(t)}_{a_ik\lambda,k'} H^{(t)}_{i,k'\lambda m}
    {
        A[t].assign(num_nodes*A_width, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto Phi_i = Phi.data()+i*Phi_width;
            auto H_i = H[t].data()+((t > 0) ? i : node_types[i])*H_width;
            auto A_i = A[t].data()+i*A_width;
            for (int ir=0; ir<A_mul[t].size(); ++ir) {
                auto A_weights_ir = A_weights[t][ir].data();
                auto A_sc_weights_ir = A_sc_weights[t][ir].data();
                for (int m=0; m<2*(ir/2)+1; ++m) {
                    auto Phi_i_irm = Phi_i+Phi_offset[t][ir]+m*Phi_mul[t][ir];
                    auto H_i_irm = H_i+H_offset[t][ir]+m*H_mul[t][ir];
                    auto A_i_irm = A_i+A_offset[t][ir]+m*A_mul[t][ir];
                    for (int kp=0; kp<Phi_mul[t][ir]; ++kp)
                        for (int k=0; k<A_mul[t][ir]; ++k)
                            A_i_irm[k] += Phi_i_irm[kp] * A_weights_ir[kp*A_mul[t][ir]+k];
                    if (A_sc_weights[t][ir].size() > 0)
                        for (int kp=0; kp<H_mul[t][ir]; ++kp)
                            for (int k=0; k<A_mul[t][ir]; ++k)
                                A_i_irm[k] += H_i_irm[kp] * A_sc_weights_ir[(node_types[i]*H_mul[t][ir]+kp)*A_mul[t][ir]+k];
                }
            }
        }
    }

    // gate
    // H^{(t+1)}_{i,k(0e)} = c_e \mathrm{SiLU}(A^{(t)}_{i,k(0e)}),  H^{(t+1)}_{i,k(0o)} = c_o \tanh(A^{(t)}_{i,k(0o)})
    // H^{(t+1)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) A^{(t)}_{i,k\lambda m} (l > 0),  g_{k\lambda} = N^{(t+1)}_{0e} + \sum_{0o < \lambda' < \lambda} N^{(t+1)}_{\lambda'} + k
    {
        const int width = H_offset[t+1].back();
        H[t+1].resize(num_nodes*width);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A_i = A[t].data()+i*A_width;
            auto H_i = H[t+1].data()+i*width;
            for (int k=0; k<H_mul[t+1][0]; ++k) {
                const Precision e = std::exp(-std::abs(A_i[k]));
                H_i[k] = gate_cst * A_i[k] * ((A_i[k] >= 0) ? 1 : e) / (1 + e);
            }
            for (int k=0; k<H_mul[t+1][1]; ++k)
                H_i[H_offset[t+1][1]+k] = gate_cst_odd * std::tanh(A_i[A_offset[t][1]+k]);
            int g = H_mul[t+1][0];
            for (int ir=2; ir<A_mul[t].size(); ++ir)
                for (int k=0; k<A_mul[t][ir]; ++k) {
                    const Precision e = std::exp(-std::abs(A_i[g]));
                    const Precision gate = gate_cst * A_i[g] * ((A_i[g] >= 0) ? 1 : e) / (1 + e);
                    for (int m=0; m<2*(ir/2)+1; ++m)
                        H_i[H_offset[t+1][ir]+m*A_mul[t][ir]+k] = gate * A_i[A_offset[t][ir]+m*A_mul[t][ir]+k];
                    g += 1;
                }
        }
    }

    // X
    // X^{(t+1)}_{i,k\lambda m} = \sum_{k'} w^{(t+1)}_{k\lambda,k'} H^{(t+1)}_{i,k'\lambda m}
    if (t+1 < num_layers) {
        const int width = H_offset[t+1].back();
        X[t+1].assign(num_nodes*width, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            for (int ir=0; ir<H_mul[t+1].size(); ++ir) {
                auto X_weights_ir = X_weights[t+1][ir].data();
                for (int m=0; m<2*(ir/2)+1; ++m) {
                    auto H_i_irm = H[t+1].data()+i*width+H_offset[t+1][ir]+m*H_mul[t+1][ir];
                    auto X_i_irm = X[t+1].data()+i*width+H_offset[t+1][ir]+m*H_mul[t+1][ir];
                    for (int kp=0; kp<H_mul[t+1][ir]; ++kp)
                        for (int k=0; k<H_mul[t+1][ir]; ++k)
                            X_i_irm[k] += H_i_irm[kp] * X_weights_ir[kp*H_mul[t+1][ir]+k];
                }
            }
        }
    }
}

template <typename Precision>
void NequIP<Precision>::compute_readout(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    // ZBL
    if (has_zbl) {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const double Z_i = atomic_numbers[node_types[i]];
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const double Z_j = atomic_numbers[neigh_types[ij]];
                const double d = (std::pow(Z_i, 0.23) + std::pow(Z_j, 0.23)) / 0.46850;
                const double x = d * r[ij];
                const double e_1 = 0.02817 * std::exp(-0.20162*x);
                const double e_2 = 0.28022 * std::exp(-0.40290*x);
                const double e_3 = 0.50986 * std::exp(-0.94229*x);
                const double e_4 = 0.18175 * std::exp(-3.19980*x);
                const double psi = e_1 + e_2 + e_3 + e_4;
                const double psi_deriv = -d * (0.20162*e_1 + 0.40290*e_2 + 0.94229*e_3 + 3.19980*e_4);
                const double p = zbl_p;
                const double u = r[ij] / r_cut;
                double f_c = 0.0;
                double f_c_deriv = 0.0;
                if (u < 1.0) {
                    f_c = 1.0 - (p+1.0)*(p+2.0)/2.0*std::pow(u, p) + p*(p+2.0)*std::pow(u, p+1.0)
                          - p*(p+1.0)/2.0*std::pow(u, p+2.0);
                    f_c_deriv = (-(p+1.0)*(p+2.0)/2.0*p*std::pow(u, p-1.0) + p*(p+2.0)*(p+1.0)*std::pow(u, p)
                                 - p*(p+1.0)/2.0*(p+2.0)*std::pow(u, p+1.0)) / r_cut;
                }
                const double v = zbl_prefactor * Z_i * Z_j;
                node_energies[i] += v / r[ij] * psi * f_c;
                const double dE_dr = v * (-psi*f_c/(r[ij]*r[ij]) + psi_deriv*f_c/r[ij] + psi*f_c_deriv/r[ij]);
                for (int a=0; a<3; ++a)
                    node_forces[3*ij+a] -= dE_dr * xyz[3*ij+a] / r[ij];
            }
        }
    }

    // readout
    {
        const int width = H_offset[num_layers].back();
        H_adj.resize(num_nodes*width);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto H_i = H[num_layers].data()+i*width;
            const double scale = per_type_scale[node_types[i]];
            node_energies[i] += per_type_shift[node_types[i]];
            for (int k=0; k<width; ++k) {
                node_energies[i] += scale * readout_weights[k] * H_i[k];
                H_adj[i*width+k] = scale * readout_weights[k];
            }
        }
    }
}

template <typename Precision>
void NequIP<Precision>::reverse_layer(
    const int t,
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    const int H_width = H_offset[t].back();
    const int Phi_width = Phi_offset[t].back();
    const int A_width = A_offset[t].back();

    // X
    // \bar{H}^{(t+1)}_{i,k'\lambda m} += \sum_k w^{(t+1)}_{k\lambda,k'} \bar{X}^{(t+1)}_{i,k\lambda m}
    if (t+1 < num_layers) {
        const int width = H_offset[t+1].back();
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            for (int ir=0; ir<H_mul[t+1].size(); ++ir) {
                auto X_weights_ir = X_weights[t+1][ir].data();
                for (int m=0; m<2*(ir/2)+1; ++m) {
                    auto X_adj_i_irm = X_adj.data()+i*width+H_offset[t+1][ir]+m*H_mul[t+1][ir];
                    auto H_adj_i_irm = H_adj.data()+i*width+H_offset[t+1][ir]+m*H_mul[t+1][ir];
                    for (int k=0; k<H_mul[t+1][ir]; ++k)
                        for (int kp=0; kp<H_mul[t+1][ir]; ++kp)
                            H_adj_i_irm[k] += X_adj_i_irm[kp] * X_weights_ir[k*H_mul[t+1][ir]+kp];
                }
            }
        }
    }

    // gate
    // \bar{A}^{(t)}_{i,k(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,k(0e)}) \bar{H}^{(t+1)}_{i,k(0e)},  \bar{A}^{(t)}_{i,k(0o)} = c_o \tanh'(A^{(t)}_{i,k(0o)}) \bar{H}^{(t+1)}_{i,k(0o)}
    // \bar{A}^{(t)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) \bar{H}^{(t+1)}_{i,k\lambda m},  \bar{A}^{(t)}_{i,g_{k\lambda}(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,g_{k\lambda}(0e)}) \sum_m A^{(t)}_{i,k\lambda m} \bar{H}^{(t+1)}_{i,k\lambda m}
    {
        const int width = H_offset[t+1].back();
        A_adj.resize(num_nodes*A_width);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A_i = A[t].data()+i*A_width;
            auto H_adj_i = H_adj.data()+i*width;
            auto A_adj_i = A_adj.data()+i*A_width;
            for (int k=0; k<H_mul[t+1][0]; ++k) {
                const Precision e = std::exp(-std::abs(A_i[k]));
                const Precision s = ((A_i[k] >= 0) ? 1 : e) / (1 + e);
                A_adj_i[k] = H_adj_i[k] * gate_cst * s * (1 + A_i[k]*(1 - s));
            }
            for (int k=0; k<H_mul[t+1][1]; ++k) {
                const Precision s = std::tanh(A_i[A_offset[t][1]+k]);
                A_adj_i[A_offset[t][1]+k] = H_adj_i[H_offset[t+1][1]+k] * gate_cst_odd * (1 - s*s);
            }
            int g = H_mul[t+1][0];
            for (int ir=2; ir<A_mul[t].size(); ++ir)
                for (int k=0; k<A_mul[t][ir]; ++k) {
                    const Precision e = std::exp(-std::abs(A_i[g]));
                    const Precision s = ((A_i[g] >= 0) ? 1 : e) / (1 + e);
                    Precision gate_adj = 0.0;
                    for (int m=0; m<2*(ir/2)+1; ++m) {
                        const int a = A_offset[t][ir]+m*A_mul[t][ir]+k;
                        const Precision h = H_adj_i[H_offset[t+1][ir]+m*A_mul[t][ir]+k];
                        A_adj_i[a] = h * gate_cst * A_i[g] * s;
                        gate_adj += h * A_i[a];
                    }
                    A_adj_i[g] = gate_adj * gate_cst * s * (1 + A_i[g]*(1 - s));
                    g += 1;
                }
        }
    }

    // A
    // \bar{\Phi}^{(t)}_{i,k'\lambda m\eta} = \sum_k w^{(t)}_{k\lambda,k'\eta} \bar{A}^{(t)}_{i,k\lambda m},  \bar{H}^{(t)}_{i,k'\lambda m} = \sum_k w^{(t)}_{a_ik\lambda,k'} \bar{A}^{(t)}_{i,k\lambda m} (t > 0)
    {
        Phi_adj.assign(num_nodes*Phi_width, 0.0);
        if (t > 0)
            H_adj.assign(num_nodes*H_width, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A_adj_i = A_adj.data()+i*A_width;
            auto Phi_adj_i = Phi_adj.data()+i*Phi_width;
            auto H_adj_i = H_adj.data()+i*H_width;
            for (int ir=0; ir<A_mul[t].size(); ++ir) {
                auto A_weights_ir = A_weights[t][ir].data();
                auto A_sc_weights_ir = A_sc_weights[t][ir].data();
                for (int m=0; m<2*(ir/2)+1; ++m) {
                    auto A_adj_i_irm = A_adj_i+A_offset[t][ir]+m*A_mul[t][ir];
                    auto Phi_adj_i_irm = Phi_adj_i+Phi_offset[t][ir]+m*Phi_mul[t][ir];
                    auto H_adj_i_irm = H_adj_i+H_offset[t][ir]+m*H_mul[t][ir];
                    for (int kp=0; kp<Phi_mul[t][ir]; ++kp)
                        for (int k=0; k<A_mul[t][ir]; ++k)
                            Phi_adj_i_irm[kp] += A_adj_i_irm[k] * A_weights_ir[kp*A_mul[t][ir]+k];
                    if (t > 0 and A_sc_weights[t][ir].size() > 0)
                        for (int kp=0; kp<H_mul[t][ir]; ++kp)
                            for (int k=0; k<A_mul[t][ir]; ++k)
                                H_adj_i_irm[kp] += A_adj_i_irm[k] * A_sc_weights_ir[(node_types[i]*H_mul[t][ir]+kp)*A_mul[t][ir]+k];
                }
            }
        }
    }

    // Phi
    // \bar{X}^{(t)}_{j,k\lambda_1m_1} = \sum_i \sum_{\lambda m\eta m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(t)}_{i,k\lambda m\eta} (t > 0)
    // \mathbf{F}_{ij} -= \sum_{k\lambda m\eta m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij})] X^{(t)}_{j,k\lambda_1m_1} \bar{\Phi}^{(t)}_{i,k\lambda m\eta}
    {
        if (t > 0)
            X_adj.assign(X[t].size(), 0.0);
        std::vector<std::mutex> X_adj_locks((t > 0) ? X[t].size()/H_width : 0);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            auto Phi_adj_i = Phi_adj.data()+i*Phi_width;
            auto X_adj_ij = std::vector<Precision>(H_width);
            auto R_ij = std::vector<Precision>(spl_set[t]->num_splines);
            auto R_deriv_ij = std::vector<Precision>(R_ij.size());
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                auto node_forces_ij = node_forces.data()+3*ij;
                auto xyz_ij = xyz.data()+3*ij;
                auto r_ij = r[ij];
                spl_set[t]->evaluate_derivs(r_ij, R_ij, R_deriv_ij);
                auto Y_ij = Y.data()+ij*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
                auto X_ij = X[t].data()+((t > 0) ? neigh_indices[ij] : neigh_types[ij])*H_width;
                std::fill(X_adj_ij.begin(), X_adj_ij.end(), 0.0);
                Precision dE_dr = 0.0;
                for (int p=0; p<Phi_spec[t].size(); ++p) {
                    const auto [a, lm, w, b, n] = Phi_spec[t][p];
                    const Precision C_Y = Phi_clebsch_gordan[t][p] * Y_ij[lm];
                    Precision g = 0.0;
                    Precision g_deriv = 0.0;
                    for (int k=0; k<n; ++k) {
                        g += Phi_adj_i[b+k] * R_ij[w+k] * X_ij[a+k];
                        g_deriv += Phi_adj_i[b+k] * R_deriv_ij[w+k] * X_ij[a+k];
                        X_adj_ij[a+k] += C_Y * R_ij[w+k] * Phi_adj_i[b+k];
                    }
                    dE_dr += C_Y * g_deriv;
                    for (int c=0; c<3; ++c)
                        node_forces_ij[c] -= Phi_clebsch_gordan[t][p] * Y_grad_ij[c*num_lm+lm] * g;
                }
                for (int c=0; c<3; ++c)
                    node_forces_ij[c] -= dE_dr * xyz_ij[c]/r_ij;
                if (t > 0) {
                    const int j = neigh_indices[ij];
                    std::scoped_lock lock(X_adj_locks[j]);
                    for (int x=0; x<H_width; ++x)
                        X_adj[j*H_width+x] += X_adj_ij[x];
                }
            }
        }
    }
}

template <typename Precision>
void NequIP<Precision>::load_from_json(
    const std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);
    if (file.value("format_version", 0) != 3)
        throw std::runtime_error(filename + " was extracted by an incompatible symmetrix version; extract it again");
    if (file.value("model_type", "") != "NequIP")
        throw std::runtime_error(filename + " is not a NequIP model");

    // Basic model information
    num_elements = file["num_elements"];
    num_layers = file["num_layers"];
    r_cut = file["r_cut"];
    l_max = file["l_max"];
    num_lm = (l_max+1)*(l_max+1);
    atomic_numbers = file["atomic_numbers"].get<std::vector<int>>();

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl) {
        zbl_prefactor = file["zbl_prefactor"];
        zbl_p = file["zbl_p"];
    }

    // H
    H_mul = file["H_mul"].get<std::vector<std::vector<int>>>();
    H_offset = file["H_offset"].get<std::vector<std::vector<int>>>();
    H.resize(num_layers+1);
    for (const auto& h : file["embedding"])
        for (const double x : h)
            H[0].push_back(x);

    // X
    X_weights = file["X_weights"].get<std::vector<std::vector<std::vector<Precision>>>>();
    X.resize(num_layers);
    {
        const auto embedding = file["embedding"].get<std::vector<std::vector<double>>>();
        const auto W = file["X_weights"][0][0].get<std::vector<double>>();
        const int dim = H_mul[0][0];
        X[0].resize(num_elements*dim);
        for (int e=0; e<num_elements; ++e)
            for (int k=0; k<dim; ++k) {
                double x = 0.0;
                for (int kp=0; kp<dim; ++kp)
                    x += embedding[e][kp] * W[kp*dim+k];
                X[0][e*dim+k] = x;
            }
    }

    // Phi
    auto spl_values = file["radial_spline_values"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs = file["radial_spline_derivs"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int t=0; t<num_layers; ++t)
        spl_set.push_back(std::make_unique<CubicSplineSet<Precision>>(0.0, r_cut, spl_values[t], spl_derivs[t]));
    Phi_mul = file["Phi_mul"].get<std::vector<std::vector<int>>>();
    Phi_offset = file["Phi_offset"].get<std::vector<std::vector<int>>>();
    Phi_spec = file["Phi_spec"].get<std::vector<std::vector<std::array<int,5>>>>();
    Phi_clebsch_gordan = file["Phi_clebsch_gordan"].get<std::vector<std::vector<Precision>>>();

    // A
    A_mul = file["A_mul"].get<std::vector<std::vector<int>>>();
    A_offset = file["A_offset"].get<std::vector<std::vector<int>>>();
    A_weights = file["A_weights"].get<std::vector<std::vector<std::vector<Precision>>>>();
    A_sc_weights = file["A_sc_weights"].get<std::vector<std::vector<std::vector<Precision>>>>();
    A.resize(num_layers);

    // Gates
    gate_cst = file["gate_cst"];
    gate_cst_odd = file.value("gate_cst_odd", 0.0);

    // Readout
    readout_weights = file["readout_weights"].get<std::vector<Precision>>();
    per_type_scale = file["per_type_scale"].get<std::vector<double>>();
    per_type_shift = file["per_type_shift"].get<std::vector<double>>();
}

template class NequIP<double>;
template class NequIP<float>;
