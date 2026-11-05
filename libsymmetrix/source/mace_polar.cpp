#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "mace_polar.hpp"

using cplx = std::complex<double>;

MACEPolar::MACEPolar(std::string filename)
{
    load_from_json(filename);
}

void MACEPolar::compute_node_energies_forces(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> positions,
    std::span<const double> cell,
    double total_charge,
    double total_spin)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    // \mathbf{F}_i = -\partial E / \partial \mathbf{x}_i through the phases e^{\pm i\mathbf{k}\cdot\mathbf{p}_i},  \mathbf{p}_i = -\mathbf{x}_i
    compute_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    compute_second_layer(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    compute_readouts(num_nodes, node_types);
    compute_charges(num_nodes, node_types, positions, cell, total_charge, total_spin);
    reverse_charges(num_nodes);
    reverse_second_layer(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
    reverse_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

void MACEPolar::compute_first_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    const int num_X = off_X.back();
    const int num_Phi0 = 16*num_elements*num_basis_0;

    // neighbor list
    {
        for (const double r_ij : r)
            if (std::isnan(r_ij) or r_ij < radial_spline_r_min or r_ij > r_cut)
                throw std::invalid_argument("Out of bounds in MACEPolar. r=" + std::to_string(r_ij));
        first_neigh.resize(num_nodes);
        std::exclusive_scan(num_neigh.begin(), num_neigh.begin()+num_nodes, first_neigh.begin(), 0);
    }

    // Y
    // Y_{lm}(\hat{\mathbf{r}}_{ij})
    if (xyz.size() > 0) {
        const int num = xyz.size()/3;
        Y.resize(num*16);
        Y_grad.resize(3*num*16);
        sphericart::SphericalHarmonics<double> sphericart(3);
        sphericart.compute_array_with_gradients(
            xyz.data(), xyz.size(), Y.data(), Y.size(), Y_grad.data(), Y_grad.size());
    }

    // Phi0
    // \Phi^{(0)}_{i,lm,bn} = \sum_{j:a_j=b} R^{(0)}_{a_ib,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij}),  s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij})
    {
        Phi0.assign(long(num_nodes)*num_Phi0, 0.0);
        A0_scale.resize(num_nodes);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            auto Phi0_i = Phi0.data()+long(i)*num_Phi0;
            auto R0_ij = std::vector<double>(num_basis_0);
            double A0_scale_factor = 1.0;
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_ij = node_types[i]*num_elements+neigh_types[ij];
                spl_set_0[type_ij]->evaluate(r[ij], R0_ij);
                A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
                for (int lm=0; lm<16; ++lm) {
                    const double Y_ij_lm = Y[ij*16+lm];
                    auto Phi0_i_lmb = Phi0_i+(lm*num_elements+neigh_types[ij])*num_basis_0;
                    for (int n=0; n<num_basis_0; ++n)
                        Phi0_i_lmb[n] += Y_ij_lm*R0_ij[n];
                }
            }
            A0_scale[i] = 1.0/A0_scale_factor;
        }
    }

    // X0
    // X^{(0)}_{i,klm} = \sum_{bn} w_{kl,bn} \Phi^{(0)}_{i,lm,bn} / s^{(0)}_i + \delta_{l0} r_{a_i,k}
    {
        X0.resize(long(num_nodes)*num_X);
        linear_0.forward(num_nodes, Phi0.data(), num_Phi0, X0.data(), num_X, false);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto X0_i = X0.data()+long(i)*num_X;
            for (int x=0; x<num_X; ++x)
                X0_i[x] *= A0_scale[i];
            for (int k=0; k<mul_X[0]; ++k)
                X0_i[k] += residual_0[node_types[i]*mul_X[0]+k];
        }
    }

    // A0
    // T_{i,k00} = c_s X^{(0)}_{i,k00} \sigma(X^{(0)}_{i,k00}),  T_{i,klm} = c_g X^{(0)}_{i,klm} \sigma(X^{(0)}_{i,(lC+k)00}) (l > 0)
    // A^{(0)}_{i,klm} = \sum_{k'} w_{kl,k'} T_{i,k'lm}
    {
        G0.resize(long(num_nodes)*gate_0.dim_out);
        gate_0.forward(num_nodes, X0.data(), G0.data());
        A0.resize(long(num_nodes)*16*num_channels);
        linear_2_0.forward(num_nodes, G0.data(), gate_0.dim_out, A0.data(), 16*num_channels, false);
    }

    // M0
    // M^{(0)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{klm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(0)}_{i,kl'_\xi m'_\xi}
    {
        M0.assign(long(num_nodes)*4*num_channels, 0.0);
        B0.assign(long(num_nodes)*num_B0*num_channels, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A0_i = A0.data()+long(i)*16*num_channels;
            auto M0_i = M0.data()+long(i)*4*num_channels;
            auto B0_i = B0.data()+long(i)*num_B0*num_channels;
            std::copy(A0_i, A0_i+16*num_channels, B0_i);
            for (int t=0; t<B0_spec.size(); ++t) {
                const auto [q, a, b] = B0_spec[t];
                for (int k=0; k<num_channels; ++k)
                    B0_i[q*num_channels+k] += B0_coefficients[t] * B0_i[a*num_channels+k] * B0_i[b*num_channels+k];
            }
            for (int e=0; e<M0_spec.size(); ++e) {
                const auto [o, q, n] = M0_spec[e];
                for (int m=0; m<n; ++m)
                    for (int k=0; k<num_channels; ++k)
                        M0_i[(o+m)*num_channels+k] += M0_weights[e*num_channels+k] * B0_i[(q+m)*num_channels+k];
            }
        }
    }

    // H1
    // H^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(0)}_{i,k'lm} + \delta_{l0} b_{a_i,k},  \tilde{H}^{(1)}_{i,klm} = \sum_{k'} \tilde{w}_{kl,k'} H^{(1)}_{i,k'lm}
    {
        H1.resize(long(num_nodes)*4*num_channels);
        product_linear_0.forward(num_nodes, M0.data(), 4*num_channels, H1.data(), 4*num_channels, false);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i)
            for (int k=0; k<num_channels; ++k)
                H1[long(i)*4*num_channels+k] += skip_0[node_types[i]*num_channels+k];
        H1e.resize(long(num_nodes)*4*num_edge_channels);
        linear_up_1.forward(num_nodes, H1.data(), 4*num_channels, H1e.data(), 4*num_edge_channels, false);
    }
}

void MACEPolar::compute_second_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    const int num_X = off_X.back();

    // Phi1
    // \Phi^{(1)}_{i,klm\eta} = \sum_j \sum_{l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \tilde{H}^{(1)}_{j,kl_2m_2},  s^{(1)}_i = 1 + \sum_j \rho^{(1)}_{a_ia_j}(r_{ij})
    {
        Phi1.assign(long(num_nodes)*num_lme*num_edge_channels, 0.0);
        A1_scale.resize(num_nodes);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1_i = Phi1.data()+long(i)*num_lme*num_edge_channels;
            auto CG_Y = std::vector<double>(Phi1_spec.size());
            auto R1_ij = std::vector<double>(spl_set_1[0]->num_splines);
            double A1_scale_factor = 1.0;
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_ij = node_types[i]*num_elements+neigh_types[ij];
                spl_set_1[type_ij]->evaluate(r[ij], R1_ij);
                A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
                auto Y_ij = Y.data()+ij*16;
                auto H1e_ij = H1e.data()+long(neigh_indices[ij])*4*num_edge_channels;
                std::fill(CG_Y.begin(), CG_Y.end(), 0.0);
                for (int p=0; p<Phi1_cg_spec.size(); ++p) {
                    const auto [q, lm1] = Phi1_cg_spec[p];
                    CG_Y[q] += Phi1_clebsch_gordan[p] * Y_ij[lm1];
                }
                for (int q=0; q<Phi1_spec.size(); ++q) {
                    const auto [lme, lel1l2, lm2] = Phi1_spec[q];
                    auto Phi1_i_lme = Phi1_i+lme*num_edge_channels;
                    auto R1_ij_lel1l2 = R1_ij.data()+lel1l2*num_edge_channels;
                    auto H1e_ij_lm2 = H1e_ij+lm2*num_edge_channels;
                    for (int k=0; k<num_edge_channels; ++k)
                        Phi1_i_lme[k] += CG_Y[q] * R1_ij_lel1l2[k] * H1e_ij_lm2[k];
                }
            }
            A1_scale[i] = 1.0/A1_scale_factor;
        }
    }

    // X1
    // X^{(1)}_{i,klm} = \sum_{k'\eta'} w_{kl,k'\eta'} \Phi^{(1)}_{i,k'lm\eta'} / s^{(1)}_i + \sum_{k'} w'_{kl,k'} \tilde{H}^{(1)}_{i,k'lm}
    {
        X1.resize(long(num_nodes)*num_X);
        linear_1_1.forward(num_nodes, Phi1.data(), num_lme*num_edge_channels, X1.data(), num_X, false);
        X1_residual.resize(long(num_nodes)*num_X);
        residual_1.forward(num_nodes, H1e.data(), 4*num_edge_channels, X1_residual.data(), num_X, false);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i)
            for (int x=0; x<num_X; ++x)
                X1[long(i)*num_X+x] = X1[long(i)*num_X+x]*A1_scale[i] + X1_residual[long(i)*num_X+x];
    }

    // A1
    // A^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} T_{i,k'lm},  T from X^{(1)} as in compute_first_layer
    {
        G1.resize(long(num_nodes)*gate_1.dim_out);
        gate_1.forward(num_nodes, X1.data(), G1.data());
        A1.resize(long(num_nodes)*16*num_channels);
        linear_2_1.forward(num_nodes, G1.data(), gate_1.dim_out, A1.data(), 16*num_channels, false);
    }

    // M1
    // M^{(1)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{klm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(1)}_{i,kl'_\xi m'_\xi}
    {
        M1.assign(long(num_nodes)*4*num_channels, 0.0);
        B1.assign(long(num_nodes)*num_B1*num_channels, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A1_i = A1.data()+long(i)*16*num_channels;
            auto M1_i = M1.data()+long(i)*4*num_channels;
            auto B1_i = B1.data()+long(i)*num_B1*num_channels;
            std::copy(A1_i, A1_i+16*num_channels, B1_i);
            for (int t=0; t<B1_spec.size(); ++t) {
                const auto [q, a, b] = B1_spec[t];
                for (int k=0; k<num_channels; ++k)
                    B1_i[q*num_channels+k] += B1_coefficients[t] * B1_i[a*num_channels+k] * B1_i[b*num_channels+k];
            }
            for (int e=0; e<M1_spec.size(); ++e) {
                const auto [o, q, n] = M1_spec[e];
                for (int m=0; m<n; ++m)
                    for (int k=0; k<num_channels; ++k)
                        M1_i[(o+m)*num_channels+k] += M1_weights[e*num_channels+k] * B1_i[(q+m)*num_channels+k];
            }
        }
    }

    // H2
    // H^{(2)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(1)}_{i,k'lm} + \sum_{k'} w'_{kl,k'} H^{(1)}_{i,k'lm}
    {
        H2.resize(long(num_nodes)*4*num_channels);
        product_linear_1.forward(num_nodes, M1.data(), 4*num_channels, H2.data(), 4*num_channels, false);
        skip_1.forward(num_nodes, H1.data(), 4*num_channels, H2.data(), 4*num_channels, true);
    }
}

void MACEPolar::compute_readouts(
    const int num_nodes,
    std::span<const int> node_types)
{
    // readouts
    // E_i = E^{(0)}_{a_i} + \sum_k w_k H^{(1)}_{i,k00} + \mathrm{MLP}(\{H^{(2)}_{i,k00}\})
    // \bar{H}^{(1)}_{i,k00} = w_k,  \bar{H}^{(2)}_{i,k00} = \partial \mathrm{MLP} / \partial H^{(2)}_{i,k00}
    node_energies.resize(num_nodes);
    H1_adj.assign(long(num_nodes)*4*num_channels, 0.0);
    H2_adj.assign(long(num_nodes)*4*num_channels, 0.0);
    auto E = std::vector<double>(num_nodes);
    readout_1.forward(num_nodes, H2.data(), 4*num_channels, E.data());
    for (int i=0; i<num_nodes; ++i) {
        node_energies[i] = atomic_energies[node_types[i]] + E[i];
        for (int k=0; k<num_channels; ++k) {
            node_energies[i] += readout_0[k]*H1[long(i)*4*num_channels+k];
            H1_adj[long(i)*4*num_channels+k] = readout_0[k];
        }
    }
    auto ones = std::vector<double>(num_nodes, 1.0);
    readout_1.reverse(num_nodes, ones.data(), H2_adj.data(), 4*num_channels);
}

void MACEPolar::compute_charges(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const double> positions,
    std::span<const double> cell,
    double total_charge,
    double total_spin)
{
    const int num_V = 8*feature_sigmas.size();
    position_forces.assign(3*num_nodes, 0.0);
    virial = {};
    if (not coulomb_hook)
        setup_kspace(num_nodes, positions, cell);
    const int num_k = kvec.size();

    // q0
    // q^{(0)}_{i,lms} = \sum_k (w_{ls,k} H^{(1)}_{i,klm} + w'_{ls,k} H^{(2)}_{i,klm}),  q^{(0)}_{i,00s} \leftarrow q^{(0)}_{i,00s} + \frac{f_{i,s}}{\sum_j f_{j,s}} (Q_s - \sum_j q^{(0)}_{j,00s})
    // f_{i,s} = \mathrm{MLP}_s(\{H^{(2)}_{i,k00}\}),  Q_{\uparrow\downarrow} = (Q \pm (S - 1))/2,  \mu_{i,klm} = \sum_{k'} (w_{kl,k'} H^{(1)}_{i,k'lm} + w'_{kl,k'} H^{(2)}_{i,k'lm})
    {
        spin_charges = {(total_charge+total_spin-1)/2, (total_charge-total_spin+1)/2};
        q[0].resize(long(num_nodes)*8);
        sources[0].forward(num_nodes, H1.data(), 4*num_channels, q[0].data(), 8, false);
        sources[1].forward(num_nodes, H2.data(), 4*num_channels, q[0].data(), 8, true);
        f0.resize(2*num_nodes);
        fukui.forward(num_nodes, H2.data(), 4*num_channels, f0.data());
        normalize(num_nodes, 0, f0.data(), 2);
        mixed.resize(long(num_nodes)*4*num_channels);
        mixer[0].forward(num_nodes, H1.data(), 4*num_channels, mixed.data(), 4*num_channels, false);
        mixer[1].forward(num_nodes, H2.data(), 4*num_channels, mixed.data(), 4*num_channels, true);
    }

    // q1, q2
    // V^{(t)}_{i,lms\sigma} = (\phi^{\sigma}_{i,lm}[q^{(t)}_s] - c_{lm\sigma} q^{(t)}_{i,lms}) / n_{lm\sigma},  \phi^{\sigma}_{i,lm}[q] = \sum_{\mathbf{k}} \frac{2\kappa}{\Omega k^2} \mathrm{Re}[\overline{B^{\sigma}_{lm}(\mathbf{k})} e^{i\mathbf{k}\cdot\mathbf{p}_i} \sum_{jl'm'} B_{l'm'}(\mathbf{k}) e^{-i\mathbf{k}\cdot\mathbf{p}_j} q_{j,l'm'}]
    // B_{lm}(\mathbf{k}) = a_l (-ik)^l e^{-\sigma_\rho^2k^2/2} Y_{lm}(\hat{\mathbf{k}}),  B^{\sigma}_{lm} with a^{\sigma}_l and \sigma,  \mathbf{k} over half the sphere k \le k_c,  \Omega the cell volume
    // e_{i,klm} = \sum_{s\sigma} w_{kl,s\sigma} V^{(t)}_{i,lms\sigma} + \sum_{k'} w_{kl,k'} \mu_{i,k'lm} + \sum_s w_{kl,s} q^{(t)}_{i,lms},  z_i = \mathrm{MLP}(\{\sum_{lm} \mu_{i,klm} \sum_{k'} D_{l,kk'} e_{i,k'lm}\}_k, u_{a_i})
    // q^{(t+1)}_{i,lms} = q^{(t)}_{i,lms} + \mathcal{G}_{lms}(\{\mu_{i,klm} \sum_{k'} M_{l,kk'} z_{i,k'}\}),  normalized as q^{(0)} with f_{i,s} = \mathcal{G}_{f,s}(\ldots)
    for (int t=0; t<2; ++t) {
        auto& u = updates[t];
        V[t].resize(long(num_nodes)*num_V);
        features(num_nodes, q[t].data(), structure_factors[t], basis_factors[t], V[t].data());
        u.embedding.resize(long(num_nodes)*4*num_channels);
        u.potential.forward(num_nodes, V[t].data(), num_V, u.embedding.data(), 4*num_channels, false);
        u.node.forward(num_nodes, mixed.data(), 4*num_channels, u.embedding.data(), 4*num_channels, true);
        u.charge.forward(num_nodes, q[t].data(), 8, u.embedding.data(), 4*num_channels, true);
        u.mlp_in.resize(long(num_nodes)*2*num_channels);
        u.dot.forward(num_nodes, mixed.data(), u.embedding.data(), u.mlp_in.data(), 2*num_channels);
        for (int i=0; i<num_nodes; ++i)
            std::copy(u.source.begin()+node_types[i]*num_channels, u.source.begin()+(node_types[i]+1)*num_channels,
                      u.mlp_in.begin()+long(i)*2*num_channels+num_channels);
        u.mlp_out.resize(long(num_nodes)*num_channels);
        u.mlp.forward(num_nodes, u.mlp_in.data(), u.mlp_out.data());
        u.modulated.resize(long(num_nodes)*4*num_channels);
        u.out.forward(num_nodes, mixed.data(), u.mlp_out.data(), u.modulated.data());
        u.readout_out.resize(long(num_nodes)*10);
        u.readout.forward(num_nodes, u.modulated.data(), 4*num_channels, u.readout_out.data());
        q[t+1] = q[t];
        for (int i=0; i<num_nodes; ++i)
            for (int s=0; s<2; ++s) {
                q[t+1][i*8+s] += u.readout_out[i*10+s];
                for (int m=0; m<3; ++m)
                    q[t+1][i*8+(1+m)*2+s] += u.readout_out[i*10+4+2*m+s];
            }
        normalize(num_nodes, t+1, u.readout_out.data()+2, 10);
    }

    // electron energy
    // E_i += E^{\mathrm{e}}_i = \mathrm{MLP}(\{\sum_{lm} H^{(2)}_{i,klm} \sum_{k'} D_{l,kk'} \hat{q}_{i,k'lm}\}_k, \{\sum_{lm} H^{(2)}_{i,klm} \sum_{k'} D'_{l,kk'} \hat{v}_{i,k'lm}\}_k)
    // \hat{q}_{i,klm} = \sum_s w_{kl,s} (q^{(0)}_{i,lms} + q^{(2)}_{i,lms}) + \delta_{l0} b_k,  \hat{v}_{i,klm} = \sum_{s\sigma} w_{kl,s\sigma} V^{(1)}_{i,lms\sigma} + \delta_{l0} b'_k
    {
        auto& e = electron;
        e.q_sum.resize(long(num_nodes)*8);
        for (long x=0; x<long(num_nodes)*8; ++x)
            e.q_sum[x] = q[2][x]+q[0][x];
        e.q_up.resize(long(num_nodes)*4*num_channels);
        e.q.forward(num_nodes, e.q_sum.data(), 8, e.q_up.data(), 4*num_channels, false);
        e.v_up.resize(long(num_nodes)*4*num_channels);
        e.v.forward(num_nodes, V[1].data(), num_V, e.v_up.data(), 4*num_channels, false);
        e.mlp_in.resize(long(num_nodes)*2*num_channels);
        e.dot_q.forward(num_nodes, H2.data(), e.q_up.data(), e.mlp_in.data(), 2*num_channels);
        e.dot_v.forward(num_nodes, H2.data(), e.v_up.data(), e.mlp_in.data()+num_channels, 2*num_channels);
        auto E = std::vector<double>(num_nodes);
        e.mlp.forward(num_nodes, e.mlp_in.data(), E.data());
        for (int i=0; i<num_nodes; ++i)
            node_energies[i] += E[i];
    }

    // electrostatic energy
    // \rho_{i,lm} = \sum_s q^{(2)}_{i,lms},  \psi_{i,lm} = \frac{1}{2} \phi_{i,lm}[\rho] (\phi^{\sigma} with B_{lm} for B^{\sigma}_{lm}),  E_i += \sum_{lm} \rho_{i,lm} \psi_{i,lm}
    {
        rho.resize(long(num_nodes)*4);
        for (int i=0; i<num_nodes; ++i)
            for (int lm=0; lm<4; ++lm)
                rho[i*4+lm] = q[2][i*8+lm*2]+q[2][i*8+lm*2+1];
        psi.assign(long(num_nodes)*4, 0.0);
        if (coulomb_hook) {
            auto rho_inv = rho;
            for (int i=0; i<num_nodes; ++i)
                for (int lm=1; lm<4; ++lm)
                    rho_inv[i*4+lm] = -rho[i*4+lm];
            coulomb_hook(rho_inv.data(), psi.data(), position_forces.data(), virial.data());
            for (int i=0; i<num_nodes; ++i)
                for (int lm=1; lm<4; ++lm)
                    psi[i*4+lm] = -psi[i*4+lm];
        } else {
            rho_structure.resize(4*num_k);
            rho_basis.resize(num_k);
            structure(num_nodes, rho.data(), 4, rho_structure.data());
            on_basis(rho_structure.data(), 1, false, rho_basis.data());
            auto Z = std::vector<cplx>(num_k);
            for (int k=0; k<num_k; ++k)
                Z[k] = field_constant/volume*k2inv[k]*rho_basis[k];
            project(num_nodes, Z.data(), 1, false, psi.data());
            phase_forces(num_nodes, rho_basis.data(), rho.data(), 1, false, 2.0*field_constant/volume,
                         position_forces.data());
            kspace_virial(rho_structure.data(), false, rho_structure.data(), false, 1,
                          field_constant/volume, virial.data());
        }
        electrostatic_energy = 0.0;
        for (int i=0; i<num_nodes; ++i) {
            const double E_i = std::inner_product(rho.begin()+i*4, rho.begin()+i*4+4, psi.begin()+i*4, 0.0);
            node_energies[i] += E_i;
            electrostatic_energy += E_i;
        }
    }
}

void MACEPolar::reverse_charges(
    const int num_nodes)
{
    const int num_V = 8*feature_sigmas.size();
    mixed_adj.assign(long(num_nodes)*4*num_channels, 0.0);
    for (int t=0; t<3; ++t)
        q_adj[t].assign(long(num_nodes)*8, 0.0);
    for (int t=0; t<2; ++t)
        V_adj[t].assign(long(num_nodes)*num_V, 0.0);

    // electron energy
    // \bar{H}^{(2)}_{i,klm} += \partial E^{\mathrm{e}}_i / \partial H^{(2)}_{i,klm},  \bar{q}^{(0)}_{i,lms} = \bar{q}^{(2)}_{i,lms} = \partial E^{\mathrm{e}}_i / \partial q^{(2)}_{i,lms},  \bar{V}^{(1)}_{i,lms\sigma} = \partial E^{\mathrm{e}}_i / \partial V^{(1)}_{i,lms\sigma}
    {
        auto& e = electron;
        auto ones = std::vector<double>(num_nodes, 1.0);
        auto mlp_in_adj = std::vector<double>(long(num_nodes)*2*num_channels);
        auto q_up_adj = std::vector<double>(long(num_nodes)*4*num_channels, 0.0);
        auto v_up_adj = std::vector<double>(long(num_nodes)*4*num_channels, 0.0);
        e.mlp.reverse(num_nodes, ones.data(), mlp_in_adj.data());
        e.dot_q.reverse(num_nodes, H2.data(), mlp_in_adj.data(), 2*num_channels, H2_adj.data(), q_up_adj.data());
        e.dot_v.reverse(num_nodes, H2.data(), mlp_in_adj.data()+num_channels, 2*num_channels,
                        H2_adj.data(), v_up_adj.data());
        e.q.reverse(num_nodes, q_up_adj.data(), 4*num_channels, q_adj[2].data(), 8);
        e.q.reverse(num_nodes, q_up_adj.data(), 4*num_channels, q_adj[0].data(), 8);
        e.v.reverse(num_nodes, v_up_adj.data(), 4*num_channels, V_adj[1].data(), num_V);
    }

    // electrostatic energy
    // \bar{q}^{(2)}_{i,lms} += 2 \psi_{i,lm}
    for (int i=0; i<num_nodes; ++i)
        for (int lm=0; lm<4; ++lm)
            for (int s=0; s<2; ++s)
                q_adj[2][i*8+lm*2+s] += 2.0*psi[i*4+lm];

    // q2, q1
    // \bar{V}^{(t)} += \bar{q}^{(t+1)} \cdot \partial q^{(t+1)} / \partial V^{(t)},  \bar{\mu} += \bar{q}^{(t+1)} \cdot \partial q^{(t+1)} / \partial \mu,  \bar{q}^{(t)} += \bar{q}^{(t+1)} \cdot \partial q^{(t+1)} / \partial q^{(t)} + \bar{V}^{(t)} \cdot \partial V^{(t)} / \partial q^{(t)}
    // \mathbf{F}_i -= \bar{V}^{(t)} \cdot \partial V^{(t)} / \partial \mathbf{x}_i
    for (int t=1; t>=0; --t) {
        auto& u = updates[t];
        auto readout_out_adj = std::vector<double>(long(num_nodes)*10);
        normalize_reverse(num_nodes, t+1, readout_out_adj.data()+2, 10);
        for (int i=0; i<num_nodes; ++i)
            for (int s=0; s<2; ++s) {
                readout_out_adj[i*10+s] = q_adj[t+1][i*8+s];
                for (int m=0; m<3; ++m)
                    readout_out_adj[i*10+4+2*m+s] = q_adj[t+1][i*8+(1+m)*2+s];
            }
        for (long x=0; x<long(num_nodes)*8; ++x)
            q_adj[t][x] += q_adj[t+1][x];
        auto modulated_adj = std::vector<double>(long(num_nodes)*4*num_channels, 0.0);
        auto mlp_out_adj = std::vector<double>(long(num_nodes)*num_channels, 0.0);
        auto mlp_in_adj = std::vector<double>(long(num_nodes)*2*num_channels);
        auto embedding_adj = std::vector<double>(long(num_nodes)*4*num_channels, 0.0);
        u.readout.reverse(num_nodes, readout_out_adj.data(), modulated_adj.data(), 4*num_channels);
        u.out.reverse(num_nodes, mixed.data(), modulated_adj.data(), mixed_adj.data(), mlp_out_adj.data());
        u.mlp.reverse(num_nodes, mlp_out_adj.data(), mlp_in_adj.data());
        u.dot.reverse(num_nodes, mixed.data(), mlp_in_adj.data(), 2*num_channels, mixed_adj.data(),
                      embedding_adj.data());
        u.potential.reverse(num_nodes, embedding_adj.data(), 4*num_channels, V_adj[t].data(), num_V);
        u.node.reverse(num_nodes, embedding_adj.data(), 4*num_channels, mixed_adj.data(), 4*num_channels);
        u.charge.reverse(num_nodes, embedding_adj.data(), 4*num_channels, q_adj[t].data(), 8);
        features_reverse(num_nodes, q[t].data(), structure_factors[t], basis_factors[t],
                         V_adj[t].data(), q_adj[t].data(),
                         position_forces.data(), virial.data());
    }

    // q0
    // \bar{H}^{(1)} += \bar{q}^{(0)} \cdot \partial q^{(0)} / \partial H^{(1)} + \bar{\mu} \cdot \partial \mu / \partial H^{(1)},  \bar{H}^{(2)} += \bar{q}^{(0)} \cdot \partial q^{(0)} / \partial H^{(2)} + \bar{\mu} \cdot \partial \mu / \partial H^{(2)}
    {
        auto f0_adj = std::vector<double>(2*num_nodes);
        normalize_reverse(num_nodes, 0, f0_adj.data(), 2);
        fukui.reverse(num_nodes, f0_adj.data(), H2_adj.data(), 4*num_channels);
        sources[0].reverse(num_nodes, q_adj[0].data(), 8, H1_adj.data(), 4*num_channels);
        sources[1].reverse(num_nodes, q_adj[0].data(), 8, H2_adj.data(), 4*num_channels);
        mixer[0].reverse(num_nodes, mixed_adj.data(), 4*num_channels, H1_adj.data(), 4*num_channels);
        mixer[1].reverse(num_nodes, mixed_adj.data(), 4*num_channels, H2_adj.data(), 4*num_channels);
    }
}

void MACEPolar::reverse_second_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    const int num_X = off_X.back();
    node_forces.assign(xyz.size(), 0.0);

    // H2
    // \bar{M}^{(1)}_{i,k'lm} = \sum_k w_{kl,k'} \bar{H}^{(2)}_{i,klm},  \bar{H}^{(1)}_{i,k'lm} += \sum_k w'_{kl,k'} \bar{H}^{(2)}_{i,klm}
    {
        M1_adj.assign(long(num_nodes)*4*num_channels, 0.0);
        product_linear_1.reverse(num_nodes, H2_adj.data(), 4*num_channels, M1_adj.data(), 4*num_channels);
        skip_1.reverse(num_nodes, H2_adj.data(), 4*num_channels, H1_adj.data(), 4*num_channels);
    }

    // M1
    // \bar{A}^{(1)}_{i,klm} = \sum_{l'm'} \bar{M}^{(1)}_{i,kl'm'} \partial M^{(1)}_{i,kl'm'} / \partial A^{(1)}_{i,klm}
    {
        A1_adj.assign(A1.size(), 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto B1_adj_i = std::vector<double>(num_B1*num_channels);
            auto A1_adj_i = A1_adj.data()+long(i)*16*num_channels;
            auto M1_adj_i = M1_adj.data()+long(i)*4*num_channels;
            auto B1_i = B1.data()+long(i)*num_B1*num_channels;
            for (int e=0; e<M1_spec.size(); ++e) {
                const auto [o, q, n] = M1_spec[e];
                for (int m=0; m<n; ++m)
                    for (int k=0; k<num_channels; ++k)
                        B1_adj_i[(q+m)*num_channels+k] += M1_weights[e*num_channels+k] * M1_adj_i[(o+m)*num_channels+k];
            }
            for (int t=B1_spec.size()-1; t>=0; --t) {
                const auto [q, a, b] = B1_spec[t];
                for (int k=0; k<num_channels; ++k) {
                    const double g = B1_coefficients[t] * B1_adj_i[q*num_channels+k];
                    B1_adj_i[a*num_channels+k] += g * B1_i[b*num_channels+k];
                    B1_adj_i[b*num_channels+k] += g * B1_i[a*num_channels+k];
                }
            }
            std::copy(B1_adj_i.begin(), B1_adj_i.begin()+16*num_channels, A1_adj_i);
        }
    }

    // A1
    // \bar{T}_{i,k'lm} = \sum_k w_{kl,k'} \bar{A}^{(1)}_{i,klm},  \bar{X}^{(1)}_{i,k00} = c_s \bar{T}_{i,k00} \sigma(x) (1 + x (1 - \sigma(x))),  x = X^{(1)}_{i,k00}
    // \bar{X}^{(1)}_{i,klm} = c_g \bar{T}_{i,klm} \sigma(\gamma),  \bar{X}^{(1)}_{i,(lC+k)00} = c_g \sigma(\gamma) (1 - \sigma(\gamma)) \sum_m X^{(1)}_{i,klm} \bar{T}_{i,klm},  \gamma = X^{(1)}_{i,(lC+k)00} (l > 0)
    {
        G1_adj.assign(G1.size(), 0.0);
        linear_2_1.reverse(num_nodes, A1_adj.data(), 16*num_channels, G1_adj.data(), gate_1.dim_out);
        X1_adj.resize(X1.size());
        gate_1.reverse(num_nodes, G1_adj.data(), X1_adj.data());
    }

    // X1
    // \bar{\tilde{H}}^{(1)}_{i,k'lm} = \sum_k w'_{kl,k'} \bar{X}^{(1)}_{i,klm},  \bar{\Phi}^{(1)}_{i,k'lm\eta'} = \sum_k w_{kl,k'\eta'} \bar{X}^{(1)}_{i,klm} / s^{(1)}_i
    // d^{(1)}_i = \sum_{klm} \bar{X}^{(1)}_{i,klm} (X^{(1)}_{i,klm} - \sum_{k'} w'_{kl,k'} \tilde{H}^{(1)}_{i,k'lm}) / s^{(1)}_i
    // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(1)}_i
    {
        H1e_adj.assign(H1e.size(), 0.0);
        residual_1.reverse(num_nodes, X1_adj.data(), num_X, H1e_adj.data(), 4*num_edge_channels);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            auto X1_i = X1.data()+long(i)*num_X;
            auto X1_residual_i = X1_residual.data()+long(i)*num_X;
            auto X1_adj_i = X1_adj.data()+long(i)*num_X;
            double dX1_dot_X1 = 0.0;
            for (int x=0; x<num_X; ++x)
                dX1_dot_X1 += X1_adj_i[x] * (X1_i[x] - X1_residual_i[x]);
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                auto [f,d] = A1_splines[type_i*num_elements+neigh_types[ij]].evaluate_deriv(r[ij]);
                for (int a=0; a<3; ++a)
                    node_forces[3*ij+a] += dX1_dot_X1*A1_scale[i]*d*xyz[3*ij+a]/r[ij];
            }
            for (int x=0; x<num_X; ++x)
                X1_adj_i[x] *= A1_scale[i];
        }
        dPhi1.assign(Phi1.size(), 0.0);
        linear_1_1.reverse(num_nodes, X1_adj.data(), num_X, dPhi1.data(), num_lme*num_edge_channels);
    }

    // Phi1
    // \bar{\tilde{H}}^{(1)}_{j,kl_2m_2} += \sum_i \sum_{lm\eta l_1m_1} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(1)}_{i,klm\eta}
    // \mathbf{F}_{ij} -= \sum_{klm\eta l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij})] \tilde{H}^{(1)}_{j,kl_2m_2} \bar{\Phi}^{(1)}_{i,klm\eta}
    {
        std::vector<std::mutex> H1e_adj_locks(H1e.size()/(4*num_edge_channels));
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            auto dPhi1_i = dPhi1.data()+long(i)*num_lme*num_edge_channels;
            auto CG_Y = std::vector<double>(Phi1_spec.size());
            auto CG_Y_adj = std::vector<double>(Phi1_spec.size());
            auto H1e_adj_ij = std::vector<double>(4*num_edge_channels);
            auto R1_ij = std::vector<double>(spl_set_1[0]->num_splines);
            auto R1_deriv_ij = std::vector<double>(R1_ij.size());
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                auto node_forces_ij = node_forces.data()+3*ij;
                auto xyz_ij = xyz.data()+3*ij;
                const double r_ij = r[ij];
                spl_set_1[type_i*num_elements+neigh_types[ij]]->evaluate_derivs(r_ij, R1_ij, R1_deriv_ij);
                auto Y_ij = Y.data()+ij*16;
                auto Y_grad_ij = Y_grad.data()+ij*3*16;
                const int j = neigh_indices[ij];
                auto H1e_ij = H1e.data()+long(j)*4*num_edge_channels;
                std::fill(CG_Y.begin(), CG_Y.end(), 0.0);
                std::fill(H1e_adj_ij.begin(), H1e_adj_ij.end(), 0.0);
                for (int p=0; p<Phi1_cg_spec.size(); ++p) {
                    const auto [q, lm1] = Phi1_cg_spec[p];
                    CG_Y[q] += Phi1_clebsch_gordan[p] * Y_ij[lm1];
                }
                double dE_dr = 0.0;
                for (int q=0; q<Phi1_spec.size(); ++q) {
                    const auto [lme, lel1l2, lm2] = Phi1_spec[q];
                    auto dPhi1_i_lme = dPhi1_i+lme*num_edge_channels;
                    auto R1_ij_lel1l2 = R1_ij.data()+lel1l2*num_edge_channels;
                    auto R1_deriv_ij_lel1l2 = R1_deriv_ij.data()+lel1l2*num_edge_channels;
                    auto H1e_ij_lm2 = H1e_ij+lm2*num_edge_channels;
                    auto H1e_adj_ij_lm2 = H1e_adj_ij.data()+lm2*num_edge_channels;
                    double g = 0.0;
                    for (int k=0; k<num_edge_channels; ++k) {
                        g += dPhi1_i_lme[k] * R1_ij_lel1l2[k] * H1e_ij_lm2[k];
                        dE_dr += dPhi1_i_lme[k] * CG_Y[q] * R1_deriv_ij_lel1l2[k] * H1e_ij_lm2[k];
                        H1e_adj_ij_lm2[k] += dPhi1_i_lme[k] * CG_Y[q] * R1_ij_lel1l2[k];
                    }
                    CG_Y_adj[q] = g;
                }
                for (int a=0; a<3; ++a)
                    node_forces_ij[a] -= dE_dr * xyz_ij[a]/r_ij;
                for (int p=0; p<Phi1_cg_spec.size(); ++p) {
                    const auto [q, lm1] = Phi1_cg_spec[p];
                    for (int a=0; a<3; ++a)
                        node_forces_ij[a] -= Phi1_clebsch_gordan[p] * Y_grad_ij[a*16+lm1] * CG_Y_adj[q];
                }
                {
                    std::scoped_lock lock(H1e_adj_locks[j]);
                    for (int x=0; x<4*num_edge_channels; ++x)
                        H1e_adj[long(j)*4*num_edge_channels+x] += H1e_adj_ij[x];
                }
            }
        }
    }
}

void MACEPolar::reverse_first_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    const int num_X = off_X.back();
    const int num_Phi0 = 16*num_elements*num_basis_0;

    // H1
    // \bar{H}^{(1)}_{i,k'lm} += \sum_k \tilde{w}_{kl,k'} \bar{\tilde{H}}^{(1)}_{i,klm},  \bar{M}^{(0)}_{i,k'lm} = \sum_k w_{kl,k'} \bar{H}^{(1)}_{i,klm}
    {
        linear_up_1.reverse(num_nodes, H1e_adj.data(), 4*num_edge_channels, H1_adj.data(), 4*num_channels);
        M0_adj.assign(M0.size(), 0.0);
        product_linear_0.reverse(num_nodes, H1_adj.data(), 4*num_channels, M0_adj.data(), 4*num_channels);
    }

    // M0
    // \bar{A}^{(0)}_{i,klm} = \sum_{l'm'} \bar{M}^{(0)}_{i,kl'm'} \partial M^{(0)}_{i,kl'm'} / \partial A^{(0)}_{i,klm}
    {
        A0_adj.assign(A0.size(), 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto B0_adj_i = std::vector<double>(num_B0*num_channels);
            auto A0_adj_i = A0_adj.data()+long(i)*16*num_channels;
            auto M0_adj_i = M0_adj.data()+long(i)*4*num_channels;
            auto B0_i = B0.data()+long(i)*num_B0*num_channels;
            for (int e=0; e<M0_spec.size(); ++e) {
                const auto [o, q, n] = M0_spec[e];
                for (int m=0; m<n; ++m)
                    for (int k=0; k<num_channels; ++k)
                        B0_adj_i[(q+m)*num_channels+k] += M0_weights[e*num_channels+k] * M0_adj_i[(o+m)*num_channels+k];
            }
            for (int t=B0_spec.size()-1; t>=0; --t) {
                const auto [q, a, b] = B0_spec[t];
                for (int k=0; k<num_channels; ++k) {
                    const double g = B0_coefficients[t] * B0_adj_i[q*num_channels+k];
                    B0_adj_i[a*num_channels+k] += g * B0_i[b*num_channels+k];
                    B0_adj_i[b*num_channels+k] += g * B0_i[a*num_channels+k];
                }
            }
            std::copy(B0_adj_i.begin(), B0_adj_i.begin()+16*num_channels, A0_adj_i);
        }
    }

    // A0
    // \bar{T}_{i,k'lm} = \sum_k w_{kl,k'} \bar{A}^{(0)}_{i,klm},  \bar{X}^{(0)} from \bar{T} as in reverse_second_layer
    {
        G0_adj.assign(G0.size(), 0.0);
        linear_2_0.reverse(num_nodes, A0_adj.data(), 16*num_channels, G0_adj.data(), gate_0.dim_out);
        X0_adj.resize(X0.size());
        gate_0.reverse(num_nodes, G0_adj.data(), X0_adj.data());
    }

    // X0
    // \bar{\Phi}^{(0)}_{i,lm,bn} = \sum_k w_{kl,bn} \bar{X}^{(0)}_{i,klm} / s^{(0)}_i,  d^{(0)}_i = \sum_{klm} \bar{X}^{(0)}_{i,klm} (X^{(0)}_{i,klm} - \delta_{l0} r_{a_i,k}) / s^{(0)}_i
    auto dX0_dot_X0 = std::vector<double>(num_nodes);
    {
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto X0_i = X0.data()+long(i)*num_X;
            auto X0_adj_i = X0_adj.data()+long(i)*num_X;
            double d = 0.0;
            for (int x=0; x<num_X; ++x)
                d += X0_adj_i[x]*X0_i[x];
            for (int k=0; k<mul_X[0]; ++k)
                d -= X0_adj_i[k]*residual_0[node_types[i]*mul_X[0]+k];
            dX0_dot_X0[i] = d;
            for (int x=0; x<num_X; ++x)
                X0_adj_i[x] *= A0_scale[i];
        }
        Phi0_adj.assign(Phi0.size(), 0.0);
        linear_0.reverse(num_nodes, X0_adj.data(), num_X, Phi0_adj.data(), num_Phi0);
    }

    // Phi0
    // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(0)}_i - \sum_{lmn} \nabla_{\mathbf{r}_{ij}}[R^{(0)}_{a_ia_j,n}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})] \bar{\Phi}^{(0)}_{i,lm,a_jn}
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            auto Phi0_adj_i = Phi0_adj.data()+long(i)*num_Phi0;
            auto R0_ij = std::vector<double>(num_basis_0);
            auto R0_deriv_ij = std::vector<double>(num_basis_0);
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_ij = node_types[i]*num_elements+neigh_types[ij];
                spl_set_0[type_ij]->evaluate_derivs(r[ij], R0_ij, R0_deriv_ij);
                auto [f,d] = A0_splines[type_ij].evaluate_deriv(r[ij]);
                double dE_dr = -dX0_dot_X0[i]*A0_scale[i]*d;
                double dE_dY[3] = {};
                for (int lm=0; lm<16; ++lm) {
                    auto Phi0_adj_i_lmb = Phi0_adj_i+(lm*num_elements+neigh_types[ij])*num_basis_0;
                    double dE_dR = 0.0;
                    double dE_dR_deriv = 0.0;
                    for (int n=0; n<num_basis_0; ++n) {
                        dE_dR += Phi0_adj_i_lmb[n]*R0_ij[n];
                        dE_dR_deriv += Phi0_adj_i_lmb[n]*R0_deriv_ij[n];
                    }
                    dE_dr += dE_dR_deriv*Y[ij*16+lm];
                    for (int a=0; a<3; ++a)
                        dE_dY[a] += dE_dR*Y_grad[(3*ij+a)*16+lm];
                }
                for (int a=0; a<3; ++a)
                    node_forces[3*ij+a] -= dE_dr*xyz[3*ij+a]/r[ij] + dE_dY[a];
            }
        }
    }

    // virial
    // W += \sum_{ij} \mathrm{sym}(\mathbf{r}_{ij} \otimes \mathbf{F}_{ij})
    for (long ij=0; ij<long(xyz.size())/3; ++ij) {
        const double* x = xyz.data()+3*ij;
        const double* f = node_forces.data()+3*ij;
        virial[0] += x[0]*f[0];
        virial[1] += x[1]*f[1];
        virial[2] += x[2]*f[2];
        virial[3] += 0.5*(x[0]*f[1]+x[1]*f[0]);
        virial[4] += 0.5*(x[0]*f[2]+x[2]*f[0]);
        virial[5] += 0.5*(x[1]*f[2]+x[2]*f[1]);
    }
}

void MACEPolar::normalize(int num_nodes, int t, const double* f, int ldf)
{
    fukui_n[t].resize(2*num_nodes);
    double sums[4] = {};
    for (int i=0; i<num_nodes; ++i)
        for (int s=0; s<2; ++s) {
            sums[s] += f[i*ldf+s];
            sums[2+s] += q[t][i*8+s];
        }
    allreduce(sums, 4);
    for (int s=0; s<2; ++s) {
        fukui_sum[t][s] = sums[s];
        fukui_c[t][s] = spin_charges[s]-sums[2+s];
        for (int i=0; i<num_nodes; ++i) {
            fukui_n[t][2*i+s] = f[i*ldf+s]/(sums[s] == 0.0 ? 1.0 : sums[s]);
            q[t][i*8+s] += fukui_n[t][2*i+s]*fukui_c[t][s];
        }
    }
}

void MACEPolar::normalize_reverse(int num_nodes, int t, double* f_adj, int ldf)
{
    double sums[2] = {};
    for (int i=0; i<num_nodes; ++i)
        for (int s=0; s<2; ++s)
            sums[s] += q_adj[t][i*8+s]*fukui_n[t][2*i+s];
    allreduce(sums, 2);
    for (int s=0; s<2; ++s) {
        const double F = fukui_sum[t][s];
        for (int i=0; i<num_nodes; ++i) {
            const double a = q_adj[t][i*8+s];
            f_adj[i*ldf+s] = fukui_c[t][s]*(F == 0.0 ? a : (a-sums[s])/F);
            q_adj[t][i*8+s] = a-sums[s];
        }
    }
}

static double sigmoid(double x)
{
    return 1.0/(1.0+std::exp(-x));
}

static double silu_deriv(double x)
{
    const double s = sigmoid(x);
    return s*(1.0+x*(1.0-s));
}

void MACEPolar::Linear::forward(int n, const double* x, int ldx, double* y, int ldy, bool add) const
{
    #pragma omp parallel for
    for (int i0=0; i0<n; i0+=4) {
        for (int l=0; l<out.size(); ++l)
            for (int m=0; m<2*l+1; ++m) {
                const double* x_ilm[4];
                for (int b=0; b<4; ++b)
                    x_ilm[b] = x+long(std::min(i0+b,n-1))*ldx+off_in[l]+m*in[l];
                for (int k=0; k<out[l]; ++k) {
                    const double* W_trans_lk = weights_trans[l].data()+k*in[l];
                    double y_ilm_k[4] = {};
                    for (int kp=0; kp<in[l]; ++kp)
                        for (int b=0; b<4; ++b)
                            y_ilm_k[b] += x_ilm[b][kp] * W_trans_lk[kp];
                    for (int b=0; b<4 and i0+b<n; ++b) {
                        double& y_ilmk = y[long(i0+b)*ldy+off_out[l]+m*out[l]+k];
                        y_ilmk = (add ? y_ilmk : 0.0) + y_ilm_k[b];
                    }
                }
            }
        for (int b=0; b<4 and i0+b<n; ++b)
            for (int k=0; k<bias.size(); ++k)
                y[long(i0+b)*ldy+k] += bias[k];
    }
}

void MACEPolar::Linear::reverse(int n, const double* y_adj, int ldy, double* x_adj, int ldx) const
{
    #pragma omp parallel for
    for (int i0=0; i0<n; i0+=4) {
        for (int l=0; l<out.size(); ++l)
            for (int m=0; m<2*l+1; ++m) {
                const double* y_adj_ilm[4];
                for (int b=0; b<4; ++b)
                    y_adj_ilm[b] = y_adj+long(std::min(i0+b,n-1))*ldy+off_out[l]+m*out[l];
                for (int kp=0; kp<in[l]; ++kp) {
                    const double* W_lkp = weights[l].data()+kp*out[l];
                    double x_adj_ilm_kp[4] = {};
                    for (int k=0; k<out[l]; ++k)
                        for (int b=0; b<4; ++b)
                            x_adj_ilm_kp[b] += y_adj_ilm[b][k] * W_lkp[k];
                    for (int b=0; b<4 and i0+b<n; ++b)
                        x_adj[long(i0+b)*ldx+off_in[l]+m*in[l]+kp] += x_adj_ilm_kp[b];
                }
            }
    }
}

void MACEPolar::MLP::forward(int n, const double* x_in, double* y)
{
    auto x = std::vector<double>(x_in, x_in+long(n)*layers[0].in);
    auto z = std::vector<double>();
    x_hat.resize(layers.size());
    rstd.resize(layers.size());
    for (int t=0; t<layers.size(); ++t) {
        const auto& L = layers[t];
        z.resize(long(n)*L.out);
        #pragma omp parallel for
        for (int i0=0; i0<n; i0+=4) {
            const double* x_i[4];
            for (int b=0; b<4; ++b)
                x_i[b] = x.data()+long(std::min(i0+b,n-1))*L.in;
            for (int k=0; k<L.out; ++k) {
                const double* W_trans_k = L.weights_trans.data()+k*L.in;
                double z_ik[4] = {};
                for (int kp=0; kp<L.in; ++kp)
                    for (int b=0; b<4; ++b)
                        z_ik[b] += x_i[b][kp] * W_trans_k[kp];
                for (int b=0; b<4 and i0+b<n; ++b)
                    z[long(i0+b)*L.out+k] = L.bias[k] + z_ik[b];
            }
        }
        if (L.ln_weights.empty())
            continue;
        x_hat[t].resize(long(n)*L.out);
        rstd[t].resize(n);
        x.resize(long(n)*L.out);
        #pragma omp parallel for
        for (int i=0; i<n; ++i) {
            auto z_i = z.data()+long(i)*L.out;
            const double mean = std::accumulate(z_i, z_i+L.out, 0.0)/L.out;
            double var = 0.0;
            for (int k=0; k<L.out; ++k)
                var += (z_i[k]-mean)*(z_i[k]-mean);
            rstd[t][i] = 1.0/std::sqrt(var/L.out+L.ln_eps);
            for (int k=0; k<L.out; ++k) {
                const double xh = (z_i[k]-mean)*rstd[t][i];
                const double v = L.ln_weights[k]*xh+L.ln_bias[k];
                x_hat[t][long(i)*L.out+k] = xh;
                x[long(i)*L.out+k] = v*sigmoid(v);
            }
        }
    }
    std::copy(z.begin(), z.end(), y);
}

void MACEPolar::MLP::reverse(int n, const double* y_adj, double* x_adj) const
{
    auto g = std::vector<double>(y_adj, y_adj+long(n)*layers.back().out);
    auto h = std::vector<double>();
    for (int t=layers.size()-1; t>=0; --t) {
        const auto& L = layers[t];
        if (not L.ln_weights.empty()) {
            #pragma omp parallel for
            for (int i=0; i<n; ++i) {
                auto g_i = g.data()+long(i)*L.out;
                auto xh = x_hat[t].data()+long(i)*L.out;
                double m1 = 0.0;
                double m2 = 0.0;
                for (int k=0; k<L.out; ++k) {
                    g_i[k] *= silu_deriv(L.ln_weights[k]*xh[k]+L.ln_bias[k])*L.ln_weights[k];
                    m1 += g_i[k];
                    m2 += g_i[k]*xh[k];
                }
                for (int k=0; k<L.out; ++k)
                    g_i[k] = rstd[t][i]*(g_i[k]-m1/L.out-xh[k]*m2/L.out);
            }
        }
        h.resize(long(n)*L.in);
        #pragma omp parallel for
        for (int i0=0; i0<n; i0+=4) {
            const double* g_i[4];
            for (int b=0; b<4; ++b)
                g_i[b] = g.data()+long(std::min(i0+b,n-1))*L.out;
            for (int kp=0; kp<L.in; ++kp) {
                const double* W_kp = L.weights.data()+kp*L.out;
                double h_ikp[4] = {};
                for (int k=0; k<L.out; ++k)
                    for (int b=0; b<4; ++b)
                        h_ikp[b] += g_i[b][k] * W_kp[k];
                for (int b=0; b<4 and i0+b<n; ++b)
                    h[long(i0+b)*L.in+kp] = h_ikp[b];
            }
        }
        std::swap(g, h);
    }
    std::copy(g.begin(), g.end(), x_adj);
}

void MACEPolar::Gate::forward(int n, const double* x_in, double* y)
{
    x.assign(x_in, x_in+long(n)*dim_in);
    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
        const double* x_i = x_in+long(i)*dim_in;
        double* y_i = y+long(i)*dim_out;
        for (int k=0; k<scalars; ++k)
            y_i[k] = scalar_constant*x_i[k]*sigmoid(x_i[k]);
        int g = scalars;
        int o_in = mul[0];
        int o_out = scalars;
        for (int l=1; l<mul.size(); ++l) {
            for (int k=0; k<mul[l]; ++k) {
                const double s = gate_constant*sigmoid(x_i[g+k]);
                for (int m=0; m<2*l+1; ++m)
                    y_i[o_out+m*mul[l]+k] = s*x_i[o_in+m*mul[l]+k];
            }
            g += mul[l];
            o_in += (2*l+1)*mul[l];
            o_out += (2*l+1)*mul[l];
        }
    }
}

void MACEPolar::Gate::reverse(int n, const double* y_adj, double* x_adj) const
{
    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
        const double* x_i = x.data()+long(i)*dim_in;
        const double* y_adj_i = y_adj+long(i)*dim_out;
        double* x_adj_i = x_adj+long(i)*dim_in;
        for (int k=0; k<scalars; ++k)
            x_adj_i[k] = scalar_constant*silu_deriv(x_i[k])*y_adj_i[k];
        int g = scalars;
        int o_in = mul[0];
        int o_out = scalars;
        for (int l=1; l<mul.size(); ++l) {
            for (int k=0; k<mul[l]; ++k) {
                const double s = sigmoid(x_i[g+k]);
                double d = 0.0;
                for (int m=0; m<2*l+1; ++m) {
                    d += x_i[o_in+m*mul[l]+k]*y_adj_i[o_out+m*mul[l]+k];
                    x_adj_i[o_in+m*mul[l]+k] = gate_constant*s*y_adj_i[o_out+m*mul[l]+k];
                }
                x_adj_i[g+k] = gate_constant*s*(1.0-s)*d;
            }
            g += mul[l];
            o_in += (2*l+1)*mul[l];
            o_out += (2*l+1)*mul[l];
        }
    }
}

void MACEPolar::Dot::forward(int n, const double* x, const double* y, double* out, int ld)
{
    z.resize(long(n)*4*num_channels);
    #pragma omp parallel for
    for (int i0=0; i0<n; i0+=4)
        for (int lm=0; lm<4; ++lm) {
            const double* y_ilm[4];
            for (int b=0; b<4; ++b)
                y_ilm[b] = y+(long(std::min(i0+b,n-1))*4+lm)*num_channels;
            for (int u=0; u<num_channels; ++u) {
                const double* D_trans_u = weights_trans[lm>0].data()+u*num_channels;
                double z_ilm_u[4] = {};
                for (int v=0; v<num_channels; ++v)
                    for (int b=0; b<4; ++b)
                        z_ilm_u[b] += y_ilm[b][v] * D_trans_u[v];
                for (int b=0; b<4 and i0+b<n; ++b)
                    z[(long(i0+b)*4+lm)*num_channels+u] = z_ilm_u[b];
            }
        }
    #pragma omp parallel for
    for (int i=0; i<n; ++i)
        for (int u=0; u<num_channels; ++u) {
            double s = 0.0;
            for (int lm=0; lm<4; ++lm)
                s += x[(long(i)*4+lm)*num_channels+u]*z[(long(i)*4+lm)*num_channels+u];
            out[long(i)*ld+u] = s;
        }
}

void MACEPolar::Dot::reverse(int n, const double* x, const double* out_adj, int ld, double* x_adj, double* y_adj) const
{
    #pragma omp parallel for
    for (int i0=0; i0<n; i0+=4) {
        auto z_adj = std::vector<double>(4*4*num_channels);
        for (int b=0; b<4; ++b)
            for (int lm=0; lm<4; ++lm)
                for (int u=0; u<num_channels; ++u) {
                    const int i = std::min(i0+b,n-1);
                    const long x_ilmu = (long(i)*4+lm)*num_channels+u;
                    z_adj[(b*4+lm)*num_channels+u] = out_adj[long(i)*ld+u]*x[x_ilmu];
                    if (i0+b < n)
                        x_adj[x_ilmu] += out_adj[long(i)*ld+u]*z[x_ilmu];
                }
        for (int lm=0; lm<4; ++lm)
            for (int v=0; v<num_channels; ++v) {
                const double* D_v = weights[lm>0].data()+v*num_channels;
                double y_adj_ilm_v[4] = {};
                for (int u=0; u<num_channels; ++u)
                    for (int b=0; b<4; ++b)
                        y_adj_ilm_v[b] += z_adj[(b*4+lm)*num_channels+u] * D_v[u];
                for (int b=0; b<4 and i0+b<n; ++b)
                    y_adj[(long(i0+b)*4+lm)*num_channels+v] += y_adj_ilm_v[b];
            }
    }
}

void MACEPolar::Modulation::forward(int n, const double* x, const double* s, double* out)
{
    w.resize(long(n)*2*num_channels);
    #pragma omp parallel for
    for (int i0=0; i0<n; i0+=4) {
        const double* s_i[4];
        for (int b=0; b<4; ++b)
            s_i[b] = s+long(std::min(i0+b,n-1))*num_channels;
        for (int l=0; l<2; ++l)
            for (int u=0; u<num_channels; ++u) {
                const double* M_trans_lu = weights_trans[l].data()+u*num_channels;
                double w_ilu[4] = {};
                for (int v=0; v<num_channels; ++v)
                    for (int b=0; b<4; ++b)
                        w_ilu[b] += s_i[b][v] * M_trans_lu[v];
                for (int b=0; b<4 and i0+b<n; ++b)
                    w[(long(i0+b)*2+l)*num_channels+u] = w_ilu[b];
            }
        for (int b=0; b<4 and i0+b<n; ++b)
            for (int lm=0; lm<4; ++lm)
                for (int u=0; u<num_channels; ++u)
                    out[(long(i0+b)*4+lm)*num_channels+u] =
                        x[(long(i0+b)*4+lm)*num_channels+u]*w[(long(i0+b)*2+(lm>0))*num_channels+u];
    }
}

void MACEPolar::Modulation::reverse(int n, const double* x, const double* out_adj, double* x_adj, double* s_adj) const
{
    #pragma omp parallel for
    for (int i0=0; i0<n; i0+=4) {
        auto w_adj = std::vector<double>(4*2*num_channels, 0.0);
        for (int b=0; b<4 and i0+b<n; ++b)
            for (int lm=0; lm<4; ++lm)
                for (int u=0; u<num_channels; ++u) {
                    const long x_ilmu = (long(i0+b)*4+lm)*num_channels+u;
                    x_adj[x_ilmu] += out_adj[x_ilmu]*w[(long(i0+b)*2+(lm>0))*num_channels+u];
                    w_adj[(b*2+(lm>0))*num_channels+u] += out_adj[x_ilmu]*x[x_ilmu];
                }
        for (int v=0; v<num_channels; ++v) {
            double s_adj_iv[4] = {};
            for (int l=0; l<2; ++l) {
                const double* M_lv = weights[l].data()+v*num_channels;
                for (int u=0; u<num_channels; ++u)
                    for (int b=0; b<4; ++b)
                        s_adj_iv[b] += w_adj[(b*2+l)*num_channels+u] * M_lv[u];
            }
            for (int b=0; b<4 and i0+b<n; ++b)
                s_adj[long(i0+b)*num_channels+v] += s_adj_iv[b];
        }
    }
}

void MACEPolar::BiasReadout::forward(int n, const double* x, int ldx, double* out)
{
    const int h = linear_1.out[0];
    auto a = std::vector<double>(long(n)*h);
    y1.resize(long(n)*h);
    y2.resize(long(n)*h);
    linear_1.forward(n, x, ldx, y1.data(), h, false);
    for (long x=0; x<long(n)*h; ++x)
        a[x] = act*y1[x]*sigmoid(y1[x]);
    linear_mid.forward(n, a.data(), h, y2.data(), h, false);
    for (long x=0; x<long(n)*h; ++x)
        a[x] = act*y2[x]*sigmoid(y2[x]);
    linear_2.forward(n, a.data(), h, out, linear_2.out[0], false);
}

void MACEPolar::BiasReadout::reverse(int n, const double* out_adj, double* x_adj, int ldx) const
{
    const int h = linear_1.out[0];
    auto a = std::vector<double>(long(n)*h, 0.0);
    auto b = std::vector<double>(long(n)*h, 0.0);
    linear_2.reverse(n, out_adj, linear_2.out[0], a.data(), h);
    for (long x=0; x<long(n)*h; ++x)
        a[x] *= act*silu_deriv(y2[x]);
    linear_mid.reverse(n, a.data(), h, b.data(), h);
    for (long x=0; x<long(n)*h; ++x)
        b[x] *= act*silu_deriv(y1[x]);
    linear_1.reverse(n, b.data(), h, x_adj, ldx);
}

void MACEPolar::GatedReadout::forward(int n, const double* x, int ldx, double* out)
{
    auto y = std::vector<double>(long(n)*gate_1.dim_in);
    auto g = std::vector<double>(long(n)*gate_1.dim_out);
    linear_1.forward(n, x, ldx, y.data(), gate_1.dim_in, false);
    gate_1.forward(n, y.data(), g.data());
    linear_mid.forward(n, g.data(), gate_1.dim_out, y.data(), gate_1.dim_in, false);
    gate_2.forward(n, y.data(), g.data());
    linear_2.forward(n, g.data(), gate_1.dim_out, out, linear_2.off_out.back(), false);
}

void MACEPolar::GatedReadout::reverse(int n, const double* out_adj, double* x_adj, int ldx) const
{
    auto g = std::vector<double>(long(n)*gate_1.dim_out, 0.0);
    auto y = std::vector<double>(long(n)*gate_1.dim_in);
    linear_2.reverse(n, out_adj, linear_2.off_out.back(), g.data(), gate_1.dim_out);
    gate_2.reverse(n, g.data(), y.data());
    std::fill(g.begin(), g.end(), 0.0);
    linear_mid.reverse(n, y.data(), gate_1.dim_in, g.data(), gate_1.dim_out);
    gate_1.reverse(n, g.data(), y.data());
    linear_1.reverse(n, y.data(), gate_1.dim_in, x_adj, ldx);
}

static cplx weighted(const cplx* density_basis, const cplx* feature_basis, int num_sigmas, int k, const double* w,
                     int num_columns, int c, bool features)
{
    cplx b = 0.0;
    if (not features)
        for (int lm=0; lm<4; ++lm)
            b += density_basis[k*4+lm]*w[lm*num_columns+c];
    else
        for (int lm=0; lm<4; ++lm)
            for (int s=0; s<num_sigmas; ++s)
                b += feature_basis[(k*4+lm)*num_sigmas+s]*w[(lm*num_columns+c)*num_sigmas+s];
    return b;
}

void MACEPolar::structure(int n, const double* w, int num_weights, cplx* structure_factors) const
{
    const int num_k = kvec.size();
    std::fill(structure_factors, structure_factors+long(num_weights)*num_k, 0.0);
    #pragma omp parallel
    {
        auto s = std::vector<cplx>(long(num_weights)*num_k, 0.0);
        #pragma omp for
        for (int i=0; i<n; ++i)
            for (int k=0; k<num_k; ++k) {
                const double* p_i = inverted_positions.data()+3*i;
                const cplx e = std::polar(1.0, -(kvec[k][0]*p_i[0]+kvec[k][1]*p_i[1]+kvec[k][2]*p_i[2]));
                for (int x=0; x<num_weights; ++x)
                    s[x*num_k+k] += w[long(i)*num_weights+x]*e;
            }
        #pragma omp critical
        for (long x=0; x<long(num_weights)*num_k; ++x)
            structure_factors[x] += s[x];
    }
}

void MACEPolar::on_basis(const cplx* structure_factors, int num_columns, bool features, cplx* X) const
{
    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    for (int c=0; c<num_columns; ++c)
        for (int k=0; k<num_k; ++k) {
            cplx x = 0.0;
            for (int lm=0; lm<4; ++lm)
                if (not features)
                    x += density_basis[k*4+lm]*structure_factors[(lm*num_columns+c)*num_k+k];
                else
                    for (int s=0; s<num_sigmas; ++s)
                        x += feature_basis[(k*4+lm)*num_sigmas+s]
                             *structure_factors[((lm*num_columns+c)*num_sigmas+s)*num_k+k];
            X[c*num_k+k] = x;
        }
}

void MACEPolar::project(int n, const cplx* Z, int num_columns, bool features, double* out) const
{
    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    const int ld = 4*num_columns*(features ? num_sigmas : 1);
    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
        double* out_i = out+long(i)*ld;
        const double* p_i = inverted_positions.data()+3*i;
        for (int k=0; k<num_k; ++k) {
            const cplx e = std::polar(1.0, kvec[k][0]*p_i[0]+kvec[k][1]*p_i[1]+kvec[k][2]*p_i[2]);
            for (int c=0; c<num_columns; ++c) {
                const cplx z = Z[c*num_k+k]*e;
                for (int lm=0; lm<4; ++lm)
                    if (not features)
                        out_i[lm*num_columns+c] += std::real(std::conj(density_basis[k*4+lm])*z);
                    else
                        for (int s=0; s<num_sigmas; ++s)
                            out_i[(lm*num_columns+c)*num_sigmas+s] +=
                                std::real(std::conj(feature_basis[(k*4+lm)*num_sigmas+s])*z);
            }
        }
    }
}

void MACEPolar::phase_forces(int n, const cplx* A, const double* w, int num_columns, bool features,
                             double prefactor, double* f) const
{
    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    const int ld = 4*num_columns*(features ? num_sigmas : 1);
    #pragma omp parallel for
    for (int i=0; i<n; ++i) {
        double g[3] = {};
        const double* p_i = inverted_positions.data()+3*i;
        for (int k=0; k<num_k; ++k) {
            const cplx e = std::polar(1.0, -(kvec[k][0]*p_i[0]+kvec[k][1]*p_i[1]+kvec[k][2]*p_i[2]));
            double s = 0.0;
            for (int c=0; c<num_columns; ++c)
                s += std::imag(std::conj(A[c*num_k+k])*e*weighted(density_basis.data(),
                                                                  feature_basis.data(), num_sigmas, k,
                                                                    w+long(i)*ld, num_columns, c, features));
            for (int a=0; a<3; ++a)
                g[a] += k2inv[k]*s*kvec[k][a];
        }
        for (int a=0; a<3; ++a)
            f[3*i+a] += prefactor*g[a];
    }
}

void MACEPolar::on_basis_deriv(int k, int c, const cplx* structure_factors, bool features, int num_columns, cplx& value,
                               cplx* deriv) const
{
    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    constexpr int axis[4] = {0, 1, 2, 0};
    value = 0.0;
    deriv[0] = deriv[1] = deriv[2] = 0.0;
    for (int lm=0; lm<4; ++lm)
        for (int s=0; s<(features ? num_sigmas : 1); ++s) {
            const cplx r = features ? structure_factors[((lm*num_columns+c)*num_sigmas+s)*num_k+k]
                                    : structure_factors[(lm*num_columns+c)*num_k+k];
            const cplx B = (features ? feature_basis[(k*4+lm)*num_sigmas+s] : density_basis[k*4+lm])*r;
            const double sigma = features ? feature_sigmas[s] : density_sigma;
            value += B;
            for (int a=0; a<3; ++a)
                deriv[a] -= sigma*sigma*kvec[k][a]*B;
            if (lm > 0)
                deriv[axis[lm]] += (features ? feature_basis_deriv[k*num_sigmas+s] : density_basis_deriv[k])*r;
        }
}

void MACEPolar::kspace_virial(const cplx* Rx, bool fx, const cplx* Ry, bool fy, int num_columns,
                              double prefactor, double* W) const
{
    const int num_k = kvec.size();
    double G[3][3] = {};
    double L = 0.0;
    for (int k=0; k<num_k; ++k)
        for (int c=0; c<num_columns; ++c) {
            cplx basis_x;
            cplx basis_y;
            cplx deriv_x[3];
            cplx deriv_y[3];
            on_basis_deriv(k, c, Rx, fx, num_columns, basis_x, deriv_x);
            on_basis_deriv(k, c, Ry, fy, num_columns, basis_y, deriv_y);
            const double re = std::real(std::conj(basis_x)*basis_y);
            L += prefactor*k2inv[k]*re;
            for (int a=0; a<3; ++a) {
                const double d = prefactor*(-2*kvec[k][a]*k2inv[k]*k2inv[k]*re
                                      + k2inv[k]*std::real(std::conj(deriv_x[a])*basis_y
                                                           + std::conj(basis_x)*deriv_y[a]));
                for (int b=0; b<3; ++b)
                    G[a][b] += d*kvec[k][b];
            }
        }
    W[0] += G[0][0]+L;
    W[1] += G[1][1]+L;
    W[2] += G[2][2]+L;
    W[3] += 0.5*(G[0][1]+G[1][0]);
    W[4] += 0.5*(G[0][2]+G[2][0]);
    W[5] += 0.5*(G[1][2]+G[2][1]);
}

void MACEPolar::features(int n, const double* q, std::vector<cplx>& structure_factors,
                         std::vector<cplx>& basis_factors, double* V) const
{
    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    std::fill(V, V+long(n)*8*num_sigmas, 0.0);
    if (features_hook) {
        auto q_inv = std::vector<double>(q, q+long(n)*8);
        for (int i=0; i<n; ++i)
            for (int x=2; x<8; ++x)
                q_inv[i*8+x] = -q_inv[i*8+x];
        features_hook(q_inv.data(), V);
        for (int i=0; i<n; ++i)
            for (int x=2*num_sigmas; x<8*num_sigmas; ++x)
                V[long(i)*8*num_sigmas+x] = -V[long(i)*8*num_sigmas+x];
    } else {
        structure_factors.resize(8*num_k);
        basis_factors.resize(2*num_k);
        structure(n, q, 8, structure_factors.data());
        on_basis(structure_factors.data(), 2, false, basis_factors.data());
        auto Z = std::vector<cplx>(2*num_k);
        for (int x=0; x<2*num_k; ++x)
            Z[x] = 2.0*field_constant/volume*k2inv[x%num_k]*basis_factors[x];
        project(n, Z.data(), 2, true, V);
    }
    for (int i=0; i<n; ++i)
        for (int lm=0; lm<4; ++lm)
            for (int s=0; s<2; ++s)
                for (int g=0; g<num_sigmas; ++g) {
                    double& v = V[((long(i)*4+lm)*2+s)*num_sigmas+g];
                    v = (v-field_self[lm*num_sigmas+g]*q[(i*4+lm)*2+s])/field_norms[lm*num_sigmas+g];
                }
}

void MACEPolar::features_reverse(int n, const double* q, const std::vector<cplx>& structure_factors,
                                 const std::vector<cplx>& basis_factors, const double* V_adj, double* q_adj,
                                 double* f, double* W) const
{
    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    const double prefactor = 2.0*field_constant/volume;
    auto v = std::vector<double>(long(n)*8*num_sigmas);
    for (int i=0; i<n; ++i)
        for (int lm=0; lm<4; ++lm)
            for (int s=0; s<2; ++s)
                for (int g=0; g<num_sigmas; ++g) {
                    const long x = ((long(i)*4+lm)*2+s)*num_sigmas+g;
                    v[x] = V_adj[x]/field_norms[lm*num_sigmas+g];
                    q_adj[(i*4+lm)*2+s] -= v[x]*field_self[lm*num_sigmas+g];
                }
    if (features_reverse_hook) {
        auto q_inv = std::vector<double>(q, q+long(n)*8);
        auto v_inv = v;
        auto a = std::vector<double>(long(n)*8, 0.0);
        for (int i=0; i<n; ++i) {
            for (int x=2; x<8; ++x)
                q_inv[i*8+x] = -q_inv[i*8+x];
            for (int x=2*num_sigmas; x<8*num_sigmas; ++x)
                v_inv[long(i)*8*num_sigmas+x] = -v_inv[long(i)*8*num_sigmas+x];
        }
        features_reverse_hook(q_inv.data(), v_inv.data(), a.data(), f, W);
        for (int i=0; i<n; ++i)
            for (int x=0; x<8; ++x)
                q_adj[i*8+x] += x < 2 ? a[i*8+x] : -a[i*8+x];
        return;
    }
    auto Rv = std::vector<cplx>(8*num_sigmas*num_k);
    structure(n, v.data(), 8*num_sigmas, Rv.data());
    auto T = std::vector<cplx>(2*num_k);
    on_basis(Rv.data(), 2, true, T.data());
    auto Z = std::vector<cplx>(2*num_k);
    for (int x=0; x<2*num_k; ++x)
        Z[x] = prefactor*k2inv[x%num_k]*T[x];
    project(n, Z.data(), 2, false, q_adj);
    phase_forces(n, T.data(), q, 2, false, prefactor, f);
    phase_forces(n, basis_factors.data(), v.data(), 2, true, prefactor, f);
    kspace_virial(Rv.data(), true, structure_factors.data(), false, 2, prefactor, W);
}

void MACEPolar::setup_kspace(int num_nodes, std::span<const double> positions, std::span<const double> cell)
{
    const double* a = cell.data();
    const double det = a[0]*(a[4]*a[8]-a[5]*a[7]) - a[1]*(a[3]*a[8]-a[5]*a[6]) + a[2]*(a[3]*a[7]-a[4]*a[6]);
    volume = std::abs(det);
    double b[3][3];
    for (int i=0; i<3; ++i) {
        const double* u = a+3*((i+1)%3);
        const double* w = a+3*((i+2)%3);
        b[i][0] = 2*std::numbers::pi*(u[1]*w[2]-u[2]*w[1])/det;
        b[i][1] = 2*std::numbers::pi*(u[2]*w[0]-u[0]*w[2])/det;
        b[i][2] = 2*std::numbers::pi*(u[0]*w[1]-u[1]*w[0])/det;
    }
    int nmax[3];
    for (int i=0; i<3; ++i) {
        const double norm = std::sqrt(a[3*i]*a[3*i]+a[3*i+1]*a[3*i+1]+a[3*i+2]*a[3*i+2]);
        nmax[i] = std::ceil(kspace_cutoff/((b[i][0]*a[3*i]+b[i][1]*a[3*i+1]+b[i][2]*a[3*i+2])/norm));
    }
    kvec.clear();
    for (int n1=0; n1<nmax[0]; ++n1)
        for (int n2=(n1 > 0 ? -nmax[1] : 0); n2<nmax[1]; ++n2)
            for (int n3=(n1 > 0 or n2 > 0 ? -nmax[2] : 1); n3<nmax[2]; ++n3) {
                std::array<double,3> k;
                for (int d=0; d<3; ++d)
                    k[d] = n1*b[0][d]+n2*b[1][d]+n3*b[2][d];
                if (k[0]*k[0]+k[1]*k[1]+k[2]*k[2] <= kspace_cutoff*kspace_cutoff)
                    kvec.push_back(k);
            }

    const int num_k = kvec.size();
    const int num_sigmas = feature_sigmas.size();
    const double y0 = 0.5/std::sqrt(std::numbers::pi);
    const double y1 = std::sqrt(0.75/std::numbers::pi);
    k2inv.resize(num_k);
    density_basis.resize(4*num_k);
    feature_basis.resize(4*num_k*num_sigmas);
    density_basis_deriv.resize(num_k);
    feature_basis_deriv.resize(num_k*num_sigmas);
    for (int k=0; k<num_k; ++k) {
        const double k2 = kvec[k][0]*kvec[k][0]+kvec[k][1]*kvec[k][1]+kvec[k][2]*kvec[k][2];
        const double kn = std::sqrt(k2);
        const double Yk[4] = {y0, y1*kvec[k][1]/kn, y1*kvec[k][2]/kn, y1*kvec[k][0]/kn};
        k2inv[k] = 1.0/k2;
        const double fd = std::exp(-0.5*density_sigma*density_sigma*k2);
        density_basis[k*4] = cplx(density_a[0]*fd*Yk[0], 0.0);
        for (int lm=1; lm<4; ++lm)
            density_basis[k*4+lm] = cplx(0.0, -density_a[1]*kn*fd*Yk[lm]);
        density_basis_deriv[k] = cplx(0.0, -density_a[1]*y1*fd);
        for (int s=0; s<num_sigmas; ++s) {
            const double ff = std::exp(-0.5*feature_sigmas[s]*feature_sigmas[s]*k2);
            feature_basis[k*4*num_sigmas+s] = cplx(feature_a[s][0]*ff*Yk[0], 0.0);
            for (int lm=1; lm<4; ++lm)
                feature_basis[(k*4+lm)*num_sigmas+s] = cplx(0.0, -feature_a[s][1]*kn*ff*Yk[lm]);
            feature_basis_deriv[k*num_sigmas+s] = cplx(0.0, -feature_a[s][1]*y1*ff);
        }
    }
    inverted_positions.resize(3*num_nodes);
    for (int x=0; x<3*num_nodes; ++x)
        inverted_positions[x] = -positions[x];
}

static std::vector<int> offsets(const std::vector<int>& mul)
{
    auto off = std::vector<int>(mul.size()+1, 0);
    for (int l=0; l<mul.size(); ++l)
        off[l+1] = off[l] + (2*l+1)*mul[l];
    return off;
}

static std::vector<std::vector<double>> transposed(const std::vector<std::vector<double>>& weights,
                                                  const std::vector<int>& in, const std::vector<int>& out)
{
    auto weights_trans = weights;
    for (int l=0; l<weights.size(); ++l)
        for (int kp=0; kp<in[l]; ++kp)
            for (int k=0; k<out[l]; ++k)
                weights_trans[l][k*in[l]+kp] = weights[l][kp*out[l]+k];
    return weights_trans;
}

static MACEPolar::Linear linear_from_json(const nlohmann::json& j)
{
    MACEPolar::Linear L;
    L.in = j["in"].get<std::vector<int>>();
    L.out = j["out"].get<std::vector<int>>();
    L.off_in = offsets(L.in);
    L.off_out = offsets(L.out);
    L.weights = j["weights"].get<std::vector<std::vector<double>>>();
    L.weights_trans = transposed(L.weights, L.in, L.out);
    L.bias = j["bias"].get<std::vector<double>>();
    return L;
}

static MACEPolar::MLP mlp_from_json(const nlohmann::json& j)
{
    MACEPolar::MLP mlp;
    for (const auto& l : j) {
        MACEPolar::MLP::Layer L;
        L.in = l["in"];
        L.out = l["out"];
        L.weights = l["weights"].get<std::vector<double>>();
        L.weights_trans = L.weights;
        for (int kp=0; kp<L.in; ++kp)
            for (int k=0; k<L.out; ++k)
                L.weights_trans[k*L.in+kp] = L.weights[kp*L.out+k];
        L.bias = l["bias"].get<std::vector<double>>();
        if (l.contains("ln_weight")) {
            L.ln_weights = l["ln_weight"].get<std::vector<double>>();
            L.ln_bias = l["ln_bias"].get<std::vector<double>>();
            L.ln_eps = l["ln_eps"];
        }
        mlp.layers.push_back(L);
    }
    return mlp;
}

static MACEPolar::Gate gate_from_json(const nlohmann::json& j, const std::vector<int>& mul)
{
    MACEPolar::Gate g;
    g.scalars = j["scalars"];
    g.scalar_constant = j["csts"][0];
    g.gate_constant = j["csts"][1];
    g.mul = mul;
    g.dim_in = offsets(mul).back();
    g.dim_out = g.dim_in-(mul[0]-g.scalars);
    return g;
}

static MACEPolar::BiasReadout bias_readout_from_json(const nlohmann::json& j)
{
    MACEPolar::BiasReadout r;
    r.linear_1 = linear_from_json(j["linear_1"]);
    r.linear_mid = linear_from_json(j["linear_mid"]);
    r.linear_2 = linear_from_json(j["linear_2"]);
    r.act = j["act"];
    return r;
}

static MACEPolar::GatedReadout gated_readout_from_json(const nlohmann::json& j)
{
    MACEPolar::GatedReadout r;
    r.linear_1 = linear_from_json(j["linear_1"]);
    r.linear_mid = linear_from_json(j["linear_mid"]);
    r.linear_2 = linear_from_json(j["linear_2"]);
    r.gate_1 = gate_from_json(j["gate"], r.linear_1.out);
    r.gate_2 = r.gate_1;
    return r;
}

static std::vector<double> flatten(const nlohmann::json& j)
{
    auto v = std::vector<double>();
    if (j.is_number())
        v.push_back(j);
    else
        for (const auto& x : j)
            for (const double y : flatten(x))
                v.push_back(y);
    return v;
}

void MACEPolar::load_from_json(std::string filename)
{
    std::ifstream f(filename);
    if (not f)
        throw std::runtime_error("MACEPolar: cannot open " + filename);
    nlohmann::json file = nlohmann::json::parse(f);
    if (file.value("format_version", 0) != 3)
        throw std::runtime_error(filename + " was extracted by an incompatible symmetrix version; extract it again");
    if (file.value("model_type", "") != "PolarMACE")
        throw std::invalid_argument("MACEPolar: " + filename + " is not a PolarMACE model file");
    if (file["l_max"] != 3 or file["L_max"] != 1 or file["updates"].size() != 2)
        throw std::invalid_argument("MACEPolar requires l_max = 3, L_max = 1 and two charge updates");

    // Basic model information
    num_elements = file["num_elements"];
    num_channels = file["num_channels"];
    num_edge_channels = file["num_edge_channels"];
    r_cut = file["r_cut"];
    atomic_numbers = file["atomic_numbers"].get<std::vector<int>>();
    atomic_energies = file["atomic_energies"].get<std::vector<double>>();

    // Radial functions
    radial_spline_r_min = file["radial_spline_r_min"];
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int ab=0; ab<spl_values_0.size(); ++ab)
        spl_set_0.push_back(
            std::make_unique<CubicSplineSet<double>>(radial_spline_r_min, r_cut, spl_values_0[ab], spl_derivs_0[ab]));
    num_basis_0 = spl_set_0[0]->num_splines;
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto radial_weights_1 = file["radial_weights_1"].get<std::vector<std::vector<double>>>();
    for (int ab=0; ab<spl_values_1.size(); ++ab) {
        const int num_knots = spl_values_1[ab][0].size();
        auto R1_values = std::vector<std::vector<double>>(radial_weights_1[0].size(),
                                                          std::vector<double>(num_knots, 0.0));
        auto R1_derivs = R1_values;
        for (int n=0; n<radial_weights_1.size(); ++n)
            for (int x=0; x<R1_values.size(); ++x)
                for (int s=0; s<num_knots; ++s) {
                    R1_values[x][s] += radial_weights_1[n][x]*spl_values_1[ab][n][s];
                    R1_derivs[x][s] += radial_weights_1[n][x]*spl_derivs_1[ab][n][s];
                }
        spl_set_1.push_back(std::make_unique<CubicSplineSet<double>>(radial_spline_r_min, r_cut, R1_values, R1_derivs));
    }
    for (int t=0; t<2; ++t) {
        auto values = file["A"+std::to_string(t)+"_spline_values"].get<std::vector<std::vector<double>>>();
        auto derivs = file["A"+std::to_string(t)+"_spline_derivs"].get<std::vector<std::vector<double>>>();
        auto& splines = t == 0 ? A0_splines : A1_splines;
        for (int ab=0; ab<values.size(); ++ab)
            splines.push_back(CubicSpline(radial_spline_r_min, r_cut, values[ab], derivs[ab]));
    }

    // X0
    linear_1_1 = linear_from_json(file["linear_1_1"]);
    mul_X = linear_1_1.out;
    off_X = offsets(mul_X);
    auto X0_weights = file["X0_weights"].get<std::vector<std::vector<std::vector<double>>>>();
    const int width_X0 = std::accumulate(mul_X.begin(), mul_X.end(), 0);
    bool shapes = X0_weights.size() == num_elements and mul_X.size() == 4;
    for (const auto& w : X0_weights) {
        shapes = shapes and w.size() == num_basis_0;
        for (const auto& row : w)
            shapes = shapes and row.size() == width_X0;
    }
    if (not shapes)
        throw std::invalid_argument("MACEPolar: " + filename + " has X0_weights of another shape than MACE-POLAR-1's");
    linear_0.in = std::vector<int>(4, num_elements*num_basis_0);
    linear_0.out = mul_X;
    linear_0.off_in = offsets(linear_0.in);
    linear_0.off_out = off_X;
    linear_0.weights.resize(4);
    int q = 0;
    for (int l=0; l<4; ++l) {
        for (int b=0; b<num_elements; ++b)
            for (int n=0; n<num_basis_0; ++n)
                for (int k=0; k<mul_X[l]; ++k)
                    linear_0.weights[l].push_back(X0_weights[b][n][q+k]);
        q += mul_X[l];
    }
    linear_0.weights_trans = transposed(linear_0.weights, linear_0.in, linear_0.out);
    residual_0 = flatten(file["residual_0"]);

    // A0, A1
    gate_0 = gate_from_json(file["gate_0"], mul_X);
    gate_1 = gate_from_json(file["gate_1"], mul_X);
    linear_2_0 = linear_from_json(file["linear_2_0"]);
    linear_2_1 = linear_from_json(file["linear_2_1"]);

    // M0, M1
    B0_spec = file["B0_spec"].get<std::vector<std::array<int,3>>>();
    B0_coefficients = file["B0_coefficients"].get<std::vector<double>>();
    M0_spec = file["M0_spec"].get<std::vector<std::array<int,3>>>();
    M0_weights = flatten(file["M0_weights"]);
    num_B0 = 16;
    for (const auto& [q, a, b] : B0_spec)
        num_B0 = std::max(num_B0, q+1);
    B1_spec = file["B1_spec"].get<std::vector<std::array<int,3>>>();
    B1_coefficients = file["B1_coefficients"].get<std::vector<double>>();
    M1_spec = file["M1_spec"].get<std::vector<std::array<int,3>>>();
    M1_weights = flatten(file["M1_weights"]);
    num_B1 = 16;
    for (const auto& [q, a, b] : B1_spec)
        num_B1 = std::max(num_B1, q+1);

    // H1, H2
    product_linear_0 = linear_from_json(file["product_linear_0"]);
    product_linear_1 = linear_from_json(file["product_linear_1"]);
    skip_0 = flatten(file["skip_0"]);
    linear_up_1 = linear_from_json(file["linear_up_1"]);
    residual_1 = linear_from_json(file["residual_1"]);
    skip_1 = linear_from_json(file["skip_1"]);

    // Phi1
    Phi1_l = file["Phi1_l"].get<std::vector<int>>();
    Phi1_spec = file["Phi1_spec"].get<std::vector<std::array<int,3>>>();
    Phi1_cg_spec = file["Phi1_cg_spec"].get<std::vector<std::array<int,2>>>();
    Phi1_clebsch_gordan = file["Phi1_clebsch_gordan"].get<std::vector<double>>();
    num_lme = 0;
    for (auto l : Phi1_l)
        num_lme += 2*l+1;

    // Readouts
    readout_0 = file["readout_0"].get<std::vector<double>>();
    readout_1 = bias_readout_from_json(file["readout_1"]);

    // Charges
    for (int t=0; t<2; ++t) {
        sources[t] = linear_from_json(file["sources"][t]);
        mixer[t] = linear_from_json(file["mixer"][t]);
        const auto& j = file["updates"][t];
        auto& u = updates[t];
        u.potential = linear_from_json(j["potential"]);
        u.node = linear_from_json(j["node"]);
        u.charge = linear_from_json(j["charge"]);
        u.source = flatten(j["source"]);
        u.dot.num_channels = num_channels;
        u.dot.weights = j["dot"].get<std::vector<std::vector<double>>>();
        u.dot.weights_trans = transposed(u.dot.weights, {num_channels, num_channels}, {num_channels, num_channels});
        u.mlp = mlp_from_json(j["mlp"]);
        u.out.num_channels = num_channels;
        u.out.weights = j["out"].get<std::vector<std::vector<double>>>();
        u.out.weights_trans = transposed(u.out.weights, {num_channels, num_channels}, {num_channels, num_channels});
        u.readout = gated_readout_from_json(j["readout"]);
    }
    fukui = bias_readout_from_json(file["fukui"]);

    // Electron energy
    const auto& j = file["electron"];
    electron.q = linear_from_json(j["q"]);
    electron.v = linear_from_json(j["v"]);
    electron.dot_q.num_channels = num_channels;
    electron.dot_q.weights = j["dot_q"].get<std::vector<std::vector<double>>>();
    electron.dot_q.weights_trans = transposed(electron.dot_q.weights, {num_channels, num_channels},
                                              {num_channels, num_channels});
    electron.dot_v.num_channels = num_channels;
    electron.dot_v.weights = j["dot_v"].get<std::vector<std::vector<double>>>();
    electron.dot_v.weights_trans = transposed(electron.dot_v.weights, {num_channels, num_channels},
                                              {num_channels, num_channels});
    electron.mlp = mlp_from_json(j["mlp"]);

    // Electrostatics
    kspace_cutoff = file["kspace_cutoff"];
    field_constant = file["field_constant"];
    density_sigma = file["density_basis"]["sigmas"][0];
    density_a = file["density_basis"]["a"][0].get<std::array<double,2>>();
    feature_sigmas = file["feature_basis"]["sigmas"].get<std::vector<double>>();
    feature_a = file["feature_basis"]["a"].get<std::vector<std::array<double,2>>>();
    field_self = flatten(file["field_self"]);
    field_norms = flatten(file["field_norms"]);

    const std::vector<int> hidden = {num_channels, num_channels};
    const std::vector<int> edge = {num_edge_channels, num_edge_channels};
    const std::vector<int> target(4, num_channels);
    int width_Phi1 = 0;
    for (int l=0; l<linear_1_1.in.size(); ++l)
        width_Phi1 += (2*l+1)*linear_1_1.in[l];
    if (product_linear_0.in != hidden or product_linear_0.out != hidden or product_linear_1.in != hidden
        or product_linear_1.out != hidden or skip_1.in != hidden or skip_1.out != hidden or linear_up_1.in != hidden
        or linear_up_1.out != edge or residual_1.in != std::vector<int>{num_edge_channels, num_edge_channels, 0, 0}
        or linear_2_0.out != target or linear_2_1.out != target or skip_0.size() != num_elements*num_channels
        or readout_0.size() != num_channels or width_Phi1 != num_lme*num_edge_channels)
        throw std::invalid_argument("MACEPolar: " + filename + " has feature shapes other than MACE-POLAR-1's");
}
