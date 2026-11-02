#include <iostream> //TODO
#include <fstream>
#include <mutex>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "mace.hpp"

template <typename Precision>
MACE<Precision>::MACE(std::string filename)
{
    load_from_json(filename);
}

template <typename Precision>
void MACE<Precision>::compute_node_energies_forces(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    compute_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    compute_second_layer(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    compute_readouts(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_second_layer(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
    reverse_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

template <typename Precision>
void MACE<Precision>::compute_first_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // neighbor list
    {
        for (const double r_ij : r)
            if (std::isnan(r_ij) or r_ij < R0_spline_r_min or r_ij > r_cut)
                throw std::invalid_argument("Out of bounds in MACE. r=" + std::to_string(r_ij));
        first_neigh.resize(num_nodes);
        std::exclusive_scan(num_neigh.begin(), num_neigh.begin()+num_nodes, first_neigh.begin(), 0);
    }

    // Y
    // Y_{lm}(\hat{\mathbf{r}}_{ij})
    if (xyz.size() > 0) {
        const int num = xyz.size()/3;
        Y.resize(num*num_lm);
        Y_grad.resize(3*num*num_lm);
        const auto xyz_p = std::vector<Precision>(xyz.begin(), xyz.end());
        sphericart::SphericalHarmonics<Precision> sphericart(l_max);
        sphericart.compute_array_with_gradients(
            xyz_p.data(), xyz_p.size(), Y.data(), Y.size(), Y_grad.data(), Y_grad.size());
    }

    // A0
    // \Phi^{(0)}_{i,klm} = \sum_j R^{(0)}_{a_ia_j,kl}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij}) H^{(0)}_{a_j,k}
    // A^{(0)}_{i,klm} = \sum_{k'} w_{a_ikl,k'} \Phi^{(0)}_{i,k'lm}
    {
        A0.resize(num_nodes*num_lm*num_channels);

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];

            // compute Phi0_i
            auto Phi0_i = std::vector<Precision>(num_lm*num_channels, 0.0);
            auto R0_ij = std::vector<Precision>(spl_set_0[0]->num_splines);
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                spl_set_0[type_ij]->evaluate(r[ij], R0_ij);
                auto Y_ij = Y.data()+ij*num_lm;
                auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
                for (int l=0; l<=l_max; ++l) {
                    auto R0_ij_l = R0_ij.data()+l*num_channels;
                    for (int m=-l; m<=l; ++m) {
                        const int lm = l*l+l+m;
                        const Precision Y_ij_lm = Y_ij[lm];
                        auto Phi0_i_lm = Phi0_i.data()+lm*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            Phi0_i_lm[k] += R0_ij_l[k] * Y_ij_lm * H0_ij[k];
                        }
                    }
                }
            }

            // [A0_il]_mk = \sum_k' [Phi0_il]_mk' [W_il]_k'k
            for (int l=0; l<=l_max; ++l) {
                auto A0_weights_il = A0_weights[node_types[i]][l].data();
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                    auto Phi0_i_lm = Phi0_i.data()+lm*num_channels;
                    auto A0_i_lm = A0.data()+(i*num_lm+lm)*num_channels;
                    std::fill(A0_i_lm, A0_i_lm+num_channels, 0.0);
                    for (int kp=0; kp<num_channels; ++kp)
                        for (int k=0; k<num_channels; ++k)
                            A0_i_lm[k] += Phi0_i_lm[kp] * A0_weights_il[kp*num_channels+k];
                }
            }
        }
    }

    // A0 scaling
    // A^{(0)}_{i,klm} \leftarrow A^{(0)}_{i,klm} / s^{(0)}_i,  s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij})
    if (A0_scaled) {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            double A0_scale_factor = 1.0;
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            }
            auto A0_i = A0.data()+i*num_lm*num_channels;
            for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                A0_i[lmk] /= A0_scale_factor;
        }
    }

    // M0
    // M^{(0)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_iklm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(0)}_{i,kl'_\xi m'_\xi}
    {
        M0.assign(num_nodes*num_LM*num_channels, 0.0);
        B0.assign(num_nodes*num_B0*num_channels, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A0_i = A0.data()+i*num_lm*num_channels;
            auto M0_i = M0.data()+i*num_LM*num_channels;
            auto B0_i = B0.data()+i*num_B0*num_channels;
            auto M0_weights_i = M0_weights.data()+node_types[i]*M0_spec.size()*num_channels;
            std::copy(A0_i, A0_i+num_lm*num_channels, B0_i);
            for (int t=0; t<B0_spec.size(); ++t) {
                const auto [q, a, b] = B0_spec[t];
                for (int k=0; k<num_channels; ++k)
                    B0_i[q*num_channels+k] += B0_coefficients[t] * B0_i[a*num_channels+k] * B0_i[b*num_channels+k];
            }
            for (int e=0; e<M0_spec.size(); ++e) {
                const auto [o, q, n] = M0_spec[e];
                for (int m=0; m<n; ++m)
                    for (int k=0; k<num_channels; ++k)
                        M0_i[(o+m)*num_channels+k] += M0_weights_i[e*num_channels+k] * B0_i[(q+m)*num_channels+k];
            }
        }
    }

    // H1
    // H^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(0)}_{i,k'lm}
    {
        // [H1_il]_mk = \sum_k' [M0_il]_mk' [W_l]_k'k
        H1.assign(M0.size(), 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                    const auto M0_ilm = M0.data()+(i*num_LM+lm)*num_channels;
                    auto H1_ilm = H1.data()+(i*num_LM+lm)*num_channels;
                    for (int kp=0; kp<num_channels; ++kp)
                        for (int k=0; k<num_channels; ++k)
                            H1_ilm[k] += M0_ilm[kp] * H1_weights_l[kp*num_channels+k];
                }
            }
        }
    }
}

template <typename Precision>
void MACE<Precision>::compute_second_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    // Phi1
    // \Phi^{(1)}_{i,klm\eta} = \sum_j \sum_{l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) H^{(1)}_{j,kl_2m_2}
    {
        // Compute Phi1 using CG coefficients contracted with Y (named CG_Y)
        Phi1.resize(num_nodes*num_lme*num_channels);
        std::fill(Phi1.begin(), Phi1.end(), 0.0);
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            auto Phi1_i = Phi1.data()+i*num_lme*num_channels;
            auto CG_Y = std::vector<Precision>(Phi1_spec.size());
            auto R1_ij = std::vector<Precision>(spl_set_1[0]->num_splines);
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                spl_set_1[type_ij]->evaluate(r[ij], R1_ij);
                auto Y_ij = Y.data()+ij*num_lm;
                auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
                std::fill(CG_Y.begin(), CG_Y.end(), 0.0);
                for (int p=0; p<Phi1_cg_spec.size(); ++p) {
                    const auto [q, lm1] = Phi1_cg_spec[p];
                    CG_Y[q] += Phi1_clebsch_gordan[p] * Y_ij[lm1];
                }
                for (int q=0; q<Phi1_spec.size(); ++q) {
                    const auto [lme, lel1l2, lm2] = Phi1_spec[q];
                    auto Phi1_i_lme = Phi1_i+lme*num_channels;
                    auto R1_ij_lel1l2 = R1_ij.data()+lel1l2*num_channels;
                    auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                    for (int k=0; k<num_channels; ++k)
                        Phi1_i_lme[k] += CG_Y[q] * R1_ij_lel1l2[k] * H1_ij_lm2[k];
                }
            }
        }
    }

    // A1
    // A^{(1)}_{i,klm} = \sum_{k'\eta'} w_{kl,k'\eta'} \Phi^{(1)}_{i,k'lm\eta'}
    {
        // The core matrix multiplication is:
        //         [A1_il]_mk = \sum_k' [Phi1_il]_m(ek') [W_il]_(ek')k
        // for four nodes at a time, so each weight is loaded once for all four
        A1.resize(num_nodes*num_lm*num_channels);
        std::vector<int> num_e(l_max+1,0);
        for (auto l : Phi1_l)
            num_e[l] += 1;
        #pragma omp parallel for
        for (int i0=0; i0<num_nodes; i0+=4) {
            int lme = 0;
            for (int l=0; l<=l_max; ++l) {
                const int num_ek = num_e[l]*num_channels;
                for (int m=0; m<2*l+1; ++m) {
                    const Precision* Phi1_ilm[4];
                    for (int b=0; b<4; ++b)
                        Phi1_ilm[b] = Phi1.data()+(std::min(i0+b,num_nodes-1)*num_lme+lme+m*num_e[l])*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        auto A1_weights_trans_lk = A1_weights_trans[l].data()+k*num_ek;
                        Precision A1_ilm_k[4] = {};
                        for (int ekp=0; ekp<num_ek; ++ekp)
                            for (int b=0; b<4; ++b)
                                A1_ilm_k[b] += Phi1_ilm[b][ekp] * A1_weights_trans_lk[ekp];
                        for (int b=0; b<4 and i0+b<num_nodes; ++b)
                            A1[((i0+b)*num_lm+l*l+m)*num_channels+k] = A1_ilm_k[b];
                    }
                }
                lme += (2*l+1)*num_e[l];
            }
        }
    }

    // A1 scaling
    // A^{(1)}_{i,klm} \leftarrow A^{(1)}_{i,klm} / s^{(1)}_i,  s^{(1)}_i = 1 + \sum_j \rho^{(1)}_{a_ia_j}(r_{ij})
    if (A1_scaled) {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            double A1_scale_factor = 1.0;
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            }
            auto A1_i = A1.data()+i*num_lm*num_channels;
            for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                A1_i[lmk] /= A1_scale_factor;
        }
    }

    // M1
    // M^{(1)}_{i,k} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_ik,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(1)}_{i,kl'_\xi m'_\xi}
    {
        M1.assign(num_nodes*1*num_channels, 0.0);
        B1.assign(num_nodes*num_B1*num_channels, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto A1_i = A1.data()+i*num_lm*num_channels;
            auto M1_i = M1.data()+i*num_channels;
            auto B1_i = B1.data()+i*num_B1*num_channels;
            auto M1_weights_i = M1_weights.data()+node_types[i]*M1_spec.size()*num_channels;
            std::copy(A1_i, A1_i+num_lm*num_channels, B1_i);
            for (int t=0; t<B1_spec.size(); ++t) {
                const auto [q, a, b] = B1_spec[t];
                for (int k=0; k<num_channels; ++k)
                    B1_i[q*num_channels+k] += B1_coefficients[t] * B1_i[a*num_channels+k] * B1_i[b*num_channels+k];
            }
            for (int e=0; e<M1_spec.size(); ++e) {
                const auto [o, q, n] = M1_spec[e];
                for (int k=0; k<num_channels; ++k)
                    M1_i[k] += M1_weights_i[e*num_channels+k] * B1_i[q*num_channels+k];
            }
        }
    }

    // H2
    // H^{(2)}_{i,k} = \sum_{k'} w_{k,k'} M^{(1)}_{i,k'} + \sum_{k'} w_{a_ik,k'} M^{(0)}_{i,k'00}
    {
        H2.assign(num_nodes*num_channels, 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto H2_i = H2.data()+i*num_channels;
            auto M0_i = M0.data()+i*num_LM*num_channels;
            auto M1_i = M1.data()+i*num_channels;
            auto H2_weights_for_M0_i = H2_weights_for_M0[node_types[i]].data();
            for (int kp=0; kp<num_channels; ++kp)
                for (int k=0; k<num_channels; ++k)
                    H2_i[k] += H2_weights_for_M0_i[kp*num_channels+k] * M0_i[kp]
                             + H2_weights_for_M1[kp*num_channels+k] * M1_i[kp];
        }
    }
}

template <typename Precision>
void MACE<Precision>::compute_readouts(
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
    // E_i += \frac{1}{2} \sum_j V^{\mathrm{ZBL}}_{Z_iZ_j}(r_{ij})
    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, first_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    // readouts
    // E_i += E^{(0)}_{a_i} + \sum_k w_k M^{(0)}_{i,k00} + \mathrm{MLP}(\{H^{(2)}_{i,k}\})
    // \bar{M}^{(0)}_{i,k00} = w_k,  \bar{H}^{(2)}_{i,k} = \partial \mathrm{MLP} / \partial H^{(2)}_{i,k}
    {
        M0_adj.assign(M0.size(), 0.0);
        H2_adj.resize(H2.size());
        for (int i=0; i<num_nodes; ++i) {
            // atomic energies
            node_energies[i] += atomic_energies[node_types[i]];
            // first readout
            for (int k=0; k<num_channels; ++k) {
                node_energies[i] += readout_1_weights_for_M0[k]*M0[i*num_LM*num_channels+k];
                M0_adj[i*num_LM*num_channels+k] = readout_1_weights_for_M0[k];
            }
            // second readout
            auto x = std::vector<double>(H2.begin()+i*num_channels, H2.begin()+(i+1)*num_channels);
            auto [f, g] = readout_2->evaluate_gradient(x);
            node_energies[i] += f[0];
            for (int k=0; k<num_channels; ++k) {
                H2_adj[i*num_channels+k] = g[k];
            }
        }
    }
}

template <typename Precision>
void MACE<Precision>::reverse_second_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // H2
    // \bar{M}^{(1)}_{i,k'} = \sum_k w_{k,k'} \bar{H}^{(2)}_{i,k},  \bar{M}^{(0)}_{i,k'00} += \sum_k w_{a_ik,k'} \bar{H}^{(2)}_{i,k}
    {
        M0_adj.resize(M0.size());
        M1_adj.resize(M1.size());
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto H2_adj_i = H2_adj.data()+i*num_channels;
            auto M0_adj_i = M0_adj.data()+i*num_LM*num_channels;
            auto M1_adj_i = M1_adj.data()+i*num_channels;
            auto H2_weights_for_M0_i = H2_weights_for_M0[node_types[i]].data();
            for (int kp=0; kp<num_channels; ++kp) {
                Precision M0_adj_i_kp = 0.0;
                Precision M1_adj_i_kp = 0.0;
                for (int k=0; k<num_channels; ++k) {
                    M0_adj_i_kp += H2_weights_for_M0_i[kp*num_channels+k] * H2_adj_i[k];
                    M1_adj_i_kp += H2_weights_for_M1[kp*num_channels+k] * H2_adj_i[k];
                }
                M0_adj_i[kp] += M0_adj_i_kp;
                M1_adj_i[kp] = M1_adj_i_kp;
            }
        }
    }

    // M1
    // \bar{A}^{(1)}_{i,klm} = \bar{M}^{(1)}_{i,k} \partial M^{(1)}_{i,k} / \partial A^{(1)}_{i,klm}
    {
        A1_adj.assign(A1.size(), 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto B1_adj_i = std::vector<Precision>(num_B1*num_channels);
            auto A1_adj_i = A1_adj.data()+i*num_lm*num_channels;
            auto M1_adj_i = M1_adj.data()+i*num_channels;
            auto B1_i = B1.data()+i*num_B1*num_channels;
            auto M1_weights_i = M1_weights.data()+node_types[i]*M1_spec.size()*num_channels;
            for (int e=0; e<M1_spec.size(); ++e) {
                const auto [o, q, n] = M1_spec[e];
                for (int k=0; k<num_channels; ++k)
                    B1_adj_i[q*num_channels+k] += M1_weights_i[e*num_channels+k] * M1_adj_i[k];
            }
            for (int t=B1_spec.size()-1; t>=0; --t) {
                const auto [q, a, b] = B1_spec[t];
                for (int k=0; k<num_channels; ++k) {
                    const Precision g = B1_coefficients[t] * B1_adj_i[q*num_channels+k];
                    B1_adj_i[a*num_channels+k] += g * B1_i[b*num_channels+k];
                    B1_adj_i[b*num_channels+k] += g * B1_i[a*num_channels+k];
                }
            }
            std::copy(B1_adj_i.begin(), B1_adj_i.begin()+num_lm*num_channels, A1_adj_i);
        }
    }

    // A1 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} \sum_{klm} \bar{A}^{(1)}_{i,klm} A^{(1)}_{i,klm} / s^{(1)}_i
    // \bar{A}^{(1)}_{i,klm} \leftarrow \bar{A}^{(1)}_{i,klm} / s^{(1)}_i
    if (A1_scaled) {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            auto A1_i = A1.data()+i*num_lm*num_channels;
            auto A1_adj_i = A1_adj.data()+i*num_lm*num_channels;
            // recompute the scale factor
            double A1_scale_factor = 1.0;
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            }
            // update dE/dxyz
            double dA1_dot_A1 = 0.0;
            for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                dA1_dot_A1 += A1_adj_i[lmk] * A1_i[lmk];
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                auto [f,d] = A1_splines[type_ij].evaluate_deriv(r[ij]);
                auto xyz_ij = xyz.data()+ij*3;
                auto node_forces_ij = node_forces.data()+ij*3;
                node_forces_ij[0] += dA1_dot_A1/A1_scale_factor*d*xyz_ij[0]/r[ij];
                node_forces_ij[1] += dA1_dot_A1/A1_scale_factor*d*xyz_ij[1]/r[ij];
                node_forces_ij[2] += dA1_dot_A1/A1_scale_factor*d*xyz_ij[2]/r[ij];
            }
            // update dE/dA1
            for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                A1_adj_i[lmk] /= A1_scale_factor;
        }
    }

    // A1
    // \bar{\Phi}^{(1)}_{i,k'lm\eta'} = \sum_k w_{kl,k'\eta'} \bar{A}^{(1)}_{i,klm}
    {
        // The core matrix multiplication is:
        //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
        // for four nodes at a time, so each weight is loaded once for all four
        dPhi1.resize(Phi1.size());
        std::vector<int> num_e(l_max+1,0);
        for (auto l : Phi1_l)
            num_e[l] += 1;
        #pragma omp parallel for
        for (int i0=0; i0<num_nodes; i0+=4) {
            int lme = 0;
            for (int l=0; l<=l_max; ++l) {
                const int num_ek = num_e[l]*num_channels;
                for (int m=0; m<2*l+1; ++m) {
                    const Precision* A1_adj_ilm[4];
                    for (int b=0; b<4; ++b)
                        A1_adj_ilm[b] = A1_adj.data()+(std::min(i0+b,num_nodes-1)*num_lm+l*l+m)*num_channels;
                    for (int ek=0; ek<num_ek; ++ek) {
                        auto A1_weights_lek = A1_weights[l].data()+ek*num_channels;
                        Precision dPhi1_ilm_ek[4] = {};
                        for (int kp=0; kp<num_channels; ++kp)
                            for (int b=0; b<4; ++b)
                                dPhi1_ilm_ek[b] += A1_adj_ilm[b][kp] * A1_weights_lek[kp];
                        for (int b=0; b<4 and i0+b<num_nodes; ++b)
                            dPhi1[((i0+b)*num_lme+lme+m*num_e[l])*num_channels+ek] = dPhi1_ilm_ek[b];
                    }
                }
                lme += (2*l+1)*num_e[l];
            }
        }
    }

    // Phi1
    // \bar{H}^{(1)}_{j,kl_2m_2} = \sum_i \sum_{lm\eta l_1m_1} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(1)}_{i,klm\eta}
    // \mathbf{F}_{ij} -= \sum_{klm\eta l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij})] H^{(1)}_{j,kl_2m_2} \bar{\Phi}^{(1)}_{i,klm\eta}
    {
        H1_adj.resize(H1.size());
        // Warning: Although it doesn't appear necessary to set H1_adj to zero,
        //          it matters when the number of nodes associated with H1 is greater than num_nodes.
        //          There is probably a better way to manage this.
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
        // Compute partial forces and dE/dH1
        std::vector<std::mutex> H1_adj_locks(H1.size()/(num_LM*num_channels));
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            auto dPhi1_i = dPhi1.data()+i*num_lme*num_channels;
            auto CG_Y = std::vector<Precision>(Phi1_spec.size());
            auto CG_Y_adj = std::vector<Precision>(Phi1_spec.size());
            auto H1_adj_ij = std::vector<Precision>(num_LM*num_channels);
            auto R1_ij = std::vector<Precision>(spl_set_1[0]->num_splines);
            auto R1_deriv_ij = std::vector<Precision>(R1_ij.size());
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                auto node_forces_ij = node_forces.data()+3*ij;
                auto xyz_ij = xyz.data()+3*ij;
                auto r_ij = r[ij];
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                spl_set_1[type_ij]->evaluate_derivs(r_ij, R1_ij, R1_deriv_ij);
                auto Y_ij = Y.data()+ij*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
                auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
                std::fill(CG_Y.begin(), CG_Y.end(), 0.0);
                std::fill(H1_adj_ij.begin(), H1_adj_ij.end(), 0.0);
                for (int p=0; p<Phi1_cg_spec.size(); ++p) {
                    const auto [q, lm1] = Phi1_cg_spec[p];
                    CG_Y[q] += Phi1_clebsch_gordan[p] * Y_ij[lm1];
                }
                Precision dE_dr = 0.0;
                for (int q=0; q<Phi1_spec.size(); ++q) {
                    const auto [lme, lel1l2, lm2] = Phi1_spec[q];
                    auto dPhi1_i_lme = dPhi1_i+lme*num_channels;
                    auto R1_ij_lel1l2 = R1_ij.data()+lel1l2*num_channels;
                    auto R1_deriv_ij_lel1l2 = R1_deriv_ij.data()+lel1l2*num_channels;
                    auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                    auto H1_adj_ij_lm2 = H1_adj_ij.data()+lm2*num_channels;
                    Precision g = 0.0;
                    for (int k=0; k<num_channels; ++k) {
                        g += dPhi1_i_lme[k] * R1_ij_lel1l2[k] * H1_ij_lm2[k];
                        dE_dr += dPhi1_i_lme[k] * CG_Y[q] * R1_deriv_ij_lel1l2[k] * H1_ij_lm2[k];
                        H1_adj_ij_lm2[k] += dPhi1_i_lme[k] * CG_Y[q] * R1_ij_lel1l2[k];
                    }
                    CG_Y_adj[q] = g;
                }
                for (int a=0; a<3; ++a)
                    node_forces_ij[a] -= dE_dr * xyz_ij[a]/r_ij;
                for (int p=0; p<Phi1_cg_spec.size(); ++p) {
                    const auto [q, lm1] = Phi1_cg_spec[p];
                    for (int a=0; a<3; ++a)
                        node_forces_ij[a] -= Phi1_clebsch_gordan[p] * Y_grad_ij[a*num_lm+lm1] * CG_Y_adj[q];
                }
                const int j = neigh_indices[ij];
                {
                    std::scoped_lock lock(H1_adj_locks[j]);
                    for (int LMk=0; LMk<num_LM*num_channels; ++LMk)
                        H1_adj[j*num_LM*num_channels+LMk] += H1_adj_ij[LMk];
                }
            }
        }
    }
}

template <typename Precision>
void MACE<Precision>::reverse_first_layer(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // H1
    // \bar{M}^{(0)}_{i,k'lm} += \sum_k w_{kl,k'} \bar{H}^{(1)}_{i,klm}
    {
        // [dE/dM0_il]_mk += \sum_k' [dE/dH1_il]_mk' [trans(W_l)]_k'k
        M0_adj.resize(M0.size());
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                    const auto H1_adj_ilm = H1_adj.data()+(i*num_LM+lm)*num_channels;
                    auto M0_adj_ilm = M0_adj.data()+(i*num_LM+lm)*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        Precision M0_adj_ilm_k = 0.0;
                        for (int kp=0; kp<num_channels; ++kp)
                            M0_adj_ilm_k += H1_adj_ilm[kp] * H1_weights_l[k*num_channels+kp];
                        M0_adj_ilm[k] += M0_adj_ilm_k;
                    }
                }
            }
        }
    }

    // M0
    // \bar{A}^{(0)}_{i,klm} = \sum_{l'm'} \bar{M}^{(0)}_{i,kl'm'} \partial M^{(0)}_{i,kl'm'} / \partial A^{(0)}_{i,klm}
    {
        A0_adj.assign(A0.size(), 0.0);
        #pragma omp parallel for
        for (int i=0; i<num_nodes; ++i) {
            auto B0_adj_i = std::vector<Precision>(num_B0*num_channels);
            auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
            auto M0_adj_i = M0_adj.data()+i*num_LM*num_channels;
            auto B0_i = B0.data()+i*num_B0*num_channels;
            auto M0_weights_i = M0_weights.data()+node_types[i]*M0_spec.size()*num_channels;
            for (int e=0; e<M0_spec.size(); ++e) {
                const auto [o, q, n] = M0_spec[e];
                for (int m=0; m<n; ++m)
                    for (int k=0; k<num_channels; ++k)
                        B0_adj_i[(q+m)*num_channels+k] += M0_weights_i[e*num_channels+k] * M0_adj_i[(o+m)*num_channels+k];
            }
            for (int t=B0_spec.size()-1; t>=0; --t) {
                const auto [q, a, b] = B0_spec[t];
                for (int k=0; k<num_channels; ++k) {
                    const Precision g = B0_coefficients[t] * B0_adj_i[q*num_channels+k];
                    B0_adj_i[a*num_channels+k] += g * B0_i[b*num_channels+k];
                    B0_adj_i[b*num_channels+k] += g * B0_i[a*num_channels+k];
                }
            }
            std::copy(B0_adj_i.begin(), B0_adj_i.begin()+num_lm*num_channels, A0_adj_i);
        }
    }

    // A0 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} \sum_{klm} \bar{A}^{(0)}_{i,klm} A^{(0)}_{i,klm} / s^{(0)}_i
    // \bar{A}^{(0)}_{i,klm} \leftarrow \bar{A}^{(0)}_{i,klm} / s^{(0)}_i
    if (A0_scaled) {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            auto A0_i = A0.data()+i*num_lm*num_channels;
            auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
            // recompute the scale factor
            double A0_scale_factor = 1.0;
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            }
            // update dE/dxyz
            double dA0_dot_A0 = 0.0;
            for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                dA0_dot_A0 += A0_adj_i[lmk] * A0_i[lmk];
            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                auto [f,d] = A0_splines[type_ij].evaluate_deriv(r[ij]);
                auto xyz_ij = xyz.data()+ij*3;
                auto node_forces_ij = node_forces.data()+ij*3;
                node_forces_ij[0] += dA0_dot_A0/A0_scale_factor*d*xyz_ij[0]/r[ij];
                node_forces_ij[1] += dA0_dot_A0/A0_scale_factor*d*xyz_ij[1]/r[ij];
                node_forces_ij[2] += dA0_dot_A0/A0_scale_factor*d*xyz_ij[2]/r[ij];
            }
            // update dE/dA0
            for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                A0_adj_i[lmk] /= A0_scale_factor;
        }
    }

    // A0
    // \bar{\Phi}^{(0)}_{i,k'lm} = \sum_k w_{a_ikl,k'} \bar{A}^{(0)}_{i,klm}
    // \mathbf{F}_{ij} -= \sum_{klm} \nabla_{\mathbf{r}_{ij}}[R^{(0)}_{a_ia_j,kl}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})] H^{(0)}_{a_j,k} \bar{\Phi}^{(0)}_{i,klm}
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];

            auto Phi0_adj_i = std::vector<Precision>(num_lm*num_channels);
            auto R0_ij = std::vector<Precision>(spl_set_0[0]->num_splines);
            auto R0_deriv_ij = std::vector<Precision>(R0_ij.size());

            // [dE/dPhi0_il]_mk = \sum_k' [dE/dA0_il]_mk' [trans(W_il)]_k'k
            for (int l=0; l<=l_max; ++l) {
                auto A0_weights_il = A0_weights[node_types[i]][l].data();
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                    auto Phi0_adj_i_lm = Phi0_adj_i.data()+lm*num_channels;
                    auto A0_adj_i_lm = A0_adj.data()+(i*num_lm+lm)*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        Precision Phi0_adj_i_lm_k = 0.0;
                        for (int kp=0; kp<num_channels; ++kp)
                            Phi0_adj_i_lm_k += A0_adj_i_lm[kp] * A0_weights_il[k*num_channels+kp];
                        Phi0_adj_i_lm[k] = Phi0_adj_i_lm_k;
                    }
                }
            }

            for (int ij=first_neigh[i]; ij<first_neigh[i]+num_neigh[i]; ++ij) {
                auto xyz_ij = xyz.data()+ij*3;
                auto r_ij = r[ij];
                auto Y_ij = Y.data()+ij*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
                auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
                auto node_forces_ij = node_forces.data()+ij*3;
                const int type_j = neigh_types[ij];
                const int type_ij = (type_i <= type_j)
                    ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                    : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
                spl_set_0[type_ij]->evaluate_derivs(r_ij, R0_ij, R0_deriv_ij);
                Precision dE_dr = 0.0;
                for (int l=0; l<=l_max; ++l) {
                    auto R0_ij_l = R0_ij.data()+l*num_channels;
                    auto R0_deriv_ij_l = R0_deriv_ij.data()+l*num_channels;
                    for (int m=-l; m<=l; ++m) {
                        const int lm = l*l+l+m;
                        auto Phi0_adj_i_lm = Phi0_adj_i.data()+lm*num_channels;
                        Precision dE_dr_lm = 0.0;
                        Precision dE_dY_lm = 0.0;
                        for (int k=0; k<num_channels; ++k) {
                            dE_dr_lm += Phi0_adj_i_lm[k] * R0_deriv_ij_l[k] * H0_ij[k];
                            dE_dY_lm += Phi0_adj_i_lm[k] * R0_ij_l[k] * H0_ij[k];
                        }
                        dE_dr += dE_dr_lm * Y_ij[lm];
                        for (int a=0; a<3; ++a)
                            node_forces_ij[a] -= dE_dY_lm * Y_grad_ij[a*num_lm+lm];
                    }
                }
                for (int a=0; a<3; ++a)
                    node_forces_ij[a] -= dE_dr * xyz_ij[a]/r_ij;
            }
        }
    }
}

template <typename Precision>
void MACE<Precision>::load_from_json(
    const std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);
    if (file.value("format_version", 0) != 1)
        throw std::runtime_error(filename + " was extracted by an incompatible symmetrix version; extract it again");
    if (file.contains("model_type"))
        throw std::runtime_error(filename + " is a " + file["model_type"].get<std::string>() + " model, not a MACE model");

    // Basic model information
    num_elements = file["num_elements"];
    num_channels = file["num_channels"];
    r_cut = file["r_cut"];
    l_max = file["l_max"];
    num_lm = (l_max+1)*(l_max+1);
    L_max = file["L_max"];
    num_LM = (L_max+1)*(L_max+1);
    atomic_numbers = file["atomic_numbers"].get<std::vector<int>>();
    atomic_energies = file["atomic_energies"].get<std::vector<double>>();

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl)
        zbl = ZBL(
            file["zbl_a_exp"].get<double>(),
            file["zbl_a_prefactor"].get<double>(),
            file["zbl_c"].get<std::vector<double>>(),
            file["zbl_covalent_radii"].get<std::vector<double>>(),
            file["zbl_p"].get<int>());

    // Radial splines
    const double spl_r_min = file.at("radial_spline_r_min");
    R0_spline_r_min = spl_r_min;
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<spl_values_0.size(); ++i)
        spl_set_0.push_back(std::make_unique<CubicSplineSet<Precision>>(spl_r_min, r_cut, spl_values_0[i], spl_derivs_0[i]));
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<spl_values_1.size(); ++i)
        spl_set_1.push_back(std::make_unique<CubicSplineSet<Precision>>(spl_r_min, r_cut, spl_values_1[i], spl_derivs_1[i]));

    // H0
    H0_weights = file["H0_weights"].get<std::vector<Precision>>();

    // A0
    A0_weights = file["A0_weights"].get<std::vector<std::vector<std::vector<Precision>>>>();

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled) {
        const double A0_spline_r_min = file.at("A0_spline_r_min");
        auto A0_spline_values = file["A0_spline_values"].get<std::vector<std::vector<double>>>();
        auto A0_spline_derivs = file["A0_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A0_spline_values.size(); ++i)
            A0_splines.push_back(CubicSpline(A0_spline_r_min, r_cut, A0_spline_values[i], A0_spline_derivs[i]));
    }

    // M0
    auto load_contraction = [&] (std::string l, auto& B_spec, auto& B_coefficients, auto& M_spec, auto& M_weights) {
        B_spec = file["B"+l+"_spec"].get<std::vector<std::array<int,3>>>();
        B_coefficients = file["B"+l+"_coefficients"].get<std::vector<Precision>>();
        M_spec = file["M"+l+"_spec"].get<std::vector<std::array<int,3>>>();
        for (const auto& w_a : file["M"+l+"_weights"])
            for (const auto& w_ae : w_a)
                for (const double w : w_ae)
                    M_weights.push_back(w);
        int num_B = num_lm;
        for (const auto& [q, a, b] : B_spec)
            num_B = std::max(num_B, q+1);
        return num_B;
    };
    num_B0 = load_contraction("0", B0_spec, B0_coefficients, M0_spec, M0_weights);

    // H1
    H1_weights = file["H1_weights"].get<std::vector<Precision>>();
    if (H0_weights.size() != num_elements*num_channels or H1_weights.size() != (L_max+1)*num_channels*num_channels)
        throw std::runtime_error(filename + ": weights do not match num_elements, num_channels and L_max");

    // Phi1
    Phi1_l = file["Phi1_l"].get<std::vector<int>>();
    Phi1_spec = file["Phi1_spec"].get<std::vector<std::array<int,3>>>();
    Phi1_cg_spec = file["Phi1_cg_spec"].get<std::vector<std::array<int,2>>>();
    Phi1_clebsch_gordan = file["Phi1_clebsch_gordan"].get<std::vector<Precision>>();
    num_lme = 0;
    for (auto l : Phi1_l)
        num_lme += 2*l+1;

    // A1
    A1_weights = file["A1_weights"].get<std::vector<std::vector<Precision>>>();
    A1_weights_trans = A1_weights;
    for (int l=0; l<A1_weights.size(); ++l) {
        const int num_ek = A1_weights[l].size()/num_channels;
        for (int ek=0; ek<num_ek; ++ek)
            for (int k=0; k<num_channels; ++k)
                A1_weights_trans[l][k*num_ek+ek] = A1_weights[l][ek*num_channels+k];
    }

    // A1 scaling
    A1_scaled = file["A1_scaled"].get<bool>();
    if (A1_scaled) {
        const double A1_spline_r_min = file.at("A1_spline_r_min");
        auto A1_spline_values = file["A1_spline_values"].get<std::vector<std::vector<double>>>();
        auto A1_spline_derivs = file["A1_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A1_spline_values.size(); ++i)
            A1_splines.push_back(CubicSpline(A1_spline_r_min, r_cut, A1_spline_values[i], A1_spline_derivs[i]));
    }

    // M1
    num_B1 = load_contraction("1", B1_spec, B1_coefficients, M1_spec, M1_weights);

    // H2
    H2_weights_for_M0 = file["H2_weights_for_M0"].get<std::vector<std::vector<Precision>>>();
    H2_weights_for_M1 = file["H2_weights_for_M1"].get<std::vector<Precision>>();

    // Readouts
    // TODO! hardcoded 16
    readout_1_weights_for_M0 = file["readout_1_weights_for_M0"].get<std::vector<Precision>>();
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    readout_2 = std::make_unique<MultilayerPerceptron>(
        std::vector<int>{num_channels, 16, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
}

template class MACE<double>;
template class MACE<float>;
