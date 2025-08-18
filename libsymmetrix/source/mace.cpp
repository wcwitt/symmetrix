#include <iostream> //TODO
#include <fstream>
#include <numbers>
#include <numeric>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "cblas.hpp"
#include "mace.hpp"

MACE::MACE(std::string filename)
{
    load_from_json(filename);
}

void MACE::compute_node_energies_forces(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    // TODO: best to resize these within individual routines
    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    compute_Y(xyz);

    compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1(num_nodes);

    compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Phi1(num_nodes, num_neigh, neigh_indices);
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
    reverse_A1(num_nodes);
    reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

    reverse_H1(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

void MACE::compute_R0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    const int num_spl = spl_set_0[0]->num_splines;
    R0.resize(r.size()*num_spl);
    R0_deriv.resize(R0.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            auto R0_ij = std::span<double>(R0.data()+ij*num_spl,num_spl);
            auto R0_deriv_ij = std::span<double>(R0_deriv.data()+ij*num_spl,num_spl);
            spl_set_0[type_ij]->evaluate_derivs(r[ij], R0_ij, R0_deriv_ij);
            ij += 1;
        }
    }
}

void MACE::compute_R1(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    const int num_spl = spl_set_1[0]->num_splines;
    R1.resize(r.size()*num_spl);
    R1_deriv.resize(R1.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            auto R1_ij = std::span<double>(R1.data()+ij*num_spl,num_spl);
            auto R1_deriv_ij = std::span<double>(R1_deriv.data()+ij*num_spl,num_spl);
            spl_set_1[type_ij]->evaluate_derivs(r[ij], R1_ij, R1_deriv_ij);
            ij += 1;
        }
    }
}

void MACE::compute_Y(
    std::span<const double> xyz)
{
    if (xyz.size() == 0) return;
    const int num = xyz.size()/3;
    Y.resize(num*num_lm);
    Y_grad.resize(3*num*num_lm);
    // shuffle to match e3nn conventions
    xyz_shuffled.resize(3*num);
    for (int i=0; i<num; ++i) {
        xyz_shuffled[3*i]   = xyz[3*i+2];
        xyz_shuffled[3*i+1] = xyz[3*i];
        xyz_shuffled[3*i+2] = xyz[3*i+1];
    }
    sphericart::SphericalHarmonics<double> sphericart(l_max);
    sphericart.compute_with_gradients(xyz_shuffled, Y, Y_grad);
    // normalize to match e3nn conventions
    for (int i=0; i<Y.size(); ++i)
        Y[i] *= 2*std::sqrt(std::numbers::pi);
    for (int i=0; i<Y_grad.size(); ++i)
        Y_grad[i] *= 2*std::sqrt(std::numbers::pi);
    // unshuffle gradient
    auto Y_grad_shuffled = Y_grad;
    for (int i=0; i<num; ++i) {
        for (int lm=0; lm<num_lm; ++lm) {
            Y_grad[3*i*num_lm+0*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+1*num_lm+lm];
            Y_grad[3*i*num_lm+1*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+2*num_lm+lm];
            Y_grad[3*i*num_lm+2*num_lm+lm] = Y_grad_shuffled[3*i*num_lm+0*num_lm+lm];
        }
    }
}

void MACE::compute_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types)
{
    A0.resize(num_nodes*num_lm*num_channels);

    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {

        // compute Phi0_i
        auto Phi0_i = std::vector<double>(num_lm*num_channels, 0.0);
        for (int j=0; j<num_neigh[i]; ++j) {
            auto Y_ij = Y.data()+ij*num_lm;
            auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            for (int l=0; l<=l_max; ++l) {
                auto R0_ij_l = R0.data()+ij*(l_max+1)*num_channels+l*num_channels;
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const double Y_ij_lm = Y_ij[lm];
                    auto Phi0_i_lm = Phi0_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        Phi0_i_lm[k] += R0_ij_l[k] * Y_ij_lm * H0_ij[k];
                    }
                }
            }
            ij += 1;
        }

        // [A0_il]_mk = \sum_k' [Phi0_il]_mk' [W_il]_k'k
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_il = Phi0_i.data()+l*l*num_channels;
            auto A0_il = A0.data()+(i*num_lm+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,                        // const CBLAS_LAYOUT Layout
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transa
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transb
                (2*l+1),                              // const MKL_INT m
                num_channels,                         // const MKL_INT n
                num_channels,                         // const MKL_INT k
                1.0,                                  // const double alpha
                Phi0_il,                              // const double *a
                num_channels,                         // const MKL_INT lda
                A0_weights[node_types[i]][l].data(),  // const double *b
                num_channels,                         // const MKL_INT ldb
                0.0,                                  // const double beta
                A0_il,                                // double *c
                num_channels);                        // const MKL_INT ldc
        }
    }
}

void MACE::reverse_A0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {

        auto Phi0_adj_i = std::vector<double>(num_lm*num_channels);

        // [dE/dPhi0_il]_mk = \sum_k' [dE/dA0_il]_mk' [trans(W_il)]_k'k 
        for (int l=0; l<=l_max; ++l) {
            auto Phi0_adj_il = Phi0_adj_i.data()+l*l*num_channels;
            auto A0_adj_il = A0_adj.data()+(i*num_lm+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,                        // const CBLAS_LAYOUT Layout
                CblasNoTrans,                         // const CBLAS_TRANSPOSE transa
                CblasTrans,                           // const CBLAS_TRANSPOSE transb
                (2*l+1),                              // const MKL_INT m
                num_channels,                         // const MKL_INT n
                num_channels,                         // const MKL_INT k
                1.0,                                  // const double alpha
                A0_adj_il,                            // const double *a
                num_channels,                         // const MKL_INT lda
                A0_weights[node_types[i]][l].data(),  // const double *b
                num_channels,                         // const MKL_INT ldb
                0.0,                                  // const double beta
                Phi0_adj_il,                          // double *c
                num_channels);                        // const MKL_INT ldc
        }

        // Warning: Assumes node_forces have been initialized elsewhere
        for (int j=0; j<num_neigh[i]; ++j) {
            auto xyz_ij = xyz.data()+ij*3;
            auto r_ij = r[ij];
            auto Y_ij = Y.data()+ij*num_lm;
            auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            auto H0_ij = H0_weights.data()+neigh_types[ij]*num_channels;
            auto node_forces_ij = node_forces.data()+ij*3;
            for (int l=0; l<=l_max; ++l) {
                auto R0_ij_l = R0.data()+ij*(l_max+1)*num_channels+l*num_channels;
                auto R0_deriv_ij_l = R0_deriv.data()+ij*(l_max+1)*num_channels+l*num_channels;
                for (int m=-l; m<=l; ++m) {
                    const int lm = l*l+l+m;
                    const double Y_ij_lm = Y_ij[lm];
                    const double Y_grad_ij_lm_x = Y_grad_ij[lm];
                    const double Y_grad_ij_lm_y = Y_grad_ij[num_lm+lm];
                    const double Y_grad_ij_lm_z = Y_grad_ij[2*num_lm+lm];
                    auto Phi0_adj_i_lm = Phi0_adj_i.data()+lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        node_forces_ij[0] += -Phi0_adj_i_lm[k] * (
                            xyz_ij[0]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_x * H0_ij[k] );
                        node_forces_ij[1] += -Phi0_adj_i_lm[k] * (
                            xyz_ij[1]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_y * H0_ij[k] );
                        node_forces_ij[2] += -Phi0_adj_i_lm[k] * (
                            xyz_ij[2]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                            + R0_ij_l[k] * Y_grad_ij_lm_z * H0_ij[k]);
                    }
                }
            }
            ij += 1;
        }
    }
}

void MACE::compute_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (not A0_scaled) return;
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        double A0_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        auto A0_i = A0.data()+i*num_lm*num_channels;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A0_i[lmk] /= A0_scale_factor;
    }
}

void MACE::reverse_A0_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r)
{
    if (not A0_scaled) return;
    // Warning: Assumes node_forces have been initialized elsewhere
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        auto A0_i = A0.data()+i*num_lm*num_channels;
        auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
        // recompute the scale factor
        double A0_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A0_scale_factor += A0_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        // update dE/dxyz
        double dA0_dot_A0 = 0.0;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            dA0_dot_A0 += A0_adj_i[lmk] * A0_i[lmk];
        ij = ij - num_neigh[i];
        for (int j=0; j<num_neigh[i]; ++j) {
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
            ij += 1;
        }
        // update dE/dA0
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A0_adj_i[lmk] /= A0_scale_factor;
    }
}

void MACE::compute_M0(
    const int num_nodes,
    std::span<const int> node_types)
{
    M0.resize(num_nodes*num_LM*num_channels);
    M0_grad.resize(num_nodes*num_channels*num_LM*num_lm);
    for (int i=0; i<num_nodes; ++i) {
        auto A0_i = A0.data()+i*num_lm*num_channels;
        auto M0_i = M0.data()+i*num_LM*num_channels;
        auto M0_grad_i = M0_grad.data()+i*num_LM*num_channels*num_lm;
        auto x = std::vector<double>(num_lm);
        int lmk = 0;
        for (int lm=0; lm<num_LM; ++lm) {
            for (int k=0; k<num_channels; ++k) {
                cblas_dcopy(num_lm, A0_i+k, num_channels, x.data(), 1);
                auto [f,g] = P0[node_types[i]*num_LM*num_channels+lmk].evaluate_gradient(x);
                M0_i[lmk] = f;
                cblas_dcopy(num_lm, g.data(), 1, M0_grad_i+lm*num_lm*num_channels+k, num_channels);
                lmk += 1;
            }
        }
    }
}

void MACE::reverse_M0(
    const int num_nodes,
    std::span<const int> node_types)
{
    A0_adj.resize(A0.size());
    std::fill(A0_adj.begin(), A0_adj.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto A0_adj_i = A0_adj.data()+i*num_lm*num_channels;
        auto M0_adj_i = M0_adj.data()+i*num_LM*num_channels;
        auto M0_grad_i = M0_grad.data()+i*num_LM*num_lm*num_channels;
        for (int lm=0; lm<num_lm; ++lm) {
            auto A0_adj_ilm = A0_adj_i + lm*num_channels;
            for (int lmp=0; lmp<num_LM; ++lmp) {
                auto M0_adj_ilmp = M0_adj_i + lmp*num_channels;
                auto M0_grad_ilmplm = M0_grad_i +
                    + lmp*num_lm*num_channels
                    + lm*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    A0_adj_ilm[k] += M0_grad_ilmplm[k] * M0_adj_ilmp[k];
                }
            }
        }
    }
}

void MACE::compute_H1(
    const int num_nodes)
{
    H1.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto M0_il = M0.data()+(i*num_LM+l*l)*num_channels;
            const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,  // const CBLAS_LAYOUT Layout
                CblasNoTrans,   // const CBLAS_TRANSPOSE transa
                CblasNoTrans,   // const CBLAS_TRANSPOSE transb
                2*l+1,          // const MKL_INT m
                num_channels,   // const MKL_INT n
                num_channels,   // const MKL_INT k
                1.0,            // const double alpha
                M0_il,          // const double *a
                num_channels,   // const MKL_INT lda
                H1_weights_l,   // const double *b
                num_channels,   // const MKL_INT ldb
                0.0,            // const double beta
                H1_il,          // double *c
                num_channels);  // const MKL_INT ldc
        }
    }
}

void MACE::reverse_H1(
    const int num_nodes)
{
    M0_adj.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            const auto H1_weights_l = H1_weights.data()+l*num_channels*num_channels;
            auto M0_adj_il = M0_adj.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,  // const CBLAS_LAYOUT Layout
                CblasNoTrans,   // const CBLAS_TRANSPOSE transa
                CblasTrans,     // const CBLAS_TRANSPOSE transb
                2*l+1,          // const MKL_INT m
                num_channels,   // const MKL_INT n
                num_channels,   // const MKL_INT k
                1.0,            // const double alpha
                H1_adj_il,      // const double *a
                num_channels,   // const MKL_INT lda
                H1_weights_l,   // const double *b
                num_channels,   // const MKL_INT ldb
                0.0,            // const double beta
                M0_adj_il,      // double *c
                num_channels);  // const MKL_INT ldc
        }
    }
}

void MACE::compute_Phi1(
    const int num_nodes,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices)
{
    // Compute Phi1_lelm1lm2 (named Phi1r)
    Phi1r.resize(num_nodes*num_lelm1lm2*num_channels);
    std::fill(Phi1r.begin(), Phi1r.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const double Y_ij_lm1 = Y_ij[lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto Phi1r_i_lelm1lm2 = Phi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            Phi1r_i_lelm1lm2[k] += R1_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k];
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
    // Compute Phi1 using CG coefficients
    Phi1.resize(num_nodes*num_lme*num_channels);
    std::fill(Phi1.begin(), Phi1.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_i = Phi1.data()+i*num_lme*num_channels;
        auto Phi1r_i = Phi1r.data()+i*num_lelm1lm2*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto Phi1_i_lme = Phi1_i+Phi1_lme[p]*num_channels;
            const double C = Phi1_clebsch_gordan[p];
            auto Phi1r_i_lelm1lm2 = Phi1r_i+Phi1_lelm1lm2[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                Phi1_i_lme[k] += C * Phi1r_i_lelm1lm2[k];
        }
    }
}

void MACE::reverse_Phi1(
    const int num_nodes,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    // Compute dE/dPhi1 (named dPhi1)
    dPhi1r.resize(Phi1r.size());
    std::fill(dPhi1r.begin(), dPhi1r.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        auto dPhi1_i = dPhi1.data()+i*num_lme*num_channels;
        for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
            auto dPhi1r_i_lelm1lm2 = dPhi1r_i+Phi1_lelm1lm2[p]*num_channels;
            const double C = Phi1_clebsch_gordan[p];
            auto dPhi1_i_lme = dPhi1_i+Phi1_lme[p]*num_channels;
            for (int k=0; k<num_channels; ++k)
                dPhi1r_i_lelm1lm2[k] += C * dPhi1_i_lme[k];
        }
    }
    // Compute partial forces
    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto node_forces_ij = node_forces.data()+3*ij;
            auto xyz_ij = xyz.data()+3*ij;
            auto r_ij = r[ij];
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto R1_deriv_ij = R1_deriv.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
            auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                auto R1_deriv_ij_lel1l2 = R1_deriv_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    const double Y_ij_lm1 = Y_ij[lm1];
                    const double Y_grad_ij_x_lm1 = Y_grad_ij[0*num_lm+lm1];
                    const double Y_grad_ij_y_lm1 = Y_grad_ij[1*num_lm+lm1];
                    const double Y_grad_ij_z_lm1 = Y_grad_ij[2*num_lm+lm1];
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_ij_lm2 = H1_ij+lm2*num_channels;
                        auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            node_forces_ij[0] += -dPhi1r_i_lelm1lm2[k] * (
                                xyz_ij[0]/r_ij * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_ij_lm2[k]);
                            node_forces_ij[1] += -dPhi1r_i_lelm1lm2[k] * (
                                xyz_ij[1]/r_ij * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_ij_lm2[k]);
                            node_forces_ij[2] += -dPhi1r_i_lelm1lm2[k] * (
                                xyz_ij[2]/r_ij * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_ij_lm2[k]);
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
    // Compute dE/dH1 (named dH1)
    H1_adj.resize(H1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        auto dPhi1r_i = dPhi1r.data()+i*num_lelm1lm2*num_channels;
        for (int j=0; j<num_neigh[i]; ++j) {
            auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
            auto Y_ij = Y.data()+ij*num_lm;
            auto H1_adj_ij = H1_adj.data()+neigh_indices[ij]*num_LM*num_channels;
            int lelm1lm2 = 0;
            for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                const int l1 = Phi1_l1[lel1l2];
                const int l2 = Phi1_l2[lel1l2];
                auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                    for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                        auto H1_adj_ij_lm2 = H1_adj_ij+lm2*num_channels;
                        auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            H1_adj_ij_lm2[k] += R1_ij_lel1l2[k]*Y_ij[lm1]*dPhi1r_i_lelm1lm2[k];
                        }
                        lelm1lm2 += 1;
                    }
                }
            }
            ij += 1;
        }
    }
}

void MACE::compute_A1(
    const int num_nodes)
{
    // The core matrix multiplication is:
    //         [A1_il]_mk = \sum_k' [Phi1_il]_m(ek') [W_il]_(ek')k
    A1.resize(num_nodes*num_lm*num_channels);
    int num_lme = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme += 2*l+1;
        num_e[l] += 1;
    }
    for (int i=0; i<num_nodes; ++i) {
        auto Phi1_il = Phi1.data()+i*num_lme*num_channels;
        auto A1_il = A1.data()+i*num_lm*num_channels;
        for (int l=0; l<=l_max; ++l) {
            cblas_dgemm(
                CblasRowMajor,          // const CBLAS_LAYOUT Layout
                CblasNoTrans,           // const CBLAS_TRANSPOSE transa
                CblasNoTrans,           // const CBLAS_TRANSPOSE transb
                (2*l+1),                // const MKL_INT m
                num_channels,           // const MKL_INT n
                num_e[l]*num_channels,  // const MKL_INT k
                1.0,                    // const double alpha
                Phi1_il,                // const double *a
                num_e[l]*num_channels,  // const MKL_INT lda
                A1_weights[l].data(),   // const double *b
                num_channels,           // const MKL_INT ldb
                0.0,                    // const double beta
                A1_il,                  // double *c
                num_channels);          // const MKL_INT ldc
            Phi1_il += (2*l+1)*num_e[l]*num_channels;
            A1_il += (2*l+1)*num_channels;
        }
    }
}

void MACE::reverse_A1(
    const int num_nodes)
{
    // The core matrix multiplication is:
    //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
    dPhi1.resize(Phi1.size());
    int num_lme = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme += 2*l+1;
        num_e[l] += 1;
    }
    for (int i=0; i<num_nodes; ++i) {
        auto A1_adj_il = A1_adj.data()+i*num_lm*num_channels;
        auto dPhi1_il = dPhi1.data()+i*num_lme*num_channels;
        for (int l=0; l<=l_max; ++l) {
            cblas_dgemm(
                CblasRowMajor,          // const CBLAS_LAYOUT Layout
                CblasNoTrans,           // const CBLAS_TRANSPOSE transa
                CblasTrans,             // const CBLAS_TRANSPOSE transb
                (2*l+1),                // const MKL_INT m
                num_e[l]*num_channels,  // const MKL_INT n
                num_channels,           // const MKL_INT k
                1.0,                    // const double alpha
                A1_adj_il,              // const double *a
                num_channels,           // const MKL_INT lda
                A1_weights[l].data(),   // const double *b
                num_channels,           // const MKL_INT ldb
                0.0,                    // const double beta
                dPhi1_il,            // double *c
                num_e[l]*num_channels); // const MKL_INT ldc
            A1_adj_il += (2*l+1)*num_channels;
            dPhi1_il += (2*l+1)*num_e[l]*num_channels;
        }
    }
}

void MACE::compute_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (not A1_scaled) return;
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        double A1_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        auto A1_i = A1.data()+i*num_lm*num_channels;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A1_i[lmk] /= A1_scale_factor;
    }
}

void MACE::reverse_A1_scaled(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    bool zero_dxyz)
{
    if (not A1_scaled) return;
    node_forces.resize(xyz.size());
    if (zero_dxyz)
        std::fill(node_forces.begin(), node_forces.end(), 0.0);
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        auto A1_i = A1.data()+i*num_lm*num_channels;
        auto A1_adj_i = A1_adj.data()+i*num_lm*num_channels;
        // recompute the scale factor
        double A1_scale_factor = 1.0;
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*atomic_numbers.size()-type_i-1)/2 + type_j
                : type_j*(2*atomic_numbers.size()-type_j-1)/2 + type_i;
            A1_scale_factor += A1_splines[type_ij].evaluate(r[ij]);
            ij += 1;
        }
        // update dE/dxyz
        double dA1_dot_A1 = 0.0;
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            dA1_dot_A1 += A1_adj_i[lmk] * A1_i[lmk];
        ij = ij - num_neigh[i];
        for (int j=0; j<num_neigh[i]; ++j) {
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
            ij += 1;
        }
        // update dE/dA1
        for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
            A1_adj_i[lmk] /= A1_scale_factor;
    }
}

void MACE::compute_M1(
    const int num_nodes,
    std::span<const int> node_types)
{
    M1.resize(num_nodes*num_channels);
    M1_grad.resize(num_nodes*num_channels*num_lm*num_channels);
    for (int i=0; i<num_nodes; ++i) {
        auto A1_i = A1.data()+i*num_lm*num_channels;
        auto M1_i = M1.data()+i*num_channels;
        auto M1_grad_i = M1_grad.data()+i*num_channels*num_lm;
        auto x = std::vector<double>(num_lm);
        for (int k=0; k<num_channels; ++k) {
            cblas_dcopy(num_lm, A1_i+k, num_channels, x.data(), 1);
            auto [f,g] = P1[node_types[i]*num_channels+k].evaluate_gradient(x);
            M1_i[k] = f;
            cblas_dcopy(num_lm, g.data(), 1, M1_grad_i+k, num_channels);
        }
    }
}

void MACE::reverse_M1(
    const int num_nodes,
    std::span<const int> node_types)
{
    A1_adj.resize(A1.size());
    for (int i=0; i<num_nodes; ++i) {
        auto M1_adj_i = M1_adj.data() + i*num_channels;
        for (int lm=0; lm<num_lm; ++lm) {
            auto A1_adj_ilm = A1_adj.data() + (i*num_lm+lm)*num_channels;
            auto M1_grad_ilm = M1_grad.begin() + (i*num_lm+lm)*num_channels;
            for (int k=0; k<num_channels; ++k) {
                A1_adj_ilm[k] = M1_grad_ilm[k] * M1_adj_i[k];
            }
        }
    }
}

void MACE::compute_H2(
    const int num_nodes,
    std::span<const int> node_types)
{
    H2.resize(num_nodes*num_channels);
    for (int i=0; i<num_nodes; ++i) {
        auto H2_i = H2.data()+i*num_channels;
        auto H1_i = H1.data()+i*num_LM*num_channels;
        cblas_dgemv(
            CblasRowMajor,                            // const CBLAS_LAYOUT Layout
            CblasTrans,                               // const CBLAS_TRANSPOSE trans
            num_channels,                             // const MKL_INT m
            num_channels,                             // const MKL_INT n
            1.0,                                      // const double alpha
            H2_weights_for_H1[node_types[i]].data(),  // const double *a
            num_channels,                             // const MKL_INT lda
            H1_i,                                     // const double *x
            1,                                        // const MKL_INT incx
            0.0,                                      // const double beta
            H2_i,                                     // double *y
            1);                                       // const MKL_INT incy
        auto M1_i = M1.data()+i*num_channels;
        cblas_dgemv(
            CblasRowMajor,             // const CBLAS_LAYOUT Layout
            CblasTrans,                // const CBLAS_TRANSPOSE trans
            num_channels,              // const MKL_INT m
            num_channels,              // const MKL_INT n
            1.0,                       // const double alpha
            H2_weights_for_M1.data(),  // const double *a
            num_channels,              // const MKL_INT lda
            M1_i,                      // const double *x
            1,                         // const MKL_INT incx
            1.0,                       // const double beta
            H2_i,                      // double *y
            1);                        // const MKL_INT incy
    }
}

void MACE::reverse_H2(
    const int num_nodes,
    std::span<const int> node_types,
    bool zero_H1_adj)
{
    H1_adj.resize(H1.size());
    M1_adj.resize(M1.size());
    if (zero_H1_adj)
        std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    for (int i=0; i<num_nodes; ++i) {
        auto H2_adj_i = H2_adj.data()+i*num_channels;
        auto H1_adj_i = H1_adj.data()+i*num_LM*num_channels;
        cblas_dgemv(
            CblasRowMajor,                            // const CBLAS_LAYOUT Layout
            CblasNoTrans,                             // const CBLAS_TRANSPOSE trans
            num_channels,                             // const MKL_INT m
            num_channels,                             // const MKL_INT n
            1.0,                                      // const double alpha
            H2_weights_for_H1[node_types[i]].data(),  // const double *a
            num_channels,                             // const MKL_INT lda
            H2_adj_i,                                 // const double *x
            1,                                        // const MKL_INT incx
            1.0,                                      // const double beta
            H1_adj_i,                                 // double *y
            1);                                       // const MKL_INT incy
        auto M1_adj_i = M1_adj.data()+i*num_channels;
        cblas_dgemv(
            CblasRowMajor,             // const CBLAS_LAYOUT Layout
            CblasNoTrans,              // const CBLAS_TRANSPOSE trans
            num_channels,              // const MKL_INT m
            num_channels,              // const MKL_INT n
            1.0,                       // const double alpha
            H2_weights_for_M1.data(),  // const double *a
            num_channels,              // const MKL_INT lda
            H2_adj_i,                  // const double *x
            1,                         // const MKL_INT incx
            0.0,                       // const double beta
            M1_adj_i,                  // double *y
            1);                        // const MKL_INT incy
    }
}

void MACE::compute_readouts(
    const int num_nodes,
    std::span<const int> node_types)
{
    node_energies.resize(num_nodes);
    H1_adj.resize(H1.size());
    // Warning: Although it doesn't appear necessary to set H1_adj to zero,
    //          it matters when the number of nodes associated with H1 is greater than num_nodes.
    //          There is probably a better way to manage this.
    std::fill(H1_adj.begin(), H1_adj.end(), 0.0);
    H2_adj.resize(H2.size());
    for (int i=0; i<num_nodes; ++i) {
        // atomic energies
        node_energies[i] += atomic_energies[node_types[i]];
        // first readout
        for (int k=0; k<num_channels; ++k) {
            node_energies[i] += readout_1_weights[k]*H1[i*num_LM*num_channels+k];
            H1_adj[i*num_LM*num_channels+k] = readout_1_weights[k];
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

void MACE::load_from_json(
    const std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);

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
    const double spl_h = file["radial_spline_h"];
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<spl_values_0.size(); ++i)
        spl_set_0.push_back(std::make_unique<CubicSplineSet>(spl_h, spl_values_0[i], spl_derivs_0[i]));
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<spl_values_1.size(); ++i)
        spl_set_1.push_back(std::make_unique<CubicSplineSet>(spl_h, spl_values_1[i], spl_derivs_1[i]));

    // H0
    H0_weights = file["H0_weights"].get<std::vector<double>>();

    // A0
    A0_weights = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled) {
        const double A0_spline_h = file["A0_spline_h"];
        auto A0_spline_values = file["A0_spline_values"].get<std::vector<std::vector<double>>>();
        auto A0_spline_derivs = file["A0_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A0_spline_values.size(); ++i)
            A0_splines.push_back(CubicSpline(A0_spline_h, A0_spline_values[i], A0_spline_derivs[i]));
    }

    // M0
    auto M0_weights = file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>();
    auto M0_monomials = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
    P0 = std::vector<MultivariatePolynomial>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int lm=0; lm<num_LM; ++lm) {
            for (int k=0; k<num_channels; ++k) {
                P0.push_back(MultivariatePolynomial(
                    num_lm,
                    M0_weights[std::to_string(a)][std::to_string(lm)][std::to_string(k)],
                    M0_monomials[std::to_string(lm)]));
            }
        }
    }

    // H1
    H1_weights = file["H1_weights"].get<std::vector<double>>();

    // Phi1
    Phi1_l = file["Phi1_l"].get<std::vector<int>>();
    Phi1_l1 = file["Phi1_l1"].get<std::vector<int>>();
    Phi1_l2 = file["Phi1_l2"].get<std::vector<int>>();
    Phi1_lme = file["Phi1_lme"].get<std::vector<int>>();
    Phi1_clebsch_gordan = file["Phi1_clebsch_gordan"].get<std::vector<double>>();
    Phi1_lelm1lm2 = file["Phi1_lelm1lm2"].get<std::vector<int>>();
    num_lme = 0;
    for (auto l : Phi1_l)
        num_lme += 2*l+1;
    num_lelm1lm2 = 0;
    for (int le=0; le<Phi1_l.size(); ++le)
        num_lelm1lm2 += (2*Phi1_l1[le]+1)*(2*Phi1_l2[le]+1);

    // A1
    A1_weights = file["A1_weights"].get<std::vector<std::vector<double>>>();

    // A1 scaling
    A1_scaled = file["A1_scaled"].get<bool>();
    if (A1_scaled) {
        const double A1_spline_h = file["A1_spline_h"];
        auto A1_spline_values = file["A1_spline_values"].get<std::vector<std::vector<double>>>();
        auto A1_spline_derivs = file["A1_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A1_spline_values.size(); ++i)
            A1_splines.push_back(CubicSpline(A1_spline_h, A1_spline_values[i], A1_spline_derivs[i]));
    }

    // M1
    auto M1_weights = file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>();
    auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
    P1 = std::vector<MultivariatePolynomial>();
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int k=0; k<num_channels; ++k) {
            P1.push_back(MultivariatePolynomial(
                num_lm,
                M1_weights[std::to_string(a)][std::to_string(k)],
                M1_monomials));
        }
    }

    // H2
    H2_weights_for_H1 = file["H2_weights_for_H1"].get<std::vector<std::vector<double>>>();
    H2_weights_for_M1 = file["H2_weights_for_M1"].get<std::vector<double>>();

    // Readouts
    // TODO! hardcoded 16
    readout_1_weights = file["readout_1_weights"].get<std::vector<double>>();
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    readout_2 = std::make_unique<MultilayerPerceptron>(
        std::vector<int>{num_channels, 16, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
}
