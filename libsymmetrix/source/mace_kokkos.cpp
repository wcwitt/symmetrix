#include <fstream>
#include <numbers>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"

MACEKokkos::MACEKokkos(std::string filename)
{
    load_from_json(filename);
}

void MACEKokkos::compute_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    Kokkos::realloc(node_energies, num_nodes);
    Kokkos::deep_copy(node_energies, 0.0);
    Kokkos::realloc(node_forces, xyz.size());
    Kokkos::deep_copy(node_forces, 0.0);

    if (has_zbl)
        zbl.compute_ZBL(
            num_nodes, node_types, num_neigh, neigh_types,
            atomic_numbers, r, xyz, node_energies, node_forces);

    compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Y(xyz);

    compute_Phi0(num_nodes, num_neigh, neigh_types);
    compute_A0(num_nodes, node_types);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1(num_nodes);

    compute_Phi1(num_nodes, num_neigh, neigh_indices);
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A1(num_nodes);
    reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

    reverse_H1(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A0(num_nodes, node_types);
    reverse_Phi0(num_nodes, num_neigh, neigh_types, xyz, r);
}

void MACEKokkos::compute_R0(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (r.size() > R0.extent(0)) {
        Kokkos::realloc(R0, r.size(), (l_max+1)*num_channels);
        Kokkos::realloc(R0_deriv, r.size(), (l_max+1)*num_channels);
    }
    radial_0.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, R0, R0_deriv);
    Kokkos::fence();
}

void MACEKokkos::compute_R1(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (r.size() > R1.extent(0)) {
        Kokkos::realloc(R1, r.size(), Phi1_l.size()*num_channels);
        Kokkos::realloc(R1_deriv, r.size(), Phi1_l.size()*num_channels);
    }
    radial_1.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, R1, R1_deriv);
    Kokkos::fence();
}

void MACEKokkos::compute_Y(Kokkos::View<const double*> xyz) {

#ifndef SYMMETRIX_SPHERICART_CUDA

    const int num = xyz.extent(0) / 3;
    Kokkos::realloc(Y, num*num_lm);
    Kokkos::realloc(Y_grad, 3*num*num_lm);

    const auto num_lm = this->num_lm;
    auto Y = this->Y;
    auto Y_grad = this->Y_grad;

    // shuffle to match e3nn
    auto xyz_shuffled = Kokkos::View<double*>("xyz_unshuffled", 3*num);
    Kokkos::parallel_for("shuffle_xyz", num, KOKKOS_LAMBDA (int i) {
        xyz_shuffled(3*i) = xyz(3*i+2);
        xyz_shuffled(3*i+1) = xyz(3*i);
        xyz_shuffled(3*i+2) = xyz(3*i+1);
    });
    Kokkos::fence();

    // call sphericart on host
    auto h_xyz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xyz_shuffled);
    auto h_Y = Kokkos::create_mirror_view(Y);
    auto h_Y_grad = Kokkos::create_mirror_view(Y_grad);
    sphericart::SphericalHarmonics<double> sphericart(l_max);
    sphericart.compute_array_with_gradients(
        h_xyz.data(), 3*num, h_Y.data(), 3*num*num_lm, h_Y_grad.data(), 3*num*num_lm),
    Kokkos::deep_copy(Y, h_Y);
    Kokkos::deep_copy(Y_grad, h_Y_grad);

    // unshuffle gradient
    Kokkos::View<double*> Y_grad_shuffled("Y_grad_shuffled", 3*num*num_lm);
    Kokkos::deep_copy(Y_grad_shuffled, Y_grad);
    Kokkos::parallel_for("unshuffle_Y_grad", num, KOKKOS_LAMBDA (int i) {
        for (int lm=0; lm<num_lm; ++lm) {
            Y_grad(3*i*num_lm+0*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+1*num_lm+lm);
            Y_grad(3*i*num_lm+1*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+2*num_lm+lm);
            Y_grad(3*i*num_lm+2*num_lm+lm) = Y_grad_shuffled(3*i*num_lm+0*num_lm+lm);
        }
    });
    Kokkos::fence();

    // normalize to match e3nn conventions
    Kokkos::parallel_for("normalize_Y", num*num_lm, KOKKOS_LAMBDA (int i) {
        Y(i) *= 2*std::sqrt(M_PI);
    });
    Kokkos::parallel_for("normalize_Y_grad", 3*num*num_lm, KOKKOS_LAMBDA (int i) {
        Y_grad(i) *= 2*std::sqrt(M_PI);
    });
    Kokkos::fence();

#else // SYMMETRIX_SPHERICART_CUDA

    const int num = xyz.extent(0) / 3;
    const int num_lm = (l_max+1)*(l_max+1);
    if (Y.size() < num*num_lm) {
        Kokkos::realloc(Y, num*num_lm);
        Kokkos::realloc(Y_grad, 3*num*num_lm);
    }
    sphericart::cuda::SphericalHarmonics<double> sphericart(l_max);
    sphericart.compute_with_gradients(xyz.data(), num, Y.data(), Y_grad.data());
    // normalize to match e3nn conventions
    const double normalization_factor = 2 * std::sqrt(M_PI);
    auto Y = this->Y;
    Kokkos::parallel_for("normalize_Y", num*num_lm, KOKKOS_LAMBDA (int i) {
        Y(i) *= normalization_factor;
    });
    auto Y_grad = this->Y_grad;
    Kokkos::parallel_for("normalize_Y_grad", 3*num*num_lm, KOKKOS_LAMBDA (int i) {
        Y_grad(i) *= normalization_factor;
    });

#endif

    Kokkos::fence();
}

void MACEKokkos::compute_Phi0(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types)
{
    if (Phi0.extent(0) != num_nodes)
        Kokkos::realloc(Phi0, num_nodes, num_lm, num_channels);
    Kokkos::deep_copy(Phi0, 0.0);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // local references to class members accessed in the parallel region
    auto num_lm = this->num_lm;
    auto num_channels = this->num_channels;
    auto R0 = this->R0;
    auto Y = this->Y;
    auto H0_weights = this->H0_weights;
    auto Phi0 = this->Phi0;

    Kokkos::parallel_for("ComputePhi0",
        Kokkos::TeamPolicy<>(num_nodes*num_lm, 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_lm;
            const int lm = team_member.league_rank() % num_lm;
            const int l = Kokkos::sqrt(lm);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = first_neigh(i) + j;
                const double Y_ij_lm = Y(ij*num_lm+lm);
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        Phi0(i,lm,k) += R0(ij,l*num_channels+k)*Y_ij_lm*H0_weights(neigh_types(ij),k);
                    });
            }
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_Phi0(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    // local references to class members accessed in the parallel region
    auto num_lm = this->num_lm;
    auto num_channels = this->num_channels;
    auto R0 = this->R0;
    auto R0_deriv = this->R0_deriv;
    auto Y = this->Y;
    auto Y_grad = this->Y_grad;
    auto H0_weights = this->H0_weights;
    auto Phi0_adj = this->Phi0_adj;
    auto node_forces = this->node_forces;

    int total_num_neigh;
    Kokkos::parallel_reduce("total_num_neigh", num_nodes, KOKKOS_LAMBDA (const int i, int& sum) {
        sum += num_neigh(i);
    }, total_num_neigh);
    Kokkos::fence();
    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i); 
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();
    Kokkos::View<int*> i_list("i_list", total_num_neigh);
    Kokkos::parallel_for("ij lists",
        num_nodes,
        KOKKOS_LAMBDA (const int i) {
            int ij = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                i_list(ij) = i;
                ij += 1;
            }
        });
    Kokkos::fence();

    Kokkos::parallel_for("Reverse Phi0",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                const double r_ij = r(ij);
                const double x_ij = xyz(3*ij) / r_ij;
                const double y_ij = xyz(3*ij+1) / r_ij;
                const double z_ij = xyz(3*ij+2) / r_ij;
                const double* Y_ij = &Y(ij*num_lm);
                const double* Y_grad_ij = &Y_grad(3*ij*num_lm);
                const double* H0_ij = &H0_weights(neigh_types(ij),0);
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lm),
                    [&] (const int lm, double& f_x, double& f_y, double& f_z) {
                        const int l = Kokkos::sqrt(lm);
                        double t1, t2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [&] (const int k, double& t1, double& t2) {
                                t1 += R0_deriv(ij,l*num_channels+k) * H0_ij[k] * Phi0_adj(i,lm,k);
                                t2 += R0(ij,l*num_channels+k) * H0_ij[k] * Phi0_adj(i,lm,k);
                            }, t1, t2);
                        f_x += t1*x_ij*Y_ij[lm] + t2*Y_grad_ij[lm];
                        f_y += t1*y_ij*Y_ij[lm] + t2*Y_grad_ij[num_lm+lm];
                        f_z += t1*z_ij*Y_ij[lm] + t2*Y_grad_ij[2*num_lm+lm];
                    }, f_x, f_y, f_z);
                    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                        node_forces(3*ij)   -= f_x;
                        node_forces(3*ij+1) -= f_y;
                        node_forces(3*ij+2) -= f_z;
                    });
            }
        });

    Kokkos::fence();
}


void MACEKokkos::compute_A0(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    if (A0.extent(0) != num_nodes)
        Kokkos::realloc(A0, num_nodes, num_lm, num_channels);

    // local references to class members accessed in the parallel region
    auto l_max = this->l_max;
    auto Phi0 = this->Phi0;
    auto A0_weights = this->A0_weights;
    auto A0 = this->A0;

    Kokkos::parallel_for("Compute A0",
        Kokkos::TeamPolicy<>(num_nodes*(l_max+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (l_max+1);
            const int l = team_member.league_rank() % (l_max+1);
            auto Phi0_il = Kokkos::subview(Phi0, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(A0_weights, node_types(i), l, Kokkos::ALL, Kokkos::ALL);
            auto A0_il = Kokkos::subview(A0, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Blocked>
                ::invoke(team_member, 1.0, Phi0_il, W_il, 0.0, A0_il);
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_A0(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    if (Phi0_adj.extent(0) != num_nodes)
        Kokkos::realloc(Phi0_adj, num_nodes, num_lm, num_channels);

    // local references to class members accessed in the parallel region
    auto l_max = this->l_max;
    auto A0_adj = this->A0_adj;
    auto A0_weights = this->A0_weights;
    auto Phi0_adj = this->Phi0_adj;

    Kokkos::parallel_for("Reverse A0",
        Kokkos::TeamPolicy<>(num_nodes*(l_max+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (l_max+1);
            const int l = team_member.league_rank() % (l_max+1);
            auto A0_adj_il = Kokkos::subview(A0_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(A0_weights, node_types(i), l, Kokkos::ALL, Kokkos::ALL);
            auto Phi0_adj_il = Kokkos::subview(Phi0_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::Transpose,
                                    KokkosBatched::Algo::Gemm::Blocked>
                ::invoke(team_member, 1.0, A0_adj_il, W_il, 0.0, Phi0_adj_il);
        });
    Kokkos::fence();
}

void MACEKokkos::compute_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (not A0_scaled) return;
    // compute A0 splines
    if (r.size() > A0_spline_values.extent(0)) {
        Kokkos::realloc(A0_spline_values, r.size(), 1);
        Kokkos::realloc(A0_spline_derivs, r.size(), 1);
    }
    A0_splines.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, A0_spline_values, A0_spline_derivs);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // perform the scaling
    const auto A0_spline_values = this->A0_spline_values;
    const auto A0_spline_derivs = this->A0_spline_derivs;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto A0 = this->A0;
    Kokkos::parallel_for(
        "MACEKokkos::compute_A0_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            // compute scale factor
            double A0_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A0_spline_values(first_neigh(i)+j,0);
                }, A0_scale_factor);
            A0_scale_factor += 1.0;
            team_member.team_barrier();
            // perform the scaling
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A0(i,lmk/num_channels,lmk%num_channels) /= A0_scale_factor;
                });
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_A0_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (not A0_scaled) return;
    // compute A0 splines
    if (r.size() > A0_spline_values.extent(0)) {
        Kokkos::realloc(A0_spline_values, r.size(), 1);
        Kokkos::realloc(A0_spline_derivs, r.size(), 1);
    }
    A0_splines.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, A0_spline_values, A0_spline_derivs);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // update the deriviates
    // Warning: Assumes node_forces have been initialized elsewhere
    const auto A0 = this->A0;
    const auto A0_adj = this->A0_adj;
    const auto A0_spline_values = this->A0_spline_values;
    const auto A0_spline_derivs = this->A0_spline_derivs;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto node_forces = this->node_forces;
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A0_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            // compute scale factor
            double A0_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A0_spline_values(first_neigh(i)+j,0);
                }, A0_scale_factor);
            A0_scale_factor += 1.0;
            team_member.team_barrier();
            // update dE/dxyz
            double dA0_dot_A0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (const int lmk, double& lsum) {
                    const int lm = lmk / num_channels;
                    const int k = lmk % num_channels;
                    lsum += A0_adj(i,lm,k) * A0(i,lm,k);
                }, dA0_dot_A0);
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j) {
                    const int ij = first_neigh(i) + j;
                    const double f = A0_spline_values(ij,0);
                    const double d = A0_spline_derivs(ij,0);
                    node_forces(3*ij+0) += dA0_dot_A0/A0_scale_factor*d*xyz(3*ij+0)/r(ij);
                    node_forces(3*ij+1) += dA0_dot_A0/A0_scale_factor*d*xyz(3*ij+1)/r(ij);
                    node_forces(3*ij+2) += dA0_dot_A0/A0_scale_factor*d*xyz(3*ij+2)/r(ij);
                });
            // update dE/dA0
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A0_adj(i,lmk/num_channels,lmk%num_channels) /= A0_scale_factor;
                });
        });
    Kokkos::fence();
}

void MACEKokkos::compute_M0(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    Kokkos::resize(M0, num_nodes, num_LM, num_channels);

    auto A0 = this->A0;
    auto M0 = this->M0;
    auto M0_monomials = this->M0_monomials;
    auto M0_weights = this->M0_weights;
    auto num_LM = this->num_LM;
    auto num_channels = this->num_channels;

    Kokkos::fence();
    Kokkos::parallel_for(
        "Compute M0",
        Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right>>(
            {0,0,0}, {num_nodes,num_LM,num_channels}),
        KOKKOS_LAMBDA (const int i, const int LM, const int k) {
            M0(i,LM,k) = 0.0;
            for (int u=0; u<M0_monomials(LM).extent(0); ++u) {
                double monomial = A0(i,M0_monomials(LM)(u,0),k);
                for (int v=1; v<M0_monomials(LM).extent(1); ++v) {
                    if (M0_monomials(LM)(u,v) == -1)
                        break;
                    monomial *= A0(i,M0_monomials(LM)(u,v),k);
                }
                M0(i,LM,k) += M0_weights(LM)(node_types(i),k,u) * monomial;
            }
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_M0(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(A0_adj, A0.extent(0), A0.extent(1), A0.extent(2));
    Kokkos::deep_copy(A0_adj, 0.0);

    // local references to class members accessed in the parallel region
    auto A0 = this->A0;
    auto A0_adj = this->A0_adj;
    auto M0_adj = this->M0_adj;
    auto M0_monomials = this->M0_monomials;
    auto M0_weights = this->M0_weights;
    auto num_channels = this->num_channels;
    auto num_LM = this->num_LM;

    Kokkos::parallel_for(
        "Reverse M0",
        Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Right>>(
            {0,0,0}, {num_nodes,num_LM,num_channels}),
        KOKKOS_LAMBDA (const int i, const int LM, const int k) {
            for (int u=0; u<M0_monomials(LM).extent(0); ++u) {
                for (int v=0; v<M0_monomials(LM).extent(1); ++v) {
                    if (M0_monomials(LM)(u,v) == -1) break;
                    double deriv = M0_adj(i,LM,k);
                    for (int w=0; w<M0_monomials(LM).extent(1); ++w) {
                        if (M0_monomials(LM)(u,w) == -1) break;
                        if (v == w) continue;
                        deriv *= A0(i,M0_monomials(LM)(u,w),k);
                    }
                    Kokkos::atomic_add(
                        &A0_adj(i,M0_monomials(LM)(u,v),k),
                        M0_weights(LM)(node_types(i),k,u)*deriv);
                }
            }
        });
    Kokkos::fence();
}

void MACEKokkos::compute_H1(
    const int num_nodes)
{
    Kokkos::realloc(H1, M0.extent(0), M0.extent(1), M0.extent(2));

    auto L_max = this->L_max;
    auto H1 = this->H1;
    auto H1_weights = this->H1_weights;
    auto M0 = this->M0;

    Kokkos::parallel_for("Compute H1",
        Kokkos::TeamPolicy<>(num_nodes*(L_max+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto M0_il = Kokkos::subview(M0, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto H1_il = Kokkos::subview(H1, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, M0_il, W_il, 0.0, H1_il);
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_H1(
    const int num_nodes)
{
    Kokkos::realloc(M0_adj, M0.extent(0), M0.extent(1), M0.extent(2));

    auto L_max = this->L_max;
    auto M0_adj = this->M0_adj;
    auto H1_weights = this->H1_weights;
    auto H1_adj = this->H1_adj;

    Kokkos::parallel_for("Reverse H1",
        Kokkos::TeamPolicy<>(num_nodes*(L_max+1), Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / (L_max+1);
            const int l = team_member.league_rank() % (L_max+1);
            auto H1_adj_il = Kokkos::subview(H1_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            auto W_il = Kokkos::subview(H1_weights, l, Kokkos::ALL, Kokkos::ALL);
            auto M0_adj_il = Kokkos::subview(M0_adj, i, Kokkos::make_pair(l*l, l*(l+2)+1), Kokkos::ALL);
            KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                    KokkosBatched::Trans::NoTranspose,
                                    KokkosBatched::Trans::Transpose,
                                    KokkosBatched::Algo::Gemm::Unblocked>
                ::invoke(team_member, 1.0, H1_adj_il, W_il, 0.0, M0_adj_il);
        });
    Kokkos::fence();
}

void MACEKokkos::compute_Phi1(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices)
{
    // Compute Phi1_lelm1lm2 (named Phi1r)
    Kokkos::realloc(Phi1r, num_nodes, num_lelm1lm2, num_channels);
    Kokkos::deep_copy(Phi1r, 0.0);
    Kokkos::realloc(Phi1, num_nodes, num_lme, num_channels);
    Kokkos::deep_copy(Phi1, 0.0);

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    const auto num_lelm1lm2 = this->num_lelm1lm2;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto Phi1_lel1l2 = this->Phi1_lel1l2;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto R1 = this->R1;
    const auto Y = this->Y;
    const auto H1 = this->H1;
    auto Phi1 = this->Phi1;
    auto Phi1r = this->Phi1r;

    Kokkos::parallel_for("Compute Phi1r",
        Kokkos::TeamPolicy<>(num_nodes*num_lelm1lm2, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank() / num_lelm1lm2;
            const int lelm1lm2 = team_member.league_rank() % num_lelm1lm2;
            const int i0 = first_neigh(i);
            const int lm1 = Phi1_lm1(lelm1lm2);
            const int lm2 = Phi1_lm2(lelm1lm2);
            const int lel1l2 = Phi1_lel1l2(lelm1lm2);
            double* Phi1r_i_lelm1lm2 = &Phi1r(i,lelm1lm2,0);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                const double* R1_ij_lel1l2 = &R1(ij,lel1l2*num_channels);
                const double Y_ij_lm1 = Y(ij*num_lm+lm1);
                const double* H1_ij_lm2 = &H1(neigh_indices(ij),lm2,0);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, num_channels),
                    [=] (const int k) {
                        Phi1r_i_lelm1lm2[k] += R1_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k];
                    });
            }
        });
    Kokkos::fence();

    // Compute Phi1 using CG coefficients
    Kokkos::parallel_for("Compute Phi1",
        Kokkos::TeamPolicy<>(num_nodes, 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        Phi1(i,Phi1_lme(p),k) += C * Phi1r(i,Phi1_lelm1lm2(p),k);
                    });
            }
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_Phi1(
    const int num_nodes,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    bool zero_dxyz,
    bool zero_H1_adj)
{
    Kokkos::realloc(dPhi1r, Phi1r.extent(0), Phi1r.extent(1), Phi1r.extent(2)); 
    Kokkos::resize(node_forces, xyz.size());
    if (zero_dxyz)
        Kokkos::deep_copy(node_forces, 0.0);
    Kokkos::deep_copy(dPhi1r, 0.0);
    Kokkos::resize(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (zero_H1_adj)
        Kokkos::deep_copy(H1_adj, 0.0);

    const auto num_lm = this->num_lm;
    const auto num_channels = this->num_channels;
    const auto num_lelm1lm2 = this->num_lelm1lm2;
    const auto Phi1_lm1 = this->Phi1_lm1;
    const auto Phi1_lm2 = this->Phi1_lm2;
    const auto Phi1_lel1l2 = this->Phi1_lel1l2;
    const auto Phi1_lme = this->Phi1_lme;
    const auto Phi1_lelm1lm2 = this->Phi1_lelm1lm2;
    const auto Phi1_clebsch_gordan = this->Phi1_clebsch_gordan;
    const auto R1 = this->R1;
    const auto R1_deriv = this->R1_deriv;
    const auto Y = this->Y;
    const auto Y_grad = this->Y_grad;
    const auto H1 = this->H1;
    const auto H1_adj = this->H1_adj;
    const auto node_forces = this->node_forces;
    auto dPhi1r = this->dPhi1r;
    auto dPhi1 = this->dPhi1;

    // Compute dE/dPhi1 (named dPhi1)
    Kokkos::parallel_for("Reverse Phi1",
        Kokkos::TeamPolicy<>(num_nodes, 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan(p);
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, num_channels),
                    [&] (const int k) {
                        dPhi1r(i,Phi1_lelm1lm2(p),k) += C * dPhi1(i,Phi1_lme(p),k);
                    });
            }
        });

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            const int num_neigh_i = num_neigh(i); 
            if (final)
                first_neigh(i) = update;
            update += num_neigh_i;
        });
    Kokkos::fence();

    Kokkos::parallel_for("Reverse Phi1r",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            for (int j=0; j<num_neigh(i); ++j) {
                const int ij = i0 + j;
                const double r_ij = r(ij);
                const double x_ij = xyz(3*ij) / r_ij;
                const double y_ij = xyz(3*ij+1) / r_ij;
                const double z_ij = xyz(3*ij+2) / r_ij;
                double f_x, f_y, f_z;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team_member, num_lelm1lm2),
                    [&] (const int lelm1lm2, double& f_x, double& f_y, double& f_z) {
                        const int lm1 = Phi1_lm1(lelm1lm2);
                        const int lm2 = Phi1_lm2(lelm1lm2);
                        const int lel1l2 = Phi1_lel1l2(lelm1lm2);
                        const double* R1_ij_lel1l2 = &R1(ij,lel1l2*num_channels);
                        const double* R1_deriv_ij_lel1l2 = &R1_deriv(ij,lel1l2*num_channels);
                        const double Y_ij_lm1 = Y(ij*num_lm+lm1);
                        const double xY_ij_lm1 = x_ij * Y_ij_lm1;
                        const double yY_ij_lm1 = y_ij * Y_ij_lm1;
                        const double zY_ij_lm1 = z_ij * Y_ij_lm1;
                        const double Y_grad_ij_x_lm1 = Y_grad(3*ij*num_lm+lm1);
                        const double Y_grad_ij_y_lm1 = Y_grad((3*ij+1)*num_lm+lm1);
                        const double Y_grad_ij_z_lm1 = Y_grad((3*ij+2)*num_lm+lm1);
                        const double* H1_ij_lm2 = &H1(neigh_indices(ij),lm2,0);
                        double* H1_adj_ij_lm2 = &H1_adj(neigh_indices(ij),lm2,0);
                        const double* dPhi1r_i_lelm1lm2 = &dPhi1r(i,lelm1lm2,0);
                        double t1, t2;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team_member, num_channels),
                            [&] (const int k, double& t1, double& t2) {
                                t1 += R1_deriv_ij_lel1l2[k] * H1_ij_lm2[k] * dPhi1r_i_lelm1lm2[k]; 
                                t2 += R1_ij_lel1l2[k] * H1_ij_lm2[k] * dPhi1r_i_lelm1lm2[k];
                                Kokkos::atomic_add(// TODO: use scratch space?
                                    H1_adj_ij_lm2+k,
                                    R1_ij_lel1l2[k]  * Y_ij_lm1 * dPhi1r_i_lelm1lm2[k]);
                            }, t1, t2);
                        f_x += t1*xY_ij_lm1 + t2*Y_grad_ij_x_lm1;
                        f_y += t1*yY_ij_lm1 + t2*Y_grad_ij_y_lm1;
                        f_z += t1*zY_ij_lm1 + t2*Y_grad_ij_z_lm1;
                    }, f_x, f_y, f_z);
                    Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                        node_forces(3*ij)   -= f_x;
                        node_forces(3*ij+1) -= f_y;
                        node_forces(3*ij+2) -= f_z;
                    });
            }
        });
    Kokkos::fence();
}

void MACEKokkos::compute_A1(int num_nodes)
{
    // The core matrix multiplication is:
    //         [A1_il]_mk = \sum_(ek') [Phi1_il]_m(ek') [W_il]_(ek')k
    Kokkos::realloc(A1, num_nodes, num_lm, num_channels);
    Kokkos::deep_copy(A1, 0.0);

    const auto l_max = this->l_max;
    const auto num_channels = this->num_channels;
    const auto Phi1_l = this->Phi1_l;
    const auto Phi1 = this->Phi1;
    auto A1_weights = this->A1_weights;
    auto A1 = this->A1;

    Kokkos::parallel_for("Compute A1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            int lme = 0;
            for (int l=0; l<=l_max; ++l) {
                int num_eta = 0;
                for (int j=0; j<Phi1_l.size(); ++j)
                    num_eta += (Phi1_l(j) == l);
                auto Phi1_il = Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                    &Phi1(i,lme,0), 2*l+1, num_eta*num_channels);
                auto A1_il = Kokkos::subview(A1, i, Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                        KokkosBatched::Trans::NoTranspose,
                                        KokkosBatched::Trans::NoTranspose,
                                        KokkosBatched::Algo::Gemm::Blocked>
                    ::invoke(team_member, 1.0, Phi1_il, A1_weights(l), 0.0, A1_il);
                lme += (2*l+1)*num_eta;
            }

        });
    Kokkos::fence();
}

void MACEKokkos::reverse_A1(int num_nodes)
{
    // The core matrix multiplication is:
    //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
    Kokkos::realloc(dPhi1, Phi1.extent(0), Phi1.extent(1), Phi1.extent(2));

    // local references to class members accessed in the parallel region
    const auto l_max = this->l_max;
    const auto num_channels = this->num_channels;
    const auto Phi1_l = this->Phi1_l;
    const auto A1_adj = this->A1_adj;
    const auto A1_weights = this->A1_weights;
    auto dPhi1 = this->dPhi1;

    Kokkos::parallel_for("Reverse A1",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            int lme = 0;
            for (int l=0; l<=l_max; ++l) {
                int num_eta = 0;
                for (int j=0; j<Phi1_l.size(); ++j)
                    num_eta += (Phi1_l(j) == l);
                auto dA1_il = Kokkos::subview(A1_adj, i, Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                auto dPhi1_il = Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
                    &dPhi1(i,lme,0), 2*l+1, num_eta*num_channels);
                KokkosBatched::TeamGemm<Kokkos::TeamPolicy<>::member_type,
                                        KokkosBatched::Trans::NoTranspose,
                                        KokkosBatched::Trans::Transpose,
                                        KokkosBatched::Algo::Gemm::Blocked>
                    ::invoke(team_member, 1.0, dA1_il, A1_weights(l), 0.0, dPhi1_il);
                lme += (2*l+1)*num_eta;
            }

        });
    Kokkos::fence();
}

void MACEKokkos::compute_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (not A1_scaled) return;

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // compute A1 splines
    auto A1_spline_values = Kokkos::View<double**,Kokkos::LayoutRight>("A1_spline_values", r.size(), 1);
    auto A1_spline_derivs = Kokkos::View<double**,Kokkos::LayoutRight>("A1_spline_derivs", r.size(), 1);
    A1_splines.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, A1_spline_values, A1_spline_derivs);
    // perform the scaling
    auto A1 = this->A1;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    Kokkos::parallel_for(
        "MACEKokkos::compute_A1_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            // compute scale factor
            double A1_scale_factor;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A1_spline_values(i0+j,0);
                }, A1_scale_factor);
            A1_scale_factor += 1.0;
            team_member.team_barrier();
            // perform the scaling
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A1(i,lmk/num_channels,lmk%num_channels) /= A1_scale_factor;
                });
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_A1_scaled(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (not A1_scaled) return;

    Kokkos::View<int*> first_neigh("first_neigh", num_nodes);
    Kokkos::parallel_scan("Compute first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });

    // compute A1 splines
    auto A1_spline_values = Kokkos::View<double**,Kokkos::LayoutRight>("A1_spline_values", r.size(), 1);
    auto A1_spline_derivs = Kokkos::View<double**,Kokkos::LayoutRight>("A1_spline_derivs", r.size(), 1);
    A1_splines.evaluate(num_nodes, node_types, num_neigh, neigh_types, r, A1_spline_values, A1_spline_derivs);
    // update the deriviates
    // Warning: Assumes node_forces have been initialized elsewhere
    const auto A1 = this->A1;
    const auto A1_adj = this->A1_adj;
    const auto num_channels = this->num_channels;
    const auto num_lm = this->num_lm;
    auto node_forces = this->node_forces;
    Kokkos::parallel_for(
        "MACEKokkos::reverse_A1_scaled",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            // scale factor
            double A1_scale_factor;// = 1.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j, double& lsum) {
                    lsum += A1_spline_values(i0+j,0);
                }, A1_scale_factor);
            A1_scale_factor += 1.0;
            team_member.team_barrier();
            // update dE/dxyz
            double dA1_dot_A1;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (const int lmk, double& lsum) {
                    const int lm = lmk / num_channels;
                    const int k = lmk % num_channels;
                    lsum += A1_adj(i,lm,k) * A1(i,lm,k);
                }, dA1_dot_A1);
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j) {
                    const int ij = first_neigh(i) + j;
                    const double f = A1_spline_values(ij,0);
                    const double d = A1_spline_derivs(ij,0);
                    node_forces(3*ij+0) += dA1_dot_A1/A1_scale_factor*d*xyz(3*ij+0)/r(ij);
                    node_forces(3*ij+1) += dA1_dot_A1/A1_scale_factor*d*xyz(3*ij+1)/r(ij);
                    node_forces(3*ij+2) += dA1_dot_A1/A1_scale_factor*d*xyz(3*ij+2)/r(ij);
                });
            // update dE/dA1
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_lm*num_channels),
                [=] (int lmk) {
                    A1_adj(i,lmk/num_channels,lmk%num_channels) /= A1_scale_factor;
                });
        });
    Kokkos::fence();
}

void MACEKokkos::compute_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(M1, num_nodes, num_channels);

    auto A1 = this->A1;
    auto M1 = this->M1;
    auto M1_monomials = this->M1_monomials;
    auto M1_weights = this->M1_weights;
    auto num_channels = this->num_channels;

    Kokkos::parallel_for(
        "Compute M1",
        Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Right>>(
            {0,0}, {num_nodes,num_channels}),
        KOKKOS_LAMBDA (const int i, const int k) {
            M1(i,k) = 0.0;
            for (int u=0; u<M1_monomials.extent(0); ++u) {
                double monomial = A1(i,M1_monomials(u,0),k);
                for (int v=1; v<M1_monomials.extent(1); ++v) {
                    if (M1_monomials(u,v) == -1)
                        break;
                    monomial *= A1(i,M1_monomials(u,v),k);
                }
                M1(i,k) += M1_weights(node_types(i),k,u) * monomial;
            }
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_M1(int num_nodes, Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(A1_adj, A1.extent(0), A1.extent(1), A1.extent(2));
    Kokkos::deep_copy(A1_adj, 0.0);

    auto A1 = this->A1;
    auto A1_adj = this->A1_adj;
    auto M1_adj = this->M1_adj;
    auto M1_monomials = this->M1_monomials;
    auto M1_weights = this->M1_weights;
    auto num_channels = this->num_channels;

    Kokkos::parallel_for(
        "Reverse M1",
        Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Right>>(
            {0,0}, {num_nodes,num_channels}),
        KOKKOS_LAMBDA (const int i, const int k) {
            for (int u=0; u<M1_monomials.extent(0); ++u) {
                for (int v=0; v<M1_monomials.extent(1); ++v) {
                    if (M1_monomials(u,v) == -1) break;
                    double deriv = M1_adj(i,k);
                    for (int w=0; w<M1_monomials.extent(1); ++w) {
                        if (M1_monomials(u,w) == -1) break;
                        if (v == w) continue;
                        deriv *= A1(i,M1_monomials(u,w),k);
                    }
                    Kokkos::atomic_add(
                        &A1_adj(i,M1_monomials(u,v),k),
                        M1_weights(node_types(i),k,u)*deriv);
                }
            }
        });
    Kokkos::fence();
}

void MACEKokkos::compute_H2(int num_nodes, Kokkos::View<const int*> node_types)
{
    Kokkos::realloc(H2, num_nodes, num_channels);
    Kokkos::deep_copy(H2, 0.0);

    auto num_channels = this->num_channels;
    auto H2 = this->H2;
    auto H2_weights_for_H1 = this->H2_weights_for_H1;
    auto H1 = this->H1;
    auto H2_weights_for_M1 = this->H2_weights_for_M1;
    auto M1 = this->M1;

    Kokkos::parallel_for(
        "Compute H2 from H1",
        num_nodes*num_channels,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
                for (int kp=0; kp<num_channels; ++kp) {
                    H2(i,k) += H2_weights_for_H1(node_types(i),kp*num_channels+k) * H1(i,0,kp);
                }
        });
    Kokkos::fence();
    Kokkos::parallel_for(
        "Compute H2 from M1",
        num_nodes*num_channels,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
            for (int kp=0; kp<num_channels; ++kp) {
                H2(i,k) += H2_weights_for_M1(kp*num_channels+k) * M1(i,kp);
            }
        });
    Kokkos::fence();
}

void MACEKokkos::reverse_H2(int num_nodes, Kokkos::View<const int*> node_types, bool zero_H1_adj)
{
    Kokkos::resize(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    if (zero_H1_adj)
        Kokkos::deep_copy(H1_adj, 0.0);
    Kokkos::realloc(M1_adj, M1.extent(0), M1.extent(1));
    Kokkos::deep_copy(M1_adj, 0.0);

    auto num_channels = this->num_channels;
    auto H1_adj = this->H1_adj;
    auto M1_adj = this->M1_adj;
    auto H2_adj = this->H2_adj;
    auto H2_weights_for_H1 = this->H2_weights_for_H1;
    auto H2_weights_for_M1 = this->H2_weights_for_M1;

    Kokkos::parallel_for(
        "Reverse H2",
        num_nodes*num_channels,
        KOKKOS_LAMBDA (const int ik) {
            const int i = ik / num_channels;
            const int k = ik % num_channels;
            for (int kp=0; kp<num_channels; ++kp) {
                H1_adj(i,0,k) += H2_weights_for_H1(node_types(i),k*num_channels+kp) * H2_adj(i,kp);
                M1_adj(i,k) +=  H2_weights_for_M1(k*num_channels+kp) * H2_adj(i,kp);
            }
        });
    Kokkos::fence();
}

double MACEKokkos::compute_readouts(int num_nodes, const Kokkos::View<const int*> node_types)
{
    //Kokkos::realloc(node_energies, num_nodes);
    Kokkos::realloc(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
    // Warning: Although it doesn't appear necessary to set H1_adj to zero,
    //          it matters when the number of nodes associated with H1 is greater than num_nodes.
    //          There is probably a better way to manage this.
    Kokkos::deep_copy(H1_adj, 0.0);
    Kokkos::resize(H2_adj, num_nodes, num_channels);

    auto num_channels = this->num_channels;
    auto node_energies = this->node_energies;
    auto atomic_energies = this->atomic_energies;
    auto H1 = this->H1;
    auto H1_adj = this->H1_adj;
    auto readout_1_weights = this->readout_1_weights;
    
    // atomic energies
    Kokkos::parallel_for("Compute Readouts 1", num_nodes, KOKKOS_LAMBDA (const int i) {
        node_energies(i) += atomic_energies(node_types(i));
    });
    Kokkos::fence();
    // first readout
    Kokkos::parallel_for("Compute Readouts 1", num_nodes, KOKKOS_LAMBDA (const int i) {
        for (int k=0; k<num_channels; ++k) {
            node_energies(i) += readout_1_weights(k) * H1(i,0,k);
            H1_adj(i,0,k) = readout_1_weights(k);
        }
    });
    Kokkos::fence();
    // second readout
    Kokkos::View<double*,Kokkos::LayoutRight> readout_2_output("readout_2_output", node_energies.size());
    readout_2.evaluate_gradient(H2, readout_2_output, H2_adj);
    Kokkos::parallel_for("Compute Readouts 2", num_nodes, KOKKOS_LAMBDA (const int i) {
        node_energies(i) += readout_2_output(i);
    });

    double energy;
    Kokkos::parallel_reduce(num_nodes, KOKKOS_LAMBDA (const int& i, double& local_sum) {
        local_sum += node_energies(i);
    }, energy);

    Kokkos::fence();

    return energy;
}

void MACEKokkos::load_from_json(std::string filename)
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
    atomic_numbers = toKokkosView("atomic_numbers", file["atomic_numbers"].get<std::vector<int>>());
    atomic_energies = toKokkosView("atomic_energies", file["atomic_energies"].get<std::vector<double>>());

    // ZBL
    has_zbl = file["has_zbl"].get<bool>();
    if (has_zbl)
        zbl = ZBLKokkos(
            file["zbl_a_exp"].get<double>(),
            file["zbl_a_prefactor"].get<double>(),
            file["zbl_c"].get<std::vector<double>>(),
            file["zbl_covalent_radii"].get<std::vector<double>>(),
            file["zbl_p"].get<int>());

    // Radial splines
    const double spl_h = file["radial_spline_h"];
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    radial_0 = RadialFunctionSetKokkos(spl_h, spl_values_0, spl_derivs_0);
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    radial_1 = RadialFunctionSetKokkos(spl_h, spl_values_1, spl_derivs_1);

    // H0 weights
    set_kokkos_view(
        H0_weights,
        file["H0_weights"].get<std::vector<double>>(),
        atomic_numbers.size(),
        num_channels);

    // A0 weights
    A0_weights = Kokkos::View<double****,Kokkos::LayoutRight>(
        "A0_weights", atomic_numbers.size(), l_max+1, num_channels, num_channels);
    auto h_A0_weights = Kokkos::create_mirror_view(A0_weights);
    auto A0_weights_vec = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int i=0; i<atomic_numbers.size(); ++i)
        for (int l=0; l<=l_max; ++l)
            for (int k1=0; k1<num_channels; ++k1)
                for (int k2=0; k2<num_channels; ++k2)
                    h_A0_weights(i,l,k1,k2) = A0_weights_vec[i][l][k1*num_channels+k2];
    Kokkos::deep_copy(A0_weights,h_A0_weights);

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled) {
        const double A0_spline_h = file["A0_spline_h"];
        auto A0_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A0_spline_values"].get<std::vector<std::vector<double>>>())
            A0_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A0_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A0_spline_derivs"].get<std::vector<std::vector<double>>>())
            A0_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        A0_splines = RadialFunctionSetKokkos(A0_spline_h, A0_spline_values, A0_spline_derivs);
    }

    // M0 weights and monomials
    auto M0_weights_file = file["M0_weights"].get<std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>>>();
    auto M0_monomials_file = file["M0_monomials"].get<std::map<std::string,std::vector<std::vector<int>>>>();
    M0_weights = Kokkos::View<Kokkos::View<double***,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_weights", Kokkos::SequentialHostInit), num_LM);
    M0_monomials = Kokkos::View<Kokkos::View<int**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("M0_monomials", Kokkos::SequentialHostInit), num_LM);
    for (int LM=0; LM<num_LM; ++LM) {
        M0_weights(LM) = Kokkos::View<double***,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_weights_") + std::to_string(LM), Kokkos::WithoutInitializing),
            atomic_numbers.size(), num_channels, M0_monomials_file[std::to_string(LM)].size());
        auto h_M0_weights_LM = Kokkos::create_mirror_view(M0_weights(LM));
        for (int a=0; a<atomic_numbers.size(); ++a)
            for (int k=0; k<num_channels; ++k)
                for (int w=0; w<M0_monomials_file[std::to_string(LM)].size(); ++w)
                    h_M0_weights_LM(a,k,w) = M0_weights_file[std::to_string(a)][std::to_string(LM)][std::to_string(k)][w];
        Kokkos::deep_copy(M0_weights(LM), h_M0_weights_LM);
        M0_monomials(LM) = Kokkos::View<int**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("M0_monomials_") + std::to_string(LM), Kokkos::WithoutInitializing),
            M0_monomials_file[std::to_string(LM)].size(), 3);// TODO: hardcoded 3
        auto h_M0_monomials_LM = Kokkos::create_mirror_view(M0_monomials(LM));
        Kokkos::deep_copy(h_M0_monomials_LM, -1);
        for (int i=0; i<M0_monomials_file[std::to_string(LM)].size(); ++i) {
            for (int j=0; j<M0_monomials_file[std::to_string(LM)][i].size(); ++j) {
                h_M0_monomials_LM(i,j) = M0_monomials_file[std::to_string(LM)][i][j];
            }
        }
        Kokkos::deep_copy(M0_monomials(LM), h_M0_monomials_LM);
    }

    // H1 weights
    set_kokkos_view(
        H1_weights,
        file["H1_weights"].get<std::vector<double>>(),
        L_max+1,
        num_channels,
        num_channels);

    // Phi1
    Phi1_l = toKokkosView("Phi1_l", file["Phi1_l"].get<std::vector<int>>());
    Phi1_l1 = toKokkosView("Phi1_l1", file["Phi1_l1"].get<std::vector<int>>());
    Phi1_l2 = toKokkosView("Phi1_l2", file["Phi1_l2"].get<std::vector<int>>());
    Phi1_lme = toKokkosView("Phi1_lme", file["Phi1_lme"].get<std::vector<int>>());
    Phi1_clebsch_gordan = toKokkosView("Phi1_clebsch_gordan", file["Phi1_clebsch_gordan"].get<std::vector<double>>());
    Phi1_lelm1lm2 = toKokkosView("Phi1_lelm1lm2", file["Phi1_lelm1lm2"].get<std::vector<int>>());
    num_lme = 0;
    auto h_Phi1_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
    for (int i=0; i<h_Phi1_l.size(); ++i)
        num_lme += 2*h_Phi1_l(i)+1;
    num_lelm1lm2 = 0;
    auto h_Phi1_l1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l1);
    auto h_Phi1_l2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l2);
    for (int le=0; le<h_Phi1_l.size(); ++le)
        num_lelm1lm2 += (2*h_Phi1_l1(le)+1)*(2*h_Phi1_l2(le)+1);

    // for new approach to Phi1
    std::vector<int> Phi1_lm1, Phi1_lm2, Phi1_lel1l2;
    int lelm1lm2 = 0;
    for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
        const int l1 = h_Phi1_l1(lel1l2);
        const int l2 = h_Phi1_l2(lel1l2);
        for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
            for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                Phi1_lm1.push_back(lm1);
                Phi1_lm2.push_back(lm2);
                Phi1_lel1l2.push_back(lel1l2);
                lelm1lm2 += 1;
            }
        }
    }
    this->Phi1_lm1 = toKokkosView("Phi1_lm1", Phi1_lm1);
    this->Phi1_lm2 = toKokkosView("Phi1_lm2", Phi1_lm2);
    this->Phi1_lel1l2 = toKokkosView("Phi1_lel1l2", Phi1_lel1l2);

    
    // A1 weights
    auto file_A1_weights = file["A1_weights"].get<std::vector<std::vector<double>>>();
    A1_weights = Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights", Kokkos::SequentialHostInit), l_max+1);
    for (int l=0; l<=l_max; ++l) {
        int num_eta = 0;
        auto h_Phi1_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
        for (int i=0; i<h_Phi1_l.size(); ++i)
            num_eta += (h_Phi1_l(i) == l);
        A1_weights(l) = Kokkos::View<double**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_eta*num_channels, num_channels);
        auto h_A1_weights_l = Kokkos::create_mirror_view(A1_weights(l));
        for (int i=0; i<num_eta*num_channels; ++i)
            for (int j=0; j<num_channels; ++j)
                h_A1_weights_l(i,j) = file_A1_weights[l][i*num_channels+j];
        Kokkos::deep_copy(A1_weights(l), h_A1_weights_l);
    }

    // A1 scaling
    A1_scaled = file["A1_scaled"].get<bool>();
    if (A1_scaled) {
        const double A1_spline_h = file["A1_spline_h"];
        auto A1_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A1_spline_values"].get<std::vector<std::vector<double>>>())
            A1_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A1_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A1_spline_derivs"].get<std::vector<std::vector<double>>>())
            A1_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        A1_splines = RadialFunctionSetKokkos(A1_spline_h, A1_spline_values, A1_spline_derivs);
    }

    // M1 weights and monomials
    auto M1_weights = file["M1_weights"].get<std::map<std::string,std::map<std::string,std::vector<double>>>>();
    const int num_terms = M1_weights[std::to_string(0)][std::to_string(0)].size();
    Kokkos::resize(this->M1_weights, atomic_numbers.size(), num_channels, num_terms);
    auto h_M1_weights = Kokkos::create_mirror_view(this->M1_weights);
    for (int a=0; a<atomic_numbers.size(); ++a)
        for (int k=0; k<num_channels; ++k)
            for (int w=0; w<num_terms; ++w)
                h_M1_weights(a,k,w) = M1_weights[std::to_string(a)][std::to_string(k)][w];
    Kokkos::deep_copy(this->M1_weights, h_M1_weights);
    auto M1_monomials = file["M1_monomials"].get<std::vector<std::vector<int>>>();
    Kokkos::resize(this->M1_monomials, num_terms, 3);// TODO: hardcoded 3
    auto h_M1_monomials = Kokkos::create_mirror_view(this->M1_monomials);
    Kokkos::deep_copy(h_M1_monomials, -1);
    for (int i=0; i<num_terms; ++i)
        for (int j=0; j<M1_monomials[i].size(); ++j)
            h_M1_monomials(i,j) = M1_monomials[i][j];
    Kokkos::deep_copy(this->M1_monomials, h_M1_monomials);

    // H2
    auto H2_weights_for_H1_vec = file["H2_weights_for_H1"].get<std::vector<std::vector<double>>>();
    H2_weights_for_H1 = Kokkos::View<double**,Kokkos::LayoutRight>("H2_weights_for_H2", num_elements, num_channels*num_channels);
    auto h_H2_weights_for_H1 = Kokkos::create_mirror_view(H2_weights_for_H1);
    for (int i=0; i<num_elements; ++i) {
        for (int j=0; j<num_channels*num_channels; ++j) {
            h_H2_weights_for_H1(i,j) = H2_weights_for_H1_vec[i][j];
        }
    }
    Kokkos::deep_copy(H2_weights_for_H1, h_H2_weights_for_H1);
    H2_weights_for_M1 = toKokkosView("H2_weights_for_M1", file["H2_weights_for_M1"].get<std::vector<double>>());

    // Readouts
    // WARNING!
    // hardcoded 16
    readout_1_weights = toKokkosView("readout_1_weights", file["readout_1_weights"].get<std::vector<double>>());
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    readout_2 = MultilayerPerceptronKokkos(
        std::vector<int>{num_channels, 16, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
}
