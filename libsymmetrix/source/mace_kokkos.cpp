#include <fstream>
#include <numbers>
#include <cmath>
#include <stdexcept>

// TODO: remove some of these headers?
#include "KokkosBlas.hpp"
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"

#ifdef SYMMETRIX_SPHERICART_SYCL
#include "sphericart_sycl.hpp"
#endif

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"

using Kokkos::ALL;
using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::MemoryUnmanaged;
using Kokkos::parallel_for;
using Kokkos::PerTeam;
using Kokkos::subview;
using Kokkos::TeamPolicy;
using Kokkos::TeamVectorRange;
using Kokkos::TeamVectorMDRange;
using Kokkos::View;

template <typename Precision>
MACEKokkos<Precision>::MACEKokkos(std::string filename)
{
    load_from_json(filename);
}

template <typename Precision>
MACEKokkos<Precision>::~MACEKokkos()
{
    Kokkos::fence();

    A1_weights = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
    A1_weights_trans = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    compute_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    compute_second_layer(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    compute_readouts(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_second_layer(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
    reverse_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_first_layer(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    // neighbor list
    {
        if (first_neigh.extent(0) < num_nodes)
            Kokkos::realloc(first_neigh, num_nodes);
        auto first_neigh = this->first_neigh;
        int num_edges;
        Kokkos::parallel_scan("MACEKokkos::prepare_neigh_list::first_neigh",
            num_nodes,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final)
                    first_neigh(i) = update;
                update += num_neigh(i);
            }, num_edges);

        if (i_list.extent(0) < num_edges)
            Kokkos::realloc(i_list, num_edges);
        auto i_list = this->i_list;
        Kokkos::parallel_for("MACEKokkos::prepare_neigh_list::i_list",
            num_nodes,
            KOKKOS_LAMBDA (const int i) {
                for (int j=0; j<num_neigh(i); ++j)
                    i_list(first_neigh(i)+j) = i;
            });
    }

    // Y
    // Y_{lm}(\hat{\mathbf{r}}_{ij}), normalized to \sum_m Y_{lm}^2 = 2l+1
    {
#if !defined (SYMMETRIX_SPHERICART_CUDA) && !defined (SYMMETRIX_SPHERICART_SYCL)

        const int num = xyz.extent(0) / 3;
        if (Y.extent(0) < num*num_lm) {
            Kokkos::realloc(Y, num*num_lm);
            Kokkos::realloc(Y_grad, 3*num*num_lm);
        }

        const auto num_lm = this->num_lm;
        auto Y = this->Y;
        auto Y_grad = this->Y_grad;

        if (xyz_sphericart.extent(0) < 3*num)
            Kokkos::realloc(xyz_sphericart, 3*num);
        auto xyz_sphericart = this->xyz_sphericart;
        Kokkos::parallel_for("copy_xyz", 3*num, KOKKOS_LAMBDA (int i) {
            xyz_sphericart(i) = xyz(i);
        });

        // call sphericart on host
        auto h_xyz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xyz_sphericart);
        auto h_Y = Kokkos::create_mirror_view(Y);
        auto h_Y_grad = Kokkos::create_mirror_view(Y_grad);
        using Sphericart = sphericart::SphericalHarmonics<Precision>;
        if (not sphericart_calculator)
            sphericart_calculator = std::make_shared<Sphericart>(l_max);
        static_cast<Sphericart*>(sphericart_calculator.get())->compute_array_with_gradients(
            h_xyz.data(), 3*num, h_Y.data(), 3*num*num_lm, h_Y_grad.data(), 3*num*num_lm);
        Kokkos::deep_copy(Y, h_Y);
        Kokkos::deep_copy(Y_grad, h_Y_grad);

        // normalize to match e3nn conventions
        Kokkos::parallel_for("normalize_Y", num*num_lm, KOKKOS_LAMBDA (int i) {
            Y(i) *= 2*std::sqrt(M_PI);
        });
        Kokkos::parallel_for("normalize_Y_grad", 3*num*num_lm, KOKKOS_LAMBDA (int i) {
            Y_grad(i) *= 2*std::sqrt(M_PI);
        });

#else // SYMMETRIX_SPHERICART_CUDA or SYMMETRIX_SPHERICART_SYCL

        const int num = xyz.extent(0) / 3;
        const int num_lm = (l_max+1)*(l_max+1);
        if (Y.size() < num*num_lm) {
            Kokkos::realloc(Y, num*num_lm);
            Kokkos::realloc(Y_grad, 3*num*num_lm);
        }
        auto Y = this->Y;
        auto Y_grad = this->Y_grad;

        if (xyz_sphericart.extent(0) < 3*num)
            Kokkos::realloc(xyz_sphericart, 3*num);
        auto xyz_sphericart = this->xyz_sphericart;
        Kokkos::parallel_for("copy_xyz", 3*num, KOKKOS_LAMBDA (int i) {
            xyz_sphericart(i) = xyz(i);
        });
        Kokkos::fence();

        // call sphericart
#if defined (SYMMETRIX_SPHERICART_CUDA)
        using Sphericart = sphericart::cuda::SphericalHarmonics<Precision>;
#elif defined (SYMMETRIX_SPHERICART_SYCL)
        using Sphericart = sphericart::sycl::SphericalHarmonics<Precision>;
#endif
        if (not sphericart_calculator)
            sphericart_calculator = std::make_shared<Sphericart>(l_max);
        static_cast<Sphericart*>(sphericart_calculator.get())->compute_with_gradients(
            xyz_sphericart.data(), num, Y.data(), Y_grad.data());
        Kokkos::fence();

        // normalize to match e3nn conventions
        const double normalization_factor = 2 * std::sqrt(M_PI);
        Kokkos::parallel_for("normalize_Y", num*num_lm, KOKKOS_LAMBDA (int i) {
            Y(i) *= normalization_factor;
        });
        Kokkos::parallel_for("normalize_Y_grad", 3*num*num_lm, KOKKOS_LAMBDA (int i) {
            Y_grad(i) *= normalization_factor;
        });

#endif
    }

    // R0
    // \tilde{R}^{(0)}_{a_ia_j,kl}(r_{ij}) and \partial_r \tilde{R}^{(0)}_{a_ia_j,kl}(r_{ij}),  \tilde{R}^{(0)}_{a_ia_j,kl} = \sum_{k'} w_{a_ikl,k'} R^{(0)}_{a_ia_j,k'l} H^{(0)}_{a_j,k'}
    {
        if (r.size() > R0.extent(0)) {
            Kokkos::realloc(R0, r.size(), (l_max+1)*num_channels);
            Kokkos::realloc(R0_deriv, r.size(), (l_max+1)*num_channels);
        }

        const auto i_list = this->i_list;
        const int l_max = this->l_max;
        const int num_channels = this->num_channels;
        const auto num_types = atomic_numbers.size();
        const auto h = R0_spline_h;
        const auto r_min = R0_spline_r_min;
        const auto c = R0_spline_coefficients;
        const int num_intervals = c.extent(1);
        check_spline_bounds(r, r_min, r_cut, "MACEKokkos");
        auto R0 = this->R0;
        auto R0_deriv = this->R0_deriv;

        Kokkos::parallel_for(
            "Compute R0",
            Kokkos::TeamPolicy<>(r.size(), Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int ij = team_member.league_rank();
                const int type_i = node_types(i_list(ij));
                const int type_j = neigh_types(ij);
                const int type_ij = type_i*num_types+type_j;
                // compute x, x^2, x^3
                const int n = Kokkos::clamp(static_cast<int>((r(ij) - r_min)/h), 0, num_intervals - 1);
                const double x = r(ij) - (r_min + h*n);
                const double xx = x*x;
                const double xxx = xx*x;
                const double two_x = 2*x;
                const double three_xx = 3*xx;
                // compute function values
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team_member, (l_max+1)*num_channels),
                    [&] (const int lk) {
                        const double c0 = c(type_ij,n,0,lk);
                        const double c1 = c(type_ij,n,1,lk);
                        const double c2 = c(type_ij,n,2,lk);
                        const double c3 = c(type_ij,n,3,lk);
                        R0(ij,lk) = c0 + c1*x + c2*xx + c3*xxx;
                        R0_deriv(ij,lk) = c1 + c2*two_x + c3*three_xx;
                    });
            });
    }

    // A0
    // A^{(0)}_{i,klm} = \sum_j \tilde{R}^{(0)}_{a_ia_j,kl}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})
    {
        if (A0.extent(0) != num_nodes)
            Kokkos::realloc(A0, num_nodes, num_lm, num_channels);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), A0, 0.0);

        const auto first_neigh = this->first_neigh;
        const int num_lm = this->num_lm;
        const int num_channels = this->num_channels;
        const auto R0 = this->R0;
        const auto Y = this->Y;
        auto A0 = this->A0;

        parallel_for("Compute A0",
            TeamPolicy<>(num_nodes*num_lm, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank() / num_lm;
                const int lm = team_member.league_rank() % num_lm;
                const int l = Kokkos::sqrt(lm);
                for (int j=0; j<num_neigh(i); ++j) {
                    const int ij = first_neigh(i) + j;
                    const double Y_ij_lm = Y(ij*num_lm+lm);
                    parallel_for(
                        TeamVectorRange(team_member, num_channels),
                        [=] (const int k) {
                            A0(i,lm,k) += R0(ij,l*num_channels+k) * Y_ij_lm;
                        });
                }
            });
    }

    // A0 scaling
    // A^{(0)}_{i,klm} \leftarrow A^{(0)}_{i,klm} / s^{(0)}_i,  s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij})
    if (A0_scaled) {
        // compute A0 splines
        if (r.size() > A0_spline_values.extent(0)) {
            Kokkos::realloc(A0_spline_values, r.size(), 1);
            Kokkos::realloc(A0_spline_derivs, r.size(), 1);
        }
        A0_splines.evaluate(node_types, i_list, neigh_types, r, A0_spline_values, A0_spline_derivs);

        // perform the scaling
        const auto first_neigh = this->first_neigh;
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
    }

    // M0
    // M^{(0)}_{i,klm} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_iklm,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(0)}_{i,kl'_\xi m'_\xi}
    {
        if (M0.extent(0) < num_nodes)
            Kokkos::realloc(M0, num_nodes, num_LM, num_channels);
        if (B0.extent(0) < num_nodes)
            Kokkos::realloc(B0, num_nodes, B0.extent(1), num_channels);
        Kokkos::deep_copy(M0, 0.0);

        const auto A0 = this->A0;
        const auto B0_spec = this->B0_spec;
        const auto B0_coefficients = this->B0_coefficients;
        const auto M0_spec = this->M0_spec;
        const auto M0_weights = this->M0_weights;
        const auto B0 = this->B0;
        const auto num_channels = this->num_channels;
        const auto num_lm = this->num_lm;
        auto M0 = this->M0;

        Kokkos::parallel_for("Compute M0",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                    for (int q=0; q<B0.extent(1); ++q)
                        B0(i,q,k) = (q < num_lm) ? A0(i,q,k) : 0.0;
                });
                for (int t=0; t<B0_spec.extent(0); ++t)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        B0(i,B0_spec(t,0),k) +=
                            B0_coefficients(t) * B0(i,B0_spec(t,1),k) * B0(i,B0_spec(t,2),k);
                    });
                for (int e=0; e<M0_spec.extent(0); ++e)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        for (int m=0; m<M0_spec(e,2); ++m)
                            M0(i,M0_spec(e,0)+m,k) += M0_weights(node_types(i),e,k) * B0(i,M0_spec(e,1)+m,k);
                    });
            });
    }

    // H1
    // H^{(1)}_{i,klm} = \sum_{k'} w_{kl,k'} M^{(0)}_{i,k'lm}
    {
        if (H1.extent(0) < M0.extent(0))
            Kokkos::realloc(H1, M0.extent(0), M0.extent(1), M0.extent(2));


        const auto M0 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->M0.data(), num_nodes, num_LM*num_channels);
        const auto H1 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->H1.data(), num_nodes, num_LM*num_channels);
        for (int l=0; l<=L_max and num_nodes>0; ++l)
            for (int m=0; m<2*l+1; ++m)
                KokkosBlas::gemm("N", "N", 1.0,
                    Kokkos::subview(M0, Kokkos::ALL, Kokkos::make_pair((l*l+m)*num_channels, (l*l+m+1)*num_channels)),
                    Kokkos::subview(H1_weights, l, Kokkos::ALL, Kokkos::ALL), 0.0,
                    Kokkos::subview(H1, Kokkos::ALL, Kokkos::make_pair((l*l+m)*num_channels, (l*l+m+1)*num_channels)));
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_second_layer(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    // R1
    // R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) and \partial_r R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij})
    {
        if (r.size() > R1.extent(0)) {
            Kokkos::realloc(R1, r.size(), Phi1_lel1l2_spec.extent(0)*num_channels);
            Kokkos::realloc(R1_deriv, r.size(), Phi1_lel1l2_spec.extent(0)*num_channels);
        }
        radial_1.evaluate(node_types, i_list, neigh_types, r, R1, R1_deriv);
    }

    // Phi1
    // \Phi^{(1)}_{i,klm\eta} = \sum_j \sum_{l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) H^{(1)}_{j,kl_2m_2}
    {
        if (Phi1.extent(0) < num_nodes)
            Kokkos::realloc(Phi1, num_nodes, num_lme, num_channels);

        const auto first_neigh = this->first_neigh;
        const auto num_channels = this->num_channels;
        const auto num_lm = this->num_lm;
        const int num_lel1l2 = Phi1_lel1l2_spec.extent(0);
        const auto Phi1_lel1l2_spec = this->Phi1_lel1l2_spec;
        const auto Phi1_cg = this->Phi1_cg;
        const auto R1 = this->R1;
        const auto Y = this->Y;
        const auto H1 = this->H1;
        auto Phi1 = this->Phi1;
        const int l_max = this->l_max;
        const int L_max = this->L_max;
        const int batch_size = 32;
        using Scratch1 = Kokkos::View<Precision*,Kokkos::MemoryUnmanaged>;
        using Scratch2 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;

        Kokkos::parallel_for("Compute Phi1",
            Kokkos::TeamPolicy<>(num_nodes*num_lel1l2, Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(Scratch1::shmem_size(batch_size*(2*L_max+1)*(2*mace_l_max+1)) + Scratch2::shmem_size(2*l_max+1,num_channels))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank() / num_lel1l2;
                const int lel1l2 = team_member.league_rank() % num_lel1l2;
                const int l = Phi1_lel1l2_spec(lel1l2,0);
                const int l1 = Phi1_lel1l2_spec(lel1l2,1);
                const int l2 = Phi1_lel1l2_spec(lel1l2,2);
                const int lme = Phi1_lel1l2_spec(lel1l2,3);
                const int eta = Phi1_lel1l2_spec(lel1l2,4);
                const int num_eta = Phi1_lel1l2_spec(lel1l2,5);
                const int num_m = 2*l+1;
                const int num_m1 = 2*l1+1;
                const int num_m2 = 2*l2+1;
                const Scratch1 CG_Y(team_member.team_scratch(0), batch_size*(2*L_max+1)*(2*mace_l_max+1));
                const Scratch2 Phi1_i_lel1l2(team_member.team_scratch(0), 2*l_max+1, num_channels);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_m*num_channels), [&] (const int mk) {
                    Phi1_i_lel1l2(mk/num_channels, mk%num_channels) = 0.0;
                });
                for (int j0=0; j0<num_neigh(i); j0+=batch_size) {
                    const int ij0 = first_neigh(i) + j0;
                    const int num_j = Kokkos::min(batch_size, num_neigh(i)-j0);
                    team_member.team_barrier();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_j*num_m*num_m2), [&] (const int jmm2) {
                        const int j = jmm2/(num_m*num_m2);
                        const int m = jmm2/num_m2%num_m;
                        const int m2 = jmm2%num_m2;
                        Precision t = 0.0;
                        for (int m1=0; m1<num_m1; ++m1)
                            t += Phi1_cg(lel1l2,m,m1,m2) * Y((ij0+j)*num_lm+l1*l1+m1);
                        CG_Y((j*num_m2+m2)*(2*mace_l_max+1)+m) = t;
                    });
                    team_member.team_barrier();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_channels), [&] (const int k) {
                        Precision a[2*mace_l_max+1] = {};
                        for (int j=0; j<num_j; ++j) {
                            const int ij = ij0 + j;
                            const Precision R = R1(ij,lel1l2*num_channels+k);
                            for (int m2=0; m2<num_m2; ++m2) {
                                const Precision h = R * H1(neigh_indices(ij),l2*l2+m2,k);
                                const int jm2 = (j*num_m2+m2)*(2*mace_l_max+1);
                                for (int m=0; m<2*mace_l_max+1; ++m)
                                    if (m < num_m)
                                        a[m] += CG_Y(jm2+m) * h;
                            }
                        }
                        for (int m=0; m<2*mace_l_max+1; ++m)
                            if (m < num_m)
                                Phi1_i_lel1l2(m,k) += a[m];
                    });
                }
                team_member.team_barrier();
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_channels), [&] (const int k) {
                    for (int m=0; m<num_m; ++m)
                        Phi1(i,lme+m*num_eta+eta,k) = Phi1_i_lel1l2(m,k);
                });
            });
    }

    // A1
    // A^{(1)}_{i,klm} = \sum_{k'\eta'} w_{kl,k'\eta'} \Phi^{(1)}_{i,k'lm\eta'}
    {
        // The core matrix multiplication is:
        //         [A1_il]_mk = \sum_(ek') [Phi1_il]_m(ek') [W_il]_(ek')k
        if (A1.extent(0) < num_nodes)
            Kokkos::realloc(A1, num_nodes, num_lm, num_channels);

        const auto Phi1 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->Phi1.data(), num_nodes, num_lme*num_channels);
        const auto A1 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->A1.data(), num_nodes, num_lm*num_channels);
        int lme = 0;
        for (int l=0; l<=l_max and num_nodes>0; ++l) {
            const int num_eta = A1_weights(l).extent(0)/num_channels;
            for (int m=0; m<2*l+1; ++m)
                KokkosBlas::gemm("N", "N", 1.0,
                    Kokkos::subview(Phi1, Kokkos::ALL, Kokkos::make_pair((lme+m*num_eta)*num_channels, (lme+(m+1)*num_eta)*num_channels)),
                    A1_weights(l), 0.0,
                    Kokkos::subview(A1, Kokkos::ALL, Kokkos::make_pair((l*l+m)*num_channels, (l*l+m+1)*num_channels)));
            lme += (2*l+1)*num_eta;
        }
    }

    // A1 scaling
    // A^{(1)}_{i,klm} \leftarrow A^{(1)}_{i,klm} / s^{(1)}_i,  s^{(1)}_i = 1 + \sum_j \rho^{(1)}_{a_ia_j}(r_{ij})
    if (A1_scaled) {
        // compute A1 splines
        if (A1_spline_values.extent(0) < r.size()) {
            Kokkos::realloc(A1_spline_values, r.size(), 1);
            Kokkos::realloc(A1_spline_derivs, r.size(), 1);
        }
        A1_splines.evaluate(node_types, i_list, neigh_types, r, A1_spline_values, A1_spline_derivs);

        // perform the scaling
        const auto first_neigh = this->first_neigh;
        auto A1 = this->A1;
        auto A1_spline_values = this->A1_spline_values;
        auto A1_spline_derivs = this->A1_spline_derivs;
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
    }

    // M1
    // M^{(1)}_{i,k} = \sum_{\nu\mathbf{l}'\mathbf{m}'} w_{a_ik,\nu\mathbf{l}'\mathbf{m}'} \prod_{\xi=1}^{\nu} A^{(1)}_{i,kl'_\xi m'_\xi}
    {
        if (M1.extent(0) < num_nodes)
            Kokkos::realloc(M1, num_nodes, num_channels);
        if (B1.extent(0) < num_nodes)
            Kokkos::realloc(B1, num_nodes, B1.extent(1), num_channels);
        Kokkos::deep_copy(M1, 0.0);

        const auto A1 = this->A1;
        const auto B1_spec = this->B1_spec;
        const auto B1_coefficients = this->B1_coefficients;
        const auto M1_spec = this->M1_spec;
        const auto M1_weights = this->M1_weights;
        const auto B1 = this->B1;
        const auto num_channels = this->num_channels;
        const auto num_lm = this->num_lm;
        auto M1 = this->M1;

        Kokkos::parallel_for("Compute M1",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                    for (int q=0; q<B1.extent(1); ++q)
                        B1(i,q,k) = (q < num_lm) ? A1(i,q,k) : 0.0;
                });
                for (int t=0; t<B1_spec.extent(0); ++t)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        B1(i,B1_spec(t,0),k) +=
                            B1_coefficients(t) * B1(i,B1_spec(t,1),k) * B1(i,B1_spec(t,2),k);
                    });
                for (int e=0; e<M1_spec.extent(0); ++e)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        M1(i,k) += M1_weights(node_types(i),e,k) * B1(i,M1_spec(e,1),k);
                    });
            });
    }

    // H2
    // H^{(2)}_{i,k} = \sum_{k'} w_{k,k'} M^{(1)}_{i,k'} + \sum_{k'} w_{a_ik,k'} M^{(0)}_{i,k'00}
    {
        if (H2.extent(0) < num_nodes or H2.extent(1) != num_channels)
            Kokkos::realloc(H2, num_nodes, num_channels);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), H2, 0.0);

        auto num_channels = this->num_channels;
        auto H2 = this->H2;
        auto H2_weights_for_M0 = this->H2_weights_for_M0;
        auto M0 = this->M0;
        auto H2_weights_for_M1 = this->H2_weights_for_M1;
        auto M1 = this->M1;

        Kokkos::parallel_for(
            "Compute H2 from M0",
            num_nodes*num_channels,
            KOKKOS_LAMBDA (const int ik) {
                const int i = ik / num_channels;
                const int k = ik % num_channels;
                    for (int kp=0; kp<num_channels; ++kp) {
                        H2(i,k) += H2_weights_for_M0(node_types(i),kp*num_channels+k) * M0(i,0,kp);
                    }
            });
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
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_readouts(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (node_energies.size() < num_nodes)
        Kokkos::realloc(node_energies, num_nodes);
    if (node_forces.size() < xyz.size())
        Kokkos::realloc(node_forces, xyz.size());
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), node_energies, 0.0);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), node_forces, 0.0);

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
        if (M0_adj.extent(0) < M0.extent(0))
            Kokkos::realloc(M0_adj, M0.extent(0), M0.extent(1), M0.extent(2));
        if (H2_adj.extent(0) < num_nodes or H2_adj.extent(1) != num_channels)
            Kokkos::realloc(H2_adj, num_nodes, num_channels);
        if (readout_2_output.extent(0) < num_nodes)
            Kokkos::realloc(readout_2_output, num_nodes);

        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), M0_adj, 0.0);

        auto num_channels = this->num_channels;
        auto node_energies = this->node_energies;
        auto atomic_energies = this->atomic_energies;
        auto M0 = this->M0;
        auto M0_adj = this->M0_adj;
        auto readout_1_weights_for_M0 = this->readout_1_weights_for_M0;

        // atomic energies
        Kokkos::parallel_for("Compute Readouts 1", num_nodes, KOKKOS_LAMBDA (const int i) {
            node_energies(i) += atomic_energies(node_types(i));
        });
        // first readout
        Kokkos::parallel_for("Compute Readouts 1", num_nodes, KOKKOS_LAMBDA (const int i) {
            for (int k=0; k<num_channels; ++k) {
                node_energies(i) += readout_1_weights_for_M0(k) * M0(i,0,k);
                M0_adj(i,0,k) = readout_1_weights_for_M0(k);
            }
        });
        Kokkos::fence();
        // second readout
        auto H2 = Kokkos::subview(this->H2, make_pair(0,num_nodes), Kokkos::ALL);
        auto readout_2_output = Kokkos::subview(this->readout_2_output, make_pair(0,num_nodes));
        auto H2_adj = Kokkos::subview(this->H2_adj, make_pair(0,num_nodes), Kokkos::ALL);
        readout_2.evaluate_gradient(H2, readout_2_output, H2_adj);
        Kokkos::parallel_for("Compute Readouts 2", num_nodes, KOKKOS_LAMBDA (const int i) {
            node_energies(i) += readout_2_output(i);
        });
    }
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_second_layer(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    // H2
    // \bar{M}^{(1)}_{i,k'} = \sum_k w_{k,k'} \bar{H}^{(2)}_{i,k},  \bar{M}^{(0)}_{i,k'00} += \sum_k w_{a_ik,k'} \bar{H}^{(2)}_{i,k}
    {
        if (M0_adj.extent(0) < M0.extent(0))
            Kokkos::resize(M0_adj, M0.extent(0), M0.extent(1), M0.extent(2));
        if (M1_adj.extent(0) < M1.extent(0))
            Kokkos::realloc(M1_adj, M1.extent(0), M1.extent(1));

        auto num_channels = this->num_channels;
        auto M0_adj = this->M0_adj;
        auto M1_adj = this->M1_adj;
        auto H2_adj = this->H2_adj;
        auto H2_weights_for_M0_trans = this->H2_weights_for_M0_trans;
        auto H2_weights_for_M1_trans = this->H2_weights_for_M1_trans;

        Kokkos::parallel_for(
            "Reverse H2",
            num_nodes*num_channels,
            KOKKOS_LAMBDA (const int ik) {
                const int i = ik / num_channels;
                const int k = ik % num_channels;
                double M0_adj_ik = 0;
                double M1_adj_ik = 0;
                for (int kp=0; kp<num_channels; ++kp) {
                    M0_adj_ik += H2_weights_for_M0_trans(node_types(i),kp*num_channels+k) * H2_adj(i,kp);
                    M1_adj_ik += H2_weights_for_M1_trans(kp*num_channels+k) * H2_adj(i,kp);
                }
                M0_adj(i,0,k) += M0_adj_ik;
                M1_adj(i,k) = M1_adj_ik;
            });
    }

    // M1
    // \bar{A}^{(1)}_{i,klm} = \bar{M}^{(1)}_{i,k} \partial M^{(1)}_{i,k} / \partial A^{(1)}_{i,klm}
    {
        if (A1_adj.extent(0) < num_nodes)
            Kokkos::realloc(A1_adj, A1.extent(0), A1.extent(1), A1.extent(2));
        if (B1_adj.extent(0) < num_nodes)
            Kokkos::realloc(B1_adj, num_nodes, B1.extent(1), num_channels);

        const auto M1_adj = this->M1_adj;
        const auto B1_spec = this->B1_spec;
        const auto B1_coefficients = this->B1_coefficients;
        const auto M1_spec = this->M1_spec;
        const auto M1_weights = this->M1_weights;
        const auto B1 = this->B1;
        const auto B1_adj = this->B1_adj;
        const auto num_channels = this->num_channels;
        const auto num_lm = this->num_lm;
        auto A1_adj = this->A1_adj;

        Kokkos::parallel_for("Reverse M1",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                    for (int q=0; q<B1_adj.extent(1); ++q)
                        B1_adj(i,q,k) = 0.0;
                });
                for (int e=0; e<M1_spec.extent(0); ++e)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        B1_adj(i,M1_spec(e,1),k) += M1_weights(node_types(i),e,k) * M1_adj(i,k);
                    });
                for (int t=B1_spec.extent(0)-1; t>=0; --t)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        const Precision g = B1_coefficients(t) * B1_adj(i,B1_spec(t,0),k);
                        B1_adj(i,B1_spec(t,1),k) += g * B1(i,B1_spec(t,2),k);
                        B1_adj(i,B1_spec(t,2),k) += g * B1(i,B1_spec(t,1),k);
                    });
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                    for (int lm=0; lm<num_lm; ++lm)
                        A1_adj(i,lm,k) = B1_adj(i,lm,k);
                });
            });
    }

    // A1 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} \sum_{klm} \bar{A}^{(1)}_{i,klm} A^{(1)}_{i,klm} / s^{(1)}_i
    // \bar{A}^{(1)}_{i,klm} \leftarrow \bar{A}^{(1)}_{i,klm} / s^{(1)}_i
    if (A1_scaled) {
        // update the derivatives
        const auto first_neigh = this->first_neigh;
        const auto A1 = this->A1;
        const auto A1_adj = this->A1_adj;
        const auto A1_spline_values = this->A1_spline_values;
        const auto A1_spline_derivs = this->A1_spline_derivs;
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
    }

    // A1
    // \bar{\Phi}^{(1)}_{i,k'lm\eta'} = \sum_k w_{kl,k'\eta'} \bar{A}^{(1)}_{i,klm}
    {
        // The core matrix multiplication is:
        //         [dE/dPhi1_il]_m(ek) = \sum_k' [dE/dA1_il]_mk' [trans(W_il)]_k'(ek)
        if (dPhi1.extent(0) < num_nodes)
            Kokkos::realloc(dPhi1, num_nodes, num_lme, num_channels);

        const auto A1_adj = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->A1_adj.data(), num_nodes, num_lm*num_channels);
        const auto dPhi1 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->dPhi1.data(), num_nodes, num_lme*num_channels);
        int lme = 0;
        for (int l=0; l<=l_max and num_nodes>0; ++l) {
            const int num_eta = A1_weights_trans(l).extent(1)/num_channels;
            for (int m=0; m<2*l+1; ++m)
                KokkosBlas::gemm("N", "N", 1.0,
                    Kokkos::subview(A1_adj, Kokkos::ALL, Kokkos::make_pair((l*l+m)*num_channels, (l*l+m+1)*num_channels)),
                    A1_weights_trans(l), 0.0,
                    Kokkos::subview(dPhi1, Kokkos::ALL, Kokkos::make_pair((lme+m*num_eta)*num_channels, (lme+(m+1)*num_eta)*num_channels)));
            lme += (2*l+1)*num_eta;
        }
    }

    // Phi1
    // \bar{H}^{(1)}_{j,kl_2m_2} = \sum_i \sum_{lm\eta l_1m_1} C^{lm\eta}_{l_1m_1l_2m_2} R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(1)}_{i,klm\eta}
    // \mathbf{F}_{ij} -= \sum_{klm\eta l_1m_1l_2m_2} C^{lm\eta}_{l_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(1)}_{a_ia_j,kl\eta l_1l_2}(r_{ij}) Y_{l_1m_1}(\hat{\mathbf{r}}_{ij})] H^{(1)}_{j,kl_2m_2} \bar{\Phi}^{(1)}_{i,klm\eta}
    {
        if (H1_adj.extent(0) < H1.extent(0))
            Kokkos::realloc(H1_adj, H1.extent(0), H1.extent(1), H1.extent(2));
        // Warning: Although it doesn't appear necessary to set H1_adj to zero,
        //          it matters when the number of nodes associated with H1 is greater than num_nodes.
        //          There is probably a better way to manage this.
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), H1_adj, 0.0);

        const auto first_neigh = this->first_neigh;
        const auto num_lm = this->num_lm;
        const auto num_channels = this->num_channels;
        const int num_lel1l2 = Phi1_lel1l2_spec.extent(0);
        const auto Phi1_lel1l2_spec = this->Phi1_lel1l2_spec;
        const auto Phi1_cg = this->Phi1_cg;
        const auto R1 = this->R1;
        const auto R1_deriv = this->R1_deriv;
        const auto Y = this->Y;
        const auto Y_grad = this->Y_grad;
        const auto H1 = this->H1;
        const auto H1_adj = this->H1_adj;
        const auto dPhi1 = this->dPhi1;
        const auto node_forces = this->node_forces;
        const int L_max = this->L_max;
        const int batch_size = 16;
        using Scratch1 = Kokkos::View<Precision*,Kokkos::MemoryUnmanaged>;
        using Scratch2 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;

        Kokkos::parallel_for("Reverse Phi1",
            Kokkos::TeamPolicy<>(num_nodes*num_lel1l2, Kokkos::AUTO, 32)
                .set_scratch_size(0, Kokkos::PerTeam(Scratch1::shmem_size(batch_size*(2*L_max+1)*(2*mace_l_max+1))
                    + Scratch2::shmem_size(3,batch_size*(2*L_max+1)*(2*mace_l_max+1)))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank() / num_lel1l2;
                const int lel1l2 = team_member.league_rank() % num_lel1l2;
                const int l = Phi1_lel1l2_spec(lel1l2,0);
                const int l1 = Phi1_lel1l2_spec(lel1l2,1);
                const int l2 = Phi1_lel1l2_spec(lel1l2,2);
                const int lme = Phi1_lel1l2_spec(lel1l2,3);
                const int eta = Phi1_lel1l2_spec(lel1l2,4);
                const int num_eta = Phi1_lel1l2_spec(lel1l2,5);
                const int num_m = 2*l+1;
                const int num_m1 = 2*l1+1;
                const int num_m2 = 2*l2+1;
                const Scratch1 CG_Y(team_member.team_scratch(0), batch_size*(2*L_max+1)*(2*mace_l_max+1));
                const Scratch2 CG_Y_grad(team_member.team_scratch(0), 3, batch_size*(2*L_max+1)*(2*mace_l_max+1));
                for (int j0=0; j0<num_neigh(i); j0+=batch_size) {
                    const int ij0 = first_neigh(i) + j0;
                    const int num_j = Kokkos::min(batch_size, num_neigh(i)-j0);
                    team_member.team_barrier();
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_j*num_m*num_m2), [&] (const int jmm2) {
                        const int j = jmm2/(num_m*num_m2);
                        const int ij = ij0 + j;
                        const int m = jmm2/num_m2%num_m;
                        const int m2 = jmm2%num_m2;
                        Precision t = 0.0;
                        Precision t_x = 0.0;
                        Precision t_y = 0.0;
                        Precision t_z = 0.0;
                        for (int m1=0; m1<num_m1; ++m1) {
                            const Precision c = Phi1_cg(lel1l2,m,m1,m2);
                            t += c * Y(ij*num_lm+l1*l1+m1);
                            t_x += c * Y_grad(3*ij*num_lm+l1*l1+m1);
                            t_y += c * Y_grad((3*ij+1)*num_lm+l1*l1+m1);
                            t_z += c * Y_grad((3*ij+2)*num_lm+l1*l1+m1);
                        }
                        CG_Y((j*num_m2+m2)*(2*mace_l_max+1)+m) = t;
                        CG_Y_grad(0,(j*num_m2+m2)*(2*mace_l_max+1)+m) = t_x;
                        CG_Y_grad(1,(j*num_m2+m2)*(2*mace_l_max+1)+m) = t_y;
                        CG_Y_grad(2,(j*num_m2+m2)*(2*mace_l_max+1)+m) = t_z;
                    });
                    team_member.team_barrier();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_j), [&] (const int j) {
                        const int ij = ij0 + j;
                        const int neigh_index = neigh_indices(ij);
                        Precision f_r;
                        Precision f_x;
                        Precision f_y;
                        Precision f_z;
                        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, num_channels),
                            [&] (const int k, Precision& f_r, Precision& f_x, Precision& f_y, Precision& f_z) {
                                const Precision R = R1(ij,lel1l2*num_channels+k);
                                const Precision R_deriv = R1_deriv(ij,lel1l2*num_channels+k);
                                Precision adj[2*mace_l_max+1];
                                for (int m=0; m<2*mace_l_max+1; ++m)
                                    if (m < num_m)
                                        adj[m] = dPhi1(i,lme+m*num_eta+eta,k);
                                Precision w = 0.0;
                                Precision w_x = 0.0;
                                Precision w_y = 0.0;
                                Precision w_z = 0.0;
                                for (int m2=0; m2<num_m2; ++m2) {
                                    const int jm2 = (j*num_m2+m2)*(2*mace_l_max+1);
                                    Precision g = 0.0;
                                    Precision g_x = 0.0;
                                    Precision g_y = 0.0;
                                    Precision g_z = 0.0;
                                    for (int m=0; m<2*mace_l_max+1; ++m)
                                        if (m < num_m) {
                                            g += CG_Y(jm2+m) * adj[m];
                                            g_x += CG_Y_grad(0,jm2+m) * adj[m];
                                            g_y += CG_Y_grad(1,jm2+m) * adj[m];
                                            g_z += CG_Y_grad(2,jm2+m) * adj[m];
                                        }
                                    const Precision h = H1(neigh_index,l2*l2+m2,k);
                                    Kokkos::atomic_add(&H1_adj(neigh_index,l2*l2+m2,k), R * g);
                                    w += g * h;
                                    w_x += g_x * h;
                                    w_y += g_y * h;
                                    w_z += g_z * h;
                                }
                                f_r += w * R_deriv;
                                f_x += w_x * R;
                                f_y += w_y * R;
                                f_z += w_z * R;
                            }, f_r, f_x, f_y, f_z);
                        Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                            Kokkos::atomic_add(&node_forces(3*ij), -(f_r*xyz(3*ij)/r(ij) + f_x));
                            Kokkos::atomic_add(&node_forces(3*ij+1), -(f_r*xyz(3*ij+1)/r(ij) + f_y));
                            Kokkos::atomic_add(&node_forces(3*ij+2), -(f_r*xyz(3*ij+2)/r(ij) + f_z));
                        });
                    });
                }
            });
    }
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_first_layer(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    // H1
    // \bar{M}^{(0)}_{i,k'lm} += \sum_k w_{kl,k'} \bar{H}^{(1)}_{i,klm}
    {
        if (M0_adj.extent(0) < M0.extent(0))
            Kokkos::resize(M0_adj, M0.extent(0), M0.extent(1), M0.extent(2));


        const auto H1_adj = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->H1_adj.data(), num_nodes, num_LM*num_channels);
        const auto M0_adj = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>(
            this->M0_adj.data(), num_nodes, num_LM*num_channels);
        for (int l=0; l<=L_max and num_nodes>0; ++l)
            for (int m=0; m<2*l+1; ++m)
                KokkosBlas::gemm("N", "N", 1.0,
                    Kokkos::subview(H1_adj, Kokkos::ALL, Kokkos::make_pair((l*l+m)*num_channels, (l*l+m+1)*num_channels)),
                    Kokkos::subview(H1_weights_trans, l, Kokkos::ALL, Kokkos::ALL), l == 0 ? 1.0 : 0.0,
                    Kokkos::subview(M0_adj, Kokkos::ALL, Kokkos::make_pair((l*l+m)*num_channels, (l*l+m+1)*num_channels)));
    }

    // M0
    // \bar{A}^{(0)}_{i,klm} = \sum_{l'm'} \bar{M}^{(0)}_{i,kl'm'} \partial M^{(0)}_{i,kl'm'} / \partial A^{(0)}_{i,klm}
    {
        if (A0_adj.extent(0) < num_nodes)
            Kokkos::realloc(A0_adj, A0.extent(0), A0.extent(1), A0.extent(2));
        if (B0_adj.extent(0) < num_nodes)
            Kokkos::realloc(B0_adj, num_nodes, B0.extent(1), num_channels);

        const auto M0_adj = this->M0_adj;
        const auto B0_spec = this->B0_spec;
        const auto B0_coefficients = this->B0_coefficients;
        const auto M0_spec = this->M0_spec;
        const auto M0_weights = this->M0_weights;
        const auto B0 = this->B0;
        const auto B0_adj = this->B0_adj;
        const auto num_channels = this->num_channels;
        const auto num_lm = this->num_lm;
        auto A0_adj = this->A0_adj;

        Kokkos::parallel_for("Reverse M0",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                    for (int q=0; q<B0_adj.extent(1); ++q)
                        B0_adj(i,q,k) = 0.0;
                });
                for (int e=0; e<M0_spec.extent(0); ++e)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        for (int m=0; m<M0_spec(e,2); ++m)
                            B0_adj(i,M0_spec(e,1)+m,k) += M0_weights(node_types(i),e,k) * M0_adj(i,M0_spec(e,0)+m,k);
                    });
                for (int t=B0_spec.extent(0)-1; t>=0; --t)
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                        const Precision g = B0_coefficients(t) * B0_adj(i,B0_spec(t,0),k);
                        B0_adj(i,B0_spec(t,1),k) += g * B0(i,B0_spec(t,2),k);
                        B0_adj(i,B0_spec(t,2),k) += g * B0(i,B0_spec(t,1),k);
                    });
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_channels), [&] (const int k) {
                    for (int lm=0; lm<num_lm; ++lm)
                        A0_adj(i,lm,k) = B0_adj(i,lm,k);
                });
            });
    }

    // A0 scaling
    // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} \sum_{klm} \bar{A}^{(0)}_{i,klm} A^{(0)}_{i,klm} / s^{(0)}_i
    // \bar{A}^{(0)}_{i,klm} \leftarrow \bar{A}^{(0)}_{i,klm} / s^{(0)}_i
    if (A0_scaled) {
        // update the derivatives
        const auto first_neigh = this->first_neigh;
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
    }

    // A0
    // \mathbf{F}_{ij} -= \sum_{klm} \nabla_{\mathbf{r}_{ij}}[\tilde{R}^{(0)}_{a_ia_j,kl}(r_{ij}) Y_{lm}(\hat{\mathbf{r}}_{ij})] \bar{A}^{(0)}_{i,klm}
    {
        auto A0_adj = this->A0_adj;
        auto num_lm = this->num_lm;
        auto num_channels = this->num_channels;
        auto l_max = this->l_max;
        auto R0 = this->R0;
        auto R0_deriv = this->R0_deriv;
        auto Y = this->Y;
        auto Y_grad = this->Y_grad;
        auto node_forces = this->node_forces;
        const auto first_neigh = this->first_neigh;

        Kokkos::parallel_for("Reverse A0",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_neigh(i)), [&] (const int j) {
                    const int ij = first_neigh(i) + j;
                    double f_r, f_x, f_y, f_z;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, num_channels),
                        [&] (const int k, double& f_r, double& f_x, double& f_y, double& f_z) {
                            for (int l=0; l<=l_max; ++l) {
                                const Precision R = R0(ij,l*num_channels+k);
                                const Precision R_deriv = R0_deriv(ij,l*num_channels+k);
                                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                                    const double a = A0_adj(i,lm,k);
                                    f_r += R_deriv * a * Y(ij*num_lm+lm);
                                    f_x += R * a * Y_grad(3*ij*num_lm+lm);
                                    f_y += R * a * Y_grad((3*ij+1)*num_lm+lm);
                                    f_z += R * a * Y_grad((3*ij+2)*num_lm+lm);
                                }
                            }
                        }, f_r, f_x, f_y, f_z);
                    Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                        node_forces(3*ij) -= f_r*xyz(3*ij)/r(ij) + f_x;
                        node_forces(3*ij+1) -= f_r*xyz(3*ij+1)/r(ij) + f_y;
                        node_forces(3*ij+2) -= f_r*xyz(3*ij+2)/r(ij) + f_z;
                    });
                });
            });
    }
}

template <typename Precision>
void MACEKokkos<Precision>::load_from_json(std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);
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

    // R0
    const double spl_r_min = file.at("radial_spline_r_min");
    auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
    if (not std::isfinite(spl_r_min) or not std::isfinite(r_cut) or spl_r_min >= r_cut)
        throw std::invalid_argument("Radial splines require finite r_min < r_max.");
    if (spl_values_0.empty() or spl_values_0.size() != atomic_numbers.size()*(atomic_numbers.size()+1)/2
            or spl_values_0.size() != spl_derivs_0.size())
        throw std::invalid_argument("R0 splines require matching arrays for every edge type.");
    for (int a=0; a<spl_values_0.size(); ++a) {
        if (spl_values_0[a].size() != (l_max+1)*num_channels or spl_values_0[a].size() != spl_derivs_0[a].size())
            throw std::invalid_argument("R0 splines require consistent function counts.");
        for (int j=0; j<spl_values_0[a].size(); ++j)
            if (spl_values_0[a][j].size() < 2 or spl_values_0[a][j].size() != spl_values_0[0][0].size()
                    or spl_values_0[a][j].size() != spl_derivs_0[a][j].size())
                throw std::invalid_argument("R0 splines require consistent nodal arrays with at least two values.");
    }
    const double spl_h = (r_cut - spl_r_min) / (spl_values_0[0][0].size()-1);
    if (spl_h <= 0 or not std::isfinite(spl_h))
        throw std::invalid_argument("R0 splines require positive finite spacing.");
    auto c = Kokkos::View<Precision****,Kokkos::LayoutRight>(
        "c", atomic_numbers.size()*atomic_numbers.size(), spl_values_0[0][0].size()-1, 4, (l_max+1)*num_channels);
    auto h_c = Kokkos::create_mirror_view(c);
    for (int a=0; a<atomic_numbers.size(); ++a) {
        for (int b=0; b<atomic_numbers.size(); ++b) {
            const int ab = a*atomic_numbers.size()+b;
            const int ab_unordered = (a <= b)
                ? a*(2*atomic_numbers.size()-a-1)/2 + b
                : b*(2*atomic_numbers.size()-b-1)/2 + a;
            auto spl_values = spl_values_0[ab_unordered];
            auto spl_derivs = spl_derivs_0[ab_unordered];
            for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                for (int lk=0; lk<(l_max+1)*num_channels; ++lk) {
                    h_c(ab,i,0,lk) = spl_values[lk][i];
                    h_c(ab,i,1,lk) = spl_derivs[lk][i];
                    h_c(ab,i,2,lk) = (-3*spl_values[lk][i] -2*spl_h*spl_derivs[lk][i]
                                        + 3*spl_values[lk][i+1] - spl_h*spl_derivs[lk][i+1]) / (spl_h*spl_h);
                    h_c(ab,i,3,lk) = (2*spl_values[lk][i] + spl_h*spl_derivs[lk][i]
                                        - 2*spl_values[lk][i+1] + spl_h*spl_derivs[lk][i+1]) / (spl_h*spl_h*spl_h);
                }
            }
            // add H0_weights
            auto H0_weights = file["H0_weights"].get<std::vector<double>>();
            for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                for (int lk=0; lk<(l_max+1)*num_channels; ++lk) {
                    const int k = lk % num_channels;
                    h_c(ab,i,0,lk) *= H0_weights[b*num_channels+k];
                    h_c(ab,i,1,lk) *= H0_weights[b*num_channels+k];
                    h_c(ab,i,2,lk) *= H0_weights[b*num_channels+k];
                    h_c(ab,i,3,lk) *= H0_weights[b*num_channels+k];
                }
            }
            // add A0_weights
            auto A0_weights = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();
            for (int i=0; i<spl_values_0[0][0].size()-1; ++i) {
                for (int l=0; l<=l_max; ++l) {
                    auto c0 = std::vector<double>(&h_c(ab,i,0,l*num_channels), &h_c(ab,i,0,l*num_channels)+num_channels);
                    auto c1 = std::vector<double>(&h_c(ab,i,1,l*num_channels), &h_c(ab,i,1,l*num_channels)+num_channels);
                    auto c2 = std::vector<double>(&h_c(ab,i,2,l*num_channels), &h_c(ab,i,2,l*num_channels)+num_channels);
                    auto c3 = std::vector<double>(&h_c(ab,i,3,l*num_channels), &h_c(ab,i,3,l*num_channels)+num_channels);
                    for (int k=0; k<num_channels; ++k) {
                        h_c(ab,i,0,l*num_channels+k) = 0.0;
                        h_c(ab,i,1,l*num_channels+k) = 0.0;
                        h_c(ab,i,2,l*num_channels+k) = 0.0;
                        h_c(ab,i,3,l*num_channels+k) = 0.0;
                        for (int kp=0; kp<num_channels; ++kp) {
                            h_c(ab,i,0,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c0[kp];
                            h_c(ab,i,1,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c1[kp];
                            h_c(ab,i,2,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c2[kp];
                            h_c(ab,i,3,l*num_channels+k) += A0_weights[a][l][kp*num_channels+k]*c3[kp];
                        }
                    }
                }
            }
        }
    }
    Kokkos::deep_copy(c, h_c);
    R0_spline_r_min = spl_r_min;
    R0_spline_h = spl_h;
    R0_spline_coefficients = c;

    // R1
    auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
    radial_1 = RadialFunctionSetKokkos<Precision>(spl_r_min, r_cut, spl_values_1, spl_derivs_1);

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled) {
        const double A0_spline_r_min = file.at("A0_spline_r_min");
        auto A0_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A0_spline_values"].get<std::vector<std::vector<double>>>())
            A0_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A0_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A0_spline_derivs"].get<std::vector<std::vector<double>>>())
            A0_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        A0_splines = RadialFunctionSetKokkos<double>(A0_spline_r_min, r_cut, A0_spline_values, A0_spline_derivs);
    }

    auto load_contraction = [&] (std::string l, auto& B_spec, auto& B_coefficients, auto& M_spec, auto& M_weights,
                                 auto& B) {
        const auto file_B_spec = file["B"+l+"_spec"].get<std::vector<std::array<int,3>>>();
        const auto file_B_coefficients = file["B"+l+"_coefficients"].get<std::vector<double>>();
        const auto file_M_spec = file["M"+l+"_spec"].get<std::vector<std::array<int,3>>>();
        const auto file_M_weights = file["M"+l+"_weights"].get<std::vector<std::vector<std::vector<double>>>>();
        Kokkos::realloc(B_spec, file_B_spec.size(), 3);
        Kokkos::realloc(B_coefficients, file_B_spec.size());
        Kokkos::realloc(M_spec, file_M_spec.size(), 3);
        Kokkos::realloc(M_weights, file_M_weights.size(), file_M_spec.size(), num_channels);
        auto h_B_spec = Kokkos::create_mirror_view(B_spec);
        auto h_B_coefficients = Kokkos::create_mirror_view(B_coefficients);
        auto h_M_spec = Kokkos::create_mirror_view(M_spec);
        auto h_M_weights = Kokkos::create_mirror_view(M_weights);
        int num_B = num_lm;
        for (int t=0; t<file_B_spec.size(); ++t) {
            for (int j=0; j<3; ++j)
                h_B_spec(t,j) = file_B_spec[t][j];
            h_B_coefficients(t) = file_B_coefficients[t];
            num_B = std::max(num_B, file_B_spec[t][0]+1);
        }
        for (int e=0; e<file_M_spec.size(); ++e) {
            for (int j=0; j<3; ++j)
                h_M_spec(e,j) = file_M_spec[e][j];
            for (int a=0; a<file_M_weights.size(); ++a)
                for (int k=0; k<num_channels; ++k)
                    h_M_weights(a,e,k) = file_M_weights[a][e][k];
        }
        Kokkos::deep_copy(B_spec, h_B_spec);
        Kokkos::deep_copy(B_coefficients, h_B_coefficients);
        Kokkos::deep_copy(M_spec, h_M_spec);
        Kokkos::deep_copy(M_weights, h_M_weights);
        Kokkos::realloc(B, 0, num_B, num_channels);
    };
    load_contraction("0", B0_spec, B0_coefficients, M0_spec, M0_weights, B0);

    // H1 weights
    set_kokkos_view(
        H1_weights,
        file["H1_weights"].get<std::vector<Precision>>(),
        L_max+1,
        num_channels,
        num_channels);
    H1_weights_trans = Kokkos::View<Precision***,Kokkos::LayoutRight>("H1_weights_trans", L_max+1, num_channels, num_channels);
    auto h_H1_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_weights);
    auto h_H1_weights_trans = Kokkos::create_mirror_view(H1_weights_trans);
    for (int l=0; l<=L_max; ++l)
        for (int i=0; i<num_channels; ++i)
            for (int j=0; j<num_channels; ++j)
                h_H1_weights_trans(l,j,i) = h_H1_weights(l,i,j);
    Kokkos::deep_copy(H1_weights_trans, h_H1_weights_trans);

    // Phi1
    if (l_max > mace_l_max)
        throw std::invalid_argument("MACEKokkos requires l_max <= " + std::to_string(mace_l_max) + ".");
    const auto Phi1_l = file["Phi1_l"].get<std::vector<int>>();
    const auto Phi1_spec = file["Phi1_spec"].get<std::vector<std::array<int,3>>>();
    const auto Phi1_cg_spec = file["Phi1_cg_spec"].get<std::vector<std::array<int,2>>>();
    const auto Phi1_clebsch_gordan = file["Phi1_clebsch_gordan"].get<std::vector<double>>();
    Kokkos::realloc(Phi1_lel1l2_spec, Phi1_l.size(), 6);
    Kokkos::realloc(Phi1_cg, Phi1_l.size(), 2*l_max+1, 2*l_max+1, 2*L_max+1);
    auto h_Phi1_lel1l2_spec = Kokkos::create_mirror_view(Phi1_lel1l2_spec);
    auto h_Phi1_cg = Kokkos::create_mirror_view(Phi1_cg);
    num_lme = 0;
    for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
        int lme = 0;
        int eta = 0;
        int num_eta = 0;
        for (int other=0; other<Phi1_l.size(); ++other) {
            if (Phi1_l[other] < Phi1_l[lel1l2])
                lme += 2*Phi1_l[other]+1;
            if (Phi1_l[other] == Phi1_l[lel1l2] and other < lel1l2)
                eta += 1;
            if (Phi1_l[other] == Phi1_l[lel1l2])
                num_eta += 1;
        }
        h_Phi1_lel1l2_spec(lel1l2,0) = Phi1_l[lel1l2];
        h_Phi1_lel1l2_spec(lel1l2,3) = lme;
        h_Phi1_lel1l2_spec(lel1l2,4) = eta;
        h_Phi1_lel1l2_spec(lel1l2,5) = num_eta;
        num_lme += 2*Phi1_l[lel1l2]+1;
    }
    for (int t=0; t<Phi1_cg_spec.size(); ++t) {
        const auto [q, lm1] = Phi1_cg_spec[t];
        const auto [lme, lel1l2, lm2] = Phi1_spec[q];
        const int l1 = std::sqrt(lm1);
        const int l2 = std::sqrt(lm2);
        h_Phi1_lel1l2_spec(lel1l2,1) = l1;
        h_Phi1_lel1l2_spec(lel1l2,2) = l2;
        h_Phi1_cg(lel1l2, (lme-h_Phi1_lel1l2_spec(lel1l2,3)-h_Phi1_lel1l2_spec(lel1l2,4))/h_Phi1_lel1l2_spec(lel1l2,5), lm1-l1*l1, lm2-l2*l2) += Phi1_clebsch_gordan[t];
    }
    Kokkos::deep_copy(Phi1_lel1l2_spec, h_Phi1_lel1l2_spec);
    Kokkos::deep_copy(Phi1_cg, h_Phi1_cg);

    // A1 weights
    auto file_A1_weights = file["A1_weights"].get<std::vector<std::vector<Precision>>>();
    A1_weights = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights", Kokkos::SequentialHostInit), l_max+1);
    A1_weights_trans = Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace>(
        Kokkos::view_alloc("A1_weights_trans", Kokkos::SequentialHostInit), l_max+1);
    for (int l=0; l<=l_max; ++l) {
        int num_eta = 0;
        for (int i=0; i<Phi1_l.size(); ++i)
            num_eta += (Phi1_l[i] == l);
        A1_weights(l) = Kokkos::View<Precision**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_eta*num_channels, num_channels);
        A1_weights_trans(l) = Kokkos::View<Precision**,Kokkos::LayoutRight>(
            Kokkos::view_alloc(std::string("A1_weights_") + std::to_string(l), Kokkos::WithoutInitializing),
            num_channels, num_eta*num_channels);
        auto h_A1_weights_l = Kokkos::create_mirror_view(A1_weights(l));
        auto h_A1_weights_trans_l = Kokkos::create_mirror_view(A1_weights_trans(l));
        for (int i=0; i<num_eta*num_channels; ++i) {
            for (int j=0; j<num_channels; ++j) {
                h_A1_weights_l(i,j) = file_A1_weights[l][i*num_channels+j];
                h_A1_weights_trans_l(j,i) = file_A1_weights[l][i*num_channels+j];
            }
        }
        Kokkos::deep_copy(A1_weights(l), h_A1_weights_l);
        Kokkos::deep_copy(A1_weights_trans(l), h_A1_weights_trans_l);
    }

    // A1 scaling
    A1_scaled = file["A1_scaled"].get<bool>();
    if (A1_scaled) {
        const double A1_spline_r_min = file.at("A1_spline_r_min");
        auto A1_spline_values = std::vector<std::vector<std::vector<double>>>();
        for (auto& values : file["A1_spline_values"].get<std::vector<std::vector<double>>>())
            A1_spline_values.push_back({values});  // adds dimension to reach 3d
        auto A1_spline_derivs = std::vector<std::vector<std::vector<double>>>();
        for (auto& derivs : file["A1_spline_derivs"].get<std::vector<std::vector<double>>>())
            A1_spline_derivs.push_back({derivs});  // adds dimension to reach 3d
        A1_splines = RadialFunctionSetKokkos<double>(A1_spline_r_min, r_cut, A1_spline_values, A1_spline_derivs);
    }

    load_contraction("1", B1_spec, B1_coefficients, M1_spec, M1_weights, B1);

    // H2
    auto H2_weights_for_M0_vec = file["H2_weights_for_M0"].get<std::vector<std::vector<double>>>();
    H2_weights_for_M0 = Kokkos::View<double**,Kokkos::LayoutRight>("H2_weights_for_M0", num_elements, num_channels*num_channels);
    H2_weights_for_M0_trans = Kokkos::View<double**,Kokkos::LayoutRight>("H2_weights_for_M0_trans", num_elements, num_channels*num_channels);
    auto h_H2_weights_for_M0 = Kokkos::create_mirror_view(H2_weights_for_M0);
    auto h_H2_weights_for_M0_trans = Kokkos::create_mirror_view(H2_weights_for_M0_trans);
    for (int i=0; i<num_elements; ++i)
        for (int k=0; k<num_channels; ++k)
            for (int kp=0; kp<num_channels; ++kp) {
                h_H2_weights_for_M0(i,k*num_channels+kp) = H2_weights_for_M0_vec[i][k*num_channels+kp];
                h_H2_weights_for_M0_trans(i,kp*num_channels+k) = H2_weights_for_M0_vec[i][k*num_channels+kp];
            }
    Kokkos::deep_copy(H2_weights_for_M0, h_H2_weights_for_M0);
    Kokkos::deep_copy(H2_weights_for_M0_trans, h_H2_weights_for_M0_trans);
    auto H2_weights_for_M1_vec = file["H2_weights_for_M1"].get<std::vector<double>>();
    auto H2_weights_for_M1_trans_vec = H2_weights_for_M1_vec;
    for (int k=0; k<num_channels; ++k)
        for (int kp=0; kp<num_channels; ++kp)
            H2_weights_for_M1_trans_vec[kp*num_channels+k] = H2_weights_for_M1_vec[k*num_channels+kp];
    H2_weights_for_M1 = toKokkosView("H2_weights_for_M1", H2_weights_for_M1_vec);
    H2_weights_for_M1_trans = toKokkosView("H2_weights_for_M1_trans", H2_weights_for_M1_trans_vec);

    // Readouts
    // WARNING!
    // hardcoded 16
    readout_1_weights_for_M0 = toKokkosView("readout_1_weights_for_M0", file["readout_1_weights_for_M0"].get<std::vector<double>>());
    auto readout_2_weights_1 = file["readout_2_weights_1"].get<std::vector<double>>();
    auto readout_2_weights_2 = file["readout_2_weights_2"].get<std::vector<double>>();
    readout_2 = MultilayerPerceptronKokkos(
        std::vector<int>{num_channels, 16, 1},
        std::vector<std::vector<double>>{readout_2_weights_1, readout_2_weights_2},
        file["readout_2_scale_factor"]);
}

template class MACEKokkos<float>;
template class MACEKokkos<double>;
