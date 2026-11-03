#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "KokkosBlas.hpp"
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"

#ifdef SYMMETRIX_SPHERICART_SYCL
#include "sphericart_sycl.hpp"
#endif

#include "tools_kokkos.hpp"
#include "nequip_kokkos.hpp"

using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::subview;
using Kokkos::View;

template <typename Precision>
NequIPKokkos<Precision>::NequIPKokkos(std::string filename)
{
    load_from_json(filename);
}

template <typename Precision>
NequIPKokkos<Precision>::~NequIPKokkos()
{
    Kokkos::fence();
}

template <typename Precision>
void NequIPKokkos<Precision>::compute_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
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
void NequIPKokkos<Precision>::compute_layer(
    const int t,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    const int H_width = H_offset[t].back();
    const int Phi_width = Phi_offset[t].back();
    const int A_width = A_offset[t].back();

    // neighbor list
    if (t == 0) {
        if (first_neigh.extent(0) < num_nodes)
            Kokkos::realloc(first_neigh, num_nodes);
        auto first_neigh = this->first_neigh;
        Kokkos::parallel_scan("NequIPKokkos::first_neigh",
            num_nodes,
            KOKKOS_LAMBDA (const int i, int& update, const bool final) {
                if (final)
                    first_neigh(i) = update;
                update += num_neigh(i);
            });
        check_spline_bounds(r, std::numeric_limits<double>::min(), r_cut, "NequIPKokkos");
    }

    // Y
    // Y_{lm}(\hat{\mathbf{r}}_{ij})
    if (t == 0) {
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
        if (num > 0)
            static_cast<Sphericart*>(sphericart_calculator.get())->compute_with_gradients(
                xyz_sphericart.data(), num, Y.data(), Y_grad.data());
        Kokkos::fence();

#endif
    }

    // Phi
    // \Phi^{(t)}_{i,k\lambda m\eta} = \sum_j \sum_{m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) X^{(t)}_{j,k\lambda_1m_1},  \eta = (\lambda_1, l_2)
    {
        if (Phi.extent(0) < num_nodes or Phi.extent(1) < Phi_width)
            Kokkos::realloc(Phi, num_nodes, Phi_width);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), Phi, 0.0);

        const auto first_neigh = this->first_neigh;
        const int num_lm = this->num_lm;
        const auto radial = this->radial[t];
        const auto Phi_spec = this->Phi_spec[t];
        const auto Phi_rows = this->Phi_rows[t];
        const auto Phi_columns = this->Phi_columns[t];
        const auto Phi_clebsch_gordan = this->Phi_clebsch_gordan[t];
        const auto Y = this->Y;
        const auto X = this->X[t];
        auto Phi = this->Phi;

        const int B = std::min(int(Kokkos::TeamPolicy<>::scratch_size_max(0)/sizeof(Precision))/(H_width+num_lm), 16);
        const int num_e_max = (B > 0) ? B : 16;
        using Scratch2 = Kokkos::View<Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        Kokkos::parallel_for("Compute Phi",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32)
                .set_scratch_size(0, Kokkos::PerTeam(Scratch2::shmem_size(B,H_width) + Scratch2::shmem_size(B,num_lm))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                const Scratch2 X_ij(team_member.team_scratch(0), B, H_width);
                const Scratch2 Y_ij(team_member.team_scratch(0), B, num_lm);
                for (int e0=0; e0<num_neigh(i); e0+=num_e_max) {
                    const int ij0 = first_neigh(i) + e0;
                    const int num_e = Kokkos::min(num_e_max, num_neigh(i)-e0);
                    if (B > 0) {
                        team_member.team_barrier();
                        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_e*H_width), [&] (const int x) {
                            const int ij = ij0 + x/H_width;
                            X_ij(x/H_width,x%H_width) = X((t > 0) ? neigh_indices(ij) : neigh_types(ij), x%H_width);
                        });
                        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_e*num_lm), [&] (const int x) {
                            Y_ij(x/num_lm,x%num_lm) = Y(ij0*num_lm+x);
                        });
                        team_member.team_barrier();
                    }
                    Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, Phi_width), [&] (const int b) {
                        const int q = Phi_columns(b);
                        if (q < 0)
                            return;
                        const int p0 = Phi_rows(q);
                        const int p1 = Phi_rows(q+1);
                        const int k = b - Phi_spec(p0,3);
                        const int w = Phi_spec(p0,2) + k;
                        Precision phi = 0.0;
                        for (int e=0; e<num_e; ++e) {
                            const int ij = ij0 + e;
                            const Precision* X_e = (B > 0) ? &X_ij(e,0) : &X((t > 0) ? neigh_indices(ij) : neigh_types(ij), 0);
                            const Precision* Y_e = (B > 0) ? &Y_ij(e,0) : &Y(ij*num_lm);
                            const int n = radial.interval(r(ij));
                            const Precision R = radial.value(0, n, radial.offset(r(ij), n), w);
                            Precision s = 0.0;
                            for (int p=p0; p<p1; ++p)
                                s += Phi_clebsch_gordan(p) * Y_e[Phi_spec(p,1)] * X_e[Phi_spec(p,0)+k];
                            phi += R * s;
                        }
                        Phi(i,b) += phi;
                    });
                }
            });
    }

    // A
    // A^{(t)}_{i,k\lambda m} = \sum_{k'\eta} w^{(t)}_{k\lambda,k'\eta} \Phi^{(t)}_{i,k'\lambda m\eta} + \sum_{k'} w^{(t)}_{a_ik\lambda,k'} H^{(t)}_{i,k'\lambda m}
    {
        if (A[t].extent(0) < num_nodes)
            Kokkos::realloc(A[t], num_nodes, A_width);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), A[t], 0.0);
        for (int ir=0; ir<A_mul[t].size() and num_nodes>0; ++ir)
            for (int m=0; m<2*(ir/2)+1 and A_weights[t][ir].size()>0; ++m)
                KokkosBlas::gemm("N", "N", 1.0,
                    subview(Phi, make_pair(0,num_nodes),
                            make_pair(Phi_offset[t][ir]+m*Phi_mul[t][ir], Phi_offset[t][ir]+(m+1)*Phi_mul[t][ir])),
                    A_weights[t][ir], 0.0,
                    subview(A[t], make_pair(0,num_nodes),
                            make_pair(A_offset[t][ir]+m*A_mul[t][ir], A_offset[t][ir]+(m+1)*A_mul[t][ir])));
        for (int ir=0; ir<A_mul[t].size(); ++ir) {
            if (A_sc_weights[t][ir].size() == 0)
                continue;
            const int l = ir/2;
            const auto W = A_sc_weights[t][ir];
            const int H_off = H_offset[t][ir];
            const int H_m = H_mul[t][ir];
            const int A_off = A_offset[t][ir];
            const int A_m = A_mul[t][ir];
            const auto H = this->H[t];
            auto A = this->A[t];
            Kokkos::parallel_for("Compute A (self-connection)",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    const int h = (t > 0) ? i : node_types(i);
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 2*l+1), [&] (const int m) {
                        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, A_m), [&] (const int v) {
                            Precision s = 0.0;
                            for (int u=0; u<H_m; ++u)
                                s += H(h,H_off+m*H_m+u) * W(node_types(i),u,v);
                            A(i,A_off+m*A_m+v) += s;
                        });
                    });
                });
        }
    }

    // gate
    // H^{(t+1)}_{i,k(0e)} = c_e \mathrm{SiLU}(A^{(t)}_{i,k(0e)}),  H^{(t+1)}_{i,k(0o)} = c_o \tanh(A^{(t)}_{i,k(0o)})
    // H^{(t+1)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) A^{(t)}_{i,k\lambda m} (l > 0),  g_{k\lambda} = N^{(t+1)}_{0e} + \sum_{0o < \lambda' < \lambda} N^{(t+1)}_{\lambda'} + k
    {
        const int width = H_offset[t+1].back();
        if (H[t+1].extent(0) < num_nodes)
            Kokkos::realloc(H[t+1], num_nodes, width);
        const Precision gate_cst = this->gate_cst;
        const Precision gate_cst_odd = this->gate_cst_odd;
        const int A_odd = A_offset[t][1];
        const int H_odd = H_offset[t+1][1];
        const auto A = this->A[t];
        auto H = this->H[t+1];
        Kokkos::parallel_for("Compute H (scalars)",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,H_mul[t+1][0]}),
            KOKKOS_LAMBDA (const int i, const int k) {
                const Precision e = Kokkos::exp(-Kokkos::abs(A(i,k)));
                H(i,k) = gate_cst * A(i,k) * ((A(i,k) >= 0) ? 1 : e) / (1 + e);
            });
        Kokkos::parallel_for("Compute H (odd scalars)",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,H_mul[t+1][1]}),
            KOKKOS_LAMBDA (const int i, const int k) {
                H(i,H_odd+k) = gate_cst_odd * Kokkos::tanh(A(i,A_odd+k));
            });
        int g = H_mul[t+1][0];
        for (int ir=2; ir<A_mul[t].size(); ++ir) {
            const int l = ir/2;
            const int mul = A_mul[t][ir];
            const int A_off = A_offset[t][ir];
            const int H_off = H_offset[t+1][ir];
            Kokkos::parallel_for("Compute H (gates)",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,mul}),
                KOKKOS_LAMBDA (const int i, const int k) {
                    const Precision e = Kokkos::exp(-Kokkos::abs(A(i,g+k)));
                    const Precision gate = gate_cst * A(i,g+k) * ((A(i,g+k) >= 0) ? 1 : e) / (1 + e);
                    for (int m=0; m<2*l+1; ++m)
                        H(i,H_off+m*mul+k) = gate * A(i,A_off+m*mul+k);
                });
            g += mul;
        }
    }

    // X
    // X^{(t+1)}_{i,k\lambda m} = \sum_{k'} w^{(t+1)}_{k\lambda,k'} H^{(t+1)}_{i,k'\lambda m}
    if (t+1 < num_layers) {
        const int width = H_offset[t+1].back();
        if (X[t+1].extent(0) < num_nodes)
            Kokkos::realloc(X[t+1], num_nodes, width);
        for (int ir=0; ir<H_mul[t+1].size() and num_nodes>0; ++ir)
            for (int m=0; m<2*(ir/2)+1 and H_mul[t+1][ir]>0; ++m) {
                const auto cols = make_pair(H_offset[t+1][ir]+m*H_mul[t+1][ir], H_offset[t+1][ir]+(m+1)*H_mul[t+1][ir]);
                KokkosBlas::gemm("N", "N", 1.0, subview(H[t+1], make_pair(0,num_nodes), cols), X_weights[t+1][ir],
                    0.0, subview(X[t+1], make_pair(0,num_nodes), cols));
            }
    }
}

template <typename Precision>
void NequIPKokkos<Precision>::compute_readout(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (node_energies.size() < num_nodes)
        Kokkos::realloc(node_energies, num_nodes);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), node_energies, 0.0);
    if (node_forces.size() < xyz.size())
        Kokkos::realloc(node_forces, xyz.size());
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), node_forces, 0.0);

    // ZBL
    if (has_zbl) {
        const auto first_neigh = this->first_neigh;
        const auto atomic_numbers = this->atomic_numbers;
        const double zbl_prefactor = this->zbl_prefactor;
        const double p = zbl_p;
        const double r_cut = this->r_cut;
        auto node_energies = this->node_energies;
        auto node_forces = this->node_forces;
        Kokkos::parallel_for("Compute ZBL", num_nodes, KOKKOS_LAMBDA (const int i) {
            const double Z_i = atomic_numbers(node_types(i));
            for (int ij=first_neigh(i); ij<first_neigh(i)+num_neigh(i); ++ij) {
                const double Z_j = atomic_numbers(neigh_types(ij));
                const double d = (Kokkos::pow(Z_i, 0.23) + Kokkos::pow(Z_j, 0.23)) / 0.46850;
                const double x = d * r(ij);
                const double e_1 = 0.02817 * Kokkos::exp(-0.20162*x);
                const double e_2 = 0.28022 * Kokkos::exp(-0.40290*x);
                const double e_3 = 0.50986 * Kokkos::exp(-0.94229*x);
                const double e_4 = 0.18175 * Kokkos::exp(-3.19980*x);
                const double psi = e_1 + e_2 + e_3 + e_4;
                const double psi_deriv = -d * (0.20162*e_1 + 0.40290*e_2 + 0.94229*e_3 + 3.19980*e_4);
                const double u = r(ij) / r_cut;
                double f_c = 0.0;
                double f_c_deriv = 0.0;
                if (u < 1.0) {
                    f_c = 1.0 - (p+1.0)*(p+2.0)/2.0*Kokkos::pow(u, p) + p*(p+2.0)*Kokkos::pow(u, p+1.0)
                          - p*(p+1.0)/2.0*Kokkos::pow(u, p+2.0);
                    f_c_deriv = (-(p+1.0)*(p+2.0)/2.0*p*Kokkos::pow(u, p-1.0) + p*(p+2.0)*(p+1.0)*Kokkos::pow(u, p)
                                 - p*(p+1.0)/2.0*(p+2.0)*Kokkos::pow(u, p+1.0)) / r_cut;
                }
                const double v = zbl_prefactor * Z_i * Z_j;
                node_energies(i) += v / r(ij) * psi * f_c;
                const double dE_dr = v * (-psi*f_c/(r(ij)*r(ij)) + psi_deriv*f_c/r(ij) + psi*f_c_deriv/r(ij));
                node_forces(3*ij) -= dE_dr * xyz(3*ij) / r(ij);
                node_forces(3*ij+1) -= dE_dr * xyz(3*ij+1) / r(ij);
                node_forces(3*ij+2) -= dE_dr * xyz(3*ij+2) / r(ij);
            }
        });
    }

    // readout
    {
        const int width = H_offset[num_layers].back();
        if (H_adj.extent(0) < num_nodes or H_adj.extent(1) < width)
            Kokkos::realloc(H_adj, num_nodes, width);
        const auto H = this->H[num_layers];
        const auto readout_weights = this->readout_weights;
        const auto per_type_scale = this->per_type_scale;
        const auto per_type_shift = this->per_type_shift;
        auto H_adj = this->H_adj;
        auto node_energies = this->node_energies;
        Kokkos::parallel_for("Compute Readout", num_nodes, KOKKOS_LAMBDA (const int i) {
            const double scale = per_type_scale(node_types(i));
            double e = per_type_shift(node_types(i));
            for (int k=0; k<width; ++k) {
                e += scale * readout_weights(k) * H(i,k);
                H_adj(i,k) = scale * readout_weights(k);
            }
            node_energies(i) += e;
        });
    }
}

template <typename Precision>
void NequIPKokkos<Precision>::reverse_layer(
    const int t,
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    const int H_width = H_offset[t].back();
    const int Phi_width = Phi_offset[t].back();
    const int A_width = A_offset[t].back();

    // X
    // \bar{H}^{(t+1)}_{i,k'\lambda m} += \sum_k w^{(t+1)}_{k\lambda,k'} \bar{X}^{(t+1)}_{i,k\lambda m}
    if (t+1 < num_layers) {
        for (int ir=0; ir<H_mul[t+1].size() and num_nodes>0; ++ir)
            for (int m=0; m<2*(ir/2)+1 and H_mul[t+1][ir]>0; ++m) {
                const auto cols = make_pair(H_offset[t+1][ir]+m*H_mul[t+1][ir], H_offset[t+1][ir]+(m+1)*H_mul[t+1][ir]);
                KokkosBlas::gemm("N", "N", 1.0, subview(X_adj, make_pair(0,num_nodes), cols), X_weights_trans[t+1][ir],
                    1.0, subview(H_adj, make_pair(0,num_nodes), cols));
            }
    }

    // gate
    // \bar{A}^{(t)}_{i,k(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,k(0e)}) \bar{H}^{(t+1)}_{i,k(0e)},  \bar{A}^{(t)}_{i,k(0o)} = c_o \tanh'(A^{(t)}_{i,k(0o)}) \bar{H}^{(t+1)}_{i,k(0o)}
    // \bar{A}^{(t)}_{i,k\lambda m} = c_e \mathrm{SiLU}(A^{(t)}_{i,g_{k\lambda}(0e)}) \bar{H}^{(t+1)}_{i,k\lambda m},  \bar{A}^{(t)}_{i,g_{k\lambda}(0e)} = c_e \mathrm{SiLU}'(A^{(t)}_{i,g_{k\lambda}(0e)}) \sum_m A^{(t)}_{i,k\lambda m} \bar{H}^{(t+1)}_{i,k\lambda m}
    {
        if (A_adj.extent(0) < num_nodes or A_adj.extent(1) < A_width)
            Kokkos::realloc(A_adj, num_nodes, A_width);
        const Precision gate_cst = this->gate_cst;
        const Precision gate_cst_odd = this->gate_cst_odd;
        const int A_odd = A_offset[t][1];
        const int H_odd = H_offset[t+1][1];
        const auto A = this->A[t];
        const auto H_adj = this->H_adj;
        auto A_adj = this->A_adj;
        Kokkos::parallel_for("Reverse H (scalars)",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,H_mul[t+1][0]}),
            KOKKOS_LAMBDA (const int i, const int k) {
                const Precision e = Kokkos::exp(-Kokkos::abs(A(i,k)));
                const Precision s = ((A(i,k) >= 0) ? 1 : e) / (1 + e);
                A_adj(i,k) = H_adj(i,k) * gate_cst * s * (1 + A(i,k)*(1 - s));
            });
        Kokkos::parallel_for("Reverse H (odd scalars)",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,H_mul[t+1][1]}),
            KOKKOS_LAMBDA (const int i, const int k) {
                const Precision s = Kokkos::tanh(A(i,A_odd+k));
                A_adj(i,A_odd+k) = H_adj(i,H_odd+k) * gate_cst_odd * (1 - s*s);
            });
        int g = H_mul[t+1][0];
        for (int ir=2; ir<A_mul[t].size(); ++ir) {
            const int l = ir/2;
            const int mul = A_mul[t][ir];
            const int A_off = A_offset[t][ir];
            const int H_off = H_offset[t+1][ir];
            Kokkos::parallel_for("Reverse H (gates)",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,mul}),
                KOKKOS_LAMBDA (const int i, const int k) {
                    const Precision e = Kokkos::exp(-Kokkos::abs(A(i,g+k)));
                    const Precision s = ((A(i,g+k) >= 0) ? 1 : e) / (1 + e);
                    Precision gate_adj = 0.0;
                    for (int m=0; m<2*l+1; ++m) {
                        const Precision h = H_adj(i,H_off+m*mul+k);
                        A_adj(i,A_off+m*mul+k) = h * gate_cst * A(i,g+k) * s;
                        gate_adj += h * A(i,A_off+m*mul+k);
                    }
                    A_adj(i,g+k) = gate_adj * gate_cst * s * (1 + A(i,g+k)*(1 - s));
                });
            g += mul;
        }
    }

    // A
    // \bar{\Phi}^{(t)}_{i,k'\lambda m\eta} = \sum_k w^{(t)}_{k\lambda,k'\eta} \bar{A}^{(t)}_{i,k\lambda m},  \bar{H}^{(t)}_{i,k'\lambda m} = \sum_k w^{(t)}_{a_ik\lambda,k'} \bar{A}^{(t)}_{i,k\lambda m} (t > 0)
    {
        if (Phi_adj.extent(0) < num_nodes or Phi_adj.extent(1) < Phi_width)
            Kokkos::realloc(Phi_adj, num_nodes, Phi_width);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), Phi_adj, 0.0);
        for (int ir=0; ir<A_mul[t].size() and num_nodes>0; ++ir)
            for (int m=0; m<2*(ir/2)+1 and A_weights[t][ir].size()>0; ++m)
                KokkosBlas::gemm("N", "N", 1.0,
                    subview(A_adj, make_pair(0,num_nodes),
                            make_pair(A_offset[t][ir]+m*A_mul[t][ir], A_offset[t][ir]+(m+1)*A_mul[t][ir])),
                    A_weights_trans[t][ir], 0.0,
                    subview(Phi_adj, make_pair(0,num_nodes),
                            make_pair(Phi_offset[t][ir]+m*Phi_mul[t][ir], Phi_offset[t][ir]+(m+1)*Phi_mul[t][ir])));
        if (t > 0) {
            if (H_adj.extent(0) < num_nodes or H_adj.extent(1) < H_width)
                Kokkos::realloc(H_adj, num_nodes, H_width);
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), H_adj, 0.0);
        }
        for (int ir=0; ir<A_mul[t].size() and t>0; ++ir) {
            if (A_sc_weights[t][ir].size() == 0)
                continue;
            const int l = ir/2;
            const auto W = A_sc_weights[t][ir];
            const int H_off = H_offset[t][ir];
            const int H_m = H_mul[t][ir];
            const int A_off = A_offset[t][ir];
            const int A_m = A_mul[t][ir];
            const auto A_adj = this->A_adj;
            auto H_adj = this->H_adj;
            Kokkos::parallel_for("Reverse A (self-connection)",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                    const int i = team_member.league_rank();
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 2*l+1), [&] (const int m) {
                        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, H_m), [&] (const int u) {
                            Precision s = 0.0;
                            for (int v=0; v<A_m; ++v)
                                s += A_adj(i,A_off+m*A_m+v) * W(node_types(i),u,v);
                            H_adj(i,H_off+m*H_m+u) += s;
                        });
                    });
                });
        }
    }

    // Phi
    // \bar{X}^{(t)}_{j,k\lambda_1m_1} = \sum_i \sum_{\lambda m\eta m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij}) \bar{\Phi}^{(t)}_{i,k\lambda m\eta} (t > 0)
    // \mathbf{F}_{ij} -= \sum_{k\lambda m\eta m_1m_2} C^{\lambda m}_{\lambda_1m_1l_2m_2} \nabla_{\mathbf{r}_{ij}}[R^{(t)}_{k\lambda\eta}(r_{ij}) Y_{l_2m_2}(\hat{\mathbf{r}}_{ij})] X^{(t)}_{j,k\lambda_1m_1} \bar{\Phi}^{(t)}_{i,k\lambda m\eta}
    {
        if (t > 0) {
            if (X_adj.extent(0) < X[t].extent(0) or X_adj.extent(1) < H_width)
                Kokkos::realloc(X_adj, X[t].extent(0), H_width);
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), X_adj, 0.0);
        }

        const auto first_neigh = this->first_neigh;
        const int num_lm = this->num_lm;
        const auto radial = this->radial[t];
        const auto Phi_spec = this->Phi_spec_trans[t];
        const auto Phi_rows = this->Phi_rows_trans[t];
        const auto Phi_columns = this->Phi_columns_trans[t];
        const auto Phi_clebsch_gordan = this->Phi_clebsch_gordan_trans[t];
        const auto Y = this->Y;
        const auto Y_grad = this->Y_grad;
        const auto X = this->X[t];
        const auto Phi_adj = this->Phi_adj;
        auto X_adj = this->X_adj;
        auto node_forces = this->node_forces;

        const int num_staged = (Phi_width*sizeof(Precision) <= Kokkos::TeamPolicy<>::scratch_size_max(0)) ? Phi_width : 0;
        using Scratch1 = Kokkos::View<Precision*,Kokkos::MemoryUnmanaged>;
        Kokkos::parallel_for("Reverse Phi",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32)
                .set_scratch_size(0, Kokkos::PerTeam(Scratch1::shmem_size(num_staged))),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
                const int i = team_member.league_rank();
                const Scratch1 Phi_adj_i(team_member.team_scratch(0), num_staged);
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, num_staged), [&] (const int b) {
                    Phi_adj_i(b) = Phi_adj(i,b);
                });
                team_member.team_barrier();
                const Precision* adj_i = (num_staged > 0) ? Phi_adj_i.data() : &Phi_adj(i,0);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_neigh(i)), [&] (const int jj) {
                    const int ij = first_neigh(i) + jj;
                    const int j = (t > 0) ? neigh_indices(ij) : neigh_types(ij);
                    const int n = radial.interval(r(ij));
                    const Precision x = radial.offset(r(ij), n);
                    double f_r;
                    double f_x;
                    double f_y;
                    double f_z;
                    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, H_width),
                        [&] (const int a, double& f_r, double& f_x, double& f_y, double& f_z) {
                            const int q = Phi_columns(a);
                            if (q < 0)
                                return;
                            const int p0 = Phi_rows(q);
                            const int p1 = Phi_rows(q+1);
                            const int k = a - Phi_spec(p0,0);
                            const Precision X_ja = X(j,a);
                            Precision X_adj_ja = 0.0;
                            Precision R = 0.0;
                            Precision R_deriv = 0.0;
                            for (int p=p0; p<p1; ++p) {
                                const int lm = Phi_spec(p,1);
                                const int w = Phi_spec(p,2)+k;
                                const Precision C = Phi_clebsch_gordan(p);
                                if (p == p0 or Phi_spec(p,2) != Phi_spec(p-1,2))
                                    radial.value_deriv(0, n, x, w, R, R_deriv);
                                const Precision adj = adj_i[Phi_spec(p,3)+k];
                                const Precision y = C * Y(ij*num_lm+lm);
                                const Precision u = adj * X_ja;
                                X_adj_ja += y * R * adj;
                                f_r += y * R_deriv * u;
                                f_x += C * Y_grad(3*ij*num_lm+lm) * R * u;
                                f_y += C * Y_grad((3*ij+1)*num_lm+lm) * R * u;
                                f_z += C * Y_grad((3*ij+2)*num_lm+lm) * R * u;
                            }
                            if (t > 0)
                                Kokkos::atomic_add(&X_adj(j,a), X_adj_ja);
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
void NequIPKokkos<Precision>::load_from_json(std::string filename)
{
    std::ifstream f(filename);
    nlohmann::json file = nlohmann::json::parse(f);
    if (file.value("model_type", "") != "NequIP")
        throw std::runtime_error(filename + " is not a NequIP model");

    // Basic model information
    num_elements = file["num_elements"];
    num_layers = file["num_layers"];
    r_cut = file["r_cut"];
    l_max = file["l_max"];
    num_lm = (l_max+1)*(l_max+1);
    atomic_numbers = toKokkosView("atomic_numbers", file["atomic_numbers"].get<std::vector<int>>());

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

    // X
    X_weights.resize(num_layers, std::vector<View<Precision**,LayoutRight>>(2*l_max+2));
    X_weights_trans.resize(num_layers, std::vector<View<Precision**,LayoutRight>>(2*l_max+2));
    for (int t=0; t<num_layers; ++t)
        for (int ir=0; ir<2*l_max+2; ++ir) {
            const auto w = file["X_weights"][t][ir].get<std::vector<double>>();
            if (w.size() == 0)
                continue;
            const int n = H_mul[t][ir];
            X_weights[t][ir] = View<Precision**,LayoutRight>("X_weights", n, n);
            X_weights_trans[t][ir] = View<Precision**,LayoutRight>("X_weights_trans", n, n);
            auto h_w = Kokkos::create_mirror_view(X_weights[t][ir]);
            auto h_w_trans = Kokkos::create_mirror_view(X_weights_trans[t][ir]);
            for (int kp=0; kp<n; ++kp)
                for (int k=0; k<n; ++k) {
                    h_w(kp,k) = w[kp*n+k];
                    h_w_trans(k,kp) = w[kp*n+k];
                }
            Kokkos::deep_copy(X_weights[t][ir], h_w);
            Kokkos::deep_copy(X_weights_trans[t][ir], h_w_trans);
        }
    X.resize(num_layers);
    {
        const auto embedding = file["embedding"].get<std::vector<std::vector<double>>>();
        const auto W = file["X_weights"][0][0].get<std::vector<double>>();
        const int dim = H_mul[0][0];
        H[0] = View<Precision**,LayoutRight>("H0", num_elements, dim);
        X[0] = View<Precision**,LayoutRight>("X0", num_elements, dim);
        auto h_H0 = Kokkos::create_mirror_view(H[0]);
        auto h_X0 = Kokkos::create_mirror_view(X[0]);
        for (int e=0; e<num_elements; ++e)
            for (int k=0; k<dim; ++k) {
                double x = 0.0;
                for (int kp=0; kp<dim; ++kp)
                    x += embedding[e][kp] * W[kp*dim+k];
                h_H0(e,k) = embedding[e][k];
                h_X0(e,k) = x;
            }
        Kokkos::deep_copy(H[0], h_H0);
        Kokkos::deep_copy(X[0], h_X0);
    }

    // Phi
    auto spl_values = file["radial_spline_values"].get<std::vector<std::vector<std::vector<double>>>>();
    auto spl_derivs = file["radial_spline_derivs"].get<std::vector<std::vector<std::vector<double>>>>();
    for (int t=0; t<num_layers; ++t)
        radial.push_back(RadialFunctionSetKokkos<Precision>(0.0, r_cut, {spl_values[t]}, {spl_derivs[t]}));
    Phi_mul = file["Phi_mul"].get<std::vector<std::vector<int>>>();
    Phi_offset = file["Phi_offset"].get<std::vector<std::vector<int>>>();
    for (int t=0; t<num_layers; ++t) {
        const auto spec = file["Phi_spec"][t].get<std::vector<std::array<int,5>>>();
        const auto spec_trans = file["Phi_spec_trans"][t].get<std::vector<std::array<int,5>>>();
        Phi_spec.push_back(View<int**,LayoutRight>("Phi_spec", spec.size(), 5));
        Phi_spec_trans.push_back(View<int**,LayoutRight>("Phi_spec_trans", spec_trans.size(), 5));
        auto h_spec = Kokkos::create_mirror_view(Phi_spec[t]);
        auto h_spec_trans = Kokkos::create_mirror_view(Phi_spec_trans[t]);
        for (int p=0; p<spec.size(); ++p)
            for (int x=0; x<5; ++x) {
                h_spec(p,x) = spec[p][x];
                h_spec_trans(p,x) = spec_trans[p][x];
            }
        Kokkos::deep_copy(Phi_spec[t], h_spec);
        Kokkos::deep_copy(Phi_spec_trans[t], h_spec_trans);
        Phi_clebsch_gordan.push_back(toKokkosView("Phi_clebsch_gordan", file["Phi_clebsch_gordan"][t].get<std::vector<Precision>>()));
        Phi_rows.push_back(toKokkosView("Phi_rows", file["Phi_rows"][t].get<std::vector<int>>()));
        Phi_columns.push_back(toKokkosView("Phi_columns", file["Phi_columns"][t].get<std::vector<int>>()));
        Phi_clebsch_gordan_trans.push_back(toKokkosView("Phi_clebsch_gordan_trans", file["Phi_clebsch_gordan_trans"][t].get<std::vector<Precision>>()));
        Phi_rows_trans.push_back(toKokkosView("Phi_rows_trans", file["Phi_rows_trans"][t].get<std::vector<int>>()));
        Phi_columns_trans.push_back(toKokkosView("Phi_columns_trans", file["Phi_columns_trans"][t].get<std::vector<int>>()));
    }

    // A
    A_mul = file["A_mul"].get<std::vector<std::vector<int>>>();
    A_offset = file["A_offset"].get<std::vector<std::vector<int>>>();
    A_weights.resize(num_layers, std::vector<View<Precision**,LayoutRight>>(2*l_max+2));
    A_weights_trans.resize(num_layers, std::vector<View<Precision**,LayoutRight>>(2*l_max+2));
    A_sc_weights.resize(num_layers, std::vector<View<Precision***,LayoutRight>>(2*l_max+2));
    for (int t=0; t<num_layers; ++t)
        for (int ir=0; ir<2*l_max+2; ++ir) {
            const auto w = file["A_weights"][t][ir].get<std::vector<double>>();
            if (w.size() > 0) {
                const int n0 = Phi_mul[t][ir];
                const int n1 = A_mul[t][ir];
                A_weights[t][ir] = View<Precision**,LayoutRight>("A_weights", n0, n1);
                A_weights_trans[t][ir] = View<Precision**,LayoutRight>("A_weights_trans", n1, n0);
                auto h_w = Kokkos::create_mirror_view(A_weights[t][ir]);
                auto h_w_trans = Kokkos::create_mirror_view(A_weights_trans[t][ir]);
                for (int kp=0; kp<n0; ++kp)
                    for (int k=0; k<n1; ++k) {
                        h_w(kp,k) = w[kp*n1+k];
                        h_w_trans(k,kp) = w[kp*n1+k];
                    }
                Kokkos::deep_copy(A_weights[t][ir], h_w);
                Kokkos::deep_copy(A_weights_trans[t][ir], h_w_trans);
            }
            const auto w_sc = file["A_sc_weights"][t][ir].get<std::vector<double>>();
            if (w_sc.size() > 0) {
                auto v = View<Precision***,LayoutRight>("A_sc_weights", num_elements, H_mul[t][ir], A_mul[t][ir]);
                auto h_v = Kokkos::create_mirror_view(v);
                for (int x=0; x<w_sc.size(); ++x)
                    h_v.data()[x] = w_sc[x];
                Kokkos::deep_copy(v, h_v);
                A_sc_weights[t][ir] = v;
            }
        }
    A.resize(num_layers);

    // Gates
    gate_cst = file["gate_cst"];
    gate_cst_odd = file.value("gate_cst_odd", 0.0);

    // Readout
    readout_weights = toKokkosView("readout_weights", file["readout_weights"].get<std::vector<double>>());
    per_type_scale = toKokkosView("per_type_scale", file["per_type_scale"].get<std::vector<double>>());
    per_type_shift = toKokkosView("per_type_shift", file["per_type_shift"].get<std::vector<double>>());
}

template class NequIPKokkos<float>;
template class NequIPKokkos<double>;
