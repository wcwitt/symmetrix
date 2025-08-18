#include <cmath>

#include "tools_kokkos.hpp"

#include "zbl_kokkos.hpp"

ZBLKokkos::ZBLKokkos()
{
}

ZBLKokkos::ZBLKokkos(
        double a_exp,
        double a_prefactor,
        std::vector<double> c,
        std::vector<double> covalent_radii,
        int p)
    : a_exp(a_exp),
      a_prefactor(a_prefactor),
      p(p)
{
    this->c = toKokkosView("c", c);
    this->covalent_radii = toKokkosView("covalent_radii", covalent_radii);
}

KOKKOS_FUNCTION
double ZBLKokkos::compute(const int Z_u, const int Z_v, const double r) const
{
    const double cov_rad_u = covalent_radii(Z_u);
    const double cov_rad_v = covalent_radii(Z_v);

    double Z_u_f = Z_u, Z_v_f = Z_v;
    double a = a_prefactor * 0.529 / (std::pow(Z_u_f, a_exp) + std::pow(Z_v_f, a_exp));
    double r_over_a = r / a;
    double phi = c(0) * std::exp(c_exps_0 * r_over_a) +
                 c(1) * std::exp(c_exps_1 * r_over_a) +
                 c(2) * std::exp(c_exps_2 * r_over_a) +
                 c(3) * std::exp(c_exps_3 * r_over_a);
    double v_edges = v_prefactor * Z_u_f * Z_v_f / r * phi;
    double r_max = cov_rad_u + cov_rad_v;
    double envelope = compute_envelope(r, r_max, p);
    v_edges *= 0.5 * envelope;

    return v_edges;
}

KOKKOS_FUNCTION
double ZBLKokkos::compute_gradient(const int Z_u, const int Z_v, const double r) const
{
    const double cov_rad_u = covalent_radii(Z_u);
    const double cov_rad_v = covalent_radii(Z_v);

    double Z_u_f = Z_u, Z_v_f = Z_v;
    double a = a_prefactor * 0.529 / (std::pow(Z_u_f, a_exp) + std::pow(Z_v_f, a_exp));
    double r_over_a = r / a;

    double e_0 = c(0) * exp(c_exps_0 * r_over_a);
    double e_1 = c(1) * exp(c_exps_1 * r_over_a);
    double e_2 = c(2) * exp(c_exps_2 * r_over_a);
    double e_3 = c(3) * exp(c_exps_3 * r_over_a);

    double phi = e_0 + e_1 + e_2 + e_3;
    double d_phi__d_r = c_exps_0 * e_0 + c_exps_1 * e_1 + c_exps_2 * e_2 + c_exps_3 * e_3;
    d_phi__d_r /= a;

    double v_edges = v_prefactor * Z_u_f * Z_v_f / r * phi;
    double d_v_edges__d_r = v_prefactor * Z_u_f * Z_v_f * (r * d_phi__d_r - phi) / (r * r);

    double r_max = cov_rad_u + cov_rad_v;
    double envelope = compute_envelope(r, r_max, p);
    double d_envelope__d_r = compute_envelope_gradient(r, r_max, p);

    // v_edges = 0.5 * v_edges * envelope
    double d_V_ZBL__d_r = 0.5 * (v_edges * d_envelope__d_r + d_v_edges__d_r * envelope);

    return d_V_ZBL__d_r;
}

KOKKOS_FUNCTION
double ZBLKokkos::compute_envelope(const double r, const double r_max, const int p) const
{
    if (r >= r_max) {
        return 0.0;
    }
    double r_over_r_max = r / r_max;
    double v = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * std::pow(r_over_r_max, p)
                    + p * (p + 2.0) * std::pow(r_over_r_max, p + 1)
                    - (p * (p + 1.0) / 2.0) * std::pow(r_over_r_max, p + 2));
    return v;
}

KOKKOS_FUNCTION
double ZBLKokkos::compute_envelope_gradient(const double r, const double r_max, const int p) const
{
    if (r >= r_max) {
        return 0.0;
    }
    double r_over_r_max = r / r_max;
    double v = (- ((p + 1.0) * (p + 2.0) / 2.0) * p * std::pow(r_over_r_max, p - 1)
                + p * (p + 2.0) * (p + 1.0) * std::pow(r_over_r_max, p)
                - (p * (p + 1.0) / 2.0) * (p + 2.0) * std::pow(r_over_r_max, p + 1));
    v /= r_max;
    return v;
}

void ZBLKokkos::compute_ZBL(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const int*> atomic_numbers,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> xyz,
    Kokkos::View<double*> node_energies,
    Kokkos::View<double*> node_forces)
{
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

    Kokkos::parallel_for("ZBLKokkos::compute_ZBL",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            const int type_i = node_types(i);
            const int Z_i = atomic_numbers(type_i);
            double e_i;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [&] (const int j, double& e_i) {
                    const int ij = i0 + j;
                    const int type_j = neigh_types(ij);
                    const int Z_j = atomic_numbers(type_j);
                    // energy
                    e_i += compute(Z_i, Z_j, r(ij));
                    // forces
                    const double g = compute_gradient(Z_i, Z_j, r(ij));
                    node_forces(3*ij)   -= xyz(3*ij)   / r(ij) * g;
                    node_forces(3*ij+1) -= xyz(3*ij+1) / r(ij) * g;
                    node_forces(3*ij+2) -= xyz(3*ij+2) / r(ij) * g;
                }, e_i);
            Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                node_energies(i) += e_i;
            });
        });
}
