#include <cmath>

#include "zbl.hpp"

ZBL::ZBL()
{
}

ZBL::ZBL(double a_exp,
         double a_prefactor,
         std::vector<double> c,
         std::vector<double> covalent_radii,
         int p)
    : a_exp(a_exp),
      a_prefactor(a_prefactor),
      c(c),
      covalent_radii(covalent_radii),
      p(p)
{
}

double ZBL::compute(const int Z_u, const int Z_v, const double r)
{
    const double cov_rad_u = covalent_radii[Z_u];
    const double cov_rad_v = covalent_radii[Z_v];

    double Z_u_f = Z_u, Z_v_f = Z_v;
    double a = a_prefactor * 0.529 / (std::pow(Z_u_f, a_exp) + std::pow(Z_v_f, a_exp));
    double r_over_a = r / a;
    double phi = c[0] * std::exp(c_exps_0 * r_over_a) +
                 c[1] * std::exp(c_exps_1 * r_over_a) +
                 c[2] * std::exp(c_exps_2 * r_over_a) +
                 c[3] * std::exp(c_exps_3 * r_over_a);
    double v_edges = v_prefactor * Z_u_f * Z_v_f / r * phi;
    double r_max = cov_rad_u + cov_rad_v;
    double envelope = compute_envelope(r, r_max, p);
    v_edges *= 0.5 * envelope;

    return v_edges;
}

double ZBL::compute_gradient(const int Z_u, const int Z_v, const double r)
{
    const double cov_rad_u = covalent_radii[Z_u];
    const double cov_rad_v = covalent_radii[Z_v];

    double Z_u_f = Z_u, Z_v_f = Z_v;
    double a = a_prefactor * 0.529 / (std::pow(Z_u_f, a_exp) + std::pow(Z_v_f, a_exp));
    double r_over_a = r / a;

    double e_0 = c[0] * exp(c_exps_0 * r_over_a);
    double e_1 = c[1] * exp(c_exps_1 * r_over_a);
    double e_2 = c[2] * exp(c_exps_2 * r_over_a);
    double e_3 = c[3] * exp(c_exps_3 * r_over_a);

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

double ZBL::compute_envelope(const double r, const double r_max, const int p) {
    if (r >= r_max) {
        return 0.0;
    }
    double r_over_r_max = r / r_max;
    double v = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * std::pow(r_over_r_max, p)
                    + p * (p + 2.0) * std::pow(r_over_r_max, p + 1)
                    - (p * (p + 1.0) / 2.0) * std::pow(r_over_r_max, p + 2));
    return v;
}

double ZBL::compute_envelope_gradient(const double r, const double r_max, const int p) {
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

void ZBL::compute_ZBL(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const int> atomic_numbers,
    std::span<const double> r,
    std::span<const double> xyz,
    std::span<double> node_energies,
    std::span<double> node_forces)
{
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        const int Z_i = atomic_numbers[type_i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int Z_j = atomic_numbers[type_j];
            // energy
            const double f = compute(Z_i, Z_j, r[ij]);
            node_energies[i] += f;
            // forces
            const double g = compute_gradient(Z_i, Z_j, r[ij]);
            node_forces[3*ij]   -= xyz[3*ij]   / r[ij] * g;
            node_forces[3*ij+1] -= xyz[3*ij+1] / r[ij] * g;
            node_forces[3*ij+2] -= xyz[3*ij+2] / r[ij] * g;
            ij += 1;
        }
    }
}
