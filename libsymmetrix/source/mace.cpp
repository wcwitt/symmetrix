#include <algorithm>
#include <iostream> //TODO
#include <fstream>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "sphericart.hpp"

#include "cblas.hpp"
#include "mace.hpp"

MACE::MACE(std::string filename)
{
    load_from_json(filename);
}

void MACE::prepare_active_types(std::span<const int> node_types)
{
    if (!uses_compact_radial)
        return;

    auto requested_types = std::vector<int>(node_types.begin(), node_types.end());
    std::sort(requested_types.begin(), requested_types.end());
    requested_types.erase(
        std::unique(requested_types.begin(), requested_types.end()),
        requested_types.end());
    if (requested_types.empty())
        throw std::invalid_argument("MACE compact radial cache requires at least one active type.");
    for (const int type : requested_types)
        if (type < 0 || type >= atomic_numbers.size())
            throw std::out_of_range("MACE active type index is out of range.");
    if (requested_types == active_types)
        return;

    auto new_spl_set_0 = std::vector<std::unique_ptr<CubicSplineSet>>();
    auto new_spl_set_1 = std::vector<std::unique_ptr<CubicSplineSet>>();
    auto new_A0_splines = std::vector<CubicSpline>();
    auto new_A1_splines = std::vector<CubicSpline>();
    const int pair_count = requested_types.size()*(requested_types.size()+1)/2;
    new_spl_set_0.reserve(pair_count);
    new_spl_set_1.reserve(pair_count);
    if (A0_scaled)
        new_A0_splines.reserve(pair_count);
    if (A1_scaled)
        new_A1_splines.reserve(pair_count);

    const double h = compact_radial_model->spline_h();
    const double x0 = compact_radial_model->spline_min();
    for (int local_i=0; local_i<requested_types.size(); ++local_i) {
        for (int local_j=local_i; local_j<requested_types.size(); ++local_j) {
            auto tables = compact_radial_model->materialize_pair(
                requested_types[local_i], requested_types[local_j]);
            const auto expected_R0 = static_cast<std::size_t>((l_max+1)*num_channels);
            const auto expected_R1 = static_cast<std::size_t>(Phi1_l.size()*num_channels);
            if (tables.R0.values.size() != expected_R0
                || tables.R0.derivatives.size() != expected_R0)
                throw std::runtime_error("Compact radial R0 output has an invalid size.");
            if (tables.R1.values.size() != expected_R1
                || tables.R1.derivatives.size() != expected_R1)
                throw std::runtime_error("Compact radial R1 output has an invalid size.");
            new_spl_set_0.push_back(std::make_unique<CubicSplineSet>(
                h, std::move(tables.R0.values), std::move(tables.R0.derivatives), x0));
            new_spl_set_1.push_back(std::make_unique<CubicSplineSet>(
                h, std::move(tables.R1.values), std::move(tables.R1.derivatives), x0));
            if (A0_scaled) {
                if (tables.A0.values.size() != 1 || tables.A0.derivatives.size() != 1)
                    throw std::runtime_error("Compact radial A0 network must have one output.");
                new_A0_splines.emplace_back(
                    h, std::move(tables.A0.values[0]), std::move(tables.A0.derivatives[0]), x0);
            }
            if (A1_scaled) {
                if (tables.A1.values.size() != 1 || tables.A1.derivatives.size() != 1)
                    throw std::runtime_error("Compact radial A1 network must have one output.");
                new_A1_splines.emplace_back(
                    h, std::move(tables.A1.values[0]), std::move(tables.A1.derivatives[0]), x0);
            }
        }
    }

    auto new_type_to_active = std::vector<int>(atomic_numbers.size(), -1);
    auto new_active_atomic_numbers = std::vector<int>();
    new_active_atomic_numbers.reserve(requested_types.size());
    for (int active=0; active<requested_types.size(); ++active) {
        new_type_to_active[requested_types[active]] = active;
        new_active_atomic_numbers.push_back(atomic_numbers[requested_types[active]]);
    }

    spl_set_0 = std::move(new_spl_set_0);
    spl_set_1 = std::move(new_spl_set_1);
    A0_splines = std::move(new_A0_splines);
    A1_splines = std::move(new_A1_splines);
    type_to_active = std::move(new_type_to_active);
    active_atomic_numbers = std::move(new_active_atomic_numbers);
    active_types = std::move(requested_types);
}

int MACE::radial_pair_index(int type_i, int type_j) const
{
    if (type_i < 0 || type_i >= type_to_active.size()
        || type_j < 0 || type_j >= type_to_active.size())
        throw std::out_of_range("MACE radial type index is out of range.");
    const int active_i = type_to_active[type_i];
    const int active_j = type_to_active[type_j];
    if (active_i < 0 || active_j < 0)
        throw std::runtime_error("MACE radial cache is not prepared for an active type.");
    const int num_active = active_types.size();
    return (active_i <= active_j)
        ? active_i*(2*num_active-active_i-1)/2+active_j
        : active_j*(2*num_active-active_j-1)/2+active_i;
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

void MACE::compute_node_energies_forces_field(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field)
{
    if (!has_field_coupling)
        throw std::invalid_argument(
            "MACE::compute_node_energies_forces_field requires field coupling.");

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
    compute_H1_product(num_nodes);
    compute_field_H1(num_nodes, electric_field);
    compute_H1_linear_up(num_nodes);

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
    reverse_H1_linear_up(num_nodes);
    reverse_field_H1(num_nodes, electric_field);

    reverse_H1_product(num_nodes);
    reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
}

void MACE::compute_electric_field_hessian(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field)
{
    if (not has_field_coupling)
        throw std::invalid_argument("MACE::compute_electric_field_hessian requires field coupling.");
    if (electric_field.size() != 3)
        throw std::invalid_argument("MACE::compute_electric_field_hessian requires a graph-level electric field.");

    node_energies.resize(num_nodes);
    std::fill(node_energies.begin(), node_energies.end(), 0.0);
    node_forces.resize(xyz.size());
    std::fill(node_forces.begin(), node_forces.end(), 0.0);

    compute_Y(xyz);
    compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M0(num_nodes, node_types);
    compute_H1_product(num_nodes);
    compute_field_H1(num_nodes, electric_field);
    compute_H1_linear_up(num_nodes);
    compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Phi1(num_nodes, num_neigh, neigh_indices);
    compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);
    compute_readouts(num_nodes, node_types);

    electric_field_hessian.assign(9, 0.0);
    electric_field_force_derivative.assign(3*xyz.size(), 0.0);

    const int channel_pairs = num_channels*num_channels;
    const double inv_sqrt_3 = 1.0/std::sqrt(3.0);
    const auto h1_index = [this](int i, int lm, int k) {
        return (i*num_LM + lm)*num_channels + k;
    };
    const auto vector_index = [this](int i, int k, int component) {
        return (i*num_channels + k)*3 + component;
    };

    auto A1_scale_factors = std::vector<double>(num_nodes, 1.0);
    if (A1_scaled) {
        int ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            for (int j=0; j<num_neigh[i]; ++j) {
                const int type_j = neigh_types[ij];
                const int type_ij = radial_pair_index(type_i, type_j);
                A1_scale_factors[i] += A1_splines[type_ij].evaluate(r[ij]);
                ij += 1;
            }
        }
    }

    auto A0_scale_factors = std::vector<double>(num_nodes, 1.0);
    if (A0_scaled) {
        int ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            const int type_i = node_types[i];
            for (int j=0; j<num_neigh[i]; ++j) {
                const int type_j = neigh_types[ij];
                const int type_ij = radial_pair_index(type_i, type_j);
                A0_scale_factors[i] += A0_splines[type_ij].evaluate(r[ij]);
                ij += 1;
            }
        }
    }

    int num_lme_local = 0;
    std::vector<int> num_e(l_max+1,0);
    for (auto l : Phi1_l) {
        num_lme_local += 2*l+1;
        num_e[l] += 1;
    }

    for (int seed=0; seed<3; ++seed) {
        auto H1_dot = std::vector<double>(H1.size(), 0.0);
        auto delta_scalar_dot = std::vector<double>(num_nodes*num_channels, 0.0);
        auto delta_vector_dot = std::vector<double>(num_nodes*num_channels*3, 0.0);
        auto linear_scalar_dot = std::vector<double>(num_nodes*num_channels, 0.0);
        auto linear_vector_dot = std::vector<double>(num_nodes*num_channels*3, 0.0);

        for (int i=0; i<num_nodes; ++i) {
            for (int u=0; u<num_channels; ++u) {
                const double scalar_in = H1_pre_field[h1_index(i, 0, u)];
                const double vector_in = H1_pre_field[h1_index(i, 1+seed, u)];
                for (int w=0; w<num_channels; ++w) {
                    const int weight_index = u*num_channels + w;
                    delta_vector_dot[vector_index(i, w, seed)] +=
                        field_feats_scalar_to_vector_path_weight
                        * field_feats_weight[weight_index]
                        * inv_sqrt_3
                        * scalar_in;
                    delta_scalar_dot[i*num_channels + w] +=
                        -field_feats_vector_to_scalar_path_weight
                        * field_feats_weight[channel_pairs + weight_index]
                        * inv_sqrt_3
                        * vector_in;
                }
            }
            for (int u=0; u<num_channels; ++u) {
                for (int w=0; w<num_channels; ++w) {
                    const int weight_index = u*num_channels + w;
                    linear_scalar_dot[i*num_channels + w] +=
                        field_linear_scalar_path_weight
                        * field_linear_weight[weight_index]
                        * delta_scalar_dot[i*num_channels + u];
                    for (int component=0; component<3; ++component) {
                        linear_vector_dot[vector_index(i, w, component)] +=
                            field_linear_vector_path_weight
                            * field_linear_weight[channel_pairs + weight_index]
                            * delta_vector_dot[vector_index(i, u, component)];
                    }
                }
            }
            for (int k=0; k<num_channels; ++k) {
                H1_dot[h1_index(i, 0, k)] = -linear_scalar_dot[i*num_channels + k];
                for (int component=0; component<3; ++component)
                    H1_dot[h1_index(i, 1+component, k)] =
                        linear_vector_dot[vector_index(i, k, component)];
            }
        }

        auto H1_dot_before_linear_up = H1_dot;
        std::fill(H1_dot.begin(), H1_dot.end(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_dot_in_il = H1_dot_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
                const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
                auto H1_dot_il = H1_dot.data()+(i*num_LM+l*l)*num_channels;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_dot_in_il, num_channels,
                            weights_l, num_channels,
                            0.0, H1_dot_il, num_channels);
            }
        }

        auto Phi1r_dot = std::vector<double>(Phi1r.size(), 0.0);
        auto Phi1_dot = std::vector<double>(Phi1.size(), 0.0);
        int ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1r_dot_i = Phi1r_dot.data()+i*num_lelm1lm2*num_channels;
            for (int j=0; j<num_neigh[i]; ++j) {
                auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
                auto Y_ij = Y.data()+ij*num_lm;
                auto H1_dot_ij = H1_dot.data()+neigh_indices[ij]*num_LM*num_channels;
                int lelm1lm2 = 0;
                for (int lel1l2=0; lel1l2<Phi1_l.size(); ++lel1l2) {
                    const int l1 = Phi1_l1[lel1l2];
                    const int l2 = Phi1_l2[lel1l2];
                    auto R1_ij_lel1l2 = R1_ij+lel1l2*num_channels;
                    for (int lm1=l1*l1; lm1<=l1*(l1+2); ++lm1) {
                        const double Y_ij_lm1 = Y_ij[lm1];
                        for (int lm2=l2*l2; lm2<=l2*(l2+2); ++lm2) {
                            auto H1_dot_ij_lm2 = H1_dot_ij+lm2*num_channels;
                            auto Phi1r_dot_i_lelm1lm2 = Phi1r_dot_i+lelm1lm2*num_channels;
                            for (int k=0; k<num_channels; ++k)
                                Phi1r_dot_i_lelm1lm2[k] +=
                                    R1_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k];
                            lelm1lm2 += 1;
                        }
                    }
                }
                ij += 1;
            }
        }
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1_dot_i = Phi1_dot.data()+i*num_lme*num_channels;
            auto Phi1r_dot_i = Phi1r_dot.data()+i*num_lelm1lm2*num_channels;
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan[p];
                auto Phi1_dot_i_lme = Phi1_dot_i+Phi1_lme[p]*num_channels;
                auto Phi1r_dot_i_lelm1lm2 = Phi1r_dot_i+Phi1_lelm1lm2[p]*num_channels;
                for (int k=0; k<num_channels; ++k)
                    Phi1_dot_i_lme[k] += C * Phi1r_dot_i_lelm1lm2[k];
            }
        }

        auto A1_dot = std::vector<double>(A1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto Phi1_dot_il = Phi1_dot.data()+i*num_lme_local*num_channels;
            auto A1_dot_il = A1_dot.data()+i*num_lm*num_channels;
            for (int l=0; l<=l_max; ++l) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            2*l+1, num_channels, num_e[l]*num_channels,
                            1.0, Phi1_dot_il, num_e[l]*num_channels,
                            A1_weights[l].data(), num_channels,
                            0.0, A1_dot_il, num_channels);
                Phi1_dot_il += (2*l+1)*num_e[l]*num_channels;
                A1_dot_il += (2*l+1)*num_channels;
            }
        }
        if (A1_scaled) {
            for (int i=0; i<num_nodes; ++i) {
                auto A1_dot_i = A1_dot.data()+i*num_lm*num_channels;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                    A1_dot_i[lmk] /= A1_scale_factors[i];
            }
        }

        auto M1_dot = std::vector<double>(M1.size(), 0.0);
        auto M1_grad_dot = std::vector<double>(M1_grad.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto A1_i = A1.data()+i*num_lm*num_channels;
            auto A1_dot_i = A1_dot.data()+i*num_lm*num_channels;
            auto M1_grad_dot_i = M1_grad_dot.data()+i*num_channels*num_lm;
            auto x = std::vector<double>(num_lm);
            auto x_dot = std::vector<double>(num_lm);
            for (int k=0; k<num_channels; ++k) {
                cblas_dcopy(num_lm, A1_i+k, num_channels, x.data(), 1);
                cblas_dcopy(num_lm, A1_dot_i+k, num_channels, x_dot.data(), 1);
                auto [f, g, g_dot] =
                    P1[node_types[i]*num_channels+k].evaluate_gradient_directional(x, x_dot);
                M1_dot[i*num_channels+k] = cblas_ddot(num_lm, g.data(), 1, x_dot.data(), 1);
                cblas_dcopy(num_lm, g_dot.data(), 1, M1_grad_dot_i+k, num_channels);
            }
        }

        auto H2_dot = std::vector<double>(H2.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto H2_dot_i = H2_dot.data()+i*num_channels;
            auto H1_dot_i = H1_dot.data()+i*num_LM*num_channels;
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_H1[node_types[i]].data(), num_channels,
                        H1_dot_i, 1,
                        0.0, H2_dot_i, 1);
            auto M1_dot_i = M1_dot.data()+i*num_channels;
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_M1.data(), num_channels,
                        M1_dot_i, 1,
                        1.0, H2_dot_i, 1);
        }

        auto H1_adj_local = std::vector<double>(H1.size(), 0.0);
        auto H1_adj_dot = std::vector<double>(H1.size(), 0.0);
        auto H2_adj_local = std::vector<double>(H2.size(), 0.0);
        auto H2_adj_dot = std::vector<double>(H2.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int k=0; k<num_channels; ++k)
                H1_adj_local[i*num_LM*num_channels+k] = readout_1_weights[k];
            auto x = std::vector<double>(H2.begin()+i*num_channels, H2.begin()+(i+1)*num_channels);
            auto x_dot = std::vector<double>(H2_dot.begin()+i*num_channels, H2_dot.begin()+(i+1)*num_channels);
            auto [f, g, g_dot] = readout_2->evaluate_gradient_directional(x, x_dot);
            for (int k=0; k<num_channels; ++k) {
                H2_adj_local[i*num_channels+k] = g[k];
                H2_adj_dot[i*num_channels+k] = g_dot[k];
            }
        }

        auto M1_adj_local = std::vector<double>(M1.size(), 0.0);
        auto M1_adj_dot = std::vector<double>(M1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto H2_adj_i = H2_adj_local.data()+i*num_channels;
            auto H2_adj_dot_i = H2_adj_dot.data()+i*num_channels;
            auto H1_adj_i = H1_adj_local.data()+i*num_LM*num_channels;
            auto H1_adj_dot_i = H1_adj_dot.data()+i*num_LM*num_channels;
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_H1[node_types[i]].data(), num_channels,
                        H2_adj_i, 1,
                        1.0, H1_adj_i, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_H1[node_types[i]].data(), num_channels,
                        H2_adj_dot_i, 1,
                        1.0, H1_adj_dot_i, 1);
            auto M1_adj_i = M1_adj_local.data()+i*num_channels;
            auto M1_adj_dot_i = M1_adj_dot.data()+i*num_channels;
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_M1.data(), num_channels,
                        H2_adj_i, 1,
                        0.0, M1_adj_i, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        num_channels, num_channels,
                        1.0, H2_weights_for_M1.data(), num_channels,
                        H2_adj_dot_i, 1,
                        0.0, M1_adj_dot_i, 1);
        }

        auto A1_adj_local = std::vector<double>(A1.size(), 0.0);
        auto A1_adj_dot = std::vector<double>(A1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto M1_adj_i = M1_adj_local.data()+i*num_channels;
            auto M1_adj_dot_i = M1_adj_dot.data()+i*num_channels;
            for (int lm=0; lm<num_lm; ++lm) {
                auto A1_adj_ilm = A1_adj_local.data()+(i*num_lm+lm)*num_channels;
                auto A1_adj_dot_ilm = A1_adj_dot.data()+(i*num_lm+lm)*num_channels;
                auto M1_grad_ilm = M1_grad.data()+(i*num_lm+lm)*num_channels;
                auto M1_grad_dot_ilm = M1_grad_dot.data()+(i*num_lm+lm)*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    A1_adj_ilm[k] = M1_grad_ilm[k] * M1_adj_i[k];
                    A1_adj_dot_ilm[k] =
                        M1_grad_dot_ilm[k] * M1_adj_i[k]
                        + M1_grad_ilm[k] * M1_adj_dot_i[k];
                }
            }
        }
        if (A1_scaled) {
            int ij_scale = 0;
            for (int i=0; i<num_nodes; ++i) {
                const int type_i = node_types[i];
                auto A1_i = A1.data()+i*num_lm*num_channels;
                auto A1_dot_i = A1_dot.data()+i*num_lm*num_channels;
                auto A1_adj_i = A1_adj_local.data()+i*num_lm*num_channels;
                auto A1_adj_dot_i = A1_adj_dot.data()+i*num_lm*num_channels;
                double dA1_dot_A1_dot = 0.0;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk) {
                    dA1_dot_A1_dot +=
                        A1_adj_dot_i[lmk] * A1_i[lmk]
                        + A1_adj_i[lmk] * A1_dot_i[lmk];
                }
                for (int j=0; j<num_neigh[i]; ++j) {
                    const int type_j = neigh_types[ij_scale];
                    const int type_ij = radial_pair_index(type_i, type_j);
                    auto [f,d] = A1_splines[type_ij].evaluate_deriv(r[ij_scale]);
                    auto xyz_ij = xyz.data()+ij_scale*3;
                    auto force_deriv_ij =
                        electric_field_force_derivative.data()+seed*xyz.size()+ij_scale*3;
                    force_deriv_ij[0] += dA1_dot_A1_dot/A1_scale_factors[i]*d*xyz_ij[0]/r[ij_scale];
                    force_deriv_ij[1] += dA1_dot_A1_dot/A1_scale_factors[i]*d*xyz_ij[1]/r[ij_scale];
                    force_deriv_ij[2] += dA1_dot_A1_dot/A1_scale_factors[i]*d*xyz_ij[2]/r[ij_scale];
                    ij_scale += 1;
                }
            }
            for (int i=0; i<num_nodes; ++i) {
                auto A1_adj_i = A1_adj_local.data()+i*num_lm*num_channels;
                auto A1_adj_dot_i = A1_adj_dot.data()+i*num_lm*num_channels;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk) {
                    A1_adj_i[lmk] /= A1_scale_factors[i];
                    A1_adj_dot_i[lmk] /= A1_scale_factors[i];
                }
            }
        }

        auto dPhi1_local = std::vector<double>(Phi1.size(), 0.0);
        auto dPhi1_dot = std::vector<double>(Phi1.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto A1_adj_il = A1_adj_local.data()+i*num_lm*num_channels;
            auto A1_adj_dot_il = A1_adj_dot.data()+i*num_lm*num_channels;
            auto dPhi1_il = dPhi1_local.data()+i*num_lme_local*num_channels;
            auto dPhi1_dot_il = dPhi1_dot.data()+i*num_lme_local*num_channels;
            for (int l=0; l<=l_max; ++l) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_e[l]*num_channels, num_channels,
                            1.0, A1_adj_il, num_channels,
                            A1_weights[l].data(), num_channels,
                            0.0, dPhi1_il, num_e[l]*num_channels);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_e[l]*num_channels, num_channels,
                            1.0, A1_adj_dot_il, num_channels,
                            A1_weights[l].data(), num_channels,
                            0.0, dPhi1_dot_il, num_e[l]*num_channels);
                A1_adj_il += (2*l+1)*num_channels;
                A1_adj_dot_il += (2*l+1)*num_channels;
                dPhi1_il += (2*l+1)*num_e[l]*num_channels;
                dPhi1_dot_il += (2*l+1)*num_e[l]*num_channels;
            }
        }

        auto dPhi1r_local = std::vector<double>(Phi1r.size(), 0.0);
        auto dPhi1r_dot = std::vector<double>(Phi1r.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto dPhi1r_i = dPhi1r_local.data()+i*num_lelm1lm2*num_channels;
            auto dPhi1r_dot_i = dPhi1r_dot.data()+i*num_lelm1lm2*num_channels;
            auto dPhi1_i = dPhi1_local.data()+i*num_lme*num_channels;
            auto dPhi1_dot_i = dPhi1_dot.data()+i*num_lme*num_channels;
            for (int p=0; p<Phi1_clebsch_gordan.size(); ++p) {
                const double C = Phi1_clebsch_gordan[p];
                auto dPhi1r_i_lelm1lm2 = dPhi1r_i+Phi1_lelm1lm2[p]*num_channels;
                auto dPhi1r_dot_i_lelm1lm2 = dPhi1r_dot_i+Phi1_lelm1lm2[p]*num_channels;
                auto dPhi1_i_lme = dPhi1_i+Phi1_lme[p]*num_channels;
                auto dPhi1_dot_i_lme = dPhi1_dot_i+Phi1_lme[p]*num_channels;
                for (int k=0; k<num_channels; ++k) {
                    dPhi1r_i_lelm1lm2[k] += C * dPhi1_i_lme[k];
                    dPhi1r_dot_i_lelm1lm2[k] += C * dPhi1_dot_i_lme[k];
                }
            }
        }

        ij = 0;
        for (int i=0; i<num_nodes; ++i) {
            auto dPhi1r_i = dPhi1r_local.data()+i*num_lelm1lm2*num_channels;
            auto dPhi1r_dot_i = dPhi1r_dot.data()+i*num_lelm1lm2*num_channels;
            for (int j=0; j<num_neigh[i]; ++j) {
                auto R1_ij = R1.data()+ij*spl_set_1[0]->num_splines;
                auto R1_deriv_ij = R1_deriv.data()+ij*spl_set_1[0]->num_splines;
                auto Y_ij = Y.data()+ij*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij*3*num_lm;
                auto H1_ij = H1.data()+neigh_indices[ij]*num_LM*num_channels;
                auto H1_dot_ij = H1_dot.data()+neigh_indices[ij]*num_LM*num_channels;
                auto H1_adj_ij = H1_adj_local.data()+neigh_indices[ij]*num_LM*num_channels;
                auto H1_adj_dot_ij = H1_adj_dot.data()+neigh_indices[ij]*num_LM*num_channels;
                auto xyz_ij = xyz.data()+ij*3;
                auto force_deriv_ij = electric_field_force_derivative.data()+seed*xyz.size()+ij*3;
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
                            auto H1_dot_ij_lm2 = H1_dot_ij+lm2*num_channels;
                            auto H1_adj_ij_lm2 = H1_adj_ij+lm2*num_channels;
                            auto H1_adj_dot_ij_lm2 = H1_adj_dot_ij+lm2*num_channels;
                            auto dPhi1r_i_lelm1lm2 = dPhi1r_i+lelm1lm2*num_channels;
                            auto dPhi1r_dot_i_lelm1lm2 = dPhi1r_dot_i+lelm1lm2*num_channels;
                            for (int k=0; k<num_channels; ++k) {
                                const double force_factor_x =
                                    xyz_ij[0]/r[ij] * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_ij_lm2[k];
                                const double force_factor_y =
                                    xyz_ij[1]/r[ij] * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_ij_lm2[k];
                                const double force_factor_z =
                                    xyz_ij[2]/r[ij] * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_ij_lm2[k];
                                const double force_factor_dot_x =
                                    xyz_ij[0]/r[ij] * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_x_lm1 * H1_dot_ij_lm2[k];
                                const double force_factor_dot_y =
                                    xyz_ij[1]/r[ij] * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_y_lm1 * H1_dot_ij_lm2[k];
                                const double force_factor_dot_z =
                                    xyz_ij[2]/r[ij] * R1_deriv_ij_lel1l2[k] * Y_ij_lm1 * H1_dot_ij_lm2[k]
                                    + R1_ij_lel1l2[k] * Y_grad_ij_z_lm1 * H1_dot_ij_lm2[k];
                                force_deriv_ij[0] += -(
                                    dPhi1r_dot_i_lelm1lm2[k] * force_factor_x
                                    + dPhi1r_i_lelm1lm2[k] * force_factor_dot_x);
                                force_deriv_ij[1] += -(
                                    dPhi1r_dot_i_lelm1lm2[k] * force_factor_y
                                    + dPhi1r_i_lelm1lm2[k] * force_factor_dot_y);
                                force_deriv_ij[2] += -(
                                    dPhi1r_dot_i_lelm1lm2[k] * force_factor_z
                                    + dPhi1r_i_lelm1lm2[k] * force_factor_dot_z);
                                H1_adj_ij_lm2[k] +=
                                    R1_ij_lel1l2[k]*Y_ij_lm1*dPhi1r_i_lelm1lm2[k];
                                H1_adj_dot_ij_lm2[k] +=
                                    R1_ij_lel1l2[k]*Y_ij_lm1*dPhi1r_dot_i_lelm1lm2[k];
                            }
                            lelm1lm2 += 1;
                        }
                    }
                }
                ij += 1;
            }
        }

        auto H1_adj_before_linear_up = H1_adj_local;
        auto H1_adj_dot_before_linear_up = H1_adj_dot;
        std::fill(H1_adj_local.begin(), H1_adj_local.end(), 0.0);
        std::fill(H1_adj_dot.begin(), H1_adj_dot.end(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_adj_il = H1_adj_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
                const auto H1_adj_dot_il = H1_adj_dot_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
                const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
                auto H1_pre_adj_il = H1_adj_local.data()+(i*num_LM+l*l)*num_channels;
                auto H1_pre_adj_dot_il = H1_adj_dot.data()+(i*num_LM+l*l)*num_channels;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_il, num_channels,
                            weights_l, num_channels,
                            0.0, H1_pre_adj_il, num_channels);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_dot_il, num_channels,
                            weights_l, num_channels,
                            0.0, H1_pre_adj_dot_il, num_channels);
            }
        }

        auto delta_scalar_adj = std::vector<double>(num_nodes*num_channels, 0.0);
        auto delta_vector_adj = std::vector<double>(num_nodes*num_channels*3, 0.0);
        auto delta_scalar_adj_dot = std::vector<double>(num_nodes*num_channels, 0.0);
        auto delta_vector_adj_dot = std::vector<double>(num_nodes*num_channels*3, 0.0);
        auto H1_pre_adj = H1_adj_local;
        auto H1_pre_adj_dot = H1_adj_dot;
        for (int i=0; i<num_nodes; ++i) {
            for (int u=0; u<num_channels; ++u) {
                for (int w=0; w<num_channels; ++w) {
                    const int weight_index = u*num_channels + w;
                    delta_scalar_adj[i*num_channels + u] +=
                        -field_linear_scalar_path_weight
                        * field_linear_weight[weight_index]
                        * H1_adj_local[h1_index(i, 0, w)];
                    delta_scalar_adj_dot[i*num_channels + u] +=
                        -field_linear_scalar_path_weight
                        * field_linear_weight[weight_index]
                        * H1_adj_dot[h1_index(i, 0, w)];
                    for (int component=0; component<3; ++component) {
                        delta_vector_adj_dot[vector_index(i, u, component)] +=
                            field_linear_vector_path_weight
                            * field_linear_weight[channel_pairs + weight_index]
                            * H1_adj_dot[h1_index(i, 1+component, w)];
                        delta_vector_adj[vector_index(i, u, component)] +=
                            field_linear_vector_path_weight
                            * field_linear_weight[channel_pairs + weight_index]
                            * H1_adj_local[h1_index(i, 1+component, w)];
                    }
                }
            }
            for (int u=0; u<num_channels; ++u) {
                const double scalar_in = H1_pre_field[h1_index(i, 0, u)];
                for (int w=0; w<num_channels; ++w) {
                    const int weight_index = u*num_channels + w;
                    const double scalar_to_vector_weight =
                        field_feats_scalar_to_vector_path_weight
                        * field_feats_weight[weight_index]
                        * inv_sqrt_3;
                    const double vector_to_scalar_weight =
                        field_feats_vector_to_scalar_path_weight
                        * field_feats_weight[channel_pairs + weight_index]
                        * inv_sqrt_3;
                    const double scalar_delta_adj =
                        delta_scalar_adj[i*num_channels + w];
                    const double scalar_delta_adj_dot =
                        delta_scalar_adj_dot[i*num_channels + w];
                    for (int component=0; component<3; ++component) {
                        const double field_dot = (component == seed) ? 1.0 : 0.0;
                        const double vector_delta_adj =
                            delta_vector_adj[vector_index(i, w, component)];
                        const double vector_delta_adj_dot =
                            delta_vector_adj_dot[vector_index(i, w, component)];
                        H1_pre_adj[h1_index(i, 0, u)] +=
                            vector_delta_adj*scalar_to_vector_weight*electric_field[component];
                        H1_pre_adj_dot[h1_index(i, 0, u)] +=
                            vector_delta_adj_dot*scalar_to_vector_weight*electric_field[component]
                            + vector_delta_adj*scalar_to_vector_weight*field_dot;
                        electric_field_hessian[component*3 + seed] +=
                            vector_delta_adj_dot*scalar_to_vector_weight*scalar_in;
                        H1_pre_adj[h1_index(i, 1+component, u)] +=
                            -scalar_delta_adj*vector_to_scalar_weight*electric_field[component];
                        H1_pre_adj_dot[h1_index(i, 1+component, u)] +=
                            -scalar_delta_adj_dot*vector_to_scalar_weight*electric_field[component]
                            - scalar_delta_adj*vector_to_scalar_weight*field_dot;
                        electric_field_hessian[component*3 + seed] +=
                            -scalar_delta_adj_dot*vector_to_scalar_weight
                            * H1_pre_field[h1_index(i, 1+component, u)];
                    }
                }
            }
        }
        H1_adj_local = std::move(H1_pre_adj);
        H1_adj_dot = std::move(H1_pre_adj_dot);

        auto M0_adj_local = std::vector<double>(M0.size(), 0.0);
        auto M0_adj_dot = std::vector<double>(M0.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            for (int l=0; l<=L_max; ++l) {
                const auto H1_adj_il = H1_adj_local.data()+(i*num_LM+l*l)*num_channels;
                const auto H1_adj_dot_il = H1_adj_dot.data()+(i*num_LM+l*l)*num_channels;
                const auto weights_l = H1_product_weights.data()+l*num_channels*num_channels;
                auto M0_adj_il = M0_adj_local.data()+(i*num_LM+l*l)*num_channels;
                auto M0_adj_dot_il = M0_adj_dot.data()+(i*num_LM+l*l)*num_channels;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_il, num_channels,
                            weights_l, num_channels,
                            0.0, M0_adj_il, num_channels);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, H1_adj_dot_il, num_channels,
                            weights_l, num_channels,
                            0.0, M0_adj_dot_il, num_channels);
            }
        }

        auto A0_adj_local = std::vector<double>(A0.size(), 0.0);
        auto A0_adj_dot = std::vector<double>(A0.size(), 0.0);
        for (int i=0; i<num_nodes; ++i) {
            auto A0_adj_i = A0_adj_local.data()+i*num_lm*num_channels;
            auto A0_adj_dot_i = A0_adj_dot.data()+i*num_lm*num_channels;
            auto M0_adj_i = M0_adj_local.data()+i*num_LM*num_channels;
            auto M0_adj_dot_i = M0_adj_dot.data()+i*num_LM*num_channels;
            auto M0_grad_i = M0_grad.data()+i*num_LM*num_channels*num_lm;
            for (int lm=0; lm<num_lm; ++lm) {
                auto A0_adj_ilm = A0_adj_i + lm*num_channels;
                auto A0_adj_dot_ilm = A0_adj_dot_i + lm*num_channels;
                for (int lmp=0; lmp<num_LM; ++lmp) {
                    auto M0_adj_ilmp = M0_adj_i + lmp*num_channels;
                    auto M0_adj_dot_ilmp = M0_adj_dot_i + lmp*num_channels;
                    auto M0_grad_ilmplm =
                        M0_grad_i + lmp*num_lm*num_channels + lm*num_channels;
                    for (int k=0; k<num_channels; ++k) {
                        A0_adj_ilm[k] += M0_grad_ilmplm[k] * M0_adj_ilmp[k];
                        A0_adj_dot_ilm[k] += M0_grad_ilmplm[k] * M0_adj_dot_ilmp[k];
                    }
                }
            }
        }

        if (A0_scaled) {
            int ij_scale = 0;
            for (int i=0; i<num_nodes; ++i) {
                const int type_i = node_types[i];
                auto A0_i = A0.data()+i*num_lm*num_channels;
                auto A0_adj_dot_i = A0_adj_dot.data()+i*num_lm*num_channels;
                double dA0_dot_A0_dot = 0.0;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk)
                    dA0_dot_A0_dot += A0_adj_dot_i[lmk] * A0_i[lmk];
                for (int j=0; j<num_neigh[i]; ++j) {
                    const int type_j = neigh_types[ij_scale];
                    const int type_ij = radial_pair_index(type_i, type_j);
                    auto [f,d] = A0_splines[type_ij].evaluate_deriv(r[ij_scale]);
                    auto xyz_ij = xyz.data()+ij_scale*3;
                    auto force_deriv_ij =
                        electric_field_force_derivative.data()+seed*xyz.size()+ij_scale*3;
                    force_deriv_ij[0] += dA0_dot_A0_dot/A0_scale_factors[i]*d*xyz_ij[0]/r[ij_scale];
                    force_deriv_ij[1] += dA0_dot_A0_dot/A0_scale_factors[i]*d*xyz_ij[1]/r[ij_scale];
                    force_deriv_ij[2] += dA0_dot_A0_dot/A0_scale_factors[i]*d*xyz_ij[2]/r[ij_scale];
                    ij_scale += 1;
                }
            }
            for (int i=0; i<num_nodes; ++i) {
                auto A0_adj_i = A0_adj_local.data()+i*num_lm*num_channels;
                auto A0_adj_dot_i = A0_adj_dot.data()+i*num_lm*num_channels;
                for (int lmk=0; lmk<num_lm*num_channels; ++lmk) {
                    A0_adj_i[lmk] /= A0_scale_factors[i];
                    A0_adj_dot_i[lmk] /= A0_scale_factors[i];
                }
            }
        }

        int ij_a0 = 0;
        for (int i=0; i<num_nodes; ++i) {
            auto Phi0_adj_dot_i = std::vector<double>(num_lm*num_channels, 0.0);
            for (int l=0; l<=l_max; ++l) {
                auto Phi0_adj_dot_il = Phi0_adj_dot_i.data()+l*l*num_channels;
                auto A0_adj_dot_il = A0_adj_dot.data()+(i*num_lm+l*l)*num_channels;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            2*l+1, num_channels, num_channels,
                            1.0, A0_adj_dot_il, num_channels,
                            A0_weights[node_types[i]][l].data(), num_channels,
                            0.0, Phi0_adj_dot_il, num_channels);
            }

            for (int j=0; j<num_neigh[i]; ++j) {
                auto xyz_ij = xyz.data()+ij_a0*3;
                const double r_ij = r[ij_a0];
                auto Y_ij = Y.data()+ij_a0*num_lm;
                auto Y_grad_ij = Y_grad.data()+ij_a0*3*num_lm;
                auto H0_ij = H0_weights.data()+neigh_types[ij_a0]*num_channels;
                auto force_deriv_ij = electric_field_force_derivative.data()+seed*xyz.size()+ij_a0*3;
                for (int l=0; l<=l_max; ++l) {
                    auto R0_ij_l = R0.data()+ij_a0*(l_max+1)*num_channels+l*num_channels;
                    auto R0_deriv_ij_l = R0_deriv.data()+ij_a0*(l_max+1)*num_channels+l*num_channels;
                    for (int m=-l; m<=l; ++m) {
                        const int lm = l*l+l+m;
                        const double Y_ij_lm = Y_ij[lm];
                        const double Y_grad_ij_lm_x = Y_grad_ij[lm];
                        const double Y_grad_ij_lm_y = Y_grad_ij[num_lm+lm];
                        const double Y_grad_ij_lm_z = Y_grad_ij[2*num_lm+lm];
                        auto Phi0_adj_dot_i_lm = Phi0_adj_dot_i.data()+lm*num_channels;
                        for (int k=0; k<num_channels; ++k) {
                            force_deriv_ij[0] += -Phi0_adj_dot_i_lm[k] * (
                                xyz_ij[0]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                                + R0_ij_l[k] * Y_grad_ij_lm_x * H0_ij[k]);
                            force_deriv_ij[1] += -Phi0_adj_dot_i_lm[k] * (
                                xyz_ij[1]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                                + R0_ij_l[k] * Y_grad_ij_lm_y * H0_ij[k]);
                            force_deriv_ij[2] += -Phi0_adj_dot_i_lm[k] * (
                                xyz_ij[2]/r_ij * R0_deriv_ij_l[k] * Y_ij_lm * H0_ij[k]
                                + R0_ij_l[k] * Y_grad_ij_lm_z * H0_ij[k]);
                        }
                    }
                }
                ij_a0 += 1;
            }
        }
    }
}

void MACE::compute_electric_field_force_derivative(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_indices,
    std::span<const int> neigh_types,
    std::span<const double> xyz,
    std::span<const double> r,
    std::span<const double> electric_field)
{
    compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz,
        r,
        electric_field);
}

void MACE::compute_R0(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (spl_set_0.empty())
        throw std::runtime_error(
            "MACE compact radial cache is not prepared; call prepare_active_types first.");
    const int num_spl = spl_set_0[0]->num_splines;
    R0.resize(r.size()*num_spl);
    R0_deriv.resize(R0.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
            auto R0_ij = std::span<double>(R0.data()+ij*num_spl,num_spl);
            auto R0_deriv_ij = std::span<double>(R0_deriv.data()+ij*num_spl,num_spl);
            spl_set_0[type_ij]->evaluate_derivs(r[ij], R0_ij, R0_deriv_ij);
            ij += 1;
        }
    }
}

void MACE::compute_field_H1(
    const int num_nodes,
    std::span<const double> electric_field)
{
    if (!has_field_coupling)
        return;

    if (L_max != 1 || num_LM != 4)
        throw std::runtime_error("MACEField H1 coupling currently requires L_max == 1.");

    if (electric_field.size() != 3 && electric_field.size() != 3*num_nodes)
        throw std::runtime_error("MACEField electric_field must have shape (3,) or (num_nodes, 3).");

    const int hidden_size = num_LM*num_channels;
    if (H1.size() != num_nodes*hidden_size)
        throw std::runtime_error("MACEField H1 buffer size does not match num_nodes.");

    const int channel_pairs = num_channels*num_channels;
    if (field_feats_weight.size() != 2*channel_pairs
        || field_linear_weight.size() != 2*channel_pairs)
        throw std::runtime_error("MACEField field coupling weights do not match num_channels.");

    H1_pre_field = H1;
    std::vector<double> delta_scalar(num_nodes*num_channels, 0.0);
    std::vector<double> delta_vector(num_nodes*num_channels*3, 0.0);
    std::vector<double> linear_scalar(num_nodes*num_channels, 0.0);
    std::vector<double> linear_vector(num_nodes*num_channels*3, 0.0);

    const double inv_sqrt_3 = 1.0/std::sqrt(3.0);
    const auto h1_index = [this](int i, int lm, int k) {
        return (i*num_LM + lm)*num_channels + k;
    };
    const auto vector_index = [this](int i, int k, int component) {
        return (i*num_channels + k)*3 + component;
    };

    for (int i=0; i<num_nodes; ++i) {
        const double* field_i = electric_field.data() + (electric_field.size() == 3 ? 0 : 3*i);

        for (int u=0; u<num_channels; ++u) {
            const double scalar_in = H1_pre_field[h1_index(i, 0, u)];
            double vector_dot_field = 0.0;
            for (int component=0; component<3; ++component)
                vector_dot_field += -H1_pre_field[h1_index(i, 1+component, u)]*field_i[component];

            for (int w=0; w<num_channels; ++w) {
                const int weight_index = u*num_channels + w;
                const double scalar_to_vector_weight =
                    field_feats_scalar_to_vector_path_weight
                    * field_feats_weight[weight_index]
                    * inv_sqrt_3;
                const double vector_to_scalar_weight =
                    field_feats_vector_to_scalar_path_weight
                    * field_feats_weight[channel_pairs + weight_index]
                    * inv_sqrt_3;

                for (int component=0; component<3; ++component)
                    delta_vector[vector_index(i, w, component)] +=
                        scalar_to_vector_weight*scalar_in*field_i[component];

                delta_scalar[i*num_channels + w] += vector_to_scalar_weight*vector_dot_field;
            }
        }

        for (int u=0; u<num_channels; ++u) {
            const double scalar_in = delta_scalar[i*num_channels + u];
            for (int w=0; w<num_channels; ++w) {
                const int weight_index = u*num_channels + w;
                linear_scalar[i*num_channels + w] +=
                    field_linear_scalar_path_weight
                    * field_linear_weight[weight_index]
                    * scalar_in;
                for (int component=0; component<3; ++component)
                    linear_vector[vector_index(i, w, component)] +=
                        field_linear_vector_path_weight
                        * field_linear_weight[channel_pairs + weight_index]
                        * delta_vector[vector_index(i, u, component)];
            }
        }

        for (int k=0; k<num_channels; ++k) {
            H1[h1_index(i, 0, k)] =
                H1_pre_field[h1_index(i, 0, k)] - linear_scalar[i*num_channels + k];
            for (int component=0; component<3; ++component)
                H1[h1_index(i, 1+component, k)] =
                    H1_pre_field[h1_index(i, 1+component, k)]
                    + linear_vector[vector_index(i, k, component)];
        }
    }
}

void MACE::reverse_field_H1(
    const int num_nodes,
    std::span<const double> electric_field)
{
    if (!has_field_coupling)
        return;

    if (electric_field.size() != 3 && electric_field.size() != 3*num_nodes)
        throw std::runtime_error("MACEField electric_field must have shape (3,) or (num_nodes, 3).");

    const int hidden_size = num_LM*num_channels;
    if (H1_adj.size() != num_nodes*hidden_size || H1_pre_field.size() != num_nodes*hidden_size)
        throw std::runtime_error("MACEField reverse_field_H1 requires H1_adj and saved pre-field H1 buffers.");

    const int channel_pairs = num_channels*num_channels;
    const bool global_field = electric_field.size() == 3;
    electric_field_adj.assign(electric_field.size(), 0.0);

    std::vector<double> delta_scalar_adj(num_nodes*num_channels, 0.0);
    std::vector<double> delta_vector_adj(num_nodes*num_channels*3, 0.0);
    std::vector<double> H1_pre_adj = H1_adj;

    const double inv_sqrt_3 = 1.0/std::sqrt(3.0);
    const auto h1_index = [this](int i, int lm, int k) {
        return (i*num_LM + lm)*num_channels + k;
    };
    const auto vector_index = [this](int i, int k, int component) {
        return (i*num_channels + k)*3 + component;
    };

    for (int i=0; i<num_nodes; ++i) {
        const double* field_i = electric_field.data() + (global_field ? 0 : 3*i);
        double* field_adj_i = electric_field_adj.data() + (global_field ? 0 : 3*i);

        // Reverse field_linear. H1_post = H1_pre - field_linear(delta).
        for (int u=0; u<num_channels; ++u) {
            for (int w=0; w<num_channels; ++w) {
                const int weight_index = u*num_channels + w;
                delta_scalar_adj[i*num_channels + u] +=
                    -field_linear_scalar_path_weight
                    * field_linear_weight[weight_index]
                    * H1_adj[h1_index(i, 0, w)];
                for (int component=0; component<3; ++component)
                    delta_vector_adj[vector_index(i, u, component)] +=
                        field_linear_vector_path_weight
                        * field_linear_weight[channel_pairs + weight_index]
                        * H1_adj[h1_index(i, 1+component, w)];
            }
        }

        // Reverse field_feats scalar->vector and vector->scalar paths.
        for (int u=0; u<num_channels; ++u) {
            const double scalar_in = H1_pre_field[h1_index(i, 0, u)];
            for (int w=0; w<num_channels; ++w) {
                const int weight_index = u*num_channels + w;
                const double scalar_to_vector_weight =
                    field_feats_scalar_to_vector_path_weight
                    * field_feats_weight[weight_index]
                    * inv_sqrt_3;
                const double vector_to_scalar_weight =
                    field_feats_vector_to_scalar_path_weight
                    * field_feats_weight[channel_pairs + weight_index]
                    * inv_sqrt_3;
                const double scalar_delta_adj = delta_scalar_adj[i*num_channels + w];

                for (int component=0; component<3; ++component) {
                    const double vector_delta_adj =
                        delta_vector_adj[vector_index(i, w, component)];
                    H1_pre_adj[h1_index(i, 0, u)] +=
                        vector_delta_adj*scalar_to_vector_weight*field_i[component];
                    field_adj_i[component] +=
                        vector_delta_adj*scalar_to_vector_weight*scalar_in;

                    H1_pre_adj[h1_index(i, 1+component, u)] +=
                        -scalar_delta_adj*vector_to_scalar_weight*field_i[component];
                    field_adj_i[component] +=
                        -scalar_delta_adj*vector_to_scalar_weight
                        * H1_pre_field[h1_index(i, 1+component, u)];
                }
            }
        }
    }

    H1_adj = std::move(H1_pre_adj);
}

void MACE::compute_R1(
    const int num_nodes,
    std::span<const int> node_types,
    std::span<const int> num_neigh,
    std::span<const int> neigh_types,
    std::span<const double> r)
{
    if (spl_set_1.empty())
        throw std::runtime_error(
            "MACE compact radial cache is not prepared; call prepare_active_types first.");
    const int num_spl = spl_set_1[0]->num_splines;
    R1.resize(r.size()*num_spl);
    R1_deriv.resize(R1.size());
    int ij = 0;
    for (int i=0; i<num_nodes; ++i) {
        const int type_i = node_types[i];
        for (int j=0; j<num_neigh[i]; ++j) {
            const int type_j = neigh_types[ij];
            const int type_ij = radial_pair_index(type_i, type_j);
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
            const int type_ij = radial_pair_index(type_i, type_j);
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
            const int type_ij = radial_pair_index(type_i, type_j);
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
            const int type_ij = radial_pair_index(type_i, type_j);
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

void MACE::compute_H1_product(
    const int num_nodes)
{
    H1.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto M0_il = M0.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_product_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                M0_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                H1_il,
                num_channels);
        }
    }
}

void MACE::compute_H1_linear_up(
    const int num_nodes)
{
    auto H1_before_linear_up = H1;
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_in_il = H1_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
            auto H1_il = H1.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                H1_in_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                H1_il,
                num_channels);
        }
    }
}

void MACE::reverse_H1_linear_up(
    const int num_nodes)
{
    auto H1_adj_before_linear_up = H1_adj;
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj_before_linear_up.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_linear_up_weights.data()+l*num_channels*num_channels;
            auto H1_pre_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                H1_adj_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                H1_pre_adj_il,
                num_channels);
        }
    }
}

void MACE::reverse_H1_product(
    const int num_nodes)
{
    M0_adj.resize(M0.size());
    for (int i=0; i<num_nodes; ++i) {
        for (int l=0; l<=L_max; ++l) {
            const auto H1_adj_il = H1_adj.data()+(i*num_LM+l*l)*num_channels;
            const auto weights_l = H1_product_weights.data()+l*num_channels*num_channels;
            auto M0_adj_il = M0_adj.data()+(i*num_LM+l*l)*num_channels;
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                2*l+1,
                num_channels,
                num_channels,
                1.0,
                H1_adj_il,
                num_channels,
                weights_l,
                num_channels,
                0.0,
                M0_adj_il,
                num_channels);
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
            const int type_ij = radial_pair_index(type_i, type_j);
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
            const int type_ij = radial_pair_index(type_i, type_j);
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
            const int type_ij = radial_pair_index(type_i, type_j);
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

    // Radial representation
    const int format_version = file.value("symmetrix_format_version", 1);
    uses_compact_radial = format_version == 2;
    if (uses_compact_radial) {
        if (file.value("radial_representation", std::string()) != "compact")
            throw std::invalid_argument("Symmetrix format version 2 requires compact radial data.");
        compact_radial_model = std::make_unique<CompactRadialModel>(
            file.at("compact_radial").dump(), atomic_numbers, r_cut);
        type_to_active.assign(atomic_numbers.size(), -1);
    } else if (format_version == 1) {
        const double spl_h = file["radial_spline_h"];
        const double spl_min = file.value("radial_spline_min", 0.0);
        auto spl_values_0 = file["radial_spline_values_0"].get<std::vector<std::vector<std::vector<double>>>>();
        auto spl_derivs_0 = file["radial_spline_derivs_0"].get<std::vector<std::vector<std::vector<double>>>>();
        for (int i=0; i<spl_values_0.size(); ++i)
            spl_set_0.push_back(std::make_unique<CubicSplineSet>(
                spl_h, spl_values_0[i], spl_derivs_0[i], spl_min));
        auto spl_values_1 = file["radial_spline_values_1"].get<std::vector<std::vector<std::vector<double>>>>();
        auto spl_derivs_1 = file["radial_spline_derivs_1"].get<std::vector<std::vector<std::vector<double>>>>();
        for (int i=0; i<spl_values_1.size(); ++i)
            spl_set_1.push_back(std::make_unique<CubicSplineSet>(
                spl_h, spl_values_1[i], spl_derivs_1[i], spl_min));
        active_types.resize(atomic_numbers.size());
        std::iota(active_types.begin(), active_types.end(), 0);
        type_to_active = active_types;
        active_atomic_numbers = atomic_numbers;
    } else {
        throw std::invalid_argument("Unsupported Symmetrix model format version.");
    }

    // H0
    H0_weights = file["H0_weights"].get<std::vector<double>>();

    // A0
    A0_weights = file["A0_weights"].get<std::vector<std::vector<std::vector<double>>>>();

    // A0 scaling
    A0_scaled = file["A0_scaled"].get<bool>();
    if (A0_scaled && !uses_compact_radial) {
        const double A0_spline_h = file["A0_spline_h"];
        const double A0_spline_min = file.value("A0_spline_min", 0.0);
        auto A0_spline_values = file["A0_spline_values"].get<std::vector<std::vector<double>>>();
        auto A0_spline_derivs = file["A0_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A0_spline_values.size(); ++i)
            A0_splines.push_back(CubicSpline(
                A0_spline_h, A0_spline_values[i], A0_spline_derivs[i], A0_spline_min));
    }
    if (uses_compact_radial && A0_scaled != compact_radial_model->has_A0())
        throw std::invalid_argument("Compact radial A0 network does not match A0_scaled.");

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
    H1_product_weights = file.value("H1_product_weights", std::vector<double>{});
    H1_linear_up_weights = file.value("H1_linear_up_weights", std::vector<double>{});

    // MACEField H1 coupling
    has_field_coupling = file.value("has_field_coupling", false);
    field_feats_scalar_to_vector_path_weight = 0.0;
    field_feats_vector_to_scalar_path_weight = 0.0;
    field_linear_scalar_path_weight = 0.0;
    field_linear_vector_path_weight = 0.0;
    if (has_field_coupling) {
        auto field_couplings = file["field_couplings"];
        if (field_couplings.size() != 1)
            throw std::runtime_error("MACEField JSON must contain exactly one field coupling.");
        auto coupling = field_couplings[0];
        const auto hidden_irreps = std::to_string(num_channels)
            + "x0e+" + std::to_string(num_channels) + "x1o";
        if (coupling["field_feats_irreps_in1"].get<std::string>() != hidden_irreps
            || coupling["field_feats_irreps_in2"].get<std::string>() != "1x1o"
            || coupling["field_feats_irreps_out"].get<std::string>() != hidden_irreps
            || coupling["field_linear_irreps_in"].get<std::string>() != hidden_irreps
            || coupling["field_linear_irreps_out"].get<std::string>() != hidden_irreps)
            throw std::runtime_error("Unsupported MACEField field coupling irreps.");
        if (L_max != 1)
            throw std::runtime_error("MACEField JSON field coupling currently requires L_max == 1.");
        if (H1_product_weights.size() != H1_weights.size()
            || H1_linear_up_weights.size() != H1_weights.size())
            throw std::runtime_error("MACEField JSON must contain split H1 product and linear_up weights.");

        field_feats_weight = coupling["field_feats_weight"].get<std::vector<double>>();
        field_feats_output_mask = coupling["field_feats_output_mask"].get<std::vector<double>>();
        field_linear_weight = coupling["field_linear_weight"].get<std::vector<double>>();
        field_linear_bias = coupling["field_linear_bias"].get<std::vector<double>>();
        field_linear_output_mask = coupling["field_linear_output_mask"].get<std::vector<double>>();

        const int channel_pairs = num_channels*num_channels;
        if (field_feats_weight.size() != 2*channel_pairs
            || field_linear_weight.size() != 2*channel_pairs
            || field_feats_output_mask.size() != 4*num_channels
            || field_linear_output_mask.size() != 4*num_channels
            || !field_linear_bias.empty())
            throw std::runtime_error("MACEField JSON field coupling tensor sizes are unsupported.");
        for (double mask_value : field_feats_output_mask)
            if (mask_value != 1.0)
                throw std::runtime_error("Unsupported MACEField field_feats output mask.");
        for (double mask_value : field_linear_output_mask)
            if (mask_value != 1.0)
                throw std::runtime_error("Unsupported MACEField field_linear output mask.");

        auto field_feats_instructions = coupling["field_feats_instructions"];
        if (field_feats_instructions.size() != 2)
            throw std::runtime_error("MACEField field_feats must contain exactly two instructions.");
        for (const auto& instruction : field_feats_instructions) {
            if (instruction["connection_mode"].get<std::string>() != "uvw")
                throw std::runtime_error("MACEField field_feats only supports uvw instructions.");
            auto path_shape = instruction["path_shape"].get<std::vector<int>>();
            if (path_shape != std::vector<int>{num_channels, 1, num_channels})
                throw std::runtime_error("Unsupported MACEField field_feats path shape.");
            const int i_in1 = instruction["i_in1"].get<int>();
            const int i_in2 = instruction["i_in2"].get<int>();
            const int i_out = instruction["i_out"].get<int>();
            if (i_in1 == 0 && i_in2 == 0 && i_out == 1)
                field_feats_scalar_to_vector_path_weight = instruction["path_weight"].get<double>();
            else if (i_in1 == 1 && i_in2 == 0 && i_out == 0)
                field_feats_vector_to_scalar_path_weight = instruction["path_weight"].get<double>();
            else
                throw std::runtime_error("Unsupported MACEField field_feats instruction.");
        }

        auto field_linear_instructions = coupling["field_linear_instructions"];
        if (field_linear_instructions.size() != 2)
            throw std::runtime_error("MACEField field_linear must contain exactly two instructions.");
        for (const auto& instruction : field_linear_instructions) {
            auto path_shape = instruction["path_shape"].get<std::vector<int>>();
            if (path_shape != std::vector<int>{num_channels, num_channels})
                throw std::runtime_error("Unsupported MACEField field_linear path shape.");
            const int i_in = instruction["i_in"].get<int>();
            const int i_out = instruction["i_out"].get<int>();
            if (i_in == 0 && i_out == 0)
                field_linear_scalar_path_weight = instruction["path_weight"].get<double>();
            else if (i_in == 1 && i_out == 1)
                field_linear_vector_path_weight = instruction["path_weight"].get<double>();
            else
                throw std::runtime_error("Unsupported MACEField field_linear instruction.");
        }
    }

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
    if (A1_scaled && !uses_compact_radial) {
        const double A1_spline_h = file["A1_spline_h"];
        const double A1_spline_min = file.value("A1_spline_min", 0.0);
        auto A1_spline_values = file["A1_spline_values"].get<std::vector<std::vector<double>>>();
        auto A1_spline_derivs = file["A1_spline_derivs"].get<std::vector<std::vector<double>>>();
        for (int i=0; i<A1_spline_values.size(); ++i)
            A1_splines.push_back(CubicSpline(
                A1_spline_h, A1_spline_values[i], A1_spline_derivs[i], A1_spline_min));
    }
    if (uses_compact_radial && A1_scaled != compact_radial_model->has_A1())
        throw std::invalid_argument("Compact radial A1 network does not match A1_scaled.");

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
