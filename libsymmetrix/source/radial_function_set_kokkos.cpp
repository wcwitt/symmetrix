#include <vector>

#include "radial_function_set_kokkos.hpp"


RadialFunctionSetKokkos::RadialFunctionSetKokkos()
{
}


RadialFunctionSetKokkos::RadialFunctionSetKokkos(
    double h,
    std::vector<std::vector<std::vector<double>>> node_values,
    std::vector<std::vector<std::vector<double>>> node_derivatives)
{
    // TODO: sanitize input
    this->h = h;
    num_edge_types = node_values.size();
    num_functions = node_values[0].size();
    num_nodes = node_values[0][0].size();

    auto c = Kokkos::View<double****, Kokkos::LayoutRight>(
        "coefficients", num_edge_types, num_nodes-1, 4, num_functions);
    auto h_c = Kokkos::create_mirror_view(c);
    for (int a=0; a<num_edge_types; ++a) {
        for (int i=0; i<num_nodes-1; ++i) {
            for (int j=0; j<num_functions; ++j) {
                h_c(a,i,0,j) = node_values[a][j][i];
                h_c(a,i,1,j) = node_derivatives[a][j][i];
                h_c(a,i,2,j) = (-3*node_values[a][j][i] -2*h*node_derivatives[a][j][i]
                                + 3*node_values[a][j][i+1] - h*node_derivatives[a][j][i+1]) / (h*h);
                h_c(a,i,3,j) = (2*node_values[a][j][i] + h*node_derivatives[a][j][i]
                                - 2*node_values[a][j][i+1] + h*node_derivatives[a][j][i+1]) / (h*h*h);
            }
        }
    }
    Kokkos::deep_copy(c, h_c);
    coefficients = c;
}


// This older, cleaner version of the function lacks edge-type dependence
#if 0
void RadialFunctionSetKokkos::evaluate(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    Kokkos::View<double**,Kokkos::LayoutRight> R,
    Kokkos::View<double**,Kokkos::LayoutRight> R_deriv) const
{
    const auto h = this->h;
    const auto num_functions = this->num_functions;
    const auto c = this->coefficients;

    Kokkos::parallel_for(
        "RadialFunctionSetKokkos::evaluate",
        // TODO: empirically, this vector_length appears to be the best choice,
        //       provided num_functions is sufficiently large, and from what i understand
        //       this parameter is essentially ignored on cpu, making it okay there too.
        //       but it would be nice to have something smarter.
        Kokkos::TeamPolicy<>(r.size(), 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int j = static_cast<int>(r(i)/h); // TODO: bounds checking?
            const double x = r(i) - h*j;
            const double xx = x*x;
            const double xxx = xx*x;
            const double two_x = 2*x;
            const double three_xx = 3*xx;
            // compute function values
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R(i,k) = c(0,j,0,k);
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R(i,k) += c(0,j,1,k)*x;
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R(i,k) += c(0,j,2,k)*xx;
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R(i,k) += c(0,j,3,k)*xxx;
                });
            // compute derivatives
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R_deriv(i,k) = c(0,j,1,k);
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R_deriv(i,k) += c(0,j,2,k)*two_x;
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    R_deriv(i,k) += c(0,j,3,k)*three_xx;
                });
        });
        Kokkos::fence();
}
#endif


void RadialFunctionSetKokkos::evaluate(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r,
    Kokkos::View<double**,Kokkos::LayoutRight> R,
    Kokkos::View<double**,Kokkos::LayoutRight> R_deriv) const
{
    const auto h = this->h;
    const auto num_functions = this->num_functions;
    const auto c = this->coefficients;

    // TODO: shouldn't need all this
    // Build i_list
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
    Kokkos::View<int*> i_list("i_list", r.size());
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

    const int num_unique_types = (std::sqrt(8*num_edge_types+1)-1)/2;

    Kokkos::parallel_for(
        "RadialFunctionSetKokkos::evaluate",
        // TODO: empirically, this vector_length appears to be the best choice,
        //       provided num_functions is sufficiently large, and from what i understand
        //       this parameter is essentially ignored on cpu, making it okay there too.
        //       but it would be nice to have something smarter.
        Kokkos::TeamPolicy<>(r.size(), 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int ij = team_member.league_rank();
            // determine edge type
            const int type_i = node_types(i_list(ij));
            const int type_j = neigh_types(ij);
            const int type_ij = (type_i <= type_j)
                ? type_i*(2*num_unique_types-type_i-1)/2 + type_j
                : type_j*(2*num_unique_types-type_j-1)/2 + type_i;
            // compute x, x^2, x^3
            const int n = static_cast<int>(r(ij)/h); // TODO: bounds checking?
            const double x = r(ij) - h*n;
            const double xx = x*x;
            const double xxx = xx*x;
            const double two_x = 2*x;
            const double three_xx = 3*xx;
            // compute function values
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_functions),
                [&] (const int k) {
                    const double c0 = c(type_ij,n,0,k);
                    const double c1 = c(type_ij,n,1,k); 
                    const double c2 = c(type_ij,n,2,k); 
                    const double c3 = c(type_ij,n,3,k); 
                    R(ij,k) = c0 + c1*x + c2*xx + c3*xxx;
                    R_deriv(ij,k) = c1 + c2*two_x + c3*three_xx;
                });
        });
        Kokkos::fence();
}
