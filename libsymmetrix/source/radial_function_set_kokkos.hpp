#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

template <typename Precision>
class RadialFunctionSetKokkos
{
public:
    
    RadialFunctionSetKokkos();
    RadialFunctionSetKokkos(
        double h,
        std::vector<std::vector<std::vector<double>>> node_values,
        std::vector<std::vector<std::vector<double>>> node_derivatives);
    void evaluate(
        const int num_nodes,
        Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,
        Kokkos::View<const int*> neigh_types,
        Kokkos::View<const double*> r,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R_deriv) const;
    
private:
    
    double h;
    int num_edge_types;
    int num_functions;
    int num_nodes;
    Kokkos::View<const Precision****,Kokkos::LayoutRight> coefficients;
};
