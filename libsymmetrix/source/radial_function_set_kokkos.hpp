#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

template <typename Precision>
class RadialFunctionSetKokkos
{
public:

    RadialFunctionSetKokkos();
    RadialFunctionSetKokkos(
        double r_min,
        double r_max,
        std::vector<std::vector<std::vector<double>>> node_values,
        std::vector<std::vector<std::vector<double>>> node_derivatives);
    void evaluate(
        Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> i_list,
        Kokkos::View<const int*> neigh_types,
        Kokkos::View<const double*> r,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R_deriv) const;

    KOKKOS_INLINE_FUNCTION int type_ij(const int type_i, const int type_j) const
    {
        return (type_i <= type_j)
            ? type_i*(2*num_types-type_i-1)/2 + type_j
            : type_j*(2*num_types-type_j-1)/2 + type_i;
    }

    KOKKOS_INLINE_FUNCTION int interval(const double r) const
    {
        return Kokkos::clamp(static_cast<int>((r - r_min)/h), 0, num_nodes - 2);
    }

    KOKKOS_INLINE_FUNCTION Precision offset(const double r, const int n) const
    {
        return r - (r_min + h*n);
    }

    KOKKOS_INLINE_FUNCTION Precision value(const int type, const int n, const Precision x, const int f) const
    {
        return coefficients(type,n,0,f) + x*(coefficients(type,n,1,f) + x*(coefficients(type,n,2,f) + x*coefficients(type,n,3,f)));
    }

    KOKKOS_INLINE_FUNCTION void value_deriv(const int type, const int n, const Precision x, const int f, Precision& R, Precision& R_deriv) const
    {
        const Precision c1 = coefficients(type,n,1,f);
        const Precision c2 = coefficients(type,n,2,f);
        const Precision c3 = coefficients(type,n,3,f);
        R = coefficients(type,n,0,f) + x*(c1 + x*(c2 + x*c3));
        R_deriv = c1 + x*(2*c2 + x*3*c3);
    }

private:

    double r_min;
    double r_max;
    double h;
    int num_types;
    int num_edge_types;
    int num_functions;
    int num_nodes;
    Kokkos::View<const Precision****,Kokkos::LayoutRight> coefficients;
};
