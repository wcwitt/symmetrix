#pragma once

#include <tuple>
#include <vector>

#include "Kokkos_Core.hpp"

class CubicSplineKokkos {

public:

CubicSplineKokkos(double h,
                  std::vector<double> nodal_values,
                  std::vector<double> nodal_derivs);
CubicSplineKokkos(double h,
                  Kokkos::View<double*> nodal_values,
                  Kokkos::View<double*> nodal_derivs);
            
double evaluate(double r);
std::tuple<double,double> evaluate_deriv(double r);
std::tuple<double,double> evaluate_deriv_divided(double r);

double h;
Kokkos::View<double*> c;
size_t num_coeffs;

void generate_coefficients(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs);
void generate_coefficients(
    double h,
    Kokkos::View<double*> nodal_values,
    Kokkos::View<double*> nodal_derivs);
};
