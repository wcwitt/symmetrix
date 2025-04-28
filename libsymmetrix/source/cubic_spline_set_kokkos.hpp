#pragma once

#include <span>
#include <vector>

#include <Kokkos_Core.hpp>

class CubicSplineSetKokkos {

public:
double h;
int num_nodes;
int num_splines;
Kokkos::View<double***,Kokkos::LayoutRight> c;

CubicSplineSetKokkos(double h,
                     Kokkos::View<double**> nodal_values,
                     Kokkos::View<double**> nodal_derivs);
CubicSplineSetKokkos(double h,
                     std::vector<std::vector<double>> nodal_values,
                     std::vector<std::vector<double>> nodal_derivs);
static void initialize_coefficients(double h_local,
                                    int num_nodes_local,
                                    int num_splines_local,
                                    Kokkos::View<double**> nodal_values,
                                    Kokkos::View<double**> nodal_derivs,
                                    Kokkos::View<double***> c_local);
void evaluate(double r, Kokkos::View<double*> values);
void evaluate_derivs(double r, Kokkos::View<double*> values, Kokkos::View<double*> derivs) const;
void evaluate_derivs(Kokkos::View<const double*> r,
                     Kokkos::View<double**,Kokkos::LayoutRight> values,
                     Kokkos::View<double**,Kokkos::LayoutRight> derivs) const;

// these routines accept std::span and translate into Kokkos views
void evaluate(double r, std::span<double> values);
void evaluate_derivs(double r, std::span<double> values, std::span<double> derivs);
};
