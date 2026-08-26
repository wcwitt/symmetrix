#pragma once

#include <Kokkos_Core.hpp>

class MultilayerPerceptronKokkos
{
public:

MultilayerPerceptronKokkos();
MultilayerPerceptronKokkos(
    std::vector<int> shape,
    std::vector<std::vector<double>> weights,
    double activation_scale_factor = 1.0);
~MultilayerPerceptronKokkos();

void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> x, Kokkos::View<double*,Kokkos::LayoutRight> f);
void evaluate_gradient(Kokkos::View<const double**,Kokkos::LayoutRight> x, Kokkos::View<double*,Kokkos::LayoutRight> f, Kokkos::View<double**,Kokkos::LayoutRight> g);
void evaluate_gradient_directional(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot);

private:

Kokkos::View<int*,Kokkos::SharedSpace> shape;
Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> weights;
double activation_scale;

Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> node_values;
Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> node_derivatives;
Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> node_value_dots;
Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> node_derivative_dots;

};
