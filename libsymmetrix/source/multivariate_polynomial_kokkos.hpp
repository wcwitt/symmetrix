#pragma once

#include <vector>
#include <Kokkos_Core.hpp>

class MultivariatePolynomialKokkos
{

public:

MultivariatePolynomialKokkos(int num_variables, 
                             std::vector<double> coefficients,
                             std::vector<std::vector<int>> monomials);
double evaluate(const Kokkos::View<const double*>& x);
double evaluate_gradient(const Kokkos::View<const double*>& x, Kokkos::View<double*>& g);
double evaluate_simple(const Kokkos::View<const double*>& x);
double evaluate_gradient_simple(const Kokkos::View<const double*>& x, Kokkos::View<double*>& g);

// polynomial specification
int num_variables;
Kokkos::View<double*> coefficients;
Kokkos::View<int**> monomials;

// graph-related variables
int num_auxiliary_nodes;
Kokkos::View<int**,Kokkos::LayoutRight> nodes;
Kokkos::View<int**,Kokkos::LayoutRight> edges;
Kokkos::View<double*> node_coefficients;
Kokkos::View<double*> node_values;
Kokkos::View<double*> node_adjoints;

// used during recursive evaluation
void initialize_forward_pass(const Kokkos::View<const double*>& x);
void forward_pass();
void initialize_backward_pass();
void backward_pass();
void extract_gradient_from_graph(Kokkos::View<double*>& g);

};
