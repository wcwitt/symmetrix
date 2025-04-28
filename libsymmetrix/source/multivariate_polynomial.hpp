#pragma once

#include <array>
#include <tuple>
#include <vector>

class MultivariatePolynomial
{

public:

MultivariatePolynomial(int num_variables, 
                       std::vector<double> coefficients,
                       std::vector<std::vector<int>> monomials);

auto evaluate(const std::vector<double>& x) -> double;
auto evaluate_gradient(const std::vector<double>& x) -> std::tuple<double,std::vector<double>>;
auto evaluate_batch(const std::vector<double>& x, const int batch_size) -> std::tuple<std::vector<double>,std::vector<double>>;

// polynomial specification
int num_variables;
std::vector<double> coefficients;
std::vector<std::vector<int>> monomials;

// graph-related variables
int num_auxiliary_nodes;
std::vector<std::vector<int>> nodes;
std::vector<std::array<int,2>> edges;
std::vector<double> node_coefficients;
std::vector<double> node_values;
std::vector<double> node_adjoints;

// used during recursive evaluation
void initialize_forward_pass(const std::vector<double>& x);
void forward_pass();
void initialize_backward_pass();
void backward_pass();
auto extract_gradient_from_graph() -> std::vector<double>;

// used during recursive evaluation with batching
void batched_initialize_forward_pass(const std::vector<double>& x, const int batch_size);
void batched_forward_pass(const int batch_size);
void batched_initialize_backward_pass(const int batch_size);
void batched_backward_pass(const int batch_size);
auto batched_extract_gradient_from_graph(const int batch_size) -> std::vector<double>;

// non-recursive evaluation, used for testing
auto evaluate_simple(const std::vector<double>& x) -> double;
auto evaluate_gradient_simple(const std::vector<double>& x) -> std::tuple<double,std::vector<double>>;

};
