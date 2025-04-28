#pragma once

#include <tuple>
#include <vector>

class MultilayerPerceptron {

public:

MultilayerPerceptron();
MultilayerPerceptron(
    std::vector<int> shape,
    std::vector<std::vector<double>> weights,
    double activation_scale_factor = 1.0);

auto evaluate(std::vector<double> input) -> std::vector<double>;
auto evaluate_gradient(std::vector<double> input) -> std::tuple<std::vector<double>,std::vector<double>>;
auto evaluate_batch(std::vector<double> input, const int batch_size) -> std::vector<double>;
auto evaluate_gradient_batch(std::vector<double> input, const int batch_size) -> std::tuple<std::vector<double>,std::vector<double>>;

private:

std::vector<int> shape;
std::vector<std::vector<double>> weights;
double activation_scale_factor;

std::vector<std::vector<double>> node_values;
std::vector<std::vector<double>> node_derivs;
std::vector<std::vector<double>> node_activation_derivs;

};
