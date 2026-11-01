#include <cmath>

#include "multilayer_perceptron.hpp"

MultilayerPerceptron::MultilayerPerceptron()
{
    // TODO: add sanity checks to default constructor
}

MultilayerPerceptron::MultilayerPerceptron(
    std::vector<int> shape,
    std::vector<std::vector<double>> weights,
    double activation_scale_factor)
    : shape(shape),
      weights(weights),
      activation_scale_factor(activation_scale_factor)
{
    // TODO: check and sanitize input
    // TODO: double check that something sensible happens here for i==shape.size()-1
    for (int i=0; i<shape.size(); ++i) {
        node_values.push_back(std::vector<double>(shape[i]));
        node_derivs.push_back(std::vector<double>(shape.back()*shape[i]));
        node_activation_derivs.push_back(std::vector<double>(shape[i]));
    }
}

auto MultilayerPerceptron::evaluate(
    std::vector<double> input
    )-> std::vector<double>
{
    // TODO: Check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers (no nonlinearity in the final one)
    for (int l=0; l<shape.size()-1; ++l) {
        for (int i=0; i<shape[l+1]; ++i) {
            double x = 0.0;
            for (int j=0; j<shape[l]; ++j)
                x += weights[l][i*shape[l]+j] * node_values[l][j];
            node_values[l+1][i] = (l < shape.size()-2) ? activation_scale_factor*x/(1.0+std::exp(-x)) : x;
        }
    }
    return node_values.back();
}

auto  MultilayerPerceptron::evaluate_gradient(
    std::vector<double> input
    )-> std::tuple<std::vector<double>,std::vector<double>>
{
    // TODO: check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(shape[i]);
        node_activation_derivs[i].resize(shape[i]);
        node_derivs[i].resize(shape.back()*shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers (no nonlinearity in the final one)
    for (int l=0; l<shape.size()-1; ++l) {
        for (int i=0; i<shape[l+1]; ++i) {
            double x = 0.0;
            for (int j=0; j<shape[l]; ++j)
                x += weights[l][i*shape[l]+j] * node_values[l][j];
            const double sigmoid = 1.0/(1.0+std::exp(-x));
            node_values[l+1][i] = (l < shape.size()-2) ? activation_scale_factor*x*sigmoid : x;
            node_activation_derivs[l+1][i] = activation_scale_factor*sigmoid
                + activation_scale_factor*x*sigmoid*(1-sigmoid);
        }
    }
    // Differentiate backwards
    node_derivs.end()[-2] = weights.back();
    for (int l=shape.size()-3; l>=0; --l) {
        for (int i=0; i<shape.back(); ++i) {
            for (int j=0; j<shape[l+1]; ++j) {
                node_derivs[l+1][i*shape[l+1]+j] *= node_activation_derivs[l+1][j];
            }
        }
        for (int i=0; i<shape.back(); ++i)
            for (int j=0; j<shape[l]; ++j) {
                double d = 0.0;
                for (int p=0; p<shape[l+1]; ++p)
                    d += node_derivs[l+1][i*shape[l+1]+p] * weights[l][p*shape[l]+j];
                node_derivs[l][i*shape[l]+j] = d;
            }
    }
    return {node_values.back(), node_derivs[0]};
}

auto MultilayerPerceptron::evaluate_batch(
    std::vector<double> input,
    const int batch_size
    )-> std::vector<double>
{
    // TODO: Check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(batch_size*shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers (no nonlinearity in the final one)
    for (int l=0; l<shape.size()-1; ++l) {
        for (int b=0; b<batch_size; ++b)
            for (int i=0; i<shape[l+1]; ++i) {
                double x = 0.0;
                for (int j=0; j<shape[l]; ++j)
                    x += weights[l][i*shape[l]+j] * node_values[l][b*shape[l]+j];
                node_values[l+1][b*shape[l+1]+i] = (l < shape.size()-2) ? activation_scale_factor*x/(1.0+std::exp(-x)) : x;
            }
    }
    return node_values.end()[-1];
}

auto MultilayerPerceptron::evaluate_gradient_batch(
    std::vector<double> input,
    const int batch_size
    )-> std::tuple<std::vector<double>,std::vector<double>>
{
    // TODO: Check/sanitize input
    // Reshape node arrays and send input to nodes
    for (int i=0; i<shape.size(); ++i) {
        node_values[i].resize(batch_size*shape[i]);
        node_activation_derivs[i].resize(batch_size*shape[i]);
        node_derivs[i].resize(batch_size*shape.back()*shape[i]);
    }
    std::copy(input.begin(), input.end(), node_values[0].begin());
    // Evaluate layers (no nonlinearity in the final one)
    for (int l=0; l<shape.size()-1; ++l) {
        for (int b=0; b<batch_size; ++b)
            for (int i=0; i<shape[l+1]; ++i) {
                double x = 0.0;
                for (int j=0; j<shape[l]; ++j)
                    x += weights[l][i*shape[l]+j] * node_values[l][b*shape[l]+j];
                const double sigmoid = 1.0/(1.0+std::exp(-x));
                node_values[l+1][b*shape[l+1]+i] = (l < shape.size()-2) ? activation_scale_factor*x*sigmoid : x;
                node_activation_derivs[l+1][b*shape[l+1]+i] = activation_scale_factor*sigmoid
                    + activation_scale_factor*x*sigmoid*(1-sigmoid);
            }
    }
    // Differentiate backwards
    for (int i=0; i<batch_size; ++i) {
        std::copy(weights.back().begin(), weights.back().end(), node_derivs.end()[-2].begin()+i*weights.back().size());
    }
    for (int l=shape.size()-3; l>=0; --l) {
        for (int i=0; i<batch_size; ++i) {
            for (int j=0; j<shape.back(); ++j) {
                for (int p=0; p<shape[l+1]; ++p) {
                    node_derivs[l+1][i*shape.back()*shape[l+1]+j*shape[l+1]+p] *= node_activation_derivs[l+1][i*shape[l+1]+p];
                }
            }
        }
        for (int i=0; i<batch_size*shape.back(); ++i)
            for (int j=0; j<shape[l]; ++j) {
                double d = 0.0;
                for (int p=0; p<shape[l+1]; ++p)
                    d += node_derivs[l+1][i*shape[l+1]+p] * weights[l][p*shape[l]+j];
                node_derivs[l][i*shape[l]+j] = d;
            }
    }
    return {node_values.back(), node_derivs[0]};
}
