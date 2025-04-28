#include <cmath>

#include "cblas.hpp"

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
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        cblas_dgemv(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE trans
            shape[l+1],               // const MKL_INT m
            shape[l],                 // const MKL_INT n
            1.0,                      // const double alpha
            weights[l].data(),        // const double *a
            shape[l],                 // const MKL_INT lda
            node_values[l].data(),    // const double *x
            1,                        // const MKL_INT incx
            0.0,                      // const double beta
            node_values[l+1].data(),  // double *y
            1);                       // const MKL_INT incy
        for (int i=0; i<shape[l+1]; ++i) {
            const double x = node_values[l+1][i];
            node_values[l+1][i] = activation_scale_factor*x/(1.0+std::exp(-x));
        }
    }
    // Evaluate final layer (no nonlinearity)
    cblas_dgemv(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE trans
        shape.end()[-1],               // const MKL_INT m
        shape.end()[-2],               // const MKL_INT n
        1.0,                           // const double alpha
        weights.back().data(),         // const double *a
        shape.end()[-2],               // const MKL_INT lda
        node_values.end()[-2].data(),  // const double *x
        1,                             // const MKL_INT incx
        0.0,                           // const double beta
        node_values.end()[-1].data(),  // double *y
        1);                            // const MKL_INT incy
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
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        cblas_dgemv(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE trans
            shape[l+1],               // const MKL_INT m
            shape[l],                 // const MKL_INT n
            1.0,                      // const double alpha
            weights[l].data(),        // const double *a
            shape[l],                 // const MKL_INT lda
            node_values[l].data(),    // const double *x
            1,                        // const MKL_INT incx
            0.0,                      // const double beta
            node_values[l+1].data(),  // double *y
            1);                       // const MKL_INT incy
        for (int i=0; i<shape[l+1]; ++i) {
            const double x = node_values[l+1][i];
            const double sigmoid = 1.0/(1.0+std::exp(-x));
            node_values[l+1][i] = activation_scale_factor*x*sigmoid;
            node_activation_derivs[l+1][i] = activation_scale_factor*sigmoid
                + activation_scale_factor*x*sigmoid*(1-sigmoid);
        }
    }
    // Evaluate final layer (no nonlinearity)
    cblas_dgemv(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE trans
        shape.end()[-1],               // const MKL_INT m
        shape.end()[-2],               // const MKL_INT n
        1.0,                           // const double alpha
        weights.back().data(),         // const double *a
        shape.end()[-2],               // const MKL_INT lda
        node_values.end()[-2].data(),  // const double *x
        1,                             // const MKL_INT incx
        0.0,                           // const double beta
        node_values.end()[-1].data(),  // double *y
        1);                            // const MKL_INT incy
    // Differentiate backwards
    node_derivs.end()[-2] = weights.back();
    for (int l=shape.size()-3; l>=0; --l) {
        for (int i=0; i<shape.back(); ++i) {
            for (int j=0; j<shape[l+1]; ++j) {
                node_derivs[l+1][i*shape[l+1]+j] *= node_activation_derivs[l+1][j];
            }
        }
        cblas_dgemm(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasNoTrans,             // const CBLAS_TRANSPOSE transb
            shape.back(),             // const MKL_INT m
            shape[l],                 // const MKL_INT n
            shape[l+1],               // const MKL_INT k
            1.0,                      // const double alpha
            node_derivs[l+1].data(),  // const double *a
            shape[l+1],               // const MKL_INT lda
            weights[l].data(),        // const double *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const double beta
            node_derivs[l].data(),    // double *c
            shape[l]);                // const MKL_INT ldc
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
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        cblas_dgemm(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasTrans,               // const CBLAS_TRANSPOSE transb
            batch_size,               // const MKL_INT m
            shape[l+1],               // const MKL_INT n
            shape[l],                 // const MKL_INT k
            1.0,                      // const double alpha
            node_values[l].data(),    // const double *a
            shape[l],                 // const MKL_INT lda
            weights[l].data(),        // const double *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const double beta
            node_values[l+1].data(),  // double *c
            shape[l+1]);              // const MKL_INT ldc
        for (int i=0; i<node_values[l+1].size(); ++i) {
            const double x = node_values[l+1][i];
            node_values[l+1][i] = activation_scale_factor*x/(1.0+std::exp(-x));
        }
    }
    // Evaluate final layer (no nonlinearity)
    cblas_dgemm(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE transa
        CblasTrans,                    // const CBLAS_TRANSPOSE transb
        batch_size,                    // const MKL_INT m
        shape.end()[-1],               // const MKL_INT n
        shape.end()[-2],               // const MKL_INT k
        1.0,                           // const double alpha
        node_values.end()[-2].data(),  // const double *a
        shape.end()[-2],               // const MKL_INT lda
        weights.end()[-1].data(),      // const double *b
        shape.end()[-2],               // const MKL_INT ldb
        0.0,                           // const double beta
        node_values.end()[-1].data(),  // double *c
        shape.end()[-1]);              // const MKL_INT ldc
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
    // Evaluate layers
    for (int l=0; l<shape.size()-2; ++l) {
        cblas_dgemm(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasTrans,               // const CBLAS_TRANSPOSE transb
            batch_size,               // const MKL_INT m
            shape[l+1],               // const MKL_INT n
            shape[l],                 // const MKL_INT k
            1.0,                      // const double alpha
            node_values[l].data(),    // const double *a
            shape[l],                 // const MKL_INT lda
            weights[l].data(),        // const double *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const double beta
            node_values[l+1].data(),  // double *c
            shape[l+1]);              // const MKL_INT ldc
        for (int i=0; i<node_values[l+1].size(); ++i) {
            const double x = node_values[l+1][i];
            const double sigmoid = 1.0/(1.0+std::exp(-x));
            node_values[l+1][i] = activation_scale_factor*x*sigmoid;
            node_activation_derivs[l+1][i] = activation_scale_factor*sigmoid
                + activation_scale_factor*x*sigmoid*(1-sigmoid);
        }
    }
    // Evaluate final layer (no nonlinearity)
    cblas_dgemm(
        CblasRowMajor,                 // const CBLAS_LAYOUT Layout
        CblasNoTrans,                  // const CBLAS_TRANSPOSE transa
        CblasTrans,                    // const CBLAS_TRANSPOSE transb
        batch_size,                    // const MKL_INT m
        shape.end()[-1],               // const MKL_INT n
        shape.end()[-2],               // const MKL_INT k
        1.0,                           // const double alpha
        node_values.end()[-2].data(),  // const double *a
        shape.end()[-2],               // const MKL_INT lda
        weights.end()[-1].data(),      // const double *b
        shape.end()[-2],               // const MKL_INT ldb
        0.0,                           // const double beta
        node_values.end()[-1].data(),  // double *c
        shape.end()[-1]);              // const MKL_INT ldc
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
        cblas_dgemm(
            CblasRowMajor,            // const CBLAS_LAYOUT Layout
            CblasNoTrans,             // const CBLAS_TRANSPOSE transa
            CblasNoTrans,             // const CBLAS_TRANSPOSE transb
            batch_size*shape.back(),  // const MKL_INT m
            shape[l],                 // const MKL_INT n
            shape[l+1],               // const MKL_INT k
            1.0,                      // const double alpha
            node_derivs[l+1].data(),  // const double *a
            shape[l+1],               // const MKL_INT lda
            weights[l].data(),        // const double *b
            shape[l],                 // const MKL_INT ldb
            0.0,                      // const double beta
            node_derivs[l].data(),    // double *c
            shape[l]);                // const MKL_INT ldc
    }
    return {node_values.back(), node_derivs[0]};
}
