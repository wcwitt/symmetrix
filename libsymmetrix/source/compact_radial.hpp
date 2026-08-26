#pragma once

#include <memory>
#include <string>
#include <vector>

#include "multilayer_perceptron.hpp"

struct RadialSplineData {
    std::vector<std::vector<double>> values;
    std::vector<std::vector<double>> derivatives;
};

struct CompactRadialPairTables {
    RadialSplineData R0;
    RadialSplineData R1;
    RadialSplineData A0;
    RadialSplineData A1;
};

class CompactRadialModel {
public:
    struct Network {
        std::vector<int> shape;
        std::vector<std::vector<double>> weights;
        double activation_scale;
        bool tanh_square;
        std::unique_ptr<MultilayerPerceptron> mlp;
    };

    CompactRadialModel(
        std::string definition_json,
        std::vector<int> atomic_numbers,
        double r_cut);

    CompactRadialPairTables materialize_pair(int type_i, int type_j);

    double spline_h() const;
    double spline_min() const;
    int num_spline_points() const;
    bool has_A0() const;
    bool has_A1() const;

private:
    double grid_min;
    int spline_points;
    double cutoff_r_max;
    int cutoff_p;
    double h;
    std::vector<int> atomic_numbers;
    std::vector<double> bessel_weights;
    double bessel_prefactor;

    bool use_agnesi;
    double agnesi_a;
    double agnesi_q;
    double agnesi_p;
    std::vector<double> covalent_radii;

    Network R0_network;
    Network R1_network;
    std::unique_ptr<Network> A0_network;
    std::unique_ptr<Network> A1_network;

    std::vector<double> radial_features(int type_i, int type_j) const;
    RadialSplineData evaluate_network(Network& network, const std::vector<double>& features);
    std::vector<double> spline_derivatives(const std::vector<double>& values) const;
};
