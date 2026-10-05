#pragma once

#include <tuple>
#include <vector>

class CubicSpline {

public:

CubicSpline(double r_min,
            double r_max,
            std::vector<double> nodal_values,
            std::vector<double> nodal_derivs);

auto evaluate(double r) -> double;
auto evaluate_deriv(double r) -> std::tuple<double,double>;
auto evaluate_deriv_divided(double r) -> std::tuple<double,double>;

private:

double r_min;
double r_max;
double h;
int num_intervals;
std::vector<double> c;

auto generate_coefficients(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs)
    -> std::vector<double>;
};
