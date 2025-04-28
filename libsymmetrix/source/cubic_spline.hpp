#pragma once

#include <tuple>
#include <vector>

class CubicSpline {

public:

CubicSpline(double h,
            std::vector<double> nodal_values,
            std::vector<double> nodal_derivs);

auto evaluate(double r) -> double;
auto evaluate_deriv(double r) -> std::tuple<double,double>;
auto evaluate_deriv_divided(double r) -> std::tuple<double,double>;

private:

double h;
std::vector<double> c;

auto generate_coefficients(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs)
    -> std::vector<double>;
};
