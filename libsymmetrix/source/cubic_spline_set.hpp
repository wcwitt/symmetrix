#pragma once

#include <span>
#include <vector>

class CubicSplineSet {

public:

CubicSplineSet(double h,
               std::vector<std::vector<double>> nodal_values,
               std::vector<std::vector<double>> nodal_derivs);

void evaluate(double r, std::span<double> values);
void evaluate_derivs(double r, std::span<double> values, std::span<double> derivs);

// TODO: protect this with accessor
int num_splines;

private:

double h;
int num_nodes;
std::vector<double> c;

};
