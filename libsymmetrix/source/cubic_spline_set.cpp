#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cubic_spline_set.hpp"

CubicSplineSet::CubicSplineSet(
    double h,
    std::vector<std::vector<double>> nodal_values,
    std::vector<std::vector<double>> nodal_derivs)
{
    if (h<=0 or not std::isfinite(h))
        throw std::invalid_argument("CubicSplineSet requires positive finite spacing.");
    if (nodal_values.size()<1 or nodal_values.size()!=nodal_derivs.size())
        throw std::invalid_argument("CubicSplineSet requires at least one spline and matching derivatives.");
    for (int j=0; j<nodal_values.size(); ++j)
        if (nodal_values[j].size()<2 or nodal_values[j].size()!=nodal_values[0].size()
                or nodal_values[j].size()!=nodal_derivs[j].size())
            throw std::invalid_argument("CubicSplineSet requires at least two values per spline, consistent across splines, with matching derivatives.");
    this->h = h;
    num_nodes = nodal_values[0].size();
    num_splines = nodal_values.size();
    c = std::vector<double>(4*num_splines*(num_nodes-1), 0.0);
    for (int i=0; i<num_nodes-1; ++i) {
        for (int j=0; j<num_splines; ++j) {
            c[(4*i)*num_splines+j] = nodal_values[j][i];
            c[(4*i+1)*num_splines+j] = nodal_derivs[j][i];
            c[(4*i+2)*num_splines+j] = (-3*nodal_values[j][i] -2*h*nodal_derivs[j][i]
                                        + 3*nodal_values[j][i+1] - h*nodal_derivs[j][i+1]) / (h*h);
            c[(4*i+3)*num_splines+j] = (2*nodal_values[j][i] + h*nodal_derivs[j][i]
                                        - 2*nodal_values[j][i+1] + h*nodal_derivs[j][i+1]) / (h*h*h);
        }
    }
}

void CubicSplineSet::evaluate(
    double r,
    std::span<double> values)
{
    if (r<0 or r>h*(num_nodes-1) or std::isnan(r))
        throw std::invalid_argument("Out of bounds in CubicSplineSet::evaluate. r=" + std::to_string(r));
    const int i = std::clamp(static_cast<int>(r / h), 0, num_nodes-2);
    const double x = r - h*i;
    const double xx = x*x;
    const double xxx = xx*x;
    double* c_i = c.data() + 4*i*num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] = c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += c_i[j]*x;
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += c_i[j]*xx;
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += c_i[j]*xxx;
}

void CubicSplineSet::evaluate_derivs(double r,
                                     std::span<double> values,
                                     std::span<double> derivs)
{
    if (r<0 or r>h*(num_nodes-1) or std::isnan(r))
        throw std::invalid_argument("Out of bounds in CubicSplineSet::evaluate_derivs. r=" + std::to_string(r));
    const int i = std::clamp(static_cast<int>(r / h), 0, num_nodes-2);
    const double x = r - h*i;
    const double xx = x*x;
    const double xxx = xx*x;
    const double two_x = 2*x;
    const double three_xx = 3*xx;
    // compute values
    double* c_i = c.data() + 4*i*num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] = c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += x*c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += xx*c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] += xxx*c_i[j];
    // compute derivs
    c_i = c.data() + (4*i+1)*num_splines;
    for (int j=0; j<num_splines; ++j)
        derivs[j] = c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        derivs[j] += two_x*c_i[j];
    c_i += num_splines;
    for (int j=0; j<num_splines; ++j)
        derivs[j] += three_xx*c_i[j];
}
