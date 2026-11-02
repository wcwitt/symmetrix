#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cubic_spline_set.hpp"

template <typename Precision>
CubicSplineSet<Precision>::CubicSplineSet(
    double r_min,
    double r_max,
    std::vector<std::vector<double>> nodal_values,
    std::vector<std::vector<double>> nodal_derivs)
    : r_min(r_min), r_max(r_max)
{
    if (not std::isfinite(r_min) or not std::isfinite(r_max) or r_min >= r_max)
        throw std::invalid_argument("CubicSplineSet requires finite r_min < r_max.");
    if (nodal_values.empty() or nodal_values.size() != nodal_derivs.size())
        throw std::invalid_argument("CubicSplineSet requires at least one spline and matching derivatives.");
    for (int j=0; j<nodal_values.size(); ++j)
        if (nodal_values[j].size() < 2 or nodal_values[j].size() != nodal_values[0].size()
                or nodal_values[j].size() != nodal_derivs[j].size())
            throw std::invalid_argument("CubicSplineSet requires consistent nodal arrays with at least two values.");
    num_nodes = nodal_values[0].size();
    num_splines = nodal_values.size();
    h = (r_max - r_min) / (num_nodes - 1);
    if (h <= 0 or not std::isfinite(h))
        throw std::invalid_argument("CubicSplineSet requires positive finite spacing.");
    c = std::vector<Precision>(4*num_splines*(num_nodes-1), 0.0);
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

template <typename Precision>
void CubicSplineSet<Precision>::evaluate(
    double r,
    std::span<Precision> values)
{
    if (std::isnan(r) or r < r_min or r > r_max)
        throw std::invalid_argument("Out of bounds in CubicSplineSet::evaluate. r=" + std::to_string(r));
    if (values.size() < num_splines)
        throw std::invalid_argument("CubicSplineSet::evaluate requires space for all splines.");
    const int i = std::clamp(static_cast<int>((r - r_min) / h), 0, num_nodes - 2);
    const Precision x = r - (r_min + h*i);
    const Precision xx = x*x;
    const Precision xxx = xx*x;
    const Precision* c0 = c.data() + 4*i*num_splines;
    const Precision* c1 = c0 + num_splines;
    const Precision* c2 = c1 + num_splines;
    const Precision* c3 = c2 + num_splines;
    for (int j=0; j<num_splines; ++j)
        values[j] = c0[j] + x*c1[j] + xx*c2[j] + xxx*c3[j];
}

template <typename Precision>
void CubicSplineSet<Precision>::evaluate_derivs(double r,
                                                std::span<Precision> values,
                                                std::span<Precision> derivs)
{
    if (std::isnan(r) or r < r_min or r > r_max)
        throw std::invalid_argument("Out of bounds in CubicSplineSet::evaluate_derivs. r=" + std::to_string(r));
    if (values.size() < num_splines or derivs.size() < num_splines)
        throw std::invalid_argument("CubicSplineSet::evaluate_derivs requires space for all splines.");
    const int i = std::clamp(static_cast<int>((r - r_min) / h), 0, num_nodes - 2);
    const Precision x = r - (r_min + h*i);
    const Precision xx = x*x;
    const Precision xxx = xx*x;
    const Precision two_x = 2*x;
    const Precision three_xx = 3*xx;
    const Precision* c0 = c.data() + 4*i*num_splines;
    const Precision* c1 = c0 + num_splines;
    const Precision* c2 = c1 + num_splines;
    const Precision* c3 = c2 + num_splines;
    for (int j=0; j<num_splines; ++j) {
        values[j] = c0[j] + x*c1[j] + xx*c2[j] + xxx*c3[j];
        derivs[j] = c1[j] + two_x*c2[j] + three_xx*c3[j];
    }
}

template class CubicSplineSet<double>;
template class CubicSplineSet<float>;
