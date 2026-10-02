#include <vector>
#include <algorithm>

#include "cubic_spline_set.hpp"

CubicSplineSet::CubicSplineSet(
    double h,
    std::vector<std::vector<double>> nodal_values,
    std::vector<std::vector<double>> nodal_derivs)
{
    // TODO: sanitize input
    this->h = h;
    num_pieces = nodal_values[0].size() - 1;
    num_splines = nodal_values.size();
    c = std::vector<double>(4*num_splines*num_pieces, 0.0);
    for (int i=0; i<num_pieces; ++i) {
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
    if (r<0 or r>h*num_pieces or std::isnan(r))
        throw std::invalid_argument("Out of bounds in CubicSplineSet::evaluate. r=" + std::to_string(r));
    const int i = std::clamp(static_cast<int>(r / h), 0, num_pieces - 1);
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
    if (r<0 or r>h*num_pieces or std::isnan(r))
        throw std::invalid_argument("Out of bounds in CubicSplineSet::evaluate_derivs. r=" + std::to_string(r));
    const int i = std::clamp(static_cast<int>(r / h), 0, num_pieces - 1);
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
