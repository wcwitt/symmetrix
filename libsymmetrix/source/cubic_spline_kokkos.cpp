#include <stdexcept>
#include<iostream>
#include<cmath>
#include "cubic_spline_kokkos.hpp"

CubicSplineKokkos::CubicSplineKokkos(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs)
    : h(h), num_coeffs(4*(nodal_values.size() - 1))
{
    c = Kokkos::View<double*>("coeffs",num_coeffs);
    generate_coefficients(h, nodal_values, nodal_derivs);
}

CubicSplineKokkos::CubicSplineKokkos(
    double h,
    Kokkos::View<double*> nodal_values,
    Kokkos::View<double*> nodal_derivs)
    : h(h), num_coeffs(4*(nodal_values.size() - 1))
{
    c = Kokkos::View<double*>("coeffs",num_coeffs);
    generate_coefficients(h, nodal_values, nodal_derivs);
}

double CubicSplineKokkos::evaluate(double r)
{
    const int i = static_cast<int>(r / h);
    // TODO: something better with this bounds checking
    if (i < 0 || i >= num_coeffs / 4)
        throw std::invalid_argument("Out of bounds in CubicSplineKokkos::evaluate.");
    
    const double x = r - h * i;
    const double xx = x * x;
    const double xxx = xx * x;
    const int i4 = 4 * i;

    auto h_c = Kokkos::create_mirror_view(c);
    
    double ret = 0;
    const double c0 = h_c(i4);
    const double c1 = h_c(i4 + 1);
    const double c2 = h_c(i4 + 2);
    const double c3 = h_c(i4 + 3);
    ret = c0 + c1 * x + c2 * xx + c3 * xxx;

    return ret;
}

std::tuple<double, double> CubicSplineKokkos::evaluate_deriv(double r)
{
    const int i = static_cast<int>(r / h);
    // TODO: something better with this bounds checking
    if (i < 0 || i > num_coeffs / 4)
        throw std::invalid_argument("Out of bounds in CubicSplineKokkos::evaluate_deriv.");

    const double x = r - h * i;
    const double xx = x * x;
    const double xxx = xx * x;
    const int i4 = 4 * i;

    auto h_c = Kokkos::create_mirror_view(c);

    const double c0 = h_c(i4);
    const double c1 = h_c(i4 + 1);
    const double c2 = h_c(i4 + 2);
    const double c3 = h_c(i4 + 3);

    double spline_value = c0 + c1 * x + c2 * xx + c3 * xxx;
    double spline_derivative = c1 + 2 * c2 * x + 3 * c3 * xx;

    return {spline_value, spline_derivative};
}

std::tuple<double,double> CubicSplineKokkos::evaluate_deriv_divided(double r)
{
    const int i = static_cast<int>(r / h);
    // TODO: something better with this bounds checking
    if (i<0 or i> num_coeffs)
        throw std::invalid_argument("Out of bounds in CubicSplineKokkos::evaluate_deriv.");

    const double x = r - h*i;
    const double xx = x*x;
    const double xxx = xx*x;
    const int i4 = 4*i;

    auto h_c = Kokkos::create_mirror_view(c);

    const double c0 = h_c(i4);
    const double c1 = h_c(i4+1);
    const double c2=h_c(i4+2);
    const double c3=h_c(i4+3);
    
    return {c0 + c1*x + c2*xx + c3*xxx, (c1 + 2*c2*x + 3*c3*xx) / r};
}

void CubicSplineKokkos::generate_coefficients(
    double h,
    std::vector<double> nodal_values,
    std::vector<double> nodal_derivs)
{
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> nodal_values_host(
        nodal_values.data(), nodal_values.size());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> nodal_derivs_host(
        nodal_derivs.data(), nodal_derivs.size());

    Kokkos::View<double*> nodal_values_device("nodal_values_device", nodal_values.size());
    Kokkos::View<double*> nodal_derivs_device("nodal_derivs_device", nodal_derivs.size());

    Kokkos::deep_copy(nodal_values_device, nodal_values_host);
    Kokkos::deep_copy(nodal_derivs_device, nodal_derivs_host);

    generate_coefficients(h, nodal_values_device, nodal_derivs_device);
}

void CubicSplineKokkos::generate_coefficients(
    double h,
    Kokkos::View<double*> nodal_values,
    Kokkos::View<double*> nodal_derivs)
{
    auto c = this->c;
    Kokkos::parallel_for("CubicSpline_generate_coefficients_parallel_for", nodal_values.size()-1, KOKKOS_LAMBDA (const int i){
        c(4*i) = nodal_values(i);
        c(4*i+1) = nodal_derivs(i);
        c(4*i+2) = (-3*nodal_values(i) - 2*h*nodal_derivs(i) + 3*nodal_values(i+1) - h*nodal_derivs(i+1)) / (h*h);
        c(4*i+3) = (2*nodal_values(i) + h*nodal_derivs(i) - 2*nodal_values(i+1) + h*nodal_derivs(i+1)) / (h*h*h);
    });
}
