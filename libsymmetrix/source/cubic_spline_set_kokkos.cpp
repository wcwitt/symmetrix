#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cubic_spline_set_kokkos.hpp"
#include "tools_kokkos.hpp"

CubicSplineSetKokkos::CubicSplineSetKokkos(
    double r_min,
    double r_max,
    std::vector<std::vector<double>> nodal_values,
    std::vector<std::vector<double>> nodal_derivs)
    : r_min(r_min), r_max(r_max)
{
    if (not std::isfinite(r_min) or not std::isfinite(r_max) or r_min >= r_max)
        throw std::invalid_argument("CubicSplineSetKokkos requires finite r_min < r_max.");
    if (nodal_values.empty() or nodal_values.size() != nodal_derivs.size())
        throw std::invalid_argument("CubicSplineSetKokkos requires at least one spline and matching derivatives.");
    for (int j=0; j<nodal_values.size(); ++j)
        if (nodal_values[j].size() < 2 or nodal_values[j].size() != nodal_values[0].size()
                or nodal_values[j].size() != nodal_derivs[j].size())
            throw std::invalid_argument("CubicSplineSetKokkos requires consistent nodal arrays with at least two values.");
    num_splines = nodal_values.size();
    num_nodes = nodal_values[0].size();
    h = (r_max - r_min) / (num_nodes - 1);
    if (h <= 0 or not std::isfinite(h))
        throw std::invalid_argument("CubicSplineSetKokkos requires positive finite spacing.");

    c = Kokkos::View<double***,Kokkos::LayoutRight>("coeffs", num_nodes-1, 4, num_splines);
    auto h_c = Kokkos::create_mirror_view(c);

    for (int i=0; i<num_nodes-1; ++i) {
        for (int j=0; j<num_splines; ++j) {
            h_c(i,0,j) = nodal_values[j][i];
            h_c(i,1,j) = nodal_derivs[j][i];
            h_c(i,2,j) = (-3*nodal_values[j][i] -2*h*nodal_derivs[j][i]
                          + 3*nodal_values[j][i+1] - h*nodal_derivs[j][i+1]) / (h*h);
            h_c(i,3,j) = (2*nodal_values[j][i] + h*nodal_derivs[j][i]
                          - 2*nodal_values[j][i+1] + h*nodal_derivs[j][i+1]) / (h*h*h);
        }
    }
    //copy it back to device
    Kokkos::deep_copy(c,h_c);
}

struct InitializeCoefficientsFunctor {
    double h;
    Kokkos::View<double**> nodal_values;
    Kokkos::View<double**> nodal_derivs;
    Kokkos::View<double***,Kokkos::LayoutRight> c;

    InitializeCoefficientsFunctor(double h_, Kokkos::View<double**> nodal_values_,
                                  Kokkos::View<double**> nodal_derivs_, Kokkos::View<double***,Kokkos::LayoutRight> c_)
        : h(h_), nodal_values(nodal_values_), nodal_derivs(nodal_derivs_), c(c_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j) const {
        c(i, 0, j) = nodal_values(j, i);
        c(i, 1, j) = nodal_derivs(j, i);
        c(i, 2, j) = (-3 * nodal_values(j, i) - 2 * h * nodal_derivs(j, i)
                    + 3 * nodal_values(j, i + 1) - h * nodal_derivs(j, i + 1)) / (h * h);
        c(i, 3, j) = (2 * nodal_values(j, i) + h * nodal_derivs(j, i)
                    - 2 * nodal_values(j, i + 1) + h * nodal_derivs(j, i + 1)) / (h * h * h);
    }
};


CubicSplineSetKokkos::CubicSplineSetKokkos(
    double r_min,
    double r_max,
    Kokkos::View<double**> nodal_values,
    Kokkos::View<double**> nodal_derivs)
    : r_min(r_min), r_max(r_max)
{
    if (not std::isfinite(r_min) or not std::isfinite(r_max) or r_min >= r_max)
        throw std::invalid_argument("CubicSplineSetKokkos requires finite r_min < r_max.");
    if (nodal_values.extent(0) < 1 or nodal_values.extent(1) < 2
            or nodal_values.extent(0) != nodal_derivs.extent(0)
            or nodal_values.extent(1) != nodal_derivs.extent(1))
        throw std::invalid_argument("CubicSplineSetKokkos requires matching nodal arrays with at least two values.");
    num_splines = nodal_values.extent(0);
    num_nodes = nodal_values.extent(1);
    h = (r_max - r_min) / (num_nodes - 1);
    if (h <= 0 or not std::isfinite(h))
        throw std::invalid_argument("CubicSplineSetKokkos requires positive finite spacing.");
    Kokkos::realloc(c, num_nodes-1, 4, num_splines);

    // Create and use the functor for initialization
    InitializeCoefficientsFunctor functor(h, nodal_values, nodal_derivs, c);
    Kokkos::parallel_for("InitializeCoefficients",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_nodes - 1, num_splines}),
                            functor);
    Kokkos::fence();
}

void CubicSplineSetKokkos::evaluate(
    double r,
    Kokkos::View<double*> values)
{
    if (std::isnan(r) or r < r_min or r > r_max)
        throw std::invalid_argument("Out of bounds in CubicSplineSetKokkos::evaluate. r=" + std::to_string(r));
    if (values.size() < num_splines)
        throw std::invalid_argument("CubicSplineSetKokkos::evaluate requires space for all splines.");
    const int i = std::clamp(static_cast<int>((r - r_min) / h), 0, num_nodes - 2);
    const double x = r - (r_min + h*i);
    const double xx = x*x;
    const double xxx = xx*x;

    // make a local copy
    //Kokkos::View<double***> local_c("local_c", c.extent(0), c.extent(1), c.extent(2));
    //Kokkos::deep_copy(local_c,this->c);

    // Parallel computation of values
    Kokkos::parallel_for("EvaluateSpline", num_splines, KOKKOS_CLASS_LAMBDA(const int j) {
        values(j) = c(i, 0, j) + c(i, 1, j) * x + c(i, 2, j) * xx + c(i, 3, j) * xxx;
    });
    Kokkos::fence();

}

void CubicSplineSetKokkos::evaluate_derivs(double r,
                                           Kokkos::View<double*> values,
                                           Kokkos::View<double*> derivs) const
{
    if (std::isnan(r) or r < r_min or r > r_max)
        throw std::invalid_argument("Out of bounds in CubicSplineSetKokkos::evaluate_derivs. r=" + std::to_string(r));
    if (values.size() < num_splines or derivs.size() < num_splines)
        throw std::invalid_argument("CubicSplineSetKokkos::evaluate_derivs requires space for all splines.");
    const int i = std::clamp(static_cast<int>((r - r_min) / h), 0, num_nodes - 2);
    const double x = r - (r_min + h*i);
    const double xx = x*x;
    const double xxx = xx*x;
    const double two_x = 2*x;
    const double three_xx = 3*xx;

    // Parallel computation of values and derivatives
    Kokkos::parallel_for("evaluate_derivs", num_splines, KOKKOS_CLASS_LAMBDA(const int j) {
        double c0 = c(i, 0, j);  // Coefficient of x^0
        double c1 = c(i, 1, j);  // Coefficient of x^1
        double c2 = c(i, 2, j);  // Coefficient of x^2
        double c3 = c(i, 3, j);  // Coefficient of x^3

        // Calculate the value at x
        values(j) = c0 + c1 * x + c2 * xx + c3 * xxx;

        // Calculate the derivative at x
        derivs(j) = c1 + two_x * c2 + three_xx * c3;
    });
    Kokkos::fence();
}

void CubicSplineSetKokkos::evaluate_derivs(Kokkos::View<const double*> r,
                                           Kokkos::View<double**,Kokkos::LayoutRight> values,
                                           Kokkos::View<double**,Kokkos::LayoutRight> derivs) const
{
    check_spline_bounds(r, r_min, r_max, "CubicSplineSetKokkos::evaluate_derivs");
    if (values.extent(0) < r.size() or derivs.extent(0) < r.size()
            or values.extent(1) < num_splines or derivs.extent(1) < num_splines)
        throw std::invalid_argument("CubicSplineSetKokkos::evaluate_derivs requires space for all splines.");
    const auto c = this->c;
    const auto h = this->h;
    const auto r_min = this->r_min;
    const int num_intervals = num_nodes - 1;
    const auto num_splines = this->num_splines;

    Kokkos::parallel_for(
        "CubicSplineSet::evaluate_derivs",
        // TODO: empirically, this vector_length appears to be the best choice for nvidia,
        //       provided num_splines is sufficiently large, and from what i understand
        //       this parameter is essentially ignored on cpu, making it okay there too.
        //       but it would be nice to have something smarter.
        Kokkos::TeamPolicy<>(r.size(), 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int ii = Kokkos::clamp(static_cast<int>((r(i) - r_min)/h), 0, num_intervals - 1);
            const double x = r(i) - (r_min + h*ii);
            const double xx = x*x;
            const double xxx = xx*x;
            const double two_x = 2*x;
            const double three_xx = 3*xx;
            // compute values
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    values(i,j) = c(ii,0,j);
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    values(i,j) += c(ii,1,j)*x;
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    values(i,j) += c(ii,2,j)*xx;
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    values(i,j) += c(ii,3,j)*xxx;
                });
            // compute derivs
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    derivs(i,j) = c(ii,1,j);
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    derivs(i,j) += c(ii,2,j)*two_x;
                });
            team_member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team_member, num_splines),
                [&] (const int j) {
                    derivs(i,j) += c(ii,3,j)*three_xx;
                });
         });
}


void CubicSplineSetKokkos::evaluate(
    double r,
    std::span<double> values)
{
    if (values.size() < num_splines)
        throw std::invalid_argument("CubicSplineSetKokkos::evaluate requires space for all splines.");
    Kokkos::View<double*> d_values("values", num_splines);
    evaluate(r, d_values);
    auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_values);
    std::copy_n(h_values.data(), num_splines, values.begin());
}

void CubicSplineSetKokkos::evaluate_derivs(
    double r,
    std::span<double> values,
    std::span<double> derivs)
{
    if (values.size() < num_splines or derivs.size() < num_splines)
        throw std::invalid_argument("CubicSplineSetKokkos::evaluate_derivs requires space for all splines.");
    Kokkos::View<double*> d_values("values", num_splines), d_derivs("derivs", num_splines);
    evaluate_derivs(r, d_values, d_derivs);
    auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_values);
    auto h_derivs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_derivs);
    std::copy_n(h_values.data(), num_splines, values.begin());
    std::copy_n(h_derivs.data(), num_splines, derivs.begin());
}
