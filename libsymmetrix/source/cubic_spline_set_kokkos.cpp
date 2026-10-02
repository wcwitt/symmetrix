#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cubic_spline_set_kokkos.hpp"

CubicSplineSetKokkos::CubicSplineSetKokkos(
    double h,
    std::vector<std::vector<double>> nodal_values,
    std::vector<std::vector<double>> nodal_derivs)
{
    if (h<=0 or not std::isfinite(h))
        throw std::invalid_argument("CubicSplineSetKokkos requires positive finite spacing.");
    if (nodal_values.size()<1 or nodal_values.size()!=nodal_derivs.size())
        throw std::invalid_argument("CubicSplineSetKokkos requires at least one spline and matching derivatives.");
    for (int j=0; j<nodal_values.size(); ++j)
        if (nodal_values[j].size()<2 or nodal_values[j].size()!=nodal_values[0].size()
                or nodal_values[j].size()!=nodal_derivs[j].size())
            throw std::invalid_argument("CubicSplineSetKokkos requires at least two values per spline, consistent across splines, with matching derivatives.");
    this->h = h;
    num_splines = nodal_values.size();
    num_nodes = nodal_values[0].size();

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
    double h,
    Kokkos::View<double**> nodal_values,
    Kokkos::View<double**> nodal_derivs)
{
    if (h<=0 or not std::isfinite(h))
        throw std::invalid_argument("CubicSplineSetKokkos requires positive finite spacing.");
    if (nodal_values.extent(0)<1 or nodal_values.extent(1)<2
            or nodal_values.extent(0)!=nodal_derivs.extent(0)
            or nodal_values.extent(1)!=nodal_derivs.extent(1))
        throw std::invalid_argument("CubicSplineSetKokkos requires at least one spline with at least two values, and matching derivatives.");
    this->h = h;
    num_splines = nodal_values.extent(0);
    num_nodes = nodal_values.extent(1);
    Kokkos::realloc(c, num_nodes-1, num_splines);

    // Create and use the functor for initialization
    InitializeCoefficientsFunctor functor(h, nodal_values, nodal_derivs, c);
    Kokkos::parallel_for("InitializeCoefficients",
                            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_nodes - 1, num_splines}),
                            functor);
    //Kokkos::fence();
}

void CubicSplineSetKokkos::evaluate(
    double r,
    Kokkos::View<double*> values)
{
    if (r<0 or r>h*(num_nodes-1) or std::isnan(r))
        throw std::invalid_argument("Out of bounds in CubicSplineSetKokkos::evaluate. r=" + std::to_string(r));
    const int i = std::clamp(static_cast<int>(r / h), 0, num_nodes-2);
    const double x = r - h*i;
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
    if (r<0 or r>h*(num_nodes-1) or std::isnan(r))
        throw std::invalid_argument("Out of bounds in CubicSplineSetKokkos::evaluate_derivs. r=" + std::to_string(r));
    const int i = std::clamp(static_cast<int>(r / h), 0, num_nodes-2);
    const double x = r - h*i;
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
    const auto c = this->c;
    const auto h = this->h;
    const auto num_splines = this->num_splines;
    const int num_intervals = this->num_nodes - 1;
    const double r_max = h*num_intervals;

    // bounds checking (cannot throw from device code, so reduce first)
    int num_out_of_bounds = 0;
    Kokkos::parallel_reduce(
        "CubicSplineSet::evaluate_derivs::bounds",
        r.size(),
        KOKKOS_LAMBDA (const int i, int& count) {
            const double r_i = r(i);
            if (r_i<0 or r_i>r_max or Kokkos::isnan(r_i))
                count += 1;
        },
        num_out_of_bounds);
    Kokkos::fence();
    if (num_out_of_bounds>0)
        throw std::invalid_argument("Out of bounds in CubicSplineSetKokkos::evaluate_derivs for "
                                    + std::to_string(num_out_of_bounds) + " distances.");

    Kokkos::parallel_for(
        "CubicSplineSet::evaluate_derivs",
        // TODO: empirically, this vector_length appears to be the best choice for nvidia,
        //       provided num_splines is sufficiently large, and from what i understand
        //       this parameter is essentially ignored on cpu, making it okay there too.
        //       but it would be nice to have something smarter.
        Kokkos::TeamPolicy<>(r.size(), 1, 32),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int ii = Kokkos::clamp(static_cast<int>(r(i)/h), 0, num_intervals-1);
            const double x = r(i) - h*ii;
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
    // create test device side view
    Kokkos::View<double*> test_view("test_view",values.size());

    // test view's host mirror
    auto h_test_view = Kokkos::create_mirror_view(test_view);

    // view to copy the values
    Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>> input_value_view(values.data(),values.size());

    // copy input values to host mirror
    Kokkos::deep_copy(h_test_view,input_value_view);

    //copy it back to device side view
    Kokkos::deep_copy(test_view,h_test_view);

    evaluate(
        r,
        test_view);

    // copy values back to host mirror
    Kokkos::deep_copy(h_test_view,test_view);

    // copy back to values unmanaged view
    Kokkos::deep_copy(input_value_view,h_test_view);
    // evaluate(
    //     r,
    //     Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>>(values.data(),values.size()));
}

void CubicSplineSetKokkos::evaluate_derivs(
    double r,
    std::span<double> values,
    std::span<double> derivs)
{
    // create test device side view
    Kokkos::View<double*> test_view_values("test_view_values",values.size());
    Kokkos::View<double*> test_view_derivs("test_view_derivs",derivs.size());

    // test view's host mirror
    auto h_test_view_values = Kokkos::create_mirror_view(test_view_values);
    auto h_test_view_derivs = Kokkos::create_mirror_view(test_view_derivs);

    // view to copy the values
    Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>> input_values_view(values.data(),values.size());
    Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>> input_derivs_view(derivs.data(),derivs.size());


    // copy input values to host mirror
    Kokkos::deep_copy(h_test_view_values,input_values_view);
    Kokkos::deep_copy(h_test_view_derivs,input_derivs_view);

    // copy to device side view
    Kokkos::deep_copy(test_view_values,h_test_view_values);
    Kokkos::deep_copy(test_view_values,h_test_view_values);

    evaluate_derivs(
        r,
        test_view_values,
        test_view_derivs);

     // copy values back to host mirror
    Kokkos::deep_copy(h_test_view_values,test_view_values);
    Kokkos::deep_copy(h_test_view_derivs,test_view_derivs);

    // copy back to values unmanaged view
    Kokkos::deep_copy(input_values_view,h_test_view_values);
    Kokkos::deep_copy(input_derivs_view,h_test_view_derivs);
}
