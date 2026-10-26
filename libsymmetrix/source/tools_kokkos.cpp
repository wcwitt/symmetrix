#include <algorithm>
#include <numeric>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "tools_kokkos.hpp"

void _init_kokkos()
{
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(1);
    settings.set_disable_warnings(true);
    settings.set_map_device_id_by("random");
    Kokkos::initialize(settings);
}

void _finalize_kokkos()
{
    Kokkos::finalize();
}

bool _kokkos_is_initialized()
{
    return Kokkos::is_initialized();
}

void check_spline_bounds(Kokkos::View<const double*> r, double r_min, double r_max, const char* name)
{
    int invalid = 0;
    Kokkos::parallel_reduce("spline_bounds", r.size(), KOKKOS_LAMBDA (const int i, int& count) {
        if (r(i) < r_min or r(i) > r_max or Kokkos::isnan(r(i)))
            ++count;
    }, invalid);
    if (invalid)
        throw std::invalid_argument(std::string("Out of bounds in ") + name + ".");
}

struct InitView {
  explicit InitView(view_type _v) : m_view(_v) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    m_view(i) = -(i + 1);
    //m_view(i) = (i + 1);
  }

 private:
  view_type m_view;
};

struct ModifyView {
  explicit ModifyView(view_type _v) : m_view(_v) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    m_view(i) *= 2;
  }

 private:
  view_type m_view;
};

using exec_space = typename view_type::traits::execution_space;

view_type generate_view(size_t n)
{
    if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
    }
    std::cerr << "[user-bindings]> Generating View..." << std::flush;
    view_type _v("user_view", n);
    Kokkos::RangePolicy<exec_space, int> range(0, n);
    Kokkos::parallel_for("generate_view", range, InitView{_v});
    std::cerr << " Done." << std::endl;
    return _v;
}

void modify_view(view_type _v) {
  std::cerr << "[user-bindings]> Modifying View..." << std::flush;
  Kokkos::RangePolicy<exec_space, int> range(0, _v.extent(0));
  Kokkos::parallel_for("modify_view", range, ModifyView{_v});
  std::cerr << " Done." << std::endl;
}
