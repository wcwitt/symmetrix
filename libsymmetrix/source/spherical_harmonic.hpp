#pragma once

#include <complex>
#include <vector>

std::complex<double> sph_harm(int l, int m, double theta, double phi);

std::complex<double> sph_harm_xyz(int l, int m, double x, double y, double z);

double real_sph_harm(int l, int m, double theta, double phi);

double real_sph_harm_xyz(int l, int m, double x, double y, double z);

std::complex<double> sphericart_sph_harm(int l, int m, double x, double y, double z);

double sphericart_real_sph_harm(int l, int m, double x, double y, double z);

std::vector<double> sphericart_real(int l_max, std::vector<double> xyz);

std::vector<std::complex<double>> sphericart_complex(
    int l_max,
    std::vector<double> xyz);
