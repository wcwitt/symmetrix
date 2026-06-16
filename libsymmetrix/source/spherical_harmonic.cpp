#include <cmath>

#include "sphericart.hpp"

#include "spherical_harmonic.hpp"

// Associated Legendre polynomial P_l^m(x), computed via the standard
// upward recurrence (no Condon-Shortley phase, matching std::assoc_legendre).
// Provided here because libc++ (e.g. on macOS) does not implement the C++17
// mathematical special functions, unlike libstdc++.
static double assoc_legendre(int l, int m, double x) {
    if (m < 0 || m > l) return 0.0;
    double pmm = 1.0;
    if (m > 0) {
        const double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= fact * somx2;
            fact += 2.0;
        }
    }
    if (l == m) return pmm;
    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

std::complex<double> sph_harm(int l, int m, double polar, double azimuth) {
    if (m >=0) {
        return std::pow(-1,m)
            * std::sqrt((2*l+1)/(4*M_PI)*std::tgamma(1+l-m)/std::tgamma(1+l+m))
            * assoc_legendre(l,m,std::cos(polar))
            * std::exp(std::complex<double>(0,1)*double(m)*azimuth);
    } else {
        return std::pow(-1,m)
            * std::sqrt((2*l+1)/(4*M_PI)*std::tgamma(1+l-m)/std::tgamma(1+l+m))
            * std::pow(-1,m)*std::tgamma(1+l+m)/std::tgamma(1+l-m)*assoc_legendre(l,-m,std::cos(polar))
            * std::exp(std::complex<double>(0,1)*double(m)*azimuth);
    }
}

std::complex<double> sph_harm_xyz(int l, int m, double x, double y, double z) {
    double r = std::sqrt(x*x+y*y+z*z);
    double polar = std::acos(z/r);
    double azimuth = ((y>0)-(y<0)) * std::acos(x/std::sqrt(x*x+y*y));
    return sph_harm(l, m, polar, azimuth);
}

double real_sph_harm(int l, int m, double polar, double azimuth) {
    if (m < 0) {
        return sqrt(2) * std::pow(-1,m) * sph_harm(l,-m,polar,azimuth).imag();
    } else if (m == 0) {
        return sph_harm(l,m,polar,azimuth).real();
    } else {
        return sqrt(2) * std::pow(-1,m) * sph_harm(l,m,polar,azimuth).real();
    }
}

double real_sph_harm_xyz(int l, int m, double x, double y, double z) {
    double r = std::sqrt(x*x+y*y+z*z);
    double polar = std::acos(z/r);
    double azimuth = ((y>0)-(y<0)) * std::acos(x/std::sqrt(x*x+y*y));
    return real_sph_harm(l, m, polar, azimuth);
}

std::complex<double> sphericart_sph_harm(int l, int m, double x, double y, double z) {
    if (m < 0) {
        return 1.0/sqrt(2)*(sphericart_real_sph_harm(l,-m,x,y,z)
            - std::complex<double>{0.0,1.0}*sphericart_real_sph_harm(l,m,x,y,z));
    } else if (m == 0) {
        return sphericart_real_sph_harm(l,m,x,y,z);
    } else {
        return std::pow(-1,m)/sqrt(2)*(sphericart_real_sph_harm(l,m,x,y,z)
            + std::complex<double>{0.0,1.0}*sphericart_real_sph_harm(l,-m,x,y,z));
    }
}

double sphericart_real_sph_harm(int l, int m, double x, double y, double z) {
    sphericart::SphericalHarmonics<double> sphericart(l);
    auto xyz = std::vector<double>{x, y, z};
    auto sph = std::vector<double>((l+1)*(l+1));
    sphericart.compute(xyz, sph);
    return sph[l*(l+1)+m];
}

std::vector<double> sphericart_real(int l_max, std::vector<double> xyz)
{
    sphericart::SphericalHarmonics<double> sphericart(l_max);
    auto sh = std::vector<double>((l_max+1)*(l_max+1));
    sphericart.compute(xyz, sh);
    return sh;
}

std::vector<std::complex<double>> sphericart_complex(
    int l_max,
    std::vector<double> xyz)
{
    const auto imag = std::complex<double>{0.0,1.0};
    const double inv_sq_2 = 1.0 / std::sqrt(2.0);
    const int n_samples = xyz.size() / 3;
    sphericart::SphericalHarmonics<double> sphericart(l_max);
    auto sph_real = std::vector<double>(n_samples*(l_max+1)*(l_max+1));
    auto sph_complex = std::vector<std::complex<double>>(n_samples*(l_max+1)*(l_max+1));
    sphericart.compute(xyz, sph_real);
    for (int i=0; i<n_samples; ++i) {
        const int i0 = i*(l_max+1)*(l_max+1);
        for (int l=0; l<=l_max; ++l) {
            for (int m=-l; m<=l; ++m) {
                if (m < 0) {
                    sph_complex[i0+l*(l+1)+m] = inv_sq_2 * (
                        sph_real[i0+l*(l+1)-m] - imag*sph_real[i0+l*(l+1)+m]);
                } else if (m == 0) {
                    sph_complex[i0+l*(l+1)] = sph_real[i0+l*(l+1)];
                } else {
                    sph_complex[i0+l*(l+1)+m] = std::pow(-1,m) * inv_sq_2 * (
                        sph_real[i0+l*(l+1)+m] + imag*sph_real[i0+l*(l+1)-m]);
                }
            }
        }
    }
    return sph_complex;
}
