#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

#include "symmetrix_gromacs.h"

namespace symmetrix::gromacs {

std::unique_ptr<Model> polar_model(const std::string& file, const Options& options);
std::unique_ptr<Model> mace_foundation_model(const std::string& file, const Options& options);
std::unique_ptr<Model> nequip_foundation_model(const std::string& file, const Options& options);

std::unique_ptr<Model> Model::load(const std::string& file, const Options& options)
{
    if (options.dtype != "float64" and options.dtype != "float32" and options.dtype != "float32-mixed")
        throw std::runtime_error("Symmetrix: dtype must be float64, float32 or float32-mixed, not " + options.dtype);
    if (not (options.skin >= 0) or not std::isfinite(options.skin))
        throw std::runtime_error("Symmetrix: the skin must be a finite number >= 0");
    // the extracted models write model_type near the top, as symmetrix.calculator reads it
    std::ifstream in(file);
    if (not in)
        throw std::runtime_error("Symmetrix: cannot open " + file);
    std::string head(4096, '\0');
    in.read(head.data(), std::streamsize(head.size()));
    head.resize(size_t(in.gcount()));
    std::smatch match;
    const std::string type = std::regex_search(head, match, std::regex(R"re("model_type"\s*:\s*"([^"]*)")re"))
        ? match[1].str() : "";
    if (type == "PolarMACE")
        return polar_model(file, options);
    if (type == "NequIP")
        return nequip_foundation_model(file, options);
    if (type.empty())
        return mace_foundation_model(file, options);
    throw std::runtime_error("Symmetrix: " + file + " has the unsupported model_type " + type);
}

double Model::compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                      double* forces, double* virial)
{
    // Move each atom by lattice vectors to the image nearest its previous position, so the positions are continuous
    if (unwrapped.size() != 3*size_t(n)) {
        unwrapped.assign(x, x + 3*size_t(n));
    } else {
        // rows of the inverse transpose: fractional coordinate d of a vector v is v . b_d
        const double* a = cell;
        const double c[9] = {a[4]*a[8] - a[5]*a[7], a[5]*a[6] - a[3]*a[8], a[3]*a[7] - a[4]*a[6],
                             a[7]*a[2] - a[8]*a[1], a[8]*a[0] - a[6]*a[2], a[6]*a[1] - a[7]*a[0],
                             a[1]*a[5] - a[2]*a[4], a[2]*a[3] - a[0]*a[5], a[0]*a[4] - a[1]*a[3]};
        const double det = a[0]*c[0] + a[1]*c[1] + a[2]*c[2];
        for (int i = 0; i < n; ++i) {
            double* u = unwrapped.data() + 3*i;
            const double d[3] = {u[0] - x[3*i], u[1] - x[3*i + 1], u[2] - x[3*i + 2]};
            double s[3] = {0, 0, 0};
            for (int k = 0; k < 3; ++k)
                if (pbc[k] and det != 0)
                    s[k] = std::nearbyint((d[0]*c[3*k] + d[1]*c[3*k + 1] + d[2]*c[3*k + 2])/det);
            for (int e = 0; e < 3; ++e)
                u[e] = x[3*i + e] + s[0]*a[e] + s[1]*a[3 + e] + s[2]*a[6 + e];
        }
    }
    return evaluate(n, unwrapped.data(), numbers, cell, pbc, forces, virial);
}

}  // namespace symmetrix::gromacs
