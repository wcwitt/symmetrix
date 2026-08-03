#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "compact_radial.hpp"

namespace {

CompactRadialModel::Network parse_network(
    const nlohmann::json& definition,
    const std::string& name)
{
    CompactRadialModel::Network network;
    network.shape = definition.at("shape").get<std::vector<int>>();
    network.weights = definition.at("weights").get<std::vector<std::vector<double>>>();
    network.activation_scale = definition.at("activation_scale").get<double>();
    const auto postprocess = definition.value("postprocess", std::string());
    if (!postprocess.empty() && postprocess != "tanh-square")
        throw std::invalid_argument(
            "Compact radial network " + name + " has unsupported postprocessing.");
    network.tanh_square = postprocess == "tanh-square";

    if (definition.at("activation").get<std::string>() != "silu")
        throw std::invalid_argument("Compact radial network " + name + " must use SiLU.");
    if (network.shape.size() < 2 || network.weights.size() != network.shape.size()-1)
        throw std::invalid_argument("Compact radial network " + name + " has an invalid shape.");
    if (!std::isfinite(network.activation_scale) || network.activation_scale <= 0.0)
        throw std::invalid_argument(
            "Compact radial network " + name + " has an invalid activation scale.");
    for (int layer=0; layer<network.weights.size(); ++layer) {
        if (network.shape[layer] <= 0 || network.shape[layer+1] <= 0)
            throw std::invalid_argument(
                "Compact radial network " + name + " has a non-positive layer size.");
        const auto expected = static_cast<std::size_t>(network.shape[layer]*network.shape[layer+1]);
        if (network.weights[layer].size() != expected)
            throw std::invalid_argument("Compact radial network " + name + " has invalid weights.");
        if (!std::all_of(
                network.weights[layer].begin(),
                network.weights[layer].end(),
                [] (double value) { return std::isfinite(value); }))
            throw std::invalid_argument(
                "Compact radial network " + name + " has non-finite weights.");
    }

    network.mlp = std::make_unique<MultilayerPerceptron>(
        network.shape,
        network.weights,
        network.activation_scale);
    return network;
}

}  // namespace

CompactRadialModel::CompactRadialModel(
    std::string definition_json,
    std::vector<int> atomic_numbers,
    double r_cut)
    : atomic_numbers(std::move(atomic_numbers))
{
    const auto definition = nlohmann::json::parse(definition_json);
    grid_min = definition.at("spline_grid_min").get<double>();
    spline_points = definition.at("num_spline_points").get<int>();
    if (!std::isfinite(grid_min) || grid_min <= 0.0)
        throw std::invalid_argument("Compact radial model has an invalid spline grid minimum.");
    if (spline_points < 4)
        throw std::invalid_argument("Compact radial model requires at least four spline points.");
    h = (r_cut-grid_min)/(spline_points-1);
    if (!(h > 0.0))
        throw std::invalid_argument("Compact radial model has an invalid spline grid.");

    const auto& basis = definition.at("basis");
    if (basis.at("type").get<std::string>() != "bessel")
        throw std::invalid_argument("Compact radial model only supports a Bessel basis.");
    bessel_weights = basis.at("weights").get<std::vector<double>>();
    bessel_prefactor = basis.at("prefactor").get<double>();
    if (bessel_weights.empty()
        || !std::isfinite(bessel_prefactor)
        || !std::all_of(
            bessel_weights.begin(), bessel_weights.end(),
            [] (double value) { return std::isfinite(value); }))
        throw std::invalid_argument("Compact radial model has invalid Bessel data.");

    const auto& cutoff = definition.at("cutoff");
    if (cutoff.at("type").get<std::string>() != "polynomial")
        throw std::invalid_argument("Compact radial model only supports a polynomial cutoff.");
    cutoff_r_max = cutoff.at("r_max").get<double>();
    cutoff_p = cutoff.at("p").get<int>();
    if (!std::isfinite(cutoff_r_max) || cutoff_p <= 0)
        throw std::invalid_argument("Compact radial model has invalid cutoff data.");
    if (std::abs(cutoff_r_max-r_cut) > 1e-12)
        throw std::invalid_argument("Compact radial cutoff does not match model r_cut.");

    const auto& transform = definition.at("distance_transform");
    const auto transform_type = transform.at("type").get<std::string>();
    use_agnesi = transform_type == "agnesi";
    if (use_agnesi) {
        agnesi_a = transform.at("a").get<double>();
        agnesi_q = transform.at("q").get<double>();
        agnesi_p = transform.at("p").get<double>();
        covalent_radii = transform.at("covalent_radii").get<std::vector<double>>();
        if (covalent_radii.size() != this->atomic_numbers.size())
            throw std::invalid_argument("Compact radial covalent radii do not match atomic numbers.");
        if (!std::isfinite(agnesi_a)
            || !std::isfinite(agnesi_q)
            || !std::isfinite(agnesi_p)
            || !std::all_of(
                covalent_radii.begin(), covalent_radii.end(),
                [] (double value) { return std::isfinite(value) && value > 0.0; }))
            throw std::invalid_argument("Compact radial model has invalid Agnesi data.");
    } else if (transform_type != "none") {
        throw std::invalid_argument("Compact radial model has an unsupported distance transform.");
    }

    const auto& networks = definition.at("networks");
    R0_network = parse_network(networks.at("R0"), "R0");
    R1_network = parse_network(networks.at("R1"), "R1");
    if (R0_network.shape.front() != bessel_weights.size()
        || R1_network.shape.front() != bessel_weights.size())
        throw std::invalid_argument("Compact radial network input does not match the Bessel basis.");
    if (R0_network.tanh_square || R1_network.tanh_square)
        throw std::invalid_argument("Compact radial R0/R1 networks cannot use postprocessing.");
    if (networks.contains("A0")) {
        A0_network = std::make_unique<Network>(parse_network(networks.at("A0"), "A0"));
        if (A0_network->shape.front() != bessel_weights.size()
            || A0_network->shape.back() != 1
            || !A0_network->tanh_square)
            throw std::invalid_argument("Compact radial A0 network has an invalid shape or postprocessing.");
    }
    if (networks.contains("A1")) {
        A1_network = std::make_unique<Network>(parse_network(networks.at("A1"), "A1"));
        if (A1_network->shape.front() != bessel_weights.size()
            || A1_network->shape.back() != 1
            || !A1_network->tanh_square)
            throw std::invalid_argument("Compact radial A1 network has an invalid shape or postprocessing.");
    }
}

double CompactRadialModel::spline_h() const
{
    return h;
}

double CompactRadialModel::spline_min() const
{
    return grid_min;
}

int CompactRadialModel::num_spline_points() const
{
    return spline_points;
}

bool CompactRadialModel::has_A0() const
{
    return static_cast<bool>(A0_network);
}

bool CompactRadialModel::has_A1() const
{
    return static_cast<bool>(A1_network);
}

std::vector<double> CompactRadialModel::radial_features(int type_i, int type_j) const
{
    if (type_i < 0 || type_i >= atomic_numbers.size()
        || type_j < 0 || type_j >= atomic_numbers.size())
        throw std::out_of_range("Compact radial model type index is out of range.");

    double r0 = 1.0;
    if (use_agnesi)
        r0 = 0.5*(covalent_radii[type_i]+covalent_radii[type_j]);

    auto features = std::vector<double>(spline_points*bessel_weights.size());
    for (int node=0; node<spline_points; ++node) {
        const double r = grid_min+node*h;
        double transformed_r = r;
        if (use_agnesi) {
            const double scaled_r = r/r0;
            transformed_r = 1.0/(
                1.0
                + agnesi_a*std::pow(scaled_r, agnesi_q)
                    /(1.0+std::pow(scaled_r, agnesi_q-agnesi_p)));
        }

        double cutoff_value = 0.0;
        if (r < cutoff_r_max) {
            const double scaled_r = r/cutoff_r_max;
            cutoff_value = 1.0
                - ((cutoff_p+1.0)*(cutoff_p+2.0)/2.0)*std::pow(scaled_r, cutoff_p)
                + cutoff_p*(cutoff_p+2.0)*std::pow(scaled_r, cutoff_p+1)
                - (cutoff_p*(cutoff_p+1.0)/2.0)*std::pow(scaled_r, cutoff_p+2);
        }

        for (int basis=0; basis<bessel_weights.size(); ++basis) {
            features[node*bessel_weights.size()+basis] =
                bessel_prefactor
                * std::sin(bessel_weights[basis]*transformed_r)
                / transformed_r
                * cutoff_value;
        }
    }
    return features;
}

RadialSplineData CompactRadialModel::evaluate_network(
    Network& network,
    const std::vector<double>& features)
{
    auto batch_values = network.mlp->evaluate_batch(features, spline_points);
    const int num_functions = network.shape.back();
    if (network.tanh_square) {
        for (auto& value : batch_values)
            value = std::tanh(value*value);
    }

    RadialSplineData result;
    result.values.resize(num_functions, std::vector<double>(spline_points));
    result.derivatives.resize(num_functions, std::vector<double>(spline_points));
    for (int function=0; function<num_functions; ++function) {
        for (int node=0; node<spline_points; ++node)
            result.values[function][node] = batch_values[node*num_functions+function];
        result.derivatives[function] = spline_derivatives(result.values[function]);
    }
    return result;
}

std::vector<double> CompactRadialModel::spline_derivatives(
    const std::vector<double>& values) const
{
    if (values.size() != spline_points)
        throw std::invalid_argument("Compact radial spline values have an invalid size.");

    const int unknowns = spline_points-2;
    auto lower = std::vector<double>(unknowns, 0.0);
    auto diagonal = std::vector<double>(unknowns, 4.0);
    auto upper = std::vector<double>(unknowns, 0.0);
    auto rhs = std::vector<double>(unknowns, 0.0);

    const double left_not_a_knot =
        2.0*(-values[0]+2.0*values[1]-values[2])/h;
    upper[0] = 2.0;
    rhs[0] = 3.0*(values[2]-values[0])/h-left_not_a_knot;
    for (int node=2; node<spline_points-1; ++node) {
        const int row = node-1;
        lower[row] = 1.0;
        if (node+1 <= spline_points-2)
            upper[row] = 1.0;
        rhs[row] = 3.0*(values[node+1]-values[node-1])/h;
    }

    for (int row=1; row<unknowns; ++row) {
        const double factor = lower[row]/diagonal[row-1];
        diagonal[row] -= factor*upper[row-1];
        rhs[row] -= factor*rhs[row-1];
    }
    auto solution = std::vector<double>(unknowns, 0.0);
    solution.back() = rhs.back()/diagonal.back();
    for (int row=unknowns-2; row>=0; --row)
        solution[row] = (rhs[row]-upper[row]*solution[row+1])/diagonal[row];

    auto derivatives = std::vector<double>(spline_points, 0.0);
    for (int node=1; node<spline_points-1; ++node)
        derivatives[node] = solution[node-1];
    derivatives[0] = derivatives[2]+left_not_a_knot;
    derivatives.back() = 0.0;
    return derivatives;
}

CompactRadialPairTables CompactRadialModel::materialize_pair(int type_i, int type_j)
{
    const auto features = radial_features(type_i, type_j);
    CompactRadialPairTables result;
    result.R0 = evaluate_network(R0_network, features);
    result.R1 = evaluate_network(R1_network, features);
    if (A0_network)
        result.A0 = evaluate_network(*A0_network, features);
    if (A1_network)
        result.A1 = evaluate_network(*A1_network, features);
    return result;
}
