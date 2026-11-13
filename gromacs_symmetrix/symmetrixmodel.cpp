// The GROMACS side of the Symmetrix NNPot model: units, periodicity, and the distribution of the forces to the ranks
// that own the atoms. The model runs on the main rank, after GROMACS has collected the positions.

#include "symmetrixmodel.h"

#include "gmxpre.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "gromacs/math/units.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/mpicomm.h"

#include "nnpotoptions.h"

namespace gmx
{

namespace
{

std::string environment(const char* name, const std::string& otherwise)
{
    const char* value = std::getenv(name);
    return value && *value ? value : otherwise;
}

symmetrix::gromacs::Options options(const NNPotParameters& parameters)
{
    symmetrix::gromacs::Options o;
    o.dtype             = environment("SYMMETRIX_DTYPE", o.dtype);
    o.skin              = std::stod(environment("SYMMETRIX_SKIN", std::to_string(o.skin)));
    o.spin_multiplicity = std::stod(environment("SYMMETRIX_SPIN_MULTIPLICITY", "1"));
    o.total_charge      = parameters.nnpCharge_;
    return o;
}

} // namespace

void validateSymmetrixParameters(const NNPotParameters& parameters)
{
    if (parameters.embeddingScheme_ != NNPotEmbedding::Mechanical || !parameters.mmIndices_.empty()
        || !parameters.linkFrontier_.empty())
    {
        GMX_THROW(InconsistentInputError(
                "Symmetrix models need the whole System as nnpot-input-group, with mechanical embedding"));
    }
    std::set<std::string> inputs;
    for (const auto& input : parameters.modelInput_)
    {
        if (!input.empty() && !inputs.insert(input).second)
        {
            GMX_THROW(InconsistentInputError("Duplicate Symmetrix model input: " + input));
        }
    }
    if (inputs != std::set<std::string>{ "atom-positions", "atom-numbers", "box", "pbc" })
    {
        GMX_THROW(InconsistentInputError(
                "Symmetrix models take exactly the inputs atom-positions, atom-numbers, box and pbc; "
                "they build their own neighbor list from the model's cutoff"));
    }
    const std::filesystem::path path(parameters.modelFileName_);
    if (path.extension() != ".json" || !std::filesystem::is_regular_file(path))
    {
        GMX_THROW(InconsistentInputError("nnpot-modelfile must name an existing Symmetrix model (.json)"));
    }
}

SymmetrixModel::SymmetrixModel(const NNPotParameters& parameters, const MpiComm& comm) : comm_(comm)
{
    validateSymmetrixParameters(parameters);
    if (comm_.isMainRank())
    {
        try
        {
            model_ = symmetrix::gromacs::Model::load(parameters.modelFileName_, options(parameters));
        }
        catch (const std::exception& e)
        {
            GMX_THROW(InconsistentInputError(e.what()));
        }
    }
}

void SymmetrixModel::evaluateModel(gmx_enerdata_t*     energy,
                                   ArrayRef<RVec>      forces,
                                   ArrayRef<const int> lookup,
                                   ArrayRef<const int>,
                                   ArrayRef<const std::string>,
                                   ArrayRef<RVec> positions,
                                   ArrayRef<int>  numbers,
                                   ArrayRef<int>,
                                   ArrayRef<RVec>,
                                   ArrayRef<RVec>,
                                   ArrayRef<real>,
                                   real,
                                   ArrayRef<const LinkFrontierAtom>,
                                   matrix*  box,
                                   PbcType* pbcType)
{
    if (*pbcType != PbcType::Xyz && *pbcType != PbcType::XY && *pbcType != PbcType::No)
    {
        GMX_THROW(InconsistentInputError("Symmetrix models support pbc = xyz, xy or no"));
    }
    GMX_RELEASE_ASSERT(positions.size() == numbers.size() && numbers.size() == lookup.size(),
                       "Symmetrix input atom arrays must have the same length");
    // GROMACS uses nm and kJ/mol, Symmetrix A and eV
    constexpr double energyFactor = c_electronCharge * c_avogadro / c_kilo;
    constexpr double forceFactor  = energyFactor * c_nm2A;
    std::vector<double> globalForces(3 * positions.size(), 0);
    for (auto& row : virial_)
    {
        for (auto& value : row)
        {
            value = 0;
        }
    }
    if (comm_.isMainRank())
    {
        std::array<double, 9> cell{};
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                cell[3 * i + j] = (*box)[i][j] * c_nm2A;
            }
        }
        const bool pbc[3] = { *pbcType != PbcType::No, *pbcType != PbcType::No, *pbcType == PbcType::Xyz };
        std::vector<double> x(3 * positions.size());
        for (size_t i = 0; i < positions.size(); ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                x[3 * i + j] = positions[i][j] * c_nm2A;
            }
        }
        std::vector<double>   f(x.size());
        std::array<double, 6> w{};
        double                e = 0;
        try
        {
            e = model_->compute(positions.ssize(), x.data(), numbers.data(), cell.data(), pbc, f.data(), w.data());
        }
        catch (const std::exception& error)
        {
            GMX_THROW(InternalError(std::string("Symmetrix: ") + error.what()));
        }
        energy->term[InteractionFunction::NeuralNetworkPotentialEnergy] = e * energyFactor;
        for (size_t k = 0; k < f.size(); ++k)
        {
            globalForces[k] = f[k] * forceFactor;
        }
        // GROMACS's pressure is 2*(Ekin-Xi)/V, so Xi = -W/2 for the virial W (xx yy zz xy xz yz) of the model;
        // it is added once, on the main rank
        constexpr int a[6] = { 0, 1, 2, 0, 0, 1 }, b[6] = { 0, 1, 2, 1, 2, 2 };
        for (int k = 0; k < 6; ++k)
        {
            virial_[a[k]][b[k]] = virial_[b[k]][a[k]] = -0.5 * energyFactor * w[k];
        }
    }
    if (comm_.isParallel())
    {
        comm_.sumReduce(globalForces.size(), globalForces.data());
    }
    for (size_t i = 0; i < lookup.size(); ++i)
    {
        const int local = lookup[i];
        if (local < 0)
        {
            continue;
        }
        GMX_RELEASE_ASSERT(local < forces.ssize(), "Symmetrix local atom index exceeds force buffer");
        for (int j = 0; j < 3; ++j)
        {
            forces[local][j] += globalForces[3 * i + j];
        }
    }
}

} // namespace gmx
