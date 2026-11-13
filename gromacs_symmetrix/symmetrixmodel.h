// The INNPotModel of GROMACS's NNPot interface for Symmetrix models (built with GMX_NNPOT=SYMMETRIX).
#ifndef SYMMETRIX_GROMACS_MODEL_H
#define SYMMETRIX_GROMACS_MODEL_H

#include <memory>

#include "nnpotmodel.h"
#include "symmetrix_gromacs.h"

namespace gmx
{

struct NNPotParameters;

//! Refuses the NNPot inputs and embeddings that Symmetrix models do not support
void validateSymmetrixParameters(const NNPotParameters& parameters);

class SymmetrixModel final : public INNPotModel
{
public:
    SymmetrixModel(const NNPotParameters& parameters, const MpiComm& comm);
    void evaluateModel(gmx_enerdata_t*                  energy,
                       ArrayRef<RVec>                   forces,
                       ArrayRef<const int>              lookup,
                       ArrayRef<const int>              mmIndices,
                       ArrayRef<const std::string>      inputs,
                       ArrayRef<RVec>                   positions,
                       ArrayRef<int>                    atomNumbers,
                       ArrayRef<int>                    atomPairs,
                       ArrayRef<RVec>                   pairShifts,
                       ArrayRef<RVec>                   positionsMM,
                       ArrayRef<real>                   chargesMM,
                       real                             nnpCharge,
                       ArrayRef<const LinkFrontierAtom> linkFrontier,
                       matrix*                          box,
                       PbcType*                         pbcType) override;
    bool outputsForces() const override { return true; }
    //! The virial contribution of the last evaluation (GROMACS convention, kJ/mol)
    const matrix& virial() const { return virial_; }

private:
    std::unique_ptr<symmetrix::gromacs::Model> model_;
    const MpiComm&                             comm_;
    matrix                                     virial_{};
};

} // namespace gmx

#endif
