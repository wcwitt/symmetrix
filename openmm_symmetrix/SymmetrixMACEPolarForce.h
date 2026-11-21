#pragma once

// An OpenMM Force evaluating a PolarMACE model (a Symmetrix JSON) with the GPU arithmetic of
// pair_style symmetrix/mace/polar/cuda. On the OpenMM CUDA platform, coordinates and forces stay on the GPU:
// the force reads OpenMM's positions and adds to its force buffer. Other platforms are not supported.

#include <openmm/Force.h>

#include <string>
#include <vector>

namespace symmetrix {

class SymmetrixMACEPolarForce : public OpenMM::Force {
public:
    // atomic_numbers follow the order of the System's particles, all of which the force applies to.
    // total_charge and spin_multiplicity are those of the whole System; dtype is "float32" or "float32-mixed"
    // (fp16 tensor-core GEMMs to float32 accuracy); skin (A) sets how far atoms move before the neighbor list is
    // built again.
    SymmetrixMACEPolarForce(std::string model_file, std::vector<int> atomic_numbers, double total_charge = 0.0,
                            double spin_multiplicity = 1.0, std::string dtype = "float32", double skin = 1.0);

    const std::string& getModelFile() const { return model_file; }
    const std::vector<int>& getAtomicNumbers() const { return atomic_numbers; }
    double getTotalCharge() const { return total_charge; }
    double getSpinMultiplicity() const { return spin_multiplicity; }
    double getSkin() const { return skin; }
    const std::string& getDtype() const { return dtype; }

    bool usesPeriodicBoundaryConditions() const override { return true; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    std::string model_file;
    std::vector<int> atomic_numbers;
    double total_charge;
    double spin_multiplicity;
    std::string dtype;
    double skin;
};

}  // namespace symmetrix
