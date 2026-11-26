#pragma once

// An OpenMM Force evaluating a MACE or NequIP foundation model (a Symmetrix JSON) with the GPU arithmetic of
// pair_style symmetrix/mace/foundation and symmetrix/nequip/foundation. On the OpenMM CUDA platform, coordinates
// and forces stay on the GPU: the force reads OpenMM's positions and adds to its force buffer. Other platforms are
// not supported.

#include <openmm/Force.h>

#include <string>
#include <vector>

namespace symmetrix {

class SymmetrixFoundationForce : public OpenMM::Force {
public:
    // family is "mace" or "nequip". atomic_numbers follow the order of the System's particles, all of which the
    // force applies to. dtype is "float64", "float32" or "float32-mixed"; skin (A) sets how far atoms move before
    // the neighbor list is built again. A periodic force uses the periodic box vectors of the Context in all three
    // directions; otherwise the System is an isolated molecule.
    SymmetrixFoundationForce(std::string family, std::string model_file, std::vector<int> atomic_numbers,
                             std::string dtype = "float64", double skin = 1.0, bool periodic = true);

    const std::string& getFamily() const { return family; }
    const std::string& getModelFile() const { return model_file; }
    const std::vector<int>& getAtomicNumbers() const { return atomic_numbers; }
    const std::string& getDtype() const { return dtype; }
    double getSkin() const { return skin; }
    bool getPeriodic() const { return periodic; }

    bool usesPeriodicBoundaryConditions() const override { return periodic; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    std::string family;
    std::string model_file;
    std::vector<int> atomic_numbers;
    std::string dtype;
    double skin;
    bool periodic;
};

}  // namespace symmetrix
