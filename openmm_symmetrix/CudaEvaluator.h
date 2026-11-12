#pragma once

// The interface between the platform-independent forces and their CUDA modules, which are loaded with RTLD_LOCAL:
// OpenMM's platform plugins export overlapping private names, so the CUDA platform must not become global.

#include <string>
#include <vector>

namespace OpenMM {
class ContextImpl;
}

namespace symmetrix {

class CudaEvaluator {
public:
    virtual ~CudaEvaluator() = default;
    // Adds the forces to OpenMM's force buffer and returns the energy in kJ/mol
    virtual double execute(OpenMM::ContextImpl& context, bool include_forces, bool include_energy) = 0;
};

struct CudaEvaluatorArgs {
    std::string name;  // of the force, for messages
    std::string model_file;
    std::vector<int> atomic_numbers;
    double skin;
    bool periodic = true;
    // PolarMACE
    double total_charge = 0;
    double spin_multiplicity = 1;
    bool float32_mixed = false;
    // foundation models: "float64", "float32" or "float32-mixed"
    std::string dtype;
};

// In libOpenMMSymmetrix: checks the platform and the OpenMM version, then creates the evaluator of the CUDA module
// `module`, which is next to libOpenMMSymmetrix or in the lib directory above it
CudaEvaluator* create_cuda_evaluator(OpenMM::ContextImpl& context, const std::string& module,
                                     const CudaEvaluatorArgs& args);

}  // namespace symmetrix

// Each CUDA module exports this
extern "C" symmetrix::CudaEvaluator* symmetrix_create_cuda_evaluator(OpenMM::ContextImpl& context,
                                                                     const symmetrix::CudaEvaluatorArgs& args);
