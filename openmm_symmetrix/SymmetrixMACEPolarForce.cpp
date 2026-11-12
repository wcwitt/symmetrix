#include "SymmetrixMACEPolarForce.h"

#include <memory>

#include <openmm/OpenMMException.h>
#include <openmm/System.h>
#include <openmm/internal/ContextImpl.h>
#include <openmm/internal/ForceImpl.h>
#include <openmm/serialization/SerializationNode.h>
#include <openmm/serialization/SerializationProxy.h>

#include "CudaEvaluator.h"

namespace symmetrix {

namespace {

class SymmetrixMACEPolarForceImpl : public OpenMM::ForceImpl {
public:
    explicit SymmetrixMACEPolarForceImpl(const SymmetrixMACEPolarForce& owner) : owner(owner) {}

    const OpenMM::Force& getOwner() const override { return owner; }
    std::map<std::string, double> getDefaultParameters() override { return {}; }
    std::vector<std::string> getKernelNames() override { return {}; }

    void initialize(OpenMM::ContextImpl& context) override
    {
        CudaEvaluatorArgs args;
        args.name = "SymmetrixMACEPolarForce";
        args.model_file = owner.getModelFile();
        args.atomic_numbers = owner.getAtomicNumbers();
        args.skin = owner.getSkin();
        args.total_charge = owner.getTotalCharge();
        args.spin_multiplicity = owner.getSpinMultiplicity();
        args.float32_mixed = owner.getFloat32Mixed();
        evaluator.reset(create_cuda_evaluator(context, SYMMETRIX_MACE_POLAR_MODULE, args));
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool include_forces, bool include_energy,
                               int groups) override
    {
        if ((groups & (1 << owner.getForceGroup())) == 0)
            return 0;
        return evaluator->execute(context, include_forces, include_energy);
    }

private:
    const SymmetrixMACEPolarForce& owner;
    std::unique_ptr<CudaEvaluator> evaluator;
};

class SymmetrixMACEPolarForceProxy : public OpenMM::SerializationProxy {
public:
    SymmetrixMACEPolarForceProxy() : OpenMM::SerializationProxy("SymmetrixMACEPolarForce") {}

    void serialize(const void* object, OpenMM::SerializationNode& node) const override
    {
        const auto& force = *static_cast<const SymmetrixMACEPolarForce*>(object);
        node.setIntProperty("version", 1);
        node.setStringProperty("modelFile", force.getModelFile());
        node.setDoubleProperty("totalCharge", force.getTotalCharge());
        node.setDoubleProperty("spinMultiplicity", force.getSpinMultiplicity());
        node.setDoubleProperty("skin", force.getSkin());
        node.setIntProperty("float32Mixed", force.getFloat32Mixed() ? 1 : 0);
        node.setIntProperty("forceGroup", force.getForceGroup());
        node.setStringProperty("name", force.getName());
        auto& numbers = node.createChildNode("AtomicNumbers");
        for (int z : force.getAtomicNumbers())
            numbers.createChildNode("Atom").setIntProperty("z", z);
    }

    void* deserialize(const OpenMM::SerializationNode& node) const override
    {
        if (node.getIntProperty("version") != 1)
            throw OpenMM::OpenMMException("SymmetrixMACEPolarForce: unsupported version");
        std::vector<int> numbers;
        for (const auto& atom : node.getChildNode("AtomicNumbers").getChildren())
            numbers.push_back(atom.getIntProperty("z"));
        auto* force = new SymmetrixMACEPolarForce(
            node.getStringProperty("modelFile"), numbers, node.getDoubleProperty("totalCharge", 0.0),
            node.getDoubleProperty("spinMultiplicity", 1.0), node.getDoubleProperty("skin", 1.0),
            node.getIntProperty("float32Mixed", 0) != 0);
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", "SymmetrixMACEPolarForce"));
        return force;
    }
};

}  // namespace

SymmetrixMACEPolarForce::SymmetrixMACEPolarForce(std::string model_file, std::vector<int> atomic_numbers,
                                                 double total_charge, double spin_multiplicity, double skin,
                                                 bool float32_mixed)
    : model_file(std::move(model_file)), atomic_numbers(std::move(atomic_numbers)), total_charge(total_charge),
      spin_multiplicity(spin_multiplicity), skin(skin), float32_mixed(float32_mixed)
{
    if (this->atomic_numbers.empty())
        throw OpenMM::OpenMMException("SymmetrixMACEPolarForce: no atomic numbers");
    if (skin < 0)
        throw OpenMM::OpenMMException("SymmetrixMACEPolarForce: negative skin");
    setName("SymmetrixMACEPolarForce");
}

OpenMM::ForceImpl* SymmetrixMACEPolarForce::createImpl() const
{
    return new SymmetrixMACEPolarForceImpl(*this);
}

void register_mace_polar_proxy()
{
    OpenMM::SerializationProxy::registerProxy(typeid(SymmetrixMACEPolarForce), new SymmetrixMACEPolarForceProxy());
}

}  // namespace symmetrix
