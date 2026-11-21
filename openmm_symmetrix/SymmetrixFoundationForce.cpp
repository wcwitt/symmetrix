#include "SymmetrixFoundationForce.h"

#include <memory>

#include <openmm/OpenMMException.h>
#include <openmm/internal/ContextImpl.h>
#include <openmm/internal/ForceImpl.h>
#include <openmm/serialization/SerializationNode.h>
#include <openmm/serialization/SerializationProxy.h>

#include "CudaEvaluator.h"

namespace symmetrix {

namespace {

constexpr const char* force_name = "SymmetrixFoundationForce";

class SymmetrixFoundationForceImpl : public OpenMM::ForceImpl {
public:
    explicit SymmetrixFoundationForceImpl(const SymmetrixFoundationForce& owner) : owner(owner) {}

    const OpenMM::Force& getOwner() const override { return owner; }
    std::map<std::string, double> getDefaultParameters() override { return {}; }
    std::vector<std::string> getKernelNames() override { return {}; }

    void initialize(OpenMM::ContextImpl& context) override
    {
        CudaEvaluatorArgs args;
        args.name = force_name;
        args.model_file = owner.getModelFile();
        args.atomic_numbers = owner.getAtomicNumbers();
        args.skin = owner.getSkin();
        args.periodic = owner.getPeriodic();
        args.dtype = owner.getDtype();
        const char* module = owner.getFamily() == "mace" ? SYMMETRIX_MACE_FOUNDATION_MODULE
                                                         : SYMMETRIX_NEQUIP_FOUNDATION_MODULE;
        evaluator.reset(create_cuda_evaluator(context, module, args));
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool include_forces, bool include_energy,
                               int groups) override
    {
        if ((groups & (1 << owner.getForceGroup())) == 0)
            return 0;
        return evaluator->execute(context, include_forces, include_energy);
    }

private:
    const SymmetrixFoundationForce& owner;
    std::unique_ptr<CudaEvaluator> evaluator;
};

class SymmetrixFoundationForceProxy : public OpenMM::SerializationProxy {
public:
    SymmetrixFoundationForceProxy() : OpenMM::SerializationProxy(force_name) {}

    void serialize(const void* object, OpenMM::SerializationNode& node) const override
    {
        const auto& force = *static_cast<const SymmetrixFoundationForce*>(object);
        node.setIntProperty("version", 1);
        node.setStringProperty("family", force.getFamily());
        node.setStringProperty("modelFile", force.getModelFile());
        node.setStringProperty("dtype", force.getDtype());
        node.setDoubleProperty("skin", force.getSkin());
        node.setIntProperty("periodic", force.getPeriodic() ? 1 : 0);
        node.setIntProperty("forceGroup", force.getForceGroup());
        node.setStringProperty("name", force.getName());
        auto& numbers = node.createChildNode("AtomicNumbers");
        for (int z : force.getAtomicNumbers())
            numbers.createChildNode("Atom").setIntProperty("z", z);
    }

    void* deserialize(const OpenMM::SerializationNode& node) const override
    {
        if (node.getIntProperty("version") != 1)
            throw OpenMM::OpenMMException(std::string(force_name) + ": unsupported version");
        std::vector<int> numbers;
        for (const auto& atom : node.getChildNode("AtomicNumbers").getChildren())
            numbers.push_back(atom.getIntProperty("z"));
        auto* force = new SymmetrixFoundationForce(
            node.getStringProperty("family"), node.getStringProperty("modelFile"), numbers,
            node.getStringProperty("dtype", "float64"), node.getDoubleProperty("skin", 1.0),
            node.getIntProperty("periodic", 1) != 0);
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force_name));
        return force;
    }
};

}  // namespace

SymmetrixFoundationForce::SymmetrixFoundationForce(std::string family, std::string model_file,
                                                   std::vector<int> atomic_numbers, std::string dtype, double skin,
                                                   bool periodic)
    : family(std::move(family)), model_file(std::move(model_file)), atomic_numbers(std::move(atomic_numbers)),
      dtype(std::move(dtype)), skin(skin), periodic(periodic)
{
    const std::string name = force_name;
    if (this->family != "mace" and this->family != "nequip")
        throw OpenMM::OpenMMException(name + ": family must be 'mace' or 'nequip', not '" + this->family + "'");
    if (this->dtype != "float64" and this->dtype != "float32" and this->dtype != "float32-mixed")
        throw OpenMM::OpenMMException(name + ": dtype must be 'float64', 'float32' or 'float32-mixed', not '" +
                                      this->dtype + "'");
    if (this->atomic_numbers.empty())
        throw OpenMM::OpenMMException(name + ": no atomic numbers");
    if (not (skin >= 0))
        throw OpenMM::OpenMMException(name + ": the skin must be a number >= 0");
    setName(name);
}

OpenMM::ForceImpl* SymmetrixFoundationForce::createImpl() const
{
    return new SymmetrixFoundationForceImpl(*this);
}

void register_foundation_proxy()
{
    OpenMM::SerializationProxy::registerProxy(typeid(SymmetrixFoundationForce), new SymmetrixFoundationForceProxy());
}

}  // namespace symmetrix
