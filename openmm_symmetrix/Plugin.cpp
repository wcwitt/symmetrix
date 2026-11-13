// The entry points of an OpenMM plugin library. Platform.loadPluginLibrary() and the plugin directory loaded at
// startup call registerPlatforms(), which makes the forces available to XmlSerializer, and so to Python.

namespace symmetrix {
void register_mace_polar_proxy();
void register_foundation_proxy();
}  // namespace symmetrix

extern "C" void registerPlatforms()
{
    static bool done = false;
    if (done)
        return;
    done = true;
    symmetrix::register_mace_polar_proxy();
    symmetrix::register_foundation_proxy();
}

extern "C" void registerKernelFactories() {}
