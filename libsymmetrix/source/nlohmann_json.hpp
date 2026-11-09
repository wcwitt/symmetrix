#pragma once

// Include the vendored nlohmann/json, isolated from LAMMPS's renamed copy.
//
// LAMMPS (since mid-2025, via atom.h -> json_fwd.h) ships its own nlohmann/json
// in namespace nlohmann_lmp. It uses its own include guards but defines the
// shared NLOHMANN_JSON_NAMESPACE* macros, which json.hpp only sets #ifndef. If
// LAMMPS's copy comes first, ours is silently placed in nlohmann_lmp and breaks
// on its hard-coded ::nlohmann:: references. Hide those macros while including
// our copy, then restore them so LAMMPS's copy is unaffected in either order.

#pragma push_macro("NLOHMANN_JSON_NAMESPACE")
#pragma push_macro("NLOHMANN_JSON_NAMESPACE_BEGIN")
#pragma push_macro("NLOHMANN_JSON_NAMESPACE_END")
#undef NLOHMANN_JSON_NAMESPACE
#undef NLOHMANN_JSON_NAMESPACE_BEGIN
#undef NLOHMANN_JSON_NAMESPACE_END

#include "../external/json/single_include/nlohmann/json.hpp"

#pragma pop_macro("NLOHMANN_JSON_NAMESPACE")
#pragma pop_macro("NLOHMANN_JSON_NAMESPACE_BEGIN")
#pragma pop_macro("NLOHMANN_JSON_NAMESPACE_END")
