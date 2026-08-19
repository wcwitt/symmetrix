#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(symmetrix/mace/atom,ComputeSymmetrixMACEatom)
// clang-format on
#else

#ifndef LMP_COMPUTE_SYMMETRIX_MACE_ATOM_H
#define LMP_COMPUTE_SYMMETRIX_MACE_ATOM_H

#include "compute.h"
#include <unordered_map>
#include "mace.hpp"

class MACE;

namespace LAMMPS_NS {

class ComputeSymmetrixMACEatom : public Compute {
 public:
  ComputeSymmetrixMACEatom(class LAMMPS *, int, char **);
  ~ComputeSymmetrixMACEatom() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;

  int pack_forward_comm(int n, int *list, double *buf,
                        int pbc_flag, int *pbc) override;
  void unpack_forward_comm(int n, int first, double *buf) override;

 private:
  class NeighList *list = nullptr;

  std::unique_ptr<MACE> mace;

  std::vector<int> mace_types;

  std::vector<int> node_types;
  std::vector<int> num_neigh;
  std::vector<int> neigh_types;
  std::vector<int> node_i;      
  std::vector<int> neigh_j;     
  std::vector<double> xyz;     
  std::vector<double> r;       

  std::vector<double> H1_comm;

  int num_channels = 0;
  int num_LM = 0;
  double r_cut = 0.0;
  int nmax = 0;

  void build_graph_from_neighlist(int &num_nodes, int &num_edges);

  const std::array<std::string,118> periodic_table =
    { "H", "He",
        "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
        "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
        "K",  "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                        "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

};

}  // namespace LAMMPS_NS

#endif
#endif
