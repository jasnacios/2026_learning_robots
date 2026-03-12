/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(multiLayerPerceptron,FixMLP)

#else

#ifndef LMP_FIX_MLP_H
#define LMP_FIX_MLP_H

#include "fix.h"
#include <vector>

namespace LAMMPS_NS {

struct FlatView { // Flatten arrays of inputs and outputs
    double* ptr;
    int stride;
};

enum ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    LINEAR
};

class FixMLP : public Fix {
 public:
  FixMLP(class LAMMPS *, int, char **);
  int setmask();
  void setup(int);
  void post_force(int);

  void MultiLayerPerceptron(const double* w, const int atomIndex) const;
  void CheckWeightsFit() const;

 private:
  unsigned int _nbLayers;
  unsigned int _nbNeuronsPerLayer;
  std::vector<std::string> _activationFunctions;
  std::vector<std::string> _outputs;
  std::vector<std::string> _inputs;

  // Flat views for inputs and outputs, allowing to retrieve pointers
  // from the requested properties as string and use them in the MLP
  std::vector<LAMMPS_NS::FlatView> _features;
  std::vector<LAMMPS_NS::FlatView> _out;
  std::vector<ActivationFunction> _activationFunctionEnums;
  void _buildViews();
  void _buildActivationFunctionEnums();
  std::vector<std::string> _split_csv(const std::string &s);

  int _nevery;
};

}

#endif
#endif
