/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#ifdef FIX_CLASS
// clang-format off
FixStyle(IntegrationLearning,FixIntegrationLearning);
// clang-format on
#else

#ifndef LMP_FIX_INTEGRATION_LEARNING_H
#define LMP_FIX_INTEGRATION_LEARNING_H

#include "fix.h"

namespace LAMMPS_NS {

class FixIntegrationLearning : public Fix {
 public:
  FixIntegrationLearning(class LAMMPS *, int, char **);
  ~FixIntegrationLearning();
  int setmask();
  void init();
  void setup(int);
  void initial_integrate(int);
  

 private:
  double Dp;
  int Nn;
  int seed;
  double dt;
  class RanMars *random;
};

}

#endif
#endif
