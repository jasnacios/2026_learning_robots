/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(test_learning,FixTestLearning)

#else

#ifndef LMP_FIX_TEST_LEARNING_H
#define LMP_FIX_TEST_LEARNING_H

#include "fix.h"

namespace LAMMPS_NS {

class FixTestLearning : public Fix {
 public:
  FixTestLearning(class LAMMPS *, int, char **);
  ~FixTestLearning();
  int setmask();
  void setup(int);
  void init();
  void post_force(int);

 private:
 class RanMars *random;
 int seed;
 int Nn;
 double alphaq;
 

 char* idregion0;
 class Region *region0;
 char* idregion1;
 class Region *region1;

 double alphaT;

 double ilow;
 double ihigh;
 double comm_radius;
 double dt;
 double communication_flag;

};

}

#endif
#endif