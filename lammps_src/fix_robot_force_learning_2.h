/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(robot_force_learning2,FixRobotForceLearning2)

#else

#ifndef LMP_FIX_ROBOT_FORCE_LEARNING_2_H
#define LMP_FIX_ROBOT_FORCE_LEARNING_2_H

#include "fix.h"


namespace LAMMPS_NS {

class FixRobotForceLearning2: public Fix {
 public:
  FixRobotForceLearning2(class LAMMPS *, int, char **);
  ~FixRobotForceLearning2();
  int setmask();
  void setup(int);
  void init();
  void post_force(int) override;

 private:
 class RanMars *random;
 int seed;
 int Nn;
 double alphaq;
 double Dp;
 char* idregion0;
 class Region *region0;
 char* idregion1;
 class Region *region1;
 double alpha;
 double comm_radius;
 double dt;
 int numforce;
};

}

#endif
#endif
