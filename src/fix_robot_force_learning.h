/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(robot_force_learning,FixRobotForceLearning)

#else

#ifndef LMP_FIX_ROBOT_FORCE_LEARNING_H
#define LMP_FIX_ROBOT_FORCE_LEARNING_H

#include "fix.h"
#include "neigh_request.h"

namespace LAMMPS_NS {

class FixRobotForceLearning : public Fix {
 public:
  FixRobotForceLearning(class LAMMPS *, int, char **);
  ~FixRobotForceLearning();
  int setmask();
  void setup(int);
  void init();
  void init_list(int, class NeighList *)override;
  void post_force(int)override;

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
 NeighList *list = nullptr;

};

}

#endif
#endif
