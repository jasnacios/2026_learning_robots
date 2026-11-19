/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(robot_force_learning_k,FixRobotForceLearningK)

#else

#ifndef LMP_FIX_ROBOT_FORCE_LEARNING_K_H
#define LMP_FIX_ROBOT_FORCE_LEARNING_K_H

#include "fix.h"
#include "neigh_request.h"

namespace LAMMPS_NS {

class FixRobotForceLearningK : public Fix {
 public:
  FixRobotForceLearningK(class LAMMPS *, int, char **);
  ~FixRobotForceLearningK();
  int setmask();
  void setup(int);
  void setup_pre_reverse(int,int)override;
  void init();
  void init_list(int, class NeighList *)override;
  void post_force(int)override;
  void pre_reverse(int, int)override;
 
 private:
 class RanMars *random;
 int seed;
 int Nn;
 double alphaq;

 
 double comm_radius;
 double dt;
 
 char* idregion0;
 class Region *region0;
 char* idregion1;
 class Region *region1;

 double ilow;
 double ihigh;
 
 
 int communication;
 int comm_period;
 int comm_clock;

 NeighList *list = nullptr;
 
 int N;
 double alpha_c;

};

}

#endif
#endif
