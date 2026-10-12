/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(robot_force_learning_g,FixRobotForceLearningG)

#else

#ifndef LMP_FIX_ROBOT_FORCE_LEARNING_G_H
#define LMP_FIX_ROBOT_FORCE_LEARNING_G_H

#include "fix.h"
#include "neigh_request.h"

namespace LAMMPS_NS {

class FixRobotForceLearningG : public Fix {
 public:
  FixRobotForceLearningG(class LAMMPS *, int, char **);
  ~FixRobotForceLearningG();
  int setmask();
  void setup(int);
  void setup_pre_reverse(int,int);
  void init();
  void init_list(int, class NeighList *)override;
  void post_force(int)override;
  void pre_reverse_comm(int, int);
 
 private:
 class RanMars *random;
 int seed;
 int Nn;
 double alphaq;

 char* idregion0;
 class Region *region0;
 char* idregion1;
 class Region *region1;
 double comm_radius;
 double dt;
 int numforce;
 double ilow;
 double ihigh;
 double alpha_T;
 int communication;
 int comm_period;
 int comm_clock;
 NeighList *list = nullptr;
 int N;
 double lambda_T;

};

}

#endif
#endif
