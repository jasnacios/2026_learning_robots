/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(robot_force_learning_j,FixRobotForceLearningJ)

#else

#ifndef LMP_FIX_ROBOT_FORCE_LEARNING_J_H
#define LMP_FIX_ROBOT_FORCE_LEARNING_J_H

#include "fix.h"
#include "neigh_request.h"

namespace LAMMPS_NS {

class FixRobotForceLearningJ : public Fix {
 public:
  FixRobotForceLearningJ(class LAMMPS *, int, char **);
  ~FixRobotForceLearningJ();
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
