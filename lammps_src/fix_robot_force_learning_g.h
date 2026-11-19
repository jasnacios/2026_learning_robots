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

 NeighList *list = nullptr;
 
 int N;
 double lambda_T;

};

}

#endif
#endif
