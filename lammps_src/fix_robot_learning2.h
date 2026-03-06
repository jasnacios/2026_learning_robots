/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(robot_learning2,FixRobotLearning2)

#else

#ifndef LMP_FIX_ROBOT_LEARNING2_H
#define LMP_FIX_ROBOT_LEARNING2_H

#include "fix.h"

namespace LAMMPS_NS {

class FixRobotLearning2 : public Fix {
 public:
  FixRobotLearning2(class LAMMPS *, int, char **);
  ~FixRobotLearning2();
  void grow_arrays();                                                      //grow arrays au cas ou le nombre d'atomes change (pathologique pr l'instant) 
  int setmask();                                                           //on va le mettre dans pre_force pr modifier poids et score avant de calculer la sortie des controler
  void setup(int);
  void init();                                                             // remet les dq dw a 0 et initialise poids/score + fait demande de voisins
  //void init_list(int id, NeighList *ptr);                                  // pointe vers la liste de voisins
  void post_force(int);
  int pack_reverse_comm(int n, int first, double *buf) override;           // renvoie les variations de score/ poids aux ghosts
  void unpack_reverse_comm(int n, int *list, double *buf) override;

  

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
 double **dw = nullptr;   
 double *dq  = nullptr; 

 int maxatom_ = 0;
 //NeighList *list_ = nullptr;
  bool initialized_ = false;
 

};
}
#endif
#endif
