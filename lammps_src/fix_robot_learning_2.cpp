/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_robot_force_learning_2.h"
#include <cmath>
#include "atom.h"
#include "update.h"
#include "error.h"
#include "comm.h"
#include "random_mars.h"
#include "region.h"
#include "domain.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRobotForceLearning2::FixRobotForceLearning2(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 12) error->all(FLERR,"Illegal fix robot_force_learning_2    command");
  
 
  alphaq = utils::numeric(FLERR,arg[3],false,lmp);
  Dp = utils::numeric(FLERR,arg[4],false,lmp);
  idregion0 = utils::strdup(arg[5]);
  region0 = domain->get_region_by_id(arg[5]);
  idregion1 = utils::strdup(arg[6]);
  region1 = domain->get_region_by_id(arg[6]);
  comm_radius = utils::numeric(FLERR,arg[7],false,lmp);
  alpha = utils::numeric(FLERR,arg[8],false,lmp);
  Nn = utils::numeric(FLERR,arg[9],false,lmp);
  seed = utils::numeric(FLERR,arg[10],false,lmp);
  numforce = utils::numeric(FLERR,arg[11],false,lmp);
  
  random = new RanMars(lmp, seed + comm->me);

}

/* ---------------------------------------------------------------------- */

FixRobotForceLearning2::~FixRobotForceLearning2()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearning2::init()
{
  dt = update->dt;
}

int FixRobotForceLearning2::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearning2::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearning2::post_force(int vflag)
{
  NeighList *list = neighbor->lists[0];

  int i, j, ii, jj, inum, jnum;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  inum = list ->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double **poidsnn = atom->poidsnn;
  double **dw = atom->dpoids;

  double *qreward = atom->qreward;
  double *dreward = atom->dreward;
  double *lightintensity = atom->lightintensity;

  double **x = atom->x;

  int step = update->ntimestep;
  
  int *mask = atom->mask;
  int nlocal = atom->nlocal;


  const double epsilon = 1.0e-10;
  const double comm_radius_sq = comm_radius * comm_radius;
  if (step<1){
    for (int i = 0; i < nlocal; i++) {
        qreward[i] = 0.0;
        dreward[i] = 0.0;
        for (int k = 0; k < Nn; k++){
          poidsnn[i][k] = random->uniform();
          dw[i][k] = 0.0;
        }
    }
  }
  if (step >= 1) {

    for(int i = 0; i < nlocal; i++) {
      dreward[i] = 0.0;
      for (int k = 0; k < Nn; k++){
        dw[i][k] = 0.0;
      }
    }

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
        
      jlist = firstneigh[i];
      jnum = numneigh[i];

      double maxscore = 1e-8;
      int maxj = -1;

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
    
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
        
        if (rsq < comm_radius_sq) {
          if (qreward[j] > maxscore) {
            maxscore = qreward[j];
            maxj = j;
          }
        }
      }
      if (maxj>=0 && qreward[i]<qreward[maxj]-epsilon) {
        dreward[i] = alpha*(qreward[maxj] - qreward[i])*dt;
        for (int k = 0; k<Nn; k++) {
          dw[i][k] = alpha*(poidsnn[maxj][k] - poidsnn[i][k])*dt;
        }
    }
    }

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if(region0->match(x[i][0], x[i][1], x[i][2])|| region1->match(x[i][0], x[i][1], x[i][2])) {
          lightintensity[i] = 0.7;
        } else {
          lightintensity[i] = 0.3;
        } 
        dreward[i] += alphaq*(lightintensity[i] - qreward[i]) *dt;
      }
    }  
  }
}

//if (numforce == 0){
          //if (rsq < comm_radius_sq) {
            //if (qreward[i] < qreward[j]-epsilon) {
              //dreward[i] += alpha*(qreward[j] - qreward[i])*dt/jnum;
              //for (int k = 0; k<Nn; k++) {
                //dw[i][k] += alpha*(poidsnn[j][k] - poidsnn[i][k])*dt/jnum;
              //}
            //}
          //}
        //}
        //else if (numforce == 1) {}
//for (int i = 0; i < nlocal; i++) {
      //if (mask[i] & groupbit) {
        //for(int k=0;k<Nn;k++) {
          //poidsnn[i][k] += dw[i][k] + random->gaussian() * sqrt(2*dt*Dp);
        //}
        //for (int k=0;k<Nn;k++) {
         // if(poidsnn[i][k] > 1.0) poidsnn[i][k] = 2 - poidsnn[i][k];
         // if(poidsnn[i][k] < 0.0) poidsnn[i][k] = - poidsnn[i][k];
        //}
      //}
    //}
