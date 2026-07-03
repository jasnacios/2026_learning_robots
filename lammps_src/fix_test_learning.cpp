/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_test_learning.h"
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

FixTestLearning::FixTestLearning(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix brownian/2d command");
  
 
  alphaq = utils::numeric(FLERR,arg[3],false,lmp);
  idregion0 = utils::strdup(arg[4]);
  idregion1 = utils::strdup(arg[5]);
  comm_radius = utils::numeric(FLERR,arg[6],false,lmp);
  Nn = utils::numeric(FLERR,arg[7],false,lmp);
  seed = utils::numeric(FLERR,arg[8],false,lmp);
  ilow = utils::numeric(FLERR,arg[9],false,lmp);
  ihigh = utils::numeric(FLERR,arg[10],false,lmp);
  alphaT = utils::numeric(FLERR,arg[11],false,lmp);
  communication_flag = utils::numeric(FLERR,arg[12],false,lmp);

  random = new RanMars(lmp, seed + comm->me);
  region0 = domain->get_region_by_id(arg[4]);
  region1 = domain->get_region_by_id(arg[5]);

}

/* ---------------------------------------------------------------------- */

FixTestLearning::~FixTestLearning()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

void FixTestLearning::init()
{
  dt = update->dt;
  
}

/* ---------------------------------------------------------------------- */

int FixTestLearning::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTestLearning::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixTestLearning::post_force(int vflag)
{
  int i, j, ii, jj, inum, jnum;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **poidsnn = atom->poidsnn;
  double *ztorque = atom->ztorque;
  double *qreward = atom->qreward;
  double *lightintensity = atom->lightintensity;

  double **x = atom->x;
  int step = update->ntimestep;

  NeighList *list = neighbor->lists[0];
  
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double **dw = atom->dpoids;
  double *dreward = atom->dreward;
  
  if (step < 1) {
    for (int i = 0; i < nlocal; i++) {
    qreward[i] = 0.0;
    for (int k = 0; k < Nn; k++){
      poidsnn[i][k] = random->uniform();}}
  }
  if (step >= 1) {
    int nlocal_seen = 0;
    int n_attempt = 0;
    int n_ppv = 0;
    int n_exchange = 0;
    if (communication_flag ==1){
      int count_exchange = 0;
      int count_minj = 0;
      for (ii = 0; ii < inum; ii++) {
        if (random->uniform()<alphaT){
          i = ilist[ii];
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
            
          jlist = firstneigh[i];
          jnum = numneigh[i];

          int minj = -1;
          double min_dist = 10000;
          
          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;
            if (j == i) continue;
            if (atom->tag[j] == atom->tag[i]) continue;
        
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
              
            if(rsq == 0) continue;
            else if (rsq > comm_radius*comm_radius) continue;
            else if (rsq <min_dist){
              min_dist = rsq;
              minj = j;
            }
          }
          if (minj >= 0) count_minj++;
          if (minj>=0){
            if (qreward[i] < qreward[minj]) {
              count_exchange++;
              dreward[i] = qreward[minj] - qreward[i];
              for (int k = 0; k<Nn; k++) {
                dw[i][k]  = poidsnn[minj][k] - poidsnn[i][k]; 
              }
            }
          }
          
              
        }
      }
      if (update->ntimestep % 1000 == 0 && comm->me == 0)
      fprintf(screen, "Step %ld: échanges=%d\n", 
          update->ntimestep, count_exchange, inum);
      if (update->ntimestep % 1000 == 0 && comm->me == 0)
      fprintf(screen, "Step %ld: ppv trouvé pour %d/%d atomes\n", 
          update->ntimestep, count_minj, inum);
    }

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        qreward[i] += dreward[i];
        for (int k = 0; k < Nn; k++) {
          poidsnn[i][k] += dw[i][k];
        }
        if(region0->match(x[i][0], x[i][1], x[i][2])|| region1->match(x[i][0], x[i][1], x[i][2])) {
          lightintensity[i] = ihigh;;
        } else {
          lightintensity[i] = ilow;
        } 
        // Update reward
        dreward[i] = alphaq*(lightintensity[i] - qreward[i]) *dt;
      }
    }  
  }
}