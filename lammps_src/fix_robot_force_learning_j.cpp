/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_robot_force_learning_j.h"
#include <cmath>
#include "atom.h"
#include "update.h"
#include "error.h"
#include "comm.h"
#include "random_mars.h"
#include "region.h"
#include "domain.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRobotForceLearningJ::FixRobotForceLearningJ(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 11) error->all(FLERR,"Illegal fix robot_force_learning_j command");
  
 
  alphaq = utils::numeric(FLERR,arg[3],false,lmp);
  lambda_T = utils::numeric(FLERR,arg[4],false,lmp);
  N = utils::numeric(FLERR,arg[5],false,lmp);
  comm_radius = utils::numeric(FLERR,arg[6],false,lmp);
  communication = utils::numeric(FLERR,arg[7],false,lmp);
  Nn = utils::numeric(FLERR,arg[8],false,lmp);
  
  seed = utils::numeric(FLERR,arg[9],false,lmp);
  comm_period = utils::numeric(FLERR,arg[10],false,lmp);


  random = new RanMars(lmp, seed + comm->me);


}

/* ---------------------------------------------------------------------- */

FixRobotForceLearningJ::~FixRobotForceLearningJ()
{
  delete random;
}
/* ---------------------------------------------------------------------- */

void FixRobotForceLearningJ::init()
{
  dt = update->dt;
  comm_clock = 0;

  double *qreward = atom->qreward;
  double **poidsnn = atom->poidsnn;

  auto req = neighbor->add_request(this,  NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL); // 
  req->set_cutoff(comm_radius);

  
}
void FixRobotForceLearningJ::init_list(int id, NeighList *ptr)
{
list = ptr;
}

/* ---------------------------------------------------------------------- */

int FixRobotForceLearningJ::setmask()
{
  int mask = 0;
  mask |= PRE_REVERSE;
  mask |= POST_FORCE;

  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningJ::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningJ::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningJ::pre_reverse(int eflag, int vflag)
{ 
  /*if (comm->me == 0) {
        printf("PRE_REVERSE : step = %lld\n",
               (long long) update->ntimestep);
        fflush(stdout);
    }*/

  neighbor->build_one(list);

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
  int nghost = atom->nghost;

  int nall = nlocal + nghost;


  const double epsilon = 1.0e-10;


  if (step<1){
    for (int i = 0; i < nall; i++) {

      qreward[i] = random->uniform();
      dreward[i] = 0.0;

      /*for (int k = 0; k < Nn-2; k++) {
        poidsnn[i][k] = random->uniform();
        dw[i][k] = 0.0;
      }*/
      poidsnn[i][0] = random->uniform();
      poidsnn[i][1] = 0.0;
      poidsnn[i][2] = 0.0;
      dw[i][0] = 0.0;
      dw[i][1] = 0.0;
      dw[i][2] = 0.0;
    }
  }
  if (step >= 1) {

    comm_clock++;
    for(int i = 0; i < nall; i++) {
        dreward[i] = 0.0;
        for (int k = 0; k < Nn; k++){
          dw[i][k] = 0.0;
        }
      }
      
    if (comm_clock >= comm_period) {

      
      if (communication ==1){ // Communication 1: échange de poids et de score entre agents
        for (ii = 0; ii < inum; ii++) {
              
          i = ilist[ii];
          
          jlist = firstneigh[i];
          jnum = numneigh[i];
          

          poidsnn[i][2] = 0.0; 
          
          if (jnum == 0) continue;

          jlist = firstneigh[i];
          jnum = numneigh[i];

          int nclose = 0;

          for (int jj = 0; jj < jnum; jj++) {

              int j = jlist[jj] & NEIGHMASK;

              if (atom->tag[i] == atom->tag[j]) continue;
              if (j == i) continue;

              double dx = x[i][0] - x[j][0];
              double dy = x[i][1] - x[j][1];

              double rsq = dx*dx + dy*dy;

              if (rsq < comm_radius*comm_radius) {
                  nclose++;
              }
          }

          poidsnn[i][2] = nclose;
          poidsnn[i][1]++;

          int jj = static_cast<int>(random->uniform() * jnum);

          int j = jlist[jj];
          j &= NEIGHMASK;

          if (j == i) continue;
          if (atom->tag[j] == atom->tag[i]) continue;

          if (random->uniform()<lambda_T) {

            if (qreward[i]<qreward[j] -epsilon){
              dreward[i] = qreward[j] - qreward[i];
              dw[i][0] = poidsnn[j][0] - poidsnn[i][0];
              /*for (int k = 0; k<Nn; k++) {
                dw[i][k] = poidsnn[j][k] - poidsnn[i][k];
              }*/
            }
            else if (qreward[i]>qreward[j] + epsilon){
              dreward[j] = qreward[i] - qreward[j];
              dw[j][0] = poidsnn[i][0] - poidsnn[j][0];
              /*for (int k = 0; k<Nn; k++) {
                dw[j][k] = poidsnn[i][k] - poidsnn[j][k];
              }*/
            }
            else {
              if (random->uniform() < 0.5) {
                dreward[i] = qreward[j] - qreward[i];
                dw[i][0] = poidsnn[j][0] - poidsnn[i][0];
                /*for (int k = 0; k<Nn; k++) {
                  dw[i][k] = poidsnn[j][k] - poidsnn[i][k];
                }*/
              }
              else {
                dreward[j] = qreward[i] - qreward[j];
                dw[j][0] = poidsnn[i][0] - poidsnn[j][0];
              /*for (int k = 0; k<Nn; k++) {
                dw[j][k] = poidsnn[i][k] - poidsnn[j][k];
              }*/
              }
            }
          }
          
        } 
      }

      comm_clock = 0;
    }
  }

}

void FixRobotForceLearningJ::post_force(int vflag)
{ 
  int i;
  double *qreward = atom->qreward;
  double *dreward = atom->dreward;
  double *lightintensity = atom->lightintensity;

  double **x = atom->x;

  int step = update->ntimestep;
  
  int *mask = atom->mask;
  int nlocal = atom->nlocal;


  
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      /*if(region0->match(x[i][0], x[i][1], x[i][2])|| region1->match(x[i][0], x[i][1], x[i][2])) {
        lightintensity[i] = ihigh;
      } else {
        lightintensity[i] = ilow;
      }*/
      lightintensity[i] = (1 - std::abs(x[i][0])/50);
      qreward[i] += dreward[i];
      dreward[i] = alphaq*(lightintensity[i] - qreward[i]) *dt;
    }
  } 
}
      
    
  