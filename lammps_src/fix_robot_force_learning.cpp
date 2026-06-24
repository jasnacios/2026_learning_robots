/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_robot_force_learning.h"
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

FixRobotForceLearning::FixRobotForceLearning(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 14) error->all(FLERR,"Illegal fix robot_force_learning command");
  
 
  alphaq = utils::numeric(FLERR,arg[3],false,lmp);
  idregion0 = utils::strdup(arg[4]);
  idregion1 = utils::strdup(arg[5]);
  comm_radius = utils::numeric(FLERR,arg[6],false,lmp);
  communication = utils::numeric(FLERR,arg[7],false,lmp);
  Nn = utils::numeric(FLERR,arg[8],false,lmp);
  seed = utils::numeric(FLERR,arg[9],false,lmp);
  numforce = utils::numeric(FLERR,arg[10],false,lmp);
  ilow = utils::numeric(FLERR,arg[11],false,lmp);
  ihigh = utils::numeric(FLERR,arg[12],false,lmp);
  alpha_T = utils::numeric(FLERR,arg[13],false,lmp);

  random = new RanMars(lmp, seed + comm->me);
  region0 = domain->get_region_by_id(arg[4]);
  region1 = domain->get_region_by_id(arg[5]);

}

/* ---------------------------------------------------------------------- */

FixRobotForceLearning::~FixRobotForceLearning()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearning::init()
{
  dt = update->dt;
  int nall = atom->nlocal + atom->nghost;
  double *qreward = atom->qreward;
  double **poidsnn = atom->poidsnn;

  
  auto req = neighbor->add_request(this, NeighConst::REQ_FULL);
  req->set_id(1);
  req->set_cutoff(comm_radius);
  
}
void FixRobotForceLearning::init_list(int id, NeighList *ptr)
{
  if (id == 1) list = ptr;
}

/* ---------------------------------------------------------------------- */

int FixRobotForceLearning::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearning::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearning::post_force(int vflag)
{

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
        
        for (int k = 0; k < Nn; k++) {
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

    if (communication ==1){  // Communication 1: échange de poids et de score entre agents

      if (numforce == 1){

        for (ii = 0; ii < inum; ii++) {
          
          i = ilist[ii];
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
            
          jlist = firstneigh[i];
          jnum = numneigh[i];

          double min_dist = 100;
          int ppv = -1;
          std::vector<int> pareilquei;

          if (random -> uniform() < alpha_T) {

            for (jj = 0; jj < jnum; jj++) {
              j = jlist[jj];
              j &= NEIGHMASK;
          
              delx = xtmp - x[j][0];
              dely = ytmp - x[j][1];
              delz = ztmp - x[j][2];
              rsq = delx * delx + dely * dely + delz * delz;
              
              if (rsq <= comm_radius_sq) {
                if (rsq< min_dist)  {
                  min_dist = rsq;
                  ppv = j;
                }
              }
            }
            if (ppv>=0 && qreward[i]<qreward[ppv]-epsilon) {

              dreward[i] = qreward[ppv] - qreward[i];
              for (int k = 0; k<Nn; k++) {
                dw[i][k] = poidsnn[ppv][k] - poidsnn[i][k]; 
              }

            }
            else if (ppv>=0 && fabs(qreward[ppv] - qreward[i]) <= epsilon) {

              if (random->uniform() < 0.5) {
                dreward[i] = qreward[ppv] - qreward[i];
                for (int k = 0; k<Nn; k++) {
                  dw[i][k] = poidsnn[ppv][k] - poidsnn[i][k]; 
                }
              }
            }
          }
        }
      }
      else if (numforce == 3){

        for (ii = 0; ii < inum; ii++) {
          
          i = ilist[ii];
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
            
          jlist = firstneigh[i];
          jnum = numneigh[i];

          double min_dist = 100;
          int ppv = -1;
          std::vector<int> pareilquei;

          if (random -> uniform() < alpha_T) {
            for (jj = 0; jj < jnum; jj++) {
              j = jlist[jj];
              j &= NEIGHMASK;
          
              delx = xtmp - x[j][0];
              dely = ytmp - x[j][1];
              delz = ztmp - x[j][2];
              rsq = delx * delx + dely * dely + delz * delz;
              
              if (rsq <= comm_radius_sq) {
                if (rsq< min_dist)  {
                  min_dist = rsq;
                  ppv = j;
                }
              }
            }
            if (ppv>=0 && qreward[i]<qreward[ppv]-epsilon) {

              dreward[i] = qreward[ppv] - qreward[i]; // alpha*(qreward[ppv] - qreward[i])*dt;
              for (int k = 0; k<Nn; k++) {
                dw[i][k] = poidsnn[ppv][k] - poidsnn[i][k];// alpha*(poidsnn[ppv][k] - poidsnn[i][k])*dt;
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if(region0->match(x[i][0], x[i][1], x[i][2])|| region1->match(x[i][0], x[i][1], x[i][2])) {
          lightintensity[i] = ihigh;
        } else {
          lightintensity[i] = ilow;
        } 
        qreward[i] += dreward[i];
        dreward[i] = alphaq*(lightintensity[i] - qreward[i]) *dt;
      }
    }  
  }
}


/* if (numforce == 0){    // numforce 0: copie le meilleur ou un random parmi les voisins qui ont un score similaire
        for (ii = 0; ii < inum; ii++) {
          
          i = ilist[ii];
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
            
          jlist = firstneigh[i];
          jnum = numneigh[i];

          double maxscore = -2.0;
          int maxj = -1;
          std::vector<int> pareilquei;


          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;
        
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            
            if (rsq <= comm_radius_sq) {
              if (qreward[j] > maxscore) {
                maxscore = qreward[j];
                maxj = j;
              }
              if (qreward[j]<= qreward[i]+epsilon && qreward[j]>= qreward[i] - epsilon){
                pareilquei.push_back(j);
              }
              
            }
          }
          if (maxj>=0 && qreward[i]<qreward[maxj]-epsilon) {

            dreward[i] = qreward[maxj] - qreward[i];
            
            for (int k = 0; k<Nn; k++) {
              dw[i][k] = poidsnn[maxj][k] - poidsnn[i][k];
            }
          }
          else if (maxj>=0 && fabs(qreward[maxj] - qreward[i]) <= epsilon) {

            if (random->uniform() < 0.5) {

              int semblable_size  =  pareilquei.size();
              int idx = (int)(random->uniform() * semblable_size);
              int chosen_j = pareilquei[idx];

              dreward[i] = qreward[chosen_j] - qreward[i]; 
              
              for (int k = 0; k<Nn; k++) {
                dw[i][k] = poidsnn[chosen_j][k] - poidsnn[i][k]; 
              }
            }
          }
        }
      }

      else if (numforce == 2){
        for (ii = 0; ii < inum; ii++) {
          
          i = ilist[ii];
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
            
          jlist = firstneigh[i];
          jnum = numneigh[i];

          double maxscore = -2.0;
          int maxj = -1;
          std::vector<int> pareilquei;


          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;
        
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;
            
            if (rsq <= comm_radius_sq) {
              if (qreward[j] > maxscore) {
                maxscore = qreward[j];
                maxj = j;
              }
            }
          }
          if (maxj>=0 && qreward[i]<qreward[maxj]-epsilon) {

            dreward[i] = qreward[maxj] - qreward[i]; // alpha*(qreward[maxj] - qreward[i])*dt;
            
            for (int k = 0; k<Nn; k++) {
              dw[i][k] = poidsnn[maxj][k] - poidsnn[i][k];// alpha*(poidsnn[maxj][k] - poidsnn[i][k])*dt;
            }
          }
        }
      }

*/
