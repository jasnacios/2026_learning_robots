/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_robot_force_learning_k.h"
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

FixRobotForceLearningK::FixRobotForceLearningK(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 11) error->all(FLERR,"Illegal fix robot_force_learning_k command");
  
 
  alphaq = utils::numeric(FLERR,arg[3],false,lmp);
  alpha_c = utils::numeric(FLERR,arg[4],false,lmp);
  N = utils::numeric(FLERR,arg[5],false,lmp);
  comm_radius = utils::numeric(FLERR,arg[6],false,lmp);
  communication = utils::numeric(FLERR,arg[7],false,lmp);
  Nn = utils::numeric(FLERR,arg[8],false,lmp);
  
  seed = utils::numeric(FLERR,arg[9],false,lmp);
  comm_period = utils::numeric(FLERR,arg[10],false,lmp);

  idregion0 = utils::strdup(arg[11]);
  idregion1 = utils::strdup(arg[12]);

  ilow = utils::numeric(FLERR,arg[13],false,lmp);
  ihigh = utils::numeric(FLERR,arg[14],false,lmp);

  region0 = domain->get_region_by_id(arg[11]);
  region1 = domain->get_region_by_id(arg[12]);

  random = new RanMars(lmp, seed + comm->me);


}

/* ---------------------------------------------------------------------- */

FixRobotForceLearningK::~FixRobotForceLearningK()
{
  delete random;
}
/* ---------------------------------------------------------------------- */

void FixRobotForceLearningK::init()
{
  dt = update->dt;
  comm_clock = 0;

  double *qreward = atom->qreward;
  double **poidsnn = atom->poidsnn;

  auto req = neighbor->add_request(this,  NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL); 
  req->set_cutoff(comm_radius);

  
}
void FixRobotForceLearningK::init_list(int id, NeighList *ptr)
{
list = ptr;
}

/* ---------------------------------------------------------------------- */

int FixRobotForceLearningK::setmask()
{
  int mask = 0;
  mask |= PRE_REVERSE;
  mask |= POST_FORCE;

  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningK::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningK::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningK::pre_reverse(int eflag, int vflag)
{ neighbor->build_one(list);

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

      qreward[i] = 0.6*random->uniform()+0.2;
      dreward[i] = 0.0;

      poidsnn[i][0] = random->uniform();
      dw[i][0] = 0.0;
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
      
    if (communication ==1) {// Communication 1: échange de poids et de score entre agents

      if (comm_clock >= comm_period){ 

        for (ii = 0; ii < inum; ii++) {
              
          i = ilist[ii];
          
          jlist = firstneigh[i];
          jnum = numneigh[i];
          
          //poidsnn[i][2] = 0.0;
          
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
              if (random->uniform()<alpha_c) {

                if (qreward[i]>= 0.65 && qreward[j]>=0.65){
                  if (random->uniform() < 0.5) {
                    qreward[i] = qreward[j];
                    poidsnn[i][0] = poidsnn[j][0];
                  }
                }

                else if (qreward[i] <=0.35 && qreward[j]<=0.35){
                  if (random->uniform() < 0.5) {
                    qreward[i] = qreward[j];
                    poidsnn[i][0] = poidsnn[j][0];
                  }
                }

                else if (qreward[i]<qreward[j] -epsilon){
                  qreward[i] = qreward[j];
                  poidsnn[i][0] = poidsnn[j][0];
                  /*dreward[i] = qreward[j] - qreward[i];
                  dw[i][0] = poidsnn[j][0] - poidsnn[i][0];*/
                }
                else if (fabs(qreward[i]-qreward[j])<= epsilon){
                  if (random->uniform() < 0.5) {
                    qreward[i] = qreward[j];
                    poidsnn[i][0] = poidsnn[j][0];

                    /*dreward[i] = qreward[j] - qreward[i];
                    dw[i][0] = poidsnn[j][0] - poidsnn[i][0];*/
                  }
                }
              }

            }
          }
          //poidsnn[i][2] = nclose;
          //poidsnn[i][1]++;
        }
        comm_clock = 0;
      }
    }
  }

}

void FixRobotForceLearningK::post_force(int vflag)
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
      
    
  