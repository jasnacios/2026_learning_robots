/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_robot_learning2.h"
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
#include "neigh_request.h" 
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixRobotLearning2::FixRobotLearning2(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg){
  if (narg < 9) error->all(FLERR,"Illegal fix robot_learning2 command");
 
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
  
 
  random = new RanMars(lmp, seed + comm->me);
  dq = nullptr;
  dw = nullptr;
  this->comm_reverse = Nn + 1;
}

/* ---------------------------------------------------------------------- */

FixRobotLearning2::~FixRobotLearning2(){
  delete random;
  
  
}

void FixRobotLearning2::grow_arrays(){
  int nmax = atom->nmax;
  if (nmax <= maxatom_) return;

  maxatom_ = nmax;

  memory->grow(dq, maxatom_, "fix_robot_learning2:dq");
  memory->grow(dw, maxatom_, Nn, "fix_robot_learning2:dw");
}

void FixRobotLearning2::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotLearning2::init(){

  dt = update->dt;
  maxatom_ = atom->nmax;

  memory->create(dq, maxatom_, "fix:dq");
  memory->create(dw, maxatom_, Nn, "fix:dw");

  double *qreward = atom->qreward;
  double **poidsnn = atom->poidsnn;

  int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++) {
        dq[i] = 0.0;
        for (int k = 0; k < Nn; k++) dw[i][k] = 0.0;
    }
  if (!initialized_) {
        double *qreward   = atom->qreward;
        double **poidsnn  = atom->poidsnn;
        int nlocal        = atom->nlocal;
        for (int i = 0; i < nlocal; i++) {
            qreward[i] = 0.0;
            for (int k = 0; k < Nn; k++)
                poidsnn[i][k] = random->uniform();
        }
        initialized_ = true;
    }
  //auto req = neighbor->add_request(this, NeighConst::REQ_GHOST);
  //req->set_cutoff(comm_radius);
}

//void FixRobotLearning2::init_list(int id, NeighList *ptr){
 // list_ = ptr;
//}


int FixRobotLearning2::setmask(){
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

int FixRobotLearning2::pack_reverse_comm(int n, int first, double *buf) {
    int m = 0;
    for (int i = first; i < first + n; i++) {
        buf[m++] = dq[i];
        for (int k = 0; k < Nn; k++)
            buf[m++] = dw[i][k];
    }
    int expected = n * (Nn + 1);
    if (m != expected) error->all(FLERR,"pack_reverse_comm wrong size");
    return m;
}

void FixRobotLearning2::unpack_reverse_comm(int n, int *list, double *buf) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int j = list[i];
        dq[j] += buf[m++];  // += pour accumuler
        for (int k = 0; k < Nn; k++)
            dw[j][k] += buf[m++];
    }
}

void FixRobotLearning2::post_force(int vflag)
{
  grow_arrays();
  int nall = atom->nlocal + atom->nghost;
  if (nall > maxatom_) error->one(FLERR,"dq/dw too small after grow");
  //this->list_;
  NeighList *list_ = neighbor->lists[0];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  int i, j, ii, jj, inum, jnum;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  inum = list_->inum;
  ilist = list_->ilist;
  numneigh = list_->numneigh;
  firstneigh = list_->firstneigh;

  double **poidsnn = atom->poidsnn;
  double *qreward = atom->qreward;
  double *lightintensity = atom->lightintensity;

  double **x = atom->x;
  int step = update->ntimestep;

  const double epsilon = 1.0e-10;
  const double comm_radius_sq = comm_radius * comm_radius;
  
  

  if (step >= 1) {
    int nall = atom->nlocal + atom->nghost;
    for (int i = 0; i < nall; i++) {
      dq[i] = 0.0;
      for (int k = 0; k < Nn; k++){
        dw[i][k] = 0.0;}}

    for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (i >= nlocal) continue;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;

        if (rsq == 0) continue;
        if (rsq < comm_radius_sq) {
            if (qreward[i] < qreward[j] - epsilon) {
                dq[i] += alpha*(qreward[j] - qreward[i])*dt/jnum;
                for (int k = 0; k < Nn; k++)
                    dw[i][k] += alpha*(poidsnn[j][k] - poidsnn[i][k])*dt/jnum;
            }
        }  // ← ferme if(rsq)
    }      // ← ferme for(jj)
}          // ← ferme for(ii)
        
      
    comm->reverse_comm(this);


    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        for(int k=0;k<Nn;k++) {
          poidsnn[i][k] += dw[i][k]+random->gaussian() * sqrt(2*dt*Dp);
        }
        for (int k=0;k<Nn;k++) {
          if(poidsnn[i][k] > 1.0) poidsnn[i][k] = 2 - poidsnn[i][k];
          if(poidsnn[i][k] < 0.0) poidsnn[i][k] = - poidsnn[i][k];
        }
      }
    }
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if(region0->match(x[i][0], x[i][1], x[i][2])|| region1->match(x[i][0], x[i][1], x[i][2])) {
          lightintensity[i] = 0.9;
        } else {
          lightintensity[i] = 0.1;
        } 
        // Update reward
        qreward[i] += dq[i]+alphaq*(lightintensity[i] - qreward[i]) *dt;
      }
    }  
  }
}

