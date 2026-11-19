/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_robot_force_learning_g.h"
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

FixRobotForceLearningG::FixRobotForceLearningG(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 10) error->all(FLERR,"Illegal fix robot_force_learning_g command");
  
 
  alphaq = utils::numeric(FLERR,arg[3],false,lmp);
  lambda_T = utils::numeric(FLERR,arg[4],false,lmp);
  N = utils::numeric(FLERR,arg[5],false,lmp);
  comm_radius = utils::numeric(FLERR,arg[6],false,lmp);
  communication = utils::numeric(FLERR,arg[7],false,lmp);
  Nn = utils::numeric(FLERR,arg[8],false,lmp);
  
  seed = utils::numeric(FLERR,arg[9],false,lmp);
  


  random = new RanMars(lmp, seed + comm->me);


}

/* ---------------------------------------------------------------------- */

FixRobotForceLearningG::~FixRobotForceLearningG()
{
  delete random;
}
/* ---------------------------------------------------------------------- */

void FixRobotForceLearningG::init()
{
  dt = update->dt;
  

  double *qreward = atom->qreward;
  double **poidsnn = atom->poidsnn;

  auto req = neighbor->add_request(this,   NeighConst::REQ_OCCASIONAL);
  req->set_cutoff(comm_radius);

  
}
void FixRobotForceLearningG::init_list(int id, NeighList *ptr)
{
list = ptr;
}

/* ---------------------------------------------------------------------- */

int FixRobotForceLearningG::setmask()
{
  int mask = 0;
  mask |= PRE_REVERSE;
  mask |= POST_FORCE;

  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningG::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningG::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixRobotForceLearningG::pre_reverse(int eflag, int vflag)
{ 
  neighbor->build_one(list);
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

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

  double pi = 3.14159265358979323846;
  double rho = static_cast<double>(N)/(10000);
  double scomm = pi*comm_radius*comm_radius;


  const double epsilon = 1.0e-10;


  if (step<1){
    for (int i = 0; i < nall; i++) {

      qreward[i] = random->uniform();
      dreward[i] = 0.0;

      for (int k = 0; k < Nn; k++) {
        poidsnn[i][k] = random->uniform();
        dw[i][k] = 0.0;
      }
    }
  }
  if (step >= 1) {

    for(int i = 0; i < nall; i++) {
      dreward[i] = 0.0;
      for (int k = 0; k < Nn; k++){
        dw[i][k] = 0.0;
      }
    }
      
    if (communication ==1){ // Communication 1: échange de poids et de score entre agents
     

      bigint Npairs_local = 0;
      for (int ii = 0; ii < inum; ii++) Npairs_local += numneigh[ilist[ii]];

      bigint Npairs_global = 0;
      MPI_Allreduce(&Npairs_local, &Npairs_global, 1, MPI_LMP_BIGINT, MPI_SUM, world);
      if (Npairs_global == 0) return;
      
      double p = lambda_T * Npairs_global * dt*rho*scomm; // probabilité de communication par pas de temps

      int do_communication = 0;

      // Tirage uniquement sur le processus 0
      if (comm->me == 0) {
        do_communication = (random->uniform() < lambda_T*dt*Npairs_global*rho*scomm);
    
        printf("lambda_T = %.15e\n", static_cast<double>(lambda_T));
        printf("dt       = %.15e\n", static_cast<double>(dt));
        printf("Npairs   = %lld\n", static_cast<long long>(Npairs_global));
        printf("N      = %.15e\n", static_cast<double>(N));
        printf("rho      = %.15e\n", static_cast<double>(rho));
        printf("scomm    = %.15e\n", static_cast<double>(scomm));
        printf("p        = %.15e\n", p);
      }

      // Communiquer le résultat à tous les processus
      MPI_Bcast(&do_communication, 1, MPI_INT, 0, world);

      if (do_communication) {

        bigint cumulative = 0;
        MPI_Scan(&Npairs_local, &cumulative, 1, MPI_LMP_BIGINT, MPI_SUM, world);
        bigint offset = cumulative - Npairs_local;

        bigint k_global = 0;
        if (comm->me == 0) {
          k_global = static_cast<bigint>(random->uniform() * Npairs_global);
          if (k_global >= Npairs_global) k_global = Npairs_global - 1;
        }
        MPI_Bcast(&k_global, 1, MPI_LMP_BIGINT, 0, world);

        if (k_global >= offset && k_global < offset + Npairs_local) {
          bigint k_local = k_global - offset;
          for (int ii = 0; ii < inum; ii++) {
            int i = ilist[ii];
            int jnum = numneigh[i];
            if (k_local >= jnum) { k_local -= jnum; continue; }


            int j = firstneigh[i][static_cast<int>(k_local)] & NEIGHMASK;
            
            if (qreward[i]<qreward[j] -epsilon){
              dreward[i] = qreward[j] - qreward[i];
              for (int k = 0; k<Nn; k++) {
                dw[i][k] = poidsnn[j][k] - poidsnn[i][k];
              }
            }
            else if (qreward[i]>qreward[j] + epsilon){
              dreward[j] = qreward[i] - qreward[j];
              for (int k = 0; k<Nn; k++) {
                dw[j][k] = poidsnn[i][k] - poidsnn[j][k];
              }
            }
            else {
              if (random->uniform() < 0.5) {
                dreward[i] = qreward[j] - qreward[i];
                for (int k = 0; k<Nn; k++) {
                  dw[i][k] = poidsnn[j][k] - poidsnn[i][k];
                }
              }
              else {
                dreward[j] = qreward[i] - qreward[j];
                for (int k = 0; k<Nn; k++) {
                  dw[j][k] = poidsnn[i][k] - poidsnn[j][k];
                }
              }
            break;
            }
          }
        }
      } 
    }
  }
}


void FixRobotForceLearningG::post_force(int vflag)
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
      
    
  