/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */
#include "fix_integration_learning.h"
#include <cmath>
#include "atom.h"
#include "update.h"
#include "error.h"
#include "comm.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixIntegrationLearning::FixIntegrationLearning(LAMMPS *lmp, int narg, char **arg) :
Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix integration_learning command");
  
  Dp = utils::numeric(FLERR,arg[3],false,lmp);
  Nn = utils::numeric(FLERR,arg[4],false,lmp);
  seed = utils::numeric(FLERR,arg[5],false,lmp);


  random = new RanMars(lmp, seed + comm->me);
  
  // Indiquer que ce fix réalise une intégration temporelle 
  time_integrate = 0;
}

/* ---------------------------------------------------------------------- */

FixIntegrationLearning::~FixIntegrationLearning()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

void FixIntegrationLearning::init()
{
  dt = update->dt;
  
}

/* ---------------------------------------------------------------------- */

int FixIntegrationLearning::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixIntegrationLearning::setup(int vflag)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixIntegrationLearning::end_of_step()
{
  double **poidsnn = atom->poidsnn;
  double **dpoids = atom->dpoids;
  double *qreward = atom->qreward;
  double *dreward = atom->dreward;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  int step = update->ntimestep;
  
  if (step > 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        qreward[i] += dreward[i];
        for(int k=0;k<Nn;k++) {
          poidsnn[i][k] +=  dpoids[i][k] + random->gaussian() * sqrt(2*dt*Dp);
        }
        for (int k=0;k<Nn;k++) {
          if(poidsnn[i][k] > 1.0) poidsnn[i][k] = 2 - poidsnn[i][k];
          if(poidsnn[i][k] < 0.0) poidsnn[i][k] = - poidsnn[i][k];
        }
      }
    }
  }
}
