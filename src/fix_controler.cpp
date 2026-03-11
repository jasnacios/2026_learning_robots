/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#include "fix_controler.h"
#include <cmath>
#include "atom.h"
#include "update.h"
#include "error.h"
#include <span>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixControler::FixControler(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix /active/force command");
  controler_type = utils::numeric(FLERR,arg[3],false,lmp);
  total_neurons = utils::numeric(FLERR,arg[4],false,lmp);
  hidden_size = utils::numeric(FLERR,arg[5],false,lmp);
}

/* ---------------------------------------------------------------------- */

int FixControler::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixControler::setup(int vflag)
{
  post_force(vflag);
} 

/* ---------------------------------------------------------------------- */

void FixControler::post_force(int vflag)
{
  double *lightintensity = atom->lightintensity;
  double *clock = atom->clock;
  double *Fa = atom->Fa;
  double *ztorque = atom->ztorque;
  double *zeta = atom->zeta;
  double **poidsnn = atom->poidsnn;

  int *mask = atom->mask;    
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;
    double I = lightintensity[i];
    double t = clock[i];
    double *w = poidsnn[i];

    Fa[i] = 0.0;
    ztorque[i] = 0.0;
    double slope;
    double sigm;

    double slopeg;
    double sigmg;
    double sloped;
    double sigmd;
    double f_d, f_g;
    
    switch(controler_type) {
    case 0: // Controler du stage: Vitesse déja fixé: seul le seuil est à apprendre
      Fa[i] = I >= w[0] ? 0.0 : 1.0;
      break;

    case 1:
      Fa[i] = I >= w[0] ? 0.2 : 1.0;
      break;

    case 2:
      // test pour END to END 
      if (w[1]> w[2]){std::swap(w[1], w[2]); std::swap(w[0], w[3]);}
      if (w[1] == w[2]){Fa[i] = 0.5*(w[0]+w[3]);}

      if (I<= w[1]){
        Fa[i] = w[0];
      }
      else if (I >= w[2]) {
        Fa[i] = w[3];
      }
      else {
        Fa[i] = w[0] + (w[3]-w[0])*(I-w[1])/(w[2]-w[1]);
      }
      break;
    
    case 3:
      slope = f_slope(w[1]);
      sigm = sigmoid(slope*(I - w[2]));
      Fa[i] = w[0] + (w[3] - w[0])*sigm;
      break;

    case 4:
      slopeg = f_slope(w[1]);
      sigmg = sigmoid(slopeg*(I - w[2]));
      f_g = w[0] + (w[3] - w[0])*sigmg;
      sloped = f_slope(w[5]);
      sigmd = sigmoid(sloped*(I - w[6]));
      f_d =  w[4] + (w[7] - w[4])*sigmd;
      Fa[i] = th_trun((f_g+f_d)/2); //+ 0.1;
      ztorque[i] = 0.5*(f_d - f_g);
      break;

    case 5:
      slopeg = f_slope(w[1]);
      sigmg  = sigmoid(slopeg * (I - w[2]));
      f_g = w[0] + (w[3] - w[0]) * sigmg;

      sloped = f_slope(w[5]);
      sigmd  = sigmoid(sloped * (I - w[6]));
      f_d = w[4] + (w[7] - w[4]) * sigmd;

      Fa[i]      = 0.5 * (f_g + f_d);
      ztorque[i] = 0.5 * (f_d - f_g);
      break;

    case 6:
      Fa[i] = 0.5;
      break;

    case 7:
      Fa[i] = w[0];
      break;

    // --- Controller MLP (end-to-end) ---
    case 8:
      SimpleMultiLayerPerceptron(w, I, &f_d, &f_g);
      Fa[i]      =  th_trun(0.5 * (f_d + f_g));
      ztorque[i] = 0.5 * (f_d - f_g);
      break;
    }
  }
}

// Computes left and right motor forces based on a simple MLP with one hidden layer
void FixControler::SimpleMultiLayerPerceptron(double *w, double I, double* out_left, double* out_right) {
  int index = 0;
  double layer_1[hidden_size];
  double layer_2[hidden_size];
  double layer_out[2];
  for (int l = 0; l < hidden_size; l++) {
    double Wl1 = w[index++];   // weights
    double b1l = w[index++];   // biases
    double x1l = Wl1 * I + b1l;
    layer_1[l] = Leaky_Relu(x1l);
  }

  for (int l = 0; l < hidden_size; l++) {
    double x2l = 0.0;
    for (int c = 0; c < hidden_size; c++) {
      double W2lc = w[index++];
      x2l += W2lc * layer_1[c];
    }
    x2l += w[index++]; // bias
    layer_2[l] = Leaky_Relu(x2l);
  }

  for (int l = 0; l < 2; l++) {
    double x_out_l = 0.0;
    for (int c = 0; c < hidden_size; c++) {
      double W_out_lc = w[index++];
      x_out_l += W_out_lc * layer_2[c];
    }
    x_out_l += w[index++]; // bias

    if (l == 0) layer_out[l] = sigmoid(x_out_l);
    if (l == 1) layer_out[l] = sigmoid(x_out_l);
  }

  *out_left = layer_out[0];
  *out_right = layer_out[1];
}