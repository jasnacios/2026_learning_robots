/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
------------------------------------------------------------------------- */

#include "fix_mlp.h"
#include "library.h"
#include <cmath>
#include "atom.h"
#include "update.h"
#include "error.h"
#include <span>
#include <sstream>

using namespace LAMMPS_NS;
using namespace FixConst;

inline double act_motorlike(double x)
{
  // Gompertz function approximating robot motor response for x in [-1, 1]
  double y = std::exp(-0.35*std::exp(-4*x));
  if (y < 0.01) return 0.0;
  return y;
}

inline double act_relu(double x)
{
  return x > 0.0 ? x : 0.0;
}

inline double act_sigmoid(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

inline double act_tanh(double x)
{
  return std::tanh(x);
}

inline double act_linear(double x)
{
  return x;
}

inline double act_heaviside(double x)
{
  return x >0 ? 1.0 : -1.0;
}

inline double apply_activation(const ActivationFunction& act, double x)
{
  if (act == RELU) return act_relu(x);
  if (act == SIGMOID) return act_sigmoid(x);
  if (act == TANH) return act_tanh(x);
  if (act == MOTORLIKE) return act_motorlike(x);
  if (act == HEAVISIDE) return act_heaviside(x);
  return act_linear(x);
}

const ActivationFunction FixMLP::get_activation_enum(const std::string &a)
{
  if (a == "relu") return RELU;
  if (a == "sigmoid") return SIGMOID;
  if (a == "tanh") return TANH;
  if (a == "linear") return LINEAR;
  if (a == "motorlike") return MOTORLIKE;
  if (a == "heaviside") return HEAVISIDE;
  error->all(FLERR,"Fix MLP: invalid activation function " + a);
  return LINEAR; // Unreachable, but silences compiler warning
}

/* ---------------------------------------------------------------------- */

FixMLP::FixMLP(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 9)
    error->all(FLERR,"Illegal fix mlp command");

  // Parse integers
  _nbLayers = utils::inumeric(FLERR,arg[3],false,lmp);
  _nbNeuronsPerLayer = utils::inumeric(FLERR,arg[4],false,lmp);
  _nevery = utils::inumeric(FLERR,arg[8],false,lmp);

  if (_nbLayers < 1)
    error->all(FLERR,"Fix MLP: nbLayers must be >= 1");

  if (_nbNeuronsPerLayer < 1)
    error->all(FLERR,"Fix MLP: nbNeuronsPerLayer must be >= 1");

  // Parse CSV vectors
  _inputs = _split_csv(arg[5], ';');
  _outputs = _split_csv(arg[6], ';');

  std::vector<std::string> str_activations = _split_csv(arg[7], ';');

  // Activation count check
  if (str_activations.size() != _nbLayers)
    error->all(FLERR,
      "Fix MLP: number of activation functions must match nbLayers");

  _activationFunctionsInternal.reserve(str_activations.size()-1);
  for (size_t i = 0; i < str_activations.size() - 1; i++) {
    const auto &a = str_activations[i];
    _activationFunctionsInternal.push_back(get_activation_enum(a));
  }

  // ------------------------------
  // Parse output activations
  // ------------------------------

  _activationFunctionsOutput.clear();
  std::string last = str_activations.back();
  bool bracketed = (!last.empty() && last.front() == '[' && last.back() == ']');

  if (bracketed) {
    // Remove brackets
    std::string inside = last.substr(1, last.size() - 2);
    std::vector<std::string> outActs = _split_csv(inside, ',');

    if (outActs.size() != _outputs.size())
      error->all(FLERR,
        "Fix MLP: number of output activation functions must match number of outputs");

    _activationFunctionsOutput.reserve(outActs.size());

    for (const auto &a : outActs) {
      _activationFunctionsOutput.push_back(get_activation_enum(a));
    }

  } else {
    ActivationFunction f = get_activation_enum(last);
    _activationFunctionsOutput.resize(_outputs.size(), f);
  }


  // Basic sanity checks
  if (_inputs.empty())
    error->all(FLERR,"Fix MLP: at least one input required");

  if (_outputs.empty())
    error->all(FLERR,"Fix MLP: at least one output required");

  _buildViews();

  CheckWeightsFit();
}

/* ---------------------------------------------------------------------- */

int FixMLP::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMLP::setup(int vflag)
{
  post_force(vflag);
} 

/* ---------------------------------------------------------------------- */

void FixMLP::post_force(int vflag)
{
  if (update->ntimestep % _nevery) return;
  double *clock = atom->clock;
  double **poidsnn = atom->poidsnn;
  int *mask = atom->mask;    
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;
    double t = clock[i];
    double *w = poidsnn[i];
    
    MultiLayerPerceptron(w, i);
  }
}

void FixMLP::_buildViews(){
  for (auto &name : _inputs) {
      int dtype = atom->extract_datatype(name.c_str());
      if (dtype == LAMMPS_DOUBLE) {
          double* arr = (double*) atom->extract(name.c_str());
          if (!arr) error->all(FLERR,"Fix MLP: unknown atom property " + name);
          _features.push_back({arr, 1});
      } else if (dtype == LAMMPS_DOUBLE_2D) {
          double** arr = (double**) atom->extract(name.c_str());
          if (!arr) error->all(FLERR,"Fix MLP: unknown atom property " + name);
          int cols = atom->extract_size(name.c_str(), LMP_SIZE_COLS);
          for (int c=0; c<cols; c++)
              _features.push_back({&arr[0][c], cols});
      }
      printf("input dtype : %d\n", dtype);
  }
  for (auto &name : _outputs) {
      int dtype = atom->extract_datatype(name.c_str());
      if (dtype == LAMMPS_DOUBLE) {
          double* arr = (double*) atom->extract(name.c_str());
          if (!arr) error->all(FLERR,"Fix MLP: unknown atom property " + name);
          _out.push_back({arr, 1});
      } else if (dtype == LAMMPS_DOUBLE_2D) {
          double** arr = (double**) atom->extract(name.c_str());
          if (!arr) error->all(FLERR,"Fix MLP: unknown atom property " + name);
          int cols = atom->extract_size(name.c_str(), LMP_SIZE_COLS);
          for (int c=0; c<cols; c++)
              _out.push_back({&arr[0][c], cols});
      }
      printf("output dtype : %d\n", dtype);
  }
}

void FixMLP::MultiLayerPerceptron(const double* w, const int atomIndex) const
{
  const int inputSize = _features.size();
  const int outputSize = _out.size();
  const int H = _nbNeuronsPerLayer;

  double hidden1[H];
  double hidden2[H];

  double* prev = nullptr;
  int prevSize = inputSize;

  double* curr = hidden1;

  int wpos = 0;

  for (unsigned int layer = 0; layer < _nbLayers; layer++) {

    int currSize = (layer == _nbLayers - 1) ? outputSize : H;

    if (layer % 2 == 0)
      curr = hidden1;
    else
      curr = hidden2;

    for (int i = 0; i < currSize; i++) {

      double sum = 0.0;

      if (layer == 0) {
        // read inputs directly using stride
        for (int j = 0; j < inputSize; j++) {
          double v = _features[j].ptr[atomIndex * _features[j].stride];
          sum += w[wpos++] * v;
        }
      }
      else {
        for (int j = 0; j < prevSize; j++) {
          sum += w[wpos++] * prev[j];
        }
      }

      sum += w[wpos++]; // bias
      
      if (layer == _nbLayers - 1)
        _out[i].ptr[atomIndex * _out[i].stride] = apply_activation(_activationFunctionsOutput[i], sum);
      else
        curr[i] = apply_activation(_activationFunctionsInternal[layer], sum);
    }

    prev = curr;
    prevSize = currSize;
  }
}

void FixMLP::CheckWeightsFit() const
{
  const int inputSize = _features.size();
  const int outputSize = _out.size();
  const int H = _nbNeuronsPerLayer;

  size_t required = 0;

  required += H * inputSize + H;

  for (unsigned int l = 1; l < _nbLayers - 1; l++)
    required += H * H + H;

  required += outputSize * H + outputSize;

  if (required > MAX_NEURONS) {
    error->all(FLERR,
      "Fix MLP: weight buffer too small. Required "
      + std::to_string(required) +
      " but MAX_NEURONS = " + std::to_string(MAX_NEURONS));
  }
  printf("Fix MLP: using %ld weights (MAX_NEURONS=%d)\n", required, MAX_NEURONS);
  printf("Input size : %d, output size : %d, hidden layers : %d, neurons/layer : %d\n",
    inputSize, outputSize, _nbLayers, _nbNeuronsPerLayer);
}

std::vector<std::string> FixMLP::_split_csv(const std::string &s, char delimiter) const
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delimiter)) {
    if (item.empty())
      error->all(FLERR,"Fix MLP: empty entry in CSV list");
    out.push_back(item);
  }

  return out;
}