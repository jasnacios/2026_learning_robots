/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_learner.h"

#include "atom.h"
#include "domain.h"
#include "memory.h"

#include <cmath>
 
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecLearner::AtomVecLearner(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  atom->poidsnn_flag = 1;
  atom->lightintensity_flag = 1;
  atom->qreward_flag = 1;
  atom->dpoids_flag = 1;
  atom->dreward_flag = 1;
  
  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file
  
  fields_create = { "poidsnn","dpoids", "qreward", "dreward", "lightintensity"};
  fields_exchange = { "poidsnn", "dpoids", "qreward", "dreward", "lightintensity"};
  fields_grow = { "poidsnn", "dpoids", "qreward", "dreward", "lightintensity"};
  
  fields_copy = { "poidsnn", "dpoids", "qreward", "dreward", "lightintensity"};
  fields_comm = { "poidsnn", "qreward"};
  // fields_reverse = { "poidsnn", "qreward", "lightintensity"};


  fields_border = { "poidsnn", "qreward", "lightintensity"};
  fields_restart = { "poidsnn", "dpoids", "qreward", "dreward", "lightintensity"};
 
 fields_data_atom = {"id", "type", "x", "qreward", "poidsnn"};
 fields_data_vel = {"id", "v"};

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecLearner::grow_pointers()
{
  poidsnn = atom->poidsnn;
  dpoids = atom->dpoids;
  qreward = atom->qreward;
  dreward = atom->dreward;
  lightintensity = atom->lightintensity;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecLearner::data_atom_post(int ilocal)
{
    
}

