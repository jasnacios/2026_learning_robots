#ifndef LMP_GIT_VERSION_H
#define LMP_GIT_VERSION_H
bool LAMMPS_NS::LAMMPS::has_git_info() { return true; }
const char *LAMMPS_NS::LAMMPS::git_commit() { return "e06ab4ecede6591e395043509de9ec52f60f1e28"; }
const char *LAMMPS_NS::LAMMPS::git_branch() { return "develop"; }
const char *LAMMPS_NS::LAMMPS::git_descriptor() { return ""; }
#endif
