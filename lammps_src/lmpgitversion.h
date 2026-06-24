#ifndef LMP_GIT_VERSION_H
#define LMP_GIT_VERSION_H
bool LAMMPS_NS::LAMMPS::has_git_info() { return true; }
const char *LAMMPS_NS::LAMMPS::git_commit() { return "b8ddf05c9c95e615b08dc26661a391ec3950456d"; }
const char *LAMMPS_NS::LAMMPS::git_branch() { return "develop"; }
const char *LAMMPS_NS::LAMMPS::git_descriptor() { return ""; }
#endif
