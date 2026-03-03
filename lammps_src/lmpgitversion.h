#ifndef LMP_GIT_VERSION_H
#define LMP_GIT_VERSION_H
bool LAMMPS_NS::LAMMPS::has_git_info() { return true; }
const char *LAMMPS_NS::LAMMPS::git_commit() { return "a8f49153d13f89fe28cc411583a6d4343330b9a5"; }
const char *LAMMPS_NS::LAMMPS::git_branch() { return "develop"; }
const char *LAMMPS_NS::LAMMPS::git_descriptor() { return ""; }
#endif
