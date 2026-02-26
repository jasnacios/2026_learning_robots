#ifndef LMP_GIT_VERSION_H
#define LMP_GIT_VERSION_H
bool LAMMPS_NS::LAMMPS::has_git_info() { return true; }
const char *LAMMPS_NS::LAMMPS::git_commit() { return "HEAD"; }
const char *LAMMPS_NS::LAMMPS::git_branch() { return "HEAD"; }
const char *LAMMPS_NS::LAMMPS::git_descriptor() { return ""; }
#endif
