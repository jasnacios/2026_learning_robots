#ifndef LMP_GIT_VERSION_H
#define LMP_GIT_VERSION_H
bool LAMMPS_NS::LAMMPS::has_git_info() { return true; }
const char *LAMMPS_NS::LAMMPS::git_commit() { return "aba619361d26c1785ace2c86827f275c769118b1"; }
const char *LAMMPS_NS::LAMMPS::git_branch() { return "develop"; }
const char *LAMMPS_NS::LAMMPS::git_descriptor() { return ""; }
#endif
