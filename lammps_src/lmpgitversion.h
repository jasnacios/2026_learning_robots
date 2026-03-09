#ifndef LMP_GIT_VERSION_H
#define LMP_GIT_VERSION_H
bool LAMMPS_NS::LAMMPS::has_git_info() { return true; }
const char *LAMMPS_NS::LAMMPS::git_commit() { return "eee4ac3129a23baf464696f8368ca6ae4f1b8efc"; }
const char *LAMMPS_NS::LAMMPS::git_branch() { return "develop"; }
const char *LAMMPS_NS::LAMMPS::git_descriptor() { return ""; }
#endif
