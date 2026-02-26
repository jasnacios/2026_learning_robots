#!/bin/bash

mkdir -p build/
cd lammps_src
make serial
make mpi
cp lmp_serial ../build/lmp_serial
cp lmp_mpi ../build/lmp_mpi
cd ..
