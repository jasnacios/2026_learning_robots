#!/bin/bash

mkdir -p build/
cp -r ./src/* ./lammps_src/
cd lammps_src
make serial
make mpi
cp lmp_serial ../build/lmp_serial
cp lmp_mpi ../build/lmp_mpi
cd ..