#!/bin/bash

mkdir -p build/
cp -a ./src/* ./lammps_src/
cd lammps_src
make -j8 serial
make -j8 mpi
cp lmp_serial ../build/lmp_serial
cp lmp_mpi ../build/lmp_mpi
cd -
