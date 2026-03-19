

###########################################
# LANCER UN RUN AVEC PYTHON ---------------
###########################################

import subprocess
import os

# chemin vers l'exécutable LAMMPS
lammps_exec = "/home/johann1/Desktop/2026_learning_robots/lammps_src/lmp_mpi"

# fichier input LAMMPS
input_file = "test_learning_phototaxis.in"

# nombre de coeurs MPI
nproc = 12

cmd = [
    "mpirun",
    "-np", str(nproc),
    lammps_exec,
    "-in", input_file
]

try:
    subprocess.run(cmd, check=True)
    print("Simulation terminée avec succès")

except subprocess.CalledProcessError as e:
    print("Erreur lors de l'exécution :", e)


"""
###############################################
# LANCER UN SWEEP DE RUNS EN SERIE AVEC PYTHON 
###############################################
import subprocess

lammps_exec = "/home/johann1/Desktop/2026_learning_robots/lammps_src/lmp_mpi"
input_file = "test_learning_phototaxis.in"

alpha_values = [1,10,100]

for alpha in alpha_values:

    cmd = [
        "mpirun", "-np", "8",
        lammps_exec,
        "-in", input_file,
        "-var", "alpha", str(alpha)
    ]

    print("Running alpha =", alpha)

    subprocess.run(cmd)

###########################################
# LANCER SWEEP DE RUN PTYHON EN PARALLELE -
###########################################

from concurrent.futures import ProcessPoolExecutor
import subprocess

def run_sim(alpha):

    cmd = [
        "mpirun", "-np", "8",
        "/home/johann1/Desktop/2026_learning_robots/lammps_src/lmp_mpi",
        "-in", "test_learning_phototaxis.in",
        "-var", "alpha", str(alpha)
    ]

    subprocess.run(cmd)

alphas = [0.1,0.2,0.3,0.4]

with ProcessPoolExecutor(max_workers=3) as executor:
    executor.map(run_sim, alphas)"""