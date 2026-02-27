import subprocess
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil



def filtrer_fichier(input_filepath, natoms, parasites, output_filepath):
    lignesn = []
    with open(input_filepath, 'r') as f:
        lignes = f.readlines()

    for i in range(len(lignes)):
        if i % (natoms + parasites) > parasites - 1:
            lignesn.append(lignes[i])
    dumpsteps = len(lignesn) // natoms

    with open(output_filepath, "w") as fa:
        fa.writelines(lignesn)

    return output_filepath, dumpsteps



def fichier_to_matrice(N, dumpsteps, input_filepath):
    datamat = np.loadtxt(input_filepath)
    numelements = datamat.shape[1]
    data_reshaped = datamat.reshape((dumpsteps, N, numelements)) 
    return data_reshaped

def rwd(data_reshaped, dumpsteps,N):
    Weights = np.zeros((dumpsteps, N, Nn))
    Qreward = np.zeros((dumpsteps, N))
    Positions = np.zeros((dumpsteps, N, 2))
    Speeds = np.zeros((dumpsteps, N, 2))
    for t in range(dumpsteps):
        for i in range(N):
            c = 1
            Qreward[t,i] = data_reshaped[t,i,c]
            c +=1 
            for k in range(Nn):
                Weights[t, i, k] = data_reshaped[t, i, c]  
                c +=1 
            Positions[t,i,0] = data_reshaped[t,i,c]
            c +=1
            Positions[t,i,1] = data_reshaped[t,i,c]
            c+=1
            Speeds[t,i,0] = data_reshaped[t,i,c]
            c +=1
            Speeds[t,i,1] = data_reshaped[t,i,c]


    return Qreward, Weights, Positions, Speeds
def rwd_fast(data_reshaped, Nn):
    Qreward  = data_reshaped[:, :, 1]
    Weights  = data_reshaped[:, :, 2:2+Nn]
    Positions = data_reshaped[:, :, 2+Nn:2+Nn+2]
    Speeds    = data_reshaped[:, :, 2+Nn+2:2+Nn+4]
    return Qreward, Weights, Positions, Speeds
# === PARAMÈTRES ===
parasites = 9                                                           # Lignes parasites par timestep
time = 18000000
Nn = 8
ncore = 12
input_file = "evo.in"

phi_robots = 0.066
phi_lum = 0.058

natoms = [ 1000,100,300,3000,10000]                              # nombres d'atomes
numrep = [10,50,30, 5,1]                                       # nombre de répétitions des simus100
dumpfreq = [5000,5000,5000,5000,5000]                           # fréquence de dump  
Dr = 0.1

Rarene = [54.5,17.25,29.9,94.5,172.5]                                 # demi coté de l'arène
rlum = [14.8,4.7,8.1, 25.7,46.9]                             # rayon de la zone lumineuse



dp = 1e-05
alpha_q = 0.03



for n in range(len(natoms)):
    N = natoms[n]
    dump_freq = dumpfreq[n]
    l = rlum[n]
    L = Rarene[n]
    Seed = np.zeros((numrep[n]))
    path = f"/media/johann1/DATA21/johann/simus_avec_tolerance/Scaling_N_controler_th_sans_int/{N}"
    for rep in range(numrep[n]):
        seed = random.randint(10000, 99999)
        Seed[rep] = seed
        log_file = f"log_seed_{seed}_{dp}_{alpha_q}_{Dr}.lmp"

        print(f"Lancement de la simulation avec N = {N}, seed = {seed}, Dp = {dp}, alphaq = {alpha_q}, Dr = {Dr}, {rep+1}/{numrep[n]}")

        # Exécute LAMMPS avec la seed variable
        subprocess.run(["mpirun", "--oversubscribe","-np",str(ncore),
            "/home/johann1/Desktop/lammps/lammps_src/lmp_mpi",
            "-in", input_file,
            "-var", "Dr", str(Dr),
            "-var", "Seed", str(seed),
            "-var", "Dp", str(dp),
            "-var", "alphaq", str(alpha_q),
            "-var", "N", str(N),
            "-var", "dump_freq", str(dump_freq),
            "-var", "L", str(L),
            "-var", "l", str(l),
            "-var", "runtime_Main", str(time)

        ], stdout=open(log_file, "w"))

        
        
        # enlever les lignes parasites
        monfichier, dumpsteps = filtrer_fichier(f'dumpnn_{dp}_{alpha_q}_{Dr}_{seed}.txt', N, parasites, f'filtered_{dp}_{alpha_q}_{Dr}_{seed}.txt')
        # fichier brut vers disque pour visualiser ovito
        shutil.move(f'dumpnn_{dp}_{alpha_q}_{Dr}_{seed}.txt', f'{path}/dumpnn_{dp}_{alpha_q}_{Dr}_{seed}.txt')

        
        simu = fichier_to_matrice(N, dumpsteps, f'filtered_{dp}_{alpha_q}_{Dr}_{seed}.txt')
        score, Poids, Pos, Speeds = rwd_fast(simu, Nn)
        os.remove(f'filtered_{dp}_{alpha_q}_{Dr}_{seed}.txt')




        np.save(f'Weights_{dp}_{alpha_q}_{Dr}_{seed}.npy', Poids)
        shutil.move(f'Weights_{dp}_{alpha_q}_{Dr}_{seed}.npy', f'{path}/Weights_{dp}_{alpha_q}_{Dr}_{seed}.npy')
        np.save(f'Rewards_{dp}_{alpha_q}_{Dr}_{seed}.npy', score)
        shutil.move(f'Rewards_{dp}_{alpha_q}_{Dr}_{seed}.npy', f'{path}/Rewards_{dp}_{alpha_q}_{Dr}_{seed}.npy')
        np.save(f'Positions_{dp}_{alpha_q}_{Dr}_{seed}.npy', Pos)
        shutil.move(f'Positions_{dp}_{alpha_q}_{Dr}_{seed}.npy', f'{path}/Positions_{dp}_{alpha_q}_{Dr}_{seed}.npy')
        np.save(f'Speeds_{dp}_{alpha_q}_{Dr}_{seed}.npy', Speeds)
        shutil.move(f'Speeds_{dp}_{alpha_q}_{Dr}_{seed}.npy', f'{path}/Speeds_{dp}_{alpha_q}_{Dr}_{seed}.npy')
        os.remove(log_file)

    np.save(f'{path}/Seed_{N}.npy',Seed)