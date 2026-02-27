import os
import random
import shutil
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- helpers ----------
def filtrer_fichier(input_filepath, natoms, parasites, output_filepath):
    lignesn = []
    with open(input_filepath, "r") as f:
        lignes = f.readlines()

    for i, line in enumerate(lignes):
        if i % (natoms + parasites) > parasites - 1:
            lignesn.append(line)

    dumpsteps = len(lignesn) // natoms

    with open(output_filepath, "w") as fa:
        fa.writelines(lignesn)

    return output_filepath, dumpsteps


def fichier_to_matrice(N, dumpsteps, input_filepath):
    datamat = np.loadtxt(input_filepath)
    numelements = datamat.shape[1]
    return datamat.reshape((dumpsteps, N, numelements))


def rwd_fast(data_reshaped, Nn):
    Qreward   = data_reshaped[:, :, 1]
    Weights   = data_reshaped[:, :, 2:2+Nn]
    Positions = data_reshaped[:, :, 2+Nn:2+Nn+2]
    Speeds    = data_reshaped[:, :, 2+Nn+2:2+Nn+4]
    return Qreward, Weights, Positions, Speeds


# ---------- one simulation job ----------
def run_one_sim(*, N, dump_freq, l, L, seed, dp, alpha_q, Dr, time, parasites, Nn, ncore,
                input_file, lmp_mpi_path, out_dir, keep_raw_dump=True):
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, f"log_seed_{seed}_{dp}_{alpha_q}_{Dr}.lmp")

    dump_name = f"dumpnn_{dp}_{alpha_q}_{Dr}_{seed}.txt"
    filtered_name = f"filtered_{dp}_{alpha_q}_{Dr}_{seed}.txt"

    cmd = [
        "mpirun", "-np", str(ncore),
        lmp_mpi_path,
        "-in", input_file,
        "-var", "Dr", str(Dr),
        "-var", "Seed", str(seed),
        "-var", "Dp", str(dp),
        "-var", "alphaq", str(alpha_q),
        "-var", "N", str(N),
        "-var", "dump_freq", str(dump_freq),
        "-var", "L", str(L),
        "-var", "l", str(l),
        "-var", "runtime_Main", str(time),
    ]

    # run
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)

    if proc.returncode != 0:
        raise RuntimeError(f"LAMMPS failed (N={N}, seed={seed}). See {log_file}")

    # filter dump
    _, dumpsteps = filtrer_fichier(dump_name, N, parasites, filtered_name)

    # archive or delete raw dump
    if keep_raw_dump:
        shutil.move(dump_name, os.path.join(out_dir, dump_name))
    else:
        os.remove(dump_name)

    # parse + save npy
    simu = fichier_to_matrice(N, dumpsteps, filtered_name)
    score, Poids, Pos, Speeds = rwd_fast(simu, Nn)
    os.remove(filtered_name)

    np.save(os.path.join(out_dir, f"Weights_{dp}_{alpha_q}_{Dr}_{seed}.npy"), Poids)
    np.save(os.path.join(out_dir, f"Rewards_{dp}_{alpha_q}_{Dr}_{seed}.npy"), score)
    np.save(os.path.join(out_dir, f"Positions_{dp}_{alpha_q}_{Dr}_{seed}.npy"), Pos)
    np.save(os.path.join(out_dir, f"Speeds_{dp}_{alpha_q}_{Dr}_{seed}.npy"), Speeds)

    return seed


# ---------- main ----------
if __name__ == "__main__":
    # === PARAMÈTRES ===
    parasites = 9
    time = 18_000_000
    Nn = 8
    ncore = 6
    input_file = "/home/johann1/Desktop/simus_avec_tolerance/Scaling_N/evo.in"
    lmp_mpi_path = "/home/johann1/Desktop/2026_learning_robots/lammps_src/lmp_mpi"

    natoms   = [ 300, 3000, 10000]# [1000, 100, 300, 3000, 10000]
    numrep   = [30,  5,    1] #10,   
    dumpfreq = [5000, 5000,5000,5000,5000]
    Dr = 0.1

    Rarene = [29.9, 94.5, 172.5] #54.5,
    rlum   = [8.1,  25.7, 46.9]#14.8, 

    dp = 1e-05
    alpha_q = 0.03

    base_path = "/media/johann1/DATA21/johann/simus_avec_tolerance/Scaling_N_controler_th_arret_sans_int"

    TOTAL_CORES = 24

    for idx in range(len(natoms)):
        ncore = 6
        max_workers = TOTAL_CORES // ncore
        N = natoms[idx]
        dump_freq = dumpfreq[idx]
        l = rlum[idx]
        L = Rarene[idx]
                  
        out_dir = os.path.join(base_path, str(N))
        os.makedirs(out_dir, exist_ok=True)

          
        if N == 3000:
            max_workers = 2
            ncore = 10
        if N >=10000:
            max_workers = 1
            ncore = 12

        # seeds uniques
        seeds = random.sample(range(10000, 99999), k=numrep[idx])
        np.save(os.path.join(out_dir, f"Seed_{N}.npy"), np.array(seeds, dtype=int))

        print(f"\n=== N={N} : {len(seeds)} runs | ncore={ncore} | parallel={max_workers} ===")

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for rep, seed in enumerate(seeds, start=1):
                print(f"  submit {rep}/{len(seeds)} seed={seed}")
                futures.append(ex.submit(
                    run_one_sim,
                    N=N, dump_freq=dump_freq, l=l, L=L, seed=seed,
                    dp=dp, alpha_q=alpha_q, Dr=Dr, time=time,
                    parasites=parasites, Nn=Nn, ncore=ncore,
                    input_file=input_file, lmp_mpi_path=lmp_mpi_path,
                    out_dir=out_dir,
                    keep_raw_dump=True
                ))

            for fut in as_completed(futures):
                seed_done = fut.result()
                print(f"  ✅ finished seed={seed_done} (N={N})")