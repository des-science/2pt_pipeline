#!/bin/bash
#SBATCH -N 50
#SBATCH -q regular
##SBATCH -q debug
#SBATCH -J OUT_2ptfull.log
#SBATCH -o ERR_2pfull.log
#SBATCH --mail-user=jderose@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00
##SBATCH -t 00:30:00
#SBATCH -C haswell  
#SBATCH -A des
#SBATCH --qos premium
#SBATCH --image=docker:jderose/2pt_pipeline
#SBATCH--volume="/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3:/output"

#SBATCH --cpus-per-task=1

#OpenMP settings:
#export OMP_NUM_THREADS=64

#export OMP_PROC_BIND=close


export OMP_NUM_THREADS=64
#run the application:
srun -u -N 50 -n 50 shifter python3 -s -m pipeline --mpi --stage 2pt BuzzardY3_pixelized_sompz_bin_true_zs_true_zl.yaml


