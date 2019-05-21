#!/bin/bash
#SBATCH -N 50
#SBATCH -q debug
#SBATCH -J OUT_2ptfull.log
#SBATCH -o ERR_2pfull.log
#SBATCH --mail-user=amon2018@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -C haswell  
#SBATCH -A des

#SBATCH --cpus-per-task=1

#OpenMP settings:
#export OMP_NUM_THREADS=64
#export OMP_PLACES=threads
#export OMP_PROC_BIND=close

source /global/common/software/des/zuntz/setup-nompi
 
#PYTHONPATH=$PYTHONPATH:/global/homes/s/seccolf/des-science/destest
export OMP_NUM_THREADS=64
#run the application:
srun -u -N 50 -n 160 python -s -m pipeline --mpi --stage 2pt BuzzardY3_pixelized_pp.yaml


