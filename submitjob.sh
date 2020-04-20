#!/bin/bash
#SBATCH -N 40
###SBATCH -q premium
#SBATCH -q premium
#SBATCH -J OUT_2ptfull.log
#SBATCH -o ERR_2pfull_reg.log
#SBATCH --mail-user=amon2018@stanford.edu
#SBATCH --mail-type=ALL
###SBATCH -t 08:00:00
#SBATCH -t 08:00:00
#SBATCH -C haswell  
#SBATCH -A des
#SBATCH --qos premium
#SBATCH --image=docker:jderose/2pt_pipeline
###SBATCH--volume="/global/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.8/sampleselection:/cat"
#SBATCH--volume="/global/project/projectdirs/des/www/y3_cats/:/project"

#SBATCH --cpus-per-task=1

#OpenMP settings:
#export OMP_NUM_THREADS=64

#export OMP_PROC_BIND=close
export HDF5_USE_FILE_LOCKING=FALSE 
export OMP_NUM_THREADS=64
#run the application:
srun -u -N 40 -n 160 shifter python3 -s -m pipeline --mpi --stage 2pt mcalY3_pixellized_full.yaml
###BuzzardY3_pixelized_true_zs_true_zl_weighted.yaml

