#!/bin/bash -l
#SBATCH --job-name=stationarity_v2.py
#SBATCH --partition=CLUSTER
#SBATCH -N 1
#SBATCH -c 56
#SBATCH -o out.out
#SBATCH -e err.log
#SBATCH --exclusive
#SBATCH --time=504:00:00
#SBATCH --mail-user=brian.greene@ou.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

#Set stack size, etc.
ulimit -s unlimited
ulimit -c unlimited

echo ' ****** START OF JOB ****** '
date
srun  /home/bgreene/anaconda3/bin/python -u /home/bgreene/SBL_LES/python/stationarity_v2.py
date
echo ' ****** END OF JOB ****** '