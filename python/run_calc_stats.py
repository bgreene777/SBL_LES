# --------------------------------
# Name: run_calc_stats.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 6 May 2021
# Purpose: Monitor status of simulation and run calc_stats.f90 when finished
# can also rsync spinup directory and run interp3d on vel_sc.out
# --------------------------------
import numpy as np
import os
from time import sleep
from datetime import datetime

fdir = "/home/bgreene/simulations/F_128_interp/"
fslurm = f"{fdir}slurm-2100.out"
finished = False

while not finished:
    # print timestamp
    print(f"Attempting at {datetime.utcnow()} UTC")
    # read slurm file
    with open(fslurm) as f:
        txt = f.read()
    # check for completion
    if "END" in txt:
        print("Simulation finished! Continue to run calc_stats")
        finished=True
        break
    else:
        print("Simulation still running; wait 1 hour")
        sleep(3600)

# # rsync spinup directory into new one
# os.system("rsync -avr /home/bgreene/simulations/F_spinup /home/bgreene/simulations/F_128_interp/ --exclude-from=/home/bgreene/simulations/F_spinup/output/*")
# # move everything up one level 
# os.system("mv /home/bgreene/simulations/F_128_interp/F_spinup/* /home/bgreene/simulations/F_128_interp/")
# # create the new fields_3d directory
# os.system("mkdir /home/bgreene/simulations/F_128_interp/output/fields_3d/")
# # rename vel_sc.out
# os.system("mv /home/bgreene/simulations/F_128_interp/vel_sc.out /home/bgreene/simulations/F_128_interp/vel_sc_spinup.out")
# # run the interp3d script
# os.system("/home/bgreene/fortran/interp3d")

os.system("/home/bgreene/fortran/analysis/calc_stats")
print(f"Finished! Time: {datetime.utcnow()} UTC")