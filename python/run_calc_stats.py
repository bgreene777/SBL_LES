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
from datetime import datetime
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument()

# --------------------------------
# Define functions for processing
# --------------------------------
def check_slurm_finished(fdir, fslurm):
    # Read fdir+fslurm to check for completion
    # Returns boolean
    print(f"Reading {fdir}{fslurm} to determine if finished...")
    with open(f"{fdir}{fslurm}") as f:
        txt = f.read()
    # check for completion
    if "END" in txt:
        print(f"Simulation finished! Time: {datetime.utcnow()}")
        return True
    else:
        print(f"Simulation still running, try later. Time: {datetime.utcnow()}")
        return False
# --------------------------------    
def check_already_processed(fdir):
    # check for presence of calc_stats.txt in fdir, signifying 
    # that calc_stats() has already been run
    # Returns boolean
    if os.path.exists(f"{fdir}calc_stats.txt"):
        return True
    elif os.path.exists(f"{fdir}interp3d.txt"):
        return True
    else:
        return False
# --------------------------------
def calc_stats(fdir,exe):
    # run the pre-compiled fortran executable
    os.system(f"/home/bgreene/SBL_LES/fortran/executables/{exe}")
    
    # spit out a calc_stats.txt file so this isn't repeated
    np.savetxt(f"{fdir}calc_stats.txt", fmt="%s",
               [f"Calc stats complete! Time: {datetime.utcnow()}"])
    
    return
# --------------------------------
def run_interp3d(fdir,exe,dimsize):
    # create new interp directory based on fdir (which points to spinup directory)
    simdir, spinup, blah = fdir.rsplit("/", 2)
    fnew = os.path.join(simdir, f"{spinup[0]}_{dimsize}_interp") + os.sep
    print(f"Creating new directory: {fnew}")
   
    # rsync spinup directory into new one
    os.system(f"rsync -avr {fdir} {fnew} --exclude-from={fdir}output/*")
    
    # move everything up one level 
    os.system(f"mv {fnew}{spinup[0]}_spinup/* {fnew}")
    
    # create the new fields_3d directory
    os.system(f"mkdir {fnew}output/fields_3d/")
    
    # rename vel_sc.out
    os.system(f"mv {fnew}vel_sc.out {fnew}vel_sc_spinup.out")

    # run the interp3d script
    os.system(f"/home/bgreene/SBL_LES/fortran/executables/{exe}")
    
    # spit out interp3d.txt file so this isn't repreated
    np.savetxt(f"{fdir}interp3d.txt", fmt="%s",
           [f"interp3d complete! Time: {datetime.utcnow()}"])
    
    print(f"Finished run_interp3d! Time: {datetime.utcnow()}")
    
    return
# --------------------------------
def main(dsim):
    # before anything else, check if sim finished
    sim_finished = check_slurm_finished(dsim[0], dsim[1])
    # also check to see if calc_stats or interp3d have been run already
    task_complete = check_already_processed(dsim[0])
    if not sim_finished:
        # not done -- print time and quit
        print("run_calc_stats.py finished.")
        return
    elif (sim_finished & task_complete):
        # already been performed -- print time and quit
        print("Simulation finished and task already complete!")
        print("run_calc_stats.py finished.")
        return
    elif (sim_finished & (not task_complete)):
        # task not complete -- decide which to do
        
        return
    
    else:
        print("idk how you got here")
        


# --------------------------------
# Load simulations.txt to determine tasks
# --------------------------------

fsim = "/home/bgreene/SBL_LES/python/simulations.txt"
dsim = np.genfromtxt(fsim, delimiter=",", dtype=str, skip_headers=1)




print(f"Finished! Time: {datetime.utcnow()} UTC")