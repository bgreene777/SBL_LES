# --------------------------------
# Name: plot_spectra.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 21 May 2021
# Purpose: plot spectra from given npz file output by spectra.py
# --------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from simulation import simulation

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/spectra/"
plt.close("all")

# useful plotting stuff
fstr = ["-k", "--k", ":k", ".-k", "-r", "--r", ":r", ".-r", "-b", "--b", ":b", ".-b"]
colors = [(225./255, 156./255, 131./255),
          (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
          (0., 85./255, 80./255), (20./255, 33./255, 61./255), (252./255, 193./255, 219./255)]

#
# Create simulation objects
#
s_all = []
for stab in ["A", "F"]:
    for res in ["128", "192", "160"]:
        if (stab=="F") & (res=="160"):
            break
        s = simulation(f"/home/bgreene/simulations/{stab}_{res}_interp/output/average_statistics.csv",
                      int(res), int(res), int(res), 800., 800., 400., stab)
        s.read_csv()
        s.read_spectra(f"/home/bgreene/SBL_LES/output/spectra_{s.stab}_{s.lab}.npz")
        s_all.append(s)

# --------------------------------
# Begin plotting
# --------------------------------

#
# Figure 1: E_uu for select simulation
# loop over all s in s_all and create individual figure
#
for s in s_all:
    fig1, ax1 = plt.subplots(1, figsize=(12, 8))
    imax = int(np.sqrt(s.i_h))
    xmid = s.nx//2
    # loop over heights in s, up to maximum of imax
    for i, jz in enumerate(np.arange(imax, dtype=int)**2):
        ax1.plot(s.spec["freqz"][1:xmid]*s.z[jz], 
                 s.spec["E_uu"][1:xmid,jz]/s.cov["ustar"][jz]/s.cov["ustar"][jz]/s.z[jz], 
                 fstr[i], label=f"$z={{{s.z[jz]:4.1f}}}$m")
    # clean up figure
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid()
    ax1.set_xlabel("$k_x z$")
    ax1.set_ylabel("$E_{11}(z)u_{*}^{-2}z^{-1}$")
    ax1.set_title(f"{s.stab}{s.lab} Streamwise Velocity Energy Spectra")
    ax1.legend()
    # save and close
    fsave1 = f"{fdir_save}{s.stab}{s.lab}_streamwise_u.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
    plt.close(fig1)