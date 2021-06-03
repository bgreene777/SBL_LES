# --------------------------------
# Name: UAS.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 03 June 2021
# Purpose: load npz files from timeseries output to emulate UAS profiles
# and plot output
# --------------------------------
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from simulation import *

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/UAS/"
plt.close("all")

#
# Create simulation objects
#
# A
sA = UAS_emulator("/home/bgreene/simulations/A_192_interp/output/",
                192, 192, 192, 800., 800., 400., "A")
# F
sF = UAS_emulator("/home/bgreene/simulations/F_192_interp/output/",
                192, 192, 192, 800., 800., 400., "F")

# combine in list for looping
s_all = [sA, sF]
for s in s_all:
    s.read_timeseries(720000, 0.02, raw=False)
    s.profile(ascent_rate=1.0, time_constant=0.0)
    
# plot F uas versus mean
fig1, ax1 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(12, 8))
# u
ax1[0].plot(sF.xytavg["u"], sF.z, "-k", label="$\\langle u \\rangle$")
ax1[0].plot(sF.prof["u"], sF.z, "-r", label="UAS")
# v
ax1[1].plot(sF.xytavg["v"], sF.z, "-k", label="$\\langle v \\rangle$")
ax1[1].plot(sF.prof["v"], sF.z, "-r", label="UAS")
# w
ax1[2].plot(sF.xytavg["w"], sF.z, "-k", label="$\\langle w \\rangle$")
ax1[2].plot(sF.prof["w"], sF.z, "-r", label="UAS")
# theta
ax1[3].plot(sF.xytavg["theta"], sF.z, "-k", label="$\\langle \\theta \\rangle$")
ax1[3].plot(sF.prof["theta"], sF.z, "-r", label="UAS")

# clean up
for iax in ax1:
    iax.grid()
    iax.legend(loc="upper left")
ax1[0].set_xlabel("$u$ [m/s]")
ax1[0].set_ylabel("$z$ [m]")
ax1[1].set_xlabel("$v$ [m/s]")
ax1[2].set_xlabel("$w$ [m/s]")
ax1[3].set_xlabel("$\\theta$ [K]")
# save figure
fsave1 = f"{fdir_save}uvwtheta.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
plt.close(fig1)