#!/usr/bin/env python3
# --------------------------------
# Name: u_v_w_timeseries.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 7 September 2021
# Purpose: plot timeseries of xy planar averaged profiles of
# u, v, and w
# Update 8 September 2021: switch to use raw .out files and 
# also load from spinup to see full simulation u, v, and wdir
# --------------------------------
import os
import cmocean
import seaborn
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from simulation import read_f90_bin

# --------------------------------
# Settings and Configuration
# --------------------------------
fdir = "/home/bgreene/simulations/A_192_interp/output/netcdf/"
timesteps = np.arange(900000, 1260000+1, 1000, dtype=np.int32)
# determine files to read from timesteps
fall = [f"{fdir}all_{tt:07d}.nc" for tt in timesteps]
nf = len(fall)
# calculate array of times represented by each file
times = np.array([i*0.02*1000 for i in range(nf)])

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')
colors = seaborn.color_palette("colorblind")
# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/stationarity/"
plt.close("all")

# --------------------------------
# Load files and clean up
# --------------------------------
print("Reading files...")
dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
dd.coords["time"] = times
dd.time.attrs["units"] = "s"
dd["time_hr"] = dd.time/3600.
dd.time_hr.attrs["units"] = "hr"
# calculate xyt averages
dd["u_avg"] = dd.u.mean(dim=("x", "y"))
dd["v_avg"] = dd.v.mean(dim=("x", "y"))
dd["w_avg"] = dd.w.mean(dim=("x", "y"))

# also load 1hr stats file to find h
dstat = xr.open_dataset(f"{fdir}average_statistics.nc")
# calculate ustar and h
dstat["ustar"] = ((dstat.uw_cov_tot**2.) + (dstat.vw_cov_tot**2.)) ** 0.25
dstat["ustar2"] = dstat.ustar ** 2.
dstat["h"] = dstat.z.where(dstat.ustar2 <= 0.05*dstat.ustar2[0], 
                           drop=True)[0] / 0.95

# --------------------------------
# Plot
# --------------------------------
#
# Figure 1: lowest jz**2 levels
#
fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
# loop and plot various heights
for i, jz in enumerate(np.arange(6, dtype=np.int64)**2):
    # u
    ax1[0].plot(dd.time_hr, dd.u_avg.isel(z=jz), ls="-", c=colors[i])
    # v
    ax1[1].plot(dd.time_hr, dd.v_avg.isel(z=jz), ls="--", c=colors[i])
    # w
    ax1[2].plot(dd.time_hr, dd.w_avg.isel(z=jz), ls="-", c=colors[i])
    
ax1[0].grid()
ax1[0].set_ylabel("$u$ [m s$^{-1}$]")

ax1[1].grid()
ax1[1].set_ylabel("$v$ [m s$^{-1}$]")

ax1[2].grid()
ax1[2].set_xlabel("Time [hrs]")
ax1[2].set_ylabel("$w$ [m s$^{-1}$]", fontsize=16)
ax1[2].legend()

fig1.suptitle("Simulation A velocity Timeseries - Last 2 Hours")
fig1.tight_layout()
# save and close
fsave1 = f"{fdir_save}A192_u_v_w_timeseries.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf")
plt.close(fig1)

#
# Figure 2: 6 levels near h
#
fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
# loop and plot various heights
for i, jz in enumerate(np.arange(72, 81, 2, dtype=np.int64)):
    # u
    ax2[0].plot(dd.time_hr, dd.u_avg.isel(z=jz), ls="-", c=colors[i])
    # v
    ax2[1].plot(dd.time_hr, dd.v_avg.isel(z=jz), ls="--", c=colors[i])
    # w
    ax2[2].plot(dd.time_hr, dd.w_avg.isel(z=jz), ls="-", c=colors[i])
    
ax2[0].grid()
ax2[0].set_ylabel("$u$ [m s$^{-1}$]")

ax2[1].grid()
ax2[1].set_ylabel("$v$ [m s$^{-1}$]")

ax2[2].grid()
ax2[2].set_xlabel("Time [hrs]")
ax2[2].set_ylabel("$w$ [m s$^{-1}$]", fontsize=16)
ax2[2].legend()

fig2.suptitle("Simulation A velocity Timeseries - Last 2 Hours")
fig2.tight_layout()
# save and close
fsave2 = f"{fdir_save}A192_u_v_w_h_timeseries.pdf"
print(f"Saving figure: {fsave2}")
fig2.savefig(fsave2, format="pdf")
plt.close(fig2)