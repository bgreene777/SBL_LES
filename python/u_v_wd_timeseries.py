#!/usr/bin/env python3
# --------------------------------
# Name: u_v_wd_timeseries.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 7 September 2021
# Purpose: plot timeseries of xy planar averaged profiles of
# u, v, and w
# Update 8 September 2021: switch to use raw .out files and 
# also load from spinup to see full simulation u, v, and wdir
# --------------------------------
import os
import yaml
import cmocean
import seaborn
import numpy as np
import xarray as xr
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from simulation import read_f90_bin

# --------------------------------
# Settings and Configuration
# --------------------------------
# load yaml settings file (will be in same dir)
fyaml = "timeseries.yaml"
with open(fyaml) as f:
    config = yaml.safe_load(f)
    
# grab directories and resolutions
dspin = config["dspin"]
rspin = config["rspin"]
dint = config["dint"]
rint = config["rint"]

# glob files -- just need u and v
fspin_u = np.sort(glob(os.path.join(dspin, "u_*.out")))
fspin_v = np.sort(glob(os.path.join(dspin, "v_*.out")))
fint_u = np.sort(glob(os.path.join(dint, "u_*.out")))
fint_v = np.sort(glob(os.path.join(dint, "v_*.out")))
# throw out all the timeseries files
fspin_u = np.array([f for f in fspin_u if "timeseries" not in f])
fspin_v = np.array([f for f in fspin_v if "timeseries" not in f])
fint_u = np.array([f for f in fint_u if "timeseries" not in f])
fint_v = np.array([f for f in fint_v if "timeseries" not in f])
# total number of files/timesteps
nf_spin = len(fspin_u)
nf_int = len(fint_u)
nf_all = nf_spin + nf_int
# initialize empty time list + u and v arrays to fill during big loop
time_spin, time_int = ([], [])
u_mean_spin, v_mean_spin = (np.zeros((rspin, nf_spin), 
                                     dtype=np.float64) for _ in range(2))
u_mean_int, v_mean_int = (np.zeros((rint, nf_int), 
                                   dtype=np.float64) for _ in range(2))
# dimensions
dz_spin = config["Lz"]/rspin
dz_int = config["Lz"]/rint
z_spin = np.linspace(dz_spin, config["Lz"]-dz_spin, rspin)
z_int = np.linspace(dz_int, config["Lz"]-dz_int, rint)

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')
colors = seaborn.color_palette("colorblind")
# figure save directory
dsave = config["dsave"]
plt.close("all")

# --------------------------------
# Load files and calc xy averages
# --------------------------------
print("Begin reading spinup files...")
for jt in range(nf_spin):
    # load files
    u_in = read_f90_bin(fspin_u[jt],rspin,rspin,rspin,8) * config["uscale"]
    v_in = read_f90_bin(fspin_v[jt],rspin,rspin,rspin,8) * config["uscale"]
    # calc averages and store
    u_mean_spin[:,jt] = np.mean(u_in, axis=(0,1))
    v_mean_spin[:,jt] = np.mean(v_in, axis=(0,1))
    # grab timestep
    t_in = int(fspin_u[jt].split(".")[0][-7:])
    time_spin.append(t_in * config["dt_spin"])
    
print("Begin reading interpolated files...")
for jt in range(nf_int):
    # load files
    u_in = read_f90_bin(fint_u[jt],rint,rint,rint,8) * config["uscale"]
    v_in = read_f90_bin(fint_v[jt],rint,rint,rint,8) * config["uscale"]
    # calc averages and store
    u_mean_int[:,jt] = np.mean(u_in, axis=(0,1))
    v_mean_int[:,jt] = np.mean(v_in, axis=(0,1))
    # grab timestep
    t_in = int(fint_u[jt].split(".")[0][-7:])
    # some janky math to get times to transition properly
    # for first timestep, the previous one was the last from spinup
    if jt==0:
        t_in_prev = int(fspin_u[-1].split(".")[0][-7:])
        t_prev = time_spin[-1]
    # otherwise, calculate number of timesteps in between during interp
    else:
        t_in_prev = int(fint_u[jt-1].split(".")[0][-7:])
        t_prev = time_int[-1]
    time_int.append(t_prev + (t_in-t_in_prev) * config["dt_int"])
    
# convert times to array and calculate hours
time_spin = np.array(time_spin)
time_int = np.array(time_int)
hrs_spin = time_spin/3600.
hrs_int = time_int/3600.

# calculate wdirs
wd_spin = np.arctan2(-u_mean_spin, -v_mean_spin) * 180./np.pi
wd_spin[wd_spin < 0.] += 360.
wd_int = np.arctan2(-u_mean_int, -v_mean_int) * 180./np.pi
wd_int[wd_int < 0.] += 360.

# --------------------------------
# Save to netcdf file
# --------------------------------
fsave = config["fnetcdf"]
ds = xr.Dataset(
        {
            "u_mean_spin": (["z_spin", "t_spin"], u_mean_spin),
            "v_mean_spin": (["z_spin", "t_spin"], v_mean_spin),
            "wd_mean_spin": (["z_spin", "t_spin"], wd_spin),
            "u_mean_int": (["z_int", "t_int"], u_mean_int),
            "v_mean_int": (["z_int", "t_int"], v_mean_int),
            "wd_mean_int": (["z_int", "t_int"], wd_int)
        },
    coords={
        "z_spin": z_spin,
        "t_spin": hrs_spin,
        "z_int": z_int,
        "t_int": hrs_int
    },
    attrs={
        "Lz": config["Lz"],
        "nz_spin": rspin,
        "nz_int": rint,
        "stability": config["stab"]
    })
# loop and assign attributes
for var in config["var_attrs"].keys():
    ds[var].attrs["units"] = config["var_attrs"][var]["units"]
# save to netcdf file and continue
print(f"Saving file: {fsave}")
ds.to_netcdf(fsave)

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
    ax1[0].plot(hrs_spin, u_mean_spin[jz,:], ls="-", c=colors[i])
    ax1[0].plot(hrs_int, u_mean_int[jz*2,:], ls="-", c=colors[i])
    # v
    ax1[1].plot(hrs_spin, v_mean_spin[jz,:], ls="-", c=colors[i])
    ax1[1].plot(hrs_int, v_mean_int[jz*2,:], ls="-", c=colors[i])
    # wdir
    ax1[2].plot(hrs_spin, wd_spin[jz,:], ls="", marker="x",
                markeredgecolor=colors[i])
    ax1[2].plot(hrs_int, wd_int[jz*2,:], ls="", marker="x",
                markeredgecolor=colors[i])
# plot vertical line at crossover
for ax in ax1:
    ax.axvline(hrs_spin[-1], c="k", lw=2)
    
ax1[0].grid()
ax1[0].set_ylabel("$u$ [m s$^{-1}$]")

ax1[1].grid()
ax1[1].set_ylabel("$v$ [m s$^{-1}$]")

ax1[2].grid()
ax1[2].set_xlabel("Time [hrs]")
ax1[2].set_ylabel("$\\alpha$ [deg]")

fig1.suptitle("Simulation A Timeseries - Low Levels")
fig1.tight_layout()
# save and close
fsave1 = f"{dsave}A192_u_v_wd_timeseries.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf")
plt.close(fig1)

#
# Figure 2: 5 levels near h
#
fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
# loop and plot various heights in spinup
for i, jz in enumerate(np.arange(36, 41, 1, dtype=np.int64)):
    # u
    ax2[0].plot(hrs_spin, u_mean_spin[jz,:], ls="-", c=colors[i])
    # v
    ax2[1].plot(hrs_spin, v_mean_spin[jz,:], ls="-", c=colors[i])
    # wdir
    ax2[2].plot(hrs_spin, wd_spin[jz,:], ls="", marker="x",
                markeredgecolor=colors[i])
# loop and plot various heights in interp
for i, jz in enumerate(np.arange(72, 81, 2, dtype=np.int64)):
    # u
    ax2[0].plot(hrs_int, u_mean_int[jz,:], ls="-", c=colors[i])
    # v
    ax2[1].plot(hrs_int, v_mean_int[jz,:], ls="-", c=colors[i])
    # wdir
    ax2[2].plot(hrs_int, wd_int[jz,:], ls="", marker="x",
                markeredgecolor=colors[i])
# plot vertical line at crossover
for ax in ax2:
    ax.axvline(hrs_spin[-1], c="k", lw=2)
    
ax2[0].grid()
ax2[0].set_ylabel("$u$ [m s$^{-1}$]")

ax2[1].grid()
ax2[1].set_ylabel("$v$ [m s$^{-1}$]")

ax2[2].grid()
ax2[2].set_xlabel("Time [hrs]")
ax2[2].set_ylabel("$\\alpha$ [deg]")

fig2.suptitle("Simulation A Timeseries - Near $z = h$")
fig2.tight_layout()
# save and close
fsave2 = f"{dsave}A192_u_v_wd_h_timeseries.pdf"
print(f"Saving figure: {fsave2}")
fig2.savefig(fsave2, format="pdf")
plt.close(fig2)

#
# Figure 3: time-height of wdir
#
tt_spin, zz_spin = np.meshgrid(hrs_spin, z_spin)
tt_int, zz_int = np.meshgrid(hrs_int, z_int)
fig3, ax3 = plt.subplots(1, figsize=(14.8, 5))
# plot spinup
cfax31=ax3.contourf(tt_spin, zz_spin, wd_spin,
                    cmap=seaborn.color_palette("viridis", as_cmap=True),
                    levels=np.arange(200, 301, 5), extend="both")
# plot interp
cfax32=ax3.contourf(tt_int, zz_int, wd_int,
                      cmap=seaborn.color_palette("viridis", as_cmap=True),
                      levels=np.arange(200, 301, 5), extend="both")
# clean up
ax3.set_xlabel("Time [hrs]")
ax3.set_ylabel("$z$ [m]")
# cfax31.set_edgecolor("face")
# cfax32.set_edgecolor("face")
cbar3 = fig3.colorbar(cfax32, ax=ax3)
cbar3.ax.set_ylabel("Wind Direction [deg]")
ax3.set_xlim([0, 10])
ax3.xaxis.set_major_locator(MultipleLocator(1))
ax3.xaxis.set_minor_locator(MultipleLocator(1./6))
ax3.set_ylim([0, 400])
ax3.yaxis.set_major_locator(MultipleLocator(50))
ax3.yaxis.set_minor_locator(MultipleLocator(10))
ax3.axvline(hrs_spin[-1], c="k", lw=2)

fig3.tight_layout()
# save figure
fsave3 = f"{dsave}A192_wd_2d.pdf"
print(f"Saving figure: {fsave3}")
fig3.savefig(fsave3, format="pdf")
plt.close(fig3)

#
# Figure 4: time-height of u and v
#
fig4, ax4 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14.8, 10))
# u
# plot spinup
cfax40=ax4[0].contourf(tt_spin, zz_spin, u_mean_spin,
                    cmap=seaborn.color_palette("viridis", as_cmap=True),
                    levels=np.arange(0, 11, 1), extend="both")
# plot interp
cfax41=ax4[0].contourf(tt_int, zz_int, u_mean_int,
                      cmap=seaborn.color_palette("viridis", as_cmap=True),
                      levels=np.arange(0, 11, 1), extend="both")
# v
# plot spinup
cfax42=ax4[1].contourf(tt_spin, zz_spin, v_mean_spin,
                    cmap=seaborn.color_palette("coolwarm", as_cmap=True),
                    levels=np.arange(-3, 3.1, 0.5), extend="both")
# plot interp
cfax43=ax4[1].contourf(tt_int, zz_int, v_mean_int,
                      cmap=seaborn.color_palette("coolwarm", as_cmap=True),
                      levels=np.arange(-3, 3.1, 0.5), extend="both")
# clean up
ax4[1].set_xlabel("Time [hrs]")
ax4[0].set_ylabel("$z$ [m]")
ax4[1].set_ylabel("$z$ [m]")
cbar41 = fig4.colorbar(cfax41, ax=ax4[0])
cbar41.ax.set_ylabel("$u$ [m/s]")
cbar43 = fig4.colorbar(cfax43, ax=ax4[1])
cbar43.ax.set_ylabel("$v$ [m/s]")
ax4[1].set_xlim([0, 10])
ax4[1].xaxis.set_major_locator(MultipleLocator(1))
ax4[1].xaxis.set_minor_locator(MultipleLocator(1./6))
ax4[0].set_ylim([0, 400])
ax4[0].yaxis.set_major_locator(MultipleLocator(50))
ax4[0].yaxis.set_minor_locator(MultipleLocator(10))
ax4[0].axvline(hrs_spin[-1], c="k", lw=2)
ax4[1].set_ylim([0, 400])
ax4[1].yaxis.set_major_locator(MultipleLocator(50))
ax4[1].yaxis.set_minor_locator(MultipleLocator(10))
ax4[1].axvline(hrs_spin[-1], c="k", lw=2)

fig4.tight_layout()
# save figure
fsave4 = f"{dsave}A192_u_v_2d.pdf"
print(f"Saving figure: {fsave4}")
fig4.savefig(fsave4, format="pdf")
plt.close(fig4)