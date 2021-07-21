#!/usr/bin/env python3
# --------------------------------
# Name: stationarity.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 14 July 2021
# Purpose: calculate fluxes and variances for last 3 hours of simulation as
# as a function of averaging time
# Update 19 July 2021: Include u and v variances (unrotated)
# --------------------------------
import os
import cmocean
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from simulation import read_f90_bin

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/stationarity/"
plt.close("all")

# --------------------------------
@njit
def var(var1, var1mean, nx, ny, nz):
    # input variable and corresponding xyt mean
    # output 1d profile of xy-avg "instantaneous" var
    # initialize var1_fluc to hold data
    var1_fluc = np.zeros((nx,ny,nz), dtype=np.float64)
    # loop and calculate fluctuating components
    for jz in range(nz):
        var1_fluc[:,:,jz] = var1[:,:,jz] - var1mean[jz]
    # now multiply them
    var1var1_fluc = var1_fluc * var1_fluc
    # average in x and y
    to_return = np.zeros(nz, dtype=np.float64)
    for jz in range(nz):
        to_return[jz] = np.mean(var1var1_fluc[:,:,jz])
    return to_return
    
# --------------------------------
# Main
# --------------------------------
#
# define some simulation parameters
#
# domain size
Lx, Ly, Lz = 800., 800., 400.
nx, ny, nz = 192, 192, 192
dx = Lx/nx
dy = Ly/ny
dz = Lz/nz
x = np.linspace(0., Lx, nx, dtype=np.float64)
y = np.linspace(0., Ly, ny, dtype=np.float64)
z = np.linspace(dz, Lz-dz, nx, dtype=np.float64)
# time - last 3 hours
timesteps = np.arange(721000, 1261000, 1000, dtype=np.int64)
nt = len(timesteps)
dt = 0.02  # seconds
t = (timesteps - timesteps[0]) * dt
t_hr = t / 3600.  # hours
# dimensional scales
u_scale = 0.4
theta_scale = 300.

# start with A_192_interp and evaluate theta
fdir = "/home/bgreene/simulations/F_192_interp/output/"

#
# loop data
#
# if A192_stationarity2.npz exists, then load
fdat = "/home/bgreene/SBL_LES/output/F192_stationarity2.npz"

# initialize empty arrays for looping and averaging
# ex: T_mean will have shape(nz, nt) where nt is the number of timesteps
# averaged over
T_mean = np.zeros((nz, nt), dtype=np.float64)
TT_var = np.zeros((nz, nt), dtype=np.float64)
u_mean = np.zeros((nz, nt), dtype=np.float64)
uu_var = np.zeros((nz, nt), dtype=np.float64)
v_mean = np.zeros((nz, nt), dtype=np.float64)
vv_var = np.zeros((nz, nt), dtype=np.float64)
# keep running sum to calc averages over different durations
T_sum = np.zeros((nx, ny, nz), dtype=np.float64)
TT_var_sum = np.zeros((nz, nt), dtype=np.float64)
u_sum = np.zeros((nx, ny, nz), dtype=np.float64)
uu_var_sum = np.zeros((nz, nt), dtype=np.float64)
v_sum = np.zeros((nx, ny, nz), dtype=np.float64)
vv_var_sum = np.zeros((nz, nt), dtype=np.float64)

# Begin loop 1 to calculate averages
# NOTE: loop in reverse temporal order from end!
print("Beginning First loop to calculate averages")
for jt, ts in enumerate(timesteps[::-1]):
    # load data
    f1 = f"{fdir}theta_{ts:07d}.out"
    T_in = read_f90_bin(f1,nx,ny,nz,8) * theta_scale
    f2 = f"{fdir}u_{ts:07d}.out"
    u_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
    f3 = f"{fdir}v_{ts:07d}.out"
    v_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
    
    # accumulate to T_sum
    T_sum += T_in
    # average temporally with jt and spatially with np.mean, assign to T_mean
    T_mean[:,jt] = np.mean((T_sum/(jt+1)), axis=(0,1))
    # accumulate to u_sum
    u_sum += u_in
    # average temporally with jt and spatially with np.mean, assign to u_mean
    u_mean[:,jt] = np.mean((u_sum/(jt+1)), axis=(0,1))
    # accumulate to v_sum
    v_sum += v_in
    # average temporally with jt and spatially with np.mean, assign to v_mean
    v_mean[:,jt] = np.mean((v_sum/(jt+1)), axis=(0,1))
    
# Begin loop 2 to evaluate variances
# NOTE: reverse order again!
print("Beginning second loop to calculate variances")
for jt, ts in enumerate(timesteps[::-1]):
    # load data
    f1 = f"{fdir}theta_{ts:07d}.out"
    T_in = read_f90_bin(f1,nx,ny,nz,8) * theta_scale
    f2 = f"{fdir}u_{ts:07d}.out"
    u_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
    f3 = f"{fdir}v_{ts:07d}.out"
    v_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
        
    # loop over T_mean averages
    for it in range(jt, nt):
        # calculate "instantaneous" variance relative to increasing average time
        TT_var_sum[:, it] += var(T_in, T_mean[:,it], nx, ny, nz)
        uu_var_sum[:, it] += var(u_in, u_mean[:,it], nx, ny, nz)
        vv_var_sum[:, it] += var(v_in, v_mean[:,it], nx, ny, nz)
        
# Begin loop 3 to calculate averages of cumulative variances
print("Beginning third loop to finalize variances")
for jt in range(nt):
    TT_var[:, jt] = TT_var_sum[:, jt] / (jt+1)
    uu_var[:, jt] = uu_var_sum[:, jt] / (jt+1)
    vv_var[:, jt] = vv_var_sum[:, jt] / (jt+1)
        
# export npz file
np.savez(fdat, TT_var=TT_var, uu_var=uu_var, vv_var=vv_var)

#
# Figure 1: 1d plots of every 30 timesteps overlaid 
# theta var
#
fig1, ax1 = plt.subplots(1, figsize=(6, 8))
alpha = np.linspace(0., 1., nt)
# loop over times and plot
for jt in np.arange(0, nt, 30, dtype=np.int64):
    # plot every hour in red
    if jt % 180. == 0.:
        c = "r"
    elif jt == nt:
        c = "r"
    else:
        c = "k"
    ax1.plot(TT_var[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    
# labels
ax1.grid()
ax1.set_ylim([0, 400])
ax1.set_xlabel("$\\sigma_{\\theta}$ [K$^2$]")
ax1.set_ylabel("$z$ [m]")
# save and close
fsave1 = f"{fdir_save}F192_thetavar.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
plt.close(fig1)

#
# Figure 2: 1d plots of every 30 timesteps overlaid 
# uvar, vvar
#
fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8))
alpha = np.linspace(0., 1., nt)
# loop over times and plot
for jt in np.arange(0, nt, 30, dtype=np.int64):
    # plot every hour in red
    if jt % 180. == 0.:
        c = "r"
    elif jt == nt:
        c = "r"
    else:
        c = "k"
    ax2[0].plot(uu_var[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    ax2[1].plot(vv_var[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    
# labels
ax2[0].grid()
ax2[0].set_ylim([0, 400])
ax2[0].set_xlabel("$\\sigma_u^2$ [m$^2$ s$^{-2}$]")
ax2[0].set_ylabel("$z$ [m]")
ax2[1].grid()
ax2[1].set_xlabel("$\\sigma_v^2$ [m$^2$ s$^{-2}$]")
# save and close
fsave2 = f"{fdir_save}F192_u_v_var.pdf"
print(f"Saving figure: {fsave2}")
fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
plt.close(fig2)








