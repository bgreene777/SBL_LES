#!/usr/bin/env python3
# --------------------------------
# Name: var_covar_timeseries.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 21 July 2021
# Purpose: plot timeseries of instantaneous vars and covars calculated from 
# xy planar averages to evaluate stationarity
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
def covar(var1, var2, sgs, nx, ny, nz):
    # input variable and corresponding sgs component
    # output 1d profile of xy-avg "instantaneous" covar
    # initialize var1_fluc to hold data
    var1_fluc = np.zeros((nx,ny,nz), dtype=np.float64)
    var2_fluc = np.zeros((nx,ny,nz), dtype=np.float64)
    # loop and calculate fluctuating components
    for jz in range(nz):
        var1_fluc[:,:,jz] = var1[:,:,jz] - np.mean(var1[:,:,jz])
        var2_fluc[:,:,jz] = var2[:,:,jz] - np.mean(var2[:,:,jz])
    # now multiply them
    var1var2_fluc = var1_fluc * var2_fluc
    # add in sgs component
    var1var2_fluc_tot = var1var2_fluc + sgs
    # average in x and y
    to_return = np.zeros(nz, dtype=np.float64)
    for jz in range(nz):
        to_return[jz] = np.mean(var1var2_fluc_tot[:,:,jz])
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

# if F192_stationarity3.npz exists, then load
fdat = "/home/bgreene/SBL_LES/output/F192_stationarity3.npz"

if os.path.exists(fdat):
    print(f"Reading file: {fdat}")
    dat = np.load(fdat)
    uu_var = dat["uu_var"]
    vv_var = dat["vv_var"]
    ww_var = dat["ww_var"]
    tt_var = dat["tt_var"]
    uw_cov_tot = dat["uw_cov_tot"]
    vw_cov_tot = dat["vw_cov_tot"]
    tw_cov_tot = dat["tw_cov_tot"]
    dT_dz = dat["dT_dz"]
else:
    # initialize empty arrays to store xy planar averages of vars and covars
    uu_var, vv_var, ww_var, tt_var, uw_cov_tot, vw_cov_tot, tw_cov_tot =\
    (np.zeros((nz,nt), dtype=np.float64) for _ in range(7))
    # initialize empty array to use as sgs for vars
    zero = np.zeros((nx,ny,nz), dtype=np.float64)
    # calculate timeseries of prescribed surface temp from simulation
    T0 = 265.  # K
    Tc = 2.5   # K/hr
    # starting temp for last 3 hours:
    # 6 hours before interp + 1 hour after interp
    T3h0 = T0 - (Tc * 7.)
    # now create array of surface temps
    Ts = T3h0 - (Tc * t_hr)
    # initialize empty dt_dz
    dT_dz = np.zeros(nt, dtype=np.float64)

    # begin looping through files
    for jt, ts in enumerate(timesteps):
        # load files - DONT FORGET SCALES!
        f1 = f"{fdir}u_{ts:07d}.out"
        u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
        f2 = f"{fdir}v_{ts:07d}.out"
        v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
        f3 = f"{fdir}w_{ts:07d}.out"
        w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
        f4 = f"{fdir}theta_{ts:07d}.out"
        theta_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
        f5 = f"{fdir}txz_{ts:07d}.out"
        txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
        f6 = f"{fdir}tyz_{ts:07d}.out"
        tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
        f7 = f"{fdir}q3_{ts:07d}.out"
        q3_in = read_f90_bin(f7,nx,ny,nz,8) * u_scale * theta_scale

        # calculate xy-averaged vars and covars and store
        # start with vars
        # <u'u'>
        uu_var[:,jt] = covar(u_in, u_in, zero, nx, ny, nz)
        # <v'v'>
        vv_var[:,jt] = covar(v_in, v_in, zero, nx, ny, nz)
        # <w'w'>
        ww_var[:,jt] = covar(w_in, w_in, zero, nx, ny, nz)
        # <theta'theta'>
        tt_var[:,jt] = covar(theta_in, theta_in, zero, nx, ny, nz)
        # covars
        # <u'w'>
        uw_cov_tot[:,jt] = covar(u_in, w_in, txz_in, nx, ny, nz)
        # <v'w'>
        vw_cov_tot[:,jt] = covar(v_in, w_in, tyz_in, nx, ny, nz)
        # <theta'w'>
        tw_cov_tot[:,jt] = covar(theta_in, w_in, q3_in, nx, ny, nz)
        
        # calculate dT/dz between surface and lowest grid point
        dT = np.mean(theta_in[:,:,0]) - Ts[jt]
        dT_dz[jt] = dT/dz

    # save npz file for future use to avoid long loop
    print(f"Saving file: {fdat}")
    np.savez(fdat, uu_var=uu_var, vv_var=vv_var, ww_var=ww_var, 
             tt_var=tt_var,
             uw_cov_tot=uw_cov_tot, vw_cov_tot=vw_cov_tot, 
             tw_cov_tot=tw_cov_tot, dT_dz=dT_dz)
    
#
# Plot timeseries
#
colors = [(252./255, 193./255, 219./255), (225./255, 156./255, 131./255),
          (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
          (0., 85./255, 80./255), (20./255, 33./255, 61./255), 
          ]

#
# Figure 1: u, v, theta var
#
fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
# loop and plot various heights
for i, jz in enumerate(np.arange(6, dtype=np.int64)**2):
    # u var
    ax1[0].plot(t_hr, uu_var[jz,:], ls="-", c=colors[i])
    # v var
    ax1[0].plot(t_hr, vv_var[jz,:], ls="--", c=colors[i])
    # w var
    ax1[1].plot(t_hr, ww_var[jz,:], ls="-", c=colors[i])
    # theta var
    ax1[2].plot(t_hr, tt_var[jz,:], ls="-", c=colors[i], label=f"{z[jz]:4.1f} m")
    
ax1[0].grid()
ax1[0].set_ylabel("$u'u'$, $v'v'$ [m$^2$ s$^{-2}$]")
ax1[0].set_title("$u'u'$ (solid) and $v'v'$ (dashed)")

ax1[1].grid()
ax1[1].set_ylabel("$w'w'$ [m$^2$ s$^{-2}$]")

ax1[2].grid()
ax1[2].set_xlabel("Time [hrs]")
ax1[2].set_ylabel("$\\theta'\\theta'$ [K$^2$]", fontsize=16)
ax1[2].legend()

fig1.suptitle("Simulation F Variances Timeseries - Last 3 Hours")
# save and close
fsave1 = f"{fdir_save}F192_u_v_w_theta_var_timeseries.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
plt.close(fig1)

#
# Figure 2: uw, vw, thetaw
#
fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
# loop and plot various heights
for i, jz in enumerate(np.arange(6, dtype=np.int64)**2):
    # uw
    ax2[0].plot(t_hr, uw_cov_tot[jz,:], ls="-", c=colors[i])
    # vw
    ax2[0].plot(t_hr, vw_cov_tot[jz,:], ls="--", c=colors[i])
    # thetaw
    ax2[1].plot(t_hr, tw_cov_tot[jz,:], ls="-", c=colors[i], label=f"{z[jz]:4.1f} m")
    
ax2[0].grid()
ax2[0].set_ylabel("$u'w'$, $v'w'$ [m$^2$ s$^{-2}$]")
ax2[0].set_title("$u'w'$ (solid) and $v'w'$ (dashed)")

ax2[1].grid()
ax2[1].set_xlabel("Time [hrs]")
ax2[1].set_ylabel("$\\theta'w'$ [K m s$^{-1}$]", fontsize=16)
ax2[1].legend()

fig2.suptitle("Simulation F Covariances Timeseries - Last 3 Hours")
# save and close
fsave2 = f"{fdir_save}F192_uw_vw_thetaw_timeseries.pdf"
print(f"Saving figure: {fsave2}")
fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
plt.close(fig2)

#
# Figure 3: dT/dz
#
fig3, ax3 = plt.subplots(1, figsize=(8, 4))
ax3.plot(t_hr, dT_dz, ls="-", c="k")
# labels
ax3.grid()
ax3.set_xlabel("Time [hrs]")
ax3.set_ylabel("$\\partial T / \\partial z$ [K m$^{-1}$]")
# save and close
fsave3 = f"{fdir_save}F192_dtheta_dz_timeseries.pdf"
print(f"Saving figure: {fsave3}")
fig3.savefig(fsave3, format="pdf", bbox_inches="tight")
plt.close(fig3)