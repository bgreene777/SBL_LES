# --------------------------------
# Name: stationarity.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 13 July 2021
# Purpose: loop through 3d output volumes from interpolated sims
# and plot x-y mean profiles of temperature at each step to check
# for stationarity
# also load in velocity profiles to calc mean stress to determine sbl depth h
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
    # input 2 variables and corresponding subgrid data
    # output 1d profile of xy-avg "instantaneous" covar (filtered + subgrid)
    # initialize var1_fluc, var2_fluc to hold data
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
# time
timesteps = np.arange(541000, 1261000, 1000, dtype=np.int64)
nt = len(timesteps)
dt = 0.02  # seconds
t = (timesteps - timesteps[0]) * dt
t_hr = t / 3600.  # hours
# dimensional scales
u_scale = 0.4
theta_scale = 300.

# start with A_192_interp and evaluate theta
fdir = "/home/bgreene/simulations/A_192_interp/output/"

#
# loop data
#
# if A192_stationarity.npz exists, then load
fdat = "/home/bgreene/SBL_LES/output/A192_stationarity.npz"
if os.path.exists(fdat):
    dat = np.load(fdat)
    T_in_mean = dat["T_in_mean"]
    u_in_mean = dat["u_in_mean"]
    v_in_mean = dat["v_in_mean"]
    uw_mean = dat["uw_mean"]
    vw_mean = dat["vw_mean"]
    uu_mean = dat["uu_mean"]
    vv_mean = dat["vv_mean"]
    ww_mean = dat["ww_mean"]
    tt_mean = dat["tt_mean"]
    h = dat["h"]
    
else:
    # else, loop through timesteps and read
    # initialize 2d array shape(nz,nt) and average in x and y for each step
    T_in_mean, u_in_mean, v_in_mean, uw_mean, vw_mean,\
    uu_mean, vv_mean, ww_mean, tt_mean =\
    (np.zeros((nz, nt), dtype=np.float64) for _ in range(9))
    h = np.zeros(nt, dtype=np.float64)
    zeros = np.zeros((nx,ny,nz), dtype=np.float64)
    for jt, t in enumerate(timesteps):
        # load data
        f1 = f"{fdir}u_{timesteps[jt]:07d}.out"
        u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
        f2 = f"{fdir}v_{timesteps[jt]:07d}.out"
        v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
        f3 = f"{fdir}w_{timesteps[jt]:07d}.out"
        w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
        f4 = f"{fdir}theta_{timesteps[jt]:07d}.out"
        T_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
        f5 = f"{fdir}txz_{timesteps[jt]:07d}.out"
        txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
        f6 = f"{fdir}tyz_{timesteps[jt]:07d}.out"
        tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
        # average in x and y and assign to T_in_mean
        u_in_mean[:,jt] = np.mean(u_in, axis=(0,1))
        v_in_mean[:,jt] = np.mean(v_in, axis=(0,1))
        T_in_mean[:,jt] = np.mean(T_in, axis=(0,1))
        # calculate covariances
        uw_mean[:,jt] = covar(u_in, w_in, txz_in, nx, ny, nz)
        vw_mean[:,jt] = covar(v_in, w_in, tyz_in, nx, ny, nz)
        uu_mean[:,jt] = covar(u_in, u_in, zeros, nx, ny, nz)  # unrotated
        vv_mean[:,jt] = covar(v_in, v_in, zeros, nx, ny, nz)  # unrotated
        ww_mean[:,jt] = covar(w_in, w_in, zeros, nx, ny, nz)
        tt_mean[:,jt] = covar(T_in, T_in, zeros, nx, ny, nz)
        # now calculate h from uw and vw
        ustar = (uw_mean[:,jt]**2. + vw_mean[:,jt]**2.) ** 0.25
        i_h = np.where(ustar**2. <= 0.05*ustar[0]*ustar[0])[0][0]
        h[jt] = z[i_h] / 0.95
        
    # save npz file
    np.savez(fdat, u_in_mean=u_in_mean, v_in_mean=v_in_mean, 
             T_in_mean=T_in_mean,
             uw_mean=uw_mean, vw_mean=vw_mean, h=h,
             uu_mean=uu_mean, vv_mean=vv_mean, ww_mean=ww_mean,
             tt_mean=tt_mean)
    
# calculate ustar in correct dimensions for plotting
ustar = (uw_mean**2. + vw_mean**2.) ** 0.25    
# calculate wind directions
wdir = 270. - (np.arctan2(v_in_mean, u_in_mean) * 180./np.pi)

#
# Figure 1: time-height of all these temperature profiles (lol)
#
tt, zz = np.meshgrid(t_hr, z)

fig1, ax1 = plt.subplots(1, figsize=(16, 10))
cfax1 = ax1.pcolormesh(tt, zz, T_in_mean, cmap=cmocean.cm.amp)
cfax1.set_edgecolor("face")
cax1 = ax1.contour(tt, zz, T_in_mean, levels=np.arange(262, 268.5, 0.5), colors="white")
ax1.clabel(cax1, cax1.levels, inline=True, fontsize=14, fmt="%3.0f")
cbar1 = fig1.colorbar(cfax1, ax=ax1)
# plot h vs time overlaid
ax1.plot(t_hr, h, ":k", linewidth=4)
# labels
cbar1.ax.set_ylabel("$\\theta$ [K]")
ax1.set_xlabel("Time [hr]")
ax1.set_ylabel("$z$ [m]")
ax1.set_xlim([0, 4])
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.set_ylim([0, 400])
ax1.yaxis.set_major_locator(MultipleLocator(50))
ax1.yaxis.set_minor_locator(MultipleLocator(10))
# save and close
fsave1 = f"{fdir_save}A192_theta2d.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
plt.close(fig1)

#
# Figure 2: 1d plots of every 30 timesteps (10 minutes) overlaid
#
fig2, ax2 = plt.subplots(1, figsize=(6, 8))
alpha = np.linspace(0., 1., 720)
# loop over times and plot
for jt in np.arange(0, 720, 30, dtype=np.int64):
    # plot every hour in red
    if jt % 180. == 0.:
        c = "r"
    else:
        c = "k"
    ax2.plot(T_in_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    
# labels
ax2.grid()
ax2.set_xlim([262, 268])
ax2.set_ylim([0, 400])
ax2.set_xlabel("$\\theta$ [K]")
ax2.set_ylabel("$z$ [m]")
# save and close
fsave2 = f"{fdir_save}A192_theta1d.pdf"
print(f"Saving figure: {fsave2}")
fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
plt.close(fig2)

#
# Figure 3: 1d plots of every 30 timesteps (10 minutes) overlaid
# u'w', v'w', ustar
#
fig3, ax3 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
alpha = np.linspace(0., 1., 720)
# loop over times and plot
for jt in np.arange(0, 720, 30, dtype=np.int64):
    # plot every hour in red
    if jt % 180. == 0.:
        c = "r"
    else:
        c = "k"
    ax3[0].plot(uw_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    ax3[1].plot(vw_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    ax3[2].plot(ustar[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    
# labels
ax3[0].grid()
ax3[0].set_ylim([0, 400])
ax3[0].set_ylabel("$z$ [m]")
ax3[0].set_xlabel("$\\langle u'w' \\rangle$ [m$^2$ s$^{-2}$]")

ax3[1].grid()
ax3[1].set_xlabel("$\\langle v'w' \\rangle$ [m$^2$ s$^{-2}$]")

ax3[2].grid()
ax3[2].set_xlabel("$u_{*}$ [m s$^{-1}$]")
# save and close
fsave3 = f"{fdir_save}A192_uw_vw.pdf"
print(f"Saving figure: {fsave3}")
fig3.savefig(fsave3, format="pdf", bbox_inches="tight")
plt.close(fig3)

#
# Figure 4: 1d plots of every 30 timesteps (10 minutes) overlaid
# u'u', v'v', w'w', theta'theta'
#
fig4, ax4 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 8))
alpha = np.linspace(0., 1., 720)
# loop over times and plot
for jt in np.arange(0, 720, 30, dtype=np.int64):
    # plot every hour in red
    if jt % 180. == 0.:
        c = "r"
    else:
        c = "k"
    ax4[0].plot(uu_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    ax4[1].plot(vv_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    ax4[2].plot(ww_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    ax4[3].plot(tt_mean[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    
# labels
ax4[0].grid()
ax4[0].set_ylim([0, 400])
ax4[0].set_ylabel("$z$ [m]")
ax4[0].set_xlabel("$\\sigma_u^2$ [m$^2$ s$^{-2}$]")

ax4[1].grid()
ax4[1].set_xlabel("$\\sigma_v^2$ [m$^2$ s$^{-2}$]")

ax4[2].grid()
ax4[2].set_xlabel("$\\sigma_w^2$ [m$^2$ s$^{-2}$]")

ax4[3].grid()
ax4[3].set_xlabel("$\\sigma_{\\theta}^2$ [K$^2$]")
# save and close
fsave4 = f"{fdir_save}A192_var.pdf"
print(f"Saving figure: {fsave4}")
fig4.savefig(fsave4, format="pdf", bbox_inches="tight")
plt.close(fig4)

#
# Figure 5: 1d plots of every 30 timesteps (10 minutes) overlaid
# Wind Direction
#
fig5, ax5 = plt.subplots(1, figsize=(6, 8))
alpha = np.linspace(0., 1., 720)
# loop over times and plot
for jt in np.arange(0, 720, 30, dtype=np.int64):
    # plot every hour in red
    if jt % 180. == 0.:
        c = "r"
    else:
        c = "k"
    ax5.plot(wdir[:,jt], z, ls="-", c=c, alpha=alpha[jt])
    
# labels
ax5.grid()
# ax2.set_xlim([262, 268])
ax5.set_ylim([0, 400])
ax5.set_xlabel("Wind Direction [$^\circ$]")
ax5.set_ylabel("$z$ [m]")
# save and close
fsave5 = f"{fdir_save}A192_wdir.pdf"
print(f"Saving figure: {fsave5}")
fig5.savefig(fsave5, format="pdf", bbox_inches="tight")
plt.close(fig5)