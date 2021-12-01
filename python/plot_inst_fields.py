# --------------------------------
# Name: plot_inst_fields.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 8 November 2021
# Purpose: Read instantaneous file output to plot x-z cross-sections
# --------------------------------
import os
import sys
import yaml
import numpy as np
import xarray as xr
import seaborn
import cmocean
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, LogLocator
from dask.diagnostics import ProgressBar
# ---------------------------------------------
class MidPointNormalize(Normalize):
    """Defines the midpoint of diverging colormap.
    Usage: Allows one to adjust the colorbar, e.g. 
    using contouf to plot data in the range [-3,6] with
    a diverging colormap so that zero values are still white.
    Example usage:
        norm=MidPointNormalize(midpoint=0.0)
        f=plt.contourf(X,Y,dat,norm=norm,cmap=colormap)
     """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
# ---------------------------------------------
#
# Directories
#
fsim = "/home/bgreene/simulations/"
figdir = "/home/bgreene/SBL_LES/figures/inst_fields/"
#        
# Configure plots
#
rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
rc('text',usetex='True')
props=dict(boxstyle='square',facecolor='white',alpha=0.85)
colors = seaborn.color_palette("crest")
plt.close("all")

# --------------------------------
# 9-panel of u/ustar, w/ustar, theta for A, C, F
# --------------------------------
# load last timestep for simulations A, C, F
fA = f"{fsim}A_192_interp/output/netcdf/all_1260000.nc"
print(f"Loading file: {fA}")
A = xr.load_dataset(fA)
fC = f"{fsim}C_192_interp/output/netcdf/all_1260000.nc"
print(f"Loading file: {fC}")
C = xr.load_dataset(fC)
fF = f"{fsim}F_192_interp/output/netcdf/all_1260000.nc"
print(f"Loading file: {fF}")
F = xr.load_dataset(fF)
# load mean statistics file for each
fAstat = f"{fsim}A_192_interp/output/netcdf/average_statistics.nc"
print(f"Loading file: {fAstat}")
Astat = xr.load_dataset(fAstat)
fCstat = f"{fsim}C_192_interp/output/netcdf/average_statistics.nc"
print(f"Loading file: {fCstat}")
Cstat = xr.load_dataset(fCstat)
fFstat = f"{fsim}F_192_interp/output/netcdf/average_statistics.nc"
print(f"Loading file: {fFstat}")
Fstat = xr.load_dataset(fFstat)
# calculate important parameters
# ustar
Astat["ustar"] = ((Astat.uw_cov_tot ** 2.) + (Astat.vw_cov_tot ** 2.)) ** 0.25
Astat["ustar2"] = Astat.ustar ** 2.
Cstat["ustar"] = ((Cstat.uw_cov_tot ** 2.) + (Cstat.vw_cov_tot ** 2.)) ** 0.25
Cstat["ustar2"] = Cstat.ustar ** 2.
Fstat["ustar"] = ((Fstat.uw_cov_tot ** 2.) + (Fstat.vw_cov_tot ** 2.)) ** 0.25
Fstat["ustar2"] = Fstat.ustar ** 2.
# SBL height
Astat["h"] = Astat.z.where(Astat.ustar2 <= 0.05*Astat.ustar2[0], drop=True)[0]/0.95
Cstat["h"] = Cstat.z.where(Cstat.ustar2 <= 0.05*Cstat.ustar2[0], drop=True)[0]/0.95
Fstat["h"] = Fstat.z.where(Fstat.ustar2 <= 0.05*Fstat.ustar2[0], drop=True)[0]/0.95
# z indices within sbl
Astat["isbl"] = np.where(Astat.z <= Astat.h)[0]
Cstat["isbl"] = np.where(Cstat.z <= Cstat.h)[0]
Fstat["isbl"] = np.where(Fstat.z <= Fstat.h)[0]
# calculate rotated velocities
for sim, stat in zip([A, C, F], [Astat, Cstat, Fstat]):
    # calculate inst u_rot, v_rot using stat
    sim["u_rot"] = sim.u*np.cos(stat.alpha) + sim.v*np.sin(stat.alpha)
    sim["v_rot"] =-sim.u*np.sin(stat.alpha) + sim.v*np.cos(stat.alpha)

#
# Begin plotting
#
# grab seaborn color palette
sbmako = seaborn.color_palette("mako", as_cmap=True)
sbvlag = seaborn.color_palette("vlag", as_cmap=True)
sbrckt = seaborn.color_palette("rocket", as_cmap=True)
cmbal = cmocean.cm.balance
cmtpo = cmocean.cm.tempo
cmthm = cmocean.cm.thermal
# figure 1
# rows = A,  F
# columns = u/ustar, theta
fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(14.8, 5), 
                         constrained_layout=True)
# column 1: u/ustar
cax1 = ax1[0,0].contourf(A.x/Astat.h, A.z/Astat.h, A.u_rot.isel(y=96).T/Astat.ustar.isel(z=0),
                         levels=np.arange(0, 40.1, 0.5, np.float64),
                         extend="max", cmap=cmtpo)
cax2 = ax1[1,0].contourf(F.x/Fstat.h, F.z/Fstat.h, F.u_rot.isel(y=96).T/Fstat.ustar.isel(z=0),
                         levels=np.arange(0, 60.1, 0.5, np.float64),
                         extend="max", cmap=cmtpo)
# column 2: theta
cax3 = ax1[0,1].contourf(A.x/Astat.h, A.z/Astat.h, A.theta.isel(y=96).T,
                         levels=np.arange(263., 265.1, 0.1, np.float64), extend="both",
                         cmap=cmthm)
cax4 = ax1[1,1].contourf(F.x/Fstat.h, F.z/Fstat.h, F.theta.isel(y=96).T,
                         levels=np.arange(244., 262.1, 0.1, np.float64), extend="both",
                         cmap=cmthm)    

# colorbars
cb1 = fig1.colorbar(cax1, ax=ax1[0,0], location="right", 
                    ticks=MultipleLocator(10), pad=0, aspect=15)
cb1.ax.set_title("$u/u_{*}$", fontsize=16)
cb1.ax.tick_params(labelsize=16)
cb2 = fig1.colorbar(cax2, ax=ax1[1,0], location="right", 
                    ticks=MultipleLocator(15), pad=0, aspect=15)
cb2.ax.set_title("$u/u_{*}$", fontsize=16)
cb2.ax.tick_params(labelsize=16)
cb3 = fig1.colorbar(cax3, ax=ax1[0,1], location="right", 
                    ticks=MultipleLocator(1), pad=0, aspect=15)
cb3.ax.set_title("$\\theta$ [K]", fontsize=16)
cb3.ax.tick_params(labelsize=16)
cb4 = fig1.colorbar(cax4, ax=ax1[1,1], location="right", 
                    ticks=MultipleLocator(5), pad=0, aspect=15)
cb4.ax.set_title("$\\theta$ [K]", fontsize=16)
cb4.ax.tick_params(labelsize=16)
# labels
ax1[0,0].set_ylim([0, 1])
ax1[0,0].set_xlim([0, 5])
ax1[0,1].set_xlim([0, 5])
ax1[1,0].set_xlim([0, 5])
ax1[1,1].set_xlim([0, 5])
for iax in ax1[1,:]:
    iax.set_xlabel("$x/h$")
for iax in ax1.flatten():
    iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
    iax.tick_params(which="major", length=6, width=0.5)
    iax.tick_params(which="minor", length=3, width=0.5)
ax1[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
ax1[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax1[0,0].set_ylabel("$z/h$")
ax1[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
ax1[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax1[0,1].xaxis.set_major_locator(MultipleLocator(0.5))
ax1[0,1].xaxis.set_minor_locator(MultipleLocator(0.1))
ax1[1,0].set_ylabel("$z/h$")
ax1[1,0].xaxis.set_major_locator(MultipleLocator(0.5))
ax1[1,0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax1[1,1].xaxis.set_major_locator(MultipleLocator(0.5))
ax1[1,1].xaxis.set_minor_locator(MultipleLocator(0.1))
# label subpanels
for iax, s in zip(ax1.flatten(), list("abcd")):
    iax.text(0.02,0.84,f"$\\textbf{{({s})}}$",fontsize=16,bbox=props,
              transform=iax.transAxes)

# fig1.tight_layout()
# save and close
fsave1 = f"{figdir}u_theta.png"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, dpi=600)
plt.close(fig1)

# figure 2
# rows = A, F
# columns = u'/ustar, w'/ustar, theta'
fig2, ax2 = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(14.8, 5), 
                         constrained_layout=True)
# column 1: u'/ustar
# make the norm:  Note the center is offset so that the land has more
# dynamic range:
norm=MidPointNormalize(midpoint=0.0)
cax1 = ax2[0,0].contourf(A.x/Astat.h, A.z/Astat.h, 
                         (A.u_rot.isel(y=96)-Astat.u_mean_rot).T/Astat.ustar.isel(z=0),
                         levels=np.arange(-4., 6.1, 0.1, np.float64),
                         extend="both", cmap=cmbal, norm=norm)
cax2 = ax2[1,0].contourf(F.x/Fstat.h, F.z/Fstat.h, 
                         (F.u_rot.isel(y=96)-Fstat.u_mean_rot).T/Fstat.ustar.isel(z=0),
                         levels=np.arange(-4., 4.1, 0.1, np.float64),
                         extend="both", cmap=cmbal)
# column 2: theta'
norm=MidPointNormalize(midpoint=0.0)
cax3 = ax2[0,1].contourf(A.x/Astat.h, A.z/Astat.h, 
                         (A.theta.isel(y=96)-Astat.theta_mean).T,
                         levels=np.arange(-.5, 0.21, 0.01, np.float64), 
                         extend="both", cmap=cmbal, norm=norm)
norm=MidPointNormalize(midpoint=0.0)
cax4 = ax2[1,1].contourf(F.x/Fstat.h, F.z/Fstat.h, 
                         (F.theta.isel(y=96)-Fstat.theta_mean).T,
                         levels=np.arange(-1.5, 0.11, 0.01, np.float64), 
                         extend="both", cmap=cmbal, norm=norm)    

# colorbars
cb1 = fig2.colorbar(cax1, ax=ax2[0,0], location="right", 
                    ticks=MultipleLocator(2), pad=0, aspect=15)
cb1.ax.set_title("$u'/u_{*}$", fontsize=16)
cb1.ax.tick_params(labelsize=16)
cb2 = fig2.colorbar(cax2, ax=ax2[1,0], location="right", 
                    ticks=MultipleLocator(2), pad=0, aspect=15)
cb2.ax.set_title("$u'/u_{*}$", fontsize=16)
cb2.ax.tick_params(labelsize=16)
cb3 = fig2.colorbar(cax3, ax=ax2[0,1], location="right", 
                    ticks=MultipleLocator(0.2), pad=0, aspect=15)
cb3.ax.set_title("$\\theta'$ [K]", fontsize=16)
cb3.ax.tick_params(labelsize=16)
cb4 = fig2.colorbar(cax4, ax=ax2[1,1], location="right", 
                    ticks=MultipleLocator(0.5), pad=0, aspect=15)
cb4.ax.set_title("$\\theta'$ [K]", fontsize=16)
cb4.ax.tick_params(labelsize=16)
# labels
ax2[0,0].set_ylim([0, 1])
ax2[0,0].set_xlim([0, 5])
ax2[0,1].set_xlim([0, 5])
ax2[1,0].set_xlim([0, 5])
ax2[1,1].set_xlim([0, 5])
for iax in ax2.flatten():
    iax.set_xlabel("$x/h$")
ax2[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
ax2[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax2[0,0].set_ylabel("$z/h$")
ax2[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
ax2[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax2[0,1].xaxis.set_major_locator(MultipleLocator(0.5))
ax2[0,1].xaxis.set_minor_locator(MultipleLocator(0.1))
ax2[1,0].set_ylabel("$z/h$")
ax2[1,0].xaxis.set_major_locator(MultipleLocator(0.5))
ax2[1,0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax2[1,1].xaxis.set_major_locator(MultipleLocator(0.5))
ax2[1,1].xaxis.set_minor_locator(MultipleLocator(0.1))

# save and close
fsave2 = f"{figdir}u_theta_fluc.png"
print(f"Saving figure: {fsave2}")
fig2.savefig(fsave2, dpi=300)
plt.close(fig2)