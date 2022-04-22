# --------------------------------
# Name: plot_mean.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 10 May 2021
# Purpose: Read xyt averaged files from calc_stats.f90 to plot profiles of
# quantities output by LES. Loops over multiple grid sizes and stabilities
# for comparisons
# 19 July 2021 Update: now that resolution has been selected, can plot same
# figures to compare stabilities
# 3 September 2021 Update: use the averaged netcdf files and xarray to plot
# more concisely
# 4 April 2022 Update: use load_stats from LESnc.py to load average statistics
# files and calculate important parameters behind the scenes
# --------------------------------
import os
import seaborn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from LESnc import load_stats

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
rc('text',usetex='True')
# colors = [(225./255, 156./255, 131./255),
#           (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
#           (0., 85./255, 80./255), (20./255, 33./255, 61./255), (252./255, 193./255, 219./255)]
colors = seaborn.color_palette("crest")
# colors = seaborn.color_palette("colorblind")
fdir_save = "/home/bgreene/SBL_LES/figures/mean_profiles/"
if not os.path.exists(fdir_save):
    os.mkdir(fdir_save)
plt.close("all")
# --------------------------------
# Load netcdf stats files
# --------------------------------
stabs = ["A", "B", "C", "D", "E", "F"]
s_all = []
for s in stabs:
    fread = f"/home/bgreene/simulations/{s}_192_interp/output/netcdf/average_statistics.nc"
    s_all.append(load_stats(fread, SBL=True, display=True))
# # --------------------------------
# Begin plotting
# --------------------------------
#
props=dict(boxstyle="square",facecolor="white",edgecolor="white",alpha=0.0)
# Figure 1: 9-panel everything
# (a) <u>, <v>; (b) wdir; (c) <\Theta>;
# (d) <u'w'>, <v'w'>; (e) <\theta'w'>; (f) <u'^2>;
# (g) <v'^2>; (h) <w'^2>; (i) <\theta'^2>
fig1, ax1 = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(14.8, 14.8))
for i, s in enumerate(s_all):
    # now plot
    # row 1
    # (a) <u>, <v>
    # ax1[0,0].plot(s.u_mean, s.z/s.h, ls="-", c=colors[i], lw=2)
    # ax1[0,0].plot(s.v_mean, s.z/s.h, ls=":", c=colors[i], lw=2)
    ax1[0,0].plot(s.uh, s.z/s.h, ls="-", c=colors[i], lw=2)
    # (b) wind direction
    ax1[0,1].plot(s.wdir, s.z/s.h, ls="-", c=colors[i], lw=2, 
                  label=f"{s.stability}")
    # (c) <\Theta>
    ax1[0,2].plot(s.theta_mean, s.z/s.h, ls="-", c=colors[i], lw=2)
    # row 2
    # (d) ustar^2
    # ax1[1,0].plot(s.uw_cov_tot/s.ustar0/s.ustar0, s.z/s.h, ls="-", c=colors[i], lw=2)
    # ax1[1,0].plot(s.vw_cov_tot/s.ustar0/s.ustar0, s.z/s.h, ls=":", c=colors[i], lw=2)
    ax1[1,0].plot(s.ustar2/s.ustar0/s.ustar0, s.z/s.h, ls="-", c=colors[i], lw=2)
    # (e) <\theta'w'>
    ax1[1,1].plot(s.tw_cov_tot/s.ustar0/s.tstar0, s.z/s.h, ls="-", c=colors[i], lw=2)
    # (f) Rig, Rif
    ax1[1,2].plot(s.Rig, s.z/s.h, ls="-", c=colors[i], lw=2)
    ax1[1,2].plot(s.Rif, s.z/s.h, ls="--", c=colors[i], lw=2)
    # row 3
    # (g) <u'^2> ROTATED
    ax1[2,0].plot(s.u_var_rot/s.ustar0/s.ustar0, s.z/s.h, ls="-", c=colors[i], lw=2)
    # (h) <w'^2>
    ax1[2,1].plot(s.w_var/s.ustar0/s.ustar0, s.z/s.h, ls="-", c=colors[i], lw=2)
    # (i) <\epsilon>
    ax1[2,2].plot(s.dissip_mean*s.z/s.ustar0/s.ustar0/s.ustar0, s.z/s.h, 
                  ls="-", c=colors[i], lw=2)
# clean up
# (a)
ax1[0,0].set_xlabel("$\\langle u_h \\rangle$ [m s$^{-1}$]")
ax1[0,0].set_ylabel("$z/h$")
ax1[0,0].set_xlim([0, 12.])
ax1[0,0].xaxis.set_major_locator(MultipleLocator(2))
ax1[0,0].xaxis.set_minor_locator(MultipleLocator(0.5))
ax1[0,0].set_ylim([0, 1.2])
ax1[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
ax1[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
# ax1[0,0].axvline(0., c="k", alpha=0.5)
ax1[0,0].text(0.87,0.05,r'\textbf{(a)}',fontsize=20,bbox=props, 
              transform=ax1[0,0].transAxes)
# (b)
ax1[0,1].set_xlabel("$\\langle \\alpha \\rangle$ [deg]")
ax1[0,1].set_xlim([220, 280.])
ax1[0,1].xaxis.set_major_locator(MultipleLocator(10))
ax1[0,1].xaxis.set_minor_locator(MultipleLocator(5))
ax1[0,1].axvline(270., c="k", alpha=0.5)
ax1[0,1].text(0.87,0.05,r'\textbf{(b)}',fontsize=20,bbox=props, 
              transform=ax1[0,1].transAxes)
ax1[0,1].legend(loc="upper left", labelspacing=0.10, 
                handletextpad=0.4, shadow=True)
# (c)
ax1[0,2].set_xlabel("$\\langle \\Theta \\rangle$ [K]")
ax1[0,2].set_xlim([240, 270.])
ax1[0,2].xaxis.set_major_locator(MultipleLocator(5))
ax1[0,2].xaxis.set_minor_locator(MultipleLocator(1))
ax1[0,2].text(0.87,0.05,r'\textbf{(c)}',fontsize=20,bbox=props, 
              transform=ax1[0,2].transAxes)
# (d)
ax1[1,0].set_xlabel("$u_*^2 / u_{*0}^2$")
ax1[1,0].set_ylabel("$z/h$")
ax1[1,0].set_xlim([0, 1.2])
ax1[1,0].xaxis.set_major_locator(MultipleLocator(0.2))
ax1[1,0].xaxis.set_minor_locator(MultipleLocator(0.05))
ax1[1,0].axvline(0., c="k", alpha=0.5)
ax1[1,0].text(0.03,0.05,r'\textbf{(d)}',fontsize=20,bbox=props, 
              transform=ax1[1,0].transAxes)
# (e)
ax1[1,1].set_xlabel("$\\langle \\theta'w' \\rangle / u_{*0} \\theta_{*0}$")
ax1[1,1].set_xlim([-1.2, 0])
ax1[1,1].xaxis.set_major_locator(MultipleLocator(0.2))
ax1[1,1].xaxis.set_minor_locator(MultipleLocator(0.05))
# ax1[1,1].axvline(0., c="k", alpha=0.5)
ax1[1,1].text(0.87,0.05,r'\textbf{(e)}',fontsize=20,bbox=props, 
              transform=ax1[1,1].transAxes)
# (f)
ax1[1,2].set_xlabel("$Ri_g$, $Ri_f$")
ax1[1,2].set_xlim([0, 1.2])
ax1[1,2].xaxis.set_major_locator(MultipleLocator(0.2))
ax1[1,2].xaxis.set_minor_locator(MultipleLocator(0.05))
# ax1[1,2].axvline(0., c="k", alpha=0.5)
ax1[1,2].text(0.87,0.05,r'\textbf{(f)}',fontsize=20,bbox=props, 
              transform=ax1[1,2].transAxes)
# (g)
ax1[2,0].set_xlabel("$\\langle u'^2 \\rangle / u_{*0}^2$")
ax1[2,0].set_ylabel("$z/h$")
ax1[2,0].set_xlim([0, 5])
ax1[2,0].xaxis.set_major_locator(MultipleLocator(1))
ax1[2,0].xaxis.set_minor_locator(MultipleLocator(0.25))
# ax1[2,0].axvline(0., c="k", alpha=0.5)
ax1[2,0].text(0.03,0.05,r'\textbf{(g)}',fontsize=20,bbox=props, 
              transform=ax1[2,0].transAxes)
# (h)
ax1[2,1].set_xlabel("$\\langle w'^2 \\rangle / u_{*0}^2$")
ax1[2,1].set_xlim([0, 1.8])
ax1[2,1].xaxis.set_major_locator(MultipleLocator(0.5))
ax1[2,1].xaxis.set_minor_locator(MultipleLocator(0.1))
# ax1[2,1].axvline(0., c="k", alpha=0.5)
ax1[2,1].text(0.87,0.05,r'\textbf{(h)}',fontsize=20,bbox=props, 
              transform=ax1[2,1].transAxes)
# (i)
ax1[2,2].set_xlabel("$-∏\\langle \\varepsilon \\rangle z/u_{*0}^3$ ")
ax1[2,2].set_xlim([-12, 0])
ax1[2,2].xaxis.set_major_locator(MultipleLocator(3))
ax1[2,2].xaxis.set_minor_locator(MultipleLocator(0.5))
# ax1[2,2].axvline(0., c="k", alpha=0.5)
ax1[2,2].text(0.03,0.05,r'\textbf{(i)}',fontsize=20,bbox=props, 
              transform=ax1[2,2].transAxes)
# add horizontal line at z/h = 1 for all
# tick lines inside plot
for iax in ax1.flatten():
    iax.axhline(1.0, c="k", alpha=0.5, ls="--")
    iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
# save and close
fig1.tight_layout()
fig1.savefig(f"{fdir_save}mean_prof_3x3_v3.pdf", format="pdf")
plt.close(fig1)

#
# Other experimental plots
#

# Figure 2: 2-panel TKE and ustar
# (a) TKE; (b) ustar
fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 6))
for i, s in enumerate(s_all):
    TKE = (s.u_var_rot**2.) + (s.v_var_rot**2.) + (s.w_var**2.)
    ax2[0].plot(TKE, s.z/s.h, ls="-", c=colors[i], lw=2, 
                label=f"{s.stability}")
    ax2[1].plot(s.ustar, s.z/s.h, ls="-", c=colors[i], lw=2)
# clean up
ax2[0].set_xlabel("TKE [m2/s2]")
ax2[0].set_ylabel("$z/h$")
ax2[0].set_ylim([0, 1.5])
ax2[0].grid()
ax2[0].legend()
ax2[1].set_xlabel("$u_{*}$ [m/s]")
ax2[1].grid()
# save and close
fig2.tight_layout()
fig2.savefig(f"{fdir_save}mean_tke_ustar.pdf", format="pdf")
plt.close(fig2)

# Figure 3: <u>, <v>, <u'w'>, <v'w'>, ustar2 versus z
fig3, ax3 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u, v
    ax3[0].plot(s.u_mean, s.z, ls="-", c=colors[i], lw=2)
    ax3[0].plot(s.v_mean, s.z, ls=":", c=colors[i], lw=2)
    # <u'w'>, <v'w'>
    ax3[1].plot(s.uw_cov_tot, s.z, ls="-", c=colors[i], lw=2)
    ax3[1].plot(s.vw_cov_tot, s.z, ls=":", c=colors[i], lw=2)
    # ustar2
    ax3[2].plot(s.ustar, s.z, ls="-", c=colors[i], lw=2, label=s.stability)
    # add horizontal lines for h based on ustar2
    for iax in ax3.flatten():
        iax.axhline(s.h, c=colors[i], ls="--")
# clean up
ax3[0].set_xlabel("$u$, $v$")
ax3[0].set_ylabel("$z$")
ax3[0].set_ylim([0, 400])
ax3[1].set_xlabel("$\\langle u'w' \\rangle$, $\\langle v'w' \\rangle$")
ax3[2].set_xlabel("$u_{*}^2$")
ax3[2].legend()
# save and close
fig3.tight_layout()
fig3.savefig(f"{fdir_save}h_test.png", dpi=300)
plt.close(fig3)

# Figure 4: Ri_g, Ri_f, Pr_t
fig4, ax4 = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
     # (a) Rig
    ax4[0].plot(s.Rig, s.z/s.h, ls="-", c=colors[i], lw=2, label=s.stability)
    # (b) Rif
    ax4[1].plot(s.Rif, s.z/s.h, ls="-", c=colors[i], lw=2)
    # (c) Pr
    ax4[2].plot(s.Rig/s.Rif, s.z/s.h, ls="-", c=colors[i], lw=2)
# clean up
ax4[0].legend(loc=0, labelspacing=0.10, handletextpad=0.4, shadow=True)
ax4[0].set_xlabel("$Ri_g$")
ax4[0].set_xlim([0, 1.2])
ax4[0].xaxis.set_major_locator(MultipleLocator(0.2))
ax4[0].xaxis.set_minor_locator(MultipleLocator(0.05))
ax4[1].set_xlabel("$Ri_f$")
ax4[2].set_xlabel("$Pr_t$")
ax4[0].set_ylim([0, 1.2])
ax4[0].yaxis.set_major_locator(MultipleLocator(0.2))
ax4[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax4[0].set_ylabel("$z/h$")
for iax in ax4.flatten():
    iax.axhline(1.0, c="k", alpha=0.5, ls="--")

# save and close
fig4.tight_layout()
fig4.savefig(f"{fdir_save}prandtl.png", dpi=300)
plt.close(fig4)
