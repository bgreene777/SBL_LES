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
# --------------------------------
import os
import pickle
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from datetime import datetime, timedelta
from simulation import simulation

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')
# colors = [(225./255, 156./255, 131./255),
#           (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
#           (0., 85./255, 80./255), (20./255, 33./255, 61./255), (252./255, 193./255, 219./255)]
colors = seaborn.color_palette("crest")
fdir_save = "/home/bgreene/SBL_LES/figures/grid_sensitivity/"
plt.close("all")

# --------------------------------
# initialize simulation objects
# A
# s128A = simulation("/home/bgreene/simulations/A_128_interp/output/average_statistics.csv",
#                   128, 128, 128, 800., 800., 400., "A")
# s160A = simulation("/home/bgreene/simulations/A_160_interp/output/average_statistics.csv",
#                   160, 160, 160, 800., 800., 400., "A")
sA = simulation("/home/bgreene/simulations/A_192_interp/output/",
                192, 192, 192, 800., 800., 400., "A")
# B
sB = simulation("/home/bgreene/simulations/B_192_interp/output/",
                192, 192, 192, 800., 800., 400., "B")
# C
sC = simulation("/home/bgreene/simulations/C_192_interp/output/",
                192, 192, 192, 800., 800., 400., "C")
# D
sD = simulation("/home/bgreene/simulations/D_192_interp/output/",
                192, 192, 192, 800., 800., 400., "D")
# E
sE = simulation("/home/bgreene/simulations/E_192_interp/output/",
                192, 192, 192, 800., 800., 400., "E")
# F
# s128F = simulation("/home/bgreene/simulations/F_128_interp/output/average_statistics.csv",
#                   128, 128, 128, 800., 800., 400., "F")
# s160F = simulation("/home/bgreene/simulations/F_160_interp/output/average_statistics.csv",
#                   160, 160, 160, 800., 800., 400., "F")
sF = simulation("/home/bgreene/simulations/F_192_interp/output/",
                192, 192, 192, 800., 800., 400., "F")

# put everything into a list for looping
# s_all = [s128A, s160A, s192A]
s_all = [sA, sB, sC, sD, sE, sF]
for s in s_all:
    s.read_csv()
    s.calc_Ri()
    s.calc_most()
#     # save as pickle files
#     with open(f"/home/bgreene/SBL_LES/output/pickle/{s.stab}{s.lab}.pickle", "wb") as fn:
#         pickle.dump(s, fn)
    
# --------------------------------
# Begin plotting
# --------------------------------

#
# Figure 1: unrotated u, v, wspd; wdir; theta
#
fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u
    ax1[0].plot(s.xytavg["u"], s.z, color=colors[i], linestyle="--", lw=2,)
    # v
    ax1[0].plot(s.xytavg["v"], s.z, color=colors[i], linestyle=":", lw=2,)
    # ws
    ax1[0].plot(s.xytavg["ws"], s.z, color=colors[i], linestyle="-", lw=2, label=s.stab)
    # wdir
    ax1[1].plot(s.xytavg["wd"], s.z, color=colors[i], linestyle="-", lw=2,)
    # theta
    ax1[2].plot(s.xytavg["theta"], s.z, color=colors[i], linestyle="-")
    ax1[2].axhline(s.h, color=colors[i], linestyle=":", linewidth=2, 
                   label=f"$h = {s.h:4.1f}$ m")
# clean up
ax1[0].grid()
ax1[0].legend()
ax1[0].set_xlabel(r"Wind Speed, $\langle u \rangle$, $\langle v \rangle$ [m s$^{-1}$]")
ax1[0].set_ylabel("$z$ [m]")

ax1[1].grid()
ax1[1].set_xlabel(r"Wind Direction [$^{\circ}$]")

ax1[2].grid()
ax1[2].legend()
ax1[2].set_xlabel(r"$\langle \theta \rangle$ [K]")

# save figure
fig1.savefig(f"{fdir_save}all_u_v_theta.pdf", format="pdf", bbox_inches="tight")
plt.close(fig1)

#
# Figure 2: unrotated <u'w'>, <v'w'>, <theta'w'>
#
fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u'w'
    ax2[0].plot(s.cov["uw_cov_tot"], s.z/s.h, color=colors[i], linestyle="-", label=s.stab, lw=2)
    # v'w'
    ax2[1].plot(s.cov["vw_cov_tot"], s.z/s.h, color=colors[i], linestyle="-", lw=2)
    # theta'w'
    ax2[2].plot(s.cov["thetaw_cov_tot"], s.z/s.h, color=colors[i], linestyle="-", lw=2)
# clean up
ax2[0].grid()
ax2[0].legend()
ax2[0].set_xlabel(r"$\langle u'w' \rangle$ [m$^2$ s$^{-2}$]")
ax2[0].set_ylabel("$z/h$")
ax2[0].set_ylim([-0.05, 1.2])

ax2[1].grid()
ax2[1].set_xlabel(r"$\langle v'w' \rangle$ [m$^2$ s$^{-2}$]")

ax2[2].grid()
ax2[2].set_xlabel(r"$\langle \theta'w' \rangle$ [K m s$^{-1}$]")

# save figure
fig2.savefig(f"{fdir_save}all_covars.pdf", format="pdf", bbox_inches="tight")
plt.close(fig2)

#
# Figure 3: *rotated* <u'u'>, <v'v'>, <w'w'>
#
fig3, ax3 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u'u'
    ax3[0].plot(s.var["u_var_tot"], s.z/s.h, color=colors[i], linestyle="-", label=s.stab, lw=2)
    # v'v'
    ax3[1].plot(s.var["v_var_tot"], s.z/s.h, color=colors[i], linestyle="-", lw=2)
    # theta'w'
    ax3[2].plot(s.var["w_var_tot"], s.z/s.h, color=colors[i], linestyle="-", lw=2)
# clean up
ax3[0].grid()
ax3[0].legend()
ax3[0].set_xlabel(r"$\langle u'^2 \rangle$ [m$^2$ s$^{-2}$]")
ax3[0].set_ylabel("$z/h$")
ax3[0].set_ylim([-0.05, 1.2])

ax3[1].grid()
ax3[1].set_xlabel(r"$\langle v'^2 \rangle$ [m$^2$ s$^{-2}$]")

ax3[2].grid()
ax3[2].set_xlabel(r"$\langle w'^2 \rangle$ [m$^2$ s$^{-2}$]")

# save figure
fig3.savefig(f"{fdir_save}all_vars.pdf", format="pdf", bbox_inches="tight")
plt.close(fig3)

#
# Figure 4: TKE, <theta'theta'>, ustar
#
fig4, ax4 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # TKE
    ax4[0].plot(s.var["TKE_tot"]/s.cov["ustar"][0]/s.cov["ustar"][0], s.z/s.h, color=colors[i], 
                linestyle="-", label=s.stab, lw=2)
    # theta var
    ax4[1].plot(s.var["theta_var_tot"], s.z/s.h, color=colors[i], linestyle="-", lw=2)
    # ustar
    ax4[2].plot(s.cov["ustar"], s.z/s.h, color=colors[i], linestyle="-", lw=2)
    
# clean up
ax4[0].grid()
ax4[0].legend()
ax4[0].set_xlabel(r"TKE / $u_{*}^2$")
ax4[0].set_ylabel("$z/h$")
ax4[0].set_ylim([-0.05, 1.2])

ax4[1].grid()
ax4[1].set_xlabel(r"$\langle \theta'^2 \rangle$ [K$^2$]")

ax4[2].grid()
ax4[2].set_xlabel(r"$u_{*}$ [m s$^{-1}$]")

# save figure
fig4.savefig(f"{fdir_save}all_tke.pdf", format="pdf", bbox_inches="tight")
plt.close(fig4)


#
# Figure 5: S2 and N2; Ri
#
fig5, ax5 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # S2
    ax5[0].plot(s.Ri["S2"], s.z[1:-1]/s.h, color=colors[i], linestyle="-", label=s.stab, lw=2)
    # N2
    ax5[0].plot(s.Ri["N2"], s.z[1:-1]/s.h, color=colors[i], linestyle=":", lw=2)
    # Ri
    ax5[1].plot(s.Ri["Ri"], s.z[1:-1]/s.h, color=colors[i], linestyle="-", lw=2)
    # Ri_f
    ax5[1].plot(s.Ri["Ri_f"], s.z[1:-1]/s.h, color=colors[i], linestyle=":", lw=2)
# clean up
ax5[0].grid()
ax5[0].legend()
ax5[0].set_xlabel(r"$S^2, N^2$ [s$^{-2}$]")
ax5[0].set_ylabel("$z/h$")
ax5[0].set_ylim([0, 1.2])
# ax5[0].set_xscale("log")
# ax5[0].set_xlim([1e-4, 1e-1])

ax5[1].grid()
ax5[1].set_xlabel(r"$Ri_b, Ri_f$")
ax5[1].set_xlim([-0.1, 5])

# save figure
fig5.savefig(f"{fdir_save}all_N2_S2_Ri.pdf", format="pdf", bbox_inches="tight")
plt.close(fig5)

#
# Figure 6: TKE Budget terms: one per stability
#
for s in s_all:
    fig6, ax6 = plt.subplots(1, figsize=(12,8))

    ax6.plot(s.tke["shear"][1:]/s.tke["scale"], s.tke["z"][1:], label="Shear Production", lw=2)
    ax6.plot(s.tke["buoy"][1:]/s.tke["scale"], s.tke["z"][1:], label="Buoyancy Production", lw=2)
    ax6.plot(s.tke["trans"][1:]/s.tke["scale"], s.tke["z"][1:], label="Turbulent Transport", lw=2)
    ax6.plot(s.tke["diss"][1:]/s.tke["scale"], s.tke["z"][1:], label="3D Dissipation", lw=2)
    ax6.plot(s.tke["residual"][1:]/s.tke["scale"], s.tke["z"][1:], label="Residual", lw=2)
    ax6.axhline(s.h, color="k", linestyle="--", label="h", lw=2)
    ax6.axhline(s.xytavg["zj"], color="k", linestyle=":", label="LLJ", lw=2)
    ax6.grid()
    ax6.legend(loc="upper right")
    ax6.set_xlabel("Dimensionless TKE Budget Terms [-]")
    ax6.set_ylabel("z [m]")
    ax6.set_title("TKE Budget (z-direction)")
    ax6.set_ylim([0., 200.])
    ax6.set_xlim([-5., 5.])

    # save figure
    fig6.savefig(f"{fdir_save}{s.stab}{s.lab}_tke_budget.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig6)

#
# Figure 7: Ozmidov length scale Lo
#
fig7, ax7 = plt.subplots(1, figsize=(8, 6))
for i, s in enumerate(s_all):
    ax7.plot(s.Ri["Lo"], s.z[1:-1], "-", label=s.stab, c=colors[i], lw=2)
#     ax7.axhline(s.xytavg["zj"], linestyle=":", color=colors[i], label=f"LLJ$^{{{s.nx}}}$")
    ax7.axhline(s.h, linestyle=":", color=colors[i], lw=2)
ax7.axvline(s.dd, linestyle="--", color=colors[i], label=f"$\Delta$", lw=2)
ax7.grid()
ax7.legend()
ax7.set_ylabel("z [m]")
ax7.set_xlabel(r"$L_o = \sqrt{ \langle \epsilon \rangle / \langle N^2 \rangle ^{3/2} }$ [m]")

# save figure
fig7.savefig(f"{fdir_save}all_Lo.pdf", format="pdf", bbox_inches="tight")