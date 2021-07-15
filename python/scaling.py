# --------------------------------
# Name: scaling.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 15 July 2021
# Purpose: plot gradient-based SBL scales proposed by Sorbjan (2010) and
# expanded upon in Sorbjan (2017)
# --------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import cmocean
from simulation import simulation

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/scaling/"
plt.close("all")

# useful plotting stuff
fstr = ["-k", "--k", ":k", ".-k", "-r", "--r", ":r", ".-r", "-b", "--b", ":b", ".-b"]
colors = [(225./255, 156./255, 131./255),
          (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
          (0., 85./255, 80./255), (20./255, 33./255, 61./255), (252./255, 193./255, 219./255)]

#
# Create simulation objects
#
# A
s192A = simulation("/home/bgreene/simulations/A_192_interp/output/",
                  192, 192, 192, 800., 800., 400., "A")
# F
s192F = simulation("/home/bgreene/simulations/F_192_interp/output/",
                  192, 192, 192, 800., 800., 400., "F")

# put everything into a list for looping
# s_all = [s128A, s160A, s192A]
# s_all = [s128F, s160F, s192F]
s_all = [s192A, s192F]
for s in s_all:
    s.read_csv()
    s.calc_Ri()
    s.calc_most()
    
#
# Figure 1: Ls as function of z
#
for s in s_all:
    fig1, ax1 = plt.subplots(1, figsize=(6, 8))
    ax1.plot(s.most["Ls"], s.z[1:-1], "-k")
    # labels
    ax1.set_xlabel("$L_s$ [m]")
    ax1.set_ylabel("$z$ [m]")
    ax1.grid()
    # save and close
    fsave1 = f"{fdir_save}{s.stab}{s.lab}_Ls.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
    plt.close(fig1)
    
#
# Figure 2: 4-panel gradient-based scales versus Ri
# Loop over all sims and plot on same axes to increase range of Ri
#
fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
# loop over sims
for i, s in enumerate(s_all):
    # only plot within sbl
    isbl = s.isbl
    # Gm
    ax2[0,0].plot(s.Ri["Ri"][isbl], s.most["Gm"][isbl], ls="", marker="x",
                  markeredgecolor=colors[i], alpha=0.75, label=s.stab)
    # Gh
    ax2[0,1].plot(s.Ri["Ri"][isbl], s.most["Gh"][isbl], ls="", marker="x",
                  markeredgecolor=colors[i], alpha=0.75, label=s.stab)
    # Gw
    ax2[1,0].plot(s.Ri["Ri"][isbl], s.most["Gw"][isbl], ls="", marker="x",
                  markeredgecolor=colors[i], alpha=0.75, label=s.stab)
    # Gtheta
    ax2[1,1].plot(s.Ri["Ri"][isbl], s.most["Gtheta"][isbl], ls="", marker="x",
                  markeredgecolor=colors[i], alpha=0.75, label=s.stab)

# plot empirical curves
n_bins = 21
Ri_bins_plot = np.logspace(-3, 1, num=n_bins)
# Gm_e
Gm_e = 1. / (Ri_bins_plot * ((1. + 300.*(Ri_bins_plot**2.))**1.5))
ax2[0,0].plot(Ri_bins_plot, Gm_e, linestyle='-', color='k')
# Gh_e
Gh_e = 1. / (0.9 * (Ri_bins_plot**0.5) * ((1. + 250.*(Ri_bins_plot**2.))**1.5))
ax2[0,1].plot(Ri_bins_plot, Gh_e, linestyle='-', color='k')
# Gw_e
Gw_e = 1. / (0.85*(Ri_bins_plot**0.5)*((1. + 450.*(Ri_bins_plot**2.))**0.5))
ax2[1,0].plot(Ri_bins_plot, Gw_e, linestyle='-', color='k')
# Gtheta_e
Gtheta_e = 5. / ((1. + 2500.*(Ri_bins_plot**2.))**0.5)
ax2[1,1].plot(Ri_bins_plot, Gtheta_e, linestyle='-', color='k')
Gtheta_e2 = 3. / ((1. + 1000.*(Ri_bins_plot**2.))**0.5)
ax2[1,1].plot(Ri_bins_plot, Gtheta_e2, linestyle='--', color='k')

# labels
ax2[0,0].set_xscale("log")
ax2[0,0].set_yscale("log")
ax2[0,0].set_xlabel("Ri", fontsize=30)
ax2[0,0].set_ylabel("$G_m = \\overline{u'w'} / U_s^2$", fontsize=30, labelpad=0.7)
ax2[0,0].set_xlim([0.001, 1])
ax2[0,0].set_ylim([0.005, 1000])
ax2[0,0].set_xticks([0.001, 0.01, 0.1, 1.0])
ax2[0,0].grid(which="major", axis="both")
ax2[0,0].tick_params(labelsize=30)
props=dict(boxstyle="square", facecolor="white", alpha=0.85)
ax2[0,0].text(0.05, 0.96, r"\textbf{(a)}", fontsize=20, bbox=props, 
              transform=ax2[0,0].transAxes, ha='center', va='center')
ax2[0,0].legend(loc=0, fontsize=20, shadow=True)

ax2[0,1].set_xscale("log")
ax2[0,1].set_yscale("log")
ax2[0,1].set_xlabel("Ri", fontsize=30)
ax2[0,1].set_ylabel("$G_h = -\\overline{\\theta'w'} / U_s T_s$", fontsize=30, labelpad=0.7)
ax2[0,1].set_xlim([0.001, 1])
ax2[0,1].set_ylim([0.005, 1000])
ax2[0,1].set_xticks([0.001, 0.01, 0.1, 1.0])
ax2[0,1].grid(which="major", axis="both")
# ax3[0,1].legend(loc=0, fontsize=20)
ax2[0,1].tick_params(labelsize=30)
ax2[0,1].text(0.05, 0.96, r"\textbf{(b)}", fontsize=20, bbox=props, 
              transform=ax2[0,1].transAxes, ha='center', va='center')
              
ax2[1,0].set_xscale("log")
ax2[1,0].set_yscale("log")
ax2[1,0].set_xlabel("Ri", fontsize=30)
ax2[1,0].set_ylabel("$G_w = \sigma_w / U_s$", fontsize=30, labelpad=0.7)
ax2[1,0].set_xlim([0.001, 1])
ax2[1,0].set_ylim([0.1, 100])
ax2[1,0].set_xticks([0.001, 0.01, 0.1, 1.0])
ax2[1,0].grid(which="major", axis="both")
ax2[1,0].tick_params(labelsize=30)
ax2[1,0].text(0.05, 0.96, r"\textbf{(c)}", fontsize=20, bbox=props, 
              transform=ax2[1,0].transAxes, ha='center', va='center')

ax2[1,1].set_xscale("log")
ax2[1,1].set_yscale("log")
ax2[1,1].set_xlabel("Ri", fontsize=30)
ax2[1,1].set_ylabel("$G_\\theta = \sigma_\\theta / T_s$", fontsize=30, labelpad=0.7)
ax2[1,1].set_xlim([0.001, 1])
ax2[1,1].set_ylim([0.1, 100])
ax2[1,1].set_xticks([0.001, 0.01, 0.1, 1.0])
ax2[1,1].grid(which="major", axis="both")
ax2[1,1].tick_params(labelsize=30)
ax2[1,1].text(0.05, 0.96, r"\textbf{(d)}", fontsize=20, bbox=props, 
              transform=ax2[1,1].transAxes, ha='center', va='center')

# save and close
fsave2 = f"{fdir_save}master_gradient_scaling_grid.pdf"
print(f"Saving figure: {fsave2}")
fig2.tight_layout()
fig2.savefig(fsave2, format="pdf")
plt.close(fig2)



