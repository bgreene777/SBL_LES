# --------------------------------
# Name: plot_random_errors.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 6 August 2021
# Purpose: plot random error profiles to compare across simulations/stabilities
# --------------------------------
import os
import pickle
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, LogLocator
from datetime import datetime, timedelta
from simulation import simulation

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
rc('text',usetex='True')
props=dict(boxstyle='square',facecolor='white',alpha=0.5)
# colors = [(252./255, 193./255, 219./255), (225./255, 156./255, 131./255),
#           (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
#           (0., 85./255, 80./255), (20./255, 33./255, 61./255) ]
colors = seaborn.color_palette("crest")
fdir_save = "/home/bgreene/SBL_LES/figures/random_errors/Mean_LH/"
plt.close("all")

# --------------------------------
# initialize simulation objects
# A
sA = simulation("/home/bgreene/simulations/A_192_interp/output/",
                192, 192, 192, 800., 800., 400., "A")
# # B
# sB = simulation("/home/bgreene/simulations/B_192_interp/output/",
#                 192, 192, 192, 800., 800., 400., "B")
# # C
# sC = simulation("/home/bgreene/simulations/C_192_interp/output/",
#                 192, 192, 192, 800., 800., 400., "C")
# # D
# sD = simulation("/home/bgreene/simulations/D_192_interp/output/",
#                 192, 192, 192, 800., 800., 400., "D")
# # E
# sE = simulation("/home/bgreene/simulations/E_192_interp/output/",
#                 192, 192, 192, 800., 800., 400., "E")
# # F
# sF = simulation("/home/bgreene/simulations/F_192_interp/output/",
#                 192, 192, 192, 800., 800., 400., "F")

# put everything into a list for looping
# s_all = [sA, sB, sC, sD, sE, sF]
s_all = [sA]
for s in s_all:
    s.read_csv()
    s.calc_Ri()
    s.calc_most()
    s.read_RFM(f"/home/bgreene/SBL_LES/output/Mean_LH/RFM_{s.stab}{s.lab}.npz", ierr_ws=True)
    s.print_sim_stats()
    
# --------------------------------
# Begin Plotting
# --------------------------------

#
# Figure 1
# streamwise u errors
#
fig1, ax1 = plt.subplots(1, figsize=(6, 8))
# loop over all sims
for i, s in enumerate(s_all):
    ax1.plot(100.*s.RFM["err_u"], s.z[s.isbl]/s.h, 
             ls="-", lw=2, c=colors[i], label=s.stab)
# labels
# ax1.set_xlim()
ax1.set_ylim([0., 1.])
ax1.grid()
ax1.legend()
ax1.set_xlabel("$\\epsilon_u$ [$\%$]")
ax1.set_ylabel("$z/h$")
# save and close
fsave1 = f"{fdir_save}err_u.pdf"
print(f"Saving figure: {fsave1}")
fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
plt.close(fig1)

#
# Figure 2
# ws, wd, theta errors
#
fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
# loop over all sims
for i, s in enumerate(s_all):
    ax2[0].plot(100.*s.RFM["err_ws"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax2[1].plot(100.*s.RFM["err_wd"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax2[2].plot(100.*s.RFM["err_theta"], s.z[s.isbl]/s.h, 
             ls="-", lw=2, c=colors[i], label=s.stab)
# labels
# ax2.set_xlim()
ax2[0].set_ylim([0., 1.])
# ax2[0].grid()
ax2[0].legend(loc=0, labelspacing=0.25, 
              handletextpad=0.4, shadow=True)
ax2[0].set_xlabel("$\\epsilon_{u_h}$ [$\%$]")
ax2[0].set_xlim([0, 26])
ax2[0].xaxis.set_major_locator(MultipleLocator(10))
ax2[0].xaxis.set_minor_locator(MultipleLocator(2))
ax2[0].set_ylabel("$z/h$")
ax2[0].set_ylim([0, 1])
ax2[0].yaxis.set_major_locator(MultipleLocator(0.2))
ax2[0].yaxis.set_minor_locator(MultipleLocator(0.1))
ax2[0].text(0.05,0.95,r'\textbf{(a)}',fontsize=20,bbox=props, 
              transform=ax2[0].transAxes)
# ax2[1].grid()
ax2[1].set_xlabel("$\\epsilon_{\\alpha}$ [$\%$]")
ax2[1].set_xlim([0, 7])
ax2[1].xaxis.set_major_locator(MultipleLocator(2))
ax2[1].xaxis.set_minor_locator(MultipleLocator(0.5))
ax2[1].text(0.05,0.95,r'\textbf{(b)}',fontsize=20,bbox=props, 
              transform=ax2[1].transAxes)
# ax2[2].grid()
ax2[2].set_xlabel("$\\epsilon_{\\theta}$ [$\%$]")
ax2[2].set_xlim([0, 0.16])
ax2[2].xaxis.set_major_locator(MultipleLocator(0.04))
ax2[2].xaxis.set_minor_locator(MultipleLocator(0.01))
ax2[2].text(0.05,0.95,r'\textbf{(c)}',fontsize=20,bbox=props, 
              transform=ax2[2].transAxes)
# save and close
fsave2 = f"{fdir_save}err_ws_wd_theta.pdf"
print(f"Saving figure: {fsave2}")
fig2.tight_layout()
fig2.savefig(fsave2, format="pdf")
plt.close(fig2)

#
# Figure 3
# theta errors
#
fig3, ax3 = plt.subplots(1, figsize=(6, 8))
# loop over all sims
for i, s in enumerate(s_all):
    ax3.plot(100.*s.RFM["err_theta"], s.z[s.isbl]/s.h, 
             ls="-", lw=2, c=colors[i], label=s.stab)
# labels
# ax3.set_xlim()
ax3.set_ylim([0., 1.])
ax3.grid()
ax3.legend()
ax3.set_xlabel("$\\epsilon_{\\theta}$ [$\%$]")
ax3.set_ylabel("$z/h$")
# save and close
fsave3 = f"{fdir_save}err_theta.pdf"
print(f"Saving figure: {fsave3}")
fig3.savefig(fsave3, format="pdf", bbox_inches="tight")
plt.close(fig3)

#
# Figure 4
# covariance errors: u'w', v'w', theta'w'
#
fig4, ax4 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
# loop over all sims
for i, s in enumerate(s_all):
    ax4[0].plot(100.*s.RFM["err_uw"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax4[1].plot(100.*s.RFM["err_vw"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax4[2].plot(100.*s.RFM["err_tw"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
#     for iax in ax4:
#         iax.axhline(s.xytavg["zj"]/s.h, ls=":", c=colors[i])
    
# labels
ax4[0].set_xlim([1., 5000.])
ax4[0].set_ylim([0., 1.])
# ax4[0].grid()
ax4[0].legend(loc=0, labelspacing=0.25, handletextpad=0.4, shadow=True)
ax4[0].set_xscale("log")
ax4[0].xaxis.set_major_locator(LogLocator(base=10, numticks=10, 
                                          subs=np.linspace(1,10,10)))
ax4[0].set_xlabel("$\\epsilon_{\overline{u'w'}}$ [$\%$]")
ax4[0].set_ylabel("$z/h$")
ax4[0].text(0.05,0.95,r'\textbf{(a)}',fontsize=20,bbox=props, 
              transform=ax4[0].transAxes)
ax4[1].set_xlim([1., 20000.])
# ax4[1].grid()
ax4[1].set_xscale("log")
ax4[1].set_xlabel("$\\epsilon_{\overline{v'w'}}$ [$\%$]")
ax4[1].xaxis.set_major_locator(LogLocator(base=10, numticks=15,
                                          subs=np.linspace(1,10,10)))
ax4[1].text(0.05,0.95,r'\textbf{(b)}',fontsize=20,bbox=props, 
              transform=ax4[1].transAxes)
ax4[2].set_xlim([1., 200.])
# ax4[2].grid()
ax4[2].set_xscale("log")
ax4[2].set_xlabel("$\\epsilon_{\overline{\\theta'w'}}$ [$\%$]")
ax4[2].xaxis.set_major_locator(LogLocator(base=10, numticks=10, 
                                          subs=np.linspace(1,10,10)))
ax4[2].text(0.05,0.95,r'\textbf{(c)}',fontsize=20,bbox=props, 
              transform=ax4[2].transAxes)
# save and close
fsave4 = f"{fdir_save}err_covars.pdf"
print(f"Saving figure: {fsave4}")
fig4.tight_layout()
fig4.savefig(fsave4, format="pdf")
plt.close(fig4)

#
# Figure 5
# variance errors: u'u', v'v', w'w', theta'theta'
#
fig5, ax5 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 8))
# loop over all sims
for i, s in enumerate(s_all):
    ax5[0].plot(100.*s.RFM["err_uu"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax5[1].plot(100.*s.RFM["err_vv"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax5[2].plot(100.*s.RFM["err_ww"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
    ax5[3].plot(100.*s.RFM["err_tt"], s.z[s.isbl]/s.h, 
                ls="-", lw=2, c=colors[i], label=s.stab)
#     for iax in ax5:
#         iax.axhline(s.xytavg["zj"]/s.h, ls=":", c=colors[i])
    
# labels
# ax5[0].set_xlim([1., 5000.])
ax5[0].set_ylim([0., 1.])
# ax5[0].grid()
ax5[0].legend(loc=0, labelspacing=0.25, handletextpad=0.4, shadow=True)
ax5[0].set_xlabel("$\\epsilon_{\overline{u'u'}}$ [$\%$]")
ax5[0].set_ylabel("$z/h$")
ax5[0].set_xlim([0, 50])
ax5[0].xaxis.set_major_locator(MultipleLocator(10))
ax5[0].xaxis.set_minor_locator(MultipleLocator(2))
ax5[0].text(0.05,0.95,r'\textbf{(a)}',fontsize=20,bbox=props, 
              transform=ax5[0].transAxes)
ax5[1].set_xlim([0, 40.])
# ax5[1].grid()
ax5[1].set_xlabel("$\\epsilon_{\overline{v'v'}}$ [$\%$]")
ax5[1].xaxis.set_major_locator(MultipleLocator(10))
ax5[1].xaxis.set_minor_locator(MultipleLocator(2))
ax5[1].text(0.05,0.95,r'\textbf{(b)}',fontsize=20,bbox=props, 
              transform=ax5[1].transAxes)
ax5[2].set_xlim([0, 40])
# ax5[2].grid()
ax5[2].set_xlabel("$\\epsilon_{\overline{w'w'}}$ [$\%$]")
ax5[2].xaxis.set_major_locator(MultipleLocator(10))
ax5[2].xaxis.set_minor_locator(MultipleLocator(2))
ax5[2].text(0.05,0.95,r'\textbf{(c)}',fontsize=20,bbox=props, 
              transform=ax5[2].transAxes)
# ax5[3].grid()
ax5[3].set_xlabel("$\\epsilon_{\overline{\\theta'\\theta'}}$ [$\%$]")
ax5[3].set_xlim([0, 50])
ax5[3].xaxis.set_major_locator(MultipleLocator(10))
ax5[3].xaxis.set_minor_locator(MultipleLocator(2))
ax5[3].text(0.05,0.95,r'\textbf{(d)}',fontsize=20,bbox=props, 
              transform=ax5[3].transAxes)
# save and close
fsave5 = f"{fdir_save}err_vars.pdf"
print(f"Saving figure: {fsave5}")
fig5.tight_layout()
fig5.savefig(fsave5, format="pdf")
plt.close(fig5)

#
# Figure 6, 7
# L_H for each of the variance profiles
# L_H for mean variables
#
fig6, ax6 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 8))
fig7, ax7 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
for i, s in enumerate(s_all):
    L_H_u = s.RFM["delta_x"][0] / s.RFM["dx_LH_u"][0]
    L_H_v = s.RFM["delta_x"][0] / s.RFM["dx_LH_v"][0]
    L_H_t = s.RFM["delta_x"][0] / s.RFM["dx_LH_t"][0]
    L_H_uu = s.RFM["delta_x"][0] / s.RFM["dx_LH_uu"][0]
    L_H_vv = s.RFM["delta_x"][0] / s.RFM["dx_LH_vv"][0]
    L_H_ww = s.RFM["delta_x"][0] / s.RFM["dx_LH_ww"][0]
    L_H_tt = s.RFM["delta_x"][0] / s.RFM["dx_LH_tt"][0]
    # fig 6
    ax6[0].plot(L_H_uu, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i], label=s.stab)
    ax6[1].plot(L_H_vv, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i])
    ax6[2].plot(L_H_ww, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i])
    ax6[3].plot(L_H_tt, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i])
    # fig 7
    ax7[0].plot(L_H_u, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i], label=s.stab)
    ax7[1].plot(L_H_v, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i])
    ax7[2].plot(L_H_t, s.z[s.isbl]/s.h, ls="-", lw=2, c=colors[i],
                marker="x", markeredgecolor=colors[i])

# labels
ax6[0].set_xlabel("$\mathcal{L}_H^{uu}$ [m]")
ax6[0].set_ylabel("$z/h$")
ax6[0].set_ylim([0, 1])
ax6[0].legend(loc=0)
ax6[1].set_xlabel("$\mathcal{L}_H^{vv}$ [m]")
ax6[2].set_xlabel("$\mathcal{L}_H^{ww}$ [m]")
ax6[3].set_xlabel("$\mathcal{L}_H^{\\theta\\theta}$ [m]")

ax7[0].set_xlabel("$\mathcal{L}_H^{u}$ [m]")
ax7[0].set_ylabel("$z/h$")
ax7[0].set_ylim([0, 1])
ax7[0].legend(loc=0)
ax7[1].set_xlabel("$\mathcal{L}_H^{v}$ [m]")
ax7[2].set_xlabel("$\mathcal{L}_H^{\\theta}$ [m]")

# save and close
fsave6 = f"{fdir_save}LH_vars.pdf"
print(f"Saving figure: {fsave6}")
fig6.tight_layout()
fig6.savefig(fsave6, format="pdf")
plt.close(fig6)

fsave7 = f"{fdir_save}LH_uvtheta.pdf"
print(f"Saving figure: {fsave7}")
fig7.tight_layout()
fig7.savefig(fsave7, format="pdf")
plt.close(fig7)