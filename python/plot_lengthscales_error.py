# --------------------------------
# Name: plot_lengthscales_error.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 12 May 2021
# Purpose: plot output from random_errors_filter.py and integral_lengthscale.py
# Updated: 11 June 2021 - now decided on sim resolution and RFM method
# so create plotting routines to make it easier for plotting all params
# --------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from simulation import simulation

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/lengthscales/"
plt.close("all")

# useful plotting stuff
fstr = ["-k", "--k", ":k", ".-k", "-r", "--r", ":r", ".-r", "-b", "--b", ":b", ".-b"]
colors = [(225./255, 156./255, 131./255),
          (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
          (0., 85./255, 80./255), (20./255, 33./255, 61./255), (252./255, 193./255, 219./255)]
# --------------------------------
# Define plotting routines
# --------------------------------

def plot_int_len(sim, fsave=fdir_save):
    """
    3 separate figures: 
    1) u, v, theta 
    2) u'w', v'w', theta'w' 
    3) u'u', v'v', w'w', theta'theta'
    all of these live within sim.RFM
    """
    #
    # 1) u, v, theta
    #
    # create figure and axes handles
    fig1, ax1 = plt.subplots(nrows=2,ncols=3, sharey=True, figsize=(16, 12))
    # row 1: length scales
    # u length scale
    ax1[0,0].plot(s.RFM["len_u"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
    # v length scale
    ax1[0,1].plot(s.RFM["len_v"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
    # theta length scale
    ax1[0,2].plot(s.RFM["len_theta"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
    # row 2: time scales
    # u length scale
    ax1[1,0].plot(s.RFM["len_u"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                  color=colors[1], linestyle="-")
    # v length scale
    ax1[1,1].plot(s.RFM["len_v"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                  color=colors[1], linestyle="-")
    # theta length scale
    ax1[1,2].plot(s.RFM["len_theta"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                  color=colors[1], linestyle="-") 
    # also plot zj as horiz line
    for iax in ax1.ravel():
        iax.axhline(s.xytavg["zj"]/s.h, color=colors[1], linestyle="--", label="$z_{LLJ}$")

    # format figures
    # u length scale
    ax1[0,0].set_xlabel("$\mathcal{L}_u$ [m]")
    ax1[0,0].set_ylabel("$z/h$")
    ax1[0,0].grid()
#     ax1[0,0].legend()
    ax1[0,0].set_ylim([-0.05, 1.2])
    # v length scale
    ax1[0,1].set_xlabel("$\mathcal{L}_v$ [m]")
    ax1[0,1].grid()
    # theta length scale
    ax1[0,2].set_xlabel("$\mathcal{L}_{\\theta}$ [m]")
    ax1[0,2].grid()
    # u time scale
    ax1[1,0].set_xlabel("$\mathcal{T}_u$ [s]")
    ax1[1,0].set_ylabel("$z/h$")
    ax1[1,0].grid()
#     ax1[1,0].legend()
    # v time scale
    ax1[1,1].set_xlabel("$\mathcal{T}_v$ [s]")
    ax1[1,1].grid()
    # theta time scale
    ax1[1,2].set_xlabel("$\mathcal{T}_{\\theta}$ [s]")
    ax1[1,2].grid()

    # save figure
    fsave1 = f"{fsave}{s.stab}_u_v_theta_lengthscales.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
    plt.close(fig1)
    
    #
    # 2) u'w', v'w', theta'w'
    #
    # create figure and axes handles
    fig2, ax2 = plt.subplots(nrows=2,ncols=3, sharey=True, figsize=(16, 12))
    # row 1: length scales
    # u'w' length scale
    ax2[0,0].plot(s.RFM["len_uw"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
    # v'w' length scale
    ax2[0,1].plot(s.RFM["len_vw"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
    # theta'w' length scale
    ax2[0,2].plot(s.RFM["len_thetaw"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
    # row 2: time scales
    # u'w' length scale
    ax2[1,0].plot(s.RFM["len_uw"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                  color=colors[1], linestyle="-")
    # v length scale
    ax2[1,1].plot(s.RFM["len_vw"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                  color=colors[1], linestyle="-")
    # theta length scale
    ax2[1,2].plot(s.RFM["len_thetaw"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                  color=colors[1], linestyle="-") 
    # also plot zj as horiz line
    for iax in ax2.ravel():
        iax.axhline(s.xytavg["zj"]/s.h, color=colors[1], linestyle="--", label="$z_{LLJ}$")

    # format figures
    # u'w' length scale
    ax2[0,0].set_xlabel("$\mathcal{L}_{u'w'}$ [m]")
    ax2[0,0].set_ylabel("$z/h$")
    ax2[0,0].grid()
#     ax1[0,0].legend()
    ax2[0,0].set_ylim([-0.05, 1.2])
    # v'w' length scale
    ax2[0,1].set_xlabel("$\mathcal{L}_{v'w'}$ [m]")
    ax2[0,1].grid()
    # theta'w' length scale
    ax2[0,2].set_xlabel("$\mathcal{L}_{\\theta'w'}$ [m]")
    ax2[0,2].grid()
    # u'w' time scale
    ax2[1,0].set_xlabel("$\mathcal{T}_{u'w'}$ [m] [s]")
    ax2[1,0].set_ylabel("$z/h$")
    ax2[1,0].grid()
#     ax1[1,0].legend()
    # v'w' time scale
    ax2[1,1].set_xlabel("$\mathcal{T}_{u'w'}$ [m] [s]")
    ax2[1,1].grid()
    # theta'w' time scale
    ax2[1,2].set_xlabel("$\mathcal{T}_{\\theta'w'}$ [s]")
    ax2[1,2].grid()

    # save figure
    fsave2 = f"{fsave}{s.stab}_cov_lengthscales.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
    plt.close(fig2)


# --------------------------------
#
# Create simulation objects
#
# A
s128A = simulation("/home/bgreene/simulations/A_128_interp/output/",
                  128, 128, 128, 800., 800., 400., "A")
s160A = simulation("/home/bgreene/simulations/A_160_interp/output/",
                  160, 160, 160, 800., 800., 400., "A")
s192A = simulation("/home/bgreene/simulations/A_192_interp/output/",
                  192, 192, 192, 800., 800., 400., "A")
# F
s128F = simulation("/home/bgreene/simulations/F_128_interp/output/",
                  128, 128, 128, 800., 800., 400., "F")
s160F = simulation("/home/bgreene/simulations/F_160_interp/output/",
                  160, 160, 160, 800., 800., 400., "F")
s192F = simulation("/home/bgreene/simulations/F_192_interp/output/",
                  192, 192, 192, 800., 800., 400., "F")

# put everything into a list for looping
# s_all = [s128A, s160A, s192A]
# s_all = [s128F, s160F, s192F]
s_all = [s192F]
for s in s_all:
    s.read_csv()
    s.calc_Ri()
    s.calc_most()
    s.read_RFM(f"/home/bgreene/SBL_LES/output/RFM_{s.stab}{s.lab}.npz")
    
# --------------------------------
# Begin plotting
# --------------------------------

#
# Figure 1: sigma_u versus all delta_x for filtered
# Loop over s_all and create unique plot for each
#

for s in s_all:
    fig1, ax1 = plt.subplots(1, figsize=(12, 8))
    imax = int(np.sqrt(s.i_h))
    # loop over heights in s, up to maximum of imax
    for i, jz in enumerate(np.arange(imax, dtype=int)**2):
        ax1.plot(s.RFM["delta_x"], s.RFM["var_u"][:,jz]**0.5,
                 fstr[i], label=f"jz={jz}")
        # also plot Lo for reference
        # find closest value of sigma_u at given value of Lo to plot on curve
        i_dx = np.argmin([abs(s.Ri["Lo"][jz] - xx) for xx in s.RFM["delta_x"]])
        ax1.plot(s.Ri["Lo"][jz], s.RFM["var_u"][i_dx,jz]**0.5, "ok")
    # plot the last Lo again to get in legend (bc I'm lazy)
    ax1.plot(s.Ri["Lo"][jz], s.RFM["var_u"][i_dx,jz]**0.5, "ok",
             label="$L_o (jz)$")

    # plot -1/2 power law
    ax1.plot(s.RFM["delta_x"], s.RFM["delta_x"]**(-0.5), 
             lw=4, c="m", ls="-", label="$\Delta_x^{-1/2}$")
    # plot dx and Lx and annotate
    ax1.axvline(s.dx, c="r", lw=4)
    ax1.axvline(s.Lx, c="r", lw=4)
    ax1.annotate('$dx$', xy=(s.dx, 0.005), 
                 xytext=(s.dx, 0.005), rotation=270)
    ax1.annotate('$L_x$', xy=(s.Lx, 0.005), 
                 xytext=(s.Lx, 0.005), rotation=270)
    # clean up figure
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim([0.1, 1000])
    ax1.legend(loc="lower left")
    ax1.grid()
    ax1.set_xlabel("$\Delta_x$")
    ax1.set_ylabel("$\sigma_u$")
    ax1.set_title(f"{s.stab} {s.lab} u")
    # save and close
    fsave1 = f"{fdir_save}{s.stab}{s.lab}_sigma_u_filtered.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
    plt.close(fig1)

#
# Figure 2: sigma_theta versus all delta_x for filtered
# Loop over s_all and create unique plot for each
#

for s in s_all:
    fig2, ax2 = plt.subplots(1, figsize=(12, 8))
    imax = int(np.sqrt(s.i_h))
    # loop over heights in s, up to maximum of imax
    for i, jz in enumerate(np.arange(imax, dtype=int)**2):
        ax2.plot(s.RFM["delta_x"], s.RFM["var_theta"][:,jz]**0.5,
                 fstr[i], label=f"jz={jz}")
        # also plot Lo for reference
        # find closest value of sigma_u at given value of Lo to plot on curve
        i_dx = np.argmin([abs(s.Ri["Lo"][jz] - xx) for xx in s.RFM["delta_x"]])
        ax2.plot(s.Ri["Lo"][jz], s.RFM["var_theta"][i_dx,jz]**0.5, "ok")
    # plot the last Lo again to get in legend (bc I'm lazy)
    ax2.plot(s.Ri["Lo"][jz], s.RFM["var_theta"][i_dx,jz]**0.5, "ok",
             label="$L_o (jz)$")

    # plot -1/2 power law
    ax2.plot(s.RFM["delta_x"], s.RFM["delta_x"]**(-0.5), 
             lw=4, c="m", ls="-", label="$\Delta_x^{-1/2}$")
    # plot dx and Lx and annotate
    ax2.axvline(s.dx, c="r", lw=4)
    ax2.axvline(s.Lx, c="r", lw=4)
    ax2.annotate('$dx$', xy=(s.dx, 0.02), 
                 xytext=(s.dx, 0.02), rotation=270)
    ax2.annotate('$L_x$', xy=(s.Lx, 0.02), 
                 xytext=(s.Lx, 0.02), rotation=270)
    # clean up figure
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim([0.1, 1000])
    ax2.legend(loc="lower left")
    ax2.grid()
    ax2.set_xlabel("$\Delta_x$")
    ax2.set_ylabel("$\sigma_{\\theta}$")
    ax2.set_title(f"{s.stab} {s.lab} $\\theta$")
    # save and close
    fsave2 = f"{fdir_save}{s.stab}{s.lab}_sigma_theta_filtered.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
    plt.close(fig2)
    
#
# Figure 3: length and timescales for u and theta from autocorr
# only one figure per stability
#
# create figure and axes handles
fig3, ax3 = plt.subplots(nrows=2,ncols=2, sharey=True, figsize=(16, 12))
# loop through resolutions
for i, s in enumerate(s_all):
    # row 1: length scales
    # u length scale
    ax3[0,0].plot(s.RFM["len_u"], s.z[s.RFM["isbl"]]/s.h, color=colors[i], linestyle="-", label=s.lab)
    # theta length scale
    ax3[0,1].plot(s.RFM["len_theta"], s.z[s.RFM["isbl"]]/s.h, color=colors[i], linestyle="-", label=s.lab)
    # row 2: time scales
    # u length scale
    ax3[1,0].plot(s.RFM["len_u"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, color=colors[i], linestyle="-", label=s.lab)
    # theta length scale
    ax3[1,1].plot(s.RFM["len_theta"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, color=colors[i], linestyle="-", label=s.lab) 
    # also plot zj as horiz line
    for iax in ax3.ravel():
        iax.axhline(s.xytavg["zj"]/s.h, color=colors[i], linestyle="--", label="$z_{LLJ}$")
    
# format figures
# u length scale
ax3[0,0].set_xlabel("$u$ Integral Length Scale [m]")
ax3[0,0].set_ylabel("$z/h$")
ax3[0,0].grid()
ax3[0,0].legend()
ax3[0,0].set_ylim([-0.05, 1.2])
# theta length scale
ax3[0,1].set_xlabel("$\\theta$ Integral Length Scale [m]")
ax3[0,1].grid()
# u time scale
ax3[1,0].set_xlabel("$u$ Integral Time Scale [s]")
ax3[1,0].set_ylabel("$z/h$")
ax3[1,0].grid()
ax3[1,0].legend()
# theta time scale
ax3[1,1].set_xlabel("$\\theta$ Integral Time Scale [s]")
ax3[1,1].grid()

# save figure
fsave3 = f"{fdir_save}{s.stab}_u_theta_lengthscales.pdf"
print(f"Saving figure: {fsave3}")
fig3.savefig(fsave3, format="pdf", bbox_inches="tight")
plt.close(fig3)

#
# Figure 4: MSE{x~_delta}/var{x} vs. delta/T_H, x=u'w'
# Loop over s_all and create unique plot for each
#
for s in s_all:
    fig4, ax4 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8))
    imax = int(np.sqrt(s.i_h))
    # loop over heights in s, up to maximum of imax
    for i, jz in enumerate(np.arange(imax, dtype=int)**2):
        # plot versus delta_x/L_H
        ax4[0].plot(s.RFM["dx_LH_t"][:,jz], 
                    s.RFM["var_u"][:,jz]/s.var["u_var_tot"][jz],
                 fstr[i], label=f"jz={jz}")
        # plot versus delta_x/integral length scale
        ax4[1].plot(s.RFM["delta_x"]/s.RFM["len_u"][jz], 
                    s.RFM["var_u"][:,jz]/s.var["u_var_tot"][jz],
                    fstr[i])

    # clean up figure
    # ax4[0]
    ax4[0].set_xscale("log")
    ax4[0].set_yscale("log")
#         ax4.set_xlim([0.1, 1000])
    ax4[0].legend(loc="lower left")
    ax4[0].grid()
    ax4[0].set_xlabel("$\Delta_x / \mathcal{L}_H$")
    ax4[0].set_ylabel("$\sigma_{u}^2(\Delta_x) / Var\{u\}$")
    ax4[0].set_title(f"{s.stab} {s.lab} $u$")
    # ax4[1]
    ax4[1].set_xscale("log")
    ax4[1].grid()
    ax4[1].set_xlabel("$\Delta_x / \mathcal{L}_{u}$")
#     ax4[1].set_xlabel("$\Delta_t = \Delta_x / \\langle u \\rangle$")
    # save and close
    fsave4 = f"{fdir_save}{s.stab}{s.lab}_u_RFM.pdf"
    print(f"Saving figure: {fsave4}")
    fig4.savefig(fsave4, format="pdf", bbox_inches="tight")
    plt.close(fig4)
        
#
# Figure 5: compare rel rand err for u from RFM and autocorr
# Loop over s_all and create unique plot for each
#
for s in s_all:
    fig5, ax5 = plt.subplots(1, figsize=(12, 8))
    ax5.plot(s.RFM["err_u"]*100., s.RFM["z"][s.RFM["isbl"]], "-k", label="RFM")
    ax5.plot(s.RFM["err_u_LP"]*100., s.RFM["z"][s.RFM["isbl"]], "-r", label="autocorr")
    
    # clean up figure
    ax5.grid()
    ax5.legend()
    ax5.set_xlabel("$\epsilon_u$ [$\%$]")
    ax5.set_ylabel("$z$ [m]")
    ax5.set_title(f"Relative Random Error in u, {s.stab}{s.lab}, $T=3$s")
    # save figure
    fsave5 = f"{fdir_save}{s.stab}{s.lab}_u_err.pdf"
    print(f"Saving figure: {fsave5}")
    fig5.savefig(fsave5, format="pdf", bbox_inches="tight")
    plt.close(fig5)
    
#
# Figure 6: compare rel rand err for theta from RFM and autocorr
# Loop over s_all and create unique plot for each
#
for s in s_all:
    fig6, ax6 = plt.subplots(1, figsize=(12, 8))
    ax6.plot(s.RFM["err_theta"]*100., s.RFM["z"][s.RFM["isbl"]], "-k", label="RFM")
    ax6.plot(s.RFM["err_theta_LP"]*100., s.RFM["z"][s.RFM["isbl"]], "-r", label="autocorr")
    
    # clean up figure
    ax6.grid()
    ax6.legend()
    ax6.set_xlabel("$\epsilon_{\\theta}$ [$\%$]")
    ax6.set_ylabel("$z$ [m]")
    ax6.set_title(f"Relative Random Error in $\\theta$, {s.stab}{s.lab}, $T=3$s")
    # save figure
    fsave6 = f"{fdir_save}{s.stab}{s.lab}_theta_err.pdf"
    print(f"Saving figure: {fsave6}")
    fig6.savefig(fsave6, format="pdf", bbox_inches="tight")
    plt.close(fig6)

#
# Figure 7: compare rel rand err for u'w' from RFM and autocorr
# Loop over s_all and create unique plot for each
#
for s in s_all:
    fig7, ax7 = plt.subplots(1, figsize=(12, 8))
    ax7.plot(abs(s.RFM["err_uw"])*100., s.RFM["z"][s.RFM["isbl"]], "-k", label="RFM")
    ax7.plot(s.RFM["err_uw_LP"]*100., s.RFM["z"][s.RFM["isbl"]], "-r", label="autocorr")
    # plot llj and h
    ax7.axhline(s.h, linestyle="--", color="b", label="$h$")
    ax7.axhline(s.xytavg["zj"], linestyle=":", color="b", label="$z_{LLJ}$")
    
    # clean up figure
    ax7.grid()
    ax7.legend()
#     ax7.set_xlim([0, 500])
    ax7.set_xscale("log")
    ax7.set_xlabel("$\epsilon_{u'w'}$ [$\%$]")
    ax7.set_ylabel("$z$ [m]")
    ax7.set_title(f"Relative Random Error in $u'w'$, {s.stab}{s.lab}, $T=3$s")
    # save figure
    fsave7 = f"{fdir_save}{s.stab}{s.lab}_uw_err.pdf"
    print(f"Saving figure: {fsave7}")
    fig7.savefig(fsave7, format="pdf", bbox_inches="tight")
    plt.close(fig7)