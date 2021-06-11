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

def plot_int_len(slist, fsave=fdir_save):
    """
    3 separate figures: 
    1) u, v, theta 
    2) u'w', v'w', theta'w' 
    3) u'u', v'v', w'w', theta'theta'
    all of these live within sim.RFM
    loop over s in slist to compare on same figures
    """
    #
    # 1) u, v, theta
    #
    # create figure and axes handles
    fig1, ax1 = plt.subplots(nrows=2,ncols=3, sharey=True, figsize=(16, 12))
    for s in slist:
        # row 1: length scales
        # u length scale
        ax1[0,0].plot(s.RFM["len_u"], s.z[s.RFM["isbl"]]/s.h, 
                      color=colors[1], linestyle="-", label=s.lab)
        # v length scale
        ax1[0,1].plot(s.RFM["len_v"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
        # theta length scale
        ax1[0,2].plot(s.RFM["len_theta"], s.z[s.RFM["isbl"]]/s.h, color=colors[1], linestyle="-")
        # row 2: time scales
        # u length scale
        ax1[1,0].plot(s.RFM["len_u"]/s.xytavg["ws"][s.RFM["isbl"]], s.z[s.RFM["isbl"]]/s.h, 
                      color=colors[1], linestyle="-", label=s.lab)
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
    ax1[0,0].legend()
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
    ax1[1,0].legend()
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
    
    return
# --------------------------------

def plot_sigma_filt(s, fsave=fdir_save):
    """
    6 total figures to loop over:
    u, v, theta, u'w', v'w', theta'w'
    """
    # first determine plotting parameters
    kplot = []
    for key in s.RFM.keys():
        if key[:4] == "var_":
            kplot.append(key)
    splot = ["$\sigma_u$", "$\sigma_v$", "$\sigma_{\\theta}$", 
             "$\sigma_{u'w'}$", "$\sigma_{v'w'}$", "$\sigma_{\\theta'w'}$"]
    # begin looping over parameters
    for ivar, var in enumerate(kplot):
        svar = var.split("_")[-1]
        fig1, ax1 = plt.subplots(1, figsize=(12, 8))
        imax = int(np.sqrt(s.i_h))
        # loop over heights in s, up to maximum of imax
        for i, jz in enumerate(np.arange(imax, dtype=int)**2):
            ax1.plot(s.RFM["delta_x"], s.RFM[var][:,jz]**0.5,
                     fstr[i], label=f"jz={jz}")
            # also plot Lo for reference
            # find closest value of sigma_u at given value of Lo to plot on curve
            i_dx = np.argmin([abs(s.Ri["Lo"][jz] - xx) for xx in s.RFM["delta_x"]])
            ax1.plot(s.Ri["Lo"][jz], s.RFM[var][i_dx,jz]**0.5, "ok")
        # plot the last Lo again to get in legend (bc I'm lazy)
        ax1.plot(s.Ri["Lo"][jz], s.RFM[var][i_dx,jz]**0.5, "ok",
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
        ax1.set_ylabel(splot[ivar])
        ax1.set_title(f"{s.stab} {s.lab} {svar}")
        # save and close
        fsave1 = f"{fsave}{s.stab}{s.lab}_sigma_{svar}_filtered.pdf"
        print(f"Saving figure: {fsave1}")
        fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
        plt.close(fig1)
    return
# --------------------------------

def plot_err(s, fsave=fdir_save):
    """
    6 total figures to loop over:
    u, v, theta, u'w', v'w', theta'w'
    """
    vplot = ["u", "v", "\\theta", "u'w'", "v'w'", "\\theta'w'"]
    for ivar, var in enumerate(["u", "v", "theta", "uw", "vw", "tw"]):
        # initialize figure
        fig1, ax1 = plt.subplots(1, figsize=(12, 8))
        # plot RFM errors
        ax1.plot(abs(s.RFM[f"err_{var}"])*100., s.RFM["z"][s.RFM["isbl"]], 
                 "-k", label="RFM")
        # plot autocorr errors
        if var != "v":
            ax1.plot(abs(s.RFM[f"err_{var}_LP"])*100., s.RFM["z"][s.RFM["isbl"]], 
                     "-r", label="autocorr")

        # clean up figure
        ax1.grid()
        ax1.legend()
        if var in ["uw", "vw"]:
            ax1.set_xscale("log")
        ax1.set_xlabel(f"$\epsilon_{{{vplot[ivar]}}}$ [$\%$]")
        ax1.set_ylabel("$z$ [m]")
        T_sample = s.RFM["yaml"][()]["T_sample"]
        ax1.set_title(f"Relative Random Error in ${{{vplot[ivar]}}}$, {s.stab}{s.lab}, $T={{{T_sample}}}$ s")
        # save figure
        fsave1 = f"{fsave}{s.stab}{s.lab}_{var}_err.pdf"
        print(f"Saving figure: {fsave1}")
        fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
        plt.close(fig1)
    
    return
# --------------------------------

def plot_MSE(s, fsave=fdir_save):
    """
    6 total figures to loop over:
    u, v, theta, u'w', v'w', theta'w'
    """
    vplot = ["u", "v", "\\theta", "u'w'", "v'w'", "\\theta'w'"]
    variance = {"u": s.var["u_var_tot"],
                "v": s.var["v_var_tot"],
                "t": s.var["theta_var_tot"],
                "uw": s.RFM["uwuw_var_xytavg"],
                "vw": s.RFM["vwvw_var_xytavg"],
                "tw": s.RFM["twtw_var_xytavg"]}
    
    for ivar, var in enumerate(["u", "v", "t", "uw", "vw", "tw"]):
        # annoying bit of code to keep t/theta consistent in var names
        if var == "t":
            var2 = "theta"
            var3 = "theta"
        elif var == "tw":
            var2 = var
            var3 = "thetaw"
        else:
            var2 = var
            var3 = var
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8))
        imax = int(np.sqrt(s.i_h))
        # loop over heights in s, up to maximum of imax
        for i, jz in enumerate(np.arange(imax, dtype=int)**2):
            # plot versus delta_x/L_H
            ax1[0].plot(s.RFM[f"dx_LH_{var}"][:,jz], 
                        s.RFM[f"var_{var2}"][:,jz]/variance[var][jz],
                     fstr[i], label=f"jz={jz}")
            # plot versus delta_x/integral length scale
            ax1[1].plot(s.RFM["delta_x"]/s.RFM[f"len_{var3}"][jz], 
                        s.RFM[f"var_{var2}"][:,jz]/variance[var][jz],
                        fstr[i])

        # clean up figure
        ax1[0].set_xscale("log")
        ax1[0].set_yscale("log")
    #         ax4.set_xlim([0.1, 1000])
        ax1[0].legend(loc="lower left")
        ax1[0].grid()
        ax1[0].set_xlabel("$\Delta_x / \mathcal{L}_H$")
        ax1[0].set_ylabel(f"$\sigma_{{{vplot[ivar]}}}^2(\\Delta_x) / Var {{{vplot[ivar]}}} $")
        ax1[0].set_title(f"{s.stab} {s.lab} ${{{vplot[ivar]}}}$")
        # ax4[1]
        ax1[1].set_xscale("log")
        ax1[1].grid()
        ax1[1].set_xlabel(f"$\Delta_x / \mathcal{{L}}_{{{vplot[ivar]}}}$")
    #     ax4[1].set_xlabel("$\Delta_t = \Delta_x / \\langle u \\rangle$")
        # save and close
        fsave1 = f"{fsave}{s.stab}{s.lab}_{var}_RFM.pdf"
        print(f"Saving figure: {fsave1}")
        fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
        plt.close(fig1)    

    return

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
s_all = [s192A]
for s in s_all:
    s.read_csv()
    s.calc_Ri()
    s.calc_most()
    s.read_RFM(f"/home/bgreene/SBL_LES/output/RFM_{s.stab}{s.lab}.npz")
    
# --------------------------------
# Begin plotting
# --------------------------------

for s in s_all:
    # sigma_f versus all delta_x for filtered
    plot_sigma_filt(s)
    # MSE{x~_delta}/var{x} vs. delta/T_H
    plot_MSE(s)
    # compare rel rand err from RFM and autocorr
    plot_err(s)
    
# length and timescales from autocorr
plot_int_len(s_all)
