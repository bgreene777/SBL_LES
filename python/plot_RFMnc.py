# --------------------------------
# Name: plot_RFMnc.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 8 November 2021
# Purpose: Read netcdf output from RFMnc.py to plot profiles of
# relative random errors and integral lengthscales for each
# simulation A-F
# --------------------------------
import os
import sys
import yaml
import numpy as np
import xarray as xr
import seaborn
from cmocean import cm
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, LogLocator
from dask.diagnostics import ProgressBar
#        
# Configure plots
#
rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
rc('text',usetex='True')
props=dict(boxstyle='square',facecolor='white',alpha=0.5)
colors = seaborn.color_palette("crest")
plt.close("all")

# --------------------------------
# plot_err_prof: loop through available processed sims and plot comparison profiles
# --------------------------------
def plot_err_prof():
    # load stat and error files
    # loop through directories A-F and see if err.nc exists; if so, append dataset to list
    stat = []
    err = []
    for stab in list("ABCDEF"):
        fdir = f"/home/bgreene/simulations/{stab}_192_interp/output/netcdf/"
        ferr = f"{fdir}err.nc"
        fstat = f"{fdir}average_statistics.nc"
        if os.path.exists(ferr):
            print(f"Loading file: {ferr}")
            err.append(xr.load_dataset(ferr))
            # also only load stat file if there is corresponding error file
            stat.append(xr.load_dataset(fstat))
    # loop through stat files and calculate important parameters
    for s in stat:
        # ustar
        s["ustar"] = ((s.uw_cov_tot ** 2.) + (s.vw_cov_tot ** 2.)) ** 0.25
        s["ustar2"] = s.ustar ** 2.
        # SBL height
        s["h"] = s.z.where(s.ustar2 <= 0.05*s.ustar2[0], drop=True)[0]/0.95
        # z indices within sbl
        s["isbl"] = np.where(s.z <= s.h)[0]
        s["nz_sbl"] = len(s.isbl)
        s["z_sbl"] = s.z.isel(z=s.isbl)
    #
    # Figure 1: 3-panel 1st order moments
    # wind speed, wind direction, potential temperature
    #
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, e in enumerate(err):
        # uh
        ax1[0].plot(100.*e.uh, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # alpha
        ax1[1].plot(100.*e.alpha, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # theta
        ax1[2].plot(100.*e.theta, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
    # labels
    ax1[0].set_xlabel("$\\epsilon_{u_h}$ [\%]")
    ax1[0].set_ylabel("$z/h$")
    ax1[0].set_xlim([0, 40])
    ax1[0].set_ylim([0, 1])
    ax1[0].legend()
    ax1[1].set_xlabel("$\\epsilon_{\\alpha}$ [\%]")
    ax1[1].set_xlim([0, 10])
    ax1[2].set_xlabel("$\\epsilon_{\\theta}$ [\%]")
    ax1[2].set_xlim([0, 0.15])
    fig1.tight_layout()
    # save and close
    fsave1 = f"{figdir}errors/uh_alpha_theta.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1)
    plt.close(fig1)
    
    #
    # Figure 2: 3-panel covariances
    # u'w', v'w', theta'w'
    #
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, e in enumerate(err):
        # u'w'
        ax2[0].plot(100.*e.uw_cov_tot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # v'w'
        ax2[1].plot(100.*e.vw_cov_tot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # theta'w'
        ax2[2].plot(100.*e.tw_cov_tot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
    # labels
    ax2[0].set_xlabel("$\\epsilon_{u'w'}$ [\%]")
    ax2[0].set_ylabel("$z/h$")
    ax2[0].set_xlim([0, 100])
    ax2[0].set_ylim([0, 1])
    ax2[0].legend()
    ax2[1].set_xlabel("$\\epsilon_{v'w'}$ [\%]")
    ax2[1].set_xlim([0, 100])
    ax2[2].set_xlabel("$\\epsilon_{\\theta'w'}$ [\%]")
    ax2[2].set_xlim([0, 100])
    fig2.tight_layout()
    # save and close
    fsave2 = f"{figdir}errors/covars.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2)
    plt.close(fig2)
    
    #
    # Figure 3: 4-panel variances
    # u'u' rotated, v'v' rotated, w'w', theta'theta'
    #
    fig3, ax3 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, e in enumerate(err):
        # u'u' rotated
        ax3[0].plot(100.*e.uu_var_rot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # v'v' rotated
        ax3[1].plot(100.*e.vv_var_rot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # w'w'
        ax3[2].plot(100.*e.ww_var, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # theta'theta'
        ax3[3].plot(100.*e.tt_var, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
    # labels
    ax3[0].set_xlabel("$\\epsilon_{u'u'}$ [\%]")
    ax3[0].set_ylabel("$z/h$")
    ax3[0].set_xlim([0, 25])
    ax3[0].set_ylim([0, 1])
    ax3[0].legend()
    ax3[1].set_xlabel("$\\epsilon_{v'v'}$ [\%]")
    ax3[1].set_xlim([0, 25])
    ax3[2].set_xlabel("$\\epsilon_{w'w'}$ [\%]")
    ax3[2].set_xlim([0, 25])
    ax3[3].set_xlabel("$\\epsilon_{\\theta'\\theta'}$ [\%]")
    ax3[3].set_xlim([0, 25])
    fig3.tight_layout()
    # save and close
    fsave3 = f"{figdir}errors/vars.pdf"
    print(f"Saving figure: {fsave3}")
    fig3.savefig(fsave3)
    plt.close(fig3)
    
    #
    # Figure 4: 3-panel u'w', v'w', ustar2
    #
    fig4, ax4 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, e in enumerate(err):
        # u'w'
        ax4[0].plot(100.*e.uw_cov_tot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # v'w'
        ax4[1].plot(100.*e.vw_cov_tot, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
        # ustar2
        ax4[2].plot(100.*e.ustar2, e.z/stat[i].h, c=colors[i], ls="-", lw=2, label=e.stability)
    # labels
    ax4[0].set_xlabel("$\\epsilon_{u'w'}$ [\%]")
    ax4[0].set_ylabel("$z/h$")
    ax4[0].set_xlim([0, 100])
    ax4[0].set_ylim([0, 1])
    ax4[0].legend()
    ax4[1].set_xlabel("$\\epsilon_{v'w'}$ [\%]")
    ax4[1].set_xlim([0, 100])
    ax4[2].set_xlabel("$\\epsilon_{u_{*}^2}$ [\%]")
    ax4[2].set_xlim([0, 100])
    fig4.tight_layout()
    # save and close
    fsave4 = f"{figdir}errors/ustar2.pdf"
    print(f"Saving figure: {fsave4}")
    fig4.savefig(fsave4)
    plt.close(fig4)
    
    return

# --------------------------------
# plot_L_prof: loop through available processed sims and plot comparison profiles
# --------------------------------
def plot_L_prof():
    # load stat and L files
    # loop through directories A-F and see if L.nc exists; if so, append dataset to list
    stat = []
    L = []
    for stab in list("ABCDEF"):
        fdir = f"/home/bgreene/simulations/{stab}_192_interp/output/netcdf/"
        fL = f"{fdir}L.nc"
        fstat = f"{fdir}average_statistics.nc"
        if os.path.exists(fL):
            print(f"Loading file: {fL}")
            L.append(xr.load_dataset(fL))
            # also only load stat file if there is corresponding error file
            stat.append(xr.load_dataset(fstat))
    # loop through stat files and calculate important parameters
    for s in stat:
        # ustar
        s["ustar"] = ((s.uw_cov_tot ** 2.) + (s.vw_cov_tot ** 2.)) ** 0.25
        s["ustar2"] = s.ustar ** 2.
        # SBL height
        s["h"] = s.z.where(s.ustar2 <= 0.05*s.ustar2[0], drop=True)[0]/0.95
        # z indices within sbl
        s["isbl"] = np.where(s.z <= s.h)[0]
        s["nz_sbl"] = len(s.isbl)
        s["z_sbl"] = s.z.isel(z=s.isbl)
    
    #
    # Figure 1: 3-panel 1st order moments
    # u_rot, v_rot, potential temperature
    #
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, iL in enumerate(L):
        # u_rot
        ax1[0].plot(iL.u_rot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # v_rot
        ax1[1].plot(iL.v_rot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # theta
        ax1[2].plot(iL.theta, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
    # labels
    ax1[0].set_xlabel("$\mathcal{L}_{u}$ [m]")
    ax1[0].set_ylabel("$z/h$")
    ax1[0].set_xlim([0, 40])
    ax1[0].set_ylim([0, 1])
    ax1[0].legend()
    ax1[1].set_xlabel("$\mathcal{L}_{v}$ [m]")
    ax1[1].set_xlim([0, 40])
    ax1[2].set_xlabel("$\mathcal{L}_{\\theta}$ [m]")
    ax1[2].set_xlim([0, 50])
    fig1.tight_layout()
    # save and close
    fsave1 = f"{figdir}integral_length/u_v_theta.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1)
    plt.close(fig1)
    
    #
    # Figure 2: 3-panel covariances
    # u'w', v'w', theta'w'
    #
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, iL in enumerate(L):
        # u'w'
        ax2[0].plot(iL.uw_cov_tot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # v'w'
        ax2[1].plot(iL.vw_cov_tot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # theta'w'
        ax2[2].plot(iL.tw_cov_tot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
    # labels
    ax2[0].set_xlabel("$\mathcal{L}_{u'w'}$ [m]")
    ax2[0].set_ylabel("$z/h$")
    ax2[0].set_xlim([0, 30])
    ax2[0].set_ylim([0, 1])
    ax2[0].legend()
    ax2[1].set_xlabel("$\mathcal{L}_{v'w'}$ [m]")
    ax2[1].set_xlim([0, 30])
    ax2[2].set_xlabel("$\mathcal{L}_{\\theta'w'}$ [m]")
    ax2[2].set_xlim([0, 30])
    fig2.tight_layout()
    # save and close
    fsave2 = f"{figdir}integral_length/covars.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2)
    plt.close(fig2)
    
    #
    # Figure 3: 4-panel variances
    # u'u' rotated, v'v' rotated, w'w', theta'theta'
    #
    fig3, ax3 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(14.8, 5))
    # loop through simulations
    for i, iL in enumerate(L):
        # u'u' rotated
        ax3[0].plot(iL.uu_var_rot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # v'v' rotated
        ax3[1].plot(iL.vv_var_rot, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # w'w'
        ax3[2].plot(iL.ww_var, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
        # theta'theta'
        ax3[3].plot(iL.tt_var, iL.z/stat[i].h, c=colors[i], ls="-", lw=2, label=iL.stability)
    # labels
    ax3[0].set_xlabel("$\mathcal{L}_{u'u'}$ [m]")
    ax3[0].set_ylabel("$z/h$")
    ax3[0].set_xlim([0, 30])
    ax3[0].set_ylim([0, 1])
    ax3[0].legend()
    ax3[1].set_xlabel("$\mathcal{L}_{v'v'}$ [m]")
    ax3[1].set_xlim([0, 30])
    ax3[2].set_xlabel("$\mathcal{L}_{w'w'}$ [m]")
    ax3[2].set_xlim([0, 15])
    ax3[3].set_xlabel("$\mathcal{L}_{\\theta'\\theta'}$ [m]")
    ax3[3].set_xlim([0, 30])
    fig3.tight_layout()
    # save and close
    fsave3 = f"{figdir}integral_length/vars.pdf"
    print(f"Saving figure: {fsave3}")
    fig3.savefig(fsave3)
    plt.close(fig3)
    
    return
    
# --------------------------------
# plot_2d_err: contours of relative random errors versus sample time for A & F
# --------------------------------
def plot_2d_err():
    # import recalc_err from RFMnc.py
    from RFMnc import recalc_err
    # construct Tnew from config file
    Tnew0 = config["recalc_lo"]
    Tnew1 = config["recalc_hi"]
    Tnewdt = config["recalc_dt"]
    Tnew = np.arange(Tnew0, Tnew1, Tnewdt, dtype=np.float64)
    #
    # Loop over stabilities, recalc error, and plot uh, alpha, theta
    #
    for stab in list("AF"):
        err = recalc_err(stab, Tnew)
        # plot
        # figure 1: uh
        fig1, ax1 = plt.subplots(1, figsize=(12, 8))
        # shade errors
        cfax1 = ax1.contourf(err.Tsample, err.z/err.h, 100*err.uh,
                             cmap=cm.matter, extend="max",
                             levels=np.linspace(0, 100, 21))
        # contour 10% level
        cax1 = ax1.contour(err.Tsample, err.z/err.h, 100.*err.uh,
                           "-k", levels=[10.], linewidths=4.)
        # colorbar
        cbar1 = fig1.colorbar(cfax1, ax=ax1, ticks=np.linspace(0, 100, 11))
        cbar1.ax.set_ylabel("$\\epsilon_{u_h}$ [$\%$]")
        # plot blue line outside of window to use in legend
        ax1.axhline(-10, ls="-", lw=4, c="k", label="$\\epsilon_{u_h}=10\%$")
        # plot vertical dashed line at T = 3 s
        ax1.axvline(3., ls="--", lw=4, c="k", label="$T = 3$ s")
        ax1.set_xlabel("Averaging Time [s]")
        ax1.set_ylabel("$z/h$")
        ax1.set_xlim([0, Tnew1])
        ax1.set_ylim([0.01, 1.0])
        ax1.set_yscale("log")
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.5)) 
#         ax1.yaxis.set_major_locator(MultipleLocator(0.1))
#         ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.legend(loc="upper right")
        # save and close
        fsave1 = f"{figdir}errors2d/{stab}_uh.pdf"
        print(f"Saving figure: {fsave1}")
        fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
        plt.close(fig1)
        
        # figure 2: alpha
        fig2, ax2 = plt.subplots(1, figsize=(12, 8))
        # shade errors
        cfax2 = ax2.contourf(err.Tsample, err.z/err.h, 100*err.alpha,
                             cmap=cm.matter, extend="max",
                             levels=np.linspace(0, 25, 26))
        # contour 2% level
        cax2 = ax2.contour(err.Tsample, err.z/err.h, 100.*err.alpha,
                           "-k", levels=[2.], linewidths=4.)
        # colorbar
        cbar2 = fig2.colorbar(cfax2, ax=ax2, ticks=np.linspace(0, 25, 6))
        cbar2.ax.set_ylabel("$\\epsilon_{\\alpha}$ [$\%$]")
        # plot blue line outside of window to use in legend
        ax2.axhline(-10, ls="-", lw=4, c="k", label="$\\epsilon_{\\alpha}=2\%$")
        # plot vertical dashed line at T = 3 s
        ax2.axvline(3., ls="--", lw=4, c="k", label="$T = 3$ s")
        ax2.set_xlabel("Averaging Time [s]")
        ax2.set_ylabel("$z/h$")
        ax2.set_xlim([0, Tnew1])
        ax2.set_ylim([0.01, 1.0])
        ax2.set_yscale("log")
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.5)) 
#         ax1.yaxis.set_major_locator(MultipleLocator(0.1))
#         ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.legend(loc="upper right")
        # save and close
        fsave2 = f"{figdir}errors2d/{stab}_alpha.pdf"
        print(f"Saving figure: {fsave2}")
        fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
        plt.close(fig2)
        
        # figure 3: theta
        fig3, ax3 = plt.subplots(1, figsize=(12, 8))
        # shade errors
        cfax3 = ax3.contourf(err.Tsample, err.z/err.h, 100*err.theta,
                             cmap=cm.matter, extend="max",
                             levels=np.linspace(0, 0.5, 26))
        # contour 2% level
       #  cax2 = ax2.contour(err.Tsample, err.z/err.h, 100.*err.alpha,
#                            "-k", levels=[2.], linewidths=4.)
        # colorbar
        cbar3 = fig3.colorbar(cfax3, ax=ax3, ticks=np.linspace(0, 0.5, 11))
        cbar3.ax.set_ylabel("$\\epsilon_{\\theta}$ [$\%$]")
        # plot blue line outside of window to use in legend
#         ax2.axhline(-10, ls="-", lw=4, c="k", label="$\\epsilon_{\\alpha}=2\%$")
        # plot vertical dashed line at T = 3 s
        ax3.axvline(3., ls="--", lw=4, c="k", label="$T = 3$ s")
        ax3.set_xlabel("Averaging Time [s]")
        ax3.set_ylabel("$z/h$")
        ax3.set_xlim([0, Tnew1])
        ax3.set_ylim([0.01, 1.0])
        ax3.set_yscale("log")
        ax3.xaxis.set_major_locator(MultipleLocator(1))
        ax3.xaxis.set_minor_locator(MultipleLocator(0.5)) 
#         ax1.yaxis.set_major_locator(MultipleLocator(0.1))
#         ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax3.legend(loc="upper right")
        # save and close
        fsave3 = f"{figdir}errors2d/{stab}_theta.pdf"
        print(f"Saving figure: {fsave3}")
        fig3.savefig(fsave3, format="pdf", bbox_inches="tight")
        plt.close(fig3)
        
    return
# --------------------------------
# Run script
# --------------------------------
if __name__ == "__main__":
    # load yaml file in global scope
    with open("/home/bgreene/SBL_LES/python/RFMnc.yaml") as f:
        config = yaml.safe_load(f)
    # only thing we care about is where to save figures
    figdir = config["figdir"]
#     plot_err_prof()
#     plot_L_prof()
    plot_2d_err()
    