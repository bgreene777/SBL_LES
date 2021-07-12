# --------------------------------
# Name: UAS.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 03 June 2021
# Purpose: load npz files from timeseries output to emulate UAS profiles
# and plot output
# --------------------------------
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from simulation import *

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/UAS/"
plt.close("all")

#
# Create simulation objects
#
# A
sA = UAS_emulator("/home/bgreene/simulations/A_192_interp/output/",
                192, 192, 192, 800., 800., 400., "A")
# F
sF = UAS_emulator("/home/bgreene/simulations/F_192_interp/output/",
                192, 192, 192, 800., 800., 400., "F")

# combine in list for looping
s_all = [sA, sF]
for s in s_all:
    s.read_timeseries(720000, 0.02, raw=False)
    s.profile(ascent_rate=1.0, time_average=3.0, time_start=0.0)
    s.calc_ec(time_average=1800., time_start=0.0)
    
# plot uas versus mean (loop over all)
for s in s_all:
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
    # ws
    ax1[0].plot(s.xytavg["ws"], s.z/s.h, "-k", label="$\\langle ws \\rangle$")
    ax1[0].plot(s.prof["ws"], s.prof["z"]/s.h, "-r", label="UAS")
    # shade errors
    err_hi_ws = (1. + s.RFM["err_ws_interp"]) * s.prof["ws"][s.prof["isbl"]]
    err_lo_ws = (1. - s.RFM["err_ws_interp"]) * s.prof["ws"][s.prof["isbl"]]
    ax1[0].fill_betweenx(s.prof["z"][s.prof["isbl"]]/s.h, err_lo_ws, err_hi_ws, 
                         alpha=0.3, color="r", label="$\\epsilon_{ws}$")
    # wd
    ax1[1].plot(s.xytavg["wd"], s.z/s.h, "-k", label="$\\langle wd \\rangle$")
    ax1[1].plot(s.prof["wd"], s.prof["z"]/s.h, "-r", label="UAS")
    # shade errors
    err_hi_wd = (1. + s.RFM["err_wd_interp"]) * s.prof["wd"][s.prof["isbl"]]
    err_lo_wd = (1. - s.RFM["err_wd_interp"]) * s.prof["wd"][s.prof["isbl"]]
    ax1[1].fill_betweenx(s.prof["z"][s.prof["isbl"]]/s.h, err_lo_wd, err_hi_wd, 
                         alpha=0.3, color="r", label="$\\epsilon_{wd}$")
    # theta
    ax1[2].plot(s.xytavg["theta"], s.z/s.h, "-k", label="$\\langle \\theta \\rangle$")
    ax1[2].plot(s.prof["theta"], s.prof["z"]/s.h, "-r", label="UAS")
    # shade errors
    err_hi_theta = (1. + s.RFM["err_theta_interp"]) * s.prof["theta"][s.prof["isbl"]]
    err_lo_theta = (1. - s.RFM["err_theta_interp"]) * s.prof["theta"][s.prof["isbl"]]
    ax1[2].fill_betweenx(s.prof["z"][s.prof["isbl"]]/s.h, err_lo_theta, err_hi_theta, 
                         alpha=0.3, color="r", label="$\\epsilon_{\\theta}$")

    # clean up
    for iax in ax1:
        iax.grid()
        iax.legend(loc="upper left")
    ax1[0].set_xlabel("Wind Speed [m/s]")
    ax1[0].set_ylabel("$z/h$")
    ax1[0].set_ylim([0, 1.2])
    ax1[0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax1[0].yaxis.set_minor_locator(MultipleLocator(0.05))    
    ax1[0].set_xlim([0, 11])
    ax1[0].xaxis.set_major_locator(MultipleLocator(2.))
    ax1[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1[1].set_xlabel("Wind Direction [$^{\circ}$]")
    ax1[1].set_xlim([210., 285.])
    ax1[1].xaxis.set_major_locator(MultipleLocator(30.))
    ax1[1].xaxis.set_minor_locator(MultipleLocator(5.))    
    ax1[2].set_xlabel("$\\theta$ [K]")
    # calc min max for theta
    itheta = np.where(s.z/s.h <= 1.2)[0]
    Tmin, Tmax, Timaj, Timin = min_max_plot(s.xytavg["theta"][itheta], 5.)
    ax1[2].set_xlim([Tmin, Tmax])
    ax1[2].xaxis.set_major_locator(MultipleLocator(Timaj))
    ax1[2].xaxis.set_minor_locator(MultipleLocator(Timin))
    fig1.suptitle(f"Simulation {s.stab} Winds and Theta")
    # save figure
    fsave1 = f"{fdir_save}{s.stab}_ws_wd_theta.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1, format="pdf", bbox_inches="tight")
    plt.close(fig1)

# plot raw uas versus mean (loop over all)
for s in s_all:
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
    # ws
    ax2[0].plot(s.xytavg["ws"], s.z, "-k", label="Mean LES")
    ax2[0].plot(s.raw_prof["ws"], s.raw_prof["z"], "-r", label="Raw UAS")
    ax2[0].grid()
    ax2[0].legend(loc="upper left")
    ax2[0].set_xlabel("Wind Speed [m/s]")
    ax2[0].set_ylabel("$z$ [m]")
    # wd
    ax2[1].plot(s.xytavg["wd"], s.z, "-k", label="Mean LES")
    ax2[1].plot(s.raw_prof["wd"], s.raw_prof["z"], "-r", label="Raw UAS")
    ax2[1].grid()
    ax2[1].set_xlabel("Wind Direction [$^{\circ}$]")
    # theta
    ax2[2].plot(s.xytavg["theta"], s.z, "-k", label="Mean LES")
    ax2[2].plot(s.raw_prof["theta"], s.raw_prof["z"], "-r", label="Raw UAS")
    ax2[2].grid()
    ax2[2].set_xlabel("$\\theta$ [K]")
    fig2.suptitle(f"Simulation {s.stab} Raw Profile")
    # save figure
    fsave2 = f"{fdir_save}{s.stab}_raw_ws_wd_theta.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2, format="pdf", bbox_inches="tight")
    plt.close(fig2)
    
# plot virtual tower EC covariances versus mean (loop over all)
for s in s_all:
    fig3, ax3 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
    # u'w'
    ax3[0].plot(s.cov["uw_cov_tot"], s.z/s.h, "-k", label="$\\langle u'w' \\rangle$")
    ax3[0].plot(s.ec["uw_cov_tot"], s.z/s.h, "-r", label="UAS")
    # shade errors
    err_hi_uw = (1. + s.RFM["err_uw"]) * s.ec["uw_cov_tot"][s.RFM["isbl"]]
    err_lo_uw = (1. - s.RFM["err_uw"]) * s.ec["uw_cov_tot"][s.RFM["isbl"]]
    ax3[0].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_uw, err_hi_uw, 
                         alpha=0.3, color="r", label="$\\epsilon_{u'w'}$")
    # v'w'
    ax3[1].plot(s.cov["vw_cov_tot"], s.z/s.h, "-k", label="$\\langle v'w' \\rangle$")
    ax3[1].plot(s.ec["vw_cov_tot"], s.z/s.h, "-r", label="UAS")
    # shade errors
    err_hi_vw = (1. + s.RFM["err_vw"]) * s.ec["vw_cov_tot"][s.RFM["isbl"]]
    err_lo_vw = (1. - s.RFM["err_vw"]) * s.ec["vw_cov_tot"][s.RFM["isbl"]]
    ax3[1].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_vw, err_hi_vw, 
                         alpha=0.3, color="r", label="$\\epsilon_{v'w'}$")
    # theta'w'
    ax3[2].plot(s.cov["thetaw_cov_tot"], s.z/s.h, "-k", label="$\\langle \\theta'w' \\rangle$")
    ax3[2].plot(s.ec["thetaw_cov_tot"], s.z/s.h, "-r", label="UAS")
    # shade errors
    err_hi_thetaw = (1. + s.RFM["err_tw"]) * s.ec["thetaw_cov_tot"][s.RFM["isbl"]]
    err_lo_thetaw = (1. - s.RFM["err_tw"]) * s.ec["thetaw_cov_tot"][s.RFM["isbl"]]
    ax3[2].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_thetaw, err_hi_thetaw, 
                         alpha=0.3, color="r", label="$\\epsilon_{\\theta'w'}$")

    # clean up
    for iax in ax3:
        iax.grid()
        iax.legend(loc="upper left")
    ax3[0].set_ylim([0, 1])
    ax3[0].set_xlabel("$u'w'$ [m$^2$ s$^{-2}$]")
    ax3[0].set_ylabel("$z/h$")
    ax3[1].set_xlabel("$v'w'$ [m$^2$ s$^{-2}$]")
    ax3[2].set_xlabel("$\\theta'w'$ [K m s$^{-1}$]")
    fig3.suptitle(f"Simulation {s.stab} Covariances")
    # save figure
    fsave3 = f"{fdir_save}{s.stab}_covar.pdf"
    print(f"Saving figure: {fsave3}")
    fig3.savefig(fsave3, format="pdf", bbox_inches="tight")
    plt.close(fig3)
    
# plot virtual tower EC variances versus mean (loop over all)
for s in s_all:
    fig4, ax4 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 8))
    # u'u'
    ax4[0].plot(s.var["u_var_tot"], s.z/s.h, "-k", label="$\\langle u'^2 \\rangle$")
    ax4[0].plot(s.ec["u_var"], s.z/s.h, "-r", label="EC")
    # shade errors
    err_hi_uu = (1. + s.RFM["err_uu"]) * s.ec["u_var"][s.RFM["isbl"]]
    err_lo_uu = (1. - s.RFM["err_uu"]) * s.ec["u_var"][s.RFM["isbl"]]
    ax4[0].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_uu, err_hi_uu, 
                         alpha=0.3, color="r", label="$\\epsilon_{u'^2}$")
    # v'v'
    ax4[1].plot(s.var["v_var_tot"], s.z/s.h, "-k", label="$\\langle v'^2 \\rangle$")
    ax4[1].plot(s.ec["v_var"], s.z/s.h, "-r", label="EC")
    # shade errors
    err_hi_vv = (1. + s.RFM["err_vv"]) * s.ec["v_var"][s.RFM["isbl"]]
    err_lo_vv = (1. - s.RFM["err_vv"]) * s.ec["v_var"][s.RFM["isbl"]]
    ax4[1].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_vv, err_hi_vv, 
                         alpha=0.3, color="r", label="$\\epsilon_{v'^2}$")
    # w'w'
    ax4[2].plot(s.var["w_var_tot"], s.z/s.h, "-k", label="$\\langle w'^2 \\rangle$")
    ax4[2].plot(s.ec["w_var"], s.z/s.h, "-r", label="EC")
    # shade errors
    err_hi_ww = (1. + s.RFM["err_ww"]) * s.ec["w_var"][s.RFM["isbl"]]
    err_lo_ww = (1. - s.RFM["err_ww"]) * s.ec["w_var"][s.RFM["isbl"]]
    ax4[2].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_ww, err_hi_ww, 
                         alpha=0.3, color="r", label="$\\epsilon_{w'^2}$")
    # theta'theta'
    ax4[3].plot(s.var["theta_var_tot"], s.z/s.h, "-k", label="$\\langle \\theta'^2 \\rangle$")
    ax4[3].plot(s.ec["theta_var"], s.z/s.h, "-r", label="EC")
    # shade errors
    err_hi_tt = (1. + s.RFM["err_tt"]) * s.ec["theta_var"][s.RFM["isbl"]]
    err_lo_tt = (1. - s.RFM["err_tt"]) * s.ec["theta_var"][s.RFM["isbl"]]
    ax4[3].fill_betweenx(s.z[s.RFM["isbl"]]/s.h, err_lo_tt, err_hi_tt, 
                         alpha=0.3, color="r", label="$\\epsilon_{\\theta'^2}$")

    # clean up
    for iax in ax4:
        iax.grid()
        iax.legend(loc="upper right")
    ax4[0].set_ylim([0, 1])
    ax4[0].set_xlabel("$u'^2$ [m$^2$ s$^{-2}$]")
    ax4[0].set_ylabel("$z/h$")
    ax4[1].set_xlabel("$v'^2$ [m$^2$ s$^{-2}$]")
    ax4[2].set_xlabel("$w'^2$ [m$^2$ s$^{-2}$]")
    ax4[3].set_xlabel("$\\theta'^2$ [K$^2$]")
    fig4.suptitle(f"Simulation {s.stab} Variances")
    # save figure
    fsave4 = f"{fdir_save}{s.stab}_var.pdf"
    print(f"Saving figure: {fsave4}")
    fig4.savefig(fsave4, format="pdf", bbox_inches="tight")
    plt.close(fig4)
    