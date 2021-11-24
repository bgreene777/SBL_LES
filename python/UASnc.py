# --------------------------------
# Name: UASnc.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 22 November 2021
# Purpose: load netcdf files from timeseries_to_netcdf.py to emulate UAS 
# profiles and plot output with random errors from RFMnc.py
# For now only look at A and F
# --------------------------------
import os
import numpy as np
import xarray as xr
from numba import njit
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
# --------------------------------
# Define Functions
# --------------------------------
@njit
def interp_uas(dat, z_LES, z_UAS):
    """Interpolate LES virtual tower timeseries data in the vertical to match
    ascent rate of emulated UAS to create timeseries of ascent data
    :param float dat: 2d array of the field to interpolate, shape(nt, nz)
    :param float z_LES: 1d array of LES grid point heights
    :param float z_UAS: 1d array of new UAS z from ascent & sampling rates
    Outputs 2d array of interpolated field, shape(nt, len(z_UAS))
    """
    nt = np.shape(dat)[0]
    nz = len(z_UAS)
    dat_interp = np.zeros((nt, nz), dtype=np.float64)
    for i in range(nt):
        dat_interp[i,:] = np.interp(z_UAS, z_LES, dat[i,:])
    return dat_interp
# --------------------------------
def profile(df, err, ascent_rate=1.0, time_average=3.0, time_start=0.0, quicklook=False):
    """Emulate a vertical profile from rotary-wing UAS sampling through
    simulated SBL, including random errors
    :param xr.Dataset df: dataset with virtual tower data to construct UAS prof
    :param xr.Dataset err: dataset with relative random errors corr. to df
    :param float ascent_rate: UAS ascent rate in m/s; default=1.0
    :param float time_average: time range in s to avg UAS profile; default=3.0 s
    :param float time_start: when to initialize ascending profile; default=0.0 s
    :param bool quicklook: flag to make quicklook of raw vs averaged profiles
    Outputs new xarray Dataset with emulated profile and errors
    """
    # First, calculate array of theoretical altitudes based on the base time
    # vector and ascent_rate while keeping account for time_start
    zuas = ascent_rate * df.t.values
    # find the index in df.time that corresponds to time_start
    istart = int(time_start / df.dt)
    # set zuas[:istart] = 0 and then subtract everything after that
    zuas[:istart] = 0
    zuas[istart:] -= zuas[istart]
    # now only grab indices where 3 m <= zuas <= 396 m
    iuse = np.where((zuas >= 3.) & (zuas <= 396.))[0]
    zuas = zuas[iuse]
    # calculate dz_uas from ascent_rate and time_average
    dz_uas = ascent_rate * time_average
    # loop over keys and interpolate
    interp_keys = ["u", "v", "theta"]
    d_interp = {} # define empty dictionary for looping
    for key in interp_keys:
        print(f"Interpolating {key}...")
        d_interp[key] = interp_uas(df[key].isel(t=iuse).values,
                                   df.z.values, zuas)

    # grab data from interpolated arrays to create simulated raw UAS profiles
    # define xarray dataset to eventually store all
    uas_raw = xr.Dataset(data_vars=None, coords=dict(z=zuas))
    # begin looping through keys
    for key in interp_keys:
        # define empty list
        duas = []
        # loop over altitudes/times
        for i in range(len(iuse)):
            duas.append(d_interp[key][i,i])
        # assign to uas_raw
        uas_raw[key] = xr.DataArray(data=np.array(duas), coords=dict(z=zuas))
    
    # emulate post-processing and average over altitude bins
    # can do this super easily with xarray groupby_bins
    # want bins to be at the midpoint between dz_uas grid
    zbin = np.arange(dz_uas/2, 400.-dz_uas/2, dz_uas)
    # group by altitude bins and calculate mean in one line
    uas_mean = uas_raw.groupby_bins("z", zbin).mean("z", skipna=True)
    # fix z coordinates: swap z_bins out for dz_uas grid
    znew = np.arange(dz_uas, 400.-dz_uas, dz_uas)
    # create new coordinate "z" from znew that is based on z_bins, then swap and drop
    uas_mean = uas_mean.assign_coords({"z": ("z_bins", znew)}).swap_dims({"z_bins": "z"})
    # only save data for z <= h
    h = err.z.max()
    uas_mean = uas_mean.where(uas_mean.z <= h, drop=True)
    # calculate uh, alpha from uas_mean profile
    uas_mean["uh"] = (uas_mean.u**2. + uas_mean.v**2.) ** 0.5
    alpha = np.arctan2(-uas_mean.u, -uas_mean.v) * 180./np.pi
    alpha[alpha < 0.] += 360.
    uas_mean["alpha"] = alpha
    #
    # interpolate errors for everything in uas_mean
    #
    uas_mean["uh_err"] = err.uh.interp(z=uas_mean.z)
    uas_mean["alpha_err"] = err.alpha.interp(z=uas_mean.z)
    uas_mean["theta_err"] = err.theta.interp(z=uas_mean.z)
    
    # quicklook plot
    if quicklook:
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
        # u
        ax[0].plot(uas_raw.u, uas_raw.z/h, "-k", label="raw")
        ax[0].plot(uas_mean.u, uas_mean.z/h, "-xb", label="mean")
        # v
        ax[1].plot(uas_raw.v, uas_raw.z/h, "-k", label="raw")
        ax[1].plot(uas_mean.v, uas_mean.z/h, "-xb", label="mean")
        # theta
        ax[2].plot(uas_raw.theta, uas_raw.z/h, "-k", label="raw")
        ax[2].plot(uas_mean.theta, uas_mean.z/h, "-xb", label="mean")
        # clean up
        ax[0].set_ylim([0, 1])
        ax[0].set_ylabel("$z/h$")
        ax[0].set_xlabel("$u$")
        ax[0].legend()
        ax[1].set_xlabel("$v$")
        ax[2].set_xlabel("$\\theta$")
        fig.tight_layout()
        # save and close
        fsave = f"{fdir_save}{err.stability}_u_v_theta_raw_mean.png"
        print(f"Saving figure: {fsave}")
        fig.savefig(fsave, dpi=300)
        plt.close(fig)
        
    return uas_mean
# --------------------------------
def ec(df, h, time_average=1800.0, time_start=0.0, quicklook=False):
    """Emulate a tower extending throughout SBL with EC system at each vertical
    gridpoint and calculate variances and covariances
    :param xr.Dataset df: dataset with virtual tower data to construct UAS prof
    :param float h: SBL depth in m
    :param float time_average: time range in s to avg timeseries; default=1800
    :param float time_start: when to initialize averaging; default=0.0
    :param bool quicklook: flag to make quicklook of raw vs averaged profiles
    Outputs new xarray Dataset with emulated vars and covars
    """
    # first find the index in df.t that corresponds to time_start
    istart = int(time_start / df.dt)
    # determine how many indices to use from time_average
    nuse = int(time_average / df.dt)
    # create array of indices to use
    iuse = np.linspace(istart, istart+nuse-1, nuse, dtype=np.int32)
    # initialize empty dataset to hold everything
    ec = xr.Dataset(data_vars=None, coords=dict(z=df.z))
    # begin calculating statistics
    # u'w'
    ec["uw_cov_res"] = xr.cov(df.u.isel(t=iuse), df.w.isel(t=iuse), dim=("t"))
    ec["uw_cov_tot"] = ec.uw_cov_res + df.txz.isel(t=iuse).mean("t")
    # v'w'
    ec["vw_cov_res"] = xr.cov(df.v.isel(t=iuse), df.w.isel(t=iuse), dim=("t"))
    ec["vw_cov_tot"] = ec.vw_cov_res + df.tyz.isel(t=iuse).mean("t")
    # theta'w'
    ec["tw_cov_res"] = xr.cov(df.theta.isel(t=iuse), df.w.isel(t=iuse), dim=("t"))
    ec["tw_cov_tot"] = ec.tw_cov_res + df.q3.isel(t=iuse).mean("t")
    # ustar^2 = sqrt(u'w'^2 + v'w'^2)
    ec["ustar2"] = ((ec.uw_cov_tot**2.) + (ec.vw_cov_tot**2.)) ** 0.5
    # variances
    for v in ["u", "v", "w", "theta"]:
        ec[f"{v}_var"] = df[v].isel(t=iuse).var("t")
    # rotate u and v
    angle = np.arctan2(df.v.isel(t=iuse).mean("t"), df.u.isel(t=iuse).mean("t"))
    u_rot = df.u.isel(t=iuse)*np.cos(angle) + df.v.isel(t=iuse)*np.sin(angle)
    v_rot =-df.u.isel(t=iuse)*np.sin(angle) + df.v.isel(t=iuse)*np.cos(angle)
    ec["u_var_rot"] = u_rot.var("t")
    ec["v_var_rot"] = v_rot.var("t")
    
    # only return ec where z <= h
    return ec.where(ec.z <= h, drop=True)
    
# --------------------------------
# Configure plots and define directories
# --------------------------------
# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
rc('text',usetex='True')

# figure save directory
fdir_save = "/home/bgreene/SBL_LES/figures/UASnc/"
if not os.path.exists(fdir_save):
    os.mkdir(fdir_save)
plt.close("all")

# simulation base directory
fsim = "/home/bgreene/simulations/"

# --------------------------------
# Load average stats and timeseries files
# --------------------------------
# load 1hr stats files
fstatA = f"{fsim}A_192_interp/output/netcdf/average_statistics.nc"
Astat = xr.load_dataset(fstatA)
fstatF = f"{fsim}F_192_interp/output/netcdf/average_statistics.nc"
Fstat = xr.load_dataset(fstatF)
stat_all = [Astat, Fstat]
# calculate ustar and h for each
for s in [Astat, Fstat]:
    # ustar
    s["ustar"] = ((s.uw_cov_tot ** 2.) + (s.vw_cov_tot ** 2.)) ** 0.25
    s["ustar2"] = s.ustar ** 2.
    # SBL height
    s["h"] = s.z.where(s.ustar2 <= 0.05*s.ustar2[0], drop=True)[0]/0.95
    # z indices within sbl
    s["isbl"] = np.where(s.z <= s.h)[0]
    s["nz_sbl"] = len(s.isbl)
    s["z_sbl"] = s.z.isel(z=s.isbl)
    # uh
    s["uh"] = ((s.u_mean**2.) + (s.v_mean**2.)) ** 0.5
    # alpha
    wdir = np.arctan2(-s.u_mean, -s.v_mean) * 180./np.pi
    wdir[wdir < 0.] += 360.
    s["wd"] = wdir

# load timeseries files
ftsA = f"{fsim}A_192_interp/output/netcdf/timeseries_all.nc"
Ats = xr.load_dataset(ftsA)
ftsF = f"{fsim}F_192_interp/output/netcdf/timeseries_all.nc"
Fts = xr.load_dataset(ftsF)

# load error profile files
ferrA = f"{fsim}A_192_interp/output/netcdf/err.nc"
Aerr = xr.load_dataset(ferrA)
ferrF = f"{fsim}F_192_interp/output/netcdf/err.nc"
Ferr = xr.load_dataset(ferrF)
err_all = [Aerr, Ferr]

# --------------------------------
# Perform calculations
# --------------------------------

# run profile for each sim
Auas = profile(Ats, Aerr, quicklook=True)
Fuas = profile(Fts, Ferr, quicklook=True)
uas_all = [Auas, Fuas]

# run ec for each sim
Aec = ec(Ats, Astat.h)
Fec = ec(Fts, Fstat.h)
ec_all = [Aec, Fec]

# calculate error bounds 
for s in uas_all:
    # first order
    # 1 sigma
    s["err_uh_hi"] = (1. + s.uh_err) * s.uh
    s["err_uh_lo"] = (1. - s.uh_err) * s.uh
    s["err_alpha_hi"] = (1. + s.alpha_err) * s.alpha
    s["err_alpha_lo"] = (1. - s.alpha_err) * s.alpha
    s["err_theta_hi"] = (1. + s.theta_err) * s.theta
    s["err_theta_lo"] = (1. - s.theta_err) * s.theta
    # 3 sigma
    s["err_uh_hi3"] = (1. + 3*s.uh_err) * s.uh
    s["err_uh_lo3"] = (1. - 3*s.uh_err) * s.uh
    s["err_alpha_hi3"] = (1. + 3*s.alpha_err) * s.alpha
    s["err_alpha_lo3"] = (1. - 3*s.alpha_err) * s.alpha
    s["err_theta_hi3"] = (1. + 3*s.theta_err) * s.theta
    s["err_theta_lo3"] = (1. - 3*s.theta_err) * s.theta
for s, err in zip(ec_all, err_all):
    # covariances
    # 1 sigma
    s["err_uw_hi"] = (1. + err.uw_cov_tot) * s.uw_cov_tot
    s["err_uw_lo"] = (1. - err.uw_cov_tot) * s.uw_cov_tot
    s["err_vw_hi"] = (1. + err.vw_cov_tot) * s.vw_cov_tot
    s["err_vw_lo"] = (1. - err.vw_cov_tot) * s.vw_cov_tot
    s["err_tw_hi"] = (1. + err.tw_cov_tot) * s.tw_cov_tot
    s["err_tw_lo"] = (1. - err.tw_cov_tot) * s.tw_cov_tot
    s["err_ustar2_hi"] = (1. + err.ustar2) * s.ustar2
    s["err_ustar2_lo"] = (1. - err.ustar2) * s.ustar2
    # 3 sigma
    s["err_uw_hi3"] = (1. + 3*err.uw_cov_tot) * s.uw_cov_tot
    s["err_uw_lo3"] = (1. - 3*err.uw_cov_tot) * s.uw_cov_tot
    s["err_vw_hi3"] = (1. + 3*err.vw_cov_tot) * s.vw_cov_tot
    s["err_vw_lo3"] = (1. - 3*err.vw_cov_tot) * s.vw_cov_tot
    s["err_tw_hi3"] = (1. + 3*err.tw_cov_tot) * s.tw_cov_tot
    s["err_tw_lo3"] = (1. - 3*err.tw_cov_tot) * s.tw_cov_tot
    s["err_ustar2_hi3"] = (1. + 3*err.ustar2) * s.ustar2
    s["err_ustar2_lo3"] = (1. - 3*err.ustar2) * s.ustar2
    # variances
    # 1 sigma
    s["err_uu_hi"] = (1. + err.uu_var) * s.u_var
    s["err_uu_lo"] = (1. - err.uu_var) * s.u_var
    s["err_uu_rot_hi"] = (1. + err.uu_var_rot) * s.u_var_rot
    s["err_uu_rot_lo"] = (1. - err.uu_var_rot) * s.u_var_rot
    s["err_vv_hi"] = (1. + err.vv_var) * s.v_var
    s["err_vv_lo"] = (1. - err.vv_var) * s.v_var
    s["err_vv_rot_hi"] = (1. + err.vv_var_rot) * s.v_var_rot
    s["err_vv_rot_lo"] = (1. - err.vv_var_rot) * s.v_var_rot
    s["err_ww_hi"] = (1. + err.ww_var) * s.w_var
    s["err_ww_lo"] = (1. - err.ww_var) * s.w_var
    s["err_tt_hi"] = (1. + err.tt_var) * s.theta_var
    s["err_tt_lo"] = (1. - err.tt_var) * s.theta_var
    # 3 sigma
    s["err_uu_hi3"] = (1. + 3*err.uu_var) * s.u_var
    s["err_uu_lo3"] = (1. - 3*err.uu_var) * s.u_var
    s["err_uu_rot_hi3"] = (1. + 3*err.uu_var_rot) * s.u_var_rot
    s["err_uu_rot_lo3"] = (1. - 3*err.uu_var_rot) * s.u_var_rot
    s["err_vv_hi3"] = (1. + 3*err.vv_var) * s.v_var
    s["err_vv_lo3"] = (1. - 3*err.vv_var) * s.v_var
    s["err_vv_rot_hi3"] = (1. + 3*err.vv_var_rot) * s.v_var_rot
    s["err_vv_rot_lo3"] = (1. - 3*err.vv_var_rot) * s.v_var_rot
    s["err_ww_hi3"] = (1. + 3*err.ww_var) * s.w_var
    s["err_ww_lo3"] = (1. - 3*err.ww_var) * s.w_var
    s["err_tt_hi3"] = (1. + 3*err.tt_var) * s.theta_var
    s["err_tt_lo3"] = (1. - 3*err.tt_var) * s.theta_var
# --------------------------------
# Plot
# --------------------------------
#
# Figure 1: uh, alpha, theta profiles from both UAS and LES mean
#
for s, stat in zip(uas_all, stat_all):
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
    # uh
    ax1[0].plot(stat.uh.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle u_h \\rangle$")
    ax1[0].plot(s.uh, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax1[0].fill_betweenx(s.z/stat.h, s.err_uh_lo, s.err_uh_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u_h}$")
    ax1[0].fill_betweenx(s.z/stat.h, s.err_uh_lo3, s.err_uh_hi3, alpha=0.1,
                         color="r")
    # alpha
    ax1[1].plot(stat.wd.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h,
                c="k", ls="-", lw=2, label="$\\langle \\alpha \\rangle$")
    ax1[1].plot(s.alpha, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax1[1].fill_betweenx(s.z/stat.h, s.err_alpha_lo, s.err_alpha_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{\\alpha}$")
    ax1[1].fill_betweenx(s.z/stat.h, s.err_alpha_lo3, s.err_alpha_hi3, alpha=0.1,
                         color="r")
    # theta
    ax1[2].plot(stat.theta_mean.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h,
                c="k", ls="-", lw=2, label="$\\langle \\theta \\rangle$")
    ax1[2].plot(s.theta, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax1[2].fill_betweenx(s.z/stat.h, s.err_theta_lo, s.err_theta_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{\\theta}$")
    ax1[2].fill_betweenx(s.z/stat.h, s.err_theta_lo3, s.err_theta_hi3, alpha=0.1,
                         color="r")
    # clean up
    for iax in ax1:
        iax.legend(loc="upper left")
    ax1[0].set_xlabel("$u_h$ [m s$^{-1}$]")
    ax1[0].set_ylabel("$z/h$")
    ax1[0].set_ylim([0, 1])
    ax1[0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax1[0].yaxis.set_minor_locator(MultipleLocator(0.05))
#     ax1[0].set_xlim([0, 10])
#     ax1[0].xaxis.set_major_locator(MultipleLocator(2))
#     ax1[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1[1].set_xlabel("$\\alpha$ [$^\circ$]")
#     ax1[1].set_xlim([210, 270])
#     ax1[1].xaxis.set_major_locator(MultipleLocator(15))
#     ax1[1].xaxis.set_minor_locator(MultipleLocator(5))
    ax1[2].set_xlabel("$\\theta$ [K]")
#     ax1[2].set_xlim([263, 265])
#     ax1[2].xaxis.set_major_locator(MultipleLocator(0.5))
#     ax1[2].xaxis.set_minor_locator(MultipleLocator(0.1))
    fig1.tight_layout()
    # save and close
    fsave1 = f"{fdir_save}{stat.stability}_uh_alpha_theta.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1)
    plt.close(fig1)
    
#
# Figure 2: covariances
#
for s, stat in zip(ec_all, stat_all):
    fig2, ax2 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(14.8, 5))
    # u'w'
    ax2[0].plot(stat.uw_cov_tot.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle u'w' \\rangle$")
    ax2[0].plot(s.uw_cov_tot, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax2[0].fill_betweenx(s.z/stat.h, s.err_uw_lo, s.err_uw_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u'w'}$")
    ax2[0].fill_betweenx(s.z/stat.h, s.err_uw_lo3, s.err_uw_hi3, alpha=0.1,
                         color="r")
    # v'w'
    ax2[1].plot(stat.vw_cov_tot.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle v'w' \\rangle$")
    ax2[1].plot(s.vw_cov_tot, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax2[1].fill_betweenx(s.z/stat.h, s.err_vw_lo, s.err_vw_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{v'w'}$")
    ax2[1].fill_betweenx(s.z/stat.h, s.err_vw_lo3, s.err_vw_hi3, alpha=0.1,
                         color="r")
    # ustar^2
    ax2[2].plot(stat.ustar2.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$u_{*}^2$")
    ax2[2].plot(s.ustar2, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax2[2].fill_betweenx(s.z/stat.h, s.err_ustar2_lo, s.err_ustar2_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u_{*}^2}$")
    ax2[2].fill_betweenx(s.z/stat.h, s.err_ustar2_lo3, s.err_ustar2_hi3, alpha=0.1,
                         color="r")
    # theta'w'
    ax2[3].plot(stat.tw_cov_tot.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle \\theta'w' \\rangle$")
    ax2[3].plot(s.tw_cov_tot, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax2[3].fill_betweenx(s.z/stat.h, s.err_tw_lo, s.err_tw_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{\\theta'w'}$")
    ax2[3].fill_betweenx(s.z/stat.h, s.err_tw_lo3, s.err_tw_hi3, alpha=0.1,
                         color="r")
    # clean up
    for iax in ax2[[0,1,3]]:
        iax.legend(loc="upper left")
    ax2[2].legend(loc="upper right")
    ax2[0].set_xlabel("$u'w'$ [m$^2$ s$^{-2}$]")
    ax2[0].set_ylabel("$z/h$")
    ax2[0].set_ylim([0, 1])
#     ax2[0].yaxis.set_major_locator(MultipleLocator(0.2))
#     ax2[0].yaxis.set_minor_locator(MultipleLocator(0.05))
#     ax2[0].set_xlim([0, 10])
#     ax2[0].xaxis.set_major_locator(MultipleLocator(2))
#     ax2[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax2[1].set_xlabel("$v'w'$ [m$^2$ s$^{-2}$]")
#     ax2[1].set_xlim([210, 270])
#     ax2[1].xaxis.set_major_locator(MultipleLocator(15))
#     ax2[1].xaxis.set_minor_locator(MultipleLocator(5))
    ax2[2].set_xlabel("$u_{*}^2$ [m$^2$ s$^{-2}$]")
    ax2[3].set_xlabel("$\\theta'w'$ [K m s$^{-1}$]")
#     ax2[2].set_xlim([263, 265])
#     ax2[2].xaxis.set_major_locator(MultipleLocator(0.5))
#     ax2[2].xaxis.set_minor_locator(MultipleLocator(0.1))
    fig2.tight_layout()
    # save and close
    fsave2 = f"{fdir_save}{stat.stability}_covars.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2)
    plt.close(fig2)
    
#
# Figure 3: variances
#
for s, stat in zip(ec_all, stat_all):
    fig3, ax3 = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(14.8, 10))
    # u'u' UNROTATED
    ax3[0,0].plot(stat.u_var.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle u'u' \\rangle$")
    ax3[0,0].plot(s.u_var, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[0,0].fill_betweenx(s.z/stat.h, s.err_uu_lo, s.err_uu_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u'u'}$")
    ax3[0,0].fill_betweenx(s.z/stat.h, s.err_uu_lo3, s.err_uu_hi3, alpha=0.1,
                         color="r")
    # v'v' UNROTATED
    ax3[0,1].plot(stat.v_var.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle v'v' \\rangle$")
    ax3[0,1].plot(s.v_var, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[0,1].fill_betweenx(s.z/stat.h, s.err_vv_lo, s.err_vv_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{v'v'}$")
    ax3[0,1].fill_betweenx(s.z/stat.h, s.err_vv_lo3, s.err_vv_hi3, alpha=0.1,
                         color="r")
    # w'w'
    ax3[0,2].plot(stat.w_var.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle w'w' \\rangle$")
    ax3[0,2].plot(s.w_var, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[0,2].fill_betweenx(s.z/stat.h, s.err_ww_lo, s.err_ww_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{w'w'}$")
    ax3[0,2].fill_betweenx(s.z/stat.h, s.err_ww_lo3, s.err_ww_hi3, alpha=0.1,
                         color="r")
    # u'u' ROTATED
    ax3[1,0].plot(stat.u_var_rot.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle u'u' \\rangle_{rot}$")
    ax3[1,0].plot(s.u_var_rot, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[1,0].fill_betweenx(s.z/stat.h, s.err_uu_rot_lo, s.err_uu_rot_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u'u'_{rot}}$")
    ax3[1,0].fill_betweenx(s.z/stat.h, s.err_uu_rot_lo3, s.err_uu_rot_hi3, alpha=0.1,
                         color="r")
    # v'v' ROTATED
    ax3[1,1].plot(stat.v_var_rot.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle v'v' \\rangle_{rot}$")
    ax3[1,1].plot(s.v_var_rot, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[1,1].fill_betweenx(s.z/stat.h, s.err_vv_rot_lo, s.err_vv_rot_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{v'v'_{rot}}$")
    ax3[1,1].fill_betweenx(s.z/stat.h, s.err_vv_rot_lo3, s.err_vv_rot_hi3, alpha=0.1,
                         color="r")
    # theta'theta'
    ax3[1,2].plot(stat.theta_var.isel(z=stat.isbl), stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle \\theta'\\theta' \\rangle$")
    ax3[1,2].plot(s.theta_var, s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[1,2].fill_betweenx(s.z/stat.h, s.err_tt_lo, s.err_tt_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{\\theta'\\theta'}$")
    ax3[1,2].fill_betweenx(s.z/stat.h, s.err_tt_lo3, s.err_tt_hi3, alpha=0.1,
                         color="r")
    # clean up
    for iax in ax3.flatten():
        iax.legend(loc="upper right")
    ax3[0,0].set_xlabel("$u'u'$ [m$^2$ s$^{-2}$]")
    ax3[0,0].set_ylabel("$z/h$")
    ax3[0,0].set_ylim([0, 1])
#     ax3[0].yaxis.set_major_locator(MultipleLocator(0.2))
#     ax3[0].yaxis.set_minor_locator(MultipleLocator(0.05))
#     ax3[0].set_xlim([0, 10])
#     ax3[0].xaxis.set_major_locator(MultipleLocator(2))
#     ax3[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax3[0,1].set_xlabel("$v'v'$ [m$^2$ s$^{-2}$]")
#     ax3[1].set_xlim([210, 270])
#     ax3[1].xaxis.set_major_locator(MultipleLocator(15))
#     ax3[1].xaxis.set_minor_locator(MultipleLocator(5))
    ax3[0,2].set_xlabel("$w'w'$ [m$^2$ s$^{-2}$]")
#     ax3[2].set_xlim([263, 265])
#     ax3[2].xaxis.set_major_locator(MultipleLocator(0.5))
#     ax3[2].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax3[1,0].set_ylabel("$z/h$")
    ax3[1,0].set_xlabel("$u'u'_{rot}$ [m$^2$ s$^{-2}$]")
    ax3[1,1].set_xlabel("$v'v'_{rot}$ [m$^2$ s$^{-2}$]")
    ax3[1,2].set_xlabel("$\\theta'\\theta'}$ [K$^2$]")
    fig3.tight_layout()
    # save and close
    fsave3 = f"{fdir_save}{stat.stability}_vars.pdf"
    print(f"Saving figure: {fsave3}")
    fig3.savefig(fsave3)
    plt.close(fig3)