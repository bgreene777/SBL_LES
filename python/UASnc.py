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
import yaml
import seaborn
import numpy as np
import xarray as xr
from cmocean import cm
from numba import njit
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from RFMnc import recalc_err
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
def profile(df, err, ascent_rate=1.0, time_average=3.0, time_start=0.0, 
            quicklook=False, timeheight=False):
    """Emulate a vertical profile from rotary-wing UAS sampling through
    simulated SBL, including random errors
    :param xr.Dataset df: dataset with virtual tower data to construct UAS prof
    :param xr.Dataset err: dataset with relative random errors corr. to df
    :param float ascent_rate: UAS ascent rate in m/s; default=1.0
    :param float time_average: time range in s to avg UAS profile; default=3.0 s
    :param float time_start: when to initialize ascending profile; default=0.0 s
    :param bool quicklook: flag to make quicklook of raw vs averaged profiles
    :param bool timeheight: flag to produce time-height figure of u-velocity and
    uas ascent profile overlaid
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
    # time-height plot
    if timeheight:
        fig, ax = plt.subplots(1, figsize=(7.4, 2.5), constrained_layout=True)
        # contourf u
        cfax = ax.contourf(df.t/60., df.z/err.h, df.u.T, cmap=cm.tempo,
                           levels=np.arange(0., 12.1, 0.25))
        # instantaneous vertical profile
        ax.axvline(time_start/60., c="k", ls="-", lw=2)
        # UAS profile
        t_total = err.h/ascent_rate
        t_uas = np.array([time_start, time_start+t_total])/60. # minutes
        z_uas = np.array([0., 1.])
        ax.plot(t_uas, z_uas, ls="-", c="r", lw=2)
        # ticks
        ax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
        ax.tick_params(which="major", length=6, width=0.5)
        ax.tick_params(which="minor", length=3, width=0.5)
        # colorbar
        cb = fig.colorbar(cfax, ax=ax, location="right", 
                          ticks=MultipleLocator(4), pad=0, aspect=15)
        cb.ax.set_ylabel("$u$ [m s$^{-1}$]", fontsize=16)
        cb.ax.tick_params(labelsize=16)
        # labels
        ax.set_xlabel("Time [min]")
        ax.set_xlim([0, 60])
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel("$z/h$")
        ax.set_ylim([0, 1])
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        # save and close
        fsave = f"{fdir_save}{err.stability}_profile_timeheight.png"
        print(f"Saving figure: {fsave}")
        fig.savefig(fsave, dpi=600)
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
    # calculate TKE
    ec["e"] = 0.5 * (ec.u_var + ec.v_var + ec.w_var)
    
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

# load yaml file with plotting parameters
with open("/home/bgreene/SBL_LES/python/UASnc.yaml") as f:
    config = yaml.safe_load(f)

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
    # ustar0 at lowest z
    s["ustar0"] = s.ustar.isel(z=0)
    # calculate thetastar0 at lowest level
    s["tstar0"] = -s.tw_cov_tot.isel(z=0)/s.ustar0
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
    # calculate TKE
    s["e"] = 0.5 * (s.u_var + s.v_var + s.w_var)

# load timeseries files
ftsA = f"{fsim}A_192_interp/output/netcdf/timeseries_all.nc"
Ats = xr.load_dataset(ftsA)
ftsF = f"{fsim}F_192_interp/output/netcdf/timeseries_all.nc"
Fts = xr.load_dataset(ftsF)

# load error profile files
# ferrA = f"{fsim}A_192_interp/output/netcdf/err.nc"
# Aerr = xr.load_dataset(ferrA)
# ferrF = f"{fsim}F_192_interp/output/netcdf/err.nc"
# Ferr = xr.load_dataset(ferrF)

# calc error based on config file
Aerr = recalc_err("A", config["Tavg_uv"], config["Tavg_ec"])
Ferr = recalc_err("F", config["Tavg_uv"], config["Tavg_ec"])
err_all = [Aerr, Ferr]

# --------------------------------
# Perform calculations
# --------------------------------

# run profile for each sim
Auas = profile(Ats, Aerr, time_start=config["tstart"], timeheight=False)
Fuas = profile(Fts, Ferr, time_start=config["tstart"], timeheight=False)
uas_all = [Auas.isel(Tsample=0), Fuas.isel(Tsample=0)]

# run ec for each sim
Aec = ec(Ats, Astat.h, time_average=config["Tavg_ec"], 
         time_start=config["T0_ec"], quicklook=True)
Fec = ec(Fts, Fstat.h, time_average=config["Tavg_ec"], 
         time_start=config["T0_ec"], quicklook=True)
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

for s, err, stat in zip(ec_all, err_all, stat_all):
    # covariances
    # 1 sigma
    s["err_uw_hi"] = (1. + err.uw_cov_tot.isel(Tsample_ec=0)) * (s.uw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_uw_lo"] = (1. - err.uw_cov_tot.isel(Tsample_ec=0)) * (s.uw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_vw_hi"] = (1. + err.vw_cov_tot.isel(Tsample_ec=0)) * (s.vw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_vw_lo"] = (1. - err.vw_cov_tot.isel(Tsample_ec=0)) * (s.vw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_tw_hi"] = (1. + err.tw_cov_tot.isel(Tsample_ec=0)) * (s.tw_cov_tot/stat.tstar0/stat.ustar0)
    s["err_tw_lo"] = (1. - err.tw_cov_tot.isel(Tsample_ec=0)) * (s.tw_cov_tot/stat.tstar0/stat.ustar0)
    s["err_ustar2_hi"] = (1. + err.ustar2.isel(Tsample_ec=0)) * (s.ustar2/stat.ustar0/stat.ustar0)
    s["err_ustar2_lo"] = (1. - err.ustar2.isel(Tsample_ec=0)) * (s.ustar2/stat.ustar0/stat.ustar0)
    # 3 sigma
    s["err_uw_hi3"] = (1. + 3*err.uw_cov_tot.isel(Tsample_ec=0)) * (s.uw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_uw_lo3"] = (1. - 3*err.uw_cov_tot.isel(Tsample_ec=0)) * (s.uw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_vw_hi3"] = (1. + 3*err.vw_cov_tot.isel(Tsample_ec=0)) * (s.vw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_vw_lo3"] = (1. - 3*err.vw_cov_tot.isel(Tsample_ec=0)) * (s.vw_cov_tot/stat.ustar0/stat.ustar0)
    s["err_tw_hi3"] = (1. + 3*err.tw_cov_tot.isel(Tsample_ec=0)) * (s.tw_cov_tot/stat.tstar0/stat.ustar0)
    s["err_tw_lo3"] = (1. - 3*err.tw_cov_tot.isel(Tsample_ec=0)) * (s.tw_cov_tot/stat.tstar0/stat.ustar0)
    s["err_ustar2_hi3"] = (1. + 3*err.ustar2.isel(Tsample_ec=0)) * (s.ustar2/stat.ustar0/stat.ustar0)
    s["err_ustar2_lo3"] = (1. - 3*err.ustar2.isel(Tsample_ec=0)) * (s.ustar2/stat.ustar0/stat.ustar0)
    # variances
    # 1 sigma
    s["err_uu_hi"] = (1. + err.uu_var.isel(Tsample_ec=0)) * (s.u_var/stat.ustar0/stat.ustar0)
    s["err_uu_lo"] = (1. - err.uu_var.isel(Tsample_ec=0)) * (s.u_var/stat.ustar0/stat.ustar0)
    s["err_uu_rot_hi"] = (1. + err.uu_var_rot.isel(Tsample_ec=0)) * (s.u_var_rot/stat.ustar0/stat.ustar0)
    s["err_uu_rot_lo"] = (1. - err.uu_var_rot.isel(Tsample_ec=0)) * (s.u_var_rot/stat.ustar0/stat.ustar0)
    s["err_vv_hi"] = (1. + err.vv_var.isel(Tsample_ec=0)) * (s.v_var/stat.ustar0/stat.ustar0)
    s["err_vv_lo"] = (1. - err.vv_var.isel(Tsample_ec=0)) * (s.v_var/stat.ustar0/stat.ustar0)
    s["err_vv_rot_hi"] = (1. + err.vv_var_rot.isel(Tsample_ec=0)) * (s.v_var_rot/stat.ustar0/stat.ustar0)
    s["err_vv_rot_lo"] = (1. - err.vv_var_rot.isel(Tsample_ec=0)) * (s.v_var_rot/stat.ustar0/stat.ustar0)
    s["err_ww_hi"] = (1. + err.ww_var.isel(Tsample_ec=0)) * (s.w_var/stat.ustar0/stat.ustar0)
    s["err_ww_lo"] = (1. - err.ww_var.isel(Tsample_ec=0)) * (s.w_var/stat.ustar0/stat.ustar0)
    s["err_tt_hi"] = (1. + err.tt_var.isel(Tsample_ec=0)) * (s.theta_var/stat.tstar0/stat.tstar0)
    s["err_tt_lo"] = (1. - err.tt_var.isel(Tsample_ec=0)) * (s.theta_var/stat.tstar0/stat.tstar0)
    # 3 sigma
    s["err_uu_hi3"] = (1. + 3*err.uu_var.isel(Tsample_ec=0)) * (s.u_var/stat.ustar0/stat.ustar0)
    s["err_uu_lo3"] = (1. - 3*err.uu_var.isel(Tsample_ec=0)) * (s.u_var/stat.ustar0/stat.ustar0)
    s["err_uu_rot_hi3"] = (1. + 3*err.uu_var_rot.isel(Tsample_ec=0)) * (s.u_var_rot/stat.ustar0/stat.ustar0)
    s["err_uu_rot_lo3"] = (1. - 3*err.uu_var_rot.isel(Tsample_ec=0)) * (s.u_var_rot/stat.ustar0/stat.ustar0)
    s["err_vv_hi3"] = (1. + 3*err.vv_var.isel(Tsample_ec=0)) * (s.v_var/stat.ustar0/stat.ustar0)
    s["err_vv_lo3"] = (1. - 3*err.vv_var.isel(Tsample_ec=0)) * (s.v_var/stat.ustar0/stat.ustar0)
    s["err_vv_rot_hi3"] = (1. + 3*err.vv_var_rot.isel(Tsample_ec=0)) * (s.v_var_rot/stat.ustar0/stat.ustar0)
    s["err_vv_rot_lo3"] = (1. - 3*err.vv_var_rot.isel(Tsample_ec=0)) * (s.v_var_rot/stat.ustar0/stat.ustar0)
    s["err_ww_hi3"] = (1. + 3*err.ww_var.isel(Tsample_ec=0)) * (s.w_var/stat.ustar0/stat.ustar0)
    s["err_ww_lo3"] = (1. - 3*err.ww_var.isel(Tsample_ec=0)) * (s.w_var/stat.ustar0/stat.ustar0)
    s["err_tt_hi3"] = (1. + 3*err.tt_var.isel(Tsample_ec=0)) * (s.theta_var/stat.tstar0/stat.tstar0)
    s["err_tt_lo3"] = (1. - 3*err.tt_var.isel(Tsample_ec=0)) * (s.theta_var/stat.tstar0/stat.tstar0)
    # calculate TKE error
    err["e"] = np.sqrt(0.25 * ((err.uu_var.isel(Tsample_ec=0)*stat.u_var)**2. +\
                               (err.vv_var.isel(Tsample_ec=0)*stat.v_var)**2. +\
                               (err.ww_var.isel(Tsample_ec=0)*stat.w_var)**2.) ) / stat.e
    # calculate TKE bounds
    s["err_e_hi"] = (1. + err.e) * (s.e/stat.ustar0/stat.ustar0)
    s["err_e_lo"] = (1. - err.e) * (s.e/stat.ustar0/stat.ustar0)
    s["err_e_hi3"] = (1. + 3*err.e) * (s.e/stat.ustar0/stat.ustar0)
    s["err_e_lo3"] = (1. - 3*err.e) * (s.e/stat.ustar0/stat.ustar0)
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
        iax.legend(loc="upper left", labelspacing=0.10, 
                   handletextpad=0.4, shadow=True)
    ax1[0].set_xlabel("$u_h$ [m s$^{-1}$]")
    ax1[0].set_ylabel("$z/h$")
    ax1[0].set_ylim([0, 1])
    ax1[0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax1[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1[0].set_xlim(config[stat.stability]["xlim"]["ax1_0"])
    ax1[0].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax1_0"]))
    ax1[0].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax1_0"]))
    ax1[1].set_xlabel("$\\alpha$ [$^\circ$]")
    ax1[1].set_xlim(config[stat.stability]["xlim"]["ax1_1"])
    ax1[1].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax1_1"]))
    ax1[1].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax1_1"]))
    ax1[2].set_xlabel("$\\theta$ [K]")
    ax1[2].set_xlim(config[stat.stability]["xlim"]["ax1_2"])
    ax1[2].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax1_2"]))
    ax1[2].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax1_2"]))
    # edit ticks and add subplot labels
    for iax, p in zip(ax1, list("abc")):
        iax.tick_params(which="both", direction="in", top=True, right=True)
        iax.text(0.88,0.05,f"$\\textbf{{({p})}}$",fontsize=20,
                 transform=iax.transAxes)
    fig1.tight_layout()
    # save and close
    fsave1 = f"{fdir_save}{stat.stability}_uh_alpha_theta_{int(config['Tavg_uv']):02d}s.pdf"
    print(f"Saving figure: {fsave1}")
    fig1.savefig(fsave1)
    plt.close(fig1)
    
#
# Figure 2: covariances
# ustar^2 and theta'w', normalized with ustar0 and thetastar0
#
for s, stat in zip(ec_all, stat_all):
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(9.87, 5))
    # ustar^2
    ax2[0].plot(stat.ustar2.isel(z=stat.isbl)/stat.ustar0/stat.ustar0,
                stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$u_{*}^2$")
    ax2[0].plot(s.ustar2/stat.ustar0/stat.ustar0,
                s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax2[0].fill_betweenx(s.z/stat.h, s.err_ustar2_lo, s.err_ustar2_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u_{*}^2}$")
    ax2[0].fill_betweenx(s.z/stat.h, s.err_ustar2_lo3, s.err_ustar2_hi3, alpha=0.1,
                         color="r")
    # theta'w'
    ax2[1].plot(stat.tw_cov_tot.isel(z=stat.isbl)/stat.ustar0/stat.tstar0,
                stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle \\theta'w' \\rangle$")
    ax2[1].plot(s.tw_cov_tot/stat.ustar0/stat.tstar0,
                s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax2[1].fill_betweenx(s.z/stat.h, s.err_tw_lo, s.err_tw_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{\\theta'w'}$")
    ax2[1].fill_betweenx(s.z/stat.h, s.err_tw_lo3, s.err_tw_hi3, alpha=0.1,
                         color="r")
    # clean up
    # for iax in ax2[[0,1,3]]:
    ax2[0].legend(loc="upper right", labelspacing=0.10, 
                handletextpad=0.4, shadow=True)
    ax2[1].legend(loc="upper left", labelspacing=0.10, 
                   handletextpad=0.4, shadow=True)
    ax2[0].set_ylabel("$z/h$")
    ax2[0].set_ylim([0, 1])
    ax2[0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax2[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax2[0].set_xlabel("$u_{*}^2/u_{*0}^2$")
    ax2[0].set_xlim(config[stat.stability]["xlim"]["ax2_0"])
    ax2[0].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax2_0"]))
    ax2[0].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax2_0"]))
    ax2[1].set_xlabel("$\\overline{\\theta'w'}/u_{*0} \\theta_{*0}$")
    ax2[1].set_xlim(config[stat.stability]["xlim"]["ax2_1"])
    ax2[1].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax2_1"]))
    ax2[1].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax2_1"]))
    # edit ticks and add subplot labels
    for iax in ax2:
        iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
    if config["Tavg_ec"] > 1000.:
        ax2[0].text(0.03,0.05,"$\\textbf{(a)}$",fontsize=20,
                    transform=ax2[0].transAxes)
        ax2[1].text(0.85,0.05,"$\\textbf{(b)}$",fontsize=20,
                    transform=ax2[1].transAxes)
    else:
        ax2[0].text(0.03,0.05,"$\\textbf{(c)}$",fontsize=20,
                    transform=ax2[0].transAxes)
        ax2[1].text(0.85,0.05,"$\\textbf{(d)}$",fontsize=20,
                    transform=ax2[1].transAxes)
    fig2.tight_layout()
    # save and close
    fsave2 = f"{fdir_save}{stat.stability}_ustar_tw_covars_{int(config['Tavg_ec']):04d}s.pdf"
    print(f"Saving figure: {fsave2}")
    fig2.savefig(fsave2)
    plt.close(fig2)

#
# Figure 3: variances
# u_rot, v_rot, w, TKE
#
for s, stat in zip(ec_all, stat_all):
    fig3, ax3 = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(14.8, 5))
    # u'u' ROTATED
    ax3[0].plot(stat.u_var_rot.isel(z=stat.isbl)/stat.ustar0/stat.ustar0,
                stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle u'u' \\rangle$")
    ax3[0].plot(s.u_var_rot/stat.ustar0/stat.ustar0,
                s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[0].fill_betweenx(s.z/stat.h, s.err_uu_rot_lo, s.err_uu_rot_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{u'u'}$")
    ax3[0].fill_betweenx(s.z/stat.h, s.err_uu_rot_lo3, s.err_uu_rot_hi3, alpha=0.1,
                         color="r")
    # v'v' ROTATED
    ax3[1].plot(stat.v_var_rot.isel(z=stat.isbl)/stat.ustar0/stat.ustar0,
                stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle v'v' \\rangle$")
    ax3[1].plot(s.v_var_rot/stat.ustar0/stat.ustar0,
                s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[1].fill_betweenx(s.z/stat.h, s.err_vv_rot_lo, s.err_vv_rot_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{v'v'}$")
    ax3[1].fill_betweenx(s.z/stat.h, s.err_vv_rot_lo3, s.err_vv_rot_hi3, alpha=0.1,
                         color="r")
    # w'w'
    ax3[2].plot(stat.w_var.isel(z=stat.isbl)/stat.ustar0/stat.ustar0,
                stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle w'w' \\rangle$")
    ax3[2].plot(s.w_var/stat.ustar0/stat.ustar0,
                s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[2].fill_betweenx(s.z/stat.h, s.err_ww_lo, s.err_ww_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{w'w'}$")
    ax3[2].fill_betweenx(s.z/stat.h, s.err_ww_lo3, s.err_ww_hi3, alpha=0.1,
                         color="r")
    # TKE
    ax3[3].plot(stat.e.isel(z=stat.isbl)/stat.ustar0/stat.ustar0,
                stat.z.isel(z=stat.isbl)/stat.h, 
                c="k", ls="-", lw=2, label="$\\langle e \\rangle$")
    ax3[3].plot(s.e/stat.ustar0/stat.ustar0,
                s.z/stat.h, c="r", ls="-", lw=2, label="UAS")
    # shade errors
    ax3[3].fill_betweenx(s.z/stat.h, s.err_e_lo, s.err_e_hi, alpha=0.3,
                         color="r", label="$\\epsilon_{e}$")
    ax3[3].fill_betweenx(s.z/stat.h, s.err_e_lo3, s.err_e_hi3, alpha=0.1,
                         color="r")
    # clean up
    if config["Tavg_ec"] > 1000.:
        for iax, p in zip(ax3, list("abcd")):
            iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
            iax.text(0.03,0.05,f"$\\textbf{{({p})}}$",fontsize=20,
                    transform=iax.transAxes)
            # move legend in panel b for simulation F
            if ((stat.stability == "F") & (p == "b")):
                iax.legend(loc="right", labelspacing=0.10, 
                        handletextpad=0.4, shadow=True)
            else:
                iax.legend(loc="upper right", labelspacing=0.10, 
                        handletextpad=0.4, shadow=True)
    else:
        for iax, p in zip(ax3, list("efgh")):
            iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
            iax.text(0.03,0.05,f"$\\textbf{{({p})}}$",fontsize=20,
                    transform=iax.transAxes)
            # move legend in panel b for simulation F
            if ((stat.stability == "F") & (p == "f")):
                iax.legend(loc="right", labelspacing=0.10, 
                        handletextpad=0.4, shadow=True)
            else:
                iax.legend(loc="upper right", labelspacing=0.10, 
                        handletextpad=0.4, shadow=True)
    ax3[0].set_xlabel("$\\overline{u'u'}/u_{*0}^2$")
    ax3[0].set_ylabel("$z/h$")
    ax3[0].set_ylim([0, 1])
    ax3[0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax3[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax3[0].set_xlim(config[stat.stability]["xlim"]["ax3_0"])
    ax3[0].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax3_0"]))
    ax3[0].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax3_0"]))
    ax3[1].set_xlabel("$\\overline{v'v'}/u_{*0}^2$")
    ax3[1].set_xlim(config[stat.stability]["xlim"]["ax3_1"])
    ax3[1].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax3_1"]))
    ax3[1].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax3_1"]))
    ax3[2].set_xlabel("$\\overline{w'w'}/u_{*0}^2$")
    ax3[2].set_xlim(config[stat.stability]["xlim"]["ax3_2"])
    ax3[2].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax3_2"]))
    ax3[2].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax3_2"]))
    ax3[3].set_xlabel("$e/u_{*0}^2$")
    ax3[3].set_xlim(config[stat.stability]["xlim"]["ax3_3"])
    ax3[3].xaxis.set_major_locator(MultipleLocator(config[stat.stability]["xmaj"]["ax3_3"]))
    ax3[3].xaxis.set_minor_locator(MultipleLocator(config[stat.stability]["xmin"]["ax3_3"]))

    fig3.tight_layout()
    # save and close
    fsave3 = f"{fdir_save}{stat.stability}uvw_tke_vars_{int(config['Tavg_ec']):04d}s.pdf"
    print(f"Saving figure: {fsave3}")
    fig3.savefig(fsave3)
    plt.close(fig3)

# --------------------------------
# Calculate optimal ascent rates and plot
# --------------------------------
print("Begin optimal UAS ascent rate calculations")
# construct range of recalculated errors
Tnew0 = config["recalc_lo"]
Tnew1 = config["recalc_hi"]
Tnewdt = config["recalc_dt"]
Tnew = np.arange(Tnew0, Tnew1, Tnewdt, dtype=np.float64)
# recalc errors within this range for cases A and F
Aasc = recalc_err("A", Tnew)
Fasc = recalc_err("F", Tnew)
# grab error ranges for comparison
err_range = config["err_range"]
ne = len(err_range)
# delta z for averaging
delta_z = config["delta_z"]
# determine averaging time to be at/below err for each z
# loop over A and F sims
for asc in [Aasc, Fasc]:
    # create empty dataarrays within the datasets for storing
    asc["t_uh"] = xr.DataArray(np.zeros((asc.z.size, ne), dtype=np.float64),
                               coords=dict(z=asc.z, err=err_range))
    asc["t_alpha"] = xr.DataArray(np.zeros((asc.z.size, ne), dtype=np.float64),
                                  coords=dict(z=asc.z, err=err_range))
    asc["t_theta"] = xr.DataArray(np.zeros((asc.z.size, ne), dtype=np.float64),
                                  coords=dict(z=asc.z, err=err_range))
    # loop over error level
    for je, e in enumerate(err_range):
        # loop over height
        for jz in range(asc.z.size):
            # uh
            iuh = np.where(asc.uh.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(iuh) > 0:
                asc["t_uh"][jz,je] = Tnew[iuh[0]]
            else:
                asc["t_uh"][jz,je] = np.nan
            # alpha
            ialpha = np.where(asc.alpha.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(ialpha) > 0:
                asc["t_alpha"][jz,je] = Tnew[ialpha[0]]
            else:
                asc["t_alpha"][jz,je] = np.nan
            # theta
            itheta = np.where(asc.theta.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(itheta) > 0:
                asc["t_theta"][jz,je] = Tnew[itheta[0]]
            else:
                asc["t_theta"][jz,je] = np.nan
    
    # now can calculate ascent rate based on fixed delta_z
    asc["vz_uh"] = delta_z / asc.t_uh
    asc["vz_alpha"] = delta_z / asc.t_alpha
    asc["vz_theta"] = delta_z / asc.t_theta

    # also calculate minimum ascent rate to reach h in 15 minutes
    # vz_min = h / 15 min
    asc.attrs["vz_min"] = asc.h / (15. * 60)

#
# plot optimal ascent rates
#
# define colors and linestyles to loop over
colors = seaborn.color_palette("crest", 6)
lines = ["-", "--", ":", "-."]

fig4, ax4 = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, 
                        figsize=(7.4,7.5), constrained_layout=True)
# loop over error levels
for je, e in enumerate(err_range):
    # A
    # uh
    ax4[0,0].plot(Aasc.vz_uh.isel(err=je), Aasc.z/Aasc.h,
                       ls=lines[je], c=colors[0], lw=2)
    # alpha
    ax4[1,0].plot(Aasc.vz_alpha.isel(err=je), Aasc.z/Aasc.h,
                       ls=lines[je], c=colors[0], lw=2)
    # theta
    # l3 = ax4[0].plot(Aasc.vz_theta.isel(err=je), Aasc.z/Aasc.h,
    #                  ls=lines[je], c=colors[2], lw=2, label="$\\theta$")
    # F
    # uh
    ax4[0,1].plot(Fasc.vz_uh.isel(err=je), Fasc.z/Fasc.h,
                  ls=lines[je], c=colors[5], lw=2)
    # alpha
    ax4[1,1].plot(Fasc.vz_alpha.isel(err=je), Fasc.z/Fasc.h,
                  ls=lines[je], c=colors[5], lw=2)
    # theta
    # ax4[1].plot(Fasc.vz_theta.isel(err=je), Fasc.z/Fasc.h,
    #             ls=lines[je], c=colors[2], lw=2, label="$\\theta$")
# create line handles to explain linestyle for error ranges
l4=ax4[0,0].plot([], [], ls=lines[0], c="k", 
               label=f"$\\epsilon={err_range[0]*100.:3.0f}\%$")
l5=ax4[0,0].plot([], [], ls=lines[1], c="k", 
               label=f"$\\epsilon={err_range[1]*100.:3.0f}\%$")
l6=ax4[0,0].plot([], [], ls=lines[2], c="k", 
               label=f"$\\epsilon={err_range[2]*100.:3.0f}\%$")
l7=ax4[0,0].plot([], [], ls=lines[3], c="k", 
               label=f"$\\epsilon={err_range[3]*100.:3.0f}\%$")
# combine
ltot = l4 + l5 + l6 + l7
# add legend
# fig4.legend(handles=ltot, loc="upper center", ncol=5, 
#               borderaxespad=0.15,
#               columnspacing=1, bbox_to_anchor=(0.55, 1),
#               handletextpad=0.4, fontsize=14, frameon=False)
ax4[1,1].legend(handles=ltot, loc="upper right",
                labelspacing=0.10, handletextpad=0.4, shadow=True)
# clean up
for iax, p in zip(ax4.flatten(), list("abcd")):
    iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
    iax.text(0.03,0.90,f"$\\textbf{{({p})}}$",fontsize=20,
             transform=iax.transAxes)
    # vertical solid gray line for vz = 1 m/s
    iax.axvline(1, c="k", ls="-", alpha=0.5)
    # vertical red dashed line for minimum ascent rate to reach h in 15 min
    if p in "ac":
        # sim A
        iax.axvline(Aasc.vz_min, c="#800000", ls="--", alpha=0.8)
    else:
        # sim F
        iax.axvline(Fasc.vz_min, c="#800000", ls="--", alpha=0.8)
ax4[0,0].set_xlim([0.04, 10])
# ax4[0].xaxis.set_major_locator(MultipleLocator(5))
# ax4[0].xaxis.set_minor_locator(MultipleLocator(0.5))
ax4[0,0].set_xscale("log")
ax4[0,0].set_ylabel("$z/h$")
ax4[0,0].set_ylim([0, 1])
ax4[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
ax4[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax4[1,0].set_ylabel("$z/h$")
ax4[1,0].set_xlabel("Ascent Rate [m s$^{-1}$]")
ax4[1,1].set_xlabel("Ascent Rate [m s$^{-1}$]")
# fig4.tight_layout()
# save and close
fsave4 = f"{fdir_save}AF_vz_optimal.pdf"
print(f"Saving figure: {fsave4}")
fig4.savefig(fsave4)
plt.close(fig4)

# --------------------------------
# Calculate optimal ec averaging times
# --------------------------------
print("Begin calculating optimal eddy-covariance averaging times")
# construct range of recalculated errors
Tnew0ec = config["recalc_lo_ec"]
Tnew1ec = config["recalc_hi_ec"]
Tnewdtec = config["recalc_dt_ec"]
Tnew_ec = np.arange(Tnew0ec, Tnew1ec, Tnewdtec, dtype=np.float64)
# recalc errors within this range for cases A and F
Aecavg = recalc_err("A", config["Tavg_uv"], Tnew_ec)
Fecavg = recalc_err("F", config["Tavg_uv"], Tnew_ec)
# grab err_range_ec
err_range_ec = config["err_range_ec"]

# determine averaging time to be at/below err for each z
# loop over A and F sims
for ec in [Aecavg, Fecavg]:
    # create empty dataarrays within the datasets for storing
    ec["t_ustar2"] = xr.DataArray(np.zeros((ec.z.size, ne), dtype=np.float64),
                              coords=dict(z=ec.z, err=err_range_ec))
    ec["t_tw_cov_tot"] = xr.DataArray(np.zeros((ec.z.size, ne), dtype=np.float64),
                                 coords=dict(z=ec.z, err=err_range_ec))
    ec["t_uu_var_rot"] = xr.DataArray(np.zeros((ec.z.size, ne), dtype=np.float64),
                                      coords=dict(z=ec.z, err=err_range_ec))
    ec["t_vv_var_rot"] = xr.DataArray(np.zeros((ec.z.size, ne), dtype=np.float64),
                                      coords=dict(z=ec.z, err=err_range_ec))
    ec["t_ww_var"] = xr.DataArray(np.zeros((ec.z.size, ne), dtype=np.float64),
                                  coords=dict(z=ec.z, err=err_range_ec))
    ec["t_e"] = xr.DataArray(np.zeros((ec.z.size, ne), dtype=np.float64),
                             coords=dict(z=ec.z, err=err_range_ec))
    # loop over error level
    for je, e in enumerate(err_range_ec):
        # loop over height
        for jz in range(ec.z.size):
            # uh
            iustar2 = np.where(ec.ustar2.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(iustar2) > 0:
                ec["t_ustar2"][jz,je] = Tnew_ec[iustar2[0]]
            else:
                ec["t_ustar2"][jz,je] = np.nan
            # tw_cov_tot
            itw = np.where(ec.tw_cov_tot.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(itw) > 0:
                ec["t_tw_cov_tot"][jz,je] = Tnew_ec[itw[0]]
            else:
                ec["t_tw_cov_tot"][jz,je] = np.nan
            # uu_var_rot
            iuu = np.where(ec.uu_var_rot.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(iuu) > 0:
                ec["t_uu_var_rot"][jz,je] = Tnew_ec[iuu[0]]
            else:
                ec["t_uu_var_rot"][jz,je] = np.nan
            # vv_var_rot
            ivv = np.where(ec.vv_var_rot.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(ivv) > 0:
                ec["t_vv_var_rot"][jz,je] = Tnew_ec[ivv[0]]
            else:
                ec["t_vv_var_rot"][jz,je] = np.nan
            # ww_var
            iww = np.where(ec.ww_var.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(iww) > 0:
                ec["t_ww_var"][jz,je] = Tnew_ec[iww[0]]
            else:
                ec["t_ww_var"][jz,je] = np.nan
            # uh
            iee = np.where(ec.e.isel(z=jz) <= e)[0]
            # check for empty array
            if np.size(iee) > 0:
                ec["t_e"][jz,je] = Tnew_ec[iee[0]]
            else:
                ec["t_e"][jz,je] = np.nan

#
# plot covariances
#
fig5, ax5 = plt.subplots(nrows=2, ncols=3, sharex="col", sharey=True, 
                        figsize=(10,7.5), constrained_layout=True)
# loop over error levels
for je, e in enumerate(err_range_ec):
    # A
    # ustar2
    ax5[0,0].plot(Aecavg.t_ustar2.isel(err=je)/60., Aecavg.z/Aecavg.h,
                       ls=lines[je], c=colors[0], lw=2)
    # tw_cov_tot
    ax5[0,1].plot(Aecavg.t_tw_cov_tot.isel(err=je)/60., Aecavg.z/Aecavg.h,
                       ls=lines[je], c=colors[0], lw=2)
    # TKE
    ax5[0,2].plot(Aecavg.t_e.isel(err=je)/60., Aecavg.z/Aecavg.h,
                       ls=lines[je], c=colors[0], lw=2)
    # F
    # ustar2
    ax5[1,0].plot(Fecavg.t_ustar2.isel(err=je)/60., Fecavg.z/Fecavg.h,
                       ls=lines[je], c=colors[5], lw=2)
    # tw_cov_tot
    ax5[1,1].plot(Fecavg.t_tw_cov_tot.isel(err=je)/60., Fecavg.z/Fecavg.h,
                       ls=lines[je], c=colors[5], lw=2)
    # TKE
    ax5[1,2].plot(Fecavg.t_e.isel(err=je)/60., Fecavg.z/Fecavg.h,
                       ls=lines[je], c=colors[5], lw=2)

# create line handles to explain linestyle for error ranges
l4=ax5[0,0].plot([], [], ls=lines[0], c="k", 
               label=f"$\\epsilon={err_range_ec[0]*100.:3.0f}\%$")
l5=ax5[0,0].plot([], [], ls=lines[1], c="k", 
               label=f"$\\epsilon={err_range_ec[1]*100.:3.0f}\%$")
l6=ax5[0,0].plot([], [], ls=lines[2], c="k", 
               label=f"$\\epsilon={err_range_ec[2]*100.:3.0f}\%$")
l7=ax5[0,0].plot([], [], ls=lines[3], c="k", 
               label=f"$\\epsilon={err_range_ec[3]*100.:3.0f}\%$")
# combine
ltot = l4 + l5 + l6 + l7
# add legend
ax5[1,2].legend(handles=ltot, loc="upper right",
                labelspacing=0.10, handletextpad=0.4, shadow=True)
# clean up
for iax, p in zip(ax5.flatten(), list("abcdef")):
    iax.tick_params(which="both", direction="in", top=True, right=True, pad=8)
    iax.text(0.85,0.03,f"$\\textbf{{({p})}}$",fontsize=20,
             transform=iax.transAxes)
ax5[0,0].set_xlim([0, 120])
ax5[0,0].xaxis.set_major_locator(MultipleLocator(30))
ax5[0,0].xaxis.set_minor_locator(MultipleLocator(5))
# ax4[0,0].set_xscale("log")
ax5[0,0].set_ylabel("$z/h$")
ax5[0,0].set_ylim([0, 1])
ax5[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
ax5[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax5[1,0].set_ylabel("$z/h$")
ax5[1,0].set_xlabel("Averaging Time [min]")
ax5[1,1].set_xlim([0, 120])
ax5[1,1].xaxis.set_major_locator(MultipleLocator(30))
ax5[1,1].xaxis.set_minor_locator(MultipleLocator(5))
ax5[1,1].set_xlabel("Averaging Time [min]")
ax5[1,2].set_xlim([0, 60])
ax5[1,2].xaxis.set_major_locator(MultipleLocator(20))
ax5[1,2].xaxis.set_minor_locator(MultipleLocator(5))
ax5[1,2].set_xlabel("Averaging Time [min]")
# save and close
fsave5 = f"{fdir_save}AF_tavg_covar_tke.pdf"
print(f"Saving figure: {fsave5}")
fig5.savefig(fsave5)
plt.close(fig5)