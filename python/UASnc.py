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
def profile(df, ascent_rate=1.0, time_average=3.0, time_start=0.0):
    """Emulate a vertical profile from rotary-wing UAS sampling through
    simulated SBL, including random errors
    :param xr.Dataset df: dataset with virtual tower data to construct UAS prof
    :param float ascent_rate: UAS ascent rate in m/s; default=1.0
    :param float time_average: time range in s to avg UAS profile; default=3.0 s
    :param float time_start: when to initialize ascending profile; default=0.0 s
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
    interp_keys = ["u", "v", "w", "theta"]
    d_interp = {} # define empty dictionary for looping
    for key in interp_keys:
        print(f"Interpolating {key}...")
        d_interp[key] = interp_uas(df[key].isel(t=iuse).values,
                                   df.z.values, zuas)

    # grab data from interpolated arrays to create simulated raw UAS profiles
    # define xarray dataset to eventually store all
    uas_raw = xr.Datset(data_vars=None, coords=dict(z=zuas))
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
    uas_mean = uas_mean.assign_coords({"z": ("z_bins", znew)}).swap_dims({"z_bins", "z"})
    
    #
    # TODO: calculate uh, alpha from mean profile; calculate errors and return
    #
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

# load timeseries files
ftsA = f"{fsim}A_192_interp/output/netcdf/timeseries_all.nc"
Ats = xr.load_dataset(ftsA)
# ftsF = f"{fsim}F_192_interp/output/netcdf/timeseries_all.nc"
# Fts = xr.load_dataset(ftsF)

profile(Ats)

