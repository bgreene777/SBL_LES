# --------------------------------
# Name: timeseries_to_netcdf.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 22 November 2021
# Purpose: Read timeseries files from each simulation and save individual
# files for each stability for the *last 1 hr of simulation* to save space
# --------------------------------
import os
import numpy as np
import xarray as xr
from RFMnc import print_both
from dask.diagnostics import ProgressBar
# --------------------------------
# Define directories and constants
# --------------------------------
fsim = "/home/bgreene/simulations/"
fprint = "/home/bgreene/SBL_LES/output/Print/timeseries_nc.txt"
u_scale = 0.4
T_scale = 300.
nz = 192
Lz = 400.
dz = Lz/nz
# define z array
z = np.linspace(dz, Lz-dz, nz, dtype=np.float64)  # meters
# grab last hour of simulation
# for this group, there are 720k timesteps with dt=0.02 seconds
dt = 0.02  # s
nt_tot = 720000
# 1 hour is 3600/dt
nt = int(3600./dt)
istart = nt_tot - nt
# define time in seconds
time = np.linspace(0., 3600.-dt, nt, dtype=np.float64)  # seconds

# --------------------------------
# Begin big loop over stabilities and timesteps
# --------------------------------

for stab in list("ABCDEF"):
    print_both(f"Begin loading simulation {stab}", fprint)
    fout = f"{fsim}{stab}_192_interp/output/"
    # define empty arrays to use with DataArrays 
    empty = np.zeros((nt, nz), dtype=np.float64)  # shape(nt,nz)
    # now define DataArrays for u, v, w, theta, txz, tyz, q3
    u_ts, v_ts, w_ts, theta_ts, txz_ts, tyz_ts, q3_ts =\
    (xr.DataArray(empty, dims=("t", "z"), coords=dict(t=time, z=z)) for _ in range(7))
    # now loop through each file (one for each jz)
    for jz in range(nz):
        print_both(f"Loading timeseries data, jz={jz}", fprint)
        fu = f"{fout}u_timeseries_c{jz:03d}.out"
        u_ts[:,jz] = np.loadtxt(fu, skiprows=istart, usecols=1)
        fv = f"{fout}v_timeseries_c{jz:03d}.out"
        v_ts[:,jz] = np.loadtxt(fv, skiprows=istart, usecols=1)
        fw = f"{fout}w_timeseries_c{jz:03d}.out"
        w_ts[:,jz] = np.loadtxt(fw, skiprows=istart, usecols=1)
        ftheta = f"{fout}t_timeseries_c{jz:03d}.out"
        theta_ts[:,jz] = np.loadtxt(ftheta, skiprows=istart, usecols=1)
        ftxz = f"{fout}txz_timeseries_c{jz:03d}.out"
        txz_ts[:,jz] = np.loadtxt(ftxz, skiprows=istart, usecols=1)
        ftyz = f"{fout}tyz_timeseries_c{jz:03d}.out"
        tyz_ts[:,jz] = np.loadtxt(ftyz, skiprows=istart, usecols=1)
        fq3 = f"{fout}q3_timeseries_c{jz:03d}.out"
        q3_ts[:,jz] = np.loadtxt(fq3, skiprows=istart, usecols=1)
    # apply scales
    u_ts *= u_scale
    v_ts *= u_scale
    w_ts *= u_scale
    theta_ts *= T_scale
    txz_ts *= (u_scale * u_scale)
    tyz_ts *= (u_scale * u_scale)
    q3_ts *= (u_scale * T_scale)
    # define dictionary of attributes
    attrs = {"stability": stab, "dt": dt, "nt": nt, "nz": nz, "total_time": "1hr"}
    # combine DataArrays into Dataset and save as netcdf
    # initialize empty Dataset
    ts_all = xr.Dataset(data_vars=None, coords=dict(t=time, z=z), attrs=attrs)
    # now store
    ts_all["u"] = u_ts
    ts_all["v"] = v_ts
    ts_all["w"] = w_ts
    ts_all["theta"] = theta_ts
    ts_all["txz"] = txz_ts
    ts_all["tyz"] = tyz_ts
    ts_all["q3"] = q3_ts
    # save to netcdf
    fsave_ts = f"{fout}netcdf/timeseries_all.nc"
    with ProgressBar():
        ts_all.to_netcdf(fsave_ts, mode="w")
        
print_both("Finished saving all simulations!", fprint)
    