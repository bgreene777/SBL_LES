# --------------------------------
# Name: calc_stats.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 2 September 2021
# Purpose: use xarray to read netcdf files created from sim2netcdf.py
# and conveniently calculate statistics to output new netcdf file
# --------------------------------
import os
import yaml
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
# --------------------------------
# Settings and Configuration
# --------------------------------
# load yaml settings file (will be in same dir)
fyaml = "calc_stats.yaml"
with open(fyaml) as f:
    config = yaml.safe_load(f)

# directories and configuration
fdir = config["fdir"]
timesteps = np.arange(config["t0"], config["t1"]+1, config["dt"], dtype=np.int32)
# determine files to read from timesteps
fall = [f"{fdir}all_{tt:07d}.nc" for tt in timesteps]
nf = len(fall)
# calculate array of times represented by each file
times = np.array([i*config["delta_t"]*config["dt"] for i in range(nf)])

# --------------------------------
# Load files and clean up
# --------------------------------
print("Reading files...")
dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
dd.coords["time"] = times
dd.time.attrs["units"] = "s"

# --------------------------------
# Calculate statistics
# --------------------------------
print("Beginning calculations")
# create empty dataset that will hold everything
dd_stat = xr.Dataset()
# list of base variables
base = ["u", "v", "w", "theta", "dissip"]
# calculate means
for s in base:
    dd_stat[f"{s}_mean"] = dd[s].mean(dim=("time", "x", "y"))
# calculate covars
# u'w'
dd_stat["uw_cov_res"] = xr.cov(dd.u, dd.w, dim=("time", "x", "y"))
dd_stat["uw_cov_tot"] = dd_stat.uw_cov_res + dd.txz.mean(dim=("time","x","y"))
# v'w'
dd_stat["vw_cov_res"] = xr.cov(dd.v, dd.w, dim=("time", "x", "y"))
dd_stat["vw_cov_tot"] = dd_stat.vw_cov_res + dd.tyz.mean(dim=("time","x","y"))
# theta'w'
dd_stat["tw_cov_res"] = xr.cov(dd.theta, dd.w, dim=("time", "x", "y"))
dd_stat["tw_cov_tot"] = dd_stat.tw_cov_res + dd.q3.mean(dim=("time","x","y"))
# calculate vars
for s in base[:-1]:
    dd_stat[f"{s}_var"] = dd[s].var(dim=("time", "x", "y"))
    
# --------------------------------
# Save output file
# --------------------------------
fsave = f"{fdir}{config['fsave']}"
print(f"Saving file: {fsave}")
with ProgressBar():
    dd_stat.to_netcdf(fsave, mode="w")
print("Finished!")