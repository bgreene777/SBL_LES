# --------------------------------
# Name: sim2netcdf.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 31 August 2021
# Purpose: binary output files from LES code and combine into netcdf
# files using xarray for future reading and easier analysis
# --------------------------------
import os
import yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from simulation import read_f90_bin
# --------------------------------
# Settings and Configuration
# --------------------------------
# load yaml settings file (will be in same dir)
fyaml = "sim2netcdf.yaml"
with open(fyaml) as f:
    config = yaml.safe_load(f)

# directories and configuration
fdir = config["fdir"]
sdir = config["sdir"]
nx, ny, nz = [config["res"]] * 3
Lx, Ly, Lz = config["Lx"], config["Ly"], config["Lz"]
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
u_scale = config["uscale"]
theta_scale = config["Tscale"]
# define timestep array
timesteps = np.arange(config["t0"], config["t1"]+1, config["dt"], dtype=np.int)
nt = len(timesteps)
# dimensions
x, y = np.linspace(0., Lx, nx), np.linspace(0, Ly, ny)
z = np.linspace(dz, Lz-dz, nz)

# --------------------------------
# Loop over timesteps to load and save new files
# --------------------------------
for i in range(nt):
    # load files - DONT FORGET SCALES!
    f1 = f"{fdir}u_{timesteps[i]:07d}.out"
    u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
    f2 = f"{fdir}v_{timesteps[i]:07d}.out"
    v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
    f3 = f"{fdir}w_{timesteps[i]:07d}.out"
    w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
    f4 = f"{fdir}theta_{timesteps[i]:07d}.out"
    theta_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
    f5 = f"{fdir}txz_{timesteps[i]:07d}.out"
    txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
    f6 = f"{fdir}tyz_{timesteps[i]:07d}.out"
    tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
    f7 = f"{fdir}q3_{timesteps[i]:07d}.out"
    q3_in = read_f90_bin(f7,nx,ny,nz,8) * u_scale * theta_scale
    # construct dataset from these variables
    ds = xr.Dataset(
         {
             "u": (["x","y","z"], u_in),
             "v": (["x","y","z"], v_in),
             "w": (["x","y","z"], w_in),
             "theta": (["x","y","z"], theta_in),
             "txz": (["x","y","z"], txz_in),
             "tyz": (["x","y","z"], tyz_in),
             "q3": (["x","y","z"], q3_in)
         },
         coords={
             "x": x,
             "y": y,
             "z": z
         },
         attrs={
             "nx": nx,
             "ny": ny,
             "nz": nz,
             "Lx": Lx,
             "Ly": Ly,
             "Lz": Lz,
             "dx": dx,
             "dy": dy,
             "dz": dz,
             "stability": config["stab"]
         })
    # loop and assign attributes
    for var in config["var_attrs"].keys():
        ds[var].attrs["units"] = config["var_attrs"][var]["units"]
    # save to netcdf file and continue
    fsave = f"{sdir}all_{timesteps[i]:07d}.nc"
    print(f"Saving file: {fsave.split(os.sep)[-1]}")
    ds.to_netcdf(fsave)

print("Finished saving all files!")