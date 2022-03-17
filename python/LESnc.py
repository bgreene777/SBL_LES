# --------------------------------
# Name: LESnc.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 15 March 2022
# Purpose: collection of functions and scripts for use in the following:
# 1) processing output binary files into netcdf (replaces sim2netcdf.py)
# 2) calculating mean statistics of profiles (replaces calc_stats.py)
# 3) reading functions for binary and netcdf files
# --------------------------------
import os
import yaml
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
# --------------------------------
# Define functions
# --------------------------------
def read_f90_bin(path,nx,ny,nz,precision):
    print(f"Reading file: {path}")
    f=open(path,'rb')
    if (precision==4):
        dat=np.fromfile(f,dtype='float32',count=nx*ny*nz)
    elif (precision==8):
        dat=np.fromfile(f,dtype='float64',count=nx*ny*nz)
    else:
        raise ValueError('Precision must be 4 or 8')
    dat=np.reshape(dat,(nx,ny,nz),order='F')
    f.close()
    return dat
# ---------------------------------------------
def sim2netcdf():
    """
    Adapted from sim2netcdf.py
    Purpose: binary output files from LES code and combine into netcdf
    files using xarray for future reading and easier analysis
    """
    # directories and configuration
    dout = config["dout"]
    dnc = config["dnc"]
    # check if dnc exists
    if not os.path.exists(dnc):
        os.mkdir(dnc)
    # grab relevent parameters
    nx, ny, nz = [config["res"]] * 3
    Lx, Ly, Lz = config["Lx"], config["Ly"], config["Lz"]
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    u_scale = config["uscale"]
    theta_scale = config["Tscale"]
    # define timestep array
    timesteps = np.arange(config["t0"], config["t1"]+1, config["dt"], 
                          dtype=np.int32)
    nt = len(timesteps)
    # dimensions
    x, y = np.linspace(0., Lx, nx), np.linspace(0, Ly, ny)
    z = np.linspace(dz, Lz-dz, nz)
    # --------------------------------
    # Loop over timesteps to load and save new files
    # --------------------------------
    for i in range(nt):
        # load files - DONT FORGET SCALES!
        f1 = f"{dout}u_{timesteps[i]:07d}.out"
        u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
        f2 = f"{dout}v_{timesteps[i]:07d}.out"
        v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
        f3 = f"{dout}w_{timesteps[i]:07d}.out"
        w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
        f4 = f"{dout}theta_{timesteps[i]:07d}.out"
        theta_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
        f5 = f"{dout}txz_{timesteps[i]:07d}.out"
        txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
        f6 = f"{dout}tyz_{timesteps[i]:07d}.out"
        tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
        f7 = f"{dout}q3_{timesteps[i]:07d}.out"
        q3_in = read_f90_bin(f7,nx,ny,nz,8) * u_scale * theta_scale
        # construct dictionary of data to save
        data_save = {
                        "u": (["x","y","z"], u_in),
                        "v": (["x","y","z"], v_in),
                        "w": (["x","y","z"], w_in),
                        "theta": (["x","y","z"], theta_in),
                        "txz": (["x","y","z"], txz_in),
                        "tyz": (["x","y","z"], tyz_in),
                        "q3": (["x","y","z"], q3_in)
                    }
        # check fo using dissipation files
        if config["use_dissip"]:
            # read binary file
            f8 = f"{dout}dissip_{timesteps[i]:07d}.out"
            diss_in = read_f90_bin(f8,nx,ny,nz,8) * u_scale * u_scale * u_scale / Lz
            # add to data_save
            data_save["dissip"] = (["x","y","z"], diss_in)
        # construct dataset from these variables
        ds = xr.Dataset(
            data_save,
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
        for var in list(data_save.keys())+["x", "y", "z"]:
            ds[var].attrs["units"] = config["var_attrs"][var]["units"]
        # save to netcdf file and continue
        fsave = f"{dnc}all_{timesteps[i]:07d}.nc"
        print(f"Saving file: {fsave.split(os.sep)[-1]}")
        ds.to_netcdf(fsave)

    print("Finished saving all files!")
    return
# ---------------------------------------------
def calc_stats():
    """
    Adapted from calc_stats.py 
    Purpose: use xarray to read netcdf files created from sim2netcdf()
    and conveniently calculate statistics to output new netcdf file
    """
    # directories and configuration
    dnc = config["dnc"]
    timesteps = np.arange(config["t0"], config["t1"]+1, config["dt"], dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
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
    base = ["u", "v", "w", "theta"]
    # check for dissip
    if config["use_dissip"]:
        base.append("dissip")
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
    # rotate u_mean and v_mean so <v> = 0
    angle = np.arctan2(dd_stat.v_mean, dd_stat.u_mean)
    dd_stat["alpha"] = angle
    dd_stat["u_mean_rot"] = dd_stat.u_mean*np.cos(angle) + dd_stat.v_mean*np.sin(angle)
    dd_stat["v_mean_rot"] =-dd_stat.u_mean*np.sin(angle) + dd_stat.v_mean*np.cos(angle)
    # rotate instantaneous u and v
    u_rot = dd.u*np.cos(angle) + dd.v*np.sin(angle)
    v_rot =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
    # recalculate u_var_rot, v_var_rot
    dd_stat["u_var_rot"] = u_rot.var(dim=("time", "x", "y"))
    dd_stat["v_var_rot"] = v_rot.var(dim=("time", "x", "y"))
    # --------------------------------
    # Add attributes
    # --------------------------------
    # copy from dd
    dd_stat.attrs = dd.attrs
    dd_stat.attrs["delta"] = (dd.dx * dd.dy * dd.dz) ** (1./3.)
    dd_stat.attrs["tavg"] = config["tavg"]
    # --------------------------------
    # Save output file
    # --------------------------------
    fsave = f"{dnc}{config['fstats']}"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_stat.to_netcdf(fsave, mode="w")
    print("Finished!")
    return
# ---------------------------------------------
def load_stats(fstats, SBL=True, display=False):
    """
    Reading function for average statistics files created from calc_stats()
    Load netcdf files using xarray and calculate numerous relevant parameters
    input fstats: absolute path to netcdf file for reading
    input display: boolean flag to print statistics from files, default=False
    return dd: xarray dataset
    """
    print(f"Reading file: {fstats}")
    dd = xr.load_dataset(fstats)
    # calculate ustar and h
    dd["ustar"] = ((dd.uw_cov_tot**2.) + (dd.vw_cov_tot**2.)) ** 0.25
    dd["ustar2"] = dd.ustar ** 2.
    if SBL:
        dd["h"] = dd.z.where(dd.ustar2 <= 0.05*dd.ustar2[0], drop=True)[0] / 0.95
    else:
        dd["h"] = 0. # TODO: fix this later
    # grab ustar0 and calc tstar0 for normalizing in plotting
    dd["ustar0"] = dd.ustar.isel(z=0)
    dd["tstar0"] = -dd.tw_cov_tot.isel(z=0)/dd.ustar0
    # calculate Obukhov length L
    dd["L"] = -(dd.ustar0**3) * dd.theta_mean.isel(z=0) / (0.4 * 9.81 * dd.tw_cov_tot.isel(z=0))
    if SBL:
        # calculate Richardson numbers
        # sqrt((du_dz**2) + (dv_dz**2))
        dd["du_dz"] = np.sqrt(dd.u_mean.differentiate("z", 2)**2. + dd.v_mean.differentiate("z", 2)**2.)
        # Rig = N^2 / S^2
        dd["N2"] = dd.theta_mean.differentiate("z", 2) * 9.81 / dd.theta_mean.isel(z=0)
        dd["Rig"] = dd.N2 / dd.du_dz / dd.du_dz
        # Rif = beta * w'theta' / (u'w' du/dz + v'w' dv/dz)
        dd["Rif"] = (9.81/dd.theta_mean.isel(z=0)) * dd.tw_cov_tot /\
                                (dd.uw_cov_tot*dd.u_mean.differentiate("z", 2) +\
                                dd.vw_cov_tot*dd.v_mean.differentiate("z", 2))
        # calc Ozmidov scale real quick
        dd["Lo"] = np.sqrt(-dd.dissip_mean / (dd.N2 ** (3./2.)))
        # calculate gradient scales from Sorbjan 2017, Greene et al. 2022
        l0 = 19.22 # m
        l1 = 1./(dd.Rig**(3./2.)).where(dd.z <= dd.h, drop=True)
        kz = 0.4 * dd.z.where(dd.z <= dd.h, drop=True)
        dd["Ls"] = kz / (1 + (kz/l0) + (kz/l1))
        dd["Us"] = dd.Ls * np.sqrt(dd.N2)
        dd["Ts"] = dd.Ls * dd.theta_mean.differentiate("z", 2)
    # calculate uh and wdir
    dd["uh"] = np.sqrt(dd.u_mean**2. + dd.v_mean**2.)
    dd["wdir"] = np.arctan2(-dd.u_mean, -dd.v_mean) * 180./np.pi
    dd["wdir"] = dd.wdir.where(dd.wdir < 0.) + 360.
    # calculate mean lapse rate between lowest grid point and z=h
    delta_T = dd.theta_mean.sel(z=dd.h, method="nearest") - dd.theta_mean[0]
    delta_z = dd.z.sel(z=dd.h, method="nearest") - dd.z[0]
    dd["dT_dz"] = delta_T / delta_z
    # calculate eddy turnover time TL
    dd["TL"] = dd.h / dd.ustar0
    dd["nTL"] = 3600. / dd.TL
    # calculate TKE
    dd["e"] = 0.5 * (dd.u_var + dd.v_var + dd.w_var)
    # print table statistics
    if display:
        print(f"---{dd.stability}---")
        print(f"u*: {dd.ustar0.values:4.3f} m/s")
        print(f"theta*: {dd.tstar0.values:5.4f} K")
        print(f"Q*: {1000*dd.tw_cov_tot.isel(z=0).values:4.3f} K m/s")
        print(f"h: {dd.h.values:4.3f} m")
        print(f"L: {dd.L.values:4.3f} m")
        print(f"h/L: {(dd.h/dd.L).values:4.3f}")
        print(f"zj/h: {(dd.z.isel(z=dd.uh.argmax())/dd.h).values:4.3f}")
        print(f"dT/dz: {1000*dd.dT_dz.values:4.1f} K/km")
        print(f"TL: {dd.TL.values:4.1f} s")
        print(f"nTL: {dd.nTL.values:4.1f}")

    return dd
# --------------------------------
# Run script if desired
# --------------------------------
if __name__ == "__main__":
    # load yaml file in global scope
    fyaml = "/home/bgreene/SBL_LES/python/LESnc.yaml"
    with open(fyaml) as f:
        config = yaml.safe_load(f)
    # run sim2netcdf
    if config["run_sim2netcdf"]:
        sim2netcdf()
    # run calc_stats
    if config["run_calc_stats"]:
        calc_stats()
