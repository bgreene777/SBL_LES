#!/home/bgreene/anaconda3/bin/python
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
import sys
import yaml
import xrft
import xarray as xr
import numpy as np
from numpy.fft import fft, ifft
from datetime import datetime
from scipy.signal import detrend
from scipy.optimize import curve_fit
from dask.diagnostics import ProgressBar
from matplotlib.colors import Normalize
# --------------------------------
# Define plotting class for custom diverging colorbars
# --------------------------------
class MidPointNormalize(Normalize):
    """Defines the midpoint of diverging colormap.
    Usage: Allows one to adjust the colorbar, e.g. 
    using contouf to plot data in the range [-3,6] with
    a diverging colormap so that zero values are still white.
    Example usage:
        norm=MidPointNormalize(midpoint=0.0)
        f=plt.contourf(X,Y,dat,norm=norm,cmap=colormap)
     """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
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
def print_both(s, fsave):
    """
    Print statements to both the command line (sys.stdout) and to a
    text file (fsave) for future reference on running time
    -input-
    s: string to print
    fsave: text file to append text
    """
    with open(fsave, "a") as f:
        # print to command line
        print(s, file=sys.stdout)
        # print to file with a UTC timestamp
        print(datetime.utcnow(), file=f)
        print(s, file=f)
    return
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
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
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
        # interpolate w, txz, tyz, q3 to u grid
        # create DataArrays
        w_da = xr.DataArray(w_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        txz_da = xr.DataArray(txz_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        tyz_da = xr.DataArray(tyz_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        q3_da = xr.DataArray(q3_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        # perform interpolation
        w_interp = w_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
        txz_interp = txz_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
        tyz_interp = tyz_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
        q3_interp = q3_da.interp(z=zu, method="linear", 
                                 kwargs={"fill_value": "extrapolate"})
        # construct dictionary of data to save -- u-node variables only!
        data_save = {
                        "u": (["x","y","z"], u_in),
                        "v": (["x","y","z"], v_in),
                        "theta": (["x","y","z"], theta_in),
                    }
        # check fo using dissipation files
        if config["use_dissip"]:
            # read binary file
            f8 = f"{dout}dissip_{timesteps[i]:07d}.out"
            diss_in = read_f90_bin(f8,nx,ny,nz,8) * u_scale * u_scale * u_scale / Lz
            # interpolate to u-nodes
            diss_da = xr.DataArray(diss_in, dims=("x", "y", "z"), 
                                   coords=dict(x=x, y=y, z=zw))
            diss_interp = diss_da.interp(z=zu, method="linear", 
                                         kwargs={"fill_value": "extrapolate"})
        # construct dataset from these variables
        ds = xr.Dataset(
            data_save,
            coords={
                "x": x,
                "y": y,
                "z": zu
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
        # now assign interpolated arrays that were on w-nodes
        ds["w"] = w_interp
        ds["txz"] = txz_interp
        ds["tyz"] = tyz_interp
        ds["q3"] = q3_interp
        if config["use_dissip"]:
            ds["dissip"] = diss_interp
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
    base1 = ["u", "v", "w", "theta"] # use for looping over vars in case dissip not used
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
    for s in base1:
        if config["detrend_stats"]:
            vv = np.var(detrend(dd[s], axis=0, type="linear"), axis=(0,1,2))
            dd_stat[f"{s}_var"] = xr.DataArray(vv, dims=("z"), coords=dict(z=dd.z))
        else:
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
    if config["detrend_stats"]:
        uvar_rot = np.var(detrend(u_rot, axis=0, type="linear"), axis=(0,1,2))
        dd_stat["u_var_rot"] = xr.DataArray(uvar_rot, dims=("z"), coords=dict(z=dd.z))
        vvar_rot = np.var(detrend(v_rot, axis=0, type="linear"), axis=(0,1,2))
        dd_stat["v_var_rot"] = xr.DataArray(vvar_rot, dims=("z"), coords=dict(z=dd.z))
    else:
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
def timeseries2netcdf():
    dout = config["dout"]
    dnc = config["dnc"]
    fprint = config["fprint"]
    # grab relevent parameters
    u_scale = config["uscale"]
    theta_scale = config["Tscale"]
    delta_t = config["delta_t"]
    nz = config["res"]
    Lz = config["Lz"]
    dz = Lz/nz
    # define z array
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # only load last hour of simulation
    nt_tot = config["tf"]
    # determine number of hours to process from tavg
    tavg = config["tavg"]
    nhr = float(tavg.split("h")[0])
    nt = int(nhr*3600./delta_t)
    istart = nt_tot - nt
    # define array of time in seconds
    time = np.linspace(0., nhr*3600.-delta_t, nt, dtype=np.float64)
    print_both(f"Loading {nt} timesteps = {tavg}", fprint)
    # begin looping over heights
    print_both(f"Begin loading simulation {config['stab']}", fprint)   
    # define DataArrays for u, v, w, theta, txz, tyz, q3
    # shape(nt,nz)
    # u, v, theta
    u_ts, v_ts, theta_ts =\
    (xr.DataArray(np.zeros((nt, nz), dtype=np.float64),
                  dims=("t", "z"), coords=dict(t=time, z=zu)) for _ in range(3))
    # w, txz, tyz, q3
    w_ts, txz_ts, tyz_ts, q3_ts =\
    (xr.DataArray(np.zeros((nt, nz), dtype=np.float64),
                  dims=("t", "z"), coords=dict(t=time, z=zw)) for _ in range(4))
    # now loop through each file (one for each jz)
    for jz in range(nz):
        print_both(f"Loading timeseries data, jz={jz}", fprint)
        fu = f"{dout}u_timeseries_c{jz:03d}.out"
        u_ts[:,jz] = np.loadtxt(fu, skiprows=istart, usecols=1)
        fv = f"{dout}v_timeseries_c{jz:03d}.out"
        v_ts[:,jz] = np.loadtxt(fv, skiprows=istart, usecols=1)
        fw = f"{dout}w_timeseries_c{jz:03d}.out"
        w_ts[:,jz] = np.loadtxt(fw, skiprows=istart, usecols=1)
        ftheta = f"{dout}t_timeseries_c{jz:03d}.out"
        theta_ts[:,jz] = np.loadtxt(ftheta, skiprows=istart, usecols=1)
        ftxz = f"{dout}txz_timeseries_c{jz:03d}.out"
        txz_ts[:,jz] = np.loadtxt(ftxz, skiprows=istart, usecols=1)
        ftyz = f"{dout}tyz_timeseries_c{jz:03d}.out"
        tyz_ts[:,jz] = np.loadtxt(ftyz, skiprows=istart, usecols=1)
        fq3 = f"{dout}q3_timeseries_c{jz:03d}.out"
        q3_ts[:,jz] = np.loadtxt(fq3, skiprows=istart, usecols=1)
    # apply scales
    u_ts *= u_scale
    v_ts *= u_scale
    w_ts *= u_scale
    theta_ts *= theta_scale
    txz_ts *= (u_scale * u_scale)
    tyz_ts *= (u_scale * u_scale)
    q3_ts *= (u_scale * theta_scale)
    # define dictionary of attributes
    attrs = {"stability": config["stab"], "dt": delta_t, "nt": nt, "nz": nz, "total_time": config["tavg"]}
    # combine DataArrays into Dataset and save as netcdf
    # initialize empty Dataset
    ts_all = xr.Dataset(data_vars=None, coords=dict(t=time, z=zu), attrs=attrs)
    # now store
    ts_all["u"] = u_ts
    ts_all["v"] = v_ts
    ts_all["w"] = w_ts.interp(z=zu, method="linear", 
                              kwargs={"fill_value": "extrapolate"})
    ts_all["theta"] = theta_ts
    ts_all["txz"] = txz_ts.interp(z=zu, method="linear", 
                                  kwargs={"fill_value": "extrapolate"})
    ts_all["tyz"] = tyz_ts.interp(z=zu, method="linear", 
                                  kwargs={"fill_value": "extrapolate"})
    ts_all["q3"] = q3_ts.interp(z=zu, method="linear", 
                                kwargs={"fill_value": "extrapolate"})
    # save to netcdf
    fsave_ts = f"{dnc}{config['fts']}"
    with ProgressBar():
        ts_all.to_netcdf(fsave_ts, mode="w")
        
    print_both("Finished saving all simulations!", fprint)

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
    # define new label attr that removes "_"
    dd.attrs["label"] = "-".join(dd.stability.split("_"))
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
    # local thetastar
    dd["tstar"] = -dd.tw_cov_tot / dd.ustar
    # calculate TKE
    dd["e"] = 0.5 * (dd.u_var + dd.v_var + dd.w_var)
    # calculate Obukhov length L
    dd["L"] = -(dd.ustar0**3) * dd.theta_mean.isel(z=0) / (0.4 * 9.81 * dd.tw_cov_tot.isel(z=0))
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
    if SBL:
        # calculate TKE-based sbl depth
        dd["he"] = dd.z.where(dd.e <= 0.05*dd.e[0], drop=True)[0]
        # calculate h/L as global stability parameter
        dd["hL"] = dd.he / dd.L
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
        # calculate local Obukhov length Lambda
        dd["LL"] = -(dd.ustar**3.) * dd.theta_mean / (0.4 * 9.81 * dd.tw_cov_tot)
        # calculate level of LLJ: zj
        dd["zj"] = dd.z.isel(z=dd.uh.argmax())
        # save number of points within sbl
        dd.attrs["nzsbl"] = dd.z.where(dd.z <= dd.he, drop=True).size
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
# ---------------------------------------------
def load_full(dnc, t0, t1, dt, delta_t, use_stats, SBL):
    """
    Reading function for multiple instantaneous volumetric netcdf files
    Load netcdf files using xarray
    input dnc: string path directory for location of netcdf files
    input t0, t1, dt: start, end, and spacing of file names
    input delta_t: simulation timestep in seconds
    input use_stats: optional flag to use statistics file for u,v rotation
    input SBL: flag for calculating SBL parameters
    return dd: xarray dataset of 4d volumes
    return s: xarray dataset of statistics file
    """
    # load final hour of individual files into one dataset
    # note this is specific for SBL simulations
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
    # read files
    print("Loading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    if use_stats:
        # load stats file
        s = load_stats(dnc+"average_statistics.nc", SBL=SBL)
        # calculate rotated u, v based on alpha in stats
        dd["u_rot"] = dd.u*np.cos(s.alpha) + dd.v*np.sin(s.alpha)
        dd["v_rot"] =-dd.u*np.sin(s.alpha) + dd.v*np.cos(s.alpha)
        # return both dd and s
        return dd, s
    # just return dd if no SBL
    return dd
# ---------------------------------------------
def load_timeseries(dnc, detrend=True, tavg="1h"):
    """
    Reading function for timeseries files created from timseries2netcdf()
    Load netcdf files using xarray and calculate numerous relevant parameters
    input dnc: path to netcdf directory for simulation
    input detrend: detrend timeseries for calculating variances, default=True
    input tavg: select which timeseries file to use in hours, default="1h"
    return d: xarray dataset
    """
    # load timeseries_all.nc
    if tavg == "1h":
        fts = "timeseries_all.nc"
    else:
        fts = f"timeseries_all_{tavg}r.nc"

    d = xr.open_dataset(dnc+fts)
    # calculate means
    for v in ["u", "v", "w", "theta"]:
        d[f"{v}_mean"] = d[v].mean("t") # average in time
    # rotate coords so <v> = 0
    angle = np.arctan2(d.v_mean, d.u_mean)
    d["u_mean_rot"] = d.u_mean*np.cos(angle) + d.v_mean*np.sin(angle)
    d["v_mean_rot"] =-d.u_mean*np.sin(angle) + d.v_mean*np.cos(angle)
    # rotate instantaneous u and v
    d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
    d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
    # calculate "inst" covars
    # covars not affected by detrend
    d["uw"] = (d.u - d.u_mean) * (d.w - d.w_mean) + d.txz
    d["vw"] = (d.v - d.v_mean) * (d.w - d.w_mean) + d.tyz
    d["tw"] = (d.theta - d.theta_mean) * (d.w - d.w_mean) + d.q3
    # calculate "inst" vars
    if detrend:
        ud = xrft.detrend(d.u, dim="t", detrend_type="linear")
        udr = xrft.detrend(d.u_rot, dim="t", detrend_type="linear")
        vd = xrft.detrend(d.v, dim="t", detrend_type="linear")
        vdr = xrft.detrend(d.v_rot, dim="t", detrend_type="linear")
        wd = xrft.detrend(d.w, dim="t", detrend_type="linear")
        td = xrft.detrend(d.theta, dim="t", detrend_type="linear")
        # store these detrended variables
        d["ud"] = ud
        d["udr"] = udr
        d["vd"] = vd
        d["vdr"] = vdr
        d["wd"] = wd
        d["td"] = td
        # now calculate vars
        d["uu"] = ud * ud
        d["uur"] = udr * udr
        d["vv"] = vd * vd
        d["vvr"] = vdr * vdr
        d["ww"] = wd * wd
        d["tt"] = td * td
    else:
        d["uu"] = (d.u - d.u_mean) * (d.u - d.u_mean)
        d["uur"] = (d.u_rot - d.u_mean_rot) * (d.u_rot - d.u_mean_rot)
        d["vv"] = (d.v - d.v_mean) * (d.v - d.v_mean)
        d["vvr"] = (d.v_rot - d.v_mean_rot) * (d.v_rot - d.v_mean_rot)
        d["ww"] = (d.w - d.w_mean) * (d.w - d.w_mean)
        d["tt"] = (d.theta - d.theta_mean) * (d.theta - d.theta_mean)
    
    return d
# ---------------------------------------------
def autocorr_from_timeseries(dnc, savenc=True, nblock=2):
    """
    Calculate autocorrelation and corresponding integral length/timescales
    for first- and second-order parameters based on timeseries data
    input dnc: directory where netcdf data stored
    input savenc: flag to save netcdf files of output data
    input nblock: number of blocks to segment timeseries data for averaging
    output R: xarray Dataset of autocorrelation versus z and t (lag)
    output L: xarray Dataset of integral lengthscales versus z
    """
    # define exponential autocorrelation function for curve fitting
    def acf(time, Tint):
        return np.exp(-np.abs(time) / Tint)

    # filename for print_both function
    fprint = f"/home/bgreene/SBL_LES/output/Print/autocorr_ts_{config['stab']}.txt"
    # begin by loading timeseries data
    print_both("Begin loading timeseries data", fprint)
    ts = load_timeseries(dnc, detrend=True, tavg=config['tavg'])
    # grab important parameters
    nz = ts.nz
    nt_tot = ts.nt
    nt = nt_tot // nblock
    # define list of first-order parameters to calculate
    param1 = ["udr", "vdr", "wd", "td"] # u_rot, v_rot, w, theta
    var1 = ["uur", "vvr", "ww", "tt"] # corresponding instantaneous variances
    name1 = ["u", "v", "w", "theta"] # names for saving variables
    # define empty list to hold all R
    R_all = []
    # begin looping over blocks
    for ib in range(nblock):
        print_both(f"Block {ib}", fprint)
        # define indices for block
        jt0 = ib * nt
        jt1 = nt + jt0
        jts = np.arange(jt0, jt1, 1, dtype=np.int32)
        xt = ts.t.isel(t=jts)
        # define empty Dataset R to hold autocorrs
        R = xr.Dataset(data_vars=None, coords=dict(t=xt, z=ts.z), attrs=ts.attrs)
        # loop over variables and calculate autocorrelation
        for s, svar, sname in zip(param1, var1, name1):
            print_both(f"Begin calculating autocorr for {sname}", fprint)
            # grab data
            x = ts[s].isel(t=jts)
            xvar = ts[svar].isel(t=jts).mean(dim="t")
            # store this variance
            # ts[svar+"var1"] = xvar
            # forward FFT in time
            f = fft(x, axis=0)
            # calculate PSD
            PSD = np.zeros((nt, nz), dtype=np.float64)
            for jt in range(1, nt//2):
                for jz in range(nz):
                    PSD[jt,jz] = np.real( f[jt,jz] * np.conj(f[jt,jz]) )
                    PSD[nt-jt,jz] = np.real( f[nt-jt,jz] * np.conj(f[nt-jt,jz]) )
            # normalize by variance
            for jz in range(nz):
                PSD[:,jz] /= xvar[jz].values
            # ifft to get autocorrelation
            # normalize by length of timeseries
            r = np.real( ifft(PSD, axis=0) ) / nt
            # convert to DataArray and assign to R
            R[sname] = xr.DataArray(r, dims=("t", "z"), coords=dict(t=xt, z=ts.z))
        # append R to R_all
        R_all.append(R)
    # print_both("Finished calculating 1st-order autocorrelations!", fprint)
    # # calculate autocorr for 2nd-order: variances
    # param2 = ["uur", "vvr", "ww", "tt"] # instantaneous variances
    # var2 = ["uurvar4", "vvrvar4", "wwvar4", "ttvar4"] # corresponding 4th-order variances
    # name2 = ["uuvar2", "vvvar2", "wwvar2", "ttvar2"] # names for saving autocorr variables
    # # need to calculate 4th-order variances (var{var{x}}) for normalization
    # print_both("Calculate 4th-order variances", fprint)
    # for s, svar in zip(param2, var2):
    #     xvar = ts[s+"var1"] # grab variances
    #     xvar4 = (ts[s] - xvar) * (ts[s] - xvar) # calculate inst variance of variances
    #     ts[svar] = xvar4.mean("t") # save mean value
    # # calculate autocorr for 2nd-order: covariances
    # param2c = ["uw", "vw", "tw"] # instantaneous covariances
    # var2c = ["uwvar4", "vwvar4", "twvar4"] # corresponding 4th-order variances
    # name2c = ["uwcov2", "vwcov2", "twcov2"] # names for saving autocorr variables
    # # need to calculate 4th-order variances of covar (var{cov{x}}) for normalization
    # for s, svar in zip(param2c, var2c):
    #     xcov = ts[s].mean("t") # grab inst covar and average to get mean cov
    #     xvar4 = (ts[s] - xcov) * (ts[s] - xcov) # calculate inst variance of variances
    #     ts[svar] = xvar4.mean("t") # save mean value
    # # now can calculate autocorr
    # for s, svar, sname in zip(param2+param2c, var2+var2c, name2+name2c):
    #     print_both(f"Begin calculating autocorr for {sname}", fprint)
    #     # grab data
    #     x = ts[s] # inst variances/covar
    #     xvar = ts[svar] # variance of variance/covar (4th order)
    #     # forward FFT in time
    #     f = fft(x, axis=0)
    #     # calculate PSD
    #     PSD = np.zeros((nt, nz), dtype=np.float64)
    #     for jt in range(1, nt//2):
    #         for jz in range(nz):
    #             PSD[jt,jz] = np.real( f[jt,jz] * np.conj(f[jt,jz]) )
    #             PSD[nt-jt,jz] = np.real( f[nt-jt,jz] * np.conj(f[nt-jt,jz]) )
    #     # normalize by variance
    #     for jz in range(nz):
    #         PSD[:,jz] /= xvar[jz].values
    #     # ifft to get autocorrelation
    #     # normalize by length of timeseries
    #     r = np.real( ifft(PSD, axis=0) ) / nt
    #     # convert to DataArray and assign to R
    #     R[sname] = xr.DataArray(r, dims=("t", "z"), coords=dict(t=ts.t, z=ts.z))        

    # now calculate integral scales
    # define empty Dataset T to hold timescales
    T = xr.Dataset(data_vars=None, coords=dict(z=R.z), attrs=R.attrs)
    # define empty Dataset L to hold lengthscales
    L = xr.Dataset(data_vars=None, coords=dict(z=R.z), attrs=R.attrs)
    # loop over parameters
    for sname in name1:#+name2+name2c:
        print_both(f"Calculate integral scales for {sname}", fprint)
        T_sname = np.zeros(R.z.size, np.float64)
        # loop over heights
        for jz in range(nz):
            # loop over blocks
            TT_count = 0.
            for R in R_all:
                # fit autocorr function to form R = exp(-t/Tint)
                # store in T_sname
                TT, _ = curve_fit(f=acf, xdata=R.t[:nt//2], 
                                  ydata=R[sname].isel(z=jz,t=range(nt//2)))
                TT_count += TT
            T_sname[jz] = TT_count / nblock
        
        # add to T dataset
        T[sname] = xr.DataArray(T_sname, dims="z", coords=dict(z=R.z))
        # calculate lengthscales using Taylor's hypothesis
        L_sname = T[sname] * ts.u_mean_rot
        # add to L dataset
        L[sname] = xr.DataArray(L_sname, dims="z", coords=dict(z=R.z))

    # now optionally save out netcdf files
    if savenc:
        fsaveR = f"{dnc}R_ts_{nblock}.nc"
        print_both(f"Saving file: {fsaveR}", fprint)
        with ProgressBar():
            R.to_netcdf(fsaveR, mode="w")
        fsaveL = f"{dnc}L_ts_{nblock}.nc"
        print_both(f"Saving file: {fsaveL}", fprint)
        with ProgressBar():
            L.to_netcdf(fsaveL, mode="w")
    
    print_both("Finshed with all calculations! Returning [R, L]...", fprint)
    return [R, L]
# ---------------------------------------------
def autocorr_from_volume():
    # define print file
    fprint = f"/home/bgreene/SBL_LES/output/Print/autocorr_vol_{config['stab']}.txt"
    # grab parameters from yaml file
    dnc = config["dnc"]
    t0 = config["t0"]
    t1 = config["t1"]
    dt = config["dt"]
    nint = config["nint"]
    # timestep array for loading files
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # new size of arrays to use after spectral interpolation
    nx2 = nint * s.nx
    # calculate atocorrelation function along x-dimension as pencils
    # integrate to calc length scale then average in y and t
    # initialize empty dataset for storing integral scales
    Lsave = xr.Dataset(data_vars=None, coords=dict(z=s.z[range(s.nzsbl)]), 
                       attrs=s.attrs)
    # variables to process
    # 1st order
    vall1 = ["u_rot", "v_rot", "w", "theta"]
    # 2nd order
    vall2 = ["uw", "vw", "tw", "uu", "vv", "ww", "tt"]
    # combine
    vall = vall1 + vall2
    # dictionary for cumulative values for each var
    Lall = {}
    for v in vall:
        Lall[v] = np.zeros(s.nzsbl, dtype=np.float64)
    # BEGIN LOOPING
    # load one volume file at a time
    for jt, tfile in enumerate(fall):
        # load file
        print_both(f"Loading file: {tfile}", fprint)
        dd = xr.load_dataset(tfile)
        # rotate velocities so <v>_xy = 0
        u_mean = dd.u.mean(dim=("x","y"))
        v_mean = dd.v.mean(dim=("x","y"))
        angle = np.arctan2(v_mean, u_mean)
        dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
        dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
        # SECOND ORDER PARAMETERS
        # calculate instantaneous uw, vw, tw, uu, vv, ww, tt
        # fluxes
        w_mean = dd.w.mean(dim=("x","y"))
        theta_mean = dd.theta.mean(dim=("x","y"))
        dd["uw"] = (dd.u - u_mean) * (dd.w - w_mean) + dd.txz
        dd["vw"] = (dd.v - v_mean) * (dd.w - w_mean) + dd.tyz
        dd["tw"] = (dd.theta - theta_mean) * (dd.w - w_mean) + dd.q3
        # variances - use urot, vrot
        # detrend
        udet = detrend(dd.u_rot, axis=0, type="linear")
        vdet = detrend(dd.v_rot, axis=0, type="linear")
        wdet = detrend(dd.w, axis=0, type="linear")
        tdet = detrend(dd.theta, axis=0, type="linear")
        dd["uu"] = xr.DataArray(data=udet*udet, dims=("x","y","z"), 
                                coords=dict(x=dd.x, y=dd.y, z=dd.z))
        dd["vv"] = xr.DataArray(data=vdet*vdet, dims=("x","y","z"), 
                                coords=dict(x=dd.x, y=dd.y, z=dd.z))
        dd["ww"] = xr.DataArray(data=wdet*wdet, dims=("x","y","z"), 
                                coords=dict(x=dd.x, y=dd.y, z=dd.z))
        dd["tt"] = xr.DataArray(data=tdet*tdet, dims=("x","y","z"), 
                                coords=dict(x=dd.x, y=dd.y, z=dd.z))

        print_both("Begin looping over all parameters...", fprint)
        ## BIG LOOP OVER VARIABLES HERE
        for v in vall:
            print_both(f"Begin {v}", fprint)
            # loop over z - only within SBL
            for jz in range(s.nzsbl):
                print_both(f"jz={jz}/{s.nzsbl-1}", fprint)
                # initialize empty PSD arrays
                PSD = np.zeros((nx2, s.ny), dtype=np.float64)
                # grab data
                d = dd[v].isel(z=jz).to_numpy()
                # detrend data linearly for first order
                if v in vall1:
                    d2 = detrend(d, axis=0, type="linear")
                else:
                    d2 = d
                # spectral interpolate
                dint = np.zeros((nx2, s.ny), dtype=np.float64)
                # loop over y
                for jy in range(s.ny):
                    dint[:,jy], xnew = interp_spec(d2[:,jy], s.Lx, nint)
                # fft
                f = fft(dint, axis=0)
                # calculate PSD
                for jx in range(1, nx2//2):
                    PSD[jx,:] = np.real( f[jx,:] * np.conj(f[jx,:]) )
                    PSD[nx2-jx,:] = np.real( f[nx2-jx,:] * np.conj(f[nx2-jx,:]) )
                # normalize by variance
                PSD /= np.var(dint, axis=0)
                # ifft to get autocorrelation and norm by nx
                R = np.real( ifft(PSD, axis=0) ) / nx2
                # initialize empty lengthscale == 0 to append and average later
                Lcum = 0
                # loopy over y to integrate autocorr for lengthscale
                for jy in range(s.ny):
                    # integrate up through first zero crossing
                    # find indices
                    i0 = np.where(R[:,jy] < 0.)[0]
                    if len(i0) == 0:
                        i0 = 1
                    else:
                        i0 = i0[0]
                    # integrate
                    Lint = np.trapz(R[:i0,jy], xnew[:i0])
                    # store value
                    Lcum += Lint
                # after looping over y and t, divide Lcum to average
                # store in Lall dictionary by adding to average by nf later
                Lall[v][jz] += Lcum / s.ny
            print_both(f"L{v}: {Lall[v]/(jt+1)}", fprint)
    # after looping through files, average Lall in time
    # convert to individual dataarrays to store in Lsave
    for v in vall:
        Lall[v] /= nf
        Lsave[v] = xr.DataArray(data=Lall[v], dims="z", coords=dict(z=Lsave.z))

    # save L and return
    fsaveL = f"{dnc}L_vol.nc"
    print_both(f"Saving file: {fsaveL}", fprint)
    with ProgressBar():
        Lsave.to_netcdf(fsaveL, mode="w")    
    print_both("Finshed with all processes!", fprint)

    return
# ---------------------------------------------
def interp_spec(d, Lx, nf):
    """Interpolate a 1d array d spectrally
    input d: original data
    input Lx: length of domain
    input nf: factor by which to increase number of points
    output d2: interpolated array
    output Lx2: interpolated array of x values
    """
    # define new parameters
    nx = len(d)
    nx2 = nf*nx
    # new position array
    Lx2 = np.linspace(0., Lx, nx2)
    # initialize empty arrays for fft and ifft
    f_big = np.zeros(nx2, dtype=np.complex128)
    d2 = np.zeros(nx2, dtype=np.float64)
    # begin interpolation
    # take FFT
    f_d = fft(d, axis=0)
    # zero-pad fft array
    f_big[0] = f_d[0]                  # zero wavenumber
    f_big[1:nx//2] = f_d[1:nx//2]      # positive wavenumbers
    f_big[nx2-nx//2:nx2] = f_d[nx//2:] # negative wavenumbers
    # normalize by number of points and ifft
    d2[:] = np.real( ifft(f_big * (nx2/nx)) )
    # return
    return [d2, Lx2]

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
    # run timeseries2netcdf
    if config["run_timeseries"]:
        timeseries2netcdf()
    # run autocorr_from_timeseries
    # if config["run_autocorr"]:
    #     autocorr_from_timeseries(config["dnc"], 
    #                              nblock=config["nblock_ac"])
    # run autocorr_from_volume
    if config["run_autocorr"]:
        autocorr_from_volume()