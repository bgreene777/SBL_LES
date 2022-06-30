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
import xrft
import numpy as np
import xarray as xr
from scipy.signal import detrend
from dask.diagnostics import ProgressBar
from matplotlib.colors import Normalize
from RFMnc import print_both
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
    # 1 hour is 3600/delta_t
    nt = int(3600./delta_t)
    istart = nt_tot - nt
    # define array of time in seconds
    time = np.linspace(0., 3600.-delta_t, nt, dtype=np.float64)

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
def load_timeseries(dnc, detrend=True):
    """
    Reading function for timeseries files created from timseries2netcdf()
    Load netcdf files using xarray and calculate numerous relevant parameters
    input dnc: path to netcdf directory for simulation
    input detrend: detrend timeseries for calculating variances, default=True
    return d: xarray dataset
    """
    # load timeseries_all.nc
    d = xr.open_dataset(dnc+"timeseries_all.nc")
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
        ud = xrft.detrend(d.u_rot, dim="t", detrend_type="linear")
        vd = xrft.detrend(d.v_rot, dim="t", detrend_type="linear")
        wd = xrft.detrend(d.w, dim="t", detrend_type="linear")
        td = xrft.detrend(d.theta, dim="t", detrend_type="linear")
        d["uu"] = ud * ud
        d["vv"] = vd * vd
        d["ww"] = wd * wd
        d["tt"] = td * td
    else:
        d["uu"] = (d.u - d.u_mean) * (d.u - d.u_mean)
        d["vv"] = (d.v - d.v_mean) * (d.v - d.v_mean)
        d["ww"] = (d.w - d.w_mean) * (d.w - d.w_mean)
        d["tt"] = (d.theta - d.theta_mean) * (d.theta - d.theta_mean)
    
    return d

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