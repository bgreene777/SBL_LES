# --------------------------------
# Name: RFMnc.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 27 October 2021
# Purpose: Read volumetric output from LES to calculate autocorrelations,
# integral length scales, and random error profiles by using the 
# relaxed filtering method from Dias et al. 2018
# Combines code from random_errors_filter.py and integral_lengthscales.py
# Remastered from RFM.py to load netcdf files and streamline
# --------------------------------
import os
import sys
import yaml
import numpy as np
import xarray as xr
import xrft
from numpy.fft import fft, ifft
from scipy.signal import fftconvolve, correlate
from datetime import datetime, timedelta
from numba import njit
from dask.diagnostics import ProgressBar
# --------------------------------
# Define Functions
# --------------------------------
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

# --------------------------------
def autocorrelation(f):
    # input 4d parameter, e.g. u(x,y,z,t)
    # output DataArray yt-averaged autocorrelation R_ff(xlag,z)
    # R_ff.shape = (nx, nz)
    # calculate along the x-dimension
    # keep track of how long it takes
    dt0 = datetime.utcnow()
    print_both(f"Calculating autocorrelation for: {f.name}", fprint)
    # calculate normalized perturbations
    temp = (f - f.mean("x")) / f.std("x")
    # fftconvolve along axis=1, i.e. x
    corr_xx = fftconvolve(temp, temp[:,::-1,:,:]/f.x.size, mode="full", axes=1)
    # average along y and t
    corr_xx_ytavg = np.mean(corr_xx, axis=(0,2))
    # grab only lags >= 0
    imid = (2*f.x.size-1) // 2
    # store in new xarray DataArray with same coords as f and return
    R_ff = xr.DataArray(data=corr_xx_ytavg[imid:,:],
                        dims=["x","z"],
                        coords=dict(
                            x = f.x,
                            z = f.z)
                       )
    dt1 = datetime.utcnow()
    print_both(f"Calc time for {f.name} autocorr: {(dt1-dt0).total_seconds()/60.:5.2f} min", fprint)
    return R_ff
# --------------------------------
def relaxed_filter(f, delta_x, Lx):
    """
    -input-
    f: 4D xarray DataArray in t,x,y,z (e.g., u(t,x,y,z), theta(t,x,y,z))
    delta_x: 1d array of filter widths in m
    Lx: size of domain in x (m)
    -output-
    var_f_all: xarray DataArray of yt-averaged var along x as function of delta_x and z
    """
    # keep track of time
    dt0 = datetime.utcnow()
    print_both(f"Performing relaxed filter method for: {f.name}", fprint)
    # initialize numpy array of shape(nfilt, nz) to store all sigma_f
    nfilt = len(delta_x)
    nx = f.x.size
    ny = f.y.size
    nz = f.z.size
    nt = f.time.size
    var_f_all = xr.DataArray(data=np.zeros((nfilt, nz), dtype=np.float64),
                             coords=dict(delta_x=delta_x,
                                         z=f.z)
                            )
    # forward FFT of f
    f_fft = xrft.fft(f, dim="x", true_phase=True, true_amplitude=True)
    
    # construct box filter transfer function
    dk = 2.*np.pi/Lx
    filt = np.zeros((nfilt, nx), dtype=np.float64)  # shape(nfilt, nx), nx=numfreqs
    # loop over filter sizes
    for i, idx in enumerate(delta_x):
        filt[i,0] = 1.
        filt[i,-1] = 1.
        # loop over frequencies
        for j in range(1, nx//2):
            filt[i,j] = np.sin(j*dk*idx/2.) / (j*dk*idx/2.)
            filt[i,nx-j-1] = np.sin(j*dk*idx/2.) / (j*dk*idx/2.)
    # fftshift so that zero frequency is in center (convention used by xrft)
    filt = np.fft.fftshift(filt, axes=1)
    # convert filt to xarray DataArray
    filt = xr.DataArray(data=filt,
                        coords=dict(delta_x=delta_x,
                                    freq_x=f_fft.freq_x)
                       )
    # apply filter in x-wavenumber space: loop over time and filter widths delta_x
    # create new DataArray
    f_fft_filt = xr.DataArray(data=np.zeros((nx,ny,nz,nfilt), dtype=np.complex128),
                              coords=dict(freq_x=f_fft.freq_x,
                                          y=f.y,
                                          z=f.z,
                                          delta_x=delta_x)
                             )
    for jt in range(nt):
        for i in range(len(delta_x)):
            f_fft_filt[:,:,:,i] = f_fft.isel(time=jt) * filt.isel(delta_x=i)
        # after looping over filter widths, calculate inverse fft on x-axis
        f_ifft_filt = xrft.ifft(f_fft_filt, dim="freq_x", true_phase=True, 
                                true_amplitude=True,
                                lag=f_fft.freq_x.direct_lag)
        var_f_all += f_ifft_filt.var(dim="x").mean(dim="y")
    # divide by nt to get time-averaged sigma_f_all
    var_f_all /= nt
    
    dt1 = datetime.utcnow()
    print_both(f"Calc time for {f.name} RFM: {(dt1-dt0).total_seconds()/60.:5.2f} min", fprint)
    return var_f_all
    
# --------------------------------
# Main: calculate 4th order vars, autocorrelations, and RFM
# --------------------------------
def main():
    # config loaded in global scope
    # simulation output directory
    fdir = config["fdir"]
    # check to see if files have been created and can skip main()
    if os.path.exists(f"{fdir}RFM.nc"):
        print_both("Files already created...moving on to curve fitting and error calculations!",
                   fprint)
        return
    
    # load average statistics netcdf file
    fstat = config["fstat"]
    stat = xr.load_dataset(fdir+fstat)
    # calculate important parameters
    # ustar
    stat["ustar"] = ((stat.uw_cov_tot ** 2.) + (stat.vw_cov_tot ** 2.)) ** 0.25
    stat["ustar2"] = stat.ustar ** 2.
    # SBL height
    stat["h"] = stat.z.where(stat.ustar2 <= 0.05*stat.ustar2[0], drop=True)[0]/0.95
    # z indices within sbl
    isbl = np.where(stat.z <= stat.h)[0]
    nz_sbl = len(isbl)

    #
    # Load files and clean up
    #
    # prepare to load all individual netcdf timestep files
    timesteps = np.arange(config["t0"], config["t1"]+1, config["dt"], dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{fdir}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*config["delta_t"]*config["dt"] for i in range(nf)])
    # load all files into mfdataset
    print_both("Reading files...", fprint)
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    # array of delta_x spaced logarithmic from dx to Lx
    nfilt = config["nfilt"]
    delta_x = np.logspace(np.log10(dd.dx), np.log10(dd.Lx),
                          num=nfilt, base=10.0, dtype=np.float64)

    #
    # Rotate instantaneous (u,v) using u_mean and v_mean from stat
    #
    angle = np.arctan2(stat.v_mean, stat.u_mean)
    dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)  # xarray takes care of dimensions
    dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)

    #
    # Calculate "instantaneous" vars and covars to store in dd
    #
    print_both("Calculate 'instantaneous' vars and covars", fprint)
    # u'w'
    dd["uw_cov_res"] = (dd.u - stat.u_mean) * (dd.w - stat.w_mean)
    dd["uw_cov_tot"] = dd.uw_cov_res + dd.txz
    # v'w'
    dd["vw_cov_res"] = (dd.v - stat.v_mean) * (dd.w - stat.w_mean)
    dd["vw_cov_tot"] = dd.vw_cov_res + dd.tyz
    # t'w'
    dd["tw_cov_res"] = (dd.theta - stat.theta_mean) * (dd.w - stat.w_mean)
    dd["tw_cov_tot"] = dd.tw_cov_res + dd.q3
    # u'u' UNROTATED
    dd["uu_var"] = (dd.u - stat.u_mean) * (dd.u - stat.u_mean)
    # u'u' ROTATED
    dd["uu_var_rot"] = (dd.u_rot - stat.u_mean_rot) * (dd.u_rot - stat.u_mean_rot)
    # v'v' UNROTATED
    dd["vv_var"] = (dd.v - stat.v_mean) * (dd.v - stat.v_mean)
    # v'v' ROTATED
    dd["vv_var_rot"] = (dd.v_rot - stat.v_mean_rot) * (dd.v_rot - stat.v_mean_rot)
    # w'w'
    dd["ww_var"] = (dd.w - stat.w_mean) * (dd.w - stat.w_mean)
    # t't'
    dd["tt_var"] = (dd.theta - stat.theta_mean) * (dd.theta - stat.theta_mean)

    #
    # Use "instantaneous" vars and covars to calculate 4th order variances
    #
    print("Calculate 4th order vars")
    # u'w'u'w' = var{u'w'}(x,y,z,t) = (u'w' - <u'w'>_xyzt)**2
    dd["uwuw_var"] = (dd.uw_cov_tot - stat.uw_cov_tot) * (dd.uw_cov_tot - stat.uw_cov_tot)
    # v'w'v'w' = var{v'w'}(x,y,z,t) = (v'w' - <v'w'>_xyzt)**2
    dd["vwvw_var"] = (dd.vw_cov_tot - stat.vw_cov_tot) * (dd.vw_cov_tot - stat.vw_cov_tot)
    # t'w't'w' = var{t'w'}(x,y,z,t) = (t'w' - <t'w'>_xyzt)**2
    dd["twtw_var"] = (dd.tw_cov_tot - stat.tw_cov_tot) * (dd.tw_cov_tot - stat.tw_cov_tot)
    # UNROTATED u'u'u'u' = var{u'u'}(x,y,z,t) = (u'u' - <u'u'>_xyzt)**2
    dd["uuuu_var"] = (dd.uu_var - stat.u_var) * (dd.uu_var - stat.u_var)
    # ROTATED u'u'u'u' = var{u'u'}(x,y,z,t) = (u'u' - <u'u'>_xyzt)**2
    dd["uuuu_var_rot"] = (dd.uu_var_rot - stat.u_var_rot) * (dd.uu_var_rot - stat.u_var_rot)
    # UNROTATED v'v'v'v' = var{v'v'}(x,y,z,t) = (v'v' - <v'v'>_xyzt)**2
    dd["vvvv_var"] = (dd.vv_var - stat.v_var) * (dd.vv_var - stat.v_var)
    # ROTATED v'v'v'v' = var{v'v'}(x,y,z,t) = (v'v' - <v'v'>_xyzt)**2
    dd["vvvv_var_rot"] = (dd.vv_var_rot - stat.v_var_rot) * (dd.vv_var_rot - stat.v_var_rot)
    # w'w'w'w' = var{w'w'}(x,y,z,t) = (w'w' - <w'w'>_xyzt)**2
    dd["wwww_var"] = (dd.ww_var - stat.w_var) * (dd.ww_var - stat.w_var)
    # t't't't' = var{t't'}(x,y,z,t) = (t't' - <t't'>_xyzt)**2
    dd["tttt_var"] = (dd.tt_var - stat.theta_var) * (dd.tt_var - stat.theta_var)
    
    #
    # Calculate xyt averages of 4th order variances and store in new xarray Dataset to save
    #
    var4 = xr.Dataset(data_vars=None,
                      coords=dict(z=dd.z),
                      attrs=stat.attrs
                     )
    # define list of parameters for looping
    param_var4 = ["uwuw_var", "vwvw_var", "twtw_var", "uuuu_var", "uuuu_var_rot",
                  "vvvv_var", "vvvv_var_rot", "wwww_var", "tttt_var"]
    # loop
    for p in param_var4:
        var4[p] = dd[p].mean(dim=("x","y","time"))

    #
    # Calculate autocorrelations for _everything_
    #
    print_both("Begin calculating autocorrelations...", fprint)
    # initialize empty xarray dataset for each autocorrelation and loop
    R = xr.Dataset(data_vars=None,
                   coords=dict(x=dd.x,
                               z=dd.z),
                   attrs=dd.attrs
                  )
    # define list of parameters for looping
    param_acf = ["u", "u_rot", "v", "v_rot", "w", "theta",
                 "uw_cov_tot", "vw_cov_tot", "tw_cov_tot", "uu_var", "uu_var_rot", 
                 "vv_var", "vv_var_rot", "ww_var", "tt_var",
                 "uwuw_var", "vwvw_var", "uuuu_var", "uuuu_var_rot", 
                 "vvvv_var", "vvvv_var_rot", "wwww_var", "tttt_var"]
    for p in param_acf:
        R[p] = autocorrelation(dd[p])

    #
    # Perform relaxed filter method for _everything_
    #
    print_both("Begin RFM calculations...", fprint)
    # initialize empty xarray dataset for each RFM and loop
    RFM = xr.Dataset(data_vars=None,
                     coords=dict(delta_x=delta_x,
                                 z=dd.z),
                     attrs=dd.attrs
                    )
    # define list of parameters for looping
    param_RFM = ["u", "u_rot", "v", "v_rot", "theta", 
                 "uw_cov_tot", "vw_cov_tot", "tw_cov_tot", "uu_var", "uu_var_rot",
                 "vv_var", "vv_var_rot", "ww_var", "tt_var"]
    # loop
    for p in param_RFM:
        RFM[p] = relaxed_filter(dd[p], delta_x, dd.Lx)
        
    #
    # Save all of these Datasets which took forever to calculate
    #
    # 4th order variances, var4
    fsave_var4 = f"{fdir}variances_4_order.nc"
    print_both(f"Saving file: {fsave_var4}", fprint)
    with ProgressBar():
        var4.to_netcdf(fsave_var4, mode="w")
    # autocorrelations, R
    fsave_R = f"{fdir}autocorr.nc"
    print_both(f"Saving file: {fsave_R}", fprint)
    with ProgressBar():
        R.to_netcdf(fsave_R, mode="w")
    # relaxed filter output, RFM
    fsave_RFM = f"{fdir}RFM.nc"
    print_both(f"Saving file: {fsave_RFM}", fprint)
    with ProgressBar():
        RFM.to_netcdf(fsave_RFM, mode="w")
        
    print_both("RFMnc.py:main() finished!", fprint)
    return

# --------------------------------
# Main2: read results from main() and fit power laws to estimate errors
# --------------------------------
def main2():
    # config loaded in global scope
    # simulation output directory
    fdir = config["fdir"]
    # load average statistics netcdf file
    fstat = config["fstat"]
    stat = xr.load_dataset(fdir+fstat)
    # calculate important parameters
    # ustar
    stat["ustar"] = ((stat.uw_cov_tot ** 2.) + (stat.vw_cov_tot ** 2.)) ** 0.25
    stat["ustar2"] = stat.ustar ** 2.
    # SBL height
    stat["h"] = stat.z.where(stat.ustar2 <= 0.05*stat.ustar2[0], drop=True)[0]/0.95
    # z indices within sbl
    isbl = np.where(stat.z <= stat.h)[0]
    nz_sbl = len(isbl)    
    # load files created in main()
    var4 = xr.load_dataset(f"{fdir}variances_4_order.nc")
    R = xr.load_dataset(f"{fdir}autocorr.nc")
    RFM = xr.load_dataset(f"{fdir}RFM.nc")
    
    #
    # 
    #
    
    
    
    
    print("Rose is ugly")
        
# --------------------------------
# Run script
# --------------------------------
if __name__ == "__main__":
    # load yaml file in global scope
    with open("/home/bgreene/SBL_LES/python/RFMnc.yaml") as f:
        config = yaml.safe_load(f)
    fprint = config["fprint"]
    dt0 = datetime.utcnow()
    main()
    dt1 = datetime.utcnow()
    print_both(f"Total run time for RFMnc.py: {(dt1-dt0).total_seconds()/60.:5.2f} min", fprint)