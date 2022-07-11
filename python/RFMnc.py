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
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from dask.diagnostics import ProgressBar
from LESnc import load_stats, print_both
# --------------------------------
# Define Functions
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
def power_law(delta_x, C, p):
    # function to be used with curve_fit
    return C * (delta_x ** (-p))

# --------------------------------
# Main: calculate 4th order vars, autocorrelations, and RFM
# --------------------------------
def main():
    # config loaded in global scope
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
    # recalc planar averages so resulting instantaneous vars are detrended
    umean_det = dd.u.mean(dim=("x","y"))
    dd["uu_var"] = (dd.u - umean_det) * (dd.u - umean_det)
    # u'u' ROTATED
    umean_rot_det = dd.u_rot.mean(dim=("x","y"))
    dd["uu_var_rot"] = (dd.u_rot - umean_rot_det) * (dd.u_rot - umean_rot_det)
    # v'v' UNROTATED
    vmean_det = dd.v.mean(dim=("x","y"))
    dd["vv_var"] = (dd.v - vmean_det) * (dd.v - vmean_det)
    # v'v' ROTATED
    vmean_rot_det = dd.v_rot.mean(dim=("x","y"))
    dd["vv_var_rot"] = (dd.v_rot - vmean_rot_det) * (dd.v_rot - vmean_rot_det)
    # w'w'
    wmean_det = dd.w.mean(dim=("x","y"))
    dd["ww_var"] = (dd.w - wmean_det) * (dd.w - wmean_det)
    # t't'
    tmean_det = dd.theta.mean(dim=("x","y"))
    dd["tt_var"] = (dd.theta - tmean_det) * (dd.theta - tmean_det)

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
# Main2: read results from main() to fit power laws and save C, p files
# --------------------------------
def main2(plot_MSE=True):
    # config loaded in global scope
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
    z_sbl = stat.z.isel(z=isbl)
    # load files created in main()
    var4 = xr.load_dataset(f"{fdir}variances_4_order.nc")
    R = xr.load_dataset(f"{fdir}autocorr.nc")
    RFM = xr.load_dataset(f"{fdir}RFM.nc")
    # calculate Ozmidov scale Lo = sqrt[<dissipation>/(N^2)^3/2]
    dtheta_dz = stat.theta_mean.differentiate("z", 2)
    N2 = dtheta_dz * 9.81 / stat.theta_mean.isel(z=0)
    stat["Lo"] = np.sqrt(-stat.dissip_mean / (N2 ** (3./2.)))
    
    #
    # plot MSE versus filter width
    # 3 multi-panel figures
    #
    if plot_MSE:
        # 1) u, u_rot, v, v_rot
        fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        for jz in np.arange(0, nz_sbl, 4):
            ax1[0,0].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.u.isel(z=jz)/stat.u_var.isel(z=jz), 
                          label=f"z/h={(stat.z.isel(z=jz)/stat.h).values:4.3f}")
            ax1[0,1].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.v.isel(z=jz)/stat.v_var.isel(z=jz), label=f"jz={jz}")
            ax1[1,0].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.u_rot.isel(z=jz)/stat.u_var_rot.isel(z=jz))
            ax1[1,1].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.v_rot.isel(z=jz)/stat.v_var_rot.isel(z=jz))
        ax1[0,0].set_xlabel("$\Delta x$")
        ax1[0,0].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax1[0,0].set_title("Variable: $u$")
        ax1[0,0].legend(loc="lower left")
        ax1[0,0].set_xscale("log")
        ax1[0,0].set_yscale("log")
        ax1[0,1].set_xlabel("$\Delta x$")
        ax1[0,1].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax1[0,1].set_title("Variable: $v$")
        ax1[0,1].legend()
        ax1[0,1].set_xscale("log")
        ax1[0,1].set_yscale("log")
        ax1[1,0].set_xlabel("$\Delta x$")
        ax1[1,0].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax1[1,0].set_title("Variable: $u$ rotated")
        ax1[1,0].set_xscale("log")
        ax1[1,0].set_yscale("log")
        ax1[1,1].set_xlabel("$\Delta x$")
        ax1[1,1].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax1[1,1].set_title("Variable: $v$ rotated")
        ax1[1,1].set_xscale("log")
        ax1[1,1].set_yscale("log")
        for iax in ax1.flatten():
            iax.axvline(config["dmin_u"], c="k", ls="--")
            iax.axvline(config["dmax_u"], c="k", ls="--")

        # save and close
        figdir = config["figdir"]
        fig1.savefig(f"{figdir}{stat.stability}_MSEuv_varuv_vs_deltax.png")
        plt.close(fig1)

        # 2) theta, u'w', v'w', theta'w'
        fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
#         for jz in np.arange(9)**2:
        for jz in np.arange(0, nz_sbl, 4):
            ax2[0,0].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.theta.isel(z=jz)/stat.theta_var.isel(z=jz), 
                          label=f"z/h={(stat.z.isel(z=jz)/stat.h).values:4.3f}")
            ax2[0,1].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.tw_cov_tot.isel(z=jz)/var4.twtw_var.isel(z=jz), label=f"jz={jz}")
            ax2[1,0].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.uw_cov_tot.isel(z=jz)/var4.uwuw_var.isel(z=jz))
            ax2[1,1].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.vw_cov_tot.isel(z=jz)/var4.uwuw_var.isel(z=jz))
        ax2[0,0].set_xlabel("$\Delta x$")
        ax2[0,0].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax2[0,0].set_title("Variable: $\\theta$")
        ax2[0,0].legend(loc="lower left")
        ax2[0,0].set_xscale("log")
        ax2[0,0].set_yscale("log")
        ax2[0,1].set_xlabel("$\Delta x$")
        ax2[0,1].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax2[0,1].set_title("Variable: $\\theta'w'$")
        ax2[0,1].legend()
        ax2[0,1].set_xscale("log")
        ax2[0,1].set_yscale("log")
        ax2[1,0].set_xlabel("$\Delta x$")
        ax2[1,0].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax2[1,0].set_title("Variable: $u'w'$")
        ax2[1,0].set_xscale("log")
        ax2[1,0].set_yscale("log")
        ax2[1,1].set_xlabel("$\Delta x$")
        ax2[1,1].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax2[1,1].set_title("Variable: $v'w'$")
        ax2[1,1].set_xscale("log")
        ax2[1,1].set_yscale("log")
        # plot vertical lines to show where fitting
        ax2[0,0].axvline(config["dmin_u"], c="k", ls="--")
        ax2[0,0].axvline(config["dmax_u"], c="k", ls="--")
        for iax in ax2.flatten()[1:]:
            iax.axvline(config["dmin_cov"], c="k", ls="--")
            iax.axvline(config["dmax_cov"], c="k", ls="--")

        # save and close
        fig2.savefig(f"{figdir}{stat.stability}_MSEcov_var4_vs_deltax.png")
        plt.close(fig2)

        # 3) u var, u var rot, v var, v var rot, ww var, tt var
        fig3, ax3 = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        for jz in np.arange(0, nz_sbl, 4):
            ax3[0,0].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.uu_var.isel(z=jz)/var4.uuuu_var.isel(z=jz), 
                          label=f"z/h={(stat.z.isel(z=jz)/stat.h).values:4.3f}")
            ax3[0,1].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.vv_var.isel(z=jz)/var4.vvvv_var.isel(z=jz), label=f"jz={jz}")
            ax3[0,2].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.ww_var.isel(z=jz)/var4.wwww_var.isel(z=jz), label=f"jz={jz}")
            ax3[1,0].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.uu_var_rot.isel(z=jz)/var4.uuuu_var_rot.isel(z=jz))
            ax3[1,1].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.vv_var_rot.isel(z=jz)/var4.vvvv_var_rot.isel(z=jz))
            ax3[1,2].plot(RFM.delta_x,#/stat.u_mean_rot.isel(z=jz), 
                          RFM.tt_var.isel(z=jz)/var4.tttt_var.isel(z=jz))
        ax3[0,0].set_xlabel("$\Delta x$")
        ax3[0,0].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax3[0,0].set_title("Variable: $u'u'$")
        ax3[0,0].legend(loc="lower left")
        ax3[0,0].set_xscale("log")
        ax3[0,0].set_yscale("log")
        ax3[0,1].set_xlabel("$\Delta x$")
        ax3[0,1].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax3[0,1].set_title("Variable: $v'v'$")
        ax3[0,1].legend(loc="lower left")
        ax3[0,1].set_xscale("log")
        ax3[0,1].set_yscale("log")
        ax3[0,2].set_xlabel("$\Delta x$")
        ax3[0,2].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax3[0,2].set_title("Variable: $w'w'$")
        ax3[0,2].legend()
        ax3[0,2].set_xscale("log")
        ax3[0,2].set_yscale("log")
        ax3[1,0].set_xlabel("$\Delta x$")
        ax3[1,0].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax3[1,0].set_title("Variable: $u'u'$ rotated")
        ax3[1,0].set_xscale("log")
        ax3[1,0].set_yscale("log")
        ax3[1,1].set_xlabel("$\Delta x$")
        ax3[1,1].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax3[1,1].set_title("Variable: $v'v'$ rotated")
        ax3[1,1].set_xscale("log")
        ax3[1,1].set_yscale("log")
        ax3[1,2].set_xlabel("$\Delta x$")
        ax3[1,2].set_ylabel("$MSE(\\tilde{x}) / var\{x\}$")
        ax3[1,2].set_title("Variable: $\\theta'\\theta'$")
        ax3[1,2].set_xscale("log")
        ax3[1,2].set_yscale("log")
        for iax in ax3.flatten():
            iax.axvline(config["dmin_cov"], c="k", ls="--")
            iax.axvline(config["dmax_cov"], c="k", ls="--")
        
        # save and close
        fig3.savefig(f"{figdir}{stat.stability}_MSEvar_var4_vs_deltax.png")
        plt.close(fig3)
    #
    # skip the rest if files already exist
    #
    if os.path.exists(f"{fdir}fit_C.nc"):
        print_both("Curve fit files already created...moving on to error calculations!",
                   fprint)
        return    
    #
    # Fit power law for all variables and store C, p in separate xarray Datasets
    #
    C = xr.Dataset(data_vars=None,
                   coords=dict(z=z_sbl),
                   attrs={k:str(config[k]) for k in config.keys()})
    p = xr.Dataset(data_vars=None,
                   coords=dict(z=z_sbl),
                   attrs={k:str(config[k]) for k in config.keys()})
    # loop through variables - separately for 1st and 2nd order moments (stat vs var4)
    # first order moments and variances
    param_RFM1 = ["u", "u_rot", "v", "v_rot", "theta"]
    param_var2 = ["u_var", "u_var_rot", "v_var", "v_var_rot", "theta_var"]
    # covariances and their 4th order variances
    param_RFM2 = ["uw_cov_tot", "vw_cov_tot", "tw_cov_tot"]
    param_cov = ["uwuw_var", "vwvw_var", "twtw_var"]
    # variances and their 4th order variances
    param_RFM3 = ["uu_var", "uu_var_rot", "vv_var", "vv_var_rot", 
                  "ww_var", "tt_var"]
    param_var = ["uuuu_var", "uuuu_var_rot", "vvvv_var", "vvvv_var_rot", 
                 "wwww_var", "tttt_var"]
    # need to grab delta_x ranges based on RFMnc.yaml file 
    # separately for u/v/theta, var, covar
    ix_uvt = np.where((RFM.delta_x >= config["dmin_u"]) &\
                      (RFM.delta_x <= config["dmax_u"]))[0]
    ix_cov = np.where((RFM.delta_x >= config["dmin_cov"]) &\
                      (RFM.delta_x <= config["dmax_cov"]))[0]
    ix_var = np.where((RFM.delta_x >= config["dmin_var"]) &\
                      (RFM.delta_x <= config["dmax_var"]))[0]
    # loop through first order moments
    for i, v in enumerate(param_RFM1):
        print_both(f"Calculating power law coefficients for: {v}", fprint)
        C[v] = xr.DataArray(np.zeros(nz_sbl, dtype=np.float64), 
                              dims="z",
                              coords=dict(z=z_sbl))
        p[v] = xr.DataArray(np.zeros(nz_sbl, dtype=np.float64), 
                              dims="z",
                              coords=dict(z=z_sbl))
        for jz in range(nz_sbl):
            # need to grab delta_t ranges based on RFMnc.yaml file 
            # separately for u/v/theta, var, covar
            it_uvt = np.where((RFM.delta_x/stat.u_mean_rot.isel(z=jz) >= config["dmin_u"]) &\
                              (RFM.delta_x/stat.u_mean_rot.isel(z=jz) <= config["dmax_u"]))[0]
            xfit = RFM.delta_x.isel(delta_x=ix_uvt)
            yfit = RFM[v].isel(z=jz, delta_x=ix_uvt)/\
                              stat[param_var2[i]].isel(z=jz)
            (C[v][jz], p[v][jz]), _ = curve_fit(f=power_law, 
                                                xdata=xfit,
                                                ydata=yfit,
                                                p0=[0.001,0.001])
    # loop through second order moments - covariances
    for i, v in enumerate(param_RFM2):
        print_both(f"Calculating power law coefficients for: {v}", fprint)
        C[v] = xr.DataArray(np.zeros(nz_sbl, dtype=np.float64), 
                              dims="z",
                              coords=dict(z=z_sbl))
        p[v] = xr.DataArray(np.zeros(nz_sbl, dtype=np.float64), 
                              dims="z",
                              coords=dict(z=z_sbl))
        for jz in range(nz_sbl):
            # need to grab delta_t ranges based on RFMnc.yaml file 
            # separately for u/v/theta, var, covar
            it_cov = np.where((RFM.delta_x/stat.u_mean_rot.isel(z=jz) >= config["dmin_cov"]) &\
                              (RFM.delta_x/stat.u_mean_rot.isel(z=jz) <= config["dmax_cov"]))[0]
            xfit = RFM.delta_x.isel(delta_x=ix_cov)
            yfit = RFM[v].isel(z=jz, delta_x=ix_cov)/\
                              var4[param_cov[i]].isel(z=jz)
            (C[v][jz], p[v][jz]), _ = curve_fit(f=power_law, 
                                                xdata=xfit,
                                                ydata=yfit,
                                                p0=[0.001,0.001])
    # loop through second order moments - variances
    for i, v in enumerate(param_RFM3):
        print_both(f"Calculating power law coefficients for: {v}", fprint)
        C[v] = xr.DataArray(np.zeros(nz_sbl, dtype=np.float64), 
                              dims="z",
                              coords=dict(z=z_sbl))
        p[v] = xr.DataArray(np.zeros(nz_sbl, dtype=np.float64), 
                              dims="z",
                              coords=dict(z=z_sbl))
        for jz in range(nz_sbl):
            # need to grab delta_t ranges based on RFMnc.yaml file 
            # separately for u/v/theta, var, covar
            it_var = np.where((RFM.delta_x/stat.u_mean_rot.isel(z=jz) >= config["dmin_var"]) &\
                              (RFM.delta_x/stat.u_mean_rot.isel(z=jz) <= config["dmax_var"]))[0]
            xfit = RFM.delta_x.isel(delta_x=ix_var)
            yfit = RFM[v].isel(z=jz, delta_x=ix_var)/\
                              var4[param_var[i]].isel(z=jz)
            yfit = yfit.fillna(0.)
            (C[v][jz], p[v][jz]), _ = curve_fit(f=power_law, 
                                                xdata=xfit,
                                                ydata=yfit,
                                                p0=[0.001,0.001])
    #
    # Save C and p as netcdf files to be used in error calculations later
    #
    fsave_C = f"{fdir}fit_C.nc"
    print_both(f"Saving file: {fsave_C}", fprint)
    with ProgressBar():
        C.to_netcdf(fsave_C, mode="w")
    fsave_p = f"{fdir}fit_p.nc"
    print_both(f"Saving file: {fsave_p}", fprint)
    with ProgressBar():
        p.to_netcdf(fsave_p, mode="w")  
        
    return   

# --------------------------------
# Main3: read results from main1(), main2() to calculate errors based on LP and RFM methods
# --------------------------------
def main3(reprocess):
    #
    # Calculate RFM relative random errors: epsilon = RMSE(x_delta) / <x>
    # RMSE = C**0.5 * delta**(-p/2)
    # use T from config and convert to x/L_H via Taylor
    # separate for u,v,theta / uw,vw,tw
    #
    if not reprocess:
        print_both("Errors already calculated! Finished with main3()", fprint)
        return
        
    figdir = config["figdir"]
    # load average statistics netcdf file
    fstat = config["fstat"]
    stat = xr.load_dataset(fdir+fstat)
    # calculate important parameters
    # ustar
    stat["ustar"] = ((stat.uw_cov_tot ** 2.) + (stat.vw_cov_tot ** 2.)) ** 0.25
    stat["ustar2"] = stat.ustar ** 2.
    # SBL height
    stat["h"] = stat.z.where(stat.ustar2 <= 0.05*stat.ustar2[0], drop=True)[0]/0.95
    # calculate wind angle alpha (NOTE: THE ALPHA STORED IN STAT IS *NOT* WDIR)
    stat["alpha"] = np.arctan2(-stat.u_mean, -stat.v_mean)
    ineg = np.where(stat.alpha < 0)
    stat["alpha"][ineg] += 2.*np.pi  # alpha in radians already
    # z indices within sbl
    isbl = np.where(stat.z <= stat.h)[0]
    nz_sbl = len(isbl)    
    z_sbl = stat.z.isel(z=isbl)
    # load files created in main()
    var4 = xr.load_dataset(f"{fdir}variances_4_order.nc")
    R = xr.load_dataset(f"{fdir}autocorr.nc")
    RFM = xr.load_dataset(f"{fdir}RFM.nc")
    # load files created in main2()
    C = xr.load_dataset(f"{fdir}fit_C.nc")
    p = xr.load_dataset(f"{fdir}fit_p.nc")
    
    # 
    # Begin calculating errors
    #
    # load averaging times from config file
    T1 = config["T_sample_u"]   # s
    T2 = config["T_sample_cov"] # s
    # use Taylor hypothesis to convert time to space
    x_u = stat.u_mean_rot.isel(z=isbl) * T1   # first-order moments
    x_cov = stat.u_mean_rot.isel(z=isbl) * T2 # second-order moments
    # since new fit is using time as x-coordinate, need to use time for error calc too
    t_u = T1 * np.ones(len(isbl))
    t_cov = T2 * np.ones(len(isbl))
    # create xarray Datasets for MSE and err
    MSE = xr.Dataset(data_vars=None,
                     coords=dict(z=C.z),
                     attrs=C.attrs)
    err = xr.Dataset(data_vars=None,
                     coords=dict(z=C.z),
                     attrs=C.attrs)
    # parameters for looping through C, p + stat, var4 data
    # first order moments
    # keys for looping through C, p
    param_RFM1 = ["u", "u_rot", "v", "v_rot", "theta"] 
    # keys for looping through stat to normalize MSE
    param_var2 = ["u_var", "u_var_rot", "v_var", "v_var_rot", "theta_var"] 
    # keys for looping through stat to normalize relative errors
    param_mean1 = ["u_mean", "u_mean_rot", "v_mean", "v_mean_rot", "theta_mean"] 
    # covariances and their 4th order variances
    param_RFM2 = ["uw_cov_tot", "vw_cov_tot", "tw_cov_tot"]
    param_cov = ["uwuw_var", "vwvw_var", "twtw_var"]
    param_mean2 = param_RFM2
    # variances and their 4th order variances
    param_RFM3 = ["uu_var", "uu_var_rot", "vv_var", "vv_var_rot", 
                  "ww_var", "tt_var"]
    param_var = ["uuuu_var", "uuuu_var_rot", "vvvv_var", "vvvv_var_rot", 
                 "wwww_var", "tttt_var"]
    param_mean3 = ["u_var", "u_var_rot", "v_var", "v_var_rot", "w_var", "theta_var"]
    # Loop through first-order moments to calculate MSE and error
    for i, v in enumerate(param_RFM1):
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in param_var2
        MSE[v] = stat[param_var2[i]].isel(z=isbl) * (C[v] * (x_u**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon
        err[v] = RMSE / abs(stat[param_mean1[i]].isel(z=isbl))
    # Loop through covariances to calculate MSE and error
    for i, v in enumerate(param_RFM2):
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in param_var2
        MSE[v] = var4[param_cov[i]].isel(z=isbl) * (C[v] * (x_cov**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon
        err[v] = RMSE / abs(stat[param_mean2[i]].isel(z=isbl))    
    # Loop through variances to calculate MSE and error
    for i, v in enumerate(param_RFM3):
        # use values of C and p to extrapolate calculation of MSE/var{x}
        # renormalize with variances in param_var2
        MSE[v] = var4[param_var[i]].isel(z=isbl) * (C[v] * (x_cov**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon
        err[v] = RMSE / abs(stat[param_mean3[i]].isel(z=isbl))
    
    # now calculate errors in wind speed and wind direction through error prop.
    # wind speed = uh, wind direction = alpha
    sig_u = np.sqrt(MSE["u"])
    sig_v = np.sqrt(MSE["v"])
    # calculate ws error and assign to RFM
    err_uh = np.sqrt( (sig_u**2. * stat.u_mean.isel(z=isbl)**2. +\
                       sig_v**2. * stat.v_mean.isel(z=isbl)**2.)/\
                      (stat.u_mean.isel(z=isbl)**2. +\
                       stat.v_mean.isel(z=isbl)**2.) ) / stat.u_mean_rot.isel(z=isbl)
    err["uh"] = err_uh
    # calculate wd error and assign to RFM
    err_alpha = np.sqrt( (sig_u**2. * stat.v_mean.isel(z=isbl)**2. +\
                          sig_v**2. * stat.u_mean.isel(z=isbl)**2.)/\
                         ((stat.u_mean.isel(z=isbl)**2. +\
                           stat.v_mean.isel(z=isbl)**2.)**2.) ) /\
             (stat.alpha.isel(z=isbl))  # normalize w/ rad
    err["alpha"] = err_alpha
    
    # calculate errors in ustar^2 for coordinate-agnostic horiz Reynolds stress
    # ustar2 = ((u'w')^2 + (v'w')^2) ^ 1/2
    sig_uw = np.sqrt(MSE["uw_cov_tot"])
    sig_vw = np.sqrt(MSE["vw_cov_tot"])
    err_ustar2 = np.sqrt( (sig_uw**2. * stat.uw_cov_tot.isel(z=isbl)**2. +\
                           sig_vw**2. * stat.vw_cov_tot.isel(z=isbl)**2.)/\
                           (stat.ustar2.isel(z=isbl)**2.)) /\
                           stat.ustar2.isel(z=isbl)
    err["ustar2"] = err_ustar2

    # Save err as netcdf
    fsave_err = f"{fdir}err.nc"
    print_both(f"Saving file: {fsave_err}", fprint)
    with ProgressBar():
        err.to_netcdf(fsave_err, mode="w")  
        
    #
    # Calculate errors by integrating autocorrelation and using LP method
    # as a comparison for RFM errors
    #
    print_both("Calculate integral lengthscales...", fprint)
    # first calculate lengthscales for each variable as func of z
    # use z from C since that is only isbl
    L = xr.Dataset(data_vars=None,
                     coords=dict(z=C.z),
                     attrs=C.attrs)
    for v in (param_RFM1+param_RFM2+param_RFM3+["w"]):
        # start with zero array for each parameter
        L[v] = xr.DataArray(np.zeros(C.z.size, np.float64),
                            dims="z",
                            coords=dict(z=C.z))
        # integrate R up to first zero crossing
        # loop over altitudes
        for jz in range(C.z.size):
            # find first zero crossing
            izero = np.where(R[v].isel(z=jz) < 0.)[0]
            # make sure this isn't an empty array
            if len(izero) > 0:
                # grab first instance
                izero = izero[0]
            else:
                izero = 1
            # integrate from index 0 to izero
            L[v][jz] = R[v].isel(z=jz,x=range(izero)).integrate("x")
    # calculate integral *timescales* from L and Ubar_rot
    T = L / stat.u_mean_rot.isel(z=isbl)
    # Save L as netcdf
    fsave_L = f"{fdir}L.nc"
    print_both(f"Saving file: {fsave_L}", fprint)
    with ProgressBar():
        L.to_netcdf(fsave_L, mode="w")              
    #
    # calculate error from Lumley and Panofsky for given sample time
    # use x_u and x_cov calculated earlier
    # err_LP = sqrt[(2*int_lengh*ens_variance)/(ens_mean^2*sample_length)]
    #
    print_both("Calculate LP relative random errors...", fprint)
    err_LP = xr.Dataset(data_vars=None,
                        coords=dict(z=C.z),
                        attrs=C.attrs)    
    # Loop through first-order moments to calculate LP error
    for i, v in enumerate(param_RFM1):
        err_LP[v] = np.sqrt((2. * L[v] * stat[param_var2[i]].isel(z=isbl))/\
                            (x_u * stat[param_mean1[i]].isel(z=isbl)**2.))
    # Loop through covariances to calculate LP error
    for i, v in enumerate(param_RFM2):
        err_LP[v] = np.sqrt((2. * L[v] * var4[param_cov[i]].isel(z=isbl))/\
                            (x_cov * stat[param_mean2[i]].isel(z=isbl)**2.))
    # Loop through variances to calculate LP error
    for i, v in enumerate(param_RFM3):
        err_LP[v] = np.sqrt((2. * L[v] * var4[param_var[i]].isel(z=isbl))/\
                            (x_cov * stat[param_mean3[i]].isel(z=isbl)**2.))
    # Save err_LP as netcdf
    fsave_errLP = f"{fdir}err_LP.nc"
    print_both(f"Saving file: {fsave_errLP}", fprint)
    with ProgressBar():
        err_LP.to_netcdf(fsave_errLP, mode="w")   
            
    #
    # Plot error profiles for each parameter from both methods to compare
    #
    for v in (param_RFM1+param_RFM2+param_RFM3):
        fig, ax = plt.subplots(1)
        ax.plot(100.*err[v], err.z/stat.h, label="RFM")
        ax.plot(100.*err_LP[v], err_LP.z/stat.h, label="LP")
        ax.set_xlabel("$\\epsilon$ [%]")
        ax.set_ylabel("$z/h$")
        ax.set_title(f"Relative random error for: {err[v].name}")
        ax.legend()
        # save and close
        fig.savefig(f"{figdir}{config['stability']}_error_{err[v].name}.png")
        plt.close(fig)
        # also plot integral lengthscales
#         fig, ax = plt.subplots(1)
#         ax.plot(L[v], L.z/stat.h)
#         ax.set_xlabel("$\mathcal{L}$ [m]")
#         ax.set_ylabel("$z/h$")
#         ax.set_title(f"Integral length scale for: {err[v].name}")
#         # save and close
#         fig.savefig(f"{figdir}{config['stability']}_lengthscale_{L[v].name}.png")
#         plt.close(fig)
    
    return
    
# --------------------------------
# recalc_err: recalculate random errors with coefficients for given sample time
# --------------------------------
def recalc_err(stability, Tnew, Tnew_ec=None):
    # input stability (string) new sampling time (float), and level (integer)
    # if Tnew is array, then loop through each
    # if Tnew_ec is none, skip recalculating 2nd order moments
    # else, use Tnew_ec to recalc
    # return xr.Dataset with new errors
    # new Dataset will have dimension of Tnew
    # define directories based on stability
    fdir = f"/home/bgreene/simulations/{stability}/output/netcdf/"
    # load stat file to convert Tnew to Xnew and to renormalize new errors
    stat = load_stats(fdir+"average_statistics.nc")
    # z indices within sbl
    isbl = np.where(stat.z <= stat.h)[0]
    # load 4th order variances
    var4 = xr.load_dataset(f"{fdir}variances_4_order.nc")
    # load C and p fit coefficients
    C = xr.load_dataset(f"{fdir}fit_C.nc")
    p = xr.load_dataset(f"{fdir}fit_p.nc")
    # check if Tnew is an array or single value and convert to iterable
    if np.shape(Tnew) == ():
        Tnew = np.array([Tnew])
    # create xarray Datasets for MSE and err
    MSE = xr.Dataset(data_vars=None,
                     coords=dict(z=C.z, Tsample=Tnew),
                     attrs=C.attrs)
    err = xr.Dataset(data_vars=None,
                     coords=dict(z=C.z, Tsample=Tnew),
                     attrs=C.attrs)
    # store h in err
    err["h"] = stat.h
    #
    # now can recalc errors
    #
    # keys for looping through C, p
    param_RFM1 = ["u", "v", "theta"] 
    # keys for looping through stat to normalize MSE
    param_var2 = ["u_var", "v_var", "theta_var"] 
    # keys for looping through stat to normalize relative errors
    param_mean1 = ["u_mean", "v_mean", "theta_mean"] 
    # Loop through first-order moments to calculate MSE and error
    for i, v in enumerate(param_RFM1):
        # define empty DataArray for v in MSE
        MSE[v] =  xr.DataArray(data=np.zeros((len(isbl),len(Tnew)), dtype=np.float64), 
                                coords=dict(MSE.coords))
        # Loop through Tnew to calculate MSE(z,Tsample)
        for jt, iT in enumerate(Tnew):
            # calculate Xnew from Tnew and mean wind
            Xnew = stat.u_mean_rot.isel(z=isbl) * iT
            # use values of C and p to extrapolate calculation of MSE/var{x}
            # renormalize with variances in param_var2
            MSE[v][:,jt] = stat[param_var2[i]].isel(z=isbl) * (C[v] * (Xnew**-p[v]))
        # take sqrt to get RMSE
        RMSE = np.sqrt(MSE[v])
        # divide by <x> to get epsilon
        err[v] = RMSE / abs(stat[param_mean1[i]].isel(z=isbl))  
        
    # propagate uh and alpha errors from u and v
    # wind speed = uh, wind direction = alpha
    sig_u = np.sqrt(MSE["u"])
    sig_v = np.sqrt(MSE["v"])
    # calculate ws error and assign to RFM
    err_uh = np.sqrt( (sig_u**2. * stat.u_mean.isel(z=isbl)**2. +\
                    sig_v**2. * stat.v_mean.isel(z=isbl)**2.)/\
                    (stat.u_mean.isel(z=isbl)**2. +\
                    stat.v_mean.isel(z=isbl)**2.) ) / stat.u_mean_rot.isel(z=isbl)
    err["uh"] = err_uh
    # calculate wd error and assign to RFM
    err_alpha = np.sqrt( (sig_u**2. * stat.v_mean.isel(z=isbl)**2. +\
                        sig_v**2. * stat.u_mean.isel(z=isbl)**2.)/\
                        ((stat.u_mean.isel(z=isbl)**2. +\
                        stat.v_mean.isel(z=isbl)**2.)**2.) ) /\
            (stat.alpha.isel(z=isbl))  # normalize w/ rad
    err["alpha"] = err_alpha
    
    # assign units to err
    err["uh"].attrs["units"] = "%"
    err["alpha"].attrs["units"] = "%"
    err["theta"].attrs["units"] = "%"
    err["Tsample"].attrs["units"] = "s"
    if Tnew_ec is None:
        return err
    else:
        # check if Tnew is an array or single value and convert to iterable
        if np.shape(Tnew_ec) == ():
            Tnew_ec = np.array([Tnew_ec])
        # covariances and their 4th order variances
        param_RFM2 = ["uw_cov_tot", "vw_cov_tot", "tw_cov_tot"]
        param_cov = ["uwuw_var", "vwvw_var", "twtw_var"]
        param_mean2 = param_RFM2
        # variances and their 4th order variances
        param_RFM3 = ["uu_var", "uu_var_rot", "vv_var", "vv_var_rot", 
                    "ww_var", "tt_var"]
        param_var = ["uuuu_var", "uuuu_var_rot", "vvvv_var", "vvvv_var_rot", 
                    "wwww_var", "tttt_var"]
        param_mean3 = ["u_var", "u_var_rot", "v_var", "v_var_rot", "w_var", "theta_var"]
        # add Tnew_ec coordinate to MSE and err
        MSE = MSE.assign_coords(dict(Tsample_ec=Tnew_ec))
        err = err.assign_coords(dict(Tsample_ec=Tnew_ec))
        # Loop through covariances to calculate MSE and error
        for i, v in enumerate(param_RFM2):
            # define empty DataArray for v in MSE
            MSE[v] =  xr.DataArray(data=np.zeros((len(isbl),len(Tnew_ec)), dtype=np.float64), 
                                   coords=dict(z=MSE.z, Tsample_ec=MSE.Tsample_ec))
            for jt, iT in enumerate(Tnew_ec):
                # calculate Xnew from Tnew and mean wind
                Xnew = stat.u_mean_rot.isel(z=isbl) * iT
                # use values of C and p to extrapolate calculation of MSE/var{x}
                # renormalize with variances in param_var2
                MSE[v][:,jt] = var4[param_cov[i]].isel(z=isbl) * (C[v] * (Xnew**-p[v]))
            # take sqrt to get RMSE
            RMSE = np.sqrt(MSE[v])
            # divide by <x> to get epsilon
            err[v] = RMSE / abs(stat[param_mean2[i]].isel(z=isbl))    
        # Loop through variances to calculate MSE and error
        for i, v in enumerate(param_RFM3):
            # define empty DataArray for v in MSE
            MSE[v] =  xr.DataArray(data=np.zeros((len(isbl),len(Tnew_ec)), dtype=np.float64), 
                                   coords=dict(z=MSE.z, Tsample_ec=MSE.Tsample_ec))
            for jt, iT in enumerate(Tnew_ec):
                # calculate Xnew from Tnew and mean wind
                Xnew = stat.u_mean_rot.isel(z=isbl) * iT
                # use values of C and p to extrapolate calculation of MSE/var{x}
                # renormalize with variances in param_var2
                MSE[v][:,jt] = var4[param_var[i]].isel(z=isbl) * (C[v] * (Xnew**-p[v]))
            # take sqrt to get RMSE
            RMSE = np.sqrt(MSE[v])
            # divide by <x> to get epsilon
            err[v] = RMSE / abs(stat[param_mean3[i]].isel(z=isbl))

        # propagate errors for TKE from u_var, v_var, w_var
        err["e"] = np.sqrt(0.25 * ((err.uu_var*stat.u_var)**2. +\
                                   (err.vv_var*stat.v_var)**2. +\
                                   (err.ww_var*stat.w_var)**2.) ) / stat.e
        # propagate errors for ustar2 from uw_cov_tot, vw_cov_tot
        sig_uw = np.sqrt(MSE["uw_cov_tot"])
        sig_vw = np.sqrt(MSE["vw_cov_tot"])
        err_ustar2 = np.sqrt( (sig_uw**2. * stat.uw_cov_tot.isel(z=isbl)**2. +\
                            sig_vw**2. * stat.vw_cov_tot.isel(z=isbl)**2.)/\
                            (stat.ustar2.isel(z=isbl)**2.)) /\
                            stat.ustar2.isel(z=isbl)
        err["ustar2"] = err_ustar2

    return err
    
# --------------------------------
# Run script
# --------------------------------
if __name__ == "__main__":
    # load yaml file in global scope
    with open("/home/bgreene/SBL_LES/python/RFMnc.yaml") as f:
        config = yaml.safe_load(f)
    stability = config["stability"]
    fdir = f"/home/bgreene/simulations/{stability}/output/netcdf/"
    # text file to save print statements
    fprint = f"/home/bgreene/SBL_LES/output/Print/RFMnc_{stability}.txt"
    dt0 = datetime.utcnow()
    main()
    main2(plot_MSE=config["plot_MSE"])
    main3(reprocess=config["reprocess"])
    dt1 = datetime.utcnow()
    print_both(f"Total run time for RFMnc.py: {(dt1-dt0).total_seconds()/60.:5.2f} min", fprint)