#!/usr/bin/env python3
# --------------------------------
# Name: RFM.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 26 May 2021
# Purpose: Read volumetric output from LES to calculate autocorrelations,
# integral length scales, and random error profiles by using the 
# relaxed filtering method from Dias et al. 2018
# Combines code from random_errors_filter.py and integral_lengthscales.py
# --Updates--
# 28 May 2021: Include calculations for u'w'; begin changes to filter over
# all scales dx->Lx but only fit RFM in specified windows
# 15 June 2021: Include calculations for v'w', theta'w', 
# u'u', v'v', w'w', theta'theta'
# 16 June 2021: caught error in Bartlett converting Qdt -> Qdx via Taylor
# 8 July 2021: also calculate unrotated u and v var for error prop calc
# --------------------------------
import yaml
import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from datetime import datetime
from numba import njit
from simulation import read_f90_bin
from run_calc_stats import send_sms
# --------------------------------
# Define Functions
# --------------------------------
def relaxed_filter(f, delta_x, Lx, nx, nz):
    """
    -input-
    f: 3D variable in x,y,z (e.g., u(x,y,z), theta(x,y,z))
    delta_x: 1d array of filter widths in m
    Lx: size of domain in x (m)
    nx: number of grid points in x
    nz: number of grid points in z (ideally only values of z <= h)
    -output-
    var_f_all: dictionary of y-averaged std along x as function of delta_x and z
    """
    # initialize numpy array of shape(nfilt, nz) to store all var_f 
    nfilt = len(delta_x)
    var_f_all = np.zeros((nfilt, nz), dtype=np.float64)
    
    # begin looping over z
    for kz in range(nz):
        # initialize empty array of shape(nx)
        var_f = np.zeros(nfilt, dtype=np.float64)
        # now can loop through filter sizes
        for i, idx in enumerate(delta_x):
            # filter f at scale delta_x[k]
            # wavenumber increment
            dk = 2.*np.pi / Lx
            # set up array of filter transfer function
            filt = np.zeros(nx//2, dtype=np.float64)
            # assemble transfer function of box filter
            for j in range(1, nx//2):
                filt[j] = np.sin(j*dk*idx/2.) / (j*dk*idx/2.)
            # forward FFT
            f_fft = np.fft.fft(f[:,:,kz], axis=0)
            # filter
            for j in range(1, nx//2):
                f_fft[j,:] *= filt[j]
                f_fft[nx-j-1,:] *= filt[j]
            # inverse FFT
            f_filtered = np.fft.ifft(f_fft, axis=0)
            # calc standard deviation
            # std only along x, which returns 2d in y,z; take mean along y, new axis=0
            var_f[i] = ( calc_meanvar(np.real(f_filtered), nx) )
        # outside filter loop
        # append var_f to var_f_all to create list of variances at each filter width
        var_f_all[:,kz] = var_f
    
    return var_f_all
# --------------------------------
@njit
def calc_meanvar(f, ny):
    # initialize var_y, a 1d array of variances along x for each y
    var_y = np.zeros(ny, dtype=np.float64)

    for j in range(ny):
        var_y[j] = np.var(f[:,j])
    # calculate mean and return
    return np.mean(var_y)
# --------------------------------
def autocorr(f, nx):
    # accepts single 3d volume
    # returns y-averaged data shape(nx, nz)
    # i.e., only lags >= 0
    # calculate f' / sigma_f
    temp = (f - np.mean(f, axis=0)) / np.std(f, axis=0)
    # convolve temp to get autocorr in x,y,z
    corr_xx = fftconvolve(temp[:], temp[::-1]/nx, mode="full", axes=0)
    # average over y
    corr_xx_yavg = np.mean(corr_xx, axis=1)
    # only return lags >= 0
    imid = (2*nx-1) // 2
    return corr_xx_yavg[imid:, :]
# --------------------------------
def Bartlett(R, U, xlags):
    # input autocorrelation shape(nx, nz), mean rotated wind <U>,
    # and array of xlags
    # return estimate of L_H based on Bartlett large-lag standard error
    L_H = np.zeros(R.shape[1], dtype=np.float64)
    for jz in range(R.shape[1]):
        # Bartlett large-lag standard error
        # determine equivalent of Q*delta_t = 5 min in spatial coords
        Qdx = (5.*60.) * U[jz]  # m
        # find indices in xlags smaller than Qdx
        iQdx = np.where(xlags <= Qdx)[0]
        if np.size(iQdx) == 0:
            iQdx = 1
        else:
            iQdx = iQdx[-1]
        # calculate standard error
        varB = (1. + 2.*np.sum(R[:iQdx, jz]**2.)) / len(xlags)
        errB = np.sqrt(varB)
        # look at autocorrelation again to find first instance dipping below errB
        iLH = np.where(abs(R[:,jz]) <= errB)[0]
        if np.size(iLH) == 0:
            iLH = 1
        else:
            iLH = iLH[0]
        # xlags[iLH] is L_H, so insert this value into L_H array
        L_H[jz] = xlags[iLH]
    
    return L_H
# --------------------------------
def lengthscale(R, xlags):
    # input autocorrelation shape(nx, nz) and array of xlags
    # integrate R up to first zero crossing
    # return integral length scale
    length = np.zeros(R.shape[1], dtype=np.float64)
    for jz in range(R.shape[1]):
        # find first zero crossing
        izero = np.where(R[:,jz] < 0.)[0]
        # make sure this index exists
        if len(izero) > 0:
            izero = izero[0]
        else:
            izero = 1
        # now integrate from index 0 to izero
        length[jz] = np.trapz(R[:izero,jz], xlags[:izero])
    return length
# --------------------------------
def LP_error(length, T_sample, variance, mean, wspd):
    # calculate error from Lumley and Panofsky for given 
    # sample time T (and wspd to convert to spatial)
    # also requires ensemble variance and mean values
    # output 1d array of relative random errors
    L_sample = T_sample * wspd
    err = np.sqrt((2.*length*variance)/(L_sample*mean**2.))
    return err
# --------------------------------
def RFM(dx_LH, C, p):
    # function to be used with curve_fit
    return C * (dx_LH ** (-p))
# --------------------------------
@njit
def covar(var1, var2, var1mean, var2mean, sgs, nx, ny, nz):
    # input 2 variables, their xyt means, and corresponding subgrid data
    # output 3d field of "instantaneous" covar (filtered + subgrid)
    # initialize var1_fluc, var2_fluc to hold data
    var1_fluc = np.zeros((nx,ny,nz), dtype=np.float64)
    var2_fluc = np.zeros((nx,ny,nz), dtype=np.float64)
    # loop and calculate fluctuating components
    for jz in range(nz):
        var1_fluc[:,:,jz] = var1[:,:,jz] - var1mean[jz]
        var2_fluc[:,:,jz] = var2[:,:,jz] - var2mean[jz]
    # now multiply them
    var1var2_fluc = var1_fluc * var2_fluc
    # add in sgs component and return
    to_return = var1var2_fluc + sgs
    return to_return

# --------------------------------
# Main
# --------------------------------
# load yaml file
with open("/home/bgreene/SBL_LES/python/RFM.yaml") as f:
    config = yaml.safe_load(f)

# loop over all timesteps to get ensemble mean autocorr
# output directory
fdir = config["fdir"]

# load average statistics to get <U> and <V>
fstat = f"{fdir}average_statistics.csv"
dstat = np.loadtxt(fstat, delimiter=",", skiprows=1)
Ubar = dstat[:,1]
Uvar = dstat[:,13]
Uvar2= dstat[:,19]
Vbar = dstat[:,2]
Vvar = dstat[:,15]
Vvar2= dstat[:,20]
Wbar = dstat[:,3]
Wvar = dstat[:,17]
thetabar = dstat[:,4]
thetavar = dstat[:,18]
uw_cov_tot = dstat[:,7]
vw_cov_tot = dstat[:,9]
thetaw_cov_tot = dstat[:,11]
# dimensions
nx, ny, nz = config["res"], config["res"], config["res"]
Lx, Ly, Lz = 800., 800., 400.
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
z = np.linspace(dz, Lz-dz, nz)
# calculate ustar
ustar = ((dstat[:,7]**2.) + (dstat[:,9]**2.)) ** 0.25
# calculate SBL height h from ustar
i_h = np.where(ustar <= 0.05*ustar[0])[0][0]
h = z[i_h] / 0.95
# last index where z[i] < h; will only use these data in filtering
isbl = np.where(h-z >= 0.)[0]
nz_sbl = len(isbl)
# xlags for autocorr
xlags = np.linspace(0., Lx, nx)
timesteps = np.arange(1081000, 1261000, 1000, dtype=np.int)
nt = len(timesteps)
# U and Theta scales
u_scale = 0.4
theta_scale = 300.
# array of delta_x spaced logarithmic from dx to Lx
nfilt = config["nfilt"]
delta_x = np.logspace(np.log10(dx), np.log10(Lx),
                      num=nfilt, base=10.0, dtype=np.float64)
# grab dmin_u and dmax_u
dmin_u = config["dmin_u"]
dmax_u = config["dmax_u"]
# grab dmin_cov and dmax_cov
dmin_cov = config["dmin_cov"]
dmax_cov = config["dmax_cov"]
# grab dmin_var and dmax_var
dmin_var = config["dmin_var"]
dmax_var = config["dmax_var"]

# rotate U and V so <V> = 0
angle = np.arctan2(Vbar, Ubar)
Ubar_rot = Ubar*np.cos(angle) + Vbar*np.sin(angle)
Vbar_rot =-Ubar*np.sin(angle) + Vbar*np.cos(angle)

# initialize autocorrelation of u_rot, Ruu
Ruu = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of unrotated u, Ruu2
Ruu2 = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of v_rot, Rvv
Rvv = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of unrotated v, Rvv2
Rvv2 = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of theta, Rtt
Rtt = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of uw_cov, Ruwuw
Ruwuw = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of vw_cov, Rvwvw
Rvwvw = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of thetaw_cov, Rtwtw
Rtwtw = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of uu, vv, ww, tt variances: 
# Ruuuu, Rvvvv, Rwwww, Rtttt
Ruuuu, Rvvvv, Rwwww, Rtttt =\
(np.zeros((nx, nz_sbl), dtype=np.float64) for _ in range(4))
# initialize uwuw_var to get 4th order moment for LP errors
uwuw_var, vwvw_var, twtw_var =\
(np.zeros((nx,ny,nz_sbl), dtype=np.float64) for _ in range(3))
# initialize uuuu_var to get 4th order moment for LP errors
uuuu_var, vvvv_var, wwww_var, tttt_var =\
(np.zeros((nx,ny,nz_sbl), dtype=np.float64) for _ in range(4))
# define var_u_all arrays for time averaging later
var_u_all, var_u_all2, var_v_all, var_v_all2, var_theta_all,\
var_uw_all, var_vw_all, var_tw_all,\
var_uu_all, var_vv_all, var_ww_all, var_tt_all =\
(np.zeros((nfilt,nz_sbl), dtype=np.float64) for _ in range(12))

# begin loop
# run relaxed_filter for u_rot and theta
print("---Begin calculating autocorrelations and relaxed filtering---")
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
    
    # rotate u and v so <v> = 0
    # 3D * 1D arrays multiply along last dimension
    # use angle from above based on *unrotated* Ubar and Vbar in stats file
    u_rot = u_in*np.cos(angle) + v_in*np.sin(angle)
    v_rot =-u_in*np.sin(angle) + v_in*np.cos(angle)
    
    # calculate u'w' inst covar
    uw_cov = covar(u_in[:,:,isbl], w_in[:,:,isbl], 
                   Ubar[isbl], Wbar[isbl], txz_in[:,:,isbl], 
                   nx, ny, nz_sbl)
    # calculate variance of u'w' = u'w'u'w' to calculate average later
    uwuw_var += covar(uw_cov, uw_cov, uw_cov_tot[isbl], uw_cov_tot[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)
    
    # calculate v'w' inst covar
    vw_cov = covar(v_in[:,:,isbl], w_in[:,:,isbl], 
                   Vbar[isbl], Wbar[isbl], tyz_in[:,:,isbl], 
                   nx, ny, nz_sbl)
    # calculate variance of v'w' = v'w'v'w' to calculate average later
    vwvw_var += covar(vw_cov, vw_cov, vw_cov_tot[isbl], vw_cov_tot[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)
    
    # calculate theta'w' inst covar
    thetaw_cov = covar(theta_in[:,:,isbl], w_in[:,:,isbl], 
                       thetabar[isbl], Wbar[isbl], q3_in[:,:,isbl], 
                       nx, ny, nz_sbl)
    # calculate variance of theta'w' = theta'w'theta'w' to calculate average later
    twtw_var += covar(thetaw_cov, thetaw_cov, thetaw_cov_tot[isbl], thetaw_cov_tot[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)
            
    # calculate u'u' inst var
    uu_var = covar(u_rot[:,:,isbl], u_rot[:,:,isbl], 
                   Ubar_rot[isbl], Ubar_rot[isbl], 
                   np.zeros((nx,ny,nz_sbl), dtype=np.float64), 
                   nx, ny, nz_sbl)
    # calculate variance of u'u' = u'u'u'u' to calculate average later
    uuuu_var += covar(uu_var, uu_var, Uvar[isbl], Uvar[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)
             
    # calculate v'v' inst var
    vv_var = covar(v_rot[:,:,isbl], v_rot[:,:,isbl], 
                   Vbar_rot[isbl], Vbar_rot[isbl], 
                   np.zeros((nx,ny,nz_sbl), dtype=np.float64), 
                   nx, ny, nz_sbl)
    # calculate variance of v'v' = v'v'v'v' to calculate average later
    vvvv_var += covar(vv_var, vv_var, Vvar[isbl], Vvar[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)

    # calculate w'w' inst var
    ww_var = covar(w_in[:,:,isbl], w_in[:,:,isbl], 
                   Wbar[isbl], Wbar[isbl], 
                   np.zeros((nx,ny,nz_sbl), dtype=np.float64), 
                   nx, ny, nz_sbl)
    # calculate variance of w'w' = w'w'w'w' to calculate average later
    wwww_var += covar(ww_var, ww_var, Wvar[isbl], Wvar[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)
    
    # calculate t't' inst var
    tt_var = covar(theta_in[:,:,isbl], theta_in[:,:,isbl], 
                   thetabar[isbl], thetabar[isbl], 
                   np.zeros((nx,ny,nz_sbl), dtype=np.float64), 
                   nx, ny, nz_sbl)
    # calculate variance of t't' = t't't't' to calculate average later
    tttt_var += covar(tt_var, tt_var, thetavar[isbl], thetavar[isbl],
                      np.zeros((nx,ny,nz_sbl), dtype=np.float64),
                      nx, ny, nz_sbl)
    
    # calculate autocorrelation of u_rot, Ruu and theta, Rtt
    # accumulate in single variable, then divide by number of timesteps after
    Ruu += autocorr(u_rot[:,:,isbl], nx)
    Ruu2 += autocorr(u_in[:,:,isbl], nx) # unrotated
    Rvv += autocorr(v_rot[:,:,isbl], nx)
    Rvv2 += autocorr(v_in[:,:,isbl], nx) # unrotated
    Rtt += autocorr(theta_in[:,:,isbl], nx)
    Ruwuw += autocorr(uw_cov, nx)
    Rvwvw += autocorr(vw_cov, nx)
    Rtwtw += autocorr(thetaw_cov, nx)
    Ruuuu += autocorr(uu_var, nx)
    Rvvvv += autocorr(vv_var, nx)
    Rwwww += autocorr(ww_var, nx)
    Rtttt += autocorr(tt_var, nx)
    
    # run relaxed_filter - dont need L_H anymore to run since filtering over all scales
    var_u = relaxed_filter(u_rot, delta_x, Lx, nx, nz_sbl)
    var_u2 = relaxed_filter(u_in, delta_x, Lx, nx, nz_sbl)
    var_v = relaxed_filter(v_rot, delta_x, Lx, nx, nz_sbl)
    var_v2 = relaxed_filter(v_in, delta_x, Lx, nx, nz_sbl)
    var_theta = relaxed_filter(theta_in, delta_x, Lx, nx, nz_sbl)
    var_uw = relaxed_filter(uw_cov, delta_x, Lx, nx, nz_sbl)
    var_vw = relaxed_filter(vw_cov, delta_x, Lx, nx, nz_sbl)
    var_thetaw = relaxed_filter(thetaw_cov, delta_x, Lx, nx, nz_sbl)
    var_uu = relaxed_filter(uu_var, delta_x, Lx, nx, nz_sbl)
    var_vv = relaxed_filter(vv_var, delta_x, Lx, nx, nz_sbl)
    var_ww = relaxed_filter(ww_var, delta_x, Lx, nx, nz_sbl)
    var_tt = relaxed_filter(tt_var, delta_x, Lx, nx, nz_sbl)
    # add var_u into var_u_all
    var_u_all += var_u
    var_u_all2 += var_u2 # unrotated
    var_v_all += var_v
    var_v_all2 += var_v2 # unrotated
    var_theta_all += var_theta
    var_uw_all += var_uw
    var_vw_all += var_vw
    var_tw_all += var_thetaw
    var_uu_all += var_uu
    var_vv_all += var_vv
    var_ww_all += var_ww
    var_tt_all += var_tt
    
# divide by number of timesteps to get average in time
Ruu /= nt
Ruu2 /= nt
Rvv /= nt
Rvv2 /= nt
Rtt /= nt
Ruwuw /= nt
Rvwvw /= nt
Rtwtw /= nt
Ruuuu /= nt
Rvvvv /= nt
Rwwww /= nt
Rtttt /= nt
uwuw_var /= nt
uwuw_var_xytavg = np.mean(uwuw_var, axis=(0,1))
vwvw_var /= nt
vwvw_var_xytavg = np.mean(vwvw_var, axis=(0,1))
twtw_var /= nt
twtw_var_xytavg = np.mean(twtw_var, axis=(0,1))
uuuu_var /= nt
uuuu_var_xytavg = np.mean(uuuu_var, axis=(0,1))
vvvv_var /= nt
vvvv_var_xytavg = np.mean(vvvv_var, axis=(0,1))
wwww_var /= nt
wwww_var_xytavg = np.mean(wwww_var, axis=(0,1))
tttt_var /= nt
tttt_var_xytavg = np.mean(tttt_var, axis=(0,1))
# time average filtered variances
var_u_all /= nt
var_u_all2 /= nt
var_v_all /= nt
var_v_all2 /= nt
var_theta_all /= nt
var_uw_all /= nt
var_vw_all /= nt
var_tw_all /= nt
var_uu_all /= nt
var_vv_all /= nt
var_ww_all /= nt
var_tt_all /= nt

# calculate integral lengthscales from autocorrelation
len_u = lengthscale(Ruu, xlags)
len_u2 = lengthscale(Ruu2, xlags)
len_v = lengthscale(Rvv, xlags)
len_v2 = lengthscale(Rvv2, xlags)
len_theta = lengthscale(Rtt, xlags)
len_uw = lengthscale(Ruwuw, xlags)
len_vw = lengthscale(Rvwvw, xlags)
len_thetaw = lengthscale(Rtwtw, xlags)
len_uu = lengthscale(Ruuuu, xlags)
len_vv = lengthscale(Rvvvv, xlags)
len_ww = lengthscale(Rwwww, xlags)
len_tt = lengthscale(Rtttt, xlags)

# also calculate LP errors
err_u_LP = LP_error(len_u, config["T_sample_u"], 
                    Uvar[isbl], Ubar_rot[isbl], Ubar_rot[isbl])
err_u_LP2 = LP_error(len_u2, config["T_sample_u"], 
                    Uvar2[isbl], Ubar[isbl], Ubar_rot[isbl])
err_v_LP = LP_error(len_v, config["T_sample_u"], 
                    Vvar[isbl], Vbar_rot[isbl], Ubar_rot[isbl])
err_v_LP2 = LP_error(len_v2, config["T_sample_u"], 
                    Vvar2[isbl], Vbar[isbl], Ubar_rot[isbl])
err_theta_LP = LP_error(len_theta, config["T_sample_u"], 
                        thetavar[isbl], thetabar[isbl], Ubar_rot[isbl])
err_uw_LP = LP_error(len_uw, config["T_sample_cov"],
                     uwuw_var_xytavg, uw_cov_tot[isbl], Ubar_rot[isbl])
err_vw_LP = LP_error(len_vw, config["T_sample_cov"],
                     vwvw_var_xytavg, vw_cov_tot[isbl], Ubar_rot[isbl])
err_thetaw_LP = LP_error(len_thetaw, config["T_sample_cov"],
                         twtw_var_xytavg, thetaw_cov_tot[isbl], Ubar_rot[isbl])
err_uu_LP = LP_error(len_uu, config["T_sample_cov"],
                     uuuu_var_xytavg, Uvar[isbl], Uvar[isbl])
err_vv_LP = LP_error(len_vv, config["T_sample_cov"],
                     vvvv_var_xytavg, Vvar[isbl], Vvar[isbl])
err_ww_LP = LP_error(len_ww, config["T_sample_cov"],
                     wwww_var_xytavg, Wvar[isbl], Wvar[isbl])
err_tt_LP = LP_error(len_tt, config["T_sample_cov"],
                     tttt_var_xytavg, thetavar[isbl], thetavar[isbl])


# now can calculate L_H
L_H_u = np.mean(Bartlett(Ruu, Ubar_rot[isbl], xlags))
L_H_u2 = np.mean(Bartlett(Ruu2, Ubar_rot[isbl], xlags))
L_H_v = np.mean(Bartlett(Rvv, Ubar_rot[isbl], xlags))
L_H_v2 = np.mean(Bartlett(Rvv2, Ubar_rot[isbl], xlags))
L_H_t = np.mean(Bartlett(Rtt, Ubar_rot[isbl], xlags))
L_H_uw = np.mean(Bartlett(Ruwuw, Ubar_rot[isbl], xlags))
L_H_vw = np.mean(Bartlett(Rvwvw, Ubar_rot[isbl], xlags))
L_H_tw = np.mean(Bartlett(Rtwtw, Ubar_rot[isbl], xlags))
L_H_uu = np.mean(Bartlett(Ruuuu, Ubar_rot[isbl], xlags))
L_H_vv = np.mean(Bartlett(Rvvvv, Ubar_rot[isbl], xlags))
L_H_ww = np.mean(Bartlett(Rwwww, Ubar_rot[isbl], xlags))
L_H_tt = np.mean(Bartlett(Rtttt, Ubar_rot[isbl], xlags))

# # create 2d array of delta_x/L_H for each z
# dx_LH_u = np.array([delta_x / iLH for iLH in L_H_u]).T  # shape(len(delta_x), nz)
# dx_LH_u2 = np.array([delta_x / iLH for iLH in L_H_u2]).T  # shape(len(delta_x), nz)
# dx_LH_v = np.array([delta_x / iLH for iLH in L_H_v]).T  # shape(len(delta_x), nz)
# dx_LH_v2 = np.array([delta_x / iLH for iLH in L_H_v2]).T  # shape(len(delta_x), nz)
# dx_LH_t = np.array([delta_x / iLH for iLH in L_H_t]).T  # shape(len(delta_x), nz)
# dx_LH_uw = np.array([delta_x / iLH for iLH in L_H_uw]).T  # shape(len(delta_x), nz)
# dx_LH_vw = np.array([delta_x / iLH for iLH in L_H_vw]).T  # shape(len(delta_x), nz)
# dx_LH_tw = np.array([delta_x / iLH for iLH in L_H_tw]).T  # shape(len(delta_x), nz)
# dx_LH_uu = np.array([delta_x / iLH for iLH in L_H_uu]).T  # shape(len(delta_x), nz)
# dx_LH_vv = np.array([delta_x / iLH for iLH in L_H_vv]).T  # shape(len(delta_x), nz)
# dx_LH_ww = np.array([delta_x / iLH for iLH in L_H_ww]).T  # shape(len(delta_x), nz)
# dx_LH_tt = np.array([delta_x / iLH for iLH in L_H_tt]).T  # shape(len(delta_x), nz)

# create 2d array of delta_x/L_H for each z
dx_LH_u = delta_x / L_H_u     # shape(len(delta_x))
dx_LH_u2 = delta_x / L_H_u2   # shape(len(delta_x))
dx_LH_v = delta_x / L_H_v     # shape(len(delta_x))
dx_LH_v2 = delta_x / L_H_v2   # shape(len(delta_x))
dx_LH_t = delta_x / L_H_t     # shape(len(delta_x))
dx_LH_uw = delta_x / L_H_uw   # shape(len(delta_x))
dx_LH_vw = delta_x / L_H_vw   # shape(len(delta_x))
dx_LH_tw = delta_x / L_H_tw   # shape(len(delta_x))
dx_LH_uu = delta_x / L_H_uu   # shape(len(delta_x))
dx_LH_vv = delta_x / L_H_vv   # shape(len(delta_x))
dx_LH_ww = delta_x / L_H_ww   # shape(len(delta_x))
dx_LH_tt = delta_x / L_H_tt   # shape(len(delta_x))
            
# grab the relevant dx_LH and variance values used at each kz for fitting
dx_LH_u_fit = {}
dx_LH_u_fit2 = {}
dx_LH_v_fit = {}
dx_LH_v_fit2 = {}
dx_LH_t_fit = {}
dx_LH_uw_fit = {}
dx_LH_vw_fit = {}
dx_LH_tw_fit = {}
dx_LH_uu_fit = {}
dx_LH_vv_fit = {}
dx_LH_ww_fit = {}
dx_LH_tt_fit = {}
var_u_fit = {}
var_u_fit2 = {}
var_v_fit = {}
var_v_fit2 = {}
var_theta_fit = {}
var_uw_fit = {}
var_vw_fit = {}
var_tw_fit = {}
var_uu_fit = {}
var_vv_fit = {}
var_ww_fit = {}
var_tt_fit = {}
for kz in range(nz_sbl):
    # grab dx_LH values
    # u
    i_dx_u = np.where((dx_LH_u >= dmin_u) & (dx_LH_u <= dmax_u))[0]
    dx_LH_u_fit[kz] = dx_LH_u[i_dx_u]
    var_u_fit[kz] = var_u_all[i_dx_u,kz]
    # u unrotated
    i_dx_u2 = np.where((dx_LH_u2 >= dmin_u) & (dx_LH_u2 <= dmax_u))[0]
    dx_LH_u_fit2[kz] = dx_LH_u2[i_dx_u2]
    var_u_fit2[kz] = var_u_all2[i_dx_u2,kz]
    # v
    i_dx_v = np.where((dx_LH_v >= dmin_u) & (dx_LH_v <= dmax_u))[0]
    dx_LH_v_fit[kz] = dx_LH_v[i_dx_v]
    var_v_fit[kz] = var_v_all[i_dx_v,kz]
    # v unrotated
    i_dx_v2 = np.where((dx_LH_v2 >= dmin_u) & (dx_LH_v2 <= dmax_u))[0]
    dx_LH_v_fit2[kz] = dx_LH_v2[i_dx_v2]
    var_v_fit2[kz] = var_v_all2[i_dx_v2,kz]
    # theta
    i_dx_t = np.where((dx_LH_t >= dmin_u) & (dx_LH_t <= dmax_u))[0]
    dx_LH_t_fit[kz] = dx_LH_t[i_dx_t]
    var_theta_fit[kz] = var_theta_all[i_dx_t,kz]
    # u'w'
    i_dx_uw = np.where((dx_LH_uw >= dmin_cov) & (dx_LH_uw <= dmax_cov))[0]
    dx_LH_uw_fit[kz] = dx_LH_uw[i_dx_uw]
    var_uw_fit[kz] = var_uw_all[i_dx_uw,kz]
    # v'w'
    i_dx_vw = np.where((dx_LH_vw >= dmin_cov) & (dx_LH_vw <= dmax_cov))[0]
    dx_LH_vw_fit[kz] = dx_LH_vw[i_dx_vw]
    var_vw_fit[kz] = var_vw_all[i_dx_vw,kz]
    # theta'w'
    i_dx_tw = np.where((dx_LH_tw >= dmin_cov) & (dx_LH_tw <= dmax_cov))[0]
    dx_LH_tw_fit[kz] = dx_LH_tw[i_dx_tw]
    var_tw_fit[kz] = var_tw_all[i_dx_tw,kz]
    # u'u'
    i_dx_uu = np.where((dx_LH_uu >= dmin_var) & (dx_LH_uu <= dmax_var))[0]
    dx_LH_uu_fit[kz] = dx_LH_uu[i_dx_uu]
    var_uu_fit[kz] = var_uu_all[i_dx_uu,kz]
    # v'v'
    i_dx_vv = np.where((dx_LH_vv >= dmin_var) & (dx_LH_vv <= dmax_var))[0]
    dx_LH_vv_fit[kz] = dx_LH_vv[i_dx_vv]
    var_vv_fit[kz] = var_vv_all[i_dx_vv,kz]
    # w'w'
    i_dx_ww = np.where((dx_LH_ww >= dmin_var) & (dx_LH_ww <= dmax_var))[0]
    dx_LH_ww_fit[kz] = dx_LH_ww[i_dx_ww]
    var_ww_fit[kz] = var_ww_all[i_dx_ww,kz]
    # theta'theta'
    i_dx_tt = np.where((dx_LH_tt >= dmin_var) & (dx_LH_tt <= dmax_var))[0]
    dx_LH_tt_fit[kz] = dx_LH_tt[i_dx_tt]
    var_tt_fit[kz] = var_tt_all[i_dx_tt,kz]
    
# now can loop over z to fit power law to each set of var_u_all vs dx_LH_u
C, p, Cv, pv, Ctheta, ptheta, Cuw, puw, Cvw, pvw, Ctw, ptw,\
Cuu, puu, Cvv, pvv, Cww, pww, Ctt, ptt, Cu2, pu2, Cv2, pv2=\
(np.zeros(nz_sbl, dtype=np.float64) for _ in range(24))
for kz in range(nz_sbl):
    (C[kz], p[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_u_fit[kz], 
                                  ydata=var_u_fit[kz]/Uvar[kz], 
                                  p0=[0.001,0.001])
    (Cu2[kz], pu2[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_u_fit2[kz], 
                                      ydata=var_u_fit2[kz]/Uvar2[kz], 
                                      p0=[0.001,0.001])
    (Cv[kz], pv[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_v_fit[kz], 
                                    ydata=var_v_fit[kz]/Vvar[kz], 
                                    p0=[0.001,0.001])
    (Cv2[kz], pv2[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_v_fit2[kz], 
                                      ydata=var_v_fit2[kz]/Vvar2[kz], 
                                      p0=[0.001,0.001])
    (Ctheta[kz], ptheta[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_t_fit[kz],
                                            ydata=var_theta_fit[kz]/thetavar[kz],
                                            p0=[0.001,0.001])
    (Cuw[kz], puw[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_uw_fit[kz],
                                      ydata=var_uw_fit[kz]/uwuw_var_xytavg[kz],
                                      p0=[0.001,0.001])
    (Cvw[kz], pvw[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_vw_fit[kz],
                                      ydata=var_vw_fit[kz]/vwvw_var_xytavg[kz],
                                      p0=[0.001,0.001])
    (Ctw[kz], ptw[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_tw_fit[kz],
                                      ydata=var_tw_fit[kz]/twtw_var_xytavg[kz],
                                      p0=[0.001,0.001])
    (Cuu[kz], puu[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_uu_fit[kz],
                                      ydata=var_uu_fit[kz]/uuuu_var_xytavg[kz],
                                      p0=[0.001,0.001])
    (Cvv[kz], pvv[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_vv_fit[kz],
                                      ydata=var_vv_fit[kz]/vvvv_var_xytavg[kz],
                                      p0=[0.001,0.001])
    if kz > 0:
        (Cww[kz], pww[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_ww_fit[kz],
                                          ydata=var_ww_fit[kz]/wwww_var_xytavg[kz],
                                          p0=[0.001,0.001])
    (Ctt[kz], ptt[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_tt_fit[kz],
                                      ydata=var_tt_fit[kz]/tttt_var_xytavg[kz],
                                      p0=[0.001,0.001])
    
# now calculate relative random error: epsilon = RMSE(x_delta) / <x>
# RMSE = C**0.5 * delta**(-p/2)
# use T from config and convert to x/L_H via Taylor
# separate for u,v,theta / uw,vw,tw
T1 = config["T_sample_u"]  # s
T2 = config["T_sample_cov"]  # s
x_LH_u = (Ubar_rot[isbl] * T1) / L_H_u
x_LH_u2 = (Ubar_rot[isbl] * T1) / L_H_u2
x_LH_v = (Ubar_rot[isbl] * T1) / L_H_v
x_LH_v2 = (Ubar_rot[isbl] * T1) / L_H_v2
x_LH_t = (Ubar_rot[isbl] * T1) / L_H_t
x_LH_uw = (Ubar_rot[isbl] * T2) / L_H_uw
x_LH_vw = (Ubar_rot[isbl] * T2) / L_H_vw
x_LH_tw = (Ubar_rot[isbl] * T2) / L_H_tw
x_LH_uu = (Ubar_rot[isbl] * T2) / L_H_uu
x_LH_vv = (Ubar_rot[isbl] * T2) / L_H_vv
x_LH_ww = (Ubar_rot[isbl] * T2) / L_H_ww
x_LH_tt = (Ubar_rot[isbl] * T2) / L_H_tt

# now using the values of C and p, extrapolate to calc MSE/var{x}
MSE = Uvar[isbl] * (C * (x_LH_u**-p))
MSE_u2 = Uvar2[isbl] * (Cu2 * (x_LH_u2**-pu2))  # unrotated
MSE_v = Vvar[isbl] * (Cv * (x_LH_v**-pv))
MSE_v2 = Vvar2[isbl] * (Cv2 * (x_LH_v2**-pv2))  # unrotated
MSE_theta = thetavar[isbl] * (Ctheta * (x_LH_t**-ptheta))
MSE_uw = uwuw_var_xytavg * (Cuw * (x_LH_uw**-puw))
MSE_vw = vwvw_var_xytavg * (Cvw * (x_LH_vw**-pvw))
MSE_tw = twtw_var_xytavg * (Ctw * (x_LH_tw**-ptw))
MSE_uu = uuuu_var_xytavg * (Cuu * (x_LH_uu**-puu))
MSE_vv = vvvv_var_xytavg * (Cvv * (x_LH_vv**-pvv))
MSE_ww = wwww_var_xytavg * (Cww * (x_LH_ww**-pww))
MSE_tt = tttt_var_xytavg * (Ctt * (x_LH_tt**-ptt))
# take sqrt to get RMSE
RMSE = np.sqrt(MSE)
RMSE_u2 = np.sqrt(MSE_u2)
RMSE_v = np.sqrt(MSE_v)
RMSE_v2 = np.sqrt(MSE_v2)
RMSE_theta = np.sqrt(MSE_theta)
RMSE_uw = np.sqrt(MSE_uw)
RMSE_vw = np.sqrt(MSE_vw)
RMSE_tw = np.sqrt(MSE_tw)
RMSE_uu = np.sqrt(MSE_uu)
RMSE_vv = np.sqrt(MSE_vv)
RMSE_ww = np.sqrt(MSE_ww)
RMSE_tt = np.sqrt(MSE_tt)
# finally divide by <U> to get epsilon
err_u = RMSE / abs(Ubar_rot[isbl])
err_u2 = RMSE_u2 / abs(Ubar[isbl])
err_v = RMSE_v
err_v2 = RMSE_v2 / abs(Vbar[isbl])
err_theta = RMSE_theta / abs(thetabar[isbl])
err_uw = RMSE_uw / abs(uw_cov_tot[isbl])
err_vw = RMSE_vw / abs(vw_cov_tot[isbl])
err_tw = RMSE_tw / abs(thetaw_cov_tot[isbl])
err_uu = RMSE_uu / abs(Uvar[isbl])
err_vv = RMSE_vv / abs(Vvar[isbl])
err_ww = RMSE_ww / abs(Wvar[isbl])
err_tt = RMSE_tt / abs(thetavar[isbl])

# now save output in an npz file
fsave = config["fsave"]
print(f"Saving file: {fsave}")
np.savez(fsave, z=z, h=h, isbl=isbl, delta_x=delta_x, yaml=config,
         dx_LH_u=dx_LH_u, dx_LH_v=dx_LH_v, dx_LH_t=dx_LH_t, 
         dx_LH_u2=dx_LH_u2, dx_LH_v2=dx_LH_v2,
         dx_LH_uw=dx_LH_uw, dx_LH_vw=dx_LH_vw, dx_LH_tw=dx_LH_tw,
         dx_LH_uu=dx_LH_uu, dx_LH_vv=dx_LH_vv, dx_LH_ww=dx_LH_ww, 
         dx_LH_tt=dx_LH_tt,
         var_u=var_u_all, var_v=var_v_all, var_theta=var_theta_all, 
         var_u2=var_u_all2, var_v2=var_v_all2,
         var_uw=var_uw_all, var_vw=var_vw_all, var_tw=var_tw_all,
         var_uu=var_uu_all, var_vv=var_vv_all, var_ww=var_ww_all,
         var_tt=var_tt_all,
         uwuw_var_xytavg=uwuw_var_xytavg, vwvw_var_xytavg=vwvw_var_xytavg,
         twtw_var_xytavg=twtw_var_xytavg,
         uuuu_var_xytavg=uuuu_var_xytavg, vvvv_var_xytavg=vvvv_var_xytavg,
         wwww_var_xytavg=wwww_var_xytavg, tttt_var_xytavg=tttt_var_xytavg,
         Ruu=Ruu, len_u=len_u, err_u_LP=err_u_LP,
         Ruu2=Ruu2, len_u2=len_u2, err_u_LP2=err_u_LP2,
         Rvv=Rvv, len_v=len_v, err_v_LP=err_v_LP,
         Rvv2=Rvv2, len_v2=len_v2, err_v_LP2=err_v_LP2,
         Rtt=Rtt, len_theta=len_theta, err_theta_LP=err_theta_LP,
         Ruwuw=Ruwuw, len_uw=len_uw, err_uw_LP=err_uw_LP,
         Rvwvw=Rvwvw, len_vw=len_vw, err_vw_LP=err_vw_LP,
         Rtwtw=Rtwtw, len_thetaw=len_thetaw, err_tw_LP=err_thetaw_LP,
         Ruuuu=Ruuuu, len_uu=len_uu, err_uu_LP=err_uu_LP,
         Rvvvv=Rvvvv, len_vv=len_vv, err_vv_LP=err_vv_LP,
         Rwwww=Rwwww, len_ww=len_ww, err_ww_LP=err_ww_LP,
         Rtttt=Rtttt, len_tt=len_tt, err_tt_LP=err_tt_LP,
         err_u=err_u, C_u=C, p_u=p,
         err_u2=err_u2, C_u2=Cu2, p_u2=pu2,
         err_v=err_v, C_v=Cv, p_v=pv,
         err_v2=err_v2, C_v2=Cv2, p_v2=pv2,
         err_theta=err_theta, C_theta=Ctheta, p_theta=ptheta,
         err_uw=err_uw, C_uw=Cuw, p_uw=puw,
         err_vw=err_vw, C_vw=Cvw, p_vw=pvw,
         err_tw=err_tw, C_tw=Ctw, p_tw=ptw,
         err_uu=err_uu, C_uu=Cuu, p_uu=puu,
         err_vv=err_vv, C_vv=Cvv, p_vv=pvv,
         err_ww=err_ww, C_ww=Cww, p_ww=pww,
         err_tt=err_tt, C_tt=Ctt, p_tt=ptt
        )

# send_sms("/home/bgreene/SBL_LES/python/sms.yaml", f"Finished RFM.py, saved {config['fsave']}")