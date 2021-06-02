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
# --------------------------------
import yaml
import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from datetime import datetime
from numba import njit
from simulation import read_f90_bin
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
        Qdx = (5.*60.) / U[jz]  # m
        # find indices in xlags smaller than Qdx
        iQdx = np.where(xlags <= Qdx)[0][-1]
        # calculate standard error
        varB = (1. + 2.*np.sum(R[:iQdx, jz]**2.)) / len(xlags)
        errB = np.sqrt(varB)
        # look at autocorrelation again to find first instance dipping below errB
        iLH = np.where(abs(R[:,jz]) <= errB)[0][0]
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
Vbar = dstat[:,2]
Wbar = dstat[:,3]
thetabar = dstat[:,4]
thetavar = dstat[:,18]
uw_cov_tot = dstat[:,7]
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

# rotate U and V so <V> = 0
angle = np.arctan2(Vbar, Ubar)
Ubar_rot = Ubar*np.cos(angle) + Vbar*np.sin(angle)
Vbar_rot =-Ubar*np.sin(angle) + Vbar*np.cos(angle)

# initialize autocorrelation of u_rot, Ruu
Ruu = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of theta, Rtt
Rtt = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize autocorrelation of uw_cov, Ruwuw
Ruwuw = np.zeros((nx, nz_sbl), dtype=np.float64)
# initialize uwuw_var to get 4th order moment for LP errors
uwuw_var = np.zeros((nx,ny,nz_sbl), dtype=np.float64)
# define var_u_all arrays for time averaging later
var_u_all, var_theta_all, var_uw_all =\
(np.zeros((nfilt,nz_sbl), dtype=np.float64) for _ in range(3))

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
    
    # calculate autocorrelation of u_rot, Ruu and theta, Rtt
    # accumulate in single variable, then divide by number of timesteps after
    Ruu += autocorr(u_rot[:,:,isbl], nx)
    Rtt += autocorr(theta_in[:,:,isbl], nx)
    Ruwuw += autocorr(uw_cov, nx)
    
    # run relaxed_filter - dont need L_H anymore to run since filtering over all scales
    var_u = relaxed_filter(u_rot, delta_x, Lx, nx, nz_sbl)
    var_theta = relaxed_filter(theta_in, delta_x, Lx, nx, nz_sbl)
    var_uw = relaxed_filter(uw_cov, delta_x, Lx, nx, nz_sbl)
    # add var_u into var_u_all
    var_u_all += var_u
    var_theta_all += var_theta
    var_uw_all += var_uw
    
# divide by number of timesteps to get average in time
Ruu /= nt
Rtt /= nt
Ruwuw /= nt
uwuw_var /= nt
uwuw_var_xytavg = np.mean(uwuw_var, axis=(0,1))
# time average filtered variances
var_u_all /= nt
var_theta_all /= nt
var_uw_all /= nt

# calculate integral lengthscales from autocorrelation
len_u = lengthscale(Ruu, xlags)
len_theta = lengthscale(Rtt, xlags)
len_uw = lengthscale(Ruwuw, xlags)
# also calculate LP errors
err_u_LP = LP_error(len_u, config["T_sample"], 
                    Uvar[isbl], Ubar_rot[isbl], Ubar_rot[isbl])
err_theta_LP = LP_error(len_theta, config["T_sample"], 
                        thetavar[isbl], thetabar[isbl], Ubar_rot[isbl])
err_uw_LP = LP_error(len_uw, config["T_sample"],
                     uwuw_var_xytavg, uw_cov_tot[isbl], Ubar_rot[isbl])

# now can calculate L_H
L_H_u = Bartlett(Ruu, Ubar_rot[isbl], xlags)
L_H_t = Bartlett(Rtt, Ubar_rot[isbl], xlags)
L_H_uw = Bartlett(Ruwuw, Ubar_rot[isbl], xlags)

# create 2d array of delta_x/L_H for each z
dx_LH_u = np.array([delta_x / iLH for iLH in L_H_u]).T  # shape(len(delta_x), nz)
dx_LH_t = np.array([delta_x / iLH for iLH in L_H_t]).T  # shape(len(delta_x), nz)
dx_LH_uw = np.array([delta_x / iLH for iLH in L_H_uw]).T  # shape(len(delta_x), nz)
            
# grab the relevant dx_LH and variance values used at each kz for fitting
dx_LH_u_fit = {}
dx_LH_t_fit = {}
dx_LH_uw_fit = {}
var_u_fit = {}
var_theta_fit = {}
var_uw_fit = {}
for kz in range(nz_sbl):
    # grab dx_LH values
    i_dx_u = np.where((dx_LH_u[:,kz] >= dmin_u) & (dx_LH_u[:,kz] <= dmax_u))[0]
    dx_LH_u_fit[kz] = dx_LH_u[i_dx_u,kz]
    var_u_fit[kz] = var_u_all[i_dx_u,kz]
    i_dx_t = np.where((dx_LH_t[:,kz] >= dmin_u) & (dx_LH_t[:,kz] <= dmax_u))[0]
    dx_LH_t_fit[kz] = dx_LH_t[i_dx_t,kz]
    var_theta_fit[kz] = var_theta_all[i_dx_t,kz]
    i_dx_uw = np.where((dx_LH_uw[:,kz] >= dmin_cov) & (dx_LH_uw[:,kz] <= dmax_cov))[0]
    dx_LH_uw_fit[kz] = dx_LH_uw[i_dx_uw,kz]
    var_uw_fit[kz] = var_uw_all[i_dx_uw,kz]
    
# now can loop over z to fit power law to each set of var_u_all vs dx_LH_u
C, p, Ctheta, ptheta, Cuw, puw = (np.zeros(nz_sbl, dtype=np.float64) for _ in range(6))
for kz in range(nz_sbl):
    (C[kz], p[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_u_fit[kz], 
                                  ydata=var_u_fit[kz]/Uvar[kz], 
                                  p0=[0.001,0.001])
    (Ctheta[kz], ptheta[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_t_fit[kz],
                                            ydata=var_theta_fit[kz]/thetavar[kz],
                                            p0=[0.001,0.001])
    (Cuw[kz], puw[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_uw_fit[kz],
                                      ydata=var_uw_fit[kz]/uwuw_var_xytavg[kz],
                                      p0=[0.001,0.001])
    
# now calculate relative random error: epsilon = RMSE(x_delta) / <x>
# RMSE = C**0.5 * delta**(-p/2)
# use T = 3 sec and convert to x/L_H via Taylor
T = config["T_sample"]  # s
x_LH_u = (Ubar_rot[isbl] * T) / L_H_u
x_LH_t = (Ubar_rot[isbl] * T) / L_H_t
x_LH_uw = (Ubar_rot[isbl] * T) / L_H_uw

# now using the values of C and p, extrapolate to calc MSE/var{x}
MSE = Uvar[isbl] * (C * (x_LH_u**-p))
MSE_theta = thetavar[isbl] * (Ctheta * (x_LH_t**-ptheta))
MSE_uw = uwuw_var_xytavg * (Cuw * (x_LH_uw**-puw))
# take sqrt to get RMSE
RMSE = np.sqrt(MSE)
RMSE_theta = np.sqrt(MSE_theta)
RMSE_uw = np.sqrt(MSE_uw)
# finally divide by <U> to get epsilon
err_u = RMSE / Ubar_rot[isbl]
err_theta = RMSE_theta / thetabar[isbl]
err_uw = RMSE_uw / uw_cov_tot[isbl]

# now save output in an npz file
fsave = config["fsave"]
print(f"Saving file: {fsave}")
np.savez(fsave, z=z, h=h, isbl=isbl, delta_x=delta_x, yaml=config,
         dx_LH_u=dx_LH_u, dx_LH_t=dx_LH_t, dx_LH_uw=dx_LH_uw,
         var_u=var_u_all, var_theta=var_theta_all, var_uw=var_uw_all,
         uwuw_var_xytavg=uwuw_var_xytavg,
         Ruu=Ruu, len_u=len_u, err_u_LP=err_u_LP,
         Rtt=Rtt, len_theta=len_theta, err_theta_LP=err_theta_LP,
         Ruwuw=Ruwuw, len_uw=len_uw, err_uw_LP=err_uw_LP,
         err_u=err_u, C_u=C, p_u=p,
         err_theta=err_theta, C_theta=Ctheta, p_theta=ptheta,
         err_uw=err_uw, C_uw=Cuw, p_uw=puw)