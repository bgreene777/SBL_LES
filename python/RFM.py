# --------------------------------
# Name: RFM.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 26 May 2021
# Purpose: Read volumetric output from LES to calculate autocorrelations,
# integral length scales, and random error profiles by using the 
# relaxed filtering method from Dias et al. 2018
# Combines code from random_errors_filter.py and integral_lengthscales.py
# --------------------------------
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
def relaxed_filter(f, dx_L, dmin, dmax, delta_x, Lx, nx, nz):
    """
    -input-
    f: 3D variable in x,y,z (e.g., u(x,y,z), theta(x,y,z))
    dx_L: 2d numpy array of normalized filter width sizes vs z
    dmin: minimum value of delta_x/L_H for filtering
    dmax: maximum value of delta_x/L_H for filtering
    delta_x: 1d array of filter widths in m
    Lx: size of domain in x (m)
    nx: number of grid points in x
    nz: number of grid points in z (ideally only values of z <= h)
    -output-
    var_f_all: dictionary of y-averaged std along x as function of delta_x and z
    """
    # since number of filter widths will be variable depending on z, will
    # need to loop over z (ugh) and store output in a dictionary instead of
    # numpy array
    var_f_all = {}
    
    # begin looping over z
    for kz in range(nz):
        # determine filter widths to use
        i_filt = np.where((dx_L[:,kz] >= dmin) & (dx_L[:,kz] <= dmax))[0]
        # initialize empty array of shape(len(i_filt))
        var_f = np.zeros(len(i_filt), dtype=np.float64)
        # now can loop through filter sizes
        for i, idx in enumerate(delta_x[i_filt]):
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
        var_f_all[kz] = var_f
    
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
def RFM(dx_LH, C, p):
    # function to be used with curve_fit
    return C * (dx_LH ** (-p))

# --------------------------------
# Main
# --------------------------------

# loop over all timesteps to get ensemble mean autocorr
# output directory
fdir = "/home/bgreene/simulations/F_192_interp/output/"

# load average statistics to get <U> and <V>
fstat = f"{fdir}average_statistics.csv"
dstat = np.loadtxt(fstat, delimiter=",", skiprows=1)
Ubar = dstat[:,1]
Uvar = dstat[:,13]
Vbar = dstat[:,2]
thetabar = dstat[:,4]
thetavar = dstat[:,18]
# dimensions
nx, ny, nz = 192, 192, 192
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
delta_x = np.logspace(np.log10(dx), np.log10(Lx),
                      num=100, base=10.0, dtype=np.float64)

# rotate U and V so <V> = 0
angle = np.arctan2(Vbar, Ubar)
Ubar_rot = Ubar*np.cos(angle) + Vbar*np.sin(angle)
Vbar_rot =-Ubar*np.sin(angle) + Vbar*np.cos(angle)

# initialize autocorrelation of u_rot, Ruu
Ruu = np.zeros((nx, nz_sbl), dtype=float)

# begin loop
for i in range(nt):
    # load files - DONT FORGET SCALES!
    f1 = f"{fdir}u_{timesteps[i]:07d}.out"
    u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
    f2 = f"{fdir}v_{timesteps[i]:07d}.out"
    v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale

    # rotate u and v so <v> = 0
    # 3D * 1D arrays multiply along last dimension
    # use angle from above based on *unrotated* Ubar and Vbar in stats file
    u_rot = u_in*np.cos(angle) + v_in*np.sin(angle)
    v_rot =-u_in*np.sin(angle) + v_in*np.cos(angle)
    
    # calculate autocorrelation of u_rot, Ruu
    # accumulate in single variable, then divide by number of timesteps after
    Ruu += autocorr(u_rot[:,:,isbl], nx)
    
# divide by number of timesteps to get average in time
Ruu /= nt

# now can calculate L_H
L_H = Bartlett(Ruu, Ubar_rot[isbl], xlags)

# create 2d array of delta_x/L_H for each z
dx_LH = np.array([delta_x / iLH for iLH in L_H]).T  # shape(len(delta_x), nz)

# run relaxed_filter for u_rot
# define var_u_all dictionary for time averaging later
var_u_all = {}
# need to loop through timesteps again
for i in range(nt):
    # load files - DONT FORGET SCALES!
    f1 = f"{fdir}u_{timesteps[i]:07d}.out"
    u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
    f2 = f"{fdir}v_{timesteps[i]:07d}.out"
    v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale

    # rotate u and v so <v> = 0
    # 3D * 1D arrays multiply along last dimension
    # use angle from above based on *unrotated* Ubar and Vbar in stats file
    u_rot = u_in*np.cos(angle) + v_in*np.sin(angle)
    v_rot =-u_in*np.sin(angle) + v_in*np.cos(angle)
    
    # run relaxed_filter
    var_u = relaxed_filter(u_rot, dx_LH, 0.2, 3., delta_x, Lx, nx, nz_sbl)
    # if i==0, create var_u_all; otherwise add var_u into var_u_all
    if i==0:
        for kz in range(nz_sbl):
            var_u_all[kz] = var_u[kz]
    else:
        for kz in range(nz_sbl):
            var_u_all[kz] += var_u[kz]
            
# time average by dividing by number of timesteps
# also grab the relevant dx_LH values used at each kz
dx_LH_u = {}
for kz in range(nz_sbl):
    var_u_all[kz] /= nt
    i_dx = np.where((dx_LH[:,kz] >= 0.2) & (dx_LH[:,kz] <= 3.))[0]
    dx_LH_u[kz] = dx_LH[i_dx,kz]
    
# now can loop over z to fit power law to each set of var_u_all vs dx_LH_u
C, p = (np.zeros(nz_sbl, dtype=np.float64) for _ in range(2))
for kz in range(nz_sbl):
    (C[kz], p[kz]), _ = curve_fit(f=RFM, xdata=dx_LH_u[kz], 
                                  ydata=var_u_all[kz]/Uvar[kz], 
                                  p0=[0.001,0.001])
    
# now calculate relative random error: epsilon = RMSE(x_delta) / <x>
# RMSE = C**0.5 * delta**(-p/2)

# use T = 3 sec and convert to x/L_H via Taylor
T = 3.  # s
x_LH = (Ubar_rot[isbl] * T) / L_H

# now using the values of C and p, extrapolate to calc MSE/var{x}
MSE = Uvar[isbl] * (C * (x_LH)**-p)
# take sqrt to get RMSE
RMSE = np.sqrt(MSE)
# finally divide by <U> to get epsilon
err_u = RMSE / Ubar_rot[isbl]

# now save output in an npz file
fsave = "/home/bgreene/SBL_LES/output/RFM_F192.npz"
np.savez(fsave, z=z, h=h, isbl=isbl, err_u=err_u, C_u=C, p_u=p)