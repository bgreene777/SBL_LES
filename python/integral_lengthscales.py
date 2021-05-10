# --------------------------------
# Name: integral_lengthsales.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 13 April 2021
# Purpose: Read volumetric output from LES to calculate autocorrelations,
# integral length scales, and random error profiles
# --------------------------------

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib import rc

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

# ---------------------------------
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
# ---------------------------------
def lengthscale(fdir, timesteps, nx, ny, nz, Lx, scale, str1, str2=None):
    """
    -input-
    fdir: root directory to load files from
    timesteps: numpy array of integers to loop over for reading 3d LES files
    nx, ny, nz: number of grid points in x, y, z
    Lx: length of domain in x
    scale: value to re-dimensionalize data from LES output
    str1: string name for files to be read (e.g., "u_", "theta_")
    str2: optional (default=None) second string for v to rotate coords
    -returns-
    corr_xx: size(2*nx-1, nz) array of y- and t-averaged autocorrelation
    var_len: size(nz) array of integral lengthscales
    """
    # calculate number of timesteps to initialize an empty array
    nt = len(timesteps)
    # calculate 1D correlation in x as function of z (average over y and t)
    corr_xx_all = np.zeros((2*nx-1, ny, nz, nt), dtype=float)

    for i, step in enumerate(timesteps):
        # read str1 file
        var1_file = f"{fdir}{str1}{step:07d}.out"
        var1_in = read_f90_bin(var1_file, nx, ny, nz, 8)
        var1_in *= scale
        # read str2 file if not none
        if str2 is not None:
            var2_file = f"{fdir}{str2}{step:07d}.out"
            var2_in = read_f90_bin(var2_file, nx, ny, nz, 8)
            var2_in *= scale
            # define new u and v to fill with rotated coords
            var1, var2 = [np.zeros((nx,ny,nz), dtype=float) for _ in range(2)]

            # now rotate coordinates so <v> = 0
            for k in range(nz):
                angle = np.arctan2(np.mean(var2_in[:,:,k]), np.mean(var1_in[:,:,k]))
                var1[:,:,k] = var1_in[:,:,k]*np.cos(angle) + var2_in[:,:,k]*np.sin(angle)
                var2[:,:,k] = -var1_in[:,:,k]*np.sin(angle) + var2_in[:,:,k]*np.cos(angle)
        else:
            var1 = var1_in
            var2 = None

        # begin calculating correlations
        # loop over vertical spatial lags
        for k in range(nz):
            # loop over y-axis to average
            for jy in range(ny):  
                temp1 = np.zeros(nx, dtype=float)
                temp2 = np.zeros(nx, dtype=float)
                temp1 = (var1[:,jy,k] - np.mean(var1[:,jy,k])) / np.std(var1[:,jy,k])
                temp2 = (var1[:,jy,k] - np.mean(var1[:,jy,k])) / np.std(var1[:,jy,k])
                temp2 /= nx
                corr_xx_all[:,jy,k,i] = fftconvolve(temp1[:], temp2[::-1], mode="full")

    # average over y and t    
    corr_xx = np.mean(corr_xx_all, axis=(1,3))

    # calculate integral lengh scale for u as function of z
    var_len, var_len_abs = [np.zeros(nz, dtype=float) for _ in range(2)]
    xlags = np.linspace(-Lx, Lx, 2*nx-1)
    zlags = z
    imid = len(xlags) // 2
    for k in range(nz):
        ifin = np.where(corr_xx[imid:,k] < 0.)[0]
        # make sure this index exists
        if len(ifin) > 0:
            ifin = ifin[0]
        else:
            ifin = imid - 1
        var_len[k] = np.trapz(corr_xx[imid:imid+ifin,k], xlags[imid:imid+ifin])
        var_len_abs[k] = np.trapz(abs(corr_xx[imid:,k]), xlags[imid:])
    
    print(f"Finished integral length scale for {str1.split('_')[0]}")
    return [corr_xx, var_len]
# ---------------------------------

# TODO: convert this to loop over the different sim resolutions
# array of resolutions to be used in looping
# resolutions = ["096", "128", "160", "192", "256"]
resolutions = ["192"]
# for 096, use first one; else use second
# timesteps_all = [np.arange(901000, 991000, 1000, dtype=int), 
#                  np.arange(991000, 1171000, 1000, dtype=int)]
timesteps_all = [np.arange(1081000, 1261000, 1000, dtype=int)]
# common parameters
Lx, Ly, Lz = 800., 800., 400.
u_scale = 0.4
T_scale = 300.

# define dictionaries to shove data into
corr = {}
length = {}
# initialize keys and more dictionaries
for res in resolutions:
    corr[res] = {}
    length[res] = {}

# begin loop
for i, res in enumerate(resolutions):
    # output directory to grab data and range of files to grab
#     if i == 0:
#         fdir = "/home/bgreene/simulations/C_spinup/output/"
#         timesteps = timesteps_all[0]
#     else:
#         fdir = f"/home/bgreene/simulations/C_{res}_interp/output/"
#         timesteps = timesteps_all[1]
    fdir = f"/home/bgreene/simulations/A_{res}_interp/output/"
    timesteps = timesteps_all[0]
    print(f"Timestep range {timesteps[0]} -- {timesteps[-1]}")
    # sim params
    nx, ny, nz = int(res), int(res), int(res)
    dx, dy, dz = Lx/(nx-1), Ly/(ny-1), Lz/(nz-1)
    z = np.linspace(0.,Lz,nz)
    x = np.linspace(0.,Lx,nx)

    # call lengthscale for u, w, theta
    corr[res]["u"], length[res]["u"] = lengthscale(fdir, timesteps, nx, ny, nz, Lx, u_scale, "u_", "v_")
    corr[res]["w"], length[res]["w"] = lengthscale(fdir, timesteps, nx, ny, nz, Lx, u_scale, "w_")
    corr[res]["theta"], length[res]["theta"] = lengthscale(fdir, timesteps, nx, ny, nz, Lx, u_scale, "theta_")

    # save output for error calcs and plotting in another script
    fsave = f"./lengthscales_A_{res}.npz"
    np.savez(fsave, z=z, u_corr=corr[res]["u"], u_len=length[res]["u"], 
             w_corr=corr[res]["w"], w_len=length[res]["w"],
             theta_corr=corr[res]["theta"], theta_len=length[res]["theta"])