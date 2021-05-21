# --------------------------------
# Name: random_errors_filter.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 28 April 2021
# Purpose: Read volumetric output from LES to calculate autocorrelations,
# integral length scales, and random error profiles by using filtering
# method from Salesky et al. 2012
# Modified: 6 May 2021
# operate on 3D volumetric fields instead of looping over y and z
# --------------------------------

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib import rc
from datetime import datetime
from numba import jit, float64, vectorize, int64, njit

dt0 = datetime.utcnow()

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')

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
# --------------------------------
# @vectorize(float64(float64, float64, float64, int64, int64))
def lengthscale_filter(f, delta_x, Lx, nx, nz):
    """
    -input-
    f: 3D variable in x,y,z (e.g., u(x,y,z), theta(x,y,z))
    delta_x: numpy array of filter width sizes
    Lx: size of domain in x (m)
    nx: number of grid points in x
    nz: number of grid points in z
    -output-
    sigma_f: array of y-averaged std along x as function of z and deltax, size(nz,ndelta)
    """
    # get length of delta_x for number of filter widths
    ndelta = len(delta_x)
    
    # initialize arrays for sigma_f, f_fft, f_filtered
    sigma_f = np.zeros((nz,ndelta), dtype=np.float64)
    
    # loop through filter sizes
    for k in range(ndelta):
        # filter f at scale delta_x[k]
        # wavenumber increment
        dk = 2.*np.pi / Lx
        # set up array of filter transfer function
        filt = np.zeros(nx//2, dtype=np.float64)
        # assemble transfer function of box filter
        for i in range(1, nx//2):
            filt[i] = np.sin(i*dk*delta_x[k]/2.) / (i*dk*delta_x[k]/2.)
        # forward FFT
        f_fft = np.fft.fft(f, axis=0)
        # filter
        for i in range(1, nx//2):
            f_fft[i,:,:] *= filt[i]
            f_fft[nx-i-1,:,:] *= filt[i]
        # inverse FFT
        f_filtered = np.fft.ifft(f_fft, axis=0)
        # calc standard deviation
        # std only along x, which returns 2d in y,z; take mean along y, new axis=0
        sigma_f[:,k] = calc_meanstd(np.real(f_filtered),nx,nz)
#         sigma_f[:,k] = np.mean(np.std(f_filtered, axis=0), axis=0)
    
    return sigma_f
    
# --------------------------------
@njit
def calc_meanstd(f,nx,nz):
    std_yz = np.zeros((nx,nz), dtype=np.float64)
    to_return = np.zeros(nz, dtype=np.float64)
    
    for k in range(nz):
        for j in range(nx):
            std_yz[j,k] = np.std(f[:,j,k])
        to_return[k] = np.mean(std_yz[:,k])

    return to_return
# --------------------------------
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
timesteps = np.arange(1081000, 1261000, 1000, dtype=int)
# timesteps = np.arange(901000, 1081000, 1000, dtype=int)
nt = len(timesteps)
# U and Theta scales
u_scale = 0.4
theta_scale = 300.
# define filter min and max
# delta_x_min = 4.*dx
# delta_x_max = Lx/10.
# delta_x_min = Lx/6.
delta_x_min = dx
delta_x_max = Lx
# number of filter sizes
ndelta = 50
# logspace of filter widths
delta_x = np.logspace(np.log10(delta_x_min), np.log10(delta_x_max), 
                      num=ndelta, base=10.0, dtype=np.float64)

# rotate U and V so <V> = 0
angle = np.arctan2(Vbar, Ubar)
Ubar_rot = Ubar*np.cos(angle) + Vbar*np.sin(angle)
Vbar_rot =-Ubar*np.sin(angle) + Vbar*np.cos(angle)

# output lengthscales into sigma_f_all size(nz,ndelta,nt)
sigma_u_all = np.zeros((nz,ndelta,nt), dtype=float)
sigma_theta_all = np.zeros((nz,ndelta,nt), dtype=float)

# begin loop
for i in range(nt):
    dt2 = datetime.utcnow()
    # load files - DONT FORGET SCALES!
    f1 = f"{fdir}u_{timesteps[i]:07d}.out"
    u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
    f2 = f"{fdir}v_{timesteps[i]:07d}.out"
    v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
    f3 = f"{fdir}theta_{timesteps[i]:07d}.out"
    theta_in = read_f90_bin(f3,nx,ny,nz,8) * theta_scale
    
    # rotate u and v so <v> = 0
    # 3D * 1D arrays multiply along last dimension
    # use angle from above based on *unrotated* Ubar and Vbar in stats file
    u_rot = u_in*np.cos(angle) + v_in*np.sin(angle)
    v_rot =-u_in*np.sin(angle) + v_in*np.cos(angle)
    
    # now pass full 3D u_rot into lengthscale_filter()
    sigma_u_all[:,:,i] = lengthscale_filter(u_rot, delta_x, Lx, nx, nz)
    sigma_theta_all[:,:,i] = lengthscale_filter(theta_in, delta_x, Lx, nx, nz)

    dt3 = datetime.utcnow()
    print(f"Duration for timestep {timesteps[i]:07d}: {(dt3-dt2).total_seconds()} sec")

# average sigma_f over all t
sigma_u = np.mean(sigma_u_all, axis=2)
sigma_theta = np.mean(sigma_theta_all, axis=2)
# define A and its inverse to calculate C_u
A = delta_x ** -0.5
A_inv = np.linalg.pinv(A.reshape(1,-1))
C_u = np.zeros(nz, dtype=float)
C_theta = np.zeros(nz, dtype=float)
# loop over z to calculate C_u
# this uses the full delta_x range to fit the -1/2 power law
# this is temporary; more concerned at plotting sigma_f as function of delta_x
for k in range(nz): 
    C_u[k] = np.dot(sigma_u[k,:].reshape(1,-1), A_inv)[0][0]
    C_theta[k] = np.dot(sigma_theta[k,:].reshape(1,-1), A_inv)[0][0]
    
# calculate len_u from C_u and Uvar
len_u = (C_u**2.) / (2.*Uvar)
len_theta = (C_theta**2.) / (2.*thetavar)
    
# save npz file
fsave = f"../output/filtered_lengthscale_F_192_full.npz"
np.savez(fsave, delta_x=delta_x, dx=dx, Lx=Lx,
         C_u=C_u, len_u=len_u, sigma_u=sigma_u,
         C_theta=C_theta, len_theta=len_theta, sigma_theta=sigma_theta)

dt1 = datetime.utcnow()

print(f"Total time: {(dt1-dt0).total_seconds()/60.:5.1f} min")