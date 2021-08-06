# --------------------------------
# Name: spectra.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 21 May 2021
# Purpose: calculate total streamwise velocity energy spectra 
# for a given simulation and save npz file
# Update 23 July 2021: Calculate spectra for theta
# --------------------------------
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from simulation import *

# define stab and res
stab = "F"
res = "192"
# define simulation output directory based on stab and res
fdir = f"/home/bgreene/simulations/{stab}_{res}_interp/output/"
# define timesteps to evaluate
timesteps = np.arange(1081000, 1261000, 1000, dtype=int)
nt = len(timesteps)
# common parameters
Lx, Ly, Lz = 800., 800., 400.
u_scale = 0.4
T_scale = 300.
# sim params
nx, ny, nz = int(res), int(res), int(res)
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
z = np.linspace(dz,Lz-dz,nz)
x = np.linspace(0.,Lx-dx,nx)
# create simulation object to grab mean u and v to rotate
s = simulation(fdir, nx, ny, nz, Lx, Ly, Lz, stab)
s.read_csv()

# calculate rotation angle based on <U> and <V>
alpha = np.arctan2(s.xytavg["v"], s.xytavg["u"])

# create E_uu and E_uu_ytavg
E_uu = np.zeros((nx,ny,nz), dtype=float)
E_uu_ytavg = np.zeros((nx,nz), dtype=float)
# create E_tt and E_tt_ytavg
E_tt = np.zeros((nx,ny,nz), dtype=float)
E_tt_ytavg = np.zeros((nx,nz), dtype=float)

# begin looping over timesteps
for it, t in enumerate(timesteps):
    # load in u
    u_in = read_f90_bin(f"{fdir}u_{t:07d}.out",nx,ny,nz,8) * u_scale
    v_in = read_f90_bin(f"{fdir}v_{t:07d}.out",nx,ny,nz,8) * u_scale
    T_in = read_f90_bin(f"{fdir}theta_{t:07d}.out",nx,ny,nz,8) * T_scale
    # rotate u
    u_rot = u_in*np.cos(alpha) + v_in*np.sin(alpha)
    # forward FFT u along x-axis
    f_u = fft(u_rot, axis=0)
    f_T = fft(T_in, axis=0)
    # calculate E_uu - loop to get symmetric in x correct
    for ix in range(1, nx//2):
        E_uu[ix,:,:] = np.real( f_u[ix,:,:] * np.conj(f_u[ix,:,:]) )
        E_uu[nx-ix,:,:] = np.real( f_u[nx-ix,:,:] * np.conj(f_u[nx-ix,:,:]) )
        E_tt[ix,:,:] = np.real( f_T[ix,:,:] * np.conj(f_T[ix,:,:]) )
        E_tt[nx-ix,:,:] = np.real( f_T[nx-ix,:,:] * np.conj(f_T[nx-ix,:,:]) )
        
    # average in y and *add* to E_uu_ytavg (will divide by nt later)
    E_uu_ytavg += np.mean(E_uu, axis=1)
    E_tt_ytavg += np.mean(E_tt, axis=1)
    
# finished with time loop, now divide E_uu_ytavg by nt to get time average
E_uu_ytavg /= nt
E_tt_ytavg /= nt

# calculate array of freqz
freqz = fftfreq(nx, d=dx)

# save npz file for plotting separately
fsave = f"/home/bgreene/SBL_LES/output/spectra_{stab}_{res}.npz"
np.savez(fsave, z=z, E_uu=E_uu_ytavg, E_tt=E_tt_ytavg, freqz=freqz)