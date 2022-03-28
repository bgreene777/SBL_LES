# --------------------------------
# Name: spectrogram.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 15 March 2022
# Purpose: calculate spectra from SBL simulations as functions of
# wavelength and height above ground and optionally plot
# --------------------------------
import xrft
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from LESnc import load_stats
# --------------------------------
# Define function to calculate spectra
# --------------------------------
def calc_spectra(dnc):
    """
    Calculate power spectra for u', w', theta', u'w', theta'w'
    and save single netcdf file for plotting later
    Input dnc: string path directory for location of netcdf files
    Output netcdf file in dnc
    """
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # load final hour of individual files into one dataset
    # note this is specific for SBL simulations
    timesteps = np.arange(1080000, 1260000+1, 1000, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*0.02*1000 for i in range(nf)])
    # read files
    print("Loading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    # calculate rotated u, v based on alpha in stats
    dd["u_rot"] = dd.u*np.cos(s.alpha) + dd.v*np.sin(s.alpha)
    dd["v_rot"] =-dd.u*np.sin(s.alpha) + dd.v*np.cos(s.alpha)

    #
    # calculate power spectra
    #
    print("Begin power spectrum calculations...")
    # u_rot
    E_uu = xrft.power_spectrum(dd.u_rot, dim="x", true_phase=True, true_amplitude=True)
    # average in time and y
    E_uu_ytmean = E_uu.mean(dim=("time","y"))
    # w
    E_ww = xrft.power_spectrum(dd.w, dim="x", true_phase=True, true_amplitude=True)
    # average in time and y
    E_ww_ytmean = E_ww.mean(dim=("time","y"))
    # theta
    E_tt = xrft.power_spectrum(dd.theta, dim="x", true_phase=True, true_amplitude=True)
    # average in time and y
    E_tt_ytmean = E_tt.mean(dim=("time","y"))
    # # u'w'
    # E_uw = xrft.cross_spectrum(dd.u_rot, dd.w, dim="x", scaling="density",
    #                            true_phase=True, true_amplitude=True)
    # # average in time and y
    # E_uw_ytmean = E_uw.mean(dim=("time","y"))
    # # theta'w'
    # E_tw = xrft.cross_spectrum(dd.theta, dd.w, dim="x", scaling="density",
    #                            true_phase=True, true_amplitude=True)
    # # average in time and y
    # E_tw_ytmean = E_tw.mean(dim=("time","y"))

    #
    # Combine yt-averaged spectra into one Dataset and save nc
    #
    # initialize empty xarray dataset
    E_save = xr.Dataset(data_vars=None,
                        coords=dict(z=E_uu_ytmean.z, 
                                    freq_x=E_uu_ytmean.freq_x),
                        attrs=s.attrs)
    # assign individual spectra
    E_save["uu"] = E_uu_ytmean
    E_save["ww"] = E_ww_ytmean
    E_save["tt"] = E_tt_ytmean
    # E_save["uw"] = E_uw_ytmean
    # E_save["tw"] = E_tw_ytmean
    # only save positive frequencies
    E_save = E_save.where(E_save.freq_x > 0., drop=True)
    # save file
    fsavenc = f"{dnc}spectrogram.nc"
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        E_save.to_netcdf(fsavenc, mode="w")

    return

# --------------------------------
# Define function to read output files and plot
# --------------------------------

def plot_spectrogram(dnc, figdir):
    """
    Multi-panel plot from single simulation of all premultiplied spectra
    Input dnc: directory with nc files for given sim
    Input figdir: directory to save output figures
    Output: saved figures in figdir
    """
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # test similarity scale for plotting
    # height where Ls is maximum
    s["zLs"] = s.z.isel(z=s.Ls.argmax())
    # load spectra file
    E = xr.load_dataset(dnc+"spectrogram.nc")

    # Fig 1: E_uu, E_ww, E_tt
    print("Begin plotting Fig 1")
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(14.8, 5))
    # cax1 = ax1.contour(E_uu_nondim.z/sA.h, 1/E_uu_nondim.freq_x/sA.h,
    #                    E_uu_nondim, levels=np.linspace(0.0, 0.75, 26),
    #                    extend="max")
    # cax1 = ax1.contour(E_uu_nondim.z, 1/E_uu_nondim.freq_x,
    #                    E_uu_nondim, levels=np.linspace(0.0, 0.75, 26),
    #                    extend="max")
    # cax1 = ax1.contour(E_uu_nondim.z/sA.Lo.max(), 1/E_uu_nondim.freq_x/sA.Lo.max(),
    #                 E_uu_nondim, levels=np.linspace(0.0, 0.75, 26),
    #                 extend="max")
    # cax1 = ax1.contour(E_uu_nondim.z/sA.Ls.isel(z=0), 1/E_uu_nondim.freq_x/sA.Ls.isel(z=0),
    #                    E_uu_nondim, levels=np.linspace(0.0, 0.75, 26),
    #                    extend="max")
    # Euu
    cax1_0 = ax1[0].contour(E.z/s.zLs, 1/E.freq_x/s.zLs, E.freq_x*E.uu/s.ustar0/s.ustar0/2/np.pi)
    # Eww
    cax1_1 = ax1[1].contour(E.z/s.zLs, 1/E.freq_x/s.zLs, E.freq_x*E.ww/s.ustar0/s.ustar0/2/np.pi)
    # Ett
    cax1_2 = ax1[2].contour(E.z/s.zLs, 1/E.freq_x/s.zLs, E.freq_x*E.tt/s.tstar0/s.tstar0/2/np.pi)
    # clean up
    ax1[0].set_xlabel("$z/z_{L_s}$")
    ax1[0].set_ylabel("$\\lambda_x/z_{L_s}$")
    ax1[0].set_xscale("log")
    ax1[0].set_yscale("log")
    ax1[1].set_xlabel("$z/z_{L_s}$")
    ax1[2].set_xlabel("$z/z_{L_s}$")
    # ax1.set_xlim([0.01, 1])
    # ax1.set_ylim([0.05, 10])
    cb1_0 = fig1.colorbar(cax1_0, ax=ax1[0], location="bottom")
    cb1_1 = fig1.colorbar(cax1_1, ax=ax1[1], location="bottom")
    cb1_2 = fig1.colorbar(cax1_2, ax=ax1[2], location="bottom")

    cb1_0.ax.set_xlabel("$k_x \\Phi_{uu} / u_*^2$")
    cb1_1.ax.set_xlabel("$k_x \\Phi_{ww} / u_*^2$")
    cb1_2.ax.set_xlabel("$k_x \\Phi_{\\theta\\theta} / \\theta_*^2$")

    for iax in ax1.flatten():
        iax.axhline(1, c="k", lw=2)
    # ax1.axvline(1, c="k", lw=2)

    # save
    fsave1 = f"{figdir}{E.stability}_uu_ww_tt.png"
    print(f"Saving figure {fsave1}")
    fig1.savefig(fsave1, dpi=300)
    plt.close(fig1)

    # # Fig 2: E_uw, E_tw
    # print("Begin plotting Fig 2...")
    # fig2, ax2 = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(14.8, 5))
    # # Euw
    # cax2_0 = ax2[0].contour(E.z/s.zLs, 1/E.freq_x/s.zLs, E.freq_x*E.uw/s.ustar0/s.ustar0/2/np.pi)
    # # Etw
    # cax2_1 = ax2[1].contour(E.z/s.zLs, 1/E.freq_x/s.zLs, E.freq_x*E.ww/s.ustar0/s.tstar0/2/np.pi)
    # # clean up
    # ax2[0].set_xlabel("$z/z_{L_s}$")
    # ax2[0].set_ylabel("$\\lambda_x/z_{L_s}$")
    # ax2[0].set_xscale("log")
    # ax2[0].set_yscale("log")
    # ax2[1].set_xlabel("$z/z_{L_s}$")
    # # ax1.set_xlim([0.01, 1])
    # # ax1.set_ylim([0.05, 10])
    # cb2_0 = fig2.colorbar(cax2_0, ax=ax2[0], location="bottom")
    # cb2_1 = fig2.colorbar(cax2_1, ax=ax2[1], location="bottom")

    # cb2_0.ax.set_xlabel("$k_x \\Phi_{uw} / u_*^2$")
    # cb2_1.ax.set_xlabel("$k_x \\Phi_{tw} / u_* \\theta_*$")

    # for iax in ax2.flatten():
    #     iax.axhline(1, c="k", lw=2)
    # # ax1.axvline(1, c="k", lw=2)

    # # save
    # fsave2 = f"{figdir}{E.stability}_uw_tw.png"
    # fig2.savefig(fsave2, dpi=300)
    # plt.close(fig2)

    return

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    figdir = "/home/bgreene/SBL_LES/figures/spectrogram/"
    # loop sims A--F
    for sim in list("DEF"):
        print(f"---Begin Sim {sim}---")
        ncdir = f"/home/bgreene/simulations/{sim}_192_interp/output/netcdf/"
        # run calc_spectra
        calc_spectra(ncdir)
        # run plot_spectrogram
        plot_spectrogram(ncdir, figdir)
        print(f"---End Sim {sim}---")
