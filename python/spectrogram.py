# --------------------------------
# Name: spectrogram.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 15 March 2022
# Purpose: calculate spectra from SBL simulations as functions of
# wavelength and height above ground and optionally plot
# --------------------------------
import os
import xrft
import seaborn
import cmocean
import numpy as np
import xarray as xr
from scipy.signal import hilbert
from scipy.stats import gmean
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from LESnc import load_stats, load_full, MidPointNormalize
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
    # load data and stats files for dimensions
    dd, s = load_full(dnc, 1080000, 1260000, 1000, 0.02, True, True)

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
    # u'w'
    E_uw = xrft.cross_spectrum(dd.u_rot, dd.w, dim="x", scaling="density",
                               true_phase=True, true_amplitude=True)
    # average in time and y, only take real component
    E_uw_ytmean = np.real(E_uw.mean(dim=("time","y")))
    # theta'w'
    E_tw = xrft.cross_spectrum(dd.theta, dd.w, dim="x", scaling="density",
                               true_phase=True, true_amplitude=True)
    # average in time and y, only take real component
    E_tw_ytmean = np.real(E_tw.mean(dim=("time","y")))
    # theta'u'
    E_tu = xrft.cross_spectrum(dd.theta, dd.u_rot, dim="x", scaling="density",
                               true_phase=True, true_amplitude=True)
    # average in time and y, only take real component
    E_tu_ytmean = np.real(E_tu.mean(dim=("time","y")))

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
    E_save["uw"] = E_uw_ytmean
    E_save["tw"] = E_tw_ytmean
    E_save["tu"] = E_tu_ytmean
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
    # configure plots
    nlevel = 36
    cmap = seaborn.color_palette("cubehelix", as_cmap=True)
    cmap_r = seaborn.color_palette("cubehelix_r", as_cmap=True)
    # cmap2 = seaborn.color_palette("vlag", as_cmap=True)
    cmap2 = cmocean.cm.balance
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # test similarity scale for plotting
    # height where Ls is maximum
    s["zLs"] = s.z.isel(z=s.Ls.argmax())
    # load spectra file
    E = xr.load_dataset(dnc+"spectrogram.nc")

    # Fig 1: E_uu, E_ww, E_tt
    print("Begin plotting Fig 1")
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, 
                             figsize=(14, 5), constrained_layout=True)
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
    cax1_0 = ax1[0].contour(E.z/s.h, 1/E.freq_x/s.h, E.freq_x*E.uu/s.ustar0/s.ustar0,
                            levels=np.linspace(0.0, 0.8, nlevel), extend="max", cmap=cmap)
    # Eww
    cax1_1 = ax1[1].contour(E.z/s.h, 1/E.freq_x/s.h, E.freq_x*E.ww/s.ustar0/s.ustar0,
                            levels=np.linspace(0.0, 0.8, nlevel), extend="max", cmap=cmap)
    # Ett
    cax1_2 = ax1[2].contour(E.z/s.h, 1/E.freq_x/s.h, E.freq_x*E.tt/s.tstar0/s.tstar0,
                            levels=np.linspace(0.0, 0.5, nlevel), extend="max", cmap=cmap)
    # clean up
    ax1[0].set_xlabel("$z/h$")
    ax1[0].set_ylabel("$\\lambda_x/h$")
    ax1[0].set_xscale("log")
    ax1[0].set_yscale("log")
    ax1[1].set_xlabel("$z/h$")
    ax1[2].set_xlabel("$z/h$")
    # ax1.set_xlim([0.01, 1])
    # ax1.set_ylim([0.05, 10])
    cb1_0 = fig1.colorbar(cax1_0, ax=ax1[0], location="bottom", 
                          ticks=MultipleLocator(0.2), shrink=0.8)
    cb1_1 = fig1.colorbar(cax1_1, ax=ax1[1], location="bottom", 
                          ticks=MultipleLocator(0.2), shrink=0.8)
    cb1_2 = fig1.colorbar(cax1_2, ax=ax1[2], location="bottom", 
                          ticks=MultipleLocator(0.1), shrink=0.8)

    cb1_0.ax.set_xlabel("$k_x \\Phi_{uu} / u_*^2$")
    cb1_1.ax.set_xlabel("$k_x \\Phi_{ww} / u_*^2$")
    cb1_2.ax.set_xlabel("$k_x \\Phi_{\\theta\\theta} / \\theta_*^2$")

    # for iax in ax1.flatten():
    #     iax.axhline(s.zLs/s.h, c="k", lw=2)
    # ax1.axvline(1, c="k", lw=2)

    # save
    fsave1 = f"{figdir}{E.stability}_uu_ww_tt.png"
    print(f"Saving figure {fsave1}")
    fig1.savefig(fsave1, dpi=300)
    plt.close(fig1)

    # Fig 2: E_uw, E_tw
    print("Begin plotting Fig 2...")
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True,
                             figsize=(14, 5), constrained_layout=True)
    # Euw
    cax2_0 = ax2[0].contour(E.z/s.h, 1/E.freq_x/s.h, E.freq_x*E.uw/s.ustar0/s.ustar0,
                            levels=np.linspace(-0.2, 0.0, nlevel), extend="both", cmap=cmap_r)
    # Etw
    cax2_1 = ax2[1].contour(E.z/s.h, 1/E.freq_x/s.h, E.freq_x*E.tw/s.ustar0/s.tstar0,
                            levels=np.linspace(-0.2, 0.0, nlevel), extend="both", cmap=cmap_r)
    # Etu
    norm=MidPointNormalize(midpoint=0.0)
    cax2_2 = ax2[2].contour(E.z/s.h, 1/E.freq_x/s.h, E.freq_x*E.tu/s.ustar0/s.tstar0,
                            levels=np.linspace(-0.2, 0.4, nlevel), extend="both", cmap=cmap2, norm=norm)
    # clean up
    ax2[0].set_xlabel("$z/h$")
    ax2[0].set_ylabel("$\\lambda_x/h$")
    ax2[0].set_xscale("log")
    ax2[0].set_yscale("log")
    ax2[1].set_xlabel("$z/h$")
    ax2[2].set_xlabel("$z/h$")
    # ax1.set_xlim([0.01, 1])
    # ax1.set_ylim([0.05, 10])
    cb2_0 = fig2.colorbar(cax2_0, ax=ax2[0], location="bottom",
                          ticks=MultipleLocator(0.1), shrink=0.8)
    cb2_1 = fig2.colorbar(cax2_1, ax=ax2[1], location="bottom",
                          ticks=MultipleLocator(0.1), shrink=0.8)
    cb2_2 = fig2.colorbar(cax2_2, ax=ax2[2], location="bottom",
                          ticks=MultipleLocator(0.1), shrink=0.8)

    cb2_0.ax.set_xlabel("$k_x \\Phi_{uw} / u_*^2$")
    cb2_1.ax.set_xlabel("$k_x \\Phi_{\\theta w} / u_* \\theta_*$")
    cb2_2.ax.set_xlabel("$k_x \\Phi_{\\theta u} / u_* \\theta_*$")

    # for iax in ax2.flatten():
    #     iax.axhline(s.zLs/s.h, c="k", lw=2)
    # ax1.axvline(1, c="k", lw=2)
    # print(f"zLs/h = {(s.zLs/s.h).values:3.2f}")
    # save
    fsave2 = f"{figdir}{E.stability}_uw_tw_tu.png"
    print(f"Saving figure {fsave2}")
    fig2.savefig(fsave2, dpi=300)
    plt.close(fig2)

    return

# --------------------------------
# Define function to read output files and plot 1D spectra
# --------------------------------

def plot_1D_spectra(dnc, figdir):
    """
    Multi-panel plot from single simulation of all premultiplied spectra
    Input dnc: directory with nc files for given sim
    Input figdir: directory to save output figures
    Output: saved figures in figdir
    """
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # load spectra file
    E = xr.load_dataset(dnc+"spectrogram.nc")
    # add z/h as coordinate and swap with z
    # define array of z/h
    zh = E.z / s.he
    # assign coupled with z
    E = E.assign_coords(zh=("z",zh.values))
    # swap
    E = E.swap_dims({"z": "zh"})
    # do same for s
    s = s.assign_coords(zh=("z",zh.values)).swap_dims({"z": "zh"})
    # Kolmogorov constant
    Cu = (18./55) * 1.55
    Cw = (24./55) * 1.55

    # three panel plot
    # Euu, Eww, Ett
    # init figure
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(14.8, 5))
    # loop over heights z/h
    zhplot = np.arange(0.1, 0.9, 0.1)
    for jz in zhplot:
        # Euu
        ax[0].plot(E.freq_x * E.z.sel(zh=jz, method="nearest"), 
                   E.uu.sel(zh=jz, method="nearest") /\
                   (Cu*abs(s.dissip_mean.sel(zh=jz, method="nearest"))**(2./3)), 
                   label=f"$z/h=${jz:2.1f}")
        # Eww
        ax[1].plot(E.freq_x * E.z.sel(zh=jz, method="nearest"),
                   E.ww.sel(zh=jz, method="nearest") /\
                   (Cw*abs(s.dissip_mean.sel(zh=jz, method="nearest"))**(2./3)))
        # Ett
        ax[2].plot(E.freq_x * E.z.sel(zh=jz, method="nearest"), 
                   E.tt.sel(zh=jz, method="nearest")/\
                   (abs(s.dissip_mean.sel(zh=jz, method="nearest"))**(2./3)))

    ax[0].legend(loc=0, fontsize=12)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$k_x z$")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("$E_{uu} C_u^{-1} \\epsilon ^{-2/3}$")
    ax[1].set_xlabel("$k_x z$")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("$E_{ww} C_w^{-1} \\epsilon ^{-2/3}$")
    ax[2].set_xlabel("$k_x z$")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("$E_{\\theta \\theta} \\epsilon ^{-2/3}$")
    fig.tight_layout()
    # save
    fsave = f"{figdir}{E.stability}_1D_spectra.png"
    fig.savefig(fsave, dpi=300)
    plt.close(fig)

    return

# --------------------------------
# Define function to calculate amplitude modulation from timeseries
# --------------------------------
def amp_mod(dnc):
    """
    Calculate amplitude modulation coefficients from LES timeseries netcdf file
    Input dnc: string path directory for location of netcdf files
    Output netcdf file in dnc
    """
    # TODO: write load_timeseries() function in LESnc
    # load timeseries file
    d = xr.open_dataset(dnc+"timeseries_all.nc")
    # pre-processing ------------------------------------------------
    # calculate means
    for v in ["u", "v", "w", "theta", "txz", "tyz", "q3"]:
        d[f"{v}_mean"] = d[v].mean("t") # average in time
    # rotate coords so <v> = 0
    angle = np.arctan2(d.v_mean, d.u_mean)
    d["u_mean_rot"] = d.u_mean*np.cos(angle) + d.v_mean*np.sin(angle)
    d["v_mean_rot"] =-d.u_mean*np.sin(angle) + d.v_mean*np.cos(angle)
    # rotate instantaneous u and v
    d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
    d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
    # calculate "inst" covars
    d["uw"] = (d.u - d.u_mean) * (d.w - d.w_mean) + d.txz
    d["vw"] = (d.v - d.v_mean) * (d.w - d.w_mean) + d.tyz
    d["tw"] = (d.theta - d.theta_mean) * (d.w - d.w_mean) + d.q3
    # calculate "inst" vars
    d["uu"] = (d.u - d.u_mean) * (d.u - d.u_mean)
    d["vv"] = (d.v - d.v_mean) * (d.v - d.v_mean)
    d["ww"] = (d.w - d.w_mean) * (d.w - d.w_mean)
    d["tt"] = (d.theta - d.theta_mean) * (d.theta - d.theta_mean)
    # filtering ------------------------------------------------
    print("Begin calculating amplitude modulation coefficients")
    # lengthscale of filter to separate large and small scales
    # for now will hardcode value for sim A
    delta = 60. # m
    # cutoff frequency from Taylor hypothesis - use same for all heights
    f_c = 1./(delta/d.u_mean_rot)
    # calculate FFT of u, v, w, theta, uw, tw
    f_u = xrft.fft(d.u_rot, dim="t", true_phase=True, true_amplitude=True)
    f_v = xrft.fft(d.v_rot, dim="t", true_phase=True, true_amplitude=True)
    f_w = xrft.fft(d.w, dim="t", true_phase=True, true_amplitude=True)
    f_t = xrft.fft(d.theta, dim="t", true_phase=True, true_amplitude=True)
    f_uw = xrft.fft(d.uw, dim="t", true_phase=True, true_amplitude=True)
    f_tw = xrft.fft(d.tw, dim="t", true_phase=True, true_amplitude=True)
    # loop over heights and lowpass filter
    for jz in range(d.nz):
        # can do this all in one line
        # set high frequencies equal to zero -- sharp spectral filter
        # only keep -f_c < freq_t < f_c
        f_u[:,jz] = f_u[:,jz].where((f_u.freq_t < f_c.isel(z=jz)) &\
                                    (f_u.freq_t > -f_c.isel(z=jz)), other=0.)
        f_v[:,jz] = f_v[:,jz].where((f_v.freq_t < f_c.isel(z=jz)) &\
                                    (f_v.freq_t > -f_c.isel(z=jz)), other=0.)
        f_w[:,jz] = f_w[:,jz].where((f_w.freq_t < f_c.isel(z=jz)) &\
                                    (f_w.freq_t > -f_c.isel(z=jz)), other=0.)                                    
        f_t[:,jz] = f_t[:,jz].where((f_t.freq_t < f_c.isel(z=jz)) &\
                                    (f_t.freq_t > -f_c.isel(z=jz)), other=0.)
        f_uw[:,jz] = f_uw[:,jz].where((f_uw.freq_t < f_c.isel(z=jz)) &\
                                      (f_uw.freq_t > -f_c.isel(z=jz)), other=0.)   
        f_tw[:,jz] = f_tw[:,jz].where((f_tw.freq_t < f_c.isel(z=jz)) &\
                                      (f_tw.freq_t > -f_c.isel(z=jz)), other=0.)                                                                          

    # Inverse FFT to get large-scale component of signal
    u_l = xrft.ifft(f_u, dim="freq_t", true_phase=True, true_amplitude=True,
                    lag=f_u.freq_t.direct_lag).real
    v_l = xrft.ifft(f_v, dim="freq_t", true_phase=True, true_amplitude=True,
                    lag=f_v.freq_t.direct_lag).real
    w_l = xrft.ifft(f_w, dim="freq_t", true_phase=True, true_amplitude=True,
                    lag=f_w.freq_t.direct_lag).real
    t_l = xrft.ifft(f_t, dim="freq_t", true_phase=True, true_amplitude=True,
                    lag=f_t.freq_t.direct_lag).real
    uw_l = xrft.ifft(f_uw, dim="freq_t", true_phase=True, true_amplitude=True,
                     lag=f_uw.freq_t.direct_lag).real
    tw_l = xrft.ifft(f_tw, dim="freq_t", true_phase=True, true_amplitude=True,
                     lag=f_tw.freq_t.direct_lag).real
    # reset time coordinate
    u_l["t"] = d.t
    v_l["t"] = d.t
    w_l["t"] = d.t
    t_l["t"] = d.t
    uw_l["t"] = d.t
    tw_l["t"] = d.t

    # calculate small-scale component by subtracting large-scale from full
    u_s = d.u_rot - u_l
    v_s = d.v_rot - v_l
    w_s = d.w - w_l
    t_s = d.theta - t_l
    uw_s = d.uw - uw_l
    tw_s = d.tw - tw_l

    # envelope of small-scale signal from Hilbert transform
    E_u = xr.DataArray(data=np.abs(hilbert(u_s, axis=0)),
                       coords=dict(t=u_s.t,
                                   z=u_s.z)    )
    E_v = xr.DataArray(data=np.abs(hilbert(v_s, axis=0)),
                       coords=dict(t=v_s.t,
                                   z=v_s.z)    )
    E_w = xr.DataArray(data=np.abs(hilbert(w_s, axis=0)),
                       coords=dict(t=w_s.t,
                                   z=w_s.z)    )                                   
    E_t = xr.DataArray(data=np.abs(hilbert(t_s, axis=0)),
                       coords=dict(t=t_s.t,
                                   z=t_s.z)    )
    E_uw = xr.DataArray(data=np.abs(hilbert(uw_s, axis=0)),
                        coords=dict(t=uw_s.t,
                                    z=uw_s.z)    )                                   
    E_tw = xr.DataArray(data=np.abs(hilbert(tw_s, axis=0)),
                        coords=dict(t=tw_s.t,
                                    z=tw_s.z)    ) 
    # lowpass filter the small-scale envelope
    # fft the envelopes
    f_Eu = xrft.fft(E_u, dim="t", true_phase=True, true_amplitude=True)
    f_Ev = xrft.fft(E_v, dim="t", true_phase=True, true_amplitude=True)
    f_Ew = xrft.fft(E_w, dim="t", true_phase=True, true_amplitude=True)
    f_Et = xrft.fft(E_t, dim="t", true_phase=True, true_amplitude=True)
    f_Euw = xrft.fft(E_uw, dim="t", true_phase=True, true_amplitude=True)
    f_Etw = xrft.fft(E_tw, dim="t", true_phase=True, true_amplitude=True)
    # loop over heights and lowpass filter - copied from above
    for jz in range(d.nz):
        # can do this all in one line
        # set high frequencies equal to zero -- sharp spectral filter
        # only keep -f_c < freq_t < f_c
        f_Eu[:,jz] = f_Eu[:,jz].where((f_Eu.freq_t < f_c.isel(z=jz)) &\
                                      (f_Eu.freq_t > -f_c.isel(z=jz)), other=0.)
        f_Ev[:,jz] = f_Ev[:,jz].where((f_Ev.freq_t < f_c.isel(z=jz)) &\
                                      (f_Ev.freq_t > -f_c.isel(z=jz)), other=0.)    
        f_Ew[:,jz] = f_Ew[:,jz].where((f_Ew.freq_t < f_c.isel(z=jz)) &\
                                      (f_Ew.freq_t > -f_c.isel(z=jz)), other=0.)                                                                        
        f_Et[:,jz] = f_Et[:,jz].where((f_Et.freq_t < f_c.isel(z=jz)) &\
                                      (f_Et.freq_t > -f_c.isel(z=jz)), other=0.)
        f_Euw[:,jz] = f_Euw[:,jz].where((f_Euw.freq_t < f_c.isel(z=jz)) &\
                                        (f_Euw.freq_t > -f_c.isel(z=jz)), other=0.) 
        f_Etw[:,jz] = f_Etw[:,jz].where((f_Etw.freq_t < f_c.isel(z=jz)) &\
                                        (f_Etw.freq_t > -f_c.isel(z=jz)), other=0.)                                                                              
    # inverse fft the filtered envelopes
    E_u_f = xrft.ifft(f_Eu, dim="freq_t", true_phase=True, true_amplitude=True,
                      lag=f_Eu.freq_t.direct_lag).real
    E_v_f = xrft.ifft(f_Ev, dim="freq_t", true_phase=True, true_amplitude=True,
                      lag=f_Ev.freq_t.direct_lag).real   
    E_w_f = xrft.ifft(f_Ew, dim="freq_t", true_phase=True, true_amplitude=True,
                      lag=f_Ew.freq_t.direct_lag).real                                         
    E_t_f = xrft.ifft(f_Et, dim="freq_t", true_phase=True, true_amplitude=True,
                      lag=f_Et.freq_t.direct_lag).real
    E_uw_f = xrft.ifft(f_Euw, dim="freq_t", true_phase=True, true_amplitude=True,
                       lag=f_Euw.freq_t.direct_lag).real          
    E_tw_f = xrft.ifft(f_Etw, dim="freq_t", true_phase=True, true_amplitude=True,
                       lag=f_Etw.freq_t.direct_lag).real                                             
    # reset time coordinate
    E_u_f["t"] = d.t
    E_v_f["t"] = d.t
    E_w_f["t"] = d.t
    E_t_f["t"] = d.t
    E_uw_f["t"] = d.t
    E_tw_f["t"] = d.t

    # AM coefficients ------------------------------------------------
    # new dataset to hold corr coeff
    R = xr.Dataset(data_vars=None,
                   coords=dict(z=d.z),
                   attrs=d.attrs
                  )
    # add delta as attr
    R.attrs["cutoff"] = delta
    # correlation between large scale u and filtered envelope of small-scale variable
    R["ul_Eu"] = xr.corr(u_l, E_u_f, dim="t")
    R["ul_Ew"] = xr.corr(u_l, E_w_f, dim="t")
    R["ul_Et"] = xr.corr(u_l, E_t_f, dim="t")
    R["ul_Euw"] = xr.corr(u_l, E_uw_f, dim="t")
    R["ul_Etw"] = xr.corr(u_l, E_tw_f, dim="t")
    # correlation between large scale w and filtered envelope of small-scale variable
    R["wl_Eu"] = xr.corr(w_l, E_u_f, dim="t")
    R["wl_Ew"] = xr.corr(w_l, E_w_f, dim="t")
    R["wl_Et"] = xr.corr(w_l, E_t_f, dim="t")
    R["wl_Euw"] = xr.corr(w_l, E_uw_f, dim="t")
    R["wl_Etw"] = xr.corr(w_l, E_tw_f, dim="t")    
    # correlation between large scale u and filtered envelope of small-scale variable
    R["tl_Eu"] = xr.corr(t_l, E_u_f, dim="t")
    R["tl_Ew"] = xr.corr(t_l, E_w_f, dim="t")
    R["tl_Et"] = xr.corr(t_l, E_t_f, dim="t")
    R["tl_Euw"] = xr.corr(t_l, E_uw_f, dim="t")
    R["tl_Etw"] = xr.corr(t_l, E_tw_f, dim="t")

    # save file
    fsavenc = f"{dnc}AM_coefficients.nc"
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        R.to_netcdf(fsavenc, mode="w")

    return

def plot_AM(dnclist, figdir):
    """
    Input list of directories for plotting to loop over
    Output figures
    """
    # initialize figure before looping
    fig1, ax1 = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True,
                            figsize=(12, 16), constrained_layout=True)
    # define color palette
    nsim = len(dnclist)
    colors = seaborn.color_palette("cubehelix", nsim)
    for isim, dnc in enumerate(dnclist):
        # load stats file
        stat = load_stats(dnc+"average_statistics.nc")    
        # load AM coeff file
        R = xr.open_dataset(dnc+"AM_coefficients.nc")
        # add z/h as coordinate and swap with z
        # define array of z/h
        zh = R.z / stat.he
        # assign coupled with z
        R = R.assign_coords(zh=("z",zh.values))
        # swap
        R = R.swap_dims({"z": "zh"})
        # define new array of z/h logspace for bin averaging
        zhbin = np.logspace(-2, 0, 21)
        # from this, also need len(zhbin)-1 with midpoints of bins for plotting
        zhnew = [] # define empty array
        for iz in range(20):
            zhnew.append(gmean([zhbin[iz], zhbin[iz+1]]))
        zhnew = np.array(zhnew)
        # group by zh bins and calculate mean in one line
        Rbin = R.groupby_bins("zh", zhbin).mean("zh", skipna=True)
        # create new coordinate "zh_bins", then swap and drop
        Rbin = Rbin.assign_coords({"zh": ("zh_bins", zhnew)}).swap_dims({"zh_bins": "zh"})
        # interpolate empty values for better plotting
        Rbin = Rbin.interpolate_na(dim="zh")
        # calculate h/L parameter for plotting
        hL = (stat.he/stat.L).values
        # Plot ------------------------------------------------
        print(f"Figure 1 - Sim {isim}")
        # figure 1 - fifteen panels - modulation by u_l and w_l and t_l
        # col 1 - modulation by u_l
        # col 2 - modulation by w_l
        # col 3 - modulation by t_l
        # row 1 - modulation of small-scale u by large-scale u&w&t
        # row 2 - modulation of small-scale w by large-scale u&w&t
        # row 3 - modulation of small-scale theta by large-scale u&w&t
        # row 4 - modulation of small-scale uw by large-scale u&w&t
        # row 5 - modulation of small-scale tw by large-scale u&w&t
        # (a) R_ul_Eu
        ax1[0,0].plot(Rbin.zh, Rbin.ul_Eu, ls="-", c=colors[isim], lw=2,
                      label=f"$h/L=${hL:3.1f}")
        # (b) R_wl_Eu
        ax1[0,1].plot(Rbin.zh, Rbin.wl_Eu, ls="-", c=colors[isim], lw=2)
        # (c) R_tl_Eu
        ax1[0,2].plot(Rbin.zh, Rbin.tl_Eu, ls="-", c=colors[isim], lw=2)
        # (d) R_ul_Ew
        ax1[1,0].plot(Rbin.zh, Rbin.ul_Ew, ls="-", c=colors[isim], lw=2)
        # (e) R_wl_Ew
        ax1[1,1].plot(Rbin.zh, Rbin.wl_Ew, ls="-", c=colors[isim], lw=2)
        # (f) R_tl_Ew
        ax1[1,2].plot(Rbin.zh, Rbin.tl_Ew, ls="-", c=colors[isim], lw=2)
        # (g) R_ul_Et
        ax1[2,0].plot(Rbin.zh, Rbin.ul_Et, ls="-", c=colors[isim], lw=2)
        # (h) R_wl_Et
        ax1[2,1].plot(Rbin.zh, Rbin.wl_Et, ls="-", c=colors[isim], lw=2)
        # (i) R_tl_Et
        ax1[2,2].plot(Rbin.zh, Rbin.tl_Et, ls="-", c=colors[isim], lw=2)
        # (j) R_ul_Euw
        ax1[3,0].plot(Rbin.zh, Rbin.ul_Euw, ls="-", c=colors[isim], lw=2)
        # (k) R_wl_Euw
        ax1[3,1].plot(Rbin.zh, Rbin.wl_Euw, ls="-", c=colors[isim], lw=2)
        # (l) R_tl_Euw
        ax1[3,2].plot(Rbin.zh, Rbin.tl_Euw, ls="-", c=colors[isim], lw=2)
        # (m) R_ul_Etw
        ax1[4,0].plot(Rbin.zh, Rbin.ul_Etw, ls="-", c=colors[isim], lw=2)
        # (n) R_wl_Etw
        ax1[4,1].plot(Rbin.zh, Rbin.wl_Etw, ls="-", c=colors[isim], lw=2)
        # (o) R_wl_Etw
        ax1[4,2].plot(Rbin.zh, Rbin.tl_Etw, ls="-", c=colors[isim], lw=2)

    # OUTSIDE LOOP
    # clean up fig 1
    ax1[4,0].set_xlabel("$z/h$")
    ax1[4,0].set_xlim([1e-2, 1e0])
    ax1[0,0].set_ylim([-0.5, 0.5])
    ax1[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax1[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1[0,0].set_xscale("log")
    ax1[4,1].set_xlabel("$z/h$")
    ax1[4,2].set_xlabel("$z/h$")
    ax1[0,0].legend(loc="lower left", labelspacing=0.10, 
                    handletextpad=0.4, handlelength=0.75,
                    fontsize=14)
    # plot ref lines
    for iax in ax1.flatten():
        iax.axhline(0, c="k", ls="-", alpha=0.5)
        iax.axvline(R.cutoff/stat.he, c="k", ls="-", alpha=0.5)
    # y-axis labels
    for iax in ax1[:,0]:
        iax.set_ylabel("$R$")
    # text labels
    ax1[0,0].text(0.03, 0.90, "$R_{u_l,u_s}$", fontsize=16, transform=ax1[0,0].transAxes)
    ax1[0,1].text(0.03, 0.90, "$R_{w_l,u_s}$", fontsize=16, transform=ax1[0,1].transAxes)
    ax1[0,2].text(0.03, 0.90, "$R_{\\theta_l,u_s}$", fontsize=16, transform=ax1[0,2].transAxes)
    ax1[1,0].text(0.03, 0.90, "$R_{u_l,w_s}$", fontsize=16, transform=ax1[1,0].transAxes)
    ax1[1,1].text(0.03, 0.90, "$R_{w_l,w_s}$", fontsize=16, transform=ax1[1,1].transAxes)
    ax1[1,2].text(0.03, 0.90, "$R_{\\theta_l,w_s}$", fontsize=16, transform=ax1[1,2].transAxes)
    ax1[2,0].text(0.03, 0.90, "$R_{u_l,\\theta_s}$", fontsize=16, transform=ax1[2,0].transAxes)
    ax1[2,1].text(0.03, 0.90, "$R_{w_l,\\theta_s}$", fontsize=16, transform=ax1[2,1].transAxes)
    ax1[2,2].text(0.03, 0.90, "$R_{\\theta_l,\\theta_s}$", fontsize=16, transform=ax1[2,2].transAxes)
    ax1[3,0].text(0.03, 0.90, "$R_{u_l,(uw)_s}$", fontsize=16, transform=ax1[3,0].transAxes)
    ax1[3,1].text(0.03, 0.90, "$R_{w_l,(uw)_s}$", fontsize=16, transform=ax1[3,1].transAxes)
    ax1[3,2].text(0.03, 0.90, "$R_{\\theta_l,(uw)_s}$", fontsize=16, transform=ax1[3,2].transAxes)
    ax1[4,0].text(0.03, 0.90, "$R_{u_l,(\\theta w)_s}$", fontsize=16, transform=ax1[4,0].transAxes)
    ax1[4,1].text(0.03, 0.90, "$R_{w_l,(\\theta w)_s}$", fontsize=16, transform=ax1[4,1].transAxes)
    ax1[4,2].text(0.03, 0.90, "$R_{\\theta_l,(\\theta w)_s}$", fontsize=16, transform=ax1[4,2].transAxes)

    # save
    fsave1 = f"{figdir}all_amp_mod.png"
    print(f"Saving figure {fsave1}")
    fig1.savefig(fsave1, dpi=300)
    plt.close(fig1)

    """
    # figure 2 - timeseries with envelope
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14.8, 5))
    # (a) us, ul, El(us)
    ax2[0].plot(u_s.t*stat.ustar0/stat.he, u_s[:,1]/stat.ustar0, "-k", alpha=0.5, label="$u_s$")
    ax2[0].plot(u_l.t*stat.ustar0/stat.he, u_l[:,1]/stat.ustar0, "--b", label="$u_l$")
    ax2[0].plot(E_u.t*stat.ustar0/stat.he, E_u[:,1]/stat.ustar0, "-r", label="$E_l (u_s)$")
    # (b) ul, El'(us)
    ax2[1].plot(u_l.t*stat.ustar0/stat.he, u_l[:,1]/stat.ustar0, "--b", label="$u_l$")
    ax2[1].plot(E_u_f.t*stat.ustar0/stat.he, E_u_f[:,1]/stat.ustar0, "-r", label="$E_l' (u_s)$")
    # clean up
    ax2[0].set_ylabel("$u/u_*$")
    ax2[1].set_ylabel("$E_l'/u_*$")
    ax2[1].set_xlabel("$t u_* / h$")
    fig2.tight_layout()
     # save
    fsave2 = f"{figdir}{R.stability}_timeseries.png"
    print(f"Saving figure {fsave2}")
    fig2.savefig(fsave2, dpi=300)
    plt.close(fig2)
    """ 
    return

# --------------------------------
# Define function to perform quadrant analysis of resolved fluxes
# --------------------------------
def calc_quadrant(dnc):
    """
    Calculate quadrant components of u'w' and theta'w'
    and save single netcdf file for plotting later
    Input dnc: string path directory for location of netcdf files
    Output netcdf file in dnc
    """
    # load data and stats files for dimensions
    dd, s = load_full(dnc, 1080000, 1260000, 1000, 0.02, True, True)

    # get instantaneous u, w, theta perturbations
    u = dd.u_rot - s.u_mean_rot
    w = dd.w - s.w_mean
    theta = dd.theta - s.theta_mean

    # calculate four quadrants
    quad = xr.Dataset(data_vars=None,
                      coords=dict(z=s.z),
                      attrs=s.attrs)
    # 1) u'w'
    # u'>0, w'>0
    uw_pp = u.where(u > 0.) * w.where(w > 0.)
    # u'>0, w'<0
    uw_pn = u.where(u > 0.) * w.where(w < 0.)
    # u'<0, w'>0
    uw_np = u.where(u < 0.) * w.where(w > 0.)
    # u'<0, w'<0
    uw_nn = u.where(u < 0.) * w.where(w < 0.)
    # calculate averages and store in dataset
    quad["uw_pp"] = uw_pp.mean(dim=("time","x","y"))
    quad["uw_pn"] = uw_pn.mean(dim=("time","x","y"))
    quad["uw_np"] = uw_np.mean(dim=("time","x","y"))
    quad["uw_nn"] = uw_nn.mean(dim=("time","x","y"))

    # 2) theta'w'
    # theta'>0, w'>0
    tw_pp = theta.where(theta > 0.) * w.where(w > 0.)
    # theta'>0, w'<0
    tw_pn = theta.where(theta > 0.) * w.where(w < 0.)
    # theta'<0, w'>0
    tw_np = theta.where(theta < 0.) * w.where(w > 0.)
    # theta'<0, w'<0
    tw_nn = theta.where(theta < 0.) * w.where(w < 0.)
    # calculate averages and store in dataset
    quad["tw_pp"] = tw_pp.mean(dim=("time","x","y"))
    quad["tw_pn"] = tw_pn.mean(dim=("time","x","y"))
    quad["tw_np"] = tw_np.mean(dim=("time","x","y"))
    quad["tw_nn"] = tw_nn.mean(dim=("time","x","y"))

    # save quad Dataset as netcdf file for plotting later
    fsave = f"{dnc}uw_tw_quadrant.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        quad.to_netcdf(fsave, mode="w")
    
    #
    # Additionally save out u, w, theta at various heights in 1-d arrays
    # to plot in joint distribution 2d histograms
    #
    # first add z/h dimension for u, w, theta
    zh = s.z/s.he
    u = u.assign_coords(zh=("z",zh.values)).swap_dims({"z": "zh"})
    w = w.assign_coords(zh=("z",zh.values)).swap_dims({"z": "zh"})
    theta = theta.assign_coords(zh=("z",zh.values)).swap_dims({"z": "zh"})
    # grab data at z/h = 0.10, 0.50, 0.90
    # reshape by stacking along x, y, time
    ulong = u.sel(zh=(0.10,0.50,0.90), method="nearest").stack(index=("x","y","time")).reset_index("index", drop=True)
    wlong = w.sel(zh=(0.10,0.50,0.90), method="nearest").stack(index=("x","y","time")).reset_index("index", drop=True)
    thetalong = theta.sel(zh=(0.10,0.50,0.90), method="nearest").stack(index=("x","y","time")).reset_index("index", drop=True)
    # combine into dataset to save
    quad2d = xr.Dataset(data_vars=None,
                        coords=dict(zh=ulong.zh,
                                    index=ulong.index),
                        attrs=s.attrs)
    quad2d["u"] = ulong
    quad2d["w"] = wlong
    quad2d["theta"] = thetalong
    # save
    fsave = f"{dnc}u_w_theta_2d_quadrant.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        quad2d.to_netcdf(fsave, mode="w")
    print("Finished!")

    return

def plot_quadrant(dnc, figdir):
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # load quadrant file
    q = xr.load_dataset(dnc+"uw_tw_quadrant.nc")
    # plot
    # 1) 2-panel: uw and tw
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True, 
                             figsize=(14.8, 5), constrained_layout=True)
    # a: u'w'
    # u'>0, w'>0
    ax1[0].plot(q.uw_pp, q.z/s.he, c="b", ls="-", lw=2, label="$u_{+}w_{+}$")
    # u'>0, w'<0
    ax1[0].plot(q.uw_pn, q.z/s.he, c="b", ls="--", lw=2, label="$u_{+}w_{-}$")
    # u'<0, w'>0
    ax1[0].plot(q.uw_np, q.z/s.he, c="b", ls="-.", lw=2, label="$u_{-}w_{+}$")
    # u'<0, w'<0
    ax1[0].plot(q.uw_nn, q.z/s.he, c="b", ls=":", lw=2, label="$u_{-}w_{-}$")
    # u'w' total
    ax1[0].plot(s.uw_cov_res, s.z/s.he, c="k", ls="-", lw=2, label="$\\langle u'w' \\rangle$")
    # b: theta'w'
    # u'>0, w'>0
    ax1[1].plot(q.tw_pp, q.z/s.he, c="b", ls="-", lw=2, label="$\\theta_{+}w_{+}$")
    # theta'>0, w'<0
    ax1[1].plot(q.tw_pn, q.z/s.he, c="b", ls="--", lw=2, label="$\\theta_{+}w_{-}$")
    # theta'<0, w'>0
    ax1[1].plot(q.tw_np, q.z/s.he, c="b", ls="-.", lw=2, label="$\\theta_{-}w_{+}$")
    # theta'<0, w'<0
    ax1[1].plot(q.tw_nn, q.z/s.he, c="b", ls=":", lw=2, label="$\\theta_{-}w_{-}$")
    # theta'w' total
    ax1[1].plot(s.tw_cov_res, s.z/s.he, c="k", ls="-", lw=2, label="$\\langle \\theta'w' \\rangle$")

    # clean up
    ax1[0].legend(loc=0)
    ax1[1].legend(loc=0)
    ax1[0].set_ylim([0, 1.2])
    ax1[0].set_ylabel("$z/h$")
    ax1[0].set_xlabel("$u'w'$ [m$^2$ s$^{-2}$]")
    ax1[1].set_xlabel("$\\theta'w'$ [K m s$^{-1}$]")

    # save and close
    fsave1 = f"{figdir}{s.stability}_uw_tw_quad.png"
    print(f"Saving figure {fsave1}")
    fig1.savefig(fsave1, dpi=300)
    plt.close(fig1)

    #
    # joint distributions
    #
    # load second dataset
    q2 = xr.load_dataset(dnc+"u_w_theta_2d_quadrant.nc")
    # calculate 2d histogram bins and edges
    #
    # u'w'
    #
    uw_bins = (np.arange(-3., 3.1, 0.1), np.arange(-3., 3.1, 0.1))
    H_uw0, x_uw, y_uw = np.histogram2d(q2.u[0].values, 
                                       q2.w[0].values, 
                                       bins=uw_bins, density=True)
    H_uw1, x_uw, y_uw = np.histogram2d(q2.u[1].values, 
                                       q2.w[1].values, 
                                       bins=uw_bins, density=True)
    H_uw2, x_uw, y_uw = np.histogram2d(q2.u[2].values, 
                                       q2.w[2].values, 
                                       bins=uw_bins, density=True)                                 
    # calculate bin centers
    x2 = x_uw[:-1] + np.diff(x_uw)/2
    y2 = y_uw[:-1] + np.diff(y_uw)/2
    # normalize sum==1 by multiplying by bin area
    # also multiply by 100% for units
    uw_bin_area = np.diff(uw_bins[0])[0] * np.diff(uw_bins[1])[0]
    H_uw0 *= uw_bin_area * 100.
    H_uw1 *= uw_bin_area * 100.
    H_uw2 *= uw_bin_area * 100.
    # calculate cumulative percentages in each quadrant
    ix, iy = np.where(x2 > 0.)[0][0], np.where(y2 > 0.)[0][0]
    # level 0
    tot_upwp_0 = H_uw0[ix:, iy:].sum() # u'>0, w'>0
    tot_unwp_0 = H_uw0[:ix, iy:].sum() # u'<0, w'>0
    tot_unwn_0 = H_uw0[:ix, :iy].sum() # u'<0, w'<0
    tot_upwn_0 = H_uw0[ix:, :iy].sum() # u'>0, w'<0
    # level 1
    tot_upwp_1 = H_uw1[ix:, iy:].sum() # u'>0, w'>0
    tot_unwp_1 = H_uw1[:ix, iy:].sum() # u'<0, w'>0
    tot_unwn_1 = H_uw1[:ix, :iy].sum() # u'<0, w'<0
    tot_upwn_1 = H_uw1[ix:, :iy].sum() # u'>0, w'<0

    # plot uw quadrants
    fig2, ax2 = plt.subplots(1, figsize=(7.4, 5))
    # levels
    cmax2 = np.around(np.max([H_uw0.max(), H_uw1.max()]), 1) + 0.2
    levels2 = np.arange(0., cmax2, 0.2)
    cax2 = ax2.contour(x2, y2, H_uw0.T,
                       cmap=plt.get_cmap("Greys"),
                       levels=levels2, extend="max")
    cax2_1 = ax2.contour(x2, y2, H_uw1.T,
                         cmap=plt.get_cmap("Blues"),
                         levels=levels2, extend="max")
    # cax2_2 = ax2.contour(x2, y2, H_uw2.T,
    #                      cmap=plt.get_cmap("Reds"),
    #                      levels=np.linspace(0.0, 3.5, 15), extend="max")
    # legend labels and lines
    lines = [cax2.collections[-1], cax2_1.collections[-1]]#, cax2_2.collections[-1]]
    labels = [f"$z/h=${q2.zh[0].values:3.1f}", f"$z/h=${q2.zh[1].values:3.1f}"]#,
            #   f"$z/h=${q2.zh[2].values:3.1f}"]
    ax2.legend(lines, labels, fontsize=14)
    # clean up
    cb2 = fig2.colorbar(cax2, ax=ax2, location="right")
    cb2.ax.set_ylabel("Frequency [\\%]")
    ax2.set_xlabel("$u'$ [m s$^{-1}$]")
    ax2.set_ylabel("$w'$ [m s$^{-1}$]")
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.axhline(0., c="k", alpha=0.7)
    ax2.axvline(0., c="k", alpha=0.7)
    ax2.set_title(f"{'-'.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}")
    # add labels with percentages
    # get line colors
    # level 0
    cuw0 = cax2.collections[-1].get_color().squeeze()
    ax2.text(0.85,0.62,f"{tot_upwp_0:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw0)
    ax2.text(0.03,0.62,f"{tot_unwp_0:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw0)
    ax2.text(0.03,0.42,f"{tot_unwn_0:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw0)
    ax2.text(0.85,0.42,f"{tot_upwn_0:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw0)
    # level 1
    cuw1 = cax2_1.collections[-1].get_color().squeeze()
    ax2.text(0.85,0.55,f"{tot_upwp_1:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw1)
    ax2.text(0.03,0.55,f"{tot_unwp_1:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw1)
    ax2.text(0.03,0.36,f"{tot_unwn_1:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw1)
    ax2.text(0.85,0.36,f"{tot_upwn_1:3.1f}\\%",fontsize=14, 
             transform=ax2.transAxes, color=cuw1)
    
    fig2.tight_layout()
    fsave = f"{figdir}{s.stability}_uw_hist2d.png"
    print(f"Saving figure {fsave}")
    fig2.savefig(fsave, dpi=300)
    plt.close(fig2)

    #
    # theta'w'
    #
    tw_bins = (np.arange(-0.4, 0.41, 0.02), np.arange(-3., 3.1, 0.1))
    H_tw0, x_tw, y_tw = np.histogram2d(q2.theta[0].values, 
                                       q2.w[0].values, 
                                       bins=tw_bins, density=True)
    H_tw1, x_tw, y_tw = np.histogram2d(q2.theta[1].values, 
                                       q2.w[1].values, 
                                       bins=tw_bins, density=True)
    H_tw2, x_tw, y_tw = np.histogram2d(q2.theta[2].values, 
                                       q2.w[2].values, 
                                       bins=tw_bins, density=True)                                       
    # calculate bin centers
    x3 = x_tw[:-1] + np.diff(x_tw)/2
    y3 = y_tw[:-1] + np.diff(y_tw)/2
    # normalize sum==1 by multiplying by bin area
    # also multiply by 100% for units
    tw_bin_area = np.diff(tw_bins[0])[0] * np.diff(tw_bins[1])[0]
    H_tw0 *= tw_bin_area * 100.
    H_tw1 *= tw_bin_area * 100.
    H_tw2 *= tw_bin_area * 100.
    # calculate cumulative percentages in each quadrant
    ix, iy = np.where(x3 > 0.)[0][0], np.where(y3 > 0.)[0][0]
    # level 0
    tot_tpwp_0 = H_tw0[ix:, iy:].sum() # t'>0, w'>0
    tot_tnwp_0 = H_tw0[:ix, iy:].sum() # t'<0, w'>0
    tot_tnwn_0 = H_tw0[:ix, :iy].sum() # t'<0, w'<0
    tot_tpwn_0 = H_tw0[ix:, :iy].sum() # t'>0, w'<0
    # level 1
    tot_tpwp_1 = H_tw1[ix:, iy:].sum() # t'>0, w'>0
    tot_tnwp_1 = H_tw1[:ix, iy:].sum() # t'<0, w'>0
    tot_tnwn_1 = H_tw1[:ix, :iy].sum() # t'<0, w'<0
    tot_tpwn_1 = H_tw1[ix:, :iy].sum() # t'>0, w'<0

    # plot tw quadrants
    fig3, ax3 = plt.subplots(1, figsize=(7.4, 5))
    # contour levels
    cmax3 = np.around(np.max([H_tw0.max(), H_tw1.max()]), 1) + 0.2
    levels3 = np.arange(0., cmax3, 0.2)
    cax3 = ax3.contour(x3, y3, H_tw0.T,
                       cmap=plt.get_cmap("Greys"),
                       levels=levels3, extend="max")
    cax3_1 = ax3.contour(x3, y3, H_tw1.T,
                         cmap=plt.get_cmap("Blues"),
                         levels=levels3, extend="max")
    # cax3_2 = ax3.contour(x3, y3, H_tw2.T,
    #                      cmap=plt.get_cmap("Reds"),
    #                      levels=np.linspace(0.0, 3., 13), extend="both")
    # legend labels and lines
    lines = [cax3.collections[-1], cax3_1.collections[-1]]#, cax3_2.collections[-1]]
    labels = [f"$z/h=${q2.zh[0].values:3.1f}", f"$z/h=${q2.zh[1].values:3.1f}"]#,
            #   f"$z/h=${q2.zh[2].values:3.1f}"
    ax3.legend(lines, labels, fontsize=14)
    # clean up
    cb3 = fig3.colorbar(cax3, ax=ax3, location="right")
    cb3.ax.set_ylabel("Frequency [\\%]")
    ax3.set_xlabel("$\\theta'$ [K]")
    ax3.set_ylabel("$w'$ [m s$^{-1}$]")
    ax3.set_xlim([-0.2, 0.2])
    ax3.set_ylim([-1, 1])
    ax3.axhline(0., c="k", alpha=0.7)
    ax3.axvline(0., c="k", alpha=0.7)
    ax3.set_title(f"{'-'.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}")
    # add labels with percentages
    # get line colors
    # level 0
    ctw0 = cax3.collections[-1].get_color().squeeze()
    ax3.text(0.85,0.62,f"{tot_tpwp_0:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw0)
    ax3.text(0.03,0.62,f"{tot_tnwp_0:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw0)
    ax3.text(0.03,0.42,f"{tot_tnwn_0:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw0)
    ax3.text(0.85,0.42,f"{tot_tpwn_0:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw0)
    # level 1
    ctw1 = cax3_1.collections[-1].get_color().squeeze()
    ax3.text(0.85,0.55,f"{tot_tpwp_1:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw1)
    ax3.text(0.03,0.55,f"{tot_tnwp_1:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw1)
    ax3.text(0.03,0.36,f"{tot_tnwn_1:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw1)
    ax3.text(0.85,0.36,f"{tot_tpwn_1:3.1f}\\%",fontsize=14, 
             transform=ax3.transAxes, color=ctw1)
    
    fig3.tight_layout()
    fsave = f"{figdir}{s.stability}_tw_hist2d.png"
    print(f"Saving figure {fsave}")
    fig3.savefig(fsave, dpi=300)
    plt.close(fig3)

    #
    # theta'u'
    #
    tu_bins = (np.arange(-0.4, 0.41, 0.02), np.arange(-3., 3.1, 0.1))
    H_tu0, x_tu, y_tu = np.histogram2d(q2.theta[0].values, 
                                       q2.u[0].values, 
                                       bins=tu_bins, density=True)
    H_tu1, x_tu, y_tu = np.histogram2d(q2.theta[1].values, 
                                       q2.u[1].values, 
                                       bins=tu_bins, density=True)
    H_tu2, x_tu, y_tu = np.histogram2d(q2.theta[2].values, 
                                       q2.u[2].values, 
                                       bins=tu_bins, density=True)                                       
    # calculate bin centers
    x4 = x_tu[:-1] + np.diff(x_tu)/2
    y4 = y_tu[:-1] + np.diff(y_tu)/2
    # normalize sum==1 by multiplying by bin area
    # also multiply by 100% for units
    tu_bin_area = np.diff(tu_bins[0])[0] * np.diff(tu_bins[1])[0]
    H_tu0 *= tu_bin_area * 100.
    H_tu1 *= tu_bin_area * 100.
    H_tu2 *= tu_bin_area * 100.
    # calculate cumulative percentages in each quadrant
    ix, iy = np.where(x4 > 0.)[0][0], np.where(y4 > 0.)[0][0]
    # level 0
    tot_tpup_0 = H_tu0[ix:, iy:].sum() # t'>0, u'>0
    tot_tnup_0 = H_tu0[:ix, iy:].sum() # t'<0, u'>0
    tot_tnun_0 = H_tu0[:ix, :iy].sum() # t'<0, u'<0
    tot_tpun_0 = H_tu0[ix:, :iy].sum() # t'>0, u'<0
    # level 1
    tot_tpup_1 = H_tu1[ix:, iy:].sum() # t'>0, u'>0
    tot_tnup_1 = H_tu1[:ix, iy:].sum() # t'<0, u'>0
    tot_tnun_1 = H_tu1[:ix, :iy].sum() # t'<0, u'<0
    tot_tpun_1 = H_tu1[ix:, :iy].sum() # t'>0, u'<0

    # plot tu quadrants
    fig4, ax4 = plt.subplots(1, figsize=(7.4, 5))
    # get contour levels dynamically
    cmax4 = np.around(np.max([H_tu0.max(), H_tu1.max()]), 1) + 0.2
    levels4 = np.arange(0., cmax4, 0.2)
    cax4 = ax4.contour(x4, y4, H_tu0.T,
                       cmap=plt.get_cmap("Greys"),
                       levels=levels4, extend="max")
    cax4_1 = ax4.contour(x4, y4, H_tu1.T,
                         cmap=plt.get_cmap("Blues"),
                         levels=levels4, extend="max")
    # cax4_2 = ax4.contour(x4, y4, H_tu2.T,
    #                      cmap=plt.get_cmap("Reds"),
    #                      levels=np.linspace(0.0, 10., 11), extend="max")
    # legend labels and lines
    lines = [cax4.collections[-1], cax4_1.collections[-1]]#, cax4_2.collections[-1]]
    labels = [f"$z/h=${q2.zh[0].values:3.1f}", f"$z/h=${q2.zh[1].values:3.1f}"]#,
            #   f"$z/h=${q2.zh[2].values:3.1f}"]
    ax4.legend(lines, labels, fontsize=14)
    # clean up
    cb4 = fig4.colorbar(cax4, ax=ax4, location="right")
    cb4.ax.set_ylabel("Frequency [\\%]")
    ax4.set_xlabel("$\\theta'$ [K]")
    ax4.set_ylabel("$u'$ [m s$^{-1}$]")
    ax4.set_xlim([-0.2, 0.2])
    ax4.set_ylim([-1, 1])
    ax4.axhline(0., c="k", alpha=0.7)
    ax4.axvline(0., c="k", alpha=0.7)
    ax4.set_title(f"{'-'.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}")
    # add labels with percentages
    # get line colors
    # level 0
    ctu0 = cax4.collections[-1].get_color().squeeze()
    ax4.text(0.85,0.62,f"{tot_tpup_0:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu0)
    ax4.text(0.03,0.62,f"{tot_tnup_0:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu0)
    ax4.text(0.03,0.42,f"{tot_tnun_0:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu0)
    ax4.text(0.85,0.42,f"{tot_tpun_0:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu0)
    # level 1
    ctu1 = cax4_1.collections[-1].get_color().squeeze()
    ax4.text(0.85,0.55,f"{tot_tpup_1:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu1)
    ax4.text(0.03,0.55,f"{tot_tnup_1:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu1)
    ax4.text(0.03,0.36,f"{tot_tnun_1:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu1)
    ax4.text(0.85,0.36,f"{tot_tpun_1:3.1f}\\%",fontsize=14, 
             transform=ax4.transAxes, color=ctu1)    
    
    fig4.tight_layout()
    fsave = f"{figdir}{s.stability}_tu_hist2d.png"
    print(f"Saving figure {fsave}")
    fig4.savefig(fsave, dpi=300)
    plt.close(fig4)

    return
# --------------------------------
# Define function to calculate 2d correlations in x and z
# --------------------------------
def calc_corr2d(dnc, nmax=96):
    """
    Calculate 2-point correlations in x and z by running
    executable fortran routine and then converting binary file to netcdf
    Input dnc: directory for netcdf files
    Input nmax: maximum number of lags to calculate, default 96
    Output netcdf file
    """
    # load stats file for dimensions
    s = load_stats(dnc+"average_statistics.nc", SBL=True)
    # grab output directory 1 up from dnc
    dout = os.path.split(os.path.abspath(dnc))[0]+os.sep
    # define path to fortran executable
    fexe = "/home/bgreene/SBL_LES/fortran/executables/corr2d.exe"
    # run fortran executable with input arguments
    os.system(f"{fexe} {dout} {nmax}")
    # convert output binary file into netcdf file
    # load data from fortran program (binary file)
    # R_uu.out
    fopen = dnc+"R_uu.out"
    print(f"Reading file: {fopen}")
    f=open(fopen,"rb")
    datuu = np.fromfile(f,dtype="float64",count=(2*nmax+1)*(nmax+1))
    datuu = np.reshape(datuu,(2*nmax+1,nmax+1),order="F")
    f.close()
    # R_tt.out
    fopen = dnc+"R_tt.out"
    print(f"Reading file: {fopen}")
    f=open(fopen,"rb")
    dattt = np.fromfile(f,dtype="float64",count=(2*nmax+1)*(nmax+1))
    dattt = np.reshape(dattt,(2*nmax+1,nmax+1),order="F")
    f.close()
    # R_uwuw.out
    fopen = dnc+"R_uwuw.out"
    print(f"Reading file: {fopen}")
    f=open(fopen,"rb")
    datuwuw = np.fromfile(f,dtype="float64",count=(2*nmax+1)*(nmax+1))
    datuwuw = np.reshape(datuwuw,(2*nmax+1,nmax+1),order="F")
    f.close()    
    # R_twtw.out
    fopen = dnc+"R_twtw.out"
    print(f"Reading file: {fopen}")
    f=open(fopen,"rb")
    dattwtw = np.fromfile(f,dtype="float64",count=(2*nmax+1)*(nmax+1))
    dattwtw = np.reshape(dattwtw,(2*nmax+1,nmax+1),order="F")
    f.close()    
    # determine x and z lag coordinates from nmax and dx, dz
    xall = np.linspace(-nmax, nmax, nmax*2+1) * s.dx
    zall = np.linspace(0., nmax, nmax+1) * s.dz
    # construct xarray DataArrays to store in dataset
    # R_uu
    R_uu = xr.DataArray(data=datuu,
                        dims=["x", "z"],
                        coords=dict(x=xall,
                                    z=zall))
    R_uu.x.attrs["units"] = "m" # units
    R_uu.z.attrs["units"] = "m" # units
    # R_tt
    R_tt = xr.DataArray(data=dattt,
                        dims=["x", "z"],
                        coords=dict(x=xall,
                                    z=zall))
    R_tt.x.attrs["units"] = "m" # units
    R_tt.z.attrs["units"] = "m" # units
    # R_uwuw
    R_uwuw = xr.DataArray(data=datuwuw,
                        dims=["x", "z"],
                        coords=dict(x=xall,
                                    z=zall))
    R_uwuw.x.attrs["units"] = "m" # units
    R_uwuw.z.attrs["units"] = "m" # units
    # R_twtw
    R_twtw = xr.DataArray(data=dattwtw,
                        dims=["x", "z"],
                        coords=dict(x=xall,
                                    z=zall))
    R_twtw.x.attrs["units"] = "m" # units
    R_twtw.z.attrs["units"] = "m" # units
    # now store in Dataset
    R_save = xr.Dataset(data_vars=None,
                        coords=dict(x=R_uu.x,
                                    z=R_uu.z),
                        attrs=s.attrs)
    # add nmax as attribute
    R_save.attrs["nmax"] = nmax
    # assign data
    R_save["R_uu"] = R_uu
    R_save["R_tt"] = R_tt
    R_save["R_uwuw"] = R_uwuw
    R_save["R_twtw"] = R_twtw
    # save as netcdf
    fsavenc = f"{dnc}R2d_xz.nc"
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        R_save.to_netcdf(fsavenc, mode="w")
    print("Finished!")

    return

def plot_corr2d(dnc, figdir):
    # load correlation file created from calc_corr2d()
    R = xr.load_dataset(dnc+"R2d_xz.nc")

    # load data and stats files for dimensions
    dd, s = load_full(dnc, 1080000, 1260000, 1000, 0.02, True, True)

    #
    # R_uu
    #
    # calculate inclination angle for z/h <= 0.15
    imax = R.R_uu.argmax(axis=0)
    xmax = R.x[imax].where(R.z/s.he <= 0.1, drop=True)
    fit = xmax.polyfit(dim="z", deg=1)
    gamma = np.arctan2(1/fit.polyfit_coefficients[0], 1) * 180./np.pi
    print(f"Inclination angle: {gamma.values:4.2f} deg")
    # determine line of fit for plotting
    zplot = np.linspace(0, xmax.size, xmax.size+1) * s.dz
    fplot = np.poly1d(fit.polyfit_coefficients)
    xplot = fplot(zplot)

    # plot quicklook
    fig, ax = plt.subplots(1, figsize=(7.4,5))
    cfax = ax.contour(R.x/s.he, R.z/s.he, R.R_uu.T,
                      cmap=seaborn.color_palette("cubehelix_r", as_cmap=True),
                      levels=np.linspace(0.1, 0.9, 17))
    # plot location of max delta_x at each delta_z
    ax.plot(R.x[imax]/s.he, R.z/s.he, "ok")
    # plot best fit line
    ax.plot(xplot/s.he.values, zplot/s.he.values, "-k")
    cb = fig.colorbar(cfax, ax=ax, location="right")
    cb.ax.set_ylabel("$R_{uu}$")
    ax.set_xlabel("$\Delta x /h$")
    ax.set_ylabel("$\Delta z /h$")
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([0, 0.25])
    ax.set_title(f"{''.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}, $\\gamma = ${gamma.values:4.2f} deg")
    fig.tight_layout()
    fsave = f"{figdir}{s.stability}_R_uu.png"
    fig.savefig(fsave, dpi=300)
    plt.close(fig)

    #
    # R_tt
    #
    # calculate inclination angle for z/h <= 0.15
    imax = R.R_tt.argmax(axis=0)
    xmax = R.x[imax].where(R.z/s.he <= 0.1, drop=True)
    fit = xmax.polyfit(dim="z", deg=1)
    gamma = np.arctan2(1/fit.polyfit_coefficients[0], 1) * 180./np.pi
    print(f"Inclination angle theta: {gamma.values:4.2f} deg")
    # determine line of fit for plotting
    zplot = np.linspace(0, xmax.size, xmax.size+1) * s.dz
    fplot = np.poly1d(fit.polyfit_coefficients)
    xplot = fplot(zplot)

    # plot quicklook
    fig, ax = plt.subplots(1, figsize=(7.4,5))
    cfax = ax.contour(R.x/s.he, R.z/s.he, R.R_tt.T,
                      cmap=seaborn.color_palette("cubehelix_r", as_cmap=True),
                      levels=np.linspace(0.1, 0.9, 17))
    # plot location of max delta_x at each delta_z
    ax.plot(R.x[imax]/s.he, R.z/s.he, "ok")
    # plot best fit line
    ax.plot(xplot/s.he.values, zplot/s.he.values, "-k")
    cb = fig.colorbar(cfax, ax=ax, location="right")
    cb.ax.set_ylabel("$R_{\\theta \\theta}$")
    ax.set_xlabel("$\Delta x /h$")
    ax.set_ylabel("$\Delta z /h$")
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([0, 0.25])
    ax.set_title(f"{''.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}, $\\gamma = ${gamma.values:4.2f} deg")
    fig.tight_layout()
    fsave = f"{figdir}{s.stability}_R_tt.png"
    fig.savefig(fsave, dpi=300)
    plt.close(fig)

    #
    # R_uwuw
    #
    # calculate inclination angle for z/h <= 0.15
    imax = R.R_uwuw.argmax(axis=0)
    xmax = R.x[imax].where(R.z/s.he <= 0.1, drop=True)
    fit = xmax.polyfit(dim="z", deg=1)
    gamma = np.arctan2(1/fit.polyfit_coefficients[0], 1) * 180./np.pi
    print(f"Inclination angle u'w': {gamma.values:4.2f} deg")
    # determine line of fit for plotting
    zplot = np.linspace(0, xmax.size, xmax.size+1) * s.dz
    fplot = np.poly1d(fit.polyfit_coefficients)
    xplot = fplot(zplot)

    # plot quicklook
    fig, ax = plt.subplots(1, figsize=(7.4,5))
    cfax = ax.contour(R.x/s.he, R.z/s.he, R.R_uwuw.T,
                      cmap=seaborn.color_palette("cubehelix_r", as_cmap=True),
                      levels=np.linspace(0.1, 0.9, 17))
    # plot location of max delta_x at each delta_z
    ax.plot(R.x[imax]/s.he, R.z/s.he, "ok")
    # plot best fit line
    ax.plot(xplot/s.he.values, zplot/s.he.values, "-k")
    cb = fig.colorbar(cfax, ax=ax, location="right")
    cb.ax.set_ylabel("$R_{uwuw}$")
    ax.set_xlabel("$\Delta x /h$")
    ax.set_ylabel("$\Delta z /h$")
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([0, 0.25])
    ax.set_title(f"{''.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}, $\\gamma = ${gamma.values:4.2f} deg")
    fig.tight_layout()
    fsave = f"{figdir}{s.stability}_R_uwuw.png"
    fig.savefig(fsave, dpi=300)
    plt.close(fig)

    #
    # R_twtw
    #
    # calculate inclination angle for z/h <= 0.15
    imax = R.R_twtw.argmax(axis=0)
    xmax = R.x[imax].where(R.z/s.he <= 0.1, drop=True)
    fit = xmax.polyfit(dim="z", deg=1)
    gamma = np.arctan2(1/fit.polyfit_coefficients[0], 1) * 180./np.pi
    print(f"Inclination angle theta'w': {gamma.values:4.2f} deg")
    # determine line of fit for plotting
    zplot = np.linspace(0, xmax.size, xmax.size+1) * s.dz
    fplot = np.poly1d(fit.polyfit_coefficients)
    xplot = fplot(zplot)

    # plot quicklook
    fig, ax = plt.subplots(1, figsize=(7.4,5))
    cfax = ax.contour(R.x/s.he, R.z/s.he, R.R_twtw.T,
                      cmap=seaborn.color_palette("cubehelix_r", as_cmap=True),
                      levels=np.linspace(0.1, 0.9, 17))
    # plot location of max delta_x at each delta_z
    ax.plot(R.x[imax]/s.he, R.z/s.he, "ok")
    # plot best fit line
    ax.plot(xplot/s.he.values, zplot/s.he.values, "-k")
    cb = fig.colorbar(cfax, ax=ax, location="right")
    cb.ax.set_ylabel("$R_{\\theta w \\theta w}$")
    ax.set_xlabel("$\Delta x /h$")
    ax.set_ylabel("$\Delta z /h$")
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([0, 0.25])
    ax.set_title(f"{''.join(s.stability.split('_'))} $h/L = ${(s.he/s.L).values:4.3f}, $\\gamma = ${gamma.values:4.2f} deg")
    fig.tight_layout()
    fsave = f"{figdir}{s.stability}_R_twtw.png"
    fig.savefig(fsave, dpi=300)
    plt.close(fig)

    return

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # configure plotting parameters
    rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
    rc('text',usetex='True')
    # figure save directories
    figdir = "/home/bgreene/SBL_LES/figures/spectrogram/"
    figdir_AM = "/home/bgreene/SBL_LES/figures/amp_mod/"
    figdir_quad = "/home/bgreene/SBL_LES/figures/quadrant/"
    figdir_corr2d = "/home/bgreene/SBL_LES/figures/corr2d/"
    ncdirlist = []
    # loop sims A--F
    for sim in ["0.10"]:
        print(f"---Begin Sim {sim}---")
        ncdir = f"/home/bgreene/simulations/cr{sim}_u08_192/output/netcdf/"
        ncdirlist.append(ncdir)
        # calc_spectra(ncdir)
        # plot_spectrogram(ncdir, figdir)
        # plot_1D_spectra(ncdir, figdir)
        # amp_mod(ncdir)
        # calc_quadrant(ncdir)
        plot_quadrant(ncdir, figdir_quad)
        # calc_corr2d(ncdir)
        # plot_corr2d(ncdir, figdir_corr2d)
        print(f"---End Sim {sim}---")
    # plot_AM(ncdirlist, figdir_AM)