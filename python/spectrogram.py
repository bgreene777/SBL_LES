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
from LESnc import load_stats, load_full, load_timeseries
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
    # load stats file to get zj
    s = load_stats(dnc+"average_statistics.nc")
    # load timeseries file
    d = load_timeseries(dnc, detrend=True)
    # filtering ------------------------------------------------
    print("Begin calculating amplitude modulation coefficients")
    # lengthscale of filter to separate large and small scales
    # choose value of delta=zj/2
    delta =  s.zj.values/2 # m
    # cutoff frequency from Taylor hypothesis - use same for all heights
    f_c = 1./(delta/d.u_mean_rot)
    # calculate FFT of u, v, w, theta, uw, tw
    f_u = xrft.fft(d.u_rot, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
    f_v = xrft.fft(d.v_rot, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
    f_w = xrft.fft(d.w, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
    f_t = xrft.fft(d.theta, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
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
        # NOTE: using z/zj now, kept variable names for ease
        zh = R.z / stat.zj
        # assign coupled with z
        R = R.assign_coords(zh=("z",zh.values))
        # swap
        R = R.swap_dims({"z": "zh"})
        # define new array of z/h logspace for bin averaging
        zhbin = np.logspace(-2, 0.2, 25)
        # from this, also need len(zhbin)-1 with midpoints of bins for plotting
        zhnew = [] # define empty array
        for iz in range(24):
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
                      label=f"$h/L={{{hL:3.1f}}}$")
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
    ax1[4,0].set_xlabel("$z/z_j$")
    ax1[4,0].set_xlim([1e-2, 1.5e0])
    ax1[0,0].set_ylim([-0.5, 0.5])
    ax1[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax1[0,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1[0,0].set_xscale("log")
    ax1[4,1].set_xlabel("$z/z_j$")
    ax1[4,2].set_xlabel("$z/z_j$")
    ax1[0,0].legend(loc="lower left", labelspacing=0.10, 
                    handletextpad=0.4, handlelength=0.75,
                    fontsize=14)
    # plot ref lines
    for iax in ax1.flatten():
        iax.axhline(0, c="k", ls="-", alpha=0.5)
        iax.axvline(0.5, c="k", ls="-", alpha=0.5)
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
# Define functions for calculating linear coherence spectra
# --------------------------------
def LCS(dnc):
    """
    Calculate linear coherence spectra in streamwise dimension
    against reference height of lowest grid point
    and average along spanwise and time dimensions
    Input dnc: directory for netcdf files
    Output netcdf file
    G2(z,zr;lx) = |<F_u(z,lx) F_u*(zr,lx)>|**2 / <|F_u(z,lx)|**2><|F_u(zr,lx)|**2>
    """
    # load data
    dd, s = load_full(dnc, 1080000, 1260000, 1000, 0.02, True, True) 
    # forward FFT   
    F_uu = xrft.fft(dd.u_rot, dim="x", true_phase=True, true_amplitude=True)
    F_ww = xrft.fft(dd.w, dim="x", true_phase=True, true_amplitude=True)
    F_tt = xrft.fft(dd.theta, dim="x", true_phase=True, true_amplitude=True)
    # first use z = z[0] as reference
    # u
    G2u0 = np.absolute((F_uu * F_uu.isel(z=0).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_uu)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_uu.isel(z=0))**2.).mean(dim=("y","time")))
    # w
    G2w0 = np.absolute((F_ww * F_ww.isel(z=0).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_ww)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_ww.isel(z=0))**2.).mean(dim=("y","time")))
    # theta
    G2t0 = np.absolute((F_tt * F_tt.isel(z=0).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_tt)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_tt.isel(z=0))**2.).mean(dim=("y","time")))
    # use zr = zj as reference
    izr = s.uh.argmax()
    # calculate G2
    # u
    G2u1 = np.absolute((F_uu * F_uu.isel(z=izr).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_uu)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_uu.isel(z=izr))**2.).mean(dim=("y","time")))
    # w
    G2w1 = np.absolute((F_ww * F_ww.isel(z=izr).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_ww)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_ww.isel(z=izr))**2.).mean(dim=("y","time")))
    # theta
    G2t1 = np.absolute((F_tt * F_tt.isel(z=izr).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_tt)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_tt.isel(z=izr))**2.).mean(dim=("y","time")))
    # combine Gs into dataset to save as netcdf
    Gsave = xr.Dataset(data_vars=None,
                       coords=dict(z=G2u0.z, freq_x=G2u0.freq_x),
                       attrs=s.attrs)
    Gsave["u0"] = G2u0
    Gsave["w0"] = G2w0
    Gsave["theta0"] = G2t0
    Gsave["u1"] = G2u1
    Gsave["w1"] = G2w1
    Gsave["theta1"] = G2t1
    # add attr for reference heights
    Gsave.attrs["zr0"] = dd.z.isel(z=0).values
    Gsave.attrs["zr1"] = dd.z.isel(z=izr).values

    # calculate G2 for different variables at *same* height
    # uw
    G2uw = np.absolute((F_uu * F_ww.conj()).mean(dim=("y","time"))) ** 2. /\
           ((np.absolute(F_uu)**2.).mean(dim=("y","time")) *\
            (np.absolute(F_ww)**2.).mean(dim=("y","time")))    
    # tw
    G2tw = np.absolute((F_tt * F_ww.conj()).mean(dim=("y","time"))) ** 2. /\
           ((np.absolute(F_tt)**2.).mean(dim=("y","time")) *\
            (np.absolute(F_ww)**2.).mean(dim=("y","time")))    
    # store in Gsave
    Gsave["uw"] = G2uw
    Gsave["tw"] = G2tw
    # only keep freqs > 0
    Gsave = Gsave.where(Gsave.freq_x > 0., drop=True)
    # save file
    fsavenc = f"{dnc}G2.nc"
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        Gsave.to_netcdf(fsavenc, mode="w")
    print("Finished!")
    
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
    for sim in ["cr0.10_u08_192","cr0.25_u08_192","cr0.33_u08_192","cr0.50_u08_192",
                "cr1.00_u08_192","cr1.50_u08_192","cr2.00_u08_192","cr2.50_u08_192"]:
        print(f"---Begin Sim {sim}---")
        ncdir = f"/home/bgreene/simulations/{sim}/output/netcdf/"
        ncdirlist.append(ncdir)
        # calc_spectra(ncdir)
        # plot_1D_spectra(ncdir, figdir)
        # amp_mod(ncdir)
        # calc_quadrant(ncdir)
        # calc_corr2d(ncdir)
        # plot_corr2d(ncdir, figdir_corr2d)
        LCS(ncdir)
        print(f"---End Sim {sim}---")
    # plot_AM(ncdirlist, figdir_AM)