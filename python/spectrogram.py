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
import numpy as np
import xarray as xr
from datetime import datetime
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
    E_uu = xrft.power_spectrum(dd.u_rot, dim="x", true_phase=True, 
                               true_amplitude=True, detrend="linear")
    # average in time and y
    E_uu_ytmean = E_uu.mean(dim=("time","y"))
    # w
    E_ww = xrft.power_spectrum(dd.w, dim="x", true_phase=True, 
                               true_amplitude=True, detrend="linear")
    # average in time and y
    E_ww_ytmean = E_ww.mean(dim=("time","y"))
    # theta
    E_tt = xrft.power_spectrum(dd.theta, dim="x", true_phase=True, 
                               true_amplitude=True, detrend="linear")
    # average in time and y
    E_tt_ytmean = E_tt.mean(dim=("time","y"))
    # u'w'
    E_uw = xrft.cross_spectrum(dd.u_rot, dd.w, dim="x", scaling="density",
                               true_phase=True, true_amplitude=True, detrend="linear")
    # average in time and y, only take real component
    E_uw_ytmean = np.real(E_uw.mean(dim=("time","y")))
    # theta'w'
    E_tw = xrft.cross_spectrum(dd.theta, dd.w, dim="x", scaling="density",
                               true_phase=True, true_amplitude=True, detrend="linear")
    # average in time and y, only take real component
    E_tw_ytmean = np.real(E_tw.mean(dim=("time","y")))
    # theta'u'
    E_tu = xrft.cross_spectrum(dd.theta, dd.u_rot, dim="x", scaling="density",
                               true_phase=True, true_amplitude=True, detrend="linear")
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
    # delete old file for saving new one
    if os.path.exists(fsavenc):
        os.system(f"rm {fsavenc}")
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
    f_u = xrft.fft(d.udr, dim="t", true_phase=True, true_amplitude=True)
    f_v = xrft.fft(d.vdr, dim="t", true_phase=True, true_amplitude=True)
    f_w = xrft.fft(d.wd, dim="t", true_phase=True, true_amplitude=True)
    f_t = xrft.fft(d.td, dim="t", true_phase=True, true_amplitude=True)
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
    # USE DETRENDED FULL TIMESERIES OR HILBERT TRANSFORMS WILL BE NONPHYSICAL
    u_s = d.udr - u_l
    v_s = d.vdr - v_l
    w_s = d.wd - w_l
    t_s = d.td - t_l
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
    f_Eu = xrft.fft(E_u, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
    f_Ev = xrft.fft(E_v, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
    f_Ew = xrft.fft(E_w, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
    f_Et = xrft.fft(E_t, dim="t", true_phase=True, true_amplitude=True, detrend="linear")
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
    # delete old file for saving new one
    if os.path.exists(fsavenc):
        os.system(f"rm {fsavenc}")
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        R.to_netcdf(fsavenc, mode="w")

    return

# --------------------------------
# Define function to perform quadrant analysis of resolved fluxes
# --------------------------------
def calc_quadrant(dnc, save_2d_hist=False):
    """
    Calculate quadrant components of u'w', theta'w', theta'u'
    and save single netcdf file for plotting later
    Input dnc: string path directory for location of netcdf files
    Output netcdf file in dnc
    """
    # load data and stats files for dimensions
    dd, s = load_full(dnc, 1080000, 1260000, 1000, 0.02, True, True)

    # get instantaneous u, w, theta perturbations
    u = dd.u - dd.u.mean(dim=("x","y"))
    v = dd.v - dd.v.mean(dim=("x","y"))
    w = dd.w - dd.w.mean(dim=("x","y"))
    theta = dd.theta - dd.theta.mean(dim=("x","y"))
    # get SGS
    txz = dd.txz
    tyz = dd.tyz
    q3 = dd.q3

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
    # calculate instantaneous u'w'+txz
    uw_inst = (u * w) + txz
    # compute mean instances of positive and negative
    quad["uw_p_mean"] = uw_inst.where(uw_inst > 0.).mean(dim=("time","x","y"))
    quad["uw_n_mean"] = uw_inst.where(uw_inst < 0.).mean(dim=("time","x","y"))

    # 2) v'w'
    # v'>0, w'>0
    vw_pp = v.where(v > 0.) * w.where(w > 0.)
    # v'>0, w'<0
    vw_pn = v.where(v > 0.) * w.where(w < 0.)
    # v'<0, w'>0
    vw_np = v.where(v < 0.) * w.where(w > 0.)
    # v'<0, w'<0
    vw_nn = v.where(v < 0.) * w.where(w < 0.)
    # calculate averages and store in dataset
    quad["vw_pp"] = vw_pp.mean(dim=("time","x","y"))
    quad["vw_pn"] = vw_pn.mean(dim=("time","x","y"))
    quad["vw_np"] = vw_np.mean(dim=("time","x","y"))
    quad["vw_nn"] = vw_nn.mean(dim=("time","x","y"))
    # calculate instantaneous v'w'+tyz
    vw_inst = (v * w) + tyz
    # compute mean instances of positive and negative
    quad["vw_p_mean"] = vw_inst.where(vw_inst > 0.).mean(dim=("time","x","y"))
    quad["vw_n_mean"] = vw_inst.where(vw_inst < 0.).mean(dim=("time","x","y"))

    # 3) theta'w'
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
    # calculate instantaneous t'w'+q3
    tw_inst = (theta * w) + q3
    # compute mean instances of positive and negative
    quad["tw_p_mean"] = tw_inst.where(tw_inst > 0.).mean(dim=("time","x","y"))
    quad["tw_n_mean"] = tw_inst.where(tw_inst < 0.).mean(dim=("time","x","y"))

    # calculate correlation coeffs and mixing efficiencies
    # # uw and tw covariances - resolved only
    # uw_cov_res = s.uw_cov_res
    # vw_cov_res = s.vw_cov_res
    # tw_cov_res = s.tw_cov_res
    # # mean SGS values
    # txz_mean = s.uw_cov_tot - s.uw_cov_res
    # tyz_mean = s.vw_cov_tot - s.vw_cov_res
    # q3_mean = s.tw_cov_tot - s.tw_cov_res
    # # u, w, theta variances
    # uvar = u.var(dim=("time","x","y"))
    # wvar = w.var(dim=("time","x","y"))
    # tvar = theta.var(dim=("time","x","y"))
    # # R_ab = <a'b'> / (sigma_a * sigma_b)
    # quad["Ruw"] = np.abs(uw_cov_res / np.sqrt(uvar) / np.sqrt(wvar))
    # quad["Rtw"] = np.abs(tw_cov_res / np.sqrt(tvar) / np.sqrt(wvar))
    # eta_uw = <u'w' + txz> / (u-w+ + u+w- + txz_n)
    quad["eta_uw"] = np.abs(s.uw_cov_tot) / np.abs(quad.uw_n_mean)
    # eta_vw = <v'w' + tyz> / (v-w+ + v+w- + tyz_n)
    quad["eta_vw"] = np.abs(s.vw_cov_tot) / np.abs(quad.vw_n_mean)    
    # eta_tw = <theta'w' + q3> / (t+w- + t-w+ + q3_n)
    quad["eta_tw"] = np.abs(s.tw_cov_tot) / np.abs(quad.tw_n_mean)
    # attempt to calculate combined eta_ustar2
    uw_downgrad = quad.uw_n_mean
    vw_downgrad = quad.vw_n_mean
    ustar2_downgrad = (uw_downgrad**2. + vw_downgrad**2.) ** 0.5
    quad["eta_ustar2"] = s.ustar2 / ustar2_downgrad

    # calculate sum of quadrants
    quad["uw_sum"] = np.abs(quad.uw_pp) + np.abs(quad.uw_pn) +\
                     np.abs(quad.uw_np) + np.abs(quad.uw_nn)
    quad["tw_sum"] = np.abs(quad.tw_pp) + np.abs(quad.tw_pn) +\
                     np.abs(quad.tw_np) + np.abs(quad.tw_nn)

    # save quad Dataset as netcdf file for plotting later
    fsave = f"{dnc}uw_tw_quadrant_SGS.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        quad.to_netcdf(fsave, mode="w")
    
    #
    # Additionally save out u, w, theta at various heights in 1-d arrays
    # to plot in joint distribution 2d histograms
    #
    if save_2d_hist:
        # first add z/h dimension for u, w, theta
        zh = s.z/s.h
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
        # delete old file for saving new one
        if os.path.exists(fsave):
            os.system(f"rm {fsave}")
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
    F_uu = xrft.fft(dd.u_rot, dim="x", true_phase=True, true_amplitude=True, detrend="linear")
    F_ww = xrft.fft(dd.w, dim="x", true_phase=True, true_amplitude=True, detrend="linear")
    F_tt = xrft.fft(dd.theta, dim="x", true_phase=True, true_amplitude=True, detrend="linear")
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
    # use zr = h/2 as reference
    zh = s.z / s.h
    izr2 = abs(zh - 0.50).argmin()
    # calculate G2
    # u
    G2u2 = np.absolute((F_uu * F_uu.isel(z=izr2).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_uu)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_uu.isel(z=izr2))**2.).mean(dim=("y","time")))
    # w
    G2w2 = np.absolute((F_ww * F_ww.isel(z=izr2).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_ww)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_ww.isel(z=izr2))**2.).mean(dim=("y","time")))
    # theta
    G2t2 = np.absolute((F_tt * F_tt.isel(z=izr2).conj()).mean(dim=("y","time"))) ** 2. /\
          ((np.absolute(F_tt)**2.).mean(dim=("y","time")) *\
           (np.absolute(F_tt.isel(z=izr2))**2.).mean(dim=("y","time")))

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
    Gsave["u2"] = G2u2
    Gsave["w2"] = G2w2
    Gsave["theta2"] = G2t2
    # add attr for reference heights
    Gsave.attrs["zr0"] = dd.z.isel(z=0).values
    Gsave.attrs["zr1"] = dd.z.isel(z=izr).values
    Gsave.attrs["zr2"] = dd.z.isel(z=izr2).values

    # calculate G2 for different variables at *same* height
    # uw
    G2uw = np.absolute((F_uu * F_ww.conj()).mean(dim=("y","time"))) ** 2. /\
           ((np.absolute(F_uu)**2.).mean(dim=("y","time")) *\
            (np.absolute(F_ww)**2.).mean(dim=("y","time")))    
    # tw
    G2tw = np.absolute((F_tt * F_ww.conj()).mean(dim=("y","time"))) ** 2. /\
           ((np.absolute(F_tt)**2.).mean(dim=("y","time")) *\
            (np.absolute(F_ww)**2.).mean(dim=("y","time")))    
    # tu
    G2tu = np.absolute((F_tt * F_uu.conj()).mean(dim=("y","time"))) ** 2. /\
           ((np.absolute(F_tt)**2.).mean(dim=("y","time")) *\
            (np.absolute(F_uu)**2.).mean(dim=("y","time")))    
    # store in Gsave
    Gsave["uw"] = G2uw
    Gsave["tw"] = G2tw
    Gsave["tu"] = G2tu
    # only keep freqs > 0
    Gsave = Gsave.where(Gsave.freq_x > 0., drop=True)
    # save file
    fsavenc = f"{dnc}G2.nc"
    # delete old file for saving new one
    if os.path.exists(fsavenc):
        os.system(f"rm {fsavenc}")
    print(f"Saving file: {fsavenc}")
    with ProgressBar():
        Gsave.to_netcdf(fsavenc, mode="w")
    print("Finished!")
    
    return

# --------------------------------
# Define functions for conditional averaging
# --------------------------------
def cond_avg(dnc):
    """
    Calculate averages of SBL fields (u, w, theta) conditioned on: 
    u'(x,y,z=jz) <-2*sigma_u
    u'(x,y,z=jz) > 2*sigma_u
    w'(x,y,z=jz) <-2*sigma_w
    w'(x,y,z=jz) > 2*sigma_w
    jz for: z/h = 0.05, z/h = 0.50, z/zj = 1.00
    Warning: lots of hardcoding :)
    Only load one file at a time
    Input dnc: directory for netcdf files
    Output netcdf file
    """
    # sim parameters
    t0 = 1080000
    t1 = 1260000
    dt = 1000
    # timesteps for loading files
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # load stats file
    s = load_stats(dnc+"average_statistics.nc")
    # grab some useful params
    dx = s.dx
    nx = s.nx
    ny = s.ny
    nz = s.nz
    #
    # Find jz indices to use for conditioning
    #
    # find closest z to z/h = 0.05: jz1
    zh = s.z / s.h
    jz1 = np.argmin(abs(zh.values - 0.05))
    print(f"z/h = 0.05 for jz={jz1}")
    # find closest z to z/h = 0.50: jz1
    jz2 = np.argmin(abs(zh.values - 0.50))
    print(f"z/h = 0.50 for jz={jz2}")
    # find closest z to z/zh = 1: jz2
    zzj = s.z / s.zj
    jz3 = np.argmin(abs(zzj.values - 1.0))
    print(f"z/zj = 1.0 for jz={jz3}")
    #
    # Determine hi and lo cutoffs based on u and w
    #
    # read middle volume file to determine alpha cutoffs
    dd = xr.load_dataset(fall[nf//2])
    # rotate velocities so <v>_xy = 0
    u_mean = dd.u.mean(dim=("x","y"))
    v_mean = dd.v.mean(dim=("x","y"))
    angle = np.arctan2(v_mean, u_mean)
    dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
    dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
    # calculate u'/u*
    dd["u_p"] = (dd.u_rot - dd.u_rot.mean(dim=("x","y"))) / s.ustar0
    # calculate mean and std of u'/u*
    mu_u = dd.u_p.mean(dim=("x","y"))
    std_u = dd.u_p.std(dim=("x","y"))
    # calculate alpha cutoffs from u
    # lo
    alpha_u_lo_1 = mu_u[jz1] - 2.0*std_u[jz1]
    alpha_u_lo_2 = mu_u[jz2] - 2.0*std_u[jz2]
    alpha_u_lo_3 = mu_u[jz3] - 2.0*std_u[jz3]
    # hi
    alpha_u_hi_1 = mu_u[jz1] + 2.0*std_u[jz1]
    alpha_u_hi_2 = mu_u[jz2] + 2.0*std_u[jz2]
    alpha_u_hi_3 = mu_u[jz3] + 2.0*std_u[jz3]
    #
    # do the same for w' cutoffs
    #
    dd["w_p"] = (dd.w - dd.w.mean(dim=("x","y"))) / s.ustar0
    mu_w = dd.w_p.mean(dim=("x","y"))
    std_w = dd.w_p.std(dim=("x","y"))
    # calculate alpha cutoffs from u
    # lo
    alpha_w_lo_1 = mu_w[jz1] - 2.0*std_w[jz1]
    alpha_w_lo_2 = mu_w[jz2] - 2.0*std_w[jz2]
    alpha_w_lo_3 = mu_w[jz3] - 2.0*std_w[jz3]
    # hi
    alpha_w_hi_1 = mu_w[jz1] + 2.0*std_w[jz1]
    alpha_w_hi_2 = mu_w[jz2] + 2.0*std_w[jz2]
    alpha_w_hi_3 = mu_w[jz3] + 2.0*std_w[jz3]
    #
    # Prepare arrays for conditional averaging
    #
    # max points to include in conditional average
    n_delta = int(s.h/dx)
    # number of points upstream and downstream to include in cond avg
    n_min = 3*n_delta
    n_max = 3*n_delta
    # initialize conditionally averaged arrays
    # condition on u lo and hi
    # < u'/u* | u'/u* < 2*alpha > and < u'/u* | u'/u* > 2*alpha > for each jz
    u_cond_u_lo_1, u_cond_u_lo_2, u_cond_u_lo_3,\
    u_cond_u_hi_1, u_cond_u_hi_2, u_cond_u_hi_3 =\
        [np.zeros((n_min+n_max, nz), dtype=np.float64) for _ in range(6)]
    # < w'/u* | u'/u* < 2*alpha > and < w'/u* | u'/u* > 2*alpha > for each jz
    w_cond_u_lo_1, w_cond_u_lo_2, w_cond_u_lo_3,\
    w_cond_u_hi_1, w_cond_u_hi_2, w_cond_u_hi_3 =\
        [np.zeros((n_min+n_max, nz), dtype=np.float64) for _ in range(6)]
    # < T'/T* | u'/u* < 2*alpha > and < T'/T* | u'/u* > 2*alpha > for each jz
    T_cond_u_lo_1, T_cond_u_lo_2, T_cond_u_lo_3,\
    T_cond_u_hi_1, T_cond_u_hi_2, T_cond_u_hi_3 =\
        [np.zeros((n_min+n_max, nz), dtype=np.float64) for _ in range(6)]
    # condition on w
    # < u'/u* | w'/u* < 2*alpha > and < u'/u* | w'/u* > 2*alpha > for each jz
    u_cond_w_lo_1, u_cond_w_lo_2, u_cond_w_lo_3,\
    u_cond_w_hi_1, u_cond_w_hi_2, u_cond_w_hi_3 =\
        [np.zeros((n_min+n_max, nz), dtype=np.float64) for _ in range(6)]
    # < w'/u* | w'/u* < 2*alpha > and < w'/u* | w'/u* > 2*alpha > for each jz
    w_cond_w_lo_1, w_cond_w_lo_2, w_cond_w_lo_3,\
    w_cond_w_hi_1, w_cond_w_hi_2, w_cond_w_hi_3 =\
        [np.zeros((n_min+n_max, nz), dtype=np.float64) for _ in range(6)]
    # < T'/T* | w'/u* < 2*alpha > and < T'/T* | w'/u* > 2*alpha > for each jz
    T_cond_w_lo_1, T_cond_w_lo_2, T_cond_w_lo_3,\
    T_cond_w_hi_1, T_cond_w_hi_2, T_cond_w_hi_3 =\
        [np.zeros((n_min+n_max, nz), dtype=np.float64) for _ in range(6)]
    # initialize counters for number of points satisfying each condition
    n_u_lo_1 = 0; n_u_hi_1 = 0
    n_u_lo_2 = 0; n_u_hi_2 = 0
    n_u_lo_3 = 0; n_u_hi_3 = 0
    n_w_lo_1 = 0; n_w_hi_1 = 0
    n_w_lo_2 = 0; n_w_hi_2 = 0
    n_w_lo_3 = 0; n_w_hi_3 = 0
    #
    # BEGIN LOOP OVER FILES
    #
    for jt, tfile in enumerate(fall):
        # print timestep
        print(datetime.utcnow())
        # load file
        print(f"Loading file: {tfile}")
        dd = xr.load_dataset(tfile)
        # rotate velocities so <v>_xy = 0
        u_mean = dd.u.mean(dim=("x","y"))
        v_mean = dd.v.mean(dim=("x","y"))
        angle = np.arctan2(v_mean, u_mean)
        dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
        dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
        # calculate u'/u*
        dd["u_p"] = (dd.u_rot - dd.u_rot.mean(dim=("x","y"))) / s.ustar0
        # calculate w'/u*
        dd["w_p"] = (dd.w - dd.w.mean(dim=("x","y"))) / s.ustar0
        # calculate theta'/theta*
        dd["t_p"] = (dd.theta - dd.theta.mean(dim=("x","y"))) / s.tstar0
        #Create big arrays so we don't have to deal with periodicity
        # u
        u_big = np.zeros((4*nx,ny,nz), dtype=np.float64)
        u_big[0:nx,:,:] = dd.u_p[:,:,:].to_numpy()
        u_big[nx:2*nx,:,:] = dd.u_p[:,:,:].to_numpy()
        u_big[2*nx:3*nx,:,:] = dd.u_p[:,:,:].to_numpy()
        u_big[3*nx:4*nx,:,:] = dd.u_p[:,:,:].to_numpy()
        # w
        w_big = np.zeros((4*nx,ny,nz), dtype=np.float64)
        w_big[0:nx,:,:] = dd.w_p[:,:,:].to_numpy()
        w_big[nx:2*nx,:,:] = dd.w_p[:,:,:].to_numpy()
        w_big[2*nx:3*nx,:,:] = dd.w_p[:,:,:].to_numpy()
        w_big[3*nx:4*nx,:,:] = dd.w_p[:,:,:].to_numpy()
        # theta
        theta_big = np.zeros((4*nx,ny,nz), dtype=np.float64)
        theta_big[0:nx,:,:] = dd.t_p[:,:,:].to_numpy()
        theta_big[nx:2*nx,:,:] = dd.t_p[:,:,:].to_numpy()
        theta_big[2*nx:3*nx,:,:] = dd.t_p[:,:,:].to_numpy()
        theta_big[3*nx:4*nx,:,:] = dd.t_p[:,:,:].to_numpy()
        #
        # Calculate conditional averages
        #
        # loop over y
        for jy in range(ny):
            # loop over x
            for jx in range(nx):
                # include points if meeting condition
                #
                # u
                #
                # condition on u lo
                # jz1
                if dd.u_p[jx,jy,jz1] < alpha_u_lo_1:
                    n_u_lo_1 += 1
                    u_cond_u_lo_1[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_u_lo_1[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_u_lo_1[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz2
                if dd.u_p[jx,jy,jz2] < alpha_u_lo_2:
                    n_u_lo_2 += 1
                    u_cond_u_lo_2[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_u_lo_2[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_u_lo_2[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz3
                if dd.u_p[jx,jy,jz3] < alpha_u_lo_3:
                    n_u_lo_3 += 1
                    u_cond_u_lo_3[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_u_lo_3[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_u_lo_3[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # condition on u hi
                # jz1
                if dd.u_p[jx,jy,jz1] > alpha_u_hi_1:
                    n_u_hi_1 += 1
                    u_cond_u_hi_1[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_u_hi_1[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_u_hi_1[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz2
                if dd.u_p[jx,jy,jz2] > alpha_u_hi_2:
                    n_u_hi_2 += 1
                    u_cond_u_hi_2[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_u_hi_2[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_u_hi_2[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz3
                if dd.u_p[jx,jy,jz3] > alpha_u_hi_3:
                    n_u_hi_3 += 1
                    u_cond_u_hi_3[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_u_hi_3[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_u_hi_3[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                #
                # w
                #                
                # condition on w lo
                # jz1
                if dd.w_p[jx,jy,jz1] < alpha_w_lo_1:
                    n_w_lo_1 += 1
                    u_cond_w_lo_1[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_w_lo_1[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_w_lo_1[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz2
                if dd.w_p[jx,jy,jz2] < alpha_w_lo_2:
                    n_w_lo_2 += 1
                    u_cond_w_lo_2[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_w_lo_2[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_w_lo_2[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz3
                if dd.w_p[jx,jy,jz3] < alpha_w_lo_3:
                    n_w_lo_3 += 1
                    u_cond_w_lo_3[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_w_lo_3[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_w_lo_3[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # condition on w hi
                # jz1
                if dd.w_p[jx,jy,jz1] > alpha_w_hi_1:
                    n_w_hi_1 += 1
                    u_cond_w_hi_1[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_w_hi_1[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_w_hi_1[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz2
                if dd.w_p[jx,jy,jz2] > alpha_w_hi_2:
                    n_w_hi_2 += 1
                    u_cond_w_hi_2[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_w_hi_2[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_w_hi_2[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                # jz3
                if dd.w_p[jx,jy,jz3] > alpha_w_hi_3:
                    n_w_hi_3 += 1
                    u_cond_w_hi_3[:,:] += u_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    w_cond_w_hi_3[:,:] += w_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
                    T_cond_w_hi_3[:,:] += theta_big[(jx+nx-n_min):(jx+nx+n_max),jy,:]
    # Finished looping
    print ("Finished processing all timesteps")
    # Normalize each by number of samples
    # u lo 1
    u_cond_u_lo_1[:,:] /= n_u_lo_1
    w_cond_u_lo_1[:,:] /= n_u_lo_1
    T_cond_u_lo_1[:,:] /= n_u_lo_1
    # u lo 2
    u_cond_u_lo_2[:,:] /= n_u_lo_2
    w_cond_u_lo_2[:,:] /= n_u_lo_2
    T_cond_u_lo_2[:,:] /= n_u_lo_2
    # u lo 3
    u_cond_u_lo_3[:,:] /= n_u_lo_3
    w_cond_u_lo_3[:,:] /= n_u_lo_3
    T_cond_u_lo_3[:,:] /= n_u_lo_3
    # w lo 1
    u_cond_w_lo_1[:,:] /= n_w_lo_1
    w_cond_w_lo_1[:,:] /= n_w_lo_1
    T_cond_w_lo_1[:,:] /= n_w_lo_1
    # w lo 2
    u_cond_w_lo_2[:,:] /= n_w_lo_2
    w_cond_w_lo_2[:,:] /= n_w_lo_2
    T_cond_w_lo_2[:,:] /= n_w_lo_2
    # w lo 3
    u_cond_w_lo_3[:,:] /= n_w_lo_3
    w_cond_w_lo_3[:,:] /= n_w_lo_3
    T_cond_w_lo_3[:,:] /= n_w_lo_3
    # u hi 1
    u_cond_u_hi_1[:,:] /= n_u_hi_1
    w_cond_u_hi_1[:,:] /= n_u_hi_1
    T_cond_u_hi_1[:,:] /= n_u_hi_1
    # u hi 2
    u_cond_u_hi_2[:,:] /= n_u_hi_2
    w_cond_u_hi_2[:,:] /= n_u_hi_2
    T_cond_u_hi_2[:,:] /= n_u_hi_2
    # u hi 3
    u_cond_u_hi_3[:,:] /= n_u_hi_3
    w_cond_u_hi_3[:,:] /= n_u_hi_3
    T_cond_u_hi_3[:,:] /= n_u_hi_3
    # w hi 1
    u_cond_w_hi_1[:,:] /= n_w_hi_1
    w_cond_w_hi_1[:,:] /= n_w_hi_1
    T_cond_w_hi_1[:,:] /= n_w_hi_1
    # w hi 2
    u_cond_w_hi_2[:,:] /= n_w_hi_2
    w_cond_w_hi_2[:,:] /= n_w_hi_2
    T_cond_w_hi_2[:,:] /= n_w_hi_2
    # w hi 3
    u_cond_w_hi_3[:,:] /= n_w_hi_3
    w_cond_w_hi_3[:,:] /= n_w_hi_3
    T_cond_w_hi_3[:,:] /= n_w_hi_3
    # store these variables into Dataset for saving
    # define array coordinates
    xnew = np.linspace(-1*n_min*dx, n_max*dx, (n_min+n_max))
    # initialize empty dataset
    dsave = xr.Dataset(data_vars=None, coords=dict(x=xnew,z=s.z), attrs=s.attrs)
    # store each variable as DataArray
    # conditioned on u
    # u lo 1
    dsave["u_cond_u_lo_1"] = xr.DataArray(data=u_cond_u_lo_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_u_lo_1"] = xr.DataArray(data=w_cond_u_lo_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_u_lo_1"] = xr.DataArray(data=T_cond_u_lo_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # u lo 2
    dsave["u_cond_u_lo_2"] = xr.DataArray(data=u_cond_u_lo_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_u_lo_2"] = xr.DataArray(data=w_cond_u_lo_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_u_lo_2"] = xr.DataArray(data=T_cond_u_lo_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # u lo 3
    dsave["u_cond_u_lo_3"] = xr.DataArray(data=u_cond_u_lo_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_u_lo_3"] = xr.DataArray(data=w_cond_u_lo_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_u_lo_3"] = xr.DataArray(data=T_cond_u_lo_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # u hi 1
    dsave["u_cond_u_hi_1"] = xr.DataArray(data=u_cond_u_hi_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_u_hi_1"] = xr.DataArray(data=w_cond_u_hi_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_u_hi_1"] = xr.DataArray(data=T_cond_u_hi_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # u hi 2
    dsave["u_cond_u_hi_2"] = xr.DataArray(data=u_cond_u_hi_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_u_hi_2"] = xr.DataArray(data=w_cond_u_hi_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_u_hi_2"] = xr.DataArray(data=T_cond_u_hi_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # u hi 3
    dsave["u_cond_u_hi_3"] = xr.DataArray(data=u_cond_u_hi_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_u_hi_3"] = xr.DataArray(data=w_cond_u_hi_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_u_hi_3"] = xr.DataArray(data=T_cond_u_hi_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # conditioned on w
    # w lo 1
    dsave["u_cond_w_lo_1"] = xr.DataArray(data=u_cond_w_lo_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_w_lo_1"] = xr.DataArray(data=w_cond_w_lo_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_w_lo_1"] = xr.DataArray(data=T_cond_w_lo_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # w lo 2
    dsave["u_cond_w_lo_2"] = xr.DataArray(data=u_cond_w_lo_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_w_lo_2"] = xr.DataArray(data=w_cond_w_lo_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_w_lo_2"] = xr.DataArray(data=T_cond_w_lo_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # w lo 3
    dsave["u_cond_w_lo_3"] = xr.DataArray(data=u_cond_w_lo_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_w_lo_3"] = xr.DataArray(data=w_cond_w_lo_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_w_lo_3"] = xr.DataArray(data=T_cond_w_lo_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # w hi 1
    dsave["u_cond_w_hi_1"] = xr.DataArray(data=u_cond_w_hi_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_w_hi_1"] = xr.DataArray(data=w_cond_w_hi_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_w_hi_1"] = xr.DataArray(data=T_cond_w_hi_1, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # w hi 2
    dsave["u_cond_w_hi_2"] = xr.DataArray(data=u_cond_w_hi_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_w_hi_2"] = xr.DataArray(data=w_cond_w_hi_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_w_hi_2"] = xr.DataArray(data=T_cond_w_hi_2, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # w hi 3
    dsave["u_cond_w_hi_3"] = xr.DataArray(data=u_cond_w_hi_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["w_cond_w_hi_3"] = xr.DataArray(data=w_cond_w_hi_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    dsave["T_cond_w_hi_3"] = xr.DataArray(data=T_cond_w_hi_3, dims=("x","z"),
                                          coords=dict(x=xnew,z=s.z))
    # include attrs for each variable: z, n, alpha
    # z values
    dsave.attrs["z1"] = "$z/h = 0.05$"
    dsave.attrs["z2"] = "$z/h = 0.50$"
    dsave.attrs["z3"] = "$z/z_j = 1.00$"
    # corresponding jz values
    dsave.attrs["jz1"] = jz1
    dsave.attrs["jz2"] = jz2
    dsave.attrs["jz3"] = jz3
    # number of events
    # u
    dsave.attrs["n_u_lo_1"] = n_u_lo_1
    dsave.attrs["n_u_lo_2"] = n_u_lo_2
    dsave.attrs["n_u_lo_3"] = n_u_lo_3
    dsave.attrs["n_u_hi_1"] = n_u_hi_1
    dsave.attrs["n_u_hi_2"] = n_u_hi_2
    dsave.attrs["n_u_hi_3"] = n_u_hi_3
    # w
    dsave.attrs["n_w_lo_1"] = n_w_lo_1
    dsave.attrs["n_w_lo_2"] = n_w_lo_2
    dsave.attrs["n_w_lo_3"] = n_w_lo_3
    dsave.attrs["n_w_hi_1"] = n_w_hi_1
    dsave.attrs["n_w_hi_2"] = n_w_hi_2
    dsave.attrs["n_w_hi_3"] = n_w_hi_3
    # alpha
    # u
    dsave.attrs["alpha_u_lo_1"] = alpha_u_lo_1.values
    dsave.attrs["alpha_u_lo_2"] = alpha_u_lo_2.values
    dsave.attrs["alpha_u_lo_3"] = alpha_u_lo_3.values
    dsave.attrs["alpha_u_hi_1"] = alpha_u_hi_1.values
    dsave.attrs["alpha_u_hi_2"] = alpha_u_hi_2.values
    dsave.attrs["alpha_u_hi_3"] = alpha_u_hi_3.values
    # w
    dsave.attrs["alpha_w_lo_1"] = alpha_w_lo_1.values
    dsave.attrs["alpha_w_lo_2"] = alpha_w_lo_2.values
    dsave.attrs["alpha_w_lo_3"] = alpha_w_lo_3.values
    dsave.attrs["alpha_w_hi_1"] = alpha_w_hi_1.values
    dsave.attrs["alpha_w_hi_2"] = alpha_w_hi_2.values
    dsave.attrs["alpha_w_hi_3"] = alpha_w_hi_3.values
    # save and return
    fsave = f"{dnc}cond_avg.nc"
    # delete old file for saving new one
    if os.path.exists(fsave):
        os.system(f"rm {fsave}")
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dsave.to_netcdf(fsave, mode="w")
    print("Finished processing conditional averaging!")

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
        calc_quadrant(ncdir, save_2d_hist=False)
        # calc_corr2d(ncdir)
        # plot_corr2d(ncdir, figdir_corr2d)
        # LCS(ncdir)
        # cond_avg(ncdir)
        print(f"---End Sim {sim}---")