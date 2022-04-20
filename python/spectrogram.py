# --------------------------------
# Name: spectrogram.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 15 March 2022
# Purpose: calculate spectra from SBL simulations as functions of
# wavelength and height above ground and optionally plot
# --------------------------------
import xrft
import seaborn
import cmocean
import numpy as np
import xarray as xr
from scipy.signal import hilbert
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from LESnc import load_stats, MidPointNormalize
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
    rc('font',weight='normal',size=20,family='serif',serif='Times New Roman')
    rc('text',usetex='True')
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
    zh = E.z / s.h
    # assign coupled with z
    E = E.assign_coords(zh=("z",zh.values))
    # swap
    E = E.swap_dims({"z": "zh"})

    # three panel plot
    # Euu, Eww, Ett
    # init figure
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(14.8, 5))
    # loop over heights z/h
    zhplot = np.arange(0.1, 0.9, 0.1)
    for jz in zhplot:
        # Euu
        ax[0].plot(E.freq_x * E.z.sel(zh=jz, method="nearest"), 
                   E.uu.sel(zh=jz, method="nearest") * E.freq_x /\
                   (s.ustar0 * s.ustar0 ), 
                   label=f"$z/h=${jz:2.1f}")
        # Eww
        ax[1].plot(E.freq_x * E.z.sel(zh=jz, method="nearest"),
                   E.ww.sel(zh=jz, method="nearest") * E.freq_x /\
                   (s.ustar0 * s.ustar0 ))
        # Ett
        ax[2].plot(E.freq_x * E.z.sel(zh=jz, method="nearest"), 
                   E.tt.sel(zh=jz, method="nearest") * E.freq_x /\
                   (s.tstar0 * s.tstar0 ))

    ax[0].legend(loc=0)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$k_x z$")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("$k_x E_{uu} u_{*}^{-2}$")
    ax[1].set_xlabel("$k_x z$")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("$k_x E_{ww} u_{*}^{-2}$")
    ax[2].set_xlabel("$k_x z$")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("$k_x E_{\\theta \\theta} \\theta_{*}^{-2}$")
    fig.tight_layout()
    # save
    fsave = f"{figdir}{E.stability}_1D_spectra.png"
    fig.savefig(fsave, dpi=300)
    plt.close(fig)

    return

# --------------------------------
# Define function to calculate amplitude modulation from timeseries
# --------------------------------
def amp_mod(dnc, figdir):
    """
    Calculate amplitude modulation coefficients from LES timeseries netcdf file
    Input dnc: string path directory for location of netcdf files
    Output netcdf file in dnc
    """
    # TODO: write load_timeseries() function in LESnc
    # load timeseries file
    d = xr.open_dataset(dnc+"timeseries_all.nc")
    # load stats file
    stat = load_stats(dnc+"average_statistics.nc")
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
    # calculate covars
    # u'w'
    d["uw_cov_res"] = xr.cov(d.u, d.w, dim=("t"))
    d["uw_cov_tot"] = d.uw_cov_res + d.txz_mean
    # v'w'
    d["vw_cov_res"] = xr.cov(d.v, d.w, dim=("t"))
    d["vw_cov_tot"] = d.vw_cov_res + d.tyz_mean
    # theta'w'
    d["tw_cov_res"] = xr.cov(d.theta, d.w, dim=("t"))
    d["tw_cov_tot"] = d.tw_cov_res + d.q3_mean
    # calculate vars
    for s in ["u", "v", "w", "theta"]:
        d[f"{s}_var"] = d[s].var("t")
    # recalculate u_var_rot, v_var_rot
    d["u_var_rot"] = d.u_rot.var("t")
    d["v_var_rot"] = d.v_rot.var("t")
    # calculate ustar, L, and z/L
    d["ustar"] = (d.uw_cov_tot**2. + d.vw_cov_tot**2.) ** 0.25
    d["L"] = (-1. * (d.ustar**3.) * d.theta_mean) / (0.4 * 9.81 * d.tw_cov_tot)
    d["zL"] = d.z/d.L

    # filtering ------------------------------------------------
    # lengthscale of filter to separate large and small scales
    # for now will hardcode value for sim A
    delta = 60. # m
    # cutoff frequency from Taylor hypothesis - use same for all heights
    f_c = 1./(delta/d.u_mean_rot)
    # calculate FFT of u, theta
    f_u = xrft.fft(d.u_rot, dim="t", true_phase=True, true_amplitude=True)
    f_t = xrft.fft(d.theta, dim="t", true_phase=True, true_amplitude=True)
    # loop over heights and lowpass filter
    for jz in range(d.nz):
        # can do this all in one line
        # set high frequencies equal to zero -- sharp spectral filter
        # only keep -f_c < freq_t < f_c
        f_u[:,jz] = f_u[:,jz].where((f_u.freq_t < f_c.isel(z=jz)) &\
                                    (f_u.freq_t > -f_c.isel(z=jz)), other=0.)
        f_t[:,jz] = f_t[:,jz].where((f_t.freq_t < f_c.isel(z=jz)) &\
                                    (f_t.freq_t > -f_c.isel(z=jz)), other=0.)

    # Inverse FFT to get large-scale component of signal
    u_l = xrft.ifft(f_u, dim="freq_t", true_phase=True, true_amplitude=True,
                    lag=f_u.freq_t.direct_lag).real
    t_l = xrft.ifft(f_t, dim="freq_t", true_phase=True, true_amplitude=True,
                    lag=f_t.freq_t.direct_lag).real
    # reset time coordinate
    u_l["t"] = d.t
    t_l["t"] = d.t

    # calculate small-scale component by subtracting large-scale from full
    u_s = d.u_rot - u_l
    t_s = d.theta - t_l

    # envelope of small-scale signal from Hilbert transform
    E_u = xr.DataArray(data=np.abs(hilbert(u_s, axis=0)),
                       coords=dict(t=u_s.t,
                                   z=u_s.z)    )
    E_t = xr.DataArray(data=np.abs(hilbert(t_s, axis=0)),
                       coords=dict(t=t_s.t,
                                   z=t_s.z)    )

    # lowpass filter the small-scale envelope
    # fft the envelopes
    f_Eu = xrft.fft(E_u, dim="t", true_phase=True, true_amplitude=True)
    f_Et = xrft.fft(E_t, dim="t", true_phase=True, true_amplitude=True)
    # loop over heights and lowpass filter - copied from above
    for jz in range(d.nz):
        # can do this all in one line
        # set high frequencies equal to zero -- sharp spectral filter
        # only keep -f_c < freq_t < f_c
        f_Eu[:,jz] = f_Eu[:,jz].where((f_Eu.freq_t < f_c.isel(z=jz)) &\
                                      (f_Eu.freq_t > -f_c.isel(z=jz)), other=0.)
        f_Et[:,jz] = f_Et[:,jz].where((f_Et.freq_t < f_c.isel(z=jz)) &\
                                      (f_Et.freq_t > -f_c.isel(z=jz)), other=0.)
    # inverse fft the filtered envelopes
    E_u_f = xrft.ifft(f_Eu, dim="freq_t", true_phase=True, true_amplitude=True,
                      lag=f_Eu.freq_t.direct_lag).real
    E_t_f = xrft.ifft(f_Et, dim="freq_t", true_phase=True, true_amplitude=True,
                      lag=f_Et.freq_t.direct_lag).real
    # reset time coordinate
    E_u_f["t"] = d.t
    E_t_f["t"] = d.t

    # AM coefficients ------------------------------------------------
    # new dataset to hold corr coeff
    R = xr.Dataset(data_vars=None,
                   coords=dict(z=d.z),
                   attrs=d.attrs
                  )
    # correlation between large scale u and filtered envelope of small-scale variable
    R["ul_Eu"] = xr.corr(u_l, E_u_f, dim="t")
    R["ul_Et"] = xr.corr(u_l, E_t_f, dim="t")
    # correlation between large scale u and filtered envelope of small-scale variable
    R["tl_Et"] = xr.corr(t_l, E_t_f, dim="t")
    # TODO: add more variables

    # Plot ------------------------------------------------
    # figure 1 - two panels
    # (a) modulation of small-scale u by large-scale u: R_ul_Eu
    # (b) modulation of small-scale theta by large-scale u: R_tl_Et
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                             figsize=(9.8, 5))
    # (a) R_ul_Eu vs z/h
    ax1[0].plot(R.z/stat.he, R.ul_Eu, "-k")
    # (b) R_ul_Et vs z/h
    ax1[1].plot(R.z/stat.he, R.ul_Et, "-k")
    # clean up
    ax1[0].set_xlabel("$z/h$")
    ax1[0].set_ylabel("$R_{u_l,u_s}$")
    ax1[0].set_ylim([-0.5, 0.5])
    ax1[0].set_xscale("log")
    ax1[1].set_xlabel("$z/h$")
    ax1[1].set_ylabel("$R_{u_l,\\theta_s}$")
    # plot ref lines
    for iax in ax1.flatten():
        iax.axhline(0, c="k", ls="-", alpha=0.5)
        iax.axvline(delta/stat.he, c="k", ls="-", alpha=0.5)

    # save
    fig1.tight_layout()
    fsave1 = f"{figdir}{R.stability}_amp_mod.png"
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

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    figdir = "/home/bgreene/SBL_LES/figures/spectrogram/"
    figdir_AM = "/home/bgreene/SBL_LES/figures/amp_mod/"
    # loop sims A--F
    for sim in list("A"):
        print(f"---Begin Sim {sim}---")
        ncdir = f"/home/bgreene/simulations/{sim}_192_interp/output/netcdf/"
        # calc_spectra(ncdir)
        # plot_spectrogram(ncdir, figdir)
        # plot_1D_spectra(ncdir, figdir)
        amp_mod(ncdir, figdir_AM)
        print(f"---End Sim {sim}---")
