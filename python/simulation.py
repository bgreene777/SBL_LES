# --------------------------------
# Name: simulation.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 12 May 2021
# Purpose: define the simulation class to be used in other code
# --------------------------------
import os
import numpy as np
from numba import njit
# ---------------------------------------------
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
# ---------------------------------------------
@njit
def interp_uas(dat, z_LES, z_UAS):
    """Interpolate LES virtual tower timeseries data in the vertical to match
    ascent rate of emulated UAS to create timeseries of ascent data
    :param float dat: 2d array of the field to interpolate, shape(nt, nz)
    :param float z_LES: 1d array of LES grid point heights
    :param float z_UAS: 1d array of new UAS z from ascent & sampling rates
    Outputs 2d array of interpolated field, shape(nt, len(z_UAS))
    """
    nt = np.shape(dat)[0]
    nz = len(z_UAS)
    dat_interp = np.zeros((nt, nz), dtype=np.float64)
    for i in range(nt):
        dat_interp[i,:] = np.interp(z_UAS, z_LES, dat[i,:])
    return dat_interp
# ---------------------------------------------
def min_max_plot(dat, n):
    """Determine range and intervals of parameter for plotting
    :param float dat: 1d array of data to analyze
    :param float n: number of labels to display
    Returns xmin, xmax, interval
    """
    xmin = np.floor(dat.min())
    xmax = np.ceil(dat.max())
    r = xmax - xmin
    if r/n < 2.:
        dmaj = 1.
        dmin = 0.5
    else:
        dmaj = 5.
        dmin = 1.
    return [xmin, xmax, dmaj, dmin]
# ---------------------------------------------
class simulation():
    """Contains simulation parameters and can read averaged csv files
    
    :var dict xytavg: averages of instantaneous variables in x,y,t dimensions
    :var dict cov: covariances of instantaneous variables averaged in x,y,t dimensions
    :var dict most: monin-obukhov similarity parameters
    """
    def __init__(self, path, nx, ny, nz, Lx, Ly, Lz, stab):
        """Creates a simulation object with appropriate setup parameters
        :param string path: absolute path to simulation output directory
        :param int nx: number of grid points in x-dimension
        :param int ny: number of grid points in y-dimension
        :param int nz: number of grid points in z-dimension
        :param float Lx: physical length of x-dimension in meters
        :param float Ly: physical length of y-dimension in meters
        :param float Lz: physical length of z-dimension in meters
        :param str stab: HBZ cooling rate label (A-F)
        """
        # initialize with args
        self.path = path
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.stab = stab
        
        # calculate dx, dy, dz, nt
        self.dx = self.Lx/float(self.nx)
        self.dy = self.Ly/float(self.ny)
        self.dz = self.Lz/float(self.nz)
        self.dd = (self.dx * self.dy * self.dz) ** (1./3.)
        
        # create label
        self.lab = str(self.nx).zfill(3)
        
        # initialize empty dicts for averaged and covariance vars
        self.xytavg = {}
        self.cov = {}
        self.var = {}
        self.Ri = {}
        self.most = {}
        self.tke = {} # tke budget terms
        self.flen = {} # filtered lengthscale data from .npz files
        self.len = {} # autocorr lengthscale data from .npz files
        self.spec = None # spectra data from .npz files
        self.RFM = {} # relaxed filter for random errors from .npz files
        self.RFM_new = {} # recalculated errors, both LP and RFM
        
        # initialize empty z variable
        self.z = None
        self.h = None
        self.i_h = None
        self.isbl = None
        
        # initialize T_H as empty array of zeros
        self.L_H = np.zeros(self.nz, dtype=float)
        
    def read_csv(self):
        print(f"--Beginning loading data for {self.nx}^3 simulation--")
        print(f"Reading file: {self.path}average_statistics.csv")
        
        data = np.genfromtxt(f"{self.path}average_statistics.csv", 
                             dtype=np.float64, delimiter=",", skip_header=1)
        # assign each column to parameters
        # columns are:
        # z, ubar, vbar, wbar, Tbar, uw_cov_res, uw_cov_tot,
        # vw_cov_res, vw_cov_tot, thetaw_cov_res, thetaw_cov_tot,
        # u_var_res, u_var_tot, v_var_res, v_var_tot, w_var_res,
        # w_var_tot, theta_var_res
#         self.z = data[:, 0]
        self.z = np.linspace(self.dz, self.Lz-self.dz, self.nz)
        self.xytavg["u"] = data[:, 1]
        self.xytavg["v"] = data[:, 2]
        self.xytavg["w"] = data[:, 3]
        self.xytavg["theta"] = data[:, 4]
        self.xytavg["dissip"] = data[:, 5]
        self.cov["uw_cov_res"] = data[:, 6]
        self.cov["uw_cov_tot"] = data[:, 7]
        self.cov["vw_cov_res"] = data[:, 8]
        self.cov["vw_cov_tot"] = data[:, 9]
        self.cov["thetaw_cov_res"] = data[:, 10]
        self.cov["thetaw_cov_tot"] = data[:, 11]
        self.var["u_var_res"] = data[:, 12]
        self.var["u_var_tot"] = data[:, 13]
        self.var["v_var_res"] = data[:, 14]
        self.var["v_var_tot"] = data[:, 15]
        self.var["w_var_res"] = data[:, 16]
        self.var["w_var_tot"] = data[:, 17]
        self.var["theta_var_tot"] = data[:, 18]
        self.var["u_var_tot2"] = data[:, 19]
        self.var["v_var_tot2"] = data[:, 20]
        
        # calculate ustar and thetastar and assign to cov
        ustar = ((self.cov["uw_cov_tot"]**2.) + (self.cov["vw_cov_tot"]**2.)) ** 0.25
        self.cov["ustar"] = ustar
        thetastar = -self.cov["thetaw_cov_tot"] / ustar
        self.cov["thetastar"] = thetastar
        L = -(ustar ** 3.) * self.xytavg["theta"][0] / (0.4 * 9.81 * self.cov["thetaw_cov_tot"])
        self.cov["L"] = L
    
        # calculate TKE and assign to var
        TKE = 0.5 * (self.var["u_var_tot"] + self.var["v_var_tot"] + self.var["w_var_tot"])
        self.var["TKE_tot"] = TKE
        u_var_sgs = self.var["u_var_tot"] - self.var["u_var_res"]
        v_var_sgs = self.var["v_var_tot"] - self.var["v_var_res"]
        w_var_sgs = self.var["w_var_tot"] - self.var["w_var_res"]
        TKE_sgs = 0.5 * (u_var_sgs + v_var_sgs + w_var_sgs)
        self.var["TKE_sgs"] = TKE_sgs
        
        # calculate zi based on linear extrapolated method
        i_h = np.where(ustar <= 0.05*ustar[0])[0][0]
        self.h = self.z[i_h] / 0.95
        self.i_h = i_h
        self.isbl = np.where(self.z <= self.h)[0]
        
        # calculate ws and wd and assign to xytavg
        self.xytavg["ws"] = np.sqrt( self.xytavg["u"]**2. + self.xytavg["v"]**2. )
        self.xytavg["wd"] = np.arctan2(-self.xytavg["u"], -self.xytavg["v"]) * 180./np.pi
        self.xytavg["wd"][self.xytavg["wd"] < 0.] += 360.
        
        # calculate level of LLJ core using wspd
        self.xytavg["zj"] = self.z[np.argmax(self.xytavg["ws"])]
        
        # now also read csv for TKE budget terms
        ftke = os.path.join(self.path, "tke_budget.csv")
        dtke = np.genfromtxt(ftke, dtype=float, skip_header=1, delimiter=",")
        # assign data into self.tke
        self.tke["z"] = dtke[:,0]
        self.tke["shear"] = dtke[:,1]
        self.tke["buoy"] = dtke[:,2]
        self.tke["trans"] = dtke[:,3]
        self.tke["diss"] = dtke[:,4]
        self.tke["tot"] = np.sum(dtke[:,1:], axis=1)
        self.tke["residual"] = np.zeros(len(self.tke["z"])) - self.tke["tot"]
        # nondimensionalize
        self.tke["scale"] = (self.cov["ustar"][0]**3.) / self.tke["z"][1:] / 0.4
        return
    
    def calc_Ri(self):
        du_dz, dv_dz, dtheta_dz, dws_dz = ([] for _ in range(4))
        ws = ( (self.xytavg["u"]**2.) + (self.xytavg["v"]**2.) ) ** 0.5
        
        for i in range(1, self.nz-1):
            du_dz.append( (self.xytavg["u"][i+1] - self.xytavg["u"][i-1]) / (2.*self.dz) )
            dv_dz.append( (self.xytavg["v"][i+1] - self.xytavg["v"][i-1]) / (2.*self.dz) )
            dtheta_dz.append( (self.xytavg["theta"][i+1] - self.xytavg["theta"][i-1]) / (2.*self.dz) )
            dws_dz.append( (ws[i+1] - ws[i-1]) / (2.*self.dz) )
        du_dz = np.array(du_dz)
        dv_dz = np.array(dv_dz)
        dtheta_dz = np.array(dtheta_dz)
        dws_dz = np.array(dws_dz)
        
        S2 =(du_dz**2.) + (dv_dz**2.)
#         S2 = dws_dz ** 2.
        N2 = dtheta_dz * 9.81 / self.xytavg["theta"][0]
        Ri = N2 / S2
        
        self.Ri["du_dz"] = du_dz
        self.Ri["dv_dz"] = dv_dz
        self.Ri["dtheta_dz"] = dtheta_dz
        self.Ri["S2"] = S2
        self.Ri["N2"] = N2
        self.Ri["Ri"] = Ri
        
        # now calculate flux Ri
        Ri_f_num = self.cov["thetaw_cov_tot"][1:-1] * 9.81 / self.xytavg["theta"][0]
        Ri_f_den = (self.cov["uw_cov_tot"][1:-1] * du_dz) + (self.cov["vw_cov_tot"][1:-1] * dv_dz)
        self.Ri["Ri_f"] = Ri_f_num / Ri_f_den
        
        # calculate zB from van de Wiel et al 2008
        # zB = sqrt(0.5) * sigma_w/N = sqrt(0.5 * var_w / N2)
        self.Ri["zB"] = (0.5 * self.var["w_var_tot"][1:-1] / N2) ** 0.5
        
        # calculate Ozmidov scale Lo
        # Lo = sqrt[<dissipation>/(N^2)^3/2]
        self.Ri["Lo"] = np.sqrt(-self.xytavg["dissip"][1:-1] / (N2 ** (3./2.)))
        
        return
    
    def calc_most(self):
        k = 0.4
        phi_m = np.sqrt(self.Ri["S2"]) * k * self.z[1:-1] / self.cov["ustar"][0]
        phi_h = self.Ri["dtheta_dz"] * k * self.z[1:-1] / self.cov["thetastar"][0]
        phi_m_MO = 1. + 6.*self.z/self.cov["L"][0]
        phi_h_MO = 0.95 + 7.8*self.z/self.cov["L"][0]
        # calculate local phi_m, phi_h
        phi_m_l = np.sqrt(self.Ri["S2"]) * k * self.z[1:-1] / self.cov["ustar"][1:-1]
        phi_h_l = self.Ri["dtheta_dz"] * k * self.z[1:-1] / self.cov["thetastar"][1:-1]
        
        # assign
        self.most["phi_m"] = phi_m
        self.most["phi_h"] = phi_h      
        self.most["phi_m_MO"] = phi_m_MO
        self.most["phi_h_MO"] = phi_h_MO
        self.most["phi_m_l"] = phi_m_l
        self.most["phi_h_l"] = phi_h_l
        
        # now look at gradient-based scales
        kz = k * self.z[1:-1]
        # see mixing length from Sorbjan 2017!!
        L0 = 19.22  # m
        L1 = 1. / (self.Ri["Ri"]**(3./2))
        Ls = kz / (1. + (kz/L0) + (kz/L1))
        Us = Ls * np.sqrt(self.Ri["N2"])
        Ts = Ls * self.Ri["dtheta_dz"]
        # assign
        self.most["Gm"] = (self.cov["ustar"][1:-1]**2.) / (Us**2.)
        self.most["Gh"] = -self.cov["thetaw_cov_tot"][1:-1] / Us / Ts
        self.most["Gw"] = np.sqrt(self.var["w_var_tot"][1:-1]) / Us
        self.most["Gtheta"] = np.sqrt(self.var["theta_var_tot"][1:-1]) / Ts
        self.most["Us"] = Us
        self.most["Ts"] = Ts
        self.most["Ls"] = Ls
        
        return
    
    def read_filt_len(self, npz, label):
        self.flen[label] = np.load(npz)
        return
        
    def read_auto_len(self, npz, calc_LH=False):
        # read npz file of autocorrelation lengthscale
        dat = np.load(npz)
        for key in dat.keys():
            self.len[key] = dat[key]
        # calculate err_u from u_len
        L_samp = 3. * self.xytavg["ws"]
        self.len["err_u"] = np.sqrt((2.*self.len["u_len"]*self.var["u_var_tot"])/(L_samp*self.xytavg["ws"]**2.))
        self.len["err_theta"] = np.sqrt((2.*self.len["theta_len"]*self.var["theta_var_tot"])/(L_samp*self.xytavg["theta"]**2.))
            
        if calc_LH:
            x = np.linspace(0., self.Lx, self.nx)
            # calculate L_H from autocorrelation (only for u) at each z
            for jz in range(self.nz):
                # Bartlett large-lag standard error
                # determine equivalent of Q*delta_t = 5 min in spatial coords
                Qdx = (5.*60.) / self.xytavg["ws"][jz]  # m
                iQdx = np.where(x <= Qdx)[0][-1]
                imid = len(self.len["u_corr"][:,jz]) // 2
                # calculate standard error
                varB = (1. + 2.*np.sum(self.len["u_corr"][imid:iQdx,jz]**2.)) / self.nx
                # now look at autocorrelation to find spatial lag eta for L_H
                errB = np.sqrt(varB)
                # grab first instance of autocorr dipping below errB - this is L_H
                iLH = np.where(abs(self.len["u_corr"][imid:,jz]) <= errB)[0][0]
                self.L_H[jz] = x[iLH]
        return
    
    def read_spectra(self, npz):
        self.spec = np.load(npz)
        return
    
    def read_RFM(self, npz):
        dat = np.load(npz, allow_pickle=True)
        for key in dat.keys():
            self.RFM[key] = dat[key]
        return
    
    def recalc_rand_err(self, Tnew):
        # recalc random error for theta and *unrotated* u and v for UAS profs
        # first, LP error
        # err_LP = sqrt{2*L*var / (L * mean**2)}
        isbl = self.RFM["isbl"]
        Lnew = Tnew * self.xytavg["ws"][isbl]
        len_key = ["len_u2", "len_v2", "len_theta"]
        var_key = ["u_var_tot2", "v_var_tot2", "theta_var_tot"]
        avg_key = ["u", "v", "theta"]
        param = ["u2", "v2", "theta"]
        
        # loop over keys
        for ik in range(3):
        # calculate
            err_LP = np.sqrt((2.*self.RFM[len_key[ik]]*self.var[var_key[ik]][isbl])/\
                          (Lnew*self.xytavg[avg_key[ik]][isbl]))
            # assign to RFM_new
            self.RFM_new[f"err_{param[ik]}_LP"] = err_LP
        
        # now calculate RFM error
        # need to get L_H_param from dx_LH_param and delta_x
        # this indexing will give L_H as function of z (within sbl)
        dxLH_key = ["dx_LH_u2", "dx_LH_v2", "dx_LH_t"]
        
        # loop over keys
        for ik in range(3):
            L_H = self.RFM["delta_x"][0] / self.RFM[dxLH_key[ik]][0,:]
            x_LH = (self.xytavg["ws"][isbl] * Tnew) / L_H
            # calculate MSE = var * C * (x/L_H)**-p
            MSE = self.var[var_key[ik]][isbl] *\
                  (self.RFM[f"C_{param[ik]}"] * (x_LH ** -self.RFM[f"p_{param[ik]}"]))
            RMSE = np.sqrt(MSE)
            # err_RFM = RMSE / mean
            err_RFM = RMSE / self.xytavg[avg_key[ik]][isbl]
            self.RFM_new[f"err_{param[ik]}"] = err_RFM
            
        # now perform error propagation to calculate errors in ws and wd
        # err_ws = sqrt[(sig_u^2 * u^2 + sig_v^2 * v^2) / (u^2 + v^2)] / ws
        # err_wd = sqrt[(sig_u^2 * v^2 + sig_v^2 * u^2) / (u^2 + v^2)^2] / wd
        # NOTE 1: wd term in sqrt will return radians, not degrees
        # NOTE 2: these use the *unrotated* error variances for u and v
        # first recalculate error variances from error and mean quantities
        sig_u = self.RFM_new["err_u2"] * abs(self.xytavg["u"][isbl])
        sig_v = self.RFM_new["err_v2"] * abs(self.xytavg["v"][isbl])
        # calculate ws error and assign to RFM
        err_ws = np.sqrt( (sig_u**2. * self.xytavg["u"][isbl]**2. +\
                           sig_v**2. * self.xytavg["v"][isbl]**2.)/\
                          (self.xytavg["u"][isbl]**2. +\
                           self.xytavg["v"][isbl]**2.) ) /\
                 self.xytavg["ws"][isbl]
        self.RFM_new["err_ws"] = err_ws
        # calculate wd error and assign to RFM
        err_wd = np.sqrt( (sig_u**2. * self.xytavg["v"][isbl]**2. +\
                           sig_v**2. * self.xytavg["u"][isbl]**2.)/\
                          ((self.xytavg["u"][isbl]**2. +\
                            self.xytavg["v"][isbl]**2.)**2.) ) /\
                 (self.xytavg["wd"][isbl] * np.pi/180.)  # normalize w/ rad
        self.RFM_new["err_wd"] = err_wd        
            
        return
        
    def calc_time_error(self, err_range):
        """Basically invert recalc_rand_err and determine the amount of 
        averaging time necessary to achieve a predetermined error value
        """
        isbl = self.RFM["isbl"]
        var_key = ["u_var_tot2", "v_var_tot2", "theta_var_tot"]
        avg_key = ["u", "v", "theta"]
        param = ["u2", "v2", "theta"]
               
        # now calculate RFM sampling time as function of err
        # Tavg_u(err) = (L_H/<ws>) * [(err_u * <u>)**2 / (u'u' * C_u)]**-1/p_u
        # need to get L_H_param from dx_LH_param and delta_x
        # this indexing will give L_H as function of z (within sbl)
        dxLH_key = ["dx_LH_u2", "dx_LH_v2", "dx_LH_t"]

        # loop over keys
        for ik in range(3):
            L_H = self.RFM["delta_x"][0] / self.RFM[dxLH_key[ik]][0,:]
            # initialize empty array for storing Tavg, shape(nz_sbl,nerr)
            Tavg_param = np.zeros((len(isbl), len(err_range)), dtype=np.float64)
            # calculate Tavg by looping through err_range            
            for ie, err in enumerate(err_range):
                Tavg_param[:,ie] = (L_H/self.xytavg["ws"][isbl]) *\
                ( ((err * self.xytavg[f"{avg_key[ik]}"][isbl])**2.)/\
                  (self.var[f"{var_key[ik]}"][isbl] * self.RFM[f"C_{param[ik]}"])\
                ) ** (-1/self.RFM[f"p_{param[ik]}"])
                
                # store in RFM_new
                self.RFM_new[f"Tavg_{param[ik]}"] = Tavg_param            
            
        return
        
        
    
class UAS_emulator(simulation):
    """Emulate a profile from a rotary-wing UAS based on timeseries data
    Inherits simulation class to conveniently load in mean simulation
    quantities.
    """
    def __init__(self, path, nx, ny, nz, Lx, Ly, Lz, stab):
        # initialize simulation values, load mean csv, calc Ri and MOST
        simulation.__init__(self, path, nx, ny, nz, Lx, Ly, Lz, stab)
        self.read_csv()
        self.calc_Ri()
        self.calc_most()
        self.read_RFM(f"/home/bgreene/SBL_LES/output/RFM_{self.stab}{self.lab}.npz")
        # now assign additional params
        self.ts = {}  # raw timeseries data: u, v, w, theta
        self.u_scale = 0.4
        self.T_scale = 300.
        self.raw_prof = {}  # raw/unprocessed profile w/ same samp freq as LES
        self.prof = {}  # profile of sampled parameters from timeseries
        self.ec = {}  # profile of 2nd order moments at every native z from LES
        
    def read_timeseries(self, nt_tot, dt, raw=True):
        # read last hour of simulation
        # based on number of timesteps: nt_tot
        # and timestep: dt in seconds
        # if raw==True, load from individual timeseries files and
        # create new npz file for faster loading later
        # calculate number of timesteps in last half hour
        if raw:
            nt = int(1800./dt)
            istart = nt_tot - nt        
            # initialize empty arrays shape(nt, nz)
            u_ts, v_ts, w_ts, theta_ts, txz_ts, tyz_ts, q3_ts =\
            (np.zeros((nt, self.nz), dtype=np.float64) for _ in range(7))
            # now loop through files (one for each jz)
            for jz in range(self.nz):
                print(f"Loading timeseries data, jz={jz}")
                fu = f"{self.path}u_timeseries_c{jz:03d}.out"
                u_ts[:,jz] = np.loadtxt(fu, skiprows=istart, usecols=1)
                fv = f"{self.path}v_timeseries_c{jz:03d}.out"
                v_ts[:,jz] = np.loadtxt(fv, skiprows=istart, usecols=1)
                fw = f"{self.path}w_timeseries_c{jz:03d}.out"
                w_ts[:,jz] = np.loadtxt(fw, skiprows=istart, usecols=1)
                ftheta = f"{self.path}t_timeseries_c{jz:03d}.out"
                theta_ts[:,jz] = np.loadtxt(ftheta, skiprows=istart, usecols=1)
                ftxz = f"{self.path}txz_timeseries_c{jz:03d}.out"
                txz_ts[:,jz] = np.loadtxt(ftxz, skiprows=istart, usecols=1)
                ftyz = f"{self.path}tyz_timeseries_c{jz:03d}.out"
                tyz_ts[:,jz] = np.loadtxt(ftyz, skiprows=istart, usecols=1)
                fq3 = f"{self.path}q3_timeseries_c{jz:03d}.out"
                q3_ts[:,jz] = np.loadtxt(fq3, skiprows=istart, usecols=1)
            # apply scales
            u_ts *= self.u_scale
            v_ts *= self.u_scale
            w_ts *= self.u_scale
            theta_ts *= self.T_scale
            txz_ts *= (self.u_scale * self.u_scale)
            tyz_ts *= (self.u_scale * self.u_scale)
            q3_ts *= (self.u_scale * self.T_scale)
            # now create time vector and assign
            time = np.linspace(0., 1800.-dt, nt)
            # save npz file with all this data
            fsave = f"{self.path}timeseries.npz"
            np.savez(fsave, u=u_ts, v=v_ts, w=w_ts,
                     theta=theta_ts, 
                     txz=txz_ts, tyz=tyz_ts, q3=q3_ts,
                     time=time)
        else:
            # load npz file and assign to self.ts dictionary
            print(f"Reading file: {self.path}timeseries.npz")
            dat = np.load(f"{self.path}timeseries.npz")
            for key in dat.keys():
                self.ts[key] = dat[key]
            # also save dt for use later
            self.ts["dt"] = dt
                
        return
                
    def profile(self, ascent_rate=1.0, time_average=3.0, time_start=0.0):
        # specify an ascent rate in m/s, default = 1.0; if <= 0 then instantaneous
        # also specify averaging intervals to match random error analysis
        # can set time_start lag in seconds to somewhat randomize profiles
        # output data on z grid based on ascent_rate and time_average
        # assigns data to self.prof and returns
        # first check to see if time_constant != 0 and set flag
        # now check ascent rate and set flag
        inst = False
        if ascent_rate <= 0.0:
            inst = True  
        # calculate array of theoretical altitudes based on ts["time"]
        # and ascent_rate, keeping in account time_start
        zuas = ascent_rate * self.ts["time"]
        # find the index in self.ts["time"] that corresponds to time_start
        istart = int(time_start / self.ts["dt"])
        # set zuas[:istart] = 0 and then subtract everything after that
        zuas[:istart] = 0
        zuas[istart:] -= zuas[istart]
            
        # now only grab indices where 3 m <= zuas <= 396 m
        iuse = np.where((zuas >= 3.) & (zuas <= 396.))[0]
        zuas = zuas[iuse]
        # now the fun part -- interpolate in z to get continuous timeseries
        # of UAS data matching LES timestep dt
        # define empty dictionary to make it easier to loop and store
        d_interp = {}
        # define interp keys
        interp_keys = ["u", "v", "w", "theta"]
        # call the interp_uas function defined above
        for key in interp_keys: 
            print(f"Interpolating {key}...")
            d_interp[key] = interp_uas(self.ts[key][iuse, :], 
                                       self.z, zuas)
        # now grab data from interp arrays to create simulated raw UAS profiles
        uas_ts = {}
        for key in interp_keys:
            uas_ts[key] = []
            for i in range(len(iuse)):
                uas_ts[key].append(d_interp[key][i,i])
            self.raw_prof[key] = np.array(uas_ts[key])
        self.raw_prof["z"] = zuas
        # calc ws and wd
        self.raw_prof["ws"] = ((self.raw_prof["u"]**2.) +\
                               (self.raw_prof["v"]**2.)) ** 0.5
        wd = np.arctan2(-self.raw_prof["u"], -self.raw_prof["v"]) * 180./np.pi
        wd[wd < 0.] += 360.
        self.raw_prof["wd"] = wd
        
        # now post-process to new altitude bins based on specified average time
        dz_new = ascent_rate * time_average  # m/s * s = m
        znew = np.arange(3., 396.1, dz_new, dtype=np.float64)
        self.prof["z"] = znew
        uas_mean = {}
        for key in interp_keys:
            uas_mean[key] = []
            for jz, zz in enumerate(znew):
                # find indices in zuas to average over
                imean = np.where((zuas >= zz-dz_new/2.) &\
                                 (zuas < zz+dz_new/2.))[0]
                uas_mean[key].append(np.mean(self.raw_prof[key][imean]))
            self.prof[key] = np.array(uas_mean[key])
        # also calculate ws and wd and assign
        self.prof["ws"] = ((self.prof["u"]**2.) + (self.prof["v"]**2.)) ** 0.5
        wd = np.arctan2(-self.prof["u"], -self.prof["v"]) * 180./np.pi
        wd[wd < 0.] += 360.
        self.prof["wd"] = wd
        
        # calculate errors for ws and wd from error propagation
        # err_ws = sqrt[(sig_u^2 * u^2 + sig_v^2 * v^2) / (u^2 + v^2)] / ws
        # err_wd = sqrt[(sig_u^2 * v^2 + sig_v^2 * u^2) / (u^2 + v^2)^2] / wd
        # NOTE 1: wd term in sqrt will return radians, not degrees
        # NOTE 2: these use the *unrotated* error variances for u and v
        # first recalculate error variances from error and mean quantities
        isbl = self.RFM["isbl"]
        sig_u = self.RFM["err_u2"] * abs(self.xytavg["u"][isbl])
        sig_v = self.RFM["err_v2"] * abs(self.xytavg["v"][isbl])
        # calculate ws error and assign to RFM
        err_ws = np.sqrt( (sig_u**2. * self.xytavg["u"][isbl]**2. +\
                           sig_v**2. * self.xytavg["v"][isbl]**2.)/\
                          (self.xytavg["u"][isbl]**2. +\
                           self.xytavg["v"][isbl]**2.) ) /\
                 self.xytavg["ws"][isbl]
        self.RFM["err_ws"] = err_ws
        # calculate wd error and assign to RFM
        err_wd = np.sqrt( (sig_u**2. * self.xytavg["v"][isbl]**2. +\
                           sig_v**2. * self.xytavg["u"][isbl]**2.)/\
                          ((self.xytavg["u"][isbl]**2. +\
                           self.xytavg["v"][isbl]**2.)**2.) ) /\
                 (self.xytavg["wd"][isbl] * np.pi/180.)  # normalize w/ rad
        self.RFM["err_wd"] = err_wd
        
        # get isbl for znew
        isbl_new = np.where(znew <= self.h)[0]
        self.prof["isbl"] = isbl_new
        # interpolate errors from RFM to znew grid
        for key in ["ws", "wd", "theta"]:
            err_new = np.interp(znew[isbl_new], self.z[self.RFM["isbl"]], self.RFM[f"err_{key}"])
            self.RFM[f"err_{key}_interp"] = err_new
        return
    
    def calc_ec(self, time_average=1800., time_start=0.0):
        # grab profiles of sample 2nd order moments from a virtual
        # eddy covariance tower to plot along with ensemble mean
        # default time_average is 30 minutes
        # find the index in self.ts["time"] that corresponds to time_start
        istart = int(time_start / self.ts["dt"])
        # determine how many indices to use from time_average
        nuse = int(time_average / self.ts["dt"])
        # create array of indices to use
        iuse = np.linspace(istart, istart+nuse-1, nuse, dtype=int)
        # begin calculating statistics
        # calc mean quantities from selected window
        Ubar = np.mean(self.ts["u"][iuse,:], axis=0)
        Vbar = np.mean(self.ts["v"][iuse,:], axis=0)
        Wbar = np.mean(self.ts["w"][iuse,:], axis=0)
        thetabar = np.mean(self.ts["theta"][iuse,:], axis=0)
        # rotate Ubar and Vbar so <V> = 0 to use with variances
        angle = np.arctan2(Vbar, Ubar)
        Ubar_rot = Ubar*np.cos(angle) + Vbar*np.sin(angle)
        Vbar_rot =-Ubar*np.sin(angle) + Vbar*np.cos(angle)
        # alro rotate raw u and v
        u_rot = self.ts["u"]*np.cos(angle) + self.ts["v"]*np.sin(angle)
        v_rot =-self.ts["u"]*np.sin(angle) + self.ts["v"]*np.cos(angle)
        # calc fluctuating components
        u_fluc = self.ts["u"][iuse,:] - Ubar
        v_fluc = self.ts["v"][iuse,:] - Vbar
        w_fluc = self.ts["w"][iuse,:] - Wbar
        theta_fluc = self.ts["theta"][iuse,:] - thetabar
        # fluctuating rotated u and v
        u_fluc_rot = u_rot[iuse,:] - Ubar_rot
        v_fluc_rot = v_rot[iuse,:] - Vbar_rot
        # calc resolved covariances
        uw_cov_res = np.mean((u_fluc*w_fluc), axis=0)
        vw_cov_res = np.mean((v_fluc*w_fluc), axis=0)
        thetaw_cov_res = np.mean((theta_fluc*w_fluc), axis=0)
        # add sgs components to get total covariances
        uw_cov_tot = uw_cov_res + np.mean(self.ts["txz"][iuse,:], axis=0)
        vw_cov_tot = vw_cov_res + np.mean(self.ts["tyz"][iuse,:], axis=0)
        thetaw_cov_tot = thetaw_cov_res + np.mean(self.ts["q3"][iuse,:], axis=0)
        # now do variances using rotated fluctuating components
        u_var = np.mean((u_fluc_rot*u_fluc_rot), axis=0)
        v_var = np.mean((v_fluc_rot*v_fluc_rot), axis=0)
        w_var = np.mean((w_fluc*w_fluc), axis=0)
        theta_var = np.mean((theta_fluc*theta_fluc), axis=0)
        
        # store everything in self.ec
        self.ec["uw_cov_tot"] = uw_cov_tot
        self.ec["vw_cov_tot"] = vw_cov_tot
        self.ec["thetaw_cov_tot"] = thetaw_cov_tot
        self.ec["u_var"] = u_var
        self.ec["v_var"] = v_var
        self.ec["w_var"] = w_var
        self.ec["theta_var"] = theta_var
        
        return