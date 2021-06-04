# --------------------------------
# Name: simulation.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 12 May 2021
# Purpose: define the simulation class to be used in other code
# --------------------------------
import os
import numpy as np

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
        
        # initialize empty z variable
        self.z = None
        self.h = None
        
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
        Us = k * self.z[1:-1] * np.sqrt(self.Ri["N2"])
        Ts = k * self.z[1:-1] * self.Ri["dtheta_dz"]
        Ls = k * self.z[1:-1]
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
        # now assign additional params
        self.ts = {}  # raw timeseries data: u, v, w, theta
        self.u_scale = 0.4
        self.T_scale = 300.
        self.prof = {}  # profile of sampled parameters from timeseries
        
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
            u_ts, v_ts, w_ts, theta_ts =\
            (np.zeros((nt, self.nz), dtype=np.float64) for _ in range(4))
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
            # apply scales
            u_ts *= self.u_scale
            v_ts *= self.u_scale
            w_ts *= self.u_scale
            theta_ts *= self.T_scale
            # now create time vector and assign
            time = np.linspace(0., 1800.-dt, nt)
            # save npz file with all this data
            fsave = f"{self.path}timeseries.npz"
            np.savez(fsave, u=u_ts, v=v_ts, w=w_ts,
                     theta=theta_ts, time=time)
        else:
            # load npz file and assign to self.ts dictionary
            print(f"Reading file: {self.path}timeseries.npz")
            dat = np.load(f"{self.path}timeseries.npz")
            for key in dat.keys():
                self.ts[key] = dat[key]
                
        return
                
    def profile(self, ascent_rate=1.0, time_constant=0.0):
        # specify an ascent rate in m/s, default = 1.0; if <= 0 then instantaneous
        # time constant for temperature measurements in seconds, default=0.0
        # output data on same z grid as simulation
        # assigns data to self.prof and returns
        # first check to see if time_constant != 0 and set flag
        sens_lag = False
        if time_constant > 0.0:
            sens_lag = True
        # now check ascent rate and set flag
        inst = False
        if ascent_rate <= 0.0:
            inst = True  
        # calculate array of theoretical altitudes based on ts["time"]
        # and ascent_rate
        zuas = ascent_rate * self.ts["time"]
        u_mean, v_mean, w_mean, theta_mean = ([] for _ in range(4))
        for jz, zz in enumerate(self.z):
            # find indices in zuas corresponding to each simulation altitude
            imean = np.where((zuas >= zz-self.dz/2.) & (zuas < zz+self.dz/2.))[0]
            u_mean.append(np.mean(self.ts["u"][imean, jz]))
            v_mean.append(np.mean(self.ts["v"][imean, jz]))
            w_mean.append(np.mean(self.ts["w"][imean, jz]))
            theta_mean.append(np.mean(self.ts["theta"][imean, jz]))
        print(imean)
        # assign to self.profile and return
        self.prof["u"] = np.array(u_mean)
        self.prof["v"] = np.array(v_mean)
        self.prof["w"] = np.array(w_mean)
        self.prof["theta"] = np.array(theta_mean)
        # also calculate ws and wd
        self.prof["ws"] = ((self.prof["u"]**2.) + (self.prof["v"]**2.)) ** 0.5
        wd = np.arctan2(-self.prof["u"], -self.prof["v"]) * 180./np.pi
        wd[wd < 0.] += 360.
        self.prof["wd"] = wd
        return
        