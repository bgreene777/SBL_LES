# --------------------------------
# Name: plot_mean.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 10 May 2021
# Purpose: Read xyt averaged files from calc_stats.f90 to plot profiles of
# quantities output by LES. Loops over multiple grid sizes and stabilities
# for comparisons
# --------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from datetime import datetime, timedelta

# Configure plots
rc('font',weight='normal',size=20,family='serif',serif='Computer Modern Roman')
rc('text',usetex='True')
colors = [(225./255, 156./255, 131./255),
          (134./255, 149./255, 68./255), (38./255, 131./255, 63./255),
          (0., 85./255, 80./255), (20./255, 33./255, 61./255), (252./255, 193./255, 219./255)]
fdir_save = "/home/bgreene/SBL_LES/figures/grid_sensitivity/"
plt.close("all")

# --------------------------------
# define simulation class to hold all data and calculate additional params
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
        self.dz = self.Lz/float(self.nz-1)
        
        # create label
        self.lab = str(self.nx).zfill(3)
        
        # initialize empty dicts for averaged and covariance vars
        self.xytavg = {}
        self.cov = {}
        self.var = {}
        self.Ri = {}
        self.most = {}
        self.tke = {} # tke budget terms
        
        # initialize empty z variable
        self.z = None
        self.h = None
        
    def read_csv(self):
        print(f"Beginning loading data for {self.nx}^3 simulation--")
        print(f"Reading file: {self.path}")
        
        data = np.genfromtxt(self.path, dtype=float, delimiter=",", skip_header=1)
        # assign each column to parameters
        # columns are:
        # z, ubar, vbar, wbar, Tbar, uw_cov_res, uw_cov_tot,
        # vw_cov_res, vw_cov_tot, thetaw_cov_res, thetaw_cov_tot,
        # u_var_res, u_var_tot, v_var_res, v_var_tot, w_var_res,
        # w_var_tot, theta_var_res
#         self.z = data[:, 0]
        self.z = np.linspace(0., 400., self.nz)
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
        
        # calculate ws and wd and assign to xytavg
        self.xytavg["ws"] = np.sqrt( self.xytavg["u"]**2. + self.xytavg["v"]**2. )
        self.xytavg["wd"] = np.arctan2(-self.xytavg["u"], -self.xytavg["v"]) * 180./np.pi
        self.xytavg["wd"][self.xytavg["wd"] < 0.] += 360.
        
        # calculate level of LLJ core using wspd
        self.xytavg["zj"] = self.z[np.argmax(self.xytavg["ws"])]
        
        # now also read csv for TKE budget terms
        ftke = os.path.join(self.path.rsplit("/", 1)[0], "tke_budget.csv")
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
    
# --------------------------------

# initialize simulation objects
# A
s128A = simulation("/home/bgreene/simulations/A_128_interp/output/average_statistics.csv",
                  128, 128, 128, 800., 800., 400., "A")
s160A = simulation("/home/bgreene/simulations/A_160_interp/output/average_statistics.csv",
                  160, 160, 160, 800., 800., 400., "A")
s192A = simulation("/home/bgreene/simulations/A_192_interp/output/average_statistics.csv",
                  192, 192, 192, 800., 800., 400., "A")
# F
s128F = simulation("/home/bgreene/simulations/F_128_interp/output/average_statistics.csv",
                  128, 128, 128, 800., 800., 400., "F")

# put everything into a list for looping
# s_all = [s128A, s160A, s192A]
s_all = [s128F]
for s in s_all:
    s.read_csv()
    s.calc_Ri()
    s.calc_most()
    
# --------------------------------
# Begin plotting
# --------------------------------

#
# Figure 1: unrotated u, v, wspd; wdir; theta
#
fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u
    ax1[0].plot(s.xytavg["u"], s.z, color=colors[i], linestyle="--")
    # v
    ax1[0].plot(s.xytavg["v"], s.z, color=colors[i], linestyle=":")
    # ws
    ax1[0].plot(s.xytavg["ws"], s.z, color=colors[i], linestyle="-", label=s.lab)
    # wdir
    ax1[1].plot(s.xytavg["wd"], s.z, color=colors[i], linestyle="-")
    # theta
    ax1[2].plot(s.xytavg["theta"], s.z, color=colors[i], linestyle="-")
    ax1[2].axhline(s.h, color=colors[i], linestyle=":", linewidth=2, 
                   label=f"$h = {s.h:4.1f}$ m")
# clean up
ax1[0].grid()
ax1[0].legend()
ax1[0].set_xlabel(r"Wind Speed, $\langle u \rangle$, $\langle v \rangle$ [m s$^{-1}$]")
ax1[0].set_ylabel("$z$ [m]")

ax1[1].grid()
ax1[1].set_xlabel(r"Wind Direction [$^{\circ}$]")

ax1[2].grid()
ax1[2].legend()
ax1[2].set_xlabel(r"$\langle \theta \rangle$ [K]")

# save figure
fig1.savefig(f"{fdir_save}{s_all[0].stab}_u_v_theta.pdf", format="pdf", bbox_inches="tight")
plt.close(fig1)


#
# Figure 2: unrotated <u'w'>, <v'w'>, <theta'w'>
#
fig2, ax2 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u'w'
    ax2[0].plot(s.cov["uw_cov_tot"], s.z, color=colors[i], linestyle="-", label=s.lab)
    # v'w'
    ax2[1].plot(s.cov["vw_cov_tot"], s.z, color=colors[i], linestyle="-")
    # theta'w'
    ax2[2].plot(s.cov["thetaw_cov_tot"], s.z, color=colors[i], linestyle="-")
# clean up
ax2[0].grid()
ax2[0].legend()
ax2[0].set_xlabel(r"$\langle u'w' \rangle$ [m$^2$ s$^{-2}$]")
ax2[0].set_ylabel("$z/h$")
# ax2[0].set_ylim([-0.05, 1.])

ax2[1].grid()
ax2[1].set_xlabel(r"$\langle v'w' \rangle$ [m$^2$ s$^{-2}$]")

ax2[2].grid()
ax2[2].set_xlabel(r"$\langle \theta'w' \rangle$ [K m s$^{-1}$]")

# save figure
fig2.savefig(f"{fdir_save}{s_all[0].stab}_covars.pdf", format="pdf", bbox_inches="tight")
plt.close(fig2)

#
# Figure 3: *rotated* <u'u'>, <v'v'>, <w'w'>
#
fig3, ax3 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # u'u'
    ax3[0].plot(s.var["u_var_tot"], s.z, color=colors[i], linestyle="-", label=s.lab)
    # v'v'
    ax3[1].plot(s.var["v_var_tot"], s.z, color=colors[i], linestyle="-")
    # theta'w'
    ax3[2].plot(s.var["w_var_tot"], s.z, color=colors[i], linestyle="-")
# clean up
ax3[0].grid()
ax3[0].legend()
ax3[0].set_xlabel(r"$\langle u'^2 \rangle$ [m$^2$ s$^{-2}$]")
ax3[0].set_ylabel("$z/h$")
# ax3[0].set_ylim([-0.05, 1.])

ax3[1].grid()
ax3[1].set_xlabel(r"$\langle v'^2 \rangle$ [m$^2$ s$^{-2}$]")

ax3[2].grid()
ax3[2].set_xlabel(r"$\langle w'^2 \rangle$ [m$^2$ s$^{-2}$]")

# save figure
fig3.savefig(f"{fdir_save}{s_all[0].stab}_vars.pdf", format="pdf", bbox_inches="tight")
plt.close(fig3)

#
# Figure 4: TKE, <theta'theta'>, ustar
#
fig4, ax4 = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # TKE
    ax4[0].plot(s.var["TKE_tot"]/s.cov["ustar"][0]/s.cov["ustar"][0], s.z, color=colors[i], 
                linestyle="-", label=str(s.nx).zfill(3))
    # theta var
    ax4[1].plot(s.var["theta_var_tot"], s.z, color=colors[i], linestyle="-")
    # ustar
    ax4[2].plot(s.cov["ustar"], s.z, color=colors[i], linestyle="-")
    
# clean up
ax4[0].grid()
ax4[0].legend()
ax4[0].set_xlabel(r"TKE / $u_{*}^2$")
ax4[0].set_ylabel("$z/h$")
# ax4[0].set_ylim([-0.05, 1.])

ax4[1].grid()
ax4[1].set_xlabel(r"$\langle \theta'^2 \rangle$ [K$^2$]")

ax4[2].grid()
ax4[2].set_xlabel(r"$u_{*}$ [m s$^{-1}$]")

# save figure
fig4.savefig(f"{fdir_save}{s_all[0].stab}_tke.pdf", format="pdf", bbox_inches="tight")
plt.close(fig4)


#
# Figure 5: S2 and N2; Ri
#
fig5, ax5 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(14.8, 5))
for i, s in enumerate(s_all):
    # S2
    ax5[0].plot(s.Ri["S2"], s.z[1:-1]/s.h, color=colors[i], linestyle="-", label=s.lab)
    # N2
    ax5[0].plot(s.Ri["N2"], s.z[1:-1]/s.h, color=colors[i], linestyle=":")
    # Ri
    ax5[1].plot(s.Ri["Ri"], s.z[1:-1]/s.h, color=colors[i], linestyle="-")
    # Ri_f
    ax5[1].plot(s.Ri["Ri_f"], s.z[1:-1]/s.h, color=colors[i], linestyle=":")
# clean up
ax5[0].grid()
ax5[0].legend()
ax5[0].set_xlabel(r"$S^2, N^2$ [s$^{-2}$]")
ax5[0].set_ylabel("$z/h$")
ax5[0].set_ylim([0, 1.2])
# ax5[0].set_xscale("log")
# ax5[0].set_xlim([1e-4, 1e-1])

ax5[1].grid()
ax5[1].set_xlabel(r"$Ri_b, Ri_f$")
ax5[1].set_xlim([-0.1, 5])

# save figure
fig5.savefig(f"{fdir_save}{s_all[0].stab}_N2_S2_Ri.pdf", format="pdf", bbox_inches="tight")
plt.close(fig5)

#
# Figure 6: TKE Budget terms: ONLY LAST IN LIST
#
fig6, ax6 = plt.subplots(1, figsize=(12,8))

s = s_all[-1]

ax6.plot(s.tke["shear"][1:]/s.tke["scale"], s.tke["z"][1:], label="Shear Production")
ax6.plot(s.tke["buoy"][1:]/s.tke["scale"], s.tke["z"][1:], label="Buoyancy Production")
ax6.plot(s.tke["trans"][1:]/s.tke["scale"], s.tke["z"][1:], label="Turbulent Transport")
ax6.plot(s.tke["diss"][1:]/s.tke["scale"], s.tke["z"][1:], label="3D Dissipation")
ax6.plot(s.tke["residual"][1:]/s.tke["scale"], s.tke["z"][1:], label="Residual")
ax6.axhline(s.h, color="k", linestyle="--", label="h")
ax6.axhline(s.xytavg["zj"], color="k", linestyle=":", label="LLJ")
ax6.grid()
ax6.legend(loc="upper right")
ax6.set_xlabel("Dimensionless TKE Budget Terms [-]")
ax6.set_ylabel("z [m]")
ax6.set_title("TKE Budget (z-direction)")
ax6.set_ylim([0., 200.])
ax6.set_xlim([-5., 5.])

# save figure
fig6.savefig(f"{fdir_save}{s.stab}{s.lab}_tke_budget.pdf", format="pdf", bbox_inches="tight")
plt.close(fig6)

#
# Figure 7: Ozmidov length scale Lo
#
fig7, ax7 = plt.subplots(1, figsize=(8, 6))
for i, s in enumerate(s_all):
    ax7.plot(s.Ri["Lo"], s.z[1:-1], "-", label=f"$L_o^{{{s.nx}}}$", c=colors[i])
    ax7.axvline(800./(s.nx-1.), linestyle="--", color=colors[i], label=f"$\Delta^{{{s.nx}}}$")
#     ax7.axhline(s.xytavg["zj"], linestyle=":", color=colors[i], label=f"LLJ$^{{{s.nx}}}$")
    ax7.axhline(s.h, linestyle=":", color=colors[i], label=f"h$^{{{s.nx}}}$")
ax7.grid()
ax7.legend()
ax7.set_ylabel("z [m]")
ax7.set_xlabel(r"$L_o = \sqrt{ \langle \epsilon \rangle / \langle N^2 \rangle ^{3/2} }$ [m]")

# save figure
fig7.savefig(f"{fdir_save}{s_all[0].stab}_Lo.pdf", format="pdf", bbox_inches="tight")