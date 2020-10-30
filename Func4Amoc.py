"""
Created on Fri Oct/30/2020
@author: Qiyu Song, sqy2017@pku.edu.cn
"""

import numpy as np
import math as mt
import calc_ode as ode
from Constants import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# For Atmosphere
def r(sol, alpha, t):  # Radiation
    # SW = S * (1-alpha)
    # LW = A + B * Ta
    return sol * (1-alpha) - 213.35 - 2.22*t


def h(q1, q2, to, ta):  # Ocean-Atmosphere Heat Flux
    # H = Q1 - Q2 * (To - Ta)
    return q1 - q2*(to-ta)


def fs(ta_s=0, ta_n=0, deltay=0):  # Meridional Sensible Heat Flux
    # Fs = Ks * \Delta T_a / \Delta y
    return 1.0e13 * 2.5 * abs(ta_n-ta_s) / deltay  # always positive value, polewards


def tc(t_s, t_n, lat_s, lat_n, lat_b):  # Temperature at Boundary
    # linear interpolation
    return np.average([t_s, t_n], weights=[lat_n-lat_b, lat_b-lat_s])


def sat(t):  # Saturation Water Vapor
    return 6.112 * mt.exp(17.67 * t / (t+243.5))


def dqsdt(tc):  # Difference in Saturation Water Vapor
    return 243.5 * 17.67 * 0.622 * sat(tc) / 1000 / ((tc+243.5) ** 2)


def fl(dqsdt=0, deltay=0):  # Meridional Latent Heat Flux
    # Fl = Kl * RH * dQSdT / \Delta y
    return 5.1e17 * 1.5 * 0.8 * dqsdt / deltay  # always positive value, polewards


def dfdy(f_s=0., f_n=0., boundary_s=0., boundary_n=0.):  # Contribution of Meridional Flux to Total Box
    # \Delta F / \Delta y
    return (f_s * mt.cos(boundary_s) + f_n * mt.cos(boundary_n)) / 6.371e6 / (mt.sin(boundary_n) - mt.sin(boundary_s))


# For Ocean
def phi(so_s=0, so_n=0, to_s=0, to_n=0):  # AMOC Strength
    return 1.5264e10 * (8.0e-4 * (so_n - so_s) - 1.5e-4 * (to_n - to_s))


def pe_S(fl_S=0.):  # Precipitation - Evaporation, in South Box
    # Associated with Latent Heat Flux
    return 2*mt.pi*6.371e6 * Lr * mt.sqrt(3)/2 * fl_S * 80/360


def pe_N(fl_N=0.):  # Precipitation - Evaporation, in North Box
    # Associated with Latent Heat Flux
    return 2*mt.pi*6.371e6 * Lr * mt.sqrt(2)/2 * fl_N * 2.5*80/360


def Equations(X):  # Calculate Tendency
    # Initialize
    To_S, To_M, To_N, To_D, So_S, So_M, So_N, So_D, Ta_S, Ta_M, Ta_N = X
    tend = np.zeros(11)

    # Atmosphere
    # Radiation
    R = r(Sol, alpha, np.array([Ta_S, Ta_M, Ta_N]))
    # Ocean-Atmosphere Heat Flux
    H = h(Q1, Q2, np.array([To_S, To_M, To_N]), np.array([Ta_S, Ta_M, Ta_N]))
    # Calculate Meridional Heat Flux
    Fs_S = fs(Ta_S, Ta_M, deltay_SM)
    Fs_N = fs(Ta_M, Ta_N, deltay_MN)
    Tc_S = tc(Ta_S, Ta_M, MassLat[0], MassLat[1], Boudary_n[0])
    Tc_N = tc(Ta_M, Ta_N, MassLat[1], MassLat[2], Boudary_n[1])
    dQSdT_S = dqsdt(Tc_S)
    dQSdT_N = dqsdt(Tc_N)
    Fl_S = fl(dQSdT_S, deltay_SM)
    Fl_N = fl(dQSdT_N, deltay_MN)
    F_S = Fs_S+Fl_S
    F_N = Fs_N+Fl_N
    # Tendency of Atmosphere Temperature: S,M,N
    tend[8] = (dfdy(0, F_S, Boudary_s[0], Boudary_n[0]) + R[0] - SurfFrac[0] * H[0]) / c
    tend[9] = (dfdy(-F_S, -F_N, Boudary_s[1], Boudary_n[1]) + R[1] - SurfFrac[1] * H[1]) / c
    tend[10] = (dfdy(F_N, 0, Boudary_s[2], Boudary_n[2]) + R[2] - SurfFrac[2] * H[2]) / c

    # Ocean
    # AMOC Strength
    Phi = max(phi(So_S, So_N, To_S, To_N), 0.)
    # Tendency of Ocean Temperature: S,M,N,D
    tend[0] = -(To_S - To_D) * Phi / V[0] + H[0] / rho_cp / Depth[0]
    tend[1] = -(To_M - To_S) * Phi / V[1] + H[1] / rho_cp / Depth[1]
    tend[2] = -(To_N - To_M) * Phi / V[2] + H[2] / rho_cp / Depth[2]
    tend[3] = -(To_D - To_N) * Phi / V[3]
    # Water Budget
    PE_S = pe_S(Fl_S)
    PE_N = pe_N(Fl_N)
    PE_M = PE_S + PE_N  # Maintain Global Water Constant
    # Tendency of Ocean Salinity: S,M,N,D
    tend[4] = -(So_S - So_D) * Phi / V[0] - 34.9 * PE_S / V[0]
    tend[5] = -(So_M - So_S) * Phi / V[1] + 34.9 * PE_M / V[1]
    tend[6] = -(So_N - So_M) * Phi / V[2] - 34.9 * PE_N / V[2]
    tend[7] = -(So_D - So_N) * Phi / V[3]

    return tend

def plot_time_evolution(time, data, fn):
    lbs = ['S', 'M', 'N', 'D']
    fig = plt.figure(figsize=(10, 8), dpi=300)

    # Ocean Temperature (K)
    ax = plt.subplot(2, 2, 1)
    ax.set(ylabel='Ocean Temperature (K)')
    for nline in range(4):
        ax.plot(time, data[:, nline], '-', linewidth=1.0, label=lbs[nline])
    ax.legend(fontsize=12, loc='upper right')
    ax.set(xlabel='Time (a)')

    # Ocean Salinity (PSU)
    ax = plt.subplot(2, 2, 2)
    ax.set(ylabel='Salinity (PSU)')
    for nline in range(4):
        ax.plot(time, data[:, nline + 4], '-', linewidth=1.0, label=lbs[nline])
    ax.legend(fontsize=12, loc='upper right')
    ax.set(xlabel='Time (a)')

    # Atmosphere Temperature (K)
    ax = plt.subplot(2, 2, 3)
    ax.set(ylabel='Atmosphere Temperature (K)')
    for nline in range(3):
        ax.plot(time, data[:, nline + 8], '-', linewidth=1.0, label=lbs[nline])
    # Global Mean
    ax.plot(time, (0.5 * data[:, 8] + 1.207 * data[:, 9] + 0.293 * data[:, 10]) / 2, '-', linewidth=1.0, label='G')
    ax.legend(fontsize=12, loc='upper right')
    ax.set(xlabel='Time (a)')

    # Phi of AMOC
    ax = plt.subplot(2, 2, 4)
    ax.set(ylabel='Phi * 1e-6 (Sv)')
    ax.plot(time, 1e-6 * phi(data[:, 4], data[:, 6], data[:, 0], data[:, 2]), '-', linewidth=1.0)
    # ax.plot(time, 0*time, '-', linewidth=1.0)
    ax.set(xlabel='Time (a)')

    fig.tight_layout()
    plt.savefig(fn)

if __name__ == "__main__":
    # Initial run: n years
    nyear = 5000  # should no less than 3000 to ensure equilibrium
    ic = [4.777404031, 24.42876625, 2.66810894, 2.67598915, 34.40753555, 35.62585068, 34.92513657, 34.91130066, 4.67439556, 23.30437851, 0.94061828]
    a = ode.ode(ic, Equations, 0.01*365*86400, nyear*100, "rk4", 1)
    a.Integrate()
    time1 = np.arange(nyear+1)
    data1_year = a.traj[0::100, :]
    del a
    np.savetxt('./Amoc_data_0_' + str(nyear) + '.txt', data1_year, fmt='%f', delimiter='\t')

    # Add fresh water: extra years
    extra_year = 3000  # should no less than 3000 to ensure equilibrium
    ic = data1_year[-1, :]
    ic[6] -= 0.7
    a = ode.ode(ic, Equations, 0.01*365*86400, extra_year*100, "rk4", 1)
    a.Integrate()
    time2 = np.arange(extra_year + 1) + nyear
    data2_year = a.traj[0::100, :]
    del a
    np.savetxt('./Amoc_data_' + str(nyear) + '_' + str(nyear+extra_year) + '.txt', data2_year, fmt='%f', delimiter='\t')

    # Combine data
    data_year = np.concatenate((data1_year, data2_year[1:, :]), axis=0)
    np.savetxt('./Amoc_data_0_' + str(nyear+extra_year) + '.txt', data_year, fmt='%f', delimiter='\t')
    time = np.concatenate((time1, time2[1:]))

    # Plot
    plot_time_evolution(time1[:3001], data1_year[:3001, :], 'plot_Amoc_ts_0_3000.pdf')
    plot_time_evolution(time2[:3001], data2_year[:3001, :], 'plot_Amoc_ts_'+str(nyear)+'_'+str(nyear+3000)+'.pdf')
    plot_time_evolution(time, data_year, 'plot_Amoc_ts_0_'+str(nyear+extra_year)+'.pdf')


