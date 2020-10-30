"""
Created on Fri Oct/30/2020
@author: Qiyu Song, sqy2017@pku.edu.cn
"""

import math as mt
import numpy as np

def degree2rad(degree):
    return degree / 180. * mt.pi


def masslat(boundary_s, boundary_n):
    return np.arcsin(0.5*(np.sin(boundary_s)+np.sin(boundary_n)))


def v(boundary_s, boundary_n, dz):
    return 2*mt.pi*(6.371e6**2) * abs(np.sin(boundary_n) - np.sin(boundary_s)) * 80/360 * dz

# Parameters and Constants
Re = 6.371e6
cpa = 1004.0
c = 5300 * cpa
rho_o = 1025
cpo = 4200
Boudary_s = np.array(list(map(degree2rad, [-90, -30, 45])))
Boudary_n = np.array(list(map(degree2rad, [-30, 45, 90])))
MassLat = masslat(Boudary_s, Boudary_n)
deltay_SM = Re * (MassLat[1] - MassLat[0])
deltay_MN = Re * (MassLat[2] - MassLat[1])
Boudaryo_s = np.array(list(map(degree2rad, [-60, -30, 45, -30])))
Boudaryo_n = np.array(list(map(degree2rad, [-30, 45, 80, 45])))
Depth = np.array([4000, 600, 4000, 3400])
V = v(Boudaryo_s, Boudaryo_n, Depth)
Sol = np.array([320, 390, 270])
alpha = np.array([0.4, 0.25, 0.42])
SurfFrac = np.array([0.16267, 0.22222, 0.21069])
Q1 = np.array([10, 70, 20])
Q2 = np.array([50, 50, 50])

Phis = []