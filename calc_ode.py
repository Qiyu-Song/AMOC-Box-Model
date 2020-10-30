"""
Created on Tue Oct/27/2020
@author: Xinyu Wen, xwen@pku.edu.cn

The following parts are edited by Qiyu Song (sqy2017@pku.edu.cn) on Fri Oct/30/2020:
    1. Integrate code for mutiple schemes choice;
    2. Calculation code for forward Euler, leapfrog, and AB2
    3. Plotting function for time series of variables
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ode:
    def __init__(self, ic=None, tendfunc=None, dt=0.01, steps=1000, scheme="", debug=0):
        '''
        ic       : Initial condition
        tendfunc : Tendency function: feedin X, return F(X), in type "np.array"
        dt       : Time step (sec)
        steps    : Number of steps to integrate
        debug    : Debug level (0=None, 1=Basic, 2=InDetails)
        '''
        try:
            if len(ic) == np.size(tendfunc(ic)):
                self.dim = len(ic)
        except:
            print(len(ic))
            print(np.size(tendfunc(ic)))
            print("IC does not match TendFunc! Quit!")
            raise SystemExit

        self.F      = tendfunc
        self.dt     = dt
        self.steps  = steps + 1
        self.scheme = scheme
        self.debug  = debug

        self.traj = np.ndarray(shape=(self.steps, self.dim), dtype=float)
        self.traj[0, :] = np.array(ic, dtype=float)

        if self.debug >= 1:
            print('Initialization Finished. Initial Value = ', ic)

    def Integrate(self):
        if self.debug >= 1: print("Start integrating ... scheme =", self.scheme)
        if self.scheme == 'leapfrog' or self.scheme == 'ab2':
            print('use Forward Euler to calculate 1st step')
            self.traj[1, :] = self.forward(self.traj[0, :])
            if self.scheme == 'ab2': pre_F = self.F(self.traj[0, :])
            i = 2
            while i < self.steps:
                if self.scheme == 'ab2':
                    self.traj[i, :], pre_F = self.ab2(self.traj[i - 1, :], pre_F)
                elif self.scheme == 'leapfrog':
                    self.traj[i, :] = self.leapfrog(self.traj[i-2:i, :])
                if self.debug >= 2:
                    print(i, self.traj[i, :])
                i = i + 1
        else:
            i = 1
            while i < self.steps:
                if self.scheme == 'forward':
                    self.traj[i, :] = self.forward(self.traj[i - 1, :])
                elif self.scheme == "rk4":
                    self.traj[i, :] = self.rk4(self.traj[i - 1, :])
                if self.debug >= 2:
                    print(i, self.traj[i, :])
                i = i + 1
        if self.debug >= 1: print('Stop integrating ... total steps =', self.steps - 1, '\nIntegration Finished. Final Value = ', list(self.traj[-1, :]))

    def rk4(self, X):
        # Runge-Kutta 4th order
        q1 = self.dt * self.F(X)
        q2 = self.dt * self.F(X + q1 / 2)
        q3 = self.dt * self.F(X + q2 / 2)
        q4 = self.dt * self.F(X + q3)
        Xnew = X + (q1 + 2 * q2 + 2 * q3 + q4) / 6
        return Xnew

    def forward(self, X):
        # Forward Euler
        Xnew = X + self.dt * self.F(X)
        return Xnew

    def leapfrog(self, X):
        # Leapfrog (order = 2)
        Xnew = X[0, :] + 2 * self.dt * self.F(X[1, :])
        return Xnew

    def ab2(self, X, pre_F):
        # Adams - Bashforth (order = 2)
        Fnew = self.F(X)
        Xnew = X + 0.5 * self.dt * (3 * Fnew - pre_F)
        return Xnew, Fnew



###################################
##  Other Functions for Plotting ##
###################################

def plot3d(data,style="r-",lw=1.0,ti="Plot",xl="X",yl="Y",zl="Z",fn="plot3d.pdf"):
    x,y,z = data[:,0],data[:,1],data[:,2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set(title=ti,xlabel=xl,ylabel=yl,zlabel=zl)
    ax.plot(x,y,z,style,linewidth=lw)
    plt.savefig(fn)
    plt.show()
    plt.close()


def plot3dcomp2(data1,data2,lb1='L1',lb2='L2',lw=1.0,ti="Plot",xl="X",yl="Y",zl="Z",fn="plot3d.pdf"):
    x1,y1,z1 = data1[:,0],data1[:,1],data1[:,2]
    x2,y2,z2 = data2[:,0],data2[:,1],data2[:,2]
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.gca(projection='3d')
    ax.set_title(ti, fontsize=20)
    ax.set(xlabel=xl,ylabel=yl,zlabel=zl)
    ax.plot(x1,y1,z1,'r-',linewidth=lw,label=lb1)
    ax.plot(x2,y2,z2,'b-',linewidth=lw,label=lb2)
    plt.legend(fontsize=10)
    plt.savefig(fn)
    plt.show()
    plt.close()

def plotts(time,datas=[],lbs=[],lw=1.0,ti="TS",xl="Time",yls=[],fn="plotts.pdf"):
    try:
        if len(lbs) == np.size(datas[0, 0, :]): pass
    except:
        print(str(len(lbs))+str(np.size(datas[0, 0, :]))+"\nLabel number does not match variable number! Quit!")
        raise SystemExit
    Nvar = len(yls)

    fig = plt.figure(figsize=(8, 18), dpi=300)
    colors = ['k', 'b', 'r', 'orange']
    for nvar in range(Nvar):
        ax = fig.add_subplot(3, 1, nvar+1)
        ax.set(ylabel=yls[nvar])
        for nline in range(np.size(datas[0, 0, :])):
            ax.plot(time, datas[:, nvar, nline], '-', color=colors[nline], linewidth=lw, label=lbs[nline])
        ax.legend(fontsize=10, loc='upper right')
    ax.set(xlabel=xl)

    plt.suptitle(ti, fontsize=24)
    plt.savefig(fn)
    plt.show()
    plt.close()

# Main
if __name__=='__main__':
    print("Hello!")
