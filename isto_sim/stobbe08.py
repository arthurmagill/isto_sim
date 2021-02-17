# -*- coding: utf-8 -*-
"""

This demonstration uses isto_sim to reproduce figures 1 and 2 from:

Sodium Imaging Optimization Under Specific Absorption Rate Constraint
Robert Stobbe and Christian Beaulieu
Magnetic Resonance in Medicine 59:345–355 (2008)
https://doi.org/10.1002/mrm.21468

Arthur W. Magill, 2021 (arthurmagill@gmail.com)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""


from isto_sim import SpinSystem
import numpy as np
import pylab as plt
from copy import copy

cmap = copy(plt.cm.jet)
cmap.set_bad('w')
cmap.set_under('w')


# Minimum time between pulse and acquisition
ringdown = 128e-6

# Sequence repetitions to reach steady state
reps = 15


J_sal = [9.4,9.4,9.4]     # Saline
J_agar_006 = [300,31,18]  # From Stobbe 2014, 3% agar
J_agar_003 = [190,21,14]  # From Stobbe 2014, 6% agar
J_brain = [558,32,12]

J_ic = [2200, 28, 28]     # Intracellular
J_ec = [135, 11.5, 11.5]  # Extracellular
J_csf = J_sal             # Cerebrospinal fluid


def ss_exp(FA,t_pulse,TR):

    ss = SpinSystem(J_brain)
    
    #ss.wq_stats(0,736,21)
    ss.wq_stats(1,0,0)
    ss.set_acq(40e-6,1)
    ss.TR = TR

    for n in range(reps-1):
        ss.pulse(FA,0,t_pulse)
        ss.evolve(ringdown)
        ss.spoil()
        ss.evolve()

    ss.pulse(FA,0,t_pulse)
    ss.evolve(ringdown)
    sig = ss.acquire()

    return abs(sig[0])

#
# Reproduce fig 1
#

ss_xy = SpinSystem(J_brain)
ss_xy.wq_stats(1,0,0)
ss_xy.set_acq(40e-6,100)
ss_xy.TR = 1

ss_xy.pulse(np.pi/2,0,0.1e-3)
#ss.evolve(ringdown)
sig_xy = np.imag( ss_xy.acquire() )
time_xy = np.linspace(0,ss_xy.acq_time(),ss_xy.acq_pts)

ss_zz = SpinSystem(J_brain,Mz=True)
ss_zz.wq_stats(1,0,0)
ss_zz.set_acq(1500e-6,100)
ss_zz.TR = 1

ss_zz.pulse(np.pi/2,0,0.1e-3)
ss_zz.evolve(ringdown)
sig_zz = np.real( ss_zz.acquire() )
time_zz = np.linspace(0,ss_zz.acq_time(),ss_zz.acq_pts)


fig,ax = plt.subplots(1,2,figsize=(6, 3))

ax[0].plot(time_xy*1000, sig_xy)
ax[0].set_xlim(0,4)
ax[0].set_xticks([0,1,2,3,4])
ax[0].set_ylabel('Relative Mxy')

ax[1].plot(time_zz*1000, sig_zz)
ax[1].set_xlim(0,150)
ax[1].set_xticks([0,50,100,150])
ax[1].set_ylabel('Relative Mz')

for a in ax:
    a.set_ylim(0,1)
    a.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    a.set_xlabel('time /ms')
    
fig.tight_layout(h_pad=0.1)
fig.savefig('../figures/stobbe08_fig1.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)




#
# Reproduce fig 2
#

TR = np.linspace(10e-3,150e-3,50)
t_pulse = np.linspace(0.1e-3,4.0e-3,50)

FA = np.zeros((len(t_pulse),len(TR)))
Mxy = np.zeros(np.shape(FA))
Mxy_sarless = np.zeros(np.shape(FA))
relax = np.zeros(np.shape(FA))
rSNR = np.zeros(np.shape(FA))
rSAR = np.zeros(np.shape(FA))
rMxy = np.zeros(np.shape(FA))


t_pulse0 = 0.45e-3
TR0 = 150e-3
FA0 = np.pi/2

Mxy0 = ss_exp(FA0,t_pulse0,TR0)

x,y = np.meshgrid(np.array(TR)*1000,np.array(t_pulse)*1000)

for i in range(len(t_pulse)):
    for j in range(len(TR)):
        FA[i,j] = FA0 * np.sqrt( t_pulse[i]*TR[j]/(t_pulse0*TR0) )

rSNR[FA>np.pi/2] = -np.inf
rSAR[FA>np.pi/2] = -np.inf
rMxy[FA>np.pi/2] = -np.inf
relax[FA>np.pi/2] = -np.inf
FA[FA>np.pi/2] = -np.inf

for j in range(len(TR)):
    for i in range(len(t_pulse)):
        if FA[i,j] != -np.inf:
            Mxy[i,j] = ss_exp(FA[i,j],t_pulse[i],TR[j])
            Mxy_sarless[i,j] = ss_exp(FA[i,j],100e-9,TR[j])
            relax[i,j] = 100.0 * (1 - Mxy[i,j]/np.sin(FA[i,j]))
            rSNR[i,j] = 100.0 * (Mxy[i,j] / Mxy0 * np.sqrt(TR0/TR[j]))
            rSAR[i,j] = 100.0 * (1 - Mxy[i,j] / Mxy_sarless[i,j])
            rMxy[i,j] = 100.0 * Mxy[i,j] / Mxy0


fig,ax = plt.subplots(3,2,figsize=(6, 8))

ax[0,0].set_title('Flip angle /$^\circ$')
im = ax[0,0].pcolor(x,y,FA/np.pi*180,vmin=0,vmax=90,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[0,0])
cb.set_ticks(np.arange(0,91,10))
cb.solids.set_edgecolor("face")

ax[0,1].set_title('Relative Mxy /%')
im = ax[0,1].pcolor(x,y,rMxy,vmin=10,vmax=100,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[0,1])
cb.set_ticks(np.arange(10,101,10))
cb.solids.set_edgecolor("face")

ax[1,0].set_title('Relaxation weighting /%')
im = ax[1,0].pcolor(x,y,relax,vmin=0,vmax=80,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[1,0])
cb.set_ticks(np.arange(0,81,10))
cb.solids.set_edgecolor("face")

ax[1,1].set_title('SAR loss /%')
im = ax[1,1].pcolor(x,y,rSAR,vmin=0,vmax=30,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[1,1])
cb.set_ticks(np.arange(0,31,5))
cb.solids.set_edgecolor("face")

ax[2,0].set_title('Relative SNR')
im = ax[2,0].pcolor(x,y,rSNR,vmin=90,vmax=150,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[2,0])
cb.set_ticks(np.arange(90,151,10))
cb.solids.set_edgecolor("face")

for a in ax.flat:
    a.set_xlim(TR[0]*1000,TR[-1]*1000)
    a.set_xticks([10,50,100,150])
    a.set_xlabel('TR /ms')
    a.set_ylim(t_pulse[0]*1000,t_pulse[-1]*1000)
    a.set_ylabel('pulse length /ms')
    

ax[2,1].set_axis_off()

fig.tight_layout(h_pad=0.8,w_pad=0.2)
fig.savefig('../figures/stobbe08_fig2.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)

