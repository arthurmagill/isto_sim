# -*- coding: utf-8 -*-
"""

Use isto_sim to reproduce figure 2 from:

Exploring and enhancing relaxation-based sodium MRI contrast. 
Stobbe RW, Beaulieu C. 
MAGMA. 2014 Feb;27(1):21-33. 
https://doi.org/10.1007/s10334-013-0390-7

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
ringdown = 65e-6

# Sequence repetitions to reach steady state
reps = 20

J_saline = [9.4,9.4,9.4]
J_agar = [300,31,18]  # From Stobbe 2014, 6% agar
#J_agar = [190,21,14]  # From Stobbe 2014, 3% agar


def ss_exp(J,FA,t_pulse,TR):

    ss = SpinSystem(J)
    ss.set_acq(40e-6,1)
    ss.TR = TR

    for n in range(reps):
        if n%2:
            ss.pulse(FA,0,t_pulse)
        else:
            ss.pulse(FA,np.pi,t_pulse)
        
        ss.evolve(ringdown)
        if n==reps-1:
            sig = ss.acquire()

        ss.evolve()

    return abs(sig[0])


def ss_exp_converge(J,FA,t_pulse,TR,conv):

    ss = SpinSystem(J)
    ss.set_acq(40e-6,1)
    ss.TR = TR
    state = ss.ss

    n = 0
    err = np.inf

    while err>conv and n<100:
        if n%2:
            ss.pulse(FA,0,t_pulse)
        else:
            ss.pulse(FA,np.pi,t_pulse)
            prev_state,state = state,np.mean(ss.ss,axis=(1,2))
        
        # Following Kharrazian
        err = np.linalg.norm(prev_state - state) / np.linalg.norm(state)

        ss.evolve(ringdown)
        if err<conv:
            sig = ss.acquire()

        ss.evolve()
        n += 1
        
        #print('%d\t%.3f\t%.3f\t%.3f' % (n,err,np.linalg.norm(prev_state),np.linalg.norm(state)))

    return abs(sig[0]),n


#
# Reproduce fig 2
#

PTS = 19

TR = np.linspace(11e-3,100e-3,PTS)
t_pulse = np.linspace(0.5e-3,4.0e-3,PTS)

#TR += (TR[1]-TR[0])
#t_pulse += (t_pulse[1]-t_pulse[0])

FA = np.zeros((len(t_pulse),len(TR)))
Mxy_agar = np.zeros( np.shape(FA) )
Mxy_saline = np.zeros( np.shape(FA) )

steps_agar = np.zeros( np.shape(FA) )
steps_saline = np.zeros( np.shape(FA) )

##t_pulse0 = 0.38e-3
##TR0 = 100e-3
#t_pulse0 = 3.75e-3
#TR0 = 16.7e-3
#FA0 = pi/2

FA0 = np.pi/2
t_pulse0 = 0.63e-3
TR0 = 100e-3
max_delta = 1e-4

Mxy0_agar = ss_exp_converge(J_agar,FA0,t_pulse0,TR0,max_delta)[0]
Mxy0_saline = ss_exp_converge(J_saline,FA0,t_pulse0,TR0,max_delta)[0]

x,y = np.meshgrid( np.array(TR)*1000,np.array(t_pulse)*1000 )

FA = FA0 * np.sqrt( np.outer(t_pulse,TR)/(t_pulse0*TR0) )

# flip angles above 180 don't make much sense, so mask them out
FA[FA>np.pi] = -np.inf

for j in range(len(TR)):
    for i in range(len(t_pulse)):
        if FA[i,j] != -np.inf:
            Mxy_agar[i,j],steps_agar[i,j] = ss_exp_converge(J_agar,FA[i,j],t_pulse[i],TR[j],max_delta)
            Mxy_saline[i,j],steps_saline[i,j] = ss_exp_converge(J_saline,FA[i,j],t_pulse[i],TR[j],max_delta)
            
            
rSNR_agar = Mxy_agar / Mxy0_agar * np.sqrt(TR0/TR[np.newaxis,:])
rSNR_saline = Mxy_saline / Mxy0_saline * np.sqrt(TR0/TR[np.newaxis,:])
# You don't want warnings about invalid values, so I'll hide them
with np.errstate(invalid='ignore'):
    rSI = Mxy_saline / Mxy_agar
rCNR = -(Mxy_saline-Mxy_agar)/(Mxy0_saline-Mxy0_agar)*np.sqrt(TR0/TR)

rSNR_agar[np.isinf(FA)] = -np.inf
rSNR_saline[np.isinf(FA)] = -np.inf
rSI[np.isinf(FA)] = -np.inf
rCNR[np.isinf(FA)] = -np.inf



fig,ax = plt.subplots(3,2,figsize=(6, 8))

ax[0,0].set_title('Flip angle /$^\circ$')
im = ax[0,0].pcolor(x,y,FA/np.pi*180,vmin=0,vmax=180,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[0,0])
cb.set_ticks(np.arange(0,181,50))
cb.solids.set_edgecolor("face")

ax[1,0].set_title('Relative SNR (agar)')
im = ax[1,0].pcolor(x,y,rSNR_agar,vmin=0,vmax=2,linewidth=0,rasterized=True,cmap=cmap)
cb = fig.colorbar(im,ax=ax[1,0])
cb.set_ticks(np.arange(0,2.1,0.5))
cb.solids.set_edgecolor("face")

ax[1,1].set_title('Relative SNR (saline)')
im = ax[1,1].pcolor(x,y,rSNR_saline,vmin=0,vmax=2,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[1,1])
cb.set_ticks(np.arange(0,2.1,0.5))
cb.solids.set_edgecolor("face")

ax[2,0].set_title('Relative SI (saline/agar)')
im = ax[2,0].pcolor(x,y,rSI,vmin=0.7,vmax=1.4,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[2,0])
cb.set_ticks([0.8,1.0,1.2,1.4])
cb.solids.set_edgecolor("face")

ax[2,1] = plt.subplot(3,2,6)
ax[2,1].set_title('Relative CNR')
im = ax[2,1].pcolor(x,y,rCNR,vmin=-7,vmax=13,linewidth=0,rasterized=True,cmap=cmap)
cb = plt.colorbar(im,ax=ax[2,1])
cb.set_ticks(np.arange(-5,15,5))
cb.solids.set_edgecolor("face")

for a in ax.flat:
    a.set_xlim(TR[0]*1000,TR[-1]*1000)
    a.set_ylim(t_pulse[0]*1000,t_pulse[-1]*1000)
    a.set_xticks(np.arange(20,101,20))

ax[0,0].set_ylabel('pulse length /ms')
ax[1,0].set_ylabel('pulse length /ms')
ax[2,0].set_ylabel('pulse length /ms')
ax[2,0].set_xlabel('TR /ms')
ax[2,1].set_xlabel('TR /ms')

ax[0,1].set_axis_off()

fig.tight_layout(h_pad=1.0,w_pad=0.3)
fig.savefig('../figures/stobbe14_fig2.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)



#fig,ax = plt.subplots(1,2,figsize=(6, 3))
#
#ax[0].set_title('Convergence (saline)')
#im = ax[0].pcolor(x,y,steps_saline,vmin=5,vmax=40,linewidth=0,rasterized=True,cmap=cmap)
#cb = plt.colorbar(im,ax=ax[0])
#cb.set_ticks(np.arange(5,41,5))
#cb.solids.set_edgecolor("face")
#
#ax[1].set_title('Convergence (agar)')
#im = ax[1].pcolor(x,y,steps_agar,vmin=5,vmax=40,linewidth=0,rasterized=True,cmap=cmap)
#cb = plt.colorbar(im,ax=ax[1])
#cb.set_ticks(np.arange(5,41,5))
#cb.solids.set_edgecolor("face")
#
#for a in ax:
#    a.set_xlim(TR[0]*1000,TR[-1]*1000)
#    a.set_ylim(t_pulse[0]*1000,t_pulse[-1]*1000)
#    a.set_xticks(np.arange(20,101,20))
#    a.set_xlabel('TR /ms')
#    a.set_ylabel('pulse length /ms')
#
#fig.tight_layout(h_pad=0.3,w_pad=0.3)
#fig.savefig('../figures/stobbe14_fig2_converge_1e-4.pdf',
#            bbox_inches='tight', 
#            pad_inches=0.05, 
#            dpi=300)
