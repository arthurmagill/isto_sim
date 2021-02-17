# -*- coding: utf-8 -*-
"""

Simulate a simple FID sequence using the isto_sim module.

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


J_sal = [9.4,9.4,9.4]     # From Stobbe 2014
J_brain = [558,32,12]     # From Stobbe 2008

# RF pulse durations
t_pulse = [0.1e-3, 1e-3, 10e-3]

# Points in readout
acq_pts = 100

sig_xy = np.zeros((len(t_pulse),acq_pts))
sig_z = np.zeros((len(t_pulse),acq_pts))

#
# Simple FID
#

fig,ax = plt.subplots(1,2,figsize=(6, 3))

for n in range(len(t_pulse)):
    # Simulate transverse magnetisation
    ss = SpinSystem(J_brain)
    ss.set_acq(40e-6,acq_pts)
    
    ss.pulse(pi/2,0,t_pulse[n])
    sig_xy[n] = np.imag( ss.acquire() )
    
    # and longitudinal part
    ss = SpinSystem(J_brain,Mz=True)
    ss.set_acq(1500e-6,acq_pts)
    
    ss.pulse(pi/2,0,t_pulse[n])
    sig_z[n] = np.real( ss.acquire() )
    


for n in range(len(t_pulse)):
    ax[0].plot(np.linspace(0,ss.acq_time(),acq_pts)*1000, sig_xy[n],label='%.1f' % (t_pulse[n]*1000))
    ax[1].plot(np.linspace(0,ss.acq_time(),acq_pts)*1000, sig_z[n],label='%.1f' % (t_pulse[n]*1000))


ax[0].set_xlim(0,4)
ax[0].set_xticks([0,1,2,3,4])
ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax[0].set_xlabel('time /ms')
ax[0].set_ylabel('Relative Mxy')

ax[1].set_xlim(0,150)
ax[1].set_xticks([0,50,100,150])
ax[1].set_ylim(0,1)
ax[1].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax[1].set_xlabel('time /ms')
ax[1].set_ylabel('Relative Mz')
ax[1].legend(loc='lower right',title='t_pulse /ms',frameon=False)

fig.tight_layout(h_pad=0.1)
fig.savefig('../figures/simple_fid.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)



