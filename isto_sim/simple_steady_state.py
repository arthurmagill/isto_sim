# -*- coding: utf-8 -*-
"""

Simulate convergence to steady state with a simple pulse-acquire sequence using
the isto_sim module.

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


#
# Simple FID, repeated reps times
#

FA = np.pi/4    # Flip angle
t_pulse = 4e-3  # Pulse length
TR = 10e-3      # Repetition time

reps = 10       # Number of repetitions

# Build two spin systems, one to monitor transverse magnetisation, and a second
# for the longitudinal magnetisation.
ss_xy = SpinSystem(J_brain)
ss_xy.set_acq(40e-6,100)
ss_xy.TR = TR

ss_zz = SpinSystem(J_brain,Mz=True)
ss_zz.set_acq(40e-6,100)
ss_zz.TR = TR

time = np.linspace(0,ss_zz.acq_time()*reps,ss_zz.acq_pts*reps)

sig_xy = np.array([])
sig_zz = np.array([])

for n in range(reps):    
    ss_xy.pulse(FA,0,t_pulse)
    sig_xy = np.append(sig_xy,ss_xy.acquire())
    ss_xy.evolve()

    ss_zz.pulse(FA,0,t_pulse)
    sig_zz = np.append(sig_zz,ss_zz.acquire())
    ss_zz.evolve()
    

fig,ax = plt.subplots(1,2,figsize=(6, 3))

ax[0].plot(time*1000,np.imag(sig_xy))
ax[1].plot(time*1000,np.real(sig_zz))

for a in ax:
    a.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    a.set_xlabel('time /ms')
    a.set_ylim(0,1)
    
ax[0].set_xlabel('time /ms')
ax[0].set_ylabel('Relative Mxy')
ax[1].set_ylabel('Relative Mz')

fig.tight_layout(h_pad=0.1)
fig.savefig('../figures/simple_steady_state.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)



