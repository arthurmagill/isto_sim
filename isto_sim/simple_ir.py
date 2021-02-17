# -*- coding: utf-8 -*-
"""

Simulate a simple inversion recovery sequence using the isto_sim module.

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

# Minimum time between pulse and acquisition
ringdown = 128e-6

# Sequence repetitions to reach steady state
reps = 5

# Points to acquire
acq_pts = 100

J_sal = [9.4,9.4,9.4]     # Saline
J_agar_006 = [300,31,18]  # From Stobbe 2014, 3% agar
J_agar_003 = [190,21,14]  # From Stobbe 2014, 6% agar
J_brain = [558,32,12]

J_ic = [2200, 28, 28]     # Intracellular
J_ec = [135, 11.5, 11.5]  # Extracellular
J_csf = J_sal             # CSF



#
# Simple inversion recovery
#

# First simulate the transverse magnetisation

ss = SpinSystem(J_brain)
ss.wq_stats(1,0,0)
ss.set_acq(40e-6,acq_pts)
ss.TR = 1

ss.pulse(np.pi,0,0.1e-3)
#ss.evolve(ringdown)
sig_xy = ss.acquire()
time_xy = np.linspace(0,ss.acq_time(),ss.acq_pts)

# And then the longitudinal magnetisation

ss = SpinSystem(J_brain,Mz=True)
ss.wq_stats(1,0,0)
ss.set_acq(1500e-6,acq_pts)
ss.TR = 1

ss.pulse(np.pi,0,0.1e-3)
ss.evolve(ringdown)
sig_z = ss.acquire()
time_z = np.linspace(0,ss.acq_time(),ss.acq_pts)



fig,ax = plt.subplots(1,2,figsize=(6,3))

ax[0].plot(time_xy*1000, np.imag(sig_xy))
ax[0].set_xlim(0,4)
ax[0].set_xticks([0,1,2,3,4])
ax[0].set_ylim(-0.1,1.1)
ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax[0].set_xlabel('time /ms')
ax[0].set_ylabel('Relative Mxy')

ax[1].plot(time_z*1000, np.real(sig_z))
ax[1].set_xlim(0,150)
ax[1].set_xticks([0,50,100,150])
ax[1].set_ylim(-1.1,1.1)
#ax[1].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax[1].set_yticks(np.arange(-1,1.1,0.5))
ax[1].set_xlabel('time /ms')
ax[1].set_ylabel('Relative Mz')

fig.tight_layout(h_pad=0.1)
fig.savefig('../figures/simple_ir.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)
