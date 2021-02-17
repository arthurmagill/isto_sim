# -*- coding: utf-8 -*-
"""

Simulate a simple pulse-acquire sequence using the isto_sim module.

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


J_sal = [9.4,9.4,9.4]     # Saline
J_agar_006 = [300,31,18]  # From Stobbe 2014, 6% agar
J_agar_003 = [190,21,14]  # From Stobbe 2014, 6% agar
J_brain = [558,32,12]

J_ic = [2200, 28, 28]     # Intracellular compartment
J_ec = [135, 11.5, 11.5]  # Extracellular compartment
J_csf = J_sal             # Cerebrospinal fluid



FA = np.linspace(0,360,100)/180.0*pi
res = np.zeros(len(FA))


ss = SpinSystem(J_sal,Mz=False)
ss.set_acq(0.1e-3,1)
ss.TR = 1000e-3

fig,ax = plt.subplots(2,1,figsize=(6,6))

for t_pulse in linspace(0.5e-3,4.5e-3,5):
    for n in range(len(FA)):
        ss.reset()
        ss.pulse(FA[n],pi/2,t_pulse)
        ss.evolve(128e-6)
        res[n] = real(ss.acquire()[0])
        ss.evolve()
    
    ax[0].plot(FA/pi*180,res,label='%.1f' % (t_pulse*1000))
    

ss = SpinSystem(J_sal,Mz=True)
ss.set_acq(0.1e-3,1)
ss.TR = 1000e-3

for t_pulse in linspace(0.5e-3,4.5e-3,5):
    for n in range(len(FA)):
        ss.reset()
        ss.pulse(FA[n],pi/2,t_pulse)
        ss.evolve(128e-6)
        res[n] = real(ss.acquire()[0])
        ss.evolve()
    
    ax[1].plot(FA/pi*180,res,label='%.1f' % (t_pulse*1000))
    

for a in ax:
    a.set_xlim(0,360)
    a.set_ylim(-1.1,1.1)
    a.set_xticks([0,90,180,270,360])

ax[0].set_ylabel('M_xy')
ax[1].set_xlabel('Flip angle /degree')
ax[1].set_ylabel('M_z')
ax[1].legend(title='t_pulse /ms',loc='lower right',frameon=False)

fig.tight_layout()
fig.savefig('../figures/pulse_acq.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)
