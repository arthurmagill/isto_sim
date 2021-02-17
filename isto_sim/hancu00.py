# -*- coding: utf-8 -*-
"""

Use isto_sim to reproduce fig. 3 from:

A model for the dynamics of spins 3/2 in biological media: signal loss during 
radiofrequency excitation in triple-quantum-filtered sodium MRI
I. Hancua, J. R. C. van der Maarelb and F. E. Boada
JMR 147, 2000, p179-191
https://doi.org/10.1006/jmre.2000.2177

The pulse sequence is pi/2--mix--pi/2--pi/2--acq with phases phi, phi+pi/2, 0, 
with a mixing time of 2.4ms 

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

# From Hancu et al., all in Hz
J = [185, 50.4, 50.4]

fig3 = plt.figure(3)
fig5 = plt.figure(5)

#for pulselength in [0.1e-3, 0.5e-3, 0.9e-3]:
for pulselength in np.linspace(0.1e-3,0.9e-3,9):

    ss = SpinSystem(J)
    # Hancu used 2200 points, but that takes a while to run
    ss.wq_stats(21,0,733)
    # Hancu used 4096 points and 16us dwell time - but it's a bit slow, 
    # so I've reduced the resolution
    ss.set_acq(4*16e-6,4096/4)
    ss.TR = 100e-3
    sig = 0

    phi = np.array([30,90,150,-150,-90,-30]) * np.pi / 180
    #phi = np.array([30]) * np.pi / 180

    for n in range(len(phi)):
        #ss.spoil()
        ss.pulse(np.pi/2,phi[n],pulselength)
        ss.evolve(2.4e-3)
        ss.pulse(np.pi/2,phi[n]+np.pi/2,pulselength)
        ss.evolve(20e-6)
        ss.pulse(np.pi/2,0,pulselength)
        ss.evolve(20e-6)
        # receiver phase cycling
        sig += (-1 if n % 2 else 1) * ss.acquire()
        #sig += ss.acquire()
        ss.evolve()

    tt = np.arange(0,len(sig)) * ss.t_dwell * 1000
    if pulselength in [0.1e-3,0.5e-3,0.9e-3]:
        plt.figure(3)
        plt.plot(tt,-np.real(sig),label='%0.1f' % (pulselength*1000))
    
    plt.figure(5)
    # 134 is fudge-factor to match scaling of original plot
    plt.plot(pulselength*1e6,np.sum(abs(sig))/134.0,'ok')

plt.figure(3)
plt.xlim(0,60)
plt.xlabel('time /ms')
plt.ylabel('TQ signal intensity /a.u.')
plt.legend(title='pulse length /ms',frameon=False)
plt.tight_layout(h_pad=0.2,w_pad=0.2)

plt.savefig('../figures/hancu00_fig3.pdf',
        bbox_inches='tight', 
        pad_inches=0.05, 
        dpi=300)

plt.figure(5)
plt.xlim(0,1000)
plt.ylim(0.5,1.0)
plt.xlabel('pulse width /ms')
plt.ylabel('TQ signal intensity /a.u.')
plt.tight_layout(h_pad=0.2,w_pad=0.2)

plt.savefig('../figures/hancu00_fig5.pdf',
        bbox_inches='tight', 
        pad_inches=0.05, 
        dpi=300)

