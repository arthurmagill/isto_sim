# -*- coding: utf-8 -*-
"""

Use isto_sim to reproduce figure 1 from:

In vivo sodium magnetic resonance imaging of the human brain using soft 
inversion recovery fluid attenuation. 
Stobbe, R. and Beaulieu, C. 
Magn. Reson. Med., 54: 1305-1310, 2005
https://doi.org/10.1002/mrm.20696

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


J_saline = [10.64,9.24,9.24] # From Stobbe 2005, 500mM Na+
J_agar = [625,50.8,30.4]  # From Stobbe 2005, 20% agar, 100mM Na+

#
# Reproduce fig 1
#

fig,ax = plt.subplots(2,2,figsize=(6, 6))
ax = ax.flat

for i,t_pulse in enumerate([0.5e-3, 1.0e-3, 5e-3, 10e-3]):

    ss_xy_saline = SpinSystem(J_saline)
    ss_xy_saline.set_acq(0,1)
    
    ss_zz_saline = SpinSystem(J_saline,Mz=True)
    ss_zz_saline.set_acq(0,1)

    ss_xy_agar = SpinSystem(J_agar)
    ss_xy_agar.set_acq(0,1)
    
    ss_zz_agar = SpinSystem(J_agar,Mz=True)
    ss_zz_agar.set_acq(0,1)

    # break each pulse into N sections    
    N = 100
    
    sig_xy_saline = np.zeros(N)
    sig_zz_saline = np.zeros(N)
    
    sig_xy_agar = np.zeros(N)
    sig_zz_agar = np.zeros(N)

    for n in range(N):
        ss_xy_saline.pulse(np.pi/N,0,t_pulse/N)
        sig_xy_saline[n] = np.imag(ss_xy_saline.acquire()[0])
    
        ss_zz_saline.pulse(np.pi/N,0,t_pulse/N)
        sig_zz_saline[n] = np.real(ss_zz_saline.acquire()[0])
    
        ss_xy_agar.pulse(np.pi/N,0,t_pulse/N)
        sig_xy_agar[n] = np.imag(ss_xy_agar.acquire()[0])
    
        ss_zz_agar.pulse(np.pi/N,0,t_pulse/N)
        sig_zz_agar[n] = np.real(ss_zz_agar.acquire()[0])
    

    time = np.linspace(0,t_pulse*1000,N)
    ax[i].plot(time,sig_xy_saline,linestyle='-',color='b',label='$M_{xy}$, Saline')
    ax[i].plot(time,sig_xy_agar,linestyle='--',color='b',label='$M_{xy}$, 20% Agar')
    ax[i].plot(time,sig_zz_saline,linestyle='-',color='g',label='$M_z$, Saline')
    ax[i].plot(time,sig_zz_agar,linestyle='--',color='g',label='$M_z$, 20% Agar')
    ax[i].set_xlim(0,t_pulse*1000)
    ax[i].set_xticks(np.linspace(0,t_pulse*1000,6))
    ax[i].set_yticks([-1,0,1])
    ax[i].set_xlabel('Time /ms')
    ax[i].set_ylabel('Relative Magnetisation')
    
    # Because we're paranoid, check that the sum of many small 
    # pulses has the same effect as one large one
    print('Pulse length = %.1f ms' % (t_pulse*1000))
    print('Mxy,Mz after lots of short pulses:\t%.3f, %.3f' % (sig_xy_agar[-1],sig_zz_agar[-1]))
    ss_xy_agar.reset()
    ss_xy_agar.pulse(np.pi,0,t_pulse)
    sig_xy_agar = np.imag( ss_xy_agar.acquire()[0] )
    ss_zz_agar.reset()
    ss_zz_agar.pulse(np.pi,0,t_pulse)
    sig_zz_agar = np.real( ss_zz_agar.acquire()[0] )
    print('Mxy,Mz after one long pulse:\t\t%.3f, %.3f' % (sig_xy_agar,sig_zz_agar))
    
ax[3].legend(loc='lower left',fontsize='small')
    

plt.tight_layout(h_pad=0.1)
plt.savefig('../figures/stobbe05_fig1.pdf',
            bbox_inches='tight', 
            pad_inches=0.05, 
            dpi=300)
