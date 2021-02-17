#!/usr/bin/env python
""" Simulate a quadrupolar spin system using irreducible spherical tensor 
operators.

Simulate quadrupolar spin dynamics using irreducible spherical tensor operators
(ISTOs). This is useful for modelling nuclei with half-integer spin greater 
than 1/2 (e.g. sodium, oxygen, etc), particularly if you're interested in 
multiple quantum coherences.

The code implements the model described here:
    
A model for the dynamics of spins 3/2 in biological media: signal loss 
during radiofrequency excitation in triple-quantum-filtered sodium MRI
I. Hancua, J. R. C. van der Maarelb and F. E. Boada
JMR 147, 2000, p179-191
https://doi.org/10.1006/jmre.2000.2177

Longitudinal recovery model provided by Rob Stobbe, as used in:

Sodium Imaging Optimization Under Specific Absorption Rate Constraint
Robert Stobbe and Christian Beaulieu
Magnetic Resonance in Medicine 59:345-355 (2008)
https://doi.org/10.1002/mrm.21468

See also:

Thermal relaxation and coherence dynamics of spin 3/2. 
I. Static and fluctuating quadrupolar interactions in the multipole basis
J. R. C. van der Maarel
Concepts Magn Resn A 19A(2), 2003, pp97-116
https://doi.org/10.1002/cmr.a.10087

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

__author__ = "Arthur W. Magill"
__contact__ = "arthurmagill@gmail.com"
__copyright__ = "Copyright 2016-2021, Arthur W. Magill"
__date__ = "2021/02/17"
__deprecated__ = False
__license__ = "GPLv3"
__status__ = "Development"
__version__ = "0.0.1"

import numpy as np
from scipy.linalg import inv,eig,eigh


class SpinSystem:

    # describe population quadruploar coupling distribution    
    wq_sigma = 0
    wq_mean = 0
    wq_weight = np.array([1])
    wq = np.array([0])

    # Off-resonance (B0 inhomogeneity)
    woff_sigma = 0
    woff_mean = 0
    woff_weight = np.array([1])
    woff = np.array([0])

    # Acquisition parameters
    t_dwell = 0.1e-3
    acq_pts = 2048
    TR = 100e-3
    t = 0

    ss = 0    
    Mz = False

    # Initial state, thermal equilibrium
    ss0 = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=complex)
    
    state_label = ['T00', 
                   'T10', 'T11s', 'T11a', 
                   'T20', 'T21s', 'T21a', 'T22s', 'T22a', 
                   'T30', 'T31s', 'T31a', 'T32s', 'T32a', 'T33s', 'T33a']

    
    def __init__(self,J,Mz=False):
        # if Mz==True, will return longitudinal (T10) magnetisation rather than transverse (T11)
        self.J0 = J[0]
        self.J1 = J[1]
        self.J2 = J[2]
        
        # defaults to no quadrupolar coupling
        self.wq_stats(1,0,0)

		# and no off-resonance
        self.woff_stats(1,0,0)
        self.Mz = Mz
        return

         
    def wq_stats(self,wq_pts,wq_mean,wq_sigma):
        
        self.wq_mean = wq_mean
        self.wq_sigma = wq_sigma

        if wq_pts==1:
            self.wq_weight = np.array([1.0])
            self.wq = np.array([wq_mean])
        else:
            # Initialise Gaussian population of domains at different quadrupolar couplings wq
            self.wq = np.linspace(-3.0*wq_sigma,3.0*wq_sigma,wq_pts,dtype=float)
            # Hancu, Eq 16
            self.wq_weight = 1.0/(np.sqrt(2*np.pi)*wq_sigma) * np.exp(-0.5*(self.wq-wq_mean)**2/wq_sigma**2 )
            self.wq_weight /= np.sum(self.wq_weight)
        
        self.reset()
        return
        
        
    def woff_stats(self,woff_pts,woff_mean,woff_sigma):
        
        self.woff_mean = woff_mean
        self.woff_sigma = woff_sigma

        if woff_pts==1:
            self.woff_weight = np.array([1.0])
            self.woff = np.array([woff_mean])
        else:
            # Initialise Gaussian population of domains at different resonance offsets
            self.woff = np.linspace(-3.0*woff_sigma,3.0*woff_sigma,woff_pts,dtype=float)
            self.woff_weight = 1.0/(np.sqrt(2*np.pi)*woff_sigma) * np.exp(-0.5*(self.woff-woff_mean)**2/woff_sigma**2 )
            self.woff_weight /= np.sum(self.woff_weight)
        
            #self.woff_weight = np.ones( len(self.woff) ) / len(self.woff)
            #self.woff[:] = 0
            
        self.reset()
        return


    def set_acq(self,t_dwell,pts):
        self.t_dwell = t_dwell
        self.acq_pts = pts
        return


    def acq_time(self):
        return self.acq_pts * self.t_dwell

        
    def reset(self):
        #
        # ss stores the system state as co-efficients for the 
        # following irreducible tensors:
        #
        # ss = [T00, T10, T11s, T11a, T20, T21s, T21a, T22s, T22a, T30, T31s, T31a, T32s, T32a, T33s, T33a]
        #
        # Note that T00 should always be 1

        # Reset system state to thermal equilibrium
        self.ss = self.ss0[:,np.newaxis,np.newaxis] * np.ones((16,len(self.wq),len(self.woff)))        
        
        # Precalculate weighting for wq and woff distributions
        self.wq_woff_weight = self.wq_weight[:,np.newaxis] * self.woff_weight
        
        self.t = 0
        return

    
    def rot_tr(self,phi):
        # Rotate basis in transverse plane by angle phi (rads)
        ss_tmp = np.zeros(np.shape(self.ss),dtype=complex)        

        ss_tmp[ 0,:] = self.ss[ 0,:]
        ss_tmp[ 1,:] = self.ss[ 1,:]
        ss_tmp[ 2,:] = self.ss[ 2,:]*np.cos(1*phi) + 1j*self.ss[ 3,:]*np.sin(1*phi)
        ss_tmp[ 3,:] = self.ss[ 3,:]*np.cos(1*phi) + 1j*self.ss[ 2,:]*np.sin(1*phi)
        ss_tmp[ 4,:] = self.ss[ 4,:]
        ss_tmp[ 5,:] = self.ss[ 5,:]*np.cos(1*phi) + 1j*self.ss[ 6,:]*np.sin(1*phi)
        ss_tmp[ 6,:] = self.ss[ 6,:]*np.cos(1*phi) + 1j*self.ss[ 5,:]*np.sin(1*phi)
        ss_tmp[ 7,:] = self.ss[ 7,:]*np.cos(2*phi) + 1j*self.ss[ 8,:]*np.sin(2*phi)
        ss_tmp[ 8,:] = self.ss[ 8,:]*np.cos(2*phi) + 1j*self.ss[ 7,:]*np.sin(2*phi)
        ss_tmp[ 9,:] = self.ss[ 9,:]
        ss_tmp[10,:] = self.ss[10,:]*np.cos(1*phi) + 1j*self.ss[11,:]*np.sin(1*phi)
        ss_tmp[11,:] = self.ss[11,:]*np.cos(1*phi) + 1j*self.ss[10,:]*np.sin(1*phi)
        ss_tmp[12,:] = self.ss[12,:]*np.cos(2*phi) + 1j*self.ss[13,:]*np.sin(2*phi)
        ss_tmp[13,:] = self.ss[13,:]*np.cos(2*phi) + 1j*self.ss[12,:]*np.sin(2*phi)
        ss_tmp[14,:] = self.ss[14,:]*np.cos(3*phi) + 1j*self.ss[15,:]*np.sin(3*phi)
        ss_tmp[15,:] = self.ss[15,:]*np.cos(3*phi) + 1j*self.ss[14,:]*np.sin(3*phi)

        self.ss = ss_tmp
        return
    
        
    def precess(self,w1,tau=None,acq=False):
        # Calculate system precession. Not usually called directly.
    
        # One of these must be provided
        assert(tau!=None or acq==True)
    
        # Code is more readable without so many self.'s
        J0 = self.J0
        J1 = self.J1
        J2 = self.J2

        if acq:
            signal = np.zeros((self.acq_pts,len(self.wq),len(self.woff)),dtype=complex)
            self.t += self.acq_pts * self.t_dwell
        else:
            self.t += tau

        # ss = [T00, T10, T11s, T11a, T20, T21s, T21a, T22s, T22a, T30, T31s, T31a, T32s, T32a, T33s, T33a]
        
        # Excitation, Hancu eq. 10 (see also quad)
        excite = -1j * np.matrix([
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,   w1    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,   w1    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0,   np.sqrt(3)*w1,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0,   np.sqrt(3)*w1,    0    ,    0    ,   w1    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   w1    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,   w1    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   w1    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,   np.sqrt(6)*w1,    0    ,    0     ,     0     ,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,   np.sqrt(6)*w1,    0    ,    0,  np.sqrt(2.5)*w1,     0     ,     0     ,     0     ],      
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,    np.sqrt(2.5)*w1,     0     ,     0     ],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(2.5)*w1,   0    ,    0     ,     0 ,   np.sqrt(1.5)*w1,     0     ],    
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(2.5)*w1,   0     ,     0     ,     0,    np.sqrt(1.5)*w1],
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0 , np.sqrt(1.5)*w1,     0     ,     0     ,     0     ],   
                         [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0 ,   np.sqrt(1.5)*w1,     0     ,     0     ] 
                       ])

        # Hancu eq. 14, m = 0 terms, slow fluctuating EFGs
        relax_J0 = np.matrix([
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,  0.6*J0 ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0 , np.sqrt(6)*0.2*J0, 0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,  0.6*J0 ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0 , np.sqrt(6)*0.2*J0, 0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,   J0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],      
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0 , np.sqrt(6)*0.2*J0, 0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,  0.4*J0 ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0 , np.sqrt(6)*0.2*J0, 0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,  0.4*J0 ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                    [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0]
                  ])

        # Hancu eq. 13, m != 0 terms, fast EFGs
        relax_J1J2 = np.matrix([
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [-0.4*J1-1.6*J2 ,0.4*J1+1.6*J2,0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 0.8*(J1-J2), 0    ,    0    ,    0    ,    0    ,    0    ,    0],
#                       [          0    ,0.4*J1+1.6*J2,0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 0.8*(J1-J2), 0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,J1+0.4*J2,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0 ,-np.sqrt(6)*0.2*J2, 0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,J1+0.4*J2,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0 ,-np.sqrt(6)*0.2*J2, 0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,2*J1+2*J2,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    , J1+2*J2 ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    , J1+2*J2 ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],      
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 2*J1+J2 ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 2*J1+J2 ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [  -0.8*(J1-J2) , 0.8*(J1-J2) ,0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ,1.6*J1+0.4*J2,  0    ,    0    ,    0    ,    0    ,    0    ,    0],
#                       [          0    , 0.8*(J1-J2) ,0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ,1.6*J1+0.4*J2,  0    ,    0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0 ,-np.sqrt(6)*0.2*J2, 0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,J1+0.6*J2,    0    ,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0 ,-np.sqrt(6)*0.2*J2, 0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,J1+0.6*J2,    0    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J2    ,    0    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   J2    ,    0    ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,  J1+J2  ,    0],
                       [          0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,J1+J2]
                     ])

    
        for n,wq in enumerate(self.wq):

            # Quadrupolar coupling, Hancu eq. 10                                           
            quad = 1j * np.matrix([
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(0.6)*wq,   0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(0.6)*wq,   0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0,  np.sqrt(0.6)*wq,   0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(0.4)*wq,   0    ,    0    ,    0    ,    0],
                          [0    ,    0,  np.sqrt(0.6)*wq,   0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(0.4)*wq,   0    ,    0    ,    0    ,    0    ,    0],      
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   wq    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   wq    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(0.4)*wq,   0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0,  np.sqrt(0.4)*wq,   0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   wq    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   wq    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0],
                          [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0] 
                        ]) 

            for m,woff in enumerate(self.woff):

                # Off Resonance, from R. Stobbe
                offres = 1j * np.matrix([
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,   woff  ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,   woff  ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   woff  ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,   woff  ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],      
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 2*woff  ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 2*woff  ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   woff  ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,   woff  ,    0    ,    0    ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 2*woff  ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 2*woff  ,    0    ,    0    ,    0  ],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 3*woff],
                                [0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    ,    0    , 3*woff  ,    0  ] 
                              ]) 
    
    
                Md,S = eig(excite + offres + quad - relax_J0 - relax_J1J2)
                S = np.matrix(S)
                
                if acq:
                    # precess is steps of t_dwell, record observable signal (eq 19)
                    step = S*np.diag(np.exp(self.t_dwell*Md))*inv(S)
                    
                    for p in range(self.acq_pts):
                        self.ss[:,n,m] = np.dot(step,self.ss[:,n,m])
                        
                        if self.Mz:
                            # return longitudinal magnetisation
                            signal[p,n,m] = self.ss[1,n,m]  
                        else:
                            # return transverse magnetisation
                            signal[p,n,m] = self.ss[3,n,m] - self.ss[2,n,m]    
                else:
                    # precess in single step (eq 19)
                    self.ss[:,n,m] = np.dot(S*np.diag(np.exp(tau*Md))*inv(S),self.ss[:,n,m])
                
        if acq:
            return np.sum(self.wq_woff_weight * signal,axis=(1,2))
        else:
            return
    
    
    def pulse(self,alpha,phi,tau):
        # Precess under RF field
        # alpha is flip angle
        # phi is transverse phase
        # tau is duration
        w1 = alpha / tau
        self.rot_tr(-phi)
        self.precess(w1,tau)
        self.rot_tr(phi)
        return
        

    def evolve(self,tau=None):
        # Free system evolution for time tau
        # If tau not provided, evolve to end of TR
        if tau==None:
            self.precess(0.0,self.TR - self.t)
            self.t = 0
        else:
            self.precess(0.0,tau)
        return
        
        
    def acquire(self):
        # Evolve system while acquiring signal, return signal
        signal = self.precess(0.0,acq=True)
        return signal
        
    def spoil(self):
        # Kill transverse magnetisation
        self.ss[2,:] = 0
        self.ss[3,:] = 0


if __name__ == '__main__':

    # Reproduce fig. 3 from Hancu paper
    # Pulse sequence is pi/2--mix--pi/2--pi/2--acq   
    # Pulse phases are phi, phi+pi/2, 0, mixing time is 2.4ms 
    
    import pylab as plt

    fig3 = plt.figure(3)
    fig5 = plt.figure(5)
    
    # From Hancu et al., all in Hz
    J = [185, 50.4, 50.4]
    
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
    plt.xlabel('t /ms')
    plt.ylabel('TQ signal intensity /a.u.')
    plt.legend(title='pulse length /ms')
    plt.tight_layout(h_pad=0.2,w_pad=0.2)

    plt.figure(5)
    plt.xlim(0,1000)
    plt.ylim(0.5,1.0)
    plt.xlabel('pulse width /ms')
    plt.ylabel('TQ signal intensity /a.u.')
    plt.tight_layout(h_pad=0.2,w_pad=0.2)

