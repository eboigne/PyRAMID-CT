# /*-----------------------------------------------------------------------*\
# |                                                                         |
# |                                 ++                                      |
# |                               +====+                                    |
# |                            +==========+                                 |
# |                         +================+                              |
# |                      +======================+                           |
# |                   +============================+                        |
# |                +==================================+                     |
# |             +========================================+                  |
# |             |  PyRAMID-CT: DYNAMIC CT RECONSTRUCTION |                  |
# |             +========================================+                  |
# |                                                                         |
# |                                                                         |
# |   Author: Emeric Boigné                                                 |
# |                                                                         |
# |   Contact: Emeric Boigné                                                |
# |   email: emericboigne@gmail.com                                         |
# |   Department of Mechanical Engineering                                  |
# |   Stanford University                                                   |
# |   488 Escondido Mall, Stanford, CA 94305, USA                           |
# |                                                                         |
# |-------------------------------------------------------------------------|
# |                                                                         |
# |   This file is part of the PyRAMID-CT package.                          |
# |                                                                         |
# |   License                                                               |
# |                                                                         |
# |   Copyright(C) 2021 E. Boigné                                           |
# |   PyRAMID-CT is free software: you can redistribute it and/or modify    |
# |   it under the terms of the GNU General Public License as published by  |
# |   the Free Software Foundation, either version 3 of the License, or     |
# |   (at your option) any later version.                                   |
# |                                                                         |
# |   PyRAMID-CT is distributed in the hope that it will be useful,         |
# |   but WITHOUT ANY WARRANTY; without even the implied warranty of        |
# |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         |
# |   GNU General Public License for more details.                          |
# |                                                                         |
# |   You should have received a copy of the GNU General Public License     |
# |   along with PyRAMID-CT. If not, see <http://www.gnu.org/licenses/>.    |
# |                                                                         |
# /*-----------------------------------------------------------------------*/

import numpy as np

class Projector:
    '''
    A class used to define and store 2D CT projectors
    '''

    def __init__(self, n_rays, n_angles, primary_array_projector = True):
        '''
        Constructor method for a CT projector

        Parameters
        ----------
        n_rays : int
            Number of rays (=pixels) for the CT projector
        n_angles : int
            Number of angles for the CT projector
        primary_array_projector : boolean
            A boolean specifying whether or not to create the projector attributes that can be shared in an array format (to save GPU memory)
        '''

        self.n_rays = n_rays
        if isinstance(n_angles, int) or isinstance(n_angles, float):
            self.n_angles = n_angles
            self.angles = np.linspace(0, np.pi, self.n_angles, False)
        else:
            self.angles = n_angles
            self.n_angles = len(n_angles)

        self.build_breakpoints_array(1)
        self.setup_single_projector(primary_array_projector = primary_array_projector)

    def build_breakpoints_array(self, ind_breakpoints):
        '''
        Method that builds the array of breakpoints ind_breakpoints from either the number of breakpoints M or directly ind_breakpoints

        Parameters
        ----------
        ind_breakpoints : int or np.array
            Either simply the number of breakpoints M or the array of the breakpoints
        '''

        if isinstance(ind_breakpoints, int) or isinstance(ind_breakpoints, float): # Test if given directly ind_breakpoints = M, in which case build a standard array ind_breakpoints of M equidistant breakpoints
            self.M = ind_breakpoints
            self.ind_breakpoints = (np.linspace(0, self.n_angles-1, self.M)+0.5).astype('int') # index of registered images.
            self.ind_breakpoints[-1] += 1 # Add last angle at end (Last block projector is one-line taller).
        else:
            self.ind_breakpoints = ind_breakpoints
            self.M = len(ind_breakpoints)

    def setup_single_projector(self, primary_array_projector = True):
        '''
        Method to setup a single projector using all angles at once (no block-splitting)

        Parameters
        ----------
        primary_array_projector : boolean
            A boolean specifying whether or not to create the projector attributes that can be shared in an array format (to save GPU memory)
        '''

        self.setup_projector(self.angles, primary_array_projector = primary_array_projector)
        self.array_projectors = [self]

    def setup_array_projectors(self, ind_breakpoints):
        '''
        Method to setup an array of sub-projector to decompose in block the forward and backward projection, using the provided array of breakpoints.

        Parameters
        ----------
        ind_breakpoints : np.array
            The array of the breakpoints to use to decompose the projector
        '''

        if np.array_equal(self.ind_breakpoints, ind_breakpoints): # Don't do anything if already setup as requested.
            return

        self.build_breakpoints_array(ind_breakpoints)
        if self.M == 1:
            self.setup_single_projector()
        else:
            print('\tSetting up projector', end = '')

            # Destroy the previous ASTRA objects to avoid memory leaks
            self.clear()

            self.angle_breakpoints = []
            self.proj_geom = []
            self.sino_id = []

            # Setup first projector
            k = 0
            self.angle_breakpoints.append(self.angles[self.ind_breakpoints[k]:self.ind_breakpoints[k+1]])

            # Create a new instance of Projector of the same type as self
            this_projector = type(self)(self.n_rays, self.angle_breakpoints[k])
            this_projector.setup_single_projector(primary_array_projector = True)
            self.array_projectors = [this_projector]

            for k in range(1, self.M-1):
                self.angle_breakpoints.append(self.angles[self.ind_breakpoints[k]:self.ind_breakpoints[k+1]])

                # Create a new instance of Projector of the same type as self
                this_projector = type(self)(self.n_rays, self.angle_breakpoints[k])
                this_projector.setup_single_projector(primary_array_projector = False)

                # Transfer projector attributes that are shared by all projectors (to save GPU memory)
                this_projector.transfer_array_global_properties(self.array_projectors[0])

                # self.array_projectors.append(Projector(self.n_rays, self.angle_breakpoints[k], n_slices = self.n_slices, primary_array_projector = False))
                self.array_projectors.append(this_projector)

    def clear(self):
        '''
        Method that clears this projector, and free up memory space
        '''
        for proj in self.array_projectors:
            proj.clear_projector()

    def fp_array(self, slice_data, ind_block):
        '''
        Method for the forward projection of a specific block

        Parameters
        ----------
        slice_data : np.array or torch.Tensor
            The array of volume data to project of size Nz x N x N or simply N x N
        ind_block : int
            The index of the sub-block to project

        Returns
        -------
        sino : np.array or torch.Tensor
            The sinogram obtained after projection of size Nz x Na x N or simply Na x N
        '''

        return(self.array_projectors[ind_block].fp(slice_data))

    def bp_array(self, sino, ind_block):
        '''
        Method for the backward projection of a specific block

        Parameters
        ----------
        sino : np.array or torch.Tensor
            The array of sinogram data to back-project of size Nz x Na x N or simply Na x N
        ind_block : int
            The index of the sub-block to project

        Returns
        -------
        slice_out : np.array or torch.Tensor
            The volume data obtained after back-projection of size Nz x N x N or simply N x N
        '''

        return(self.array_projectors[ind_block].bp(sino))
