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

from . import projector
import numpy as np
import astra

try:
    import torch
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, CPU capabilities only')

class Projector_ASTRA(projector.Projector):
    '''
    A subclass used to define and store CT projectors from the ASTRA toolbox
    '''

    def __init__(self, n_rays, n_angles, projector_type = 'cuda'):
        '''
        Constructor method for the ASTRA projector

        Parameters
        ----------
        n_rays : int
            Number of rays (=pixels) for the CT projector
        n_angles : int
            Number of angles for the CT projector
        projector_type : str
            The projector type from the ASTRA Toolbox
        '''

        projector.Projector.__init__(self, n_rays, n_angles)
        self.projectorType = projector_type # Types are: 'cuda', 'line', 'strip' (preferred), 'linear' (Joseph's Model)

    def setup_projector(self, angles, primary_array_projector = True):
        '''
        Method to setup the projector given a specific angle arrangement

        Parameters
        ----------
        angles : np.array
            The array of the projection angles
        primary_array_projector : boolean
            A boolean specifying whether or not to create the projector attributes that can be shared in an array format (to save GPU memory)
        '''

        if not hasattr(self, 'proj_geom'):
            self.proj_geom = astra.create_proj_geom('parallel', 1.0, self.n_rays, angles)
        if not hasattr(self, 'sino_id'):
            self.sino_id = astra.data2d.create('-sino', self.proj_geom)

        if primary_array_projector:
            self.vol_geom = astra.create_vol_geom(self.n_rays, self.n_rays)
            self.rec_id = astra.data2d.create('-vol', self.vol_geom)

    def clear_projector(self):
        '''
        Method to clear the projector, and free-up the memory space
        '''

        # Destroy the previous ASTRA objects to avoid memory leaks
        if hasattr(self, 'proj_geom'):
            del self.proj_geom # Only a python dictionary
        if hasattr(self, 'vol_geom'):
            del self.vol_geom # Only a python dictionary
        if hasattr(self, 'sino_id'):
            astra.data2d.delete(self.sino_id) # GPU Memory tied to this object
            del self.sino_id
        if hasattr(self, 'rec_id'):
            astra.data2d.delete(self.rec_id) # GPU Memory tied to this object
            del self.rec_id

    def transfer_array_global_properties(self, main_projector):
        '''
        Method to transfer the global projector properties that can be shared within projectors to save GPU memory

        Parameters
        ----------
        main_projector : Projector_ASTRA
            The main projector from which the shared data is transferred
        '''

        # Transfer pointers for geometry and volume array from main projector
        self.vol_geom = main_projector.vol_geom
        self.rec_id = main_projector.rec_id

    def fp(self, slice_data):
        '''
        Method for the forward projection using the ASTRA Toolbox (operator A)

        Parameters
        ----------
        slice_data : np.array or torch.Tensor
            The array of volume data to project of size Nz x N x N or simply N x N

        Returns
        -------
        sino : np.array or torch.Tensor
            The sinogram obtained after projection of size Nz x Na x N or simply Na x N
        '''

        using_pyTorch = type(slice_data) != np.ndarray
        if using_pyTorch:
            on_gpu = 'cuda' in str(slice_data.device)
            slice_data = slice_data.detach().cpu().numpy()

        if len(slice_data.shape) <= 2: # If data only 2D
            slice_data = np.reshape(slice_data, [1, slice_data.shape[0], slice_data.shape[1]])

        if self.projectorType == 'cuda':
            self.cfg = astra.astra_dict('FP_CUDA')
        else:
            self.cfg = astra.astra_dict('FP')
            proj_id = astra.create_projector(self.projectorType, self.proj_geom, self.vol_geom)
            self.cfg['ProjectorId'] = proj_id

        self.cfg['ProjectionDataId'] = self.sino_id
        self.cfg['VolumeDataId'] = self.rec_id
        sino = np.zeros([slice_data.shape[0], self.n_angles, self.n_rays])

        for iz in range(slice_data.shape[0]):
            self.alg_id = astra.algorithm.create(self.cfg)

            astra.data2d.store(self.rec_id, slice_data[iz,:,:])
            astra.algorithm.run(self.alg_id)

            sino_2d = astra.data2d.get(self.sino_id)
            sino[iz,:,:] = sino_2d

            astra.algorithm.delete(self.alg_id) # GPU memory is tied up with algorithm object

        if not self.projectorType == 'cuda':
            astra.projector.delete(proj_id)

        sino = np.squeeze(sino)
        if using_pyTorch:
            if on_gpu:
                return(torch.tensor(sino, device = torch.device('cuda')))
            else:
                return(torch.tensor(sino, device = torch.device('cpu')))
        else:
            return(sino)


    def bp(self, sino):
        '''
        Method for the backward projection using the ASTRA Toolbox (operator A^T)

        Parameters
        ----------
        sino : np.array or torch.Tensor
            The array of sinogram data to back-project of size Nz x Na x N or simply Na x N

        Returns
        -------
        slice_out : np.array or torch.Tensor
            The volume data obtained after back-projection of size Nz x N x N or simply N x N
        '''

        using_pyTorch = type(sino) != np.ndarray
        if using_pyTorch:
            on_gpu = 'cuda' in str(sino.device)
            sino = sino.detach().cpu().numpy()

        if len(sino.shape) <= 2:
            sino = np.reshape(sino, [1, sino.shape[0], sino.shape[1]])

        slice_out = np.zeros([sino.shape[0], self.n_rays, self.n_rays])

        if self.projectorType == 'cuda':
            self.cfg = astra.astra_dict('BP_CUDA')
        else:
            self.cfg = astra.astra_dict('BP')
            proj_id = astra.create_projector(self.projectorType, self.proj_geom, self.vol_geom)
            self.cfg['ProjectorId'] = proj_id

        self.cfg['ProjectionDataId'] = self.sino_id
        self.cfg['ReconstructionDataId'] = self.rec_id

        for iz in range(sino.shape[0]):
            self.alg_id = astra.algorithm.create(self.cfg)

            astra.data2d.store(self.sino_id, sino[iz,:,:])
            astra.algorithm.run(self.alg_id)

            rec_slice = astra.data2d.get(self.rec_id)

            slice_out[iz,:,:] = rec_slice
            astra.algorithm.delete(self.alg_id) # GPU memory is tied up with algorithm object

        if not self.projectorType == 'cuda':
            astra.projector.delete(proj_id)

        slice_out = np.squeeze(slice_out)
        if using_pyTorch:
            if on_gpu:
                return(torch.tensor(slice_out, device = torch.device('cuda')))
            else:
                return(torch.tensor(slice_out, device = torch.device('cpu')))
        else:
            return(slice_out)

