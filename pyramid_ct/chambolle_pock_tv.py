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
from . import algorithm

try:
    import torch
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, CPU capabilities only')

class Chambolle_Pock_tv(algorithm.Algorithm):
    '''
    A class used to represent the Chambolle and Pock algorithm without motion (no PLI, M = 1)
    '''

    def __init__(self, case):
        '''
        Constructor method for the class
        '''

        algorithm.Algorithm.__init__(self, case)

    def initialize_variables(self):
        '''
        Method that initializes the algorithm variables
        '''

        # Nd = 2 * self.nd_tv, + 2 * self.nd_tv (for z), + 2 * self.nd_tv (for time)
        if self.n_slices > 1 and self.case.reg_z > 0:
            self.Nd = 3 * self.nd_tv
        else:
            self.Nd = 2 * self.nd_tv

        # Beta = Beta_i = \sum_{j=1}^m |D_{i,j}|
        # Eg: Upwind, Beta = |1| + |-1| = 2

        # Factors depend on tv scheme
        if self.case.tv_scheme == 'central':
            beta = 1.0 # 2 terms with (1/2) and (-1/2)
        elif self.case.tv_scheme == 'hybrid':
            beta = 2.0 / np.sqrt(2) # 4 terms with (1/sqrt(2)) and (-1/sqrt(2))
        else: # Upwind / Downwind
            beta = 2.0 # 2 terms with (1) and (-1)
        # Gamma = Gamma_j = \sum_{i=1}^n |D_{i,j}|
        # Eg: Upwind (2D, without z direction), Gamma = |1| + |-1| + |1| + |-1| = 4
        gamma = 2 * self.nd_tv * beta
        if self.case.reg_z > 0:
            gamma += 2 * self.nd_tv * beta * np.sqrt(self.case.reg_z)

        # Dual step sizes
        self.Sigma_A = 1.0 / self.projector.fp(self.module_array.ones([self.n_rays,self.n_rays], dtype = self.dtype)) # Equal to self.w
        self.Sigma_D = self.module_array.ones([1, self.Nd, 1, 1, 1], dtype = self.dtype) / beta # Typically significantly larger in magnitude than Sigma_A

        # self.Sigma_D[self.id_z:] /= np.sqrt(self.case.reg_z) # TODO: Implement np.sqrt(self.case.reg_z) factor for preconditioner

        # Primal step sizes
        a = self.module_array.ones([self.n_angles,self.n_rays], dtype = self.dtype)
        self.Tau = 1.0 / (gamma + self.projector.bp(a))

        # Over-relaxation in primal variable
        self.theta = 1.0

        # Allocate primal variables
        self.x = self.module_array.zeros([self.n_slices, 1, self.n_rays, self.n_rays], dtype = self.dtype) # Nz x M x N x N
        self.x_tilde = self.module_array.copy(self.x)
        self.primal_residual = self.module_array.zeros_like(self.sino, dtype = self.dtype) # Nz x Na x N

        # Allocate dual variables
        self.y_1 = self.module_array.zeros([self.n_slices, self.n_angles, self.n_rays], dtype = self.dtype) # Nz x Na x Np
        self.y_2 = self.module_array.zeros([self.n_slices, self.Nd, 1, self.n_rays, self.n_rays], dtype = self.dtype) # Nz x Nd x M x N x N

        # Initialize all arrays on GPU
        if self.case.use_pyTorch:
            self.Sigma_A = self.Sigma_A.cuda()
            self.Sigma_D = self.Sigma_D.cuda()
            self.w = self.w.cuda()
            self.Tau = self.Tau.cuda()

            if self.n_chunks == 1: # If only one chunk, keep these variables on GPU (Otherwise extra cost of unnecessary GPU-CPU transfer)
                self.x = self.x.cuda()
                self.x_tilde = self.x_tilde.cuda()
                self.primal_residual = self.primal_residual.cuda()
                self.y_1 = self.y_1.cuda()
                self.y_2 = self.y_2.cuda()

    def compute_update_WLS_chunk(self, i1, i2):
        '''
        Method that computes the WLS update for the chunk of slices from i1 to i2

        Parameters
        ----------
        i1 : int
            Starting slice of the chunk
        i2 : int
            End slice of the chunk
        '''

        # Local variables
        this_primal_residual = self.primal_residual[i1:i2]
        this_sino = self.sino[i1:i2]
        this_y1 = self.y_1[i1:i2]

        # Transfer to GPU
        if self.case.use_pyTorch and self.n_chunks > 1:
            this_primal_residual = this_primal_residual.cuda()
            this_sino = this_sino.cuda()
            this_y1 = this_y1.cuda()

        # Operations on GPU
        this_primal_residual += - this_sino
        this_y1 = (this_y1 + self.Sigma_A * this_primal_residual) / (1.0 + self.Sigma_A * (1/self.w))
        if self.case.compute_metrics:
            self.WLS_term += 0.5 * self.module_array.sum(self.w * self.module_array.square(this_primal_residual))

        # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
        if self.case.use_pyTorch and self.n_chunks > 1:
            self.y_1[i1:i2] = this_y1.detach().cpu()
            self.primal_residual[i1:i2]= this_primal_residual.detach().cpu()
        else:
            self.y_1[i1:i2] = this_y1
            self.primal_residual[i1:i2]= this_primal_residual

        if self.case.use_pyTorch:
            del this_sino, this_y1, this_primal_residual

    def compute_update_TV_chunk(self, i1, i2, i_chunk):
        '''
        Method that computes the WLS update for the chunk of slices from i1 to i2

        Parameters
        ----------
        i1 : int
            Starting slice of the chunk
        i2 : int
            End slice of the chunk
        i_chunk : int
            Index of the chunk, to properly manage end cases (i1 = 0, and i2 = n_slices-1)
        '''

        if self.case.reg == 0: # y_2 = 0 all the time.
            self.primal_update_reg = 0
        else:
            # Local variables
            this_y2 = self.y_2[i1:i2]
            if self.n_chunks == 1:
                this_x_tilde = self.x_tilde[i1:i2]
            elif i_chunk == 0: # First
                this_x_tilde = self.x_tilde[i1:i2+(self.stencil//2)]
            elif i_chunk == self.n_chunks-1: # Last
                this_x_tilde = self.x_tilde[i1-(self.stencil//2):i2]
            else:
                this_x_tilde = self.x_tilde[i1-(self.stencil//2):i2+(self.stencil//2)]

            # Transfer to GPU
            if self.case.use_pyTorch and self.n_chunks > 1:
                this_y2 = this_y2.cuda()
                this_x_tilde = this_x_tilde.cuda()

            # Operations
            this_D = self.D(this_x_tilde, self.case.reg_z/self.case.reg)
            del this_x_tilde
            if self.n_chunks == 1:
                this_D = this_D[i1:i2]
            elif i_chunk == 0: # First
                this_D = this_D[:-(self.stencil//2)]
            elif i_chunk == self.n_chunks-1: # Last
                this_D = this_D[(self.stencil//2):]
            else:
                this_D = this_D[(self.stencil//2):-(self.stencil//2)]

            if self.case.compute_metrics: # Not exactly the objective function: Using x_tilde and not x #TODO: Note this somewhere
                self.TV_term += self.case.reg * np.sum(self.tv_module.compute_L21_norm(this_D))

            this_prox_argument = this_y2 + self.Sigma_D * this_D # The argument of the proximal operator
            del this_D

            if self.case.use_pyTorch:
                this_y2 = this_prox_argument / self.module_array.maximum(torch.tensor([1.0], dtype = self.dtype).cuda(), self.module_array.sqrt(self.module_array.sum(this_prox_argument**2, axis = 1, keepdims = True)) / self.case.reg)
            else:
                this_y2 = this_prox_argument / self.module_array.maximum(1.0, self.module_array.sqrt(self.module_array.sum(this_prox_argument**2, axis = 1, keepdims = True)) / self.case.reg)
            del this_prox_argument
            this_primal_update_reg = self.D_T(this_y2, self.case.reg_z/self.case.reg, self.case.reg_time)

            # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
            if self.case.use_pyTorch and self.n_chunks > 1:
                self.y_2[i1:i2] = this_y2.detach().cpu()
                self.primal_update_reg[i1:i2] = this_primal_update_reg.detach().cpu()
            else:
                self.y_2[i1:i2] = this_y2
                self.primal_update_reg[i1:i2] = this_primal_update_reg

            del this_y2, this_primal_update_reg

    def update_primal_variables_chunk(self, i1, i2):
        '''
        Method that computes the updated primal variables for one iteration for the chunk of slices from i1 to i2

        Parameters
        ----------
        it : int
            Iteration number
        i1 : int
            Starting slice of the chunk
        i2 : int
            End slice of the chunk
        '''

        if self.case.use_pyTorch: # GPU operations
            if self.case.reg > 0:
                this_primal_update = -self.Tau * (self.primal_update[i1:i2].cuda()+self.primal_update_reg[i1:i2].cuda())
            else:
                this_primal_update = -self.Tau * (self.primal_update[i1:i2].cuda())
        else: # CPU operations
            this_primal_update = -self.Tau * (self.primal_update[i1:i2]+self.primal_update_reg[i1:i2])

        if self.use_mask_no_update:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_no_update, this_primal_update.shape)
            this_primal_update[mask_4D_broadcast] = 0.0

        if self.case.use_pyTorch: # GPU operations
            this_x = self.x[i1:i2].cuda() + this_primal_update
        else:
            this_x = self.x[i1:i2] + this_primal_update

        this_x_tilde = this_x + self.theta * this_primal_update # With theta = 0.0, x_tilde = x

        if self.use_mask_zero:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_zero, this_x.shape)
            this_x[mask_4D_broadcast] = 0.0
            this_x_tilde[mask_4D_broadcast] = 0.0

        # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
        if self.case.use_pyTorch:
            self.x[i1:i2] = this_x.detach().cpu()
            self.x_tilde[i1:i2] = this_x_tilde.detach().cpu()
        else:
            self.x[i1:i2] = this_x
            self.x_tilde[i1:i2] = this_x_tilde

        del this_primal_update, this_x, this_x_tilde

    def update_dual_variables_chunk(self, i1, i2, i_chunk):
        '''
        Method that computes the updated dual variables for one iteration for the chunk of slices from i1 to i2

        Parameters
        ----------
        it : int
            Iteration number
        i1 : int
            Starting slice of the chunk
        i2 : int
            End slice of the chunk
        '''

        # 1/ Compute forward projections
        for iz in range(i1, i2):
            self.primal_residual[iz] = self.projector.fp(self.x_tilde[iz])

        # 2/ Compute primal residual and dual update of fidelity term in chunks
        self.compute_update_WLS_chunk(i1, i2)

        # 3/ Compute all primal update from fidelity terms
        for iz in range(i1, i2):
            self.primal_update[iz] = self.projector.bp(self.y_1[iz])

        # 4/ Compute dual update of regularization term in chunks
        self.compute_update_TV_chunk(i1, i2, i_chunk)

    def do_one_iteration(self, it):
        '''
        Method that takes care of performing one iteration of the algorithm

        Parameters
        ----------
        it : int
            Iteration number
        '''

        self.primal_update = self.module_array.zeros_like(self.x, dtype = self.dtype)
        self.primal_update_reg = self.module_array.zeros_like(self.x, dtype = self.dtype)

        if self.case.compute_metrics:
            self.WLS_term = 0
            self.TV_term = 0

        # 1/ Compute dual variable updates for all chunks
        for i_chunk in range(self.n_chunks):
            self.update_dual_variables_chunk(self.ind_slice_1[i_chunk], self.ind_slice_2[i_chunk], i_chunk)

        # 2/ Update primal variables in chunks
        for i_chunk in range(self.n_chunks):
            self.update_primal_variables_chunk(self.ind_slice_1[i_chunk], self.ind_slice_2[i_chunk])
