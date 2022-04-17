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
from . import case
from . import algorithm

try:
    import torch
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, CPU capabilities only')

class Chambolle_Pock_tv_motion(algorithm.Algorithm):
    '''
    A class used to represent the Chambolle and Pock algorithm with motion (PLI, M > 1)
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

        self.M = self.case.M

        # Nd = 2 * self.nd_tv, + 2 * self.nd_tv (for z), + 2 * self.nd_tv (for time)
        if self.n_slices > 1 and self.case.reg_z > 0 and self.case.reg_time > 0:
            self.Nd = 4 * self.nd_tv
        elif self.n_slices > 1 and self.case.reg_z > 0:
            self.Nd = 3 * self.nd_tv
        elif self.case.reg_time > 0:
            self.Nd = 3 * self.nd_tv
        else:
            self.Nd = 2 * self.nd_tv

        # id_time: index out of Nd of differences that are computed w.r.t time
        if self.case.reg_time > 0:
            self.id_time = self.Nd - 2 - 1 # 2 terms only (only computed UP / DW for time differences)
        else:
            self.id_time = self.Nd # No difference w.r.t time

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
        # Eg: Upwind, Gamma = |1| + |-1| + |1| + |-1| = 4
        gamma = 2 * self.nd_tv * beta
        if self.case.reg_z > 0:
            gamma += 2 * self.nd_tv * beta * np.sqrt(self.case.reg_z)
        # # Dual step sizes
        self.Sigma_B = 1.0 / self.projector.fp(self.module_array.ones([self.n_rays,self.n_rays], dtype = self.dtype)) # Same as without motion
        self.Sigma_D = self.module_array.ones([1, self.Nd, 1, 1, 1], dtype = self.dtype) / beta
        self.Sigma_D[self.id_time:] /= np.sqrt(self.case.reg_time)
        # self.Sigma_D[self.id_z:] /= np.sqrt(self.case.reg_z) # TODO: Implement np.sqrt(self.case.reg_z) factor for preconditioner

        # Primal step sizes
        self.Tau = 1.0 / (gamma + 2 * self.nd_tv * np.sqrt(self.case.reg_time) + self.projector.bp(self.module_array.ones([self.n_angles,self.n_rays], dtype = self.dtype)))

        # Build extra TV preconditioner
        self.block_corrections = self.module_array.reshape(self.correction_factor_angles_per_block(), [1,self.case.M,1,1])
        self.preconditioner_TV = self.M * self.block_corrections

        # Over-relaxation in primal variable
        self.theta = 1.0

        # Allocate primal variables
        self.x = self.module_array.zeros([self.n_slices, self.case.M, self.n_rays, self.n_rays], dtype = self.dtype) # Nz x M x N x N
        self.x_tilde = self.module_array.zeros([self.n_slices, self.case.M, self.n_rays, self.n_rays], dtype = self.dtype) # Nz x M x N x N
        self.x_avg = self.module_array.zeros([self.n_slices, self.n_rays, self.n_rays], dtype = self.dtype) # Nz x N x N
        # self.primal_update = self.module_array.zeros_like(self.x, dtype = self.dtype)
        self.primal_residual = self.module_array.zeros_like(self.sino, dtype = self.dtype) # Nz x Na x N

        # Allocate dual variables
        self.y_1 = self.module_array.zeros_like(self.sino, dtype = self.dtype) # Nz x Na x N
        self.y_2 = self.module_array.zeros([self.n_slices, self.Nd, self.case.M, self.n_rays, self.n_rays], dtype = self.dtype) # Nz x Nd x M x N x N

        # Initialize all arrays on GPU
        if self.case.use_pyTorch:
            self.Sigma_B = self.Sigma_B.cuda()
            self.Sigma_D = self.Sigma_D.cuda()
            self.w = self.w.cuda()
            self.Tau = self.Tau.cuda()

            if self.n_chunks == 1: # If only one chunk, keep these variables on GPU (Otherwise extra cost of unnecessary GPU-CPU transfer)
                self.x = self.x.cuda()
                self.x_tilde = self.x_tilde.cuda()
                # self.primal_update = self.primal_update.cuda()
                self.primal_residual = self.primal_residual.cuda()
                self.y_1 = self.y_1.cuda()
                self.y_2 = self.y_2.cuda()

            # Allocate default variables
            self.x_tilde = self.module_array.copy(self.x)

        if self.case.steady_iterations_first > 0:
            print('\tComputing a guess for the initial solution')
            self.compute_initial_steady_solution()

        self.projector.setup_array_projectors(self.ind_breakpoints)

    def compute_initial_steady_solution(self):
        '''
        Method that computes the initial guess from the steady version of the algorithm
        '''

        input_parameters = {}
        for key in self.case.__dict__:
            input_parameters[key] = self.case.__dict__[key]
        input_parameters['algorithm_name'] = 'Chambolle_Pock_tv'
        input_parameters['nb_it'] = self.case.steady_iterations_first

        self.case_init = case.Case()
        self.case_init.load_input_parameters(input_parameters)
        self.case_init.run()

        print('\nFinished running initialization case - Starting time-CT algorithm:')

        self.x_avg = self.module_array.reshape(self.case_init.algo.x, self.x_avg.shape)
        self.y_1 = self.case_init.algo.y_1
        x_tilde = self.case_init.algo.x_tilde
        y_2 = self.case_init.algo.y_2

        for i in range(self.case.M):
            self.x[:,i,:,:] = self.module_array.copy(self.x_avg)
            self.x_tilde[:,i,:,:] = self.module_array.squeeze(self.module_array.copy(x_tilde))
            self.y_2[:,0:y_2.shape[-4],i,:,:] = self.module_array.squeeze(self.module_array.copy(y_2))

        # Save the obj_fct and error terms
        self.case.data['obj_fct_steadyIts'] = np.array(self.case_init.algo.obj_function_table)
        self.case.data['l2_primal_update_steadyIts'] = np.array(self.case_init.algo.l2_primal_update_table)
        self.case.data['l2_primal_update_normalized_steadyIts'] = np.array(self.case_init.algo.l2_primal_update_normalized_table)

        self.case_init.projector.clear()
        del self.case_init

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
        this_y1 = (this_y1 + self.Sigma_B * this_primal_residual) / (1.0 + self.Sigma_B * (1/self.w))
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
            self.primal_update_reg = 0.0
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
            this_D = self.D(this_x_tilde, self.case.reg_z/self.case.reg, self.case.reg_time, self.mask_static, self.factor_reg_static)
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
                self.TV_term += self.case.reg / self.M * np.sum(self.tv_module.compute_L21_norm(this_D))

            this_prox_argument = this_y2 + self.Sigma_D * this_D # The argument of the proximal operator
            del this_D

            if self.case.use_pyTorch:
                this_y2 = this_prox_argument / self.module_array.maximum(torch.tensor([1.0], dtype = self.dtype).cuda(), self.module_array.sqrt(self.module_array.sum(this_prox_argument**2, axis = 1, keepdims = True)) / (self.case.reg / self.M))
            else:
                this_y2 = this_prox_argument / self.module_array.maximum(1.0, self.module_array.sqrt(self.module_array.sum(this_prox_argument**2, axis = 1, keepdims = True)) / (self.case.reg / self.M))
            del this_prox_argument
            this_primal_update_reg = self.D_T(this_y2, self.case.reg_z/self.case.reg, self.case.reg_time, self.mask_static, self.factor_reg_static)

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

        if self.case.use_pyTorch:
            if self.case.reg > 0:
                this_primal_update = -self.Tau * (self.primal_update[i1:i2].cuda() + self.preconditioner_TV * self.primal_update_reg[i1:i2].cuda())
            else:
                this_primal_update = -self.Tau * self.primal_update[i1:i2].cuda()
        else: # CPU operations
            this_primal_update = -self.Tau * (self.primal_update[i1:i2] + self.preconditioner_TV * +self.primal_update_reg[i1:i2])

        if self.use_mask_no_update:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_no_update, this_primal_update.shape)
            this_primal_update[mask_4D_broadcast] = 0.0

        if self.case.use_pyTorch: # GPU
            this_x = self.x[i1:i2].cuda() + this_primal_update
        else: # CPU
            this_x = self.x[i1:i2] + this_primal_update

        this_x_avg = self.module_array.sum(this_x * self.block_corrections, axis = 1) / self.module_array.sum(self.block_corrections) # Time-integral
        this_x_tilde = this_x + self.theta * this_primal_update # With theta = 0.0, x_tilde = x

        if self.use_mask_zero:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_zero, this_x_avg.shape)
            this_x_avg[mask_4D_broadcast] = 0.0
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_zero, this_x.shape)
            this_x[mask_4D_broadcast] = 0.0
            this_x_tilde[mask_4D_broadcast] = 0.0

        # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
        if self.case.use_pyTorch:
            self.x[i1:i2] = this_x.detach().cpu()
            self.x_tilde[i1:i2] = this_x_tilde.detach().cpu()
            self.x_avg[i1:i2] = this_x_avg.detach().cpu()
        else:
            self.x[i1:i2] = this_x
            self.x_tilde[i1:i2] = this_x_tilde
            self.x_avg[i1:i2] = this_x_avg

        del this_primal_update, this_x, this_x_tilde, this_x_avg

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
            self.primal_residual[iz] = self.C(self.x_tilde[iz], self.projector, self.ind_breakpoints)

        # 2/ Compute primal residual and dual update of fidelity term in chunks
        self.compute_update_WLS_chunk(i1, i2)

        # 3/ Compute all primal update from fidelity terms
        for iz in range(i1, i2):
            self.primal_update[iz] = self.C_T(self.y_1[iz], self.projector, self.ind_breakpoints)

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

        # if self.case.compute_metrics: # TODO: Consider doing this? Extra cost. Maybe as an option? The current self.TV_term isn't the actual obj function (x_tilde instead of x). Also would have to chunk it
        #     this_D = self.D(self.x, self.case.reg_z/self.case.reg, self.case.reg_time, self.mask_static, self.factor_reg_static)
        #     self.TV_term = self.case.reg / self.M * np.sum(self.tv_module.compute_L21_norm(this_D))

