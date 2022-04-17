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

class Gradient_descent_tv_motion(algorithm.Algorithm):
    '''
    A class used to represent the gradient descent algorithm with motion (PLI, M > 1)
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

        # Allocate variables
        self.x = self.module_array.zeros([self.n_slices, self.case.M, self.n_rays, self.n_rays])
        self.x_avg = self.module_array.zeros([self.n_slices, self.n_rays, self.n_rays])
        self.res = self.module_array.zeros([self.n_slices, self.n_angles, self.n_rays])
        self.primal_update = self.module_array.zeros_like(self.x)

        if self.case.steady_iterations_first > 0:
            print('Computing a guess for the initial solution')
            self.compute_initial_steady_solution()
        self.projector.setup_array_projectors(self.ind_breakpoints)

        # Preconditioner defined similarly to SIRT
        self.preconditioner = self.C_T(self.module_array.ones([self.n_angles,self.n_rays]), self.projector, self.ind_breakpoints)
        self.preconditioner[self.preconditioner < 1.0] = 1.0 # Clean some low and zero values near edges
        self.preconditioner = 1.0 / self.preconditioner

        # Alternative preconditioner choice: Kronecker product of the static SIRT algorithm, Q_M = I_M \otimes Q
        # self.preconditioner = self.module_array.tile(self.p, [self.case.M, 1, 1])

        # Build extra TV preconditioner
        self.block_corrections = self.module_array.reshape(self.correction_factor_angles_per_block(), [1,self.case.M,1,1])
        self.preconditioner_TV = self.M * self.block_corrections

        # Initialize all arrays on GPU
        if self.case.use_pyTorch:
            self.w = self.w.cuda()
            self.preconditioner = self.preconditioner.cuda()

            if self.n_chunks == 1: # If only one chunk, keep these variables on GPU (Otherwise extra cost of unnecessary GPU-CPU transfer)
                self.x = self.x.cuda()
                self.x_avg = self.x_avg.cuda()
                self.res = self.res.cuda()
                self.primal_update = self.primal_update.cuda()

    def compute_initial_steady_solution(self):
        '''
        Method that computes the initial guess from the steady version of the algorithm
        '''

        input_parameters = {}
        for key in self.case.__dict__:
            input_parameters[key] = self.case.__dict__[key]
        input_parameters['algorithm_name'] = 'Gradient_descent_tv'
        input_parameters['nb_it'] = self.case.steady_iterations_first

        self.case_init = case.Case()
        self.case_init.load_input_parameters(input_parameters)
        self.case_init.run()

        print('\nFinished running initialization case - Starting time-CT algorithm:')

        self.x_avg = self.module_array.reshape(self.case_init.algo.x, [self.n_slices, self.n_rays, self.n_rays])
        for i in range(self.case.M):
            self.x[:,i,:,:] = self.module_array.squeeze(self.module_array.copy(self.x_avg))

        # Save the obj_fct terms
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
        this_res = self.res[i1:i2]
        this_sino = self.sino[i1:i2]

        # Transfer to GPU
        if self.case.use_pyTorch and self.n_chunks > 1:
            this_res = this_res.cuda()
            this_sino = this_sino.cuda()

        # Operations on GPU
        this_res += - this_sino
        if self.case.compute_metrics:
            self.WLS_term += 0.5 * self.module_array.sum(self.w * self.module_array.square(this_res))
        this_res = self.w * this_res

        # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
        if self.case.use_pyTorch and self.n_chunks > 1:
            self.res[i1:i2]= this_res.detach().cpu()
        else:
            self.res[i1:i2]= this_res

        if self.case.use_pyTorch:
            del this_sino, this_res

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

        if self.case.reg == 0:
            self.primal_update_reg = 0.0
        else:
            # Local variables
            if self.n_chunks == 1:
                this_x = self.x[i1:i2]
            elif i_chunk == 0: # First
                this_x = self.x[i1:i2+(self.stencil//2)]
            elif i_chunk == self.n_chunks-1: # Last
                this_x = self.x[i1-(self.stencil//2):i2]
            else:
                this_x = self.x[i1-(self.stencil//2):i2+(self.stencil//2)]

            # Transfer to GPU
            if self.case.use_pyTorch and self.n_chunks > 1:
                this_x = this_x.cuda()

            # Operations
            _, this_primal_update_reg, this_grad_norms = self.tv_fct(this_x, reg_z_over_reg=self.case.reg_z/self.case.reg, reg_time = self.case.reg_time, mask_static = self.mask_static, factor_reg_static = self.factor_reg_static, return_pytorch_tensor = self.case.use_pyTorch, return_grad_norms = True)
            del this_x
            if self.n_chunks == 1:
                this_grad_norms = this_grad_norms[i1:i2]
                this_primal_update_reg = this_primal_update_reg[i1:i2]
            elif i_chunk == 0: # First
                this_grad_norms = this_grad_norms[:-(self.stencil//2)]
                this_primal_update_reg = this_primal_update_reg[:-(self.stencil//2)]
            elif i_chunk == self.n_chunks-1: # Last
                this_grad_norms = this_grad_norms[(self.stencil//2):]
                this_primal_update_reg = this_primal_update_reg[(self.stencil//2):]
            else:
                this_grad_norms = this_grad_norms[(self.stencil//2):-(self.stencil//2)]
                this_primal_update_reg = this_primal_update_reg[(self.stencil//2):-(self.stencil//2)]

            if self.case.compute_metrics: # Exactly the objective function, unlike CP
                self.TV_term += self.case.reg / self.M * self.module_array.sum(this_grad_norms)

            # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
            if self.case.use_pyTorch and self.n_chunks > 1:
                self.primal_update_reg[i1:i2] = this_primal_update_reg.detach().cpu()
            else:
                self.primal_update_reg[i1:i2] = this_primal_update_reg

            del this_primal_update_reg, this_grad_norms

    def update_primal_variables_chunk(self, i1, i2, it):
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

        # Set GD step size
        if isinstance(self.case.step, float) or isinstance(self.case.step, int):
            step = self.case.step
        else:
            if it < len(self.case.step):
                step = self.case.step[it]
            else: # Keep using last step size if not enough step sizes given
                step = self.case.step[-1]

        if self.case.use_pyTorch: # GPU operations
            if self.case.reg > 0:
                this_primal_update = - self.primal_update[i1:i2].cuda() - self.case.reg / self.M * self.preconditioner_TV * self.primal_update_reg[i1:i2].cuda()
            else:
                this_primal_update = - self.primal_update[i1:i2].cuda()
            this_primal_update *= self.preconditioner * step
        else: # CPU operations
            this_primal_update = - self.primal_update[i1:i2] - self.case.reg / self.M * self.preconditioner_TV * self.primal_update_reg[i1:i2]
            this_primal_update *= self.preconditioner * step

        if self.use_mask_no_update:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_no_update, this_primal_update.shape)
            this_primal_update[mask_4D_broadcast] = 0.0

        if self.case.use_pyTorch:
            this_x = self.x[i1:i2].cuda() + this_primal_update
        else:
            this_x = self.x[i1:i2] + this_primal_update

        this_x_avg = self.module_array.sum(this_x * self.block_corrections, axis = 1) / self.module_array.sum(self.block_corrections) # Time-integral

        if self.use_mask_zero:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_zero, this_x.shape)
            this_x[mask_4D_broadcast] = 0.0

        # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
        if self.case.use_pyTorch:
            self.x[i1:i2] = this_x.detach().cpu()
            self.x_avg[i1:i2] = this_x_avg.detach().cpu()
        else:
            self.x[i1:i2] = this_x
            self.x_avg[i1:i2] = this_x_avg

        del this_primal_update, this_x, step, this_x_avg

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
            self.res[iz] = self.C(self.x[iz], self.projector, self.ind_breakpoints)

        # 2/ Compute primal residual and dual update of fidelity term in chunks
        self.compute_update_WLS_chunk(i1, i2)

        # 3/ Compute all primal update from fidelity terms
        for iz in range(i1, i2):
            self.primal_update[iz] = self.C_T(self.res[iz], self.projector, self.ind_breakpoints)

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

        self.primal_update = self.module_array.zeros_like(self.x)
        self.primal_update_reg = self.module_array.zeros_like(self.x)

        if self.case.compute_metrics:
            self.WLS_term = 0
            self.TV_term = 0

        # 1/ Compute dual variable updates for all chunks
        for i_chunk in range(self.n_chunks):
            self.update_dual_variables_chunk(self.ind_slice_1[i_chunk], self.ind_slice_2[i_chunk], i_chunk)

        # 2/ Update primal variables in chunks
        for i_chunk in range(self.n_chunks):
            self.update_primal_variables_chunk(self.ind_slice_1[i_chunk], self.ind_slice_2[i_chunk], it)
