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

from . import algorithm

try:
    import torch
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, CPU capabilities only')

class Gradient_descent_tv(algorithm.Algorithm):
    '''
    A class used to represent the gradient descent algorithm without motion (no PLI, M = 1)
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

        # Allocate variables
        self.x = self.module_array.zeros([self.n_slices, 1, self.n_rays, self.n_rays])
        self.res = self.module_array.zeros_like(self.sino)
        self.primal_update = self.module_array.zeros_like(self.x)

        # Initialize all arrays on GPU
        if self.case.use_pyTorch:
            self.w = self.w.cuda()
            self.p = self.p.cuda()

            if self.n_chunks == 1: # If only one chunk, keep these variables on GPU (Otherwise extra cost of unnecessary GPU-CPU transfer)
                self.x = self.x.cuda()
                self.res = self.res.cuda()
                self.primal_update = self.primal_update.cuda()

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
            _, this_primal_update_reg, this_grad_norms = self.tv_fct(this_x, reg_z_over_reg=self.case.reg_z/self.case.reg, return_pytorch_tensor = self.case.use_pyTorch, return_grad_norms = True)
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
                self.TV_term += self.case.reg * self.module_array.sum(this_grad_norms)

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

         # If using matrix p also for TV term, results in non-proper convergence
        min_p = self.p.min() # The constant value in the center circle

        if self.case.use_pyTorch: # GPU operations
            if self.case.reg > 0:
                this_primal_update = -self.p * self.primal_update[i1:i2].cuda() - min_p * self.case.reg * self.primal_update_reg[i1:i2].cuda()
            else:
                this_primal_update = -self.p * self.primal_update[i1:i2].cuda()
            this_primal_update *= step
        else: # CPU operations
            this_primal_update = -self.p * self.primal_update[i1:i2] - min_p * self.case.reg * self.primal_update_reg[i1:i2]
            this_primal_update *= step

        if self.use_mask_no_update:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_no_update, this_primal_update.shape)
            this_primal_update[mask_4D_broadcast] = 0.0

        if self.case.use_pyTorch: # GPU operations
            this_x = self.x[i1:i2].cuda() + this_primal_update
        else:
            this_x = self.x[i1:i2] + this_primal_update

        if self.use_mask_zero:
            mask_4D_broadcast = self.module_array.broadcast_to(self.mask_zero, this_x.shape)
            this_x[mask_4D_broadcast] = 0.0

        # Update (These lines are responsible for CPU usage, thread-parallel array copy from numpy`)
        if self.case.use_pyTorch:
            self.x[i1:i2] = this_x.detach().cpu()
        else:
            self.x[i1:i2] = this_x

        del this_primal_update, this_x, step, min_p

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
            self.res[iz] = self.projector.fp(self.x[iz])

        # 2/ Compute residual in chunks
        self.compute_update_WLS_chunk(i1, i2)

        # 3/ Back-project the weighted residual
        for iz in range(i1, i2):
            self.primal_update[iz] = self.projector.bp(self.res[iz])

        # # 4/ Compute update of regularization term in chunks
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
