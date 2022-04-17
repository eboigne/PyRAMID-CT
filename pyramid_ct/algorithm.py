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
import time
import sys
import tifffile
import pytv

try:
    import torch
except (ImportError, ModuleNotFoundError) as error:
    print('PyTorch not properly imported, CPU capabilities only')
    print(error)

class Algorithm:
    '''
    A class used to represent the CT algorithms
    '''
    
    def __init__(self, case):
        '''
        Constructor method for the Algorithm class

        Parameters
        ----------
        case : Case
            The parent Case object in which the Algorithm occurs
        '''

        self.case = case
        self.n_slices, self.n_angles, self.n_rays = case.sino.shape
        self.projector = case.projector
        self.nb_it = case.nb_it
        self.sino = case.sino

        self.set_dtype()
        self.table_it_save_manual = []

        # Setup PyTorch or Numpy as the module to handle array operations
        if self.case.use_pyTorch:
            self.sino = torch.as_tensor(self.sino, dtype = self.dtype)
            if self.n_slices <= self.case.slices_per_chunk_max: # If one chunk, transfer sino to GPU
                self.sino = self.sino.cuda()
            self.module_array = torch
            torch.copy = torch.clone # Such that torch.copy (doesn't exist) -> torch.clone (exists)
            def torch_as_type(tensor, dtype):
                tensor = tensor.type(dtype)
                return(tensor)
            torch.as_type = torch_as_type # Such that np.as_type (doesn't exist) -> tensor.type (exists)
        else:
            def numpy_as_type(array, dtype):
                array = array.astype(dtype)
                return(array)
            np.as_type = numpy_as_type # Such that np.as_type (doesn't exist) -> array.as_type (exists)
            self.sino = np.as_type(self.sino, self.dtype)
            self.module_array = np

        # Some setup
        self.set_tv()
        self.set_masks()
        self.prepare_chunk_indices()
        self.init_metrics()

        # Index breakpoints
        if 'ind_breakpoints' in self.case.__dict__: # Use specified breakpoints
            self.ind_breakpoints = np.array(self.case.ind_breakpoints)
        else: # Or build equidistant breakpoints
            self.ind_breakpoints = (np.linspace(0, self.n_angles-1, self.case.M)+0.5).astype('int')
            self.ind_breakpoints[-1] += 1 # Add last angle at end (Last block projector is one-line taller).

    def set_masks(self):
        '''
        Method that setups the masks from the input parameters
        '''

        if isinstance(self.case.mask_zero_path, str):
            self.use_mask_zero = self.case.mask_zero_path != ''
            if self.use_mask_zero:
                self.mask_zero = tifffile.imread(self.case.mask_zero_path).astype('bool')
            else:
                self.mask_zero = False
        else: # If mask array directly given
            self.use_mask_zero = True
            self.mask_zero = self.case.mask_zero_path.astype('bool')
        if self.case.use_pyTorch and self.use_mask_zero:
            self.mask_zero = torch.as_type(torch.Tensor(self.mask_zero), torch.bool)

        if isinstance(self.case.mask_static_path, str):
            self.use_mask_static = self.case.mask_static_path != ''
            if self.use_mask_static:
                self.mask_static = tifffile.imread(self.case.mask_static_path).astype('bool')
                self.factor_reg_static = self.case.factor_reg_static
            else:
                self.mask_static = False
                self.factor_reg_static = 0
        else: # If mask array directly given
            self.use_mask_static = True
            self.mask_static = self.case.mask_static_path.astype('bool')
            self.factor_reg_static = self.case.factor_reg_static
        if self.case.use_pyTorch and self.use_mask_static:
            self.mask_static = torch.as_type(torch.Tensor(self.mask_static), torch.bool)

        if isinstance(self.case.mask_no_update_path, str):
            self.use_mask_no_update = self.case.mask_no_update_path != ''
            if self.use_mask_no_update:
                self.mask_no_update = tifffile.imread(self.case.mask_no_update_path).astype('bool')
            else:
                self.mask_no_update = False
        else: # If mask array directly given
            self.use_mask_no_update = True
            self.mask_no_update = self.case.mask_no_update_path.astype('bool')
        if self.case.use_pyTorch and self.use_mask_no_update:
            self.mask_no_update = torch.as_type(torch.Tensor(self.mask_no_update), torch.bool)

    def set_dtype(self):
        '''
        Method that setups the dtype from the input parameters
        '''

        self.dtype = self.case.dtype
        if self.case.use_pyTorch:
            if self.dtype == 'single':
                self.dtype = torch.float32
            elif self.dtype == 'double':
                self.dtype = torch.float64
        # Numpy recognises `dtype = 'single'` or `dtype = 'double'`

    def prepare_chunk_indices(self):
        '''
        Method that prepares the indices for the splitting along z in "chunks" given the input parameters
        '''

        # Prepare chunk indices
        self.ind_slice_1 = []
        self.ind_slice_2 = []
        self.n_chunks = 0

        this_ind_slice_1 = 0
        this_ind_slice_2 = min(self.case.slices_per_chunk_max, self.n_slices)
        while this_ind_slice_1 < self.n_slices:
            # Store
            self.ind_slice_1.append(this_ind_slice_1)
            self.ind_slice_2.append(this_ind_slice_2)

            # Update
            self.n_chunks += 1
            this_ind_slice_1 += self.case.slices_per_chunk_max
            this_ind_slice_2 = min(this_ind_slice_2+self.case.slices_per_chunk_max, self.n_slices)

        # Fix ends to avoid last chunk with only a few elements
        if self.case.slices_per_chunk_max > 5 and self.n_slices > 5:
            while (self.ind_slice_2[-1] - self.ind_slice_1[-1]) < 3:
                self.ind_slice_1[-1] = self.ind_slice_1[-1] - 1

        print(' (using '+str(self.n_chunks)+' chunks) ', end ='')

    def save_case_and_probes(self, it):
        '''
        Method that initiates the saving of the case and probes at a given iteration, calling the parent Case object

        Parameters
        ----------
        it : int
            The iteration number
        '''

        self.table_it_save.append(it)
        light_data = {'obj_fct': np.array(self.obj_function_table), 'table_it_save': np.array(self.table_it_save), 'l2_primal_update': np.array(self.l2_primal_update_table), 'l2_primal_update_normalized': np.array(self.l2_primal_update_normalized_table), 'avg_time_per_it': self.total_time_run / max(1, it)}
        self.case.save_case(light_data, it)

        # Make sure not probing for non-existing slices
        slices_to_probe = [e for e in self.case.slices_to_probe if e < self.n_slices]

        if hasattr(self, 'x_avg'): # Motion algorithm
            probe_data = {'x': self.x[slices_to_probe], 'x_avg': self.x_avg[slices_to_probe]}
        else:
            probe_data = {'x_avg': self.x[slices_to_probe]}
        self.case.save_data(probe_data, it, folder = 'probes/', slice_table = slices_to_probe)

    def set_tv(self):
        '''
        A method that sets the TV functions
        '''

        # Gradient descent
        if self.case.tv_use_pyTorch:
            self.tv_module = pytv.tv_GPU
        else:
            self.tv_module = pytv.tv_CPU

        if self.case.tv_scheme == 'upwind':
            self.tv_fct = self.tv_module.tv_upwind
            self.stencil = 2
        elif self.case.tv_scheme == 'downwind':
            self.tv_fct = self.tv_module.tv_downwind
            self.stencil = 2
        elif self.case.tv_scheme == 'central':
            self.tv_fct = self.tv_module.tv_central
            self.stencil = 3
        elif self.case.tv_scheme == 'hybrid':
            self.tv_fct = self.tv_module.tv_hybrid
            self.stencil = 3
        else: # Hybrid
            print('/!\ Unrecognized TV discretization scheme, using ''hybrid'' instead')
            self.tv_fct = self.tv_module.tv_hybrid

        # Chambolle-Pock
        if 'Chambolle_Pock' in self.case.algorithm_name:
            if self.case.tv_use_pyTorch:
                self.tv_module = pytv.tv_operators_GPU
            else:
                self.tv_module = pytv.tv_operators_CPU

            if self.case.tv_scheme == 'upwind':
                self.D = self.tv_module.D_upwind
                self.D_T = self.tv_module.D_T_upwind
                self.nd_tv = 1
                self.stencil = 2
            elif self.case.tv_scheme == 'downwind':
                self.D = self.tv_module.D_downwind
                self.D_T = self.tv_module.D_T_downwind
                self.nd_tv = 1
                self.stencil = 2
            elif self.case.tv_scheme == 'central':
                self.D = self.tv_module.D_central
                self.D_T = self.tv_module.D_T_central
                self.nd_tv = 1
                self.stencil = 3
            else: # Hybrid
                self.D = self.tv_module.D_hybrid
                self.D_T = self.tv_module.D_T_hybrid
                self.nd_tv = 2
                self.stencil = 3

    def compute_SIRT_weights(self):
        '''
        A method that computes the SIRT weights for the given projector
        '''

        if self.case.use_pyTorch:
            x_w = torch.ones([self.n_rays, self.n_rays], dtype = self.dtype)
        else:
            x_w = np.ones([self.n_rays, self.n_rays], dtype = self.dtype)
        return(1.0/self.projector.fp(x_w))

    def compute_SIRT_preconditioner(self):
        '''
        A method that computes the SIRT preconditioner for the given projector
        '''

        if self.case.use_pyTorch:
            s_p = torch.ones([self.n_angles,self.n_rays], dtype = self.dtype)
        else:
            s_p = np.ones([self.n_angles,self.n_rays], dtype = self.dtype)
        return(1.0/self.projector.bp(s_p))
    
    def init_metrics(self):
        '''
        A method that initializes the metrics
        '''

        self.WLS_term = 0.0
        self.TV_term = 0.0
        self.obj_function_table = []
        self.table_it_save = []
        self.l2_primal_update_table = []
        self.l2_primal_update_normalized_table = []
        self.it_time_counter = 0
        
    def do_I_save_probes(self, it):
        '''
        Method that checks whether to save probes at the given iteration

        Parameters
        ----------
        it : int
            The iteration number

        Returns
        -------
        bool
            True indicates saving probes
        '''

        table_save_log = np.array((self.case.save_every_log)**np.linspace(0, 150)).astype('int')
        if it % self.case.save_every_linear == 0:
            return True
        elif it in table_save_log: 
            return True
        else:
            return False

    def compute_metrics(self):
        '''
        Method that computes and stores metrics
        '''

        self.obj_function = []

        if isinstance(self.WLS_term, torch.Tensor):
            self.WLS_term = self.WLS_term.cpu()
        if isinstance(self.TV_term, torch.Tensor):
            self.TV_term = self.TV_term.cpu()

        self.obj_function.append(np.array(self.WLS_term))
        self.obj_function.append(np.array(self.TV_term))
        self.obj_function.append(np.array(self.obj_function[0]+self.obj_function[1]))
        self.obj_function_table.append(self.obj_function)

        if self.case.use_pyTorch:
            self.l2_primal_update_table.append(np.array(torch.sqrt(torch.sum(torch.square(self.primal_update))).cpu()))
            self.l2_primal_update_normalized_table.append(np.array(torch.sqrt(torch.mean(torch.square(self.primal_update))).cpu()))
        else:
            self.l2_primal_update_table.append(np.array(np.sqrt(np.sum(np.square(self.primal_update)))))
            self.l2_primal_update_normalized_table.append(np.array(np.sqrt(np.mean(np.square(self.primal_update)))))

    def print_metrics(self, it, bool_force = False):
        '''
        Method that checks whether to print the metrics at the given iteration, and does so if judged appropriate

        Parameters
        ----------
        it : int
            The iteration number
        bool_force : bool
            Use to force the printing, whatever the iteration
        '''

        if (bool_force or (it % self.case.print_every == 0)):
            if self.case.compute_metrics:
                print("[%d] \t Total %e \t WLS %e \t TV %e \t Time [s] %.1f \t Avg time/it [ms] %.3f" %(it,self.obj_function_table[-1][2], self.obj_function_table[-1][0], self.obj_function_table[-1][1], self.total_time_run, self.total_time_run / max(1, it-2) * 1000))
            else:
                print("[%d] \t Time [s] %.1f \t Avg time/it [ms] %.3f" %(it, self.total_time_run, self.total_time_run / max(1, it-2) * 1000))

    def return_data(self):
        '''
        Method that returns the metric data

        Returns
        -------
        dict
            A dictionary of the metrics data
        '''

        out_data = {}
        out_data['obj_fct'] = np.array(self.obj_function_table)
        out_data['l2_primal_update'] = np.array(self.l2_primal_update_table)
        out_data['l2_primal_update_normalized'] = np.array(self.l2_primal_update_normalized_table)
        return out_data

    def update_counter(self, it):
        '''
        Method that updates the counter used to time the algorithm iterations

        Parameters
        ----------
        it : int
            The iteration number
        '''

        if it < 2: # Start counter at third it, to avoid counting setup-time in cost per it.
            self.tic = time.time()
        else:
            self.it_time_counter+=1
            self.toc = time.time()
            self.total_time_run += self.toc - self.tic
            self.tic = time.time()

    def run(self):
        '''
        Method that runs the algorithm
        '''

        # Initialize metrics
        self.init_metrics()
        sys.stdout.flush()
        
        # Compute SIRT weight and preconditioner
        print('Computing SIRT weights and preconditioner', end = '')
        self.w = self.compute_SIRT_weights()
        self.p = self.compute_SIRT_preconditioner()
        print('\tDone')
        
        # Initialize primal & dual variables
        print('Initialization', end = '')
        self.initialize_variables()

        print('\tDone')
        print('\nStarting iterations:')

        self.total_time_run = 0
        for it in range(self.nb_it):

            if self.do_I_save_probes(it): # Check if it's time to save
                self.save_case_and_probes(it)
                sys.stdout.flush() # Not at every it, otherwise stops multithreading and slows down a lot.

            self.do_one_iteration(it)

            self.update_counter(it) # update counter used to estimate avg_time_per_it

            # Compute and print metrics
            if self.case.compute_metrics:
                self.compute_metrics()

            self.print_metrics(it+1)

        self.print_metrics(self.nb_it, bool_force = True)
        self.save_case_and_probes(self.nb_it)

    def build_motion_interpolation_coeffs(self, ind_breakpoints, n_angles, k):
        '''
        Method that builds the time interpolation coefficients for the motion algorithms at the given block k

        Parameters
        ----------
        ind_breakpoints : np.ndarray
            The array of the breakpoint indices
        n_angles : int
            The number of angles
        k : int
            The number of the block considered

        Returns
        -------
        np.ndarray or torch.Tensor
            The array of the weights for the k^th block
        '''

        M = len(ind_breakpoints)

        t = np.linspace(0, np.pi, n_angles, False) # a table of the times t_a for angle #a. Size is [self.nAng,]. Time is equal to angle for linear sampling (exluding pi entry).
        if self.case.use_pyTorch:
            t = torch.as_tensor(t, dtype = self.dtype).cuda()
        t = self.module_array.as_type(t, self.dtype)

        if k == M-2: # Last block
            ta = t[ind_breakpoints[k]:ind_breakpoints[k+1]]
            t_a_k = t[ind_breakpoints[k]]
            t_a_kp1 = t[ind_breakpoints[k+1]-1]
        else:
            ta = t[ind_breakpoints[k]:ind_breakpoints[k+1]]
            t_a_k = t[ind_breakpoints[k]]
            t_a_kp1 = t[ind_breakpoints[k+1]]

        w_k = (ta-t_a_k)/(t_a_kp1-t_a_k)
        w_k = self.module_array.reshape(w_k, [w_k.shape[0] , 1])
        return(w_k)

    def correction_factor_angles_per_block(self):
        '''
        Method that builds the array of the correction factor for the angles (used in $L_M$ preconditioner)

        Returns
        -------
        np.ndarray or torch.Tensor
            The array of the correction factors
        '''

        n_angles_in_block = []
        for ii in range(self.case.M-1): # Rebalancing the regularization update w.r.t block fidelity term
            n_angles_in_block.append(self.ind_breakpoints[ii+1] - self.ind_breakpoints[ii])
        n_angles_in_block = np.reshape(np.array(n_angles_in_block), [self.case.M-1, 1])
        update_multiplier = self.module_array.zeros([self.case.M,1], dtype = self.dtype)
        update_multiplier[0:-1] += n_angles_in_block
        update_multiplier[1:] += n_angles_in_block

        update_multiplier /= self.module_array.sum(update_multiplier) # Normalization to keep sum equal to 1.0.

        if self.case.use_pyTorch:
            update_multiplier = update_multiplier.cuda()
        return(update_multiplier)

    def C(self, x, projector, ind_breakpoints):
        '''
        Method that calculates the image of the operator C applied to the variable x

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Primal variable of dimensions M x N x N
        projector : Projector
            The Projector object used
        ind_breakpoints : np.ndarray
            The array of breakpoint indices

        Returns
        -------
        np.ndarray or torch.Tensor
            The array of the image of x by C of dimensions Na x N
        '''

        M = x.shape[0]
        C_x = self.module_array.zeros([projector.n_angles, projector.n_rays], dtype = self.dtype)
        projector.setup_array_projectors(ind_breakpoints)

        if self.case.use_pyTorch:
            C_x = C_x.cuda()

        for k in range(M-1): # Only M-1 projectors

            # Compute required forward projections
            A_k_x_k = projector.fp_array(x[k,:,:], k)
            A_k_x_kp1 = projector.fp_array(x[k+1,:,:], k)

            w_k = self.build_motion_interpolation_coeffs(ind_breakpoints, projector.n_angles, k)

            if self.case.use_pyTorch:
                A_k_x_k = A_k_x_k.cuda()
                A_k_x_kp1 = A_k_x_kp1.cuda()

            C_x[ind_breakpoints[k]:ind_breakpoints[k+1], :] = (1-w_k)*A_k_x_k+w_k*A_k_x_kp1

        if k == M-2 and self.case.use_pyTorch:
            del A_k_x_k, A_k_x_kp1

        return(C_x)

    def C_T(self, y, projector, ind_breakpoints):
        '''
        Method that calculates the image of the operator C_T applied to the variable y

        Parameters
        ----------
        y : np.ndarray or torch.Tensor
            Dual variable of dimensions Na x N
        projector : Projector
            The Projector object used
        ind_breakpoints : np.ndarray
            The array of breakpoint indices

        Returns
        -------
        np.ndarray or torch.Tensor
            The array of the image of y by C_T of dimensions M x N x N
        '''

        M = len(ind_breakpoints)
        projector.setup_array_projectors(ind_breakpoints)
        n_rays = projector.n_rays
        x = self.module_array.zeros([M, n_rays, n_rays], dtype = self.dtype)

        if self.case.use_pyTorch:
            x = x.cuda()
            y = y.cuda()

        projector.setup_array_projectors(ind_breakpoints)

        for k in range(1,M-1):
            w_k = self.build_motion_interpolation_coeffs(ind_breakpoints, projector.n_angles, k)
            w_km1 = self.build_motion_interpolation_coeffs(ind_breakpoints, projector.n_angles, k-1)
            x[k,:,:] = projector.bp_array((1-w_k) * y[ind_breakpoints[k]:ind_breakpoints[k+1],:], k) + projector.bp_array(w_km1 * y[ind_breakpoints[k-1]:ind_breakpoints[k],:], k-1)

        # k = 0 case (first)
        w_k = self.build_motion_interpolation_coeffs(ind_breakpoints, projector.n_angles, 0)
        x[0,:,:] = projector.bp_array((1-w_k) * y[ind_breakpoints[0]:ind_breakpoints[1],:], 0)

        # k = M-1 case (last)
        w_km1 = self.build_motion_interpolation_coeffs(ind_breakpoints, projector.n_angles, M-2)
        x[M-1,:,:] = projector.bp_array(w_km1 * y[ind_breakpoints[M-2]:ind_breakpoints[M-1],:], M-2)

        return(x)
