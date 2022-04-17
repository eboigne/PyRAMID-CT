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

import os
import numpy as np

# Algorithm options:
# algorithm_name = 'Gradient_descent_tv' # Algorithm without any motion model (no PLI)
algorithm_name = 'Gradient_descent_tv_motion' # Algorithm with motion model (PLI)
M = 2 # Number of breakpoints for the piece-wise linear interpolation in time
nb_it = 10000 # Number of iterations

# Options for output folder
path_save = os.getcwd()+'/PyRAMID_output/' # Path where to save the solution folder
case_name_prefix = 'phantom_'+algorithm_name # Prefix of the output folder
case_name_suffix = '' # Suffix of the output folder

# Regularization parameters
reg = pow(2.0, -4) # (\lambda) Main regularization parameter
reg_z = reg # (\lambda_z) Regularization in the z-direction (z = axis of CT rotation)
reg_time = pow(2.0,-5) # Regularization in time (\mu)

# Directly specify the index of the breakpoints:
# Reqs: first is 0, last is n_angles, and length is M
# ind_breakpoints = [0, 100, 150, 200]
# ind_breakpoints = [0, 90, 91, 200]

# Slices options:
slices_to_probe = [0,] # Indices of the slices that are saved along z
slices_per_chunk_max = 100 # Maximum number of slices processed at once in memory.

# =========== Step size options ===========

# Option 1: Fixed step size.
step = 1.0 * np.ones([nb_it])

# Option 2: Directly provide the learning rates as an array. If array too small, will keep using the last value.
# step_init = 1.0
# step = step_init * np.ones([nb_it])
# n_it_decrease_step = 100
# alpha = 0.5 # Between 0.5 and 1.0. At 1.0: decrease step faster
# for it in range(n_it_decrease_step, nb_it):
#     # Theoretically (to guarantee convergence of the sub-gradient descent),
#     # steps must be defined such that \sum_i step_i = \infty, while \sum_i step_i^2 < \infty
#     step[it] *= (step_init * n_it_decrease_step / it) ** alpha

# =========== Advanced options ===========

# For 'motion' algorithms, specify:
steady_iterations_first = 200 # Number of 'steady' iterations performed first before allowing dynamic motion.
steady_iterations_first_reg = reg # TV regularization parameter to use in the 'steady' iterations

# TV options
tv_scheme = 'hybrid' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
tv_use_pyTorch = True # Use pyTorch implementations for TV calculations (runs on GPU, faster).

# Float-point precision
dtype = 'double' # Recommended: 'double' for final results. Use 'single' for faster prototyping. 'single' or 'double' for 32-bit and 64-bit floats.
dtype_save = 'single' # Recommended: 'single'. dtype for saving output TIFF image data

# Algorithm options
use_pyTorch = True # Recommended: True. Adjust slices_per_chunk_max to fit with GPU memory (can check nvidia-smi during run)

# Control log and metrics. Metrics: objective function terms and primal update.
print_every = 1000 # Log output (metrics) is printed every X iterations
compute_metrics = True # Whether to compute metrics (done at every it, slight loss of performance)
print_input_variables = False # Whether to print a summary of the input variables before starting each algorithm
skip_logger = True # Whether to add a logger Tee junction to split the log both in the console and log file
print_log_only = False # Whether to print the outputs in the log file only, or also in the console
skip_all_disk_saves = False # Whether to skip all saves on disk

# Probes are saved according to:
save_every_linear = 500 # Eg: for 500, images #500, #1000, ... are saved.
save_every_log = 1.1 # Eg: for 1.2: If image #100 is saved, then image #120 is also saved.

# Specify masks to adjust the algorithm
mask_zero_path = '' # Path of a 2D mask used to enforce 0 in the solution
mask_no_update_path = '' # Path of a 2D mask used to enforce no update in the solution
mask_static_path = '' # Path of a 2D mask used to enforce a static region in the solution
factor_reg_static = 500 # The reg_time in the mask_static_path is multiplied by this constant. Eg: set it to 500 to force this region to remain static (i.e. no variation in time).



