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

import pyramid_ct
import pathlib
import numpy as np

def run_tests():

    sino = pyramid_ct.utils.example_moving_phantom(motion_type=2)
    path_pyramid = str(pathlib.Path(__file__).parent.resolve())

    print('Taking tests for PyRAMID-CT:\n')

    tests_taken = 0
    tests_passed = 0

    # === Chambolle Pock tests without TV ===

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.reg = 0.0
    case.reg_z = 0.0
    case.reg_time = 0.0
    case.algorithm_name = 'Chambolle_Pock_tv'
    case.M = 1
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [399.127805395475661498, 0.000000000000000000], 'CP M=1 noTV')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.reg = 0.0
    case.reg_z = 0.0
    case.reg_time = 0.0
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [318.130800606706657163, 0.000000000000000000], 'CP M=2 noTV')

    # === Chambolle Pock tests ===

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.dtype = 'single'
    case.algorithm_name = 'Chambolle_Pock_tv'
    case.M = 1
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [447.631686386942703848, 480.078247070312500000], 'CP M=1 single')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv'
    case.M = 1
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [447.631688391705210961, 480.078257459567737442], 'CP M=1')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [369.169328739952845808, 484.299159301065060390], 'CP M=2')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.mask_zero_path = ''
    case.mask_no_update_path = ''
    case.mask_static_path = ''
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [129.607359269398216384, 88.032160067014189053], 'CP M=2 unmasked')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 4
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [346.514544747496017862, 479.085655287753866105], 'CP M=4')

    # === Subgradient Descent tests ===

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Gradient_descent_tv'
    case.M = 1
    case.mask_zero_path = ''
    case.mask_no_update_path = '' # Random speckle masks makes SD unstable
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [165.582997958354155799, 89.251022338867187500], 'SD M=1')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Gradient_descent_tv_motion'
    case.M = 2
    case.mask_zero_path = ''
    case.mask_no_update_path = '' # Random speckle masks makes SD unstable
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [65.595230438636264125, 103.771781921386718750], 'SD M=2')

    # === Different TV tests ===

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'upwind' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [366.799087620057662207, 432.198625123028477901], 'CP M=2 upwindTV GPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'upwind' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    case.tv_use_pyTorch = False # Use pyTorch implementations for TV calculations (runs on GPU, faster).
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [366.799087620057662207, 432.198625123028534745], 'CP M=2 upwindTV CPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'downwind' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [366.832488346558648118, 432.046635000850187680], 'CP M=2 downwindTV GPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'downwind' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    case.tv_use_pyTorch = False # Use pyTorch implementations for TV calculations (runs on GPU, faster).
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [366.832488346558704961, 432.046635000850244523], 'CP M=2 downwindTV CPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'central' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [339.822330696223730229, 252.595542930170637419], 'CP M=2 centralTV GPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'central' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    case.tv_use_pyTorch = False # Use pyTorch implementations for TV calculations (runs on GPU, faster).
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [339.822330696223730229, 252.595542930170637419], 'CP M=2 centralTV CPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.tv_scheme = 'hybrid' # Scheme of discretization used to compute the TV. Options are 'downwind', 'upwind', 'central', 'hybrid'
    case.tv_use_pyTorch = False # Use pyTorch implementations for TV calculations (runs on GPU, faster).
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [369.169328739952902652, 484.299159301065117234], 'CP M=2 hybridTV CPU')

    # === PyRAMID CPU version ===

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv'
    case.M = 1
    case.use_pyTorch = False # Use pyTorch implementations for TV calculations (runs on GPU, faster).
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [447.631688391705154118, 480.078257459567737442], 'CP M=1 CPU')

    case = pyramid_ct.Case()
    case.load_input_parameters(path_pyramid+'/input_tests.py')
    case.algorithm_name = 'Chambolle_Pock_tv_motion'
    case.M = 2
    case.use_pyTorch = False # Use pyTorch implementations for TV calculations (runs on GPU, faster).
    tests_taken += 1
    tests_passed += compare_case_output_to_reference(case, sino, [369.169328739952902652, 484.299159301065060390], 'CP M=2 CPU')

    print('\nTests finished: PASSED '+str(tests_passed)+' / '+str(tests_taken))

def compare_case_output_to_reference(case, sino, ref_values, test_name):
    case.set_sinogram_data(sino)
    case.run()

    last_TV = case.obj_fct_TV[-1]
    last_WLS = case.obj_fct_WLS[-1]

    if case.dtype == 'single':
        precision = 1e-7
    else:
        precision = 1e-15
    pass_test = np.abs(last_WLS-ref_values[0]) < precision and np.abs(last_TV-ref_values[1]) < precision

    if pass_test:
        print('\t[PASS] Test: '+test_name)
        return(1)
    else:
        print('\t[FAIL] Test: '+test_name+'')
        print('\t\tReference: \t'+'{:.18f}'.format(ref_values[0])+'\t'+'{:.18f}'.format(ref_values[1]))
        print('\t\tObtained: \t'+'{:.18f}'.format(last_WLS)+'\t'+'{:.18f}'.format(last_TV))
        print('\t\tDifference: \t'+'{:.18f}'.format(last_WLS-ref_values[0])+'\t'+'{:.18f}'.format(last_TV-ref_values[1]))
        return(0)
