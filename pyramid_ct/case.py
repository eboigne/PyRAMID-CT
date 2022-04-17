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

import pickle
import tifffile
import os
import numpy as np
import time
import shutil
import importlib
import pyramid_ct

class Case:
    '''
    A class used to define a PyRAMID case
    '''
    
    def __init__(self, path_case = None):
        '''
        Constructor method for a PyRAMID case

        Parameters
        ----------
        path_case : str
            Path of a previously run case to load
        '''

        # Default input parameters
        self.algorithm_name = 'Chambolle_Pock_tv'
        self.nb_it = 100
        self.M = 1
        self.step = 1.0
        self.reg = 0.0
        self.reg_z = 0.0
        self.reg_time = 0.0
        self.slices_per_chunk_max = 25
        self.use_pyTorch = True
        self.dtype = 'double'
        self.dtype_save = 'single'
        self.tv_scheme = 'hybrid'
        self.tv_use_pyTorch = True
        self.print_every = 50
        self.compute_metrics = True
        self.print_log_only = False
        self.print_input_variables = False
        self.skip_logger = False
        self.skip_all_disk_saves = False
        self.slices_to_probe = [0,]
        self.save_every_linear = 500
        self.save_every_log = 1.1
        self.mask_zero_path = ''
        self.mask_no_update_path = ''
        self.mask_static_path = ''
        self.factor_reg_static = 0
        self.path_save = os.getcwd()+'/PyRAMID_output/'
        self.case_name_prefix = 'test_case'
        self.case_name_suffix = ''
        self.projector_class = pyramid_ct.projector_ASTRA.Projector_ASTRA

        # Initialize variables
        self.data = {}
        self.data['obj_fct'] = [[],[],[]]
        self.data['obj_fct_steadyIts'] = [[],[],[]]
        self.data['l2_primal_update'] = []
        self.data['l2_primal_update_normalized'] = []
        self.data['l2_primal_update_steadyIts'] = []
        self.data['l2_primal_update_normalized_steadyIts'] = []
        self.table_it_save = []
        self.last_saved_iteration = 0

        # Set sinogram, which calls self.update(), sets the name and ensure case is consistent
        self.set_sinogram_data(np.zeros([1, 50, 64]))

        # If path_case provided, load case
        if path_case != None:
            self.load_case_from_file(path_case)

    def update(self):
        '''
        Method to update the case given any changes in parameters, to enforce that the case is consistent
        '''

        # Force M = 1 for static algorithms
        if not 'motion' in self.algorithm_name:
            self.M = 1

        # Set the case names from given parameters
        self.set_name()

        self.case = self # Used to pass pointer to the Case to Algorithm

        # Get metrics
        obj_fct = self.total_obj_fct()
        if len(obj_fct[0] > 0):
            self.obj_fct_WLS = obj_fct[:,0]
            self.obj_fct_TV = obj_fct[:,1]
            self.obj_fct_sum = obj_fct[:,2]
            self.l2_primal_update = self.total_l2_primal_update()
            self.l2_primal_update_normalized = self.total_l2_primal_update_normalized()
            self.n = len(self.obj_fct_sum)
            self.table_it = np.linspace(1, self.n, self.n)

        # Renew sizes from sinogram
        if type(self.sino) != type(None) :
            self.n_slices, self.n_angles, self.n_rays = self.sino.shape

        # Renew projector (n_angles and n_rays may have changed)
        try:
            self.projector.clear()
            del self.projector
        except:
            pass
        self.projector = self.projector_class(self.n_rays, self.n_angles)

        # Can't use CPU PyTV-4D with GPU PyRAMID
        if self.tv_use_pyTorch == False:
            self.use_pyTorch = False

    def set_sinogram_data(self, sino):
        '''
        Method to specify the sinogram data to reconstruct

        Parameters
        ----------
        sino : np.ndarray
            The sinogram data to reconstruct of size Nz x Na x N or simply Na x N
        '''

        self.sino = sino
        if len(self.sino.shape) == 2: # Reshape 2D input into 3D.
            self.sino = np.reshape(self.sino, [1, self.sino.shape[0], self.sino.shape[1]])
        self.update()

    def set_name(self):
        '''
        Method that updates the name of the case for consistency
        '''

        self.name = self.set_case_name()
        self.path = self.path_save + self.name + '/'
        self.path_case = self.path + self.name + '.case'
        self.path_log = self.path + self.name + '.log'

    def load_input_parameters(self, path_to_input_parameters):
        '''
        Method that loads input parameters for the case from an input.py file, or directly from a Python dictionary

        Parameters
        ----------
        path_to_input_parameters : str or dict
            A string of the path to the input.py file, or the Python dictionary of the input parameters
        '''

        if isinstance(path_to_input_parameters, dict): # If given directly the dict of input parameters
            input_parameters_dict = path_to_input_parameters
        else: # Given a path to an input.py file, imported as a module
            spec = importlib.util.spec_from_file_location('input', path_to_input_parameters)
            input_parameters_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(input_parameters_module)
            input_parameters_dict = input_parameters_module.__dict__

        # Copy input parameters into attributes of case
        self.input_parameter_keys = [e for e in input_parameters_dict.keys() if e[0:2] != '__']
        for key in input_parameters_dict:
            if key[0:2] != '__' and type(input_parameters_dict[key]) != type(np):
                self.__dict__[key] = input_parameters_dict[key]
        self.update()

    def load_case_from_file(self, path_file):
        '''
        Method that takes care of the loading of the case from a .case file

        Parameters
        ----------
        path_file : str
            A string of the path to the .case file
        '''

        name = path_file.split('/')
        if name[-1] == '': # In case path_file ends with '/'
            self.path_save = '/'.join(name[0:-2])+'/'
            self.name = name[-2]
        else:
            self.path_save = '/'.join(name[0:-1])+'/'
            self.name = name[-1]

        self.path = self.path_save + self.name + '/'
        self.path_case = self.path + self.name + '.case' # Needed to import

        self.load_case()
        self.update()

    def load_case(self):
        '''
        Method that loads the .case file
        '''

        try:
            pickle_case = open(self.path_case+'.temp','rb')
        except:
            try:
                pickle_case = open(self.path_case,'rb')
            except:
                return(False)

        # Load case from pickle
        case_loaded = pickle.load(pickle_case)
        for key in case_loaded.__dict__.keys(): # Copy attributes one by one (self = case_loaded doesn't work)
            self.__dict__[key] = case_loaded.__dict__[key]
        pickle_case.close()
        self.case = self

    def set_case_name(self):
        '''
        Method that builds the case name given the input variables

        Returns
        -------
        str
            The name of the case with added variables
        '''

        # Dictionary of variables to add within the case name. Format: [0]+'_'+[1]
        self.args_in_name = []

        if 'motion' in self.algorithm_name:
            self.args_in_name.append(['M', str(self.M)])
            self.args_in_name.append(['regTime', str('%0.3E' % self.reg_time).replace('.', 'p')])
        self.args_in_name.append(['reg', str('%0.3E' % self.reg).replace('.', 'p')])

        self.name = self.case_name_prefix
        for e in self.args_in_name:
            if e[1] != '':
                self.name = self.name + '_' + e[0] + '_' + e[1]
        self.name = self.name + self.case_name_suffix

        return(self.name)
    
    def total_obj_fct(self):
        '''
        Method that builds an array of the objective function values over the iterations

        Returns
        -------
        np.ndarray
            An array of the objective function values with 3 columns: WLS term, TV term, and WLS+TV term.
        '''

        obj_fct_1 = np.array(self.data['obj_fct_steadyIts'])[:,:]
        obj_fct_2 = np.array(self.data['obj_fct'])[:,:]
        if len(obj_fct_1[0]) > 0:
            return(np.concatenate((obj_fct_1, obj_fct_2)))
        else:
            return(obj_fct_2)
        
    def total_l2_primal_update(self):
        '''
        Method that builds an array of L2 norms of the primal update. The primal update is the chance in the solution field variable x at each iteration.

        Returns
        -------
        np.ndarray
            An array of the L2 norms of the primal update
        '''

        l2_primal_update_1 = self.data.get('l2_primal_update_steadyIts')
        l2_primal_update_2 = self.data['l2_primal_update']
        if len(l2_primal_update_1) > 0:
            return(np.concatenate((l2_primal_update_1, l2_primal_update_2)))
        else:
            return(l2_primal_update_2)
        
    def total_l2_primal_update_normalized(self):
        '''
        Method that builds an array of L2 norms of the normalized primal update. The primal update is the chance in the solution field variable x at each iteration.

        Returns
        -------
        np.ndarray
            An array of the L2 norms of the normalized primal update
        '''

        l2_primal_update_1 = self.data['l2_primal_update_normalized_steadyIts']
        l2_primal_update_2 = self.data['l2_primal_update_normalized']
        if len(l2_primal_update_1) > 0:
            return(np.concatenate((l2_primal_update_1, l2_primal_update_2)))
        else:
            return(l2_primal_update_2)
    
    def status(self):
        '''
        Method that prints the status of the current case
        '''

        self.update()

        print('Status of case: '+str(self.name))
        print('\tSetup up for using algorithm '+self.algorithm_name+' for '+str(self.nb_it)+' iterations.')
        if self.did_case_finish():
            print('\tAll iterations were performed. The last iteration is: '+str(self.last_saved_iteration))
        else:
            print('\tThe algorithm did NOT finish. The last saved iteration is: '+str(self.last_saved_iteration))
        self.print_parameters()

    def print_parameters(self):
        '''
        Method that prints the input parameters of the case
        '''

        print('\nParameters:')
        attributes_to_not_show = ['l2_primal_update', 'l2_primal_update_normalized', 'obj_fct', 'n', 'data', 'args_in_name', 'input_parameter_keys', 'time_started', 'time_ended', 'algo', 'time_taken_to_run', 'path_case', 'path_log', 'table_it_save', 'case', 'projector']
        for key in self.__dict__:
            if not (type(self.__dict__[key]) in [np.ndarray]) and key[0:2] != '__' and not key in attributes_to_not_show:
                print('\t {:<40} {:<40}'.format(key, str(self.__dict__[key])))

    def log(self):
        '''
        Method that prints the text within the logfile of the current case
        '''

        log = open(self.path_log, 'r')
        print(log.read())

    def _save(self):
        '''
        Method that saves the current case as a pickle with extension .case
        '''

        # Carry ind_breakpoints from Algorithm to Case to save
        if hasattr(self, 'algo'):
            if hasattr(self.algo, 'ind_breakpoints'):
                self.ind_breakpoints = self.algo.ind_breakpoints

        # Temporarily remove heavy data from object before saving
        vars_do_not_save = ['sino', 'algo', 'case']
        dict_save = {}
        for var in vars_do_not_save:
            dict_save[var] = self.__dict__[var]
            self.__dict__[var] = None

        with open(self.path_case+'.temp','wb') as pickle_case:
            pickle.dump(self,pickle_case)
        try:
            os.unlink(self.path_case)
        except:
            pass
        os.rename(self.path_case+'.temp', self.path_case)

        # Replace heavy data into object
        for var in vars_do_not_save:
            self.__dict__[var] = dict_save[var]

    def did_case_finish(self):
        '''
        Method that checks whether all iterations of the current case were executed
        '''

        return (self.nb_it == self.last_saved_iteration)

    def save_tiff(self, data, suffix, it, folder = ''):
        '''
        Method that saves a 2D image data as a .tif file

        Parameters
        ----------
        data : np.ndarray or Torch.Tensor
            The 2D array of the data to save in the .tif file
        suffix : str
            The suffix to the file name
        it : int
            The iteration number for the given data
        folder : str
            A path to the folder in which to save the data
        '''

        path_data = self.path + folder + '/' + suffix + '/'
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        if isinstance(data, np.ndarray):
            if self.dtype_save == 'double':
                tifffile.imsave(path_data+suffix+'_'+self.name+'_it_'+str(it).zfill(8)+'.tif', data.astype('float64'))
            else:
                tifffile.imsave(path_data+suffix+'_'+self.name+'_it_'+str(it).zfill(8)+'.tif', data.astype('float32'))
        else:
            if self.dtype_save == 'double':
                tifffile.imsave(path_data+suffix+'_'+self.name+'_it_'+str(it).zfill(8)+'.tif', np.array(data.cpu()).astype('float64'))
            else:
                tifffile.imsave(path_data+suffix+'_'+self.name+'_it_'+str(it).zfill(8)+'.tif', np.array(data.cpu()).astype('float32'))

    def save_image_data_into_tiff(self, image_data, suffix, it, folder = '', slice_table = []):
        '''
        Method that takes care of saving the image data into 2D tiff files

        Parameters
        ----------
        image_data : np.ndarray or Torch.Tensor
            An array of image data of arbitrary size that will be saved as 2D tif files
        suffix : str
            The suffix to the file name
        it : int
            The iteration number for the given data
        folder : str
            A path to the folder in which to save the data
        slice_table : list
            A list of the slice number for each slice along the first axis of the image_data
        '''

        if len(image_data.shape) == 2:
            self.save_tiff(image_data, suffix, it, folder = folder)
        else:
            for ii in range(image_data.shape[0]):
                if slice_table == []:
                    self.save_image_data_into_tiff(image_data[ii], suffix+'_'+str(ii).zfill(2), it, folder = folder)
                else:
                    self.save_image_data_into_tiff(image_data[ii], 'slice_'+str(slice_table[ii]).zfill(4)+'_'+suffix, it, folder = folder)


    def save_data(self, image_data, it, folder = '', slice_table = []):
        '''
        Method that takes care of saving the data for this case

        Parameters
        ----------
        image_data : dict
            A dictionary of arrays of image data of arbitrary size that will be saved as 2D tif files
        it : int
            The iteration number for the given data
        folder : str
            A path to the folder in which to save the data
        slice_table : list
            A list of the slice number for each slice along the first axis of the image_data
        '''

        self.table_it_save.append(it)

        if not self.skip_all_disk_saves:

            # Save image data as tiff
            for key in image_data.keys():
                self.save_image_data_into_tiff(image_data[key], key, it, folder = folder+key, slice_table = slice_table)

    def save_case(self, light_algorithm_data, it):
        '''
        Method that takes care of saving this case as a .case file

        Parameters
        ----------
        light_algorithm_data : dict
            A dictionary containing the light data from the Algorithm class used
        it : int
            The iteration number for the given data
        '''


        # Save light algorithm data in case file
        for key in light_algorithm_data.keys():
            self.data[key] = light_algorithm_data[key]
        self.last_saved_iteration = it
        self.time_ended = time.time()
        self.time_taken_to_run = self.time_ended - self.time_started

        if not self.skip_all_disk_saves:
            self._save()
        
    def clear_save(self):
        '''
        Method that clears the save folder for this case
        '''

        try:
            shutil.rmtree(self.path) # Remove everything within folder, and folder
            os.makedirs(self.path) # Rebuild folder
        except:
            pass

    def run(self):
        '''
        Method that takes care of running the execution of this case
        '''

        self.update()

        # Checking if save folder exists, and clear case
        if not self.skip_all_disk_saves:
            if os.path.exists(self.path):
                print('\tClearing previous save of this case before running')
                self.clear_save()
            else:
                os.makedirs(self.path)

        # Direct sys.stdout to both a log file and the standard ipykernel iostream
        if not self.skip_logger:
            log_file = pyramid_ct.utils.Logger(self.path_log, print_stdout = not(self.print_log_only), clear_log_file = True, skip_all_disk_saves = self.skip_all_disk_saves)

        print('\n\n\t ==================== Appending to log ====================')
        print('\nRunning case "'+str(self.name)+'"')

        if self.print_input_variables:
            self.print_parameters()
        print('')
        
        # Run algorithm
        self.time_started = time.time()
        self.algo = pyramid_ct.__dict__[self.algorithm_name.lower()].__dict__[self.algorithm_name](self) # Create an instance of the desired algorithm
        self.algo.run()
        self.time_ended = time.time()

        self.time_taken_to_run = self.time_ended - self.time_started
        print('\nThe case "'+str(self.name)+'" took: '+str(self.time_taken_to_run)+' s.')

        self.update()

        if not self.skip_all_disk_saves:
            self._save()

        if not self.skip_logger:
            log_file.close() # Redirect sys.stdout back to normal

    def load_x_avg(self, ind_iteration = -1, ind_slice = 0):
        '''
        Method that loads the time-averaged reconstructed field at a given slice and iteration

        Parameters
        ----------
        ind_iteration : int
            The index of the iteration to load
        ind_slice : int
            The index of the slice to load

        Returns
        -------
        np.ndarray
            A 2D array of the reconstructed slice of size N x N
        '''

        try:
            path_dir = self.path+'probes/x_avg/'
            path_dir = [path_dir+e+'/' for e in os.listdir(path_dir) if ('slice_'+str(ind_slice).zfill(4)) in e][0]
            list_files = os.listdir(path_dir)
            list_files.sort()
            self.rec = tifffile.imread(path_dir+list_files[ind_iteration])
        except:
            raise Exception('Failed loading reconstruction')

        return(self.rec)
        
    def load_x(self, ind_iteration = -1, ind_slice = 0):
        '''
        Method that loads the array of the reconstructed field at every breakpoints at a given slice and iteration

        Parameters
        ----------
        ind_iteration : int
            The index of the iteration to load
        ind_slice : int
            The index of the slice to load

        Returns
        -------
        np.ndarray
            A 3D array of the reconstructed slice of size M x N x N
        '''

        if not 'motion' in self.algorithm_name:
            return(self.load_x_avg(ind_iteration = ind_iteration, ind_slice = ind_slice))
        else:
            try:
                path_dir = self.path+'probes/x/'
                list_dir = [e for e in os.listdir(path_dir) if ('slice_'+str(ind_slice).zfill(4)) in e]
            except:
                raise Exception('Failed loading reconstruction')
            self.x = []
            for dir in list_dir:
                list_files = os.listdir(path_dir+dir)
                list_files.sort()
                self.x.append(tifffile.imread(path_dir+'/'+dir+'/'+list_files[ind_iteration]))
            self.x = np.array(self.x)
            return(self.x)
