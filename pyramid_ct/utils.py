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

import sys, os
import tifffile
import pathlib

def path_phantom_mask_static():
    path_pyramid = str(pathlib.Path(__file__).parent.resolve())
    return(path_pyramid+'/media/phantom_mask_static.tif')

def example_moving_phantom(motion_type = 0):
    '''
    Class that serves as a tee-junction to duplicate the output of sys.stdout into a log file.
    See: https://stackoverflow.com/q/616645
    '''
    return(tifffile.imread(os.path.join(os.path.dirname(__file__), 'media','sino_phantom_motionType'+str(motion_type)+'.tif')))

class Logger(object):
    '''Class that serves as a tee-junction to duplicate the output of sys.stdout into a log file.
    See: https://stackoverflow.com/q/616645
    '''

    def __init__(self, filename, print_stdout = True, clear_log_file = True, flush_after_write = True, mode='a', buff=1, skip_all_disk_saves = False):
        self.stdout = sys.stdout
        self.skip_all_disk_saves = skip_all_disk_saves
        self.flush_after_write = flush_after_write
        self.print_stdout = print_stdout

        if not self.skip_all_disk_saves:
            self.file = open(filename, mode, buff)
            if clear_log_file: # Clear the log file before writing
                self.file.truncate(0)

        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        '''Branch the sys.stdout.write method into two separate calls'''

        if self.print_stdout:
            self.stdout.write(message)

        if not self.skip_all_disk_saves:
            self.file.write(message)

        if self.flush_after_write:
            self.flush()

    def flush(self):
        '''Branch the sys.stdout.flush method into two separate calls'''

        # Firstly, flush internal buffers
        if self.print_stdout:
            self.stdout.flush()

        if not self.skip_all_disk_saves:
            self.file.flush()

            # Second, sync all internal buffers associated with the file object with disk (force write of file)
            os.fsync(self.file.fileno())

    def close(self):
        '''Stop branching the iostream into the logfile'''

        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if not self.skip_all_disk_saves:
            if self.file != None:
                self.file.close()
                self.file = None

if __name__ == '__main__':

    log = Logger('test.log', clear_log_file = True)
    print('1: Printed both in sys.stdout and log file')

    log.close()
    print('2: Printed only in sys.stdout')
