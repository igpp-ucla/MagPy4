import sys

import os

import shutil
from distutils import dir_util

# Get python path
py_path = os.path.dirname(sys.executable)

# Copy 'tcl' directory files to lib
tcl_dir = os.path.join(py_path, 'tcl')
dir_util.copy_tree(tcl_dir, 'lib')

# Copy DLLs into pynsist_pkgs directory
dll_files = ['_tkinter.pyd', 'tcl86t.dll', 'tk86t.dll']
dll_dir = os.path.join(py_path, 'DLLs')
for file in dll_files:
    path = os.path.join(dll_dir, file)
    shutil.copy(path, 'pynsist_pkgs')

# Copy _tkinter.lib to pynsist_pkgs
path = os.path.join(py_path, 'libs', '_tkinter.lib')
shutil.copy(path, 'pynsist_pkgs')