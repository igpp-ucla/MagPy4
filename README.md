# MagPy4 ![bird](MagPy4/images/magPy64.png)

Magnetic Field Analysis Program. Built for IGPP at UCLA.

## Installation
### OSX / Linux
1. To install MagPy via pip, run the following command in the terminal:
```
$ pip3 install https://github.com/igpp-ucla/MagPy4
```
   It may prompt you for your GitHub username/password. 

2. After this, you should be able to run MagPy4 from the terminal by simply typing in
`$ MagPy4` and pressing enter.

#### Note:
If the 'MagPy4' command is not working, then the path on your system
might not be set properly. Typically, this will be under '/Library/Frameworks/Python.framework/Versions/{PYTHONVERSION}/bin', where PYTHONVERSION is the python version number you are using. Add this to your bash profile or path so the scripts may be found.


### Windows
1. To install MagPy4 on a Windows machine, both [Python >= 3.6](https://www.python.org/downloads/release/python-376/) and [Git
For Windows](https://git-scm.com/download/win) (or alternatively, [GitHub For Windows](https://desktop.github.com/)) will need to be
installed. Take care to make sure 'Add To Path' is checked in the
first part of the Python installation process.

   Note: If these are not installed when the installation script in the
   next step is run, the installers will be downloaded and silently 
   run for you. Be sure to watch for prompts that ask whether you
   want to allow the installers to make changes to your computer.

2. From here, you will need to download the MagPy4 package as a .zip
folder from GitHub and run the 'install_windows.bat' batch script
by double clicking on it. It may prompt you for your GitHub username
and password to download the ffPy library from the private repository.

3. To run MagPy, type `MagPy4` into your Windows search bar and press enter,
making sure that the highlighted object type says 'Run command', as opposed to 'Folder'.

## Required packages
The following packages are required to run MagPy
- numpy>=1.15.0
- scipy>=1.1.0
- PyQt5>=5.11.2
- pyqtgraph>=0.10.0
- ffPy @ https://github.com/igpp-ucla/ffPy.git

To be able to plot MMS orbit data, the following packages are also needed:
- cdflib
- requests
