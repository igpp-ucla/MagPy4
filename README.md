# MagPy4 / MarsPy ![bird](MagPy4/images/magPy64.png)

Magnetic Field Analysis Program. Built for IGPP at UCLA.

## Installation
### OSX / Linux
0. Make sure Python>=3.6 and git are installed.
1. To install MagPy via pip, run the following command in the terminal:
   ```
   $ pip3 install https://github.com/igpp-ucla/MagPy4.git
   ```
   It may prompt you for your GitHub username/password. 

2. After this, you should be able to run MagPy4 from the terminal by simply 
   typing in `MagPy4` or `MarsPy` and pressing enter.

   __Note__: If the 'MagPy4' command is not working, then the path on your system might not be set properly. For OSX users, this will typically be under '/Library/Frameworks/Python.framework/Versions/{PYTHONVERSION}/bin', where PYTHONVERSION is the python version number you are using. Add this to your bash profile or path so the scripts may be found.


### Windows
0. To run MagPy4 on a Windows machine, both [Python >= 3.6](https://www.python.org/downloads/release/python-376/) and [Git For Windows](https://git-scm.com/download/win) 
(or alternatively, [GitHub For Windows](https://desktop.github.com/)) will need 
to be installed. Take care to make sure 'Add To Path' is checked in the first part 
of the Python installation process.

   __Note__: If these are not installed when the installation script in the
   next step is run, the installers will be downloaded and silently 
   run for you. Be sure to watch for prompts that ask whether you
   want to allow the installers to make changes to your computer.

1. From here, you will need to download the MagPy4 package as a .zip folder from 
GitHub and run the "__install_windows.bat__" script by double clicking on it. 
It may prompt you for your GitHub username and password to download the ffPy 
library from https://github.com/igpp-ucla/ffPy.git

2. To run MagPy, type `MagPy4` into your Windows search bar and press enter,
making sure that the highlighted object type says 'Run command'.

## Updates
### OSX/Linux
* Add the "--update" flag to the "MagPy4" command, i.e. "MagPy4 --update"
### Windows
* Click on the 'Update MagPy4' action under the Help menu. It should close
MagPy, run the update command in the Command Prompt, (possibly asking for your 
GitHub username/password for authentication), and then re-open MagPy.


## Required packages
The following packages are required to run MagPy
- numpy>=1.15.0
- scipy>=1.1.0
- PyQt5>=5.11.2
- pyqtgraph>=0.10.0
- PyQtWebEngine
- ffPy @ https://github.com/igpp-ucla/ffPy.git

To be able to download and plot MMS orbit data, the following packages are also needed:
- cdflib
- requests
