# MagPy4 / MarsPy ![bird](MagPy4/rsrc/images/magPy64.png)

Magnetic Field Analysis Program. Built for IGPP at UCLA.

## Installation
### OSX / Linux
0. Make sure Python>=3.6 and git are installed.
1. To install MagPy via pip, run the following command in the terminal:
   ```
   $ pip3 install git+https://github.com/igpp-ucla/MagPy4.git
   ```
   It may prompt you for your GitHub username/password. 

2. After this, you should be able to run MagPy4 from the terminal by simply 
   typing in `MagPy4` or `MarsPy` and pressing enter.

   __Note__: If the 'MagPy4' command is not working, then the path on your system might not be set properly. For OSX users, this will typically be under '/Library/Frameworks/Python.framework/Versions/{PYTHONVERSION}/bin', where PYTHONVERSION is the python version number you are using. Add this to your bash profile or path so the scripts may be found.


### Windows
1. Download the 'MagPy_installer.zip' file from the [latest release](https://github.com/igpp-ucla/MagPy4/releases/latest) under the 'Releases' section.
2. Unzip the folder and run the .exe installer within it. An option for 'MagPy' should be available in the Start Menu after it finishes the installation process.

## Updates
### OSX/Linux
* Add the "--update" flag to the "MagPy4" command, i.e. "MagPy4 --update"
### Windows
* Download the latest installer and run it again. This will overwrite the previous installation.

## Command line
Analysis tools can be run without the GUI with the `MagPy4-cli` command (open a new terminal after installing):

    MagPy4-cli --list-tools
    MagPy4-cli data.csv dynwave --vars Bmso --plot-type azimuth -o figure.png
    MagPy4-cli data.csv spectra-wave --vars bx by bz --window 2026-05-15T00:10 +10m -f csv

Missing options are asked for interactively. Run `MagPy4-cli -h` for details, or `MagPy4-cli FILE TOOL -h` for a tool's options.

If the command is not found (for example in an older Windows install), run it through MagPy's own Python instead, adjusting the folder to where MagPy was installed:

    "%LOCALAPPDATA%\Programs\MagPy\Python\python.exe" -m MagPy4.cli --list-tools

## Resources
Additional information is available at https://igpp-ucla.github.io/magpy-docs/
