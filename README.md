# MagPy4 ![bird](magPy/images/magPy64.png)

Generic plotting and analysis tool for flatfile and cdf data

IGPP, UCLA

### Required packages

At the time of writing this (8-3-2018) all packages are using latest versions, however I will add version numbers just incase bugs occur in future versions.

- python 3.6
- numpy 1.15.0
- scipy 1.1.0
- pytz 2018.5
- PyQt5 5.11.2
- pyqtgraph 0.10.0

### Installing the CDF C Library
[Download link](https://spdf.sci.gsfc.nasa.gov/pub/software/cdf/dist/cdf36_4/)

Open the folder corresponding to your operating system
Then download is named 'cdf36_4_0-setup-64.exe' for Windows, it is probably named similarly for other operating systems.

### Using the build script
The script is 'magPyDist.py' located under the /build folder
Requirements (could probably work with earlier versions too)
- python 3.6
- pyinstaller (for freezing)
- [Inno Setup](http://www.jrsoftware.org/isinfo.php) needs to be installed (for building)
- paramiko (ssh package)
- For deployment need to have a file called 'credentials.txt' defined in same folder as script, username on first line, password on second line. Used for connecting to the server through SSH and SFTP.
