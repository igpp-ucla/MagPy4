import os
from os import path
from appdirs import user_data_dir

os.environ['QT_MAC_WANTS_LAYER'] = '1'

# Function used to get absolute path to data/image files and directories
def getRelPath(relPath='', directory=False):
	absPath = path.join(path.dirname(__file__), relPath)
	if directory:
		absPath = absPath + path.sep
	return absPath

def get_version(file_path):
	with open(file_path, 'r') as fd:
		return fd.readline().strip('\n')

# Get package version and date
version_path = getRelPath('version.txt')
MAGPY_VERSION = get_version(version_path)

# Initialize user data directory
appname='MagPy'
appauthor='IGPP_UCLA'
USERDATALOC=user_data_dir(appname, appauthor)
