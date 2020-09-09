from os import path

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

