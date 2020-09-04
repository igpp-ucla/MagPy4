from os import path
# Function used to get absolute path to data/image files and directories
def getRelPath(relPath='', directory=False):
	absPath = path.join(path.dirname(__file__), relPath)
	if directory:
		absPath = absPath + path.sep
	return absPath