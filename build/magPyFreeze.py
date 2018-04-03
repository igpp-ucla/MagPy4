# A simple setup script to create an executable using PyQt4. This also
# demonstrates the method for creating a Windows executable that does not have
# an associated console.
#
# PyQt4app.py is a very simple type of PyQt4 application
#
# Run the build process by running the command 'python magPyFreeze.py build'
#
# If everything works well you should find a new folder with the build in it

def main():
	import os
	import sys
	import shutil
	from cx_Freeze import setup, Executable

	if len(sys.argv) != 3:
		print("Please specify version number as third argument")
		print("Call like 'python magPyFreeze.py build <version_number>")
		return

	version = sys.argv[2];
	sys.argv = sys.argv[0:2]; # take out version so cxFreeze doesnt get mad

	#this makes it so there is no console, disabling for now..
	#if sys.platform == "win32":
		#base = "Win32GUI"
	base = None

	iconLocation = "../images/magPy_blue.ico"

	#if os.path.exists(buildFolder):
	#	shutil.rmtree(buildFolder)

	#includes = ["atexit","re","zlib"]
	includes = []
	include_dirs = ["ffPy",""]
	include_dirs = ['../'+s for s in include_dirs]	#incase i need to move back to main folder and into magPy folder
	packages = ["numpy","pyqtgraph","tqdm","pytz"]
	excludes=["matplotlib"] # some packages are incorrectly added ?
	constants = ["version="+str(version)] # add version to build constants so app can know of it (only way that worked)

	zip_includes = []
	allowed_types = ['py']

	#zip together included directories into the .exe itself
	for p in include_dirs:
		for file in os.listdir(p):
			ext = file.rsplit('.',1)
			if len(ext)>1 and ext[1] in allowed_types:
				zip_includes.append(os.path.abspath(p+"/"+file))

	options = {"includes" : includes,
			   "excludes" : excludes,
			   "packages" : packages, 
			   "zip_includes" : zip_includes, 
			   "include_msvcr" : True, # includes vc++ redistributable incase they dont have it
			   "constants" : constants } 
			   
	target = Executable(
		script = "../MagPy4.py",
		targetName = "MagPy4.exe",
		base = base,
		icon = iconLocation)

	setup( # these attribs only get set if you have pywin32 installed (also pyqt app still cant detect version and stuff from this)
		name = "MagPy4", 
		version = version,
		description = "Magnetic Field Analysis Program",
		options = {"build_exe" : options},
		executables = [target])


	# move leap second table and examples into Resources folder
	exeDir = f"build/{os.listdir('build')[0]}/"

	shutil.copy2("../ffPy/tai-utc.dat", exeDir)
	
	shutil.copytree("../images", f'{exeDir}/images')

	#shutil.copytree("../magPy/BX_examples", resourceDir+"/BX_examples")
	
	sys.exit()
	
	shutil.copy2("../FFMMSMag.cfg", resourceDir)

	imageDir = resourceDir + "images/";
	os.makedirs(imageDir)
	shutil.copy2(iconLocation, imageDir)


if __name__ == "__main__":
    main()