# -*- mode: python -*-
#using dev version PyInstaller3.4 as it fixed problem with scipy import (if 3.4 or later is out by time someone else uses this code then just go with that)
#pip install https://github.com/pyinstaller/pyinstaller/tarball/develop

import os
import sys
import shutil

block_cipher = None

os.chdir('..') # move one dir up out of build folder
workDir = os.getcwd()

a = Analysis(['MagPy4.py'],
             pathex=[workDir, f'{workDir}\\ffPy', f'{workDir}\\cdfPy'],
             binaries=[],
             datas=[],
             hiddenimports=['scipy','scipy.fftpack','PyQt5.sip'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib'], 
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='MarsPy',
          debug=False,
          strip=False,
          upx=True,
          console=True,
		  version='build/version.rc',
		  icon='images/mars.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='MarsPy')
			   
exeDir = "build/dist/MarsPy/"

shutil.copy2("ffPy/tai-utc.dat", exeDir)
shutil.copy2("README.md", exeDir)
shutil.copytree("images/", f'{exeDir}images')

os.makedirs(f'{exeDir}testData/')
	
shutil.copy2('testData/mms15092720.ffh',f'{exeDir}testData/mms15092720.ffh')
shutil.copy2('testData/mms15092720.ffd',f'{exeDir}testData/mms15092720.ffd')

shutil.copytree('testData/BX_examples/',f'{exeDir}testData/BX_examples/')
shutil.copytree('testData/insight/', f'{exeDir}testData/insight/')