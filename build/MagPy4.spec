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
             pathex=[workDir, f'{workDir}\\ffPy'],
             binaries=[],
             datas=[],
             hiddenimports=['scipy','scipy.fftpack'],
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
          name='MagPy4',
          debug=False,
          strip=False,
          upx=True,
          console=True,
		  version='build/version.rc',
		  icon='images/magPy_blue.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='MagPy4')
			   
exeDir = "build/dist/MagPy4/"
shutil.copy2("ffPy/tai-utc.dat", exeDir)
shutil.copytree("images/", f'{exeDir}images')

fname = 'mms15092720'
dpath = 'testData/'
os.makedirs(f'{exeDir}{dpath}')
shutil.copy(f'{dpath}{fname}.ffh', f'{exeDir}{dpath}{fname}.ffh')
shutil.copy(f'{dpath}{fname}.ffd', f'{exeDir}{dpath}{fname}.ffd')

exampDir = 'testData/BX_examples/'
shutil.copytree(exampDir,f'{exeDir}{exampDir}')