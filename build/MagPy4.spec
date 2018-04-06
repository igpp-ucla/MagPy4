# -*- mode: python -*-

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
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib','scipy'], 
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
shutil.copytree("build/styles/", f'{exeDir}PyQt5/Qt/plugins/styles')

fname = 'mms15092720'
dpath = 'mmsTestData/L2/merged/2015/09/27/'
os.makedirs(f'{exeDir}{dpath}')
shutil.copy(f'{dpath}{fname}.ffh', f'{exeDir}{dpath}{fname}.ffh')
shutil.copy(f'{dpath}{fname}.ffd', f'{exeDir}{dpath}{fname}.ffd')