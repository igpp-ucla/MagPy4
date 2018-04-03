# -*- mode: python -*-

import os
import sys
import shutil

block_cipher = None

a = Analysis(['MagPy4.py'],
             pathex=['C:\\Users\\Bucky\\Dropbox\\workspace\\python\\MagPy4',
					 'C:\\Users\\Bucky\\Dropbox\\workspace\\python\\MagPy4\\ffPy'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
	
print('--------------------')
for s in a.scripts:
	print(s)
print('--------------------')
for p in a.pure:
	print(p)
print('--------------------')

#for file in os.listdir('ffPy/'):
#	if file.endswith('.py') or file.endswith('.dat'):
#		path = os.path.abspath(os.path.join('ffPy/',file))
#		print(f'{file} {path}')
#		a.pure.append((file, path, 'PYMODULE'))

#a.pure.append(('pytz','pytz','PYMODULE'))

#exeDir = f"build/{os.listdir('build')[0]}/"
		
# also manually added images, mmstestdata

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
		  version='version.rc',
		  icon='images/magPy_blue.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='MagPy4')
			   
exeDir = "dist/MagPy4/"
shutil.copy2("ffPy/tai-utc.dat", exeDir)
shutil.copytree("images/", f'{exeDir}images')
shutil.copytree("styles/", f'{exeDir}PyQt5/Qt/plugins/styles')