import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
import os

with open('README.md', 'r') as fh:
	long_description = fh.read()

with open('MagPy4/version.txt', 'r') as fh:
	versionNum = fh.read()

setuptools.setup(
	name='MagPy4',
	version=versionNum,
	author='UCLA/IGGP',
	author_email='',
	description='Magnetic Field Analysis Program',
	long_description=long_description,
	url='https://github.com/igpp-ucla/MagPy4',
	install_requires=['fflib @ git+https://github.com/igpp-ucla/fflib.git',
		'numpy>=1.15.0', 'scipy>=1.1.0', 'pyqtgraph>=0.11.0',
		'PyQt5>=5.15.0', 'cdflib', 'PyQtWebEngine',
		'requests', 'python-dateutil', 'matplotlib'],
	packages=['MagPy4', 'MagPy4/geopack/geopack'],
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent"
	],
	python_requires='>=3.6',
	include_package_data=True,
	entry_points={
		'console_scripts': [
			'MagPy4=MagPy4.MagPy4:runMagPy',
			'MarsPy=MagPy4.MagPy4:runMarsPy',
		],
	},
)
