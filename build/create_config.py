import requests

# Libraries that aren't properly found on PyPI
special_libs = ['scipy', 'matplotlib', 'numpy', 'pillow', 'cdflib', 'PyQt5-sip']

# Get MagPy version number
with open('MagPy4/version.txt', 'r') as fd:
    lines = fd.readlines()
    version = lines[0].strip('\n')

# Update version number in file
with open('installer.cfg', 'r+') as fd:
    lines = fd.readlines()
    fd.seek(0)
    lines[2] = f'version={version}\n'
    fd.write(''.join(lines))
