config_str = """\
[Application]
name=MagPy
version={0}
entry_point=MagPy4.MagPy4:runMagPy
console=true
icon=MagPy4/rsrc/images/magPy_blue.ico
[Python]
version=3.8.6
[Include]
packages=
    tkinter
    _tkinter
pypi_wheels = {1}
extra_wheel_sources =
    fflib
    pip_wheels
files=lib
exclude=
    MagPy4/testData\
"""

def main():
    with open('MagPy4/version.txt', 'r') as fd:
        version = fd.read().strip('\n')
    
    with open('requirements.txt', 'r') as fd:
        requirements = fd.readlines()
        requirements = '\t'.join(requirements).rstrip('\n')
    
    txt = config_str.format(version, requirements)
    with open('installer.cfg', 'w') as fd:
        fd.write(txt)

main()