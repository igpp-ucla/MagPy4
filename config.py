config_template = """\
[Application]
name = MagPy
version = {version}
entry_point = MagPy4.MagPy4:runMagPy
console = true
icon = MagPy4/rsrc/images/magPy_blue.ico
[Python]
version = 3.8.6
[Include]
packages =
    tkinter
    _tkinter
files =
    lib
    poppler
pypi_wheels =
{pypi_wheels_block}
extra_wheel_sources =
    fflib
    pip_wheels
exclude=
    MagPy4/testData\
"""

def main():
    with open('MagPy4/version.txt') as f:
        version = f.read().strip()
        
    reqs = []
    with open('requirements.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '-e')):
                continue
            reqs.append(line)

    pypi_wheels_block = "\n".join(f"    {req}" for req in reqs)

    cfg = config_template.format(
        version=version,
        pypi_wheels_block=pypi_wheels_block
    )
    with open('installer.cfg', 'w') as f:
        f.write(cfg)

if __name__ == '__main__':
    main()
