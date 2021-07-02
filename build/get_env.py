import os, sys

def main():
    path = os.path.join('MagPy4', 'version.txt')
    if not os.path.exists(path):
        path = os.path.join('..', 'version.txt')

    with open(path, 'r') as fd:
        version = fd.read()
    
    env_vars = {
        'version' : version,
        'vtag_name' : f'v{version}',
        'installer_name' : f'MagPy_v{version}.exe',
        'installer_path' : f'MagPy_{version}.exe'
    }

    for key, value in env_vars.items():
        print (f'{key}={value}')

main()