
# REQUIREMENTS
# python 3.6 (would prob work with earlier versions)
# paramiko package
# pyinstaller
# have a credentials.txt file defined in the same directory as this file (dont worry it wont be tracked in the repository)
    # username on first line, password on second line

import sys
import os
import shutil
import paramiko

server = 'aten.igpp.ucla.edu'

_indexFile = 'index.html'
_remotePath = '/webdocs/docs/magpy/'
_buildPath = 'builds/windows/'

def tryConnectSSH(ssh):
    if not os.path.isfile('credentials.txt'):
        print('Error: cannot find "credentials.txt" (file with username on first line and password on second line)')
        return False
    with open('credentials.txt', 'r') as file:
        user = file.readline().strip()
        pw = file.readline().strip()

    try:
        ssh.connect(server, username=user, password=pw)
    except paramiko.ssh_exception.AuthenticationException:
        print('Error: ssh authentication failed (bad username / pw)')
        return False
    except:
        print('Error: unknown ssh failure')
        return False

    return True

def printHelp():
    print('usage format: python magPyDist.py <command> <version>')
    print('')
    print('[commands]')
    print('freeze -freezes program and makes an executable')
    print('build  -compiles and makes an installer for given version')
    print('deploy -sends version to webserver')
    print('remove -deletes version from webserver')
    print('full -freezes builds and deploys version')
    print('')
	
def freeze(version):
	with open('version.rc','r') as file:
		lines = file.readlines()
	for li,line in enumerate(lines):
		if line.strip().startswith('filevers'):
			vers = 'filevers=('
			vt = version.split('.')
			for i in range(4):
				v = '0'
				if i < len(vt):
					v = vt[i]
				end = ',' if i < 3 else '),\n'
				vers = f'{vers}{v}{end}'
			lines[li] = vers
			break
	
	# write back to file
	with open('version.rc','w') as file:
		for line in lines:
			file.write(line)
	
	#print('\n'.join(lines))
	
	#remove these folders made by pyinstaller before remaking
	if os.path.exists('build'):
		shutil.rmtree('build')
	if os.path.exists('dist'):
		shutil.rmtree('dist')
	
	os.system('pyinstaller MagPy4.spec')	

	print(f'Successfully froze version {version}')

def getVersionString(version):
    vers = ''
    vt = version.split('.')
    for i in range(4):
        v = '0'
        if i < len(vt):
            v = vt[i]
        end = '.' if i < 3 else ''
        vers = f'{vers}{v}{end}'
    return vers
	
def build(version):
    #find innosetup file in the build directory
    issFile = None
    for file in os.listdir('.'):
        if file.endswith('.iss'):
            issFile = file
            break

    if not issFile:
        print('Error: cannot find InnoSetup .iss file!')
        return

    #open file
    with open(issFile, 'r') as file:
        data = file.readlines()

    version = getVersionString(version)

    #change version number
    appVersionLine = '#define MyAppVersion'
    for i, line in enumerate(data):
        if line.startswith(appVersionLine):
            data[i] = f'{appVersionLine} "{version}"\n'
            break

    #save file
    with open(issFile, 'w') as file:
        file.writelines(data)

    # run innosetup script to compile MagPy installer
    # need to add innosetup to system path for this to work!!!
    # also need to restart VS2017 once you do before it registers
    os.system('iscc ' + issFile)

    print(f'Built version {version} successfully')

	
def deploy(version):
    # find the .exe in Output folder
    version = getVersionString(version)

    fileName = None
    if not os.path.isdir('Output'):
        print('Error: cannot find Output directory')
        return
    for file in os.listdir('Output'):
        if file.endswith(f'{version}.exe'):
            fileName = file
            break

    # check to see file found
    if not fileName:
        print(f'Error: no .exe with version {version} found in Output folder!')
        return

    # open ssh session with webserver
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if not tryConnectSSH(ssh):
        return

    # get copy of index.html
    sftp = ssh.open_sftp()
    sftp.get(_remotePath+_indexFile, _indexFile)

    # edit the index to add new file entry
    with open(_indexFile, 'r') as file:
        data = file.readlines()

    # calc current date
    import datetime
    now = datetime.datetime.now()

    #double check no such entry exists yet
    entryName = fileName[:-4]
    found = False
    for line in data:
        if entryName in line:
            print('Failed: version {version} already deployed on website!')
            # cleanup
            sftp.close()
            ssh.close()
            os.remove(_indexFile)
            return

    # find and add new entry into html
    startPhrase = '<!--MMSDLStart-->'
    for i, line in enumerate(data):
        if startPhrase in line:
            htmlStr = f'<tr><td><a href="{_buildPath}{fileName}" download>{entryName}</a></td><td>{now.strftime("%m/%d/%Y")}</td></tr>\n'
            data.insert(i+1, htmlStr)
            break

    #save file
    with open(_indexFile, 'w') as file:
        file.writelines(data)

    #upload changed html
    sftp.put(_indexFile, _remotePath+_indexFile)
    #upload zip to builds folder
    sftp.put(f'Output/{fileName}', _remotePath+_buildPath+fileName)
    
    # cleanup
    sftp.close()
    ssh.close()
    os.remove(_indexFile)
    #os.remove(fileName)

    print(f'Deployed version {version} to website successfully')

def remove(version):
    # open ssh session with webserver
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if not tryConnectSSH(ssh):
        return

    # get copy of index.html
    sftp = ssh.open_sftp()
    sftp.get(_remotePath+_indexFile, _indexFile)

    # read index file
    with open(_indexFile, 'r') as file:
        data = file.readlines()

    # delete all lines with specific search phrase
    searchPhrase = version

    newData = [line for line in data if searchPhrase not in line]
    if data == newData: # if no differences then can return early
        print(f'No entries with version {version} found')
        # close stuff
        sftp.close()
        ssh.close()
        os.remove(_indexFile)
        return

    #save file
    with open(_indexFile, 'w') as file:
        file.writelines(newData)

    #upload changed html
    sftp.put(_indexFile, _remotePath+_indexFile)

    # remove matched versions in build folder as well
    for fileName in sftp.listdir(_remotePath+_buildPath):
        if searchPhrase in fileName:
            sftp.remove(_remotePath+_buildPath+fileName)
    
    # close stuff
    sftp.close()
    ssh.close()
    os.remove(_indexFile)

    print(f'Removed version {version} from website successfully')
    

def main():

    myargs = sys.argv
    #myargs = ["programName", "bnd", "0.2"] #if running through visual studio

    if len(myargs) != 3:
        printHelp();
        return
    
    command = myargs[1]
    version = myargs[2]

    if (command == 'full'):
        freeze(version)
        build(version)
        deploy(version)
    elif command == 'freeze': # puts all required files in one folder
        freeze(version)
    elif command == 'build':  # makes a single installer .exe
        build(version)
    elif command == 'deploy': # uploads to website
        deploy(version)
    elif command == 'remove':
        remove(version)    
    else:
        printHelp()

if __name__ == "__main__":
    main()
