
#usage format: python magPyDist.py <command> <version>
#commands
    #build  -compiles and makes an installer for given version
    #deploy -sends version to webserver
    #remove -deletes version from webserver
    #bnd    -builds and deploys version to webserver

# given a version number it will trigger innosetup to compile a new installer
# it will find the .exe in the build/Output folder with according version and make a .zip out of it
# then it will connect to the webserver and upload it to the build folder
# and add a entry to it in the html file with the current date

# REQUIREMENTS
# need to have followed README_WindowsBuildTutorial already and have Innosetup installed with an .iss file prepared in this folder
# need the paramiko package installed, i did it using python 3.6 so there might be problems if you use 3.4 but hopefully not
# have a credentials.txt file defined in the same directory as this file (dont worry it wont be tracked in the repository)
    # username on first line, password on second line

import sys
import os
import paramiko

server = 'aten.igpp.ucla.edu'

_entryName = "MagPy4_Windows_v"
_indexFile = "index.html"
_remotePath = "/webdocs/docs/magpy/"
_buildPath = "builds/windows/"

def tryConnectSSH(ssh):
    if not os.path.isfile("credentials.txt"):
        print('Error: cannot find "credentials.txt" (file with username on first line and password on second line)')
        return False
    with open("credentials.txt", 'r') as file:
        user = file.readline().strip()
        pw = file.readline().strip()

    try:
        ssh.connect(server, username=user, password=pw)
    except paramiko.ssh_exception.AuthenticationException:
        print("Error: ssh authentication failed (bad username / pw)")
        return False
    except:
        print("Error: unknown ssh failure")
        return False

    return True

def printHelp():
    print("usage format: python magPyDist.py <command> <version>")
    print("")
    print("[commands]")
    print("build  -compiles and makes an installer for given version")
    print("deploy -sends version to webserver")
    print("remove -deletes version from webserver")
    print("bnd    -builds and deploys version to webserver")
    print("")

def build(version):
    #find innosetup file in the build directory
    issFile = None
    for file in os.listdir("."):
        if file.endswith(".iss"):
            issFile = file
            break

    if not issFile:
        print("Error: cannot find InnoSetup .iss file!")
        return

    #open file
    with open(issFile, 'r') as file:
        data = file.readlines()

    #change version number
    appVersionLine = "#define MyAppVersion"
    for i, line in enumerate(data):
        if line.startswith(appVersionLine):
            data[i] = appVersionLine + ' "'+version+'"\n'
            break

    #save file
    with open(issFile, 'w') as file:
        file.writelines(data)

    # run innosetup script to compile MagPy installer
    # need to add innosetup to system path for this to work!!!
    # also need to restart VS2017 once you do before it registers
    os.system('iscc ' + issFile)

    print("Built version " + version + " successfully")

def deploy(version):
    # find the .exe in Output folder
    fileName = None
    if not os.path.isdir("Output"):
        print("Error: cannot find Output directory")
        return
    for file in os.listdir("Output"):
        if file.endswith(version+".exe"):
            fileName = file
            break

    # check to see file found
    if not fileName:
        print("Error: no .exe with version "+version+" found in Output folder!")
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
    searchPhrase = _entryName + version
    found = False
    for line in data:
        if searchPhrase in line:
            print("Failed: version " + version + " already deployed on website!")
            # cleanup
            sftp.close()
            ssh.close()
            os.remove(_indexFile)
            return

    # add .exe to .zip
    zipName = _entryName + version
    zipFile = zipName + ".zip"
    from zipfile import ZipFile
    with ZipFile(zipFile, "w") as myzip:
        myzip.write("Output/"+fileName, fileName)

    # find and add new entry into html
    startPhrase = "<!--MMSDLStart-->"
    for i, line in enumerate(data):
        if startPhrase in line:
            htmlStr = ('<tr>'
                '<td><a href="' + _buildPath+zipFile+'" download>'+zipName+'</a></td>'
                '<td>'+now.strftime("%m/%d/%Y")+'</td>'
	            '</tr>\n')
            data.insert(i+1, htmlStr)
            break

    #save file
    with open(_indexFile, 'w') as file:
        file.writelines(data)

    #upload changed html
    sftp.put(_indexFile, _remotePath+_indexFile)
    #upload zip to builds folder
    sftp.put(zipFile, _remotePath+_buildPath+zipFile)
    
    # cleanup
    sftp.close()
    ssh.close()
    os.remove(_indexFile)
    os.remove(zipFile)

    print("Deployed version " + version + " to website successfully")

def buildAndDeploy(version):
    build()
    deploy()

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
    searchPhrase = _entryName + version

    #found = False
    #for line in data:
    #    if searchPhrase not in line:
    #        newData.append(line)
    #    else:
    #        found = True;
    newData = [line for line in data if searchPhrase not in line]
    if data == newData: # if no differences then can return early
        print("No entries with version "+version+" found")
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

    print("Removed version " + version + " from website successfully")
    

def main():

    myargs = sys.argv
    #myargs = ["programName", "bnd", "0.2"] #if running through visual studio

    if len(myargs) != 3:
        printHelp();
        return
    
    command = myargs[1]
    version = myargs[2]
    if version == "0.1":
        print("Error: dont mess with the first version you heathen")
        #return

    if (command == "bnd" or
        command == "buildAndDeploy"):
        build(version)
        deploy(version)
    elif command == "build":
        build(version)
    elif command == "deploy":
        deploy(version)
    elif command == "remove":
        remove(version)    
    else:
        printHelp()

if __name__ == "__main__":
    main()
