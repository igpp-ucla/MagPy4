:: Install python silently, adding to path
WHERE py
if %ERRORLEVEL% NEQ 0 (
    curl https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe > python-3.7.6-amd64.exe 
    python-3.7.6-amd64.exe /quiet PrependPath=1
)

:: Install git silently
WHERE git
if %ERRORLEVEL% NEQ 0 (
    curl -L https://github.com/git-for-windows/git/releases/download/v2.25.1.windows.1/Git-2.25.1-64-bit.exe > Git-2.25.1-64-bit.exe
    Git-2.25.1-64-bit.exe /VERYSILENT
)

:: Update/install build tools
py -m pip install setuptools
py -m pip install wheel

:: Download MagPy, install package and required dependencies
py -m pip install .

WHERE MagPy4.exe
WHERE MagPy4.exe > tmp.txt
if %ERRORLEVEL% NEQ 0 (
	:: If path is not set correctly, run win_add2path.py to set PATH variable
	py -c "import sys; import os; basePath = os.path.dirname(sys.executable); print (os.path.join(basePath, 'Tools', 'scripts', 'win_add2path.py'))">path_name.txt
	set /p scriptPath=<path_name.txt
	%scriptPath%

	:: Get .exe file path with python
	py -c "import sys; import os; basePath = os.path.dirname(sys.executable); print (os.path.join(basePath, 'scripts', 'MagPy4.exe'))">tmp.txt
)

:: Copy the .exe script to the desktop
set /p scriptLoc=<tmp.txt
copy %scriptLoc% %HOMEPATH%\Desktop

:: Delete temporary file
erase tmp.txt
