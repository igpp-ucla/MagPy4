
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg
from .. import USERDATALOC
from ..qtthread import TaskRunner
from ..selectbase import TimeFormatWidget
from fflib import ff_time
from ..layouttools import LabeledProgress

import requests
import json
import os
import traceback, sys
import functools
import tempfile
import numpy as np
import zipfile
import shutil
from copy import copy

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, time

import cdflib

class PromptDialog(QtWidgets.QDialog):
    ''' Displays a prompt with yes/no buttons '''
    def __init__(self, prompt='', title='', *args):
        QtWidgets.QDialog.__init__(self, *args)

        # Set up layout
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Prompt label
        label = QtWidgets.QLabel(prompt)

        # Yes button
        yesBtn = QtWidgets.QPushButton('Yes')
        yesBtn.clicked.connect(self.accept)

        # No button
        noBtn = QtWidgets.QPushButton('No')
        noBtn.clicked.connect(self.reject)

        layout.addWidget(label, 0, 0, 1, 2)
        layout.addWidget(yesBtn, 1, 0, 1, 1)
        layout.addWidget(noBtn, 1, 1, 1, 1)

        self.setWindowTitle(title)

class StatusBox(QtWidgets.QTextEdit):
    ''' Modified text edit that appends messages to existing text '''
    def __init__(self, parent=None):
        QtWidgets.QTextEdit.__init__(self, parent)
        self.setReadOnly(True)

    def addMessage(self, msg):
        text = self.toPlainText() + '\n'
        text = text + msg
        self.setText(text)

class MMSLibrary():
    ''' Facilitates search for CDF files with chosen parameters '''
    def __init__(self, load_dir):
        self.base_dir = load_dir

    def get_months(self, start_dt, end_dt):
        ''' Returns the months between start_dt and end_dt '''
        # Get first day of month of month that start_dt is in
        ref_dt = datetime(start_dt.year, start_dt.month, 1)

        # Adjust end date to one month ahead
        if end_dt.month == 12:
            end_dt = datetime(end_dt.year+1, 1, 1)
        else:
            end_dt = datetime(end_dt.year, end_dt.month+1, 1)

        # Increment months for each date between ref_dt and end_date
        months = []
        while ref_dt < end_dt:
            year = ref_dt.year
            month = ref_dt.month
            months.append((year, month))
            if ref_dt.month == 12:
                ref_dt = datetime(ref_dt.year+1, 1, 1)
            else:
                ref_dt = datetime(ref_dt.year, ref_dt.month+1, 1)

        return months

    def find_data(self, params):
        ''' Looks for files matching given parameters
            Returns a list of matching files, a list of their
            corresponding data attributes, and a list of data parameters
            for which no data could be found
        '''
        # Map start/end times to flat file ticks
        start_dt, end_dt = params['Time Range']

        start_base_tick = ff_time.date_to_tick(start_dt, 'J2000')
        end_base_tick = ff_time.date_to_tick(end_dt, 'J2000')

        # Extract other parameters
        prods = params['Data Types']
        rate = params['Data Rate']
        sc_ids = params['Spacecraft']

        # Get the paths (and corresp. attrs) to look for data in
        path_grps, attrs = self.get_paths(start_dt, end_dt, prods, rate, sc_ids)

        # Search for files in each path group, keeping record of whether or not
        # any matching files were found
        matching_files = []
        date_range = []
        matching_attrs = []
        missing_data_attrs = []
        for paths, grp_attrs in zip(path_grps, attrs):
            data_found = False
            for path in paths:

                # If path exists
                if os.path.exists(path):
                    # Get latest files
                    files = os.listdir(path)
                    files = self.latest_files(files)

                    # Check if each file is in requested time range
                    for f in files:
                        filepath = os.path.join(path, f)
                        start_tick, end_tick = self.read_cdf_range(filepath)
                        if start_tick <= end_base_tick and end_tick >= start_base_tick:
                            matching_files.append(filepath)
                            date_range.append((start_tick, end_tick))
                            matching_attrs.append(grp_attrs)
                            data_found = True

            # If no data is found, save info about this set of paths
            if not data_found:
                missing_data_attrs.append(grp_attrs)

        return matching_files, matching_attrs, missing_data_attrs

    def latest_files(self, files):
        ''' Filters files list so that only latest CDF versions are selected '''
        files.sort()
        file_dict = {}
        for file in files:
            # Use pre 'v[CDFversion].cdf' info as a key in dict
            # and update with 'v[CDFversion]' (sorted so latest should be
            # in dict in the end)
            items = file.split('_')
            desc = '_'.join(items[:-1])
            vers = items[-1]
            file_dict[desc] = vers

        # Map dictionary key/values back to filenames
        return ['_'.join([desc, vers]) for desc, vers in file_dict.items()]

    def read_cdf_range(self, cdf_path):
        ''' Reads in the CDF's time range in flat file time ticks '''
        # Read in CDF
        cdf = cdflib.CDF(cdf_path)

        # Determine epoch keyword
        if '_edp_' in cdf_path:
            # Split name by underscores to get individual elements
            cdf_name = os.path.basename(cdf_path)
            elems = cdf_name.split('_')

            # Add epoch kw to list
            elems.insert(2, 'epoch')

            # Get new epoch keyword
            epoch_kw = '_'.join(elems[0:5])
        else:
            epoch_kw = 'Epoch'

        # Get epoch start/end values
        epoch_info = cdf.varinq(epoch_kw)
        last_rec = epoch_info['Last_Rec']
        start_tick = cdf.varget(epoch_kw, startrec=0, endrec=0)[0]
        end_tick = cdf.varget(epoch_kw, startrec=last_rec)[0]

        # Map ticks to SCET
        start_tick = start_tick / 1e9 - 32.184
        end_tick = end_tick / 1e9 - 32.184

        return (start_tick, end_tick)

    def get_paths(self, start_dt, end_dt, data_products, data_rate, sc_ids=[]):
        ''' Determines path to look for data in based on chosen parameters '''
        # Use all spacecraft is sc_ids is empty
        if len(sc_ids) == 0:
            sc_ids = [1,2,3,4]

        # Get base path
        path = os.path.join(self.base_dir, 'mms', 'data')

        # Determine directory path each data product should be in
        # given parameters
        path_grps = []
        path_attrs = []
        # Iterate over each spacecraft first
        for sc_id in sc_ids:
            # Set spacecraft and data rate attributes
            attrs = {'sc_id':sc_id}
            sc_path = os.path.join(path, f'mms{sc_id}')
            attrs['data_rate'] = data_rate

            # Iterate over data product types and descriptors
            for data_key, data_desc in data_products:
                # Create a copy of attribute information and store data type
                attrs = {key:val for key, val in attrs.items()}
                attrs['data_type'] = (data_key, data_desc)

                # Narrow down by data type and data rate and data level
                dt_path = os.path.join(sc_path, data_key, data_rate, 'l2')

                # Narrow by any data descriptors (for FPI data)
                if data_desc is not None:
                    dt_path = os.path.join(dt_path, data_desc)

                # Iterate over months/days depending on data rate and
                # generate paths for each date
                path_lst = []
                # Narrow by month if survey mode, and by day if burst mode
                if data_rate == 'brst':
                    # Iterate over all days between start/end date
                    td = timedelta(days=1)
                    ref_dt = start_dt
                    while ref_dt < end_dt:
                        # Get date as a string
                        year = str(ref_dt.year)
                        month = '{:02d}'.format(ref_dt.month)
                        day = '{:02d}'.format(ref_dt.day)

                        # Create path by specifying date
                        time_path = os.path.join(dt_path, year, month, day)
                        path_lst.append(time_path)

                        # Increment by a day
                        ref_dt = ref_dt + td
                else:
                    # Iterate over all months between start/end date inclusive
                    months = self.get_months(start_dt, end_dt)
                    for year, month in months:
                        # Create path from date string
                        year = str(year)
                        month = '{:02d}'.format(month)
                        time_path = os.path.join(dt_path, str(year), str(month))
                        path_lst.append(time_path)

                # Store the paths and attributes corresponding to those paths
                path_grps.append(path_lst)
                path_attrs.append(attrs)

        return path_grps, path_attrs

def reverse_dict(d, name):
    ''' Returns the key in dict d with the value equal to name '''
    rdict = {d[key]:key for key in d}
    return rdict[name]

class MMSDataUI():
    def setupUI(self, frame, window):
        self.baseFrm = frame
        # Set up directory chooser
        dirFrm = self.setupDirFrame()

        # Set up data products options
        dtypeFrm = self.setupDataTypes()
        rateFrm = self.setupDataRates()
        scFrm = self.setupSpacecrafts()

        # Set up time list text box
        timeListFrm = self.setupTimeList()

        # Set up message frame
        msgFrm = self.setupMessageView()

        # Set up query and load button button
        self.queryBtn = QtWidgets.QPushButton('Query')
        self.downBtn = QtWidgets.QPushButton('Load Data')
        self.downBtn.setVisible(False)

        # Set up loading bar
        self.loadBar = LabeledProgress()
        self.loadBar.setVisible(False)
        self.loadBar.setMinimum(0)
        self.loadBar.setMaximum(0)

        # Set up cancel button
        self.cancelBtn = QtWidgets.QPushButton('X')
        self.cancelBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Create main layout and add widgets to it
        layout = QtWidgets.QGridLayout(frame)

        optionsLt = QtWidgets.QGridLayout()
        optionsLt.addWidget(dtypeFrm, 0, 0, 1, 1)
        optionsLt.addWidget(rateFrm, 1, 0, 1, 1)
        optionsLt.addWidget(scFrm, 2, 0, 1, 1)
        optionsLt.addWidget(timeListFrm, 3, 0, 1, 1)
        layout.addWidget(dirFrm, 0, 0, 1, 1)
        layout.addLayout(optionsLt, 1, 0, 1, 1)
        layout.addWidget(msgFrm, 2, 0, 1, 1)

        # Set up buttons at bottom of window
        btnLt = QtWidgets.QHBoxLayout()
        btnLt.addWidget(self.queryBtn)
        btnLt.addWidget(self.downBtn)
        layout.addLayout(btnLt, 3, 0, 1, 1)
        layout.addWidget(self.loadBar, 4, 0, 1, 1)

        # Set window title
        frame.setWindowTitle('Load MMS Data')

    def setupDirFrame(self):
        ''' Set up frame for selecting and displaying directory
            to load data from
        '''
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(frame)
        label = QtWidgets.QLabel('Directory: ')

        # Selected directory label
        self.dirLabel = QtWidgets.QLabel(self.baseFrm.load_dir)
        ss = 'QLabel { color : #323a80; }'
        self.dirLabel.setStyleSheet(ss)

        # Folder icon button
        self.dirBtn = QtWidgets.QPushButton()
        icon_prov = QtWidgets.QFileIconProvider()
        icon = icon_prov.icon(QtWidgets.QFileIconProvider.Folder)
        self.dirBtn.setIcon(icon)

        # Add items to layout and make sure size is minimal
        for elem in [label, self.dirLabel, self.dirBtn]:
            layout.addWidget(elem)
            elem.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)

        return frame

    def setupDataTypes(self):
        ''' Set up frame for choosing data types '''
        frame = QtWidgets.QGroupBox('Data Types')
        layout = QtWidgets.QVBoxLayout(frame)

        # Get list of data types from frame
        dtypes = sorted(list(self.baseFrm.dtype_map.keys()))

        # Create a checkbox for each frame and set tooltip
        self.dtype_boxes = []
        for dtype in dtypes:
            box = QtWidgets.QCheckBox(dtype)
            tt = self.baseFrm.dtype_ttips[dtype]
            box.setToolTip(tt)
            layout.addWidget(box)
            self.dtype_boxes.append(box)

        return frame

    def setupDataRates(self):
        ''' Set up frame for selecting data rates '''
        frame = QtWidgets.QGroupBox('Data Rate')
        layout = QtWidgets.QHBoxLayout(frame)

        # Get list of data rates from main frame
        rates = sorted(list(self.baseFrm.rate_map.keys()))

        # Create a radio button for each data rate
        self.rate_btns = []
        for rate in rates:
            btn = QtWidgets.QRadioButton(rate)
            self.rate_btns.append(btn)
            layout.addWidget(btn)
        self.rate_btns[0].setChecked(True)

        return frame

    def setupSpacecrafts(self):
        ''' Set up frame for selecting spacecraft options '''
        frame = QtWidgets.QGroupBox('Spacecraft')
        layout = QtWidgets.QHBoxLayout(frame)

        self.sc_btns = []
        for sc_id in [1,2,3,4]:
            btn = QtWidgets.QCheckBox(f'MMS{sc_id}')
            self.sc_btns.append(btn)
            layout.addWidget(btn)
            btn.setChecked(True)

        return frame

    def setupTimeList(self):
        # Set up frame
        frame = QtWidgets.QGroupBox('Event Time')
        layout = QtWidgets.QGridLayout(frame)

        # Set up time format line edit
        self.timeFmt = TimeFormatWidget()

        # Set up event list box
        self.timeList = QtWidgets.QLineEdit()
        test_text = ''' 2018 Feb 01 10:44:10, 2018 Feb 01 10:46:20'''
        self.timeList.setPlaceholderText(test_text)

        # Add to layout
        layout.addWidget(self.timeFmt, 1, 0, 1, 1)
        layout.addWidget(self.timeList, 0, 0, 1, 1)

        return frame

    def setupMessageView(self):
        ''' Set up status message box '''
        frame = QtWidgets.QGroupBox('Status')
        layout = QtWidgets.QVBoxLayout(frame)

        self.messageBox = StatusBox()
        layout.addWidget(self.messageBox)

        return frame

class MMSDataDownloader(QtWidgets.QFrame):
    def __init__(self, window):
        self.window = window
        QtWidgets.QFrame.__init__(self)
        self.threadpool = QtCore.QThreadPool()
        self.task_obj = None
        self.load_dir = os.path.join(USERDATALOC, 'mms_data')

        # Get load directory from state
        state = window.readStateFile()
        if 'mms' in state and 'load_dir' in state['mms']:
            self.load_dir = state['mms']['load_dir']

        # List of data products
        # Key = label, Value = (query name, descriptor values)
        self.dtype_map = {
            'FGM' : ('fgm', None),
            'EDP DCE' : ('edp', 'dce'),
            'FPI DES MOMS' : ('fpi', 'des-moms'),
            'FPI DIS MOMS' : ('fpi', 'dis-moms')
        }

        # Tooltips for each data type
        self.dtype_ttips = {
            'FGM' : 'Fluxgate Magnetometer',
            'EDP DCE' : 'Electric field Double Probe - DC Electric Field',
            'FPI DES MOMS' : 'Fast Plasma Investigation - Dual Electron Spectrometer Moments',
            'FPI DIS MOMS' : 'Fast Plasma Investigation - Dual Ion Spectrometer Moments',
        }

        # Maps data rate label to
        self.rate_map = {
            'Burst' : 'brst',
            'Survey' : 'srvy',
            'Fast' : 'fast',
        }

        self.dtype_options = {
            'fgm' : {
                'modes' : ['Survey', 'Burst'],
            },
            'edp' : {
                'modes' : ['Survey', 'Burst'],
            },
            'fpi' : {
                'modes' : ['Fast', 'Burst'],
            },
        }

        # Parameters for current query
        self.currParams = {}
        self.foundFiles = None

        # Set up user interface
        self.ui = MMSDataUI()
        self.ui.setupUI(self, self.window)

        self.ui.downBtn.clicked.connect(self.loadFiles)
        self.ui.queryBtn.clicked.connect(self.queryLASP)
        self.ui.dirBtn.clicked.connect(self.setLoadDir)

    def closeEvent(self, ev):
        self.ui.timeFmt.closeFormatInfo()
        self.close()

    def setLoadDir(self):
        ''' Opens file dialog to set directory to load files from '''
        dirLoc = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dirLoc is not None and dirLoc != '':
            self.load_dir = dirLoc
            self.ui.dirLabel.setText(self.load_dir)   
            self.window.updateStateFile('mms', {'load_dir' : self.load_dir})    

    def getExistingFiles(self):
        ''' Gets a list of all existing files in load_dir '''
        # Create load_dir if it does not exist
        if not os.path.exists(self.load_dir):
            os.mkdir(self.load_dir)

        # Recursively get list of files in load_dir folders
        files = []
        for r, d, f in os.walk(self.load_dir):
            files.extend(f)
        return files

    def getParams(self):
        ''' Extract parameters from UI interface '''
        params = {}

        # Extract data type selections
        dtypes = [box.text() for box in self.ui.dtype_boxes if box.isChecked()]
        dtypes = [self.dtype_map[label] for label in dtypes]
        params['Data Types'] = dtypes

        # Extract data rate selection
        rate = 'Burst'
        for btn in self.ui.rate_btns:
            if btn.isChecked():
                rate = btn.text()
        rate = self.rate_map[rate]
        params['Data Rate'] = rate

        # Extract time range
        events = self.parseEventList()
        if events is None:
            self.ui.messageBox.setText('Error: Could not interpret event times')
            return None

        params['Time Range'] = events

        # Extract spacecraft selections
        sc_ids = [int(btn.text()[-1]) for btn in self.ui.sc_btns if btn.isChecked()]
        params['Spacecraft'] = sc_ids

        return params

    def parseEventList(self):
        ''' Reads in event list times and maps to datetime objects '''
        # Read in format and event list text
        fmt = self.ui.timeFmt.text()
        txt = self.ui.timeList.text()
        if ',' not in txt:
            return None

        # Parse into datetimes
        dt0, dt1 = txt.split(',')
        try:
            dt0 = datetime.strptime(dt0.strip(' '), fmt)
            dt1 = datetime.strptime(dt1.strip(' '), fmt)
        except:
            return None

        # Sort
        dt0, dt1 = min(dt0, dt1), max(dt0, dt1)

        return (dt0, dt1)

    def queryLASP(self):
        ''' Starts process for querying LASP MMS SDC for files to download '''
        # Get user-set parameters
        params = self.getParams()
        if params is None:
            return

        # Save current parameters to load data
        self.currParams = params

        # Check if it's possible to write to the load_dir
        try:
            if not os.path.exists(self.load_dir):
                os.makedirs(self.load_dir)
            if not os.access(self.load_dir, os.W_OK):
                raise Exception()
        except:
            # If not, read from local files only
            msg = f'{self.load_dir} not writable. Skipping file download...'
            self.downloadFinished(msg=msg)
            return

        # Show progress bar
        self.showProgress(True)
        self.ui.loadBar.setText('Running query...')

        # Start task for building and running query
        task = TaskRunner(self.buildQuery, params) # Any other args, kwargs are passed to the run function
        task.signals.finished.connect(self.queryEnded)
        task.signals.result.connect(self.reviewQuery)

        # Execute
        self.threadpool.start(task)

    def queryEnded(self):
        ''' Hide progress bar when finished '''
        self.showProgress(False)

    def showProgress(self, val, msg=''):
        ''' Shows progress bar, with option message '''
        self.ui.loadBar.setVisible(val)
        self.ui.queryBtn.setVisible(not val)

        if msg != '':
            self.ui.messageBox.addMessage(msg)

        self.ui.downBtn.setVisible(not val)

    def reviewQuery(self, full_files):
        ''' Reviews result(s) from query to LASP MMS SDC '''
        self.queryEnded()

        # Calculate total size of files to download
        if full_files is None:
            self.downloadFinished(msg='Could not query LASP...')
        elif len(full_files) == 0:
            self.downloadFinished(msg='No new files to download...')
            return
        else:
            # Exclude all files that are currently local
            curr_files = self.getExistingFiles()
            new_files = []
            for info in full_files:
                if info['file_name'] not in curr_files:
                    new_files.append(info)
            full_files = new_files

            # Return if all files are on disk
            if len(full_files) == 0:
                self.downloadFinished(msg='No new files to download...')
                return

            # Calculate the size of all files and format into a string
            size = self.file_list_size(full_files)
            size = np.round(size / 1e6, 2)

            if size > 1000: # GB format
                size = np.round(size / 1e3, 2)
                size_str = f'{size} GB'
            else: # MB format
                size_str = f'{size} MB'

            # Create a dialog displaying download size and asking whether to proceed
            question = f'Total download size is {size_str}. Proceed with download?'
            dialog = PromptDialog(question, 'Download Data', self)

            # If user wants to download data, start download process
            downloadFunc = functools.partial(self.startDownload, full_files)
            dialog.accepted.connect(downloadFunc)

            # Otherwise, start looking at local files for matches
            msg = 'Download canceled. Searching local files...'
            localFunc = functools.partial(self.startReviewLocalFiles, msg)
            dialog.rejected.connect(localFunc)

            # Open dialog
            dialog.open()

    def buildQuery(self, params):
        ''' Builds and runs queries to pass to LASP MMS SDC for the files
            matching the chosen parameters 
        '''
        base_url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/file_info/science?'

        # Extract data types and data rates, default level is L2 data
        data_level = 'l2'
        dtypes = params['Data Types']
        data_rate = params['Data Rate']

        # Extract spacecraft choices
        sc_ids = params['Spacecraft']
        if len(sc_ids) == 0:
            sc_ids = [1,2,3,4]
        sc_ids = ','.join([f'mms{sc_id}' for sc_id in sc_ids])

        # Get start/end dates and convert to timestamps
        start_date, end_date = params['Time Range']
        start_ts = start_date.date().isoformat()

        # End date should be incremented if not exact date since it is 
        # non-inclusive when used to query LASP
        if end_date.time() == time.min:
            end_ts = end_date.date().isoformat()
        else:
            end_ts = (end_date.date() + timedelta(days=1)).isoformat()

        # For each data product, run the query with the chosen
        # parameters and save the result
        results = []
        for name, desc in dtypes:
            # Build query string from parameters
            query = f'{base_url}sc_ids={sc_ids}&data_level={data_level}&instrument_ids={name}&start_date={start_ts}&end_date={end_ts}'

            # Add descriptors if necessary for data type
            if desc is not None:
                query = f'{query}&descriptors={desc}'

            # Set data rate
            query = f'{query}&data_rate_mode={data_rate}'

            # Query LASP database, returning None if fails
            try:
                result = requests.get(query)
            except:
                return None

            # Get dictionary of items from JSON result and store
            result = result.json()
            results.append(result)

        # Get full list of files to download
        full_files = []
        for result in results:
            full_files.extend(result['files'])

        # Below filters out files that are outside of times range
        ## Get start date for each file
        ts_fmt = '%Y-%m-%dT%H:%M:%S'
        dates = []
        for file_info in full_files:
            file_start = file_info['timetag']
            file_dt = datetime.strptime(file_start, ts_fmt)
            dates.append(file_dt)

        ## Sort files by their start dates
        sort_order = np.argsort(dates)
        dates = [dates[i] for i in sort_order]
        full_files = [full_files[i] for i in sort_order]

        ## Get start/end indices if in burst mode
        if data_rate == 'brst':
            index = max(bisect_left(dates, start_date)-1, 0)
            last_index = bisect_right(dates, end_date)
            return full_files[index:last_index]
        else:
            return full_files

    def updateProgress(self, val):
        self.ui.loadBar.setText(f'{val}%')

    def startDownload(self, full_files):
        ''' Called when list of files to download is determined and
            starts to download the files asynchronously 
        '''
        # Show the progress bar and clear it
        self.showProgress(True)
        self.ui.loadBar.setText('Starting download...')

        # Start thread that downloads files and connect its progress updates
        task = TaskRunner(self.downloadData, full_files, update_progress=True) # Any other args, kwargs are passed to the run function
        task.signals.progress.connect(self.updateProgress)
        task.signals.result.connect(self.downloadFinished)

        # Execute
        self.threadpool.start(task)

    def splitFilesByCount(self, files, max_files):
        ''' Splits files into groups smaller than max_files '''
        if len(files) < max_files:
            return [files]

        # Calculate number of groups and split
        n_grps = int(np.ceil(len(files)/max_files))
        subsets = [files[i*max_files:(i+1)*max_files] for i in range(n_grps)]
        return subsets

    def splitFilesBySize(self, files, max_size):
        ''' Splits the list of files into groups of <= max_size '''
        if len(files) == 0:
            return []

        if len(files) < max_size:
            return [files]

        max_size = max_size * 1e6 # Max group size
        b_count = 0 # Current byte count
        curr_lst = []
        subsets = [curr_lst]

        # Iterate over all files in list
        for file in files:
            file_size = file['file_size']
            # If group size > max size
            if (b_count + file_size) >= max_size:
                # Create a new list and reset byte count
                curr_lst = [file]
                b_count = file_size
                subsets.append(curr_lst)
            else:
                # Otherwise, append file to current group
                curr_lst.append(file)
                b_count += file_size

        return subsets

    def file_list_size(self, file_list):
        ''' Returns the total size of the number of files '''
        return sum([file_info['file_size'] for file_info in file_list])

    def downloadData(self, full_files, progress_func):
        ''' Requests list of files to download from LASP and extracts
            them to load_dir
        '''
        if len(full_files) == 0:
            return

        # Get size of files to download
        full_size = self.file_list_size(full_files)

        # If more than 500 files or more than 1GB of data (LASP MMS SDC limits)
        # split into multiple queries
        max_request = 1e3 - 50 # Additional leeway room
        max_files = 150 - 1

        # Split files by max_files (returns [full_files] if len < max_files)
        subsets = self.splitFilesByCount(full_files, max_files)

        # Split each subset further by size if total_size > max_size
        next_sets = []
        for subset in subsets:
            split_set = self.splitFilesBySize(subset, max_request)
            next_sets.extend(split_set)
        subsets = next_sets

        # Adjust full size slightly to account for compression
        full_size = full_size*.95

        size = 0 # Current byte count
        # Download each subset of files from LASP
        for file_subset in subsets:
            # Assemble list of files and format download request link
            download_url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?files='
            file_list = [file_info['file_name'] for file_info in file_subset]
            file_str = ','.join(file_list)
            download_link = f'{download_url}{file_str}'

            # Request files from LASP
            response = requests.get(download_link, stream=True)

            # Create a temporary file
            temp_fd, temp_name = tempfile.mkstemp()
            fd = open(temp_fd, 'wb')

            # Stream response and write to the temporary file
            for chunk in response.iter_content(chunk_size=None):
                fd.write(chunk)

                # Update progress function
                size += len(chunk)
                progress_func(int(size/full_size*100))

            # Close temporary file
            fd.close()

            # Try to extract zip file to load_dir
            singleFile = False
            try:
                zf = zipfile.ZipFile(temp_name)
                zf.extractall(self.load_dir)
                zf.close()
            except:
                # If loading a single file, need to determine load_dir subpath
                if len(file_subset) == 1:
                    # Get filename
                    filename = file_subset[0]['file_name']

                    # Extract subpath directories from filename
                    elems = filename.split('_')
                    subdirs = os.path.join(*elems[:-2])
                    savepath = os.path.join(self.load_dir, 'mms', 'data', subdirs)

                    # Determine date subdirectories based on timestamp
                    date_str = elems[-2]
                    year = date_str[0:4]
                    month = date_str[4:6]
                    day = date_str[6:8]

                    if elems[2] in ['srvy', 'fast']:
                        date_dirs = os.path.join(savepath, year, month)
                    else:
                        date_dirs = os.path.join(savepath,year,month,day)

                    # Create path if it does not exist
                    if not os.path.exists(date_dirs):
                        os.makedirs(date_dirs)

                    # TODO: Single file not correctly loaded for fpi des, dis = 2020 Jan 01 00:00:00, 2020 Jan 01 01:00:00
                    # Move file to directory
                    shutil.move(temp_name, os.path.join(date_dirs, filename))
                    singleFile = True

            # Remove zip file if extracted successfully
            if not singleFile:
                os.remove(temp_name)

    def downloadFinished(self, msg=None):
        ''' After download is finished, show progress bar and
            initiate search for local files
        '''
        self.showProgress(True)

        if msg is not None:
            self.ui.messageBox.setText(msg)
        else:
            self.ui.messageBox.setText('Download finished...')

        self.startReviewLocalFiles()

    def startReviewLocalFiles(self, msg=None):
        ''' Initiate search for local files '''
        # Set status bar/box messages
        self.ui.loadBar.setText('Locating local files...')
        if msg is not None:
            self.ui.messageBox.setText(msg)

        # Create task for reviewing local files
        task = TaskRunner(self.reviewLocalFiles)

        # Function will return message to print out to status box,
        # and connecting it to showProgress will hide progress bar
        # and update the message box
        endfunc = functools.partial(self.showProgress, False)
        task.signals.result.connect(endfunc)

        # Start task
        self.threadpool.start(task)

    def reviewLocalFiles(self):
        ''' Identifies which files are available and notifies
            user if data is missing
        '''
        # Get list of available files and info about missing data
        lib = MMSLibrary(self.load_dir)
        files, infos, missing_infos = lib.find_data(self.currParams)

        # Write out missing data information to message box if missing
        missing_text = ''
        if len(missing_infos) > 0:
            missing_text = ''
            for desc in missing_infos:
                # Write out spacecraft, data type, and data rate
                sc_id = desc['sc_id']
                data_type = reverse_dict(self.dtype_map, desc['data_type'])
                data_rate = reverse_dict(self.rate_map, desc['data_rate'])
                txt = f'No data for MMS{sc_id} {data_type} {data_rate}\n'
                missing_text += txt
        else:
            # Otherwise, notify user that data is available for all products
            missing_text = 'Data available for all data products'

        # Save list of found files
        self.foundFiles = (files, infos)

        # Show button for loading files
        self.ui.downBtn.setVisible(True)

        return missing_text

    def loadFiles(self):
        ''' Loads files in from list of saved files '''
        # If param is None, return
        param = self.foundFiles
        if param is None:
            self.ui.messageBox.setText('No files to load')
            return

        # If list of files is empty, return as well
        files, infos = param
        if files == [] or infos == []:
            self.ui.messageBox.setText('No data to load...')
            return

        # Keys (regex expressions) for variables to ignore in file
        exclude_keys = ['.*_dmpa_.*', '.*_bcs_.*', '.*_dbcs_.*', '.*energy_delta.*']

        # Generate labeling functions for variable names
        label_funcs = []
        for file, info in zip(files, infos):
            lblfunc = functools.partial(MMSDataDownloader.name_func, info)
            label_funcs.append(lblfunc)

        # Load list of files in main window
        clip_range = self.currParams['Time Range']
        self.window.addCDF(files, exclude_keys, label_funcs, clearPrev=True,
            clip_range=clip_range)

    def name_func(info, s):
        ''' Adjusts variable names according to data type '''
        # Add spacecraft number as a suffix if not present in variable name
        sc_id = str(info['sc_id'])
        if f'mms{sc_id}' not in s.lower():
            s += sc_id

        # If FPI DES/DIS data, prefix variables with 'e' or 'i'
        data_type = info['data_type']
        if data_type[0] == 'fpi':
            prefix = data_type[1][1]
            if f'd{prefix}s' not in s.lower():
                s = prefix + s

        return s