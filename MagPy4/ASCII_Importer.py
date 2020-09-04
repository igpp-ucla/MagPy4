from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from FF_Time import FFTIME, leapFile
from datetime import datetime
from .layoutTools import BaseLayout
import numpy as np
from bisect import bisect
import time
import re
import numpy.lib.recfunctions as rfn

import dateutil
import numpy as np
import numpy.lib.recfunctions as rfn
from .timeManager import TimeManager

def check_if_timestamp(s):
    ''' Checks if in timestamp format by attempting
        to convert string to a float 
    '''
    try:
        float(s)
    except:
        return True
    return False

def map_dtype_to_fmt(dtype):
    ''' Maps a numpy dtype to a formatting string '''
    if dtype[1] == 'U':
        return '%s'
    elif dtype[1] == 'f':
        prec = dtype[2]
        return f'%.{prec}f'
    return '%s'

class TextReader():
    def __init__(self, name):
        self.name = name

    def _read_first_lines(self, n=5):
        ''' Read first n lines from file '''
        with open(self.name, 'r') as fd:
            lines = []
            for i in range(n):
                lines.append(fd.readline())
        return lines

    def _get_dtype(self, cols, time_dtype):
        ''' Creates a numpy dtype based on column names and
            number of columns and the given time_dtype
        '''
        dtype = [(cols[0], time_dtype)] 
        dtype += [(col_name, 'f8') for col_name in cols[1:]]
        dtype = np.dtype(dtype)
        return dtype

    def get_header(self):
        ''' Returns first line in file '''
        return self._read_first_lines(n=1)[0]
    
    def guess_type(self):
        ''' Tries to determine if file is a CSV file or
            fixed-width tab file
        '''
        if self.name.endswith('csv'):
            file_type = 'CSV'
        else:
            file_type = 'Fixed-Width'
        return file_type

    def get_data(self, epoch=None):
        ''' Read data into a structured numpy array '''
        if epoch is None:
            epoch = 'J2000'

        # Guess file type and set up reader
        file_type = self.guess_type()
        if file_type == 'CSV':
            reader = CSVReader(self.name)
        else:
            reader = TabReader(self.name)

        # Read in data table
        data = reader.read()

        # Format times 
        time_lbl = data.dtype.names[0]
        times = data[time_lbl]
        dates = self.map_to_dates(times)
        ticks = self.map_to_ticks(dates, epoch)

        # Create a new data table
        old_type = data.dtype.descr
        old_type[0] = (time_lbl, 'f8')
        dtype = np.dtype(old_type)

        data = rfn.drop_fields(data, [time_lbl])
        data = rfn.structured_to_unstructured(data)

        table_data = np.hstack([np.vstack(ticks), data])
        table_data = rfn.unstructured_to_structured(table_data, dtype=dtype)
        return table_data

    def map_to_dates(self, times):
        ''' Map timestamps to datetimes '''
        return list(map(dateutil.parser.isoparse, times))
    
    def map_to_ticks(self, dates, epoch):
        ''' Map datetimes to seconds since epoch '''
        tm = TimeManager(0, 0, epoch)
        ticks = list(map(tm.getTickFromDateTime, dates))
        return ticks

    def get_reader(self):
        ''' Returns the reader for the specific ASCII file subtype '''
        file_type = self.guess_type()
        if file_type == 'CSV':
            reader = CSVReader(self.name)
        else:
            reader = TabReader(self.name)
        return reader

    def getRecords(self, epoch):
        ''' Returns the data table with time in seconds
            since the given epoch
        '''
        return self.get_data(epoch)

    def getFileType(self):
        return 'ASCII'

    def getEpoch(self):
        return None
    
    def close(self):
        pass
    
class TabReader(TextReader):
    def _find_slices(self, lines):
        ''' Attempts to identify the indices indicating
            the start and end of each column in the file
            based on the first few lines
        '''
        # Get header and first record of data
        header = lines[0]
        row = lines[1]
        n = min(len(header), len(row))

        # Iterate over each character index in 
        # the header and first record
        slices = []
        start = 0
        spaces = False
        for i in range(0, n):
            h = header[i]
            r = row[i]
            # If not preceded by spaces and spaces are found,
            # this marks a full column so add the slice to the list
            if (not spaces) and h == ' ' and r == ' ':
                slices.append((start, i))
                start = i
                spaces = True
            # If no longer looking at spaces, reset this value
            # so the next column end will be detected
            elif spaces and h != ' ':
                spaces = False
    
        # Add in last slice if not added
        if slices[-1][1] != (n-1):
            slices.append((start, max(len(header), len(row))))
        
        return slices

    def get_slices(self):
        ''' Returns a list of indices specifying the start and
            end of each column
        '''
        lines = self._read_first_lines(n=2)
        slices = self._find_slices(lines)
        return slices

    def read(self):
        ''' Reads in data and returns it as a structured numpy array '''
        lines = self._read_first_lines()
        header = lines[0]
        slices = self._find_slices(lines)

        # Split first row and check if time is a tick or timestamp
        time_dtype = 'f8'
        row = lines[1]
        row = [row[slice(*s)] for s in slices]
        if check_if_timestamp(row[0]):
            time_dtype = 'U72'
        
        # Get column names
        cols = [header[slice(*s)].strip(' ').strip('\n') for s in slices]

        # Set up dtype
        dtype = self._get_dtype(cols, time_dtype)

        # Load data
        data = np.loadtxt(self.name, dtype=dtype, skiprows=1)

        return data
    
    def subtype(self):
        return 'Fixed-width'

class CSVReader(TextReader):
    def read(self):
        ''' Reads in data and returns it as a structured numpy array '''
        # Get header and first record
        lines = self._read_first_lines(n=2)
        header = lines[0]
        row = lines[1]
        row = row.split(',')

        # Get column names
        cols = [col.strip(' ').strip('\n') for col in header.split(',')]

        # Check if times are in timestamp format
        time_dtype = 'f8'
        if check_if_timestamp(row[0]):
            time_dtype = 'U72'

        # Set up dtype
        dtype = self._get_dtype(cols, time_dtype)

        # Read in data
        data = np.genfromtxt(self.name, dtype=dtype, skip_header=1,
            filling_values=np.nan, delimiter=',')

        return data
    
    def subtype(self):
        return 'CSV'