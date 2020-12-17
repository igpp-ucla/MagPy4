from datetime import datetime
import numpy as np
from bisect import bisect
import re
import numpy.lib.recfunctions as rfn
import dateutil
import numpy as np
from fflib import ff_time, isoparser
from .general_data_import import FileData
from enum import Enum

def load_text_file(filename, epoch=None):
    ''' Helper function that creates a TextFileInfo object
        and returns it along with the FileData object
    '''
    if epoch is None:
        epoch = 'J2000'
    reader = TextFileInfo(filename)
    data = reader.read_data(epoch)
    return (data, reader)

class FORMATS(Enum):
    ''' ASCII file formats '''
    TAB = 0 # Fixed-width columns
    CSV = 1 # Comma-separated values

def is_float(s):
    ''' Checks if a string s can be converted to a float value '''
    try:
        float(s)
    except:
        return False
    return True

def is_int(s):
    ''' Checks if a string can be converted to an integer value '''
    try:
        int(s)
    except:
        return False
    
    return True

def find_cols(l):
    ''' Finds indices of all points where there is
        a non-whitespace character followed by whitespace
    '''
    matches = list(re.finditer('[^ ] ', l))
    cols = [m.start()+1 for m in matches]
    return cols

def guess_cols(hdr, first_line):
    ''' Calls find_cols() on header line or first data record line
        based on spacing
    '''
    if hdr.startswith(' '):
        cols = find_cols(hdr)
    else:
        cols = find_cols(first_line)
    
    return cols

def guess_file_info(hdr, first_line):
    ''' Determine file type and related info such as columns and
        the column number of the timestamps based on the header
        line and the first record
    '''
    # Determine format and delimeter, then split first record into items
    if ',' in hdr:
        t = FORMATS.CSV
        delimeter = ','
        record = first_line.split(delimeter) # Split by commas
    else:
        t = FORMATS.TAB
        cols = guess_cols(hdr, first_line)
        if len(cols) == 1:
            raise Exception('Could not determine columns')
        cols = [0] + cols
        delimeter = np.diff(cols)
        record = [first_line[slice(*t)] for t in zip(cols, cols[1:])]

    # Identify first column that can be converted into a set of timestamps
    time_col = None
    parser = isoparser.ISOParser('T')
    for entry, num in zip(record, np.arange(len(record), dtype=int)):
        # Skip if time component is missing
        if ':' not in entry:
            continue

        # Try to convert element into datetime
        try:
            parser.isoparse(entry)
            time_col = num
        except:
            continue

    # Save file info into a dictionary
    file_info = {
        'type' : t,
        'delimiter' : delimeter,
        'time_col' : time_col
    }

    return file_info

def map_to_dates(times):
    ''' Map timestamps to datetimes '''
    parser = isoparser.ISOParser('T')
    return list(map(parser.isoparse, times))

def read_text_file(filename, epoch='J2000'):
    ''' Reads in text file data (with time ticks relative to given epoch)
        and returns the structured data and a dictionary of file info
    '''
    # Open file and read in header line and first record
    with open(filename, 'r') as fd:
        skip_header = 0 # Comment lines to skip
        hdr = fd.readline()
        while skip_header < 100 and hdr.startswith('#'):
            hdr = fd.readline()
            skip_header += 1

        line = fd.readline()

    # Determine file type and delimeter
    file_info = guess_file_info(hdr, line)
    delimeter = file_info['delimiter']

    # Read in data using numpy
    data = np.genfromtxt(filename, names=True, comments='#',
            filling_values=[np.nan, int(1e32), ''], delimiter=delimeter, 
            dtype=None, encoding=None, skip_header=skip_header)
    
    # Adjust dtype so missing / non-numerical values are converted to strings
    dtype = data.dtype.descr
    new_dtype = []
    for label, t in dtype:
        if not (t.startswith('<U') or t.startswith('<i') or t.startswith('<f')):
            t = 'U8'
        new_dtype.append((label, t))

    if new_dtype != dtype:
        data = np.array(data, dtype=np.dtype(new_dtype))

    # Get the time column data and convert to ticks
    time_col = file_info['time_col']
    time_lbl = data.dtype.names[time_col]

    times = data[time_lbl]
    dates = map_to_dates(times)
    ticks = ff_time.dates_to_ticks(dates, 'J2000', fold_mode=True)

    # Create a new data table with mapped ticks
    old_type = data.dtype.descr
    old_type[time_col] = (time_lbl, 'f8')
    dtype = np.dtype(old_type)

    data = rfn.drop_fields(data, [time_lbl])
    data = rfn.structured_to_unstructured(data)

    table_data = np.insert(data, time_col, ticks, axis=1)
    table_data = rfn.unstructured_to_structured(table_data, dtype=dtype)

    # Assemble file data object
    file_data = FileData(table_data, table_data.dtype.names, epoch=epoch, time_col=time_col)

    return file_data, file_info

class TextFileInfo():
    ''' Text File Descriptor '''
    def __init__(self, filename):
        self.name = filename
        self.filename = filename
        self.epoch = None
        self.subtype = FORMATS.TAB
        self.delimiter = None
        self.data_obj = None
    
    def __repr__(self):
        return f'TextFileInfo({self.name}) at {hex(id(self))}'

    def read_data(self, epoch='J2000'):
        ''' Reads in the file data '''
        data, info = read_text_file(self.filename, epoch=epoch)
        self.epoch = epoch
        self.delimiter = info.get('delimiter')
        return data

    def get_data(self, epoch_var=None):
        ''' Returns the file data object '''
        if self.data_obj is None:
            self.data_obj = self.read_data()
        return self.data_obj

    def getRecords(self, epoch_var):
        ''' Returns the file data object '''
        return self.get_data()
    
    def getEpoch(self):
        ''' Returns the file epoch '''
        return self.epoch

    def close(self):
        return

    def getFileType(self):
        return 'ASCII'