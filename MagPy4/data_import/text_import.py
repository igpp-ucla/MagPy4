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

# Specific to PDS4 labels
import os
import xml.etree.ElementTree as ET
import re

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

def read_pds_xml_info(filename, line=None):
    ''' Looks for and attempts to read a corresponding XML file
        for PDS4 formatted ASCII files if one exists
    '''
    # Skip if file has no extension
    basename = os.path.basename(filename)
    if '.' not in basename:
        return None

    # Check if XML file of same name / loc exists
    name = basename.split('.')[0]
    dir_loc = filename.split(name)[0]
    xml_name = os.path.join(dir_loc, f'{name}.xml')

    if not os.path.exists(xml_name):
        return None
    
    # Read in data from file as an XML tree
    try:
        tree = ET.parse(xml_name)
        root = tree.getroot()
    except:
        return None

    # Try to find file_observational info
    children = root.getchildren()
    tag_expr = r'{http://pds.nasa.gov/pds4/pds/v[0-9]+}File_Area_Observational'
    file_area_node = None
    for c in children:
        if re.fullmatch(tag_expr, c.tag):
            file_area_node = c
            break

    if file_area_node is None:
        return None
    
    # Get PDS version string
    version = file_area_node.tag.split('}')[0] + '}'
    
    # Look for file info nodes
    file_node = file_area_node.find(f'{version}File')
    header = file_area_node.find(f'{version}Header')
    table = file_area_node.find(f'{version}Table_Character')
    if file_node is None or table is None:
        return None

    # Determine delimiter
    delim_node = table.find(f'{version}record_delimiter')

    # Check that corresponding file is correct
    file_info = file_node.getchildren()
    if len(file_info) > 0 and file_info[0].tag.endswith('file_name'):
        if file_info[0].text != os.path.basename(filename):
            return None

    # Read in each column info from table
    record_char = table.find(f'{version}Record_Character')
    field_chars = record_char.iter(f'{version}Field_Character')
    keys = ['name', 'field_number', 'field_location', 'data_type', 
        'field_length', 'unit']

    field_descs = []
    field_chars = list(field_chars)
    for f in field_chars:
        d = {}
        for key in keys:
            node = f.find(f'{version}{key}')
            if node is not None:
                d[key] = node.text

        if 'name' in d:
            field_descs.append(d)

    # Get missing flags if given
    for desc, f in zip(field_descs, field_chars):
        node = f.find(f'{version}Special_Constants')
        if node is not None:
            subnode = node.find(f'{version}missing_constant')
            if subnode is not None:
                desc['flag'] = float(subnode.text)

    # Sort fields by field number
    for desc in field_descs:
        desc['field_number'] = int(desc['field_number'])
        desc['field_length'] = int(desc['field_length'])
        desc['field_location'] = int(desc['field_location'])
    field_descs = sorted(field_descs, key=lambda d : d['field_number'])

    # Determine time column and map field types
    time_col = None
    for desc in field_descs:
        field_length = desc['field_length']
        dtype = desc['data_type']
        if dtype == 'ASCII_Real':
            dtype = 'f8'
        elif dtype == 'ASCII_Integer':
            dtype = 'd'
        elif re.fullmatch(r'ASCII_Date_Time_\w+', dtype) and time_col is None:
            dtype = f'U{field_length}'
            time_col = desc['field_number'] - 1
        else:
            dtype = f'U{field_length}'
        desc['data_type'] = dtype
    
    if time_col is None:
        return None

    # Iterate over flags to get max
    max_flag = 1e32
    for desc in field_descs:
        flag = desc.get('flag')
        if flag is not None and flag > max_flag:
            max_flag = flag

    # Determine delimeter type (if separated by commas or fixed column-widths)
    cols = [desc['field_location'] for desc in field_descs]
    widths = [desc['field_length'] for desc in field_descs]
    last_chars = [start+width-1 for start, width in zip(cols, widths)]
    last_chars = [line[i] for i in last_chars]
    cols += [cols[-1] + field_descs[-1]['field_length']]

    fmt = FORMATS.TAB
    delim = np.diff(cols)
    if ',' in last_chars:
        fmt = FORMATS.CSV
        delim = ','
    
    # Assemble file info
    file_info = {
        'type' : fmt,
        'delimiter' : delim,
        'time_col' : time_col,
        'field_descs' : field_descs,
        'label_header' : header is not None,
        'flag' : max_flag
    }

    return file_info

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
    if len(times) > 0 and times[0].endswith(' '):
        times = list(map(lambda s : s.strip(' '), times))
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
    pds_info = read_pds_xml_info(filename, line=line)
    if pds_info is None:
        file_info = guess_file_info(hdr, line)
    else:
        file_info = pds_info

    delimeter = file_info['delimiter']
    names = True if file_info.get('label_header') is None else file_info['label_header']
    names = None if names == False else names
    
    # Read in data using numpy
    data = np.genfromtxt(filename, names=names, comments='#',
            filling_values=[np.nan, int(1e32), ''], delimiter=delimeter, 
            dtype=None, encoding=None, skip_header=skip_header)
    
    # Adjust labels if given by pds_info
    dtype = data.dtype.descr
    units = None
    if pds_info is not None:
        # Use PDS given data dtypes 
        new_dtype = []
        for desc in pds_info['field_descs']:
            new_dtype.append((desc['name'], desc['data_type']))

        # Create new array
        if new_dtype != dtype:
            data = np.array(data, dtype=np.dtype(new_dtype))

        # Get units if given
        units = []
        for desc in pds_info['field_descs']:
            unit = desc.get('unit')
            if unit is not None:
                units.append(unit)
            else:
                units.append('')

    # Otherwise, adjust dtype so missing / non-numerical values are converted to strings
    else:
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
    ticks = ff_time.dates_to_ticks(dates, 'J2000')

    # Create a new data table with mapped ticks
    old_type = data.dtype.descr
    old_type[time_col] = (time_lbl, 'f8')
    dtype = np.dtype(old_type)

    data = rfn.drop_fields(data, [time_lbl])
    data = rfn.structured_to_unstructured(data)

    table_data = np.insert(data, time_col, ticks, axis=1)
    table_data = rfn.unstructured_to_structured(table_data, dtype=dtype)

    # Remove flag values if given
    if pds_info is not None:
        max_flag = pds_info.get('flag')
        max_flag = 1e32 if max_flag is None else max_flag
        descs = pds_info.get('field_descs')
        for desc in descs:
            key = desc['name']
            flag = desc.get('flag')
            if flag is not None:
                data = table_data[key]
                if flag >= 0:
                    data[data >= flag] = max_flag
                else:
                    data[data <= flag] = max_flag
    else:
        max_flag = 1e32

    # Assemble file data object
    file_data = FileData(table_data, table_data.dtype.names, epoch=epoch, 
        time_col=time_col, units=units, error_flag=max_flag)

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