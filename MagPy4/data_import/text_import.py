from datetime import datetime, timedelta
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

def load_text_file(filename, epoch=None, time_config=None):
    ''' Helper function that creates a TextFileInfo object
        and returns it along with the FileData object
    '''
    if epoch is None:
        epoch = 'J2000'
    reader = TextFileInfo(filename, time_config=time_config)
    try:
        data = reader.read_data(epoch)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
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

    # If no ISO timestamp found, check for year + doy column pair which can
    # be used to synthesize timestamps
    if time_col is None:
        col_names = [c.strip().lower() for c in hdr.split(delimeter if isinstance(delimeter, str) else ',')]
        year_col = next((i for i, n in enumerate(col_names) if n == 'year'), None)
        doy_col = next((i for i, n in enumerate(col_names) if n in ('doy', 'day_of_year')), None)

        if year_col is not None and doy_col is not None:
            fday_col = next(
                (i for i, n in enumerate(col_names) if n in ('fday', 'frac_day', 'fractional_day')),
                None
            )
            file_info['year_col'] = year_col
            file_info['doy_col'] = doy_col
            file_info['fday_col'] = fday_col
            file_info['time_col'] = doy_col

    return file_info


def build_ticks_from_year_doy(data, year_col, doy_col, fday_col, epoch):
    ''' Build time ticks from year, day-of-year, and optional fractional day columns '''
    year_lbl = data.dtype.names[year_col]
    doy_lbl = data.dtype.names[doy_col]

    years = data[year_lbl].astype(int)
    doys = data[doy_lbl].astype(int)

    if fday_col is not None:
        fday_lbl = data.dtype.names[fday_col]
        fdays = data[fday_lbl].astype(float)
    else:
        fdays = np.zeros(len(years))

    dates = [
        datetime(int(y), 1, 1) + timedelta(days=int(d) - 1 + float(f))
        for y, d, f in zip(years, doys, fdays)
    ]
    return ff_time.dates_to_ticks(dates, epoch)

def normalize_time_spec(file_info):
    ''' Converts the auto-detected timestamp fields in file_info (as produced
        by guess_file_info / read_pds_xml_info) into a normalized time_spec
        dict describing how to compute time ticks. Returns {'mode': None}
        if no timestamp could be auto-detected.
    '''
    if file_info.get('time_col') is None:
        return {'mode': None}

    if 'year_col' in file_info:
        return {
            'mode': 'year_doy',
            'year_col': file_info['year_col'],
            'doy_col': file_info['doy_col'],
            'fday_col': file_info.get('fday_col'),
        }

    return {'mode': 'iso', 'col': file_info['time_col']}

def parse_custom_format_dates(values, fmt, filename=''):
    ''' Parses a list of strings into datetimes using a strptime format string '''
    dates = []
    for v in values:
        v = v.strip()
        try:
            dates.append(datetime.strptime(v, fmt))
        except ValueError as e:
            raise ValueError(
                f"'{filename}': failed to parse timestamp '{v}' using format '{fmt}'. {e}"
            ) from e
    return dates

def compute_time_ticks(data, time_spec, epoch, filename=''):
    ''' Computes time ticks (relative to epoch) from a structured array
        according to the given time_spec. Returns (ticks, cols_consumed),
        where cols_consumed are the column indices that should be replaced
        by / collapsed into the resulting tick array.

        time_spec modes:
          {'mode': 'iso', 'col': <int>}
          {'mode': 'year_doy', 'year_col': <int>, 'doy_col': <int>, 'fday_col': <int or None>}
          {'mode': 'custom', 'cols': [<int>] or [<int>, <int>], 'format': <strptime format>}
    '''
    mode = time_spec.get('mode')

    if mode == 'year_doy':
        try:
            ticks = build_ticks_from_year_doy(
                data, time_spec['year_col'], time_spec['doy_col'],
                time_spec.get('fday_col'), epoch
            )
        except Exception as e:
            raise ValueError(
                f"'{filename}': failed to build timestamps from year/doy columns. "
                f"Original error: {e}"
            ) from e
        return ticks, [time_spec['doy_col']]

    if mode == 'custom':
        cols = time_spec['cols']
        fmt = time_spec['format']
        labels = [data.dtype.names[c] for c in cols]
        if len(labels) == 2:
            raw_values = [f'{a} {b}' for a, b in zip(data[labels[0]], data[labels[1]])]
        else:
            raw_values = list(data[labels[0]])
        dates = parse_custom_format_dates(raw_values, fmt, filename)
        ticks = ff_time.dates_to_ticks(dates, epoch)
        return ticks, list(cols)

    # ISO mode (default)
    col = time_spec['col']
    lbl = data.dtype.names[col]
    times = data[lbl]
    try:
        dates = map_to_dates(times)
    except Exception as e:
        sample = times[0] if len(times) > 0 else '<empty>'
        raise ValueError(
            f"'{filename}': failed to parse timestamps in column '{lbl}'. "
            f"Expected ISO 8601 format (e.g. '2023-01-15T12:30:00'). "
            f"First value seen: '{sample}'. Original error: {e}"
        ) from e
    ticks = ff_time.dates_to_ticks(dates, epoch)
    return ticks, [col]

def map_to_dates(times):
    ''' Map timestamps to datetimes '''
    if len(times) > 0 and times[0].endswith(' '):
        times = list(map(lambda s : s.strip(' '), times))
    parser = isoparser.ISOParser('T')
    return list(map(parser.isoparse, times))

def _read_header_info(filename):
    ''' Reads past comment lines to find the header row and the first data
        record, validating that the header is not blank. Returns
        (hdr, line, skip_header, last_comment).
    '''
    with open(filename, 'r') as fd:
        skip_header = 0 # Comment lines to skip
        last_comment = None
        hdr = fd.readline()
        while skip_header < 100 and hdr.startswith('#'):
            last_comment = hdr
            hdr = fd.readline()
            skip_header += 1

        line = fd.readline()

    # Detect blank/degenerate header (e.g. a row of bare commas between comments
    # and a commented-out column-name line, which is a common CSV export artifact)
    hdr_content = hdr.strip().strip(',').strip()
    if not hdr_content:
        msg = (
            f"'{filename}': the first non-comment line (line {skip_header + 1}) "
            f"is blank. Column names could not be determined."
        )
        if last_comment is not None:
            candidate = last_comment.lstrip('#').strip()
            tokens = [t.strip() for t in candidate.split(',') if t.strip()]
            if tokens and not all(is_float(t) for t in tokens):
                msg += (
                    f"\n  Hint: the column header may be on the preceding commented "
                    f"line (starts with '#'). Found: '{candidate[:100]}'"
                )
        raise ValueError(msg)

    return hdr, line, skip_header, last_comment

def preview_text_file(filename, n_preview=10):
    ''' Performs a lightweight partial read of a text file for use in import
        preview dialogs: returns the detected columns, a handful of raw
        sample rows, and the auto-detected time_spec, without running the
        full numpy parse of the entire file.
    '''
    hdr, line, skip_header, last_comment = _read_header_info(filename)

    with open(filename, 'r') as fd:
        for _ in range(skip_header + 1):
            fd.readline()
        sample_lines = []
        for raw in fd:
            sample_lines.append(raw.rstrip('\n'))
            if len(sample_lines) >= n_preview:
                break

    pds_info = read_pds_xml_info(filename, line=line)
    if pds_info is None:
        file_info = guess_file_info(hdr, line)
    else:
        file_info = pds_info

    if file_info['type'] == FORMATS.CSV:
        columns = [c.strip() for c in hdr.split(',')]
        rows = [[v.strip() for v in l.split(',')] for l in sample_lines]
    else:
        cols = [0] + guess_cols(hdr, line) + [None]
        spans = list(zip(cols, cols[1:]))
        columns = [hdr[slice(a, b)].strip() for a, b in spans]
        rows = [[l[slice(a, b)].strip() for a, b in spans] for l in sample_lines]

    return {
        'columns': columns,
        'rows': rows,
        'file_info': file_info,
        'time_spec': normalize_time_spec(file_info),
        'skip_header': skip_header,
    }

def read_text_file(filename, epoch='J2000', time_config=None):
    ''' Reads in text file data (with time ticks relative to given epoch)
        and returns the structured data and a dictionary of file info.

        time_config, if given, overrides automatic timestamp detection and
        must be a dict in one of these shapes:
          {'mode': 'iso', 'col': <int>}
          {'mode': 'year_doy', 'year_col': <int>, 'doy_col': <int>, 'fday_col': <int or None>}
          {'mode': 'custom', 'cols': [<int>] or [<int>, <int>], 'format': <strptime format str>}
    '''
    hdr, line, skip_header, last_comment = _read_header_info(filename)

    # Determine file type and delimeter
    pds_info = read_pds_xml_info(filename, line=line)
    if pds_info is None:
        file_info = guess_file_info(hdr, line)
    else:
        file_info = pds_info

    # Resolve how timestamps will be derived, either from an explicit
    # user-supplied override or from auto-detection
    if time_config is not None:
        time_spec = time_config
    else:
        time_spec = normalize_time_spec(file_info)
        if time_spec['mode'] is None:
            if ',' in hdr:
                col_names = [t.strip() for t in hdr.split(',') if t.strip()]
            else:
                col_names = hdr.split()
            raise ValueError(
                f"'{filename}': no timestamp column found. "
                f"The reader requires either:\n"
                f"  - A column containing ISO 8601 datetime strings (e.g. '2023-01-15T12:30:00'), or\n"
                f"  - Columns named 'year' and 'doy' (day-of-year) for automatic timestamp synthesis.\n"
                f"Detected column names: {col_names[:10]}"
            )

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

    # Get time ticks according to the resolved time_spec. cols_consumed holds
    # the column index(es) used to derive the timestamp (more than one for
    # 'custom' mode when date/time are split across two columns) - all of
    # them collapse into a single time column in the output.
    ticks, cols_consumed = compute_time_ticks(data, time_spec, epoch, filename)
    cols_sorted = sorted(cols_consumed)
    time_col = cols_sorted[0]
    extra_cols = cols_sorted[1:]

    # Create a new data table with mapped ticks, dropping any extra
    # columns that were consumed (e.g. a second date/time column)
    old_type = data.dtype.descr
    time_lbl = old_type[time_col][0]
    old_type[time_col] = (time_lbl, 'f8')
    for c in sorted(extra_cols, reverse=True):
        del old_type[c]
    dtype = np.dtype(old_type)

    drop_lbls = [data.dtype.names[c] for c in cols_sorted]
    data = rfn.drop_fields(data, drop_lbls)
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
    def __init__(self, filename, time_config=None):
        self.name = filename
        self.filename = filename
        self.epoch = None
        self.subtype = FORMATS.TAB
        self.delimiter = None
        self.data_obj = None
        self.time_config = time_config

    def __repr__(self):
        return f'TextFileInfo({self.name}) at {hex(id(self))}'

    def read_data(self, epoch='J2000'):
        ''' Reads in the file data '''
        data, info = read_text_file(self.filename, epoch=epoch, time_config=self.time_config)
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