import cdflib
import re
import numpy as np
from datetime import datetime, timedelta
import numpy.lib.recfunctions as rfn
from .general_data_import import FileData
from fflib import ff_time
import bisect
from ..dynBase import SpecData

class CDF_ID():
    def __init__(self, cdf, data_dict, vec_grps):
        self.name = cdf
        self.epoch_vars = list(data_dict.keys())
        self.datas = data_dict
        self.vec_grps = vec_grps
    
    def get_grps(self):
        return self.vec_grps

    def getFileType(self):
        return 'CDF'

    def getName(self):
        return self.name

    def open(self):
        return

    def getEpoch(self):
        return self.datas[self.epoch_vars[0]].get_epoch()

    def getLabels(self, epoch_var=None):
        if epoch_var is None:
            epoch_var = self.epoch_vars[0]
        table = self.datas[epoch_var]
        return table.get_labels()

    def close(self):
        return

    def getEpochVars(self):
        return self.epoch_vars
    
    def getRecords(self, epoch_var=None):
        if epoch_var is None:
            epoch_var = self.epoch_vars[0]
        table = self.datas[epoch_var]
        return table

    def getTimes(self, epoch_var=None):
        if epoch_var is None:
            epoch_var = self.epoch_vars[0]
        table = self.datas[epoch_var]
        times = table.get_times()
        return times   

def get_cdf_datas(cdf, labels, time_len, exclude_keys=[], clip=None):
    ''' Reads in variables from the labels list in the given cdf
        and returns the data, labels, and units
        exclude_keys specifies a set of regular expressions or
            variable names to exclude
    '''
    datas = []
    data_labels = []
    data_units = []
    vec_grps = {}

    if clip:
        startIndex, endIndex = clip
    else:
        startIndex = 0
        endIndex = None

    # For each label in the CDF
    for label in labels:
        # Check if this variable should be excluded
        exclude = False
        for exclude_key in exclude_keys:
            if label == exclude_key or re.fullmatch(exclude_key, label):
                exclude = True
                break
        if exclude:
            continue

        # Get information about variable
        info = cdf.varinq(label)

        num_dims = info.get('Num_Dims')
        dims = info.get('Dim_Sizes')
        attrs = cdf.varattsget(label)

        if 'DISPLAY_TYPE' in attrs and 'spectogram' == attrs['DISPLAY_TYPE'].lower():
            continue

        # Skip non-time-series data
        if 'DEPEND_0' not in attrs:
            continue

        # Single column data is added directly
        if num_dims == 0:
            # Get data from cdf
            data = cdf.varget(label, startrec=startIndex, endrec=endIndex)

            # Make sure data shape is correct
            if len(data) != time_len:
                continue

            # Replace nans with error flag
            data = np.array(data, dtype='f4')
            data[np.isnan(data)] = 1.0e32

            # Determine base data label and units
            data_lbl = label if 'FIELDNAM' not in attrs else attrs['FIELDNAM']
            units = '' if 'UNITS' not in attrs else attrs['UNITS']

            # Store data, reshaping so it is in column format
            data_labels.append(data_lbl)
            data_units.append(units)
            datas.append(np.reshape(data, (len(data), 1)))

        # Multi-dimensional data
        if num_dims == 1:
            # Get data and replace nans with error flags
            data = cdf.varget(label, startrec=startIndex, endrec=endIndex)
            data = np.array(data, dtype='f4')
            data[np.isnan(data)] = 1.0e32

            # Make sure data shape is correct
            if len(data) != time_len:
                continue

            # Get number of columns
            cols = dims[0]
            plot_label = label

            # Use fieldnam attr for base label if present
            if 'FIELDNAM' in attrs:
                plot_label = attrs['FIELDNAM']

            # Otherwise check if lablaxis is present
            if 'LABLAXIS' in attrs:
                plot_label = attrs['LABLAXIS']

            # Lastly, use labl_ptr_1 to get a set of labels to apply to data
            if 'LABL_PTR_1' in attrs:
                plot_labels = cdf.varget(attrs['LABL_PTR_1']).tolist()
                plot_labels = np.reshape(plot_labels, (cols))
            else:
                plot_labels = [f'{plot_label}_{i}' for i in range(0, cols)]

            # Store data and make sure units are correct length
            units = '' if 'UNITS' not in attrs else attrs['UNITS']
            datas.append(data)
            data_units.extend([units]*len(plot_labels))
            data_labels.extend(plot_labels)

            # Save vector grouping
            vec_grps[label] = [lbl.strip(' ').replace(' ', '_') for lbl in plot_labels]

    # Remove extra whitespace and replace spaces with underscores
    data_labels = [lbl.strip(' ').replace(' ', '_') for lbl in data_labels]

    # Count up how many times each data label appears
    label_counts = {lbl:0 for lbl in data_labels}
    for label in data_labels:
        label_counts[label] += 1

    datas = np.hstack(datas)

    # Remove duplicates for any variables that appear more than once, like Bt
    for key in label_counts:
        if label_counts[key] > 1:
            repeat_indices = []
            # Gather repeated indices
            for i in range(0, len(data_labels)):
                if data_labels[i] == key:
                    repeat_indices.append(i)
            # Delete repeat indices from all lists
            datas = np.delete(datas, repeat_indices[1:], axis=1)
            data_units = np.delete(np.array(data_units), repeat_indices[1:]).tolist()
            data_labels = np.delete(np.array(data_labels), repeat_indices[1:]).tolist()

    return datas, data_labels, data_units, vec_grps

def load_spec_data(cdf, label, times, clip=None):
    # Get general spectrogram information
    attrs = cdf.varattsget(label)

    # Determine clip range
    if clip:
        startIndex, endIndex = clip
    else:
        startIndex, endIndex = 0, None

    if endIndex is None:
        endIndex = len(times) - 1

    # Get spectrogram values and y bins
    grid = cdf.varget(label, startrec=startIndex, endrec=endIndex)

    yvar = attrs['DEPEND_1']
    yvals = cdf.varget(yvar)

    # Get y-axis label, and units
    yattrs = cdf.varattsget(yvar)

    ylabel = yattrs.get('FIELDNAM')
    if ylabel is None:
        ylabel = yattrs.get('LABLAXIS')
    if ylabel is None:
        ylabel = ''

    yunits = cdf.varattsget(yvar)['UNITS']

    ylbltxt = f'{ylabel} ({yunits})'

    # Adjust y-variable if clip range is specified
    if clip:
        shape = yvals.shape
        if len(shape) > 0 and shape[0] > 1:
            yvals = yvals[startIndex:endIndex+1]

    # Get plot name
    name = attrs.get('FIELDNAM')
    if name is None:
        name = label

    # Get legend label and units
    legend_label = attrs.get('LABLAXIS')
    legend_units = attrs.get('UNITS')
    legend_lbls = [legend_label, legend_units]

    # Create specData object
    specData = SpecData(yvals, times, grid.T, log_color=True, log_y=True, 
        y_label=ylbltxt, legend_label=legend_lbls, name=name)

    return name, specData

def get_cdf_indices(times, clip_range):
    start_dt, end_dt = clip_range
    start_tick = ff_time.date_to_tick(start_dt, 'J2000')
    end_tick = ff_time.date_to_tick(end_dt, 'J2000')

    start_index = 0
    if start_tick > times[0]:
        start_index = bisect.bisect_left(times, start_tick)
        start_index = max(start_index-1, 0)
    
    end_index = len(times) - 1
    if end_tick < times[-1]:
        end_index = bisect.bisect_right(times, end_tick)
    
    return (start_index, end_index)

def load_cdf(path, exclude_expr=[], label_func=None, clip_range=None):
    # Open CDF and get list of variables
    cdf = cdflib.CDF(path)
    labels = cdf.cdf_info()['zVariables']
    labels.sort()

    # Get all epochs in file
    time_dict = {} # Each epoch and corresponding variables
    time_specs = {} # Spectrograms assoc. with epoch
    for label in labels:
        # Get attributes for each variable
        info = cdf.varinq(label)
        attrs = cdf.varattsget(label)

        # Skip non-time-series data
        if 'Data_Type_Description' in info and info['Data_Type_Description'] == 'CDF_CHAR':
            continue

        if 'DEPEND_0' not in attrs:
            continue

        # Get epoch variable
        time_var = attrs['DEPEND_0']

        # Check if this is a simple 2D spectrogram 
        specVar = False
        dtype = attrs.get('DISPLAY_TYPE')
        specVar = (dtype is not None) and (dtype.lower() == 'spectrogram')

        # Exclude spectrogram and support variables from data to be loaded later
        if specVar:
            if info.get('Num_Dims') == 1:
                yvar = attrs['DEPEND_1']
                exclude_expr.append(yvar)
            exclude_expr.append(label)

        # Save in dictionaries if not already present
        if time_var not in time_dict:
            time_dict[time_var] = []
            time_specs[time_var] = []
        # Keep track of which variables correspond to which epoch
        if specVar:
            time_specs[time_var].append(label)
        else:
            time_dict[time_var].append(label)

    # Read in CDF variable info for variables corresponding to each epoch
    tables = {}
    for epoch_lbl in time_dict:
        # Skip non-TT2000 formatted epochs
        mode = cdf.varinq(epoch_lbl)['Data_Type_Description']
        if mode not in ['CDF_TIME_TT2000', 'CDF_EPOCH']:
            continue

        # Get times and map from nanoseconds to SCET
        times = cdf.varget(epoch_lbl)

        if mode == 'CDF_TIME_TT2000':
            epoch = 'J2000'
            times = times * 1e-9 - 32.184 # Nanoseconds to seconds, then remove leap ofst
        elif mode == 'CDF_EPOCH':
            epoch = 'Y1966'
            ofst = cdflib.cdfepoch.parse([datetime(1966, 1, 1).isoformat()+'.000'])[0]
            times = (times - ofst) * 1e-3

        # Get indices for clipping if time range specified
        if clip_range:
            indices = get_cdf_indices(times, clip_range)
            times = times[indices[0]:indices[1]+1]
        else:
            indices = [0, None]

        # Get data variables with this epoch and extract data
        keys = time_dict[epoch_lbl]
        tlen = len(times)
        data_info = get_cdf_datas(cdf, keys, tlen, exclude_expr, 
            clip=indices)
        datas, data_labels, data_units, vec_grps = data_info

        # Extract spectrogram data
        specs = {}
        for spec_var in time_specs[epoch_lbl]:
            spec_name, spec_data = load_spec_data(cdf, spec_var, 
                times, clip=indices)
            specs[spec_name] = spec_data

        # Skip epochs with no data to load
        if len(datas) == 0:
            continue

        # If a label formatting function is passed, apply to variable labels
        if label_func is not None:
            data_labels = list(map(label_func, data_labels))
            vec_grps = {k:list(map(label_func, v)) for k, v in vec_grps.items()}

        # Create file data object
        dtype = np.dtype([(name, 'f8') for name in [epoch_lbl] + data_labels])
        table = np.hstack([np.reshape(times, (len(times), 1)), datas])
        table = rfn.unstructured_to_structured(table, dtype=dtype)
        cols = [epoch_lbl] + data_labels
        table_data = FileData(table, cols, 
            units=['Sec'] + data_units, epoch=epoch, specs=specs)
        tables[epoch_lbl] = table_data


    # # Update list of files
    cdf_id = CDF_ID(path, tables, vec_grps)
    return (tables, cdf_id)

