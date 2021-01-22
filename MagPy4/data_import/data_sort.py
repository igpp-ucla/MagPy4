import numpy as np
from bisect import bisect_left, bisect_right

class DataTuple():
    ''' Enables quick access to a pair of arrays, one for the time ticks
        and one for the corresponding data values
    '''
    def __init__(self, times, data):
        self.times = times
        self.data = data
    
    def __add__(self, other):
        times = np.concatenate([self.times, other.times])
        data = np.concatenate([self.data, other.data])
        return DataTuple(times, data)

    def __getitem__(self, s):
        times = self.times[s]
        data = self.data[s]
        return DataTuple(times, data)

    def __str__(self):
        return f'DataTuple(\n\t{str(self.times)}, \n\t{str(self.data)}\n)'

def single_fill_flags(ta, tb, dta_a, dta_b, flag):
    ''' Finds error flag values and fills in with non-error
        flag values from the other array if there is data
        available;

        Helper function for fill_flags; 
        Operates on a single column of data
    '''

    # Find indices where there are error flags
    n_a = len(dta_a)
    n_b = len(dta_b)
    flags_a = np.indices([n_a])[0][abs(dta_a) >= flag]
    flags_b = np.indices([n_b])[0][abs(dta_b) >= flag]

    # Make copies of data arrays
    dta_a = np.array(dta_a)
    dta_b = np.array(dta_b)

    # For each error flag value in dta_a
    for index in flags_a:
        time_a = ta[index]
        # Find the closest time in t_b to this error flag's time
        comp_index = bisect_left(tb, time_a)
        time_b = tb[comp_index]
        # If times are the same and dta_b's corresponding value
        # is not an error flag, replace it in dta_a
        if time_a == time_b and abs(dta_b[comp_index]) < flag:
            dta_a[index] = dta_b[comp_index]

    # Repeat above for dta_b
    for index in flags_b:
        time_b = tb[index]
        comp_index = bisect_left(ta, time_b)
        time_a = ta[comp_index]
        if time_a == time_b and abs(dta_a[comp_index]) < flag:
            dta_b[index] = dta_a[comp_index]

    return dta_a, dta_b

def fill_flags(ta, tb, dta_a, dta_b, flag):
    ''' Fills flag values with non-flag values it finds for each column
        in dta_a and dta_b
    '''
    if flag is None:
        return

    shape_a = dta_a.shape
    shape_b = dta_b.shape

    # Make sure number of columns are the same
    valid_shape = False
    if len(shape_a) == 1 and len(shape_b) == 1:
        valid_shape = True
        ncols = 0
    elif len(shape_a) == 2 and len(shape_b) == 2:
        if shape_a[1] == shape_b[1]:
            valid_shape = True
            ncols = shape_b[1]
    
    # Replace flags for single row
    if ncols == 0:
        dta_a, dta_b = single_fill_flags(ta, tb, dta_a, dta_b, flag)
    else:
        # Replace flags for each column of data
        for i in range(0, ncols):
            ref_a = dta_a[:,i]
            ref_b = dta_b[:,i]
            new_a, new_b = single_fill_flags(ta, tb, ref_a, ref_b, flag)
            dta_a[:,i] = new_a
            dta_b[:,i] = new_b

    return dta_a, dta_b

def sort_by_start(vals):
    ''' Sorts a set of DataTuples by their first time tick '''
    vals = [v for v in vals if len(v.times) > 0]
    keys = [v.times[0] for v in vals]
    order = np.argsort(keys)
    new_vals = [vals[i] for i in order]
    return new_vals

def merge_events(a, b, flag=None):
    ''' Merges two DataTuple events '''
    # Do not overlap at all
    if a.times[-1] < b.times[0]:
        return a + b

    # Initialize the two arrays to concatenate
    first = a
    second = DataTuple([], [])

    # b extends event
    if b.times[-1] > a.times[-1]:
        eI = bisect_right(b.times, a.times[-1])
        second = b[eI:]
        b = b[:eI]

    # a and b overlap
    sI = bisect_left(a.times, b.times[0])
    eI = bisect_right(a.times, b.times[-1])
    a_width = eI - sI
    b_width = len(b.times)

    # Fill flags in data
    dta_a, dta_b = fill_flags(a.times[sI:eI], b.times, a.data[sI:eI], b.data, flag)
    a.data[sI:eI] = dta_a
    b.data = dta_b

    # b's overlapping region has a higher resolution
    if b_width > a_width:
        first = a[:sI] + b

    return first + second

def merge_sort(time_slices, data_slices, flag):
    ''' Sorts and merges a set of time slices and their corresponding data slcies '''
    # Sort events by first time tick and convert to DataTuple objects
    events = [DataTuple(t, e) for t, e in zip(time_slices, data_slices)]
    sorted_times = sort_by_start(events)
    if len(sorted_times) == 0:
        return [], []

    # Merge each event in list with the previous one
    prev = sorted_times[0]
    i = 1
    while i < len(sorted_times):
        curr = sorted_times[i]
        prev = merge_events(prev, curr, flag)
        i += 1
    
    return (prev.times, prev.data)
