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

def sort_by_start(vals):
    ''' Sorts a set of DataTuples by their first time tick '''
    vals = [v for v in vals if len(v.times) > 0]
    keys = [v.times[0] for v in vals]
    order = np.argsort(keys)
    new_vals = [vals[i] for i in order]
    return new_vals

def merge_events(a, b):
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
    a_width = len(a.times) - sI
    b_width = len(b.times)

    # b's overlapping region has a higher resolution
    if b_width > a_width:
        first = a[:sI] + b

    return first + second

def merge_sort(time_slices, data_slices):
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
        prev = merge_events(prev, curr)
        i += 1
    
    return (prev.times, prev.data)
