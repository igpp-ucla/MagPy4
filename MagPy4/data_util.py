import numpy as np
from bisect import bisect_right, bisect_left
from fflib import ff_time

def get_resolution_diffs(times):
    ''' Computes the time resolution in seconds and first differences '''
    # Compute diffs
    if len(times) == 0:
        return 1.0, [0]

    last_val = times[0] - (times[1] - times[0])
    diffs = np.diff(times, prepend=[last_val])

    # Check if start, center, end diffs are same to use
    # as resolution
    n = len(diffs)
    left = diffs[0]
    right = diffs[-1]
    center = diffs[int(n/2)]

    if left == right and right == center:
        resolution = center
    else:
        # Use the median value if the values are different
        resolution = np.median(diffs)
    
    return resolution, diffs

def _get_res_changes(diffs):
    ''' Helper function for get_res_changes '''
    # Determines the indices where the next resolution
    # diff value is > 1.5 times or < 1/2 of the previous
    # resolution dif value and returns the mask array
    yield False
    last_val = None
    for a, b in zip(diffs, diffs[1:]):
        if b > a*1.5 or b < a*.5:
            val = True if last_val is not True else False
        else:
            val = False
        yield val
        last_val = val

def get_res_changes(diffs):
    return list(_get_res_changes(diffs))

def find_gaps(times, comp_res=None):
    if len(times) == 0:
        return []

    # Compute the first difference to find the resolutions
    diffs = np.diff(times)

    # Compute the indices where the resolution is > 1.5 times the previous res
    break_mask = diffs[1:] > diffs[:-1] * 1.5
    breaks = np.concatenate([[False, False], break_mask])
    gap_indices = np.indices(times.shape)[0][breaks]
    
    return gap_indices

def get_data_index(times, t):
    assert(len(times) >= 2)
    if t <= times[0]:
        return 0
    if t >= times[-1]:
        return len(times)
    b = bisect_left(times, t) # can bin search because times are sorted
    if b < 0:
        return 0
    elif b >= len(times):
        return len(times) - 1
    return b

def get_ticks_from_edit(timeEdit, epoch):
    start = timeEdit.start.dateTime().toPyDateTime()
    end = timeEdit.end.dateTime().toPyDateTime()
    t0 = ff_time.date_to_tick(start, epoch)
    t1 = ff_time.date_to_tick(end, epoch)
    return (t0, t1) if t0 < t1 else (t0, t1)