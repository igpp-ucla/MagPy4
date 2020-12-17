import numpy as np
def get_resolution_diffs(times):
    ''' Computes the time resolution in seconds and first differences '''
    # Compute diffs
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

def find_gaps(times, comp_res=None):
    if len(times) == 0:
        return []

    # Get the differences between times and resolution
    resolution, diffs = get_resolution_diffs(times)
    
    # Get comparison resolution
    comp_res = resolution*1.5 if comp_res is None else comp_res

    # Get mask for time gaps based on resolution
    gap_indices = np.indices([len(times)])[0][diffs >= comp_res]

    return gap_indices