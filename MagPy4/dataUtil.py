from bisect import bisect, bisect_left, bisect_right
import numpy as np

def find_gaps(times):
    if len(times) == 0:
        return []

    # Get the differences between times
    last_val = times[0] - (times[1] - times[0])
    diffs = np.diff(times, prepend=[last_val])
    n = len(diffs)

    # Check if start, center, end diffs are same to use
    # as resolution
    left = diffs[0]
    right = diffs[-1]
    center = diffs[int(n/2)]

    if left == right and right == center:
        resolution = center
    else:
        # Use the average if the values are different
        resolution = np.mean(diffs)
    print ('RES', resolution)
    
    # Get mask for time gaps based on resolution
    gap_indices = np.indices([len(times)])[0][diffs > resolution*1.5]

    return gap_indices

def merge_datas(ta, tb, dta_a, dta_b, flag):
    # Fill flag values before merging
    # print ('GAPS PRE', dta_a, dta_b, len(dta_a), len(dta_b))
    dta_a, dta_b = fill_flags(ta, tb, dta_a, dta_b, flag)
    # print ('GAPS FILLED', dta_a, dta_b, len(dta_a), len(dta_b))

    # Find indices corresponding to time gaps for each data set
    gaps_a = find_gaps(ta)
    gaps_b = find_gaps(tb)

    # Split data and times along gaps
    t_slices = np.split(ta, gaps_a)
    dta_slices = np.split(dta_a, gaps_a)
    # print ('A Slices', t_slices, dta_slices, sep='\n')

    tb_slices = np.split(tb, gaps_b)
    dta_b_slices = np.split(dta_b, gaps_b)
    # print ('B Slices', tb_slices, dta_b_slices, gaps_b, sep='\n')

    t_grps = []
    dta_grps = []

    for tb, db in zip(tb_slices, dta_b_slices):
        if len(tb) <= 0:
            continue
        t0, t1 = tb[0], tb[-1]
        for i in range(0, len(t_slices)):
            t = t_slices[i]
            dta = dta_slices[i]
            # Clip end so it's replaced by start of b array
            if t0 >= t[0] and t0 <= t[-1]:
                index = bisect_left(t, t0)
                t_slices[i] = t[:index]
                dta_slices[i] = dta[:index]
            
            # Clip beginning so it's replaced by end of b array
            if t1 <= t[-1] and t1 >= t[0]:
                index = bisect_right(t, t1)
                t_slices[i] = t[index:]
                dta_slices[i] = dta[index:]
            
            if t0 < t[0] and t1 > t[-1]:
                t_slices[i] = []
                dta_slices[i] = []
            
            # print ('Clipped', t, t_slices[i], i)
        
    
    t_slices = [t for t in t_slices if len(t) > 0]
    dta_slices = [d for d in dta_slices if len(d) > 0]

    tb_slices = [t for t in tb_slices if len(t) > 0]
    dta_b_slices = [d for d in dta_b_slices if len(d) > 0]
    # print ('NEW SLICES', t_slices)

    # new_t_slices = []
    # new_dta_slices = []
    # for t, dta in zip(t_slices, dta_slices):
    #     t0, t1 = t[0], t[-1]
    #     for other_t, other_dta in zip(tb_slices, dta_b_slices):
    #         ta, tb = other_t[0], other_t[-1]

    #         if ta < t1 and 

    # Sort times and data
    t_grps = t_slices[:]
    dta_grps = dta_slices[:]
    keys = [t[0] for t in t_grps]
    for t, dta in zip(tb_slices, dta_b_slices):
        index = bisect_left(keys, t[0])
        t_grps.insert(index, t)
        dta_grps.insert(index, dta)
        keys.insert(index, t[0])
    
    times = np.concatenate(t_grps)
    data = np.concatenate(dta_grps)

    # last_t, last_dta = None, None
    # next_t, next_dta = t_slices.pop(0), dta_slices.pop(0)
    # i = 0
    # while (len(t_slices) > 0) and i < len(tb_slices):
    #     t_slice = tb_slices[i]
    #     dta_slice = dta_b_slices[i]
    #     # Get end time of previous slice
    #     if last_t is None:
    #         t0 = 0
    #     else:
    #         t0 = last_t[-1]

    #     # Get start time of next slice
    #     t1 = next_t[0]

    #     print( 'SLICE', t_slice, last_t, next_t)
    #     if last_t is not None:
    #         t_grps.append(last_t)
    #         dta_grps.append(last_dta)

    #     # If current b slice fits between selected slices
    #     if t_slice[0] > t0 and t_slice[-1] < t1:
    #         # Set current slice as the b slice
    #         last_t = t_slice
    #         last_dta = dta_slice
    #         i += 1
    #     else:
    #         # Set current slice as the next data slice and
    #         # move forward
    #         last_t = next_t
    #         last_dta = next_dta
    #         next_t = t_slices.pop(0)
    #         next_dta = dta_slices.pop(0)
        
    # if last_t is not None:
    #     t_grps.append(last_t)
    #     dta_grps.append(last_dta)
    
    # if i < len(tb_slices):
    #     t_grps.extend(tb_slices[i:])
    #     dta_grps.extend(dta_b_slices[i:])

    print ('GRPS', times, data, '\n')
    return times, data        





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
    shape_a = dta_a.shape
    shape_b = dta_b.shape

    print ('SHAPES', shape_a, shape_b)
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

def merge_records(ta, tb, dta_a, dta_b):
    # If an array is empty, return the other one
    if len(ta) == 0:
        return tb, dta_b
    elif len(tb) == 0:
        return ta, dta_a

    # Determine which array goes first
    if ta[0] > tb[0]:
        ta, tb = tb, ta
        dta_a, dta_b = dta_b, dta_a

    # Get the sections where times do not overlap
    start = bisect.bisect(ta, tb[0])
    pre_t = ta[:start]
    pre_data = dta_a[:start]

    if tb[-1] < ta[-1]:
        end = bisect.bisect(ta, tb[-1])
        mid_a = (start, end)
        mid_b = (0, len(dta_b))
    else:
        end = bisect.bisect(tb, ta[-1])
        mid_a = (start, len(ta))
        mid_b = (0, end)

    # Get sections where times overlap    
    a, b = mid_a
    mid_ta = ta[a:b]
    mid_dta_a = dta_a[a:b]
    del a
    del b

    a, b = mid_b
    mid_tb = tb[a:b]
    mid_dta_b = dta_b[a:b]

    sort_order = np.argsort(np.concatenate(mid_ta, mid_tb))
    initial_mask = sort_order[sort_order < len(mid_ta)]

# def merge_datas(times, datas):
#     # Empty lists
#     if len(times) == 0 or len(datas) == 0:
#         return [], []

#     # Single set of records
#     if len(times) <= 1 and len(datas) <= 1:
#         return times[0], datas[0]

#     # Sort a pair of time arrays
#     if len(times) == 2 and len(datas) == 2:
#         pass


def tests():
    # Test filling in small gaps
    flag = 100
    arr_a = np.array([1,2,3,flag,flag,4])
    t_a = np.arange(len(arr_a))

    arr_b = np.array([flag,4,4,8,9])
    t_b = np.arange(2, len(arr_b) + 2)

    new_arr_a, new_arr_b = single_fill_flags(t_a, t_b, arr_a, arr_b, flag)

    assert (np.array_equal(new_arr_a, np.array([1,2,3,4,4,4])))
    assert (np.array_equal(new_arr_b, np.array([3,4,4,8,9])))
    
    # Test finding gap indices
    ## Create a time array with gaps
    res = 1
    arr = np.arange(0, 50, res)
    rmv_indices = [5, 10, 11, 14, 26]
    arr = np.delete(arr, rmv_indices)
    
    ## Get the expected gap indices
    exp_gaps = set([rmv_indices[i] - i for i in range(len(rmv_indices))])

    ## Test gaps against expected gaps
    gaps = find_gaps(arr)

    assert(set(gaps) == set(exp_gaps))

    # Test merging non-overlapping data
    t_a = np.array([0, 1, 2, 6, 7, 10, 11])
    t_b = np.array([3, 4, 8, 9])
    dta_a = np.array([11, 10, 9, 6, 5, 2, 1])
    dta_b = np.array([8, 7, 4, 3])

    t, dta = merge_datas(t_a, t_b, dta_a, dta_b, flag)
    assert (np.array_equal(t, np.delete(np.arange(0, 12), 5)))
    assert (np.array_equal(dta, (np.arange(1, 12)[::-1])))

    # Test overlapping data
    # Simple partial overlap
    ta = np.arange(0, 5)
    tb = np.arange(3, 5+3)
    dta_a = np.array(ta)
    dta_b = np.array(tb)

    t, dta = merge_datas(ta, tb, dta_a, dta_b, flag)
    t_rev, dta_rev = merge_datas(ta, tb, dta_a, dta_b, flag)

    assert(np.array_equal(t, np.arange(0, 5+3)))
    assert(np.array_equal(dta, np.arange(0, 5+3)))

    # Make sure reverse overlap works the same
    assert(np.array_equal(t, t_rev))
    assert(np.array_equal(dta, dta_rev))

    # Full overlap
    tb = np.arange(-1, 10)
    dta_b = np.arange(-1, 10)
    t, dta = merge_datas(ta, tb, dta_a, dta_b, flag)

    assert(np.array_equal(tb, t))

    # Full overlap with error flags
    dta_b[0:4] = flag
    t, dta = merge_datas(ta, tb, dta_a, dta_b, flag)
    test_arr = np.array(dta_b)
    test_arr[1:4] = [0,1,2]
    assert(np.array_equal(dta, test_arr))

    # Test with  full overlap with error flags with gaps
    tb = np.delete(tb, [2,3,4])
    dta_b = np.delete(dta_b, [2,3,4])
    t, dta = merge_datas(ta, tb, dta_a, dta_b, flag)
    print (len(t), len(dta))
    assert(np.array_equal(dta, test_arr))


# tests()
