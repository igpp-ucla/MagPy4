from bisect import bisect, bisect_left, bisect_right
import numpy as np
import re
from fflib import ff_time

class FileData():
    ''' Time series data (and related info) from a given file '''
    def __init__(self, data, headers, time_col=0, units=None, epoch='Y1966', res=None,
        error_flag=1e32):
        self.data = data
        self.headers = headers
        self.units = units if units is not None else ['']*len(headers)
        self.epoch = epoch
        self.time_col = time_col
        self.error_flag = error_flag
        self.resolution = res if res is not None else self._guess_resolution()
        self._num_indices = None
    
    def _get_num_indices(self):
        ''' Determine the column numbers that contain strictly
            numerical data based on their dtype
        '''
        expr = r'[<>][if]'
        if self._num_indices is None:
            # Iterate over each column type and check if it matches
            # the regular expression for float and integer data
            desc = self.data.dtype.descr
            indices = []
            for i in range(0, len(desc)):
                d = desc[i][1]

                if re.match(expr, d):
                    indices.append(i)

            if self.time_col in indices:
                indices.remove(self.time_col)

            self._num_indices = indices
        return self._num_indices

    def _guess_resolution(self):
        ''' Guess the resolution for the data by checking if the
            start/middle/end resolutions are the same; Otherwise
            use the median of the first differences
        '''
        times = self.get_times()
        diff = np.diff(times)
        start = diff[0]
        end = diff[-1]
        center = diff[int(len(times)/2)]

        if start == end and end == center:
            resolution = start
        else:
            resolution = np.median(diff)
        
        return resolution

    def get_times(self):
        ''' Returns the time array '''
        col = self.data.dtype.names[self.time_col]
        times = self.data[col]
        return times
    
    def get_flag(self):
        ''' Returns the error flag '''
        return self.error_flag

    def get_epoch(self):
        ''' Returns the data file epoch '''
        return self.epoch
    
    def get_labels(self):
        ''' Returns the labels for each column '''
        return self.headers

    def get_units(self):
        ''' Return units for each column '''
        return self.units
    
    def get_num_units(self):
        ''' Returns the units for each data slice
            in the numerical data
        '''
        units = self.get_units()
        indices = self._get_num_indices()
        num_units = [units[i] for i in indices]
        return num_units

    def get_numerical_data(self):
        ''' Returns the data slices for strictly numerical data '''
        keys = self.get_numerical_keys()
        data = [self.data[key] for key in keys]
        return data

    def get_numerical_keys(self):
        ''' Returns the column keys for each data slice in the
            numerical data
        '''
        keys = self.get_labels()
        indices = self._get_num_indices()
        num_keys = [keys[i] for i in indices]
        return num_keys

    def get_data_table(self):
        ''' Returns the structured records array (with the time column) '''
        return self.data

    def get_dtype(self):
        ''' Returns the array dtype '''
        return self.data.dtype

    def get_resolution(self):
        ''' Returns the resolution for the data '''
        return self.resolution

    def get_time_col(self):
        ''' Returns the column number corresponding to the time array '''
        return self.time_col
    
    def find_record_by_time(self, dt):
        ''' Finds the row where date should fall before/at
            in the records
        '''
        tick = ff_time.date_to_tick(dt, self.get_epoch())
        times = self.get_times()
        index = bisect_left(times, tick)
        return index

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

def merge_datas(ta, tb, dta_a, dta_b, flag):
    # Fill flag values before merging
    dta_a, dta_b = fill_flags(ta, tb, dta_a, dta_b, flag)

    # Find indices corresponding to time gaps for each data set
    gaps_a = find_gaps(ta)
    gaps_b = find_gaps(tb)

    # Split data and times along gaps
    t_slices = np.split(ta, gaps_a)
    dta_slices = np.split(dta_a, gaps_a)

    tb_slices = np.split(tb, gaps_b)
    dta_b_slices = np.split(dta_b, gaps_b)

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

    # Remove any empty slices
    t_slices = [t for t in t_slices if len(t) > 0]
    dta_slices = [d for d in dta_slices if len(d) > 0]

    tb_slices = [t for t in tb_slices if len(t) > 0]
    dta_b_slices = [d for d in dta_b_slices if len(d) > 0]

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

def find_vec_grps(labels):
    ''' Attempts to find groups of three-axis
        vector data variable names 
    '''
    # Set up expressions for finding magnetic field vectors
    axes = ['X', 'Y', 'Z']
    ax_exprs = [f'{c}{c.lower()}' for c in axes]
    b_expr = [f'^[Bb]_?[{ax}]' for ax in ax_exprs]

    # Find all labels matching Bx/By/Bz format and place
    # in dict according to matching axis
    groups, suffixes = find_groups(b_expr, axes, labels)

    # Try to match b_expr pattern w/ found suffixes
    # in each axis group
    b_grps = find_full_groups(b_expr, axes, groups, suffixes)

    # Get leftover variables and check if 'BTotal' is
    # in variable list
    unused = get_unused(labels, b_grps)
    btot_expr = '[Bb][Tt]'
    btot_var = None
    for label in unused:
        if re.fullmatch(btot_expr, label):
            btot_var = label
            break
    
    # If btotal variable found, add to each group in b_grps
    if btot_var:
        for grp in b_grps:
            b_grps[grp].append(btot_var)

    # Repeat above process for position groups using
    # leftover variables
    pos_exprs = [f'[^{ax}]*[{ax}]' for ax in ax_exprs]
    groups, pos_suffixes = find_groups(pos_exprs, axes, unused)
    pos_grps = find_full_groups(groups, axes, groups, pos_suffixes)

    return (b_grps, pos_grps)

def get_unused(labels, groups):
    ''' Returns list of unused strings from labels
        based on variables found in groups dict
        Returns a list
    '''
    used = []
    for group in groups:
        used.extend(groups[group])

    unused = set(labels) - set(used)    
    return list(unused)

def find_groups(group_exprs, axes, labels):
    ''' Finds variables in labels matching each group expression
        and places them in a dictionary with
        key = axis corresponding to the expression

        Returns a tuple -> (dict, list)
    '''
    groups = {}
    suffixes = set()
    for expr, ax in zip(group_exprs, axes):
        ax_grp = []
        for label in labels:
            # If label matches, add to axis group and
            # save rest of string as potential pattern
            match_res = re.match(expr, label)
            if match_res:
                ax_grp.append(label)
                suffix = label[match_res.end():]
                if suffix != ' ':
                    suffixes.add(suffix)
        groups[ax] = ax_grp
    
    return groups, suffixes

def find_full_groups(group_expr, axes, groups, suffixes):
    ''' Uses the suffixes given to generate new regular
        expressions from group_expr that must generate
        a full match when compared against variables
        in the groups dict to be considered part of
        the vector group with that suffix
        
        Returns a dict
    '''
    full_groups = {}
    for suffix in suffixes:
        suffix_grp = []
        # Iterate over each axis to form a new regular
        # expression from axis group_expr and suffix
        for expr, ax in zip(group_expr, axes):
            ax_grp = groups[ax]
            full_expr = f'{expr}{suffix}'
            ax_matches = []
            # Add variable to axis subgroup if it fully matches
            for label in ax_grp:
                if re.fullmatch(full_expr, label):
                    ax_matches.append(label)

            # Only add to list if only one variable matches expression
            if len(ax_matches) == 1:
                suffix_grp.extend(ax_matches)
        
        # Add to list of vector groups if number of variables in
        # group is equal to number of axes
        if len(suffix_grp) == len(axes):
            suffix = suffix.strip(' ').strip('_')
            full_groups[suffix] = suffix_grp
    
    return full_groups

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
    gaps = find_gaps(arr, res*1.5)
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
    assert(np.array_equal(dta, test_arr))

    # Test general groups
    find_vec_grps(['BX_GSM1', 'BY_GSM1', 'BZ_GSM1', 'Bt', 'BX_GSE1', 'BY_GSE1', 'BZ_GSE1', 'X_GSM1', 'Y_GSM1', 'Z_GSM1',
        'BX' ,'BY', 'BZ'])
