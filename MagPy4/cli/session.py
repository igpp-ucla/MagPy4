''' Headless stand-in for the main MagPy4 window's data handling.

    A Session loads a single data file (ASCII/CSV, CDF, or flat file) using
    the same readers as the GUI and exposes its variables and time slicing
    without creating any Qt widgets. Times are kept as ticks relative to the
    session epoch, as in the main window.
'''
import os
from bisect import bisect_left
from dataclasses import dataclass, field
import numpy as np
from dateutil import parser as date_parser
from datetime import timezone
from fflib import ff_time

from ..alg.mth import Mth
from ..data_import import (load_text_file, load_flat_file, load_cdf,
    find_vec_grps, get_resolution_diffs)

DEFAULT_ERROR_FLAG = 1e32

class SessionError(Exception):
    ''' Raised for problems loading files or selecting data '''

@dataclass
class Variable():
    ''' A single numerical column and the time array it is sampled on '''
    name: str
    units: str
    table: int # Index of the time array shared by this variable
    raw: np.ndarray # Values as read from the file
    flag: float # Values >= flag are treated as errors/fill
    _clean: np.ndarray = field(default=None, repr=False)

    def cleaned(self):
        ''' Returns the data with error values interpolated over, which
            is what the GUI plots and passes to the analysis tools
        '''
        if self._clean is None:
            self._clean = Mth.interpolateErrors(self.raw, self.flag)
        return self._clean

@dataclass
class TimeSeries():
    times: np.ndarray
    data: np.ndarray

class Session():
    FILE_TYPES = ('ascii', 'cdf', 'flat')

    def __init__(self):
        self.epoch = None
        self.path = None
        self.file_type = None
        self.vars = {} # name -> Variable
        self.tables = [] # [times, resolution] for each distinct time array
        self.vec_grps = {}
        self.warnings = []

    # ---- Loading ----
    @staticmethod
    def detect_file_type(path):
        ''' Picks a reader the same way the GUI does: .cdf files are CDFs,
            files with no extension (or .ffh/.ffd) are flat files, and
            everything else is treated as ASCII
        '''
        name = os.path.basename(path)
        lower = name.lower()
        if lower.endswith('.cdf'):
            return 'cdf'
        if '.' not in name or lower.endswith('.ffh') or lower.endswith('.ffd'):
            return 'flat'
        return 'ascii'

    @staticmethod
    def _exists(path, file_type):
        ''' Flat files are named by their base (without .ffh/.ffd) '''
        if file_type != 'flat':
            return os.path.exists(path)
        base, ext = os.path.splitext(path)
        if ext.lower() in ('.ffh', '.ffd'):
            path = base
        return os.path.exists(path + '.ffh') and os.path.exists(path + '.ffd')

    @classmethod
    def from_file(cls, path, **kwargs):
        session = cls()
        session.load(path, **kwargs)
        return session

    def load(self, path, file_type=None, time_config=None):
        ''' Reads in a file.

            file_type - one of FILE_TYPES, guessed from the name if None
            time_config - timestamp column description for ASCII files
                (see read_text_file); auto-detected if None
        '''
        if self.path is not None:
            raise SessionError('A Session holds a single file; create a new '
                'Session to open another')
        file_type = file_type or self.detect_file_type(path)
        if file_type not in self.FILE_TYPES:
            raise SessionError(f"Unknown file type '{file_type}', expected "
                f"one of {', '.join(self.FILE_TYPES)}")

        if not self._exists(path, file_type):
            raise SessionError(f"File not found: '{path}'")

        try:
            if file_type == 'cdf':
                tables, _ = load_cdf(path, exclude_expr=[])
                file_datas = list(tables.values())
            elif file_type == 'flat':
                file_datas = [load_flat_file(path)[0]]
            else:
                file_datas = [load_text_file(path, time_config=time_config)[0]]
        except Exception as e:
            raise SessionError(f"Failed to read '{path}': {e}") from e

        if len(file_datas) == 0:
            raise SessionError(f"No time series data found in '{path}'")

        self.path = path
        self.file_type = file_type
        for file_data in file_datas:
            self._add_file_data(file_data)

        if len(self.vars) == 0:
            raise SessionError(f"No numerical variables found in '{path}'")
        self._find_vector_groups()

    def _add_file_data(self, file_data):
        times = np.asarray(file_data.get_times(), dtype=float)
        if len(times) < 3:
            raise SessionError('Data must contain at least 3 time samples')

        # Map to the session epoch (set by the first table read)
        epoch = file_data.get_epoch()
        if self.epoch is None:
            self.epoch = epoch
        elif epoch != self.epoch:
            dates = ff_time.ticks_to_dates(times, epoch)
            times = ff_time.dates_to_ticks(dates, self.epoch)

        res = get_resolution_diffs(times)[0]
        self.tables.append([times, res])
        table = len(self.tables) - 1

        labels = file_data.get_numerical_keys()
        units = file_data.get_num_units()
        datas = file_data.get_numerical_data()
        flag = self._error_flag(file_data.get_flag())
        for label, unit, data in zip(labels, units, datas):
            if label in self.vars:
                self.warnings.append(f"Skipped duplicate variable '{label}'")
                continue
            self.vars[label] = Variable(label, unit, table,
                np.asarray(data, dtype=float), flag)

        self.vec_grps.update(file_data.vec_grps)

    def _error_flag(self, flag):
        ''' Flat file headers give the error flag as text; coerce to a float
            and fall back to the default flag if it can't be parsed
        '''
        try:
            return float(flag)
        except (TypeError, ValueError):
            self.warnings.append(f"Could not parse error flag '{flag}', "
                f"using {DEFAULT_ERROR_FLAG:g}")
            return DEFAULT_ERROR_FLAG

    def _find_vector_groups(self):
        ''' Finds field/position vector groups, named as in the GUI '''
        b_grps, pos_grps = find_vec_grps(self.variables)
        self.vec_grps.update({f'B{k}': v for k, v in b_grps.items()})
        self.vec_grps.update({f'R{k}': v for k, v in pos_grps.items()})

    # ---- Variable info ----
    @property
    def variables(self):
        return list(self.vars.keys())

    @property
    def vector_groups(self):
        return dict(self.vec_grps)

    def _var(self, name):
        if name not in self.vars:
            raise SessionError(f"Unknown variable '{name}'")
        return self.vars[name]

    def units(self, name):
        return self._var(name).units

    def times(self, name):
        ''' Full time array (ticks) for the variable '''
        return self.tables[self._var(name).table][0]

    def resolution(self, name):
        ''' Sample spacing in seconds '''
        return self.tables[self._var(name).table][1]

    def same_grid(self, names):
        ''' True if all variables share one time array '''
        return len({self._var(n).table for n in names}) <= 1

    # ---- Time conversion ----
    def to_tick(self, value):
        ''' Converts a datetime or ISO 8601 string to ticks '''
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, str):
            try:
                value = date_parser.isoparser().isoparse(value.strip())
            except ValueError as e:
                raise SessionError(f"Invalid timestamp '{value}': expected "
                    "ISO 8601, e.g. 2026-05-15T00:10:00") from e
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return ff_time.date_to_tick(value, self.epoch)

    def to_datetime(self, tick):
        return ff_time.tick_to_date(tick, self.epoch)

    def time_range(self, name=None):
        ''' (start, end) ticks of one variable, or of all of them '''
        names = [name] if name else self.variables
        arrs = [self.times(n) for n in names]
        return (min(a[0] for a in arrs), max(a[-1] for a in arrs))

    def datetime_range(self, name=None):
        return tuple(self.to_datetime(t) for t in self.time_range(name))

    # ---- Slicing ----
    def indices(self, name, start=None, end=None):
        ''' Start/end indices for the window [start, end), found the same
            way as the Spectra tool (bisect_left at both ends). None for
            start/end uses the variable's full extent.
        '''
        times = self.times(name)
        t0 = times[0] if start is None else self.to_tick(start)
        t1 = times[-1] if end is None else self.to_tick(end)
        if t0 >= t1:
            raise SessionError('Start time must be before end time')

        sI = 0 if start is None else bisect_left(times, t0)
        eI = len(times) if end is None else bisect_left(times, t1)
        return sI, eI

    def get(self, name, start=None, end=None, clean=True):
        ''' Returns the TimeSeries for a variable within [start, end)

            clean - interpolate over error-flag values (as the GUI does)
        '''
        var = self._var(name)
        sI, eI = self.indices(name, start, end)
        if eI - sI < 1:
            times = self.times(name)
            raise SessionError(f"No data for '{name}' between the given times "
                f"(data spans {self.to_datetime(times[0])} to "
                f"{self.to_datetime(times[-1])})")
        data = var.cleaned() if clean else var.raw
        return TimeSeries(self.times(name)[sI:eI], data[sI:eI])

    def get_many(self, names, start=None, end=None, clean=True):
        ''' Returns {name: TimeSeries}; variables must share a time grid '''
        if not self.same_grid(names):
            raise SessionError('Variables are sampled on different time '
                f"arrays ({', '.join(names)}); this is not supported yet")
        return {n: self.get(n, start, end, clean) for n in names}

    def describe(self):
        ''' Summary of the loaded file, e.g. for prompts and output metadata '''
        start, end = self.datetime_range()
        return {
            'path': self.path,
            'file_type': self.file_type,
            'epoch': self.epoch,
            'start': start.isoformat(),
            'end': end.isoformat(),
            'variables': {n: {'units': v.units,
                'resolution': self.resolution(n), 'samples': len(v.raw)}
                for n, v in self.vars.items()},
            'vector_groups': self.vector_groups,
        }
