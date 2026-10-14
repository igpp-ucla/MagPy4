''' Time window parsing for the command line '''
import re
from .prompt import CLIError

_DURATION = re.compile(r'^\+\s*(\d+(?:\.\d+)?)\s*([smhd])$', re.IGNORECASE)
_UNITS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}

def parse_duration(text):
    ''' Parses '+90s', '+10m', '+2h', '+1d' to seconds; None if not a duration '''
    match = _DURATION.match(text.strip())
    if not match:
        return None
    return float(match.group(1)) * _UNITS[match.group(2).lower()]

def make_window(session, start, end):
    ''' Resolves a (start, end) pair into ticks. Either may be None (use the
        data's extent) or, for end, a '+10m'-style duration after start.
        Returns (start_tick_or_None, end_tick_or_None).
    '''
    try:
        s = None if start in (None, '') else session.to_tick(start)
        duration = parse_duration(end) if isinstance(end, str) else None
        if duration is not None:
            if s is None:
                raise CLIError("A duration end ('+10m') needs a start time")
            e = s + duration
        else:
            e = None if end in (None, '') else session.to_tick(end)
    except Exception as ex:
        raise CLIError(str(ex)) from ex

    if s is not None and e is not None and s >= e:
        raise CLIError(f'Window start {start} must be before end {end}')
    return (s, e)

def read_windows_file(session, path):
    ''' Reads windows from a text file with one 'start,end' pair per line;
        blank lines and lines starting with # are ignored
    '''
    try:
        with open(path, 'r') as fd:
            lines = fd.read().splitlines()
    except OSError as e:
        raise CLIError(f"Could not read windows file '{path}': {e}") from e

    windows = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 2:
            raise CLIError(f"{path}:{n}: expected 'start,end'")
        windows.append(make_window(session, *parts))
    if not windows:
        raise CLIError(f"No windows found in '{path}'")
    return windows
