''' MagPy4 command line interface.

    MagPy4-cli [FILE] [TOOL] [options] [tool options]

    Anything not given on the command line is asked for interactively when
    running in a terminal (use --no-prompt to disable that for scripts).
    Prompts go to stderr so results written to stdout stay clean.
'''
import argparse
import os
import sys
import traceback

from .. import MAGPY_VERSION
from .session import Session, SessionError
from .prompt import Prompter, CLIError
from .windows import make_window, read_windows_file
from . import outputs
from . import tools as tool_registry

ALL_FORMATS = outputs.FORMATS + tool_registry.IMAGE_FORMATS

EXAMPLES = '''
examples:
  MagPy4-cli data.csv stats --vars bx by bz --window 2026-05-15T00:10 +10m -f table
  MagPy4-cli data.cdf stats --vars Bmso --windows windows.txt -f csv -o out.csv
  MagPy4-cli data.csv            (prompts for the tool, variables, windows, format)

Put FILE and TOOL first; tool options (see 'MagPy4-cli FILE TOOL -h') follow.
'''

def build_base_parser():
    p = argparse.ArgumentParser(prog='MagPy4-cli', add_help=False,
        description='Run MagPy4 analysis tools without the GUI.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('file', nargs='?', help='data file (ASCII/CSV, CDF, or flat file)')
    p.add_argument('tool', nargs='?', help='tool to run (see --list-tools)')
    p.add_argument('--window', nargs=2, action='append', default=[],
        metavar=('START', 'END'), help='time window as ISO 8601 timestamps; '
        "END may be a duration after START such as '+10m' (s/m/h/d). "
        'Repeat for several windows. Default: the full time range')
    p.add_argument('--windows', metavar='FILE',
        help="file of windows, one 'start,end' per line")
    p.add_argument('-f', '--format', choices=ALL_FORMATS,
        help='output format; tools list what they support (default: from the '
        '-o extension, otherwise the default for the tool: json for numeric tools, '
        'png for plots)')
    p.add_argument('-o', '--output', metavar='PATH',
        help='write results to PATH (overwriting it) instead of stdout; a '
        'folder gets a file named after the tool. For images, PATH is a file '
        'name or folder, and several windows give one file each with the '
        'window start time added to the name. Plots go to the current '
        'directory when -o is not given. In a terminal you are asked for the '
        'location if you do not pass -o')
    p.add_argument('--file-type', choices=Session.FILE_TYPES,
        help='override file type detection')
    p.add_argument('--time-col', nargs='+', type=int, metavar='N',
        help='ASCII: 0-based timestamp column (two with --time-format if the '
        'date and time are in separate columns)')
    p.add_argument('--time-format', metavar='FMT',
        help="ASCII: strptime format of the timestamp column(s), e.g. '%%Y-%%m-%%d %%H:%%M:%%S'")
    p.add_argument('--year-doy', nargs='+', type=int, metavar='COL',
        help='ASCII: 0-based year, day-of-year and optional fractional-day columns')
    p.add_argument('--no-prompt', action='store_true',
        help='never prompt; fail if a required value is missing')
    p.add_argument('--list-tools', action='store_true', help='list tools and exit')
    p.add_argument('--list-vars', action='store_true',
        help='list the variables in FILE and exit')
    p.add_argument('--debug', action='store_true', help='show tracebacks')
    p.add_argument('--version', action='store_true')
    p.add_argument('-h', '--help', action='store_true')
    return p

def build_tool_parser(tool):
    p = argparse.ArgumentParser(prog=f'MagPy4-cli FILE {tool.name}',
        description=tool.description, add_help=False)
    for param in tool.params:
        kwargs = {'help': param.help or None, 'default': None, 'dest': param.name}
        if param.kind == 'bool':
            p.add_argument(param.flag, action='store_true', default=None, **{
                k: v for k, v in kwargs.items() if k != 'default'})
            continue
        if param.kind == 'vars':
            # Count is checked in resolve_vars so a group name like Bmso
            # (one argument) can expand into several variables
            kwargs['nargs'] = '+'
        elif param.kind == 'int':
            kwargs['type'] = int
        elif param.kind == 'float':
            kwargs['type'] = float
        elif param.kind == 'choice':
            kwargs['choices'] = param.choices
        if param.default is not None:
            kwargs['help'] = f"{kwargs['help'] or ''} (default: {param.default})".strip()
        p.add_argument(param.flag, **kwargs)
    p.add_argument('-h', '--help', action='store_true')
    return p

def print_tool_list(out):
    width = max(len(n) for n in tool_registry.TOOLS)
    for name, tool in tool_registry.TOOLS.items():
        print(f'  {name.ljust(width)}  {tool.description}', file=out)

# ---- Parameter handling ----
def match_variable(session, name):
    ''' Exact match, or a unique case-insensitive match '''
    if name in session.vars:
        return name
    matches = [v for v in session.variables if v.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise CLIError(f"Unknown variable '{name}'. Use --list-vars to see the options")

def resolve_vars(session, values, count=None):
    ''' Expands variable names (comma or space separated) and vector group
        names such as Bmso into a list of variables
    '''
    items = []
    for v in values:
        items.extend(x for x in v.replace(',', ' ').split() if x)

    groups = {k.lower(): v for k, v in session.vector_groups.items()}
    names = []
    for item in items:
        if item.lower() in groups and item not in session.vars:
            names.extend(groups[item.lower()])
        else:
            names.append(match_variable(session, item))
    if count is not None and len(names) != count:
        raise CLIError(f'Expected {count} variables, got {len(names)}')
    if not names:
        raise CLIError('No variables given')
    return names

def convert_text(param, text, session):
    ''' Converts a prompt answer into a value for the parameter '''
    if param.kind == 'int':
        return int(text)
    if param.kind == 'float':
        return float(text)
    if param.kind == 'bool':
        return text.lower() in ('y', 'yes', 'true', '1')
    if param.kind == 'choice':
        if text not in param.choices:
            raise ValueError(f"choose one of: {', '.join(map(str, param.choices))}")
        return text
    if param.kind == 'vars':
        return resolve_vars(session, [text], param.count)
    return text

def collect_params(tool, tool_args, session, prompter):
    ''' Gets a value for each tool parameter from the parsed flags, prompting
        for required ones (and ones marked ask) that were not given
    '''
    values = {}
    for param in tool.params:
        value = getattr(tool_args, param.name, None)
        if value is not None:
            if param.kind == 'vars':
                value = resolve_vars(session, value, param.count)
        elif param.required or (param.ask and prompter.interactive):
            if param.kind == 'vars':
                show_variables(session, prompter)
            if param.kind == 'choice':
                labels = param.labels or {}
                options = [(c, labels.get(c, '')) for c in param.choices]
                value = prompter.choose(param.help or param.name, options,
                    default=param.default, flag=param.flag)
            else:
                value = prompter.ask(param.help or param.name, flag=param.flag,
                    parse=lambda t, p=param: convert_text(p, t, session))
        elif param.kind == 'bool':
            value = bool(param.default)
        else:
            value = param.default
        values[param.name] = value
    return values

def show_variables(session, prompter):
    if not prompter.interactive:
        return
    prompter.say('Variables: ' + ', '.join(session.variables))
    if session.vector_groups:
        groups = ', '.join(f"{k} ({' '.join(v)})"
            for k, v in session.vector_groups.items())
        prompter.say('Vector groups (enter the name to use all three): ' + groups)
    prompter.say('Separate several variables with spaces or commas.')

# ---- Time configuration for ASCII files ----
def build_time_config(args):
    ''' Builds the read_text_file time_config dict from flags, or None '''
    if args.year_doy and (args.time_col or args.time_format):
        raise CLIError('--year-doy cannot be combined with --time-col/--time-format')

    if args.year_doy:
        if len(args.year_doy) not in (2, 3):
            raise CLIError('--year-doy takes year, doy, and optionally fraction-of-day columns')
        year, doy = args.year_doy[:2]
        fday = args.year_doy[2] if len(args.year_doy) == 3 else None
        return {'mode': 'year_doy', 'year_col': year, 'doy_col': doy, 'fday_col': fday}

    if args.time_format:
        if not args.time_col or len(args.time_col) not in (1, 2):
            raise CLIError('--time-format needs --time-col with one or two columns')
        return {'mode': 'custom', 'cols': args.time_col, 'format': args.time_format}

    if args.time_col:
        if len(args.time_col) != 1:
            raise CLIError('Two --time-col columns need --time-format')
        return {'mode': 'iso', 'col': args.time_col[0]}
    return None

# ---- Windows ----
def collect_windows(args, session, tool_vars, prompter):
    windows = [make_window(session, s, e) for s, e in args.window]
    if args.windows:
        windows += read_windows_file(session, args.windows)
    if windows:
        return windows

    if not prompter.interactive:
        return [(None, None)] # Full time range

    first = session.to_datetime(session.time_range(tool_vars[0])[0]).isoformat() \
        if tool_vars else ''
    prompter.say(f'Data starts at {first}' if first else '')
    while True:
        start = prompter.ask('Window start (ISO 8601, Enter for start of data)',
            allow_empty=True)
        end = prompter.ask("Window end (ISO 8601 or '+10m', Enter for end of data)",
            allow_empty=True)
        try:
            windows.append(make_window(session, start, end))
        except CLIError as e:
            prompter.say(f'Invalid window: {e}')
            continue
        if not prompter.confirm('Add another window?'):
            return windows

# ---- Running ----
def window_info(session, window, names):
    ''' ISO start/end for a window, filling in the data extent for None '''
    t0, t1 = session.time_range(names[0]) if names else session.time_range()
    s = t0 if window[0] is None else window[0]
    e = t1 if window[1] is None else window[1]
    return {'start': session.to_datetime(s).isoformat(),
        'end': session.to_datetime(e).isoformat()}

def run_tool(tool, session, windows, params):
    ''' Runs the tool over every window; a failing window is reported in its
        result entry rather than aborting the others
    '''
    names = tool.variables(params)
    results = []
    for window in windows:
        entry = {'window': window_info(session, window, names)}
        try:
            entry['values'] = tool.run_window(session, window[0], window[1], **params)
        except (SessionError, CLIError, ValueError, ArithmeticError) as e:
            entry['error'] = str(e)
        results.append(entry)
    return results

def make_meta(tool, session, params):
    info = session.describe()
    return {
        'magpy_version': MAGPY_VERSION,
        'tool': tool.name,
        'params': params,
        'file': {k: info[k] for k in ('path', 'file_type', 'epoch', 'start', 'end')},
    }

EXTENSION_FORMATS = {'.json': 'json', '.csv': 'csv', '.txt': 'table',
    '.png': 'png', '.pdf': 'pdf', '.svg': 'svg'}

def format_from_path(path):
    ''' Output format implied by the file extension, or None '''
    if not path:
        return None
    return EXTENSION_FORMATS.get(os.path.splitext(path)[1].lower())

def data_for_output(data):
    ''' Copy of the results without the working data kept for rendering '''
    results = []
    for res in data['results']:
        res = dict(res)
        if 'values' in res:
            res['values'] = outputs.public_values(res['values'])
        results.append(res)
    return {**data, 'results': results}

def timestamp_slug(iso):
    ''' 2026-05-15T00:10:00.5 -> 20260515T001000 '''
    return iso.replace('-', '').replace(':', '')[:15]

def join_path(directory, name):
    return name if directory == '.' else os.path.join(directory, name)

def image_paths(tool, params, results, fmt, output):
    ''' File path for the figure of each result without an error (None for
        failed windows). A single window gets the plain name; several get the
        window start time added so the files are distinguishable.
    '''
    stem = tool.file_stem(params) if tool.file_stem else tool.name
    directory = '.'
    if output:
        if os.path.isdir(output) or output.endswith(('/', os.sep)):
            directory = output
        else:
            directory, base = os.path.split(output)
            directory = directory or '.'
            stem = os.path.splitext(base)[0]

    paths, used = [], set()
    for res in results:
        if 'error' in res:
            paths.append(None)
            continue
        name = stem
        if len(results) > 1:
            name = f"{stem}_{timestamp_slug(res['window']['start'])}"
        path, k = join_path(directory, f'{name}.{fmt}'), 1
        while path in used:
            k += 1
            path = join_path(directory, f'{name}_{k}.{fmt}')
        used.add(path)
        paths.append(path)
    return paths

def write_images(tool, session, results, params, fmt, output, stderr):
    ''' Saves one figure per successful window; returns the paths written '''
    paths = image_paths(tool, params, results, fmt, output)
    written = []
    for res, path in zip(results, paths):
        if path is None:
            continue
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            fig = tool.render(session, res['values'])
            fig.savefig(path, dpi=150)
        except OSError as e:
            raise CLIError(f"Could not write '{path}': {e}") from e
        written.append(path)
        print(f'Wrote {path}', file=stderr)
    return written

DATA_EXTENSIONS = {'json': 'json', 'csv': 'csv', 'table': 'txt'}

def data_output_path(path, tool, fmt):
    ''' A folder (existing, or given with a trailing separator) gets a file
        named after the tool inside it
    '''
    if path and (os.path.isdir(path) or path.endswith(('/', os.sep))):
        return os.path.join(path, f'{tool.name}.{DATA_EXTENSIONS[fmt]}')
    return path

def write_output(data, tool, fmt, path, stdout):
    ''' Writes formatted results to path, or stdout if there is none;
        returns the path written
    '''
    text = outputs.format_results(data_for_output(data), fmt)
    path = data_output_path(path, tool, fmt)
    if path:
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', newline='') as fd:
                fd.write(text)
        except OSError as e:
            raise CLIError(f"Could not write '{path}': {e}") from e
    else:
        stdout.write(text)
    return path

def prompt_output_path(prompter, tool, params, fmt):
    ''' Asks where to save the results. Returns a path, or None for the
        default: the terminal for data, or a file named after the tool in the
        current directory for images
    '''
    if fmt in tool_registry.IMAGE_FORMATS:
        stem = tool.file_stem(params) if tool.file_stem else tool.name
        default = f'save {stem}.{fmt} in the current directory'
    else:
        default = 'print to the terminal'
    prompter.say('Output location: a file name, or a folder to save into '
        '(end a new folder name with / to create it).')
    while True:
        answer = prompter.ask(f'Save to (Enter to {default})', allow_empty=True)
        path = answer.strip().strip('"\'') or None
        if path and os.path.isfile(path) and not prompter.confirm(
                f"'{path}' already exists. Overwrite?"):
            continue
        return path

def pick_file_and_tool(args):
    ''' The first positional may be a tool name if no file was given '''
    if args.file and not args.tool and args.file in tool_registry.TOOLS:
        if not os.path.exists(args.file):
            return None, args.file
    return args.file, args.tool

def run(argv, prompter, stdout, stderr):
    base = build_base_parser()
    args, extras = base.parse_known_args(argv)
    args.file, args.tool = pick_file_and_tool(args)

    if args.version:
        print(f'MagPy4 {MAGPY_VERSION}', file=stdout)
        return 0

    tool = None
    if args.tool is not None:
        tool = tool_registry.get_tool(args.tool)
        if tool is None:
            raise CLIError(f"Unknown tool '{args.tool}'. Available tools:\n"
                + '\n'.join(f'  {n}' for n in tool_registry.tool_names())
                + '\n(FILE and TOOL must come before any other options)')

    if args.help:
        if tool:
            build_tool_parser(tool).print_help(stdout)
        else:
            base.print_help(stdout)
            print('\ntools:', file=stdout)
            print_tool_list(stdout)
        return 0

    if args.list_tools:
        print_tool_list(stdout)
        return 0

    # Open the file
    path = args.file
    if not path:
        path = prompter.ask('Data file', flag='FILE')
    time_config = build_time_config(args)
    file_type = args.file_type or Session.detect_file_type(path)
    if time_config and file_type != 'ascii':
        print('Warning: timestamp options only apply to ASCII files; ignoring',
            file=stderr)
        time_config = None
    session = Session.from_file(path, file_type=file_type, time_config=time_config)
    for warning in session.warnings:
        print(f'Warning: {warning}', file=stderr)

    if args.list_vars:
        for name in session.variables:
            print(f'{name}\t{session.units(name)}\t{session.resolution(name):g}s\t'
                f'{len(session.times(name))} samples', file=stdout)
        return 0

    # Pick the tool
    if tool is None:
        options = [(n, t.description) for n, t in tool_registry.TOOLS.items()]
        tool = tool_registry.get_tool(prompter.choose('Which tool?', options,
            flag='TOOL'))

    tool_parser = build_tool_parser(tool)
    tool_args = tool_parser.parse_args(extras)
    if tool_args.help:
        tool_parser.print_help(stdout)
        return 0

    params = collect_params(tool, tool_args, session, prompter)
    windows = collect_windows(args, session, tool.variables(params), prompter)

    fmt = args.format or format_from_path(args.output)
    if fmt is None:
        fmt = prompter.choose('Output format?', list(tool.formats),
            default=tool.default_format) if prompter.interactive else tool.default_format
    if fmt not in tool.formats:
        raise CLIError(f"The {tool.name} tool cannot write {fmt} output "
            f"(supported: {', '.join(tool.formats)})")

    output = args.output
    if output is None and prompter.interactive:
        output = prompt_output_path(prompter, tool, params, fmt)

    results = run_tool(tool, session, windows, params)
    data = {'meta': make_meta(tool, session, params), 'results': results}
    if fmt in tool_registry.IMAGE_FORMATS:
        write_images(tool, session, results, params, fmt, output, stderr)
    else:
        written = write_output(data, tool, fmt, output, stdout)
        if written:
            n = len(results)
            print(f"Wrote {n} window{'s' if n != 1 else ''} to {written} ({fmt})",
                file=stderr)

    failed = [r for r in results if 'error' in r]
    for r in failed:
        print(f"Window {r['window']['start']} to {r['window']['end']}: "
            f"{r['error']}", file=stderr)
    return 1 if failed else 0

def main(argv=None, input_fn=None, stdin=None, stdout=None, stderr=None):
    ''' Console script entry point; returns the process exit code '''
    argv = sys.argv[1:] if argv is None else argv
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    interactive = stdin.isatty() and '--no-prompt' not in argv
    if input_fn is not None:
        interactive = '--no-prompt' not in argv
    prompter = Prompter(interactive, input_fn=input_fn, out=stderr)

    debug = '--debug' in argv
    try:
        return run(argv, prompter, stdout, stderr)
    except SystemExit as e: # argparse usage errors
        return e.code if isinstance(e.code, int) else 2
    except (CLIError, SessionError) as e:
        print(f'Error: {e}', file=stderr)
        if debug:
            traceback.print_exc(file=stderr)
        return 2
    except KeyboardInterrupt:
        print('\nCancelled', file=stderr)
        return 130
    except Exception as e:
        print(f'Unexpected error: {type(e).__name__}: {e}', file=stderr)
        if debug:
            traceback.print_exc(file=stderr)
        else:
            print('Run with --debug for details', file=stderr)
        return 1

def run_cli():
    ''' Entry point for launchers that ignore the return value of the target
        function (such as the pynsist Windows installer's commands)
    '''
    sys.exit(main())

if __name__ == '__main__':
    run_cli()
