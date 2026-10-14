''' Output writers for tool results.

    A result set is {'meta': {...}, 'results': [{'window': {...},
    'values': {...} or 'error': str}, ...]} where values may be nested
    dicts, lists, numpy arrays, and real/complex numbers.
'''
import csv
import json
import io
import math
import numpy as np

FORMATS = ('json', 'csv', 'table')

def public_values(values):
    ''' Drops keys starting with '_' (working data for rendering) '''
    return {k: v for k, v in values.items() if not k.startswith('_')}

def to_plain(value):
    ''' Converts numpy/complex values to JSON-compatible Python values.
        Complex numbers become {"real": r, "imag": i}; NaN/inf become None.
    '''
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, (complex, np.complexfloating)):
        return {'real': to_plain(float(value.real)),
            'imag': to_plain(float(value.imag))}
    if isinstance(value, np.generic):
        return to_plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value

def flatten(value, prefix=''):
    ''' Yields (key, scalar) pairs for a nested value. Lists are indexed as
        key[0], key[1][2], ...; complex numbers give key.real and key.imag
    '''
    value = to_plain(value)
    if isinstance(value, dict):
        if set(value) == {'real', 'imag'} and not any(
                isinstance(v, (list, dict)) for v in value.values()):
            yield f'{prefix}.real', value['real']
            yield f'{prefix}.imag', value['imag']
            return
        for k, v in value.items():
            yield from flatten(v, f'{prefix}.{k}' if prefix else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from flatten(v, f'{prefix}[{i}]')
    else:
        yield prefix, value

def write_json(data, stream):
    json.dump(to_plain(data), stream, indent=2)
    stream.write('\n')

def write_csv(data, stream):
    ''' One row per window; columns are the flattened values '''
    rows = []
    columns = ['window_start', 'window_end']
    for res in data['results']:
        row = {'window_start': res['window']['start'],
            'window_end': res['window']['end']}
        if 'error' in res:
            row['error'] = res['error']
        else:
            row.update(dict(flatten(res['values'])))
        for key in row:
            if key not in columns:
                columns.append(key)
        rows.append(row)

    writer = csv.DictWriter(stream, fieldnames=columns, restval='',
        lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

def write_table(data, stream):
    meta = data['meta']
    stream.write(f"Tool: {meta['tool']}    File: {meta['file']['path']}\n")
    for res in data['results']:
        win = res['window']
        stream.write(f"\nWindow: {win['start']}  to  {win['end']}\n")
        if 'error' in res:
            stream.write(f"  error: {res['error']}\n")
            continue
        items = [(k, '' if v is None else str(v))
            for k, v in flatten(res['values'])]
        width = max((len(k) for k, _ in items), default=0)
        for key, val in items:
            stream.write(f'  {key.ljust(width)}  {val}\n')

WRITERS = {'json': write_json, 'csv': write_csv, 'table': write_table}

def format_results(data, fmt):
    ''' Returns the formatted results as a string '''
    buf = io.StringIO()
    WRITERS[fmt](data, buf)
    return buf.getvalue()
